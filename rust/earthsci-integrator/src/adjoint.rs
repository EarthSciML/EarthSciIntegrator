//! The discrete adjoint (spec/adjoint.md): explicit Runge–Kutta steps
//! (§4.1), Rosenbrock steps (§4.2), SDIRK/ESDIRK steps (§4.3), IMEX Euler
//! (§4.4) and the θ-method (§4.5), maps and clips (§5.7), the combinators
//! (§5.1–§5.4, §5.8) and the accumulation of `ū0` and `p̄` (§6).
//!
//! The reverse pass runs macro steps last first. For each it restores the
//! checkpoint at the macro step's start (spec/tape.md §4: M1 checkpoints
//! every macro step in memory), replays the step from the tape recording
//! every invocation's stage values, and then applies the adjoint of each
//! invocation in reverse execution order. The reverse of `sequence`,
//! `strang` and `subcycle` is that reversal (§5.2–§5.4); `bind_dt`
//! parameters are tape constants and get no cotangent (§5.8).

use crate::leaf::{LeafMethod, LeafSpec, LeafWork, StepData};
use crate::operator::{Call, MapOperator, MatrixStorage, MuPair};
use crate::schedule::{AdjRecord, ForwardRun, RunSpec};
use crate::{Error, Scalar};
use earthsci_integrator_solvers::erk::{ErkForm, ExplicitRk};
use earthsci_integrator_solvers::implicit::{Implicit, ImplicitForm, ImplicitWork, Part};
use earthsci_integrator_solvers::rosenbrock::{RosForm, RosWork, Rosenbrock};

/// The functional and its gradient.
#[derive(Debug, Clone)]
pub struct Gradient {
    /// `J`.
    pub functional: f64,
    /// `∂J/∂u0` (all zero unless `u0` is in `wrt`).
    pub u0: Vec<f64>,
    /// `∂J/∂p` (zero for parameters not in `wrt`).
    pub p: Vec<f64>,
}

impl<T: Scalar> RunSpec<T> {
    /// The functional `J` of a run.
    pub fn functional_value(&self, run: &ForwardRun<T>) -> Result<f64, Error> {
        let f = self
            .functional
            .as_ref()
            .ok_or_else(|| Error::Config("the run config has no adjoint functional".into()))?;
        let mut j = 0.0;
        for &tau in &f.times {
            let u = run
                .checkpoints
                .iter()
                .find(|c| c.t == tau)
                .map(|c| &c.u)
                .ok_or_else(|| Error::Run(format!("no state at observation time {tau}")))?;
            j += f.phi(u);
        }
        if f.integral {
            j += run.integral;
        }
        Ok(j)
    }

    /// The gradient of the functional by the discrete adjoint of the
    /// frozen-tape map of `run` (spec/adjoint.md §2.2).
    pub fn gradient(&self, run: &ForwardRun<T>, p: &[f64]) -> Result<Gradient, Error> {
        let f = self
            .functional
            .as_ref()
            .ok_or_else(|| Error::Config("the run config has no adjoint functional".into()))?;
        let n = self.problem.n;
        let mut lam = vec![0.0f64; n];
        let mut pbar = vec![0.0f64; p.len()];
        let index = run.tape.index();
        let ncp = run.checkpoints.len();
        // Parameters bound by bind_dt get no cotangent (§5.8).
        let mut bound = vec![false; p.len()];
        for l in &self.schedule.leaves {
            for &(i, _) in &l.binds {
                bound[i] = true;
            }
        }
        let add_obs = |lam: &mut [f64], t: f64| {
            for &tau in &f.times {
                if tau == t {
                    for &j in &f.dofs {
                        lam[j] += f.weight;
                    }
                }
            }
        };
        for k in (0..ncp - 1).rev() {
            let cp = &run.checkpoints[k];
            let t1 = run.checkpoints[k + 1].t;
            add_obs(&mut lam, t1);
            let (t0, delta) = match index.get(k.to_string().as_str()) {
                Some(crate::tape::TapeRecord::Macro { t, dt, .. }) => (*t, *dt),
                _ => return Err(Error::Run(format!("adjoint: no macro record {k}"))),
            };
            let mut u = cp.u.clone();
            let mut persist = cp.leaves.clone();
            let mut recs = Vec::new();
            self.macro_step(
                k,
                &mut u,
                &mut persist,
                p,
                t0,
                t1,
                delta,
                Some(&index),
                Some(&mut recs),
            )?;
            for rec in recs.iter().rev() {
                match rec {
                    AdjRecord::Leaf { leaf, steps } => {
                        let l = &self.schedule.leaves[*leaf];
                        leaf_adjoint(l, steps, &mut lam, &mut pbar, &bound)?;
                    }
                    AdjRecord::Map {
                        map,
                        u_before,
                        t,
                        dt,
                        mask,
                    } => {
                        let m = &self.schedule.maps[*map];
                        map_adjoint(
                            m.op.as_ref(),
                            &mut lam,
                            &mut pbar,
                            &bound,
                            u_before,
                            p,
                            *t,
                            *dt,
                            mask,
                        );
                    }
                }
            }
        }
        add_obs(&mut lam, self.tspan.0);
        let u0 = if self.wrt_u0 { lam } else { vec![0.0; n] };
        let pg = pbar
            .iter()
            .zip(&self.wrt_p)
            .map(|(&g, &w)| if w { g } else { 0.0 })
            .collect();
        Ok(Gradient {
            functional: self.functional_value(run)?,
            u0,
            p: pg,
        })
    }
}

fn accumulate(pbar: &mut [f64], contrib: &[f64], bound: &[bool]) {
    for ((a, &c), &b) in pbar.iter_mut().zip(contrib).zip(bound) {
        if !b {
            *a += c;
        }
    }
}

/// Reverse of one map invocation (§5.7): `λ ← (∂M/∂u)ᵀ λ`, `p̄ += (∂M/∂p)ᵀ λ`.
/// The map passes the DOFs it does not write through unchanged, so their
/// cotangents carry over; `map_vjp` gives the written rows' contribution.
#[allow(clippy::too_many_arguments)]
fn map_adjoint<T: Scalar>(
    op: &dyn MapOperator<T>,
    lam: &mut [f64],
    pbar: &mut [f64],
    bound: &[bool],
    u_before: &[T],
    p: &[f64],
    t: f64,
    dt: f64,
    mask: &[u8],
) {
    let w: Vec<T> = lam.iter().map(|&x| T::from_f64(x)).collect();
    let mut ub = w.clone();
    for j in op.writes() {
        ub[j] = T::zero();
    }
    let mut pb = vec![0.0; p.len()];
    op.map_vjp(&mut ub, &mut pb, u_before, p, t, dt, mask, &w);
    for (l, x) in lam.iter_mut().zip(&ub) {
        *l = x.to_f64();
    }
    accumulate(pbar, &pb, bound);
}

/// Reverse of one leaf invocation (§5.1): the family's adjoint step for
/// `n = N−1, …, 0`. `lam` crosses the leaf boundary in f64; inside, stage
/// cotangents are in `T` (§6.4).
fn leaf_adjoint<T: Scalar>(
    leaf: &LeafSpec<T>,
    steps: &[StepData<T>],
    lam: &mut [f64],
    pbar: &mut [f64],
    bound: &[bool],
) -> Result<(), Error> {
    let mut l: Vec<T> = lam.iter().map(|&x| T::from_f64(x)).collect();
    // A Rosenbrock leaf refactors W at each step's u_n, which reproduces the
    // forward factorization bit for bit (spec/adjoint.md §3, `solve_Wt`); an
    // implicit leaf refactors W at each converged stage (§4.3, D42).
    let mut lw = leaf.take_work(l.len())?;
    let out = leaf_adjoint_steps(leaf, steps, &mut l, pbar, bound, &mut lw);
    leaf.put_work(lw);
    out?;
    for (a, x) in lam.iter_mut().zip(&l) {
        *a = x.to_f64();
    }
    Ok(())
}

fn leaf_adjoint_steps<T: Scalar>(
    leaf: &LeafSpec<T>,
    steps: &[StepData<T>],
    l: &mut [T],
    pbar: &mut [f64],
    bound: &[bool],
    lw: &mut LeafWork<T>,
) -> Result<(), Error> {
    for st in steps.iter().rev() {
        // The clip after the step: λ ← m ⊙ λ, zero on the clipped DOFs (§5.7).
        if let Some(clipped) = &st.clip {
            for &j in clipped {
                l[j] = T::zero();
            }
        }
        if st.skipped {
            // A tiny stop step taken without computing it: the identity.
            continue;
        }
        let pc = match &leaf.method {
            LeafMethod::Erk(m) => erk_adjoint_step(leaf, m, st, l),
            LeafMethod::Ros(m, _) => {
                let w = lw.ros().expect("a Rosenbrock leaf's workspace");
                ros_adjoint_step(leaf, m, w, st, l)?
            }
            LeafMethod::Imp(m, _) => {
                let w = lw.imp().expect("an implicit leaf's workspace");
                imp_adjoint_step(leaf, m, w, st, l)?
            }
        };
        accumulate(pbar, &pc, bound);
    }
    Ok(())
}

/// `y += c·g` with `g = ∇ψ` the integrand's gradient (`w` on its DOFs):
/// the cotangent the integrated auxiliary state `q` sends into a stage
/// state, `c` being the stage cotangent of `q` (spec/adjoint.md §2.1).
fn add_psi_grad<T: Scalar>(y: &mut [T], leaf: &LeafSpec<T>, c: f64) {
    if let Some((dofs, w)) = &leaf.integrand {
        if c != 0.0 {
            let v = T::from_f64(c * w);
            for &j in dofs {
                y[j] += v;
            }
        }
    }
}

/// `y += c·x` with the coefficient rounded to `T`.
fn axpy<T: Scalar>(y: &mut [T], c: f64, x: &[T]) {
    let c = T::from_f64(c);
    for (a, &b) in y.iter_mut().zip(x) {
        *a += c * b;
    }
}

/// One explicit Runge–Kutta adjoint step (spec/adjoint.md §4.1): maps
/// `λ_{n+1}` (in `lam`) to `λ_n` and returns the step's `p̄` contribution,
/// summed over stages in ascending order (§6.4).
pub fn erk_adjoint_step<T: Scalar>(
    leaf: &LeafSpec<T>,
    m: &ExplicitRk,
    st: &StepData<T>,
    lam: &mut [T],
) -> Vec<f64> {
    let n = lam.len();
    let np = st.p.len();
    let (t, h, p) = (st.t, st.h, st.p.as_slice());
    let s = m.stages;
    match &m.form {
        ErkForm::Butcher { a, b, c } => {
            let mut sigma: Vec<Option<Vec<T>>> = vec![None; s];
            let mut pcs: Vec<Option<Vec<f64>>> = vec![None; s];
            for i in (0..s).rev() {
                // κ_i = h(b_i λ_{n+1} + Σ_{j>i} a_ji σ_j); a stage nothing reads is skipped.
                let reads = b[i] != 0.0 || (i + 1..s).any(|j| a[j][i] != 0.0 && sigma[j].is_some());
                if !reads {
                    continue;
                }
                let mut kappa = vec![T::zero(); n];
                if b[i] != 0.0 {
                    axpy(&mut kappa, h * b[i], lam);
                }
                for j in i + 1..s {
                    if a[j][i] != 0.0 {
                        if let Some(sj) = &sigma[j] {
                            axpy(&mut kappa, h * a[j][i], sj);
                        }
                    }
                }
                // Stage 1 at the time it was evaluated (a reused value's time).
                let ti = if i == 0 { st.tk1 } else { t + c[i] * h };
                let mut si = vec![T::zero(); n];
                leaf.vjp_u(&mut si, &st.y[i], p, ti, &kappa);
                // The integral term: q_{n+1} = q_n + Σ_i (h b_i) ψ(U_i), q̄ = 1.
                add_psi_grad(&mut si, leaf, h * b[i]);
                let mut pc = vec![0.0; np];
                if leaf.has_vjp_p() {
                    leaf.vjp_p(&mut pc, &st.y[i], p, ti, &kappa);
                }
                sigma[i] = Some(si);
                pcs[i] = Some(pc);
            }
            // λ_n = λ_{n+1} + Σ_i σ_i, ascending i.
            for si in sigma.iter().flatten() {
                for (a, &x) in lam.iter_mut().zip(si) {
                    *a += x;
                }
            }
            sum_ascending(pcs, np)
        }
        ErkForm::ShuOsher { alpha, beta, c } => {
            // ū^(s) = λ_{n+1}; for j = s−1 … 0, every ū^(i), i > j, is complete:
            // ū^(j) = Σ_i α_ij ū^(i) + J(u^(j))ᵀ v_j with v_j = Σ_i (h β_ij) ū^(i).
            let mut ubar: Vec<Vec<T>> = vec![vec![T::zero(); n]; s + 1];
            ubar[s].copy_from_slice(lam);
            // The integral term's cotangents q̄^(j): q̄^(s) = 1, and
            // q̄^(j) = Σ_i α_ij q̄^(i); ū^(j) gains (Σ_i h β_ij q̄^(i)) ∇ψ.
            let mut qbar = vec![0.0f64; s + 1];
            qbar[s] = 1.0;
            let mut pcs: Vec<Option<Vec<f64>>> = vec![None; s];
            for j in (0..s).rev() {
                let mut aj = vec![T::zero(); n];
                let mut vj = vec![T::zero(); n];
                let mut any_v = false;
                let mut qv = 0.0;
                for i in j + 1..=s {
                    let (al, be) = (alpha[i - 1][j], beta[i - 1][j]);
                    if al != 0.0 {
                        axpy(&mut aj, al, &ubar[i]);
                        qbar[j] += al * qbar[i];
                    }
                    if be != 0.0 {
                        axpy(&mut vj, h * be, &ubar[i]);
                        qv += (h * be) * qbar[i];
                        any_v = true;
                    }
                }
                add_psi_grad(&mut aj, leaf, qv);
                if any_v {
                    let tj = t + c[j] * h;
                    let mut sj = vec![T::zero(); n];
                    leaf.vjp_u(&mut sj, &st.y[j], p, tj, &vj);
                    for (a, &x) in aj.iter_mut().zip(&sj) {
                        *a += x;
                    }
                    let mut pc = vec![0.0; np];
                    if leaf.has_vjp_p() {
                        leaf.vjp_p(&mut pc, &st.y[j], p, tj, &vj);
                    }
                    pcs[j] = Some(pc);
                }
                ubar[j] = aj;
            }
            lam.copy_from_slice(&ubar[0]);
            sum_ascending(pcs, np)
        }
    }
}

/// Whether a Rosenbrock leaf forms parameter cotangents: every operator
/// provides `vjp_p` and `hess_p`, and `ft_vjp_p` where `f_t` is formed.
fn ros_has_p_adjoint<T: Scalar>(leaf: &LeafSpec<T>, forms_ft: bool) -> bool {
    leaf.has_vjp_p()
        && leaf.ops.iter().all(|o| {
            o.provides(Call::HessP) && (!forms_ft || !o.depends_on_t() || o.provides(Call::FtVjpP))
        })
}

/// `x ← c·x` with the coefficient rounded to `T`.
fn scale<T: Scalar>(x: &mut [T], c: f64) {
    let c = T::from_f64(c);
    for a in x.iter_mut() {
        *a *= c;
    }
}

/// One Rosenbrock adjoint step (spec/adjoint.md §4.2): maps `λ_{n+1}` (in
/// `lam`) to `λ_n` and returns the step's `p̄` contribution.
///
/// `W` is refactored at the step's `u_n` exactly as the forward step formed
/// it, so `solve_Wt` uses the forward factorization (spec/adjoint.md §3).
/// Stage terms accumulate in ascending stage order (§6.4): `λ_n = λ_{n+1} +
/// Σ_i σ_i`, then `hess_u`, then `ft_vjp_u`; `p̄` likewise.
///
/// - Transformed form: seed `k̄_i = b_i λ_{n+1}`; for `i = s … 1`,
///   `μ_i = Ŵ⁻ᵀ k̄_i`, `σ_i = J(U_i)ᵀ μ_i` at `t_n + c_i h`, and for `j < i`,
///   `k̄_j += A_ij σ_i + (C_ij/h) μ_i` (the `A` term first). `hess_u` takes
///   the pairs `(μ_i, k_i)` and `ft_vjp_u` takes `ν = Σ_i (h d_i) μ_i`, both
///   at `(u_n, t_n)`.
/// - Rosenbrock23, with `Ŵ = I − γJ = γ M` (`γ = h d`, `M = σI − J`) so
///   `Ŵ⁻ᵀ v = (M⁻ᵀ v)·σ`: `k̄_2 = h λ_{n+1}`, `μ_2 = Ŵ⁻ᵀ k̄_2`,
///   `σ_2 = J(U_2)ᵀ μ_2` at `t_n + h/2`; `k̄_1 = (k̄_2 − μ_2) + (h/2) σ_2`,
///   `μ_1 = Ŵ⁻ᵀ k̄_1`, `σ_1 = J(u_n)ᵀ μ_1` at the time `f_0` was evaluated
///   (`t_n`, or the previous step's end when it was the stored value, §3.3);
///   `hess_u` takes `(μ_1, γ k_1)` and `(μ_2, γ (k_2 − k_1))`, and
///   `ft_vjp_u` takes `ν = γ μ_1`.
pub fn ros_adjoint_step<T: Scalar>(
    leaf: &LeafSpec<T>,
    m: &Rosenbrock,
    w: &mut RosWork<T>,
    st: &StepData<T>,
    lam: &mut [T],
) -> Result<Vec<f64>, Error> {
    let n = lam.len();
    let np = st.p.len();
    let (t, h, p) = (st.t, st.h, st.p.as_slice());
    let un = &st.y[0];
    let mut jac = |o: &mut MatrixStorage<T>, uu: &[T], tt: f64| leaf.jac(o, uu, p, tt);
    let sigma = m.factor_w(w, &mut jac, un, t, h).ok_or_else(|| {
        Error::Run(format!(
            "adjoint: leaf {}: W is singular at the step from t = {t}",
            leaf.id
        ))
    })?;
    let forms_ft = leaf.forms_ft();
    let with_p = ros_has_p_adjoint(leaf, forms_ft);
    let mut pc = vec![0.0; np];
    match &m.form {
        RosForm::Transformed {
            a, cmat, c, d, b, ..
        } => {
            let s = m.stages;
            let mut kbar = vec![vec![T::zero(); n]; s];
            for (kb, &bi) in kbar.iter_mut().zip(b) {
                if bi != 0.0 {
                    axpy(kb, bi, lam);
                }
            }
            // The integral term (spec/adjoint.md §2.1): with q̄ = 1, k̄_q,i is
            // seeded b_i; Ŵ_augᵀ = [[Ŵᵀ, −g], [0, σ]] gives μ_q,i = k̄_q,i/σ
            // and μ_i = Ŵ⁻ᵀ(k̄_i + μ_q,i g); σ_i gains μ_q,i g, and k̄_q,j
            // gains (C_ij/h) μ_q,i. ψ is linear and t- and p-independent, so
            // it adds nothing to the second-order and f_t terms.
            let integ = leaf.integrand.is_some();
            let mut kq: Vec<f64> = b.to_vec();
            let mut mu = vec![Vec::new(); s];
            let mut sig = vec![Vec::new(); s];
            let mut pcs: Vec<Option<Vec<f64>>> = vec![None; s];
            for i in (0..s).rev() {
                let mut mui = std::mem::take(&mut kbar[i]);
                let muq = if integ { kq[i] / sigma } else { 0.0 };
                add_psi_grad(&mut mui, leaf, muq);
                w.lu.solve_t(&mut mui);
                let ti = t + c[i] * h;
                let mut si = vec![T::zero(); n];
                leaf.vjp_u(&mut si, &st.y[i], p, ti, &mui);
                add_psi_grad(&mut si, leaf, muq);
                for j in 0..i {
                    if a[i][j] != 0.0 {
                        axpy(&mut kbar[j], a[i][j], &si);
                    }
                    if cmat[i][j] != 0.0 {
                        axpy(&mut kbar[j], cmat[i][j] / h, &mui);
                        kq[j] += (cmat[i][j] / h) * muq;
                    }
                }
                if with_p {
                    let mut pi = vec![0.0; np];
                    leaf.vjp_p(&mut pi, &st.y[i], p, ti, &mui);
                    pcs[i] = Some(pi);
                }
                mu[i] = mui;
                sig[i] = si;
            }
            for si in &sig {
                for (l, &x) in lam.iter_mut().zip(si) {
                    *l += x;
                }
            }
            let pairs: Vec<MuPair<'_, T>> = (0..s).map(|i| (&mu[i][..], &st.k[i][..])).collect();
            leaf.hess_u(lam, un, p, t, &pairs);
            let mut nu = None;
            if forms_ft {
                let mut v = vec![T::zero(); n];
                for (i, &di) in d.iter().enumerate() {
                    if di != 0.0 {
                        axpy(&mut v, h * di, &mu[i]);
                    }
                }
                leaf.ft_vjp_u(lam, un, p, t, &v);
                nu = Some(v);
            }
            if with_p {
                pc = sum_ascending(pcs, np);
                leaf.hess_p(&mut pc, un, p, t, &pairs);
                if let Some(v) = &nu {
                    leaf.ft_vjp_p(&mut pc, un, p, t, v);
                }
            }
        }
        RosForm::Ode23s { d, .. } => {
            let gam = h * d;
            let t2 = t + h / 2.0;
            // μ_2 = Ŵ⁻ᵀ (h λ_{n+1}).
            let mut kbar2 = vec![T::zero(); n];
            axpy(&mut kbar2, h, lam);
            // The integral term (spec/adjoint.md §2.1), with Ŵ_augᵀ =
            // [[Ŵᵀ, −γg], [0, 1]]: μ_q,2 = k̄_q,2 = h, μ_2 = Ŵ⁻ᵀ(k̄_2 + γ h g),
            // σ_2 gains h g; then k̄_q,1 = k̄_q,2 − μ_q,2 = 0, so stage 1
            // gains nothing.
            let muq2 = if leaf.integrand.is_some() { h } else { 0.0 };
            let mut mu2 = kbar2.clone();
            add_psi_grad(&mut mu2, leaf, gam * muq2);
            w.lu.solve_t(&mut mu2);
            scale(&mut mu2, sigma);
            let mut s2 = vec![T::zero(); n];
            leaf.vjp_u(&mut s2, &st.y[1], p, t2, &mu2);
            add_psi_grad(&mut s2, leaf, muq2);
            // k̄_1 = (k̄_2 − μ_2) + (h/2) σ_2; μ_1 = Ŵ⁻ᵀ k̄_1.
            let half = T::from_f64(h / 2.0);
            let mut mu1: Vec<T> = (0..n).map(|j| (kbar2[j] - mu2[j]) + half * s2[j]).collect();
            w.lu.solve_t(&mut mu1);
            scale(&mut mu1, sigma);
            let mut s1 = vec![T::zero(); n];
            leaf.vjp_u(&mut s1, un, p, st.tk1, &mu1);
            for (l, (&x1, &x2)) in lam.iter_mut().zip(s1.iter().zip(&s2)) {
                *l += x1;
                *l += x2;
            }
            let g = T::from_f64(gam);
            let w1: Vec<T> = st.k[0].iter().map(|&k1| g * k1).collect();
            let w2: Vec<T> = st.k[1]
                .iter()
                .zip(&st.k[0])
                .map(|(&k2, &k1)| g * (k2 - k1))
                .collect();
            let pairs: [MuPair<'_, T>; 2] = [(&mu1, &w1), (&mu2, &w2)];
            leaf.hess_u(lam, un, p, t, &pairs);
            let nu: Option<Vec<T>> = forms_ft.then(|| mu1.iter().map(|&x| g * x).collect());
            if let Some(v) = &nu {
                leaf.ft_vjp_u(lam, un, p, t, v);
            }
            if with_p {
                leaf.vjp_p(&mut pc, un, p, st.tk1, &mu1);
                leaf.vjp_p(&mut pc, &st.y[1], p, t2, &mu2);
                leaf.hess_p(&mut pc, un, p, t, &pairs);
                if let Some(v) = &nu {
                    leaf.ft_vjp_p(&mut pc, un, p, t, v);
                }
            }
        }
    }
    Ok(pc)
}

/// Refactors `M = σI − J_part(u, t)` for an implicit leaf's adjoint
/// (spec/adjoint.md §4.3, D42), naming the step on failure.
#[allow(clippy::too_many_arguments)]
fn refactor<T: Scalar>(
    leaf: &LeafSpec<T>,
    m: &Implicit,
    w: &mut ImplicitWork<T>,
    part: Part,
    u: &[T],
    p: &[f64],
    t: f64,
    h: f64,
) -> Result<f64, Error> {
    let mut jac =
        |pt: Part, o: &mut MatrixStorage<T>, uu: &[T], tt: f64| leaf.jac_part(pt, o, uu, p, tt);
    m.factor(w, &mut jac, part, u, t, h).ok_or_else(|| {
        Error::Run(format!(
            "adjoint: leaf {}: W is singular at the converged stage at t = {t}",
            leaf.id
        ))
    })
}

/// `x ← M⁻ᵀ (x·σ)`: the transposed solve with `W = I − hγJ = hγ M`.
fn solve_wt<T: Scalar>(w: &ImplicitWork<T>, x: &mut [T], sigma: f64) {
    scale(x, sigma);
    w.lu.solve_t(x);
}

/// One SDIRK, θ-method or IMEX Euler adjoint step (spec/adjoint.md
/// §4.3–§4.5): maps `λ_{n+1}` (in `lam`) to `λ_n` and returns the step's `p̄`
/// contribution. Every transposed system is the exact one at the converged
/// stage: `W` is refactored there (`block_lu`/`dense_lu`, D42), not taken from
/// the forward factorization at `u_n`.
///
/// - SDIRK: for `i = s … 1`, `R_i = h(b_i λ_{n+1} + Σ_{j>i} a_ji ρ_j)`, plus
///   `hγ ζ_i` for the integral term (`ζ_i = (h b_i) ∇ψ`, the cotangent the
///   increment `Δq = Σ_i b_i h ψ(Z_i)` sends into `Z_i`), `W_iᵀ ω_i = R_i` with
///   `W_i = I − hγ J(Z_i, t_n + c_i h)` (the identity for an explicit first
///   stage, whose `γ` is 0), `ρ_i = J(Z_i)ᵀ ω_i + ζ_i`; then
///   `λ_n = λ_{n+1} + Σ_i ρ_i` and `p̄ += Σ_i f_p(Z_i)ᵀ ω_i`, ascending `i`.
/// - θ-method: `Wᵀ ω = λ_{n+1} + (hθ)∇ψ` with `W = I − hθ J(u_{n+1}, t_n + h)`,
///   `λ_n = ω + J(u_n)ᵀ ((h(1−θ)) ω) + (h(1−θ))∇ψ`,
///   `p̄ += f_p(u_n)ᵀ((h(1−θ))ω) + f_p(u_{n+1})ᵀ((hθ)ω)`.
/// - IMEX Euler: `Wᵀ ω = λ_{n+1}` with `W = I − h J_I(u_{n+1}, t_n + h)`,
///   `λ_n = ω + J_E(u_n)ᵀ (hω)`, `p̄ += f_{E,p}(u_n)ᵀ(hω) + f_{I,p}(u_{n+1})ᵀ(hω)`.
/// - Linearized IMEX Euler, a one-stage Rosenbrock step with `Ŵ = σI − J_I(u_n)`,
///   `σ = 1/h`, `Ŵ k = f(u_n)`: `μ = Ŵ⁻ᵀ λ_{n+1}`,
///   `λ_n = λ_{n+1} + J(u_n)ᵀ μ + ∇_u(μᵀ J_I(u) k)`, `p̄ += f_p(u_n)ᵀ μ + ∇_p(μᵀ J_I k)`
///   (§4.2 with one stage, D43).
pub fn imp_adjoint_step<T: Scalar>(
    leaf: &LeafSpec<T>,
    m: &Implicit,
    w: &mut ImplicitWork<T>,
    st: &StepData<T>,
    lam: &mut [T],
) -> Result<Vec<f64>, Error> {
    let n = lam.len();
    let np = st.p.len();
    let (t, h, p) = (st.t, st.h, st.p.as_slice());
    let with_p = leaf.has_vjp_p();
    let mut pc = vec![0.0; np];
    match &m.form {
        ImplicitForm::Sdirk {
            a,
            b,
            c,
            gamma,
            explicit_first,
            ..
        } => {
            let s = m.stages;
            let mut rho: Vec<Vec<T>> = vec![Vec::new(); s];
            let mut pcs: Vec<Option<Vec<f64>>> = vec![None; s];
            for i in (0..s).rev() {
                let mut om = vec![T::zero(); n];
                if b[i] != 0.0 {
                    axpy(&mut om, h * b[i], lam);
                }
                for j in i + 1..s {
                    if a[j][i] != 0.0 {
                        axpy(&mut om, h * a[j][i], &rho[j]);
                    }
                }
                let zq = h * b[i];
                let explicit = i == 0 && *explicit_first;
                let ti = if explicit { t } else { t + c[i] * h };
                if !explicit {
                    add_psi_grad(&mut om, leaf, h * gamma * zq);
                    let sigma = refactor(leaf, m, w, Part::All, &st.y[i], p, ti, h)?;
                    solve_wt(w, &mut om, sigma);
                }
                let mut ri = vec![T::zero(); n];
                leaf.vjp_u(&mut ri, &st.y[i], p, ti, &om);
                add_psi_grad(&mut ri, leaf, zq);
                if with_p {
                    let mut pi = vec![0.0; np];
                    leaf.vjp_p(&mut pi, &st.y[i], p, ti, &om);
                    pcs[i] = Some(pi);
                }
                rho[i] = ri;
            }
            for ri in &rho {
                for (l, &x) in lam.iter_mut().zip(ri) {
                    *l += x;
                }
            }
            if with_p {
                pc = sum_ascending(pcs, np);
            }
        }
        ImplicitForm::Theta { theta } => {
            let th = *theta;
            let (un, unew) = (&st.y[0], &st.y[1]);
            let mut om = lam.to_vec();
            add_psi_grad(&mut om, leaf, h * th);
            if th != 0.0 {
                let sigma = refactor(leaf, m, w, Part::All, unew, p, t + h, h)?;
                solve_wt(w, &mut om, sigma);
            }
            lam.copy_from_slice(&om);
            if th != 1.0 {
                let mut v = om.clone();
                scale(&mut v, h * (1.0 - th));
                leaf.vjp_u_part(Part::All, lam, un, p, t, &v);
                add_psi_grad(lam, leaf, h * (1.0 - th));
                if with_p {
                    leaf.vjp_p(&mut pc, un, p, t, &v);
                }
            }
            if with_p && th != 0.0 {
                let mut v = om;
                scale(&mut v, h * th);
                leaf.vjp_p(&mut pc, unew, p, t + h, &v);
            }
        }
        ImplicitForm::ImexEuler {
            linearized: false,
            has_explicit,
        } => {
            let (un, unew) = (&st.y[0], &st.y[1]);
            let mut om = lam.to_vec();
            let sigma = refactor(leaf, m, w, Part::Implicit, unew, p, t + h, h)?;
            solve_wt(w, &mut om, sigma);
            lam.copy_from_slice(&om);
            let mut v = om;
            scale(&mut v, h);
            if *has_explicit {
                leaf.vjp_u_part(Part::Explicit, lam, un, p, t, &v);
                if with_p {
                    leaf.vjp_p_part(Part::Explicit, &mut pc, un, p, t, &v);
                }
            }
            if with_p {
                leaf.vjp_p_part(Part::Implicit, &mut pc, unew, p, t + h, &v);
            }
        }
        ImplicitForm::ImexEuler {
            linearized: true, ..
        } => {
            let un = &st.y[0];
            // Ŵ = σI − J_I(u_n) with σ = 1/h is the forward factorization, bit
            // for bit, as for Rosenbrock (spec/adjoint.md §3).
            refactor(leaf, m, w, Part::Implicit, un, p, t, h)?;
            let mut mu = lam.to_vec();
            w.lu.solve_t(&mut mu);
            let mut s1 = vec![T::zero(); n];
            leaf.vjp_u(&mut s1, un, p, t, &mu);
            for (l, &x) in lam.iter_mut().zip(&s1) {
                *l += x;
            }
            let pairs: [MuPair<'_, T>; 1] = [(&mu, &st.k[0])];
            leaf.hess_u_part(Part::Implicit, lam, un, p, t, &pairs);
            let implicit_hess_p = leaf
                .ops
                .iter()
                .enumerate()
                .filter(|(k, _)| leaf.imex_implicit.get(*k).copied().unwrap_or(true))
                .all(|(_, o)| o.provides(Call::HessP));
            if with_p && implicit_hess_p {
                leaf.vjp_p(&mut pc, un, p, t, &mu);
                leaf.hess_p_part(Part::Implicit, &mut pc, un, p, t, &pairs);
            }
        }
    }
    Ok(pc)
}

fn sum_ascending(pcs: Vec<Option<Vec<f64>>>, np: usize) -> Vec<f64> {
    let mut out = vec![0.0; np];
    for pc in pcs.into_iter().flatten() {
        for (a, x) in out.iter_mut().zip(pc) {
            *a += x;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `u₀ ← a·u₀·u₁` on a two-DOF state; `u₁` passes through.
    struct Product;

    impl MapOperator<f64> for Product {
        fn name(&self) -> &str {
            "product"
        }
        fn writes(&self) -> Vec<usize> {
            vec![0]
        }
        fn apply(
            &self,
            out: &mut [f64],
            u: &[f64],
            p: &[f64],
            t: f64,
            dt: f64,
            mask: &mut Vec<u8>,
        ) {
            mask.clear();
            self.apply_frozen(out, u, p, t, dt, mask);
        }
        fn apply_frozen(
            &self,
            out: &mut [f64],
            u: &[f64],
            p: &[f64],
            _t: f64,
            _dt: f64,
            _m: &[u8],
        ) {
            out[0] = p[0] * u[0] * u[1];
        }
        fn map_vjp(
            &self,
            out_u: &mut [f64],
            out_p: &mut [f64],
            u: &[f64],
            p: &[f64],
            _t: f64,
            _dt: f64,
            _mask: &[u8],
            w: &[f64],
        ) {
            out_u[0] += p[0] * u[1] * w[0];
            out_u[1] += p[0] * u[0] * w[0];
            out_p[0] += u[0] * u[1] * w[0];
        }
        fn map_jvp(
            &self,
            out: &mut [f64],
            u: &[f64],
            p: &[f64],
            _t: f64,
            _dt: f64,
            _mask: &[u8],
            v_u: &[f64],
            v_p: &[f64],
        ) {
            out[0] += p[0] * (u[1] * v_u[0] + u[0] * v_u[1]) + u[0] * u[1] * v_p[0];
        }
    }

    #[test]
    fn map_adjoint_keeps_untouched_dofs() {
        // M(u) = (a u₀ u₁, u₁): λ ← (a u₁ λ₀, a u₀ λ₀ + λ₁), p̄ = u₀ u₁ λ₀.
        let (u, p) = ([2.0, 3.0], [5.0]);
        let mut lam = [7.0, 11.0];
        let mut pbar = [0.0];
        map_adjoint(
            &Product,
            &mut lam,
            &mut pbar,
            &[false],
            &u,
            &p,
            0.0,
            1.0,
            &[],
        );
        assert_eq!(lam, [5.0 * 3.0 * 7.0, 5.0 * 2.0 * 7.0 + 11.0]);
        assert_eq!(pbar, [2.0 * 3.0 * 7.0]);
    }
}
