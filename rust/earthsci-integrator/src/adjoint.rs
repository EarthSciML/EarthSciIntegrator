//! The discrete adjoint (spec/adjoint.md): explicit Runge–Kutta steps
//! (§4.1), maps and clips (§5.7), the combinators (§5.1–§5.4, §5.8) and the
//! accumulation of `ū0` and `p̄` (§6).
//!
//! The reverse pass runs macro steps last first. For each it restores the
//! checkpoint at the macro step's start (spec/tape.md §4: M1 checkpoints
//! every macro step in memory), replays the step from the tape recording
//! every invocation's stage values, and then applies the adjoint of each
//! invocation in reverse execution order. The reverse of `sequence`,
//! `strang` and `subcycle` is that reversal (§5.2–§5.4); `bind_dt`
//! parameters are tape constants and get no cotangent (§5.8).

use crate::leaf::{LeafMethod, LeafSpec, StepData};
use crate::schedule::{AdjRecord, ForwardRun, RunSpec};
use crate::{Error, Scalar};
use earthsci_integrator_solvers::erk::{ErkForm, ExplicitRk};

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
            let ((_, _), _) = self.macro_step(
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
                        leaf_adjoint(l, steps, &mut lam, &mut pbar, &bound);
                    }
                    AdjRecord::Map {
                        map,
                        u_before,
                        t,
                        dt,
                        mask,
                    } => {
                        let m = &self.schedule.maps[*map];
                        let w: Vec<T> = lam.iter().map(|&x| T::from_f64(x)).collect();
                        let mut ub = vec![T::zero(); n];
                        let mut pb = vec![0.0; p.len()];
                        m.op.map_vjp(&mut ub, &mut pb, u_before, p, *t, *dt, mask, &w);
                        for (l, x) in lam.iter_mut().zip(&ub) {
                            *l = x.to_f64();
                        }
                        accumulate(&mut pbar, &pb, &bound);
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

/// Reverse of one leaf invocation (§5.1): the family's adjoint step for
/// `n = N−1, …, 0`. `lam` crosses the leaf boundary in f64; inside, stage
/// cotangents are in `T` (§6.4).
fn leaf_adjoint<T: Scalar>(
    leaf: &LeafSpec<T>,
    steps: &[StepData<T>],
    lam: &mut [f64],
    pbar: &mut [f64],
    bound: &[bool],
) {
    let mut l: Vec<T> = lam.iter().map(|&x| T::from_f64(x)).collect();
    for st in steps.iter().rev() {
        // The clip after the step: λ ← m ⊙ λ on the listed DOFs (§5.7).
        if let (Some(mask), Some(d)) = (&st.clip, &leaf.domain) {
            for (&j, &m) in d.dofs.iter().zip(mask) {
                if m == 0 {
                    l[j] = T::zero();
                }
            }
        }
        let LeafMethod::Erk(m) = &leaf.method;
        let pc = erk_adjoint_step(leaf, m, st, &mut l);
        accumulate(pbar, &pc, bound);
    }
    for (a, x) in lam.iter_mut().zip(&l) {
        *a = x.to_f64();
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
                let ti = t + c[i] * h;
                let mut si = vec![T::zero(); n];
                leaf.vjp_u(&mut si, &st.y[i], p, ti, &kappa);
                let mut pc = vec![0.0; np];
                if leaf
                    .ops
                    .iter()
                    .all(|o| o.provides(crate::operator::Call::VjpP))
                {
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
            let mut pcs: Vec<Option<Vec<f64>>> = vec![None; s];
            for j in (0..s).rev() {
                let mut aj = vec![T::zero(); n];
                let mut vj = vec![T::zero(); n];
                let mut any_v = false;
                for i in j + 1..=s {
                    let (al, be) = (alpha[i - 1][j], beta[i - 1][j]);
                    if al != 0.0 {
                        axpy(&mut aj, al, &ubar[i]);
                    }
                    if be != 0.0 {
                        axpy(&mut vj, h * be, &ubar[i]);
                        any_v = true;
                    }
                }
                if any_v {
                    let tj = t + c[j] * h;
                    let mut sj = vec![T::zero(); n];
                    leaf.vjp_u(&mut sj, &st.y[j], p, tj, &vj);
                    for (a, &x) in aj.iter_mut().zip(&sj) {
                        *a += x;
                    }
                    let mut pc = vec![0.0; np];
                    if leaf
                        .ops
                        .iter()
                        .all(|o| o.provides(crate::operator::Call::VjpP))
                    {
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

fn sum_ascending(pcs: Vec<Option<Vec<f64>>>, np: usize) -> Vec<f64> {
    let mut out = vec![0.0; np];
    for pc in pcs.into_iter().flatten() {
        for (a, x) in out.iter_mut().zip(pc) {
            *a += x;
        }
    }
    out
}
