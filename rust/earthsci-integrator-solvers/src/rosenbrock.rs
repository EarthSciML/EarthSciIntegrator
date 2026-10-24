//! Rosenbrock methods (spec/methods.md §4): the transformed form (ROS2,
//! ROS3P, Rodas3, Rodas4, Rodas5P and KPP's Ros2/3/4, Rodas3/4) and
//! Rosenbrock23's ode23s form, with their error estimates.
//!
//! The arithmetic is OrdinaryDiffEqRosenbrock 2.7.5's, term by term, with
//! no fused multiply-adds (D10):
//!
//! **Transformed form** (`src/rosenbrock_perform_step.jl:431-540`, the
//! out-of-place generic step). With `dtC_ij = C_ij / h`, `dtd_i = h·d_i`
//! and `σ = 1/(h·γ)`:
//! - `Ŵ = σI − J(u_n, t_n)`, each diagonal entry `σ − J_ii` rounded once
//!   (the exact negation of OrdinaryDiffEq's `W = J − I/(hγ)`, so solving
//!   `Ŵ k = r` gives OrdinaryDiffEq's `W \ −r` bit for bit);
//! - `U_i = u_n + Σ_{j<i} A_ij k_j`, ascending `j`, each term `A_ij·k_j`;
//! - `r_i = (f(U_i, t_n + c_i h) + dtd_i·f_t) + s_i` with
//!   `s_i = Σ_{j<i} dtC_ij·k_j` summed ascending from zero first. The `f_t`
//!   term is left out when `f_t` is not formed (§4.2) or `d_i = 0`;
//! - `u_{n+1} = u_n + Σ_i b_i k_i` ascending, and `e = Σ_i b̃_i k_i` (in f64);
//! - a stage whose row of `A` and whose `c_i` equal the previous stage's
//!   reuses its `f` value (OrdinaryDiffEq's test, from the third stage on;
//!   §4.1). This is KPP's `new_f` for the KPP tables.
//!
//! **Rosenbrock23** (`src/rosenbrock_perform_step.jl:34-140`, the ode23s
//! step). With `γ = h·d` and `σ = 1/γ`, OrdinaryDiffEq factors
//! `W = J − I/γ` and scales each solve by `−1/γ`; here `M = σI − J = −W`
//! is factored, and each solve is scaled by `σ`, which is the same
//! arithmetic with both signs flipped:
//! - `k_1 = (M⁻¹ (f_0 + γ·f_t))·σ`;
//! - `U_2 = u_n + (h/2)·k_1`, `f_1 = f(U_2, t_n + h/2)`;
//! - `k_2 = (M⁻¹ (f_1 − k_1))·σ + k_1`, `u_{n+1} = u_n + h·k_2`;
//! - `f_2 = f(u_{n+1}, t_n + h)` (the next step's `f_0`, §3.3);
//! - `k_3 = (M⁻¹ (((f_2 − c32·(k_2 − f_1)) − 2·(k_1 − f_0)) + γ·f_t))·σ`,
//!   `e = (h/6)·((k_1 − 2·k_2) + k_3)`.
//!
//! `M = Ŵ/γ` with spec/adjoint.md §4.2's `Ŵ = I − hdJ`, so the step is the
//! spec's; only the rounding follows OrdinaryDiffEq.

// Index loops mirror the elimination and stage formulas.
#![allow(clippy::needless_range_loop)]

use crate::linalg::{LinearSolver, LinsolveKind, MatrixStorage, Structure};
use crate::scalar::Scalar;
use crate::tableau::{DCoefficients, ErrorEstimate, Family, Form, Tableau};
use crate::{NormFn, RhsFn};

/// The step form and its coefficients.
#[derive(Debug, Clone)]
pub enum RosForm {
    /// `A`, `C`, `γ`, `c`, `d`, `b`, `b̃`.
    Transformed {
        a: Vec<Vec<f64>>,
        cmat: Vec<Vec<f64>>,
        gamma: f64,
        c: Vec<f64>,
        d: Vec<f64>,
        b: Vec<f64>,
        btilde: Option<Vec<f64>>,
        /// Stage `i` reuses stage `i−1`'s `f` value.
        reuse_f: Vec<bool>,
    },
    /// Rosenbrock23: `d` and `c32`.
    Ode23s { d: f64, c32: f64 },
}

/// A Rosenbrock method built from its table.
#[derive(Debug, Clone)]
pub struct Rosenbrock {
    pub name: String,
    pub form: RosForm,
    pub stages: usize,
    pub order: u32,
    pub embedded_order: Option<u32>,
    /// The table's `first_same_as_last` (Rosenbrock23).
    pub fsal: bool,
}

/// The `jac(out, u, t)` call as a step sees it: overwrites `out` with `J(u, t)`.
pub type JacFn<'a, T> = dyn FnMut(&mut MatrixStorage<T>, &[T], f64) + 'a;

/// What one Rosenbrock step attempt produced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RosOutcome {
    /// `W` is singular; nothing else was computed (spec/methods.md §4.2).
    Singular,
    /// The step was computed; the scaled error when one was asked for.
    Done(Option<f64>),
}

impl Rosenbrock {
    /// Builds the method from a table; fails for non-Rosenbrock tables.
    pub fn from_tableau(t: &Tableau) -> Result<Self, String> {
        if t.family != Family::Rosenbrock {
            return Err(format!("{} is not a Rosenbrock method", t.name));
        }
        let form = match t.form {
            Form::RosenbrockTransformed => {
                let a = t.a.clone().ok_or("missing A")?;
                let cmat = t.c_mat.clone().ok_or("missing C")?;
                let c = t.c.clone().ok_or("missing c")?;
                let d = match &t.d {
                    Some(DCoefficients::Vector(v)) => v.clone(),
                    _ => return Err(format!("{}: d must be a vector", t.name)),
                };
                let b = t.b.clone().ok_or("missing b")?;
                let btilde = match t.error_estimate {
                    ErrorEstimate::Embedded => Some(t.btilde.clone().ok_or("missing btilde")?),
                    ErrorEstimate::None => None,
                    e => return Err(format!("{}: error estimate {e:?}", t.name)),
                };
                let s = t.stages;
                if a.len() != s || cmat.len() != s || c.len() != s || d.len() != s || b.len() != s {
                    return Err(format!("{}: coefficient sizes differ from stages", t.name));
                }
                // OrdinaryDiffEqRosenbrock src/rosenbrock_perform_step.jl:470-474:
                // `stage > 2 && c[stage] == c[stage-1] && all(A[stage, j] ==
                // A[stage-1, j] for j in 1:stage-1)`.
                let reuse_f = (0..s)
                    .map(|i| i >= 2 && c[i] == c[i - 1] && (0..i).all(|j| a[i][j] == a[i - 1][j]))
                    .collect();
                RosForm::Transformed {
                    a,
                    cmat,
                    gamma: t.gamma.ok_or("missing gamma")?,
                    c,
                    d,
                    b,
                    btilde,
                    reuse_f,
                }
            }
            Form::Rosenbrock23 => {
                let d = match &t.d {
                    Some(DCoefficients::Scalar(x)) => *x,
                    _ => return Err(format!("{}: d must be a number", t.name)),
                };
                if t.error_estimate != ErrorEstimate::Rosenbrock23 {
                    return Err(format!("{}: error estimate", t.name));
                }
                RosForm::Ode23s {
                    d,
                    c32: t.c32.ok_or("missing c32")?,
                }
            }
            f => return Err(format!("{}: form {f:?} is not Rosenbrock", t.name)),
        };
        Ok(Rosenbrock {
            name: t.name.clone(),
            form,
            stages: t.stages,
            order: t.order,
            embedded_order: t.embedded_order,
            fsal: t.first_same_as_last,
        })
    }

    /// Whether the method has an error estimate.
    pub fn adaptive(&self) -> bool {
        match &self.form {
            RosForm::Transformed { btilde, .. } => btilde.is_some(),
            RosForm::Ode23s { .. } => true,
        }
    }

    /// Whether some `d_i ≠ 0`, so the step reads `f_t` (spec/methods.md §4.2).
    pub fn uses_ft(&self) -> bool {
        match &self.form {
            RosForm::Transformed { d, .. } => d.iter().any(|&x| x != 0.0),
            RosForm::Ode23s { d, .. } => *d != 0.0,
        }
    }

    /// The `γ` of the step's `W`: the table's `γ`, or Rosenbrock23's `d`.
    pub fn w_gamma(&self) -> f64 {
        match &self.form {
            RosForm::Transformed { gamma, .. } => *gamma,
            RosForm::Ode23s { d, .. } => *d,
        }
    }

    /// Time of stage `i` (the stage's `f` evaluation).
    pub fn stage_time(&self, i: usize, t: f64, h: f64) -> f64 {
        match &self.form {
            RosForm::Transformed { c, .. } => t + c[i] * h,
            RosForm::Ode23s { .. } => {
                if i == 0 {
                    t
                } else {
                    t + h / 2.0
                }
            }
        }
    }

    /// A workspace for a state of length `n` whose matrices have `structure`.
    pub fn workspace<T: Scalar>(
        &self,
        n: usize,
        structure: Structure,
        linsolve: LinsolveKind,
    ) -> Result<RosWork<T>, String> {
        let s = self.stages;
        Ok(RosWork {
            y: vec![vec![T::zero(); n]; s],
            k: vec![vec![T::zero(); n]; s],
            fs: vec![T::zero(); n],
            f0: vec![T::zero(); n],
            f1: vec![T::zero(); n],
            r: vec![T::zero(); n],
            ft: vec![T::zero(); n],
            jac: MatrixStorage::zeros(structure, n),
            w: MatrixStorage::zeros(structure, n),
            lu: LinearSolver::new(linsolve, structure, n)?,
            unew: vec![T::zero(); n],
            fend: vec![T::zero(); n],
            has_fend: false,
            e: vec![0.0; n],
            nf: 0,
        })
    }

    /// Forms `J(u_n, t_n)` with `jac` and factors `σI − J` with
    /// `σ = 1/(h·γ)` into `w.lu`. Returns `σ`, or `None` when singular. The
    /// forward step and the adjoint share it, so the adjoint's transposed
    /// solves use the forward factorization bit for bit.
    pub fn factor_w<T: Scalar>(
        &self,
        w: &mut RosWork<T>,
        jac: &mut JacFn<'_, T>,
        un: &[T],
        t: f64,
        h: f64,
    ) -> Option<f64> {
        let dtgamma = h * self.w_gamma();
        let sigma = 1.0 / dtgamma;
        w.jac.clear();
        jac(&mut w.jac, un, t);
        w.w.shifted_negation(&w.jac, T::from_f64(sigma));
        if w.lu.factor(&w.w) {
            Some(sigma)
        } else {
            None
        }
    }

    /// Takes one step of size `h` from `(un, t)`.
    ///
    /// - `f(out, u, t)` overwrites `out` with the leaf's right-hand side,
    ///   `jac` forms `J`, and `ft`, when given, overwrites its output with
    ///   `f_t(u, t)`; it is given only when spec/methods.md §4.2 forms `f_t`.
    /// - `f0`: Rosenbrock23's stored `f(u_n)` (§3.3), if valid.
    /// - `norm(e, u_n, u_{n+1})`: the error norm of §7, when the error is wanted.
    ///
    /// On return `w.unew` is `u_{n+1}`, `w.y` the stage states (`w.y[0] =
    /// u_n`), `w.k` the stage increments, and `w.fend` holds
    /// `f(u_{n+1}, t_n + h)` when `w.has_fend` (Rosenbrock23).
    #[allow(clippy::too_many_arguments)]
    pub fn step<T: Scalar>(
        &self,
        w: &mut RosWork<T>,
        f: &mut RhsFn<'_, T>,
        jac: &mut JacFn<'_, T>,
        ft: Option<&mut RhsFn<'_, T>>,
        un: &[T],
        t: f64,
        h: f64,
        f0: Option<&[T]>,
        norm: Option<&mut NormFn<'_, T>>,
    ) -> RosOutcome {
        w.has_fend = false;
        let Some(sigma) = self.factor_w(w, jac, un, t, h) else {
            return RosOutcome::Singular;
        };
        let has_ft = match ft {
            Some(ftf) => {
                ftf(&mut w.ft, un, t);
                true
            }
            None => false,
        };
        w.y[0].copy_from_slice(un);
        match &self.form {
            RosForm::Transformed {
                a,
                cmat,
                c,
                d,
                b,
                btilde,
                reuse_f,
                ..
            } => {
                let s = self.stages;
                for i in 0..s {
                    if i > 0 {
                        let (lo, hi) = w.y.split_at_mut(i);
                        let yi = &mut hi[0];
                        yi.copy_from_slice(&lo[0]);
                        for j in 0..i {
                            if a[i][j] != 0.0 {
                                let coef = T::from_f64(a[i][j]);
                                for (y, &kv) in yi.iter_mut().zip(w.k[j].iter()) {
                                    *y += coef * kv;
                                }
                            }
                        }
                    }
                    if !reuse_f[i] {
                        f(&mut w.fs, &w.y[i], t + c[i] * h);
                        w.nf += 1;
                    }
                    // r_i = (f_i + dtd_i·f_t) + Σ_j dtC_ij k_j.
                    w.r.copy_from_slice(&w.fs);
                    if has_ft && d[i] != 0.0 {
                        let dtd = T::from_f64(h * d[i]);
                        for (r, &x) in w.r.iter_mut().zip(w.ft.iter()) {
                            *r += dtd * x;
                        }
                    }
                    if (0..i).any(|j| cmat[i][j] != 0.0) {
                        let ki = &mut w.k[i];
                        ki.iter_mut().for_each(|x| *x = T::zero());
                        for j in 0..i {
                            if cmat[i][j] != 0.0 {
                                let coef = T::from_f64(cmat[i][j] / h);
                                let (lo, hi) = w.k.split_at_mut(i);
                                for (sv, &kv) in hi[0].iter_mut().zip(lo[j].iter()) {
                                    *sv += coef * kv;
                                }
                            }
                        }
                        for (r, &sv) in w.r.iter_mut().zip(w.k[i].iter()) {
                            *r += sv;
                        }
                    }
                    w.lu.solve(&mut w.r);
                    w.k[i].copy_from_slice(&w.r);
                }
                w.unew.copy_from_slice(un);
                for (i, &bi) in b.iter().enumerate() {
                    if bi != 0.0 {
                        let coef = T::from_f64(bi);
                        for (u, &kv) in w.unew.iter_mut().zip(w.k[i].iter()) {
                            *u += coef * kv;
                        }
                    }
                }
                let (Some(norm), Some(bt)) = (norm, btilde) else {
                    return RosOutcome::Done(None);
                };
                w.e.iter_mut().for_each(|v| *v = 0.0);
                for (i, &bti) in bt.iter().enumerate() {
                    if bti != 0.0 {
                        for (e, &kv) in w.e.iter_mut().zip(w.k[i].iter()) {
                            *e += bti * kv.to_f64();
                        }
                    }
                }
                RosOutcome::Done(Some(norm(&w.e, un, &w.unew)))
            }
            RosForm::Ode23s { d, c32 } => {
                let sig = T::from_f64(sigma);
                let gam = T::from_f64(h * d);
                match f0 {
                    Some(v) => w.f0.copy_from_slice(v),
                    None => {
                        f(&mut w.f0, un, t);
                        w.nf += 1;
                    }
                }
                // k1 = (M⁻¹ (f0 + γ f_t)) σ.
                w.r.copy_from_slice(&w.f0);
                if has_ft {
                    for (r, &x) in w.r.iter_mut().zip(w.ft.iter()) {
                        *r += gam * x;
                    }
                }
                w.lu.solve(&mut w.r);
                for (k, &x) in w.k[0].iter_mut().zip(w.r.iter()) {
                    *k = x * sig;
                }
                // U2 = u_n + (h/2) k1; f1 = f(U2, t + h/2).
                let dto2 = h / 2.0;
                {
                    let c = T::from_f64(dto2);
                    let (lo, hi) = w.y.split_at_mut(1);
                    for ((y, &u), &k1) in hi[0].iter_mut().zip(lo[0].iter()).zip(w.k[0].iter()) {
                        *y = u + c * k1;
                    }
                }
                f(&mut w.f1, &w.y[1], t + dto2);
                w.nf += 1;
                // k2 = (M⁻¹ (f1 − k1)) σ + k1.
                for ((r, &f1), &k1) in w.r.iter_mut().zip(w.f1.iter()).zip(w.k[0].iter()) {
                    *r = f1 - k1;
                }
                w.lu.solve(&mut w.r);
                {
                    let (lo, hi) = w.k.split_at_mut(1);
                    for ((k2, &x), &k1) in hi[0].iter_mut().zip(w.r.iter()).zip(lo[0].iter()) {
                        *k2 = x * sig + k1;
                    }
                }
                // u_{n+1} = u_n + h k2.
                let hc = T::from_f64(h);
                for ((u, &up), &k2) in w.unew.iter_mut().zip(un.iter()).zip(w.k[1].iter()) {
                    *u = up + hc * k2;
                }
                // f2 = f(u_{n+1}, t + h), the next step's f0 (§3.3).
                f(&mut w.fend, &w.unew, t + h);
                w.nf += 1;
                w.has_fend = true;
                let Some(norm) = norm else {
                    return RosOutcome::Done(None);
                };
                // k3 = (M⁻¹ (((f2 − c32 (k2 − f1)) − 2 (k1 − f0)) + γ f_t)) σ, with
                // γ = h d, which keeps the estimate third order; OrdinaryDiffEq adds h f_t,
                // which makes it second order when f depends on t (methods.md §4.3, D62).
                let c32 = T::from_f64(*c32);
                let two = T::from_f64(2.0);
                for j in 0..un.len() {
                    let mut v =
                        w.fend[j] - c32 * (w.k[1][j] - w.f1[j]) - two * (w.k[0][j] - w.f0[j]);
                    if has_ft {
                        v += gam * w.ft[j];
                    }
                    w.r[j] = v;
                }
                w.lu.solve(&mut w.r);
                let dto6 = T::from_f64(h / 6.0);
                for j in 0..un.len() {
                    let k3 = w.r[j] * sig;
                    let ej = dto6 * (w.k[0][j] - two * w.k[1][j] + k3);
                    w.e[j] = ej.to_f64();
                }
                RosOutcome::Done(Some(norm(&w.e, un, &w.unew)))
            }
        }
    }
}

/// Workspace and outputs of a Rosenbrock step.
#[derive(Debug, Clone)]
pub struct RosWork<T> {
    /// Stage states `U_i`; `y[0] = u_n`.
    pub y: Vec<Vec<T>>,
    /// Stage increments `k_i`.
    pub k: Vec<Vec<T>>,
    /// The current stage's `f` value.
    fs: Vec<T>,
    /// Rosenbrock23's `f_0` and `f_1`.
    f0: Vec<T>,
    f1: Vec<T>,
    /// Right-hand side of the current solve.
    r: Vec<T>,
    /// `f_t(u_n, t_n)`, when formed.
    ft: Vec<T>,
    /// `J(u_n, t_n)`.
    pub jac: MatrixStorage<T>,
    /// `σI − J`.
    w: MatrixStorage<T>,
    /// Its factorization.
    pub lu: LinearSolver<T>,
    /// `u_{n+1}`.
    pub unew: Vec<T>,
    /// `f(u_{n+1}, t_n + h)`, valid when `has_fend`.
    pub fend: Vec<T>,
    pub has_fend: bool,
    e: Vec<f64>,
    /// Right-hand-side evaluations made, cumulative.
    pub nf: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tableau::all_tableaus;

    // forced_decay: u' = −2u + sin 3t.
    fn forced(out: &mut [f64], u: &[f64], t: f64) {
        out[0] = -2.0 * u[0] + (3.0 * t).sin();
    }
    fn forced_jac(m: &mut MatrixStorage<f64>, _u: &[f64], _t: f64) {
        m.data_mut()[0] = -2.0;
    }
    fn forced_ft(out: &mut [f64], _u: &[f64], t: f64) {
        out[0] = 3.0 * (3.0 * t).cos();
    }
    fn exact(t: f64) -> f64 {
        let (k, w) = (2.0f64, 3.0f64);
        let a = |t: f64| (k * (w * t).sin() - w * (w * t).cos()) / (k * k + w * w);
        (1.0 - a(0.0)) * (-k * t).exp() + a(t)
    }

    fn run(m: &Rosenbrock, h: f64) -> f64 {
        let mut w = m
            .workspace::<f64>(1, Structure::Dense, LinsolveKind::DenseLu)
            .unwrap();
        let mut u = vec![1.0];
        let n = (2.0 / h).round() as usize;
        let mut cache: Option<Vec<f64>> = None;
        for i in 0..n {
            let t = i as f64 * h;
            let o = m.step(
                &mut w,
                &mut forced,
                &mut forced_jac,
                Some(&mut forced_ft),
                &u,
                t,
                h,
                cache.as_deref(),
                None,
            );
            assert_eq!(o, RosOutcome::Done(None));
            u.copy_from_slice(&w.unew);
            cache = w.has_fend.then(|| w.fend.clone());
        }
        (u[0] - exact(2.0)).abs()
    }

    #[test]
    fn every_rosenbrock_method_shows_its_order() {
        for t in all_tableaus().filter(|t| t.family == Family::Rosenbrock) {
            let m = Rosenbrock::from_tableau(t).unwrap();
            let (h1, h2) = if m.order == 2 {
                (0.01, 0.005)
            } else {
                (0.04, 0.02)
            };
            let e1 = run(&m, h1);
            let e2 = run(&m, h2);
            let p = (e1 / e2).log2();
            assert!(
                (p - m.order as f64).abs() < 0.3 || p > m.order as f64,
                "{}: observed {p}, expected {}",
                m.name,
                m.order
            );
        }
    }

    #[test]
    fn kpp_new_f_is_the_reuse_rule() {
        for t in all_tableaus().filter(|t| t.kpp.is_some()) {
            let m = Rosenbrock::from_tableau(t).unwrap();
            let RosForm::Transformed { reuse_f, .. } = &m.form else {
                panic!()
            };
            let new_f = &t.kpp.as_ref().unwrap().new_f;
            // From the third stage on, every stage KPP does not re-evaluate is
            // one this rule reuses. KPP_Rodas3's second stage (KPP reuses the
            // first stage's f there) and KPP_Rodas4's sixth (KPP evaluates
            // it) differ only in the evaluation count.
            for (i, (&r, &nf)) in reuse_f.iter().zip(new_f).enumerate().skip(2) {
                if !nf {
                    assert!(r, "{} stage {i}", t.name);
                }
            }
        }
        // ROS3P's third stage repeats its second (spec/methods.md §4.1 names
        // Rodas4, but Rodas4's last two rows of A differ: A_65 = 1).
        let rows = |name: &str| {
            let m = Rosenbrock::from_tableau(crate::tableau::tableau(name).unwrap()).unwrap();
            let RosForm::Transformed { reuse_f, .. } = m.form else {
                panic!()
            };
            reuse_f
        };
        assert_eq!(rows("ROS3P"), [false, false, true]);
        assert!(rows("Rodas4").iter().all(|r| !r));
    }

    #[test]
    fn error_estimates_scale() {
        for t in all_tableaus().filter(|t| t.family == Family::Rosenbrock) {
            let m = Rosenbrock::from_tableau(t).unwrap();
            let mut w = m
                .workspace::<f64>(1, Structure::Dense, LinsolveKind::DenseLu)
                .unwrap();
            let mut est = |h: f64| {
                let mut norm = |e: &[f64], _a: &[f64], _b: &[f64]| -> f64 { e[0].abs() };
                match m.step(
                    &mut w,
                    &mut forced,
                    &mut forced_jac,
                    Some(&mut forced_ft),
                    &[1.0],
                    0.3,
                    h,
                    None,
                    Some(&mut norm),
                ) {
                    RosOutcome::Done(Some(e)) => e,
                    o => panic!("{o:?}"),
                }
            };
            let e1 = est(0.02);
            let e2 = est(0.01);
            let ratio = (e1 / e2).log2();
            // The local error estimate of an order-p method is O(h^(p+1)),
            // Rosenbrock23's included on this non-autonomous problem because
            // its third stage adds h·d·f_t (spec/methods.md §4.3); with
            // OrdinaryDiffEq's h·f_t it would be O(h²).
            let eo = m.order.min(m.embedded_order.unwrap()) as f64 + 1.0;
            assert!(
                ratio > eo - 0.7,
                "{}: estimate order {ratio}, expected ~{eo}",
                m.name
            );
        }
    }

    #[test]
    fn singular_w_is_reported() {
        let m = Rosenbrock::from_tableau(crate::tableau::tableau("Rodas4").unwrap()).unwrap();
        let mut w = m
            .workspace::<f64>(1, Structure::Dense, LinsolveKind::DenseLu)
            .unwrap();
        // J = 1/(hγ) makes σ − J = 0.
        let h = 0.1;
        let mut jac = |mm: &mut MatrixStorage<f64>, _u: &[f64], _t: f64| {
            mm.data_mut()[0] = 1.0 / (h * 0.25);
        };
        let mut f = |o: &mut [f64], u: &[f64], _t: f64| o[0] = u[0];
        let o = m.step(&mut w, &mut f, &mut jac, None, &[1.0], 0.0, h, None, None);
        assert_eq!(o, RosOutcome::Singular);
        assert_eq!(w.nf, 0);
    }
}
