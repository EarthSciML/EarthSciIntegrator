//! Implicit methods (spec/methods.md §5–§6): SDIRK and ESDIRK
//! (BackwardEuler, TRBDF2, SDIRK2, Kvaerno3, Kvaerno4), the θ-method and
//! IMEX Euler, with the modified Newton iteration of §5.2 and their error
//! estimates (§5.5).
//!
//! **Stage equations** (§5.1), unknown `z_i = h K_i`:
//! `τ_i = u_n + Σ_{j<i} a_ij z_j` (ascending `j`, zero coefficients skipped,
//! each term `a_ij·z_j`), `z_i = h f(τ_i + γ z_i, t_n + c_i h)`,
//! `u_{n+1} = u_n + Σ_i b_i z_i`. An explicit first stage is `z_1 = h·f(u_n, t_n)`,
//! evaluated (not reused from the previous step's `z_s/h`).
//!
//! **Newton** (§5.2; OrdinaryDiffEqNonlinearSolve 2.9.9 `src/nlsolve.jl:105-247`,
//! `src/newton.jl:708-756`, `:874-894`). `J` is formed at `(u_n, t_n)` once per
//! attempt and `M = σI − J`, `σ = 1/(hγ)`, factored once (§5.3); `M` is the
//! exact negation of OrdinaryDiffEq's stored `W = J − I/(hγ)`. One iteration:
//! - `ustep = τ + γ·z` (the stage state at the current iterate);
//! - `r = (h·f(ustep, t_s) − z)·σ`, OrdinaryDiffEq's `_compute_rhs`;
//! - `Δz = M⁻¹ r`, which is exactly `−(W \ r)`, so `z ← z + Δz` equals
//!   OrdinaryDiffEq's `z .- dz` bit for bit;
//! - `δ = ‖Δz‖` in §7's norm scaled with `max(|u_n|, |ustep|)`.
//!
//! The convergence test follows `nlsolve!` exactly, which §5.2 step 5 misreads
//! (both points agreed with the Julia implementation, for the spec PR):
//! - `θ_1 = θ_prev`; for `m > 1`, `θ_m = max(0.3 θ_prev, δ_m/δ_{m−1})`, and
//!   `θ_prev ← θ_m`. `η = θ/(1 − θ)` on every iteration, including the first
//!   (the `η_old`-based initial `η` is overwritten before it is used, so `η_old`
//!   has no effect).
//! - The `η δ < κ` test is skipped on the first iteration of a fixed-step
//!   solve (`check_η_convergence`, `:224-235`), so there iteration 1 converges
//!   only when `δ_1 < 1e-5`.
//! - `θ_prev` is 1 at the start of every step attempt and carries across the
//!   stages of that attempt. OrdinaryDiffEq carries it across steps (resetting
//!   it after a rejection), which the tape cannot reproduce on replay; this
//!   keeps the step a function of `(u_n, h)` (§5.3).
//! - At the floating-point limit (`|θ − 1| ≤ 100√ε`) and on divergence
//!   (`θ > 2`), the loop stops *before* applying the iteration's `Δz`.
//!
//! **IMEX Euler and the θ-method** (§6) are one-stage instances: the θ-method
//! with `τ = u_n + (h(1−θ))·f(u_n, t_n)` (the term skipped at `θ = 1`), `γ = θ`,
//! the stage at `t_n + h`, `u_{n+1} = τ + θ·z`; IMEX Euler with
//! `τ = u_n + h·f_E(u_n, t_n)`, `γ = 1`, `f_I` at `t_n + h`, `u_{n+1} = τ + z`, and
//! `W` from `J_I`. IMEX Euler's `implicit_solve: "linearized"` is the
//! Rosenbrock–Euler step `(σI − J_I(u_n)) Δ = f_E(u_n) + f_I(u_n)`, `σ = 1/h`,
//! `u_{n+1} = u_n + Δ` (D43).

// Index loops mirror the stage formulas.
#![allow(clippy::needless_range_loop)]

use crate::linalg::{LinearSolver, LinsolveKind, MatrixStorage, Structure};
use crate::scalar::Scalar;
use crate::tableau::{ErrorEstimate, Family, Form, Tableau};
use crate::NormFn;

/// Which terms of the leaf a right-hand side or Jacobian call sums: all of
/// them, or one part of an IMEX partition (run-config §4.5). Outside IMEX
/// Euler the implicit part is every term.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    All,
    Explicit,
    Implicit,
}

/// `f(part, out, u, t)` overwrites `out` with the part's right-hand side.
pub type PartRhsFn<'a, T> = dyn FnMut(Part, &mut [T], &[T], f64) + 'a;
/// `jac(part, out, u, t)` overwrites `out` with the part's Jacobian.
pub type PartJacFn<'a, T> = dyn FnMut(Part, &mut MatrixStorage<T>, &[T], f64) + 'a;

/// The error estimate of an SDIRK method (§5.5).
#[derive(Debug, Clone)]
pub enum SdirkEstimate {
    /// Hosea–Shampine smoothing of `Σ b̃_i z_i`.
    Smoothed(Vec<f64>),
    /// BackwardEuler's divided difference, with its constant `7/12`.
    DividedDifference(f64),
}

/// The step form and its coefficients.
#[derive(Debug, Clone)]
pub enum ImplicitForm {
    Sdirk {
        a: Vec<Vec<f64>>,
        b: Vec<f64>,
        c: Vec<f64>,
        /// The diagonal `a_ii` of every implicit stage.
        gamma: f64,
        explicit_first: bool,
        estimate: SdirkEstimate,
    },
    Theta {
        theta: f64,
    },
    ImexEuler {
        /// `implicit_solve: "linearized"` (D43).
        linearized: bool,
        /// Whether the leaf has explicit terms.
        has_explicit: bool,
    },
}

/// How a Newton solve decides it has converged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NewtonMode {
    /// A fixed-step leaf: no `η` test on the first iteration.
    Fixed,
    /// An adaptive leaf.
    Adaptive,
    /// The frozen-tape replay of a finite-difference check (spec/adjoint.md
    /// §7.2): iterate to roundoff, so the stages are exact well below `ε²`.
    Tight,
}

/// What one step attempt produced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImplicitOutcome {
    /// `W` is singular (§4.2, §8.5); nothing else was computed.
    Singular,
    /// A Newton solve failed (§5.2 step 8).
    NewtonFailure,
    /// The step was computed; the scaled error when one was asked for.
    Done(Option<f64>),
}

/// An implicit method built from its table.
#[derive(Debug, Clone)]
pub struct Implicit {
    pub name: String,
    pub form: ImplicitForm,
    pub stages: usize,
    pub order: u32,
    pub embedded_order: Option<u32>,
}

/// OrdinaryDiffEqNonlinearSolve's `NLNewton` defaults (`src/type.jl:185-190`).
const KAPPA: f64 = 0.01;
const MAX_ITERS: usize = 10;
/// The tight mode's limits: an increment norm this small is roundoff.
const TIGHT_ITERS: usize = 100;
const TIGHT_DELTA: f64 = 1e-12;

impl Implicit {
    /// Builds the method from a table. `theta` is the θ-method's leaf option;
    /// `linearized` and `has_explicit` describe an IMEX Euler leaf.
    pub fn from_tableau(
        t: &Tableau,
        theta: Option<f64>,
        linearized: bool,
        has_explicit: bool,
    ) -> Result<Self, String> {
        let form = match (t.family, t.form) {
            (Family::Sdirk, Form::Sdirk) => {
                let a = t.a.clone().ok_or("missing A")?;
                let b = t.b.clone().ok_or("missing b")?;
                let c = t.c.clone().ok_or("missing c")?;
                let s = t.stages;
                if a.len() != s || b.len() != s || c.len() != s || a.iter().any(|r| r.len() != s) {
                    return Err(format!("{}: coefficient sizes differ from stages", t.name));
                }
                let explicit_first = t.explicit_first_stage.unwrap_or(false);
                let gamma = a[s - 1][s - 1];
                for i in 0..s {
                    let want = if i == 0 && explicit_first { 0.0 } else { gamma };
                    if a[i][i] != want || (i + 1..s).any(|j| a[i][j] != 0.0) {
                        return Err(format!(
                            "{}: not a singly diagonally implicit table",
                            t.name
                        ));
                    }
                }
                let estimate = match t.error_estimate {
                    ErrorEstimate::EmbeddedSmoothed => {
                        SdirkEstimate::Smoothed(t.btilde.clone().ok_or("missing btilde")?)
                    }
                    ErrorEstimate::DividedDifference => SdirkEstimate::DividedDifference(
                        t.divided_difference
                            .as_ref()
                            .ok_or("missing divided_difference")?
                            .constant,
                    ),
                    e => return Err(format!("{}: error estimate {e:?}", t.name)),
                };
                ImplicitForm::Sdirk {
                    a,
                    b,
                    c,
                    gamma,
                    explicit_first,
                    estimate,
                }
            }
            (Family::Theta, Form::Theta) => {
                let theta = theta.ok_or("the Theta method needs the leaf option theta")?;
                if !(0.0..=1.0).contains(&theta) {
                    return Err(format!("theta = {theta} is outside [0, 1]"));
                }
                ImplicitForm::Theta { theta }
            }
            (Family::Imex, Form::ImexEuler) => ImplicitForm::ImexEuler {
                linearized,
                has_explicit,
            },
            (f, fm) => {
                return Err(format!(
                    "{}: {f:?}/{fm:?} is not an implicit method",
                    t.name
                ))
            }
        };
        // The θ-method is second order at θ = 1/2 (its table's note).
        let order = match &form {
            ImplicitForm::Theta { theta } if *theta == 0.5 => 2,
            _ => t.order,
        };
        Ok(Implicit {
            name: t.name.clone(),
            form,
            stages: t.stages,
            order,
            embedded_order: t.embedded_order,
        })
    }

    /// Whether the method has an error estimate.
    pub fn adaptive(&self) -> bool {
        matches!(self.form, ImplicitForm::Sdirk { .. })
    }

    /// Whether this is IMEX Euler.
    pub fn is_imex(&self) -> bool {
        matches!(self.form, ImplicitForm::ImexEuler { .. })
    }

    /// The `γ` of the step's `W = I − hγJ`, or `None` when there is no solve
    /// (the θ-method at θ = 0).
    pub fn w_gamma(&self) -> Option<f64> {
        match &self.form {
            ImplicitForm::Sdirk { gamma, .. } => Some(*gamma),
            ImplicitForm::Theta { theta } => (*theta != 0.0).then_some(*theta),
            ImplicitForm::ImexEuler { .. } => Some(1.0),
        }
    }

    /// The terms `W`'s Jacobian covers.
    pub fn w_part(&self) -> Part {
        if self.is_imex() {
            Part::Implicit
        } else {
            Part::All
        }
    }

    /// A workspace for a state of length `n` whose matrices have `structure`.
    pub fn workspace<T: Scalar>(
        &self,
        n: usize,
        structure: Structure,
        linsolve: LinsolveKind,
    ) -> Result<ImplicitWork<T>, String> {
        let s = self.stages.max(1);
        let v = || vec![T::zero(); n];
        Ok(ImplicitWork {
            y: vec![v(); s.max(2)],
            z: vec![v(); s],
            zc: v(),
            tau: v(),
            fs: v(),
            r: v(),
            ustep: v(),
            e: vec![0.0; n],
            jac: MatrixStorage::zeros(structure, n),
            w: MatrixStorage::zeros(structure, n),
            lu: LinearSolver::new(linsolve, structure, n)?,
            unew: v(),
            fend: v(),
            prev_u: v(),
            prev_t: 0.0,
            has_prev: false,
            nf: 0,
            newton_iters: 0,
            newton_solves: 0,
        })
    }

    /// Forms `J` of `part` at `(u, t)` and factors `M = σI − J` with
    /// `σ = 1/(hγ)` into `w.lu`. Returns `σ`, or `None` when `M` is singular
    /// or the method has no solve. The forward step factors at `(u_n, t_n)`;
    /// the adjoint refactors at each converged stage (spec/adjoint.md §4.3, D42).
    pub fn factor<T: Scalar>(
        &self,
        w: &mut ImplicitWork<T>,
        jac: &mut PartJacFn<'_, T>,
        part: Part,
        u: &[T],
        t: f64,
        h: f64,
    ) -> Option<f64> {
        let sigma = 1.0 / (h * self.w_gamma()?);
        jac(part, &mut w.jac, u, t);
        w.w.shifted_negation(&w.jac, T::from_f64(sigma));
        w.lu.factor(&w.w).then_some(sigma)
    }

    /// Takes one step of size `h` from `(un, t)`.
    ///
    /// On return `w.unew` is `u_{n+1}`; `w.y` holds the stage states the
    /// adjoint linearizes at (SDIRK: `Z_i = τ_i + γ z_i`, with `Z_1 = u_n` for
    /// an explicit first stage; θ-method and IMEX Euler: `[u_n, u_{n+1}]`)
    /// and `w.z` the stage unknowns (linearized IMEX Euler: `[Δ]`).
    /// `norm(e, u_a, u_b)` is §7's norm; the error is formed when `want_err`.
    #[allow(clippy::too_many_arguments)]
    pub fn step<T: Scalar>(
        &self,
        w: &mut ImplicitWork<T>,
        f: &mut PartRhsFn<'_, T>,
        jac: &mut PartJacFn<'_, T>,
        un: &[T],
        t: f64,
        h: f64,
        mode: NewtonMode,
        want_err: bool,
        norm: &mut NormFn<'_, T>,
    ) -> ImplicitOutcome {
        let n = un.len();
        let hh = T::from_f64(h);
        // θ_prev is 1 at the start of every attempt (see the module docs).
        let mut theta_prev = 1.0f64;
        match &self.form {
            ImplicitForm::Sdirk {
                a,
                b,
                c,
                gamma,
                explicit_first,
                estimate,
            } => {
                let Some(sigma) = self.factor(w, jac, Part::All, un, t, h) else {
                    return ImplicitOutcome::Singular;
                };
                let g = T::from_f64(*gamma);
                let s = self.stages;
                let mut first_implicit = true;
                for i in 0..s {
                    if i == 0 && *explicit_first {
                        // z_1 = h·f(u_n, t_n), evaluated (§5.1).
                        f(Part::All, &mut w.fs, un, t);
                        w.nf += 1;
                        for j in 0..n {
                            w.z[0][j] = hh * w.fs[j];
                        }
                        w.y[0].copy_from_slice(un);
                        continue;
                    }
                    w.tau.copy_from_slice(un);
                    for jj in 0..i {
                        if a[i][jj] != 0.0 {
                            let coef = T::from_f64(a[i][jj]);
                            for j in 0..n {
                                w.tau[j] += coef * w.z[jj][j];
                            }
                        }
                    }
                    // Initial guess: zero for the step's first implicit stage,
                    // else the previous stage's z (§5.2 step 1).
                    if first_implicit {
                        w.zc.iter_mut().for_each(|x| *x = T::zero());
                    } else {
                        let (lo, _) = w.z.split_at(i);
                        w.zc.copy_from_slice(&lo[i - 1]);
                    }
                    first_implicit = false;
                    let ts = t + c[i] * h;
                    if !newton(
                        w,
                        f,
                        Part::All,
                        norm,
                        un,
                        *gamma,
                        sigma,
                        h,
                        ts,
                        mode,
                        &mut theta_prev,
                    ) {
                        return ImplicitOutcome::NewtonFailure;
                    }
                    w.z[i].copy_from_slice(&w.zc);
                    for j in 0..n {
                        w.y[i][j] = w.tau[j] + g * w.zc[j];
                    }
                }
                w.unew.copy_from_slice(un);
                for (i, &bi) in b.iter().enumerate() {
                    if bi != 0.0 {
                        let coef = T::from_f64(bi);
                        for j in 0..n {
                            w.unew[j] += coef * w.z[i][j];
                        }
                    }
                }
                if !want_err {
                    return ImplicitOutcome::Done(None);
                }
                match estimate {
                    SdirkEstimate::Smoothed(bt) => {
                        // e = (I − hγJ)⁻¹ Σ b̃_i z_i = M⁻¹((Σ b̃_i z_i)·σ) (§5.5, D56).
                        w.r.iter_mut().for_each(|x| *x = T::zero());
                        for (i, &bti) in bt.iter().enumerate() {
                            if bti != 0.0 {
                                let coef = T::from_f64(bti);
                                for j in 0..n {
                                    w.r[j] += coef * w.z[i][j];
                                }
                            }
                        }
                        let sg = T::from_f64(sigma);
                        w.r.iter_mut().for_each(|x| *x *= sg);
                        w.lu.solve(&mut w.r);
                        for j in 0..n {
                            w.e[j] = w.r[j].to_f64();
                        }
                        ImplicitOutcome::Done(Some(norm(&w.e, un, &w.unew)))
                    }
                    SdirkEstimate::DividedDifference(cc) => {
                        if !w.has_prev {
                            // No previous step in this invocation: E = 1 (§5.5).
                            return ImplicitOutcome::Done(Some(1.0));
                        }
                        // OrdinaryDiffEqSDIRK src/generic_imex_perform_step.jl:2401-2420,
                        // with t + dt for t_{n+1} as OrdinaryDiffEq writes it.
                        let tprev = w.prev_t;
                        let dt1 = h * ((t + h) - tprev);
                        let dt2 = (t - tprev) * ((t + h) - tprev);
                        let rr = cc * (h * h);
                        for j in 0..n {
                            let d1 = (w.unew[j].to_f64() - un[j].to_f64()) / dt1;
                            let d2 = (un[j].to_f64() - w.prev_u[j].to_f64()) / dt2;
                            w.e[j] = rr * (d1 - d2).abs();
                        }
                        ImplicitOutcome::Done(Some(norm(&w.e, un, &w.unew)))
                    }
                }
            }
            ImplicitForm::Theta { theta } => {
                let th = *theta;
                if th == 0.0 {
                    // Forward Euler: no solve.
                    f(Part::All, &mut w.fs, un, t);
                    w.nf += 1;
                    for j in 0..n {
                        w.unew[j] = un[j] + hh * w.fs[j];
                    }
                    w.y[0].copy_from_slice(un);
                    w.y[1].copy_from_slice(&w.unew);
                    return ImplicitOutcome::Done(None);
                }
                let Some(sigma) = self.factor(w, jac, Part::All, un, t, h) else {
                    return ImplicitOutcome::Singular;
                };
                w.tau.copy_from_slice(un);
                if th != 1.0 {
                    f(Part::All, &mut w.fs, un, t);
                    w.nf += 1;
                    let coef = T::from_f64(h * (1.0 - th));
                    for j in 0..n {
                        w.tau[j] += coef * w.fs[j];
                    }
                }
                w.zc.iter_mut().for_each(|x| *x = T::zero());
                if !newton(
                    w,
                    f,
                    Part::All,
                    norm,
                    un,
                    th,
                    sigma,
                    h,
                    t + h,
                    mode,
                    &mut theta_prev,
                ) {
                    return ImplicitOutcome::NewtonFailure;
                }
                let g = T::from_f64(th);
                for j in 0..n {
                    w.unew[j] = w.tau[j] + g * w.zc[j];
                }
                w.z[0].copy_from_slice(&w.zc);
                w.y[0].copy_from_slice(un);
                w.y[1].copy_from_slice(&w.unew);
                ImplicitOutcome::Done(None)
            }
            ImplicitForm::ImexEuler {
                linearized,
                has_explicit,
            } => {
                let Some(sigma) = self.factor(w, jac, Part::Implicit, un, t, h) else {
                    return ImplicitOutcome::Singular;
                };
                w.y[0].copy_from_slice(un);
                if *linearized {
                    // (σI − J_I(u_n)) Δ = f_E(u_n) + f_I(u_n), σ = 1/h (D43),
                    // with each part summed on its own and f_E first, as the
                    // Julia implementation forms it (not the leaf's operator
                    // order, which can interleave the parts).
                    if *has_explicit {
                        f(Part::Explicit, &mut w.r, un, t);
                        f(Part::Implicit, &mut w.fs, un, t);
                        w.nf += 2;
                        for j in 0..n {
                            w.r[j] += w.fs[j];
                        }
                    } else {
                        f(Part::Implicit, &mut w.r, un, t);
                        w.nf += 1;
                    }
                    w.lu.solve(&mut w.r);
                    for j in 0..n {
                        w.unew[j] = un[j] + w.r[j];
                    }
                    w.z[0].copy_from_slice(&w.r);
                    w.y[1].copy_from_slice(&w.unew);
                    return ImplicitOutcome::Done(None);
                }
                w.tau.copy_from_slice(un);
                if *has_explicit {
                    f(Part::Explicit, &mut w.fs, un, t);
                    w.nf += 1;
                    for j in 0..n {
                        w.tau[j] += hh * w.fs[j];
                    }
                }
                w.zc.iter_mut().for_each(|x| *x = T::zero());
                if !newton(
                    w,
                    f,
                    Part::Implicit,
                    norm,
                    un,
                    1.0,
                    sigma,
                    h,
                    t + h,
                    mode,
                    &mut theta_prev,
                ) {
                    return ImplicitOutcome::NewtonFailure;
                }
                for j in 0..n {
                    w.unew[j] = w.tau[j] + w.zc[j];
                }
                w.z[0].copy_from_slice(&w.zc);
                w.y[1].copy_from_slice(&w.unew);
                ImplicitOutcome::Done(None)
            }
        }
    }
}

/// One stage's Newton solve (§5.2) of `z = h f(τ + γ z, ts)` from the guess in
/// `w.zc`, with `M = σI − J` factored in `w.lu`. The solution is left in
/// `w.zc`; returns whether it converged.
#[allow(clippy::too_many_arguments)]
fn newton<T: Scalar>(
    w: &mut ImplicitWork<T>,
    f: &mut PartRhsFn<'_, T>,
    part: Part,
    norm: &mut NormFn<'_, T>,
    un: &[T],
    gamma: f64,
    sigma: f64,
    h: f64,
    ts: f64,
    mode: NewtonMode,
    theta_prev: &mut f64,
) -> bool {
    let n = un.len();
    let (g, hh, sg) = (T::from_f64(gamma), T::from_f64(h), T::from_f64(sigma));
    let tight = mode == NewtonMode::Tight;
    let maxiters = if tight { TIGHT_ITERS } else { MAX_ITERS };
    // nlsolve.jl's eps_around_one.
    let fp_limit = 100.0 * f64::EPSILON.sqrt();
    w.newton_solves += 1;
    let mut ndzprev = 1.0f64;
    for iter in 1..=maxiters {
        for j in 0..n {
            w.ustep[j] = w.tau[j] + g * w.zc[j];
        }
        f(part, &mut w.fs, &w.ustep, ts);
        w.nf += 1;
        w.newton_iters += 1;
        for j in 0..n {
            w.r[j] = (hh * w.fs[j] - w.zc[j]) * sg;
        }
        w.lu.solve(&mut w.r);
        for j in 0..n {
            w.e[j] = w.r[j].to_f64();
        }
        let ndz = norm(&w.e, un, &w.ustep);
        if !ndz.is_finite() {
            return false;
        }
        if tight {
            // Stop at roundoff: when the increment no longer shrinks (keeping
            // the previous iterate) or is negligible.
            if iter > 1 && ndz >= ndzprev {
                return ndzprev < 1e-6;
            }
            for j in 0..n {
                w.zc[j] += w.r[j];
            }
            if ndz < TIGHT_DELTA {
                return true;
            }
            ndzprev = ndz;
            continue;
        }
        let theta = if iter > 1 {
            let th = (0.3 * *theta_prev).max(ndz / ndzprev);
            if (th - 1.0).abs() <= fp_limit {
                // The floating-point limit: converged or diverged by δ alone,
                // without applying this iteration's increment.
                return ndz <= 1.0;
            }
            if th > 2.0 {
                return false;
            }
            th
        } else {
            *theta_prev
        };
        *theta_prev = theta;
        for j in 0..n {
            w.zc[j] += w.r[j];
        }
        let eta = theta / (1.0 - theta);
        let check_eta = iter > 1 || mode == NewtonMode::Adaptive;
        if (iter == 1 && ndz < 1e-5) || (check_eta && eta >= 0.0 && eta * ndz < KAPPA) {
            return true;
        }
        ndzprev = ndz;
    }
    false
}

/// Workspace and outputs of an implicit step.
#[derive(Debug, Clone)]
pub struct ImplicitWork<T> {
    /// Stage states (see [`Implicit::step`]).
    pub y: Vec<Vec<T>>,
    /// Stage unknowns `z_i`.
    pub z: Vec<Vec<T>>,
    /// The current Newton iterate.
    zc: Vec<T>,
    tau: Vec<T>,
    fs: Vec<T>,
    r: Vec<T>,
    ustep: Vec<T>,
    e: Vec<f64>,
    /// `J` at the last factorization.
    pub jac: MatrixStorage<T>,
    /// `σI − J`.
    w: MatrixStorage<T>,
    /// Its factorization.
    pub lu: LinearSolver<T>,
    /// `u_{n+1}`.
    pub unew: Vec<T>,
    /// Unused: no implicit method is first same as last.
    pub fend: Vec<T>,
    /// The previous accepted step's start `(u_{n−1}, t_{n−1})` in this
    /// invocation, for BackwardEuler's divided difference (§5.5).
    pub prev_u: Vec<T>,
    pub prev_t: f64,
    pub has_prev: bool,
    /// Right-hand-side evaluations made, cumulative.
    pub nf: usize,
    /// Newton iterations and Newton solves, cumulative (statistics only).
    pub newton_iters: usize,
    pub newton_solves: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tableau::{all_tableaus, tableau};

    // forced_decay: u' = −2u + sin 3t.
    fn forced(_p: Part, out: &mut [f64], u: &[f64], t: f64) {
        out[0] = -2.0 * u[0] + (3.0 * t).sin();
    }
    fn forced_jac(_p: Part, m: &mut MatrixStorage<f64>, _u: &[f64], _t: f64) {
        m.data_mut()[0] = -2.0;
    }
    fn exact(t: f64) -> f64 {
        let (k, w) = (2.0f64, 3.0f64);
        let a = |t: f64| (k * (w * t).sin() - w * (w * t).cos()) / (k * k + w * w);
        (1.0 - a(0.0)) * (-k * t).exp() + a(t)
    }
    fn norm(e: &[f64], a: &[f64], b: &[f64]) -> f64 {
        e[0].abs() / (1e-8 + 1e-6 * a[0].abs().max(b[0].abs()))
    }

    fn run(m: &Implicit, h: f64) -> f64 {
        let mut w = m
            .workspace::<f64>(1, Structure::Dense, LinsolveKind::DenseLu)
            .unwrap();
        let mut u = vec![1.0];
        let n = (2.0 / h).round() as usize;
        for i in 0..n {
            let t = i as f64 * h;
            let o = m.step(
                &mut w,
                &mut forced,
                &mut forced_jac,
                &u,
                t,
                h,
                NewtonMode::Fixed,
                false,
                &mut norm,
            );
            assert_eq!(o, ImplicitOutcome::Done(None));
            u.copy_from_slice(&w.unew);
        }
        (u[0] - exact(2.0)).abs()
    }

    #[test]
    fn every_implicit_method_shows_its_order() {
        for t in all_tableaus().filter(|t| matches!(t.family, Family::Sdirk | Family::Imex)) {
            let m = Implicit::from_tableau(t, None, false, false).unwrap();
            let (e1, e2) = (run(&m, 0.02), run(&m, 0.01));
            let p = (e1 / e2).log2();
            assert!(
                p > m.order as f64 - 0.3,
                "{}: observed {p}, expected {}",
                m.name,
                m.order
            );
        }
        for (th, order) in [(0.5, 2.0), (1.0, 1.0), (0.0, 1.0), (0.7, 1.0)] {
            let m =
                Implicit::from_tableau(tableau("Theta").unwrap(), Some(th), false, false).unwrap();
            let p = (run(&m, 0.02) / run(&m, 0.01)).log2();
            assert!((p - order).abs() < 0.2, "theta {th}: observed {p}");
        }
    }

    #[test]
    fn smoothed_estimate_is_the_embedded_one_when_j_is_zero() {
        // f = t³: J = 0, so smoothing changes nothing and e = Σ b̃_i z_i
        // (spec/methods.md §5.5's check of OrdinaryDiffEq's factor hγ).
        let m = Implicit::from_tableau(tableau("TRBDF2").unwrap(), None, false, false).unwrap();
        let mut w = m
            .workspace::<f64>(1, Structure::Dense, LinsolveKind::DenseLu)
            .unwrap();
        let mut f = |_p: Part, o: &mut [f64], _u: &[f64], t: f64| o[0] = t * t * t;
        let mut j =
            |_p: Part, mm: &mut MatrixStorage<f64>, _u: &[f64], _t: f64| mm.data_mut()[0] = 0.0;
        let mut nrm = |e: &[f64], _a: &[f64], _b: &[f64]| e[0];
        let h = 0.1;
        let o = m.step(
            &mut w,
            &mut f,
            &mut j,
            &[0.0],
            1.0,
            h,
            NewtonMode::Adaptive,
            true,
            &mut nrm,
        );
        let ImplicitForm::Sdirk {
            estimate: SdirkEstimate::Smoothed(bt),
            ..
        } = &m.form
        else {
            panic!()
        };
        let raw: f64 = (0..3).map(|i| bt[i] * w.z[i][0]).sum();
        let ImplicitOutcome::Done(Some(e)) = o else {
            panic!("{o:?}")
        };
        assert!(
            (e - raw).abs() <= 1e-15 * raw.abs().max(1e-300),
            "{e} vs {raw}"
        );
    }

    #[test]
    fn newton_failure_and_singular_w() {
        let m =
            Implicit::from_tableau(tableau("BackwardEuler").unwrap(), None, false, false).unwrap();
        let mut w = m
            .workspace::<f64>(1, Structure::Dense, LinsolveKind::DenseLu)
            .unwrap();
        let h = 0.5;
        // J = 1/h makes σ − J = 0.
        let mut jac =
            |_p: Part, mm: &mut MatrixStorage<f64>, _u: &[f64], _t: f64| mm.data_mut()[0] = 1.0 / h;
        let mut f = |_p: Part, o: &mut [f64], u: &[f64], _t: f64| o[0] = u[0];
        let mut nrm = |e: &[f64], _a: &[f64], _b: &[f64]| e[0].abs();
        let o = m.step(
            &mut w,
            &mut f,
            &mut jac,
            &[1.0],
            0.0,
            h,
            NewtonMode::Fixed,
            false,
            &mut nrm,
        );
        assert_eq!(o, ImplicitOutcome::Singular);
        // u' = u³ from u = 10 with h = 1 and J = 0: Newton is a fixed-point
        // iteration that diverges.
        let mut jac0 =
            |_p: Part, mm: &mut MatrixStorage<f64>, _u: &[f64], _t: f64| mm.data_mut()[0] = 0.0;
        let mut f3 = |_p: Part, o: &mut [f64], u: &[f64], _t: f64| o[0] = u[0] * u[0] * u[0];
        let o = m.step(
            &mut w,
            &mut f3,
            &mut jac0,
            &[10.0],
            0.0,
            1.0,
            NewtonMode::Fixed,
            false,
            &mut nrm,
        );
        assert_eq!(o, ImplicitOutcome::NewtonFailure);
    }

    #[test]
    fn kvaerno_stage_times_are_row_sums() {
        for name in ["TRBDF2", "SDIRK2", "Kvaerno3", "Kvaerno4", "BackwardEuler"] {
            let t = tableau(name).unwrap();
            let (a, c) = (t.a.as_ref().unwrap(), t.c.as_ref().unwrap());
            for (row, ci) in a.iter().zip(c) {
                let s: f64 = row.iter().sum();
                assert!((s - ci).abs() < 1e-9, "{name}");
            }
        }
    }
}
