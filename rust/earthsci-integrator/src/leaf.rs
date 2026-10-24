//! A leaf: advances the sum of its operators over one interval with one
//! method (spec/run-config.md §4.2, spec/methods.md).
//!
//! The method is a [`LeafMethod`]. M1 implements the explicit Runge–Kutta
//! and Rosenbrock families; SDIRK, IMEX Euler and θ-method leaves are
//! refused when the schedule is built and join as further variants, sharing
//! the step loop below (landing on stops, control, domain, return codes).

use crate::operator::{Call, MatrixStorage, MuPair, Operator, Structure};
use crate::tape::{LeafPersist, Retcode};
use crate::{Error, Scalar};
use earthsci_integrator_solvers::controller::{Controller, Decision};
use earthsci_integrator_solvers::erk::{ErkForm, ErkWork, ExplicitRk};
use earthsci_integrator_solvers::initdt::{initial_dt, InitDtInput};
use earthsci_integrator_solvers::linalg::LinsolveKind;
use earthsci_integrator_solvers::norm::{error_norm, Tolerances};
use earthsci_integrator_solvers::rosenbrock::{RosForm, RosOutcome, RosWork, Rosenbrock};
use earthsci_integrator_solvers::scalar::eps_f64;
use earthsci_integrator_solvers::{NormFn, RhsFn};
use std::sync::{Arc, Mutex};

/// The leaf's method.
#[derive(Debug, Clone)]
pub enum LeafMethod {
    /// Explicit Runge–Kutta (spec/methods.md §3).
    Erk(ExplicitRk),
    /// Rosenbrock (spec/methods.md §4), with the leaf's linear solver.
    Ros(Rosenbrock, LinsolveKind),
}

impl LeafMethod {
    /// Operator calls the method's forward step needs from an operator
    /// (operator-interface.md §3): `ft` only from operators that depend on
    /// `t`, and only when the method reads `f_t`.
    pub fn forward_calls(&self, op_depends_on_t: bool) -> Vec<Call> {
        match self {
            LeafMethod::Erk(_) => vec![Call::Rhs],
            LeafMethod::Ros(m, _) => {
                let mut v = vec![Call::Rhs, Call::Jac];
                if m.uses_ft() && op_depends_on_t {
                    v.push(Call::Ft);
                }
                v
            }
        }
    }
    /// Operator calls the method's adjoint step needs from an operator
    /// (spec/adjoint.md §3): Rosenbrock adds `hess_u` (unless the operator
    /// is linear in `u`, where it is zero), `hess_p`, and the `ft_vjp_*`
    /// of operators that depend on `t`.
    pub fn adjoint_calls(&self, wrt_p: bool, op_depends_on_t: bool, op_linear: bool) -> Vec<Call> {
        let mut v = vec![Call::VjpU];
        if wrt_p {
            v.push(Call::VjpP);
        }
        if let LeafMethod::Ros(m, _) = self {
            if !op_linear {
                v.push(Call::HessU);
            }
            if wrt_p {
                v.push(Call::HessP);
            }
            if m.uses_ft() && op_depends_on_t {
                v.push(Call::FtVjpU);
                if wrt_p {
                    v.push(Call::FtVjpP);
                }
            }
        }
        v
    }
    /// Whether the method has an error estimate.
    pub fn adaptive(&self) -> bool {
        match self {
            LeafMethod::Erk(m) => m.adaptive(),
            LeafMethod::Ros(m, _) => m.adaptive(),
        }
    }
    /// The method's order (for the first step).
    pub fn order(&self) -> u32 {
        match self {
            LeafMethod::Erk(m) => m.order,
            LeafMethod::Ros(m, _) => m.order,
        }
    }
}

/// How the leaf chooses its substeps (spec/run-config.md §4.2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DtMode {
    /// Adaptive under `control`.
    Adaptive,
    /// Fixed substeps of this size, the last shortened to land on the end.
    Fixed(f64),
    /// One step of the node's Δ.
    Step,
    /// One step of the time since the leaf last ran (Δ without `every`).
    Accumulated,
}

/// Which step a `bind_dt` binding gives (spec/run-config.md §4.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindKind {
    Leaf,
    Node,
    Macro,
}

/// `domain` of a leaf (spec/methods.md §11).
#[derive(Debug, Clone)]
pub struct DomainSpec {
    /// Listed DOFs, ascending.
    pub dofs: Vec<usize>,
    /// `on_violation: "clip"`; otherwise `reject`.
    pub clip: bool,
    /// `check_stages` (Rosenbrock only): stage states are checked too.
    pub check_stages: bool,
}

/// A built leaf.
pub struct LeafSpec<T: Scalar> {
    /// Node id.
    pub id: String,
    pub ops: Vec<Arc<dyn Operator<T>>>,
    /// The leaf's DOFs: the union of its operators' writes, ascending.
    pub dofs: Vec<usize>,
    pub structure: Structure,
    pub alg: String,
    pub method: LeafMethod,
    pub dt_mode: DtMode,
    pub tol: Tolerances,
    pub controller: Option<Controller>,
    pub dt0: Option<f64>,
    pub dtmin: f64,
    pub dtmax: Option<f64>,
    pub maxiters: Option<u64>,
    pub domain: Option<DomainSpec>,
    /// `(parameter index, binding)`.
    pub binds: Vec<(usize, BindKind)>,
    /// The integrand `ψ(u) = Σ_{j ∈ dofs} w·u_j` of a functional over
    /// `time`, as `(dofs, w)`, when this leaf integrates its auxiliary state
    /// (spec/adjoint.md §2.1, D26).
    pub integrand: Option<(Vec<usize>, f64)>,
    /// Workspaces kept across invocations, one per invocation in flight.
    pub work: Mutex<Vec<LeafWork<T>>>,
}

/// A leaf's step workspace: the method's, and the stored
/// first-same-as-last value (spec/methods.md §3.3).
pub struct LeafWork<T> {
    m: MethodWork<T>,
    fsal: Vec<T>,
}

impl<T> LeafWork<T> {
    /// The Rosenbrock workspace, for a Rosenbrock leaf.
    pub fn ros(&mut self) -> Option<&mut RosWork<T>> {
        match &mut self.m {
            MethodWork::Ros(w) => Some(w),
            MethodWork::Erk(_) => None,
        }
    }
}

/// The method's workspace.
pub enum MethodWork<T> {
    Erk(ErkWork<T>),
    Ros(Box<RosWork<T>>),
}

impl<T: Scalar> MethodWork<T> {
    fn unew(&self) -> &[T] {
        match self {
            MethodWork::Erk(w) => &w.unew,
            MethodWork::Ros(w) => &w.unew,
        }
    }
    fn unew_mut(&mut self) -> &mut Vec<T> {
        match self {
            MethodWork::Erk(w) => &mut w.unew,
            MethodWork::Ros(w) => &mut w.unew,
        }
    }
    fn has_fend(&self) -> bool {
        match self {
            MethodWork::Erk(w) => w.has_fend,
            MethodWork::Ros(w) => w.has_fend,
        }
    }
    fn fend_mut(&mut self) -> &mut Vec<T> {
        match self {
            MethodWork::Erk(w) => &mut w.fend,
            MethodWork::Ros(w) => &mut w.fend,
        }
    }
    /// Stage states.
    fn y(&self) -> &[Vec<T>] {
        match self {
            MethodWork::Erk(w) => &w.y,
            MethodWork::Ros(w) => &w.y,
        }
    }
    /// Rosenbrock stage increments (none for explicit methods).
    fn k(&self) -> &[Vec<T>] {
        match self {
            MethodWork::Erk(_) => &[],
            MethodWork::Ros(w) => &w.k,
        }
    }
    fn nf(&self) -> usize {
        match self {
            MethodWork::Erk(w) => w.nf,
            MethodWork::Ros(w) => w.nf,
        }
    }
    fn reset_nf(&mut self) {
        match self {
            MethodWork::Erk(w) => w.nf = 0,
            MethodWork::Ros(w) => w.nf = 0,
        }
    }
}

/// What one step attempt produced, for the step loop.
struct Attempt {
    /// The scaled error, when wanted.
    e: Option<f64>,
    /// The time the first stage's `f` was evaluated at.
    tk1: f64,
    /// `W` was singular (spec/methods.md §4.2); nothing else was computed.
    singular: bool,
}

/// One accepted step as the adjoint needs it.
#[derive(Debug, Clone)]
pub struct StepData<T> {
    pub t: f64,
    pub h: f64,
    /// Parameters the step's evaluations saw (bound parameters set).
    pub p: Vec<f64>,
    /// Stage states: Butcher `U_i`; Shu–Osher `u^(j)`, `j = 0..s`;
    /// Rosenbrock `U_i` (`U_1 = u_n`).
    pub y: Vec<Vec<T>>,
    /// Rosenbrock stage increments `k_i` (empty for explicit methods).
    pub k: Vec<Vec<T>>,
    /// The DOFs the clip after the step zeroed, if it clipped.
    pub clip: Option<Vec<usize>>,
    /// The time the first stage was evaluated at: `t_n`, or `t_{n−1} + h_{n−1}`
    /// when it was the stored first-same-as-last value (spec/methods.md §3.3).
    pub tk1: f64,
    /// A stop step shorter than eps(|t|), accepted without computing it
    /// (spec/methods.md §10): the identity.
    pub skipped: bool,
}

/// The recorded steps and clip masks of a leaf invocation.
pub type Replay<'a> = (&'a [[f64; 2]], &'a [(usize, Vec<usize>)]);

/// Arguments of one invocation.
pub struct LeafCall<'a> {
    pub p: &'a [f64],
    pub t0: f64,
    pub t1: f64,
    /// The node's Δ.
    pub delta: f64,
    /// The macro step.
    pub macro_dt: f64,
    /// Interior stops in `(t0, t1)`, ascending (output times).
    pub stops: &'a [f64],
    /// Replay: the recorded steps and clip masks; the controller is off.
    pub replay: Option<Replay<'a>>,
    /// Keep the stage values of every accepted step for the adjoint.
    pub record_stages: bool,
}

/// What one invocation did.
#[derive(Debug, Clone)]
pub struct LeafOutcome<T> {
    pub steps: Vec<[f64; 2]>,
    pub accepted: u64,
    pub rejected: u64,
    pub retcode: Retcode,
    pub clips: Vec<(usize, Vec<usize>)>,
    pub rhs_calls: u64,
    /// The increment of the integrated auxiliary state `q` over the
    /// invocation (zero unless the leaf has an integrand).
    pub integral: f64,
    pub stage_data: Vec<StepData<T>>,
    /// The state on landing at each interior stop.
    pub stop_states: Vec<(f64, Vec<T>)>,
}

impl<T: Scalar> LeafSpec<T> {
    /// A fresh persistent state.
    pub fn initial_persist(&self) -> LeafPersist {
        LeafPersist::new(self.controller.as_ref().map(|c| c.initial_state()))
    }

    /// Parameters with the bound ones set for a step of size `h`: `p` itself
    /// when the leaf binds nothing, else `buf` (a copy of `p`) with them set.
    fn params<'b>(
        &self,
        p: &'b [f64],
        buf: &'b mut [f64],
        h: f64,
        call: &LeafCall<'_>,
    ) -> &'b [f64] {
        if self.binds.is_empty() {
            return p;
        }
        for &(i, k) in &self.binds {
            buf[i] = match k {
                BindKind::Leaf => h,
                BindKind::Node => call.delta,
                BindKind::Macro => call.macro_dt,
            };
        }
        buf
    }

    /// `out = Σ_k f_k(u, p, t)`, operators in declared order (operator-interface.md §3).
    pub fn rhs(&self, out: &mut [T], u: &[T], p: &[f64], t: f64) {
        out.iter_mut().for_each(|x| *x = T::zero());
        for op in &self.ops {
            op.rhs(out, u, p, t);
        }
    }

    /// `out = Σ_k J_k(u)ᵀ w`.
    pub fn vjp_u(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, w: &[T]) {
        out.iter_mut().for_each(|x| *x = T::zero());
        for op in &self.ops {
            op.vjp_u(out, u, p, t, w);
        }
    }

    /// `out += Σ_k f_{k,p}(u)ᵀ w`.
    pub fn vjp_p(&self, out: &mut [f64], u: &[T], p: &[f64], t: f64, w: &[T]) {
        for op in &self.ops {
            op.vjp_p(out, u, p, t, w);
        }
    }

    /// Whether every operator provides `vjp_p`, so parameter cotangents are formed.
    pub fn has_vjp_p(&self) -> bool {
        self.ops.iter().all(|o| o.provides(Call::VjpP))
    }

    /// `out = Σ_k J_k(u)` in the leaf's matrix storage.
    pub fn jac(&self, out: &mut MatrixStorage<T>, u: &[T], p: &[f64], t: f64) {
        out.clear();
        for op in &self.ops {
            op.jac(out, u, p, t);
        }
    }

    /// Whether the leaf forms `f_t` for its Rosenbrock method: some `d_i ≠ 0`
    /// and some operator depends on `t` (spec/methods.md §4.2). At M1 there
    /// is no forcing, so explicit dependence is the only kind.
    pub fn forms_ft(&self) -> bool {
        match &self.method {
            LeafMethod::Ros(m, _) => m.uses_ft() && self.ops.iter().any(|o| o.depends_on_t()),
            LeafMethod::Erk(_) => false,
        }
    }

    /// `out = Σ_k ∂f_k/∂t` over the operators that depend on `t`.
    pub fn ft(&self, out: &mut [T], u: &[T], p: &[f64], t: f64) {
        out.iter_mut().for_each(|x| *x = T::zero());
        for op in self.ops.iter().filter(|o| o.depends_on_t()) {
            op.ft(out, u, p, t);
        }
    }

    /// `out += Σ_k ∇_u Σ_i μ_iᵀ J_k(u) w_i` (zero for operators linear in `u`).
    pub fn hess_u(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, mu: &[MuPair<'_, T>]) {
        for op in self.ops.iter().filter(|o| !o.linear_in_u()) {
            op.hess_u(out, u, p, t, mu);
        }
    }

    /// `out += Σ_k ∇_p Σ_i μ_iᵀ J_k(u) w_i`.
    pub fn hess_p(&self, out: &mut [f64], u: &[T], p: &[f64], t: f64, mu: &[MuPair<'_, T>]) {
        for op in &self.ops {
            op.hess_p(out, u, p, t, mu);
        }
    }

    /// `out += Σ_k ∇_u (νᵀ ∂f_k/∂t)` over the operators that depend on `t`.
    pub fn ft_vjp_u(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, nu: &[T]) {
        for op in self.ops.iter().filter(|o| o.depends_on_t()) {
            op.ft_vjp_u(out, u, p, t, nu);
        }
    }

    /// `out += Σ_k ∇_p (νᵀ ∂f_k/∂t)` over the operators that depend on `t`.
    pub fn ft_vjp_p(&self, out: &mut [f64], u: &[T], p: &[f64], t: f64, nu: &[T]) {
        for op in self.ops.iter().filter(|o| o.depends_on_t()) {
            op.ft_vjp_p(out, u, p, t, nu);
        }
    }

    /// `ψ(v) = Σ_j w·v_j` over the integrand's DOFs, ascending, in f64.
    pub fn psi(&self, v: &[T]) -> f64 {
        let Some((dofs, w)) = &self.integrand else {
            return 0.0;
        };
        let mut s = 0.0;
        for &j in dofs {
            s += w * v[j].to_f64();
        }
        s
    }

    /// The increment `q_{n+1} − q_n` of the auxiliary state `q' = ψ(u)` over
    /// the step just computed in `w`: the leaf's method applied to the
    /// augmented state `(u, q)` (spec/adjoint.md §2.1, D26). `q` is not part
    /// of the error norm, so the step sequence does not depend on whether a
    /// functional is requested. `ψ` is linear in `u` and does not depend on
    /// `t` or `p`, so its Jacobian row is the constant `g = ∇ψ`, and for a
    /// Rosenbrock step the augmented `W` is block lower triangular:
    /// - transformed form, `Ŵ = σI − J`: `σ k_{q,i} − g·k_i = r_{q,i}`, so
    ///   `k_{q,i} = (ψ(U_i) + Σ_j (C_ij/h) k_{q,j} + ψ(k_i)) / σ`, and
    ///   `Δq = Σ_i b_i k_{q,i}`;
    /// - Rosenbrock23, `Ŵ = I − γJ`: `k_{q,1} = ψ(u_n) + γ ψ(k_1)`,
    ///   `k_{q,2} = ((ψ(U_2) − k_{q,1}) + γ ψ(k_2 − k_1)) + k_{q,1}`, and
    ///   `Δq = h k_{q,2}`;
    /// - explicit Butcher form: `Δq = Σ_i (h b_i) ψ(U_i)`; Shu–Osher form:
    ///   the stage recurrence on `q` from `q^(0) = 0`.
    fn q_increment(&self, w: &MethodWork<T>, h: f64) -> f64 {
        if self.integrand.is_none() {
            return 0.0;
        }
        match (&self.method, w) {
            (LeafMethod::Erk(m), MethodWork::Erk(w)) => match &m.form {
                ErkForm::Butcher { b, .. } => {
                    let mut dq = 0.0;
                    for (i, &bi) in b.iter().enumerate() {
                        if bi != 0.0 {
                            dq += (h * bi) * self.psi(&w.y[i]);
                        }
                    }
                    dq
                }
                ErkForm::ShuOsher { alpha, beta, .. } => {
                    let s = m.stages;
                    let mut q = vec![0.0; s + 1];
                    for i in 1..=s {
                        let mut qi = 0.0;
                        for j in 0..i {
                            let (a, bb) = (alpha[i - 1][j], beta[i - 1][j]);
                            if a != 0.0 {
                                qi += a * q[j];
                            }
                            if bb != 0.0 {
                                qi += (h * bb) * self.psi(&w.y[j]);
                            }
                        }
                        q[i] = qi;
                    }
                    q[s]
                }
            },
            (LeafMethod::Ros(m, _), MethodWork::Ros(w)) => match &m.form {
                RosForm::Transformed { cmat, gamma, b, .. } => {
                    let sigma = 1.0 / (h * gamma);
                    let s = m.stages;
                    let mut kq = vec![0.0; s];
                    for i in 0..s {
                        let mut r = self.psi(&w.y[i]);
                        for j in 0..i {
                            if cmat[i][j] != 0.0 {
                                r += (cmat[i][j] / h) * kq[j];
                            }
                        }
                        kq[i] = (r + self.psi(&w.k[i])) / sigma;
                    }
                    let mut dq = 0.0;
                    for (i, &bi) in b.iter().enumerate() {
                        if bi != 0.0 {
                            dq += bi * kq[i];
                        }
                    }
                    dq
                }
                RosForm::Ode23s { d, .. } => {
                    let gam = h * d;
                    let kq1 = self.psi(&w.y[0]) + gam * self.psi(&w.k[0]);
                    let dk = self.psi(&w.k[1]) - self.psi(&w.k[0]);
                    let kq2 = ((self.psi(&w.y[1]) - kq1) + gam * dk) + kq1;
                    h * kq2
                }
            },
            _ => unreachable!("the leaf's workspace matches its method"),
        }
    }

    /// The domain check of spec/methods.md §11: whether a listed DOF is negative.
    fn violates(&self, u: &[T]) -> bool {
        match &self.domain {
            Some(d) => d.dofs.iter().any(|&j| u[j] < T::zero()),
            None => false,
        }
    }

    /// `check_stages` (spec/methods.md §11): whether a listed DOF of a
    /// Rosenbrock stage state `U_i`, `i ≥ 2`, is negative. `U_1 = u_n` was
    /// checked when it was accepted. A stage cannot be clipped, so this is a
    /// violation under `clip` too.
    fn stage_violates(&self, w: &MethodWork<T>) -> bool {
        match (&self.domain, w) {
            (Some(d), MethodWork::Ros(rw)) if d.check_stages => {
                rw.y.iter()
                    .skip(1)
                    .any(|y| d.dofs.iter().any(|&j| y[j] < T::zero()))
            }
            _ => false,
        }
    }

    /// Clips the listed DOFs at zero and returns the ascending indices of the
    /// DOFs it clipped, or `None` when nothing was clipped (spec/tape.md §2;
    /// spec/adjoint.md §5.7: ties at 0 keep).
    fn clip(&self, u: &mut [T]) -> Option<Vec<usize>> {
        let d = self.domain.as_ref().filter(|d| d.clip)?;
        let clipped: Vec<usize> = d
            .dofs
            .iter()
            .copied()
            .filter(|&j| u[j] < T::zero())
            .collect();
        if clipped.is_empty() {
            return None;
        }
        for &j in &clipped {
            u[j] = T::zero();
        }
        Some(clipped)
    }

    /// Applies a recorded clip (the frozen-tape clip): zeroes the listed DOFs.
    /// A recorded index outside the leaf's clip domain, or a clip on a leaf
    /// without one, is a replay mismatch.
    fn clip_frozen(&self, u: &mut [T], clipped: &[usize]) -> Result<(), Error> {
        let d = self.domain.as_ref().filter(|d| d.clip).ok_or_else(|| {
            Error::Run(format!(
                "replay: leaf {} has a recorded clip but no clip domain",
                self.id
            ))
        })?;
        if let Some(j) = clipped
            .iter()
            .find(|&&j| j >= u.len() || d.dofs.binary_search(&j).is_err())
        {
            return Err(Error::Run(format!(
                "replay: leaf {}: recorded clip of DOF {j}, outside its clip domain",
                self.id
            )));
        }
        for &j in clipped {
            u[j] = T::zero();
        }
        Ok(())
    }

    /// The bookkeeping of an accepted step, shared by the forward loop and
    /// replay so the two stay bitwise identical: the adjoint's step data, the
    /// recorded step and clip, the stored first-same-as-last value, the new
    /// state, and the state at an interior stop it lands on.
    #[allow(clippy::too_many_arguments)]
    fn accept(
        &self,
        call: &LeafCall<'_>,
        out: &mut LeafOutcome<T>,
        lw: (&mut MethodWork<T>, &mut Vec<T>),
        pbuf: &mut [f64],
        cache: &mut Option<(f64, f64)>,
        u: &mut [T],
        s: Accepted,
    ) {
        let (w, fsal) = lw;
        let Accepted {
            t,
            h,
            tk1,
            clip,
            stop,
        } = s;
        let skipped = tk1.is_none();
        if call.record_stages {
            out.stage_data.push(StepData {
                t,
                h,
                p: self.params(call.p, pbuf, h, call).to_vec(),
                y: if skipped { Vec::new() } else { w.y().to_vec() },
                k: if skipped { Vec::new() } else { w.k().to_vec() },
                clip: clip.clone(),
                tk1: tk1.unwrap_or(t),
                skipped,
            });
        }
        if skipped {
            // Nothing is computed; the stored first-same-as-last value is dropped.
            *cache = None;
        } else {
            // Keep f(u_{n+1}, t_n + h) unless clipped.
            *cache = if clip.is_none() && w.has_fend() {
                std::mem::swap(fsal, w.fend_mut());
                Some((t + h, h))
            } else {
                None
            };
            out.integral += self.q_increment(w, h);
            u.copy_from_slice(w.unew());
        }
        if let Some(mk) = clip {
            out.clips.push((out.steps.len(), mk));
        }
        out.steps.push([t, h]);
        out.accepted += 1;
        if let Some(ts) = stop {
            out.stop_states.push((ts, u.to_vec()));
        }
    }

    /// Runs one invocation over `[t0, t1]`, updating `u` and the persistent state.
    pub fn invoke(
        &self,
        u: &mut [T],
        persist: &mut LeafPersist,
        call: &LeafCall<'_>,
    ) -> Result<LeafOutcome<T>, Error> {
        let mut w = self.take_work(u.len())?;
        w.m.reset_nf();
        let out = self.invoke_with(&mut w, u, persist, call);
        self.put_work(w);
        out
    }

    /// A workspace for a state of length `n`: a pooled one, or a new one.
    pub fn take_work(&self, n: usize) -> Result<LeafWork<T>, Error> {
        let pooled = self.work.lock().ok().and_then(|mut v| v.pop());
        Ok(match pooled {
            Some(w) if w.fsal.len() == n => w,
            _ => LeafWork {
                m: match &self.method {
                    LeafMethod::Erk(m) => MethodWork::Erk(m.workspace(n)),
                    LeafMethod::Ros(m, ls) => MethodWork::Ros(Box::new(
                        m.workspace(n, self.structure, *ls)
                            .map_err(|e| Error::Config(format!("leaf {}: {e}", self.id)))?,
                    )),
                },
                fsal: vec![T::zero(); n],
            },
        })
    }

    /// Returns a workspace from [`take_work`](Self::take_work) to the pool.
    pub fn put_work(&self, w: LeafWork<T>) {
        if let Ok(mut v) = self.work.lock() {
            v.push(w);
        }
    }

    fn invoke_with(
        &self,
        lw: &mut LeafWork<T>,
        u: &mut [T],
        persist: &mut LeafPersist,
        call: &LeafCall<'_>,
    ) -> Result<LeafOutcome<T>, Error> {
        let LeafWork { m: w, fsal } = lw;
        let mut out = LeafOutcome {
            steps: Vec::new(),
            accepted: 0,
            rejected: 0,
            retcode: Retcode::Success,
            clips: Vec::new(),
            rhs_calls: 0,
            integral: 0.0,
            stage_data: Vec::new(),
            stop_states: Vec::new(),
        };
        let (t0, t1) = (call.t0, call.t1);
        // The stops of this invocation: interior output times, then the end.
        let mut stops = Stops {
            stops: call.stops,
            t1,
            next: 0,
        };
        // Bound parameters are written into a copy of p (none when nothing is bound).
        let mut pbuf: Vec<f64> = if self.binds.is_empty() {
            Vec::new()
        } else {
            call.p.to_vec()
        };
        // spec/methods.md §3.3: the stored first-same-as-last value, in `fsal`,
        // is valid while `cache` holds the time it was evaluated at and the
        // size of the step that bound its parameters. It is invalid at entry
        // to every invocation, and with `bind_dt: "leaf"` parameters only a
        // step of that same size reuses it, since they change with h.
        let mut cache: Option<(f64, f64)> = None;
        let leaf_bound = self.binds.iter().any(|b| b.1 == BindKind::Leaf);

        let forms_ft = self.forms_ft();
        // One step of size h from (u, t) into `w`: E when wanted, the time of
        // its first stage, and whether W was singular.
        let take = |w: &mut MethodWork<T>,
                    pbuf: &mut [f64],
                    fsal: &[T],
                    cache: Option<(f64, f64)>,
                    u: &[T],
                    t: f64,
                    h: f64,
                    want_err: bool|
         -> Attempt {
            let pb = self.params(call.p, pbuf, h, call);
            let reuse = cache.filter(|&(_, hb)| !leaf_bound || hb == h);
            let mut f = |o: &mut [T], uu: &[T], tt: f64| self.rhs(o, uu, pb, tt);
            let tol = &self.tol;
            let dofs = &self.dofs;
            let mut norm = |e: &[f64], un: &[T], unew: &[T]| error_norm(e, un, unew, tol, dofs);
            let norm: Option<&mut NormFn<'_, T>> = if want_err { Some(&mut norm) } else { None };
            match (&self.method, w) {
                (LeafMethod::Erk(m), MethodWork::Erk(w)) => {
                    let e = m.step(w, &mut f, u, t, h, reuse.map(|_| fsal), norm);
                    let tk1 = reuse.map_or(m.stage_time(0, t, h), |c| c.0);
                    Attempt {
                        e,
                        tk1,
                        singular: false,
                    }
                }
                (LeafMethod::Ros(m, _), MethodWork::Ros(w)) => {
                    let mut jac =
                        |o: &mut MatrixStorage<T>, uu: &[T], tt: f64| self.jac(o, uu, pb, tt);
                    let mut ftf = |o: &mut [T], uu: &[T], tt: f64| self.ft(o, uu, pb, tt);
                    let ft: Option<&mut RhsFn<'_, T>> =
                        if forms_ft { Some(&mut ftf) } else { None };
                    let reuse = if m.fsal { reuse } else { None };
                    let o = m.step(w, &mut f, &mut jac, ft, u, t, h, reuse.map(|_| fsal), norm);
                    let tk1 = reuse.map_or(t, |c| c.0);
                    match o {
                        RosOutcome::Singular => Attempt {
                            e: None,
                            tk1,
                            singular: true,
                        },
                        RosOutcome::Done(e) => Attempt {
                            e,
                            tk1,
                            singular: false,
                        },
                    }
                }
                _ => unreachable!("the leaf's workspace matches its method"),
            }
        };
        // Replay: the recorded steps, controller off (spec/tape.md §3).
        if let Some((steps, clips)) = call.replay {
            if steps.first().is_some_and(|s| s[0] != t0) {
                return Err(Error::Run(format!(
                    "replay: leaf {}: the recorded steps start at {:?}, the invocation at {t0:?}",
                    self.id, steps[0][0]
                )));
            }
            // Clips are recorded in step order, one per step at most.
            let mut clips = clips.iter().peekable();
            for (i, s) in steps.iter().enumerate() {
                let (t, h) = (s[0], s[1]);
                let (ts, interior) = stops.next_after(t);
                let (_, is_stop) = stop_rule(t, h, ts);
                let tnew = if is_stop { ts } else { t + h };
                let mask = clips.next_if(|c| c.0 == i).map(|c| c.1.clone());
                let tk1 = if is_stop && h < eps_f64(t) {
                    // A skipped tiny stop step, recognized by the same rule.
                    if mask.is_some() {
                        return Err(Error::Run(format!(
                            "replay: leaf {}: a recorded clip on skipped step {i}",
                            self.id
                        )));
                    }
                    None
                } else {
                    let a = take(w, &mut pbuf, fsal, cache, u, t, h, false);
                    if a.singular {
                        return Err(Error::Run(format!(
                            "replay: leaf {}: W is singular on recorded step {i}",
                            self.id
                        )));
                    }
                    if let Some(mk) = &mask {
                        self.clip_frozen(w.unew_mut(), mk)?;
                    }
                    Some(a.tk1)
                };
                let step = Accepted {
                    t,
                    h,
                    tk1,
                    clip: mask,
                    stop: (is_stop && interior).then_some(tnew),
                };
                self.accept(call, &mut out, (w, fsal), &mut pbuf, &mut cache, u, step);
            }
            if let Some((k, _)) = clips.next() {
                return Err(Error::Run(format!(
                    "replay: leaf {}: a recorded clip at step {k} is out of order or past its {} steps",
                    self.id,
                    steps.len()
                )));
            }
            out.rhs_calls = w.nf() as u64;
            return Ok(out);
        }

        let adaptive = self.dt_mode == DtMode::Adaptive;
        let dtmin = self.dtmin;
        let dtmax = self.dtmax.unwrap_or(t1 - t0);
        // fix_dt_at_bounds: [max(dtmin, eps(t)), dtmax].
        let fix = |h: f64, t: f64| h.min(dtmax).max(eps_f64(t).max(dtmin));
        let ctrl = if adaptive {
            Some(self.controller.as_ref().ok_or_else(|| {
                Error::Config(format!(
                    "leaf {}: {} has no error estimate, so the leaf must give dt (D52)",
                    self.id, self.alg
                ))
            })?)
        } else {
            None
        };
        if adaptive && persist.proposal.is_none() {
            let h0 = match self.dt0 {
                Some(d) => fix(d, t0),
                None => {
                    // The build refuses a "leaf" binding here (spec/methods.md §9).
                    let pb = self.params(call.p, &mut pbuf, 0.0, call);
                    let inp = InitDtInput {
                        u0: u,
                        t0,
                        dtmin,
                        // Capped at the first stop, as OrdinaryDiffEqCore's
                        // _determine_initdt does (spec/methods.md §9).
                        dtmax: dtmax.min(stops.next_after(t0).0 - t0),
                        tol: &self.tol,
                        dofs: &self.dofs,
                        order: self.method.order(),
                    };
                    let (h0, nf) = initial_dt(&inp, &mut |o, uu, tt| self.rhs(o, uu, pb, tt));
                    out.rhs_calls += nf as u64;
                    h0
                }
            };
            persist.proposal = Some(h0);
        }
        // Fixed substeps: dt, or one step of the node's Δ for "step" and
        // "accumulated" (Δ without `every`), landing on the end by the stop rule.
        let fixed_dt = match self.dt_mode {
            DtMode::Fixed(dt) => dt,
            _ => t1 - t0,
        };
        let mut hctrl = persist.proposal.unwrap_or(f64::NAN);
        let maxiters = self
            .maxiters
            .unwrap_or(if adaptive { 1_000_000 } else { u64::MAX });
        let mut t = t0;
        let mut iter: u64 = 0;
        // Whether the leaf's previous step attempt was accepted. An invocation
        // ends on an accepted step, so at entry this is whether the leaf has
        // accepted a step: the first attempt of the run counts as following a
        // rejection, as in OrdinaryDiffEq (its `accept_step` starts false).
        let mut prev_accepted = persist.ctrl.as_ref().is_some_and(|c| c.accepted_any);
        loop {
            if t >= t1 {
                break;
            }
            let (ts, interior) = stops.next_after(t);
            let hc = if adaptive { fix(hctrl, t) } else { fixed_dt };
            let (h, is_stop) = stop_rule(t, hc, ts);
            iter += 1;
            // Limits and return codes (spec/methods.md §12), checked before
            // every attempt in SciMLBase `check_error`'s order. A step at or
            // below dtmin is allowed only when the previous attempt was
            // accepted and this one lands on the stop.
            if h.is_nan() {
                out.retcode = Retcode::DtNaN;
                break;
            } else if iter > maxiters {
                out.retcode = Retcode::MaxIters;
                break;
            } else if adaptive && dtmin > 0.0 && h <= dtmin && (!prev_accepted || t + h < ts) {
                out.retcode = Retcode::DtLessThanMin;
                break;
            } else if adaptive && !prev_accepted && h <= eps_f64(t) {
                out.retcode = Retcode::Unstable;
                break;
            }
            let tnew = if is_stop { ts } else { t + h };
            if is_stop && h < eps_f64(t) {
                // handle_tstop_step!: accepted without computing it; recorded
                // and counted, and the stored first-same-as-last value dropped.
                // Nothing else changes: not the proposal, the controller's
                // history or whether the leaf has accepted a step (spec/methods.md §10).
                let step = Accepted {
                    t,
                    h,
                    tk1: None,
                    clip: None,
                    stop: interior.then_some(tnew),
                };
                self.accept(call, &mut out, (w, fsal), &mut pbuf, &mut cache, u, step);
                t = tnew;
                continue;
            }
            let Attempt { e, tk1, singular } = take(w, &mut pbuf, fsal, cache, u, t, h, adaptive);
            let violation = !singular
                && ((self.violates(w.unew()) && !self.domain.as_ref().is_some_and(|d| d.clip))
                    || self.stage_violates(w));
            if let Some(ctrl) = ctrl {
                let cs = persist
                    .ctrl
                    .as_mut()
                    .expect("adaptive leaf has controller state");
                let e = e.unwrap_or(2.0);
                let reject_to = if singular {
                    // spec/methods.md §4.2, §8.5: rejected as if E were 2,
                    // OrdinaryDiffEq's set_EEst!(integrator, 2). The step is
                    // rejected whatever the controller decides (under PID,
                    // E = 2 can pass accept_safety); the new size is the
                    // controller's rejection size for E = 2.
                    Some(ctrl.reject(&ctrl.evaluate(cs, e), h))
                } else if !e.is_finite() || violation {
                    // spec/methods.md §7 (non-finite E) and §8.5 (domain).
                    Some(h * ctrl.qmin)
                } else {
                    let ev = ctrl.evaluate(cs, e);
                    match ev.decision {
                        // h is the step attempted, shortened if it was a stop step.
                        Decision::Reject => Some(ctrl.reject(&ev, h)),
                        Decision::Accept => {
                            // D61 (spec/methods.md §8.6): after a stop step
                            // shortened from the controller's size hc, the
                            // proposal comes from the shortened h when the
                            // previous attempt was accepted, and from hc when
                            // this is the leaf's first attempt of the run or
                            // follows a rejection. OrdinaryDiffEqCore 4.19.0
                            // restores the size modify_dt_for_tstops! saved;
                            // after an accepted step its second call (in
                            // loopheader!) saves the shortened size, raised
                            // to dtmin by the fix_dt_at_bounds! before it.
                            let hb = match (is_stop, prev_accepted) {
                                (false, _) => h,
                                (true, true) => fix(h, t),
                                (true, false) => hc,
                            };
                            let prop = ctrl.accept(cs, &ev, e, hb);
                            hctrl = fix((tnew + prop) - tnew, tnew);
                            None
                        }
                    }
                };
                if let Some(hn) = reject_to {
                    out.rejected += 1;
                    hctrl = hn;
                    prev_accepted = false;
                    continue;
                }
            } else {
                // A singular W at fixed dt cannot shrink the step: Unstable.
                if singular || w.unew().iter().any(|x| !x.is_finite()) {
                    out.retcode = Retcode::Unstable;
                    break;
                }
                if violation {
                    out.retcode = Retcode::DomainViolation;
                    break;
                }
            }
            // Accepted.
            let clip = self.clip(w.unew_mut());
            let step = Accepted {
                t,
                h,
                tk1: Some(tk1),
                clip,
                stop: (is_stop && interior).then_some(tnew),
            };
            self.accept(call, &mut out, (w, fsal), &mut pbuf, &mut cache, u, step);
            prev_accepted = true;
            t = tnew;
        }
        if adaptive {
            persist.proposal = Some(hctrl);
        }
        out.rhs_calls += w.nf() as u64;
        Ok(out)
    }
}

/// An accepted step, for [`LeafSpec::accept`].
struct Accepted {
    t: f64,
    h: f64,
    /// The time of the first stage; `None` for a skipped tiny stop step.
    tk1: Option<f64>,
    clip: Option<Vec<usize>>,
    /// The interior stop the step lands on.
    stop: Option<f64>,
}

/// The stops of an invocation: its interior stops, ascending, then its end,
/// walked forward as the steps advance.
struct Stops<'a> {
    stops: &'a [f64],
    t1: f64,
    /// The first interior stop not yet passed.
    next: usize,
}

impl Stops<'_> {
    /// The first stop after `t`, and whether it is an interior stop (not the
    /// end). `t` never decreases between calls.
    fn next_after(&mut self, t: f64) -> (f64, bool) {
        while self.next < self.stops.len() && self.stops[self.next] <= t {
            self.next += 1;
        }
        match self.stops.get(self.next) {
            Some(&s) => (s, true),
            None => (self.t1, false),
        }
    }
}

/// The landing tolerance of spec/methods.md §10, `100 eps(max(|a|, |b|))`:
/// a time within it of a stop lands on the stop.
pub fn landing_tol(a: f64, b: f64) -> f64 {
    100.0 * eps_f64(a.abs().max(b.abs()))
}

/// spec/methods.md §10: with `d = ts − t` and `tol = 100 eps(max(|t|, |ts|))`,
/// an ordinary step when `h + tol < d`, otherwise a stop step of size
/// `min(h, d)` that ends at `ts` exactly.
pub fn stop_rule(t: f64, h: f64, ts: f64) -> (f64, bool) {
    let d = ts - t;
    if h + landing_tol(t, ts) < d {
        (h, false)
    } else {
        (h.min(d), true)
    }
}
