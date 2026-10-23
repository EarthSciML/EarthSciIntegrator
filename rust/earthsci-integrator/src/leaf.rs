//! A leaf: advances the sum of its operators over one interval with one
//! method (spec/run-config.md §4.2, spec/methods.md).
//!
//! The method is a [`LeafMethod`]. M1's first stage implements the explicit
//! Runge–Kutta family; Rosenbrock, SDIRK, IMEX Euler and θ-method leaves are
//! refused when the schedule is built and join as further variants, sharing
//! the step loop below (landing on stops, control, domain, return codes).

use crate::operator::{Call, Operator, Structure};
use crate::tape::{LeafPersist, Retcode};
use crate::{Error, Scalar};
use earthsci_integrator_solvers::controller::{Controller, Decision};
use earthsci_integrator_solvers::erk::{ErkWork, ExplicitRk};
use earthsci_integrator_solvers::initdt::{initial_dt, InitDtInput};
use earthsci_integrator_solvers::norm::{error_norm, Tolerances};
use earthsci_integrator_solvers::scalar::eps_f64;
use std::sync::Arc;

/// The leaf's method.
#[derive(Debug, Clone)]
pub enum LeafMethod {
    /// Explicit Runge–Kutta (spec/methods.md §3).
    Erk(ExplicitRk),
}

impl LeafMethod {
    /// Operator calls the method's forward step needs.
    pub fn forward_calls(&self) -> &'static [Call] {
        match self {
            LeafMethod::Erk(_) => &[Call::Rhs],
        }
    }
    /// Operator calls the method's adjoint step needs (spec/adjoint.md §3).
    pub fn adjoint_calls(&self, wrt_p: bool) -> Vec<Call> {
        match self {
            LeafMethod::Erk(_) => {
                let mut v = vec![Call::VjpU];
                if wrt_p {
                    v.push(Call::VjpP);
                }
                v
            }
        }
    }
    /// Whether the method has an error estimate.
    pub fn adaptive(&self) -> bool {
        match self {
            LeafMethod::Erk(m) => m.adaptive(),
        }
    }
    /// The method's order (for the first step).
    pub fn order(&self) -> u32 {
        match self {
            LeafMethod::Erk(m) => m.order,
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
}

/// One accepted step as the adjoint needs it.
#[derive(Debug, Clone)]
pub struct StepData<T> {
    pub t: f64,
    pub h: f64,
    /// Parameters the step's evaluations saw (bound parameters set).
    pub p: Vec<f64>,
    /// Stage states: Butcher `U_i`; Shu–Osher `u^(j)`, `j = 0..s`.
    pub y: Vec<Vec<T>>,
    /// The clip mask applied after the step, if it clipped.
    pub clip: Option<Vec<u8>>,
}

/// The recorded steps and clip masks of a leaf invocation.
pub type Replay<'a> = (&'a [[f64; 2]], &'a [(usize, Vec<u8>)]);

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
    pub clips: Vec<(usize, Vec<u8>)>,
    pub rhs_calls: u64,
    pub stage_data: Vec<StepData<T>>,
    /// The state on landing at each interior stop.
    pub stop_states: Vec<(f64, Vec<T>)>,
}

impl<T: Scalar> LeafSpec<T> {
    /// A fresh persistent state.
    pub fn initial_persist(&self) -> LeafPersist {
        LeafPersist::new(self.controller.as_ref().map(|c| c.initial_state()))
    }

    /// Parameters with the bound ones set for a step of size `h`.
    fn bound_params(&self, p: &[f64], h: f64, call: &LeafCall<'_>) -> Vec<f64> {
        let mut q = p.to_vec();
        for &(i, k) in &self.binds {
            q[i] = match k {
                BindKind::Leaf => h,
                BindKind::Node => call.delta,
                BindKind::Macro => call.macro_dt,
            };
        }
        q
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

    /// The domain check of spec/methods.md §11: whether a listed DOF is negative.
    fn violates(&self, u: &[T]) -> bool {
        match &self.domain {
            Some(d) => d.dofs.iter().any(|&j| u[j] < T::zero()),
            None => false,
        }
    }

    /// Clips the listed DOFs at zero and returns the mask (1 = kept), or
    /// `None` when nothing was clipped (spec/adjoint.md §5.7: ties at 0 keep).
    fn clip(&self, u: &mut [T]) -> Option<Vec<u8>> {
        let d = self.domain.as_ref().filter(|d| d.clip)?;
        let mask: Vec<u8> = d
            .dofs
            .iter()
            .map(|&j| u8::from(u[j] >= T::zero()))
            .collect();
        if mask.iter().all(|&m| m == 1) {
            return None;
        }
        for (&j, &m) in d.dofs.iter().zip(&mask) {
            if m == 0 {
                u[j] = T::zero();
            }
        }
        Some(mask)
    }

    /// Applies a recorded clip mask (the frozen-tape clip).
    fn clip_frozen(&self, u: &mut [T], mask: &[u8]) {
        if let Some(d) = &self.domain {
            for (&j, &m) in d.dofs.iter().zip(mask) {
                if m == 0 {
                    u[j] = T::zero();
                }
            }
        }
    }

    /// Runs one invocation over `[t0, t1]`, updating `u` and the persistent state.
    pub fn invoke(
        &self,
        u: &mut [T],
        persist: &mut LeafPersist,
        call: &LeafCall<'_>,
    ) -> Result<LeafOutcome<T>, Error> {
        let LeafMethod::Erk(m) = &self.method;
        let n = u.len();
        let mut w: ErkWork<T> = m.workspace(n);
        let mut out = LeafOutcome {
            steps: Vec::new(),
            accepted: 0,
            rejected: 0,
            retcode: Retcode::Success,
            clips: Vec::new(),
            rhs_calls: 0,
            stage_data: Vec::new(),
            stop_states: Vec::new(),
        };
        // spec/methods.md §3.3: the stored first-same-as-last value is
        // invalid at entry to every invocation.
        let mut cache: Option<Vec<T>> = None;
        let (t0, t1) = (call.t0, call.t1);

        // One step of size h from (u, t); returns E when wanted.
        let take = |w: &mut ErkWork<T>,
                    u: &[T],
                    t: f64,
                    h: f64,
                    cache: &Option<Vec<T>>,
                    want_err: bool|
         -> (Option<f64>, Vec<f64>) {
            let pb = self.bound_params(call.p, h, call);
            let mut f = |o: &mut [T], uu: &[T], tt: f64| self.rhs(o, uu, &pb, tt);
            let tol = &self.tol;
            let dofs = &self.dofs;
            let mut norm = |e: &[f64], un: &[T], unew: &[T]| error_norm(e, un, unew, tol, dofs);
            let e = m.step(
                w,
                &mut f,
                u,
                t,
                h,
                cache.as_deref(),
                if want_err { Some(&mut norm) } else { None },
            );
            (e, pb)
        };

        // Replay: the recorded steps, controller off (spec/tape.md §3).
        if let Some((steps, clips)) = call.replay {
            let mut t = t0;
            for (i, s) in steps.iter().enumerate() {
                let (tn, h) = (s[0], s[1]);
                t = tn;
                let (_, pb) = take(&mut w, u, t, h, &cache, false);
                let mut unew = w.unew.clone();
                let mut clip = None;
                if let Some((_, mask)) = clips.iter().find(|(k, _)| *k == i) {
                    self.clip_frozen(&mut unew, mask);
                    clip = Some(mask.clone());
                }
                if call.record_stages {
                    out.stage_data.push(StepData {
                        t,
                        h,
                        p: pb,
                        y: w.y.clone(),
                        clip: clip.clone(),
                    });
                }
                cache = if clip.is_none() && w.has_fend {
                    Some(w.fend.clone())
                } else {
                    None
                };
                u.copy_from_slice(&unew);
                t = steps.get(i + 1).map(|s| s[0]).unwrap_or(t1);
                if let Some(&st) = call.stops.iter().find(|&&st| st == t) {
                    out.stop_states.push((st, u.to_vec()));
                }
                out.steps.push(*s);
                out.accepted += 1;
                if let Some(mk) = clip {
                    out.clips.push((i, mk));
                }
            }
            let _ = t;
            out.rhs_calls = w.nf as u64;
            return Ok(out);
        }

        let adaptive = self.dt_mode == DtMode::Adaptive;
        let dtmax = self.dtmax.unwrap_or(t1 - t0);
        let mut h = match self.dt_mode {
            DtMode::Adaptive => {
                let ctrl = self.controller.as_ref().ok_or_else(|| {
                    Error::Config(format!(
                        "leaf {}: {} has no error estimate, so the leaf must give dt (D52)",
                        self.id, self.alg
                    ))
                })?;
                let _ = ctrl;
                match persist.proposal {
                    // *Assumption* (spec/methods.md §8.6 says dtmax defaults to the
                    // current interval): a proposal carried over from a longer
                    // interval is clamped to this invocation's dtmax.
                    Some(h) => h.min(dtmax),
                    None => {
                        let h0 = match self.dt0 {
                            Some(d) => d.min(dtmax).max(self.dtmin.max(eps_f64(t0))),
                            None => {
                                let pb = self.bound_params(call.p, 0.0, call);
                                let inp = InitDtInput {
                                    u0: u,
                                    t0,
                                    dtmin: self.dtmin,
                                    dtmax,
                                    tol: &self.tol,
                                    dofs: &self.dofs,
                                    order: self.method.order(),
                                };
                                let (h0, nf) =
                                    initial_dt(&inp, &mut |o, uu, tt| self.rhs(o, uu, &pb, tt));
                                out.rhs_calls += nf as u64;
                                h0
                            }
                        };
                        persist.proposal = Some(h0);
                        h0
                    }
                }
            }
            DtMode::Fixed(dt) => dt,
            DtMode::Step | DtMode::Accumulated => call.delta,
        };

        // One step of Δ: "step" and "accumulated" (spec/run-config.md §4.2).
        if matches!(self.dt_mode, DtMode::Step | DtMode::Accumulated) {
            let (_, pb) = take(&mut w, u, t0, h, &cache, false);
            let mut unew = w.unew.clone();
            out.rhs_calls += w.nf as u64;
            if unew.iter().any(|x| !x.is_finite()) {
                out.retcode = Retcode::Unstable;
                return Ok(out);
            }
            let mut clip = None;
            if self.violates(&unew) {
                match &self.domain {
                    Some(d) if d.clip => clip = self.clip(&mut unew),
                    _ => {
                        out.retcode = Retcode::DomainViolation;
                        return Ok(out);
                    }
                }
            }
            if call.record_stages {
                out.stage_data.push(StepData {
                    t: t0,
                    h,
                    p: pb,
                    y: w.y.clone(),
                    clip: clip.clone(),
                });
            }
            u.copy_from_slice(&unew);
            out.steps.push([t0, h]);
            out.accepted = 1;
            if let Some(mk) = clip {
                out.clips.push((0, mk));
            }
            return Ok(out);
        }

        let mut t = t0;
        let mut attempts: u64 = 0;
        let maxiters = self
            .maxiters
            .or(if adaptive { Some(1_000_000) } else { None });
        let mut stop_idx = 0;
        while t != t1 {
            // The next stop: interior stops, then the interval end (spec/methods.md §10).
            while stop_idx < call.stops.len() && call.stops[stop_idx] <= t {
                stop_idx += 1;
            }
            let stop = call.stops.get(stop_idx).copied().unwrap_or(t1);
            let d = stop - t;
            let tol = 100.0 * eps_f64(t.abs().max(stop.abs()));
            let (hs, is_stop) = if h + tol < d {
                (h, false)
            } else {
                (h.min(d), true)
            };
            if is_stop && hs < eps_f64(t) {
                // handle_tstop_step!: accepted without computing it.
                t = stop;
                if stop != t1 {
                    out.stop_states.push((stop, u.to_vec()));
                }
                continue;
            }
            attempts += 1;
            if let Some(mx) = maxiters {
                if attempts > mx {
                    out.retcode = Retcode::MaxIters;
                    break;
                }
            }
            let (e, pb) = take(&mut w, u, t, hs, &cache, adaptive);
            let mut unew = w.unew.clone();
            let violation = self.violates(&unew) && !self.domain.as_ref().is_some_and(|d| d.clip);

            if !adaptive {
                if unew.iter().any(|x| !x.is_finite()) {
                    out.retcode = Retcode::Unstable;
                    break;
                }
                if violation {
                    out.retcode = Retcode::DomainViolation;
                    break;
                }
            } else {
                let ctrl = self
                    .controller
                    .as_ref()
                    .expect("adaptive leaf has a controller");
                let cs = persist
                    .ctrl
                    .as_mut()
                    .expect("adaptive leaf has controller state");
                let e = e.expect("adaptive step returns an error estimate");
                let reject_to = if !e.is_finite() || violation {
                    // spec/methods.md §7 (non-finite E) and §8.5 (domain).
                    Some(hs * ctrl.qmin)
                } else {
                    let ev = ctrl.evaluate(cs, e);
                    match ev.decision {
                        Decision::Reject => Some(ctrl.reject(&ev, hs)),
                        Decision::Accept => {
                            // The proposal comes from the unshortened size
                            // the controller chose (§8.6).
                            //
                            // *Spec discrepancy, reported:* OrdinaryDiffEqCore
                            // 4.19.0 calls modify_dt_for_tstops! twice before a
                            // step (apply_step!, then loopheader! line 123), and
                            // the second call overwrites dtpropose with the
                            // already-shortened dt, so OrdinaryDiffEq in fact
                            // proposes from the shortened step. With that rule
                            // this leaf reproduces OrdinaryDiffEq's Tsit5 run of
                            // van_der_pol/nonstiff (170 accepted, 2 rejected,
                            // 1035 f calls). We follow the spec's stated rule.
                            let prop = ctrl.accept(cs, &ev, e, h);
                            let tn = if is_stop { stop } else { t + hs };
                            let mut ph = (tn + prop) - tn;
                            ph = ph.min(dtmax).max(self.dtmin.max(eps_f64(tn)));
                            h = ph;
                            persist.proposal = Some(h);
                            None
                        }
                    }
                };
                if let Some(hn) = reject_to {
                    out.rejected += 1;
                    h = hn;
                    persist.proposal = Some(h);
                    if h.is_nan() {
                        out.retcode = Retcode::DtNaN;
                        break;
                    }
                    if self.dtmin > 0.0 && h <= self.dtmin {
                        out.retcode = Retcode::DtLessThanMin;
                        break;
                    }
                    if h <= eps_f64(t) {
                        out.retcode = Retcode::Unstable;
                        break;
                    }
                    continue;
                }
            }
            // Accepted.
            let clip = self.clip(&mut unew);
            let k = out.steps.len();
            if call.record_stages {
                out.stage_data.push(StepData {
                    t,
                    h: hs,
                    p: pb,
                    y: w.y.clone(),
                    clip: clip.clone(),
                });
            }
            out.steps.push([t, hs]);
            out.accepted += 1;
            cache = if clip.is_none() && w.has_fend {
                Some(w.fend.clone())
            } else {
                None
            };
            if let Some(mk) = clip {
                out.clips.push((k, mk));
            }
            u.copy_from_slice(&unew);
            t = if is_stop { stop } else { t + hs };
            if is_stop && stop != t1 {
                out.stop_states.push((stop, u.to_vec()));
            }
        }
        out.rhs_calls += w.nf as u64;
        Ok(out)
    }
}
