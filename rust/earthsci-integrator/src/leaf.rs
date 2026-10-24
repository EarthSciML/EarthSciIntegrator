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
use std::sync::{Arc, Mutex};

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
    /// Workspaces kept across invocations, one per invocation in flight.
    pub work: Mutex<Vec<LeafWork<T>>>,
}

/// A leaf's step workspace: the method's, and the stored
/// first-same-as-last value (spec/methods.md §3.3).
pub struct LeafWork<T> {
    erk: ErkWork<T>,
    fsal: Vec<T>,
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

    /// The domain check of spec/methods.md §11: whether a listed DOF is negative.
    fn violates(&self, u: &[T]) -> bool {
        match &self.domain {
            Some(d) => d.dofs.iter().any(|&j| u[j] < T::zero()),
            None => false,
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
    fn clip_frozen(&self, u: &mut [T], clipped: &[usize]) {
        for &j in clipped {
            u[j] = T::zero();
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
        let pooled = self.work.lock().ok().and_then(|mut v| v.pop());
        let mut w = match pooled {
            Some(w) if w.fsal.len() == n => w,
            _ => LeafWork {
                erk: m.workspace(n),
                fsal: vec![T::zero(); n],
            },
        };
        w.erk.nf = 0;
        let out = self.invoke_with(m, &mut w, u, persist, call);
        if let Ok(mut v) = self.work.lock() {
            v.push(w);
        }
        out
    }

    fn invoke_with(
        &self,
        m: &ExplicitRk,
        lw: &mut LeafWork<T>,
        u: &mut [T],
        persist: &mut LeafPersist,
        call: &LeafCall<'_>,
    ) -> Result<LeafOutcome<T>, Error> {
        let LeafWork { erk: w, fsal } = lw;
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
        let (t0, t1) = (call.t0, call.t1);
        // The stops of this invocation: interior output times, then the end.
        let mut stops: Vec<f64> = call.stops.to_vec();
        stops.push(t1);
        let next_stop = |t: f64| stops.iter().copied().find(|&s| s > t).unwrap_or(t1);
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

        // One step of size h from (u, t) into `w`; returns E when wanted and
        // the time of its first stage.
        let take = |w: &mut ErkWork<T>,
                    pbuf: &mut [f64],
                    fsal: &[T],
                    cache: Option<(f64, f64)>,
                    u: &[T],
                    t: f64,
                    h: f64,
                    want_err: bool|
         -> (Option<f64>, f64) {
            let pb = self.params(call.p, pbuf, h, call);
            let reuse = cache.filter(|&(_, hb)| !leaf_bound || hb == h);
            let mut f = |o: &mut [T], uu: &[T], tt: f64| self.rhs(o, uu, pb, tt);
            let tol = &self.tol;
            let dofs = &self.dofs;
            let mut norm = |e: &[f64], un: &[T], unew: &[T]| error_norm(e, un, unew, tol, dofs);
            let e = m.step(
                w,
                &mut f,
                u,
                t,
                h,
                reuse.map(|_| fsal),
                if want_err { Some(&mut norm) } else { None },
            );
            let tk1 = reuse.map_or(m.stage_time(0, t, h), |c| c.0);
            (e, tk1)
        };
        // After an accepted step: keep f(u_{n+1}, t_n + h) unless clipped.
        let keep = |w: &mut ErkWork<T>, fsal: &mut Vec<T>, clipped: bool, t: f64, h: f64| {
            if !clipped && w.has_fend {
                std::mem::swap(fsal, &mut w.fend);
                Some((t + h, h))
            } else {
                None
            }
        };

        // Replay: the recorded steps, controller off (spec/tape.md §3).
        if let Some((steps, clips)) = call.replay {
            for (i, s) in steps.iter().enumerate() {
                let (t, h) = (s[0], s[1]);
                let ts = next_stop(t);
                let (_, is_stop) = stop_rule(t, h, ts);
                let tnew = if is_stop { ts } else { t + h };
                if is_stop && h < eps_f64(t) {
                    // A skipped tiny stop step, recognized by the same rule.
                    cache = None;
                    if call.record_stages {
                        out.stage_data.push(StepData {
                            t,
                            h,
                            p: self.params(call.p, &mut pbuf, h, call).to_vec(),
                            y: Vec::new(),
                            clip: None,
                            tk1: t,
                            skipped: true,
                        });
                    }
                } else {
                    let (_, tk1) = take(w, &mut pbuf, fsal, cache, u, t, h, false);
                    let mut clip = None;
                    if let Some((_, mask)) = clips.iter().find(|(k, _)| *k == i) {
                        self.clip_frozen(&mut w.unew, mask);
                        clip = Some(mask.clone());
                    }
                    if call.record_stages {
                        out.stage_data.push(StepData {
                            t,
                            h,
                            p: self.params(call.p, &mut pbuf, h, call).to_vec(),
                            y: w.y.clone(),
                            clip: clip.clone(),
                            tk1,
                            skipped: false,
                        });
                    }
                    cache = keep(w, fsal, clip.is_some(), t, h);
                    u.copy_from_slice(&w.unew);
                    if let Some(mk) = clip {
                        out.clips.push((i, mk));
                    }
                }
                if tnew != t1 && call.stops.contains(&tnew) {
                    out.stop_states.push((tnew, u.to_vec()));
                }
                out.steps.push(*s);
                out.accepted += 1;
            }
            out.rhs_calls = w.nf as u64;
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
                        dtmax: dtmax.min(next_stop(t0) - t0),
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
            let ts = next_stop(t);
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
                if call.record_stages {
                    out.stage_data.push(StepData {
                        t,
                        h,
                        p: self.params(call.p, &mut pbuf, h, call).to_vec(),
                        y: Vec::new(),
                        clip: None,
                        tk1: t,
                        skipped: true,
                    });
                }
                out.steps.push([t, h]);
                out.accepted += 1;
                cache = None;
                t = tnew;
                if t != t1 && call.stops.contains(&t) {
                    out.stop_states.push((t, u.to_vec()));
                }
                continue;
            }
            let (e, tk1) = take(w, &mut pbuf, fsal, cache, u, t, h, adaptive);
            let violation = self.violates(&w.unew) && !self.domain.as_ref().is_some_and(|d| d.clip);
            if let Some(ctrl) = ctrl {
                let cs = persist
                    .ctrl
                    .as_mut()
                    .expect("adaptive leaf has controller state");
                let e = e.expect("adaptive step returns an error estimate");
                let reject_to = if !e.is_finite() || violation {
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
                if w.unew.iter().any(|x| !x.is_finite()) {
                    out.retcode = Retcode::Unstable;
                    break;
                }
                if violation {
                    out.retcode = Retcode::DomainViolation;
                    break;
                }
            }
            // Accepted.
            let clip = self.clip(&mut w.unew);
            let k = out.steps.len();
            if call.record_stages {
                out.stage_data.push(StepData {
                    t,
                    h,
                    p: self.params(call.p, &mut pbuf, h, call).to_vec(),
                    y: w.y.clone(),
                    clip: clip.clone(),
                    tk1,
                    skipped: false,
                });
            }
            out.steps.push([t, h]);
            out.accepted += 1;
            cache = keep(w, fsal, clip.is_some(), t, h);
            if let Some(mk) = clip {
                out.clips.push((k, mk));
            }
            u.copy_from_slice(&w.unew);
            prev_accepted = true;
            t = tnew;
            if t != t1 && call.stops.contains(&t) {
                out.stop_states.push((t, u.to_vec()));
            }
        }
        if adaptive {
            persist.proposal = Some(hctrl);
        }
        out.rhs_calls += w.nf as u64;
        Ok(out)
    }
}

/// spec/methods.md §10: with `d = ts − t` and `tol = 100 eps(max(|t|, |ts|))`,
/// an ordinary step when `h + tol < d`, otherwise a stop step of size
/// `min(h, d)` that ends at `ts` exactly.
pub fn stop_rule(t: f64, h: f64, ts: f64) -> (f64, bool) {
    let d = ts - t;
    let tol = 100.0 * eps_f64(t.abs().max(ts.abs()));
    if h + tol < d {
        (h, false)
    } else {
        (h.min(d), true)
    }
}
