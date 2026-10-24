//! Step-size controllers (spec/methods.md §8): PI (default), I and PID.
//!
//! `q` is the factor that divides `h` (OrdinaryDiffEq's convention).

use crate::tableau::{ControllerConstants, Tableau};

/// Which controller a leaf uses (`control.controller`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControllerKind {
    PI,
    I,
    PID,
}

/// The controller's error history, part of a leaf's persistent state
/// (spec/methods.md §8.6) and of checkpoints (spec/tape.md §4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControllerState {
    /// `E_old` of the PI controller.
    pub eold: f64,
    /// The PID controller's `ε₂`, `ε₃` (previous two accepted steps).
    pub pid_eps: [f64; 2],
    /// Whether the leaf has accepted a step this run (`qmax_first_step` before).
    pub accepted_any: bool,
}

/// The outcome of the controller on one step attempt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Decision {
    /// Accept; the proposal for the next step is `h_base * factor`.
    Accept,
    /// Reject; retry with `h_attempted * factor`.
    Reject,
}

/// Controller constants plus the method's orders.
#[derive(Debug, Clone)]
pub struct Controller {
    pub kind: ControllerKind,
    pub beta1: f64,
    pub beta2: f64,
    pub beta3: f64,
    pub gamma: f64,
    pub qmin: f64,
    pub qmax: f64,
    pub qmax_first_step: f64,
    pub qsteady_min: f64,
    pub qsteady_max: f64,
    pub failfactor: f64,
    pub qoldinit: f64,
    pub accept_safety: f64,
    pub order: u32,
    pub embedded_order: u32,
}

/// Intermediate values of one evaluation, kept between the decision and the
/// update so the arithmetic is done once.
#[derive(Debug, Clone, Copy)]
pub struct Evaluation {
    pub decision: Decision,
    /// PI: `q11`; I: unused; PID: unused.
    q11: f64,
    /// PI and I: the clamped `q`; PID: `φ`.
    q: f64,
    /// PID: `ε₁`.
    eps1: f64,
}

impl Controller {
    /// The controller for a method's table, or `None` when it has no
    /// error estimate.
    pub fn new(kind: ControllerKind, t: &Tableau) -> Option<Self> {
        let c: &ControllerConstants = t.controller.as_ref()?;
        Some(Controller {
            kind,
            beta1: c.beta1,
            beta2: c.beta2,
            beta3: c.pid_beta3,
            gamma: c.gamma,
            qmin: c.qmin,
            qmax: c.qmax,
            qmax_first_step: c.qmax_first_step,
            qsteady_min: c.qsteady_min,
            qsteady_max: c.qsteady_max,
            failfactor: c.failfactor,
            qoldinit: c.qoldinit,
            accept_safety: c.pid_accept_safety,
            order: t.order,
            embedded_order: t.embedded_order.unwrap_or(t.order),
        })
    }

    /// A fresh controller state (spec/methods.md §8.2: `E_old` starts at
    /// `qoldinit`; §8.4: `ε₂ = ε₃ = 1`).
    pub fn initial_state(&self) -> ControllerState {
        ControllerState {
            eold: self.qoldinit,
            pid_eps: [1.0, 1.0],
            accepted_any: false,
        }
    }

    fn qmax_now(&self, s: &ControllerState) -> f64 {
        if s.accepted_any {
            self.qmax
        } else {
            self.qmax_first_step
        }
    }

    /// Evaluates the controller on a finite scaled error `e` (spec/methods.md §8.1–§8.4).
    pub fn evaluate(&self, s: &ControllerState, e: f64) -> Evaluation {
        match self.kind {
            ControllerKind::PI => {
                // OrdinaryDiffEqCore src/integrators/controllers.jl:805-843:
                // q11 = E^β1; q = q11 / E_old^β2; q = clamp(q/γ, 1/qmax, 1/qmin).
                // spec/methods.md §8.2 writes q11/(E_old^β2 γ); we divide in
                // OrdinaryDiffEq's order, which the formula does not fix.
                let qmax = self.qmax_now(s);
                let q11 = e.powf(self.beta1);
                let mut q = q11 / s.eold.powf(self.beta2);
                q = (1.0 / qmax).max((1.0 / self.qmin).min(q / self.gamma));
                if e == 0.0 {
                    q = 1.0 / qmax;
                }
                Evaluation {
                    decision: if e <= 1.0 {
                        Decision::Accept
                    } else {
                        Decision::Reject
                    },
                    q11,
                    q,
                    eps1: 0.0,
                }
            }
            ControllerKind::I => {
                let qmax = self.qmax_now(s);
                let expo = 1.0 / (self.embedded_order as f64 + 1.0);
                let qtmp = e.powf(expo) / self.gamma;
                let mut q = (1.0 / qmax).max((1.0 / self.qmin).min(qtmp));
                if e == 0.0 {
                    q = 1.0 / qmax;
                }
                Evaluation {
                    decision: if e <= 1.0 {
                        Decision::Accept
                    } else {
                        Decision::Reject
                    },
                    q11: 0.0,
                    q,
                    eps1: 0.0,
                }
            }
            ControllerKind::PID => {
                let k = self.order.min(self.embedded_order) as f64 + 1.0;
                let ep = e.max(f64::EPSILON);
                let eps1 = 1.0 / ep;
                let mut phi = eps1.powf(self.beta1 / k)
                    * s.pid_eps[0].powf(self.beta2 / k)
                    * s.pid_eps[1].powf(self.beta3 / k);
                phi = 1.0 + (phi - 1.0).atan();
                Evaluation {
                    decision: if phi >= self.accept_safety {
                        Decision::Accept
                    } else {
                        Decision::Reject
                    },
                    q11: 0.0,
                    q: phi,
                    eps1,
                }
            }
        }
    }

    /// Updates the history after an accepted step and returns the proposed
    /// next size from `h_base`, the size of the step just taken (shortened if it
    /// landed on a stop; D61, spec/methods.md §8.6), before rounding and clamping.
    pub fn accept(&self, s: &mut ControllerState, ev: &Evaluation, e: f64, h_base: f64) -> f64 {
        let out = match self.kind {
            ControllerKind::PI | ControllerKind::I => {
                let mut q = ev.q;
                if self.qsteady_min <= q && q <= self.qsteady_max {
                    q = 1.0;
                }
                if self.kind == ControllerKind::PI {
                    s.eold = e.max(self.qoldinit);
                }
                h_base / q
            }
            ControllerKind::PID => {
                let mut phi = ev.q;
                let inv = 1.0 / phi;
                if self.qsteady_min <= inv && inv <= self.qsteady_max {
                    phi = 1.0;
                }
                s.pid_eps[1] = s.pid_eps[0];
                s.pid_eps[0] = ev.eps1;
                h_base * phi
            }
        };
        s.accepted_any = true;
        out
    }

    /// The new size after a step of size `h` was rejected by its error estimate.
    pub fn reject(&self, ev: &Evaluation, h: f64) -> f64 {
        match self.kind {
            ControllerKind::PI => h / (1.0 / self.qmin).min(ev.q11 / self.gamma),
            ControllerKind::I => h / ev.q,
            ControllerKind::PID => h * ev.q,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tableau::tableau;

    #[test]
    fn pi_accepts_and_grows() {
        let c = Controller::new(ControllerKind::PI, tableau("Tsit5").unwrap()).unwrap();
        let mut s = c.initial_state();
        let ev = c.evaluate(&s, 0.5);
        assert_eq!(ev.decision, Decision::Accept);
        let h = c.accept(&mut s, &ev, 0.5, 0.1);
        let q = (0.5f64.powf(0.14) / 1e-4f64.powf(0.08)) / 0.9;
        assert_eq!(h, 0.1 / q.clamp(1.0 / 10000.0, 5.0));
        assert_eq!(s.eold, 0.5);
        assert!(s.accepted_any);
        let ev = c.evaluate(&s, 4.0);
        assert_eq!(ev.decision, Decision::Reject);
        let hr = c.reject(&ev, 0.1);
        assert_eq!(hr, 0.1 / (5.0f64).min(4.0f64.powf(0.14) / 0.9));
    }

    #[test]
    fn pid_limiter() {
        let c = Controller::new(ControllerKind::PID, tableau("Tsit5").unwrap()).unwrap();
        let s = c.initial_state();
        let ev = c.evaluate(&s, 1e-30);
        assert!(ev.q < 1.0 + std::f64::consts::FRAC_PI_2);
        let ev = c.evaluate(&s, 1e30);
        assert_eq!(ev.decision, Decision::Reject);
    }
}
