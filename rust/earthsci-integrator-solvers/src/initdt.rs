//! The first step (spec/methods.md §9): OrdinaryDiffEqCore's
//! `_ode_initdt_oop` (`src/initdt.jl:354-468`), computed in f64.

use crate::norm::{rms_norm, Tolerances};
use crate::scalar::{eps_f64, next_up, Scalar};

/// Inputs of the first-step algorithm.
pub struct InitDtInput<'a, T> {
    pub u0: &'a [T],
    pub t0: f64,
    pub dtmin: f64,
    pub dtmax: f64,
    pub tol: &'a Tolerances,
    /// The leaf's DOFs, ascending.
    pub dofs: &'a [usize],
    /// The method's `order`.
    pub order: u32,
}

/// Computes the first step. `f(out, u, t)` overwrites `out` with the leaf's
/// right-hand side. Returns the step and the number of `f` evaluations (2,
/// or 1 when the first evaluation has a NaN).
pub fn initial_dt<T: Scalar>(
    inp: &InitDtInput<'_, T>,
    f: &mut dyn FnMut(&mut [T], &[T], f64),
) -> (f64, usize) {
    let n = inp.u0.len();
    let nd = inp.dofs.len();
    let u0 = inp.u0;
    // 1. sk_j = abstol_j + |u0_j| reltol.
    let sk: Vec<f64> = (0..n)
        .map(|j| inp.tol.abstol[j] + u0[j].to_f64().abs() * inp.tol.reltol)
        .collect();
    let d0 = rms_norm(inp.dofs.iter().map(|&j| u0[j].to_f64() / sk[j]), nd);
    let mut f0 = vec![T::zero(); n];
    f(&mut f0, u0, inp.t0);
    let d1 = rms_norm(inp.dofs.iter().map(|&j| f0[j].to_f64() / sk[j]), nd);
    // 2.
    let dtmin = next_up(inp.dtmin.max(eps_f64(inp.t0)));
    let smalldt = dtmin.max(1e-6);
    // 3.
    if inp.dofs.iter().any(|&j| f0[j].to_f64().is_nan()) || d1.is_nan() {
        return (dtmin, 1);
    }
    // 4. OrdinaryDiffEq computes (d0/d1)/100; spec/methods.md §9 writes
    // d0/(100 d1). We follow OrdinaryDiffEq's arithmetic order.
    let mut dt0 = if d0 < 1e-5 || d1 < 1e-5 {
        smalldt
    } else {
        (d0 / d1) / 100.0
    };
    dt0 = dt0.min(inp.dtmax);
    // 5.
    let h0 = T::from_f64(dt0);
    let u1: Vec<T> = (0..n).map(|j| u0[j] + h0 * f0[j]).collect();
    let mut f1 = vec![T::zero(); n];
    f(&mut f1, &u1, inp.t0 + dt0);
    if f0 == f1 {
        return (dtmin.max(100.0 * dt0), 2);
    }
    // 6.
    let d2 = rms_norm(
        inp.dofs
            .iter()
            .map(|&j| (f1[j].to_f64() - f0[j].to_f64()) / sk[j]),
        nd,
    ) / dt0;
    let m = d1.max(d2);
    let dt1 = if m <= 1e-15 {
        1e-6f64.max(dt0 * 1e-3)
    } else {
        10.0f64.powf(-(2.0 + m.log10()) / inp.order as f64)
    };
    // 7.
    (dtmin.max((100.0 * dt0).min(dt1).min(inp.dtmax)), 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_first_step_is_reasonable() {
        let tol = Tolerances {
            abstol: vec![1e-8],
            reltol: 1e-6,
        };
        let u0 = [1.0f64];
        let inp = InitDtInput {
            u0: &u0,
            t0: 0.0,
            dtmin: 0.0,
            dtmax: 10.0,
            tol: &tol,
            dofs: &[0],
            order: 5,
        };
        let (h, nf) = initial_dt(&inp, &mut |out: &mut [f64], u: &[f64], _t| out[0] = -u[0]);
        assert_eq!(nf, 2);
        assert!(h > 1e-4 && h < 1.0, "{h}");
    }
}
