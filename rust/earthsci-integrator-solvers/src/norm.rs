//! The error norm (spec/methods.md §7).

use crate::scalar::Scalar;

/// Error tolerances of a leaf, per entry of its state vector.
#[derive(Debug, Clone)]
pub struct Tolerances {
    /// `abstol_j`, indexed by the leaf's state vector.
    pub abstol: Vec<f64>,
    /// `reltol`.
    pub reltol: f64,
}

/// The scaled RMS error norm of spec/methods.md §7:
/// `sc_j = abstol_j + reltol·max(|u_{n,j}|, |u_{n+1,j}|)`,
/// `E = sqrt((1/N) Σ_j (e_j/sc_j)^2)`, over every entry of the leaf's
/// state vector in ascending order.
///
/// Everything is computed in f64 (D13), summed sequentially in DOF order
/// with no reordering or fused multiply-add (D10). `e`, `un`, `unew` and the
/// tolerances are indexed alike, by the leaf's state vector.
pub fn error_norm<T: Scalar>(e: &[f64], un: &[T], unew: &[T], tol: &Tolerances) -> f64 {
    let mut sum = 0.0f64;
    for (((&ej, u0), u1), &at) in e.iter().zip(un).zip(unew).zip(&tol.abstol) {
        let m = u0.to_f64().abs().max(u1.to_f64().abs());
        let sc = at + tol.reltol * m;
        let r = ej / sc;
        sum += r * r;
    }
    (sum / un.len().max(1) as f64).sqrt()
}

/// The plain RMS norm `sqrt((1/N) Σ x_j^2)` over `dofs`, used by the first-step
/// algorithm (spec/methods.md §9).
pub fn rms_norm(x: impl Iterator<Item = f64>, n: usize) -> f64 {
    let mut sum = 0.0f64;
    for v in x {
        sum += v * v;
    }
    (sum / n.max(1) as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_of_two_components() {
        let tol = Tolerances {
            abstol: vec![1e-6, 1.0],
            reltol: 1e-3,
        };
        let un = [1.0f64, 5.0];
        let unew = [2.0f64, 5.0];
        let e = [1e-3, 0.5];
        let n = error_norm(&e, &un, &unew, &tol);
        let sc0 = 1e-6 + 1e-3 * 2.0;
        let sc1 = 1.0 + 1e-3 * 5.0;
        let sum = (1e-3 / sc0) * (1e-3 / sc0) + (0.5 / sc1) * (0.5 / sc1);
        assert_eq!(n, (sum / 2.0f64).sqrt());
    }
}
