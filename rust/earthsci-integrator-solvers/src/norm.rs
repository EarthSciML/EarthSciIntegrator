//! The error norm (spec/methods.md §7).

use crate::scalar::Scalar;

/// Error tolerances of a leaf, per DOF of the full state vector.
#[derive(Debug, Clone)]
pub struct Tolerances {
    /// `abstol_j`, indexed by full-state DOF.
    pub abstol: Vec<f64>,
    /// `reltol`.
    pub reltol: f64,
}

/// The scaled RMS error norm of spec/methods.md §7:
/// `sc_j = abstol_j + reltol·max(|u_{n,j}|, |u_{n+1,j}|)`,
/// `E = sqrt((1/N) Σ_j (e_j/sc_j)^2)`, over the leaf's DOFs `dofs` in
/// ascending order.
///
/// Everything is computed in f64 (D13), summed sequentially in DOF order
/// with no reordering or fused multiply-add (D10). `e` is indexed by full-state DOF.
pub fn error_norm<T: Scalar>(
    e: &[f64],
    un: &[T],
    unew: &[T],
    tol: &Tolerances,
    dofs: &[usize],
) -> f64 {
    let mut sum = 0.0f64;
    for &j in dofs {
        let m = un[j].to_f64().abs().max(unew[j].to_f64().abs());
        let sc = tol.abstol[j] + tol.reltol * m;
        let r = e[j] / sc;
        sum += r * r;
    }
    (sum / dofs.len().max(1) as f64).sqrt()
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
    fn norm_of_one_component() {
        let tol = Tolerances {
            abstol: vec![1e-6, 1.0],
            reltol: 1e-3,
        };
        let un = [1.0f64, 5.0];
        let unew = [2.0f64, 5.0];
        let e = [1e-3, 123.0];
        // Only DOF 0 is in the leaf.
        let n = error_norm(&e, &un, &unew, &tol, &[0]);
        let sc = 1e-6 + 1e-3 * 2.0;
        assert_eq!(n, ((1e-3 / sc) * (1e-3 / sc) / 1.0f64).sqrt());
    }
}
