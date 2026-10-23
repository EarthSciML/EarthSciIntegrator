//! The hand-written toy problems of spec/toy-problems.md §1 (D53).
//!
//! Every call of spec/operator-interface.md §3 is implemented with
//! hand-derived derivatives. Arithmetic is done in f64 and each result is
//! rounded to `T` as it is accumulated.

use crate::operator::{Call, MapOperator, MatrixStorage, MuPair, Operator, Structure};
use crate::{Error, Scalar};
use std::f64::consts::PI;
use std::sync::Arc;

/// A toy problem: its operators, layout, parameters, initial state,
/// interval and the facts the conformance checks use.
pub struct ToyProblem<T: Scalar> {
    /// `<problem>` or `<problem>/<variant>`.
    pub name: String,
    /// Length of the state vector.
    pub n: usize,
    /// Parameter names, in `p` order.
    pub param_names: Vec<&'static str>,
    /// Parameter values.
    pub p: Vec<f64>,
    /// Initial state.
    pub u0: Vec<f64>,
    /// The problem's interval.
    pub tspan: (f64, f64),
    /// Variable name of each DOF.
    pub variables: Vec<&'static str>,
    /// Cell index of each DOF (0-based) for problems with the index set `cell`.
    pub cells: Option<Vec<usize>>,
    /// Term operators by name.
    pub operators: Vec<Arc<dyn Operator<T>>>,
    /// Map operators by name.
    pub maps: Vec<Arc<dyn MapOperator<T>>>,
    /// Linear invariants `Σ_j w_j u_j`, as `(DOF, weight)` lists.
    pub invariants: Vec<Vec<(usize, f64)>>,
}

impl<T: Scalar> ToyProblem<T> {
    /// The term operator named `name`.
    pub fn operator(&self, name: &str) -> Option<Arc<dyn Operator<T>>> {
        self.operators.iter().find(|o| o.name() == name).cloned()
    }
    /// The map operator named `name`.
    pub fn map(&self, name: &str) -> Option<Arc<dyn MapOperator<T>>> {
        self.maps.iter().find(|o| o.name() == name).cloned()
    }
}

/// Builds the toy problem a case's `model: "toy:<problem>[/<variant>]"` names.
pub fn toy_problem<T: Scalar>(model: &str) -> Result<ToyProblem<T>, Error> {
    let name = model
        .strip_prefix("toy:")
        .ok_or_else(|| Error::Config(format!("model {model:?} is not a toy problem")))?;
    let pb = match name {
        "forced_decay" => ToyProblem {
            name: name.into(),
            n: 1,
            param_names: vec!["k", "omega"],
            p: vec![2.0, 3.0],
            u0: vec![1.0],
            tspan: (0.0, 2.0),
            variables: vec!["u"],
            cells: None,
            operators: vec![Arc::new(Decay)],
            maps: vec![],
            invariants: vec![],
        },
        "van_der_pol/nonstiff" | "van_der_pol/stiff" => {
            let stiff = name.ends_with("stiff") && !name.ends_with("nonstiff");
            ToyProblem {
                name: name.into(),
                n: 2,
                param_names: vec!["mu"],
                p: vec![if stiff { 1000.0 } else { 1.0 }],
                u0: vec![2.0, 0.0],
                tspan: (0.0, if stiff { 1.0 } else { 6.3 }),
                variables: vec!["y", "z"],
                cells: None,
                operators: vec![Arc::new(Oscillator)],
                maps: vec![],
                invariants: vec![],
            }
        }
        "robertson" => ToyProblem {
            name: name.into(),
            n: 3,
            param_names: vec!["k1", "k2", "k3"],
            p: vec![0.04, 3e7, 1e4],
            u0: vec![1.0, 0.0, 0.0],
            tspan: (0.0, 40.0),
            variables: vec!["y1", "y2", "y3"],
            cells: None,
            operators: vec![Arc::new(Kinetics)],
            maps: vec![],
            invariants: vec![vec![(0, 1.0), (1, 1.0), (2, 1.0)]],
        },
        "photostationary_cells" => {
            let nc = PHOTO_NCELLS;
            let mut u0 = vec![0.0; 3 * nc];
            let mut variables = Vec::new();
            let mut cells = Vec::new();
            for (s, (v, x0)) in [("NO2", 10.0), ("NO", 5.0), ("O3", 40.0)]
                .into_iter()
                .enumerate()
            {
                for c in 0..nc {
                    u0[s * nc + c] = x0;
                    variables.push(v);
                    cells.push(c);
                }
            }
            let mut invariants = Vec::new();
            for c in 0..nc {
                invariants.push(vec![(c, 1.0), (nc + c, 1.0)]);
                invariants.push(vec![(c, 1.0), (2 * nc + c, 1.0)]);
            }
            ToyProblem {
                name: name.into(),
                n: 3 * nc,
                param_names: vec!["j0", "k", "omega"],
                p: vec![1e-2, 4.4e-4, 2.0 * PI / 3600.0],
                u0,
                tspan: (0.0, 3600.0),
                variables,
                cells: Some(cells),
                operators: vec![Arc::new(Photochemistry)],
                maps: vec![],
                invariants,
            }
        }
        "advection_reaction" => {
            let nc = ADV_NCELLS;
            let mut u0 = vec![0.0; 2 * nc];
            let mut variables = Vec::new();
            let mut cells = Vec::new();
            for c in 0..nc {
                let x = (c as f64 + 0.5) / nc as f64;
                u0[c] = 1.0 + 0.5 * (2.0 * PI * x).sin();
            }
            for v in ["A", "B"] {
                for c in 0..nc {
                    variables.push(v);
                    cells.push(c);
                }
            }
            ToyProblem {
                name: name.into(),
                n: 2 * nc,
                param_names: vec!["v", "k1", "k2"],
                p: vec![1.0, 50.0, 25.0],
                u0,
                tspan: (0.0, 0.5),
                variables,
                cells: Some(cells),
                operators: vec![Arc::new(Transport), Arc::new(Reaction)],
                maps: vec![],
                invariants: vec![(0..2 * nc).map(|j| (j, 1.0)).collect()],
            }
        }
        "source_decay_map" => ToyProblem {
            name: name.into(),
            n: 1,
            param_names: vec!["a", "r"],
            p: vec![1.0, 0.5],
            u0: vec![0.0],
            tspan: (0.0, 4.0),
            variables: vec!["u"],
            cells: None,
            operators: vec![Arc::new(Source)],
            maps: vec![Arc::new(DecayMap)],
            invariants: vec![],
        },
        _ => {
            return Err(Error::Config(format!(
                "unknown toy problem {model:?} (spec/toy-problems.md §1)"
            )))
        }
    };
    Ok(pb)
}

/// `forced_decay`'s exact solution (spec/toy-problems.md §1.1) for
/// `p = (k, ω)` and `u(0) = u0`.
pub fn forced_decay_exact(t: f64, u0: f64, p: &[f64]) -> f64 {
    let (k, w) = (p[0], p[1]);
    let a = |t: f64| (k * (w * t).sin() - w * (w * t).cos()) / (k * k + w * w);
    (u0 - a(0.0)) * (-k * t).exp() + a(t)
}

/// `source_decay_map`'s exact discrete solution (spec/toy-problems.md §1.6):
/// `n` macro steps of size `dt` from `u0`.
pub fn source_decay_map_exact(u0: f64, p: &[f64], dt: f64, n: usize) -> f64 {
    let (a, r) = (p[0], p[1]);
    let mut u = u0;
    for _ in 0..n {
        u = (u + a * dt) * (-r * dt).exp();
    }
    u
}

#[inline]
fn f<T: Scalar>(x: T) -> f64 {
    x.to_f64()
}

#[inline]
fn acc<T: Scalar>(out: &mut T, v: f64) {
    *out += T::from_f64(v);
}

const ALL_CALLS: [Call; 10] = [
    Call::Rhs,
    Call::JvpU,
    Call::VjpU,
    Call::VjpP,
    Call::Jac,
    Call::Ft,
    Call::HessU,
    Call::HessP,
    Call::FtVjpU,
    Call::FtVjpP,
];

// ---------------------------------------------------------------- forced_decay

/// `decay`: `u' = −k u + sin(ω t)`, `p = (k, ω)`.
pub struct Decay;

impl<T: Scalar> Operator<T> for Decay {
    fn name(&self) -> &str {
        "decay"
    }
    fn structure(&self) -> Structure {
        Structure::Dense
    }
    fn depends_on_t(&self) -> bool {
        true
    }
    fn linear_in_u(&self) -> bool {
        true
    }
    fn writes(&self) -> Vec<usize> {
        vec![0]
    }
    fn provides(&self, call: Call) -> bool {
        ALL_CALLS.contains(&call)
    }
    fn rhs(&self, out: &mut [T], u: &[T], p: &[f64], t: f64) {
        acc(&mut out[0], -p[0] * f(u[0]) + (p[1] * t).sin());
    }
    fn jvp_u(&self, out: &mut [T], _u: &[T], p: &[f64], _t: f64, v: &[T]) {
        acc(&mut out[0], -p[0] * f(v[0]));
    }
    fn vjp_u(&self, out: &mut [T], _u: &[T], p: &[f64], _t: f64, w: &[T]) {
        acc(&mut out[0], -p[0] * f(w[0]));
    }
    fn vjp_p(&self, out: &mut [f64], u: &[T], p: &[f64], t: f64, w: &[T]) {
        out[0] += -f(u[0]) * f(w[0]);
        out[1] += t * (p[1] * t).cos() * f(w[0]);
    }
    fn jac(&self, out: &mut MatrixStorage<T>, _u: &[T], p: &[f64], _t: f64) {
        out.add(0, 0, T::from_f64(-p[0]));
    }
    fn ft(&self, out: &mut [T], _u: &[T], p: &[f64], t: f64) {
        acc(&mut out[0], p[1] * (p[1] * t).cos());
    }
    fn hess_u(&self, _out: &mut [T], _u: &[T], _p: &[f64], _t: f64, _mu: &[MuPair<'_, T>]) {}
    fn hess_p(&self, out: &mut [f64], _u: &[T], _p: &[f64], _t: f64, mu: &[MuPair<'_, T>]) {
        // ∇_k Σ μ_i (−k) w_i = −Σ μ_i w_i.
        for (m, w) in mu {
            out[0] += -f(m[0]) * f(w[0]);
        }
    }
    fn ft_vjp_u(&self, _out: &mut [T], _u: &[T], _p: &[f64], _t: f64, _nu: &[T]) {}
    fn ft_vjp_p(&self, out: &mut [f64], _u: &[T], p: &[f64], t: f64, nu: &[T]) {
        // ν ω cos(ω t): ∂/∂ω = ν (cos ωt − ω t sin ωt).
        let w = p[1];
        out[1] += f(nu[0]) * ((w * t).cos() - w * t * (w * t).sin());
    }
}

// ---------------------------------------------------------------- van_der_pol

/// `oscillator`: `y' = z`, `z' = μ(1 − y²)z − y`, `p = (μ)`.
pub struct Oscillator;

impl<T: Scalar> Operator<T> for Oscillator {
    fn name(&self) -> &str {
        "oscillator"
    }
    fn structure(&self) -> Structure {
        Structure::Dense
    }
    fn depends_on_t(&self) -> bool {
        false
    }
    fn linear_in_u(&self) -> bool {
        false
    }
    fn writes(&self) -> Vec<usize> {
        vec![0, 1]
    }
    fn provides(&self, call: Call) -> bool {
        ALL_CALLS.contains(&call)
    }
    fn rhs(&self, out: &mut [T], u: &[T], p: &[f64], _t: f64) {
        let (y, z, mu) = (f(u[0]), f(u[1]), p[0]);
        acc(&mut out[0], z);
        acc(&mut out[1], mu * (1.0 - y * y) * z - y);
    }
    fn jvp_u(&self, out: &mut [T], u: &[T], p: &[f64], _t: f64, v: &[T]) {
        let (y, z, mu) = (f(u[0]), f(u[1]), p[0]);
        acc(&mut out[0], f(v[1]));
        acc(
            &mut out[1],
            (-2.0 * mu * y * z - 1.0) * f(v[0]) + mu * (1.0 - y * y) * f(v[1]),
        );
    }
    fn vjp_u(&self, out: &mut [T], u: &[T], p: &[f64], _t: f64, w: &[T]) {
        let (y, z, mu) = (f(u[0]), f(u[1]), p[0]);
        acc(&mut out[0], (-2.0 * mu * y * z - 1.0) * f(w[1]));
        acc(&mut out[1], f(w[0]) + mu * (1.0 - y * y) * f(w[1]));
    }
    fn vjp_p(&self, out: &mut [f64], u: &[T], _p: &[f64], _t: f64, w: &[T]) {
        let (y, z) = (f(u[0]), f(u[1]));
        out[0] += (1.0 - y * y) * z * f(w[1]);
    }
    fn jac(&self, out: &mut MatrixStorage<T>, u: &[T], p: &[f64], _t: f64) {
        let (y, z, mu) = (f(u[0]), f(u[1]), p[0]);
        out.add(0, 1, T::from_f64(1.0));
        out.add(1, 0, T::from_f64(-2.0 * mu * y * z - 1.0));
        out.add(1, 1, T::from_f64(mu * (1.0 - y * y)));
    }
    fn ft(&self, _out: &mut [T], _u: &[T], _p: &[f64], _t: f64) {}
    fn hess_u(&self, out: &mut [T], u: &[T], p: &[f64], _t: f64, mu: &[MuPair<'_, T>]) {
        // g = Σ_i [m0 w1 + m1((−2μyz − 1) w0 + μ(1 − y²) w1)].
        let (y, z, m) = (f(u[0]), f(u[1]), p[0]);
        let (mut gy, mut gz) = (0.0, 0.0);
        for (mi, wi) in mu {
            let (m1, w0, w1) = (f(mi[1]), f(wi[0]), f(wi[1]));
            gy += m1 * (-2.0 * m * z * w0 - 2.0 * m * y * w1);
            gz += m1 * (-2.0 * m * y * w0);
        }
        acc(&mut out[0], gy);
        acc(&mut out[1], gz);
    }
    fn hess_p(&self, out: &mut [f64], u: &[T], _p: &[f64], _t: f64, mu: &[MuPair<'_, T>]) {
        let (y, z) = (f(u[0]), f(u[1]));
        for (mi, wi) in mu {
            let (m1, w0, w1) = (f(mi[1]), f(wi[0]), f(wi[1]));
            out[0] += m1 * (-2.0 * y * z * w0 + (1.0 - y * y) * w1);
        }
    }
    fn ft_vjp_u(&self, _out: &mut [T], _u: &[T], _p: &[f64], _t: f64, _nu: &[T]) {}
    fn ft_vjp_p(&self, _out: &mut [f64], _u: &[T], _p: &[f64], _t: f64, _nu: &[T]) {}
}

// ---------------------------------------------------------------- robertson

/// `kinetics`: Robertson's three-species kinetics, `p = (k1, k2, k3)`.
pub struct Kinetics;

impl<T: Scalar> Operator<T> for Kinetics {
    fn name(&self) -> &str {
        "kinetics"
    }
    fn structure(&self) -> Structure {
        Structure::Dense
    }
    fn depends_on_t(&self) -> bool {
        false
    }
    fn linear_in_u(&self) -> bool {
        false
    }
    fn writes(&self) -> Vec<usize> {
        vec![0, 1, 2]
    }
    fn provides(&self, call: Call) -> bool {
        ALL_CALLS.contains(&call)
    }
    fn rhs(&self, out: &mut [T], u: &[T], p: &[f64], _t: f64) {
        let (y1, y2, y3) = (f(u[0]), f(u[1]), f(u[2]));
        let (k1, k2, k3) = (p[0], p[1], p[2]);
        acc(&mut out[0], -k1 * y1 + k3 * y2 * y3);
        acc(&mut out[1], k1 * y1 - k3 * y2 * y3 - k2 * y2 * y2);
        acc(&mut out[2], k2 * y2 * y2);
    }
    fn jvp_u(&self, out: &mut [T], u: &[T], p: &[f64], _t: f64, v: &[T]) {
        let (_, y2, y3) = (f(u[0]), f(u[1]), f(u[2]));
        let (k1, k2, k3) = (p[0], p[1], p[2]);
        let (v1, v2, v3) = (f(v[0]), f(v[1]), f(v[2]));
        acc(&mut out[0], -k1 * v1 + k3 * y3 * v2 + k3 * y2 * v3);
        acc(
            &mut out[1],
            k1 * v1 + (-k3 * y3 - 2.0 * k2 * y2) * v2 - k3 * y2 * v3,
        );
        acc(&mut out[2], 2.0 * k2 * y2 * v2);
    }
    fn vjp_u(&self, out: &mut [T], u: &[T], p: &[f64], _t: f64, w: &[T]) {
        let (_, y2, y3) = (f(u[0]), f(u[1]), f(u[2]));
        let (k1, k2, k3) = (p[0], p[1], p[2]);
        let (w1, w2, w3) = (f(w[0]), f(w[1]), f(w[2]));
        acc(&mut out[0], -k1 * w1 + k1 * w2);
        acc(
            &mut out[1],
            k3 * y3 * w1 + (-k3 * y3 - 2.0 * k2 * y2) * w2 + 2.0 * k2 * y2 * w3,
        );
        acc(&mut out[2], k3 * y2 * w1 - k3 * y2 * w2);
    }
    fn vjp_p(&self, out: &mut [f64], u: &[T], _p: &[f64], _t: f64, w: &[T]) {
        let (y1, y2, y3) = (f(u[0]), f(u[1]), f(u[2]));
        let (w1, w2, w3) = (f(w[0]), f(w[1]), f(w[2]));
        out[0] += -y1 * w1 + y1 * w2;
        out[1] += -y2 * y2 * w2 + y2 * y2 * w3;
        out[2] += y2 * y3 * w1 - y2 * y3 * w2;
    }
    fn jac(&self, out: &mut MatrixStorage<T>, u: &[T], p: &[f64], _t: f64) {
        let (_, y2, y3) = (f(u[0]), f(u[1]), f(u[2]));
        let (k1, k2, k3) = (p[0], p[1], p[2]);
        let e = |v: f64| T::from_f64(v);
        out.add(0, 0, e(-k1));
        out.add(0, 1, e(k3 * y3));
        out.add(0, 2, e(k3 * y2));
        out.add(1, 0, e(k1));
        out.add(1, 1, e(-k3 * y3 - 2.0 * k2 * y2));
        out.add(1, 2, e(-k3 * y2));
        out.add(2, 1, e(2.0 * k2 * y2));
    }
    fn ft(&self, _out: &mut [T], _u: &[T], _p: &[f64], _t: f64) {}
    fn hess_u(&self, out: &mut [T], u: &[T], p: &[f64], _t: f64, mu: &[MuPair<'_, T>]) {
        let _ = u;
        let (k2, k3) = (p[1], p[2]);
        let (mut g2, mut g3) = (0.0, 0.0);
        for (mi, wi) in mu {
            let (m1, m2, m3) = (f(mi[0]), f(mi[1]), f(mi[2]));
            let (a2, a3) = (f(wi[1]), f(wi[2]));
            g2 += m1 * k3 * a3 + m2 * (-2.0 * k2 * a2 - k3 * a3) + m3 * 2.0 * k2 * a2;
            g3 += m1 * k3 * a2 - m2 * k3 * a2;
        }
        acc(&mut out[1], g2);
        acc(&mut out[2], g3);
    }
    fn hess_p(&self, out: &mut [f64], u: &[T], _p: &[f64], _t: f64, mu: &[MuPair<'_, T>]) {
        let (y2, y3) = (f(u[1]), f(u[2]));
        for (mi, wi) in mu {
            let (m1, m2, m3) = (f(mi[0]), f(mi[1]), f(mi[2]));
            let (a1, a2, a3) = (f(wi[0]), f(wi[1]), f(wi[2]));
            out[0] += -m1 * a1 + m2 * a1;
            out[1] += -2.0 * m2 * y2 * a2 + 2.0 * m3 * y2 * a2;
            out[2] += m1 * (y3 * a2 + y2 * a3) - m2 * (y3 * a2 + y2 * a3);
        }
    }
    fn ft_vjp_u(&self, _out: &mut [T], _u: &[T], _p: &[f64], _t: f64, _nu: &[T]) {}
    fn ft_vjp_p(&self, _out: &mut [f64], _u: &[T], _p: &[f64], _t: f64, _nu: &[T]) {}
}

// ---------------------------------------------------------------- photostationary_cells

/// Number of cells of `photostationary_cells`.
pub const PHOTO_NCELLS: usize = 16;

/// `photochemistry`: `r_c = j_c NO2 − k NO O3`, `NO2' = −r`, `NO' = r`,
/// `O3' = r`, with `j_c(t) = ½ j0 (1 + sin(ω t + φ_c))`, `p = (j0, k, ω)`.
pub struct Photochemistry;

/// Sign of `r` in each species' tendency.
const PHOTO_SGN: [f64; 3] = [-1.0, 1.0, 1.0];

impl Photochemistry {
    fn phase(c: usize) -> f64 {
        2.0 * PI * c as f64 / PHOTO_NCELLS as f64
    }
    fn jc(p: &[f64], t: f64, c: usize) -> f64 {
        0.5 * p[0] * (1.0 + (p[2] * t + Self::phase(c)).sin())
    }
    fn cell<T: Scalar>(u: &[T], c: usize) -> (f64, f64, f64) {
        let nc = PHOTO_NCELLS;
        (f(u[c]), f(u[nc + c]), f(u[2 * nc + c]))
    }
    /// `Σ_s sgn_s x_s` in cell `c`.
    fn signed<T: Scalar>(x: &[T], c: usize) -> f64 {
        let nc = PHOTO_NCELLS;
        PHOTO_SGN[0] * f(x[c]) + PHOTO_SGN[1] * f(x[nc + c]) + PHOTO_SGN[2] * f(x[2 * nc + c])
    }
}

impl<T: Scalar> Operator<T> for Photochemistry {
    fn name(&self) -> &str {
        "photochemistry"
    }
    fn structure(&self) -> Structure {
        Structure::BlockDiagonal {
            ncells: PHOTO_NCELLS,
            n: 3,
        }
    }
    fn depends_on_t(&self) -> bool {
        true
    }
    fn linear_in_u(&self) -> bool {
        false
    }
    fn writes(&self) -> Vec<usize> {
        (0..3 * PHOTO_NCELLS).collect()
    }
    fn provides(&self, call: Call) -> bool {
        ALL_CALLS.contains(&call)
    }
    fn rhs(&self, out: &mut [T], u: &[T], p: &[f64], t: f64) {
        let nc = PHOTO_NCELLS;
        for c in 0..nc {
            let (no2, no, o3) = Self::cell(u, c);
            let r = Self::jc(p, t, c) * no2 - p[1] * no * o3;
            for (s, sg) in PHOTO_SGN.iter().enumerate() {
                acc(&mut out[s * nc + c], sg * r);
            }
        }
    }
    fn jvp_u(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, v: &[T]) {
        let nc = PHOTO_NCELLS;
        for c in 0..nc {
            let (_, no, o3) = Self::cell(u, c);
            let (v0, v1, v2) = Self::cell(v, c);
            let dr = Self::jc(p, t, c) * v0 - p[1] * o3 * v1 - p[1] * no * v2;
            for (s, sg) in PHOTO_SGN.iter().enumerate() {
                acc(&mut out[s * nc + c], sg * dr);
            }
        }
    }
    fn vjp_u(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, w: &[T]) {
        let nc = PHOTO_NCELLS;
        for c in 0..nc {
            let (_, no, o3) = Self::cell(u, c);
            let ws = Self::signed(w, c);
            acc(&mut out[c], Self::jc(p, t, c) * ws);
            acc(&mut out[nc + c], -p[1] * o3 * ws);
            acc(&mut out[2 * nc + c], -p[1] * no * ws);
        }
    }
    fn vjp_p(&self, out: &mut [f64], u: &[T], p: &[f64], t: f64, w: &[T]) {
        let (mut g0, mut g1, mut g2) = (0.0, 0.0, 0.0);
        for c in 0..PHOTO_NCELLS {
            let (no2, no, o3) = Self::cell(u, c);
            let ws = Self::signed(w, c);
            let arg = p[2] * t + Self::phase(c);
            g0 += ws * 0.5 * (1.0 + arg.sin()) * no2;
            g1 += ws * (-no * o3);
            g2 += ws * 0.5 * p[0] * arg.cos() * t * no2;
        }
        out[0] += g0;
        out[1] += g1;
        out[2] += g2;
    }
    fn jac(&self, out: &mut MatrixStorage<T>, u: &[T], p: &[f64], t: f64) {
        let nc = PHOTO_NCELLS;
        for c in 0..nc {
            let (_, no, o3) = Self::cell(u, c);
            let dr = [Self::jc(p, t, c), -p[1] * o3, -p[1] * no];
            for (i, sg) in PHOTO_SGN.iter().enumerate() {
                for (j, d) in dr.iter().enumerate() {
                    out.add(i * nc + c, j * nc + c, T::from_f64(sg * d));
                }
            }
        }
    }
    fn ft(&self, out: &mut [T], u: &[T], p: &[f64], t: f64) {
        let nc = PHOTO_NCELLS;
        for c in 0..nc {
            let (no2, _, _) = Self::cell(u, c);
            let drdt = 0.5 * p[0] * p[2] * (p[2] * t + Self::phase(c)).cos() * no2;
            for (s, sg) in PHOTO_SGN.iter().enumerate() {
                acc(&mut out[s * nc + c], sg * drdt);
            }
        }
    }
    fn hess_u(&self, out: &mut [T], _u: &[T], p: &[f64], _t: f64, mu: &[MuPair<'_, T>]) {
        let nc = PHOTO_NCELLS;
        for c in 0..nc {
            let (mut gno, mut go3) = (0.0, 0.0);
            for (mi, wi) in mu {
                let m = Self::signed(mi, c);
                let (_, a_no, a_o3) = Self::cell(wi, c);
                gno += m * (-p[1] * a_o3);
                go3 += m * (-p[1] * a_no);
            }
            acc(&mut out[nc + c], gno);
            acc(&mut out[2 * nc + c], go3);
        }
    }
    fn hess_p(&self, out: &mut [f64], u: &[T], p: &[f64], t: f64, mu: &[MuPair<'_, T>]) {
        let (mut g0, mut g1, mut g2) = (0.0, 0.0, 0.0);
        for c in 0..PHOTO_NCELLS {
            let (_, no, o3) = Self::cell(u, c);
            let arg = p[2] * t + Self::phase(c);
            for (mi, wi) in mu {
                let m = Self::signed(mi, c);
                let (a_no2, a_no, a_o3) = Self::cell(wi, c);
                g0 += m * 0.5 * (1.0 + arg.sin()) * a_no2;
                g1 += m * (-o3 * a_no - no * a_o3);
                g2 += m * 0.5 * p[0] * arg.cos() * t * a_no2;
            }
        }
        out[0] += g0;
        out[1] += g1;
        out[2] += g2;
    }
    fn ft_vjp_u(&self, out: &mut [T], _u: &[T], p: &[f64], t: f64, nu: &[T]) {
        for c in 0..PHOTO_NCELLS {
            let n = Self::signed(nu, c);
            let arg = p[2] * t + Self::phase(c);
            acc(&mut out[c], n * 0.5 * p[0] * p[2] * arg.cos());
        }
    }
    fn ft_vjp_p(&self, out: &mut [f64], u: &[T], p: &[f64], t: f64, nu: &[T]) {
        let (mut g0, mut g2) = (0.0, 0.0);
        for c in 0..PHOTO_NCELLS {
            let (no2, _, _) = Self::cell(u, c);
            let n = Self::signed(nu, c);
            let arg = p[2] * t + Self::phase(c);
            g0 += n * 0.5 * p[2] * arg.cos() * no2;
            g2 += n * 0.5 * p[0] * (arg.cos() - p[2] * t * arg.sin()) * no2;
        }
        out[0] += g0;
        out[2] += g2;
    }
}

// ---------------------------------------------------------------- advection_reaction

/// Number of cells of `advection_reaction`.
pub const ADV_NCELLS: usize = 64;

/// `transport`: first-order upwind advection of `A`,
/// `A_c' = −v (A_c − A_{c−1})/Δx`, periodic, `p = (v, k1, k2)`.
pub struct Transport;

impl Transport {
    fn dx() -> f64 {
        1.0 / ADV_NCELLS as f64
    }
    fn left(c: usize) -> usize {
        (c + ADV_NCELLS - 1) % ADV_NCELLS
    }
}

impl<T: Scalar> Operator<T> for Transport {
    fn name(&self) -> &str {
        "transport"
    }
    fn structure(&self) -> Structure {
        Structure::Dense
    }
    fn depends_on_t(&self) -> bool {
        false
    }
    fn linear_in_u(&self) -> bool {
        true
    }
    fn writes(&self) -> Vec<usize> {
        (0..ADV_NCELLS).collect()
    }
    fn provides(&self, call: Call) -> bool {
        ALL_CALLS.contains(&call)
    }
    fn rhs(&self, out: &mut [T], u: &[T], p: &[f64], _t: f64) {
        for c in 0..ADV_NCELLS {
            let d = f(u[c]) - f(u[Self::left(c)]);
            acc(&mut out[c], -p[0] * d / Self::dx());
        }
    }
    fn jvp_u(&self, out: &mut [T], _u: &[T], p: &[f64], _t: f64, v: &[T]) {
        for c in 0..ADV_NCELLS {
            let d = f(v[c]) - f(v[Self::left(c)]);
            acc(&mut out[c], -p[0] * d / Self::dx());
        }
    }
    fn vjp_u(&self, out: &mut [T], _u: &[T], p: &[f64], _t: f64, w: &[T]) {
        // (Jᵀw)_c = (v/Δx)(−w_c + w_{c+1}).
        let s = p[0] / Self::dx();
        for c in 0..ADV_NCELLS {
            let right = (c + 1) % ADV_NCELLS;
            acc(&mut out[c], s * (f(w[right]) - f(w[c])));
        }
    }
    fn vjp_p(&self, out: &mut [f64], u: &[T], _p: &[f64], _t: f64, w: &[T]) {
        let mut g = 0.0;
        for c in 0..ADV_NCELLS {
            let d = f(u[c]) - f(u[Self::left(c)]);
            g += f(w[c]) * (-d / Self::dx());
        }
        out[0] += g;
    }
    fn jac(&self, out: &mut MatrixStorage<T>, _u: &[T], p: &[f64], _t: f64) {
        let s = p[0] / Self::dx();
        for c in 0..ADV_NCELLS {
            out.add(c, c, T::from_f64(-s));
            out.add(c, Self::left(c), T::from_f64(s));
        }
    }
    fn ft(&self, _out: &mut [T], _u: &[T], _p: &[f64], _t: f64) {}
    fn hess_u(&self, _out: &mut [T], _u: &[T], _p: &[f64], _t: f64, _mu: &[MuPair<'_, T>]) {}
    fn hess_p(&self, out: &mut [f64], _u: &[T], _p: &[f64], _t: f64, mu: &[MuPair<'_, T>]) {
        let mut g = 0.0;
        for (mi, wi) in mu {
            for c in 0..ADV_NCELLS {
                let d = f(wi[c]) - f(wi[Self::left(c)]);
                g += f(mi[c]) * (-d / Self::dx());
            }
        }
        out[0] += g;
    }
    fn ft_vjp_u(&self, _out: &mut [T], _u: &[T], _p: &[f64], _t: f64, _nu: &[T]) {}
    fn ft_vjp_p(&self, _out: &mut [f64], _u: &[T], _p: &[f64], _t: f64, _nu: &[T]) {}
}

/// `reaction`: `A' = −k1 A + k2 B`, `B' = k1 A − k2 B` in every cell.
pub struct Reaction;

impl<T: Scalar> Operator<T> for Reaction {
    fn name(&self) -> &str {
        "reaction"
    }
    fn structure(&self) -> Structure {
        Structure::BlockDiagonal {
            ncells: ADV_NCELLS,
            n: 2,
        }
    }
    fn depends_on_t(&self) -> bool {
        false
    }
    fn linear_in_u(&self) -> bool {
        true
    }
    fn writes(&self) -> Vec<usize> {
        (0..2 * ADV_NCELLS).collect()
    }
    fn provides(&self, call: Call) -> bool {
        ALL_CALLS.contains(&call)
    }
    fn rhs(&self, out: &mut [T], u: &[T], p: &[f64], _t: f64) {
        let nc = ADV_NCELLS;
        for c in 0..nc {
            let (a, b) = (f(u[c]), f(u[nc + c]));
            acc(&mut out[c], -p[1] * a + p[2] * b);
            acc(&mut out[nc + c], p[1] * a - p[2] * b);
        }
    }
    fn jvp_u(&self, out: &mut [T], _u: &[T], p: &[f64], _t: f64, v: &[T]) {
        let nc = ADV_NCELLS;
        for c in 0..nc {
            let (a, b) = (f(v[c]), f(v[nc + c]));
            acc(&mut out[c], -p[1] * a + p[2] * b);
            acc(&mut out[nc + c], p[1] * a - p[2] * b);
        }
    }
    fn vjp_u(&self, out: &mut [T], _u: &[T], p: &[f64], _t: f64, w: &[T]) {
        let nc = ADV_NCELLS;
        for c in 0..nc {
            let (wa, wb) = (f(w[c]), f(w[nc + c]));
            acc(&mut out[c], -p[1] * wa + p[1] * wb);
            acc(&mut out[nc + c], p[2] * wa - p[2] * wb);
        }
    }
    fn vjp_p(&self, out: &mut [f64], u: &[T], _p: &[f64], _t: f64, w: &[T]) {
        let nc = ADV_NCELLS;
        let (mut g1, mut g2) = (0.0, 0.0);
        for c in 0..nc {
            let (a, b) = (f(u[c]), f(u[nc + c]));
            let (wa, wb) = (f(w[c]), f(w[nc + c]));
            g1 += -a * wa + a * wb;
            g2 += b * wa - b * wb;
        }
        out[1] += g1;
        out[2] += g2;
    }
    fn jac(&self, out: &mut MatrixStorage<T>, _u: &[T], p: &[f64], _t: f64) {
        let nc = ADV_NCELLS;
        let e = |v: f64| T::from_f64(v);
        for c in 0..nc {
            out.add(c, c, e(-p[1]));
            out.add(c, nc + c, e(p[2]));
            out.add(nc + c, c, e(p[1]));
            out.add(nc + c, nc + c, e(-p[2]));
        }
    }
    fn ft(&self, _out: &mut [T], _u: &[T], _p: &[f64], _t: f64) {}
    fn hess_u(&self, _out: &mut [T], _u: &[T], _p: &[f64], _t: f64, _mu: &[MuPair<'_, T>]) {}
    fn hess_p(&self, out: &mut [f64], _u: &[T], _p: &[f64], _t: f64, mu: &[MuPair<'_, T>]) {
        let nc = ADV_NCELLS;
        let (mut g1, mut g2) = (0.0, 0.0);
        for (mi, wi) in mu {
            for c in 0..nc {
                let (ma, mb) = (f(mi[c]), f(mi[nc + c]));
                let (aa, ab) = (f(wi[c]), f(wi[nc + c]));
                g1 += -ma * aa + mb * aa;
                g2 += ma * ab - mb * ab;
            }
        }
        out[1] += g1;
        out[2] += g2;
    }
    fn ft_vjp_u(&self, _out: &mut [T], _u: &[T], _p: &[f64], _t: f64, _nu: &[T]) {}
    fn ft_vjp_p(&self, _out: &mut [f64], _u: &[T], _p: &[f64], _t: f64, _nu: &[T]) {}
}

// ---------------------------------------------------------------- source_decay_map

/// `source`: `u' = a`, `p = (a, r)`.
pub struct Source;

impl<T: Scalar> Operator<T> for Source {
    fn name(&self) -> &str {
        "source"
    }
    fn structure(&self) -> Structure {
        Structure::Dense
    }
    fn depends_on_t(&self) -> bool {
        false
    }
    fn linear_in_u(&self) -> bool {
        true
    }
    fn writes(&self) -> Vec<usize> {
        vec![0]
    }
    fn provides(&self, call: Call) -> bool {
        ALL_CALLS.contains(&call)
    }
    fn rhs(&self, out: &mut [T], _u: &[T], p: &[f64], _t: f64) {
        acc(&mut out[0], p[0]);
    }
    fn jvp_u(&self, _out: &mut [T], _u: &[T], _p: &[f64], _t: f64, _v: &[T]) {}
    fn vjp_u(&self, _out: &mut [T], _u: &[T], _p: &[f64], _t: f64, _w: &[T]) {}
    fn vjp_p(&self, out: &mut [f64], _u: &[T], _p: &[f64], _t: f64, w: &[T]) {
        out[0] += f(w[0]);
    }
    fn jac(&self, _out: &mut MatrixStorage<T>, _u: &[T], _p: &[f64], _t: f64) {}
    fn ft(&self, _out: &mut [T], _u: &[T], _p: &[f64], _t: f64) {}
    fn hess_u(&self, _out: &mut [T], _u: &[T], _p: &[f64], _t: f64, _mu: &[MuPair<'_, T>]) {}
    fn hess_p(&self, _out: &mut [f64], _u: &[T], _p: &[f64], _t: f64, _mu: &[MuPair<'_, T>]) {}
    fn ft_vjp_u(&self, _out: &mut [T], _u: &[T], _p: &[f64], _t: f64, _nu: &[T]) {}
    fn ft_vjp_p(&self, _out: &mut [f64], _u: &[T], _p: &[f64], _t: f64, _nu: &[T]) {}
}

/// Map `decay`: `u ← u e^{−r Δ}`. It takes no discrete decisions.
pub struct DecayMap;

impl<T: Scalar> MapOperator<T> for DecayMap {
    fn name(&self) -> &str {
        "decay"
    }
    fn apply(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, dt: f64, mask: &mut Vec<u8>) {
        mask.clear();
        self.apply_frozen(out, u, p, t, dt, mask);
    }
    fn apply_frozen(&self, out: &mut [T], u: &[T], p: &[f64], _t: f64, dt: f64, _mask: &[u8]) {
        out[0] = T::from_f64(f(u[0]) * (-p[1] * dt).exp());
    }
    fn map_vjp(
        &self,
        out_u: &mut [T],
        out_p: &mut [f64],
        u: &[T],
        p: &[f64],
        _t: f64,
        dt: f64,
        _mask: &[u8],
        w: &[T],
    ) {
        let e = (-p[1] * dt).exp();
        acc(&mut out_u[0], e * f(w[0]));
        out_p[1] += -dt * f(u[0]) * e * f(w[0]);
    }
    fn map_jvp(
        &self,
        out: &mut [T],
        u: &[T],
        p: &[f64],
        _t: f64,
        dt: f64,
        _mask: &[u8],
        v_u: &[T],
        v_p: &[f64],
    ) {
        let e = (-p[1] * dt).exp();
        acc(&mut out[0], e * f(v_u[0]) - dt * f(u[0]) * e * v_p[1]);
    }
}

#[cfg(test)]
mod tests {
    //! Every derivative against central finite differences, and every
    //! transpose by a dot-product identity, at non-default states.
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        }
        fn vec(&mut self, n: usize) -> Vec<f64> {
            (0..n).map(|_| self.next()).collect()
        }
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn rhs(op: &dyn Operator<f64>, u: &[f64], p: &[f64], t: f64) -> Vec<f64> {
        let mut out = vec![0.0; u.len()];
        op.rhs(&mut out, u, p, t);
        out
    }
    fn jvp(op: &dyn Operator<f64>, u: &[f64], p: &[f64], t: f64, v: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; u.len()];
        op.jvp_u(&mut out, u, p, t, v);
        out
    }
    fn ftv(op: &dyn Operator<f64>, u: &[f64], p: &[f64], t: f64) -> Vec<f64> {
        let mut out = vec![0.0; u.len()];
        op.ft(&mut out, u, p, t);
        out
    }
    fn axpy(x: &[f64], a: f64, y: &[f64]) -> Vec<f64> {
        x.iter().zip(y).map(|(xi, yi)| xi + a * yi).collect()
    }

    fn close(a: f64, b: f64, scale: f64, tol: f64, what: &str) {
        let err = (a - b).abs() / scale.max(1e-300);
        assert!(err < tol, "{what}: {a} vs {b} (rel err {err:e})");
    }

    /// Checks one operator's calls at state `u` and parameters `p`.
    fn check_operator(op: &dyn Operator<f64>, u: &[f64], p: &[f64], t: f64, rng: &mut Lcg) {
        let n = u.len();
        let np = p.len();
        let name = op.name().to_string();
        let v = rng.vec(n);
        let w = rng.vec(n);
        let dp = rng.vec(np);
        // Relative perturbations, so parameters of very different size are
        // perturbed comparably.
        let du: Vec<f64> = v.iter().zip(u).map(|(a, b)| a * b.abs().max(1.0)).collect();
        let dpp: Vec<f64> = dp.iter().zip(p).map(|(a, b)| a * b.abs()).collect();
        let eps = 1e-6;
        // jvp_u against finite differences of rhs.
        let jv = jvp(op, u, p, t, &du);
        let fp = rhs(op, &axpy(u, eps, &du), p, t);
        let fm = rhs(op, &axpy(u, -eps, &du), p, t);
        let scale = jv.iter().fold(1e-12f64, |m, x| m.max(x.abs()));
        for j in 0..n {
            close(jv[j], (fp[j] - fm[j]) / (2.0 * eps), scale, 1e-6, &format!("{name} jvp_u[{j}]"));
        }
        // vjp_u transposes jvp_u.
        let mut jtw = vec![0.0; n];
        op.vjp_u(&mut jtw, u, p, t, &w);
        let (l, r) = (dot(&w, &jv), dot(&jtw, &du));
        close(l, r, l.abs().max(r.abs()), 1e-12, &format!("{name} vjp_u"));
        // jac against jvp_u of unit vectors.
        let mut m = MatrixStorage::zeros(op.structure(), n);
        op.jac(&mut m, u, p, t);
        for j in 0..n {
            let mut e = vec![0.0; n];
            e[j] = 1.0;
            let col = jvp(op, u, p, t, &e);
            for (i, ci) in col.iter().enumerate() {
                assert_eq!(m.get(i, j), *ci, "{name} jac({i},{j})");
            }
        }
        // vjp_p against finite differences in p.
        let mut gp = vec![0.0; np];
        op.vjp_p(&mut gp, u, p, t, &w);
        let fpp = rhs(op, u, &axpy(p, eps, &dpp), t);
        let fpm = rhs(op, u, &axpy(p, -eps, &dpp), t);
        let fd: Vec<f64> = fpp.iter().zip(&fpm).map(|(a, b)| (a - b) / (2.0 * eps)).collect();
        let (l, r) = (dot(&gp, &dpp), dot(&w, &fd));
        close(l, r, l.abs().max(r.abs()).max(1e-10), 1e-6, &format!("{name} vjp_p"));
        // ft against finite differences in t.
        let ftt = ftv(op, u, p, t);
        let et = 1e-6 * t.abs().max(1.0);
        let fdt: Vec<f64> = rhs(op, u, p, t + et)
            .iter()
            .zip(rhs(op, u, p, t - et))
            .map(|(a, b)| (a - b) / (2.0 * et))
            .collect();
        let scale = fdt.iter().fold(1e-10f64, |m, x| m.max(x.abs()));
        for j in 0..n {
            close(ftt[j], fdt[j], scale, 1e-6, &format!("{name} ft[{j}]"));
        }
        if !op.depends_on_t() {
            assert!(ftt.iter().all(|x| *x == 0.0), "{name}: ft of a t-independent operator");
        }
        // Second-order calls: g(u, p) = Σ_i μ_iᵀ J(u, p) w_i.
        let mus: Vec<Vec<f64>> = (0..2).map(|_| rng.vec(n)).collect();
        let ws: Vec<Vec<f64>> = (0..2).map(|_| rng.vec(n)).collect();
        let pairs: Vec<MuPair<'_, f64>> = mus
            .iter()
            .zip(&ws)
            .map(|(a, b)| (a.as_slice(), b.as_slice()))
            .collect();
        let g = |uu: &[f64], pp: &[f64]| -> f64 {
            mus.iter()
                .zip(&ws)
                .map(|(mi, wi)| dot(mi, &jvp(op, uu, pp, t, wi)))
                .sum()
        };
        let mut hu = vec![0.0; n];
        op.hess_u(&mut hu, u, p, t, &pairs);
        let fd = (g(&axpy(u, eps, &du), p) - g(&axpy(u, -eps, &du), p)) / (2.0 * eps);
        let l = dot(&hu, &du);
        close(l, fd, l.abs().max(fd.abs()).max(1e-8), 1e-5, &format!("{name} hess_u"));
        if op.linear_in_u() {
            assert!(hu.iter().all(|x| *x == 0.0), "{name}: hess_u of a linear operator");
        }
        let mut hp = vec![0.0; np];
        op.hess_p(&mut hp, u, p, t, &pairs);
        let fd = (g(u, &axpy(p, eps, &dpp)) - g(u, &axpy(p, -eps, &dpp))) / (2.0 * eps);
        let l = dot(&hp, &dpp);
        close(l, fd, l.abs().max(fd.abs()).max(1e-8), 1e-5, &format!("{name} hess_p"));
        // ft_vjp: ∇ (νᵀ f_t).
        let nu = rng.vec(n);
        let h = |uu: &[f64], pp: &[f64]| dot(&nu, &ftv(op, uu, pp, t));
        let mut fu = vec![0.0; n];
        op.ft_vjp_u(&mut fu, u, p, t, &nu);
        let fd = (h(&axpy(u, eps, &du), p) - h(&axpy(u, -eps, &du), p)) / (2.0 * eps);
        let l = dot(&fu, &du);
        close(l, fd, l.abs().max(fd.abs()).max(1e-8), 1e-5, &format!("{name} ft_vjp_u"));
        let mut fpv = vec![0.0; np];
        op.ft_vjp_p(&mut fpv, u, p, t, &nu);
        let fd = (h(u, &axpy(p, eps, &dpp)) - h(u, &axpy(p, -eps, &dpp))) / (2.0 * eps);
        let l = dot(&fpv, &dpp);
        close(l, fd, l.abs().max(fd.abs()).max(1e-8), 1e-5, &format!("{name} ft_vjp_p"));
        // Writes: rhs is zero outside the written DOFs.
        let wr = op.writes();
        let f0 = rhs(op, u, p, t);
        for (j, fj) in f0.iter().enumerate() {
            if !wr.contains(&j) {
                assert_eq!(*fj, 0.0, "{name} writes DOF {j}");
            }
        }
    }

    #[test]
    fn every_toy_operator_derivative() {
        let mut rng = Lcg(12345);
        for model in [
            "toy:forced_decay",
            "toy:van_der_pol/nonstiff",
            "toy:van_der_pol/stiff",
            "toy:robertson",
            "toy:photostationary_cells",
            "toy:advection_reaction",
            "toy:source_decay_map",
        ] {
            let pb: ToyProblem<f64> = toy_problem(model).unwrap();
            for op in &pb.operators {
                for call in ALL_CALLS {
                    assert!(op.provides(call), "{model} {} {}", op.name(), call.name());
                }
                // Several non-default states (never only the initial condition).
                for k in 0..3 {
                    let u: Vec<f64> = pb
                        .u0
                        .iter()
                        .map(|x| x + (0.3 + 0.2 * k as f64) * (1.0 + x.abs()) * rng.next())
                        .collect();
                    let p: Vec<f64> = pb.p.iter().map(|x| x * (1.0 + 0.2 * rng.next())).collect();
                    let t = pb.tspan.0 + (0.25 + 0.3 * k as f64) * (pb.tspan.1 - pb.tspan.0);
                    check_operator(op.as_ref(), &u, &p, t, &mut rng);
                }
            }
        }
    }

    #[test]
    fn map_derivatives() {
        let mut rng = Lcg(7);
        let pb: ToyProblem<f64> = toy_problem("toy:source_decay_map").unwrap();
        let m = pb.map("decay").unwrap();
        for _ in 0..3 {
            let u = vec![1.0 + rng.next()];
            let p: Vec<f64> = pb.p.iter().map(|x| x * (1.0 + 0.2 * rng.next())).collect();
            let dt = 0.5 + 0.1 * rng.next();
            let mut mask = Vec::new();
            let mut out = vec![0.0];
            m.apply(&mut out, &u, &p, 0.0, dt, &mut mask);
            let (vu, vp) = (rng.vec(1), rng.vec(2));
            let mut jv = vec![0.0];
            m.map_jvp(&mut jv, &u, &p, 0.0, dt, &mask, &vu, &vp);
            let eps = 1e-6;
            let (mut op, mut om) = (vec![0.0], vec![0.0]);
            m.apply_frozen(&mut op, &axpy(&u, eps, &vu), &axpy(&p, eps, &vp), 0.0, dt, &mask);
            m.apply_frozen(&mut om, &axpy(&u, -eps, &vu), &axpy(&p, -eps, &vp), 0.0, dt, &mask);
            close(jv[0], (op[0] - om[0]) / (2.0 * eps), jv[0].abs(), 1e-8, "map_jvp");
            let w = rng.vec(1);
            let (mut gu, mut gp) = (vec![0.0], vec![0.0; 2]);
            m.map_vjp(&mut gu, &mut gp, &u, &p, 0.0, dt, &mask, &w);
            let (l, r) = (dot(&w, &jv), dot(&gu, &vu) + dot(&gp, &vp));
            close(l, r, l.abs(), 1e-14, "map_vjp");
        }
    }

    #[test]
    fn exact_solutions() {
        // forced_decay's exact solution satisfies the equation.
        let p = [2.0, 3.0];
        for &t in &[0.3, 1.1, 1.9] {
            let e = 1e-6;
            let du = (forced_decay_exact(t + e, 1.0, &p) - forced_decay_exact(t - e, 1.0, &p)) / (2.0 * e);
            let rhs = -2.0 * forced_decay_exact(t, 1.0, &p) + (3.0 * t).sin();
            assert!((du - rhs).abs() < 1e-8);
        }
        assert_eq!(forced_decay_exact(0.0, 1.0, &p), 1.0);
        assert_eq!(source_decay_map_exact(0.0, &[1.0, 0.5], 0.5, 1), 0.5 * (-0.25f64).exp());
    }

    #[test]
    fn structures_and_layout() {
        let pb: ToyProblem<f64> = toy_problem("toy:advection_reaction").unwrap();
        let s = Structure::common(pb.operators.iter().map(|o| o.structure()));
        assert_eq!(s, Structure::Dense);
        let r = pb.operator("reaction").unwrap();
        assert_eq!(Structure::common([r.structure()]), Structure::BlockDiagonal { ncells: 64, n: 2 });
        let ph: ToyProblem<f32> = toy_problem("toy:photostationary_cells").unwrap();
        assert_eq!(ph.u0[16], 5.0);
        assert_eq!(ph.u0[32], 40.0);
        assert_eq!(ph.invariants.len(), 32);
    }
}
