//! Explicit Runge–Kutta methods (spec/methods.md §3).
//!
//! Butcher form (ForwardEuler, Heun, RK4, BS3, Tsit5, WRF_RK3) and Shu–Osher
//! form (SSPRK33, SSPRK43, SSPRK104), with their error estimates and
//! first-same-as-last reuse. The arithmetic order is the spec's:
//! sums over ascending `j`, each term `(h·a_ij)·K_j` with the scalar product
//! formed first in f64, zero coefficients skipped.

use crate::scalar::Scalar;
use crate::tableau::{ErrorEstimate, Family, Form, Tableau};
use crate::{NormFn, RhsFn};

/// The step form and its coefficients.
#[derive(Debug, Clone)]
pub enum ErkForm {
    /// `A`, `b`, `c`.
    Butcher {
        a: Vec<Vec<f64>>,
        b: Vec<f64>,
        c: Vec<f64>,
    },
    /// `alpha[i-1][j]`, `beta[i-1][j]` and `c[j]`, the time of the
    /// evaluation at `u^(j)`.
    ShuOsher {
        alpha: Vec<Vec<f64>>,
        beta: Vec<Vec<f64>>,
        c: Vec<f64>,
    },
}

/// The error estimate (spec/methods.md §3.2).
#[derive(Debug, Clone)]
pub enum ErkError {
    None,
    /// `e = Σ_i (h·b̃_i) K_i`.
    Embedded {
        btilde: Vec<f64>,
    },
    /// `e = σ(ũ − u_{n+1})`.
    ShuOsherEmbedded {
        alpha: Vec<f64>,
        beta: Vec<f64>,
        scale: f64,
    },
    /// Shampine's residual estimate (RK4).
    ResidualControl {
        sigma: Vec<f64>,
        scale: f64,
    },
}

/// An explicit Runge–Kutta method built from its table.
#[derive(Debug, Clone)]
pub struct ExplicitRk {
    pub name: String,
    pub form: ErkForm,
    pub error: ErkError,
    pub stages: usize,
    pub order: u32,
    pub embedded_order: Option<u32>,
    /// The table's `first_same_as_last`.
    pub fsal: bool,
    /// The last Butcher stage is `f(u_{n+1}, t_n + h)` (row `s` of `A` equals
    /// `b` and `c_s = 1`): Tsit5 and BS3.
    pub last_stage_is_end: bool,
    /// A first-same-as-last method whose stages do not include
    /// `f(u_{n+1})` evaluates it after the step, at `t_n + h` (Heun, RK4,
    /// ForwardEuler), as OrdinaryDiffEq and the Julia implementation do.
    pub end_eval: bool,
}

impl ExplicitRk {
    /// Builds the method from a table; fails for non-explicit tables.
    pub fn from_tableau(t: &Tableau) -> Result<Self, String> {
        if t.family != Family::ExplicitRk {
            return Err(format!("{} is not an explicit Runge–Kutta method", t.name));
        }
        let form = match t.form {
            Form::Butcher => ErkForm::Butcher {
                a: t.a.clone().ok_or("missing A")?,
                b: t.b.clone().ok_or("missing b")?,
                c: t.c.clone().ok_or("missing c")?,
            },
            Form::ShuOsher => ErkForm::ShuOsher {
                alpha: t.alpha.clone().ok_or("missing alpha")?,
                beta: t.beta.clone().ok_or("missing beta")?,
                c: t.c.clone().ok_or("missing c")?,
            },
            f => return Err(format!("{}: form {f:?} is not explicit", t.name)),
        };
        let error = match t.error_estimate {
            ErrorEstimate::None => ErkError::None,
            ErrorEstimate::Embedded => ErkError::Embedded {
                btilde: t.btilde.clone().ok_or("missing btilde")?,
            },
            ErrorEstimate::ShuOsherEmbedded => {
                let e = t.embedded.as_ref().ok_or("missing embedded")?;
                ErkError::ShuOsherEmbedded {
                    alpha: e.alpha.clone(),
                    beta: e.beta.clone(),
                    scale: e.scale,
                }
            }
            ErrorEstimate::ResidualControl => {
                let r = t
                    .residual_control
                    .as_ref()
                    .ok_or("missing residual_control")?;
                ErkError::ResidualControl {
                    sigma: r.sigma.clone(),
                    scale: r.scale,
                }
            }
            e => return Err(format!("{}: error estimate {e:?} is not explicit", t.name)),
        };
        let s = t.stages;
        let last_stage_is_end = match &form {
            ErkForm::Butcher { a, b, c } => s > 1 && a[s - 1] == *b && c[s - 1] == 1.0,
            ErkForm::ShuOsher { .. } => false,
        };
        let fsal = t.first_same_as_last;
        Ok(ExplicitRk {
            name: t.name.clone(),
            form,
            error,
            stages: s,
            order: t.order,
            embedded_order: t.embedded_order,
            fsal,
            last_stage_is_end,
            end_eval: fsal && !last_stage_is_end,
        })
    }

    /// Whether the method has an error estimate.
    pub fn adaptive(&self) -> bool {
        !matches!(self.error, ErkError::None)
    }

    /// Time of stage `i` (Butcher) or of the evaluation at `u^(i)` (Shu–Osher).
    pub fn stage_time(&self, i: usize, t: f64, h: f64) -> f64 {
        let c = match &self.form {
            ErkForm::Butcher { c, .. } => c,
            ErkForm::ShuOsher { c, .. } => c,
        };
        t + c[i] * h
    }

    /// A workspace for state vectors of length `n`.
    pub fn workspace<T: Scalar>(&self, n: usize) -> ErkWork<T> {
        let s = self.stages;
        let ny = match self.form {
            ErkForm::Butcher { .. } => s,
            ErkForm::ShuOsher { .. } => s + 1,
        };
        ErkWork {
            k: vec![vec![T::zero(); n]; s],
            y: vec![vec![T::zero(); n]; ny],
            unew: vec![T::zero(); n],
            fend: vec![T::zero(); n],
            has_fend: false,
            e: vec![0.0; n],
            e2: vec![0.0; n],
            tmp: vec![T::zero(); n],
            nf: 0,
        }
    }

    /// Takes one step of size `h` from `(un, t)`.
    ///
    /// - `f(out, u, t)` overwrites `out` with the leaf's right-hand side.
    /// - `k1`: the stored first-same-as-last value `f(u_n)`, if valid
    ///   (spec/methods.md §3.3); otherwise the first stage is evaluated.
    /// - `norm(e, u_n, u_{n+1})`: the scaled error norm of §7. When given and
    ///   the method has an estimate, the step's scaled error `E` is returned.
    ///
    /// On return `w.unew` is `u_{n+1}`, `w.y` and `w.k` hold the stage
    /// states and stage derivatives, and `w.fend` holds `f(u_{n+1}, t_n + h)`
    /// when `w.has_fend`.
    #[allow(clippy::too_many_arguments)]
    pub fn step<T: Scalar>(
        &self,
        w: &mut ErkWork<T>,
        f: &mut RhsFn<'_, T>,
        un: &[T],
        t: f64,
        h: f64,
        k1: Option<&[T]>,
        norm: Option<&mut NormFn<'_, T>>,
    ) -> Option<f64> {
        w.has_fend = false;
        match &self.form {
            ErkForm::Butcher { a, b, c } => {
                let s = self.stages;
                for i in 0..s {
                    // U_i = u_n + Σ_{j<i} (h a_ij) K_j.
                    {
                        let (ks, ys) = (&w.k, &mut w.y);
                        let yi = &mut ys[i];
                        yi.copy_from_slice(un);
                        for (j, &aij) in a[i].iter().enumerate().take(i) {
                            if aij != 0.0 {
                                let coef = T::from_f64(h * aij);
                                for (y, &kv) in yi.iter_mut().zip(ks[j].iter()) {
                                    *y += coef * kv;
                                }
                            }
                        }
                    }
                    if i == 0 {
                        if let Some(k1) = k1 {
                            w.k[0].copy_from_slice(k1);
                            continue;
                        }
                    }
                    let ti = t + c[i] * h;
                    let (ys, ks) = (&w.y, &mut w.k);
                    f(&mut ks[i], &ys[i], ti);
                    w.nf += 1;
                }
                // u_{n+1} = u_n + Σ_i (h b_i) K_i.
                w.unew.copy_from_slice(un);
                for (i, &bi) in b.iter().enumerate() {
                    if bi != 0.0 {
                        let coef = T::from_f64(h * bi);
                        for (u, &kv) in w.unew.iter_mut().zip(w.k[i].iter()) {
                            *u += coef * kv;
                        }
                    }
                }
                if self.last_stage_is_end {
                    w.fend.copy_from_slice(&w.k[s - 1]);
                    w.has_fend = true;
                } else if self.end_eval {
                    f(&mut w.fend, &w.unew, t + h);
                    w.nf += 1;
                    w.has_fend = true;
                }
            }
            ErkForm::ShuOsher { alpha, beta, c } => {
                let s = self.stages;
                w.y[0].copy_from_slice(un);
                for i in 1..=s {
                    // f(u^(i-1)), the only new evaluation stage i needs.
                    {
                        let tj = t + c[i - 1] * h;
                        let (ys, ks) = (&w.y, &mut w.k);
                        f(&mut ks[i - 1], &ys[i - 1], tj);
                        w.nf += 1;
                    }
                    // u^(i) = Σ_{j<i} (α_ij u^(j) + (h β_ij) f(u^(j))), ascending j,
                    // the α term before the β term for each j.
                    let (lo, hi) = w.y.split_at_mut(i);
                    let yi = &mut hi[0];
                    yi.iter_mut().for_each(|v| *v = T::zero());
                    for j in 0..i {
                        let aij = alpha[i - 1][j];
                        let bij = beta[i - 1][j];
                        if aij != 0.0 {
                            let ca = T::from_f64(aij);
                            for (y, &uj) in yi.iter_mut().zip(lo[j].iter()) {
                                *y += ca * uj;
                            }
                        }
                        if bij != 0.0 {
                            let cb = T::from_f64(h * bij);
                            for (y, &kj) in yi.iter_mut().zip(w.k[j].iter()) {
                                *y += cb * kj;
                            }
                        }
                    }
                }
                w.unew.copy_from_slice(&w.y[s]);
            }
        }

        let norm = norm?;
        match &self.error {
            ErkError::None => None,
            ErkError::Embedded { btilde } => {
                w.e.iter_mut().for_each(|v| *v = 0.0);
                for (i, &bt) in btilde.iter().enumerate() {
                    if bt != 0.0 {
                        let coef = h * bt;
                        for (e, &kv) in w.e.iter_mut().zip(w.k[i].iter()) {
                            *e += coef * kv.to_f64();
                        }
                    }
                }
                Some(norm(&w.e, un, &w.unew))
            }
            ErkError::ShuOsherEmbedded { alpha, beta, scale } => {
                // ũ = Σ_j (α̃_j u^(j) + (h β̃_j) f(u^(j))), then e = σ(ũ − u_{n+1}).
                w.e.iter_mut().for_each(|v| *v = 0.0);
                for (j, &aj) in alpha.iter().enumerate() {
                    if aj != 0.0 {
                        for (e, &uj) in w.e.iter_mut().zip(w.y[j].iter()) {
                            *e += aj * uj.to_f64();
                        }
                    }
                    if let Some(&bj) = beta.get(j) {
                        if bj != 0.0 {
                            let coef = h * bj;
                            for (e, &kj) in w.e.iter_mut().zip(w.k[j].iter()) {
                                *e += coef * kj.to_f64();
                            }
                        }
                    }
                }
                for (e, &u) in w.e.iter_mut().zip(w.unew.iter()) {
                    *e = scale * (*e - u.to_f64());
                }
                Some(norm(&w.e, un, &w.unew))
            }
            ErkError::ResidualControl { sigma, scale } => {
                // OrdinaryDiffEqLowOrderRK src/fixed_timestep_perform_step.jl:252-316,
                // grouped as written there. k5 = f(u_{n+1}, t_n + h) is the
                // end-of-step evaluation.
                debug_assert!(w.has_fend);
                let mut emax = 0.0f64;
                let mut first = true;
                for &sg in sigma {
                    #[allow(clippy::needless_range_loop)]
                    for j in 0..un.len() {
                        let up = un[j].to_f64();
                        let u = w.unew[j].to_f64();
                        let k1 = w.k[0][j].to_f64();
                        let k5 = w.fend[j].to_f64();
                        let p = (1.0 - sg) * up
                            + sg * u
                            + sg * (sg - 1.0)
                                * ((1.0 - 2.0 * sg) * (u - up) + (sg - 1.0) * h * k1 + sg * h * k5);
                        w.tmp[j] = T::from_f64(p);
                        let inner = -4.0 * h * k1 - 2.0 * h * k5 - 6.0 * up
                            + sg * (3.0 * h * k1 + 3.0 * h * k5 + 6.0 * up - 6.0 * u)
                            + 6.0 * u;
                        w.e2[j] = k1 + sg * inner / h;
                    }
                    let (tmp, k6) = (&w.tmp, &mut w.y[0]);
                    // w.y[0] is U_1 = u_n, no longer needed after the step;
                    // reuse it for f(p_σ) and restore it below.
                    f(k6, tmp, t + sg * h);
                    w.nf += 1;
                    for j in 0..un.len() {
                        w.e[j] = h * (w.y[0][j].to_f64() - w.e2[j]);
                    }
                    let ej = norm(&w.e, un, &w.unew);
                    // max(e1, e2), NaN-propagating like Julia's max.
                    emax = if first || ej.is_nan() || ej > emax {
                        ej
                    } else {
                        emax
                    };
                    first = false;
                }
                w.y[0].copy_from_slice(un);
                Some(scale * emax)
            }
        }
    }
}

/// Workspace and outputs of an explicit Runge–Kutta step.
#[derive(Debug, Clone)]
pub struct ErkWork<T> {
    /// Butcher: `K_i`. Shu–Osher: `f(u^(j))`, `j < s`.
    pub k: Vec<Vec<T>>,
    /// Butcher: stage states `U_i`. Shu–Osher: `u^(j)`, `j = 0..=s`.
    pub y: Vec<Vec<T>>,
    /// `u_{n+1}`.
    pub unew: Vec<T>,
    /// `f(u_{n+1}, t_n + h)`, valid when `has_fend`.
    pub fend: Vec<T>,
    pub has_fend: bool,
    e: Vec<f64>,
    e2: Vec<f64>,
    tmp: Vec<T>,
    /// Right-hand-side evaluations made, cumulative.
    pub nf: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tableau::all_tableaus;

    fn forced(out: &mut [f64], u: &[f64], t: f64) {
        out[0] = -2.0 * u[0] + (3.0 * t).sin();
    }

    fn exact(t: f64) -> f64 {
        let (k, w) = (2.0f64, 3.0f64);
        let a = |t: f64| (k * (w * t).sin() - w * (w * t).cos()) / (k * k + w * w);
        (1.0 - a(0.0)) * (-k * t).exp() + a(t)
    }

    fn run(m: &ExplicitRk, h: f64) -> f64 {
        let mut w = m.workspace::<f64>(1);
        let mut u = vec![1.0];
        let n = (2.0 / h).round() as usize;
        let mut cache: Option<Vec<f64>> = None;
        for i in 0..n {
            let t = i as f64 * h;
            m.step(&mut w, &mut forced, &u, t, h, cache.as_deref(), None);
            u.copy_from_slice(&w.unew);
            cache = if w.has_fend {
                Some(w.fend.clone())
            } else {
                None
            };
        }
        (u[0] - exact(2.0)).abs()
    }

    #[test]
    fn every_explicit_method_shows_its_order() {
        for t in all_tableaus().filter(|t| t.family == Family::ExplicitRk) {
            let m = ExplicitRk::from_tableau(t).unwrap();
            let e1 = run(&m, 0.02);
            let e2 = run(&m, 0.01);
            let p = (e1 / e2).log2();
            assert!(
                (p - m.order as f64).abs() < 0.2 || p > m.order as f64,
                "{}: observed {p}, expected {}",
                m.name,
                m.order
            );
        }
    }

    #[test]
    fn error_estimates_are_small_and_scale() {
        for t in all_tableaus().filter(|t| t.family == Family::ExplicitRk) {
            let m = ExplicitRk::from_tableau(t).unwrap();
            if !m.adaptive() {
                continue;
            }
            let mut w = m.workspace::<f64>(1);
            let mut est = |h: f64| {
                let mut norm = |e: &[f64], _a: &[f64], _b: &[f64]| -> f64 { e[0].abs() };
                m.step(&mut w, &mut forced, &[1.0], 0.0, h, None, Some(&mut norm))
                    .unwrap()
            };
            let e1 = est(0.02);
            let e2 = est(0.01);
            let ratio = (e1 / e2).log2();
            // Local error of the embedded method: order embedded_order + 1.
            let eo = m.embedded_order.unwrap() as f64 + 1.0;
            assert!(
                ratio > eo - 0.6,
                "{}: estimate order {ratio}, expected ~{eo}",
                m.name
            );
        }
    }
}
