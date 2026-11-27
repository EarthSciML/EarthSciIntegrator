//! The runtime of algebraic equations: `solve` nodes and the consistent
//! initialization of index-1 DAEs (spec/methods.md §14, §15;
//! spec/run-config.md §4.3, §4.10; D95, D120–D122). The Julia
//! implementation is `algebraic_runtime.jl`; this follows it.

use crate::operator::Operator;
use crate::tape::ActiveSet;
use crate::{Error, Scalar};
use earthsci_integrator_solvers::linalg::{DenseLu, MatrixStorage, Structure};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Where a bound's value comes from (spec/plan.md §5.9.7).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bound {
    Const(f64),
    /// Index into the parameter vector.
    Param(usize),
}

/// An algebraic operator's unknowns (canonical DOFs, ascending by variable
/// then element) and, per DOF, its bound sources.
#[derive(Debug, Clone)]
pub struct AlgebraicInfo {
    pub dofs: Vec<usize>,
    /// The unknown variable of each DOF.
    pub names: Vec<String>,
    pub lower: Vec<Option<Bound>>,
    pub upper: Vec<Option<Bound>>,
}

impl AlgebraicInfo {
    /// The bounds at the parameters `p` (`±∞` for an unbounded side).
    pub fn values(&self, p: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let v = |b: &Option<Bound>, inf: f64| match b {
            None => inf,
            Some(Bound::Const(x)) => *x,
            Some(Bound::Param(i)) => p[*i],
        };
        (
            self.lower.iter().map(|b| v(b, f64::NEG_INFINITY)).collect(),
            self.upper.iter().map(|b| v(b, f64::INFINITY)).collect(),
        )
    }

    /// Whether any DOF has a bound.
    pub fn bounded(&self) -> bool {
        self.lower.iter().chain(&self.upper).any(Option::is_some)
    }

    /// Concatenates several operators' unknowns (a DAE's).
    pub fn concat(infos: &[&AlgebraicInfo]) -> AlgebraicInfo {
        let mut out = AlgebraicInfo {
            dofs: vec![],
            names: vec![],
            lower: vec![],
            upper: vec![],
        };
        for i in infos {
            out.dofs.extend(&i.dofs);
            out.names.extend(i.names.iter().cloned());
            out.lower.extend(&i.lower);
            out.upper.extend(&i.upper);
        }
        out
    }
}

/// The `default` of the parameter `name` declared in a derived document's
/// JSON, at each element of extents `ext`, column-major (nested lists are
/// outermost index first).
pub fn array_default(doc: &Value, name: &str, ext: &[usize]) -> Result<Vec<f64>, String> {
    let segs: Vec<&str> = name.split('.').collect();
    let mut node = &doc["models"][segs[0]];
    for s in &segs[1..segs.len() - 1] {
        node = &node["subsystems"][*s];
    }
    let d = &node["variables"][segs[segs.len() - 1]]["default"];
    let n: usize = ext.iter().product();
    if let Some(x) = d.as_f64() {
        return Ok(vec![x; n]);
    }
    if !d.is_array() {
        return Err(format!(
            "the bound parameter {name} has no default in its residual document"
        ));
    }
    let mut out = Vec::with_capacity(n);
    let mut idx = vec![0usize; ext.len()];
    for _ in 0..n {
        let mut x = d;
        for &i in &idx {
            x = &x[i];
        }
        out.push(
            x.as_f64()
                .ok_or_else(|| format!("the default of {name} is not numeric at {idx:?}"))?,
        );
        // Next index, first dimension fastest (column-major).
        for (k, v) in idx.iter_mut().enumerate() {
            *v += 1;
            if *v < ext[k] {
                break;
            }
            *v = 0;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// The method (spec/methods.md §15)

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MethodKind {
    Newton,
    Ptc,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Recover {
    Underrelax(f64),
    ScaleParam { param: String, by: f64, max: usize },
    RevertWarmStart,
}

/// A solve method as the resolved plan gives it (spec/plan.md §7.1).
#[derive(Debug, Clone)]
pub struct SolveMethod {
    pub kind: MethodKind,
    pub dt0: f64,
    pub growth: f64,
    pub backtracking: bool,
    pub tol: f64,
    pub maxiter: usize,
    pub recover: Vec<Recover>,
}

impl SolveMethod {
    pub fn from_json(m: &Value, id: &str) -> Result<SolveMethod, Error> {
        if m.get("picard").is_some() {
            return Err(Error::Unimplemented(format!(
                "solve node {id}: picard needs the frozen Jacobian document (spec/plan.md §6.5), which the planner does not emit yet"
            )));
        }
        let conv = m.get("converged");
        if conv.and_then(|c| c.get("rel_change")).is_some() {
            return Err(Error::Unimplemented(format!(
                "solve node {id}: converged.rel_change needs an observed's value, which the runtime does not evaluate yet"
            )));
        }
        let ptc = m.get("ptc");
        let mut recover = Vec::new();
        for a in m
            .get("recover")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(w) = a.get("underrelax").and_then(Value::as_f64) {
                recover.push(Recover::Underrelax(w));
            } else if let Some(p) = a.get("scale_param").and_then(Value::as_str) {
                recover.push(Recover::ScaleParam {
                    param: p.to_string(),
                    by: a["by"].as_f64().unwrap_or(1.0),
                    max: a.get("max").and_then(Value::as_u64).unwrap_or(1) as usize,
                });
            } else if a.get("revert_warm_start").is_some()
                || a.as_str() == Some("revert_warm_start")
            {
                recover.push(Recover::RevertWarmStart);
            }
        }
        Ok(SolveMethod {
            kind: if ptc.is_some() {
                MethodKind::Ptc
            } else {
                MethodKind::Newton
            },
            dt0: ptc
                .and_then(|p| p.get("dt0"))
                .and_then(Value::as_f64)
                .unwrap_or(1e-3),
            growth: ptc
                .and_then(|p| p.get("growth"))
                .and_then(Value::as_f64)
                .unwrap_or(2.0),
            backtracking: m.get("linesearch").and_then(Value::as_str) == Some("backtracking"),
            tol: conv
                .and_then(|c| c.get("residual"))
                .and_then(Value::as_f64)
                .unwrap_or(1e-10),
            maxiter: m.get("maxiter").and_then(Value::as_u64).unwrap_or(100) as usize,
            recover,
        })
    }

    /// The default consistent initialization of a DAE (spec/run-config.md §4.3).
    pub fn default_initialize() -> SolveMethod {
        SolveMethod {
            kind: MethodKind::Newton,
            dt0: 1e-3,
            growth: 2.0,
            backtracking: true,
            tol: 1e-10,
            maxiter: 100,
            recover: vec![],
        }
    }
}

/// The outcome of a solve.
#[derive(Debug, Clone)]
pub struct SolveOutcome {
    pub z: Vec<f64>,
    pub ok: bool,
    pub iterations: usize,
    pub recover: Vec<String>,
    /// Positions (0-based) into the unknowns.
    pub lower: Vec<usize>,
    pub upper: Vec<usize>,
    pub degenerate: Vec<usize>,
}

/// The residual problem: the operators on the model state `u` with the
/// unknowns `dofs`.
pub struct Residual<'a, T: Scalar> {
    pub ops: &'a [Arc<dyn Operator<T>>],
    pub dofs: &'a [usize],
    pub u: &'a [f64],
    pub t: f64,
}

impl<T: Scalar> Residual<'_, T> {
    fn state(&self, z: &[f64]) -> Vec<T> {
        let mut x: Vec<T> = self.u.iter().map(|&v| T::from_f64(v)).collect();
        for (&d, &v) in self.dofs.iter().zip(z) {
            x[d] = T::from_f64(v);
        }
        x
    }

    /// `G(z)` (§15.1).
    pub fn g(&self, z: &[f64], p: &[f64]) -> Vec<f64> {
        let x = self.state(z);
        let mut out = vec![T::zero(); x.len()];
        for op in self.ops {
            op.rhs(&mut out, &x, p, self.t);
        }
        self.dofs.iter().map(|&d| out[d].to_f64()).collect()
    }

    /// `G_z`, dense over the unknowns, column-major.
    pub fn gz(&self, z: &[f64], p: &[f64]) -> Vec<f64> {
        let x = self.state(z);
        let n = x.len();
        let mut jac = MatrixStorage::<T>::zeros(Structure::Dense, n);
        for op in self.ops {
            op.jac(&mut jac, &x, p, self.t);
        }
        let m = self.dofs.len();
        let mut a = vec![0.0; m * m];
        for (c, &dc) in self.dofs.iter().enumerate() {
            for (r, &dr) in self.dofs.iter().enumerate() {
                a[c * m + r] = jac.get(dr, dc).to_f64();
            }
        }
        a
    }
}

fn mid(l: f64, x: f64, h: f64) -> f64 {
    x.max(l).min(h)
}

fn proj(z: &[f64], l: &[f64], h: &[f64]) -> Vec<f64> {
    z.iter()
        .zip(l.iter().zip(h))
        .map(|(&x, (&a, &b))| mid(a, x, b))
        .collect()
}

fn reform(z: &[f64], g: &[f64], l: &[f64], h: &[f64]) -> Vec<f64> {
    (0..z.len())
        .map(|i| z[i] - mid(l[i], z[i] + g[i], h[i]))
        .collect()
}

fn ninf(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

/// Solves `A x = b` (column-major `A`) with partial pivoting; `None` when singular.
fn linsolve(a: &[f64], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut lu = DenseLu::<f64>::new(n);
    if !lu.factor(a) {
        return None;
    }
    let mut x = b.to_vec();
    lu.solve(&mut x);
    Some(x)
}

/// One attempt from `z0` with under-relaxation `omega` (§15.2–§15.4).
fn attempt<T: Scalar>(
    r: &Residual<'_, T>,
    z0: &[f64],
    p: &[f64],
    m: &SolveMethod,
    l: &[f64],
    h: &[f64],
    omega: f64,
) -> (Vec<f64>, bool, usize) {
    let nz = z0.len();
    let mut z = proj(z0, l, h);
    let mut g = r.g(&z, p);
    let mut rr = reform(&z, &g, l, h);
    let mut n = ninf(&rr);
    if !n.is_finite() {
        return (z, false, 0);
    }
    if n <= m.tol {
        return (z, true, 0);
    }
    let mut delta = m.dt0;
    for k in 1..=m.maxiter {
        let a = r.gz(&z, p);
        match m.kind {
            MethodKind::Ptc => {
                // (δ⁻¹I − G_z) s = G; a component held at a bound by an
                // outward rate does not move (§15.4).
                let mut mm: Vec<f64> = a.iter().map(|x| -x).collect();
                let mut b = g.clone();
                for i in 0..nz {
                    mm[i * nz + i] += 1.0 / delta;
                    if (z[i] <= l[i] && g[i] <= 0.0) || (z[i] >= h[i] && g[i] >= 0.0) {
                        for c in 0..nz {
                            mm[c * nz + i] = 0.0;
                        }
                        mm[i * nz + i] = 1.0;
                        b[i] = 0.0;
                    }
                }
                let Some(s) = linsolve(&mm, &b) else {
                    return (z, false, k);
                };
                let zt: Vec<f64> = (0..nz).map(|i| z[i] + omega * s[i]).collect();
                let znew = proj(&zt, l, h);
                let gnew = r.g(&znew, p);
                let rnew = reform(&znew, &gnew, l, h);
                let nnew = ninf(&rnew);
                if !nnew.is_finite() {
                    return (z, false, k);
                }
                delta *= if nnew == 0.0 {
                    m.growth
                } else {
                    m.growth.min(n / nnew)
                };
                (z, g, rr, n) = (znew, gnew, rnew, nnew);
            }
            MethodKind::Newton => {
                // The generalized Jacobian of R (§15.3).
                let mut d: Vec<f64> = a.iter().map(|x| -x).collect();
                for i in 0..nz {
                    let x = z[i] + g[i];
                    if x <= l[i] || x >= h[i] {
                        for c in 0..nz {
                            d[c * nz + i] = 0.0;
                        }
                        d[i * nz + i] = 1.0;
                    }
                }
                let b: Vec<f64> = rr.iter().map(|x| -x).collect();
                let Some(mut s) = linsolve(&d, &b) else {
                    return (z, false, k);
                };
                s.iter_mut().for_each(|x| *x *= omega);
                let mut alpha = 1.0;
                if m.backtracking {
                    let phi = 0.5 * rr.iter().map(|x| x * x).sum::<f64>();
                    alpha = 2f64.powi(-10);
                    for j in 0..=10 {
                        let aj = 2f64.powi(-j);
                        let zt = proj(&(0..nz).map(|i| z[i] + aj * s[i]).collect::<Vec<_>>(), l, h);
                        let rt = reform(&zt, &r.g(&zt, p), l, h);
                        if 0.5 * rt.iter().map(|x| x * x).sum::<f64>() <= (1.0 - 1e-4 * aj) * phi {
                            alpha = aj;
                            break;
                        }
                    }
                }
                z = proj(
                    &(0..nz).map(|i| z[i] + alpha * s[i]).collect::<Vec<_>>(),
                    l,
                    h,
                );
                g = r.g(&z, p);
                rr = reform(&z, &g, l, h);
                n = ninf(&rr);
                if !n.is_finite() {
                    return (z, false, k);
                }
            }
        }
        if n <= m.tol {
            return (z, true, k);
        }
    }
    (z, false, m.maxiter)
}

/// A solve with its `recover` ladder (§15.7) and, at convergence, the
/// active set (spec/run-config.md §4.10).
#[allow(clippy::too_many_arguments)]
pub fn run_solve<T: Scalar>(
    r: &Residual<'_, T>,
    info: &AlgebraicInfo,
    p: &[f64],
    pnames: &[String],
    m: &SolveMethod,
    warm: &[f64],
    initial: &[f64],
) -> Result<SolveOutcome, Error> {
    let (l, h) = info.values(p);
    if let Some(i) = (0..l.len()).find(|&i| l[i] > h[i]) {
        return Err(Error::Run(format!(
            "solve: the lower bound of {} exceeds its upper bound",
            info.names[i]
        )));
    }
    let mut actions = Vec::new();
    let (mut z, mut ok, mut it) = attempt(r, warm, p, m, &l, &h, 1.0);
    let mut pp = p.to_vec();
    for a in &m.recover {
        if ok {
            break;
        }
        match a {
            Recover::Underrelax(w) => {
                actions.push("underrelax".to_string());
                (z, ok, it) = attempt(r, warm, &pp, m, &l, &h, *w);
            }
            Recover::ScaleParam { param, by, max } => {
                let i = pnames.iter().position(|x| x == param).ok_or_else(|| {
                    Error::Config(format!(
                        "solve: recover scale_param {param} is not a scalar parameter"
                    ))
                })?;
                for _ in 0..*max {
                    if ok {
                        break;
                    }
                    pp[i] *= by;
                    actions.push("scale_param".to_string());
                    (z, ok, it) = attempt(r, warm, &pp, m, &l, &h, 1.0);
                }
            }
            Recover::RevertWarmStart => {
                actions.push("revert_warm_start".to_string());
                (z, ok, it) = attempt(r, initial, &pp, m, &l, &h, 1.0);
            }
        }
    }
    let (mut lower, mut upper, mut degenerate) = (vec![], vec![], vec![]);
    if ok {
        let g = r.g(&z, &pp);
        let eps = m.tol;
        for i in 0..z.len() {
            if !(l[i].is_finite() || h[i].is_finite()) {
                continue;
            }
            let atl = l[i].is_finite()
                && (z[i] + g[i] <= l[i] || (z[i] - l[i] <= eps && g[i].abs() <= eps));
            let atu = !atl
                && h[i].is_finite()
                && (z[i] + g[i] >= h[i] || (h[i] - z[i] <= eps && g[i].abs() <= eps));
            if atl {
                z[i] = l[i];
                lower.push(i);
            } else if atu {
                z[i] = h[i];
                upper.push(i);
            }
            if (atl || atu) && g[i].abs() <= eps {
                degenerate.push(i);
            }
        }
    }
    Ok(SolveOutcome {
        z,
        ok,
        iterations: it,
        recover: actions,
        lower,
        upper,
        degenerate,
    })
}

/// The active set per bounded unknown variable, element positions 0-based
/// within the variable (spec/tape.md §2).
pub fn active_sets(info: &AlgebraicInfo, o: &SolveOutcome) -> BTreeMap<String, ActiveSet> {
    let mut first: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, z) in info.names.iter().enumerate() {
        first.entry(z.as_str()).or_insert(i);
    }
    let mut out: BTreeMap<String, ActiveSet> = BTreeMap::new();
    for (set, k) in [(&o.lower, 0), (&o.upper, 1), (&o.degenerate, 2)] {
        for &i in set {
            let z = &info.names[i];
            let e = out.entry(z.clone()).or_default();
            let pos = i - first[z.as_str()];
            match k {
                0 => e.lower.push(pos),
                1 => e.upper.push(pos),
                _ => e.degenerate.push(pos),
            }
        }
    }
    out
}

/// A built `solve` node.
pub struct SolveSpec<T: Scalar> {
    pub id: String,
    pub ops: Vec<Arc<dyn Operator<T>>>,
    pub info: AlgebraicInfo,
    pub method: SolveMethod,
    pub warm_initial: bool,
    /// The model's initial value of the unknowns (`warm_start: initial`,
    /// `revert_warm_start`).
    pub initial: Vec<f64>,
    /// The parameter names (`recover` `scale_param`).
    pub pnames: Vec<String>,
}

/// A `coupled` node with algebraic operators (spec/methods.md §14).
pub struct DaeSpec<T: Scalar> {
    /// The algebraic rows over the leaf's state vector.
    pub mass: Vec<bool>,
    pub ops: Vec<Arc<dyn Operator<T>>>,
    pub info: AlgebraicInfo,
    pub init: SolveMethod,
    pub initial: Vec<f64>,
    pub pnames: Vec<String>,
}
