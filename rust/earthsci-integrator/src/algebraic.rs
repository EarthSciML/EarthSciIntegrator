//! The runtime of algebraic equations: `solve` nodes and the consistent
//! initialization of index-1 DAEs (spec/methods.md §14, §15;
//! spec/run-config.md §4.3, §4.10; D95, D120–D122). The Julia
//! implementation is `algebraic_runtime.jl`; this follows it.

use crate::operator::Operator;
use crate::tape::ActiveSet;
use crate::{Error, Scalar};
use earthsci_integrator_solvers::linalg::DenseLu;
use serde_json::Value;
use std::collections::BTreeMap;

/// A Jacobian over a solve's unknowns: `(row, column)` positions to values.
pub type Entries = BTreeMap<(usize, usize), f64>;
use std::sync::Arc;

/// Where a bound's value comes from (spec/plan.md §5.9.7).
#[derive(Debug, Clone, PartialEq)]
pub enum Bound {
    Const(f64),
    /// Index into the parameter vector.
    Param(usize),
    /// A state-free observed (D125) and the element's position in it (a
    /// scalar observed has one value for every element).
    Obs(String, usize),
}

/// The values of the frozen observeds by name, evaluated once per
/// invocation (D125).
pub type Observed = BTreeMap<String, Vec<f64>>;

/// An algebraic operator's frozen documents (spec/plan.md §6.5), evaluated
/// at a model state `x`.
pub trait FrozenEval: Send + Sync {
    /// The values of the frozen observeds `O_m` at `x`, each column-major.
    fn values(&self, x: &[f64], p: &[f64], t: f64) -> Result<Vec<Vec<f64>>, String>;
    /// Whether the frozen Jacobian document is loaded (a `picard` node).
    fn has_jacobian(&self) -> bool;
    /// The Jacobian over the `nz` unknowns with the frozen observeds held at
    /// their values at `x` (spec/methods.md §15.5), by position.
    fn jacobian(&self, x: &[f64], p: &[f64], t: f64) -> Result<Entries, String>;
    /// The frozen observeds `O_m`, in order (the plan entry's `frozen`).
    fn names(&self) -> &[String];
}

/// The frozen documents of an algebraic operator, when the plan has them.
#[derive(Clone)]
pub struct Frozen(pub Arc<dyn FrozenEval>);

impl std::fmt::Debug for Frozen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Frozen")
    }
}

/// An algebraic operator's unknowns (canonical DOFs, ascending by variable
/// then element), per DOF its bound sources, and its frozen documents.
#[derive(Debug, Clone)]
pub struct AlgebraicInfo {
    pub dofs: Vec<usize>,
    /// The unknown variable of each DOF.
    pub names: Vec<String>,
    pub lower: Vec<Option<Bound>>,
    pub upper: Vec<Option<Bound>>,
    pub frozen: Option<Frozen>,
    /// The blocks, when the plan gives the structure a `block` (spec/plan.md
    /// §5.4): per cell, the positions among the unknowns of the block's
    /// elements, in the block's order.
    pub blocks: Option<Vec<Vec<usize>>>,
    /// The residual document's JSON (an array parameter's default), when
    /// loaded from a plan.
    pub doc: Option<Arc<Value>>,
    /// Each unknown variable's extents.
    pub extents: BTreeMap<String, Vec<usize>>,
}

impl AlgebraicInfo {
    /// The bounds at the parameters `p` and the frozen observeds' values
    /// `obs` (`±∞` for an unbounded side).
    pub fn values(&self, p: &[f64], obs: &Observed) -> (Vec<f64>, Vec<f64>) {
        let v = |b: &Option<Bound>, inf: f64| match b {
            None => inf,
            Some(Bound::Const(x)) => *x,
            Some(Bound::Param(i)) => p[*i],
            Some(Bound::Obs(name, e)) => {
                let o = &obs[name];
                if o.len() == 1 {
                    o[0]
                } else {
                    o[*e]
                }
            }
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

    /// Concatenates several operators' unknowns (a DAE's). One operator's
    /// frozen documents serve a picard or rel_change initialization.
    pub fn concat(infos: &[&AlgebraicInfo]) -> AlgebraicInfo {
        let mut out = AlgebraicInfo {
            dofs: vec![],
            names: vec![],
            lower: vec![],
            upper: vec![],
            frozen: match infos {
                [one] => one.frozen.clone(),
                _ => None,
            },
            blocks: match infos {
                [one] => one.blocks.clone(),
                _ => None,
            },
            doc: match infos {
                [one] => one.doc.clone(),
                _ => None,
            },
            extents: BTreeMap::new(),
        };
        for i in infos {
            out.dofs.extend(&i.dofs);
            out.names.extend(i.names.iter().cloned());
            out.lower.extend(i.lower.iter().cloned());
            out.upper.extend(i.upper.iter().cloned());
            out.extents
                .extend(i.extents.iter().map(|(k, v)| (k.clone(), v.clone())));
        }
        out
    }

    /// The values of the frozen observeds at the model state `x` (D125), by
    /// name: what a referenced bound or scale reads, once per invocation.
    pub fn observed(&self, x: &[f64], p: &[f64], t: f64) -> Result<Observed, Error> {
        let Some(Frozen(f)) = &self.frozen else {
            return Ok(Observed::new());
        };
        let vals = f
            .values(x, p, t)
            .map_err(|e| Error::Run(format!("solve: the frozen documents: {e}")))?;
        Ok(f.names().iter().cloned().zip(vals).collect())
    }

    /// The scale of each unknown DOF under method `m` (spec/methods.md §15.1,
    /// D125): 1 where `scale` names no unknown, else a number, a parameter,
    /// an array parameter's default, or a frozen observed (scalar or one per
    /// element). A scale that is not positive and finite fails the
    /// invocation.
    pub fn scale(
        &self,
        m: &SolveMethod,
        p: &[f64],
        pnames: &[String],
        obs: &Observed,
        id: &str,
    ) -> Result<Vec<f64>, Error> {
        let mut d = vec![1.0; self.dofs.len()];
        if m.scale.is_empty() {
            return Ok(d);
        }
        let mut first: BTreeMap<&str, usize> = BTreeMap::new();
        for (i, z) in self.names.iter().enumerate() {
            first.entry(z.as_str()).or_insert(i);
        }
        for (z, v) in &m.scale {
            let Some(&f) = first.get(z.as_str()) else {
                continue;
            };
            let n = self.names.iter().filter(|x| *x == z).count();
            let vals: Vec<f64> = if let Some(x) = v.as_f64() {
                vec![x; n]
            } else {
                let name = v.as_str().unwrap_or_default();
                if let Some(k) = pnames.iter().position(|q| q == name) {
                    vec![p[k]; n]
                } else if let Some(o) = obs.get(name) {
                    if o.len() == 1 {
                        vec![o[0]; n]
                    } else {
                        o.clone()
                    }
                } else {
                    let doc = self.doc.as_ref().ok_or_else(|| {
                        Error::Config(format!(
                            "solve node {id}: the scale {name} of {z} has no value"
                        ))
                    })?;
                    array_default(doc, name, &self.extents[z]).map_err(Error::Config)?
                }
            };
            if vals.len() != n {
                return Err(Error::Config(format!(
                    "solve node {id}: the scale {v} of {z} has {} elements, {z} {n}",
                    vals.len()
                )));
            }
            d[f..f + n].copy_from_slice(&vals);
        }
        if let Some(bad) = d.iter().position(|x| !(x.is_finite() && *x > 0.0)) {
            let z = &self.names[bad];
            return Err(Error::Run(format!(
                "solve node {id}: the scale of {z} element {} is {}, not positive and finite (spec/methods.md §15.1) (SolveFailure)",
                bad - first[z.as_str()] + 1,
                d[bad]
            )));
        }
        Ok(d)
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
    /// Newton's iteration on the frozen Jacobian (§15.5).
    Picard,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Recover {
    Underrelax(f64),
    ScaleParam {
        param: String,
        by: f64,
        max: usize,
    },
    RevertWarmStart,
    /// An attempt with another method (D125): the step's method object.
    Method(Value),
}

/// A solve method as the resolved plan gives it (spec/plan.md §7.1).
#[derive(Debug, Clone)]
pub struct SolveMethod {
    pub kind: MethodKind,
    pub dt0: f64,
    pub growth: f64,
    pub backtracking: bool,
    /// `converged.residual`; infinite when only `rel_change` is given.
    pub tol: f64,
    /// The `rel_change` observed's position among the frozen observeds
    /// (spec/plan.md §6.5).
    pub rel_change: Option<usize>,
    pub rel_tol: f64,
    pub maxiter: usize,
    pub recover: Vec<Recover>,
    /// `linsolve` `block_lu` or `sparse_block_lu`: one LU per block of the
    /// plan's structure (dense within a block).
    pub block: bool,
    /// The method's `scale`: unknown to a number or a name (D125).
    pub scale: BTreeMap<String, Value>,
    /// `newton.inexact`: a Newton–Krylov step with Eisenstat–Walker forcing
    /// (D126).
    pub inexact: bool,
}

/// `ptc.growth`: a number, or infinite for "uncapped" (switched evolution
/// relaxation, D125).
fn growth_of(ptc: Option<&Value>) -> f64 {
    match ptc.and_then(|p| p.get("growth")) {
        Some(Value::String(s)) if s == "uncapped" => f64::INFINITY,
        g => g.and_then(Value::as_f64).unwrap_or(2.0),
    }
}

impl SolveMethod {
    pub fn from_json(m: &Value, id: &str) -> Result<SolveMethod, Error> {
        let _ = id;
        let conv = m.get("converged");
        // The frozen observeds, in the frozen documents' order (spec/plan.md
        // §6.5): picard's freeze list, then the rel_change observed if not
        // among them.
        let mut frozen: Vec<&str> = m["picard"]["freeze"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        let rel_change = conv
            .and_then(|c| c.get("rel_change"))
            .and_then(Value::as_str)
            .map(|o| {
                frozen.iter().position(|x| *x == o).unwrap_or_else(|| {
                    frozen.push(o);
                    frozen.len() - 1
                })
            });
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
            } else if let Some(mm) = a.get("method") {
                recover.push(Recover::Method(mm.clone()));
            }
        }
        Ok(SolveMethod {
            kind: if ptc.is_some() {
                MethodKind::Ptc
            } else if m.get("picard").is_some() {
                MethodKind::Picard
            } else {
                MethodKind::Newton
            },
            dt0: ptc
                .and_then(|p| p.get("dt0"))
                .and_then(Value::as_f64)
                .unwrap_or(1e-3),
            growth: growth_of(ptc),
            backtracking: m.get("linesearch").and_then(Value::as_str) == Some("backtracking"),
            tol: match conv {
                None => 1e-10,
                Some(c) => c
                    .get("residual")
                    .and_then(Value::as_f64)
                    .unwrap_or(f64::INFINITY),
            },
            rel_change,
            block: m
                .get("linsolve")
                .is_some_and(|l| l.get("block_lu").is_some() || l.get("sparse_block_lu").is_some()),
            rel_tol: conv
                .and_then(|c| c.get("tol"))
                .and_then(Value::as_f64)
                .unwrap_or(0.0),
            maxiter: m.get("maxiter").and_then(Value::as_u64).unwrap_or(100) as usize,
            recover,
            inexact: m["newton"]["inexact"].as_bool().unwrap_or(false),
            scale: m
                .get("scale")
                .and_then(Value::as_object)
                .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                .unwrap_or_default(),
        })
    }

    /// The method of a `recover` step `{"method": M}` (spec/methods.md
    /// §15.7): `M`'s kind, options, `linesearch` and `maxiter`, with this
    /// node's `converged`, `linsolve` and `scale`.
    pub fn recover_method(&self, mm: &Value) -> SolveMethod {
        let ptc = mm.get("ptc");
        SolveMethod {
            kind: if ptc.is_some() {
                MethodKind::Ptc
            } else {
                MethodKind::Newton
            },
            dt0: ptc
                .and_then(|p| p.get("dt0"))
                .and_then(Value::as_f64)
                .unwrap_or(1e-3),
            growth: growth_of(ptc),
            backtracking: mm.get("linesearch").and_then(Value::as_str) == Some("backtracking"),
            maxiter: mm.get("maxiter").and_then(Value::as_u64).unwrap_or(100) as usize,
            recover: vec![],
            inexact: mm["newton"]["inexact"].as_bool().unwrap_or(false),
            ..self.clone()
        }
    }

    /// The default consistent initialization of a DAE (spec/run-config.md §4.3).
    pub fn default_initialize() -> SolveMethod {
        SolveMethod {
            kind: MethodKind::Newton,
            dt0: 1e-3,
            growth: 2.0,
            backtracking: true,
            tol: 1e-10,
            rel_change: None,
            rel_tol: 0.0,
            block: false,
            maxiter: 100,
            recover: vec![],
            scale: BTreeMap::new(),
            inexact: false,
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

    /// The model state with the unknowns set to `z`, in f64.
    fn state_f64(&self, z: &[f64]) -> Vec<f64> {
        let mut x = self.u.to_vec();
        for (&d, &v) in self.dofs.iter().zip(z) {
            x[d] = v;
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

    /// `G_z` over the unknowns, by position (the unknowns' DOFs ascending).
    pub fn gz(&self, z: &[f64], p: &[f64]) -> Entries {
        let x = self.state(z);
        let mut a = Entries::new();
        for op in self.ops {
            op.jac_entries(self.dofs, &x, p, self.t, &mut |r, c, v| {
                *a.entry((r, c)).or_insert(0.0) += v.to_f64();
            });
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

/// The blocks of a solve's linear systems (spec/methods.md §15.3): with a
/// block linsolve the plan's blocks, else one block of every unknown;
/// `of[i]` is the block and place of position `i`.
struct Blocks<'a> {
    blocks: std::borrow::Cow<'a, [Vec<usize>]>,
    of: Vec<(usize, usize)>,
}

impl<'a> Blocks<'a> {
    fn new(info: &'a AlgebraicInfo, m: &SolveMethod) -> Result<Blocks<'a>, Error> {
        let nz = info.dofs.len();
        let blocks: std::borrow::Cow<'a, [Vec<usize>]> = if m.block {
            match &info.blocks {
                Some(b) => std::borrow::Cow::Borrowed(b.as_slice()),
                None => {
                    return Err(Error::Unimplemented(
                        "solve: a block linsolve, but the operator's plan entry gives no block structure (spec/plan.md §5.4)".into(),
                    ))
                }
            }
        } else {
            std::borrow::Cow::Owned(vec![(0..nz).collect()])
        };
        let mut of = vec![(usize::MAX, 0); nz];
        for (k, b) in blocks.iter().enumerate() {
            for (i, &q) in b.iter().enumerate() {
                of[q] = (k, i);
            }
        }
        if of.iter().any(|x| x.0 == usize::MAX) {
            return Err(Error::Config(
                "solve: the plan's blocks do not cover every unknown".into(),
            ));
        }
        Ok(Blocks { blocks, of })
    }
}

/// The matrix `σI − A` with the rows `held` replaced by the identity's, one
/// column-major dense matrix per block.
fn block_matrices(
    a: &Entries,
    sigma: f64,
    held: &[bool],
    bl: &Blocks<'_>,
) -> Result<Vec<Vec<f64>>, Error> {
    let mut ms: Vec<Vec<f64>> = bl
        .blocks
        .iter()
        .map(|p| vec![0.0; p.len() * p.len()])
        .collect();
    for (&(r, c), &v) in a {
        let ((kr, ir), (kc, ic)) = (bl.of[r], bl.of[c]);
        if kr != kc {
            return Err(Error::Config(format!(
                "solve: the Jacobian couples unknowns {r} and {c} of two blocks of the plan's block structure"
            )));
        }
        let n = bl.blocks[kr].len();
        ms[kr][ic * n + ir] -= v;
    }
    for (blk, mm) in bl.blocks.iter().zip(ms.iter_mut()) {
        let n = blk.len();
        for (i, &q) in blk.iter().enumerate() {
            mm[i * n + i] += sigma;
            if held[q] {
                for c in 0..n {
                    mm[c * n + i] = 0.0;
                }
                mm[i * n + i] = 1.0;
            }
        }
    }
    Ok(ms)
}

/// Solves `(σI − A) x = b` with the rows `held` replaced by the identity's,
/// one LU per block (partial pivoting); `None` when a block is singular.
fn linsolve(
    a: &Entries,
    sigma: f64,
    held: &[bool],
    b: &[f64],
    bl: &Blocks<'_>,
) -> Result<Option<Vec<f64>>, Error> {
    let ms = block_matrices(a, sigma, held, bl)?;
    let mut x = vec![0.0; b.len()];
    for (blk, mm) in bl.blocks.iter().zip(&ms) {
        let n = blk.len();
        let mut lu = DenseLu::<f64>::new(n);
        if !lu.factor(mm) {
            return Ok(None);
        }
        let mut xb: Vec<f64> = blk.iter().map(|&q| b[q]).collect();
        lu.solve(&mut xb);
        for (&q, v) in blk.iter().zip(xb) {
            x[q] = v;
        }
    }
    Ok(Some(x))
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// The inexact Newton–Krylov step of spec/methods.md §15.3 (D126): GMRES on
/// `D P⁻¹ y = b`, `s = P⁻¹ y`, with `D = −A` and the rows `held` the
/// identity's; right-preconditioned, unrestarted, at most N iterations, zero
/// initial guess, modified Gram–Schmidt with Givens rotations, stopping at
/// `‖D s − b‖₂ ≤ η ‖b‖₂`. `P` is the per-cell block LU with a block
/// linsolve, else the diagonal of `D`. `None` when `P` is singular.
fn krylov_step(
    a: &Entries,
    held: &[bool],
    b: &[f64],
    bl: &Blocks<'_>,
    m: &SolveMethod,
    eta: f64,
) -> Result<Option<Vec<f64>>, Error> {
    let n = b.len();
    let dv = |v: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; n];
        for (&(r, c), &x) in a {
            out[r] -= x * v[c];
        }
        for i in 0..n {
            if held[i] {
                out[i] = v[i];
            }
        }
        out
    };
    let lus: Option<Vec<DenseLu<f64>>> = if m.block {
        let ms = block_matrices(a, 0.0, held, bl)?;
        let mut lus = Vec::new();
        for (blk, mm) in bl.blocks.iter().zip(&ms) {
            let mut lu = DenseLu::<f64>::new(blk.len());
            if !lu.factor(mm) {
                return Ok(None);
            }
            lus.push(lu);
        }
        Some(lus)
    } else {
        None
    };
    // Jacobi: the diagonal of D, −A_ii (1 on a held row; a zero taken as 1).
    let mut d = vec![0.0; n];
    for (&(r, c), &x) in a {
        if r == c {
            d[r] -= x;
        }
    }
    for i in 0..n {
        if held[i] || d[i] == 0.0 {
            d[i] = 1.0;
        }
    }
    let prec = |v: &[f64]| -> Vec<f64> {
        match &lus {
            Some(lus) => {
                let mut out = vec![0.0; n];
                for (blk, lu) in bl.blocks.iter().zip(lus) {
                    let mut xb: Vec<f64> = blk.iter().map(|&q| v[q]).collect();
                    lu.solve(&mut xb);
                    for (&q, x) in blk.iter().zip(xb) {
                        out[q] = x;
                    }
                }
                out
            }
            None => v.iter().zip(&d).map(|(x, y)| x / y).collect(),
        }
    };
    let beta = norm2(b);
    if beta == 0.0 {
        return Ok(Some(vec![0.0; n]));
    }
    let mut vs: Vec<Vec<f64>> = vec![b.iter().map(|x| x / beta).collect()];
    let mut zs: Vec<Vec<f64>> = Vec::new();
    let mut h = vec![vec![0.0; n]; n + 1];
    let (mut cs, mut sn) = (vec![0.0; n], vec![0.0; n]);
    let mut g = vec![0.0; n + 1];
    g[0] = beta;
    let mut k = 0;
    for j in 0..n {
        let z = prec(&vs[j]);
        let mut w = dv(&z);
        zs.push(z);
        for i in 0..=j {
            h[i][j] = w.iter().zip(&vs[i]).map(|(x, y)| x * y).sum();
            let hij = h[i][j];
            w.iter_mut().zip(&vs[i]).for_each(|(x, y)| *x -= hij * y);
        }
        let hnext = norm2(&w);
        h[j + 1][j] = hnext;
        for i in 0..j {
            let t = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
            h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
            h[i][j] = t;
        }
        let dj = h[j][j].hypot(h[j + 1][j]);
        (cs[j], sn[j]) = if dj == 0.0 {
            (1.0, 0.0)
        } else {
            (h[j][j] / dj, h[j + 1][j] / dj)
        };
        h[j][j] = dj;
        h[j + 1][j] = 0.0;
        g[j + 1] = -sn[j] * g[j];
        g[j] *= cs[j];
        k = j + 1;
        if g[j + 1].abs() <= eta * beta || hnext == 0.0 {
            break;
        }
        vs.push(w.iter().map(|x| x / hnext).collect());
    }
    let mut y = vec![0.0; k];
    for i in (0..k).rev() {
        let mut acc = g[i];
        for c in (i + 1)..k {
            acc -= h[i][c] * y[c];
        }
        y[i] = if h[i][i] == 0.0 { 0.0 } else { acc / h[i][i] };
    }
    let mut s = vec![0.0; n];
    for (yi, z) in y.iter().zip(&zs) {
        s.iter_mut().zip(z).for_each(|(x, zz)| *x += yi * zz);
    }
    Ok(Some(s))
}

/// Eisenstat–Walker choice 2 (spec/methods.md §15.3, D126): the forcing
/// term from the last two residual 2-norms and the previous term.
fn forcing(nr: f64, prev: Option<(f64, f64)>) -> f64 {
    let Some((nr_prev, eta_prev)) = prev else {
        return 0.5;
    };
    let mut eta = 0.9 * (nr / nr_prev).powi(2);
    let s = 0.9 * eta_prev * eta_prev;
    if s > 0.1 {
        eta = eta.max(s);
    }
    eta.min(0.9)
}

/// The frozen documents a picard method or `converged.rel_change` needs
/// (spec/plan.md §6.5).
fn frozen_of<'a>(
    info: &'a AlgebraicInfo,
    m: &SolveMethod,
) -> Result<Option<&'a dyn FrozenEval>, Error> {
    if m.kind != MethodKind::Picard && m.rel_change.is_none() {
        return Ok(None);
    }
    let Some(Frozen(f)) = &info.frozen else {
        return Err(Error::Config(
            "solve: a picard method or converged.rel_change needs the operator's frozen documents (spec/plan.md §6.5), which the plan does not have".into(),
        ));
    };
    if m.kind == MethodKind::Picard && !f.has_jacobian() {
        return Err(Error::Config(
            "solve: a picard method needs the operator's frozen Jacobian document (spec/plan.md §6.5), which the plan does not have".into(),
        ));
    }
    Ok(Some(f.as_ref()))
}

/// `converged.rel_change` (§15.2): `‖O_k − O_{k-1}‖₂ ≤ tol ‖O_k‖₂`.
fn rel_met(o: &[f64], prev: &[f64], tol: f64) -> bool {
    let d: f64 = o.iter().zip(prev).map(|(a, b)| (a - b) * (a - b)).sum();
    let n: f64 = o.iter().map(|a| a * a).sum();
    d.sqrt() <= tol * n.sqrt()
}

/// One attempt from `z0` with under-relaxation `omega` (§15.2–§15.5). With
/// `converged.rel_change` it converges when every criterion given is met;
/// the change is measured from the previous iterate, so the starting
/// iterate cannot meet it.
#[allow(clippy::too_many_arguments)]
fn attempt<T: Scalar>(
    r: &Residual<'_, T>,
    fz: Option<&dyn FrozenEval>,
    bl: &Blocks<'_>,
    z0: &[f64],
    p: &[f64],
    m: &SolveMethod,
    l: &[f64],
    h: &[f64],
    omega: f64,
    d: &[f64],
) -> Result<(Vec<f64>, bool, usize), Error> {
    let fail = |e: String| Error::Run(format!("solve: the frozen documents: {e}"));
    // The iteration works on the scaled residual D·G (§15.1, D125); the
    // convergence test on the unscaled one.
    let gs = |z: &[f64], p: &[f64]| -> Vec<f64> {
        r.g(z, p).iter().zip(d).map(|(a, b)| a * b).collect()
    };
    let conv = |z: &[f64], g: &[f64]| -> f64 {
        let gu: Vec<f64> = g.iter().zip(d).map(|(a, b)| a / b).collect();
        ninf(&reform(z, &gu, l, h))
    };
    let observed = |z: &[f64]| -> Result<Vec<f64>, Error> {
        let (Some(f), Some(i)) = (fz, m.rel_change) else {
            return Ok(vec![]);
        };
        Ok(f.values(&r.state_f64(z), p, r.t)
            .map_err(fail)?
            .swap_remove(i))
    };
    let nz = z0.len();
    let mut z = proj(z0, l, h);
    let mut g = gs(&z, p);
    let mut rr = reform(&z, &g, l, h);
    let mut n = ninf(&rr);
    if !n.is_finite() {
        return Ok((z, false, 0));
    }
    if m.rel_change.is_none() && conv(&z, &g) <= m.tol {
        return Ok((z, true, 0));
    }
    let mut o = observed(&z)?;
    let mut delta = m.dt0;
    let mut ew: Option<(f64, f64)> = None;
    for k in 1..=m.maxiter {
        let mut a = match (m.kind, fz) {
            (MethodKind::Picard, Some(f)) => f.jacobian(&r.state_f64(&z), p, r.t).map_err(fail)?,
            _ => r.gz(&z, p),
        };
        for ((row, _), v) in a.iter_mut() {
            *v *= d[*row];
        }
        match m.kind {
            MethodKind::Ptc => {
                // (δ⁻¹I − G_z) s = G; a component held at a bound by an
                // outward rate does not move (§15.4).
                let held: Vec<bool> = (0..nz)
                    .map(|i| (z[i] <= l[i] && g[i] <= 0.0) || (z[i] >= h[i] && g[i] >= 0.0))
                    .collect();
                let b: Vec<f64> = (0..nz).map(|i| if held[i] { 0.0 } else { g[i] }).collect();
                let Some(s) = linsolve(&a, 1.0 / delta, &held, &b, bl)? else {
                    return Ok((z, false, k));
                };
                // A step that is not finite (a NaN or infinite Jacobian entry)
                // fails the attempt (§15.7).
                if !s.iter().all(|x| x.is_finite()) {
                    return Ok((z, false, k));
                }
                let zt: Vec<f64> = (0..nz).map(|i| z[i] + omega * s[i]).collect();
                let znew = proj(&zt, l, h);
                let gnew = gs(&znew, p);
                let rnew = reform(&znew, &gnew, l, h);
                let nnew = ninf(&rnew);
                if !nnew.is_finite() {
                    return Ok((z, false, k));
                }
                delta *= if nnew == 0.0 {
                    m.growth
                } else {
                    m.growth.min(n / nnew)
                };
                (z, g, rr, n) = (znew, gnew, rnew, nnew);
            }
            MethodKind::Newton | MethodKind::Picard => {
                // The generalized Jacobian of R (§15.3).
                let held: Vec<bool> = (0..nz)
                    .map(|i| z[i] + g[i] <= l[i] || z[i] + g[i] >= h[i])
                    .collect();
                let b: Vec<f64> = rr.iter().map(|x| -x).collect();
                let step = if m.inexact {
                    // Newton–Krylov with Eisenstat–Walker forcing (§15.3, D126).
                    let nr = norm2(&rr);
                    let eta = forcing(nr, ew);
                    ew = Some((nr, eta));
                    krylov_step(&a, &held, &b, bl, m, eta)?
                } else {
                    linsolve(&a, 0.0, &held, &b, bl)?
                };
                let Some(mut s) = step else {
                    return Ok((z, false, k));
                };
                if !s.iter().all(|x| x.is_finite()) {
                    return Ok((z, false, k));
                }
                s.iter_mut().for_each(|x| *x *= omega);
                let mut alpha = 1.0;
                if m.backtracking {
                    let phi = 0.5 * rr.iter().map(|x| x * x).sum::<f64>();
                    alpha = 2f64.powi(-10);
                    for j in 0..=10 {
                        let aj = 2f64.powi(-j);
                        let zt = proj(&(0..nz).map(|i| z[i] + aj * s[i]).collect::<Vec<_>>(), l, h);
                        let rt = reform(&zt, &gs(&zt, p), l, h);
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
                g = gs(&z, p);
                rr = reform(&z, &g, l, h);
                n = ninf(&rr);
                if !n.is_finite() {
                    return Ok((z, false, k));
                }
            }
        }
        if m.rel_change.is_none() {
            if conv(&z, &g) <= m.tol {
                return Ok((z, true, k));
            }
        } else {
            let onew = observed(&z)?;
            let met = conv(&z, &g) <= m.tol && rel_met(&onew, &o, m.rel_tol);
            o = onew;
            if met {
                return Ok((z, true, k));
            }
        }
    }
    Ok((z, false, m.maxiter))
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
    id: &str,
) -> Result<SolveOutcome, Error> {
    // Referenced bounds and scales are evaluated once per invocation (D125).
    let obs = info.observed(&r.state_f64(warm), p, r.t)?;
    let (l, h) = info.values(p, &obs);
    let d = info.scale(m, p, pnames, &obs, id)?;
    if let Some(i) = (0..l.len()).find(|&i| l[i] > h[i]) {
        return Err(Error::Run(format!(
            "solve: the lower bound of {} exceeds its upper bound",
            info.names[i]
        )));
    }
    let fz = frozen_of(info, m)?;
    let bl = Blocks::new(info, m)?;
    let mut actions = Vec::new();
    let (mut z, mut ok, mut it) = attempt(r, fz, &bl, warm, p, m, &l, &h, 1.0, &d)?;
    let mut pp = p.to_vec();
    for a in &m.recover {
        if ok {
            break;
        }
        match a {
            Recover::Underrelax(w) => {
                actions.push("underrelax".to_string());
                (z, ok, it) = attempt(r, fz, &bl, warm, &pp, m, &l, &h, *w, &d)?;
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
                    (z, ok, it) = attempt(r, fz, &bl, warm, &pp, m, &l, &h, 1.0, &d)?;
                }
            }
            Recover::RevertWarmStart => {
                actions.push("revert_warm_start".to_string());
                (z, ok, it) = attempt(r, fz, &bl, initial, &pp, m, &l, &h, 1.0, &d)?;
            }
            Recover::Method(mm) => {
                // An attempt with another method from the warm start (D125).
                actions.push("method".to_string());
                let m2 = m.recover_method(mm);
                (z, ok, it) = attempt(r, fz, &bl, warm, &pp, &m2, &l, &h, 1.0, &d)?;
            }
        }
    }
    let (mut lower, mut upper, mut degenerate) = (vec![], vec![], vec![]);
    if ok {
        let g = r.g(&z, &pp);
        let eps = if m.tol.is_finite() { m.tol } else { 0.0 };
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
