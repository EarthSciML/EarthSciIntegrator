//! Algebraic equations, solving nodes and their pairing (spec/plan.md §5.9,
//! §6.4; D120, D121, D122).

use crate::error::{codes, ptr, refuse, PlanError, Result};
use crate::locality::{BandEntry, RowIdx};
use crate::model::{defined_observed, differentiated, ModelInfo};
use crate::operators::Operator;
use crate::schedule::{Kind, Node};
use earthsci_ast::{Expr, TermOrigin};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};

/// One algebraic equation of the flattened model (§5.9.1).
#[derive(Debug, Clone, PartialEq)]
pub struct Residual {
    /// `<prefix>.residual[j]`, as EarthSciAST's stripped object names it.
    pub id: String,
    /// The declaring component's path, or `coupling[k]`.
    pub origin: String,
    /// Its owner for `components` selection.
    pub owner: Option<String>,
    /// Its position among the flattened equations.
    pub equation: usize,
    pub lhs: Value,
    pub rhs: Value,
    /// The extents of rhs − lhs (empty for a scalar).
    pub extents: Vec<i64>,
    /// The names it references.
    pub refs: BTreeSet<String>,
    /// The names it reads: `refs` with the observed definitions' names,
    /// transitively.
    pub reads: BTreeSet<String>,
}

/// Whether flattened equation `lhs` is an algebraic (implicit) equation:
/// neither a state equation nor an observed's definition.
fn is_implicit(info: &ModelInfo, lhs: &Expr) -> bool {
    if differentiated(lhs).is_some() {
        return false;
    }
    !defined_observed(lhs).is_some_and(|o| info.flat.observed_variables.contains_key(o))
}

/// The solved unknowns (§4): the algebraic unknowns some algebraic equation
/// reads, directly or through observeds.
pub fn solved_unknowns(info: &ModelInfo) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for eq in &info.flat.equations {
        if !is_implicit(info, &eq.lhs) {
            continue;
        }
        let mut names = info.names_in(&eq.lhs);
        names.extend(info.names_in(&eq.rhs));
        for n in info.close_over_observeds(&names) {
            if info.flat.algebraic_variables.contains_key(&n) {
                out.insert(n);
            }
        }
    }
    out
}

fn as_int(v: &Value) -> Option<i64> {
    v.as_i64().filter(|_| v.is_i64() || v.is_u64())
}

fn range_size(info: &ModelInfo, r: &Value) -> Result<i64> {
    if let Some(from) = r.get("from").and_then(Value::as_str) {
        return Ok(info.index_sets[from]);
    }
    if let Some([lo, hi]) = r.as_array().map(Vec::as_slice) {
        if let (Some(lo), Some(hi)) = (as_int(lo), as_int(hi)) {
            return Ok(hi - lo + 1);
        }
    }
    Err(PlanError::Internal(format!(
        "a faq range {r} the planner does not read"
    )))
}

/// The extents of an expression side (§5.9.1): empty for a scalar.
fn side_extents(info: &ModelInfo, e: &Value) -> Result<Vec<i64>> {
    match e {
        Value::String(n) => Ok(var_extents(info, n)),
        Value::Object(m) => {
            let op = m.get("op").and_then(Value::as_str).unwrap_or_default();
            match op {
                "faq" => {
                    let mut out = Vec::new();
                    for o in m
                        .get("output_idx")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                    {
                        if let Some(o) = o.as_str() {
                            out.push(range_size(info, &m["ranges"][o])?);
                        }
                    }
                    Ok(out)
                }
                "index" => Ok(vec![]),
                "makearray" => Ok(m["regions"][0]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|r| r[1].as_i64().unwrap_or(0) - r[0].as_i64().unwrap_or(0) + 1)
                    .collect()),
                _ => {
                    for a in m
                        .get("args")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                    {
                        let x = side_extents(info, a)?;
                        if !x.is_empty() {
                            return Ok(x);
                        }
                    }
                    Ok(vec![])
                }
            }
        }
        _ => Ok(vec![]),
    }
}

/// The extents of variable `x` (empty for a scalar or a name with no shape).
pub fn var_extents(info: &ModelInfo, x: &str) -> Vec<i64> {
    let shape = info
        .shapes
        .get(x)
        .cloned()
        .or_else(|| info.flat.parameters.get(x).and_then(|v| v.shape.clone()))
        .or_else(|| {
            info.flat
                .observed_variables
                .get(x)
                .and_then(|v| v.shape.clone())
        })
        .unwrap_or_default();
    shape.iter().map(|s| info.index_sets[s]).collect()
}

/// The algebraic equations of the model (§5.9.1), in flattened equation order.
pub fn residuals(info: &ModelInfo) -> Result<Vec<Residual>> {
    let flat = &info.flat;
    let mut out = Vec::new();
    let mut count: BTreeMap<String, usize> = BTreeMap::new();
    for (i, eq) in flat.equations.iter().enumerate() {
        if !is_implicit(info, &eq.lhs) {
            continue;
        }
        let origin = flat
            .equation_contributions
            .get(i)
            .and_then(|c| c.first())
            .and_then(|c| c.origins.first())
            .ok_or_else(|| {
                refuse(
                    codes::TERM_ORIGIN_UNKNOWN,
                    "/model",
                    format!(
                        "algebraic equation {} of the flattened model has no origin",
                        i + 1
                    ),
                )
            })?;
        let (prefix, component) = match origin {
            TermOrigin::Component { component } => (component.clone(), Some(component.clone())),
            TermOrigin::Coupling { coupling } => (format!("coupling[{coupling}]"), None),
        };
        let j = {
            let c = count.entry(prefix.clone()).or_default();
            *c += 1;
            *c
        };
        let mut refs = info.names_in(&eq.lhs);
        refs.extend(info.names_in(&eq.rhs));
        let owner = component.or_else(|| {
            let r: BTreeSet<&str> = refs.iter().map(|n| crate::terms::component_of(n)).collect();
            (r.len() == 1).then(|| r.into_iter().next().unwrap_or_default().to_string())
        });
        let lhs = serde_json::to_value(&eq.lhs).unwrap_or(Value::Null);
        let rhs = serde_json::to_value(&eq.rhs).unwrap_or(Value::Null);
        let mut extents = side_extents(info, &rhs)?;
        if extents.is_empty() {
            extents = side_extents(info, &lhs)?;
        }
        out.push(Residual {
            id: format!("{prefix}.residual[{j}]"),
            origin: prefix,
            owner,
            equation: i,
            lhs,
            rhs,
            extents,
            reads: info.close_over_observeds(&refs),
            refs,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// The residual written cell by cell (§6.4)

fn cell_names(d: usize) -> Vec<String> {
    (1..=d).map(|k| format!("_r{k}")).collect()
}

fn mentions(v: &Value, n: &str) -> bool {
    match v {
        Value::String(s) => s == n,
        Value::Array(a) => a.iter().any(|x| mentions(x, n)),
        Value::Object(m) => m.keys().any(|k| k == n) || m.values().any(|x| mentions(x, n)),
        _ => false,
    }
}

/// Rename the index names `ren` in a serialized expression; a nested `faq`
/// binding one of them shadows it.
fn rename(x: &Value, ren: &BTreeMap<String, String>) -> Value {
    if ren.is_empty() {
        return x.clone();
    }
    match x {
        Value::String(s) => Value::String(ren.get(s).cloned().unwrap_or_else(|| s.clone())),
        Value::Array(a) => Value::Array(a.iter().map(|v| rename(v, ren)).collect()),
        Value::Object(m) => {
            let mut ren = ren.clone();
            if m.get("op").and_then(Value::as_str) == Some("faq") {
                for k in m
                    .get("ranges")
                    .and_then(Value::as_object)
                    .into_iter()
                    .flat_map(|r| r.keys())
                {
                    ren.remove(k);
                }
                for o in m
                    .get("output_idx")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    if let Some(o) = o.as_str() {
                        ren.remove(o);
                    }
                }
            }
            let mut out = Map::new();
            for (k, v) in m {
                let keep = matches!(k.as_str(), "op" | "ranges" | "output_idx" | "units");
                out.insert(k.clone(), if keep { v.clone() } else { rename(v, &ren) });
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}

/// The right-hand side of the state equation of the unknown paired with
/// residual `r`, of shape (index-set names) `shape` (§6.4).
pub fn residual_rhs(info: &ModelInfo, r: &Residual, shape: &[String], at: &str) -> Result<Value> {
    if shape.is_empty() {
        return Ok(json!({"op": "-", "args": [r.rhs, r.lhs]}));
    }
    let names = cell_names(shape.len());
    let cell = |side: &Value| -> Result<Value> {
        if names.iter().any(|n| mentions(side, n)) {
            return Err(refuse(
                codes::UNSUPPORTED_SOLVE_PAIRING,
                at,
                format!("residual {} already uses an index name the cell-by-cell form binds (spec/plan.md §6.4)", r.id),
            ));
        }
        match side {
            Value::Number(_) => return Ok(side.clone()),
            Value::String(n) => {
                if var_extents(info, n).is_empty() {
                    return Ok(side.clone());
                }
                let mut args = vec![json!(n)];
                args.extend(names.iter().map(|x| json!(x)));
                return Ok(json!({"op": "index", "args": args}));
            }
            _ => {}
        }
        if side.get("op").and_then(Value::as_str) == Some("faq") {
            let o: Vec<&Value> = side["output_idx"]
                .as_array()
                .into_iter()
                .flatten()
                .collect();
            if o.len() != names.len() || o.iter().any(|x| !x.is_string()) {
                return Err(refuse(
                    codes::UNSUPPORTED_SOLVE_PAIRING,
                    at,
                    format!("residual {}: a faq side whose output indices are not one name per dimension of its unknown (spec/plan.md §6.4)", r.id),
                ));
            }
            let ren: BTreeMap<String, String> = o
                .iter()
                .zip(&names)
                .map(|(a, b)| (a.as_str().unwrap_or_default().to_string(), b.clone()))
                .collect();
            let body = rename(&side["expr"], &ren);
            let rest: Map<String, Value> = side["ranges"]
                .as_object()
                .into_iter()
                .flatten()
                .filter(|(k, _)| !ren.contains_key(*k))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            if rest.is_empty() {
                return Ok(body);
            }
            if side.get("filter").is_some() {
                return Err(refuse(
                    codes::UNSUPPORTED_SOLVE_PAIRING,
                    at,
                    format!(
                        "residual {}: a filtered reducing faq side (spec/plan.md §6.4)",
                        r.id
                    ),
                ));
            }
            return Ok(
                json!({"op": "faq", "args": [], "output_idx": [], "ranges": rest, "expr": body}),
            );
        }
        if side_extents(info, side)?.is_empty() {
            return Ok(side.clone());
        }
        Err(refuse(
            codes::UNSUPPORTED_SOLVE_PAIRING,
            at,
            format!("residual {} has an array-valued side that is neither a name nor a faq; the planner does not read it at a cell (spec/plan.md §6.4)", r.id),
        ))
    };
    let ranges: Map<String, Value> = names
        .iter()
        .zip(shape)
        .map(|(n, s)| (n.clone(), json!({"from": s})))
        .collect();
    Ok(json!({
        "op": "faq", "args": [], "output_idx": names, "ranges": ranges,
        "expr": {"op": "-", "args": [cell(&r.rhs)?, cell(&r.lhs)?]},
    }))
}

// ---------------------------------------------------------------------------
// Solving nodes (§5.9.2) and the schedule's algebraic checks (rules 18, 19)

/// The stiffly accurate methods a DAE may use (spec/run-config.md §4.3).
pub const DAE_METHODS: [&str; 6] = [
    "Rodas3", "Rodas4", "Rodas5P", "TRBDF2", "Kvaerno3", "Kvaerno4",
];

pub fn is_algebraic(ops: &BTreeMap<String, Operator>, o: &str) -> bool {
    ops.get(o).is_some_and(|op| !op.residuals.is_empty())
}

/// Whether `n` is a solving node: a `solve` node, or a `coupled` node naming
/// an algebraic operator.
pub fn is_solving(n: &Node, ops: &BTreeMap<String, Operator>) -> bool {
    match &n.kind {
        Kind::Solve => true,
        Kind::Leaf {
            ops: names,
            coupled: true,
        } => names.iter().any(|o| is_algebraic(ops, o)),
        _ => false,
    }
}

/// The operator a `solve` node names.
pub fn solve_op(n: &Node) -> String {
    n.value
        .get("solve")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// The algebraic operators a solving node names, in its order.
pub fn node_algebraic_ops(n: &Node, ops: &BTreeMap<String, Operator>) -> Vec<String> {
    match &n.kind {
        Kind::Solve => vec![solve_op(n)],
        Kind::Leaf { ops: names, .. } => names
            .iter()
            .filter(|o| is_algebraic(ops, o))
            .cloned()
            .collect(),
        _ => vec![],
    }
}

/// A `solve` node's `unknowns` as written.
fn written_unknowns(n: &Node) -> Vec<String> {
    n.value
        .get("unknowns")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect()
}

/// A solving node's unknowns (§5.9.2), in code-point order.
pub fn node_unknowns(n: &Node, ops: &BTreeMap<String, Operator>, info: &ModelInfo) -> Vec<String> {
    if matches!(n.kind, Kind::Solve) {
        let mut u = written_unknowns(n);
        u.sort();
        return u;
    }
    let mut out = BTreeSet::new();
    for o in node_algebraic_ops(n, ops) {
        for r in &ops[&o].residuals {
            out.extend(r.reads.iter().filter(|z| info.solved.contains(*z)).cloned());
        }
    }
    out.into_iter().collect()
}

/// The method objects of a solving node with their keys.
fn node_methods(n: &Node) -> Vec<(&Value, &'static str)> {
    match &n.kind {
        Kind::Solve => n
            .value
            .get("method")
            .map(|m| vec![(m, "method")])
            .unwrap_or_default(),
        Kind::Leaf { coupled: true, .. } => n
            .value
            .get("initialize")
            .map(|m| vec![(m, "initialize")])
            .unwrap_or_default(),
        _ => vec![],
    }
}

pub fn is_bounded(info: &ModelInfo, z: &str) -> bool {
    let v = &info.flat.state_variables[z];
    v.lb.is_some() || v.ub.is_some()
}

/// The name a bound refers to, when it is a reference.
fn bound_name(b: &Option<earthsci_ast::VariableBound>) -> Option<String> {
    match b {
        Some(earthsci_ast::VariableBound::Reference(s)) => Some(s.clone()),
        _ => None,
    }
}

fn need_observed(info: &ModelInfo, name: &str, at: String, what: &str) -> Result<()> {
    if !info.is_name(name) {
        return Err(refuse(
            codes::NAME_UNRESOLVED,
            at,
            format!("{what} {name} is not a name of the flattened model"),
        ));
    }
    if !info.is_observed(name) {
        return Err(refuse(
            codes::NAME_WRONG_KIND,
            at,
            format!("{what} {name} is not an observed"),
        ));
    }
    Ok(())
}

/// Rule 19 in spec/plan.md §7.1's order, then rule 2 on the names a solve
/// method gives and rule 18's checks that need no derivative.
pub fn check_schedule(
    root: &Node,
    ops: &BTreeMap<String, Operator>,
    info: &ModelInfo,
) -> Result<()> {
    let nodes = root.preorder();
    for n in &nodes {
        match &n.kind {
            Kind::Solve => {
                let o = solve_op(n);
                if !is_algebraic(ops, &o) {
                    return Err(refuse(
                        codes::OPERATOR_KIND_MISMATCH,
                        n.key_ptr("solve"),
                        format!("solve node {} names operator {o}, which holds right-hand-side terms, not algebraic equations", n.id()),
                    ));
                }
            }
            Kind::Leaf {
                ops: names,
                coupled: false,
            } => {
                for (i, o) in names.iter().enumerate() {
                    if is_algebraic(ops, o) {
                        return Err(refuse(
                            codes::OPERATOR_KIND_MISMATCH,
                            n.op_ptr(i),
                            format!("leaf {} names algebraic operator {o}; a solve node or a coupled node advances it", n.id()),
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    for n in &nodes {
        if !matches!(n.kind, Kind::Solve) {
            continue;
        }
        for (i, z) in written_unknowns(n).iter().enumerate() {
            let at = ptr(&n.key_ptr("unknowns"), i);
            if !info.is_name(z) {
                return Err(refuse(
                    codes::NAME_UNRESOLVED,
                    at,
                    format!("solve unknown {z} is not a name of the flattened model"),
                ));
            }
            if !info.solved.contains(z) {
                return Err(refuse(
                    codes::NAME_WRONG_KIND,
                    at,
                    format!(
                        "solve unknown {z} is not an algebraic unknown an algebraic equation reads"
                    ),
                ));
            }
        }
    }
    let mut solver: BTreeMap<String, String> = BTreeMap::new();
    for n in &nodes {
        if !is_solving(n, ops) {
            continue;
        }
        for z in node_unknowns(n, ops, info) {
            if let Some(first) = solver.get(&z) {
                return Err(refuse(
                    codes::ALGEBRAIC_UNKNOWN_SOLVED_TWICE,
                    n.pointer.clone(),
                    format!(
                        "algebraic unknown {z} is solved by node {first} and by node {}",
                        n.id()
                    ),
                ));
            }
            solver.insert(z, n.id().to_string());
        }
    }
    if let Some(z) = info.solved.iter().find(|z| !solver.contains_key(*z)) {
        return Err(refuse(
            codes::ALGEBRAIC_UNKNOWN_UNSOLVED,
            "/model",
            format!("algebraic unknown {z} is read by an algebraic equation, and no solve or coupled node solves it"),
        ));
    }
    for n in &nodes {
        if !(matches!(n.kind, Kind::Leaf { coupled: true, .. }) && is_solving(n, ops)) {
            continue;
        }
        let alg = n.value.get("alg").and_then(Value::as_str).unwrap_or("auto");
        if alg != "auto" && !DAE_METHODS.contains(&alg) {
            return Err(refuse(
                codes::DAE_METHOD_INVALID,
                n.key_ptr("alg"),
                format!("coupled node {} holds algebraic operators, an index-1 DAE, and {alg} is not one of the stiffly accurate methods {}", n.id(), DAE_METHODS.join(", ")),
            ));
        }
        if n.value.contains_key("imex") {
            return Err(refuse(
                codes::DAE_METHOD_INVALID,
                n.key_ptr("imex"),
                format!(
                    "coupled node {} is an index-1 DAE; imex is not specified for it",
                    n.id()
                ),
            ));
        }
        if let Some(z) = node_unknowns(n, ops, info)
            .iter()
            .find(|z| is_bounded(info, z))
        {
            return Err(refuse(
                codes::UNSUPPORTED_DAE_BOUNDS,
                n.pointer.clone(),
                format!("coupled node {} is an index-1 DAE with the bounded unknown {z}; a complementarity DAE is not specified", n.id()),
            ));
        }
    }
    for n in &nodes {
        for (meth, key) in node_methods(n) {
            if meth.get("linesearch").and_then(Value::as_str) == Some("strong_wolfe") {
                return Err(refuse(
                    codes::UNSUPPORTED_SOLVE_OPTION,
                    ptr(&n.key_ptr(key), "linesearch"),
                    "linesearch strong_wolfe is not specified in v1 (spec/methods.md §15.6)",
                ));
            }
            if meth.get("anderson").and_then(Value::as_i64).unwrap_or(0) > 0 {
                return Err(refuse(
                    codes::UNSUPPORTED_SOLVE_OPTION,
                    ptr(&n.key_ptr(key), "anderson"),
                    "Anderson acceleration is not specified in v1 (spec/methods.md §15.6)",
                ));
            }
        }
    }
    for n in &nodes {
        // The names the node's residuals read: a frozen observed must be
        // among them; a rel_change observed need only be an observed
        // (spec/plan.md §6.5).
        let rd: BTreeSet<&String> = node_algebraic_ops(n, ops)
            .iter()
            .filter_map(|o| ops.get(o))
            .flat_map(|op| op.residuals.iter().flat_map(|r| r.reads.iter()))
            .collect();
        let frozen_name = |o: &str, at: String, what: &str| -> Result<()> {
            need_observed(info, o, at.clone(), what)?;
            if !rd.contains(&o.to_string()) {
                return Err(refuse(
                    codes::NAME_WRONG_KIND,
                    at,
                    format!(
                        "{what} {o} is not read by the residuals of node {} (spec/plan.md §6.5)",
                        n.id()
                    ),
                ));
            }
            Ok(())
        };
        for (meth, key) in node_methods(n) {
            if let Some(fr) = meth
                .get("picard")
                .and_then(|p| p.get("freeze"))
                .and_then(Value::as_array)
            {
                for (i, o) in fr.iter().enumerate() {
                    let at = ptr(&ptr(&ptr(&n.key_ptr(key), "picard"), "freeze"), i);
                    frozen_name(o.as_str().unwrap_or_default(), at, "frozen observed")?;
                }
            }
            if let Some(o) = meth
                .get("converged")
                .and_then(|c| c.get("rel_change"))
                .and_then(Value::as_str)
            {
                let at = ptr(&ptr(&n.key_ptr(key), "converged"), "rel_change");
                need_observed(info, o, at, "rel_change observed")?;
            }
        }
    }
    for n in &nodes {
        if !matches!(n.kind, Kind::Solve) {
            continue;
        }
        let zs = node_unknowns(n, ops, info);
        if !zs.iter().any(|z| is_bounded(info, z)) {
            continue;
        }
        if n.value
            .get("method")
            .and_then(|m| m.get("converged"))
            .and_then(|c| c.get("rel_change"))
            .is_some()
        {
            return Err(refuse(
                codes::SOLVE_BOUNDS_INVALID,
                ptr(&ptr(&n.key_ptr("method"), "converged"), "rel_change"),
                format!("solve node {} has a bounded unknown, which converges on its reformulated residual only (spec/run-config.md §4.10)", n.id()),
            ));
        }
        for (i, z) in written_unknowns(n).iter().enumerate() {
            let v = &info.flat.state_variables[z];
            for b in [bound_name(&v.lb), bound_name(&v.ub)].into_iter().flatten() {
                let reads = info.close_over_observeds(&[b.clone()].into());
                if reads.iter().any(|y| zs.contains(y)) {
                    return Err(refuse(
                        codes::SOLVE_BOUNDS_INVALID,
                        ptr(&n.key_ptr("unknowns"), i),
                        format!("the bound {b} of {z} is, or depends on, an unknown of solve node {}, so it would move during the solve", n.id()),
                    ));
                }
            }
        }
    }
    // §5.9.7: a bound is a number or a parameter at this version.
    for n in &nodes {
        if !matches!(n.kind, Kind::Solve) {
            continue;
        }
        for (i, z) in written_unknowns(n).iter().enumerate() {
            let v = &info.flat.state_variables[z];
            for b in [bound_name(&v.lb), bound_name(&v.ub)].into_iter().flatten() {
                if !info.is_parameter(&b) {
                    return Err(refuse(
                        codes::UNSUPPORTED_BOUND_REFERENCE,
                        ptr(&n.key_ptr("unknowns"), i),
                        format!("the bound {b} of {z} is not a parameter; a bound that refers to a state or an observed is a later addition (spec/plan.md §5.9.7)"),
                    ));
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The pairing (§5.9.3–§5.9.6)

/// A solving node with its residual blocks and unknown blocks.
pub struct SolvingNode<'a> {
    pub node: &'a Node,
    pub ops: Vec<String>,
    pub residuals: Vec<Residual>,
    pub unknowns: Vec<String>,
}

/// The maximum flow of §5.9.3's network.
fn block_flow(rcap: &[i64], zcap: &[i64], edges: &[Vec<usize>]) -> i64 {
    let (nr, nz) = (rcap.len(), zcap.len());
    let n = nr + nz + 2;
    let (s, t) = (n - 2, n - 1);
    let mut cap = vec![vec![0i64; n]; n];
    let big = rcap.iter().sum::<i64>() + 1;
    for k in 0..nr {
        cap[s][k] = rcap[k];
        for &j in &edges[k] {
            cap[k][nr + j] = big;
        }
    }
    for j in 0..nz {
        cap[nr + j][t] = zcap[j];
    }
    let mut flow = 0;
    loop {
        let mut prev = vec![usize::MAX; n];
        prev[s] = s;
        let mut q = std::collections::VecDeque::from([s]);
        while let Some(u) = q.pop_front() {
            if prev[t] != usize::MAX {
                break;
            }
            for v in 0..n {
                if prev[v] == usize::MAX && cap[u][v] > 0 {
                    prev[v] = u;
                    q.push_back(v);
                }
            }
        }
        if prev[t] == usize::MAX {
            return flow;
        }
        let mut f = i64::MAX;
        let mut v = t;
        while v != s {
            f = f.min(cap[prev[v]][v]);
            v = prev[v];
        }
        let mut v = t;
        while v != s {
            cap[prev[v]][v] -= f;
            cap[v][prev[v]] += f;
            v = prev[v];
        }
        flow += f;
    }
}

fn check_singular(info: &ModelInfo, sn: &SolvingNode) -> Result<()> {
    let at = sn.node.pointer.clone();
    let rcap: Vec<i64> = sn
        .residuals
        .iter()
        .map(|r| r.extents.iter().product())
        .collect();
    let zcap: Vec<i64> = sn
        .unknowns
        .iter()
        .map(|z| var_extents(info, z).iter().product())
        .collect();
    let (ne, nz): (i64, i64) = (rcap.iter().sum(), zcap.iter().sum());
    if ne != nz {
        return Err(refuse(
            codes::SOLVE_STRUCTURALLY_SINGULAR,
            at,
            format!("node {} has {ne} residual elements and {nz} unknown elements (spec/plan.md §5.9.3)", sn.node.id()),
        ));
    }
    let edges: Vec<Vec<usize>> = sn
        .residuals
        .iter()
        .map(|r| {
            (0..sn.unknowns.len())
                .filter(|&j| r.reads.contains(&sn.unknowns[j]))
                .collect()
        })
        .collect();
    let f = block_flow(&rcap, &zcap, &edges);
    if f < nz {
        return Err(refuse(
            codes::SOLVE_STRUCTURALLY_SINGULAR,
            at,
            format!("node {}: no matching of its residuals with its unknowns can cover every unknown element (a maximum flow of {f} of {nz}, spec/plan.md §5.9.3)", sn.node.id()),
        ));
    }
    Ok(())
}

/// A band's diagonal box (§5.9.4), or `None` when it is empty.
pub fn diagonal_box(b: &BandEntry) -> Option<Vec<(i64, i64)>> {
    if b.row_idx.len() != b.col_idx.len() || b.rows.len() != b.row_idx.len() {
        return None;
    }
    let contracted: BTreeMap<&str, (i64, i64)> = b
        .contracted
        .iter()
        .map(|(n, lo, hi)| (n.as_str(), (*lo, *hi)))
        .collect();
    let mut out = Vec::new();
    for d in 0..b.rows.len() {
        let (mut lo, mut hi) = b.rows[d];
        let c = &b.col_idx[d];
        match (c, &b.row_idx[d]) {
            (Expr::Variable(v), RowIdx::Name(r)) if v == r => {}
            (Expr::Variable(v), _) if contracted.contains_key(v.as_str()) => {
                let (clo, chi) = contracted[v.as_str()];
                lo = lo.max(clo);
                hi = hi.min(chi);
            }
            _ => {
                let k = crate::locality::int_const(c)?;
                match &b.row_idx[d] {
                    RowIdx::Name(_) => {
                        lo = lo.max(k);
                        hi = hi.min(k);
                    }
                    RowIdx::Fixed(rho) => {
                        if *rho != k {
                            return None;
                        }
                    }
                }
            }
        }
        if lo > hi {
            return None;
        }
        out.push((lo, hi));
    }
    Some(out)
}

/// Whether the union of `boxes` covers the block of extents `ext` (§5.9.4).
pub fn covers_block(boxes: &[Vec<(i64, i64)>], ext: &[i64]) -> bool {
    if ext.is_empty() {
        return !boxes.is_empty();
    }
    if boxes.is_empty() {
        return false;
    }
    let dims = ext.len();
    let cuts: Vec<Vec<i64>> = (0..dims)
        .map(|d| {
            let mut c: BTreeSet<i64> = [1, ext[d] + 1].into();
            for b in boxes {
                c.insert(b[d].0);
                c.insert(b[d].1 + 1);
            }
            c.into_iter()
                .filter(|&x| 1 <= x && x <= ext[d] + 1)
                .collect()
        })
        .collect();
    let mut idx = vec![0usize; dims];
    loop {
        let p: Vec<i64> = (0..dims).map(|d| cuts[d][idx[d]]).collect();
        if !boxes
            .iter()
            .any(|b| (0..dims).all(|d| b[d].0 <= p[d] && p[d] <= b[d].1))
        {
            return false;
        }
        // Next cell, first dimension fastest.
        let mut d = 0;
        loop {
            idx[d] += 1;
            if idx[d] + 1 < cuts[d].len() {
                break;
            }
            idx[d] = 0;
            d += 1;
            if d == dims {
                return true;
            }
        }
    }
}

/// The probe's state for residual block `k` (1-based) (§5.9.4).
pub fn residual_probe_state(k: usize) -> String {
    format!("{}.res_{k}", crate::derived::PROBE_COMPONENT)
}

fn diagonal_pairs(info: &ModelInfo, sn: &SolvingNode) -> Result<Vec<Vec<usize>>> {
    let at = sn.node.pointer.clone();
    let mut cand = Vec::new();
    let mut probe: Vec<(usize, String)> = Vec::new();
    for (k, r) in sn.residuals.iter().enumerate() {
        let c: Vec<usize> = (0..sn.unknowns.len())
            .filter(|&j| {
                r.reads.contains(&sn.unknowns[j]) && var_extents(info, &sn.unknowns[j]) == r.extents
            })
            .collect();
        if let Some(&j) = c.first() {
            probe.push((k + 1, sn.unknowns[j].clone()));
        }
        cand.push(c);
    }
    let mut out = vec![Vec::new(); sn.residuals.len()];
    if probe.is_empty() {
        return Ok(out);
    }
    let mut extra = Vec::new();
    let mut refs = BTreeSet::new();
    for (k, z) in &probe {
        let r = &sn.residuals[k - 1];
        let shape = info.shapes[z].clone();
        let mut decl = Map::new();
        decl.insert("type".into(), json!("unknown"));
        decl.insert("default".into(), json!(0.0));
        if !shape.is_empty() {
            decl.insert("shape".into(), json!(shape));
        }
        extra.push((
            residual_probe_state(*k),
            Value::Object(decl),
            residual_rhs(info, r, &shape, &at)?,
        ));
        refs.extend(r.refs.iter().cloned());
    }
    let doc = crate::derived::residual_probe_document(info, &extra, &refs);
    let entries = crate::astdiff::document_structure(doc).map_err(|m| {
        refuse(
            codes::BAND_MODEL_FAILED,
            "/model",
            format!("the residuals of node {}: {m}", sn.node.id()),
        )
    })?;
    let bands: Vec<BandEntry> = entries
        .iter()
        .map(crate::astdiff::structure_entry)
        .collect();
    for (k, _) in &probe {
        let row = residual_probe_state(*k);
        for &j in &cand[k - 1] {
            let boxes: Vec<Vec<(i64, i64)>> = bands
                .iter()
                .filter(|b| b.row == row && b.col == sn.unknowns[j])
                .filter_map(diagonal_box)
                .collect();
            if covers_block(&boxes, &sn.residuals[k - 1].extents) {
                out[k - 1].push(j);
            }
        }
    }
    Ok(out)
}

fn canonical_pairing(sn: &SolvingNode, diag: &[Vec<usize>]) -> Option<Vec<usize>> {
    let order: Vec<Vec<usize>> = (0..sn.residuals.len())
        .map(|k| {
            let own = sn.residuals[k].owner.as_deref();
            let mine =
                |j: &usize| own.is_some_and(|o| crate::terms::component_of(&sn.unknowns[*j]) == o);
            let mut a: Vec<usize> = diag[k].iter().copied().filter(|j| mine(j)).collect();
            let mut b: Vec<usize> = diag[k].iter().copied().filter(|j| !mine(j)).collect();
            a.sort_by(|x, y| sn.unknowns[*x].cmp(&sn.unknowns[*y]));
            b.sort_by(|x, y| sn.unknowns[*x].cmp(&sn.unknowns[*y]));
            a.extend(b);
            a
        })
        .collect();
    let mut partner: Vec<Option<usize>> = vec![None; sn.unknowns.len()];
    fn augment(
        k: usize,
        order: &[Vec<usize>],
        partner: &mut Vec<Option<usize>>,
        seen: &mut BTreeSet<usize>,
    ) -> bool {
        for &j in &order[k] {
            if !seen.insert(j) {
                continue;
            }
            let free = match partner[j] {
                None => true,
                Some(k2) => augment(k2, order, partner, seen),
            };
            if free {
                partner[j] = Some(k);
                return true;
            }
        }
        false
    }
    for k in 0..sn.residuals.len() {
        if !augment(k, &order, &mut partner, &mut BTreeSet::new()) {
            return None;
        }
    }
    let mut pair = vec![0; sn.residuals.len()];
    for (j, k) in partner.iter().enumerate() {
        if let Some(k) = k {
            pair[*k] = j;
        }
    }
    Some(pair)
}

fn alternating_cycle(sn: &SolvingNode, diag: &[Vec<usize>], pair: &[usize]) -> Option<Vec<usize>> {
    let nr = sn.residuals.len();
    let mut partner = vec![0; sn.unknowns.len()];
    for (k, &j) in pair.iter().enumerate() {
        partner[j] = k;
    }
    let succ = |v: usize| -> Vec<usize> {
        if v < nr {
            diag[v]
                .iter()
                .filter(|&&j| j != pair[v])
                .map(|&j| nr + j)
                .collect()
        } else {
            vec![partner[v - nr]]
        }
    };
    let mut state = vec![0u8; nr + sn.unknowns.len()];
    let mut stack: Vec<usize> = Vec::new();
    fn dfs(
        v: usize,
        succ: &dyn Fn(usize) -> Vec<usize>,
        state: &mut Vec<u8>,
        stack: &mut Vec<usize>,
    ) -> Option<Vec<usize>> {
        state[v] = 1;
        stack.push(v);
        for w in succ(v) {
            if state[w] == 1 {
                let i = stack.iter().position(|&x| x == w).unwrap_or(0);
                return Some(stack[i..].to_vec());
            }
            if state[w] == 0 {
                if let Some(c) = dfs(w, succ, state, stack) {
                    return Some(c);
                }
            }
        }
        stack.pop();
        state[v] = 2;
        None
    }
    for k in 0..nr {
        if state[k] == 0 {
            if let Some(c) = dfs(k, &succ, &mut state, &mut stack) {
                return Some(c);
            }
        }
    }
    None
}

/// §5.9.3–§5.9.6 for one solving node: the unknown of each residual block.
pub fn pair_node(info: &ModelInfo, sn: &SolvingNode) -> Result<Vec<String>> {
    let at = sn.node.pointer.clone();
    check_singular(info, sn)?;
    let diag = diagonal_pairs(info, sn)?;
    let pair = canonical_pairing(sn, &diag).ok_or_else(|| {
        refuse(
            codes::UNSUPPORTED_SOLVE_PAIRING,
            at.clone(),
            format!("node {} has no diagonal pairing of its residuals with its unknowns; an element-level matching is not yet specified (spec/plan.md §5.9.5)", sn.node.id()),
        )
    })?;
    if let Some(cyc) = alternating_cycle(sn, &diag, &pair) {
        let nr = sn.residuals.len();
        if cyc
            .iter()
            .any(|&v| v >= nr && is_bounded(info, &sn.unknowns[v - nr]))
        {
            let blocks: Vec<&str> = cyc
                .iter()
                .map(|&v| {
                    if v < nr {
                        sn.residuals[v].id.as_str()
                    } else {
                        sn.unknowns[v - nr].as_str()
                    }
                })
                .collect();
            return Err(refuse(
                codes::SOLVE_PAIRING_AMBIGUOUS,
                at,
                format!("node {} can pair its residuals with its bounded unknowns more than one way (around {}), and the pairing decides the complementarity conditions (spec/plan.md §5.9.6)", sn.node.id(), blocks.join(", ")),
            ));
        }
    }
    Ok(pair.into_iter().map(|j| sn.unknowns[j].clone()).collect())
}

// ---------------------------------------------------------------------------
// Resolved solve nodes (§7.1)

/// The default consistent initialization of a DAE (spec/run-config.md §4.3).
pub fn default_initialize() -> Value {
    json!({"newton": {}, "linesearch": "backtracking", "converged": {"residual": 1.0e-10}, "maxiter": 100})
}

/// A solve method (or a DAE's `initialize`) with every option defaulted;
/// `linsolve` the resolved linear solver.
pub fn resolve_method(meth: &Value, linsolve: Value) -> Value {
    let mut out = Map::new();
    if let Some(p) = meth.get("ptc") {
        out.insert(
            "ptc".into(),
            json!({
                "dt0": p.get("dt0").and_then(Value::as_f64).unwrap_or(1.0e-3),
                "growth": p.get("growth").and_then(Value::as_f64).unwrap_or(2.0),
            }),
        );
    } else if let Some(p) = meth.get("picard") {
        out.insert("picard".into(), json!({"freeze": p["freeze"]}));
    } else {
        out.insert(
            "newton".into(),
            meth.get("newton").cloned().unwrap_or_else(|| json!({})),
        );
    }
    out.insert("linsolve".into(), linsolve);
    if meth.get("ptc").is_none() {
        out.insert(
            "linesearch".into(),
            meth.get("linesearch")
                .cloned()
                .unwrap_or_else(|| json!("none")),
        );
    }
    out.insert(
        "converged".into(),
        meth.get("converged")
            .cloned()
            .unwrap_or_else(|| json!({"residual": 1.0e-10})),
    );
    out.insert(
        "maxiter".into(),
        meth.get("maxiter").cloned().unwrap_or_else(|| json!(100)),
    );
    if let Some(r) = meth.get("recover") {
        out.insert("recover".into(), r.clone());
    }
    Value::Object(out)
}

/// The plan's `bounds` of an algebraic operator (§5.9.7): per bounded paired
/// unknown its `lb` and `ub`; `None` when none is bounded.
pub fn bounds_json(info: &ModelInfo, op: &Operator) -> Option<Value> {
    let mut out = Map::new();
    for z in op.pairing.iter().collect::<BTreeSet<_>>() {
        let v = &info.flat.state_variables[z];
        let mut b = Map::new();
        for (k, x) in [("lb", &v.lb), ("ub", &v.ub)] {
            if let Some(x) = x {
                b.insert(k.into(), serde_json::to_value(x).unwrap_or(Value::Null));
            }
        }
        if !b.is_empty() {
            out.insert(z.clone(), Value::Object(b));
        }
    }
    (!out.is_empty()).then_some(Value::Object(out))
}

// ---------------------------------------------------------------------------
// The frozen documents (spec/plan.md §6.5, D124)

/// The observeds a solving node names (picard freeze in written order, then
/// a rel_change observed not among them) and whether it is a picard solve.
pub fn node_frozen(n: &Node) -> (Vec<String>, bool) {
    let meth = match &n.kind {
        Kind::Solve => n.value.get("method"),
        _ => n.value.get("initialize"),
    };
    let Some(meth) = meth else {
        return (vec![], false);
    };
    let mut out: Vec<String> = Vec::new();
    let picard = meth.get("picard").is_some();
    if let Some(fr) = meth
        .get("picard")
        .and_then(|p| p.get("freeze"))
        .and_then(Value::as_array)
    {
        out.extend(fr.iter().filter_map(Value::as_str).map(str::to_string));
    }
    if let Some(rc) = meth
        .get("converged")
        .and_then(|c| c.get("rel_change"))
        .and_then(Value::as_str)
    {
        if !out.iter().any(|o| o == rc) {
            out.push(rc.to_string());
        }
    }
    (out, picard)
}

/// The probe component's frozen state for the k-th observed (1-based).
pub fn frozen_state(k: usize) -> String {
    format!("{}.frozen_{k}", crate::derived::FROZEN_COMPONENT)
}

/// The shape of observed `o` (empty when it has none).
fn observed_shape(info: &ModelInfo, o: &str) -> Vec<String> {
    info.flat
        .observed_variables
        .get(o)
        .and_then(|v| v.shape.clone())
        .unwrap_or_default()
}

/// A frozen state's declaration (§6.5): the observed's shape, `default` 0.0.
pub fn frozen_declaration(info: &ModelInfo, o: &str) -> Value {
    let mut d = Map::new();
    d.insert("type".into(), json!("unknown"));
    d.insert("default".into(), json!(0.0));
    let shape = observed_shape(info, o);
    if !shape.is_empty() {
        d.insert("shape".into(), json!(shape));
    }
    Value::Object(d)
}

/// `D(frozen_k) = O_k`, an array observed read cell by cell (§6.5).
pub fn frozen_rhs(info: &ModelInfo, o: &str) -> Value {
    let shape = observed_shape(info, o);
    if shape.is_empty() {
        return json!(o);
    }
    let names = cell_names(shape.len());
    let ranges: Map<String, Value> = names
        .iter()
        .zip(&shape)
        .map(|(n, s)| (n.clone(), json!({"from": s})))
        .collect();
    let mut args = vec![json!(o)];
    args.extend(names.iter().map(|x| json!(x)));
    json!({"op": "faq", "args": [], "output_idx": names, "ranges": ranges, "expr": {"op": "index", "args": args}})
}

/// The zero right-hand side over observed `o`'s shape.
pub fn frozen_zero(info: &ModelInfo, o: &str) -> Value {
    let shape = observed_shape(info, o);
    if shape.is_empty() {
        return json!(0.0);
    }
    let region: Vec<Value> = shape
        .iter()
        .map(|s| json!([1, info.index_sets[s]]))
        .collect();
    json!({"op": "makearray", "args": [], "regions": [region], "values": [0.0]})
}

/// Rename every reference to the names of `ren` in the right-hand sides and
/// operator bindings of a document's equations (`models`, recursively).
pub fn rename_refs(models: &mut Value, ren: &BTreeMap<String, String>) {
    fn rn(x: &Value, ren: &BTreeMap<String, String>) -> Value {
        match x {
            Value::String(s) => Value::String(ren.get(s).cloned().unwrap_or_else(|| s.clone())),
            Value::Array(a) => Value::Array(a.iter().map(|v| rn(v, ren)).collect()),
            Value::Object(m) => Value::Object(
                m.iter()
                    .map(|(k, v)| {
                        let keep = matches!(k.as_str(), "op" | "ranges" | "output_idx" | "units");
                        (k.clone(), if keep { v.clone() } else { rn(v, ren) })
                    })
                    .collect(),
            ),
            other => other.clone(),
        }
    }
    if let Value::Object(top) = models {
        for (_, node) in top.iter_mut() {
            if let Some(Value::Array(eqs)) = node.get_mut("equations") {
                for eq in eqs.iter_mut() {
                    let r = rn(&eq["rhs"], ren);
                    eq["rhs"] = r;
                }
            }
            if let Some(subs) = node.get_mut("subsystems") {
                rename_refs(subs, ren);
            }
        }
    }
}
