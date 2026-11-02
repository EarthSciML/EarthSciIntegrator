//! Pass 6, the derived documents (spec/plan.md §6, D68, D69).
//!
//! Both documents are JSON objects written by EarthSciASTDiff's
//! `stable_json` (§2.2). The right-hand-side document (§6.1) declares the
//! flattened model's states, parameters and live observeds in models named
//! after their components and holds one operator's terms; the Jacobian
//! document (§6.2) is EarthSciASTDiff's evaluation document of it, which
//! zeroes every state equation and adds the band fields `JacobianEval.J_k`.

use crate::astdiff::FlatView;
use crate::error::{PlanError, Result};
use crate::model::{defined_observed, differentiated, lifted_ranges, node, ModelInfo};
use crate::terms::{component_of, Term};
use earthsci_ast::{Domain, Expr, ModelVariable, VariableType};
use earthsci_astdiff::JacEntry;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};

/// The esm version a derived document declares (§6.1).
pub const ESM_VERSION: &str = "1.3.0";

/// The model the Jacobian document adds for its band fields, EarthSciASTDiff's
/// evaluation model (§6.2, §6.1.4).
pub const JACOBIAN_MODEL: &str = crate::model::RESERVED_COMPONENT;

/// A derived document's bytes: EarthSciASTDiff's `stable_json` (§2.2).
pub fn to_bytes(doc: &Value) -> Vec<u8> {
    earthsci_astdiff::stable_json(doc).into_bytes()
}

/// A model or subsystem of the component tree (§6.1.1).
#[derive(Default)]
struct Component {
    variables: Map<String, Value>,
    equations: Vec<Value>,
    subsystems: BTreeMap<String, Component>,
}

impl Component {
    fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("variables".into(), Value::Object(self.variables.clone()));
        out.insert("equations".into(), Value::Array(self.equations.clone()));
        if !self.subsystems.is_empty() {
            let subs: Map<String, Value> = self
                .subsystems
                .iter()
                .map(|(k, c)| (k.clone(), c.to_json()))
                .collect();
            out.insert("subsystems".into(), Value::Object(subs));
        }
        Value::Object(out)
    }
}

/// The component tree: top-level models by name.
#[derive(Default)]
struct Tree(BTreeMap<String, Component>);

impl Tree {
    /// The component at dotted path `path`, created with its ancestors.
    fn at(&mut self, path: &str) -> &mut Component {
        let mut segs = path.split('.');
        let first = segs.next().unwrap_or_default();
        let mut c = self.0.entry(first.to_string()).or_default();
        for s in segs {
            c = c.subsystems.entry(s.to_string()).or_default();
        }
        c
    }

    /// Declares the flattened variable `name` in its component.
    fn declare(&mut self, name: &str, var: Value) {
        let local = name.rsplit_once('.').map(|(_, v)| v).unwrap_or(name);
        self.at(component_of(name))
            .variables
            .insert(local.to_string(), var);
    }

    fn push(&mut self, subject: &str, eq: Value) {
        self.at(component_of(subject)).equations.push(eq);
    }

    fn to_json(&self) -> Value {
        Value::Object(
            self.0
                .iter()
                .map(|(k, c)| (k.clone(), c.to_json()))
                .collect(),
        )
    }
}

fn expr_json(e: &Expr) -> Value {
    serde_json::to_value(e).unwrap_or(Value::Null)
}

/// A variable's declaration (§6.1.1): `{"type": T}` plus `shape` and
/// `default` when present.
fn variable_json(v: &ModelVariable, shape: &[String]) -> Value {
    let mut out = Map::new();
    let t = if v.var_type == VariableType::Parameter {
        "parameter"
    } else {
        "unknown"
    };
    out.insert("type".into(), json!(t));
    if !shape.is_empty() {
        out.insert("shape".into(), json!(shape));
    }
    if let Some(d) = &v.default {
        out.insert(
            "default".into(),
            serde_json::to_value(d).unwrap_or(Value::Null),
        );
    }
    Value::Object(out)
}

/// The zero right-hand side of a state (§6.1.2): `0.0` for a scalar, a
/// `makearray` of `0.0` over the whole array otherwise.
pub fn zero(info: &ModelInfo, state: &str) -> Value {
    let shape = &info.shapes[state];
    if shape.is_empty() {
        return json!(0.0);
    }
    let region: Vec<Value> = shape
        .iter()
        .map(|s| json!([1, info.index_sets[s]]))
        .collect();
    json!({"op": "makearray", "args": [], "regions": [region], "values": [0.0]})
}

/// The left fold `((t_1 + t_2) + t_3) + …` of `terms` as nested binary `+`
/// (§6.1.2); `None` for no term.
fn fold<'a>(terms: impl IntoIterator<Item = &'a Expr>) -> Option<Value> {
    terms.into_iter().fold(None, |acc, t| {
        Some(match acc {
            None => expr_json(t),
            Some(a) => json!({"op": "+", "args": [a, expr_json(t)]}),
        })
    })
}

/// A state equation's left-hand side, `D(x)` on the whole state.
fn d_of(x: &str) -> Value {
    json!({"op": "D", "args": [x], "wrt": "t"})
}

/// The right-hand side of state `x`'s equation `i` holding `terms` (§6.1.2).
fn state_rhs(info: &ModelInfo, i: usize, x: &str, terms: &[&Term]) -> Value {
    let lifted = info.flat.equation_lifted.get(i).copied().unwrap_or(false);
    let Some(f) = fold(terms.iter().map(|t| &t.expr)) else {
        return zero(info, x);
    };
    if !lifted {
        return f;
    }
    let output_idx: Vec<String> = node(&info.flat.equations[i].rhs)
        .and_then(|n| n.output_idx.clone())
        .unwrap_or_default();
    json!({
        "op": "faq",
        "args": [],
        "output_idx": output_idx,
        "ranges": Value::Object(lifted_ranges(&output_idx, &info.shapes[x])),
        "expr": f,
    })
}

/// The observeds `names` reference, with every observed their definitions
/// reference, transitively: the live observeds (§6.1.1, §6.2 item 2).
pub fn live_observeds(info: &ModelInfo, names: &BTreeSet<String>) -> BTreeSet<String> {
    info.close_over_observeds(names)
        .into_iter()
        .filter(|n| info.is_observed(n))
        .collect()
}

/// The right-hand-side document's component tree: every state and parameter, the live
/// observeds, the state equations (right-hand sides from `rhs`), the live
/// observeds' definitions and every `ic` equation (§6.1.1–§6.1.2).
fn tree(info: &ModelInfo, live: &BTreeSet<String>, rhs: &dyn Fn(usize, &str) -> Value) -> Tree {
    let flat = &info.flat;
    let mut tree = Tree::default();
    for s in &info.states {
        tree.declare(s, variable_json(&flat.state_variables[s], &info.shapes[s]));
    }
    for (p, v) in &flat.parameters {
        let shape = v.shape.clone().unwrap_or_default();
        tree.declare(p, variable_json(v, &shape));
    }
    for o in live {
        let v = &flat.observed_variables[o];
        let shape = v.shape.clone().unwrap_or_default();
        tree.declare(o, variable_json(v, &shape));
    }
    for (i, eq) in flat.equations.iter().enumerate() {
        if let Some(x) = differentiated(&eq.lhs).filter(|x| info.is_state(x)) {
            tree.push(x, json!({"lhs": d_of(x), "rhs": rhs(i, x)}));
        } else if let Some(o) = defined_observed(&eq.lhs).filter(|o| live.contains(*o)) {
            tree.push(
                o,
                json!({"lhs": expr_json(&eq.lhs), "rhs": expr_json(&eq.rhs)}),
            );
        }
    }
    for (x, e) in &flat.field_ics {
        tree.push(
            x,
            json!({"lhs": {"op": "ic", "args": [x]}, "rhs": expr_json(e)}),
        );
    }
    tree
}

/// The top-level keys of a derived document (§6.1).
fn document(info: &ModelInfo, name: &str, models: Value) -> Value {
    let sets: Map<String, Value> = info
        .index_sets
        .iter()
        .map(|(k, n)| (k.clone(), json!({"kind": "interval", "size": n})))
        .collect();
    let mut out = Map::new();
    out.insert("esm".into(), json!(ESM_VERSION));
    out.insert("metadata".into(), json!({"name": name}));
    out.insert("index_sets".into(), Value::Object(sets));
    if let Some(d) = info.flat.domain.as_ref().and_then(domain_json) {
        out.insert("domain".into(), d);
    }
    out.insert("models".into(), models);
    Value::Object(out)
}

/// A derived document's `domain` (§6.1), built as EarthSciASTDiff builds the
/// evaluation document's, so both documents carry the same one: `temporal`
/// holding the `start`, `end` and `reference_time` the source gives, then
/// `element_type` and `array_type` when given, each as EarthSciAST
/// serializes it. No `independent_variable`: the derived equations are
/// written with respect to `t`. `None` when nothing is left.
pub fn domain_json(domain: &Domain) -> Option<Value> {
    let src = serde_json::to_value(domain).ok()?;
    let given = |m: &Map<String, Value>, k: &str| m.get(k).filter(|v| !v.is_null()).cloned();
    let mut out = Map::new();
    if let Some(Value::Object(tmp)) = src.get("temporal") {
        let t: Map<String, Value> = ["start", "end", "reference_time"]
            .iter()
            .filter_map(|k| given(tmp, k).map(|v| (k.to_string(), v)))
            .collect();
        if !t.is_empty() {
            out.insert("temporal".into(), Value::Object(t));
        }
    }
    if let Value::Object(src) = &src {
        for k in ["element_type", "array_type"] {
            if let Some(v) = given(src, k) {
                out.insert(k.into(), v);
            }
        }
    }
    (!out.is_empty()).then_some(Value::Object(out))
}

/// The right-hand-side document of an operator named `name` whose terms are
/// `terms` (§6.1). Also the document of a single term (§5.2.2).
pub fn rhs_document(info: &ModelInfo, name: &str, terms: &[&Term]) -> Value {
    let mut by_eq: BTreeMap<usize, Vec<&Term>> = BTreeMap::new();
    let mut names = BTreeSet::new();
    for t in terms {
        by_eq.entry(t.equation).or_default().push(t);
        names.extend(t.names.iter().cloned());
    }
    let live = live_observeds(info, &names);
    let rhs = |i: usize, x: &str| {
        let ts = by_eq.get(&i).map(Vec::as_slice).unwrap_or(&[]);
        state_rhs(info, i, x, ts)
    };
    document(info, name, tree(info, &live, &rhs).to_json())
}

/// The Jacobian document of an operator (§6.2): EarthSciASTDiff's
/// evaluation document (`evaluation_document`, coefficient hoisting off) of
/// `view`, the flattened view of the operator's right-hand-side document,
/// for its band list `entries`, without its `ic` equations. Band `k`
/// (1-based) is the field `JacobianEval.J_k`; the document's `esm` version
/// and `metadata.name` are EarthSciASTDiff's.
pub fn jacobian_document(view: &FlatView, entries: &[JacEntry]) -> Result<Value> {
    let internal = |e: earthsci_astdiff::BandError| {
        PlanError::Internal(format!("evaluation document: {}", e.0))
    };
    let sv = view.sysview().map_err(internal)?;
    let (mut doc, _) =
        earthsci_astdiff::evaluation_document(&sv, entries, false, 0).map_err(internal)?;
    if let Some(Value::Object(models)) = doc.get_mut("models") {
        models.values_mut().for_each(drop_ic);
    }
    Ok(doc)
}

/// Removes the `ic` equations of a model and of its subsystems.
fn drop_ic(model: &mut Value) {
    if let Some(Value::Array(eqs)) = model.get_mut("equations") {
        eqs.retain(|eq| eq["lhs"]["op"] != "ic");
    }
    if let Some(Value::Object(subs)) = model.get_mut("subsystems") {
        subs.values_mut().for_each(drop_ic);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn domain(v: Value) -> Domain {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn domain_keeps_what_evaluation_reads() {
        let d = domain(json!({
            "independent_variable": "time",
            "temporal": {"start": "2020-01-01T00:00:00Z", "reference_time": "2019-12-31T00:00:00Z"},
            "element_type": "Float64",
            "array_type": "Array",
        }));
        assert_eq!(
            domain_json(&d),
            Some(json!({
                "temporal": {"start": "2020-01-01T00:00:00Z", "reference_time": "2019-12-31T00:00:00Z"},
                "element_type": "Float64",
                "array_type": "Array",
            }))
        );
        // Nothing left: no `domain`.
        assert_eq!(
            domain_json(&domain(json!({"independent_variable": "t"}))),
            None
        );
        assert_eq!(domain_json(&domain(json!({"temporal": {}}))), None);
    }

    /// The right-hand-side document's `domain` is the Jacobian document's,
    /// EarthSciASTDiff's evaluation document of the same source.
    #[test]
    fn domain_matches_evaluation_document() {
        let src = json!({
            "esm": "1.3.0",
            "metadata": {"name": "d"},
            "domain": {
                "independent_variable": "t",
                "temporal": {"start": "2020-01-01T00:00:00Z", "end": "2020-01-02T00:00:00Z"},
                "element_type": "Float64",
            },
            "models": {"M": {
                "variables": {"x": {"type": "unknown", "default": 1.0}, "k": {"type": "parameter", "default": 2.0}},
                "equations": [{"lhs": {"op": "D", "args": ["x"], "wrt": "t"},
                               "rhs": {"op": "*", "args": [{"op": "-", "args": ["k"]}, "x"]}}],
            }},
        });
        let view = FlatView::new(&src.to_string()).unwrap();
        let ours = view.flat.domain.as_ref().and_then(domain_json);
        assert_eq!(
            ours,
            Some(json!({
                "temporal": {"start": "2020-01-01T00:00:00Z", "end": "2020-01-02T00:00:00Z"},
                "element_type": "Float64",
            }))
        );
        let jac = jacobian_document(&view, &view.bands().unwrap()).unwrap();
        assert_eq!(jac.get("domain").cloned(), ours);
    }
}
