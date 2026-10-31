//! Pass 6, the derived documents (spec/plan.md §6, D68, D69).
//!
//! Both documents are JSON objects written by EarthSciASTDiff's
//! `stable_json` (§2.2). The right-hand-side document (§6.1) declares the
//! flattened model's states, parameters and live observeds in models named
//! after their components and holds one operator's terms; the Jacobian
//! document (§6.2) zeroes every state equation and adds the band fields
//! `_jacobian.J_k` that EarthSciASTDiff's evaluation document builds.

use crate::astdiff::FlatView;
use crate::error::{PlanError, Result};
use crate::model::{defined_observed, differentiated, lifted_ranges, node, ModelInfo};
use crate::terms::{component_of, Term};
use earthsci_ast::{Expr, ModelVariable, VariableType};
use earthsci_astdiff::JacEntry;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};

/// The esm version a derived document declares (§6.1).
pub const ESM_VERSION: &str = "1.3.0";

/// The model the Jacobian document adds for its band fields (§6.2, §6.1.4).
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

/// The parts both documents share: every state and parameter, the live
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
fn document(info: &ModelInfo, name: &str, models: Value, extra_sets: &Map<String, Value>) -> Value {
    let mut sets: Map<String, Value> = info
        .index_sets
        .iter()
        .map(|(k, n)| (k.clone(), json!({"kind": "interval", "size": n})))
        .collect();
    for (k, v) in extra_sets {
        sets.entry(k.clone()).or_insert_with(|| v.clone());
    }
    let mut out = Map::new();
    out.insert("esm".into(), json!(ESM_VERSION));
    out.insert("metadata".into(), json!({"name": name}));
    out.insert("index_sets".into(), Value::Object(sets));
    if let Some(d) = &info.flat.domain {
        out.insert(
            "domain".into(),
            serde_json::to_value(d).unwrap_or(Value::Null),
        );
    }
    out.insert("models".into(), models);
    Value::Object(out)
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
    document(info, name, tree(info, &live, &rhs).to_json(), &Map::new())
}

/// The Jacobian document of an operator named `name` (§6.2): its band list
/// is `entries`, from `view`, the flattened view of its right-hand-side
/// document. Band `k` (1-based) is the field `_jacobian.J_k`.
pub fn jacobian_document(
    info: &ModelInfo,
    name: &str,
    view: &FlatView,
    entries: &[JacEntry],
) -> Result<Value> {
    // Observeds referenced, directly or transitively, by a band coefficient.
    let mut names = BTreeSet::new();
    for en in entries {
        names.extend(
            earthsci_ast::free_variables(&en.band.coef)
                .into_iter()
                .filter(|n| info.is_name(n)),
        );
    }
    let live = live_observeds(info, &names);
    let mut tree = tree(info, &live, &|_, x| zero(info, x));

    // The band fields as EarthSciASTDiff's evaluation document builds them,
    // coefficient hoisting off, each `J_k` moved to `_jacobian`.
    let sv = view.sysview().map_err(|e| PlanError::Internal(e.0))?;
    let (eval, fields) = earthsci_astdiff::evaluation_document(&sv, entries, false, 0)
        .map_err(|e| PlanError::Internal(format!("evaluation document: {}", e.0)))?;
    let model = &eval["models"]["JacobianEval"];
    let fields: BTreeSet<&str> = fields.iter().map(String::as_str).collect();
    let jac = tree.0.entry(JACOBIAN_MODEL.to_string()).or_default();
    for (k, v) in model["variables"].as_object().into_iter().flatten() {
        if fields.contains(k.as_str()) {
            jac.variables.insert(k.clone(), v.clone());
        }
    }
    for eq in model["equations"].as_array().into_iter().flatten() {
        let lhs = &eq["lhs"];
        let target = lhs["args"][0].as_str().unwrap_or_default();
        if lhs["op"] == "D" && fields.contains(target) {
            let mut eq = eq.clone();
            eq["lhs"]["args"][0] = json!(format!("{JACOBIAN_MODEL}.{target}"));
            jac.equations.push(eq);
        }
    }
    // Index sets for contracted dimensions, under EarthSciASTDiff's names.
    let extra: Map<String, Value> = eval["index_sets"]
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(k, _)| !info.index_sets.contains_key(*k))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    Ok(document(info, name, tree.to_json(), &extra))
}
