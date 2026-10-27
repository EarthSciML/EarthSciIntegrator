//! Reading the model (spec/plan.md §4 steps 2–6) and the facts about the
//! flattened model the passes share: states, their shapes (§6.1.3), the
//! canonical state order, observed definitions and the names an expression
//! references.

use crate::error::{codes, refuse, PlanError, Result};
use crate::runconfig::RunConfig;
use earthsci_ast::{EsmFile, Expr, FlattenedSystem};
use indexmap::IndexMap;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

/// The top-level component name the Jacobian document reserves
/// (spec/plan.md §6.1.4).
pub const RESERVED_COMPONENT: &str = "_jacobian";

/// The flattened model and what the passes read from it.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// The model's path, resolved against the run config's directory.
    pub path: PathBuf,
    /// The model file's bytes, for `inputs.model.sha256`.
    pub bytes: Vec<u8>,
    /// The loaded document.
    pub file: EsmFile,
    /// The flattened model (§4 step 5).
    pub flat: FlattenedSystem,
    /// Index-set sizes, in the flattened model's `index_sets` order.
    pub index_sets: IndexMap<String, i64>,
    /// The state variables (keys of `state_variables` that an equation
    /// differentiates), in canonical order: sorted by code point (§4).
    pub states: Vec<String>,
    /// Each state's index sets (its declared shape, or for a lifted state
    /// the shape of §6.1.3); empty for a scalar.
    pub shapes: BTreeMap<String, Vec<String>>,
    /// Observed definitions: observed name → its right-hand side.
    pub observed_defs: BTreeMap<String, Expr>,
}

impl ModelInfo {
    /// Whether `name` is a state of the flattened model.
    pub fn is_state(&self, name: &str) -> bool {
        self.shapes.contains_key(name)
    }

    /// Whether `name` is a parameter of the flattened model.
    pub fn is_parameter(&self, name: &str) -> bool {
        self.flat.parameters.contains_key(name)
    }

    /// Whether `name` is an observed of the flattened model.
    pub fn is_observed(&self, name: &str) -> bool {
        self.flat.observed_variables.contains_key(name)
    }

    /// Whether `name` is a state, parameter or observed: a *name* in the
    /// sense of spec/plan.md §5.3 (`t` and index names are not).
    pub fn is_name(&self, name: &str) -> bool {
        self.is_state(name)
            || self.is_parameter(name)
            || self.is_observed(name)
            || self.flat.state_variables.contains_key(name)
    }

    /// The number of cells of a state (1 for a scalar).
    pub fn cells(&self, state: &str) -> i64 {
        self.shapes
            .get(state)
            .map(|s| s.iter().map(|x| self.index_sets[x]).product())
            .unwrap_or(1)
    }

    /// The names (states, parameters, observeds) `e` references.
    pub fn names_in(&self, e: &Expr) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        let v = serde_json::to_value(e).unwrap_or(Value::Null);
        collect_strings(&v, &mut |s| {
            if self.is_name(s) {
                out.insert(s.to_string());
            }
        });
        out
    }

    /// `names` together with every name the definitions of the observeds
    /// among them reference, transitively (spec/plan.md §5.3).
    pub fn close_over_observeds(&self, names: &BTreeSet<String>) -> BTreeSet<String> {
        let mut out = names.clone();
        let mut todo: Vec<String> = names.iter().cloned().collect();
        while let Some(n) = todo.pop() {
            if let Some(def) = self.observed_defs.get(&n) {
                for m in self.names_in(def) {
                    if out.insert(m.clone()) {
                        todo.push(m);
                    }
                }
            }
        }
        out
    }
}

/// Every string value in a serialized expression (object keys excluded).
pub(crate) fn collect_strings(v: &Value, f: &mut impl FnMut(&str)) {
    match v {
        Value::String(s) => f(s),
        Value::Array(a) => a.iter().for_each(|x| collect_strings(x, f)),
        Value::Object(m) => m.values().for_each(|x| collect_strings(x, f)),
        _ => {}
    }
}

/// The `ranges` of a lifted equation's `faq` in a derived document
/// (spec/plan.md §6.1.2): the d-th name of `output_idx` maps to
/// `{"from": X_d}`, `X_d` the index set of dimension d of the state's shape.
/// Built rather than copied, since the bindings write the flattened ranges
/// differently.
pub fn lifted_ranges(output_idx: &[String], shape: &[String]) -> serde_json::Map<String, Value> {
    output_idx
        .iter()
        .zip(shape)
        .map(|(i, set)| (i.clone(), serde_json::json!({"from": set})))
        .collect()
}

/// The operator node of an expression, if it is one.
pub fn node(e: &Expr) -> Option<&earthsci_ast::ExpressionNode> {
    match e {
        Expr::Operator(n) => Some(n),
        _ => None,
    }
}

/// The state a state equation's left-hand side differentiates: `D(x, t)`,
/// or the pointwise lift's `faq` over `D(index(x, …), t)` (spec/plan.md
/// §5.1.1). `None` for any other left-hand side.
pub fn differentiated(lhs: &Expr) -> Option<&str> {
    let n = node(lhs)?;
    match n.op.as_str() {
        "D" => match n.args.first()? {
            Expr::Variable(x) => Some(x),
            inner => {
                let m = node(inner)?;
                if m.op != "index" {
                    return None;
                }
                match m.args.first()? {
                    Expr::Variable(x) => Some(x),
                    _ => None,
                }
            }
        },
        "faq" => differentiated(n.expr.as_deref()?),
        _ => None,
    }
}

/// The observed an observed definition defines: a bare name, or a `faq`
/// over `index(name, …)`.
fn defined_observed(lhs: &Expr) -> Option<&str> {
    match lhs {
        Expr::Variable(x) => Some(x),
        Expr::Operator(n) if n.op == "faq" => defined_observed(n.expr.as_deref()?),
        Expr::Operator(n) if n.op == "index" => match n.args.first()? {
            Expr::Variable(x) => Some(x),
            _ => None,
        },
        _ => None,
    }
}

/// `path` with `.` and `..` segments removed lexically.
fn lexical_normal(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Steps 2–6 of spec/plan.md §4: resolve, load, check `solver.external`,
/// flatten, and refuse what M2 does not plan.
pub fn read(rc: &RunConfig) -> Result<ModelInfo> {
    // Step 2.
    let model = rc.model();
    if model.starts_with("toy:") {
        return Err(refuse(
            codes::TOY_MODEL_NOT_PLANNABLE,
            "/model",
            format!(
                "{model} names a hand-written toy problem, which only the conformance harness runs"
            ),
        ));
    }
    let path = rc.dir().join(model);

    // Step 3.
    let load_failed = |m: String| refuse(codes::MODEL_LOAD_FAILED, "/model", m);
    let bytes = std::fs::read(&path)
        .map_err(|e| load_failed(format!("cannot read {}: {e}", path.display())))?;
    let file = earthsci_ast::load_path(&path).map_err(|e| {
        // A `solver.time_span` EarthSciAST rejects at load takes its rule-14
        // code (spec/plan.md §7.2).
        serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|doc| crate::time::model_span_refusal(&doc))
            .unwrap_or_else(|| load_failed(e.to_string()))
    })?;
    let validation = earthsci_ast::validate(&file);
    if !validation.is_valid {
        return Err(load_failed(format!(
            "the model does not validate: {:?} {:?}",
            validation.schema_errors, validation.structural_errors
        )));
    }

    // Step 4.
    check_external(rc, &file, &path)?;

    // Step 5. EarthSciAST's Rust `load_path` has already expanded expression
    // templates and lowered `enum` nodes; `table_lookup` lowering is not
    // public in Rust, so a model that still holds one cannot be planned here.
    let flat = earthsci_ast::flatten(&file)
        .map_err(|e| refuse(codes::MODEL_FLATTEN_FAILED, "/model", e.to_string()))?;
    if serde_json::to_string(&flat.equations)
        .unwrap_or_default()
        .contains("\"table_lookup\"")
    {
        return Err(PlanError::NotYetImplemented(
            "lowering table_lookup inside each contribution (spec/plan.md §4 step 5) has no public Rust entry point in EarthSciAST 0.6".into(),
        ));
    }

    // Step 6.
    let scope = |code, m: String| Err(refuse(code, "/model", m));
    if let Some(name) = flat.algebraic_variables.keys().next() {
        return scope(
            codes::UNSUPPORTED_ALGEBRAIC_EQUATION,
            format!("the flattened model has the algebraic unknown {name}; algebraic operators arrive at M4b"),
        );
    }
    if !flat.continuous_events.is_empty() || !flat.discrete_events.is_empty() {
        return scope(
            codes::UNSUPPORTED_EVENT,
            "the flattened model has events; condition and map operators arrive at M4b".into(),
        );
    }
    // At the parameter-update place, a `crossing` rule on any parameter
    // takes the more specific code, whatever the parameters' order (§4 step 6).
    let crossing = |v: &earthsci_ast::ModelVariable| {
        v.update.as_ref().is_some_and(|u| {
            u.rules()
                .iter()
                .any(|r| matches!(r, earthsci_ast::ParameterUpdate::Crossing { .. }))
        })
    };
    if let Some((name, _)) = flat.parameters.iter().find(|(_, v)| crossing(v)) {
        return scope(
            codes::UNSUPPORTED_CROSSING_UPDATE,
            format!("parameter {name} has a crossing update rule, which is not in v1 (D22)"),
        );
    }
    if let Some((name, _)) = flat.parameters.iter().find(|(_, v)| v.update.is_some()) {
        return scope(
            codes::UNSUPPORTED_PARAMETER_UPDATE,
            format!("parameter {name} has an update rule, which M2 does not plan (D75)"),
        );
    }
    if let Some(f) = flat.loader_fields.first() {
        return scope(
            codes::UNSUPPORTED_FORCING,
            format!("the flattened model reads the data-loader field {f:?}; forcing arrives at M3"),
        );
    }
    let mut index_sets = IndexMap::new();
    for (name, set) in &flat.index_sets {
        match (set.kind.as_str(), set.size) {
            ("interval", Some(n)) => {
                index_sets.insert(name.clone(), n);
            }
            _ => {
                return scope(
                    codes::UNSUPPORTED_INDEX_SET,
                    format!("index set {name} is not an interval with an integer size"),
                )
            }
        }
    }
    // §4 step 7 (§6.1.4, D81).
    let top_level = file
        .models
        .iter()
        .flat_map(|m| m.keys())
        .chain(file.reaction_systems.iter().flat_map(|r| r.keys()));
    if top_level.into_iter().any(|k| k == RESERVED_COMPONENT) {
        return scope(
            codes::RESERVED_COMPONENT_NAME,
            format!("the model has a top-level component named {RESERVED_COMPONENT}, which the Jacobian document reserves"),
        );
    }

    let mut states = BTreeSet::new();
    let mut observed_defs = BTreeMap::new();
    for eq in &flat.equations {
        if let Some(x) = differentiated(&eq.lhs) {
            if flat.state_variables.contains_key(x) {
                states.insert(x.to_string());
            }
        } else if let Some(o) = defined_observed(&eq.lhs) {
            if flat.observed_variables.contains_key(o) {
                observed_defs.insert(o.to_string(), eq.rhs.clone());
            }
        }
    }
    let mut info = ModelInfo {
        path,
        bytes,
        file,
        flat,
        index_sets,
        states: states.into_iter().collect(),
        shapes: BTreeMap::new(),
        observed_defs,
    };
    // Declared shapes now; lifted states get theirs in `resolve_lifted_shapes`.
    for s in info.states.clone() {
        let declared = info.flat.state_variables[&s]
            .shape
            .clone()
            .unwrap_or_default();
        info.shapes.insert(s, declared);
    }
    Ok(info)
}

/// Step 4 of spec/plan.md §4: the model's `solver.external`, resolved with
/// the model's directory as base, must equal the run config's own location
/// written the same way.
fn check_external(rc: &RunConfig, file: &EsmFile, model_path: &Path) -> Result<()> {
    if file
        .solver
        .as_ref()
        .and_then(|s| s.external.as_ref())
        .is_none()
    {
        return Ok(());
    }
    let model_dir = model_path.parent().unwrap_or(Path::new("."));
    let theirs = earthsci_ast::solver_external_location(file, Some(model_dir))
        .map_err(|e| refuse(codes::MODEL_LOAD_FAILED, "/model", e.to_string()))?;
    // The run config's location, spelled by the same accessor: an absolute,
    // dot-segment-free path given to it as an absolute `external`.
    let abs = if rc.path.is_absolute() {
        rc.path.clone()
    } else {
        std::env::current_dir()
            .map_err(|e| PlanError::Io(e.to_string()))?
            .join(&rc.path)
    };
    let abs = lexical_normal(&abs);
    let mut probe = file.clone();
    if let Some(s) = probe.solver.as_mut() {
        s.external = Some(abs.to_string_lossy().into_owned());
    }
    let ours = earthsci_ast::solver_external_location(&probe, None)
        .map_err(|e| PlanError::Io(e.to_string()))?;
    if theirs != ours {
        return Err(refuse(
            codes::SOLVER_EXTERNAL_MISMATCH,
            "/model",
            format!(
                "the model's solver.external resolves to {}, not to this run config ({})",
                theirs.unwrap_or_default(),
                ours.unwrap_or_default()
            ),
        ));
    }
    Ok(())
}

/// spec/plan.md §6.1.3: a lifted state's shape is the declared shape of the
/// first variable of the flattened model (`state_variables`, then
/// `parameters`, then `observed_variables`, each in document order) whose
/// declared shape has exactly the lift's extents, index set by index set;
/// none is `lift_shape_unresolved`.
pub fn resolve_lifted_shapes(info: &mut ModelInfo) -> Result<()> {
    let lifted: Vec<(String, Vec<i64>)> = info
        .flat
        .lifted_shapes
        .iter()
        .filter(|(k, _)| info.shapes.contains_key(*k))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    for (state, extents) in lifted {
        // Non-lifted variables only: a shape `flatten` records on a lifted
        // state itself is ignored (Rust's records none; Julia's does).
        let lifted = &info.flat.lifted_shapes;
        let candidates = info
            .flat
            .state_variables
            .iter()
            .chain(info.flat.parameters.iter())
            .chain(info.flat.observed_variables.iter())
            .filter(|(k, _)| !lifted.contains_key(*k))
            .map(|(_, v)| v);
        let found = candidates.filter_map(|v| v.shape.as_ref()).find(|shape| {
            shape.len() == extents.len()
                && shape
                    .iter()
                    .zip(&extents)
                    .all(|(set, n)| info.index_sets.get(set) == Some(n))
        });
        match found {
            Some(shape) => {
                info.shapes.insert(state, shape.clone());
            }
            None => {
                return Err(refuse(
                    codes::LIFT_SHAPE_UNRESOLVED,
                    "/model",
                    format!(
                        "no declared shape has the extents {extents:?} of the lifted state {state}"
                    ),
                ))
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifted_faq_ranges() {
        let r = lifted_ranges(
            &["i".into(), "j".into(), "k".into()],
            &["lon".into(), "lat".into(), "lev".into()],
        );
        assert_eq!(
            Value::Object(r),
            serde_json::json!({"i": {"from": "lon"}, "j": {"from": "lat"}, "k": {"from": "lev"}})
        );
    }
}
