//! Reading the model (spec/plan.md §4 steps 2–6) and the facts about the
//! flattened model the passes share: states, their shapes (§6.1.3), the
//! canonical state order, observed definitions and the names an expression
//! references.

use crate::derived::JACOBIAN_MODEL;
use crate::error::{codes, refuse, PlanError, Result};
use crate::runconfig::RunConfig;
use earthsci_ast::{EsmFile, Expr, FlattenedSystem};
use indexmap::IndexMap;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

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
    /// the shape `flatten` records, §6.1.3); empty for a scalar.
    pub shapes: BTreeMap<String, Vec<String>>,
    /// Observed definitions: observed name → its right-hand side.
    pub observed_defs: BTreeMap<String, Expr>,
    /// The derived documents' `domain` ([`crate::derived::domain_json`] of
    /// the flattened model's), built once; `None` when they have none.
    pub domain: Option<Value>,
    /// The data-fed parameters (spec/plan.md §4.1), by flattened name.
    pub data_fed: BTreeMap<String, DataFed>,
    /// The used data-fed parameters (§4.1): those a term reads, directly or
    /// through observed definitions, or an `ic` equation reads. Set by
    /// [`ModelInfo::set_used`] once the terms are known (pass 1); every
    /// term belongs to some operator, so this is the union of the
    /// operators' `forcing` (§5.3) and the `ic` equations' data-fed reads.
    pub used: BTreeSet<String>,
}

/// A data-fed parameter (spec/plan.md §4.1): what the plan's `forcing`
/// records about it, and its declared `default`.
#[derive(Debug, Clone, PartialEq)]
pub struct DataFed {
    /// Its declared shape (index-set names; empty for a scalar).
    pub shape: Vec<String>,
    /// The `data_sources` key its `loader_fields` entry names.
    pub source: String,
    /// The source's `kind` as the esm format writes it (`grid`, `points`,
    /// `static`), for the derived documents' stub (§6.3).
    pub source_kind: String,
    /// The `file_variable` its `loader_fields` entry names.
    pub file_variable: String,
    /// `discrete` when its source's `temporal` has a `start` and a
    /// `frequency` or `file_period`, else `const`.
    pub cadence: &'static str,
    /// The effective records per sample EarthSciAST reports (1 or 2).
    pub records_per_sample: u8,
}

impl DataFed {
    /// Its entry in the plan's `forcing` (§3).
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "shape": self.shape,
            "source": self.source,
            "file_variable": self.file_variable,
            "cadence": self.cadence,
            "records_per_sample": self.records_per_sample,
        })
    }
}

impl ModelInfo {
    /// Sets [`ModelInfo::used`] from the terms' names (pass 1) and the `ic`
    /// equations (§4.1).
    pub fn set_used<'a>(&mut self, term_names: impl IntoIterator<Item = &'a BTreeSet<String>>) {
        let mut names = BTreeSet::new();
        for n in term_names {
            names.extend(n.iter().cloned());
        }
        for (_, e) in &self.flat.field_ics {
            names.extend(self.names_in(e));
        }
        self.used = self
            .close_over_observeds(&names)
            .into_iter()
            .filter(|n| self.data_fed.contains_key(n))
            .collect();
    }

    /// The data-fed parameters among `names`.
    pub fn forcing_among<'a>(&self, names: impl IntoIterator<Item = &'a String>) -> Vec<String> {
        names
            .into_iter()
            .filter(|n| self.data_fed.contains_key(*n))
            .cloned()
            .collect()
    }

    /// The extents of parameter `name`'s declared shape (empty for a
    /// scalar).
    pub fn param_extents(&self, name: &str) -> Vec<usize> {
        self.flat.parameters[name]
            .shape
            .iter()
            .flatten()
            .map(|x| self.index_sets.get(x).copied().unwrap_or(0) as usize)
            .collect()
    }

    /// Parameter `name`'s `default` at every element, column-major over its
    /// declared shape (a scalar default broadcast); `None` without a
    /// `default`, or with array data that does not have the shape.
    pub fn default_column_major(&self, name: &str) -> Option<Vec<f64>> {
        let ext = self.param_extents(name);
        let len: usize = ext.iter().product();
        match self.flat.parameters[name].default.as_ref()? {
            earthsci_ast::InlineValue::Scalar(x) => Some(vec![*x; len]),
            d => {
                let (dims, row_major) = d.to_dense().ok()?;
                if dims != ext {
                    return None;
                }
                // Row-major (last index fastest) to column-major (first
                // index fastest).
                let mut out = vec![0.0; len];
                for (k, x) in row_major.into_iter().enumerate() {
                    let mut rem = k;
                    let mut off = 0;
                    let mut stride = 1;
                    let mut idx = vec![0; ext.len()];
                    for d in (0..ext.len()).rev() {
                        idx[d] = rem % ext[d];
                        rem /= ext[d];
                    }
                    for d in 0..ext.len() {
                        off += idx[d] * stride;
                        stride *= ext[d];
                    }
                    out[off] = x;
                }
                Some(out)
            }
        }
    }

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

    /// The states, and every observed whose definition reads a state,
    /// directly or through other observeds. Each definition is read once,
    /// then the set grows until no observed joins it.
    pub fn state_dependent(&self) -> BTreeSet<String> {
        let refs: Vec<(&String, BTreeSet<String>)> = self
            .observed_defs
            .iter()
            .map(|(o, d)| (o, self.names_in(d)))
            .collect();
        let mut out: BTreeSet<String> = self.states.iter().cloned().collect();
        loop {
            let grew = refs
                .iter()
                .filter(|(o, r)| !out.contains(*o) && r.iter().any(|n| out.contains(n)))
                .map(|(o, _)| (*o).clone())
                .collect::<Vec<_>>();
            if grew.is_empty() {
                return out;
            }
            out.extend(grew);
        }
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

/// The names the `ic` equations of the flattened model `flat` read,
/// directly or through observed definitions, transitively (spec/plan.md
/// §4.1; spec/operator-interface.md §8.4.5).
pub fn ic_reads(flat: &FlattenedSystem) -> BTreeSet<String> {
    let defs: BTreeMap<&str, &Expr> = flat
        .equations
        .iter()
        .filter_map(|eq| defined_observed(&eq.lhs).map(|o| (o, &eq.rhs)))
        .collect();
    let names = |e: &Expr| {
        let mut out = Vec::new();
        collect_strings(&serde_json::to_value(e).unwrap_or(Value::Null), &mut |s| {
            out.push(s.to_string())
        });
        out
    };
    let mut out = BTreeSet::new();
    let mut todo: Vec<String> = flat.field_ics.iter().flat_map(|(_, e)| names(e)).collect();
    while let Some(n) = todo.pop() {
        if out.insert(n.clone()) {
            if let Some(d) = defs.get(n.as_str()) {
                todo.extend(names(d));
            }
        }
    }
    out
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
///
/// The two have the same length: [`read`] stops with an internal error
/// when a lifted equation's do not.
pub fn lifted_ranges(output_idx: &[String], shape: &[String]) -> serde_json::Map<String, Value> {
    assert_eq!(
        output_idx.len(),
        shape.len(),
        "lifted_ranges: output_idx {output_idx:?} and shape {shape:?} differ in length"
    );
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
pub(crate) fn defined_observed(lhs: &Expr) -> Option<&str> {
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
            "lowering table_lookup inside each contribution (spec/plan.md §4 step 5) has no public Rust entry point in EarthSciAST 0.10".into(),
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
    // Data-fed parameters (§4.1): those a `loader_fields` entry names. Each
    // one's rule must be a single `data` rule with `from`; any other rule,
    // on any parameter, is refused.
    let fed: BTreeMap<&str, &earthsci_ast::LoaderField> = flat
        .loader_fields
        .iter()
        .map(|f| (f.name.as_str(), f))
        .collect();
    let allowed = |name: &str, v: &earthsci_ast::ModelVariable| match &v.update {
        None => true,
        Some(u) => {
            fed.contains_key(name)
                && matches!(u.rules(), [r @ earthsci_ast::ParameterUpdate::Data { .. }]
                    if r.value().is_some_and(|x| x.from.is_some()))
        }
    };
    if let Some((name, v)) = flat.parameters.iter().find(|(n, v)| !allowed(n, v)) {
        let kinds: Vec<&str> = v
            .update
            .iter()
            .flat_map(|u| u.rules().iter().map(|r| r.kind()))
            .collect();
        return scope(
            codes::UNSUPPORTED_PARAMETER_UPDATE,
            format!(
                "parameter {name} has the update rules {kinds:?}; M2 plans only a data-fed parameter's single data rule (D75, D87)"
            ),
        );
    }
    if let Some(f) = flat
        .loader_fields
        .iter()
        .find(|f| !flat.parameters.contains_key(&f.name))
    {
        return scope(
            codes::UNSUPPORTED_FORCING,
            format!(
                "the loader_fields entry {} (source {}, file variable {}) names no parameter of the flattened model",
                f.name, f.source, f.file_variable
            ),
        );
    }
    let mut data_fed = BTreeMap::new();
    for f in &flat.loader_fields {
        let src = file.data_sources.as_ref().and_then(|d| d.get(&f.source));
        let temporal = src.and_then(|s| s.temporal.as_ref());
        let discrete = temporal.is_some_and(|t| {
            t.start.is_some() && (t.frequency.is_some() || t.file_period.is_some())
        });
        if f.records_per_sample == 2 && !discrete {
            return scope(
                codes::FORCING_RECORDS_INVALID,
                format!(
                    "data-fed parameter {} asks for 2 records per sample, but its source {}'s temporal block is not time-varying (it needs a start and a frequency or file_period), so there is no successor record",
                    f.name, f.source
                ),
            );
        }
        let source_kind = src
            .and_then(|s| serde_json::to_value(&s.kind).ok())
            .and_then(|k| k.as_str().map(str::to_string))
            .ok_or_else(|| {
                PlanError::Internal(format!(
                    "data-fed parameter {} names the source {}, which the loaded model does not hold",
                    f.name, f.source
                ))
            })?;
        data_fed.insert(
            f.name.clone(),
            DataFed {
                shape: flat.parameters[&f.name].shape.clone().unwrap_or_default(),
                source: f.source.clone(),
                source_kind,
                file_variable: f.file_variable.clone(),
                cadence: if discrete { "discrete" } else { "const" },
                records_per_sample: f.records_per_sample,
            },
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
    if top_level.into_iter().any(|k| k == JACOBIAN_MODEL) {
        return scope(
            codes::RESERVED_COMPONENT_NAME,
            format!("the model has a top-level component named {JACOBIAN_MODEL}, which the Jacobian document reserves"),
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
        domain: None,
        data_fed,
        used: BTreeSet::new(),
    };
    if let Some(d) = &info.flat.domain {
        info.domain = crate::derived::domain_json(d)?;
    }
    // A lifted state's shape is the one `flatten` records on it (§6.1.3).
    for s in info.states.clone() {
        let declared = info.flat.state_variables[&s]
            .shape
            .clone()
            .unwrap_or_default();
        info.shapes.insert(s, declared);
    }
    check_shapes(&info)?;
    Ok(info)
}

/// The shapes the derived documents read hold together (§6.1.2, §6.1.3):
/// every index set of a state's shape is an index set of the flattened
/// model; a lifted state's recorded shape has the lift's extents, index set
/// by index set; and a lifted equation's `output_idx` names one index per
/// dimension of its state's shape. EarthSciAST 0.8's `flatten` guarantees
/// all three, so a failure is an internal error, not a refusal.
fn check_shapes(info: &ModelInfo) -> Result<()> {
    let internal = |m: String| Err(PlanError::Internal(m));
    for (s, shape) in &info.shapes {
        if let Some(set) = shape.iter().find(|set| !info.index_sets.contains_key(*set)) {
            return internal(format!(
                "state {s} has the index set {set} in its shape, which the flattened model does not declare"
            ));
        }
    }
    for (s, extents) in &info.flat.lifted_shapes {
        let Some(shape) = info.shapes.get(s) else {
            continue;
        };
        let sizes: Vec<i64> = shape.iter().map(|set| info.index_sets[set]).collect();
        if &sizes != extents {
            return internal(format!(
                "flatten recorded the shape {shape:?} (extents {sizes:?}) on the lifted state {s}, whose lift has the extents {extents:?}"
            ));
        }
    }
    for (i, eq) in info.flat.equations.iter().enumerate() {
        if !info.flat.equation_lifted.get(i).copied().unwrap_or(false) {
            continue;
        }
        let Some(x) = differentiated(&eq.lhs).filter(|x| info.shapes.contains_key(*x)) else {
            continue;
        };
        let n = node(&eq.rhs)
            .and_then(|n| n.output_idx.as_ref())
            .map_or(0, Vec::len);
        if n != info.shapes[x].len() {
            return internal(format!(
                "the lifted equation {i} of state {x} has {n} output indices, but the state's shape {:?} has {} dimensions",
                info.shapes[x],
                info.shapes[x].len()
            ));
        }
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "differ in length")]
    fn lifted_faq_ranges_lengths() {
        lifted_ranges(&["i".into(), "j".into()], &["cell".into()]);
    }

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
