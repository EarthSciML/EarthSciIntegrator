//! Operator references (esm-spec §9.6.4, spec/plan.md §4 step 5, §6.1.5):
//! the one place the planner handles `apply_operator` nodes (the Julia
//! planner's operator_refs.jl). The planner never expands a reference. The
//! flattened model keeps them, each resolving against the merged registry
//! `flatten` carries (`template_registry`, esm-spec §9.6.4 rule 7), whose
//! bodies `flatten` has closed: a body's free names are trailing `params`,
//! bound at every reference to the site's own (flattened) names. So
//!
//! - the names a reference reads are the names in its `bindings`
//!   ([`for_each_read_string`]; never its `name`, which names a definition);
//! - a reference is one summand when a contribution is split into terms
//!   (§5.1.1): it is neither `+` nor `-`;
//! - a derived document carries, in each top-level component whose subtree
//!   holds a reference, an `operators` block with the definitions its
//!   references reach, directly or through bodies (§6.1.5,
//!   [`OperatorRefs::attach_operators`]).
//!
//! What this layer reads of a definition, it reads once per definition: the
//! definitions its body references ([`OperatorRefs::callees`]) are memoized
//! by name.

use indexmap::IndexMap;
use serde_json::{Map, Value};
use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

/// The `op` of an operator reference (esm-spec §9.6.2).
pub const APPLY_OPERATOR: &str = "apply_operator";

/// The flattened model's merged operator registry
/// (`FlattenedSystem::template_registry`: definition name → its JSON
/// object) and the per-definition facts read from it, memoized by name.
/// Clones share the memo.
#[derive(Debug, Clone, Default)]
pub struct OperatorRefs {
    registry: IndexMap<String, Value>,
    callees: Arc<Mutex<HashMap<String, Arc<Vec<String>>>>>,
}

/// Whether the JSON object `m` is an operator reference.
fn is_reference(m: &Map<String, Value>) -> bool {
    m.get("op").and_then(Value::as_str) == Some(APPLY_OPERATOR)
}

/// The names of the definitions referenced in the JSON form `x` (an
/// expression, a component, or a definition's body), in first-seen order,
/// appended to `out` (each once).
pub fn referenced_definitions(x: &Value, out: &mut Vec<String>) {
    match x {
        Value::Object(m) => {
            if is_reference(m) {
                if let Some(n) = m.get("name").and_then(Value::as_str) {
                    if !out.iter().any(|o| o == n) {
                        out.push(n.to_string());
                    }
                }
            }
            m.values().for_each(|v| referenced_definitions(v, out));
        }
        Value::Array(a) => a.iter().for_each(|v| referenced_definitions(v, out)),
        _ => {}
    }
}

/// Every string of the JSON form `x` that may read a name: every string but
/// a reference's `name`, which names a definition, not a variable. A
/// reference's reads are its bindings (the registry's bodies are closed),
/// which this visits as it visits any other child.
pub fn for_each_read_string(x: &Value, f: &mut impl FnMut(&str)) {
    match x {
        Value::String(s) => f(s),
        Value::Array(a) => a.iter().for_each(|v| for_each_read_string(v, f)),
        Value::Object(m) => {
            let reference = is_reference(m);
            for (k, v) in m {
                if !(reference && k == "name") {
                    for_each_read_string(v, f);
                }
            }
        }
        _ => {}
    }
}

impl OperatorRefs {
    /// The layer over `registry` (the flattened model's merged registry).
    pub fn new(registry: &IndexMap<String, Value>) -> OperatorRefs {
        OperatorRefs {
            registry: registry.clone(),
            callees: Arc::default(),
        }
    }

    /// The merged registry, in `flatten`'s order.
    pub fn registry(&self) -> &IndexMap<String, Value> {
        &self.registry
    }

    /// The definitions the body of definition `name` references directly,
    /// in first-seen order (read once per definition). A name the registry
    /// does not hold is the planner's error: `flatten` carries every
    /// definition a reference reaches.
    pub fn callees(&self, name: &str) -> Result<Arc<Vec<String>>, String> {
        let mut memo = self.callees.lock().expect("the callees memo");
        if let Some(c) = memo.get(name) {
            return Ok(c.clone());
        }
        let entry = self.registry.get(name).ok_or_else(|| {
            format!(
                "an operator reference names {name}, which the flattened model's registry does not hold (esm-spec §9.6.4 rule 7)"
            )
        })?;
        let mut out = Vec::new();
        if let Some(body) = entry.get("body") {
            referenced_definitions(body, &mut out);
        }
        let out = Arc::new(out);
        memo.insert(name.to_string(), out.clone());
        Ok(out)
    }

    /// The names whose callees have been read, sorted (for tests: each
    /// definition's body is read once, and only when reached).
    pub fn read_definitions(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .callees
            .lock()
            .expect("the callees memo")
            .keys()
            .cloned()
            .collect();
        v.sort();
        v
    }

    /// Every definition reachable from the definitions `roots` through the
    /// bodies' references, sorted by code point.
    pub fn reachable_definitions(&self, roots: &[String]) -> Result<BTreeSet<String>, String> {
        let mut seen = BTreeSet::new();
        let mut stack: Vec<String> = roots.to_vec();
        while let Some(n) = stack.pop() {
            if seen.contains(&n) {
                continue;
            }
            let c = self.callees(&n)?;
            seen.insert(n);
            stack.extend(c.iter().cloned());
        }
        Ok(seen)
    }

    /// The entry a derived document writes for definition `name`: the
    /// registry's entry as `flatten` holds it, materialized as esm-spec
    /// §9.6.4 rule 5 writes a referenced `match` rule (without `match`,
    /// `where` and `priority`, with `"reference_only": true`) should the
    /// registry hold one.
    pub fn materialized_entry(&self, name: &str) -> Value {
        let e = &self.registry[name];
        match e {
            Value::Object(m) if m.contains_key("match") => {
                let mut out: Map<String, Value> = m
                    .iter()
                    .filter(|(k, _)| !matches!(k.as_str(), "match" | "where" | "priority"))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                out.insert("reference_only".into(), Value::Bool(true));
                Value::Object(out)
            }
            _ => e.clone(),
        }
    }

    /// §6.1.5: gives each top-level component of the derived document's
    /// component tree `models` (JSON, as written) whose subtree holds an
    /// operator reference an `operators` block with every definition its
    /// references reach. A component with no reference gets none.
    pub fn attach_operators(&self, models: &mut Value) -> Result<(), String> {
        if self.registry.is_empty() {
            return Ok(());
        }
        let Some(models) = models.as_object_mut() else {
            return Ok(());
        };
        for comp in models.values_mut() {
            let Some(c) = comp.as_object_mut() else {
                continue;
            };
            c.remove("operators");
            let mut names = Vec::new();
            referenced_definitions(comp, &mut names);
            if names.is_empty() {
                continue;
            }
            let block: Map<String, Value> = self
                .reachable_definitions(&names)?
                .into_iter()
                .map(|n| {
                    let e = self.materialized_entry(&n);
                    (n, e)
                })
                .collect();
            comp["operators"] = Value::Object(block);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn reference(n: &str, b: Value) -> Value {
        json!({"op": "apply_operator", "args": [], "name": n, "bindings": b})
    }

    /// Definitions reached through bodies, a `match` rule materialized, each
    /// body read once (the Julia planner's "the operator layer" test).
    #[test]
    fn the_operator_layer() {
        let reg: IndexMap<String, Value> = [
            (
                "a",
                json!({"params": ["x"], "body": {"op": "+", "args": ["x", reference("b", json!({}))]}}),
            ),
            (
                "b",
                json!({"params": [], "body": {"op": "*", "args": [2, reference("c", json!({}))]}}),
            ),
            (
                "c",
                json!({"params": [], "body": 1.0, "match": {"op": "foo", "args": []}, "priority": 2}),
            ),
            ("unused", json!({"params": [], "body": 0.0})),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let r = OperatorRefs::new(&reg);
        let mut models = json!({
            "M": {"variables": {}, "equations": [{"lhs": "y", "rhs": reference("a", json!({"x": "M.y"}))}]},
            "N": {"variables": {}, "equations": []},
        });
        r.attach_operators(&mut models).unwrap();
        assert!(models["N"].get("operators").is_none());
        let ops = models["M"]["operators"].as_object().unwrap();
        assert_eq!(ops.keys().collect::<Vec<_>>(), ["a", "b", "c"]);
        assert_eq!(ops["a"], reg["a"]);
        assert_eq!(
            ops["c"],
            json!({"params": [], "body": 1.0, "reference_only": true})
        );
        assert_eq!(r.read_definitions(), ["a", "b", "c"]);
        assert!(r.callees("missing").is_err());
        // A reference's reads are its bindings, never its name.
        let mut read = Vec::new();
        for_each_read_string(&models["M"]["equations"][0]["rhs"], &mut |s| {
            read.push(s.to_string())
        });
        read.sort();
        assert_eq!(read, ["M.y", "apply_operator"]);
    }
}
