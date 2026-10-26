//! Resolved settings other than the schedule (spec/plan.md §7.3–§7.7):
//! step, partition, precision, the checkpoint placeholder with the adjoint's
//! checks, and inputs.

use crate::axes::Axes;
use crate::error::{codes, ptr, refuse, Result};
use crate::locality::Class;
use crate::model::ModelInfo;
use crate::operators::Operator;
use crate::runconfig::RunConfig;
use crate::schedule::{duration, leaf_class, Kind, Node};
use crate::strategy::classical_rosenbrock;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// §7.3: `step` with `dt` or `limits` (each `{"max": seconds}`) when given,
/// `hit` always (sorted), and `hit_multiples` and `resolution` when given.
/// An observed limit is `unsupported_step_limit`. Rules 9 and 12 are
/// checked with the schedule.
pub fn step(rc: &RunConfig, info: &ModelInfo) -> Result<Value> {
    let given = rc.at("/step").and_then(Value::as_object).cloned().unwrap_or_default();
    let mut out = Map::new();
    if let Some(dt) = given.get("dt") {
        out.insert("dt".into(), json!(duration(dt, "/step/dt", info)?));
    }
    if let Some(Value::Array(limits)) = given.get("limits") {
        let mut list = Vec::new();
        for (i, l) in limits.iter().enumerate() {
            let at = ptr("/step/limits", i);
            match l.get("max") {
                Some(m) => list.push(json!({"max": duration(m, &ptr(&at, "max"), info)?})),
                None => {
                    return Err(refuse(
                        codes::UNSUPPORTED_STEP_LIMIT,
                        at,
                        format!("the observed step limit {l} needs a limit operator (U6, M4b)"),
                    ))
                }
            }
        }
        out.insert("limits".into(), Value::Array(list));
    }
    let hit: BTreeSet<String> = given
        .get("hit")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
        .unwrap_or_default();
    out.insert("hit".into(), json!(hit));
    for k in ["hit_multiples", "resolution"] {
        if let Some(v) = given.get(k) {
            out.insert(k.into(), json!(duration(v, &ptr("/step", k), info)?));
        }
    }
    Ok(Value::Object(out))
}

/// The states a leaf writes: the union of its operators' `writes`, in
/// canonical order.
pub fn leaf_states(n: &Node, ops: &BTreeMap<String, Operator>) -> Vec<String> {
    let mut out = BTreeSet::new();
    if let Kind::Leaf { ops: names, .. } = &n.kind {
        for o in names {
            if let Some(op) = ops.get(o) {
                out.extend(op.writes.iter().cloned());
            }
        }
    }
    out.into_iter().collect()
}

/// §7.4 (D77): `ranks`, `tiles`, `halo` and `shards`.
///
/// SPEC FINDING (§7.4): "the leaf's array cells" for `shard_size` is read
/// as the cells of the first array state, in canonical order, the leaf
/// writes (1 when it writes only scalars), and "columns" as those cells
/// divided by the vertical extent of that state's shape.
pub fn partition(
    rc: &RunConfig,
    info: &ModelInfo,
    axes: &Axes,
    ops: &BTreeMap<String, Operator>,
    root: &Node,
) -> Result<Value> {
    let grid = |k: &str| -> Vec<i64> {
        rc.at(&format!("/partition/{k}"))
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_i64).collect())
            .unwrap_or_else(|| vec![1, 1])
    };
    let (ranks, tiles) = (grid("ranks"), grid("tiles"));
    for a in 0..2 {
        match axes.horizontal.get(a) {
            None => {
                if ranks[a] != 1 || tiles[a] != 1 {
                    return Err(refuse(
                        codes::PARTITION_AXIS_MISSING,
                        "/partition",
                        format!("ranks and tiles entry {a} divide a horizontal axis the model does not have"),
                    ));
                }
            }
            Some(h) => {
                let n = info.index_sets[h];
                if ranks[a] * tiles[a] > n {
                    return Err(refuse(
                        codes::PARTITION_TOO_FINE,
                        "/partition",
                        format!(
                            "ranks × tiles = {} × {} exceeds the {n} cells of axis {h}",
                            ranks[a], tiles[a]
                        ),
                    ));
                }
            }
        }
    }
    let mut out = Map::new();
    out.insert("ranks".into(), json!(ranks));
    out.insert("tiles".into(), json!(tiles));
    if !axes.horizontal.is_empty() {
        let mut halo = Map::new();
        for h in &axes.horizontal {
            let w = ops
                .values()
                .filter(|o| o.locality.class == Class::Stencil)
                .filter_map(|o| o.locality.radius.get(h))
                .map(|(lo, up)| (*lo).max(*up))
                .max()
                .unwrap_or(0);
            halo.insert(h.clone(), json!(w));
        }
        out.insert("halo".into(), Value::Object(halo));
    }
    let run_shards = rc.at("/partition/shards").and_then(Value::as_object);
    let count = |v: &Value| v.as_i64().unwrap_or(1); // "auto" is 1 at M2.
    let mut shards = Map::new();
    for n in root.leaves() {
        let Kind::Leaf { ops: names, coupled: false } = &n.kind else {
            continue;
        };
        let class = leaf_class(n, ops);
        if !matches!(class, Class::Pointwise | Class::Column) {
            continue;
        }
        let own = n.value.get("partition");
        let k = if let Some(s) = own.and_then(|p| p.get("shards")) {
            count(s)
        } else if let Some(size) = own.and_then(|p| p.get("shard_size")).and_then(Value::as_i64) {
            let states = leaf_states(n, ops);
            let first = states.iter().find(|s| !info.shapes[*s].is_empty());
            let cells = first.map_or(1, |s| {
                let all = info.cells(s);
                match (&axes.vertical, class) {
                    (Some(v), Class::Column) if info.shapes[s].contains(v) => {
                        all / info.index_sets[v]
                    }
                    _ => all,
                }
            });
            (cells + size - 1) / size
        } else if let Some(s) = run_shards.and_then(|m| m.get(n.id())) {
            count(s)
        } else {
            let mut found: Option<(String, i64)> = None;
            for o in names {
                if let Some(s) = run_shards.and_then(|m| m.get(o)) {
                    let c = count(s);
                    match &found {
                        Some((first, f)) if *f != c => {
                            return Err(refuse(
                                codes::PARTITION_CONFLICT,
                                ptr("/partition/shards", o),
                                format!("partition.shards gives {first} {f} shards and {o} {c}, both advanced by leaf {}", n.id()),
                            ))
                        }
                        Some(_) => {}
                        None => found = Some((o.clone(), c)),
                    }
                }
            }
            found.map_or(1, |(_, c)| c)
        };
        shards.insert(n.id().to_string(), json!(k));
    }
    out.insert("shards".into(), Value::Object(shards));
    Ok(Value::Object(out))
}

const ROLES: [&str; 5] = ["state", "rhs", "jacobian", "factors", "tables"];

/// Bits of a precision, for "below" (f16 and bf16 rank equal).
fn bits(p: &str) -> u32 {
    match p {
        "f16" | "bf16" => 16,
        "f32" => 32,
        _ => 64,
    }
}

/// The precision an `element_type` declaration names.
fn declared(element_type: &str) -> Option<&'static str> {
    match element_type {
        "Float32" => Some("f32"),
        "Float64" => Some("f64"),
        _ => None,
    }
}

/// §7.5: every precision role for every leaf and `coupled` node, by id.
/// `algs` gives each leaf's method where it is known (a given `alg`, or
/// the resolved one); the classical-Rosenbrock check runs for those.
///
/// SPEC FINDING (§7.5): when the leaf's states declare different
/// `element_type`s, no `state` role satisfies them all; that is read as
/// `precision_contradicts_declaration`.
pub fn precision(
    rc: &RunConfig,
    info: &ModelInfo,
    ops: &BTreeMap<String, Operator>,
    root: &Node,
    algs: &BTreeMap<String, String>,
) -> Result<Value> {
    let entries = rc.at("/precision").and_then(Value::as_object);
    let entry = |k: &str| entries.and_then(|m| m.get(k)).and_then(Value::as_object);
    let domain_type = info
        .file
        .domain
        .as_ref()
        .and_then(|d| d.element_type.as_deref())
        .and_then(declared);
    let mut out = Map::new();
    for n in root.leaves() {
        let Kind::Leaf { ops: names, .. } = &n.kind else {
            continue;
        };
        // role → (value, pointer of the entry that set it)
        let mut set: BTreeMap<&str, (String, String)> = BTreeMap::new();
        if let Some(d) = entry("default") {
            for (r, v) in d {
                set.insert(
                    ROLES.iter().find(|x| *x == r).copied().unwrap_or("state"),
                    (v.as_str().unwrap_or_default().to_string(), "/precision/default".into()),
                );
            }
        }
        let mut from_ops: BTreeMap<&str, (String, String)> = BTreeMap::new();
        for o in names {
            if let Some(e) = entry(o) {
                for role in ROLES {
                    let Some(v) = e.get(role).and_then(Value::as_str) else {
                        continue;
                    };
                    let at = ptr("/precision", o);
                    match from_ops.get(role) {
                        Some((w, first)) if w != v => {
                            return Err(refuse(
                                codes::PRECISION_CONFLICT,
                                at,
                                format!("{first} and {o} disagree on the {role} role of leaf {}", n.id()),
                            ))
                        }
                        Some(_) => {}
                        None => {
                            from_ops.insert(role, (v.to_string(), at));
                        }
                    }
                }
            }
        }
        set.extend(from_ops);
        if let Some(e) = entry(n.id()) {
            for role in ROLES {
                if let Some(v) = e.get(role).and_then(Value::as_str) {
                    set.insert(role, (v.to_string(), ptr("/precision", n.id())));
                }
            }
        }
        // The declarations that govern the leaf's states.
        let decl: BTreeSet<&str> = leaf_states(n, ops)
            .iter()
            .filter_map(|s| {
                info.flat.state_variables[s]
                    .element_type
                    .as_deref()
                    .and_then(declared)
                    .or(domain_type)
            })
            .collect();
        if decl.len() > 1 {
            return Err(refuse(
                codes::PRECISION_CONTRADICTS_DECLARATION,
                "/precision",
                format!("the states of leaf {} declare different element types", n.id()),
            ));
        }
        let decl = decl.into_iter().next();
        let get = |set: &BTreeMap<&str, (String, String)>, r: &str| set.get(r).cloned();
        let state = get(&set, "state")
            .unwrap_or_else(|| (decl.unwrap_or("f64").to_string(), "/precision".into()));
        let rhs = get(&set, "rhs").unwrap_or_else(|| state.clone());
        let jacobian = get(&set, "jacobian").unwrap_or_else(|| rhs.clone());
        let factors = get(&set, "factors").unwrap_or_else(|| rhs.clone());
        let tables = get(&set, "tables").unwrap_or_else(|| ("f64".into(), "/precision".into()));
        for (role, (v, at)) in [("state", &state), ("rhs", &rhs)] {
            if let Some(d) = decl {
                if v != d {
                    return Err(refuse(
                        codes::PRECISION_CONTRADICTS_DECLARATION,
                        at.clone(),
                        format!("the {role} role {v} of leaf {} contradicts the declared element type {d}", n.id()),
                    ));
                }
            }
        }
        if bits(&rhs.0) < bits(&state.0) {
            return Err(refuse(
                codes::PRECISION_RHS_BELOW_STATE,
                rhs.1.clone(),
                format!("leaf {}: rhs {} is below state {}", n.id(), rhs.0, state.0),
            ));
        }
        if let Some(alg) = algs.get(n.id()).filter(|a| classical_rosenbrock(a)) {
            for (role, (v, at)) in [("jacobian", &jacobian), ("factors", &factors)] {
                if bits(v) < bits(&rhs.0) {
                    return Err(refuse(
                        codes::PRECISION_LOW_WITH_CLASSICAL_ROSENBROCK,
                        at.clone(),
                        format!("leaf {}: {role} {v} is below rhs {} on {alg}, a classical Rosenbrock method", n.id(), rhs.0),
                    ));
                }
            }
        }
        out.insert(
            n.id().to_string(),
            json!({
                "state": state.0, "rhs": rhs.0, "jacobian": jacobian.0,
                "factors": factors.0, "tables": tables.0,
            }),
        );
    }
    Ok(Value::Object(out))
}

/// `*` matches any characters, dots included; everything else literally.
fn glob(pattern: &str, name: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == name;
    }
    let (first, last) = (parts[0], parts[parts.len() - 1]);
    if !name.starts_with(first) || name.len() < first.len() + last.len() || !name.ends_with(last) {
        return false;
    }
    let mut rest = &name[first.len()..name.len() - last.len()];
    for mid in &parts[1..parts.len() - 1] {
        match rest.find(mid) {
            Some(i) => rest = &rest[i + mid.len()..],
            None => return false,
        }
    }
    true
}

/// The parameters bound by a `bind_dt` anywhere in the run config.
fn bound_parameters(rc: &RunConfig, root: &Node) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut add = |b: Option<&Value>| {
        if let Some(m) = b.and_then(Value::as_object) {
            out.extend(m.keys().cloned());
        }
    };
    if let Some(ops) = rc.at("/operators").and_then(Value::as_object) {
        for op in ops.values() {
            add(op.get("bind_dt"));
        }
    }
    for n in root.preorder() {
        add(n.value.get("bind_dt"));
    }
    out
}

/// §7.6 (D78): the checkpoint placeholder, present exactly when the run
/// config has `adjoint`, after the adjoint's rule 11.
pub fn checkpoint(rc: &RunConfig, info: &ModelInfo, root: &Node) -> Result<Option<Value>> {
    let Some(adjoint) = rc.at("/adjoint") else {
        return Ok(None);
    };
    let bound = bound_parameters(rc, root);
    let scalar = |v: &earthsci_ast::ModelVariable| v.shape.as_ref().is_none_or(|s| s.is_empty());
    for (i, w) in adjoint["wrt"].as_array().into_iter().flatten().enumerate() {
        let w = w.as_str().unwrap_or_default();
        let at = ptr("/adjoint/wrt", i);
        let params = info.flat.parameters.iter();
        let matched: Vec<String> = if w == "u0" {
            info.states.clone()
        } else if w == "params:numeric" {
            params
                .filter(|(_, v)| scalar(v) && v.default_scalar().is_some())
                .map(|(k, _)| k.clone())
                .collect()
        } else if let Some(p) = w.strip_prefix("params:") {
            params.filter(|(k, v)| scalar(v) && glob(p, k)).map(|(k, _)| k.clone()).collect()
        } else if let Some(p) = w.strip_prefix("fields:") {
            params.filter(|(k, v)| !scalar(v) && glob(p, k)).map(|(k, _)| k.clone()).collect()
        } else {
            // forcing:<pattern>: M2 plans no forcing fields.
            vec![]
        };
        if matched.is_empty() {
            return Err(refuse(
                codes::WRT_MATCHES_NOTHING,
                at,
                format!("adjoint.wrt entry {w} matches nothing"),
            ));
        }
        let named = w.split_once(':').map(|(_, p)| p).filter(|p| !p.contains('*'));
        if let Some(name) = named.filter(|n| bound.contains(*n)) {
            return Err(refuse(
                codes::WRT_BOUND_PARAMETER,
                at,
                format!("{name} is bound by bind_dt and cannot be differentiated"),
            ));
        }
    }
    Ok(Some(json!({"kind": "all", "macro_steps": 0})))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// §7.7: the model as the run config writes it, the run config by file
/// name, each with the SHA-256 of its bytes.
pub fn inputs(rc: &RunConfig, info: &ModelInfo) -> Value {
    json!({
        "model": {"path": rc.model(), "sha256": sha256_hex(&info.bytes)},
        "run_config": {"path": rc.file_name(), "sha256": sha256_hex(&rc.bytes)},
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn globs() {
        assert!(glob("*", "Chem.k"));
        assert!(glob("Chem.*", "Chem.k"));
        assert!(glob("Chem.*", "Chem.Gas.k"));
        assert!(!glob("Nope.*", "Chem.k"));
        assert!(glob("*.k", "A.B.k"));
        assert!(glob("A*k*", "A.B.k1"));
        assert!(!glob("A*k*z", "A.B.k1"));
        assert!(glob("Chem.k", "Chem.k"));
        assert!(!glob("Chem.k", "Chem.k2"));
    }

    #[test]
    fn hashes() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
