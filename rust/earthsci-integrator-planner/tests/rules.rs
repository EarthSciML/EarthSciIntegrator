//! Refusals and resolutions of the merged stage-1 spec (spec PR #22) that no
//! conformance fixture exercises: each case writes a run config (and, where
//! needed, a model) into Cargo's test directory and checks the code and the
//! pointer.

mod common;

use common::conformance;
use earthsci_integrator_planner::astdiff::AstDiffBands;
use earthsci_integrator_planner::error::codes;
use earthsci_integrator_planner::{plan, Options, PlanError, Planned};
use serde_json::{json, Value};
use std::path::PathBuf;

fn dir(case: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("rules")
        .join(case);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn run(case: &str, model: Option<&Value>, rc: Value) -> Result<Planned, PlanError> {
    let d = dir(case);
    if let Some(m) = model {
        std::fs::write(d.join("model.esm"), serde_json::to_vec_pretty(m).unwrap()).unwrap();
    }
    let path = d.join("run.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&rc).unwrap()).unwrap();
    plan(&path, &Options::default(), &AstDiffBands)
}

fn refused(case: &str, model: Option<&Value>, rc: Value) -> (&'static str, String) {
    match run(case, model, rc) {
        Err(PlanError::Refused(r)) => (r.code, r.pointer),
        Err(e) => panic!("{case}: expected a refusal, got {e}"),
        Ok(_) => panic!("{case}: expected a refusal, got a plan"),
    }
}

fn robertson() -> String {
    conformance()
        .join("toys/robertson.esm")
        .canonicalize()
        .unwrap()
        .to_string_lossy()
        .into_owned()
}

/// A run config on the Robertson toy with one Rodas4 leaf, with `extra`
/// merged in at the top level and `leaf` merged into the leaf.
fn rob(extra: Value, leaf: Value) -> Value {
    let mut node = json!({"leaf": "kinetics", "alg": "Rodas4"});
    for (k, v) in leaf.as_object().unwrap() {
        node[k] = v.clone();
    }
    let mut rc = json!({
        "esm_integrator": "0.1.0",
        "model": robertson(),
        "operators": {"kinetics": {"select": {"components": "Kinetics"}}},
        "schedule": node,
    });
    for (k, v) in extra.as_object().unwrap() {
        rc[k] = v.clone();
    }
    rc
}

fn box_model(params: Value, events: bool) -> Value {
    let mut m = json!({
        "esm": "1.3.0",
        "metadata": {"name": "box"},
        "models": {"Box": {
            "variables": {"u": {"type": "unknown", "default": 1.5}},
            "equations": [{"lhs": {"op": "D", "args": ["u"], "wrt": "t"}, "rhs": {"op": "*", "args": [-1, "k", "u"]}}]
        }},
        "solver": {"time_span": {"start": 0, "end": 1}}
    });
    for (k, v) in params.as_object().unwrap() {
        m["models"]["Box"]["variables"][k] = v.clone();
    }
    if events {
        m["models"]["Box"]["discrete_events"] = json!([{
            "name": "halve",
            "trigger": {"type": "preset_times", "times": [0.5]},
            "affects": [{"lhs": "u", "rhs": {"op": "*", "args": [{"op": "Pre", "args": ["u"]}, 0.5]}}]
        }]);
    }
    m
}

fn box_rc() -> Value {
    json!({"esm_integrator": "0.1.0", "model": "model.esm",
           "schedule": {"leaf": "default_pointwise", "alg": "Rodas4"}})
}

/// §4 step 6: a crossing rule on any parameter takes the more specific
/// code, whatever the parameter order, but not ahead of events.
#[test]
fn crossing_updates() {
    let params = json!({
        "a": {"type": "parameter", "default": 0.5, "shape": [],
              "update": {"kind": "schedule", "times": [0.5], "expression": 0.25}},
        "k": {"type": "parameter", "default": 0.5,
              "update": {"kind": "crossing", "when": {"op": "-", "args": ["u", 1]}, "expression": 0.25}}
    });
    assert_eq!(
        refused(
            "crossing",
            Some(&box_model(params.clone(), false)),
            box_rc()
        ),
        (codes::UNSUPPORTED_CROSSING_UPDATE, "/model".into())
    );
    assert_eq!(
        refused("crossing-event", Some(&box_model(params, true)), box_rc()),
        (codes::UNSUPPORTED_EVENT, "/model".into())
    );
}

/// §4 step 1: a schema failure is at the empty pointer.
#[test]
fn schema_failure_at_root() {
    assert_eq!(
        refused("schema", None, rob(json!({"bogus": 1}), json!({}))),
        (codes::RUN_CONFIG_INVALID, String::new())
    );
    assert_eq!(
        refused("schema-deep", None, rob(json!({}), json!({"alg": "Nope"}))),
        (codes::RUN_CONFIG_INVALID, String::new())
    );
}

/// §7.1 rule 2, in its order.
#[test]
fn rule_2_names() {
    // An operator's bind_dt key that is a state, not a parameter.
    let ops = json!({"operators": {"kinetics": {
        "select": {"components": "Kinetics"}, "bind_dt": {"Kinetics.y1": "node"}}}});
    assert_eq!(
        refused("bind-op", None, rob(ops.clone(), json!({}))),
        (
            codes::NAME_WRONG_KIND,
            "/operators/kinetics/bind_dt/Kinetics.y1".into()
        )
    );
    // It comes before a bad step duration name.
    let mut both = ops;
    both["step"] = json!({"dt": "Nope.dt"});
    assert_eq!(
        refused("bind-op-first", None, rob(both, json!({}))),
        (
            codes::NAME_WRONG_KIND,
            "/operators/kinetics/bind_dt/Kinetics.y1".into()
        )
    );
    assert_eq!(
        refused(
            "step-dt",
            None,
            rob(json!({"step": {"dt": "Nope.dt"}}), json!({}))
        ),
        (codes::NAME_UNRESOLVED, "/step/dt".into())
    );
    // A parameter duration must be positive: run_config_invalid.
    assert_eq!(
        refused(
            "output-saveat",
            None,
            rob(json!({"output": {"saveat": "Kinetics.y1"}}), json!({}))
        ),
        (codes::NAME_WRONG_KIND, "/output/saveat".into())
    );
    // Node keys: bind_dt, abstol, nonnegative.
    assert_eq!(
        refused(
            "bind-leaf",
            None,
            rob(json!({}), json!({"bind_dt": {"Nope.k": "node"}}))
        ),
        (codes::NAME_UNRESOLVED, "/schedule/bind_dt/Nope.k".into())
    );
    assert_eq!(
        refused(
            "abstol",
            None,
            rob(
                json!({}),
                json!({"control": {"abstol": {"Kinetics.y1": 1e-8, "Kinetics.k1": 1e-8}}})
            )
        ),
        (
            codes::NAME_WRONG_KIND,
            "/schedule/control/abstol/Kinetics.k1".into()
        )
    );
    assert_eq!(
        refused(
            "nonneg",
            None,
            rob(
                json!({}),
                json!({"domain": {"nonnegative": ["Kinetics.y1", "Nope.x"]}})
            )
        ),
        (
            codes::NAME_UNRESOLVED,
            "/schedule/domain/nonnegative/1".into()
        )
    );
    // Precision keys: default, an operator or a node id; else name_unresolved.
    let ok = rob(
        json!({"precision": {"default": {"state": "f64"}, "kinetics": {"rhs": "f64"}, "root": {"tables": "f64"}}}),
        json!({}),
    );
    assert!(run("precision-ok", None, ok).is_ok());
    assert_eq!(
        refused(
            "precision-key",
            None,
            rob(json!({"precision": {"bogus": {"state": "f64"}}}), json!({}))
        ),
        (codes::NAME_UNRESOLVED, "/precision/bogus".into())
    );
}

/// §7.1: observed limits come after the schedule's unsupported constructs,
/// and a limit's name is resolved before it is refused.
#[test]
fn step_limits() {
    let solve = json!({"sequence": [
        {"solve": "kinetics", "unknowns": ["Kinetics.y2"], "method": {"newton": {}}},
        {"leaf": "kinetics", "alg": "Rodas4"}]});
    let mut rc = rob(json!({"step": {"limits": ["Kinetics.y1"]}}), json!({}));
    rc["schedule"] = solve;
    assert_eq!(
        refused("limits-after-solve", None, rc),
        (codes::UNSUPPORTED_SOLVE_NODE, "/schedule/sequence/0".into())
    );
    assert_eq!(
        refused(
            "limit-state",
            None,
            rob(
                json!({"step": {"limits": [{"max": 1}, "Kinetics.y1"]}}),
                json!({})
            )
        ),
        (codes::NAME_WRONG_KIND, "/step/limits/1".into())
    );
    assert_eq!(
        refused(
            "limit-unknown",
            None,
            rob(json!({"step": {"limits": ["Nope.x"]}}), json!({}))
        ),
        (codes::NAME_UNRESOLVED, "/step/limits/0".into())
    );
    let mut rc = rob(json!({"step": {"dt": 40}}), json!({}));
    rc["schedule"] =
        json!({"subcycle": {"leaf": "kinetics", "alg": "Rodas4"}, "limits": ["Kinetics.y3"]});
    assert_eq!(
        refused("subcycle-limit", None, rc),
        (codes::NAME_WRONG_KIND, "/schedule/limits/0".into())
    );
}

/// §7.1: an IMEXEuler leaf without imex.
#[test]
fn imex_euler_without_imex() {
    assert_eq!(
        refused(
            "imex-missing",
            None,
            rob(json!({}), json!({"alg": "IMEXEuler", "dt": 1}))
        ),
        (codes::IMEX_PARTITION_INVALID, "/schedule".into())
    );
}

/// §7.1: a leaf with a dt has no controller, dt0, dtmin or dtmax.
#[test]
fn fixed_dt_leaf_drops_adaptive_controls() {
    let p = run(
        "fixed-controls",
        None,
        rob(
            json!({"step": {"dt": 40}}),
            json!({"dt": 1, "linsolve": "dense_lu", "control": {"controller": "I", "dt0": 0.1, "dtmin": 0.01, "dtmax": 2, "maxiters": 7}}),
        ),
    )
    .unwrap();
    let s = p.resolve_schedule(None).unwrap();
    assert_eq!(
        s["control"],
        json!({"abstol": 0.000001, "dt_scope": "global", "maxiters": 7, "reltol": 0.001})
    );
}

/// §7.4: a partition.shards entry for a coupled node, or an operator a
/// coupled node names, is run_config_invalid.
#[test]
fn coupled_shards_refused() {
    let mut rc = rob(json!({"partition": {"shards": {"kinetics": 2}}}), json!({}));
    rc["schedule"] = json!({"coupled": "kinetics", "alg": "Rodas4"});
    assert_eq!(
        refused("coupled-op", None, rc.clone()),
        (
            codes::RUN_CONFIG_INVALID,
            "/partition/shards/kinetics".into()
        )
    );
    rc["partition"] = json!({"shards": {"root": 2}});
    assert_eq!(
        refused("coupled-id", None, rc.clone()),
        (codes::RUN_CONFIG_INVALID, "/partition/shards/root".into())
    );
    rc["partition"] = json!({});
    let p = run("coupled-none", None, rc).unwrap();
    assert_eq!(p.partition["shards"], json!({}));
}

/// §7.4: shard_size cells count horizontal dimensions, plus the vertical one
/// for a pointwise leaf only.
#[test]
fn shard_size_cells() {
    let model = conformance()
        .join("plans/fixtures/reseact_shaped/model.esm")
        .canonicalize()
        .unwrap();
    let rc = json!({
        "esm_integrator": "0.1.0",
        "model": model.to_string_lossy(),
        "axes": {"horizontal": ["lon", "lat"], "vertical": "lev"},
        "operators": {
            "transport": {"select": {"components": "Transport"}},
            "mixing": {"select": {"components": "Mix"}},
            "chemistry": {"select": {"components": ["Chem", "Dep"]}}
        },
        "step": {"dt": 300},
        "schedule": {"sequence": [
            {"leaf": "transport", "alg": "SSPRK43"},
            {"leaf": "mixing", "alg": "Tsit5", "partition": {"shard_size": 5}},
            {"leaf": "chemistry", "alg": "Rosenbrock23", "partition": {"shard_size": 10}}
        ]}
    });
    let p = run("shard-size", None, rc).unwrap();
    // Column leaf: lon × lat = 24 columns, ⌈24/5⌉ = 5. Pointwise leaf:
    // lon × lat × lev = 96 cells, ⌈96/10⌉ = 10.
    assert_eq!(p.partition["shards"], json!({"root.1": 5, "root.2": 10}));
}

/// §7.5: states of one leaf that declare different element types.
#[test]
fn mixed_element_types() {
    let m = json!({
        "esm": "1.3.0",
        "metadata": {"name": "mixed"},
        "models": {"Box": {
            "variables": {
                "u": {"type": "unknown", "default": 1.5, "element_type": "Float32"},
                "w": {"type": "unknown", "default": 1.5, "element_type": "Float64"},
                "k": {"type": "parameter", "default": 0.5}
            },
            "equations": [
                {"lhs": {"op": "D", "args": ["u"], "wrt": "t"}, "rhs": {"op": "*", "args": [-1, "k", "u"]}},
                {"lhs": {"op": "D", "args": ["w"], "wrt": "t"}, "rhs": {"op": "*", "args": [-1, "k", "w"]}}
            ]
        }},
        "solver": {"time_span": {"start": 0, "end": 1}}
    });
    let r = run("mixed-types", Some(&m), box_rc());
    match r {
        Err(PlanError::Refused(r)) => {
            assert_eq!(
                (r.code, r.pointer.as_str()),
                (codes::PRECISION_CONTRADICTS_DECLARATION, "/schedule")
            )
        }
        // EarthSciAST may refuse to load a mixed-precision model itself.
        other => panic!("{other:?}"),
    }
    let mut rc = box_rc();
    rc["precision"] = json!({"default": {"state": "f32"}});
    assert_eq!(
        refused("mixed-types-entry", Some(&m), rc),
        (
            codes::PRECISION_CONTRADICTS_DECLARATION,
            "/precision/default".into()
        )
    );
}

/// §7.5: `precision_conflict` is reported at the first operator entry, in
/// the leaf's operator order, that disagrees with the first entry setting
/// the role.
#[test]
fn precision_conflict_pointer() {
    let model = conformance()
        .join("plans/fixtures/cross_component_term/model.esm")
        .canonicalize()
        .unwrap();
    let rc = |precision: Value| {
        json!({
            "esm_integrator": "0.1.0",
            "model": model.to_string_lossy(),
            "operators": {
                "chemistry": {"select": {"components": "Chem"}},
                "deposition": {"select": {"components": "Dep"}},
                "scavenging": {"select": {"terms": ["coupling:0"], "variables": ["Chem.O3"]}}
            },
            "step": {"dt": 60},
            "precision": precision,
            // Leaf order differs from code-point order on purpose.
            "schedule": {"leaf": ["scavenging", "chemistry", "deposition"], "alg": "Rodas4", "linsolve": "dense_lu"}
        })
    };
    let at = |p: Value| refused("precision-conflict", None, rc(p));
    // First setter scavenging (f64); chemistry agrees; deposition disagrees.
    assert_eq!(
        at(
            json!({"scavenging": {"jacobian": "f64"}, "chemistry": {"jacobian": "f64"}, "deposition": {"jacobian": "f32"}})
        ),
        (codes::PRECISION_CONFLICT, "/precision/deposition".into())
    );
    // First setter scavenging; chemistry is the first to disagree, although
    // deposition agrees with chemistry.
    assert_eq!(
        at(
            json!({"scavenging": {"factors": "f64"}, "chemistry": {"factors": "f32"}, "deposition": {"factors": "f32"}})
        ),
        (codes::PRECISION_CONFLICT, "/precision/chemistry".into())
    );
    // scavenging sets nothing: chemistry is the first setter.
    assert_eq!(
        at(json!({"chemistry": {"tables": "f32"}, "deposition": {"tables": "f16"}})),
        (codes::PRECISION_CONFLICT, "/precision/deposition".into())
    );
}

/// Rule 3 checks the operator names of a leaf's `imex`, after its `frozen`
/// entries and before rule 6 checks the partition; locality names are not
/// operator names; an `imex` on another method is still checked by rule 3
/// but not by rule 6.
#[test]
fn imex_operator_names() {
    let imex = |v: Value| {
        rob(
            json!({"step": {"dt": 40}}),
            json!({"alg": "IMEXEuler", "dt": 1, "imex": v}),
        )
    };
    // An undefined operator in a list, even with a partition that is also bad.
    assert_eq!(
        refused(
            "imex-undef-list",
            None,
            imex(json!({"implicit": ["kinetics"], "explicit": ["kinetics", "ghost"]}))
        ),
        (
            codes::OPERATOR_UNDEFINED,
            "/schedule/imex/explicit/1".into()
        )
    );
    // `explicit` before `implicit`.
    assert_eq!(
        refused(
            "imex-undef-both",
            None,
            imex(json!({"implicit": "ghost1", "explicit": ["ghost2"]}))
        ),
        (
            codes::OPERATOR_UNDEFINED,
            "/schedule/imex/explicit/0".into()
        )
    );
    // A part written as one string.
    assert_eq!(
        refused(
            "imex-undef-string",
            None,
            imex(json!({"implicit": "ghost"}))
        ),
        (codes::OPERATOR_UNDEFINED, "/schedule/imex/implicit".into())
    );
    // A locality name is not checked by rule 3; the partition then fails
    // rule 6 (the explicit part is empty, kinetics is pointwise and in both).
    assert_eq!(
        refused(
            "imex-locality",
            None,
            imex(json!({"implicit": "pointwise", "explicit": "kinetics"}))
        ),
        (codes::IMEX_PARTITION_INVALID, "/schedule/imex".into())
    );
    // Frozen entries come before imex entries.
    let both = rob(
        json!({"step": {"dt": 40}}),
        json!({"alg": "IMEXEuler", "dt": 1, "frozen": [{"every": "ghost1", "period": 10}],
               "imex": {"implicit": ["ghost2"]}}),
    );
    assert_eq!(
        refused("imex-after-frozen", None, both),
        (codes::OPERATOR_UNDEFINED, "/schedule/frozen/0/every".into())
    );
    // On a leaf whose written alg is not IMEXEuler, rule 3 still checks imex
    let undefined = rob(
        json!({}),
        json!({"linsolve": "dense_lu", "imex": {"implicit": ["ghost"]}}),
    );
    assert_eq!(
        refused("imex-other-alg", None, undefined),
        (
            codes::OPERATOR_UNDEFINED,
            "/schedule/imex/implicit/0".into()
        )
    );
    // but rule 6 does not: a bad partition of defined operators is ignored.
    let ignored = rob(
        json!({}),
        json!({"linsolve": "dense_lu", "imex": {"implicit": ["kinetics"], "explicit": ["kinetics"]}}),
    );
    assert!(run("imex-ignored", None, ignored).is_ok());
}

/// spec/plan.md §6.1.3: the `unnamed_lift` fixture's model with its grid
/// declared as the index set `cell`, which no variable's shape names: the
/// only declared index set of the lift's size, so `flatten` names the
/// lift's dimension `cell`, and every derived document declares it and the
/// lifted `faq` ranges over it. (The fixture itself, with no index set,
/// covers the synthetic `_liftdim1_4`.)
#[test]
fn lift_onto_declared_unnamed_grid() {
    let path = conformance().join("plans/fixtures/unnamed_lift/model.esm");
    let mut model: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    model["index_sets"] = json!({"cell": {"kind": "interval", "size": 4}});
    let rc = json!({"esm_integrator": "0.1.0", "model": "model.esm", "schedule": "auto"});
    let p = run("lift-declared-unnamed-grid", Some(&model), rc).unwrap();
    for x in &p.model.states {
        assert_eq!(p.model.shapes[x], ["cell"], "shape of {x}");
    }
    let sets = json!({"cell": {"kind": "interval", "size": 4}});
    for (op, d) in &p.documents {
        let rhs: Value = serde_json::from_slice(&d.rhs).unwrap();
        let jac: Value = serde_json::from_slice(&d.jacobian).unwrap();
        assert_eq!(rhs["index_sets"], sets, "{op}.rhs index_sets");
        assert_eq!(jac["index_sets"], sets, "{op}.jacobian index_sets");
        for eq in rhs["models"]["Chem"]["equations"].as_array().unwrap() {
            assert_eq!(
                eq["rhs"]["ranges"],
                json!({"i": {"from": "cell"}}),
                "{op} ranges"
            );
        }
    }
}

/// §0.1, §7.1, D94: `dt_scope` `cell` and `shard` wait for M3 and are refused
/// at the leaf's `control/dt_scope`, in tree order with the other
/// unsupported constructs; `global` and `"auto"` plan, and an omitted or
/// `"auto"` `dt_scope` resolves to `global`, the pointwise leaf included.
#[test]
fn dt_scope_at_m2() {
    for scope in ["cell", "shard"] {
        assert_eq!(
            refused(
                &format!("dt-scope-{scope}"),
                None,
                rob(json!({}), json!({"control": {"dt_scope": scope}}))
            ),
            (
                codes::UNSUPPORTED_DT_SCOPE,
                "/schedule/control/dt_scope".into()
            ),
            "{scope}"
        );
    }
    // In a sequence, the refusal points into the child, and a map node
    // earlier in tree order is refused first.
    let mut rc = rob(json!({}), json!({}));
    rc["schedule"] = json!({"sequence": [
        {"leaf": "kinetics", "alg": "Rodas4", "control": {"dt_scope": "cell"}}
    ]});
    assert_eq!(
        refused("dt-scope-seq", None, rc.clone()),
        (
            codes::UNSUPPORTED_DT_SCOPE,
            "/schedule/sequence/0/control/dt_scope".into()
        )
    );
    rc["schedule"]["sequence"]
        .as_array_mut()
        .unwrap()
        .insert(0, json!({"map": "m"}));
    assert_eq!(
        refused("dt-scope-after-map", None, rc).0,
        codes::UNSUPPORTED_MAP_OPERATOR
    );
    for (case, control) in [
        ("dt-scope-global", json!({"dt_scope": "global"})),
        ("dt-scope-auto", json!({"dt_scope": "auto"})),
        ("dt-scope-omitted", json!({})),
    ] {
        let p = run(case, None, rob(json!({}), json!({"control": control}))).unwrap();
        assert_eq!(p.operators["kinetics"].locality.class.as_str(), "pointwise");
        assert_eq!(
            p.resolved_schedule["control"]["dt_scope"], "global",
            "{case}"
        );
    }
}

/// spec/plan.md §0.2: `plan.report.json` names the upstream versions this
/// build was compiled against, not the ones the spec names: earthsci-ast's
/// own constant, and the earthsci-astdiff version of the workspace's
/// Cargo.lock (read by build.rs).
#[test]
fn report_names_the_built_versions() {
    let p = run("report-versions", None, rob(json!({}), json!({}))).unwrap();
    let up = &p.report["upstream"];
    assert_eq!(up["earthsci-ast"], earthsci_ast::LIBRARY_VERSION);
    let lock =
        std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../Cargo.lock"))
            .unwrap();
    let mut lines = lock.lines();
    let mut astdiff = None;
    while let Some(l) = lines.next() {
        if l == "name = \"earthsci-astdiff\"" {
            let v = lines.next().unwrap();
            astdiff = v
                .strip_prefix("version = \"")
                .map(|v| v.trim_end_matches('"'));
        }
    }
    assert_eq!(up["earthsci-astdiff"], astdiff.unwrap());
    assert_eq!(p.report["version"], env!("CARGO_PKG_VERSION"));
}
