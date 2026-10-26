//! The plan fixtures of `tests/conformance/plans/fixtures/`, as far as
//! stage 1 goes: every section stage 1 computes is compared with the case's
//! `expected.md` (and, for `cross_component_term/by-label`, byte for byte
//! with its hand-written `expected-plan.json`). Facts that come from
//! EarthSciASTDiff in stage 2 (`structure.block` sizes, stiffness by the
//! Gershgorin bound) are supplied from `expected.md`.

mod common;

use common::{conformance, HandBands};
use earthsci_integrator_planner::plan::StructureFacts;
use earthsci_integrator_planner::{jcs, plan, Options, PlanError, Planned};
use serde_json::{json, Value};

fn planned(fixture: &str, case: &str) -> Planned {
    let path = conformance()
        .join("plans/fixtures")
        .join(fixture)
        .join(format!("{case}.run.json"));
    plan(&path, &Options::default(), &HandBands).unwrap_or_else(|e| panic!("{fixture}/{case}: {e}"))
}

fn canon(v: &Value) -> String {
    jcs::to_string(v).unwrap()
}

fn same(got: &Value, want: &Value, what: &str) {
    assert_eq!(canon(got), canon(want), "{what}");
}

fn facts(stiff: &[(&str, bool)], blocks: &[(&str, &[&str])]) -> StructureFacts {
    StructureFacts {
        gershgorin_stiff: stiff.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        block_vars: blocks
            .iter()
            .map(|(k, v)| {
                (
                    k.to_string(),
                    Some(v.iter().map(|s| s.to_string()).collect()),
                )
            })
            .collect(),
    }
}

fn locality_classes(p: &Planned) -> Vec<(String, usize, usize, String, &'static str)> {
    p.terms
        .iter()
        .zip(&p.term_localities)
        .map(|(t, l)| {
            (
                t.variable.clone(),
                t.contribution,
                t.summand,
                t.provenance.clone(),
                l.class.as_str(),
            )
        })
        .collect()
}

#[test]
fn cross_component_term_by_label() {
    let p = planned("cross_component_term", "by-label");
    let path =
        conformance().join("plans/fixtures/cross_component_term/by-label.expected-plan.json");
    let text = std::fs::read_to_string(path).unwrap();
    let want: Value = serde_json::from_str(&text).unwrap();
    // The hand-written plan is in canonical form already.
    assert_eq!(canon(&want), text.trim_end_matches('\n'));
    same(&p.inputs, &want["inputs"], "inputs");
    same(&p.time_json(), &want["time"], "time");
    same(&p.step, &want["step"], "step");
    same(&p.partition, &want["partition"], "partition");
    same(&p.precision, &want["precision"], "precision");
    assert!(p.checkpoint_schedule.is_none());
    let f = facts(
        &[],
        &[
            ("chemistry", &["Chem.NO", "Chem.O3"]),
            ("deposition", &["Chem.NO", "Chem.O3"]),
        ],
    );
    same(
        &p.resolve_schedule(Some(&f)).unwrap(),
        &want["schedule"],
        "schedule",
    );
    assert_eq!(
        p.operators.keys().collect::<Vec<_>>(),
        ["chemistry", "deposition"]
    );
    for name in ["chemistry", "deposition"] {
        let mut w = want["operators"][name].clone();
        w.as_object_mut().unwrap().remove("structure");
        same(&p.operator_json_stage1(name), &w, name);
    }
    // Without structure the plan is not written.
    assert!(matches!(
        p.plan_json(),
        Err(PlanError::NotYetImplemented(_))
    ));
    assert!(matches!(
        p.resolve_schedule(None),
        Err(PlanError::NotYetImplemented(_))
    ));
}

#[test]
fn cross_component_term_by_owner() {
    let p = planned("cross_component_term", "by-owner");
    let w = |n: &str| p.operators[n].writes.clone();
    let r = |n: &str| p.operators[n].reads.clone();
    assert_eq!(w("deposition"), ["Chem.NO"]);
    assert_eq!(r("deposition"), ["Chem.NO", "Dep.kNO"]);
    assert_eq!(w("scavenging"), ["Chem.O3"]);
    assert_eq!(r("scavenging"), ["Chem.O3", "Dep.vd", "Met.h"]);
    assert_eq!(p.operators["chemistry"].terms.len(), 3);
    for op in ["chemistry", "deposition", "scavenging"] {
        same(
            &p.operator_json_stage1(op)["stiffness"],
            &json!({"class": "stiff", "source": "run_config"}),
            op,
        );
    }
    let f = facts(
        &[],
        &[
            ("chemistry", &["Chem.NO", "Chem.O3"]),
            ("deposition", &["Chem.NO"]),
            ("scavenging", &["Chem.O3"]),
        ],
    );
    // SPEC FINDING: expected.md shows this leaf as the root (`id` "root",
    // shards {"root": 1}), but by-owner.run.json writes it inside a
    // one-child `sequence`, and spec/plan.md §7.1 keeps a written sequence
    // (only the "auto" schedule unwraps a single leaf, D72). The literal
    // reading is tested: the leaf is `root.0`.
    same(
        &p.resolve_schedule(Some(&f)).unwrap(),
        &json!({"id": "root", "sequence": [{"alg": "Rodas4", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "cell", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root.0", "jacobian": "symbolic", "leaf": ["chemistry", "deposition", "scavenging"], "linsolve": {"block_lu": {"pivot": true, "refine": 0}}}]}),
        "schedule",
    );
    same(&p.partition["shards"], &json!({"root.0": 1}), "shards");
}

const CHEM: [&str; 3] = ["Chem.NO", "Chem.NO2", "Chem.O3"];

#[test]
fn pointwise_chemistry() {
    for case in ["auto", "named"] {
        let p = planned("pointwise_chemistry", case);
        let op = if case == "auto" {
            "default_pointwise"
        } else {
            "chemistry"
        };
        assert_eq!(p.operators.keys().collect::<Vec<_>>(), [op]);
        assert_eq!(p.model.shapes["Chem.NO"], ["cell"]);
        assert_eq!(p.axes.horizontal, ["cell"]);
        let terms = locality_classes(&p);
        let want: Vec<(&str, usize, usize, &str)> = vec![
            ("Chem.NO2", 0, 0, "Chem"),
            ("Chem.NO2", 0, 1, "Chem"),
            ("Chem.NO2", 1, 0, "Dep"),
            ("Chem.NO", 0, 0, "Chem"),
            ("Chem.NO", 0, 1, "Chem"),
            ("Chem.NO", 1, 0, "Dep"),
            ("Chem.O3", 0, 0, "Chem"),
            ("Chem.O3", 0, 1, "Chem"),
            ("Chem.O3", 1, 0, "Dep"),
        ];
        assert_eq!(terms.len(), 9);
        for (t, w) in terms.iter().zip(&want) {
            assert_eq!(
                (t.0.as_str(), t.1, t.2, t.3.as_str(), t.4),
                (w.0, w.1, w.2, w.3, "pointwise")
            );
        }
        let o = p.operator_json_stage1(op);
        same(&o["locality"], &json!({"class": "pointwise"}), "locality");
        same(&o["writes"], &json!(CHEM), "writes");
        same(
            &o["reads"],
            &json!([
                "Chem.NO",
                "Chem.NO2",
                "Chem.O3",
                "Chem.j",
                "Chem.k",
                "Dep.cellw",
                "Dep.kd"
            ]),
            "reads",
        );
        same(
            &o["stiffness"],
            &json!({"class": "stiff", "source": "reaction_systems"}),
            "stiffness",
        );
        same(&p.time_json(), &json!({"start": 0, "end": 3600}), "time");
        same(&p.step, &json!({"hit": []}), "step");
        let shards = if case == "auto" { 1 } else { 2 };
        same(
            &p.partition,
            &json!({"halo": {"cell": 0}, "ranks": [1, 1], "shards": {"root": shards}, "tiles": [1, 1]}),
            "partition",
        );
        same(
            &p.precision,
            &json!({"root": {"factors": "f64", "jacobian": "f64", "rhs": "f64", "state": "f64", "tables": "f64"}}),
            "precision",
        );
        let f = facts(&[], &[(op, &CHEM)]);
        let schedule = if case == "auto" {
            json!({"alg": "Rosenbrock23", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "cell", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "jacobian": "symbolic", "leaf": ["default_pointwise"], "linsolve": {"block_lu": {"pivot": true, "refine": 0}}})
        } else {
            json!({"alg": "Rodas4", "control": {"abstol": 1e-9, "controller": "PI", "dt_scope": "cell", "maxiters": 1000000, "reltol": 0.000001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "jacobian": "symbolic", "leaf": ["chemistry"], "linsolve": {"block_lu": {"pivot": true, "refine": 0}}})
        };
        same(&p.resolve_schedule(Some(&f)).unwrap(), &schedule, case);
    }
}

#[test]
fn periodic_stencil() {
    for case in ["auto", "partitioned"] {
        let p = planned("periodic_stencil", case);
        let op = if case == "auto" {
            "default_stencil"
        } else {
            "transport"
        };
        assert_eq!(p.operators.keys().collect::<Vec<_>>(), [op]);
        let o = p.operator_json_stage1(op);
        same(
            &o["locality"],
            &json!({"class": "stencil", "radius": {"lat": [0, 0], "lon": [3, 2]}}),
            "locality",
        );
        same(
            &o["reads"],
            &json!(["Transport.c", "Transport.dx", "Transport.q"]),
            "reads",
        );
        same(
            &o["terms"],
            &json!([{"variable": "Transport.q", "contribution": 0, "summand": 0, "provenance": "Transport", "locality": "stencil"}]),
            "terms",
        );
        same(&p.time_json(), &json!({"start": 0, "end": 0.5}), "time");
        let (grid, step, schedule) = if case == "auto" {
            assert!(
                o.get("stiffness").is_none(),
                "stiffness comes from the Gershgorin bound"
            );
            (
                [1, 1],
                json!({"hit": []}),
                json!({"alg": "SSPRK43", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "leaf": ["default_stencil"]}),
            )
        } else {
            same(
                &o["stiffness"],
                &json!({"class": "nonstiff", "source": "run_config"}),
                "stiffness",
            );
            (
                [2, 1],
                json!({"dt": 0.1, "hit": []}),
                json!({"alg": "SSPRK33", "control": {"abstol": 0.000001, "dt_scope": "global", "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "dt": 0.01, "id": "root", "leaf": ["transport"]}),
            )
        };
        same(&p.step, &step, "step");
        same(
            &p.partition,
            &json!({"halo": {"lat": 0, "lon": 3}, "ranks": grid, "shards": {}, "tiles": grid}),
            "partition",
        );
        let f = facts(&[(op, false)], &[]);
        same(&p.resolve_schedule(Some(&f)).unwrap(), &schedule, case);
        assert_eq!(
            p.reference_steps["root"],
            if case == "auto" { 0.5 } else { 0.01 }
        );
    }
}

#[test]
fn column_contraction() {
    let p = planned("column_contraction", "auto");
    let o = p.operator_json_stage1("default_column");
    same(&o["locality"], &json!({"class": "column"}), "locality");
    same(&o["reads"], &json!(["Mix.kmix", "Mix.q"]), "reads");
    same(&p.step, &json!({"dt": 600, "hit": []}), "step");
    same(
        &p.partition,
        &json!({"halo": {"lat": 0, "lon": 0}, "ranks": [1, 1], "shards": {"root": 1}, "tiles": [1, 1]}),
        "partition",
    );
    assert_eq!(p.reference_steps["root"], 600.0);
    let f = facts(&[("default_column", true)], &[]);
    same(
        &p.resolve_schedule(Some(&f)).unwrap(),
        &json!({"alg": "TRBDF2", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "jacobian": "symbolic", "leaf": ["default_column"], "linsolve": {"banded": {}}}),
        "schedule",
    );
}

#[test]
fn reseact_shaped() {
    for case in ["auto", "named"] {
        let p = planned("reseact_shaped", case);
        let names = if case == "auto" {
            ["default_column", "default_pointwise", "default_stencil"]
        } else {
            ["mixing", "chemistry", "transport"]
        };
        let [col, pw, st] = names;
        let mut sorted = names.to_vec();
        sorted.sort();
        assert_eq!(p.operators.keys().collect::<Vec<_>>(), sorted);
        let terms = locality_classes(&p);
        assert_eq!(terms.len(), 18);
        for species in CHEM {
            let mine: Vec<_> = terms.iter().filter(|t| t.0 == species).collect();
            let want = [
                (0, 0, "Chem", "pointwise"),
                (0, 1, "Chem", "pointwise"),
                (1, 0, "Dep", "pointwise"),
                (2, 0, "Transport", "stencil"),
                (2, 1, "Transport", "stencil"),
                (3, 0, "Mix", "column"),
            ];
            for (t, w) in mine.iter().zip(want) {
                assert_eq!((t.1, t.2, t.3.as_str(), t.4), w, "{species}");
            }
        }
        let o = |n| p.operator_json_stage1(n);
        same(
            &o(st)["locality"],
            &json!({"class": "stencil", "radius": {"lat": [1, 0], "lon": [3, 2]}}),
            "stencil",
        );
        same(
            &o(st)["reads"],
            &json!([
                "Chem.NO",
                "Chem.NO2",
                "Chem.O3",
                "Transport.c",
                "Transport.dx",
                "Transport.dy",
                "Transport.v"
            ]),
            "stencil reads",
        );
        same(
            &o(col)["reads"],
            &json!(["Chem.NO", "Chem.NO2", "Chem.O3", "Mix.kmix"]),
            "column reads",
        );
        same(
            &o(pw)["reads"],
            &json!(["Chem.NO", "Chem.NO2", "Chem.O3", "Chem.j", "Chem.k", "Dep.vd"]),
            "pointwise reads",
        );
        for n in names {
            same(&o(n)["writes"], &json!(CHEM), n);
        }
        let f = facts(&[(st, false), (col, false)], &[(pw, &CHEM)]);
        let schedule = p.resolve_schedule(Some(&f)).unwrap();
        if case == "auto" {
            same(&p.step, &json!({"dt": 300, "hit": []}), "step");
            same(
                &p.partition,
                &json!({"halo": {"lat": 1, "lon": 3}, "ranks": [1, 1], "shards": {"root.1": 1, "root.2": 1}, "tiles": [1, 1]}),
                "partition",
            );
            same(
                &schedule,
                &json!({"id": "root", "sequence": [
                    {"alg": "SSPRK43", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root.0", "leaf": ["default_stencil"]},
                    {"alg": "Tsit5", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root.1", "leaf": ["default_column"]},
                    {"alg": "Rosenbrock23", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "cell", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root.2", "jacobian": "symbolic", "leaf": ["default_pointwise"], "linsolve": {"block_lu": {"pivot": true, "refine": 0}}}
                ]}),
                "schedule",
            );
            same(
                &o(pw)["stiffness"],
                &json!({"class": "stiff", "source": "reaction_systems"}),
                "pw",
            );
        } else {
            same(&p.step, &json!({"dt": 300, "hit": ["output"]}), "step");
            same(
                &p.partition,
                &json!({"halo": {"lat": 1, "lon": 3}, "ranks": [2, 1], "shards": {"chem": 4, "root.1": 1}, "tiles": [1, 3]}),
                "partition",
            );
            let f64s = json!({"factors": "f64", "jacobian": "f64", "rhs": "f64", "state": "f64", "tables": "f64"});
            same(
                &p.precision,
                &json!({"root.0": f64s, "root.1": f64s, "chem": {"factors": "f32", "jacobian": "f64", "rhs": "f64", "state": "f64", "tables": "f64"}}),
                "precision",
            );
            same(
                &schedule,
                &json!({"id": "root", "sequence": [
                    {"alg": "SSPRK43", "control": {"abstol": 0.000001, "controller": "PI", "dt0": 15, "dt_scope": "global", "maxiters": 1000000, "reltol": 0.0001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root.0", "leaf": ["transport"]},
                    {"alg": "Tsit5", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root.1", "leaf": ["mixing"]},
                    {"alg": "Rosenbrock23", "control": {"abstol": 1e-9, "controller": "PI", "dt0": 0.5, "dt_scope": "global", "maxiters": 1000000, "reltol": 0.0001}, "domain": {"check_stages": false, "nonnegative": "all", "on_violation": "reject"}, "id": "chem", "jacobian": "symbolic", "leaf": ["chemistry"], "linsolve": {"block_lu": {"pivot": false, "refine": 0}}}
                ]}),
                "schedule",
            );
            same(
                &o(st)["stiffness"],
                &json!({"class": "nonstiff", "source": "run_config"}),
                "transport",
            );
            same(
                &o(pw)["stiffness"],
                &json!({"class": "stiff", "source": "run_config"}),
                "chemistry",
            );
            assert!(
                o(col)["stiffness"].is_null(),
                "mixing's stiffness is the Gershgorin bound's"
            );
        }
    }
}

#[test]
fn toy_robertson() {
    let p = planned("toy_robertson", "rodas4");
    let want = [
        ("Kinetics.y1", 0, 0),
        ("Kinetics.y1", 0, 1),
        ("Kinetics.y2", 0, 0),
        ("Kinetics.y2", 0, 1),
        ("Kinetics.y2", 0, 2),
        ("Kinetics.y3", 0, 0),
    ];
    let got: Vec<_> = p
        .terms
        .iter()
        .map(|t| (t.variable.as_str(), t.contribution, t.summand))
        .collect();
    assert_eq!(got, want);
    let o = p.operator_json_stage1("kinetics");
    same(
        &o["reads"],
        &json!([
            "Kinetics.k1",
            "Kinetics.k2",
            "Kinetics.k3",
            "Kinetics.y1",
            "Kinetics.y2",
            "Kinetics.y3"
        ]),
        "reads",
    );
    same(&p.time_json(), &json!({"start": 0, "end": 40}), "time");
    same(
        &p.partition,
        &json!({"ranks": [1, 1], "shards": {"root": 1}, "tiles": [1, 1]}),
        "partition",
    );
    same(
        &p.inputs["model"]["path"],
        &json!("../../../toys/robertson.esm"),
        "path",
    );
    same(
        &p.resolve_schedule(None).unwrap(),
        &json!({"alg": "Rodas4", "control": {"abstol": 1e-10, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 0.000001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "jacobian": "symbolic", "leaf": ["kinetics"], "linsolve": {"dense_lu": {}}}),
        "schedule",
    );
}

#[test]
fn toy_advection_reaction() {
    let p = planned("toy_advection_reaction", "strang");
    let o = p.operator_json_stage1("transport");
    same(
        &o["locality"],
        &json!({"class": "stencil", "radius": {"cell": [1, 0]}}),
        "locality",
    );
    same(
        &o["reads"],
        &json!(["Reaction.A", "Transport.dx", "Transport.v"]),
        "reads",
    );
    same(
        &p.partition,
        &json!({"halo": {"cell": 1}, "ranks": [1, 1], "shards": {"root.1": 1}, "tiles": [1, 1]}),
        "partition",
    );
    same(&p.step, &json!({"dt": 0.05, "hit": []}), "step");
    let f = facts(&[], &[("reaction", &["Reaction.A", "Reaction.B"])]);
    same(
        &p.resolve_schedule(Some(&f)).unwrap(),
        &json!({"id": "root", "strang": [
            {"alg": "SSPRK43", "control": {"abstol": 1e-12, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 1e-10}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root.0", "leaf": ["transport"]},
            {"alg": "Rodas4", "control": {"abstol": 1e-12, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 1e-10}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root.1", "jacobian": "symbolic", "leaf": ["reaction"], "linsolve": {"block_lu": {"pivot": false, "refine": 0}}}
        ]}),
        "schedule",
    );
    assert_eq!(p.reference_steps["root.0"], 0.025);
    assert_eq!(p.reference_steps["root.1"], 0.05);
}

#[test]
fn toy_photostationary_cells() {
    let p = planned("toy_photostationary_cells", "kvaerno4");
    assert_eq!(p.terms.len(), 3);
    let f = facts(
        &[],
        &[("photochemistry", &["Photo.NO", "Photo.NO2", "Photo.O3"])],
    );
    same(
        &p.resolve_schedule(Some(&f)).unwrap(),
        &json!({"alg": "Kvaerno4", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 0.000001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "jacobian": "symbolic", "leaf": ["photochemistry"], "linsolve": {"block_lu": {"pivot": true, "refine": 0}}}),
        "schedule",
    );
    same(
        &p.partition,
        &json!({"halo": {"cell": 0}, "ranks": [1, 1], "shards": {"root": 1}, "tiles": [1, 1]}),
        "partition",
    );
}
