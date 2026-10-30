//! The plan fixtures of `tests/conformance/plans/fixtures/` (spec/plan.md
//! §9). Every case is planned with EarthSciASTDiff's bands; its `plan.json`
//! must validate against `spec/schemas/plan.schema.json`, and what each
//! case's `expected.md` lists is checked here (for
//! `cross_component_term/by-label`, the whole hand-written
//! `expected-plan.json`). Where a case has a golden (`<case>.golden/`,
//! written by the Julia planner), `plan.json` and `derived/*.esm` must equal
//! it byte for byte, and the Rust band values must agree with its
//! `validation/<op>.bandvalues.json` (§8.2), a missing file being a harness
//! error.

mod common;

use common::{band_values_file, compare_band_values, conformance, HandBands};
use earthsci_integrator_planner::astdiff::{AstDiffBands, FlatView};
use earthsci_integrator_planner::numeric::{self, Dofs};
use earthsci_integrator_planner::{jcs, plan, validation, Options, Planned};
use serde_json::{json, Value};
use std::path::PathBuf;

const CASES: &[(&str, &str)] = &[
    ("column_contraction", "auto"),
    ("cross_component_term", "by-label"),
    ("cross_component_term", "by-owner"),
    ("periodic_stencil", "auto"),
    ("periodic_stencil", "partitioned"),
    ("pointwise_chemistry", "auto"),
    ("pointwise_chemistry", "named"),
    ("reseact_shaped", "auto"),
    ("reseact_shaped", "named"),
    ("toy_advection_reaction", "strang"),
    ("toy_photostationary_cells", "kvaerno4"),
    ("toy_robertson", "rodas4"),
];

fn run_config(fixture: &str, case: &str) -> PathBuf {
    conformance()
        .join("plans/fixtures")
        .join(fixture)
        .join(format!("{case}.run.json"))
}

fn planned(fixture: &str, case: &str) -> Planned {
    plan(
        &run_config(fixture, case),
        &Options::default(),
        &AstDiffBands,
    )
    .unwrap_or_else(|e| panic!("{fixture}/{case}: {e}"))
}

fn canon(v: &Value) -> String {
    jcs::to_string(v).unwrap()
}

fn same(got: &Value, want: &Value, what: &str) {
    assert_eq!(canon(got), canon(want), "{what}");
}

const F64S: &str =
    r#"{"factors": "f64", "jacobian": "f64", "rhs": "f64", "state": "f64", "tables": "f64"}"#;

fn f64s() -> Value {
    serde_json::from_str(F64S).unwrap()
}

/// `validation` for operators `ops` (already in code-point order).
fn validation(ops: &[&str]) -> Value {
    let mut v = vec![json!({"check": "rhs_sum", "passed": true})];
    for op in ops {
        v.push(json!({"check": "band_model", "operator": op, "passed": true}));
        v.push(json!({"check": "non_empty", "operator": op, "passed": true}));
    }
    Value::Array(v)
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

fn band_pairs(o: &Value) -> Vec<(String, String)> {
    o["structure"]["bands"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| {
            (
                b["row"].as_str().unwrap().to_string(),
                b["col"].as_str().unwrap().to_string(),
            )
        })
        .collect()
}

/// Resolves the schemas' `$ref`s to `spec/schemas/<file>` by file name.
struct SpecSchemas;

impl jsonschema::Retrieve for SpecSchemas {
    fn retrieve(
        &self,
        uri: &jsonschema::Uri<String>,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let name = uri.path().as_str().rsplit('/').next().unwrap_or_default();
        let path = conformance().join("../../spec/schemas").join(name);
        Ok(serde_json::from_slice(&std::fs::read(path)?)?)
    }
}

fn plan_schema() -> jsonschema::Validator {
    let path = conformance().join("../../spec/schemas/plan.schema.json");
    let schema: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    jsonschema::draft202012::options()
        .with_retriever(SpecSchemas)
        .build(&schema)
        .unwrap()
}

/// Every case plans; its plan validates; its derived documents load and
/// flatten with EarthSciAST; and where a golden exists, the plan equals it.
#[test]
fn every_case() {
    let schema = plan_schema();
    for (fixture, case) in CASES {
        let p = planned(fixture, case);
        let plan_json = p.plan_json();
        let errors: Vec<String> = schema
            .iter_errors(&plan_json)
            .map(|e| format!("{e} at {}", e.instance_path()))
            .collect();
        assert!(errors.is_empty(), "{fixture}/{case}: {errors:?}");
        // The written bytes are canonical.
        let bytes = p.plan_bytes().unwrap();
        assert_eq!(String::from_utf8(bytes.clone()).unwrap(), canon(&plan_json));
        // Each derived document loads and flattens, and the Jacobian
        // document has one band field per band.
        for (op, d) in &p.documents {
            FlatView::new(std::str::from_utf8(&d.rhs).unwrap())
                .unwrap_or_else(|e| panic!("{fixture}/{case} {op}.rhs: {e}"));
            let jac = FlatView::new(std::str::from_utf8(&d.jacobian).unwrap())
                .unwrap_or_else(|e| panic!("{fixture}/{case} {op}.jacobian: {e}"));
            let fields = jac
                .flat
                .state_variables
                .keys()
                .filter(|k| k.starts_with("_jacobian.J_"))
                .count();
            assert_eq!(
                fields,
                p.structures[op].bands.len(),
                "{fixture}/{case} {op}"
            );
        }
        check_golden(fixture, case, &p, &bytes);
    }
}

/// The golden of a case, when the Julia planner has written one (§9).
fn check_golden(fixture: &str, case: &str, p: &Planned, plan_bytes: &[u8]) {
    let golden = conformance()
        .join("plans/fixtures")
        .join(fixture)
        .join(format!("{case}.golden"));
    if !golden.is_dir() {
        return;
    }
    let want = std::fs::read(golden.join("plan.json")).unwrap();
    assert!(
        want == plan_bytes,
        "{fixture}/{case}: plan.json differs from the golden"
    );
    let mut files: Vec<String> = std::fs::read_dir(golden.join("derived"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    files.sort();
    let mut ours: Vec<String> = Vec::new();
    for (op, d) in &p.documents {
        for (kind, bytes) in [("jacobian", &d.jacobian), ("rhs", &d.rhs)] {
            let name = format!("{op}.{kind}.esm");
            let want = std::fs::read(golden.join("derived").join(&name))
                .unwrap_or_else(|e| panic!("{fixture}/{case}: golden derived/{name}: {e}"));
            assert!(
                &want == bytes,
                "{fixture}/{case}: derived/{name} differs from the golden"
            );
            ours.push(name);
        }
    }
    ours.sort();
    assert_eq!(files, ours, "{fixture}/{case}: derived documents");
    // The band values: a missing file is a harness error (§8.2).
    for op in p.operators.keys() {
        let path = golden
            .join("validation")
            .join(format!("{op}.bandvalues.json"));
        let text = std::fs::read(&path).unwrap_or_else(|e| {
            panic!(
                "harness error: {fixture}/{case} has a golden but no {}: {e}",
                path.display()
            )
        });
        let file: Value = serde_json::from_slice(&text).unwrap();
        compare_band_values(p, op, &file)
            .unwrap_or_else(|e| panic!("{fixture}/{case} {op} band values: {e}"));
    }
}

/// The comparison of §8.2 on files written from Rust's own values: equal
/// values pass, a changed value or a missing entry fails.
#[test]
fn band_value_comparison() {
    for (fixture, case) in [
        ("pointwise_chemistry", "auto"),
        ("periodic_stencil", "auto"),
    ] {
        let p = planned(fixture, case);
        let dofs = Dofs::new(&p.model);
        let u0 = numeric::initial_state(&p.model, &dofs, p.interval.start).unwrap();
        let states: Vec<(f64, Vec<f64>)> = numeric::samples(&u0, p.interval.start, p.interval.end)
            .into_iter()
            .map(|s| (s.t, s.u))
            .collect();
        let op = p.operators.keys().next().unwrap().clone();
        let file = band_values_file(&p, &op, &states);
        compare_band_values(&p, &op, &file).unwrap();
        let mut changed = file.clone();
        let v = changed["samples"][1]["entries"][0][2].as_f64().unwrap();
        changed["samples"][1]["entries"][0][2] = json!(v * (1.0 + 1e-6) + 1e-12);
        assert!(compare_band_values(&p, &op, &changed).is_err());
        let mut missing = file.clone();
        missing["samples"][0]["entries"]
            .as_array_mut()
            .unwrap()
            .pop();
        assert!(compare_band_values(&p, &op, &missing).is_err());
    }
}

/// `band_model` (§8.2, Rust) passes the planner's own bands and catches a
/// wrong band value and a band missing from the pattern.
#[test]
fn band_model_catches_errors() {
    for (fixture, case) in [
        ("pointwise_chemistry", "auto"),
        ("column_contraction", "auto"),
    ] {
        let p = planned(fixture, case);
        let (op, docs) = p.documents.iter().next().unwrap();
        let tspan = (p.interval.start, p.interval.end);
        let rhs: Value = serde_json::from_slice(&docs.rhs).unwrap();
        let jac: Value = serde_json::from_slice(&docs.jacobian).unwrap();
        let f = numeric::Program::new(&rhs, tspan).unwrap();
        let jp = numeric::Program::new(&jac, tspan).unwrap();
        let dofs = Dofs::new(&p.model);
        let names = dofs.element_names();
        let u0 = numeric::initial_state(&p.model, &dofs, p.interval.start).unwrap();
        let base = numeric::sampling_base(&u0);
        let s = &numeric::samples(&u0, tspan.0, tspan.1)[1];
        let bands = &p.structures[op].bands;
        let jb = numeric::band_jacobian(&jp, &dofs, bands, &s.u, s.t).unwrap();
        let pattern = numeric::pattern(&dofs, bands).unwrap();
        let rows = vec![true; dofs.n];
        let check = |jb: &numeric::Sparse, pattern: &numeric::Sparse| {
            let at = validation::At {
                dofs: &dofs,
                sample: s,
                base: &base,
                names: &names,
            };
            validation::band_model(&f, &at, jb, pattern, &rows)
        };
        check(&jb, &pattern).unwrap();
        // A wrong value.
        let mut wrong = jb.clone();
        let (k, v) = wrong.iter().next().map(|(k, v)| (*k, *v)).unwrap();
        wrong.insert(k, v * 1.01 + 1e-3);
        assert!(check(&wrong, &pattern).is_err(), "{fixture}: wrong value");
        // An off-diagonal entry missing from the bands and the pattern.
        let (k, _) = jb.iter().find(|((r, c), v)| r != c && **v != 0.0).unwrap();
        let (mut jb2, mut pattern2) = (jb.clone(), pattern.clone());
        jb2.remove(k);
        pattern2.remove(k);
        assert!(check(&jb2, &pattern2).is_err(), "{fixture}: missing band");
    }
}

/// The hand-written bands of stage 1 and EarthSciASTDiff's give every term
/// the same class.
#[test]
fn hand_bands_agree() {
    for (fixture, case) in CASES {
        let ours = planned(fixture, case);
        let hand = plan(&run_config(fixture, case), &Options::default(), &HandBands).unwrap();
        assert_eq!(
            locality_classes(&ours),
            locality_classes(&hand),
            "{fixture}/{case}"
        );
        for (name, op) in &ours.operators {
            assert_eq!(
                op.locality, hand.operators[name].locality,
                "{fixture}/{case} {name}"
            );
        }
    }
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
    let mut got = p.plan_json();
    for d in got["derived"].as_object_mut().unwrap().values_mut() {
        d["sha256"] = json!("0".repeat(64));
    }
    same(&got, &want, "plan.json");
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
            &p.operator_json(op)["stiffness"],
            &json!({"class": "stiff", "source": "run_config"}),
            op,
        );
    }
    for (op, var) in [("deposition", "Chem.NO"), ("scavenging", "Chem.O3")] {
        let o = p.operator_json(op);
        assert_eq!(band_pairs(&o), [(var.to_string(), var.to_string())]);
        same(
            &o["structure"]["block"],
            &json!({"vars": [var], "ordering": [1], "pattern": [[1, 1]], "lu_pattern": [[1, 1]]}),
            op,
        );
    }
    // The run config writes a one-child `sequence`, which §7.1 keeps, so the
    // leaf is `root.0` (expected.md).
    same(
        &p.resolved_schedule,
        &json!({"id": "root", "sequence": [{"alg": "Rodas4", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "cell", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root.0", "jacobian": "symbolic", "leaf": ["chemistry", "deposition", "scavenging"], "linsolve": {"block_lu": {"pivot": true, "refine": 0}}}]}),
        "schedule",
    );
    same(&p.partition["shards"], &json!({"root.0": 1}), "shards");
    same(
        &p.plan_json()["validation"],
        &validation(&["chemistry", "deposition", "scavenging"]),
        "validation",
    );
}

const CHEM: [&str; 3] = ["Chem.NO", "Chem.NO2", "Chem.O3"];

/// The nine (row, column) pairs of a 3 × 3 species block, as
/// EarthSciASTDiff orders them: equation order (NO2, NO, O3, as flattened),
/// then sorted column.
fn chem_pairs() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for r in ["Chem.NO2", "Chem.NO", "Chem.O3"] {
        for c in CHEM {
            out.push((r.to_string(), c.to_string()));
        }
    }
    out
}

fn full_block(vars: &[&str]) -> Value {
    let n = vars.len();
    let pairs: Vec<Value> = (1..=n)
        .flat_map(|i| (1..=n).map(move |j| json!([i, j])))
        .collect();
    json!({"vars": vars, "ordering": (1..=n).collect::<Vec<_>>(), "pattern": pairs, "lu_pattern": pairs})
}

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
        let o = p.operator_json(op);
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
        let s = &o["structure"];
        assert_eq!(s["class"], "block_diagonal");
        assert_eq!(s["linear_in_u"], false);
        assert_eq!(s["depends_on_t"], false);
        assert_eq!(band_pairs(&o), chem_pairs());
        for b in s["bands"].as_array().unwrap() {
            same(&b["rows"], &json!([[1, 4]]), "rows");
            same(&b["row_idx"], &json!(["i"]), "row_idx");
            same(&b["col_idx"], &json!(["i"]), "col_idx");
        }
        same(&s["block"], &full_block(&CHEM), "block");
        same(&p.time_json(), &json!({"start": 0, "end": 3600}), "time");
        same(&p.step, &json!({"hit": []}), "step");
        let shards = if case == "auto" { 1 } else { 2 };
        same(
            &p.partition,
            &json!({"halo": {"cell": 0}, "ranks": [1, 1], "shards": {"root": shards}, "tiles": [1, 1]}),
            "partition",
        );
        same(&p.precision, &json!({"root": f64s()}), "precision");
        let schedule = if case == "auto" {
            json!({"alg": "Rosenbrock23", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "cell", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "jacobian": "symbolic", "leaf": ["default_pointwise"], "linsolve": {"block_lu": {"pivot": true, "refine": 0}}})
        } else {
            json!({"alg": "Rodas4", "control": {"abstol": 1e-9, "controller": "PI", "dt_scope": "cell", "maxiters": 1000000, "reltol": 0.000001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "jacobian": "symbolic", "leaf": ["chemistry"], "linsolve": {"block_lu": {"pivot": true, "refine": 0}}})
        };
        same(&p.resolved_schedule, &schedule, case);
        same(
            &p.plan_json()["validation"],
            &validation(&[op]),
            "validation",
        );
        // The right-hand-side document: models Chem and Dep, the lifted
        // species with shape ["cell"], each equation a `faq` over `cell`.
        let rhs: Value = serde_json::from_slice(&p.documents[op].rhs).unwrap();
        let models: Vec<&String> = rhs["models"].as_object().unwrap().keys().collect();
        assert_eq!(models, ["Chem", "Dep"]);
        let chem = &rhs["models"]["Chem"];
        same(
            &chem["variables"]["NO2"],
            &json!({"type": "unknown", "shape": ["cell"], "default": 10.5}),
            "NO2",
        );
        let eqs = chem["equations"].as_array().unwrap();
        assert_eq!(eqs.len(), 3);
        for eq in eqs {
            same(
                &eq["rhs"]["ranges"],
                &json!({"i": {"from": "cell"}}),
                "ranges",
            );
            same(&eq["rhs"]["output_idx"], &json!(["i"]), "output_idx");
        }
        let jac: Value = serde_json::from_slice(&p.documents[op].jacobian).unwrap();
        let fields = jac["models"]["_jacobian"]["variables"].as_object().unwrap();
        assert_eq!(fields.len(), 9);
        for f in fields.values() {
            same(
                f,
                &json!({"type": "unknown", "default": 0.0, "shape": ["cell"]}),
                "J_k",
            );
        }
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
        let o = p.operator_json(op);
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
        let s = &o["structure"];
        assert_eq!(s["class"], "general");
        assert_eq!(s["linear_in_u"], true);
        assert_eq!(s["depends_on_t"], false);
        assert_eq!(s["bands"].as_array().unwrap().len(), 36);
        same(
            &s["stencil"],
            &json!({"footprint": [[-3, 0], [-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0]]}),
            "footprint",
        );
        assert!(s.get("block").is_none() && s.get("column").is_none());
        // Six interior bands over lon 4–6, six per seam row with a literal
        // first column index.
        let bands = s["bands"].as_array().unwrap();
        let interior = bands
            .iter()
            .filter(|b| b["rows"] == json!([[4, 6], [1, 2], [1, 2]]))
            .count();
        assert_eq!(interior, 6);
        for r in [1, 2, 3, 7, 8] {
            let seam: Vec<&Value> = bands
                .iter()
                .filter(|b| b["rows"] == json!([[r, r], [1, 2], [1, 2]]))
                .collect();
            assert_eq!(seam.len(), 6, "seam row {r}");
            assert!(seam.iter().all(|b| b["col_idx"][0].is_i64()));
        }
        same(&p.time_json(), &json!({"start": 0, "end": 0.5}), "time");
        let (grid, step, schedule, stiffness) = if case == "auto" {
            (
                [1, 1],
                json!({"hit": []}),
                json!({"alg": "SSPRK43", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "leaf": ["default_stencil"]}),
                json!({"class": "nonstiff", "source": "gershgorin"}),
            )
        } else {
            (
                [2, 1],
                json!({"dt": 0.1, "hit": []}),
                json!({"alg": "SSPRK33", "control": {"abstol": 0.000001, "dt_scope": "global", "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "dt": 0.01, "id": "root", "leaf": ["transport"]}),
                json!({"class": "nonstiff", "source": "run_config"}),
            )
        };
        same(&o["stiffness"], &stiffness, "stiffness");
        same(&p.step, &step, "step");
        same(
            &p.partition,
            &json!({"halo": {"lat": 0, "lon": 3}, "ranks": grid, "shards": {}, "tiles": grid}),
            "partition",
        );
        same(&p.resolved_schedule, &schedule, case);
        same(&p.precision, &json!({"root": f64s()}), "precision");
        assert_eq!(
            p.reference_steps["root"],
            if case == "auto" { 0.5 } else { 0.01 }
        );
        // ρ = (c/dx) Σ|w| = 1.375 (expected.md).
        if case == "auto" {
            let rho = &p.report["validation"]["gershgorin"][op]["rho"];
            for r in rho.as_array().unwrap() {
                assert!((r.as_f64().unwrap() - 1.375).abs() < 1e-12);
            }
        }
        let jac: Value = serde_json::from_slice(&p.documents[op].jacobian).unwrap();
        assert_eq!(
            jac["models"]["_jacobian"]["variables"]
                .as_object()
                .unwrap()
                .len(),
            36
        );
        let rhs: Value = serde_json::from_slice(&p.documents[op].rhs).unwrap();
        let eqs = rhs["models"]["Transport"]["equations"].as_array().unwrap();
        assert_eq!(eqs.len(), 2);
        assert_eq!(eqs[1]["lhs"], json!({"op": "ic", "args": ["Transport.q"]}));
    }
}

#[test]
fn column_contraction() {
    let p = planned("column_contraction", "auto");
    let o = p.operator_json("default_column");
    same(&o["locality"], &json!({"class": "column"}), "locality");
    same(&o["reads"], &json!(["Mix.kmix", "Mix.q"]), "reads");
    let s = &o["structure"];
    assert_eq!(s["class"], "general");
    assert_eq!(s["linear_in_u"], true);
    assert_eq!(s["depends_on_t"], false);
    same(
        &s["column"],
        &json!({"axis": "lev", "kl": 4, "ku": 4}),
        "column",
    );
    let bands = s["bands"].as_array().unwrap();
    assert_eq!(bands.len(), 2);
    same(&bands[0]["col_idx"], &json!(["i", "j", "k"]), "diagonal");
    let contracted = bands[1]["contracted"].as_array().unwrap();
    assert_eq!(contracted.len(), 1);
    assert_eq!(
        (&contracted[0][1], &contracted[0][2]),
        (&json!(1), &json!(5))
    );
    assert_eq!(bands[1]["col_idx"][2], contracted[0][0]);
    same(
        &o["stiffness"],
        &json!({"class": "stiff", "source": "gershgorin"}),
        "stiffness",
    );
    same(&p.step, &json!({"dt": 600, "hit": []}), "step");
    same(
        &p.partition,
        &json!({"halo": {"lat": 0, "lon": 0}, "ranks": [1, 1], "shards": {"root": 1}, "tiles": [1, 1]}),
        "partition",
    );
    assert_eq!(p.reference_steps["root"], 600.0);
    same(
        &p.resolved_schedule,
        &json!({"alg": "TRBDF2", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "jacobian": "symbolic", "leaf": ["default_column"], "linsolve": {"banded": {}}}),
        "schedule",
    );
    // ρ = 0.32 (expected.md).
    let rho = &p.report["validation"]["gershgorin"]["default_column"]["rho"];
    assert!((rho[0].as_f64().unwrap() - 0.32).abs() < 1e-12);
    // The contracted band field has the row's index sets and one of extent 5.
    let jac: Value = serde_json::from_slice(&p.documents["default_column"].jacobian).unwrap();
    let shape = jac["models"]["_jacobian"]["variables"]["J_2"]["shape"].clone();
    assert_eq!(
        &shape.as_array().unwrap()[..3],
        &[json!("lon"), json!("lat"), json!("lev")]
    );
    let extra = shape[3].as_str().unwrap();
    assert_eq!(
        jac["index_sets"][extra],
        json!({"kind": "interval", "size": 5})
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
        let o = |n| p.operator_json(n);
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
            &o(st)["structure"]["stencil"],
            &json!({"footprint": [[-3, 0], [-2, 0], [-1, 0], [0, -1], [0, 0], [1, 0], [2, 0]]}),
            "footprint",
        );
        assert_eq!(o(st)["structure"]["class"], "general");
        assert_eq!(o(st)["structure"]["linear_in_u"], true);
        same(
            &o(col)["reads"],
            &json!(["Chem.NO", "Chem.NO2", "Chem.O3", "Mix.kmix"]),
            "column reads",
        );
        same(
            &o(col)["structure"]["column"],
            &json!({"axis": "lev", "kl": 3, "ku": 3}),
            "column",
        );
        assert_eq!(o(col)["structure"]["class"], "general");
        same(
            &o(col)["stiffness"],
            &json!({"class": "nonstiff", "source": "gershgorin"}),
            "column stiffness",
        );
        same(
            &o(pw)["reads"],
            &json!(["Chem.NO", "Chem.NO2", "Chem.O3", "Chem.j", "Chem.k", "Dep.vd"]),
            "pointwise reads",
        );
        assert_eq!(o(pw)["structure"]["class"], "block_diagonal");
        assert_eq!(o(pw)["structure"]["linear_in_u"], false);
        same(&o(pw)["structure"]["block"], &full_block(&CHEM), "block");
        for n in names {
            same(&o(n)["writes"], &json!(CHEM), n);
        }
        let mut ops = names.to_vec();
        ops.sort();
        same(
            &p.plan_json()["validation"],
            &validation(&ops),
            "validation",
        );
        if case == "auto" {
            same(&p.step, &json!({"dt": 300, "hit": []}), "step");
            same(
                &p.partition,
                &json!({"halo": {"lat": 1, "lon": 3}, "ranks": [1, 1], "shards": {"root.1": 1, "root.2": 1}, "tiles": [1, 1]}),
                "partition",
            );
            same(
                &p.resolved_schedule,
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
            same(
                &o(st)["stiffness"],
                &json!({"class": "nonstiff", "source": "gershgorin"}),
                "stencil",
            );
            same(
                &p.precision,
                &json!({"root.0": f64s(), "root.1": f64s(), "root.2": f64s()}),
                "precision",
            );
        } else {
            same(&p.step, &json!({"dt": 300, "hit": ["output"]}), "step");
            same(
                &p.partition,
                &json!({"halo": {"lat": 1, "lon": 3}, "ranks": [2, 1], "shards": {"chem": 4, "root.1": 1}, "tiles": [1, 3]}),
                "partition",
            );
            same(
                &p.precision,
                &json!({"root.0": f64s(), "root.1": f64s(), "chem": {"factors": "f32", "jacobian": "f64", "rhs": "f64", "state": "f64", "tables": "f64"}}),
                "precision",
            );
            same(
                &p.resolved_schedule,
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
        }
        // Each right-hand-side document declares Chem, Dep, Mix, Transport.
        for n in names {
            let rhs: Value = serde_json::from_slice(&p.documents[n].rhs).unwrap();
            let models: Vec<&String> = rhs["models"].as_object().unwrap().keys().collect();
            assert_eq!(models, ["Chem", "Dep", "Mix", "Transport"], "{n}");
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
    let o = p.operator_json("kinetics");
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
    let pairs: Vec<(String, String)> = [
        ("y1", "y1"),
        ("y1", "y2"),
        ("y1", "y3"),
        ("y2", "y1"),
        ("y2", "y2"),
        ("y2", "y3"),
        ("y3", "y2"),
    ]
    .iter()
    .map(|(r, c)| (format!("Kinetics.{r}"), format!("Kinetics.{c}")))
    .collect();
    assert_eq!(band_pairs(&o), pairs);
    let pattern = json!([
        [1, 1],
        [1, 2],
        [1, 3],
        [2, 1],
        [2, 2],
        [2, 3],
        [3, 2],
        [3, 3]
    ]);
    same(
        &o["structure"]["block"],
        &json!({"vars": ["Kinetics.y1", "Kinetics.y2", "Kinetics.y3"], "ordering": [1, 2, 3], "pattern": pattern, "lu_pattern": pattern}),
        "block",
    );
    assert_eq!(o["structure"]["linear_in_u"], false);
    assert_eq!(o["structure"]["depends_on_t"], false);
    same(&p.time_json(), &json!({"start": 0, "end": 40}), "time");
    same(&p.step, &json!({"dt": 40, "hit": []}), "step");
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
        &p.resolved_schedule,
        &json!({"alg": "Rodas4", "control": {"abstol": 1e-10, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 0.000001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "jacobian": "symbolic", "leaf": ["kinetics"], "linsolve": {"dense_lu": {}}}),
        "schedule",
    );
}

#[test]
fn toy_advection_reaction() {
    let p = planned("toy_advection_reaction", "strang");
    let o = p.operator_json("transport");
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
    let s = &o["structure"];
    assert_eq!(s["class"], "general");
    assert_eq!(s["bands"].as_array().unwrap().len(), 4);
    same(
        &s["stencil"],
        &json!({"footprint": [[-1], [0]]}),
        "footprint",
    );
    assert_eq!(s["linear_in_u"], true);
    same(
        &o["stiffness"],
        &json!({"class": "nonstiff", "source": "run_config"}),
        "stiffness",
    );
    let r = p.operator_json("reaction");
    same(
        &r["structure"]["block"],
        &full_block(&["Reaction.A", "Reaction.B"]),
        "block",
    );
    assert_eq!(r["structure"]["linear_in_u"], true);
    same(
        &p.partition,
        &json!({"halo": {"cell": 1}, "ranks": [1, 1], "shards": {"root.1": 1}, "tiles": [1, 1]}),
        "partition",
    );
    same(&p.step, &json!({"dt": 0.05, "hit": []}), "step");
    same(
        &p.resolved_schedule,
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
    let o = p.operator_json("photochemistry");
    let s = &o["structure"];
    assert_eq!(s["class"], "block_diagonal");
    assert_eq!(s["depends_on_t"], true);
    assert_eq!(s["linear_in_u"], false);
    same(
        &s["block"],
        &full_block(&["Photo.NO", "Photo.NO2", "Photo.O3"]),
        "block",
    );
    same(
        &p.resolved_schedule,
        &json!({"alg": "Kvaerno4", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 0.000001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "jacobian": "symbolic", "leaf": ["photochemistry"], "linsolve": {"block_lu": {"pivot": true, "refine": 0}}}),
        "schedule",
    );
    same(
        &p.partition,
        &json!({"halo": {"cell": 0}, "ranks": [1, 1], "shards": {"root": 1}, "tiles": [1, 1]}),
        "partition",
    );
}
