//! The M3 cases (spec/toy-problems.md §5; spec/partitioning.md §9): each
//! case is planned by the Rust planner once per `invariance` variant and
//! run, and the runs must be the same bit for bit; `scope_agreement`
//! compares a case with the one that differs only in `dt_scope`.
//! `cross_language` runs in CI on both implementations' result files.

use earthsci_integrator::conformance::Outcome;
use earthsci_integrator::conformance_m2::{write_result, Case, Check, RunResult};
use earthsci_integrator::conformance_m3::{
    bitwise_same, run_plan_with_options, scope_agreement, Variant,
};
use earthsci_integrator::forcing::DataOptions;
use earthsci_integrator_planner::astdiff::AstDiffBands;
use earthsci_integrator_planner::{plan, Options};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn cases() -> Vec<Value> {
    let v: Value = serde_json::from_str(
        &std::fs::read_to_string(repo().join("tests/conformance/m3/cases.json")).unwrap(),
    )
    .unwrap();
    v["cases"].as_array().unwrap().clone()
}

fn case_json(id: &str) -> Value {
    cases()
        .into_iter()
        .find(|c| c["id"] == id)
        .unwrap_or_else(|| panic!("no M3 case {id}"))
}

/// Plans case `c` with variant `v`'s partition into the target's directory
/// and runs it on the variant's threads.
fn run(c: &Value, v: Variant, tag: &str) -> RunResult {
    run_as(c, v, tag, false)
}

/// [`run`], with the whole-state oracle in place of lane programs when
/// `whole` is set.
fn run_as(c: &Value, v: Variant, tag: &str, whole: bool) -> RunResult {
    let id = c["id"].as_str().unwrap();
    let work = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("conformance_m3");
    std::fs::create_dir_all(&work).unwrap();
    let cases_dir = repo().join("tests/conformance/m3");
    let model = cases_dir
        .join(c["run_config"]["model"].as_str().unwrap())
        .canonicalize()
        .unwrap();
    let mut rc = v.run_config(&c["run_config"]);
    rc["model"] = json!(model.to_string_lossy());
    let rc_path = work.join(format!("{id}.{tag}.run.json"));
    std::fs::write(&rc_path, rc.to_string()).unwrap();
    let plan_dir = work.join(format!("{id}.{tag}"));
    let _ = std::fs::remove_dir_all(&plan_dir);
    let tspan = &c["tspan"];
    let options = Options {
        time: Some(json!({"start": tspan[0], "end": tspan[1]})),
    };
    plan(&rc_path, &options, &AstDiffBands)
        .and_then(|p| p.write(&plan_dir))
        .unwrap_or_else(|e| panic!("{id} ({tag}): planning: {e:?}"));
    let data = DataOptions {
        model: Some(model.clone()),
        cache_dir: Some(work.join("esio-cache")),
        ..Default::default()
    };
    let case: Case = serde_json::from_value(c.clone()).unwrap();
    run_plan_with_options(&case, &plan_dir, &data, v.threads.unwrap_or(1), whole)
        .unwrap_or_else(|e| panic!("{id} ({tag}): {e}"))
}

/// Runs every check of case `id` the Rust side runs on its own.
fn check_case(id: &str) {
    let c = case_json(id);
    let case: Case = serde_json::from_value(c.clone()).unwrap();
    let base = run(&c, Variant::default(), "base");
    // Result files for `cross_language`, beside M2's.
    let result_dir = std::env::var_os("ESI_RESULT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("results-m3"));
    let paths = earthsci_integrator::conformance::default_paths(&repo(), result_dir);
    write_result(&base, &paths).unwrap();
    let mut notes = Vec::new();
    for check in &case.checks {
        match check {
            Check::Invariance { variants } => {
                let runs: Vec<(Variant, RunResult)> = variants
                    .iter()
                    .enumerate()
                    .map(|(k, v)| {
                        let v = Variant::from_json(v);
                        (v, run(&c, v, &format!("v{k}")))
                    })
                    .collect();
                for (v, r) in &runs[1..] {
                    if let Err(e) = bitwise_same(&runs[0].1, r) {
                        panic!("{id}: invariance: {v:?} differs from {:?}: {e}", runs[0].0);
                    }
                }
                notes.push(format!("invariance over {} variants", runs.len()));
            }
            Check::ScopeAgreement { against, rtol } => {
                let other = run(&case_json(against), Variant::default(), "base");
                match scope_agreement(&base, &other, *rtol) {
                    Outcome::Pass(m) => notes.push(format!("scope_agreement with {against}: {m}")),
                    o => panic!("{id}: scope_agreement with {against}: {o:?}"),
                }
            }
            Check::CrossLanguage { .. } => {}
            other => panic!("{id}: unexpected check {other:?}"),
        }
    }
    eprintln!("{id}: {}", notes.join("; "));
}

#[test]
fn reseact_mini_global() {
    check_case("m3.reseact_mini.global");
}

#[test]
fn reseact_mini_cell() {
    check_case("m3.reseact_mini.cell");
}

#[test]
fn reseact_mini_global_trbdf2() {
    check_case("m3.reseact_mini.global_trbdf2");
}

#[test]
fn reseact_mini_cell_trbdf2() {
    check_case("m3.reseact_mini.cell_trbdf2");
}

#[test]
fn reseact_mini_shard() {
    check_case("m3.reseact_mini.shard");
}

#[test]
fn photostationary_cells_global() {
    check_case("m3.photostationary_cells.global");
}

#[test]
fn photostationary_cells_cell() {
    check_case("m3.photostationary_cells.cell");
}

/// spec/partitioning.md §3.1 rule 4, §9.3: lane programs give each cell the
/// whole-domain program's value bit for bit, so a `cell` run through them
/// equals the same run evaluating every operator over the whole model state.
/// The Rosenbrock case calls `rhs`, `jac` and `ft`; the shard case runs
/// several cells per lane call.
#[test]
fn lanes_equal_the_whole_domain_program() {
    for (id, v) in [
        ("m3.reseact_mini.cell", Variant::default()),
        (
            "m3.reseact_mini.shard",
            Variant {
                lanes: Some(3),
                ..Default::default()
            },
        ),
    ] {
        let c = case_json(id);
        let lanes = run_as(&c, v, "lanes", false);
        let whole = run_as(&c, v, "whole", true);
        if let Err(e) = bitwise_same(&lanes, &whole) {
            panic!("{id}: lane programs differ from the whole-domain program: {e}");
        }
    }
}
