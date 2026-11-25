//! The M3 cases (spec/toy-problems.md §5; spec/partitioning.md §9): each
//! case is planned by the Rust planner once per `invariance` variant and
//! run, and the runs must be the same bit for bit; `scope_agreement`
//! compares a case with the one that differs only in `dt_scope`.
//! `cross_language` runs in CI on both implementations' result files.

use earthsci_integrator::conformance::Outcome;
use earthsci_integrator::conformance_m2::{write_result, Case, Check, RunResult};
use earthsci_integrator::conformance_m3::{
    bitwise_same, resume_and_concurrent_runs, run_plan_configured, scope_agreement, Variant,
};
use earthsci_integrator::forcing::DataOptions;
use earthsci_integrator::schedule::RunSpec;
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
    run_configured(c, v, tag, whole, &|_| {})
}

/// [`run_as`], with `configure` applied to the run before it starts.
fn run_configured(
    c: &Value,
    v: Variant,
    tag: &str,
    whole: bool,
    configure: &dyn Fn(&RunSpec<f64>),
) -> RunResult {
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
    run_plan_configured(
        &case,
        &plan_dir,
        &data,
        v.threads.unwrap_or(1),
        whole,
        configure,
    )
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

/// spec/partitioning.md §3.2, U19: under `dt_scope: cell` a lane pool
/// serves cells at different times in one lane-program call.
#[test]
fn a_lane_pool_batches_cells_at_different_times() {
    let c = case_json("m3.reseact_mini.cell");
    let _ = run(
        &c,
        Variant {
            lanes: Some(8),
            ..Default::default()
        },
        "pooled",
    );
    let mixed =
        earthsci_integrator::lanes::MIXED_TIME_LANES.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        mixed > 1,
        "the pool never served cells at different times together ({mixed})"
    );
}

/// spec/partitioning.md §3.2, §4: a panic in a lane of a lane pool, outside
/// the pool's evaluations or while serving one, fails the run (a panic or an
/// error) instead of leaving the pool's other lanes waiting for it.
#[test]
fn a_panicking_lane_fails_the_run_instead_of_hanging() {
    for in_serve in [false, true] {
        let c = case_json("m3.reseact_mini.cell");
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let r = std::panic::catch_unwind(|| {
                run_configured(
                    &c,
                    Variant {
                        lanes: Some(4),
                        threads: Some(2),
                        ..Default::default()
                    },
                    if in_serve {
                        "panic-serve"
                    } else {
                        "panic-lane"
                    },
                    false,
                    &|s| s.set_test_panic(Some(1), in_serve),
                )
            });
            let _ = tx.send(r.is_err());
        });
        match rx.recv_timeout(std::time::Duration::from_secs(600)) {
            Ok(failed) => assert!(failed, "the run with a panicking lane succeeded"),
            Err(_) => panic!("a panicking lane (in_serve = {in_serve}) left the run hanging"),
        }
    }
}

/// spec/partitioning.md §6.2, §11: the re-sharding signal is the run's own
/// persistent state. A run resumed from any checkpoint re-shards and steps
/// as the uninterrupted run, concurrent runs of one RunSpec do not mix, and
/// a shard's tape record lists its cells only on its first record or after
/// its composition changed.
#[test]
fn resharding_resumes_and_does_not_leak_between_runs() {
    let c = case_json("m3.reseact_mini.shard");
    let _ = run(&c, Variant::default(), "resume");
    let plan_dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("conformance_m3")
        .join("m3.reseact_mini.shard.resume");
    let data = DataOptions {
        model: Some(
            repo()
                .join("tests/conformance/m3")
                .join(c["run_config"]["model"].as_str().unwrap())
                .canonicalize()
                .unwrap(),
        ),
        cache_dir: Some(
            PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
                .join("conformance_m3")
                .join("esio-cache"),
        ),
        ..Default::default()
    };
    let case: Case = serde_json::from_value(c.clone()).unwrap();
    let relisted = resume_and_concurrent_runs(&case, &plan_dir, &data)
        .unwrap_or_else(|e| panic!("m3.reseact_mini.shard: {e}"));
    eprintln!("m3.reseact_mini.shard: {relisted} shard records list re-sharded cells");
}
