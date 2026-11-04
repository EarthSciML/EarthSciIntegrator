//! Runs `tests/conformance/m2/cases.json` (spec/toy-problems.md §4): each
//! case is planned by the Rust planner, its operators are built from the
//! plan (spec/operator-interface.md §8), its schedule is run, and the result
//! must pass `toy_equivalence` against this implementation's run of the M1
//! case on the hand-written problem, or, for the data-input cases (§4.4),
//! `reference` against `m2/reference/<case>.json`; `cross_language` is CI's.
//!
//! The planner reads the case's run config from a file, with `model`
//! relative to it, so each run config is written to
//! `<work>/cases/<id>.run.json` beside a copy of `tests/conformance/toys/`
//! at `<work>/toys/`; its plan directory is `<work>/plans/<id>/`. `<work>`
//! is `$EARTHSCI_M2_WORK_DIR` (default `rust/target/tmp/conformance/m2`).
//! Result files for both runs (the M2 case's and its M1 case's, with
//! tapes) go to `$EARTHSCI_M2_RESULTS_DIR` (default `<work>/rust`).
//! `$EARTHSCI_M2_CASES` limits the run to the case ids it lists, comma
//! separated.

use earthsci_integrator::conformance::{self, default_paths, Outcome};
use earthsci_integrator::conformance_m2::{
    check_reference, run_m1_forward, run_plan_with, toy_equivalence, write_result, CasesFile, Check,
};
use earthsci_integrator::forcing::DataOptions;
use earthsci_integrator_planner::astdiff::AstDiffBands;
use earthsci_integrator_planner::{plan, Options};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Copies `from` into `to`, subdirectories (the data-input toys' data)
/// included.
fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let ty = e.file_type().unwrap();
        if ty.is_file() {
            std::fs::copy(e.path(), to.join(e.file_name())).unwrap();
        } else if ty.is_dir() {
            copy_dir(&e.path(), &to.join(e.file_name()));
        }
    }
}

/// M2 cases that fail for a cause outside the Rust implementation, with
/// the cause.
const KNOWN_FAILURES: &[(&str, &str)] = &[
    (
        "m2.data_points.tsit5",
        "stations.csv leaves the filtered rates blank; EarthSciIO's csv reader reads a column \
         with a blank cell as text (a blank numeric cell is an error, never NaN, in every \
         language), and EarthSciAST refuses a text forcing",
    ),
    (
        "m2.data_monthly.tsit5",
        "operator-interface.md §8.4.1 gives a stage at exactly an epoch boundary the new \
         epoch, which keeps a two-record time blend continuous but puts the next month's \
         rate into the c = 1 stages of the step that ends on a month boundary of this \
         piecewise-constant forcing (2.3e-8 against rtol 1e-10)",
    ),
];

#[test]
fn m2_conformance_cases() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cases: CasesFile = serde_json::from_str(
        &std::fs::read_to_string(repo.join("tests/conformance/m2/cases.json")).unwrap(),
    )
    .expect("m2/cases.json parses");
    assert_eq!(cases.esm_integrator, earthsci_integrator::SPEC_VERSION);
    let m1: conformance::CasesFile = serde_json::from_str(
        &std::fs::read_to_string(repo.join("tests/conformance/m1/cases.json")).unwrap(),
    )
    .expect("m1/cases.json parses");

    let work = std::env::var_os("EARTHSCI_M2_WORK_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_TARGET_TMPDIR")).join("conformance/m2"));
    let result_dir = std::env::var_os("EARTHSCI_M2_RESULTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| work.join("rust"));
    let paths = default_paths(&repo, result_dir.clone());
    copy_dir(&repo.join("tests/conformance/toys"), &work.join("toys"));
    std::fs::create_dir_all(work.join("cases")).unwrap();
    let only: Option<Vec<String>> = std::env::var("EARTHSCI_M2_CASES")
        .ok()
        .map(|s| s.split(',').map(str::to_string).collect());
    let selected: Vec<_> = cases
        .cases
        .iter()
        .filter(|c| only.as_ref().is_none_or(|o| o.contains(&c.id)))
        .collect();

    // One case: plan, build, run, check. Returns its report line and
    // whether it passed.
    let run_one = |case: &earthsci_integrator::conformance_m2::Case| -> (String, bool) {
        let start = Instant::now();
        let fail = |m: String| (format!("FAIL  {}: {m}\n", case.id), false);
        let rc_path = work.join("cases").join(format!("{}.run.json", case.id));
        std::fs::write(
            &rc_path,
            serde_json::to_vec_pretty(&case.run_config).unwrap(),
        )
        .unwrap();
        let plan_dir = work.join("plans").join(&case.id);
        let _ = std::fs::remove_dir_all(&plan_dir);
        let options = Options {
            time: Some(json!({"start": case.tspan[0], "end": case.tspan[1]})),
        };
        if let Err(e) = plan(&rc_path, &options, &AstDiffBands).and_then(|p| p.write(&plan_dir)) {
            return fail(format!("planning: {e}"));
        }
        let planned = start.elapsed();
        // The plan's model, for a plan with forcing (operator-interface.md
        // §8.4.2): the run config names it relative to its own directory.
        let data = DataOptions {
            model: case.run_config["model"]
                .as_str()
                .map(|m| work.join("cases").join(m)),
            cache_dir: Some(work.join("esio-cache")),
            ..Default::default()
        };
        let m2 = match run_plan_with(case, &plan_dir, &data) {
            Ok(r) => r,
            Err(e) => return fail(format!("running the plan: {e}")),
        };
        if let Err(e) = write_result(&m2, &paths) {
            return fail(format!("result file: {e}"));
        }
        if case.checks.is_empty() {
            return fail("the case has no checks".into());
        }
        // Every check must pass; each names the M1 case it compares with.
        let mut notes = Vec::new();
        let mut ok = true;
        for check in &case.checks {
            let (against, rtol, rtol_steps_differ) = match check {
                Check::ToyEquivalence {
                    against,
                    rtol,
                    rtol_steps_differ,
                } => (against, rtol, rtol_steps_differ),
                Check::Reference { rtol } => {
                    let path = repo
                        .join("tests/conformance/m2/reference")
                        .join(format!("{}.json", case.id));
                    match check_reference(&m2, &path, *rtol) {
                        Outcome::Pass(m) => notes.push(format!("reference: {m}")),
                        Outcome::Fail(m) | Outcome::Skip(m) => {
                            ok = false;
                            notes.push(format!("reference FAILED: {m}"));
                        }
                    }
                    continue;
                }
                // Run in CI on both implementations' result files.
                Check::CrossLanguage { .. } => continue,
            };
            let Some(m1_case) = m1.cases.iter().find(|c| &c.id == against) else {
                return fail(format!("no M1 case {against}"));
            };
            let m1r = match run_m1_forward(m1_case) {
                Ok(r) => r,
                Err(e) => return fail(format!("M1 case {against}: {e}")),
            };
            if let Err(e) = write_result(&m1r, &paths) {
                return fail(format!("M1 result file: {e}"));
            }
            match toy_equivalence(&m2, &m1r, *rtol, *rtol_steps_differ) {
                Outcome::Pass(m) => notes.push(format!("toy_equivalence against {against}: {m}")),
                Outcome::Fail(m) | Outcome::Skip(m) => {
                    ok = false;
                    notes.push(format!("toy_equivalence against {against} FAILED: {m}"));
                }
            }
        }
        let t = format!(
            "plan {:.2} s, total {:.2} s",
            planned.as_secs_f64(),
            start.elapsed().as_secs_f64()
        );
        (
            format!(
                "{}  {}: {} [{t}]\n",
                if ok { "PASS" } else { "FAIL" },
                case.id,
                notes.join("; ")
            ),
            ok,
        )
    };

    // Cases are independent: run them on as many threads as there are cores.
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut lines: Vec<(usize, String, bool)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads.min(selected.len()).max(1))
            .map(|_| {
                s.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(case) = selected.get(i) else { break };
                        let (line, ok) = run_one(case);
                        print!("{line}");
                        out.push((i, line, ok));
                    }
                    out
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    });
    lines.sort_by_key(|l| l.0);
    let failed: Vec<&str> = lines
        .iter()
        .filter(|l| !l.2)
        .map(|l| selected[l.0].id.as_str())
        .collect();
    let mut report: String = lines.iter().map(|l| l.1.as_str()).collect();
    report.push_str(&format!(
        "\n{} M2 cases: {} passed, {} failed. Result files: {}\n",
        selected.len(),
        selected.len() - failed.len(),
        failed.len(),
        result_dir.display()
    ));
    println!("{report}");
    // Cases that fail for a reason outside this implementation, each
    // reported: they must fail, so an entry is removed once its cause is.
    let unexpected: Vec<&&str> = failed
        .iter()
        .filter(|id| !KNOWN_FAILURES.iter().any(|(k, _)| k == *id))
        .collect();
    let fixed: Vec<&str> = KNOWN_FAILURES
        .iter()
        .map(|(k, _)| *k)
        .filter(|k| selected.iter().any(|c| c.id == *k) && !failed.contains(k))
        .collect();
    assert!(
        unexpected.is_empty(),
        "failed M2 cases: {unexpected:?}\n{report}"
    );
    assert!(
        fixed.is_empty(),
        "known failures now pass; remove them from KNOWN_FAILURES: {fixed:?}"
    );
    assert!(!selected.is_empty());
}
