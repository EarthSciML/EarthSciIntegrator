//! Runs `tests/conformance/m1/cases.json` (spec/toy-problems.md §2).
//!
//! Every case whose leaves all use implemented methods runs with the checks
//! Rust can do; `oracle`, `dot_product` and `reverse_diff` are Julia's, and
//! `cross_language` runs in a separate CI job that compares the Julia and
//! Rust result files. Cases that need unimplemented methods
//! are listed as skipped, by name, and never counted as passing. A result
//! file per case run goes to `$EARTHSCI_RESULTS_DIR` (default
//! `rust/target/tmp/conformance/m1/rust`). `$EARTHSCI_REFERENCE_DIR`
//! overrides the directory of reference results.

use earthsci_integrator::conformance::{default_paths, run_case, CasesFile, Outcome};
use std::path::{Path, PathBuf};

#[test]
fn m1_conformance_cases() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let text = std::fs::read_to_string(repo.join("tests/conformance/m1/cases.json")).unwrap();
    let cases: CasesFile = serde_json::from_str(&text).expect("cases.json parses");
    assert_eq!(cases.esm_integrator, earthsci_integrator::SPEC_VERSION);
    let result_dir = std::env::var_os("EARTHSCI_RESULTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_TARGET_TMPDIR")).join("conformance/m1/rust"));
    let mut paths = default_paths(&repo, result_dir.clone());
    // `$EARTHSCI_REFERENCE_DIR` overrides tests/conformance/m1/reference/.
    if let Some(d) = std::env::var_os("EARTHSCI_REFERENCE_DIR") {
        paths.reference_dir = PathBuf::from(d);
    }

    let mut report = String::new();
    let (mut ran, mut skipped, mut failed_cases) = (0, Vec::new(), Vec::new());
    let (mut passed, mut failed, mut skipped_checks) = (0, 0, 0);
    for case in &cases.cases {
        let r = run_case(case, &paths);
        report.push_str(&r.to_string());
        if let Some(reason) = &r.skipped {
            skipped.push((case.id.clone(), reason.clone()));
            continue;
        }
        if r.error.is_some() {
            failed_cases.push(case.id.clone());
            continue;
        }
        ran += 1;
        let mut case_failed = false;
        for (_, o) in &r.checks {
            match o {
                Outcome::Pass(_) => passed += 1,
                Outcome::Fail(_) => {
                    failed += 1;
                    case_failed = true;
                }
                Outcome::Skip(_) => skipped_checks += 1,
            }
        }
        if case_failed {
            failed_cases.push(case.id.clone());
        }
    }
    report.push_str(&format!(
        "\n{} cases: {ran} ran ({} with a failure), {} skipped for unimplemented methods or features.\n\
         Checks in the cases that ran: {passed} passed, {failed} failed, {skipped_checks} skipped \
         (Julia-only, cross-language, or reference file absent).\n\
         Skipped cases ({}): {}\nResult files: {}\n",
        cases.cases.len(),
        failed_cases.len(),
        skipped.len(),
        skipped.len(),
        skipped.iter().map(|s| s.0.as_str()).collect::<Vec<_>>().join(", "),
        result_dir.display(),
    ));
    println!("{report}");
    assert!(
        failed_cases.is_empty(),
        "failed cases: {failed_cases:?}\n{report}"
    );
    assert!(ran > 0);
}
