//! The refusal fixtures of `tests/conformance/plans/refusals/` (spec/plan.md
//! §9): each is refused with the listed code at the listed pointer.

mod common;

use common::conformance;
use earthsci_integrator_planner::astdiff::{self, testing, AstDiffBands, Mode};
use earthsci_integrator_planner::error::codes;
use earthsci_integrator_planner::locality::{BandSourceError, NoBands};
use earthsci_integrator_planner::{model, plan, runconfig, terms, Options, PlanError};
use serde_json::Value;
use std::collections::BTreeSet;

/// Fixtures decided before pass 2, so they pass with no bands at all.
const BEFORE_BANDS: &[&str] = &[
    "toy-model",
    "solver-external-mismatch",
    "algebraic-equation",
    "discrete-event",
    "schedule-update",
    "reserved-component-name",
    "axes-unresolved",
    "axes-too-many",
    "time-interval-empty",
    "time-interval-missing",
    "time-invalid-date",
];

#[test]
fn refusal_fixtures() {
    let dir = conformance().join("plans/refusals");
    let expected: Value =
        serde_json::from_slice(&std::fs::read(dir.join("expected-codes.json")).unwrap()).unwrap();
    let mut failures = Vec::new();
    let mut passed = 0;
    for (name, want) in expected.as_object().unwrap() {
        let path = dir.join(name).join("run.json");
        let got = plan(&path, &Options::default(), &AstDiffBands);
        let (code, at) = (want["code"].as_str().unwrap(), want["at"].as_str().unwrap());
        match got {
            Err(PlanError::Refused(r)) if r.code == code && r.pointer == at => passed += 1,
            Err(e) => failures.push(format!("{name}: want {code} at {at}, got {e}")),
            Ok(_) => failures.push(format!("{name}: want {code} at {at}, got a plan")),
        }
        if BEFORE_BANDS.contains(&name.as_str()) {
            match plan(&path, &Options::default(), &NoBands) {
                Err(PlanError::Refused(r)) if r.code == code && r.pointer == at => {}
                other => failures.push(format!("{name} without bands: {other:?}")),
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!(passed, expected.as_object().unwrap().len());
}

/// `band_model_failed` (§5.2.2) is reported against the term it concerns.
/// The third term's `erf` has no derivative rule, and EarthSciASTDiff's
/// structure-only mode refuses what the full mode refuses (esm-jacobian-spec
/// §2.1, EarthSciASTDiff 0.4.0), so pass 2's probe fails and its halving
/// search names the third term. The full mode finds the same term, and so
/// does pass 4's search when it is run alone.
#[test]
fn band_model_failed_names_its_term() {
    let path = conformance().join("plans/refusals/band-model-failed/run.json");
    let rc = runconfig::read(&path).unwrap();
    let info = model::read(&rc).unwrap();
    let ts = terms::terms(&info).unwrap();
    let probe = testing::probe(&info, &ts).unwrap();
    assert!(probe.is_err(), "the probe document's structure succeeds");
    let mut found = None;
    for mode in [Mode::Structure, Mode::Full] {
        // The same halving search, in every batch size and in either mode,
        // finds the third term, as each term's own document does.
        let each = match testing::each_term(&info, &ts, mode) {
            Err(BandSourceError::Failed(f)) => f,
            other => panic!("{mode:?}: want the third term's failure, got {other:?}"),
        };
        assert_eq!(each.term, 2);
        for large in [0, 1, 3] {
            match testing::in_batches(&info, &ts, large, mode) {
                Err(BandSourceError::Failed(f)) => assert_eq!(f, each, "{mode:?}, large = {large}"),
                other => panic!(
                    "{mode:?}, large = {large}: want the third term's failure, got {other:?}"
                ),
            }
        }
        assert!(found.as_ref().is_none_or(|f| *f == each));
        found = Some(each);
    }
    let full = found.unwrap();
    assert_eq!(astdiff::first_failing_term(&info, &ts), Some(full.clone()));
    // Every term here has one operator node, so the bounds 0 and 1 give
    // single-term batches, 2 gives [t0, t1] [t2], and 3 one batch of all
    // three.
    assert_eq!(astdiff::probe_batches(&ts, 0), [0..1, 1..2, 2..3]);
    assert_eq!(astdiff::probe_batches(&ts, 1), [0..1, 1..2, 2..3]);
    assert_eq!(astdiff::probe_batches(&ts, 2), [0..2, 2..3]);
    assert_eq!(astdiff::probe_batches(&ts, 3), vec![0..3]);
    // Batches given directly, so that a failing probe's halves search also
    // runs in a later batch, from a term other than the first: [t0] [t1, t2]
    // reaches the third term only through the batch's offset.
    let batches: [&[std::ops::Range<usize>]; 3] =
        [&[0..1, 1..3], &[0..2, 2..3], &[0..1, 1..2, 2..3]];
    for b in batches {
        match testing::in_given_batches(&info, &ts, b.to_vec(), Mode::Full) {
            Err(BandSourceError::Failed(f)) => assert_eq!(f, full, "batches {b:?}"),
            other => panic!("batches {b:?}: want the third term's failure, got {other:?}"),
        }
    }
    match plan(&path, &Options::default(), &AstDiffBands) {
        Err(PlanError::Refused(r)) => {
            assert_eq!(
                (r.code, r.pointer.as_str()),
                (codes::BAND_MODEL_FAILED, "/model")
            );
            assert!(r.message.starts_with("term (K.y, 0, 1): "), "{}", r.message);
            assert!(r.message.contains("erf"), "{}", r.message);
        }
        other => panic!("want band_model_failed, got {other:?}"),
    }
}

/// On every refusal fixture that reaches pass 2, the probe document
/// (§5.2.2) fails exactly when some term's own document fails, and gives
/// the same bands when none does: a successful probe hides no term's
/// failure.
#[test]
fn probe_fails_exactly_when_a_term_fails() {
    let dir = conformance().join("plans/refusals");
    let expected: Value =
        serde_json::from_slice(&std::fs::read(dir.join("expected-codes.json")).unwrap()).unwrap();
    let mut checked = 0;
    for name in expected.as_object().unwrap().keys() {
        let Ok(rc) = runconfig::read(&dir.join(name).join("run.json")) else {
            continue;
        };
        let Ok(info) = model::read(&rc) else { continue };
        let Ok(ts) = terms::terms(&info) else {
            continue;
        };
        if ts.is_empty() {
            continue;
        }
        let probe = testing::probe(&info, &ts).unwrap();
        let each = testing::each_term(&info, &ts, Mode::Structure);
        match (probe, each) {
            (Ok(p), Ok(e)) => assert_eq!(p, e, "{name}"),
            (Err(_), Err(_)) => {}
            (p, e) => panic!(
                "{name}: probe {:?}, term documents {:?}",
                p.is_ok(),
                e.is_ok()
            ),
        }
        checked += 1;
    }
    assert!(checked > 0);
}

/// Every code the planner raises is in spec/planner-errors.json.
#[test]
fn codes_are_listed() {
    let path = conformance().join("../../spec/planner-errors.json");
    let v: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let listed: BTreeSet<&str> = v["codes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["code"].as_str().unwrap())
        .collect();
    for c in codes::ALL {
        assert!(listed.contains(c), "{c} is not in spec/planner-errors.json");
    }
}
