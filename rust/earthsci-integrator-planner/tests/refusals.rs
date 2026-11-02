//! The refusal fixtures of `tests/conformance/plans/refusals/` (spec/plan.md
//! §9): each is refused with the listed code at the listed pointer.

mod common;

use common::conformance;
use earthsci_integrator_planner::astdiff::AstDiffBands;
use earthsci_integrator_planner::error::codes;
use earthsci_integrator_planner::locality::NoBands;
use earthsci_integrator_planner::{plan, Options, PlanError};
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

/// `band_model_failed` (§5.2.2) is reported against the term it concerns:
/// the probe document fails, and the term documents find the third term,
/// whose `erf` has no derivative rule.
#[test]
fn band_model_failed_names_its_term() {
    let path = conformance().join("plans/refusals/band-model-failed/run.json");
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
