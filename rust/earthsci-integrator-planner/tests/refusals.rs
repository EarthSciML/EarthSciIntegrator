//! The refusal fixtures of `tests/conformance/plans/refusals/` (spec/plan.md
//! §9): each is refused with the listed code at the listed pointer.

mod common;

use common::{conformance, HandBands};
use earthsci_integrator_planner::error::codes;
use earthsci_integrator_planner::locality::NoBands;
use earthsci_integrator_planner::{plan, Options, PlanError};
use serde_json::Value;
use std::collections::BTreeSet;

/// Fixtures whose refusal needs what stage 2 brings, with the reason.
const DEFERRED: &[(&str, &str)] = &[(
    "auto-stiff-global",
    "the global operator's stiffness comes from its Gershgorin bound (§5.5.2 rule 4)",
)];

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
    "lift-shape-unresolved",
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
        if DEFERRED.iter().any(|(d, _)| d == name) {
            continue;
        }
        let path = dir.join(name).join("run.json");
        let got = plan(&path, &Options::default(), &HandBands);
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
    assert_eq!(passed + DEFERRED.len(), expected.as_object().unwrap().len());
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
