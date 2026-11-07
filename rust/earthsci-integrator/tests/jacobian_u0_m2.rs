//! A Jacobian document has no `ic` equations (spec/plan.md §6.2 item 4), so
//! a state that only an `ic` equation starts has no starting value in it.
//! The planner (the Gershgorin bound, pass 8) and the runtime build it with
//! every state element at zero as the caller's `u0`, which no evaluation
//! reads. The `column_contraction` fixture with its state's `default`
//! removed: it plans, its plan loads, and its initial state is the `ic`
//! equation's, as with the default.

use earthsci_integrator::planned::load_plan;
use earthsci_integrator_planner::astdiff::AstDiffBands;
use earthsci_integrator_planner::{plan, Options};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/conformance/plans/fixtures/column_contraction")
}

#[test]
fn a_state_with_an_ic_equation_and_no_default() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("jacobian_u0_m2");
    std::fs::create_dir_all(&dir).unwrap();
    let mut doc: Value =
        serde_json::from_slice(&std::fs::read(fixture().join("model.esm")).unwrap()).unwrap();
    let q = doc["models"]["Mix"]["variables"]["q"]
        .as_object_mut()
        .unwrap();
    assert!(q.remove("default").is_some());
    std::fs::write(dir.join("model.esm"), serde_json::to_vec(&doc).unwrap()).unwrap();
    std::fs::copy(fixture().join("auto.run.json"), dir.join("auto.run.json")).unwrap();

    let plan_dir = dir.join("plan");
    let _ = std::fs::remove_dir_all(&plan_dir);
    plan(
        &dir.join("auto.run.json"),
        &Options::default(),
        &AstDiffBands,
    )
    .and_then(|p| p.write(&plan_dir))
    .unwrap_or_else(|e| panic!("plan: {e}"));
    let jac: Value = serde_json::from_slice(
        &std::fs::read(plan_dir.join("derived/default_column.jacobian.esm")).unwrap(),
    )
    .unwrap();
    assert!(jac["models"]["Mix"]["variables"]["q"]
        .get("default")
        .is_none());

    let got = load_plan::<f64>(&plan_dir)
        .unwrap_or_else(|e| panic!("load_plan: {e}"))
        .problem;
    let want = load_plan::<f64>(&fixture().join("auto.golden"))
        .unwrap()
        .problem;
    assert_eq!(got.u0.len(), want.u0.len());
    assert!(got
        .u0
        .iter()
        .zip(&want.u0)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
}
