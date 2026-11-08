//! A Jacobian document has no `ic` equations (spec/plan.md §6.2 item 4), so
//! the planner gives every state without a `default` the `default` `1.0`;
//! otherwise EarthSciAST refuses to build the
//! document (`E_TREEWALK_MISSING_INITIAL_VALUE`). The `column_contraction`
//! fixture with its state's `default` removed: it plans, its Jacobian
//! document builds on its own with a plain `esm_problem`, its plan loads,
//! and its initial state is the `ic` equation's, as with the default.

use earthsci_ast::{esm_problem, rhs_program, Compiler, ProblemInput, ProblemOptions};
use earthsci_integrator::planned::load_plan;
use earthsci_integrator_planner::astdiff::AstDiffBands;
use earthsci_integrator_planner::{plan, Options};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/conformance/plans/fixtures/column_contraction")
}

#[test]
fn a_state_with_an_ic_equation_and_no_default() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("jacobian_default_m2");
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
    let jac_path = plan_dir.join("derived/default_column.jacobian.esm");
    let jac: Value = serde_json::from_slice(&std::fs::read(&jac_path).unwrap()).unwrap();
    assert_eq!(jac["models"]["Mix"]["variables"]["q"]["default"], 1.0);
    // The band fields keep EarthSciASTDiff's `default`.
    assert_eq!(
        jac["models"]["JacobianEval"]["variables"]["J_1"]["default"],
        0.0
    );
    // The right-hand-side document keeps the `ic` equation and no `default`.
    let rhs: Value = serde_json::from_slice(
        &std::fs::read(plan_dir.join("derived/default_column.rhs.esm")).unwrap(),
    )
    .unwrap();
    assert!(rhs["models"]["Mix"]["variables"]["q"]
        .get("default")
        .is_none());
    esm_problem(
        ProblemInput::Path(&jac_path),
        (0.0, 1.0),
        ProblemOptions::default(),
    )
    .unwrap_or_else(|e| panic!("esm_problem of the Jacobian document: {e}"));

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

/// §6.2 item 4, D89: a band coefficient that divides by a state with no
/// `default`, the shared fixture `inverse_decay` (`D(x) = -k/x`, `x` started
/// only by its `ic` equation). At the `default` 1.0 the Jacobian document's
/// right-hand side at its own initial state is finite; at the former zero,
/// `-k/x`'s coefficient `k/x^2` was infinite.
#[test]
fn a_band_coefficient_dividing_by_a_state_with_no_default() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/conformance/plans/fixtures/inverse_decay");
    let plan_dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("jacobian_default_m2_inverse");
    let _ = std::fs::remove_dir_all(&plan_dir);
    plan(
        &fixture.join("auto.run.json"),
        &Options::default(),
        &AstDiffBands,
    )
    .and_then(|p| p.write(&plan_dir))
    .unwrap_or_else(|e| panic!("plan: {e}"));
    let jacobians: Vec<PathBuf> = std::fs::read_dir(plan_dir.join("derived"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().ends_with(".jacobian.esm"))
        .collect();
    let [jac_path] = jacobians.as_slice() else {
        panic!("expected exactly one Jacobian document, found {jacobians:?}");
    };
    let jac: Value = serde_json::from_slice(&std::fs::read(jac_path).unwrap()).unwrap();
    assert_eq!(jac["models"]["M"]["variables"]["x"]["default"], 1.0);

    let prob = esm_problem(
        ProblemInput::Path(jac_path),
        (0.0, 1.0),
        ProblemOptions {
            compiler: Some(Compiler::Native),
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("esm_problem of the Jacobian document: {e}"));
    let prog = rhs_program(&prob).unwrap();
    let (u, aux) = prog.default_state();
    let p = prog.parameter_vector(&HashMap::new()).unwrap();
    let mut du = vec![f64::NAN; u.len()];
    let mut ws = prog.new_workspace();
    prog.eval_into(&u, &aux, &p, 0.0, &prog.default_forcing(), &mut du, &mut ws)
        .unwrap();
    assert!(du.iter().all(|x| x.is_finite()), "{du:?}");
    let b = prog
        .state_layout()
        .iter()
        .find(|b| b.name == "JacobianEval.J_1")
        .unwrap();
    assert_eq!(du[b.flat_offset], 0.5);
}
