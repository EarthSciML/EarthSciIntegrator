//! A leaf of operators a plan supplies advances only the union of their
//! writes (spec/operator-interface.md §8.2), and a block-diagonal leaf's
//! block size is the number of its own states (§8.3): a model of three
//! states over 4 cells whose `chem` operator writes X and Y and reads Z, and
//! whose `decay` operator writes Z.

use earthsci_integrator::config::RunConfig;
use earthsci_integrator::operator::Structure;
use earthsci_integrator::planned::load_plan;
use earthsci_integrator::schedule::RunSpec;
use earthsci_integrator::toy::ToyProblem;
use earthsci_integrator_planner::astdiff::AstDiffBands;
use earthsci_integrator_planner::{plan, Options};
use serde_json::{json, Value};
use std::path::Path;

fn fa(e: Value) -> Value {
    json!({"op": "faq", "args": [], "output_idx": ["i"], "ranges": {"i": {"from": "cell"}}, "expr": e})
}
fn ix(v: &str) -> Value {
    json!({"op": "index", "args": [v, "i"]})
}
fn d(x: &str) -> Value {
    json!({"op": "D", "args": [x], "wrt": "t"})
}

/// Plans `schedule` on the model and loads the plan: the run, and the
/// plan's own `plan.json`.
fn planned(dir: &Path, name: &str, schedule: Value) -> (RunSpec<f64>, Value, ToyProblem<f64>) {
    let rc = json!({
        "esm_integrator": "0.1.0", "model": "partial_leaves.esm",
        "operators": {
            "chem": {"select": {"variables": ["Box.X", "Box.Y"]}},
            "decay": {"select": {"variables": ["Box.Z"]}},
        },
        "schedule": schedule,
        "step": {"dt": 0.25},
    });
    let rc_path = dir.join(format!("{name}.run.json"));
    std::fs::write(&rc_path, serde_json::to_vec_pretty(&rc).unwrap()).unwrap();
    let plan_dir = dir.join(format!("{name}.plan"));
    let _ = std::fs::remove_dir_all(&plan_dir);
    let options = Options {
        time: Some(json!({"start": 0, "end": 1})),
    };
    plan(&rc_path, &options, &AstDiffBands)
        .and_then(|p| p.write(&plan_dir))
        .unwrap();
    let lp = load_plan::<f64>(&plan_dir).unwrap();
    let again = load_plan::<f64>(&plan_dir).unwrap().problem;
    let pj: Value =
        serde_json::from_str(&std::fs::read_to_string(plan_dir.join("plan.json")).unwrap())
            .unwrap();
    (
        RunSpec::with_problem(lp.problem, &lp.run_config, lp.tspan, None).unwrap(),
        pj,
        again,
    )
}

#[test]
fn planned_leaves_advance_their_own_states() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("leaf_state_m2");
    std::fs::create_dir_all(&dir).unwrap();
    let doc = json!({
        "esm": "1.3.0", "metadata": {"name": "partial_leaves"},
        "index_sets": {"cell": {"kind": "interval", "size": 4}},
        "models": {"Box": {
            "variables": {
                "X": {"type": "unknown", "shape": ["cell"], "default": 1},
                "Y": {"type": "unknown", "shape": ["cell"], "default": 0.5},
                "Z": {"type": "unknown", "shape": ["cell"], "default": 2},
                "k": {"type": "parameter", "default": 3.5},
                "c": {"type": "parameter", "default": 0.25},
            },
            "equations": [
                {"lhs": d("X"), "rhs": fa(json!({"op": "+", "args": [{"op": "*", "args": [-1, "k", ix("X")]}, ix("Z")]}))},
                {"lhs": d("Y"), "rhs": fa(json!({"op": "-", "args": [{"op": "*", "args": ["k", ix("X")]}, ix("Y")]}))},
                {"lhs": d("Z"), "rhs": fa(json!({"op": "*", "args": [-1, "c", ix("Z")]}))},
            ],
        }},
    });
    std::fs::write(
        dir.join("partial_leaves.esm"),
        serde_json::to_vec(&doc).unwrap(),
    )
    .unwrap();
    let ctl = json!({"dt_scope": "global", "reltol": 1e-8, "abstol": 1e-10});
    let leaf = |op: &str, alg: &str| json!({"leaf": op, "alg": alg, "linsolve": {"block_lu": {"pivot": false}}, "control": ctl});

    let (s, pj, problem) = planned(
        &dir,
        "both",
        json!({"sequence": [leaf("chem", "Rodas4"), leaf("decay", "Kvaerno4")]}),
    );
    let (chem, decay) = (&s.schedule.leaves[0], &s.schedule.leaves[1]);
    // Canonical order: Box.X, Box.Y, Box.Z, 4 cells each.
    assert_eq!(chem.dofs, (0..8).collect::<Vec<_>>());
    assert_eq!(decay.dofs, (8..12).collect::<Vec<_>>());
    assert_eq!(chem.structure, Structure::BlockDiagonal { ncells: 4, n: 2 });
    assert_eq!(
        decay.structure,
        Structure::BlockDiagonal { ncells: 4, n: 1 }
    );
    assert_eq!(chem.tol.abstol.len(), 8);
    let fwd = s.forward(&s.u0(), &s.problem.p).unwrap();
    let u1 = fwd.final_state();
    for &z in &u1[8..12] {
        assert!((z - 2.0 * (-0.25f64).exp()).abs() < 1e-7 * z, "Z = {z}");
    }

    // The chem leaf alone (the plan's resolved leaf) reads Z as a constant
    // and leaves it unchanged.
    let rc = RunConfig::from_value(&json!({
        "esm_integrator": "0.1.0", "model": "partial_leaves.esm",
        "schedule": pj["schedule"]["sequence"][0], "step": pj["step"],
    }))
    .unwrap();
    let s = RunSpec::with_problem(problem, &rc, (0.0, 1.0), None).unwrap();
    assert_eq!(s.schedule.leaves.len(), 1);
    let u0 = s.u0();
    let u1 = s.forward(&u0, &s.problem.p).unwrap().final_state().to_vec();
    for j in 8..12 {
        assert_eq!(u1[j].to_bits(), u0[j].to_bits(), "Z changed at DOF {j}");
    }
    assert!(u1[..8].iter().zip(&u0[..8]).any(|(a, b)| a != b));
}
