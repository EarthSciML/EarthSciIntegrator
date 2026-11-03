//! A leaf of operators a plan supplies advances only the union of their
//! writes (spec/operator-interface.md §8.2), and a block-diagonal leaf's
//! block size is the number of its own states (§8.3): a model of three
//! states over 4 cells whose `chem` operator writes X and Y and reads Z, and
//! whose `decay` operator writes Z.

use earthsci_integrator::config::RunConfig;
use earthsci_integrator::conformance::{check_fd, check_replay, Outcome};
use earthsci_integrator::operator::Structure;
use earthsci_integrator::planned::load_plan;
use earthsci_integrator::schedule::RunSpec;
use earthsci_integrator::toy::ToyProblem;
use earthsci_integrator::Error;
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

/// A model whose states do not share one shape: `X`, `Y` and `W` over 4
/// cells, `Z` a scalar. `chem` writes X and Y and reads only them, `decay`
/// writes Z, and `push` writes W and reads Z (`W' = −W + Z`). The same
/// document is planned by the Julia test of the same name (test/m2.jl).
fn mixed_shapes_doc() -> Value {
    json!({
        "esm": "1.3.0", "metadata": {"name": "mixed_shapes"},
        "index_sets": {"cell": {"kind": "interval", "size": 4}},
        "models": {"Box": {
            "variables": {
                "X": {"type": "unknown", "shape": ["cell"], "default": 1},
                "Y": {"type": "unknown", "shape": ["cell"], "default": 0.5},
                "W": {"type": "unknown", "shape": ["cell"], "default": 0.75},
                "Z": {"type": "unknown", "default": 2},
                "k": {"type": "parameter", "default": 3.5},
                "c": {"type": "parameter", "default": 0.25},
            },
            "equations": [
                {"lhs": d("X"), "rhs": fa(json!({"op": "*", "args": [-1, "k", ix("X")]}))},
                {"lhs": d("Y"), "rhs": fa(json!({"op": "-", "args": [{"op": "*", "args": ["k", ix("X")]}, ix("Y")]}))},
                {"lhs": d("W"), "rhs": fa(json!({"op": "+", "args": [{"op": "*", "args": [-1, ix("W")]}, "Z"]}))},
                {"lhs": d("Z"), "rhs": {"op": "*", "args": [-1, "c", "Z"]}},
            ],
        }},
    })
}

/// The mixed-shapes model planned with `chem`, `decay` and `push` leaves in
/// sequence, in the work directory `name`: the plan's directory, its
/// resolved leaves and its step.
fn mixed_shapes_plan(name: &str) -> (std::path::PathBuf, Vec<Value>, Value) {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("mixed_shapes.esm"),
        serde_json::to_vec(&mixed_shapes_doc()).unwrap(),
    )
    .unwrap();
    let ctl = json!({"dt_scope": "global", "reltol": 1e-8, "abstol": 1e-10});
    let block = json!({"block_lu": {"pivot": false}});
    let rc = json!({
        "esm_integrator": "0.1.0", "model": "mixed_shapes.esm",
        "operators": {
            "chem": {"select": {"variables": ["Box.X", "Box.Y"]}},
            "decay": {"select": {"variables": ["Box.Z"]}},
            "push": {"select": {"variables": ["Box.W"]}},
        },
        "schedule": {"sequence": [
            {"leaf": "chem", "alg": "Kvaerno4", "linsolve": block, "control": ctl},
            {"leaf": "decay", "alg": "Kvaerno4", "linsolve": block, "control": ctl},
            {"leaf": "push", "alg": "RK4", "dt": 0.05},
        ]},
        "step": {"dt": 0.25},
    });
    let rc_path = dir.join("mixed_shapes.run.json");
    std::fs::write(&rc_path, serde_json::to_vec_pretty(&rc).unwrap()).unwrap();
    let plan_dir = dir.join("mixed_shapes.plan");
    let _ = std::fs::remove_dir_all(&plan_dir);
    let options = Options {
        time: Some(json!({"start": 0, "end": 1})),
    };
    plan(&rc_path, &options, &AstDiffBands)
        .and_then(|p| p.write(&plan_dir))
        .unwrap();
    let pj: Value =
        serde_json::from_str(&std::fs::read_to_string(plan_dir.join("plan.json")).unwrap())
            .unwrap();
    let leaves = pj["schedule"]["sequence"].as_array().unwrap().clone();
    (plan_dir, leaves, pj["step"].clone())
}

/// The problem a plan's operators make.
fn problem_of(plan_dir: &Path) -> ToyProblem<f64> {
    load_plan::<f64>(plan_dir).unwrap().problem
}

/// A run config of `schedule` and `adjoint` against the mixed-shapes plan.
fn mixed_rc(schedule: Value, step: &Value, adjoint: Option<Value>) -> RunConfig {
    let mut v = json!({
        "esm_integrator": "0.1.0", "model": "mixed_shapes.esm",
        "schedule": schedule, "step": step,
    });
    if let Some(a) = adjoint {
        v["adjoint"] = a;
    }
    RunConfig::from_value(&v).unwrap()
}

/// §8.3: a leaf's storage is block_diagonal when its operators are
/// pointwise with `structure.block` and its own state variables share one
/// shape, whatever the shapes of the model's other states.
#[test]
fn block_diagonal_over_the_leafs_own_states() {
    let (plan_dir, leaves, step) = mixed_shapes_plan("leaf_state_m2_block");
    let problem = problem_of(&plan_dir);
    // Canonical order: Box.W, Box.X, Box.Y (4 cells each), Box.Z.
    assert_eq!(problem.n, 13);
    let s = RunSpec::with_problem(
        problem,
        &mixed_rc(json!({"sequence": leaves}), &step, None),
        (0.0, 1.0),
        None,
    )
    .unwrap();
    let (chem, decay, push) = (
        &s.schedule.leaves[0],
        &s.schedule.leaves[1],
        &s.schedule.leaves[2],
    );
    assert_eq!(chem.dofs, (4..12).collect::<Vec<_>>());
    assert_eq!(chem.structure, Structure::BlockDiagonal { ncells: 4, n: 2 });
    assert_eq!(decay.dofs, vec![12]);
    assert_eq!(
        decay.structure,
        Structure::BlockDiagonal { ncells: 1, n: 1 }
    );
    assert_eq!(push.dofs, (0..4).collect::<Vec<_>>());
    let fwd = s.forward(&s.u0(), &s.problem.p).unwrap();
    let u1 = fwd.final_state();
    // X(1) = e^{-k}, Z(1) = 2 e^{-c}.
    for &x in &u1[4..8] {
        assert!((x - (-3.5f64).exp()).abs() < 1e-6 * x, "X = {x}");
    }
    assert!(
        (u1[12] - 2.0 * (-0.25f64).exp()).abs() < 1e-7,
        "Z = {}",
        u1[12]
    );
}

/// §1, §8.3: an adjoint with respect to the initial state through a leaf
/// of planned operators that read only the leaf's own states is accepted,
/// and agrees with finite differences of the frozen tape; one through a
/// leaf whose operator reads a state the leaf does not advance is refused,
/// and so is one with respect to parameters (planned operators provide no
/// vjp_p at M2).
#[test]
fn adjoint_through_a_planned_leaf() {
    let (plan_dir, leaves, step) = mixed_shapes_plan("leaf_state_m2_adjoint");
    let wrt_u0 = json!({"functional": {"sum": "Box.Y", "over": "final"}, "wrt": ["u0"]});
    let erk = json!({"leaf": "chem", "alg": "Tsit5", "control": {"dt_scope": "global", "reltol": 1e-8, "abstol": 1e-10}});
    for (what, sch) in [
        ("Kvaerno4", json!({"sequence": [leaves[0], leaves[1]]})),
        ("Tsit5", json!({"sequence": [erk, leaves[1]]})),
    ] {
        let s = RunSpec::with_problem(
            problem_of(&plan_dir),
            &mixed_rc(sch, &step, Some(wrt_u0.clone())),
            (0.0, 1.0),
            None,
        )
        .unwrap();
        let u0 = s.u0();
        let p = s.problem.p.clone();
        let fwd = s.forward(&u0, &p).unwrap();
        let g = s.gradient(&fwd, &p).unwrap();
        // Σ Y(1) does not depend on W or Z: their cotangents pass through
        // the chem leaf unchanged, and no leaf sends them anything.
        assert!(g.u0[0..4].iter().all(|&x| x == 0.0), "{what}: dJ/dW0");
        assert_eq!(g.u0[12], 0.0, "{what}: dJ/dZ0");
        // dY(1)/dY0 = e^{-1}.
        for &y in &g.u0[8..12] {
            assert!((y - (-1.0f64).exp()).abs() < 1e-6, "{what}: dJ/dY0 = {y}");
        }
        for (o, k) in [
            (check_fd(&s, &fwd, &g, 1e-6, &[1e-3, 1e-4, 1e-5]), "fd"),
            (check_replay(&s, &fwd, &u0, &p), "replay"),
        ] {
            match o {
                Outcome::Pass(m) => println!("{what} {k}: {m}"),
                other => panic!("{what} {k}: {other:?}"),
            }
        }
    }

    // push reads Z, which its leaf does not advance.
    let refused = RunSpec::with_problem(
        problem_of(&plan_dir),
        &mixed_rc(
            leaves[2].clone(),
            &step,
            Some(json!({"functional": {"sum": "Box.W", "over": "final"}, "wrt": ["u0"]})),
        ),
        (0.0, 1.0),
        None,
    );
    match refused {
        Err(Error::Unimplemented(m)) => assert_eq!(
            m,
            "leaf root.2: operator push may read states the leaf does not advance; the adjoint \
             of a leaf through its constants is not implemented (operator-interface.md §1)"
        ),
        Err(e) => panic!("{e}"),
        Ok(_) => panic!("an adjoint through a leaf that reads its constants was accepted"),
    }

    // Parameter gradients need vjp_p.
    let refused = RunSpec::with_problem(
        problem_of(&plan_dir),
        &mixed_rc(
            json!({"sequence": [leaves[0], leaves[1]]}),
            &step,
            Some(
                json!({"functional": {"sum": "Box.Y", "over": "final"}, "wrt": ["params:numeric"]}),
            ),
        ),
        (0.0, 1.0),
        None,
    );
    match refused {
        Err(Error::Config(m)) => assert!(m.contains("chem") && m.contains("vjp_p"), "{m}"),
        Err(e) => panic!("{e}"),
        Ok(_) => {
            panic!("an adjoint with respect to parameters through planned operators was accepted")
        }
    }
}
