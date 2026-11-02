//! A leaf advances only its own state, the union of its operators' writes
//! (spec/operator-interface.md §1, §8.2): every other state is a constant of
//! the leaf over its interval and comes out unchanged bit for bit.
//! advection_reaction's transport leaf writes `A` only; `B` is set nonzero
//! here so that a leaf carrying it through its stage combinations would
//! round it.

use earthsci_integrator::config::RunConfig;
use earthsci_integrator::conformance::{check_fd, check_replay, Outcome};
use earthsci_integrator::operator::{Call, MatrixStorage, Operator, Structure};
use earthsci_integrator::schedule::RunSpec;
use earthsci_integrator::toy::ToyProblem;
use earthsci_integrator::Error;
use serde_json::{json, Value};
use std::sync::Arc;

const NC: usize = 64;

fn rc(schedule: Value, dt: f64) -> RunConfig {
    rc_model("toy:advection_reaction", schedule, dt, None)
}

fn rc_model(model: &str, schedule: Value, dt: f64, adjoint: Option<Value>) -> RunConfig {
    let mut v = json!({"esm_integrator": "0.1.0", "model": model, "schedule": schedule, "step": {"dt": dt}});
    if let Some(a) = adjoint {
        v["adjoint"] = a;
    }
    RunConfig::from_value(&v).unwrap()
}

/// The initial state with `B` nonzero: positive, negative and signed zeros.
fn u0_with_b(s: &RunSpec<f64>) -> Vec<f64> {
    let mut u = s.u0();
    for c in 0..NC {
        u[NC + c] = match c % 8 {
            0 => 0.0,
            1 => -0.0,
            2 => -0.3 - 1e-3 * c as f64,
            _ => 0.1 + 0.37 * ((c as f64) * 0.71).sin().abs() + 1e-17 * c as f64,
        };
    }
    u
}

/// Leaves of transport alone, one per method family and dt mode.
fn transport_leaves() -> Vec<(String, Value)> {
    let ctl = json!({"reltol": 1e-8, "abstol": 1e-10});
    let mut v = Vec::new();
    for alg in [
        "ForwardEuler",
        "Heun",
        "SSPRK33",
        "SSPRK43",
        "RK4",
        "Tsit5",
        "BS3",
        "SSPRK104",
        "WRF_RK3",
    ] {
        v.push((
            format!("{alg} fixed"),
            json!({"leaf": "transport", "alg": alg, "dt": 0.001}),
        ));
    }
    for alg in ["Heun", "SSPRK43", "RK4", "Tsit5", "BS3"] {
        v.push((
            format!("{alg} adaptive"),
            json!({"leaf": "transport", "alg": alg, "control": ctl}),
        ));
    }
    for alg in [
        "Rosenbrock23",
        "Rodas3",
        "Rodas4",
        "Rodas5P",
        "ROS2",
        "ROS3P",
        "KPP_Ros4",
        "KPP_Rodas3",
    ] {
        v.push((
            format!("{alg} adaptive"),
            json!({"leaf": "transport", "alg": alg, "control": ctl}),
        ));
        v.push((
            format!("{alg} fixed"),
            json!({"leaf": "transport", "alg": alg, "dt": 0.005}),
        ));
    }
    for alg in ["TRBDF2", "SDIRK2", "Kvaerno3", "Kvaerno4", "BackwardEuler"] {
        v.push((
            format!("{alg} adaptive"),
            json!({"leaf": "transport", "alg": alg, "control": ctl}),
        ));
        v.push((
            format!("{alg} fixed"),
            json!({"leaf": "transport", "alg": alg, "dt": 0.01}),
        ));
    }
    for th in [0.5, 0.7] {
        v.push((
            format!("Theta {th}"),
            json!({"leaf": "transport", "alg": "Theta", "theta": th, "dt": 0.01}),
        ));
    }
    for solve in ["converged", "linearized"] {
        v.push((
            format!("IMEXEuler {solve}"),
            json!({"leaf": "transport", "alg": "IMEXEuler", "imex": {"implicit": "transport"},
                   "implicit_solve": solve, "dt": 0.01}),
        ));
    }
    v.push((
        "subcycle SSPRK33".into(),
        json!({"subcycle": {"leaf": "transport", "alg": "SSPRK33", "dt": "step"}, "n": 4}),
    ));
    v.push((
        "sequence of two transport leaves".into(),
        json!({"sequence": [{"leaf": "transport", "alg": "SSPRK43", "control": ctl},
                            {"leaf": "transport", "alg": "Rodas4", "dt": 0.01}]}),
    ));
    v.push((
        "strang of transport leaves".into(),
        json!({"strang": [{"leaf": "transport", "alg": "SSPRK43", "control": ctl},
                          {"leaf": "transport", "alg": "Kvaerno4", "control": ctl}]}),
    ));
    v
}

/// The largest change of `B` over a run of `schedule`, and whether `B` came
/// out bit for bit as it went in.
fn b_change(schedule: Value) -> (f64, bool, bool) {
    let s = RunSpec::<f64>::new(&rc(schedule, 0.01), (0.0, 0.02), None).unwrap();
    let u0 = u0_with_b(&s);
    let fwd = s.forward(&u0, &s.problem.p).unwrap();
    let u1 = fwd.final_state();
    let mut d = 0.0f64;
    let mut same = true;
    for c in NC..2 * NC {
        d = d.max((u1[c] - u0[c]).abs());
        same &= u1[c].to_bits() == u0[c].to_bits();
    }
    let moved = (0..NC).any(|c| u1[c] != u0[c]);
    (d, same, moved)
}

#[test]
fn a_leaf_leaves_states_it_does_not_write_unchanged() {
    let mut bad = Vec::new();
    for (what, sch) in transport_leaves() {
        let (d, same, moved) = b_change(sch);
        println!("{what}: max |ΔB| = {d:e}, bitwise unchanged: {same}");
        assert!(moved, "{what}: A did not move");
        if !same {
            bad.push(format!("{what} ({d:e})"));
        }
    }
    assert!(bad.is_empty(), "B changed under: {}", bad.join(", "));
}

/// The transport leaf's state vector is `A`: its tolerances, error norm and
/// structure are over 64 DOFs, not the model's 128.
#[test]
fn the_transport_leaf_is_sized_to_its_own_state() {
    let ctl = json!({"reltol": 1e-8, "abstol": 1e-10});
    let s = RunSpec::<f64>::new(
        &rc(
            json!({"sequence": [{"leaf": "transport", "alg": "Rodas4", "control": ctl},
                                {"leaf": "reaction", "alg": "Rodas4", "control": ctl}]}),
            0.01,
        ),
        (0.0, 0.02),
        None,
    )
    .unwrap();
    let (tr, re) = (&s.schedule.leaves[0], &s.schedule.leaves[1]);
    assert_eq!(tr.dofs, (0..NC).collect::<Vec<_>>());
    assert!(tr.is_partial());
    assert_eq!(tr.tol.abstol.len(), NC);
    assert_eq!(tr.structure, Structure::Dense);
    assert_eq!(re.dofs.len(), 2 * NC);
    assert!(!re.is_partial());
    assert_eq!(re.structure, Structure::BlockDiagonal { ncells: NC, n: 2 });
}

/// The adjoint through the partial transport leaf: the cotangent of `B`
/// passes through it unchanged, and the gradient agrees with finite differences of the frozen tape and
/// replays bitwise.
#[test]
fn adjoint_through_a_partial_leaf() {
    let ctl = json!({"reltol": 1e-8, "abstol": 1e-10});
    for (what, sch) in [
        (
            "SSPRK43",
            json!({"leaf": "transport", "alg": "SSPRK43", "control": ctl}),
        ),
        (
            "Rodas4",
            json!({"leaf": "transport", "alg": "Rodas4", "dt": 0.005}),
        ),
        (
            "Kvaerno4",
            json!({"leaf": "transport", "alg": "Kvaerno4", "dt": 0.005}),
        ),
        (
            "strang",
            json!({"strang": [{"leaf": "transport", "alg": "SSPRK43", "control": ctl},
                              {"leaf": "reaction", "alg": "Rodas4", "control": ctl}]}),
        ),
    ] {
        let r = rc_model(
            "toy:advection_reaction",
            sch,
            0.01,
            Some(
                json!({"functional": {"sum": "B", "over": "final"}, "wrt": ["u0", "params:numeric"]}),
            ),
        );
        let s = RunSpec::<f64>::new(&r, (0.0, 0.02), None).unwrap();
        // check_fd perturbs the problem's own initial state.
        let u0 = s.u0();
        let p = s.problem.p.clone();
        let fwd = s.forward(&u0, &p).unwrap();
        let g = s.gradient(&fwd, &p).unwrap();
        if what != "strang" {
            // Σ B is untouched by transport: dJ/dB0 = 1, dJ/dA0 = 0, exactly.
            assert!(g.u0[..NC].iter().all(|&x| x == 0.0), "{what}: dJ/dA0");
            assert!(g.u0[NC..].iter().all(|&x| x == 1.0), "{what}: dJ/dB0");
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
}

/// A two-state problem whose operator `push` writes `x` and reads `y`
/// (`x' = −y`), and whose `relax` writes `y` (`y' = −y`): a leaf of `push`
/// alone reads `y` as a constant of the leaf.
struct Push;
struct Relax;

impl Operator<f64> for Push {
    fn name(&self) -> &str {
        "push"
    }
    fn structure(&self) -> Structure {
        Structure::Dense
    }
    fn depends_on_t(&self) -> bool {
        false
    }
    fn linear_in_u(&self) -> bool {
        true
    }
    fn writes(&self) -> Vec<usize> {
        vec![0]
    }
    fn reads(&self) -> Option<Vec<usize>> {
        Some(vec![1])
    }
    fn provides(&self, call: Call) -> bool {
        matches!(call, Call::Rhs | Call::JvpU | Call::VjpU | Call::Jac)
    }
    fn rhs(&self, out: &mut [f64], u: &[f64], _p: &[f64], _t: f64) {
        out[0] += -u[1];
    }
    fn jvp_u(&self, out: &mut [f64], _u: &[f64], _p: &[f64], _t: f64, v: &[f64]) {
        out[0] += -v[1];
    }
    fn vjp_u(&self, out: &mut [f64], _u: &[f64], _p: &[f64], _t: f64, w: &[f64]) {
        out[1] += -w[0];
    }
    fn jac(&self, out: &mut MatrixStorage<f64>, _u: &[f64], _p: &[f64], _t: f64) {
        out.add(0, 1, -1.0);
    }
}

impl Operator<f64> for Relax {
    fn name(&self) -> &str {
        "relax"
    }
    fn structure(&self) -> Structure {
        Structure::Dense
    }
    fn depends_on_t(&self) -> bool {
        false
    }
    fn linear_in_u(&self) -> bool {
        true
    }
    fn writes(&self) -> Vec<usize> {
        vec![1]
    }
    fn reads(&self) -> Option<Vec<usize>> {
        Some(vec![1])
    }
    fn rhs(&self, out: &mut [f64], u: &[f64], _p: &[f64], _t: f64) {
        out[1] += -u[1];
    }
    fn jvp_u(&self, out: &mut [f64], _u: &[f64], _p: &[f64], _t: f64, v: &[f64]) {
        out[1] += -v[1];
    }
    fn vjp_u(&self, out: &mut [f64], _u: &[f64], _p: &[f64], _t: f64, w: &[f64]) {
        out[1] += -w[1];
    }
}

fn push_relax() -> ToyProblem<f64> {
    ToyProblem {
        name: "push_relax".into(),
        n: 2,
        param_names: vec![],
        p: vec![],
        u0: vec![1.0, 0.75],
        tspan: (0.0, 1.0),
        variables: vec!["x".into(), "y".into()],
        cells: None,
        operators: vec![Arc::new(Push), Arc::new(Relax)],
        maps: vec![],
        invariants: vec![],
    }
}

/// A leaf reads the states it does not advance as constants (from the model
/// state at the start of its invocation) and leaves them unchanged; its
/// Jacobian drops their columns. An adjoint through it is refused, since the
/// cotangent of those constants is not implemented.
#[test]
fn a_partial_leaf_reads_its_constants() {
    for sch in [
        json!({"leaf": "push", "alg": "RK4", "dt": 0.125}),
        json!({"leaf": "push", "alg": "Rodas4", "dt": 0.125}),
        json!({"leaf": "push", "alg": "BackwardEuler", "dt": 0.125}),
    ] {
        let r = rc_model("toy:push_relax", sch.clone(), 0.5, None);
        let s = RunSpec::with_problem(push_relax(), &r, (0.0, 1.0), None).unwrap();
        let fwd = s.forward(&s.u0(), &[]).unwrap();
        let u1 = fwd.final_state();
        // x' = −0.75 exactly over each macro step, y fixed.
        assert_eq!(u1[1].to_bits(), 0.75f64.to_bits(), "{sch}");
        assert!((u1[0] - 0.25).abs() < 1e-14, "{sch}: x = {}", u1[0]);
    }
    // Split with relax, push reads y at the start of each of its invocations.
    let r = rc_model(
        "toy:push_relax",
        json!({"sequence": [{"leaf": "relax", "alg": "RK4", "dt": 0.5}, {"leaf": "push", "alg": "RK4", "dt": 0.5}]}),
        0.5,
        None,
    );
    let s = RunSpec::with_problem(push_relax(), &r, (0.0, 0.5), None).unwrap();
    let u1 = s.forward(&s.u0(), &[]).unwrap().final_state().to_vec();
    assert!((u1[0] - (1.0 - 0.5 * u1[1])).abs() < 1e-15, "x = {}", u1[0]);
    assert!(u1[1] < 0.75);
    // The adjoint is refused.
    let r = rc_model(
        "toy:push_relax",
        json!({"leaf": "push", "alg": "RK4", "dt": 0.125}),
        0.5,
        Some(json!({"functional": {"sum": "x", "over": "final"}, "wrt": ["u0"]})),
    );
    match RunSpec::with_problem(push_relax(), &r, (0.0, 1.0), None) {
        Err(Error::Unimplemented(m)) => assert!(m.contains("push"), "{m}"),
        Err(e) => panic!("{e}"),
        Ok(_) => panic!("an adjoint through a leaf that reads its constants was accepted"),
    }
}
