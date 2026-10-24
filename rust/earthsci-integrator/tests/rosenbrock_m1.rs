//! Tests of the Rosenbrock leaf and its adjoint beyond what cases.json
//! covers: every Rosenbrock table's adjoint at fixed and adaptive steps
//! under each controller, `dense_lu` against `block_lu`, the integral
//! functional, `check_stages`, refusals and f32 state.

use earthsci_integrator::config::RunConfig;
use earthsci_integrator::conformance::{check_fd, check_invariant, check_replay, Outcome};
use earthsci_integrator::schedule::RunSpec;
use earthsci_integrator::toy::forced_decay_exact;
use earthsci_integrator::Error;
use earthsci_integrator_solvers::tableau::tableau;
use serde_json::{json, Value};

const ROSENBROCK: [&str; 11] = [
    "Rosenbrock23",
    "ROS2",
    "ROS3P",
    "Rodas3",
    "Rodas4",
    "Rodas5P",
    "KPP_Ros2",
    "KPP_Ros3",
    "KPP_Ros4",
    "KPP_Rodas3",
    "KPP_Rodas4",
];

fn rc(model: &str, schedule: Value, dt: f64, adjoint: Option<Value>) -> RunConfig {
    let mut v = json!({"esm_integrator": "0.1.0", "model": model, "schedule": schedule, "step": {"dt": dt}});
    if let Some(a) = adjoint {
        v["adjoint"] = a;
    }
    RunConfig::from_value(&v).unwrap()
}

fn spec(rc: &RunConfig, tspan: (f64, f64), saveat: Option<&[f64]>) -> RunSpec<f64> {
    RunSpec::new(rc, tspan, saveat).unwrap()
}

fn assert_pass(o: Outcome, what: &str) {
    match o {
        Outcome::Pass(m) => println!("{what}: {m}"),
        other => panic!("{what}: {other:?}"),
    }
}

fn grad_all(quantity: &str, over: &str) -> Value {
    json!({"functional": {"sum": quantity, "over": over}, "wrt": ["u0", "params:numeric"]})
}

fn check_gradient(rc: &RunConfig, tspan: (f64, f64), saveat: Option<&[f64]>, what: &str) {
    let s = spec(rc, tspan, saveat);
    let u0 = s.u0();
    let p = s.problem.p.clone();
    let fwd = s.forward(&u0, &p).unwrap();
    let g = s.gradient(&fwd, &p).unwrap();
    assert_pass(
        check_fd(&s, &fwd, &g, 1e-6, &[1e-3, 1e-4, 1e-5]),
        &format!("{what} fd"),
    );
    assert_pass(check_replay(&s, &fwd, &u0, &p), &format!("{what} replay"));
}

/// Every table's adjoint at fixed dt: on van der Pol (the second-order term)
/// and on forced_decay (the f_t term and stage times), with point terms at
/// several output times.
#[test]
fn every_rosenbrock_method_adjoint_fixed_dt() {
    for alg in ROSENBROCK {
        let r = rc(
            "toy:van_der_pol/nonstiff",
            json!({"leaf": "oscillator", "alg": alg, "dt": 0.05}),
            3.0,
            Some(grad_all("y", "final")),
        );
        check_gradient(&r, (0.0, 3.0), None, alg);
        let r = rc(
            "toy:forced_decay",
            json!({"leaf": "decay", "alg": alg, "dt": 0.1}),
            1.0,
            Some(grad_all("u", "saveat")),
        );
        check_gradient(&r, (0.0, 2.0), Some(&[1.0, 2.0]), alg);
    }
}

/// Every adaptive table under each controller on the stiff van der Pol
/// oscillator: accuracy against a tight Rodas5P run, and the adjoint.
#[test]
fn every_rosenbrock_method_adaptive_and_controllers() {
    let rr = rc(
        "toy:van_der_pol/stiff",
        json!({"leaf": "oscillator", "alg": "Rodas5P", "control": {"reltol": 1e-11, "abstol": 1e-13}}),
        0.5,
        None,
    );
    let sr = spec(&rr, (0.0, 0.5), None);
    let refu = sr
        .forward(&sr.u0(), &sr.problem.p)
        .unwrap()
        .final_state()
        .to_vec();
    for alg in ROSENBROCK {
        for ctl in ["PI", "I", "PID"] {
            let r = rc(
                "toy:van_der_pol/stiff",
                json!({"leaf": "oscillator", "alg": alg,
                       "control": {"reltol": 1e-6, "abstol": 1e-8, "controller": ctl}}),
                0.5,
                Some(grad_all("y", "final")),
            );
            let s = spec(&r, (0.0, 0.5), None);
            let fwd = s.forward(&s.u0(), &s.problem.p).unwrap();
            let err = fwd
                .final_state()
                .iter()
                .zip(&refu)
                .map(|(a, b)| (a - b).abs() / b.abs().max(1e-3))
                .fold(0.0, f64::max);
            println!(
                "{alg} {ctl}: {} accepted, {} rejected, error {err:e}",
                fwd.accepted, fwd.rejected
            );
            // ROS2, ROS3P and the KPP Ros methods are not stiffly accurate;
            // at μ = 1000 their embedded estimates miss the stiff error and
            // the end state is off by up to a few per cent.
            let stiffly_accurate =
                alg == "Rosenbrock23" || tableau(alg).is_some_and(|t| t.stiffly_accurate);
            let bound = if stiffly_accurate { 1e-3 } else { 1e-1 };
            assert!(err < bound, "{alg} {ctl}: error {err}");
            check_gradient(&r, (0.0, 0.5), None, &format!("{alg} {ctl}"));
        }
    }
}

/// block_lu with and without pivoting agree with each other to roundoff on
/// the batched cells, and the adjoint goes through the transposed batched
/// solve.
#[test]
fn block_lu_pivoting_and_adjoint() {
    let run = |pivot: bool, adjoint: Option<Value>| {
        rc(
            "toy:photostationary_cells",
            json!({"leaf": "photochemistry", "alg": "Rodas4", "linsolve": {"block_lu": {"pivot": pivot}},
                   "control": {"reltol": 1e-6, "abstol": 1e-6}}),
            600.0,
            adjoint,
        )
    };
    let a = spec(&run(true, None), (0.0, 1200.0), None);
    let b = spec(&run(false, None), (0.0, 1200.0), None);
    let fa = a.forward(&a.u0(), &a.problem.p).unwrap();
    let fb = b.forward(&b.u0(), &b.problem.p).unwrap();
    let d = fa
        .final_state()
        .iter()
        .zip(fb.final_state())
        .map(|(x, y)| (x - y).abs() / y.abs())
        .fold(0.0, f64::max);
    println!("pivot vs no pivot: max relative difference {d:e}");
    assert!(d < 1e-9, "{d}");
    assert_pass(check_invariant(&a, &fa, 1e-12), "invariant");
    // A functional on a few cells only, so the cells' cotangents differ.
    let adj = json!({"functional": {"sum": "O3", "where": {"cell": [3, 5]}, "over": "final"},
                     "wrt": ["u0", "params:numeric"]});
    check_gradient(
        &run(true, Some(adj.clone())),
        (0.0, 1200.0),
        None,
        "block_lu pivot",
    );
    check_gradient(
        &run(false, Some(adj)),
        (0.0, 1200.0),
        None,
        "block_lu no pivot",
    );
}

/// The integral functional (spec/adjoint.md §2.1): the quadrature each
/// method induces converges to ∫ u dt on forced_decay, and its adjoint
/// agrees with frozen-tape finite differences.
#[test]
fn integral_functional() {
    let p = [2.0, 3.0];
    // ∫_0^2 u dt by composite Simpson on the exact solution.
    let n = 20000;
    let hh = 2.0 / n as f64;
    let mut exact = forced_decay_exact(0.0, 1.0, &p) + forced_decay_exact(2.0, 1.0, &p);
    for i in 1..n {
        let w = if i % 2 == 1 { 4.0 } else { 2.0 };
        exact += w * forced_decay_exact(i as f64 * hh, 1.0, &p);
    }
    exact *= hh / 3.0;
    for (alg, order) in [
        ("Tsit5", 5.0),
        ("SSPRK43", 3.0),
        ("Rosenbrock23", 2.0),
        ("Rodas4", 4.0),
        ("ROS3P", 3.0),
    ] {
        let mut errs = Vec::new();
        for dt in [0.1, 0.05] {
            let r = rc(
                "toy:forced_decay",
                json!({"leaf": "decay", "alg": alg, "dt": dt}),
                1.0,
                Some(
                    json!({"functional": {"mean": "u", "over": "time"}, "wrt": ["u0", "params:numeric"]}),
                ),
            );
            let s = spec(&r, (0.0, 2.0), None);
            let fwd = s.forward(&s.u0(), &s.problem.p).unwrap();
            let j = s.functional_value(&fwd).unwrap();
            errs.push((j - exact).abs());
            if dt == 0.1 {
                check_gradient(&r, (0.0, 2.0), None, &format!("{alg} integral"));
            }
        }
        let observed = (errs[0] / errs[1]).log2();
        println!("{alg}: integral errors {errs:?}, order {observed:.2}");
        assert!(observed > order - 0.4, "{alg}: order {observed}");
    }
    // Adaptive, with a stiff Rosenbrock leaf.
    let r = rc(
        "toy:van_der_pol/stiff",
        json!({"leaf": "oscillator", "alg": "Rodas4", "control": {"reltol": 1e-6, "abstol": 1e-8}}),
        0.25,
        Some(
            json!({"functional": {"sum": "z", "over": "time", "integrate_in": "root"}, "wrt": ["u0", "params:numeric"]}),
        ),
    );
    check_gradient(&r, (0.0, 0.5), None, "Rodas4 adaptive integral");
}

/// check_stages: a negative stage state is a violation, so the step is
/// rejected. On photostationary_cells with Rodas5P at reltol 1e-2 one stage
/// goes negative, which a check of the end states alone lets through.
#[test]
fn check_stages() {
    let run = |cs: bool| {
        let r = rc(
            "toy:photostationary_cells",
            json!({"leaf": "photochemistry", "alg": "Rodas5P", "control": {"reltol": 1e-2, "abstol": 1e-5},
                   "domain": {"nonnegative": "all", "check_stages": cs}}),
            600.0,
            None,
        );
        let s = spec(&r, (0.0, 3600.0), None);
        let fwd = s.forward(&s.u0(), &s.problem.p).unwrap();
        assert!(fwd.final_state().iter().all(|&x| x >= 0.0));
        assert_pass(check_invariant(&s, &fwd, 1e-12), "invariant");
        (fwd.accepted, fwd.rejected)
    };
    let (a0, r0) = run(false);
    let (a1, r1) = run(true);
    println!("check_stages off: {a0} accepted, {r0} rejected; on: {a1} accepted, {r1} rejected");
    assert_eq!(r0, 0);
    assert!(r1 > 0);
}

#[test]
fn rosenbrock_refusals() {
    let build = |model: &str, sched: Value| {
        let v = json!({"esm_integrator": "0.1.0", "model": model, "schedule": sched, "step": {"dt": 1}});
        RunConfig::from_value(&v).and_then(|r| RunSpec::<f64>::new(&r, (0.0, 2.0), None))
    };
    let err = |model: &str, sched: Value| match build(model, sched) {
        Err(e) => e,
        Ok(_) => panic!("accepted"),
    };
    let fd = "toy:forced_decay";
    let photo = "toy:photostationary_cells";
    assert!(matches!(
        err(
            fd,
            json!({"leaf": "decay", "alg": "Rodas4", "dt": 0.1, "jacobian": "ad"})
        ),
        Error::Config(_)
    ));
    assert!(matches!(
        err(
            fd,
            json!({"leaf": "decay", "alg": "Rodas4", "dt": 0.1, "jacobian": "fd"})
        ),
        Error::Unimplemented(_)
    ));
    assert!(matches!(
        err(
            fd,
            json!({"leaf": "decay", "alg": "Rodas4", "dt": 0.1, "linsolve": "gmres"})
        ),
        Error::Unimplemented(_)
    ));
    assert!(matches!(
        err(
            fd,
            json!({"leaf": "decay", "alg": "Rodas4", "dt": 0.1, "linsolve": "block_lu"})
        ),
        Error::Config(_)
    ));
    assert!(matches!(
        err(
            photo,
            json!({"leaf": "photochemistry", "alg": "Rodas4", "dt": 1, "linsolve": "dense_lu"})
        ),
        Error::Unimplemented(_)
    ));
    assert!(matches!(
        err(
            fd,
            json!({"leaf": "decay", "alg": "Tsit5", "dt": 0.1, "domain": {"nonnegative": "all", "check_stages": true}})
        ),
        Error::Config(_)
    ));
    // Defaults: block_lu for a block_diagonal leaf, dense_lu otherwise.
    assert!(build(
        photo,
        json!({"leaf": "photochemistry", "alg": "Rodas4", "dt": 1})
    )
    .is_ok());
    assert!(build(fd, json!({"leaf": "decay", "alg": "Rodas4", "dt": 0.1})).is_ok());
}

#[test]
fn f32_state() {
    let r = rc(
        "toy:robertson",
        json!({"leaf": "kinetics", "alg": "Rodas4", "control": {"reltol": 1e-4, "abstol": 1e-8}}),
        40.0,
        Some(grad_all("y3", "final")),
    );
    let s32: RunSpec<f32> = RunSpec::new(&r, (0.0, 40.0), None).unwrap();
    let s64: RunSpec<f64> = RunSpec::new(&r, (0.0, 40.0), None).unwrap();
    let f32run = s32.forward(&s32.u0(), &s32.problem.p).unwrap();
    let f64run = s64.forward(&s64.u0(), &s64.problem.p).unwrap();
    let d = (f32run.final_state()[2] as f64 - f64run.final_state()[2]).abs();
    assert!(d < 1e-4, "{d}");
    let g32 = s32.gradient(&f32run, &s32.problem.p).unwrap();
    let g64 = s64.gradient(&f64run, &s64.problem.p).unwrap();
    for (a, b) in g32.u0.iter().zip(&g64.u0) {
        assert!((a - b).abs() < 1e-3 * b.abs().max(1.0), "{a} {b}");
    }
}
