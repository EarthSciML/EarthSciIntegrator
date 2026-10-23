//! Tests of the explicit Runge–Kutta leaf, the combinators, the tape and the
//! adjoint beyond what cases.json covers: every explicit method's adjoint,
//! the combinators with explicit leaves, controllers, domain enforcement,
//! return codes, refusals and f32 state.

use earthsci_integrator::config::RunConfig;
use earthsci_integrator::conformance::{check_fd, check_invariant, check_replay, Outcome};
use earthsci_integrator::schedule::RunSpec;
use earthsci_integrator::tape::TapeRecord;
use earthsci_integrator::Error;
use serde_json::{json, Value};

const EXPLICIT: [&str; 9] = [
    "ForwardEuler",
    "Heun",
    "SSPRK33",
    "SSPRK43",
    "RK4",
    "Tsit5",
    "BS3",
    "SSPRK104",
    "WRF_RK3",
];
const ADAPTIVE: [&str; 5] = ["Heun", "SSPRK43", "RK4", "Tsit5", "BS3"];

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

/// Adjoint against finite differences of the frozen tape and bitwise replay,
/// for one run config.
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

#[test]
fn every_explicit_method_adjoint_fixed_dt() {
    for alg in EXPLICIT {
        let r = rc(
            "toy:van_der_pol/nonstiff",
            json!({"leaf": "oscillator", "alg": alg, "dt": 0.05}),
            3.0,
            Some(grad_all("y", "final")),
        );
        check_gradient(&r, (0.0, 3.0), None, alg);
        // Non-autonomous: the stage times enter the parameter gradient.
        let r = rc(
            "toy:forced_decay",
            json!({"leaf": "decay", "alg": alg, "dt": 0.1}),
            1.0,
            Some(grad_all("u", "saveat")),
        );
        check_gradient(&r, (0.0, 2.0), Some(&[1.0, 2.0]), alg);
    }
}

#[test]
fn every_adaptive_method_adjoint_and_controller() {
    for alg in ADAPTIVE {
        for ctl in ["PI", "I", "PID"] {
            let r = rc(
                "toy:van_der_pol/nonstiff",
                json!({"leaf": "oscillator", "alg": alg,
                       "control": {"reltol": 1e-6, "abstol": 1e-8, "controller": ctl}}),
                2.0,
                Some(grad_all("z", "final")),
            );
            let s = spec(&r, (0.0, 6.0), None);
            let fwd = s.forward(&s.u0(), &s.problem.p).unwrap();
            // A reference by Tsit5 at a tight tolerance.
            let rr = rc(
                "toy:van_der_pol/nonstiff",
                json!({"leaf": "oscillator", "alg": "Tsit5", "control": {"reltol": 1e-12, "abstol": 1e-14}}),
                6.0,
                None,
            );
            let sr = spec(&rr, (0.0, 6.0), None);
            let refu = sr.forward(&sr.u0(), &sr.problem.p).unwrap();
            let err = fwd
                .final_state()
                .iter()
                .zip(refu.final_state())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            println!(
                "{alg} {ctl}: {} accepted, {} rejected, error {err:e}",
                fwd.accepted, fwd.rejected
            );
            assert!(err < 1e-3, "{alg} {ctl}: error {err}");
            check_gradient(&r, (0.0, 6.0), None, &format!("{alg} {ctl}"));
        }
    }
}

fn advection(schedule: Value, dt: f64, adjoint: Option<Value>) -> RunConfig {
    rc("toy:advection_reaction", schedule, dt, adjoint)
}

#[test]
fn combinators_with_explicit_leaves() {
    let transport = json!({"leaf": "transport", "alg": "SSPRK43", "control": {"reltol": 1e-8, "abstol": 1e-10}});
    let reaction = json!({"leaf": "reaction", "alg": "RK4", "dt": 0.002});
    let schedules = [
        ("sequence", json!({"sequence": [transport, reaction]})),
        ("strang", json!({"strang": [transport, reaction]})),
        (
            "subcycle",
            json!({"sequence": [{"subcycle": {"leaf": "transport", "alg": "SSPRK33", "dt": "step"}, "n": 4},
                                {"leaf": "reaction", "alg": "Heun", "dt": "step"}]}),
        ),
        (
            "nested",
            json!({"strang": [{"subcycle": {"leaf": "transport", "alg": "SSPRK104", "dt": "step"}, "n": 2},
                              {"sequence": [reaction, {"leaf": "reaction", "alg": "WRF_RK3", "dt": 0.003}]}]}),
        ),
    ];
    for (name, sched) in schedules {
        let dt = if name == "sequence" || name == "strang" {
            0.025
        } else {
            0.005
        };
        let r = advection(
            sched,
            dt,
            Some(
                json!({"functional": {"mean": "B", "over": "saveat", "where": {"cell": [3, 40]}},
                        "wrt": ["u0", "params:numeric"]}),
            ),
        );
        let s = spec(&r, (0.0, 0.1), Some(&[0.05, 0.1]));
        let fwd = s.forward(&s.u0(), &s.problem.p).unwrap();
        assert_pass(
            check_invariant(&s, &fwd, 1e-12),
            &format!("{name} invariant"),
        );
        check_gradient(&r, (0.0, 0.1), Some(&[0.05, 0.1]), name);
    }
}

#[test]
fn tape_paths_follow_the_schedule() {
    let r = advection(
        json!({"strang": [{"subcycle": {"leaf": "transport", "alg": "SSPRK33", "dt": "step"}, "n": 2},
                          {"leaf": "reaction", "alg": "Heun", "dt": "step", "id": "chem"}]}),
        0.01,
        None,
    );
    let s = spec(&r, (0.0, 0.02), None);
    let fwd = s.forward(&s.u0(), &s.problem.p).unwrap();
    let paths: Vec<&str> = fwd.tape.records.iter().map(|r| r.path()).collect();
    assert_eq!(
        &paths[..7],
        &[
            "0/root.0.0",
            "0/root.0.0#1",
            "0/root.0",
            "0/chem",
            "0/root.0.0#2",
            "0/root.0.0#3",
            "0/root.0#1"
        ]
    );
    assert_eq!(paths[7], "0");
    assert_eq!(paths[8], "1/root.0.0");
    // Strang halves cover [t, t+Δ/2] and [t+Δ/2, t+Δ]; the middle child [t, t+Δ].
    match &fwd.tape.records[0] {
        TapeRecord::Leaf { t, dt, steps, .. } => {
            assert_eq!((*t, *dt), (0.0, 0.0025));
            assert_eq!(steps, &vec![[0.0, 0.0025]]);
        }
        r => panic!("{r:?}"),
    }
    match &fwd.tape.records[3] {
        TapeRecord::Leaf { t, dt, .. } => assert_eq!((*t, *dt), (0.0, 0.01)),
        r => panic!("{r:?}"),
    }
    match &fwd.tape.records[6] {
        TapeRecord::Subcycle { t, dt, n, .. } => assert_eq!((*t, *dt, *n), (0.005, 0.005, 2)),
        r => panic!("{r:?}"),
    }
}

#[test]
fn splitting_orders_with_explicit_leaves() {
    // Reference: both operators in one leaf, tight tolerance.
    let reference = advection(
        json!({"leaf": ["transport", "reaction"], "alg": "Tsit5", "control": {"reltol": 1e-12, "abstol": 1e-14}}),
        0.2,
        None,
    );
    let sr = spec(&reference, (0.0, 0.2), None);
    let uref = sr
        .forward(&sr.u0(), &sr.problem.p)
        .unwrap()
        .final_state()
        .to_vec();
    let transport =
        json!({"leaf": "transport", "alg": "Tsit5", "control": {"reltol": 1e-12, "abstol": 1e-14}});
    let reaction =
        json!({"leaf": "reaction", "alg": "Tsit5", "control": {"reltol": 1e-12, "abstol": 1e-14}});
    for (name, sched, expected) in [
        ("lie", json!({"sequence": [transport, reaction]}), 1.0),
        ("strang", json!({"strang": [transport, reaction]}), 2.0),
    ] {
        let mut errs = Vec::new();
        for dt in [0.02, 0.01, 0.005] {
            let r = advection(sched.clone(), dt, None);
            let s = spec(&r, (0.0, 0.2), None);
            let u = s
                .forward(&s.u0(), &s.problem.p)
                .unwrap()
                .final_state()
                .to_vec();
            errs.push(
                u.iter()
                    .zip(&uref)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max),
            );
        }
        let order = (errs[0] / errs[2]).log2() / 2.0;
        println!("{name}: errors {errs:?}, order {order}");
        assert!((order - expected).abs() < 0.2, "{name}: order {order}");
    }
}

#[test]
fn domain_reject_clip_and_return_codes() {
    // forced_decay crosses zero near t = 1.3; the exact solution is negative at 2.
    let adaptive = |domain: Value| {
        rc(
            "toy:forced_decay",
            json!({"leaf": "decay", "alg": "Tsit5", "control": {"reltol": 1e-6, "abstol": 1e-8}, "domain": domain}),
            2.0,
            Some(grad_all("u", "final")),
        )
    };
    // reject: the leaf cannot pass zero and shrinks to a return code.
    let s = spec(&adaptive(json!({"nonnegative": "all"})), (0.0, 2.0), None);
    let e = s.forward(&s.u0(), &s.problem.p).unwrap_err();
    assert!(
        matches!(&e, Error::Run(m) if m.contains("Unstable") || m.contains("MaxIters")),
        "{e}"
    );
    // fixed dt: DomainViolation.
    let r = rc(
        "toy:forced_decay",
        json!({"leaf": "decay", "alg": "RK4", "dt": 0.1, "domain": {"nonnegative": ["u"]}}),
        2.0,
        None,
    );
    let s = spec(&r, (0.0, 2.0), None);
    let e = s.forward(&s.u0(), &s.problem.p).unwrap_err();
    assert!(
        matches!(&e, Error::Run(m) if m.contains("DomainViolation")),
        "{e}"
    );
    // clip: the state stays at zero, the masks are on the tape, and the
    // adjoint (masked) agrees with frozen-tape finite differences.
    for alg in ["RK4", "SSPRK43"] {
        let r = rc(
            "toy:forced_decay",
            json!({"leaf": "decay", "alg": alg, "dt": 0.05, "domain": {"nonnegative": "all", "on_violation": "clip"}}),
            0.5,
            Some(grad_all("u", "saveat")),
        );
        let s = spec(&r, (0.0, 2.0), Some(&[1.0, 1.5, 2.0]));
        let fwd = s.forward(&s.u0(), &s.problem.p).unwrap();
        assert!(fwd.final_state()[0] >= 0.0);
        let clips: usize = fwd
            .tape
            .records
            .iter()
            .map(|r| match r {
                TapeRecord::Leaf { clips, .. } => clips.len(),
                _ => 0,
            })
            .sum();
        assert!(clips > 0, "{alg}: no clips recorded");
        check_gradient(
            &r,
            (0.0, 2.0),
            Some(&[1.0, 1.5, 2.0]),
            &format!("{alg} clip"),
        );
    }
    // MaxIters.
    let r = rc(
        "toy:van_der_pol/nonstiff",
        json!({"leaf": "oscillator", "alg": "Tsit5", "control": {"reltol": 1e-10, "abstol": 1e-12, "maxiters": 5}}),
        6.3,
        None,
    );
    let s = spec(&r, (0.0, 6.3), None);
    let e = s.forward(&s.u0(), &s.problem.p).unwrap_err();
    assert!(matches!(&e, Error::Run(m) if m.contains("MaxIters")), "{e}");
    // dtmin above what the tolerance needs: DtLessThanMin.
    let r = rc(
        "toy:van_der_pol/nonstiff",
        json!({"leaf": "oscillator", "alg": "Heun", "control": {"reltol": 1e-12, "abstol": 1e-14, "dtmin": 0.01}}),
        6.3,
        None,
    );
    let s = spec(&r, (0.0, 6.3), None);
    let e = s.forward(&s.u0(), &s.problem.p).unwrap_err();
    assert!(
        matches!(&e, Error::Run(m) if m.contains("DtLessThanMin")),
        "{e}"
    );
}

#[test]
fn refusals() {
    let build = |sched: Value| {
        let v = json!({"esm_integrator": "0.1.0", "model": "toy:forced_decay", "schedule": sched, "step": {"dt": 1}});
        RunConfig::from_value(&v).and_then(|r| RunSpec::<f64>::new(&r, (0.0, 2.0), None))
    };
    let err = |sched: Value| match build(sched) {
        Err(e) => e,
        Ok(_) => panic!("accepted"),
    };
    assert!(
        matches!(err(json!({"leaf": "decay", "alg": "ForwardEuler"})), Error::Config(m) if m.contains("D52"))
    );
    assert!(matches!(
        err(json!({"leaf": "nope", "alg": "Heun", "dt": 0.1})),
        Error::Config(_)
    ));
    assert!(
        matches!(err(json!({"leaf": "decay", "alg": "Rodas4", "dt": 0.1})), Error::Unimplemented(m) if m.contains("Rodas4"))
    );
    assert!(matches!(
        err(json!({"leaf": "decay", "alg": "Tsit5", "control": {"dt_scope": "cell"}})),
        Error::Unimplemented(_)
    ));
    assert!(matches!(
        err(json!({"leaf": "decay", "alg": "Tsit5", "theta": 0.5})),
        Error::Config(_)
    ));
    assert!(matches!(
        err(json!({"leaf": "decay", "alg": "Tsit5", "bogus": 1})),
        Error::Config(_)
    ));
    let v = json!({"esm_integrator": "9.9.9", "model": "toy:forced_decay", "schedule": {"leaf": "decay", "alg": "Heun", "dt": 0.1}});
    assert!(RunConfig::from_value(&v).is_err());
    // An operator that lacks a call the adjoint needs is refused at build;
    // every toy operator provides all of them, so a functional over "time"
    // is the remaining refusal here.
    let v = json!({"esm_integrator": "0.1.0", "model": "toy:forced_decay", "schedule": {"leaf": "decay", "alg": "Heun", "dt": 0.1},
                   "step": {"dt": 1}, "adjoint": {"functional": {"sum": "u", "over": "time"}, "wrt": ["u0"]}});
    let r = RunConfig::from_value(&v).unwrap();
    assert!(matches!(
        RunSpec::<f64>::new(&r, (0.0, 2.0), None),
        Err(Error::Unimplemented(_))
    ));
}

#[test]
fn bind_dt_parameters_are_tape_constants() {
    // Bind ω to the leaf's step: the run changes and ω gets no cotangent.
    let r = rc(
        "toy:forced_decay",
        json!({"leaf": "decay", "alg": "Heun", "dt": 0.1, "bind_dt": {"omega": "leaf"}}),
        1.0,
        Some(grad_all("u", "final")),
    );
    let s = spec(&r, (0.0, 2.0), None);
    let fwd = s.forward(&s.u0(), &s.problem.p).unwrap();
    let g = s.gradient(&fwd, &s.problem.p).unwrap();
    assert_eq!(g.p[1], 0.0);
    let plain = rc(
        "toy:forced_decay",
        json!({"leaf": "decay", "alg": "Heun", "dt": 0.1}),
        1.0,
        None,
    );
    let sp = spec(&plain, (0.0, 2.0), None);
    let fp = sp.forward(&sp.u0(), &sp.problem.p).unwrap();
    assert_ne!(fp.final_state()[0], fwd.final_state()[0]);
    // With ω = h = 0.1 the forcing is sin(0.1 t).
    let mut u = 1.0f64;
    let (h, k) = (0.1f64, 2.0f64);
    for n in 0..20 {
        let t = n as f64 * h;
        let f = |u: f64, t: f64| -k * u + (h * t).sin();
        let k1 = f(u, t);
        let k2 = f(u + h * k1, t + h);
        u = u + (h * 0.5) * k1 + (h * 0.5) * k2;
    }
    assert!(
        (u - fwd.final_state()[0]).abs() < 1e-12,
        "{u} {}",
        fwd.final_state()[0]
    );
}

#[test]
fn f32_state() {
    let r = rc(
        "toy:forced_decay",
        json!({"leaf": "decay", "alg": "Tsit5", "dt": 0.05}),
        2.0,
        None,
    );
    let s32: RunSpec<f32> = RunSpec::new(&r, (0.0, 2.0), None).unwrap();
    let s64: RunSpec<f64> = RunSpec::new(&r, (0.0, 2.0), None).unwrap();
    let u32 = s32
        .forward(&s32.u0(), &s32.problem.p)
        .unwrap()
        .final_state()[0];
    let u64 = s64
        .forward(&s64.u0(), &s64.problem.p)
        .unwrap()
        .final_state()[0];
    assert!((u32 as f64 - u64).abs() < 1e-5, "{u32} {u64}");
    // Adaptive in f32: the norm is f64, the state f32.
    let r = rc(
        "toy:van_der_pol/nonstiff",
        json!({"leaf": "oscillator", "alg": "BS3", "control": {"reltol": 1e-4, "abstol": 1e-6}}),
        6.3,
        Some(grad_all("y", "final")),
    );
    let s: RunSpec<f32> = RunSpec::new(&r, (0.0, 6.3), None).unwrap();
    let fwd = s.forward(&s.u0(), &s.problem.p).unwrap();
    let g = s.gradient(&fwd, &s.problem.p).unwrap();
    assert!(g.u0.iter().all(|x| x.is_finite()));
}

/// D61: after a step shortened to land on a stop, the proposal comes from the
/// shortened size, as in OrdinaryDiffEqCore 4.19.0. OrdinaryDiffEq's Tsit5 on
/// van_der_pol/nonstiff at reltol 1e-8, abstol 1e-10 with tstops at 1 and 3
/// takes 170 accepted and 2 rejected steps with 1035 f evaluations.
#[test]
fn tsit5_step_counts_match_ordinarydiffeq() {
    let r = rc(
        "toy:van_der_pol/nonstiff",
        json!({"leaf": "oscillator", "alg": "Tsit5", "control": {"reltol": 1e-8, "abstol": 1e-10}}),
        6.3,
        None,
    );
    let s = spec(&r, (0.0, 6.3), Some(&[1.0, 3.0, 6.3]));
    let f = s.forward(&s.u0(), &s.problem.p).unwrap();
    assert_eq!((f.accepted, f.rejected, f.rhs_calls), (170, 2, 1035));
    // OrdinaryDiffEq's end state, to roundoff.
    let ode = [1.832135188281472, 1.1618990754779062];
    for (a, b) in f.final_state().iter().zip(ode) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }
}
