//! Tests of the SDIRK, θ-method and IMEX Euler leaves and their adjoints
//! beyond what cases.json covers: every method's adjoint at fixed steps,
//! the adaptive SDIRK methods under each controller, the integral
//! functional, IMEX partitions with an explicit part and the linearized
//! solve, Newton failure handling, refusals and f32 state.

use earthsci_integrator::config::RunConfig;
use earthsci_integrator::conformance::{check_fd, check_invariant, check_replay, Outcome};
use earthsci_integrator::schedule::RunSpec;
use earthsci_integrator::toy::forced_decay_exact;
use earthsci_integrator::Error;
use serde_json::{json, Value};

const SDIRK: [&str; 5] = ["BackwardEuler", "TRBDF2", "SDIRK2", "Kvaerno3", "Kvaerno4"];

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

/// The implicit leaves as schedules: every SDIRK table, the θ-method at
/// several θ (0 is forward Euler, 1 backward Euler), and IMEX Euler with
/// everything implicit, converged and linearized.
///
/// The Newton iteration converges to the leaf's tolerances, and the adjoint
/// is exact at the converged stages, so against finite differences of
/// exactly converged stages it agrees only to the Newton tolerance
/// (spec/adjoint.md §4.3): BackwardEuler on van der Pol at the default
/// `reltol = 1e-3` measures 1.25e-6. Tight tolerances keep that below the
/// check's 1e-6.
fn implicit_leaves(op: &str, dt: f64) -> Vec<(String, Value)> {
    let ctl = json!({"reltol": 1e-8, "abstol": 1e-10});
    let mut v: Vec<(String, Value)> = SDIRK
        .iter()
        .map(|a| {
            (
                a.to_string(),
                json!({"leaf": op, "alg": a, "dt": dt, "control": ctl}),
            )
        })
        .collect();
    for th in [0.0, 0.3, 0.5, 1.0] {
        v.push((
            format!("Theta {th}"),
            json!({"leaf": op, "alg": "Theta", "theta": th, "dt": dt, "control": ctl}),
        ));
    }
    for solve in ["converged", "linearized"] {
        v.push((
            format!("IMEXEuler {solve}"),
            json!({"leaf": op, "alg": "IMEXEuler", "imex": {"implicit": op},
                   "implicit_solve": solve, "dt": dt, "control": ctl}),
        ));
    }
    v
}

/// Every implicit method's adjoint at fixed dt, on van der Pol (nonlinear,
/// so the transposed systems at the converged stages differ from the
/// forward W at u_n, and linearized IMEX Euler's second-order term is
/// nonzero) and on forced_decay (stage times), with point terms at several
/// output times.
#[test]
fn every_implicit_method_adjoint_fixed_dt() {
    for (what, sch) in implicit_leaves("oscillator", 0.05) {
        let r = rc(
            "toy:van_der_pol/nonstiff",
            sch,
            3.0,
            Some(grad_all("y", "final")),
        );
        check_gradient(&r, (0.0, 3.0), None, &what);
    }
    for (what, sch) in implicit_leaves("decay", 0.1) {
        let r = rc("toy:forced_decay", sch, 1.0, Some(grad_all("u", "saveat")));
        check_gradient(&r, (0.0, 2.0), Some(&[1.0, 2.0]), &what);
    }
}

/// Every SDIRK table under each controller on the stiff van der Pol
/// oscillator: accuracy against a tight Rodas5P run, and the adjoint.
#[test]
fn every_sdirk_method_adaptive_and_controllers() {
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
    for alg in SDIRK {
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
                "{alg} {ctl}: {} accepted, {} rejected, {} rhs calls, error {err:.2e}",
                fwd.accepted, fwd.rejected, fwd.rhs_calls
            );
            // spec/methods.md §5.5: SDIRK2 and Kvaerno3/4 are unreliable on
            // stiff problems at tight tolerances (D59); this is a loose bound.
            assert!(err < 1e-2, "{alg} {ctl}: error {err}");
            let g = s.gradient(&fwd, &s.problem.p).unwrap();
            assert_pass(
                check_fd(&s, &fwd, &g, 1e-6, &[1e-3, 1e-4, 1e-5]),
                &format!("{alg} {ctl} fd"),
            );
            assert_pass(
                check_replay(&s, &fwd, &s.u0(), &s.problem.p),
                &format!("{alg} {ctl} replay"),
            );
        }
    }
}

/// BackwardEuler's divided-difference estimate drives an adaptive leaf to
/// roughly the tolerance on forced_decay, with several invocations (the
/// history resets at each, spec/methods.md §5.5).
#[test]
fn backward_euler_adaptive() {
    for macro_dt in [2.0, 0.5] {
        let r = rc(
            "toy:forced_decay",
            json!({"leaf": "decay", "alg": "BackwardEuler",
                   "control": {"reltol": 1e-4, "abstol": 1e-6}}),
            macro_dt,
            Some(grad_all("u", "final")),
        );
        let s = spec(&r, (0.0, 2.0), None);
        let fwd = s.forward(&s.u0(), &s.problem.p).unwrap();
        let exact = forced_decay_exact(2.0, s.problem.u0[0], &s.problem.p);
        let err = (fwd.final_state()[0] - exact).abs();
        println!(
            "BackwardEuler, macro step {macro_dt}: {} accepted, {} rejected, error {err:.2e}",
            fwd.accepted, fwd.rejected
        );
        assert!(err < 1e-2 && fwd.accepted > 20, "error {err}");
        check_gradient(&r, (0.0, 2.0), None, "BackwardEuler adaptive");
    }
}

/// The integral term ∫ψ dt (spec/adjoint.md §2.1) on SDIRK and θ-method
/// leaves: the forward increment against the exact integral, and the adjoint.
#[test]
fn integral_functional() {
    // ∫₀² u dt for forced_decay, by Simpson's rule on the exact solution.
    let pb = earthsci_integrator::toy::toy_problem::<f64>("toy:forced_decay").unwrap();
    let n = 20000;
    let hq = 2.0 / n as f64;
    let mut exact = 0.0;
    for i in 0..=n {
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        exact += w * forced_decay_exact(i as f64 * hq, pb.u0[0], &pb.p);
    }
    exact *= hq / 3.0;
    let mut cases: Vec<(String, Value)> = SDIRK
        .iter()
        .map(|a| {
            (
                a.to_string(),
                json!({"leaf": "decay", "alg": a, "dt": 0.01}),
            )
        })
        .collect();
    cases.push((
        "Theta 0.5".into(),
        json!({"leaf": "decay", "alg": "Theta", "theta": 0.5, "dt": 0.01}),
    ));
    cases.push((
        "Theta 0.7".into(),
        json!({"leaf": "decay", "alg": "Theta", "theta": 0.7, "dt": 0.01}),
    ));
    cases.push((
        "TRBDF2 adaptive".into(),
        json!({"leaf": "decay", "alg": "TRBDF2", "control": {"reltol": 1e-8, "abstol": 1e-10}}),
    ));
    for (what, sch) in cases {
        let r = rc("toy:forced_decay", sch, 2.0, Some(grad_all("u", "time")));
        let s = spec(&r, (0.0, 2.0), None);
        let fwd = s.forward(&s.u0(), &s.problem.p).unwrap();
        let e = (fwd.integral - exact).abs();
        println!("{what}: ∫u dt = {} (error {e:.2e})", fwd.integral);
        // BackwardEuler is first order: 2.5e-3 at dt 0.01.
        assert!(e < 5e-3, "{what}: integral error {e}");
        check_gradient(&r, (0.0, 2.0), None, &format!("{what} integral"));
    }
}

/// IMEX Euler with an explicit part: transport explicit, reaction implicit,
/// converged and linearized (which coincide here, since reaction is linear
/// in the state), on advection_reaction; the invariant and the adjoint.
#[test]
fn imex_partition_with_explicit_part() {
    let mut ends = Vec::new();
    for (imex, solve) in [
        (json!({"implicit": "reaction"}), "converged"),
        (
            json!({"implicit": "reaction", "explicit": ["transport"]}),
            "converged",
        ),
        (json!({"implicit": ["reaction"]}), "linearized"),
    ] {
        let r = rc(
            "toy:advection_reaction",
            json!({"leaf": ["transport", "reaction"], "alg": "IMEXEuler", "imex": imex,
                   "implicit_solve": solve, "dt": 0.005}),
            0.1,
            Some(grad_all("B", "final")),
        );
        let s = spec(&r, (0.0, 0.2), None);
        let fwd = s.forward(&s.u0(), &s.problem.p).unwrap();
        assert_pass(
            check_invariant(&s, &fwd, 1e-12),
            &format!("IMEX {solve} invariant"),
        );
        ends.push(fwd.final_state().to_vec());
        check_gradient(&r, (0.0, 0.2), None, &format!("IMEX {solve}"));
    }
    // Converged and linearized differ only at the Newton tolerance here.
    let d = ends[0]
        .iter()
        .zip(&ends[2])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(d < 1e-6, "converged vs linearized: {d}");
    assert_eq!(ends[0], ends[1]);
}

/// A Newton failure rejects an adaptive step and halves it (spec/methods.md
/// §5.2 step 8), and ends a fixed-step leaf with ConvergenceFailure (§12).
#[test]
fn newton_failure() {
    // Robertson at a step of 0.5 from its initial state: the modified Newton
    // iteration with J at u_n does not converge in 10 iterations.
    let r = rc(
        "toy:robertson",
        json!({"leaf": "kinetics", "alg": "TRBDF2", "dt": 0.5}),
        1.0,
        None,
    );
    let s = spec(&r, (0.0, 1.0), None);
    match s.forward(&s.u0(), &s.problem.p) {
        Err(Error::Run(m)) => {
            assert!(m.contains("ConvergenceFailure"), "{m}");
            println!("fixed dt: {m}");
        }
        o => panic!(
            "expected ConvergenceFailure, got {:?}",
            o.map(|f| f.accepted)
        ),
    }
    // Adaptive BackwardEuler from a first step of 0.5. The first step of an
    // invocation has E = 1 exactly (spec/methods.md §5.5), so its error
    // estimate cannot reject it, and no domain is set: every rejection
    // before the first accepted step is a Newton failure. Each halves the
    // step (failfactor 2), so the first accepted step is 0.5/2^k after k
    // rejections, where any controller rejection would scale by its own factor.
    let r = rc(
        "toy:robertson",
        json!({"leaf": "kinetics", "alg": "BackwardEuler",
               "control": {"reltol": 1e-6, "abstol": 1e-10, "dt0": 0.5}}),
        1.0,
        Some(grad_all("y3", "final")),
    );
    let s = spec(&r, (0.0, 1.0), None);
    let fwd = s.forward(&s.u0(), &s.problem.p).unwrap();
    let h1 = fwd
        .tape
        .records
        .iter()
        .find_map(|rec| match rec {
            earthsci_integrator::tape::TapeRecord::Leaf { steps, .. } => Some(steps[0][1]),
            _ => None,
        })
        .unwrap();
    let k = (0.5 / h1).log2();
    println!(
        "adaptive from dt0 0.5: {} accepted, {} rejected, first accepted step {h1} = 0.5/2^{k}",
        fwd.accepted, fwd.rejected
    );
    assert!(k >= 1.0 && k.fract() == 0.0, "first step {h1}");
    assert!(fwd.rejected >= k as u64);
    check_gradient(&r, (0.0, 1.0), None, "BackwardEuler after Newton failures");
}

#[test]
fn implicit_refusals() {
    let build = |sch: Value, adj: Option<Value>| {
        RunSpec::<f64>::new(&rc("toy:forced_decay", sch, 1.0, adj), (0.0, 1.0), None)
    };
    let config_err = |r: Result<RunSpec<f64>, Error>, what: &str| match r {
        Err(Error::Config(m)) => println!("{what}: {m}"),
        Err(e) => panic!("{what}: {e}"),
        Ok(_) => panic!("{what}: accepted"),
    };
    config_err(
        build(json!({"leaf": "decay", "alg": "Theta", "dt": 0.1}), None),
        "Theta without theta",
    );
    config_err(
        build(
            json!({"leaf": "decay", "alg": "Theta", "theta": 1.5, "dt": 0.1}),
            None,
        ),
        "theta outside [0, 1]",
    );
    config_err(
        build(
            json!({"leaf": "decay", "alg": "TRBDF2", "theta": 0.5, "dt": 0.1}),
            None,
        ),
        "theta on TRBDF2",
    );
    config_err(
        build(
            json!({"leaf": "decay", "alg": "TRBDF2", "dt": 0.1, "imex": {"implicit": "decay"}}),
            None,
        ),
        "imex on TRBDF2",
    );
    config_err(
        build(
            json!({"leaf": "decay", "alg": "IMEXEuler", "dt": 0.1}),
            None,
        ),
        "IMEXEuler without imex",
    );
    config_err(
        build(
            json!({"leaf": "decay", "alg": "IMEXEuler", "dt": 0.1, "imex": {"implicit": []}}),
            None,
        ),
        "IMEXEuler with an empty implicit part",
    );
    config_err(
        build(
            json!({"leaf": "decay", "alg": "IMEXEuler", "dt": 0.1, "imex": {"implicit": "nope"}}),
            None,
        ),
        "imex naming an unknown operator",
    );
    config_err(
        build(
            json!({"leaf": "decay", "alg": "IMEXEuler", "dt": 0.1,
                     "imex": {"implicit": "decay", "explicit": "decay"}}),
            None,
        ),
        "an operator in both parts",
    );
    config_err(
        build(
            json!({"leaf": "decay", "alg": "IMEXEuler", "dt": 0.1,
                     "imex": {"implicit": "decay"}, "implicit_solve": "exact"}),
            None,
        ),
        "unknown implicit_solve",
    );
    config_err(
        build(json!({"leaf": "decay", "alg": "Theta", "theta": 0.5}), None),
        "Theta without dt (D52)",
    );
    match build(
        json!({"leaf": "decay", "alg": "IMEXEuler", "dt": 0.1, "imex": {"implicit": "decay"}}),
        Some(grad_all("u", "time")),
    ) {
        Err(Error::Unimplemented(m)) => println!("IMEX integral: {m}"),
        Err(e) => panic!("{e}"),
        Ok(_) => panic!("IMEX integral accepted"),
    }
}

/// f32 state: Robertson with TRBDF2 runs, keeps its invariant to f32
/// precision and gives an f64 gradient close to the f64 run's.
#[test]
fn f32_state() {
    let r = rc(
        "toy:robertson",
        json!({"leaf": "kinetics", "alg": "TRBDF2", "control": {"reltol": 1e-4, "abstol": 1e-8}}),
        40.0,
        Some(grad_all("y3", "final")),
    );
    let s32: RunSpec<f32> = RunSpec::new(&r, (0.0, 40.0), None).unwrap();
    let s64: RunSpec<f64> = RunSpec::new(&r, (0.0, 40.0), None).unwrap();
    let f32run = s32.forward(&s32.u0(), &s32.problem.p).unwrap();
    let f64run = s64.forward(&s64.u0(), &s64.problem.p).unwrap();
    let d = (f32run.final_state()[2] as f64 - f64run.final_state()[2]).abs();
    println!("f32 vs f64 y3: {d:.2e}");
    assert!(d < 1e-3);
    let g32 = s32.gradient(&f32run, &s32.problem.p).unwrap();
    let g64 = s64.gradient(&f64run, &s64.problem.p).unwrap();
    for (a, b) in g32.u0.iter().zip(&g64.u0) {
        assert!((a - b).abs() < 1e-2 * b.abs().max(1.0), "{a} vs {b}");
    }
    // Parameter gradients: each within 1% of its f64 value, or of 1e-3 of
    // the largest where it is much smaller (k2 is 3e7, so ∂y3/∂k2 is tiny).
    let pmax = g64.p.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    assert!(pmax > 0.0);
    for (i, (a, b)) in g32.p.iter().zip(&g64.p).enumerate() {
        println!("f32 vs f64 ∂y3/∂p[{i}]: {a:.6e} vs {b:.6e}");
        assert!(
            (a - b).abs() < 1e-2 * b.abs().max(1e-3 * pmax),
            "p[{i}]: {a} vs {b}"
        );
    }
}

/// van der Pol split into a linear rotation `y' = z, z' = −y`, which
/// provides no `jac` (an explicit part needs none), and a nonlinear damping
/// `z' = μ(1 − y²) z`, which provides every call.
mod split_vdp {
    use earthsci_integrator::operator::{Call, MatrixStorage, MuPair, Operator, Structure};

    pub struct Rotation;
    impl Operator<f64> for Rotation {
        fn name(&self) -> &str {
            "rotation"
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
            vec![0, 1]
        }
        fn provides(&self, call: Call) -> bool {
            matches!(call, Call::Rhs | Call::JvpU | Call::VjpU | Call::VjpP)
        }
        fn rhs(&self, out: &mut [f64], u: &[f64], _p: &[f64], _t: f64) {
            out[0] += u[1];
            out[1] += -u[0];
        }
        fn jvp_u(&self, out: &mut [f64], _u: &[f64], _p: &[f64], _t: f64, v: &[f64]) {
            out[0] += v[1];
            out[1] += -v[0];
        }
        fn vjp_u(&self, out: &mut [f64], _u: &[f64], _p: &[f64], _t: f64, w: &[f64]) {
            out[0] += -w[1];
            out[1] += w[0];
        }
        fn vjp_p(&self, _out: &mut [f64], _u: &[f64], _p: &[f64], _t: f64, _w: &[f64]) {}
    }

    pub struct Damping;
    impl Operator<f64> for Damping {
        fn name(&self) -> &str {
            "damping"
        }
        fn structure(&self) -> Structure {
            Structure::Dense
        }
        fn depends_on_t(&self) -> bool {
            false
        }
        fn linear_in_u(&self) -> bool {
            false
        }
        fn writes(&self) -> Vec<usize> {
            vec![1]
        }
        fn provides(&self, _call: Call) -> bool {
            true
        }
        fn rhs(&self, out: &mut [f64], u: &[f64], p: &[f64], _t: f64) {
            out[1] += p[0] * (1.0 - u[0] * u[0]) * u[1];
        }
        fn jvp_u(&self, out: &mut [f64], u: &[f64], p: &[f64], _t: f64, v: &[f64]) {
            let (y, z, m) = (u[0], u[1], p[0]);
            out[1] += -2.0 * m * y * z * v[0] + m * (1.0 - y * y) * v[1];
        }
        fn vjp_u(&self, out: &mut [f64], u: &[f64], p: &[f64], _t: f64, w: &[f64]) {
            let (y, z, m) = (u[0], u[1], p[0]);
            out[0] += -2.0 * m * y * z * w[1];
            out[1] += m * (1.0 - y * y) * w[1];
        }
        fn vjp_p(&self, out: &mut [f64], u: &[f64], _p: &[f64], _t: f64, w: &[f64]) {
            out[0] += (1.0 - u[0] * u[0]) * u[1] * w[1];
        }
        fn jac(&self, out: &mut MatrixStorage<f64>, u: &[f64], p: &[f64], _t: f64) {
            let (y, z, m) = (u[0], u[1], p[0]);
            out.add(1, 0, -2.0 * m * y * z);
            out.add(1, 1, m * (1.0 - y * y));
        }
        fn ft(&self, _out: &mut [f64], _u: &[f64], _p: &[f64], _t: f64) {}
        fn hess_u(&self, out: &mut [f64], u: &[f64], p: &[f64], _t: f64, mu: &[MuPair<'_, f64>]) {
            let (y, z, m) = (u[0], u[1], p[0]);
            for (mi, wi) in mu {
                out[0] += mi[1] * (-2.0 * m * z * wi[0] - 2.0 * m * y * wi[1]);
                out[1] += mi[1] * (-2.0 * m * y * wi[0]);
            }
        }
        fn hess_p(&self, out: &mut [f64], u: &[f64], _p: &[f64], _t: f64, mu: &[MuPair<'_, f64>]) {
            let (y, z) = (u[0], u[1]);
            for (mi, wi) in mu {
                out[0] += mi[1] * (-2.0 * y * z * wi[0] + (1.0 - y * y) * wi[1]);
            }
        }
        fn ft_vjp_u(&self, _o: &mut [f64], _u: &[f64], _p: &[f64], _t: f64, _n: &[f64]) {}
        fn ft_vjp_p(&self, _o: &mut [f64], _u: &[f64], _p: &[f64], _t: f64, _n: &[f64]) {}
    }
}

/// IMEX Euler with an explicit part and a nonlinear implicit part, converged
/// and linearized: the explicit part's `J_Eᵀ`, the transposed solves and, for
/// the linearized step, the second-order term are checked together. The
/// explicit operator provides no `jac`, which an IMEX leaf must accept.
#[test]
fn imex_nonlinear_implicit_part() {
    use earthsci_integrator::schedule::{Needs, Schedule};
    use std::sync::Arc;
    for solve in ["converged", "linearized"] {
        let r = rc(
            "toy:van_der_pol/nonstiff",
            json!({"leaf": ["rotation", "damping"], "alg": "IMEXEuler",
                   "imex": {"implicit": "damping"}, "implicit_solve": solve, "dt": 0.05,
                   "control": {"reltol": 1e-8, "abstol": 1e-10}}),
            3.0,
            Some(grad_all("y", "final")),
        );
        // The van der Pol problem with the two split operators in place of
        // `oscillator`; RunSpec::new builds its schedule from `oscillator`,
        // which is then rebuilt against the split operators.
        let base = rc(
            "toy:van_der_pol/nonstiff",
            json!({"leaf": "oscillator", "alg": "IMEXEuler",
                   "imex": {"implicit": "oscillator"}, "dt": 0.05}),
            3.0,
            Some(grad_all("y", "final")),
        );
        let mut s = spec(&base, (0.0, 3.0), None);
        s.problem.operators = vec![Arc::new(split_vdp::Rotation), Arc::new(split_vdp::Damping)];
        let needs = Needs {
            adjoint: true,
            wrt_p: true,
        };
        s.schedule = Schedule::build(&r.schedule_node().unwrap(), &s.problem, needs).unwrap();
        let u0 = s.u0();
        let p = s.problem.p.clone();
        let fwd = s.forward(&u0, &p).unwrap();
        let g = s.gradient(&fwd, &p).unwrap();
        println!(
            "IMEX {solve} split van der Pol: gradient {:?} {:?}",
            g.u0, g.p
        );
        assert!(g.p[0] != 0.0);
        assert_pass(
            check_fd(&s, &fwd, &g, 1e-6, &[1e-3, 1e-4, 1e-5]),
            &format!("IMEX {solve} split van der Pol fd"),
        );
        assert_pass(
            check_replay(&s, &fwd, &u0, &p),
            &format!("IMEX {solve} split van der Pol replay"),
        );
    }
}
