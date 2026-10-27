use earthsci_integrator::config::RunConfig;
use earthsci_integrator::schedule::RunSpec;
use earthsci_integrator::toy::toy_problem;
use earthsci_integrator::Error;
use serde_json::json;

fn main() {
    // A run config is a JSON object. This one runs the `forced_decay` toy
    // problem (u' = -k u + sin(ωt)) with RK4 at a fixed step of 0.1 over
    // t = 0 … 2.
    let config = json!({
        "esm_integrator": "0.1.0",
        "model": "toy:forced_decay",
        "schedule": {"leaf": "decay", "alg": "RK4", "dt": 0.1},
        "step": {"dt": 2},
        "adjoint": {
            "functional": {"sum": "u", "over": "final"},
            "wrt": ["u0", "params:numeric"]
        }
    });

    // region: toy ops=toy_problem
    let prob = toy_problem::<f64>("toy:forced_decay").unwrap();
    assert_eq!(prob.p, vec![2.0, 3.0]);
    assert_eq!(prob.u0, vec![1.0]);
    // endregion

    // region: forward ops=run_spec,forward_run
    let rc = RunConfig::from_value(&config).unwrap();
    let spec = RunSpec::<f64>::new(&rc, (0.0, 2.0), None).unwrap();
    let (u0, p) = (spec.u0(), spec.problem.p.clone());
    let fwd = spec.forward(&u0, &p).unwrap();
    let u_final = fwd.final_state()[0];
    // Exact solution of u' = -k u + sin(ωt), u(0) = 1.
    let (k, w) = (p[0], p[1]);
    let a = |t: f64| (k * (w * t).sin() - w * (w * t).cos()) / (k * k + w * w);
    let u_exact = (1.0 - a(0.0)) * (-k * 2.0).exp() + a(2.0);
    assert!((u_final - u_exact).abs() < 1e-4 * u_exact.abs());
    // endregion

    // region: gradient ops=gradient_result
    let g = spec.gradient(&fwd, &p).unwrap();
    // The functional is u(2), so ∂J/∂u0 is d u(2) / d u(0) = exp(-k t).
    assert!((g.u0[0] - (-k * 2.0).exp()).abs() < 1e-4 * (-k * 2.0).exp());
    assert_eq!(g.p.len(), p.len());
    // endregion

    // region: replay ops=tape
    let again = spec.replay(&u0, &p, &fwd.tape).unwrap();
    assert_eq!(again.final_state(), fwd.final_state());
    // endregion

    // region: refusal ops=error
    let mut bad = config.clone();
    bad["schedule"]["alg"] = json!("Bogus");
    let rc = RunConfig::from_value(&bad).unwrap();
    match RunSpec::<f64>::new(&rc, (0.0, 2.0), None) {
        Err(Error::Config(msg)) => assert!(msg.contains("Bogus")),
        other => panic!("expected a config error, got {:?}", other.map(|_| ())),
    }
    // endregion

    println!("forward_and_gradient: ok");
}
