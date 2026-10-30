---
title: "run_spec"
description: "Stable type · Building and running"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Building and running](/generated/api/topics/runs)

| Julia | Rust |
|---|---|
| not exported | `schedule::RunSpec` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this type.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct RunSpec<T: Scalar> {
    pub problem: ToyProblem<T>,
    pub schedule: Schedule<T>,
    pub tspan: (f64, f64),
    pub macro_dt: f64,
    pub output_times: Vec<f64>,
    pub bounds: Vec<f64>,
    pub reasons: Vec<&'static str>,
    pub functional: Option<FunctionalSpec>,
    pub wrt_u0: bool,
    pub wrt_p: Vec<bool>,
}

impl<T: Scalar> RunSpec<T> { pub fn functional_value(&self, run: &ForwardRun<T>) -> Result<f64, Error> }

impl<T: Scalar> RunSpec<T> { pub fn gradient(&self, run: &ForwardRun<T>, p: &[f64]) -> Result<Gradient, Error> }

impl<T: Scalar> RunSpec<T> { pub fn new(rc: &RunConfig, tspan: (f64, f64), saveat: Option<&[f64]>) -> Result<Self, Error> }

impl<T: Scalar> RunSpec<T> { pub fn u0(&self) -> Vec<T> }

impl<T: Scalar> RunSpec<T> { pub fn forward(&self, u0: &[T], p: &[f64]) -> Result<ForwardRun<T>, Error> }

impl<T: Scalar> RunSpec<T> { pub fn replay(&self, u0: &[T], p: &[f64], tape: &Tape) -> Result<ForwardRun<T>, Error> }

impl<T: Scalar> RunSpec<T> { pub fn replay_frozen(&self, u0: &[T], p: &[f64], tape: &Tape) -> Result<ForwardRun<T>, Error> }

impl<T: Scalar> RunSpec<T> { pub fn macro_step(&self, k: usize, u: &mut [T], persist: &mut [LeafPersist], p: &[f64], t0: f64, t1: f64, delta: f64, replay: Option<&HashMap<&str, &TapeRecord>>, adj: Option<&mut Vec<AdjRecord<T>>>) -> Result<MacroOut<T>, Error> }
```

A configured run: problem, schedule, interval, macro step, outputs and
the adjoint settings.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/rust/earthsci-integrator/src/schedule.rs#L933) · [Rust docs](https://docs.rs/earthsci-integrator/latest/earthsci_integrator/schedule/struct.RunSpec.html)

**Example** (`forward`)

```rust
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
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/docs/api/examples/rust/forward_and_gradient.rs#L29-L38">docs/api/examples/rust/forward_and_gradient.rs</a></p>

</div>

</div>
