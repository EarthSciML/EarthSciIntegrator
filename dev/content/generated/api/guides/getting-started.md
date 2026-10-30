---
title: "Getting started"
description: "Build a run from a run configuration, integrate it, differentiate it with the discrete adjoint and replay it, in Julia and Rust."
weight: 10
api_tabs: true
---

This guide runs one small problem through the library's main steps. The
integrator is specified once (see the [specification](/spec/)) and implemented
natively in Julia and Rust, so each step shows both; pick your language on any
tab and the rest of the site follows. Every snippet is cut from a program that
runs, and checks its own result, in CI.

The problem is the toy problem `forced_decay`: the scalar equation
$u' = -k u + \sin(\omega t)$ with $k = 2$, $\omega = 3$ and $u(0) = 1$, which has an
exact solution. The *toy problems* are hand-written models behind the operator
interface that the conformance suite uses; a real run configuration names an
`.esm` document instead (see [the planner](/generated/api/topics/planner)).

## A problem


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
prob = toy_problem("toy:forced_decay")
@test prob.p == [2.0, 3.0]
@test prob.u0 == [1.0]
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/docs/api/examples/julia/forward_and_gradient.jl#L18-L20">docs/api/examples/julia/forward_and_gradient.jl</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
let prob = toy_problem::<f64>("toy:forced_decay").unwrap();
assert_eq!(prob.p, vec![2.0, 3.0]);
assert_eq!(prob.u0, vec![1.0]);
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/docs/api/examples/rust/forward_and_gradient.rs#L23-L25">docs/api/examples/rust/forward_and_gradient.rs</a></p>

</div>

</div>


## Forward run

A **run configuration** is a JSON object: the model, a `schedule` that says
which operators each leaf integrates and with which method (`RK4` at a fixed
step of 0.1 here), the macro `step`, and optionally an `adjoint` block naming a
functional to differentiate. Building a run checks the configuration, and
refuses what it cannot run; running it forward integrates over the interval and
records the tape: every step, accepted or rejected, with its size.


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
run = build_run(config; tspan = (0.0, 2.0))
forward!(run)
u_final = run.outputs[end][2]
# Exact solution of u' = -k u + sin(ωt), u(0) = 1.
k, ω = prob.p
A(t) = (k * sin(ω * t) - ω * cos(ω * t)) / (k^2 + ω^2)
u_exact = (1 - A(0)) * exp(-k * 2) + A(2)
@test isapprox(only(u_final), u_exact; rtol = 1.0e-4)
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/docs/api/examples/julia/forward_and_gradient.jl#L24-L32">docs/api/examples/julia/forward_and_gradient.jl</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

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


## Gradient

With an `adjoint` block, the run's functional can be differentiated with respect
to the initial state and the parameters. The gradient is the *discrete* adjoint
of the recorded run: it differentiates exactly the steps the forward run took,
not a continuous approximation of them. See the
[adjoint chapter](/spec/adjoint/) for the definition.


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
J, ū0, p̄ = gradient(run)
# The functional is u(2), so ū0 is d u(2) / d u(0) = exp(-k t).
@test isapprox(only(ū0), exp(-k * 2); rtol = 1.0e-4)
@test length(p̄) == length(prob.p)
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/docs/api/examples/julia/forward_and_gradient.jl#L36-L39">docs/api/examples/julia/forward_and_gradient.jl</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
let g = spec.gradient(&fwd, &p).unwrap();
// The functional is u(2), so ∂J/∂u0 is d u(2) / d u(0) = exp(-k t).
assert!((g.u0[0] - (-k * 2.0).exp()).abs() < 1e-4 * (-k * 2.0).exp());
assert_eq!(g.p.len(), p.len());
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/docs/api/examples/rust/forward_and_gradient.rs#L42-L45">docs/api/examples/rust/forward_and_gradient.rs</a></p>

</div>

</div>


## Replay

A run can be replayed on its tape. Every recorded step is held fixed, so the
replay gives the same result bit for bit, and with a perturbed initial state or
parameters it is the map the adjoint differentiates.


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
u_replayed, outputs, q = replay(run)
@test u_replayed == u_final
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/docs/api/examples/julia/forward_and_gradient.jl#L43-L44">docs/api/examples/julia/forward_and_gradient.jl</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
let again = spec.replay(&u0, &p, &fwd.tape).unwrap();
assert_eq!(again.final_state(), fwd.final_state());
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/docs/api/examples/rust/forward_and_gradient.rs#L49-L50">docs/api/examples/rust/forward_and_gradient.rs</a></p>

</div>

</div>


## Refusals

A run configuration that asks for something invalid or unsupported is refused
when the run is built, not partway through the integration.


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
bad = deepcopy(config)
bad["schedule"]["alg"] = "Bogus"
@test_throws ConfigError build_run(bad; tspan = (0.0, 2.0))
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/docs/api/examples/julia/forward_and_gradient.jl#L48-L50">docs/api/examples/julia/forward_and_gradient.jl</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
let mut bad = config.clone();
bad["schedule"]["alg"] = json!("Bogus");
let rc = RunConfig::from_value(&bad).unwrap();
match RunSpec::<f64>::new(&rc, (0.0, 2.0), None) {
    Err(Error::Config(msg)) => assert!(msg.contains("Bogus")),
    other => panic!("expected a config error, got {:?}", other.map(|_| ())),
}
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/docs/api/examples/rust/forward_and_gradient.rs#L54-L60">docs/api/examples/rust/forward_and_gradient.rs</a></p>

</div>

</div>


See [`build_run`](/generated/api/ref/build_run) and
[`run_spec`](/generated/api/ref/run_spec) for the full set of options.
