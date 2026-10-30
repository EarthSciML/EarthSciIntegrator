---
title: "toy_problem (function)"
description: "Stable function · Building and running"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Building and running](/generated/api/topics/runs)

| Julia | Rust |
|---|---|
| `toy_problem` | `toy::toy_problem` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
toy_problem(model::AbstractString)
```

```julia
toy_problem(model) -> ToyProblem
```

The toy problem a conformance case's `model` names: `toy:<problem>` or
`toy:<problem>/<variant>` (toy-problems.md §2). Accepted only from the test
harness; a real run config names an `.esm` document.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/toys.jl#L593)

**Example** (`toy`)

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
pub fn toy_problem<T: Scalar>(model: &str) -> Result<ToyProblem<T>, Error>
```

Builds the toy problem a case's `model: "toy:<problem>[/<variant>]"` names.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/rust/earthsci-integrator/src/toy.rs#L51) · [Rust docs](https://docs.rs/earthsci-integrator/latest/earthsci_integrator/toy/fn.toy_problem.html)

**Example** (`toy`)

```rust
let prob = toy_problem::<f64>("toy:forced_decay").unwrap();
assert_eq!(prob.p, vec![2.0, 3.0]);
assert_eq!(prob.u0, vec![1.0]);
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/docs/api/examples/rust/forward_and_gradient.rs#L23-L25">docs/api/examples/rust/forward_and_gradient.rs</a></p>

</div>

</div>
