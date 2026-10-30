---
title: "toy_problem (type)"
description: "Stable type · Building and running"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Building and running](/generated/api/topics/runs)

| Julia | Rust |
|---|---|
| not exported | `toy::ToyProblem` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this type.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct ToyProblem<T: Scalar> {
    pub name: String,
    pub n: usize,
    pub param_names: Vec<&'static str>,
    pub p: Vec<f64>,
    pub u0: Vec<f64>,
    pub tspan: (f64, f64),
    pub variables: Vec<&'static str>,
    pub cells: Option<Vec<usize>>,
    pub operators: Vec<Arc<dyn Operator<T>>>,
    pub maps: Vec<Arc<dyn MapOperator<T>>>,
    pub invariants: Vec<Vec<(usize, f64)>>,
}

impl<T: Scalar> ToyProblem<T> { pub fn operator(&self, name: &str) -> Option<Arc<dyn Operator<T>>> }

impl<T: Scalar> ToyProblem<T> { pub fn map(&self, name: &str) -> Option<Arc<dyn MapOperator<T>>> }
```

A toy problem: its operators, layout, parameters, initial state,
interval and the facts the conformance checks use.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/rust/earthsci-integrator/src/toy.rs#L14) · [Rust docs](https://docs.rs/earthsci-integrator/latest/earthsci_integrator/toy/struct.ToyProblem.html)

**Example** (`toy`)

```rust
let prob = toy_problem::<f64>("toy:forced_decay").unwrap();
assert_eq!(prob.p, vec![2.0, 3.0]);
assert_eq!(prob.u0, vec![1.0]);
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/docs/api/examples/rust/forward_and_gradient.rs#L23-L25">docs/api/examples/rust/forward_and_gradient.rs</a></p>

</div>

</div>
