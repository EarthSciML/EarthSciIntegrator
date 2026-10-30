---
title: "rosenbrock"
description: "Stable type · Solvers"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Solvers](/generated/api/topics/solvers)

| Julia | Rust |
|---|---|
| not exported | `earthsci_integrator_solvers::rosenbrock::Rosenbrock` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this type.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct Rosenbrock {
    pub name: String,
    pub form: RosForm,
    pub stages: usize,
    pub order: u32,
    pub embedded_order: Option<u32>,
    pub fsal: bool,
}

impl Rosenbrock { pub fn from_tableau(t: &Tableau) -> Result<Self, String> }

impl Rosenbrock { pub fn adaptive(&self) -> bool }

impl Rosenbrock { pub fn uses_ft(&self) -> bool }

impl Rosenbrock { pub fn w_gamma(&self) -> f64 }

impl Rosenbrock { pub fn stage_time(&self, i: usize, t: f64, h: f64) -> f64 }

impl Rosenbrock { pub fn workspace<T: Scalar>(&self, n: usize, structure: Structure, linsolve: LinsolveKind) -> Result<RosWork<T>, String> }

impl Rosenbrock { pub fn factor_w<T: Scalar>(&self, w: &mut RosWork<T>, jac: &mut JacFn<'_, T>, un: &[T], t: f64, h: f64) -> Option<f64> }

impl Rosenbrock { pub fn step<T: Scalar>(&self, w: &mut RosWork<T>, f: &mut RhsFn<'_, T>, jac: &mut JacFn<'_, T>, ft: Option<&mut RhsFn<'_, T>>, un: &[T], t: f64, h: f64, f0: Option<&[T]>, norm: Option<&mut NormFn<'_, T>>) -> RosOutcome }
```

A Rosenbrock method built from its table.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/rust/earthsci-integrator-solvers/src/rosenbrock.rs#L67) · [Rust docs](https://docs.rs/earthsci-integrator-solvers/latest/earthsci_integrator_solvers/rosenbrock/struct.Rosenbrock.html)

</div>

</div>
