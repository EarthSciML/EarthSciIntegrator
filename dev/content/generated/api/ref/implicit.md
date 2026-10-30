---
title: "implicit"
description: "Stable type · Solvers"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Solvers](/generated/api/topics/solvers)

| Julia | Rust |
|---|---|
| not exported | `earthsci_integrator_solvers::implicit::Implicit` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this type.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct Implicit {
    pub name: String,
    pub form: ImplicitForm,
    pub stages: usize,
    pub order: u32,
    pub embedded_order: Option<u32>,
}

impl Implicit { pub fn from_tableau(t: &Tableau, theta: Option<f64>, linearized: bool, has_explicit: bool) -> Result<Self, String> }

impl Implicit { pub fn adaptive(&self) -> bool }

impl Implicit { pub fn is_imex(&self) -> bool }

impl Implicit { pub fn w_gamma(&self) -> Option<f64> }

impl Implicit { pub fn w_part(&self) -> Part }

impl Implicit { pub fn workspace<T: Scalar>(&self, n: usize, structure: Structure, linsolve: LinsolveKind) -> Result<ImplicitWork<T>, String> }

impl Implicit { pub fn factor<T: Scalar>(&self, w: &mut ImplicitWork<T>, jac: &mut PartJacFn<'_, T>, part: Part, u: &[T], t: f64, h: f64) -> Option<f64> }

impl Implicit { pub fn step<T: Scalar>(&self, w: &mut ImplicitWork<T>, f: &mut PartRhsFn<'_, T>, jac: &mut PartJacFn<'_, T>, un: &[T], t: f64, h: f64, mode: NewtonMode, want_err: bool, norm: &mut NormFn<'_, T>) -> ImplicitOutcome }
```

An implicit method built from its table.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/rust/earthsci-integrator-solvers/src/implicit.rs#L126) · [Rust docs](https://docs.rs/earthsci-integrator-solvers/latest/earthsci_integrator_solvers/implicit/struct.Implicit.html)

</div>

</div>
