---
title: "explicit_rk"
description: "Stable type · Solvers"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Solvers](/generated/api/topics/solvers)

| Julia | Rust |
|---|---|
| not exported | `earthsci_integrator_solvers::erk::ExplicitRk` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this type.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct ExplicitRk {
    pub name: String,
    pub form: ErkForm,
    pub error: ErkError,
    pub stages: usize,
    pub order: u32,
    pub embedded_order: Option<u32>,
    pub fsal: bool,
    pub last_stage_is_end: bool,
    pub end_eval: bool,
}

impl ExplicitRk { pub fn from_tableau(t: &Tableau) -> Result<Self, String> }

impl ExplicitRk { pub fn adaptive(&self) -> bool }

impl ExplicitRk { pub fn stage_time(&self, i: usize, t: f64, h: f64) -> f64 }

impl ExplicitRk { pub fn workspace<T: Scalar>(&self, n: usize) -> ErkWork<T> }

impl ExplicitRk { pub fn step<T: Scalar>(&self, w: &mut ErkWork<T>, f: &mut RhsFn<'_, T>, un: &[T], t: f64, h: f64, k1: Option<&[T]>, norm: Option<&mut NormFn<'_, T>>) -> Option<f64> }
```

An explicit Runge–Kutta method built from its table.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/rust/earthsci-integrator-solvers/src/erk.rs#L54) · [Rust docs](https://docs.rs/earthsci-integrator-solvers/latest/earthsci_integrator_solvers/erk/struct.ExplicitRk.html)

</div>

</div>
