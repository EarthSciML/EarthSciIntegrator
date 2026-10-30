---
title: "controller"
description: "Stable type · Solvers"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Solvers](/generated/api/topics/solvers)

| Julia | Rust |
|---|---|
| not exported | `earthsci_integrator_solvers::controller::Controller` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this type.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct Controller {
    pub kind: ControllerKind,
    pub beta1: f64,
    pub beta2: f64,
    pub beta3: f64,
    pub gamma: f64,
    pub qmin: f64,
    pub qmax: f64,
    pub qmax_first_step: f64,
    pub qsteady_min: f64,
    pub qsteady_max: f64,
    pub failfactor: f64,
    pub qoldinit: f64,
    pub accept_safety: f64,
    pub order: u32,
    pub embedded_order: u32,
}

impl Controller { pub fn new(kind: ControllerKind, t: &Tableau) -> Option<Self> }

impl Controller { pub fn initial_state(&self) -> ControllerState }

impl Controller { pub fn evaluate(&self, s: &ControllerState, e: f64) -> Evaluation }

impl Controller { pub fn accept(&self, s: &mut ControllerState, ev: &Evaluation, e: f64, h_base: f64) -> f64 }

impl Controller { pub fn reject(&self, ev: &Evaluation, h: f64) -> f64 }
```

Controller constants plus the method's orders.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/rust/earthsci-integrator-solvers/src/controller.rs#L38) · [Rust docs](https://docs.rs/earthsci-integrator-solvers/latest/earthsci_integrator_solvers/controller/struct.Controller.html)

</div>

</div>
