---
title: "erk_error"
description: "Stable error · Errors"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** error · **Topic:** [Errors](/generated/api/topics/errors)

| Julia | Rust |
|---|---|
| not exported | `earthsci_integrator_solvers::erk::ErkError` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this error.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub enum ErkError {
    None,
    Embedded { btilde: Vec<f64> },
    ShuOsherEmbedded { alpha: Vec<f64>, beta: Vec<f64>, scale: f64 },
    ResidualControl { sigma: Vec<f64>, scale: f64 },
}
```

The error estimate (spec/methods.md §3.2).

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/rust/earthsci-integrator-solvers/src/erk.rs#L33) · [Rust docs](https://docs.rs/earthsci-integrator-solvers/latest/earthsci_integrator_solvers/erk/enum.ErkError.html)

</div>

</div>
