---
title: "tableau (type)"
description: "Stable type · Solvers"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Solvers](/generated/api/topics/solvers)

| Julia | Rust |
|---|---|
| not exported | `earthsci_integrator_solvers::tableau::Tableau` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this type.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct Tableau {
    pub schema: String,
    pub name: String,
    pub family: Family,
    pub form: Form,
    pub ordinarydiffeq_algorithm: Option<String>,
    pub stages: usize,
    pub order: u32,
    pub embedded_order: Option<u32>,
    pub error_estimate: ErrorEstimate,
    pub first_same_as_last: bool,
    pub stiffly_accurate: bool,
    pub explicit_first_stage: Option<bool>,
    pub a: Option<Vec<Vec<f64>>>,
    pub c_mat: Option<Vec<Vec<f64>>>,
    pub b: Option<Vec<f64>>,
    pub btilde: Option<Vec<f64>>,
    pub c: Option<Vec<f64>>,
    pub c_ordinarydiffeq: Option<Vec<f64>>,
    pub d: Option<DCoefficients>,
    pub gamma: Option<f64>,
    pub c32: Option<f64>,
    pub alpha: Option<Vec<Vec<f64>>>,
    pub beta: Option<Vec<Vec<f64>>>,
    pub embedded: Option<ShuOsherEmbedded>,
    pub residual_control: Option<ResidualControl>,
    pub divided_difference: Option<DividedDifference>,
    pub kpp: Option<Kpp>,
    pub controller: Option<ControllerConstants>,
    pub source: Value,
    pub notes: Vec<String>,
}
```

One coefficient table, as `spec/schemas/tableau.schema.json` defines it.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/rust/earthsci-integrator-solvers/src/tableau.rs#L124) · [Rust docs](https://docs.rs/earthsci-integrator-solvers/latest/earthsci_integrator_solvers/tableau/struct.Tableau.html)

</div>

</div>
