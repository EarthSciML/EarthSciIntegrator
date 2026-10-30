---
title: "run_config"
description: "Stable type · Building and running"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Building and running](/generated/api/topics/runs)

| Julia | Rust |
|---|---|
| not exported | `config::RunConfig` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this type.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct RunConfig {
    pub esm_integrator: String,
    pub description: Option<String>,
    pub model: String,
    pub operators: Option<Value>,
    pub schedule: Value,
    pub step: Option<Step>,
    pub partition: Option<Value>,
    pub precision: Option<Value>,
    pub adjoint: Option<Adjoint>,
    pub output: Option<Value>,
    pub restart: Option<Value>,
}

impl RunConfig { pub fn from_value(v: &Value) -> Result<RunConfig, Error> }

impl RunConfig { pub fn schedule_node(&self) -> Result<Node, Error> }
```

A run configuration.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/rust/earthsci-integrator/src/config.rs#L15) · [Rust docs](https://docs.rs/earthsci-integrator/latest/earthsci_integrator/config/struct.RunConfig.html)

</div>

</div>
