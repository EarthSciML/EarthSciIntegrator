---
title: "planner_run_config"
description: "Stable type · The planner"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [The planner](/generated/api/topics/planner)

| Julia | Rust |
|---|---|
| not exported | `earthsci_integrator_planner::runconfig::RunConfig` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this type.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct RunConfig {
    pub path: PathBuf,
    pub bytes: Vec<u8>,
    pub value: Value,
}

impl RunConfig { pub fn dir(&self) -> PathBuf }

impl RunConfig { pub fn file_name(&self) -> String }

impl RunConfig { pub fn at(&self, pointer: &str) -> Option<&Value> }

impl RunConfig { pub fn model(&self) -> &str }
```

A run config as read from its file.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/rust/earthsci-integrator-planner/src/runconfig.rs#L21) · [Rust docs](https://docs.rs/earthsci-integrator-planner/latest/earthsci_integrator_planner/runconfig/struct.RunConfig.html)

</div>

</div>
