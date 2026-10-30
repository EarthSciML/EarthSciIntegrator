---
title: "read_run_config"
description: "Stable function · The planner"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [The planner](/generated/api/topics/planner)

| Julia | Rust |
|---|---|
| not exported | `earthsci_integrator_planner::runconfig::read` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this function.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn read(path: &Path) -> Result<RunConfig>
```

Step 1 of spec/plan.md §4: parse the run config, validate it against
`schemas/run-config.schema.json` (`run_config_invalid`) and refuse a
version this planner does not implement (`version_unsupported`).

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/rust/earthsci-integrator-planner/src/runconfig.rs#L73) · [Rust docs](https://docs.rs/earthsci-integrator-planner/latest/earthsci_integrator_planner/runconfig/fn.read.html)

</div>

</div>
