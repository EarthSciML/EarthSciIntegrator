---
title: "plan_options"
description: "Stable type · The planner"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [The planner](/generated/api/topics/planner)

| Julia | Rust |
|---|---|
| not exported | `earthsci_integrator_planner::Options` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this type.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct Options {
    pub time: Option<Value>,
}
```

What the caller may give besides the run config.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/rust/earthsci-integrator-planner/src/plan.rs#L28) · [Rust docs](https://docs.rs/earthsci-integrator-planner/latest/earthsci_integrator_planner/plan/struct.Options.html)

</div>

</div>
