---
title: "load_plan"
description: "Stable function · Other"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Other](/generated/api/topics/other)

| Julia | Rust |
|---|---|
| not exported | `planned::load_plan` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this function.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn load_plan<T: Scalar>(dir: &Path) -> Result<LoadedPlan<T>, Error>
```

Reads the plan directory `dir` and builds every operator its schedule
names (operator-interface.md §8). A plan with `forcing` needs the data
options' model: see `load_plan_with`.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/2f2815c8579399275fdf110ee093aeb2fed8c2f1/rust/earthsci-integrator/src/planned.rs#L488) · [Rust docs](https://docs.rs/earthsci-integrator/latest/earthsci_integrator/planned/fn.load_plan.html)

</div>

</div>
