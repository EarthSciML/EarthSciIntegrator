---
title: "load_plan_with"
description: "Stable function · Other"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Other](/generated/api/topics/other)

| Julia | Rust |
|---|---|
| not exported | `planned::load_plan_with` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this function.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn load_plan_with<T: Scalar>(dir: &Path, data: &DataOptions) -> Result<LoadedPlan<T>, Error>
```

`load_plan`, fetching the data of a plan with `forcing` as `data`
says (operator-interface.md §8.4).

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/2f2815c8579399275fdf110ee093aeb2fed8c2f1/rust/earthsci-integrator/src/planned.rs#L494) · [Rust docs](https://docs.rs/earthsci-integrator/latest/earthsci_integrator/planned/fn.load_plan_with.html)

</div>

</div>
