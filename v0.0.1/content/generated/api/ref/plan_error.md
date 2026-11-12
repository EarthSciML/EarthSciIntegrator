---
title: "plan_error"
description: "Stable error · Errors"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** error · **Topic:** [Errors](/generated/api/topics/errors)

| Julia | Rust |
|---|---|
| not exported | `earthsci_integrator_planner::PlanError` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this error.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub enum PlanError {
    Refused(Refusal),
    NotYetImplemented(String),
    Io(String),
    Internal(String),
}
```

Why planning stopped.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/2f2815c8579399275fdf110ee093aeb2fed8c2f1/rust/earthsci-integrator-planner/src/error.rs#L145) · [Rust docs](https://docs.rs/earthsci-integrator-planner/latest/earthsci_integrator_planner/error/enum.PlanError.html)

</div>

</div>
