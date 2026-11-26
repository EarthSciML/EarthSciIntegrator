---
title: "planner_refusal"
description: "Stable error · Errors"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** error · **Topic:** [Errors](/generated/api/topics/errors)

| Julia | Rust |
|---|---|
| `PlannerRefusal` | `earthsci_integrator_planner::Refusal` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
struct PlannerRefusal <: Exception
PlannerRefusal(code::String, at::String, msg::String)
PlannerRefusal(code, at, msg)
```

```julia
PlannerRefusal(code, at, msg)
```

The planner refuses its inputs: `code` is one of spec/planner-errors.json's
codes, `at` the JSON pointer into the run config of the offending key
(`/model` for a refusal about the model), and `msg` a message for people.
Conformance fixtures compare `code` and `at` only.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/2f2815c8579399275fdf110ee093aeb2fed8c2f1/julia/EarthSciIntegrator.jl/src/planner/refusal.jl#L3)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct Refusal {
    pub code: &'static str,
    pub pointer: String,
    pub message: String,
}
```

A refusal: a code of spec/planner-errors.json, the JSON pointer into the
run config it concerns (`/model` for the model), and a message. Only the
code and the pointer are conformance-tested (spec/plan.md §4).

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/2f2815c8579399275fdf110ee093aeb2fed8c2f1/rust/earthsci-integrator-planner/src/error.rs#L131) · [Rust docs](https://docs.rs/earthsci-integrator-planner/latest/earthsci_integrator_planner/error/struct.Refusal.html)

</div>

</div>
