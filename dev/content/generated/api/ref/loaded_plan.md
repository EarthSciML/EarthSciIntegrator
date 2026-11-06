---
title: "loaded_plan"
description: "Stable type · Other"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Other](/generated/api/topics/other)

| Julia | Rust |
|---|---|
| not exported | `planned::LoadedPlan` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this type.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct LoadedPlan<T: Scalar> {
    pub problem: ToyProblem<T>,
    pub run_config: RunConfig,
    pub tspan: (f64, f64),
    pub state_dofs: BTreeMap<String, Vec<usize>>,
    pub timeline: Arc<ForcingTimeline>,
}
```

A plan loaded for running: the problem its operators make and the run
config of its resolved schedule and step.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/69547f8c5f25e3fdeacb226a938ea3942e3cd8ef/rust/earthsci-integrator/src/planned.rs#L416) · [Rust docs](https://docs.rs/earthsci-integrator/latest/earthsci_integrator/planned/struct.LoadedPlan.html)

</div>

</div>
