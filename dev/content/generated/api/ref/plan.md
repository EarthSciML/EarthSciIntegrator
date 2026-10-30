---
title: "plan"
description: "Stable function · The planner"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [The planner](/generated/api/topics/planner)

| Julia | Rust |
|---|---|
| `plan` | `earthsci_integrator_planner::plan` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
plan(config_path::AbstractString; tspan, backend)
```

```julia
plan(config_path; tspan=nothing, backend=PendingBackend()) -> draft
```

Plan the run config at `config_path` (spec/plan.md): read and flatten its
model, run passes 1–4, check and resolve the schedule and the settings, and
resolve strategies. `tspan` is the caller's run interval (§7.2), a pair of
numbers or of RFC 3339 UTC timestamps. A refusal is a
[`PlannerRefusal`](/generated/api/ref/planner_refusal); a part of the plan that needs EarthSciASTDiff, with
a backend that cannot give it, is a [`PlannerStageError`](/generated/api/ref/planner_stage_error).

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/planner/planner.jl#L40)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn plan(path: &Path, options: &Options, bands: &dyn BandSource) -> Result<Planned>
```

Plans the run config at `path` as far as stage 1 goes.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/rust/earthsci-integrator-planner/src/plan.rs#L77) · [Rust docs](https://docs.rs/earthsci-integrator-planner/latest/earthsci_integrator_planner/plan/fn.plan.html)

</div>

</div>
