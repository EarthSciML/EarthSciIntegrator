---
title: "build_plan_run"
description: "Stable function · Other"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Other](/generated/api/topics/other)

| Julia | Rust |
|---|---|
| `build_plan_run` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
build_plan_run(dir::AbstractString; saveat, tspan, cache, model, cache_root, url_overrides)
```

```julia
build_plan_run(dir; saveat = nothing, tspan = nothing, cache = nothing) -> (Run, PlanProblem)
```

A run of the plan in `dir` (spec/operator-interface.md §8, spec/toy-problems.md
§4): its operators built by [`plan_problem`](/generated/api/ref/plan_problem), its resolved schedule and
step, over the plan's `time`. A caller's `tspan` other than the plan's is
refused (the plan is stale, spec/plan.md §3). Run it with [`forward!`](/generated/api/ref/forward).

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/2f2815c8579399275fdf110ee093aeb2fed8c2f1/julia/EarthSciIntegrator.jl/src/plan_runtime.jl#L730)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
