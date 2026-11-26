---
title: "plan_problem"
description: "Stable function · Other"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Other](/generated/api/topics/other)

| Julia | Rust |
|---|---|
| `plan_problem` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
plan_problem(dir::AbstractString; cache, model, cache_root, url_overrides)
```

```julia
plan_problem(dir; cache = nothing) -> PlanProblem
```

Build every operator the resolved schedule of the plan in `dir` names, from
its entry `operators.<name>` and its documents `derived/<name>.rhs.esm` and
`derived/<name>.jacobian.esm` (spec/operator-interface.md §8). A derived
document whose bytes do not hash to the plan's `derived` entry is refused:
the plan is stale. `cache`, a dictionary, keeps built documents by their hash
across plans, so a document many plans share is built once.

A plan with `forcing` reads data (§8.4): its model, at `model` (default: its
`inputs.model.path` relative to `dir`) and checked against the plan's hash,
gives the providers, through EarthSciAST's `providers_from_document` with
EarthSciIO's cache under `cache_root` (default `dir/data-cache`) and the
source-name-to-URL-template map `url_overrides`. These three are run-time
options, in neither the run config nor the plan.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/2f2815c8579399275fdf110ee093aeb2fed8c2f1/julia/EarthSciIntegrator.jl/src/plan_runtime.jl#L521)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
