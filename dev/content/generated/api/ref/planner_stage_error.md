---
title: "planner_stage_error"
description: "Stable error · Errors"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** error · **Topic:** [Errors](/generated/api/topics/errors)

| Julia | Rust |
|---|---|
| `PlannerStageError` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
struct PlannerStageError <: Exception
PlannerStageError(what::String)
PlannerStageError(what)
```

```julia
PlannerStageError(what)
```

A part of the planner that M2 stage 1 does not implement yet: everything that
needs EarthSciASTDiff (term and operator bands, structure, stiffness by the
Gershgorin bound, the strategies that read structure, derived documents and
validation). Raised instead of writing a value that would only look right.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/planner/refusal.jl#L20)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this error.</p>

</div>

</div>
