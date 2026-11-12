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

A part of the plan that needs EarthSciASTDiff (term and operator bands,
structure, the Gershgorin bound, derived documents and validation), asked of
a [`PlanBackend`](/generated/api/ref/plan_backend) that cannot give it, such as [`PendingBackend`](/generated/api/ref/pending_backend)
or a test backend. Raised instead of writing a value that would only look
right. The production backend, [`DiffBackend`](/generated/api/ref/diff_backend), gives every part.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/2f2815c8579399275fdf110ee093aeb2fed8c2f1/julia/EarthSciIntegrator.jl/src/planner/refusal.jl#L20)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this error.</p>

</div>

</div>
