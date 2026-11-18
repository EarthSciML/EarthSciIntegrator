---
title: "plan_backend"
description: "Stable type · The planner"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [The planner](/generated/api/topics/planner)

| Julia | Rust |
|---|---|
| `PlanBackend` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
abstract type PlanBackend
```

```julia
PlanBackend
```

What the planner needs from EarthSciASTDiff: a term's bands (§5.2.2), an
operator's structure (§5.4), the Gershgorin bound (§5.5.2), and the derived
documents with their validation (§6, §8). [`DiffBackend`](/generated/api/ref/diff_backend) gives them
all. A test backend may return `nothing` for "not available"; the planner
raises [`PlannerStageError`](/generated/api/ref/planner_stage_error) where it needs the value.

Methods:
- `term_bands(backend, ctx, term) -> Union{Vector{BandEntry}, Nothing}`
- `all_term_bands(backend, ctx, terms)`: every term's `term_bands`, in term
  order; [`DiffBackend`](/generated/api/ref/diff_backend) computes them from one probe document per batch
- `operator_structure(backend, ctx, op) -> Union{AbstractDict, Nothing}` (the
  plan's `structure` object)
- `gershgorin_bound(backend, ctx, op) -> Union{Float64, Nothing}` (`max_s ρ(u_s, p, t_s)`)
- `jacobian_reads_forcing(backend, ctx, op) -> Union{Bool, Nothing}` (a band
  coefficient references a data-fed parameter, directly or through an observed; §5.5.2)
- `derived_and_validation(backend, ctx) -> Union{Tuple{AbstractDict, AbstractVector}, Nothing}`

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/2f2815c8579399275fdf110ee093aeb2fed8c2f1/julia/EarthSciIntegrator.jl/src/planner/bands.jl#L259)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this type.</p>

</div>

</div>
