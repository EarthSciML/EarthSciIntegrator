---
title: "diff_backend"
description: "Stable type · Other"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Other](/generated/api/topics/other)

| Julia | Rust |
|---|---|
| `DiffBackend` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
mutable struct DiffBackend <: PlanBackend
DiffBackend(ctx, ops::Dict{String, EarthSciIntegrator.OperatorArtifacts}, layout::Union{Nothing, EarthSciIntegrator.DofLayout}, u0::Union{Nothing, Vector{Float64}}, samples::Union{Nothing, Vector{Tuple{Float64, Vector{Float64}}}}, forcing::Union{Nothing, Vector{Dict{String, Vector{Float64}}}}, bandvalues::Dict{String, Any}, report::Dict{String, Any})
DiffBackend(ctx, ops, layout, u0, samples, forcing, bandvalues, report)
DiffBackend()
```

```julia
DiffBackend()
```

The planner's backend on EarthSciASTDiff (M2 stage 2), holding what one
planning run derives: per operator its documents, bands and built
evaluators, the initial state and the sampled states, and,
after validation, the band values (§8.2) and measured errors for
`plan.report.json`. All of it belongs to the planning context `ctx` it was
derived from; a backend passed to `plan` again starts afresh.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c7d0a137caf25b61f681f823e84b032886ea4fb9/julia/EarthSciIntegrator.jl/src/planner/backend.jl#L126)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this type.</p>

</div>

</div>
