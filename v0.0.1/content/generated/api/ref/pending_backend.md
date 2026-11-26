---
title: "pending_backend"
description: "Stable type · The planner"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [The planner](/generated/api/topics/planner)

| Julia | Rust |
|---|---|
| `PendingBackend` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
struct PendingBackend <: PlanBackend
PendingBackend()
```

```julia
PendingBackend()
```

A backend without EarthSciASTDiff, for tests of the passes that need
none of it (M2 stage 1's): nothing is derived. One case needs no bands: a model whose states are all scalars has
only scalar-row, scalar-column bands, which §5.2.3 gives no reach, so every
term is pointwise whatever its band list; `term_bands` returns an empty list
for it, which classifies identically. It returns `nothing` otherwise.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/2f2815c8579399275fdf110ee093aeb2fed8c2f1/julia/EarthSciIntegrator.jl/src/planner/bands.jl#L281)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this type.</p>

</div>

</div>
