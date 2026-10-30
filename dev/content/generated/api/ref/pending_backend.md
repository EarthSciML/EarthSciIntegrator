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

M2 stage 1's backend: EarthSciASTDiff is not available, so nothing is
derived. One case needs no bands: a model whose states are all scalars has
only scalar-row, scalar-column bands, which §5.2.3 gives no reach, so every
term is pointwise whatever its band list; `term_bands` returns an empty list
for it, which classifies identically. It returns `nothing` otherwise.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/planner/bands.jl#L271)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this type.</p>

</div>

</div>
