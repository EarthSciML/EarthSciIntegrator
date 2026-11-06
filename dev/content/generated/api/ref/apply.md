---
title: "apply"
description: "Stable function · The operator interface"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [The operator interface](/generated/api/topics/operators)

| Julia | Rust |
|---|---|
| `apply!` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
apply!(::EarthSciIntegrator.DecayMap, out, mask, u, p, t, Δ)
```

```julia
apply!(op, out, mask, u, p, t, Δ)
```

Apply the map operator: `out = M(u, p, t, Δ)`, recording in `mask` (one byte per decision) the discrete decisions taken; the adjoint holds them fixed (operator-interface.md §7).

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/69547f8c5f25e3fdeacb226a938ea3942e3cd8ef/julia/EarthSciIntegrator.jl/src/operators.jl#L135)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
