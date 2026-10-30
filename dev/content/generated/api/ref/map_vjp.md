---
title: "map_vjp"
description: "Stable function · The operator interface"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [The operator interface](/generated/api/topics/operators)

| Julia | Rust |
|---|---|
| `map_vjp!` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
map_vjp!(::EarthSciIntegrator.DecayMap, out_u, out_p, u, p, t, Δ, mask, w)
```

```julia
map_vjp!(op, out_u, out_p, u, p, t, Δ, mask, w)
```

Accumulate the vector-Jacobian product of a map operator, with the decisions in `mask` held fixed, into `out_u` and `out_p` (operator-interface.md §7).

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/operators.jl#L141)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
