---
title: "map_jvp"
description: "Stable function · The operator interface"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [The operator interface](/generated/api/topics/operators)

| Julia | Rust |
|---|---|
| `map_jvp!` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
map_jvp!(::EarthSciIntegrator.DecayMap, out, u, p, t, Δ, mask, v_u, v_p)
```

```julia
map_jvp!(op, out, u, p, t, Δ, mask, v_u, v_p)
```

The Jacobian-vector product `(∂M/∂u) v_u + (∂M/∂p) v_p` of a map operator, with the decisions in `mask` held fixed, written to `out`, for tangent checks (operator-interface.md §7).

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/operators.jl#L147)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
