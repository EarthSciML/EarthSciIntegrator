---
title: "vjp_u"
description: "Stable function · The operator interface"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [The operator interface](/generated/api/topics/operators)

| Julia | Rust |
|---|---|
| `vjp_u!` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
vjp_u!(s::EarthSciIntegrator.OperatorSum, out, u, p, t, x)
vjp_u!(::EarthSciIntegrator.ForcedDecay, out, u, p, t, w)
vjp_u!(::EarthSciIntegrator.Oscillator, out, u, p, t, w)
vjp_u!(::EarthSciIntegrator.Kinetics, out, u, p, t, w)
vjp_u!(op::EarthSciIntegrator.Photochemistry, out, u, p, t, w)
vjp_u!(op::EarthSciIntegrator.Transport, out, u, p, t, w)
vjp_u!(op::EarthSciIntegrator.Reaction, out, u, p, t, w)
vjp_u!(::EarthSciIntegrator.Source, out, u, p, t, w)
```

```julia
vjp_u!(op, out, u, p, t, w)
```

`out += J_k(u)ᵀ w`: the vector-Jacobian product with respect to the state.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/operators.jl#L87)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
