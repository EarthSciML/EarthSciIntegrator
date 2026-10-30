---
title: "vjp_p"
description: "Stable function · The operator interface"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [The operator interface](/generated/api/topics/operators)

| Julia | Rust |
|---|---|
| `vjp_p!` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
vjp_p!(s::EarthSciIntegrator.OperatorSum, out, u, p, t, x)
vjp_p!(::EarthSciIntegrator.ForcedDecay, out, u, p, t, w)
vjp_p!(::EarthSciIntegrator.Oscillator, out, u, p, t, w)
vjp_p!(::EarthSciIntegrator.Kinetics, out, u, p, t, w)
vjp_p!(op::EarthSciIntegrator.Photochemistry, out, u, p, t, w)
vjp_p!(op::EarthSciIntegrator.Transport, out, u, p, t, w)
vjp_p!(op::EarthSciIntegrator.Reaction, out, u, p, t, w)
vjp_p!(::EarthSciIntegrator.Source, out, u, p, t, w)
```

```julia
vjp_p!(op, out, u, p, t, w)
```

`out += f_{k,p}(u)ᵀ w`: the vector-Jacobian product with respect to the parameters.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/operators.jl#L93)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
