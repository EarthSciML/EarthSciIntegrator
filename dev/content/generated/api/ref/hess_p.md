---
title: "hess_p"
description: "Stable function · The operator interface"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [The operator interface](/generated/api/topics/operators)

| Julia | Rust |
|---|---|
| `hess_p!` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
hess_p!(s::EarthSciIntegrator.OperatorSum, out, u, p, t, x)
hess_p!(::EarthSciIntegrator.ForcedDecay, out, u, p, t, μ)
hess_p!(::EarthSciIntegrator.Oscillator, out, u, p, t, μs)
hess_p!(::EarthSciIntegrator.Kinetics, out, u, p, t, μs)
hess_p!(op::EarthSciIntegrator.Photochemistry, out, u, p, t, μs)
hess_p!(op::EarthSciIntegrator.Transport, out, u, p, t, μs)
hess_p!(op::EarthSciIntegrator.Reaction, out, u, p, t, μs)
hess_p!(::EarthSciIntegrator.Source, out, u, p, t, μ)
```

```julia
hess_p!(op, out, u, p, t, μ)
```

`out += ∇_p Σᵢ μᵢᵀ J_k(u) wᵢ`, for `μ` a vector of `(μᵢ, wᵢ)` pairs.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/operators.jl#L117)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
