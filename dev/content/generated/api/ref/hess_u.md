---
title: "hess_u"
description: "Stable function · The operator interface"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [The operator interface](/generated/api/topics/operators)

| Julia | Rust |
|---|---|
| `hess_u!` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
hess_u!(s::EarthSciIntegrator.OperatorSum, out, u, p, t, x)
hess_u!(::EarthSciIntegrator.ForcedDecay, out, u, p, t, μ)
hess_u!(::EarthSciIntegrator.Oscillator, out, u, p, t, μs)
hess_u!(::EarthSciIntegrator.Kinetics, out, u, p, t, μs)
hess_u!(op::EarthSciIntegrator.Photochemistry, out, u, p, t, μs)
hess_u!(::EarthSciIntegrator.Transport, out, u, p, t, μ)
hess_u!(::EarthSciIntegrator.Reaction, out, u, p, t, μ)
hess_u!(::EarthSciIntegrator.Source, out, u, p, t, μ)
```

```julia
hess_u!(op, out, u, p, t, μ)
```

`out += ∇_u Σᵢ μᵢᵀ J_k(u) wᵢ`, for `μ` a vector of `(μᵢ, wᵢ)` pairs. Zero when `linear_in_u` is true.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/operators.jl#L111)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
