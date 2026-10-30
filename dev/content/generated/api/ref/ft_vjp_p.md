---
title: "ft_vjp_p"
description: "Stable function · The operator interface"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [The operator interface](/generated/api/topics/operators)

| Julia | Rust |
|---|---|
| `ft_vjp_p!` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
ft_vjp_p!(s::EarthSciIntegrator.OperatorSum, out, u, p, t, x)
ft_vjp_p!(::EarthSciIntegrator.ForcedDecay, out, u, p, t, ν)
ft_vjp_p!(::EarthSciIntegrator.Oscillator, out, u, p, t, ν)
ft_vjp_p!(::EarthSciIntegrator.Kinetics, out, u, p, t, ν)
ft_vjp_p!(op::EarthSciIntegrator.Photochemistry, out, u, p, t, ν)
ft_vjp_p!(::EarthSciIntegrator.Transport, out, u, p, t, ν)
ft_vjp_p!(::EarthSciIntegrator.Reaction, out, u, p, t, ν)
ft_vjp_p!(::EarthSciIntegrator.Source, out, u, p, t, ν)
```

```julia
ft_vjp_p!(op, out, u, p, t, ν)
```

`out += ∇_p (νᵀ ∂f_k/∂t)`.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/operators.jl#L129)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
