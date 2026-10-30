---
title: "ft"
description: "Stable function · The operator interface"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [The operator interface](/generated/api/topics/operators)

| Julia | Rust |
|---|---|
| `ft!` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
ft!(s::EarthSciIntegrator.OperatorSum, out, u, p, t)
ft!(::EarthSciIntegrator.ForcedDecay, out, u, p, t)
ft!(::EarthSciIntegrator.Oscillator, out, u, p, t)
ft!(::EarthSciIntegrator.Kinetics, out, u, p, t)
ft!(op::EarthSciIntegrator.Photochemistry, out, u, p, t)
ft!(::EarthSciIntegrator.Transport, out, u, p, t)
ft!(::EarthSciIntegrator.Reaction, out, u, p, t)
ft!(::EarthSciIntegrator.Source, out, u, p, t)
```

```julia
ft!(op, out, u, p, t)
```

`out += ∂f_k/∂t`: the explicit time derivative. Zero when `depends_on_t` is false.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/operators.jl#L105)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
