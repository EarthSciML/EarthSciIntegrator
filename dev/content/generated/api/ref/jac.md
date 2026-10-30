---
title: "jac"
description: "Stable function · The operator interface"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [The operator interface](/generated/api/topics/operators)

| Julia | Rust |
|---|---|
| `jac!` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
jac!(s::EarthSciIntegrator.OperatorSum, out, u, p, t)
jac!(::EarthSciIntegrator.ForcedDecay, out, u, p, t)
jac!(::EarthSciIntegrator.Oscillator, out, u, p, t)
jac!(::EarthSciIntegrator.Kinetics, out, u, p, t)
jac!(op::EarthSciIntegrator.Photochemistry, out, u, p, t)
jac!(op::EarthSciIntegrator.Transport, out, u, p, t)
jac!(op::EarthSciIntegrator.Reaction, out, u, p, t)
jac!(::EarthSciIntegrator.Source, out, u, p, t)
```

```julia
jac!(op, out::JacobianStorage, u, p, t)
```

`J_k(u)` accumulated into the leaf's matrix storage (operator-interface.md §4).

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/operators.jl#L99)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
