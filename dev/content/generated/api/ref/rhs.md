---
title: "rhs"
description: "Stable function · The operator interface"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [The operator interface](/generated/api/topics/operators)

| Julia | Rust |
|---|---|
| `rhs!` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
rhs!(s::EarthSciIntegrator.OperatorSum, out, u, p, t)
rhs!(::EarthSciIntegrator.ForcedDecay, out, u, p, t)
rhs!(::EarthSciIntegrator.Oscillator, out, u, p, t)
rhs!(::EarthSciIntegrator.Kinetics, out, u, p, t)
rhs!(op::EarthSciIntegrator.Photochemistry, out, u, p, t)
rhs!(op::EarthSciIntegrator.Transport, out, u, p, t)
rhs!(op::EarthSciIntegrator.Reaction, out, u, p, t)
rhs!(::EarthSciIntegrator.Source, out, u, p, t)
```

```julia
rhs!(op, out, u, p, t)
```

`out += f_k(u, p, t)`: the operator's contribution to the right-hand side.

Operator calls are methods of these generic functions on an operator type (operator-interface.md §5). The term-operator calls accumulate into `out`; [`apply!`](/generated/api/ref/apply) overwrites it.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/operators.jl#L73)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
