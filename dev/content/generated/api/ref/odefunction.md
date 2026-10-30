---
title: "odefunction"
description: "Stable function · Building and running"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Building and running](/generated/api/topics/runs)

| Julia | Rust |
|---|---|
| `odefunction` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
odefunction(ops::EarthSciIntegrator.OperatorSum; inplace)
odefunction(leaf::EarthSciIntegrator.LeafNode; kw...)
```

```julia
odefunction(ops::OperatorSum; inplace=true)
odefunction(leaf::LeafNode; inplace=true)
```

The sum of a leaf's operators as a `SciMLBase.ODEFunction` with `jac`, `tgrad`,
`vjp` and `jvp` filled from the operator calls (operator-interface.md §5,
D33). Needs `SciMLBase` loaded (package extension).

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/EarthSciIntegrator.jl#L106)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
