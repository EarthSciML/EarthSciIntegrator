---
title: "tableaus"
description: "Stable constant · Building and running"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** constant · **Topic:** [Building and running](/generated/api/topics/runs)

| Julia | Rust |
|---|---|
| `TABLEAUS` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
const TABLEAUS::Dict{String, EarthSciIntegrator.Tableau}
```

```julia
TABLEAUS
```

Every method's `Tableau`, keyed by run-config `alg` name, read from
`data/tableaus/*.json` (a copy of `spec/tableaus`) at precompile time.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/99d5977103a92934e0e4e523acd665bb0517ceb2/julia/EarthSciIntegrator.jl/src/tableaus.jl#L145)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this constant.</p>

</div>

</div>
