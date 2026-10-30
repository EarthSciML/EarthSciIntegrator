---
title: "jcs"
description: "Stable function · The planner"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [The planner](/generated/api/topics/planner)

| Julia | Rust |
|---|---|
| `jcs` | `earthsci_integrator_planner::jcs::to_string` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
jcs(x)
```

```julia
jcs(x) -> String
```

`x` (dictionaries, vectors, strings, numbers, booleans, `nothing`) serialized
as RFC 8785 canonical JSON (spec/plan.md §2.1).

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/planner/jcs.jl#L136)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn to_string(v: &Value) -> Result<String, String>
```

The canonical form of `v`. Fails on a number that is not finite, which
JSON cannot carry.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/rust/earthsci-integrator-planner/src/jcs.rs#L15) · [Rust docs](https://docs.rs/earthsci-integrator-planner/latest/earthsci_integrator_planner/jcs/fn.to_string.html)

</div>

</div>
