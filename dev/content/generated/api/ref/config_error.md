---
title: "config_error"
description: "Stable error · Errors"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** error · **Topic:** [Errors](/generated/api/topics/errors)

| Julia | Rust |
|---|---|
| `ConfigError` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
struct ConfigError <: Exception
ConfigError(msg::String)
ConfigError(msg)
```

```julia
ConfigError(msg)
```

Raised when a run config is invalid for a reason the schema cannot check
(spec/run-config.md §8).

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/EarthSciIntegrator.jl#L51)

**Example** (`refusal`)

```julia
bad = deepcopy(config)
bad["schedule"]["alg"] = "Bogus"
@test_throws ConfigError build_run(bad; tspan = (0.0, 2.0))
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/docs/api/examples/julia/forward_and_gradient.jl#L48-L50">docs/api/examples/julia/forward_and_gradient.jl</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this error.</p>

</div>

</div>
