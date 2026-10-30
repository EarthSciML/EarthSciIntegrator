---
title: "gradient"
description: "Stable function · Replay and adjoint"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Replay and adjoint](/generated/api/topics/adjoint)

| Julia | Rust |
|---|---|
| `gradient` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
gradient(run::EarthSciIntegrator.Run)
```

```julia
gradient(run) -> (J, ū₀, p̄)
```

The discrete adjoint gradient of the run's functional on the frozen tape
(spec/adjoint.md §2.2): `ū₀ = λ(t₀)` (§6.2) and `p̄` over every parameter,
with bind_dt-bound parameters set to zero (§5.8). Needs [`forward!`](/generated/api/ref/forward)
first. Point terms are added on reaching their output times (§2.1).

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/adjoint.jl#L59)

**Example** (`gradient`)

```julia
J, ū0, p̄ = gradient(run)
# The functional is u(2), so ū0 is d u(2) / d u(0) = exp(-k t).
@test isapprox(only(ū0), exp(-k * 2); rtol = 1.0e-4)
@test length(p̄) == length(prob.p)
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/docs/api/examples/julia/forward_and_gradient.jl#L36-L39">docs/api/examples/julia/forward_and_gradient.jl</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
