---
title: "build_run"
description: "Stable function · Building and running"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Building and running](/generated/api/topics/runs)

| Julia | Rust |
|---|---|
| `build_run` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
build_run(config::AbstractDict; tspan, saveat, prob, p, u0)
```

```julia
build_run(config; tspan, saveat=nothing, prob=toy_problem(config["model"]), p=prob.p, u0=prob.u0) -> Run
```

Build a run from a run config (a parsed JSON object). `tspan` and `saveat`
come from the conformance case (spec/toy-problems.md §2). Raises
[`UnsupportedError`](/generated/api/ref/unsupported_error) for constructs not implemented in this stage, and
[`ConfigError`](/generated/api/ref/config_error) for invalid ones.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/interpreter.jl#L80)

**Example** (`forward`)

```julia
run = build_run(config; tspan = (0.0, 2.0))
forward!(run)
u_final = run.outputs[end][2]
# Exact solution of u' = -k u + sin(ωt), u(0) = 1.
k, ω = prob.p
A(t) = (k * sin(ω * t) - ω * cos(ω * t)) / (k^2 + ω^2)
u_exact = (1 - A(0)) * exp(-k * 2) + A(2)
@test isapprox(only(u_final), u_exact; rtol = 1.0e-4)
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/docs/api/examples/julia/forward_and_gradient.jl#L24-L32">docs/api/examples/julia/forward_and_gradient.jl</a></p>

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

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
