---
title: "forward"
description: "Stable function · Building and running"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Building and running](/generated/api/topics/runs)

| Julia | Rust |
|---|---|
| `forward!` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
forward!(run::EarthSciIntegrator.Run)
```

```julia
forward!(run) -> run
```

Run the schedule forward over the case's interval with every leaf's
controller on, recording the tape, a checkpoint at every macro-step boundary,
and the outputs. A non-success return code from any leaf ends the run with
that code (spec/methods.md §12), stored in `run.retcode` and `run.failure`.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/interpreter.jl#L443)

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

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
