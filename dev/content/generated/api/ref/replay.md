---
title: "replay"
description: "Stable function · Replay and adjoint"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Replay and adjoint](/generated/api/topics/adjoint)

| Julia | Rust |
|---|---|
| `replay` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
replay(run::EarthSciIntegrator.Run; ...)
replay(run::EarthSciIntegrator.Run, u0; ...)
replay(run::EarthSciIntegrator.Run, u0, p; tape)
```

```julia
replay(run, u0=run.u0, p=run.p; tape=run.tape) -> (u_final, outputs, q)
```

Replay the whole run on the frozen tape (spec/adjoint.md §2.2), generic over
the element types of `u0` and `p`. `outputs` holds the state at every output
time, in order; `q` is the integrated functional of a functional over time
(`nothing` otherwise).

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/interpreter.jl#L514)

**Example** (`replay`)

```julia
u_replayed, outputs, q = replay(run)
@test u_replayed == u_final
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/docs/api/examples/julia/forward_and_gradient.jl#L43-L44">docs/api/examples/julia/forward_and_gradient.jl</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
