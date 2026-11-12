---
title: "planned_operator"
description: "Stable type · Other"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Other](/generated/api/topics/other)

| Julia | Rust |
|---|---|
| `PlannedOperator` | `planned::PlannedOperator` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
struct PlannedOperator <: EarthSciIntegrator.AbstractOperator
PlannedOperator(name::String, writes::Vector{Int64}, structure::EarthSciIntegrator.Structure, depends_on_t::Bool, linear_in_u::Bool, rhs::R, jac::J, points::Vector{Tuple{Int64, Int64, Int64}}, cache::EarthSciIntegrator.BandCache, layout::EarthSciIntegrator.DofLayout, blockable::Bool, forcing::Union{Nothing, EarthSciIntegrator.ForcingRuntime}, rhs_fs::Base.RefValue{Union{Nothing, EarthSciIntegrator.EpochSet}}, jac_fs::Base.RefValue{Union{Nothing, EarthSciIntegrator.EpochSet}}) where {R<:EarthSciIntegrator.PlanDocument, J<:EarthSciIntegrator.PlanDocument}
```

```julia
PlannedOperator
```

An operator a plan supplies (spec/operator-interface.md §8): the right-hand
side and Jacobian documents built by EarthSciAST, the canonical DOFs it
writes, its structure, and every point of its bands as `(slot, row, col)`,
the slot of the band field `J_k` cell in the Jacobian document and the
canonical row and column DOFs it adds into. It provides `rhs`, `jvp_u`,
`vjp_u` and `jac` (§8.3); the other calls are not provided at M2.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/2f2815c8579399275fdf110ee093aeb2fed8c2f1/julia/EarthSciIntegrator.jl/src/plan_runtime.jl#L212)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct PlannedOperator {
    /* private fields */
}
```

An operator built from a plan (§8).

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/2f2815c8579399275fdf110ee093aeb2fed8c2f1/rust/earthsci-integrator/src/planned.rs#L162) · [Rust docs](https://docs.rs/earthsci-integrator/latest/earthsci_integrator/planned/struct.PlannedOperator.html)

</div>

</div>
