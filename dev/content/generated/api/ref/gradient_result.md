---
title: "gradient_result"
description: "Stable type · Replay and adjoint"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Replay and adjoint](/generated/api/topics/adjoint)

| Julia | Rust |
|---|---|
| not exported | `adjoint::Gradient` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this type.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct Gradient {
    pub functional: f64,
    pub u0: Vec<f64>,
    pub p: Vec<f64>,
}
```

The functional and its gradient.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/rust/earthsci-integrator/src/adjoint.rs#L24) · [Rust docs](https://docs.rs/earthsci-integrator/latest/earthsci_integrator/adjoint/struct.Gradient.html)

**Example** (`gradient`)

```rust
let g = spec.gradient(&fwd, &p).unwrap();
// The functional is u(2), so ∂J/∂u0 is d u(2) / d u(0) = exp(-k t).
assert!((g.u0[0] - (-k * 2.0).exp()).abs() < 1e-4 * (-k * 2.0).exp());
assert_eq!(g.p.len(), p.len());
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/docs/api/examples/rust/forward_and_gradient.rs#L42-L45">docs/api/examples/rust/forward_and_gradient.rs</a></p>

</div>

</div>
