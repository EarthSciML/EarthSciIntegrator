---
title: "map_operator"
description: "Stable type · The operator interface"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [The operator interface](/generated/api/topics/operators)

| Julia | Rust |
|---|---|
| not exported | `operator::MapOperator` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this type.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub trait MapOperator<T: Scalar>: Send + Sync {
    fn name(&self) -> &str;
    fn writes(&self) -> Vec<usize>;
    fn apply(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, dt: f64, mask: &mut Vec<u8>);
    fn apply_frozen(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, dt: f64, mask: &[u8]);
    fn map_vjp(&self, out_u: &mut [T], out_p: &mut [f64], u: &[T], p: &[f64], t: f64, dt: f64, mask: &[u8], w: &[T]);
    fn map_jvp(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, dt: f64, mask: &[u8], v_u: &[T], v_p: &[f64]);
}
```

A map operator `u ← M(u, p, t, Δ)` (§7).

The mask records the discrete decisions `apply` takes, as one byte per
decision; the adjoint holds them fixed (spec/adjoint.md §5.7).

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/rust/earthsci-integrator/src/operator.rs#L126) · [Rust docs](https://docs.rs/earthsci-integrator/latest/earthsci_integrator/operator/trait.MapOperator.html)

</div>

</div>
