---
title: "scalar"
description: "Stable type · Solvers"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Solvers](/generated/api/topics/solvers)

| Julia | Rust |
|---|---|
| not exported | `earthsci_integrator_solvers::Scalar` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this type.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub trait Scalar: Copy + Send + Sync + Debug + Display + PartialOrd + Default + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self> + AddAssign + SubAssign + MulAssign + 'static {
    const NAME: &'static str;
    const EPSILON: f64;
    fn zero() -> Self;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn is_finite(self) -> bool;
    fn to_bits_u64(self) -> u64;
}
```

A floating-point element type for state vectors: `f32` or `f64`.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/rust/earthsci-integrator-solvers/src/scalar.rs#L10) · [Rust docs](https://docs.rs/earthsci-integrator-solvers/latest/earthsci_integrator_solvers/scalar/trait.Scalar.html)

</div>

</div>
