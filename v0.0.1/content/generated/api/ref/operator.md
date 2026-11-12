---
title: "operator"
description: "Stable type · The operator interface"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [The operator interface](/generated/api/topics/operators)

| Julia | Rust |
|---|---|
| not exported | `operator::Operator` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this type.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub trait Operator<T: Scalar>: Send + Sync {
    fn name(&self) -> &str;
    fn structure(&self) -> Structure;
    fn depends_on_t(&self) -> bool;
    fn linear_in_u(&self) -> bool;
    fn writes(&self) -> Vec<usize>;
    fn reads(&self) -> Option<Vec<usize>>;
    fn structure_over(&self, dofs: &[usize], n_model: usize) -> Result<Structure, String>;
    fn forcing_refresh_times(&self) -> Vec<f64>;
    fn provides(&self, call: Call) -> bool;
    fn fault(&self) -> Option<String>;
    fn rhs(&self, out: &mut [T], u: &[T], p: &[f64], t: f64);
    fn jvp_u(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, v: &[T]);
    fn vjp_u(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, w: &[T]);
    fn vjp_p(&self, out: &mut [f64], u: &[T], p: &[f64], t: f64, w: &[T]);
    fn jac(&self, out: &mut MatrixStorage<T>, u: &[T], p: &[f64], t: f64);
    fn jac_over(&self, out: &mut MatrixStorage<T>, dofs: &[usize], u: &[T], p: &[f64], t: f64, scratch: &mut JacOverScratch<T>);
    fn ft(&self, out: &mut [T], u: &[T], p: &[f64], t: f64);
    fn hess_u(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, mu: &[MuPair<'_, T>]);
    fn hess_p(&self, out: &mut [f64], u: &[T], p: &[f64], t: f64, mu: &[MuPair<'_, T>]);
    fn ft_vjp_u(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, nu: &[T]);
    fn ft_vjp_p(&self, out: &mut [f64], u: &[T], p: &[f64], t: f64, nu: &[T]);
}
```

An additive term operator `f_k(u, p, t)` (§1, §3).

`rhs`, `jvp_u` and `vjp_u` are required. The others have default bodies
that panic; an operator that implements one says so in
`provides`, and a leaf that needs a call its
operators do not provide is refused when it is built.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/2f2815c8579399275fdf110ee093aeb2fed8c2f1/rust/earthsci-integrator/src/operator.rs#L123) · [Rust docs](https://docs.rs/earthsci-integrator/latest/earthsci_integrator/operator/trait.Operator.html)

</div>

</div>
