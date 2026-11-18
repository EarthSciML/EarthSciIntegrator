---
title: "rhs_fn"
description: "Stable type · Solvers"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Solvers](/generated/api/topics/solvers)

| Julia | Rust |
|---|---|
| not exported | `earthsci_integrator_solvers::RhsFn` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this type.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub type RhsFn<'a, T> = dyn FnMut(&mut [T], &[T], f64) + 'a;
```

The leaf's right-hand side as methods see it: `f(out, u, t)` overwrites `out`.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/2f2815c8579399275fdf110ee093aeb2fed8c2f1/rust/earthsci-integrator-solvers/src/lib.rs#L32) · [Rust docs](https://docs.rs/earthsci-integrator-solvers/latest/earthsci_integrator_solvers/type.RhsFn.html)

</div>

</div>
