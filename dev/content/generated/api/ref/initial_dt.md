---
title: "initial_dt"
description: "Stable function · Solvers"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Solvers](/generated/api/topics/solvers)

| Julia | Rust |
|---|---|
| not exported | `earthsci_integrator_solvers::initdt::initial_dt` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this function.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn initial_dt<T: Scalar>(inp: &InitDtInput<'_, T>, f: &mut RhsFn<'_, T>) -> (f64, usize)
```

Computes the first step. `f(out, u, t)` overwrites `out` with the leaf's
right-hand side. Returns the step and the number of `f` evaluations (2,
or 1 when the first evaluation has a NaN).

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/rust/earthsci-integrator-solvers/src/initdt.rs#L24) · [Rust docs](https://docs.rs/earthsci-integrator-solvers/latest/earthsci_integrator_solvers/initdt/fn.initial_dt.html)

</div>

</div>
