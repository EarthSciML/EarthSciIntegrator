---
title: "all_tableaus"
description: "Stable function · Solvers"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Solvers](/generated/api/topics/solvers)

| Julia | Rust |
|---|---|
| not exported | `earthsci_integrator_solvers::tableau::all_tableaus` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this function.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn all_tableaus() -> impl Iterator<Item = &'static Tableau>
```

Every table, by name.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/99d5977103a92934e0e4e523acd665bb0517ceb2/rust/earthsci-integrator-solvers/src/tableau.rs#L243) · [Rust docs](https://docs.rs/earthsci-integrator-solvers/latest/earthsci_integrator_solvers/tableau/fn.all_tableaus.html)

</div>

</div>
