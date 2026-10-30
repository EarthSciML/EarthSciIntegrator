---
title: "spec_version"
description: "Stable constant · Building and running"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** constant · **Topic:** [Building and running](/generated/api/topics/runs)

| Julia | Rust |
|---|---|
| `SPEC_VERSION` | `SPEC_VERSION` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
const SPEC_VERSION::VersionNumber
```

Version of the integrator spec (the run config's `esm_integrator` key) this package implements.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/EarthSciIntegrator.jl#L36)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub const SPEC_VERSION: &str = "0.1.0";
```

Version of the integrator spec (the run config's `esm_integrator` key)
this crate implements.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/rust/earthsci-integrator/src/lib.rs#L27) · [Rust docs](https://docs.rs/earthsci-integrator/latest/earthsci_integrator/constant.SPEC_VERSION.html)

</div>

</div>
