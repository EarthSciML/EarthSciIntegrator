---
title: "data_options"
description: "Stable type · Other"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Other](/generated/api/topics/other)

| Julia | Rust |
|---|---|
| not exported | `forcing::DataOptions` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this type.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct DataOptions {
    pub model: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
    pub url_overrides: HashMap<String, String>,
}
```

The run-time options of the data layer (spec/operator-interface.md
§8.4.2 item 1): in neither the run config nor the plan, since none
changes what is computed. EarthSciIO's offline mode is its own
`EARTHSCI_OFFLINE` environment variable.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/2f2815c8579399275fdf110ee093aeb2fed8c2f1/rust/earthsci-integrator/src/forcing.rs#L76) · [Rust docs](https://docs.rs/earthsci-integrator/latest/earthsci_integrator/forcing/struct.DataOptions.html)

</div>

</div>
