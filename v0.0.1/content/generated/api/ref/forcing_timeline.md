---
title: "forcing_timeline"
description: "Stable type · Other"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Other](/generated/api/topics/other)

| Julia | Rust |
|---|---|
| not exported | `forcing::ForcingTimeline` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this type.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct ForcingTimeline {
    /* private fields */
}

impl ForcingTimeline { pub fn empty(t0: f64) -> ForcingTimeline }

impl ForcingTimeline { pub fn at(&self, t: f64) -> &Forcing }

impl ForcingTimeline { pub fn select(&self, span: Option<StepSpan>, t: f64) -> &Forcing }

impl ForcingTimeline { pub fn refresh_times(&self) -> &[f64] }

impl ForcingTimeline { pub fn boundaries(&self) -> &[f64] }

impl ForcingTimeline { pub fn epoch(&self, k: usize) -> &Forcing }
```

The forcing epochs of a run (§8.4.3) and each epoch's buffers.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/2f2815c8579399275fdf110ee093aeb2fed8c2f1/rust/earthsci-integrator/src/forcing.rs#L128) · [Rust docs](https://docs.rs/earthsci-integrator/latest/earthsci_integrator/forcing/struct.ForcingTimeline.html)

</div>

</div>
