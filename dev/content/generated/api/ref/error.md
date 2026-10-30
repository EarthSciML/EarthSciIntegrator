---
title: "error"
description: "Stable error · Errors"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** error · **Topic:** [Errors](/generated/api/topics/errors)

| Julia | Rust |
|---|---|
| not exported | `Error` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this error.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub enum Error {
    Config(String),
    Unimplemented(String),
    Run(String),
}
```

Errors: refusals of a run config or schedule, and failed runs.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/rust/earthsci-integrator/src/lib.rs#L31) · [Rust docs](https://docs.rs/earthsci-integrator/latest/earthsci_integrator/enum.Error.html)

**Example** (`refusal`)

```rust
let mut bad = config.clone();
bad["schedule"]["alg"] = json!("Bogus");
let rc = RunConfig::from_value(&bad).unwrap();
match RunSpec::<f64>::new(&rc, (0.0, 2.0), None) {
    Err(Error::Config(msg)) => assert!(msg.contains("Bogus")),
    other => panic!("expected a config error, got {:?}", other.map(|_| ())),
}
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/docs/api/examples/rust/forward_and_gradient.rs#L54-L60">docs/api/examples/rust/forward_and_gradient.rs</a></p>

</div>

</div>
