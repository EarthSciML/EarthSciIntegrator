---
title: "tape"
description: "Stable type · Replay and adjoint"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Replay and adjoint](/generated/api/topics/adjoint)

| Julia | Rust |
|---|---|
| not exported | `tape::Tape` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this type.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct Tape {
    pub records: Vec<TapeRecord>,
}

impl Tape { pub fn to_jsonl(&self) -> String }

impl Tape { pub fn from_jsonl(text: &str) -> Result<Tape, String> }

impl Tape { pub fn index(&self) -> HashMap<&str, &TapeRecord> }

impl Tape { pub fn macro_segment(&self, k: usize) -> &[TapeRecord] }

impl Tape { pub fn step_counts(&self) -> (u64, u64) }
```

The tape of a run.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/rust/earthsci-integrator/src/tape.rs#L242) · [Rust docs](https://docs.rs/earthsci-integrator/latest/earthsci_integrator/tape/struct.Tape.html)

**Example** (`replay`)

```rust
let again = spec.replay(&u0, &p, &fwd.tape).unwrap();
assert_eq!(again.final_state(), fwd.final_state());
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/docs/api/examples/rust/forward_and_gradient.rs#L49-L50">docs/api/examples/rust/forward_and_gradient.rs</a></p>

</div>

</div>
