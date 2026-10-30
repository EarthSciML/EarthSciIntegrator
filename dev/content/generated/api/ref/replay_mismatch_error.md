---
title: "replay_mismatch_error"
description: "Stable error · Errors"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** error · **Topic:** [Errors](/generated/api/topics/errors)

| Julia | Rust |
|---|---|
| `ReplayMismatchError` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
struct ReplayMismatchError <: Exception
ReplayMismatchError(msg::String)
ReplayMismatchError(msg)
```

```julia
ReplayMismatchError(msg)
```

Raised when a tape does not fit the run it is replayed for (spec/tape.md §3):
a record missing or of the wrong kind, an interval other than the one the
schedule gives the invocation, a map decision that flips, or clips outside the
leaf's steps or its clip domain.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/EarthSciIntegrator.jl#L62)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this error.</p>

</div>

</div>
