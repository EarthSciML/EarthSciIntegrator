---
title: "unsupported_error"
description: "Stable error · Errors"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** error · **Topic:** [Errors](/generated/api/topics/errors)

| Julia | Rust |
|---|---|
| `UnsupportedError` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
struct UnsupportedError <: Exception
UnsupportedError(msg::String)
UnsupportedError(msg)
```

```julia
UnsupportedError(msg)
```

Raised when a run config asks for something this implementation does not
support yet (spec/run-config.md §8 rule 16): a method family not implemented
in this stage, a combinator not yet interpreted, and the like.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/julia/EarthSciIntegrator.jl/src/EarthSciIntegrator.jl#L39)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this error.</p>

</div>

</div>
