---
title: "write_plan"
description: "Stable function · Other"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Other](/generated/api/topics/other)

| Julia | Rust |
|---|---|
| `write_plan` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
write_plan(d::EarthSciIntegrator.PlanDraft, dir::AbstractString; bandvalues)
```

```julia
write_plan(draft, dir; bandvalues = false)
```

Write the plan directory (spec/plan.md §1): `plan.json`, `derived/*.esm` and
`plan.report.json`, and with `bandvalues = true` the conformance band-value
files `validation/<operator>.bandvalues.json` (§8.2). Needs a
[`DiffBackend`](/generated/api/ref/diff_backend) draft. The planner's own outputs already in `dir`
(`plan.json`, `plan.report.json`, and the `derived/` and `validation/`
directories) are removed first, so no file of an earlier plan survives;
anything else in `dir` is left alone.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/2f2815c8579399275fdf110ee093aeb2fed8c2f1/julia/EarthSciIntegrator.jl/src/planner/planner.jl#L205)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
