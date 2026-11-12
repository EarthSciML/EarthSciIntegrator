---
title: "band_entry"
description: "Stable type · The planner"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [The planner](/generated/api/topics/planner)

| Julia | Rust |
|---|---|
| `BandEntry` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
struct BandEntry
BandEntry(en::EarthSciASTDiff.JacEntry)
BandEntry(en::EarthSciASTDiff.JacEntry, row::AbstractString)
BandEntry(row::String, col::String, rows::Vector{Tuple{Int64, Int64}}, row_idx::Vector{Union{Int64, String}}, col_idx::Vector{Any}, contracted::Vector{Tuple{String, Int64, Int64}}, guard)
BandEntry(row, col, rows, row_idx, col_idx, contracted, guard)
BandEntry(row, col, rows, row_idx, col_idx)
BandEntry(row, col, rows, row_idx, col_idx, contracted)
BandEntry(d::AbstractDict)
```

```julia
BandEntry
```

One entry of EarthSciASTDiff's band list (esm-jacobian-spec §4), in the form
its `serialize_band` writes, without `coef`:

- `row`, `col`: the row and column variables (flattened names);
- `rows`: per dimension of `row`, the inclusive range `(lo, hi)`; empty for a scalar row;
- `row_idx`: per dimension of `row`, an index name (`String`) or an integer;
- `col_idx`: per dimension of `col`, an expression in EarthSciAST's serialized
  form (a name `String`, an `Integer` literal, a `Float64` literal, or a
  `Dict` with `"op"` and `"args"`); empty for a scalar column;
- `contracted`: free contracted dimensions `(name, lo, hi)`, possibly empty;
- `guard`: the band's `where` guard in serialized form, or `nothing`. A
  cell where it evaluates to zero is not a point of the band.

[`DiffBackend`](/generated/api/ref/diff_backend) builds these from `EarthSciASTDiff.serialize_band`'s output.

---

A `BandEntry` from `serialize_band`'s dictionary (its `coef`, if present, is ignored).

---

```julia
BandEntry(en::EarthSciASTDiff.JacEntry, row = en.u)
```

The [`BandEntry`](/generated/api/ref/band_entry) of `BandEntry(band_json(en))`, with row `row`. Its
coefficient, which pass 2 never reads (§5.2.2; pass 2's entries are
structure-only and have none), is replaced by the literal 0 before
`serialize_band` writes the entry, so a full entry's coefficient is never
serialized; every other field is read through `serialize_band`, as for the
operator's bands.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/2f2815c8579399275fdf110ee093aeb2fed8c2f1/julia/EarthSciIntegrator.jl/src/planner/bands.jl#L30)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this type.</p>

</div>

</div>
