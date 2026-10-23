---
title: "`solver.time_span`: the run interval a document asks for"
description: "One optional `{start, end}` field in the top-level `solver` block giving the simulation interval the document is meant to run over, resolved like `abstol` and `reltol`: an explicit argument at the call site wins, then this field. Purely additive and advisory; it changes no flattened system."
---

Status: draft (proposed by EarthSciIntegrator, 2026-10-23)
Target: EarthSciAST `esm-spec.md` §2.2 and `esm-schema.json` `$defs.Solver`; additive, in the same format version as U1 (`solver.external`) if that has not shipped, otherwise the next minor version
Needed by: EarthSciIntegrator milestone M2
Integrator refs: spec/run-config.md §1.1; decisions D60, D2, D24

## Summary

Add one optional field to the existing `solver` block:

```json
{
  "solver": {
    "reltol": 1e-4,
    "time_span": {"start": "2019-07-01T00:00:00Z", "end": "2019-07-03T00:00:00Z"}
  }
}
```

`time_span` is the interval the document's author means it to be simulated over. It is a fact about the model's intended use in the same sense as `abstol` and `reltol`, and it resolves the same way: an explicit argument at the call site wins, then this field. Without it a document that only makes sense over a particular period (a forcing-driven chemistry case, a spin-up) cannot say so, and every runner restates the interval.

## Field

| Field | Type | Description |
|---|---|---|
| `time_span` | `{start, end}` | Each a number in the document's time units (the units of the independent variable, §11.3), or an RFC 3339 timestamp in UTC. `end` is after `start`. |

- **Units.** A number is in the document's time units, as `tests[].time_span` already is (§6.6.2). This proposal reuses that shape on purpose, so `{start, end}` means one thing across the format.
- **Timestamps.** A string is an RFC 3339 UTC timestamp (`Z` suffix required). It means the POSIX time of that instant (seconds since `1970-01-01T00:00:00Z`, no leap seconds), converted to the document's time units. This matches how data loaders already describe their available range (§8.3 `temporal.start`/`end`, ISO 8601 datetimes) and how EarthSciMLBase spells a `tspan` from `DateTime`s. *Assumption:* EarthSciAST has no other time origin a timestamp could be relative to; if one exists, the timestamp should be relative to it instead.
- **Advisory.** Like every `solver` field, a binding MAY ignore it. A binding that honours it MUST let an explicit interval at the call site override it.
- **Round-trip and versioning** as for the other `solver` fields: parsed, validated, round-tripped verbatim, rejected below the version that introduces it.
- **Mounted components.** Solver hints on mounted component files are dropped at assembly (U1), so only the top-level document's `time_span` survives. This is the intended behaviour: the interval belongs to the run, not to a component.

## Validation

- `start` and `end` each a number or a string matching the RFC 3339 UTC pattern.
- `end` after `start` (a load-time check, since the schema cannot compare a timestamp with a number).
- A document mixing a number and a timestamp is legal; both convert to the document's time units.

## How EarthSciIntegrator uses it

The integrator's run interval comes from, in order of preference: the caller (API argument or command-line option), the run config's `time` key, then this field (D60, spec/run-config.md §1.1). The most preferred source that gives an interval gives all of it. The plan records the resolved interval and its source.

## Alternatives

- **Run config only.** The run config already can carry the interval, but a document then cannot state the period it is built for, and simple runners that take no run config (EarthSciAST's own `solve()`) have no place to read it.
- **A top-level `time_span` outside `solver`.** `solver` is where the format puts numerics a binding resolves against call-site arguments; the interval fits that rule exactly.
