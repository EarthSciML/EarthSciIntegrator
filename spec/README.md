# EarthSciIntegrator specification

**Spec version 0.1.0** (M0 draft, 2026-10-23).

One spec, implemented natively in Julia ([`../julia/EarthSciIntegrator.jl`](../julia/EarthSciIntegrator.jl)) and Rust ([`../rust`](../rust)), checked against each other by a shared conformance suite ([`../tests/conformance`](../tests/conformance)) (D1, D8). The design and its reasons are in [`../docs/architecture.md`](../docs/architecture.md) and [`../docs/decisions.md`](../docs/decisions.md); this directory holds what implementations must do.

| Chapter | Contents | Status |
|---|---|---|
| [`run-config.md`](run-config.md) | The run configuration: operators, combinator semantics, macro-step control, partition, precision, adjoint and output settings, and the rules the planner checks | M0 draft |
| [`plan.md`](plan.md) | The planner's output: files, canonical form, contents | M0 skeleton; operator structure at M2 |
| [`adjoint.md`](adjoint.md) | The discrete adjoint: functional, tape, primitives, the adjoint step of every method family and combinator, accumulation, verification identities | M0 draft |
| [`schemas/run-config.schema.json`](schemas/run-config.schema.json) | JSON Schema (2020-12) for run configs | 0.1.0 |
| [`schemas/plan.schema.json`](schemas/plan.schema.json) | JSON Schema (2020-12) for `plan.json` | 0.1.0 skeleton |

Chapters still to come: methods and controllers (tableau sources, error norms, step acceptance; M1), linear and nonlinear solvers (M1), partitioning and executors (M3), events and callbacks (M1).

## Conventions

- **MUST**, **MUST NOT** and **MAY** are normative. Paragraphs marked *informative*, *assumption* or *open question* are not.
- Decisions are cited as D*n* ([`../docs/decisions.md`](../docs/decisions.md)), architecture sections as "architecture §*n*", and upstream asks as U*n* ([`../docs/rfcs/`](../docs/rfcs/)).
- A chapter's statements about the model refer to the flattened model EarthSciAST produces.

## Versioning

- The spec version is the run config's `esm_integrator` key. It is semver and independent of the esm format version (D24).
- Both schemas carry the version in their `$id`. A change that makes a previously valid run config invalid, or changes what a valid one means, is a major change once 1.0.0 is released; before that, a minor one.
- Each implementation exposes the version it implements (`EarthSciIntegrator.SPEC_VERSION`, `earthsci_integrator::SPEC_VERSION`) and refuses a run config for a version it does not implement.

## Conformance tiers

Architecture §12 defines tiers 1–4. M0 adds a tier 0 that runs today: [`../tests/conformance/validate_schemas.py`](../tests/conformance/validate_schemas.py) checks that both schemas are valid 2020-12 schemas, that every fixture under `*/valid/` validates and that every fixture under `*/invalid/` does not.
