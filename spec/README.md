# EarthSciIntegrator specification

**Spec version 0.1.0** (M0 draft, 2026-10-23).

One spec, implemented natively in Julia ([`../julia/EarthSciIntegrator.jl`](../julia/EarthSciIntegrator.jl)) and Rust ([`../rust`](../rust)), checked against each other by a shared conformance suite ([`../tests/conformance`](../tests/conformance)) (D1, D8). The design and its reasons are in [`../docs/architecture.md`](../docs/architecture.md) and [`../docs/decisions.md`](../docs/decisions.md); this directory holds what implementations must do.

| Chapter | Contents | Status |
|---|---|---|
| [`run-config.md`](run-config.md) | The run configuration: operators, combinator semantics, macro-step control, partition, precision, adjoint and output settings, and the rules the planner checks | M0 draft |
| [`plan.md`](plan.md) | The planner's output: files, canonical form, contents | M0 skeleton; operator structure at M2 |
| [`adjoint.md`](adjoint.md) | The discrete adjoint: functional, tape, primitives, the adjoint step of every method family and combinator, accumulation, verification identities | M0 draft |
| [`methods.md`](methods.md) | Methods and controllers: the method catalogue, forward step forms, error estimates and norms, step-size control, first step, return codes | M1 draft |
| [`tableaus/`](tableaus/) | Method coefficients and controller constants as data, extracted from OrdinaryDiffEq (D54) | M1 |
| [`operator-interface.md`](operator-interface.md) | What a leaf calls on an operator: right-hand side, derivative products, Jacobian storage, maps | M1 draft |
| [`tape.md`](tape.md) | The tape, replay, checkpoints, and a serialized form for tests | M1 draft |
| [`toy-problems.md`](toy-problems.md) | The hand-written toy problems, the M1 conformance cases and checks, and result files | M1 draft |
| [`schemas/run-config.schema.json`](schemas/run-config.schema.json) | JSON Schema (2020-12) for run configs | 0.1.0 |
| [`schemas/plan.schema.json`](schemas/plan.schema.json) | JSON Schema (2020-12) for `plan.json` | 0.1.0 skeleton |
| [`schemas/tableau.schema.json`](schemas/tableau.schema.json), [`schemas/cases.schema.json`](schemas/cases.schema.json), [`schemas/result.schema.json`](schemas/result.schema.json) | Coefficient tables, conformance cases, and result files | 0.1.0 |

Chapters still to come: linear and nonlinear solvers beyond what methods.md states (M1 batched block LU; Krylov and `solve` nodes at M4b), events and callbacks (with the M1 implementations), partitioning and executors (M3).

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
