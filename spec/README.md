# EarthSciIntegrator specification

**Spec version 0.1.0** (M0 draft, 2026-10-23).

One spec, implemented natively in Julia ([`../julia/EarthSciIntegrator.jl`](../julia/EarthSciIntegrator.jl)) and Rust ([`../rust`](../rust)), checked against each other by a shared conformance suite ([`../tests/conformance`](../tests/conformance)) (D1, D8). The design and its reasons are in [`../docs/architecture.md`](../docs/architecture.md) and [`../docs/decisions.md`](../docs/decisions.md); this directory holds what implementations must do.

| Chapter | Contents | Status |
|---|---|---|
| [`run-config.md`](run-config.md) | The run configuration: operators, combinator semantics, macro-step control, axes, partition, precision, adjoint and output settings, and the rules the planner checks | M0 draft, M2 additions |
| [`plan.md`](plan.md) | The planner and its output: scope at M2, the passes, derived documents, resolved settings, validation, canonical form, conformance tier 1 | M2 draft |
| [`planner-errors.json`](planner-errors.json) | The planner's refusal codes, one per rule (D79) | M2 |
| [`adjoint.md`](adjoint.md) | The discrete adjoint: functional, tape, primitives, the adjoint step of every method family and combinator, accumulation, verification identities | M0 draft |
| [`methods.md`](methods.md) | Methods and controllers: the method catalogue, forward step forms, error estimates and norms, step-size control, first step, return codes | M1 draft |
| [`tableaus/`](tableaus/) | Method coefficients and controller constants as data, extracted from OrdinaryDiffEq (D54). The Julia package and the Rust solvers crate read copies in their own `data/` (a released package cannot reach `spec/`); refresh them with `tools/sync-package-data.sh` | M1 |
| [`operator-interface.md`](operator-interface.md) | What a leaf calls on an operator: right-hand side, derivative products, Jacobian storage, maps; operators a plan supplies | M1 draft, M2 §8 |
| [`tape.md`](tape.md) | The tape, replay, checkpoints, and a serialized form for tests | M1 draft |
| [`toy-problems.md`](toy-problems.md) | The hand-written toy problems, the M1 conformance cases and checks, and result files; their `.esm` versions and the M2 end-to-end cases | M1 draft, M2 §4 |
| [`schemas/run-config.schema.json`](schemas/run-config.schema.json) | JSON Schema (2020-12) for run configs | 0.1.0 |
| [`schemas/plan.schema.json`](schemas/plan.schema.json) | JSON Schema (2020-12) for `plan.json` | 0.1.0, M2 |
| [`schemas/tableau.schema.json`](schemas/tableau.schema.json), [`schemas/cases.schema.json`](schemas/cases.schema.json), [`schemas/result.schema.json`](schemas/result.schema.json), [`schemas/planner-errors.schema.json`](schemas/planner-errors.schema.json) | Coefficient tables, conformance cases, result files, and the refusal-code list | 0.1.0 |

Chapters still to come: linear and nonlinear solvers beyond what methods.md states (M1 batched block LU; Krylov and `solve` nodes at M4b), events and callbacks (with the M1 implementations), partitioning and executors (M3).

## Upstream versions

The spec is written against these sibling releases; conformance tier 0 pins the first in CI (`earthsci-ast==0.6.0`, the Python binding):

| Package | Version | Used for |
|---|---|---|
| EarthSciAST (Julia, Rust, Python bindings) | 0.6.0, esm format 1.3.0 | loading, flattening with per-term provenance (U2) and the reaction lowering of esm-libraries-spec §4.6.2, `solver.external` (U1), `solver.time_span` and `resolve_time_span` (U16), right-hand-side kernels, the public reentrant right-hand side `rhs_program` (U4) |
| EarthSciASTDiff (Julia, Rust) | 0.1.0 (commit 743d140; on EarthSciAST 0.6 in both languages, spec/plan.md §10 item 12) | band models, structure classes, the block factorization plan, `stable_json` (spec/plan.md §0.2) |

## Conventions

- **MUST**, **MUST NOT** and **MAY** are normative. Paragraphs marked *informative*, *assumption* or *open question* are not.
- Decisions are cited as D*n* ([`../docs/decisions.md`](../docs/decisions.md)), architecture sections as "architecture §*n*", and upstream asks as U*n* ([`../docs/rfcs/`](../docs/rfcs/)).
- A chapter's statements about the model refer to the flattened model EarthSciAST produces.

## Versioning

- The spec version is the run config's `esm_integrator` key. It is semver and independent of the esm format version (D24).
- Both schemas carry the version in their `$id`. A change that makes a previously valid run config invalid, or changes what a valid one means, is a major change once 1.0.0 is released; before that, a minor one.
- Each implementation exposes the version it implements (`EarthSciIntegrator.SPEC_VERSION`, `earthsci_integrator::SPEC_VERSION`) and refuses a run config for a version it does not implement.

## Conformance tiers

Architecture §12 defines tiers 1–4. M0 adds a tier 0 that runs today: [`../tests/conformance/validate_schemas.py`](../tests/conformance/validate_schemas.py) checks that the schemas are valid 2020-12 schemas, that every fixture under `*/valid/` validates and that every fixture under `*/invalid/` does not. From M2 it also checks the planner fixtures' structure and loads, validates and flattens every `.esm` fixture with EarthSciAST's Python binding, 0.6.0 (spec/plan.md §0.2, §9). Tier 1, the planner goldens, is spec/plan.md §9.
