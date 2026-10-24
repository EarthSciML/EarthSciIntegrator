# The plan

**Status:** M0 skeleton, 2026-10-23. The field set is fixed by [`schemas/plan.schema.json`](schemas/plan.schema.json) so that conformance tier 1 has a target. The contents of `operators.*.structure` are specified with the planner at M2, and the checkpoint action vocabulary at M4. Background: architecture §2, §3, D8, D34.

The planner reads a model and a run config and writes a **plan**: `plan.json` plus a directory `derived/` of `.esm` documents. The runtime executes only the plan. The plan is where the Julia and Rust implementations are checked against each other (D8): the same inputs and spec version must give byte-identical files in both (architecture §3.3).

## 1. Files

```
<plan dir>/
  plan.json            canonical JSON (§2), golden-gated
  derived/*.esm        one document per derived artifact, golden-gated
  plan.report.json     implementation-specific report, not golden-gated
```

- `derived/` documents are named `<operator>.<kind>.esm`, where `kind` is one of the `derived` kinds in the schema: `rhs`, `jacobian`, `frozen_jacobian`, `vjp`, `second_order`, `param_jacobian`, `forcing_jacobian`, `residual`, `map`, `limit`, `condition` (architecture §3.2 pass 6). They are serialized in EarthSciAST's canonical form, as EarthSciASTDiff's byte-identical goldens are.
- `frozen_jacobian` is emitted for an operator used by a `solve` node with `picard.freeze`, the only place freezing is written (D51). The forward iteration uses the frozen Jacobian; the adjoint needs the full residual Jacobian at the converged solution, emitted as `jacobian` (spec/adjoint.md §4.6).
- `plan.report.json` holds everything that legitimately differs between implementations: the planner's name and version, timings, and the measured errors behind each validation check. Its format is not specified.

## 2. Canonical form

`plan.json` is serialized as **RFC 8785 JSON Canonicalization Scheme** (JCS): UTF-8, object keys sorted by UTF-16 code units, no insignificant whitespace, and numbers in ECMAScript shortest round-trip form. Both implementations write JCS directly; neither post-processes the other's output.

Beyond JCS, the content is canonical:
- Name lists (`writes`, `reads`) are sorted by Unicode code point and duplicate-free.
- `terms` follow flattened-equation order, then contribution order within an equation (U2).
- `validation` is sorted by `(operator, check)`.
- Every default is made explicit (§3); nothing is omitted because it equals a default.
- Paths are relative to the plan directory or, for `inputs`, as given in the run config. No absolute paths, hostnames, timestamps or implementation names appear.
- `sha256` is the lowercase hex SHA-256 of a file's bytes.

## 3. Contents

| Key | Contents |
|---|---|
| `esm_integrator` | The spec version, equal to the run config's. |
| `inputs` | The model and run config, by path and hash. A plan is stale when either hash changes. |
| `operators` | Per operator: `kind` (`rhs`, `algebraic`, `map`, `limit`, `condition`); `locality` with its per-axis `radius` for stencils (architecture §3.2 pass 2); sorted `writes` and `reads` (pass 3); `terms` (provenance, U2); `structure` (pass 4, specified at M2); `stiffness` with the source of the classification (pass 5). |
| `schedule` | The **resolved** schedule: the run-config grammar (spec/run-config.md §4) with no `"auto"` anywhere, every leaf's options explicit (method, controller, `dt_scope`, `jacobian`, `linsolve`, `domain`), every default operator named, and an `id` on every node. Ids the run config gave are kept; the planner names the rest by their path from the root (`root`, `root.0`, `root.1.every`, `root.2.subcycle`, …). The path is built from positions alone, whatever ids the ancestors have: the unnamed first child of a node `transport` at `root.0` is `root.0.0`. The schema checks that every node has an id and that no value is `"auto"`. |
| `step` | The macro-step control, as resolved. |
| `partition` | Rank and tile grids, halo width per axis (the maximum stencil radius over the operators a rank runs), shard counts, and the output stores' shard length that rank boundaries align to (U12). |
| `precision` | Every precision role for every leaf, by node id (spec/run-config.md §7.1). |
| `checkpoint_schedule` | The adjoint's checkpoint schedule, computed once by the planner with Checkpointing.jl's algorithms and executed by both runtimes (D34). |
| `derived` | Every derived document, by path, with its kind, operator and hash. |
| `validation` | The checks of architecture §3.2 pass 8 that ran: `rhs_sum`, `band_model`, `non_empty`. A plan is written only when every check passes, so each entry records `passed: true`; measured errors go to `plan.report.json`. |

## 4. Conformance

Tier 1 (architecture §12): for each fixture under `tests/conformance/`, the Julia and Rust planners write `plan.json` and `derived/*.esm`, and both are compared byte for byte with the checked-in goldens. Until M2 the suite checks only that the schema accepts `tests/conformance/plans/valid/` and rejects `tests/conformance/plans/invalid/`.
