---
title: "`solver.external`: a document points at its own run configuration"
description: "One optional string field in the top-level `solver` block that names an external run-configuration document (a schedule, per-operator algorithms, tolerances, adjoint settings). EarthSciAST parses, validates, version-gates, round-trips and resolves the reference, and never opens the file. The `.esm` stays free of algorithm names; the run config has its own schema and version."
---

Status: draft (proposed by EarthSciIntegrator, 2026-10-13); done upstream: released in EarthSciAST v0.4.0 (esm 1.3.0)
Target: EarthSciAST `esm-spec.md` §2.2 and `esm-schema.json` `$defs.Solver`; format version 1.3.0 (additive; gated with `solver_external_version_too_old`)
Needed by: EarthSciIntegrator milestone M2
Integrator refs: architecture.md §2, §4.1, §4.2, §13 (U1); decisions D2, D7, D8, D24

## Summary

Add one optional field to the existing `solver` block:

```json
{
  "esm": "1.3.0",
  "solver": {
    "stiffness": "high",
    "reltol": 1e-4,
    "external": "reseact.run.json"
  }
}
```

`external` is a reference to a separate run-configuration document. That
document holds what the `.esm` deliberately does not: the splitting schedule,
the algorithm per operator, per-operator tolerances, step limits, partitioning
and adjoint settings (architecture.md §4.2). Its format belongs to the consumer.
For EarthSciIntegrator it is JSON with its own schema and an `esm_integrator`
version key (D24).

EarthSciAST's job is small and fully normative:
- parse and validate the field;
- reject it under `esm` < 1.3.0;
- round-trip it verbatim;
- resolve it to a location on request, by the same lexical rule as other
  relative references;
- never open, fetch or interpret the referenced file.

For every EarthSciAST binding the field is advisory. Each built-in `solve()`
ignores it and keeps its current behaviour.

## Motivation

**The schedule already lives outside the document, in code.** reseact's
integrator setup is spread through a run script. The per-operator algorithm
choice is a comment and a tuple (`reseact.esm/run_reseact.jl:14-15`, `:251-252`).
The macro step comes from an environment variable (`run_reseact.jl:139`). The
tolerances are constants (`run_reseact.jl:166`). The document itself declares
`esm: 1.0.0` and has no `solver` block (`reseact.esm/reseact.esm:2`). None of
this is visible to a second runtime, and none of it is versioned with the model.

**eqice encodes PISM's step order as model structure.** The harness refuses the
algebraic SSA (`eqice/PLAN.md:152-154`), and PISM operator-splits every step
(`eqice/PLAN.md:154-157`). So the plan is to wire "each substep [as] its own
model ... sequentially" (`eqice/PLAN.md:201-202`). That is a schedule expressed
through coupling, because the format has nowhere else to put it. The stage-2
component files each carry a `solver` block with their own tolerances
(`eqice/stage2/ssa.esm:7-11`, `eqice/stage2/ice_energy.esm:7-11`). Those blocks
do not survive mounting into the stage-3 assembly (`eqice/PLAN.md:196`): a mount
hoists only `function_tables` and `data_sources`
(`EarthSciAST/pkg/earthsci-ast-rs/src/ref_loading.rs:894`,
`EarthSciAST/pkg/EarthSciAST.jl/src/resolve.jl:1302`).

**EqWeFiC states outright that time integration is not in the `.esm`.** "The
*time integration* (RK3 with acoustic sub-stepping) is the solver's job and is
not written in `.esm`; the runner picks the integrator"
(`EqWeFiC/PLAN.md:2251-2253`). 3-D runs are to be "solved by a different method"
(`EqWeFiC/PLAN.md:1496-1497`). The WRF schedule (acoustic substeps, `radt`,
microphysics after the RK loop, `chemdt`) has to be written down somewhere, and
it has to be found from the document.

**The format excludes the obvious alternatives.** The solver-block RFC rules out
algorithm names (`EarthSciAST/docs/content/rfcs/solver-block.md:126-129`;
`esm-spec.md:266-268`). It defers per-component placement
(`solver-block.md:298-302`) and executable splitting, which "would amend §9's
single-flat-system model and needs its own design" (`solver-block.md:304-306`).
EarthSciIntegrator agrees with all three (D2). The schedule goes in a separate
document, and the `.esm` only points at it.

**The integrator design depends on the pointer.** The `.esm` is the entry point
(architecture.md:56). A user pins a plan by pointing `solver.external` at a
resolved run config (architecture.md:83). D7 lists `solver.external` as
unavoidable even under the smallest compiler contract (decisions.md:164).

## Decisions of record

| # | Question | Decision | Why |
|---|---|---|---|
| 1 | Where the field lives | **In the existing top-level `solver` block** | It is numerics configuration. The block, its gate pattern and its "not for" list already exist. A new top-level key would add a second container for one fact. |
| 2 | Shape of the value | **One non-empty string** | One run per document is the only case anyone has. An object form (`{ref, format}`) can be added later as `oneOf`. The referenced document identifies its own format by its contents (`esm_integrator`). |
| 3 | Inline or by reference | **By reference only** | Inline content would put algorithm names and a foreign schema into the `.esm`. It would also tie the run config's version to the format's (contrary to D24). |
| 4 | Does EarthSciAST read the file? | **Never** | The format of the target is not EarthSciAST's. Go and TypeScript have no ingest. A missing file is the consumer's error, not a load error. |
| 5 | Round-trip | **Verbatim; the resolved location is exposed by an accessor, not written back** | §2.2.4 already requires the block to round-trip verbatim (`esm-spec.md:286-292`). Writing an absolute path back would make emitted documents differ by machine, which defeats byte-identical goldens over emitted documents (architecture.md §3.3). This is the `coupling_import` pattern (`esm-spec.md:5253-5256`). |
| 6 | Resolution rule | **The §8.2.1 lexical table without the substitution row; base = directory of the file read; cwd for string loads; no `${VAR}`** | One rule for every relative location in the format. `${VAR}` is refused for the reason §8.2.1 gives (`esm-spec.md:3390`): numerics chosen by the environment are not reproducible. A deployment-chosen run config is a call-site argument. |
| 7 | Mounting | **Does not cross a mount edge**, like the rest of `solver` | A run config describes one run of one assembly. Rust and Julia already drop a leaf's `solver` block at a mount; Python, TypeScript and Go are assumed to, not checked. This makes the rule explicit. |
| 8 | Coexistence with the other four fields | **All may be present; no cross-validation** | EarthSciAST cannot see the run config, so it cannot check agreement. The four hints keep their meaning for every consumer that ignores `external`. |
| 9 | Versioning | **esm 1.3.0; new code `solver_external_version_too_old`** | A new field on an existing construct gets its own code, as `const` `units` did at 1.2.0 (`esm-spec.md:1575-1579`). |
| 10 | Which bindings | **All five parse, validate, gate, round-trip and resolve** | Resolution is a pure string function. Making it required everywhere gives one answer per document, which is the §5.22 argument (`CONFORMANCE_SPEC.md:3561-3566`). |

## Design

### Field

| Field | Type | Meaning |
|---|---|---|
| `external` | non-empty string | A reference to a run-configuration document for this document. Resolved by §2.2.6. Advisory for every EarthSciAST binding. |

Schema addition to `$defs.Solver` (`esm-schema.json:2574`), which stays closed:

```json
"external": {
  "type": "string",
  "minLength": 1,
  "description": "Reference to an external run-configuration document (esm-spec §2.2.6). Relative paths resolve against the directory of the file this document was read from. EarthSciAST never opens it. Round-trips verbatim. Arrives at esm 1.3.0."
}
```

The empty-block rule is unchanged (`esm-spec.md:185-194`). A block holding only
`external` is not empty.

### Resolution (new esm-spec §2.2.6)

Let *D* be the directory of the file the document was read from. For a document
loaded from a string or built in memory, *D* is the process working directory,
as §8.2.1 already rules for a relative `ref` (`esm-spec.md:3375`). A caller may
pass an explicit base instead.

| Form | Test on the string | Resolved location |
|---|---|---|
| URL | matches `^[A-Za-z][A-Za-z0-9+.-]*://` | unchanged |
| Absolute path | begins with `/` | dot segments removed, prefixed `file://` |
| Relative path | anything else | joined onto *D*, dot segments removed, prefixed `file://` |

- Dot-segment removal is lexical (RFC 3986 §5.2.4), never `realpath`. This is
  the rule of `esm-spec.md:3384`.
- A string containing `${` is refused at load with `solver_external_unresolved`.
  The message names `/solver/external` and quotes the string.
- `{` has no meaning here. There are no substitutions.
- Resolution never touches the filesystem or the network. A location that does
  not exist resolves normally.

Resolution is computed at load and kept beside the document, outside the wire
form. The authored string is what `emit` writes. Precedent: Rust records
`coupling_import_base` as a `#[serde(skip)]` field
(`pkg/earthsci-ast-rs/src/types/document.rs:140-156`, set at
`pkg/earthsci-ast-rs/src/parse.rs:355-358`). Julia keeps a weak side table
(`pkg/EarthSciAST.jl/src/coupling_imports.jl:224-230`). Python keeps a
non-compared dataclass field
(`pkg/earthsci-ast-py/src/earthsci_ast/esm_types.py:1358`).

A tool that writes a document to a different directory than it read it from
must rewrite a relative `external`. The same holds today for a verbatim
`coupling_import` ref; the spec does not state it there either (assumption: no
binding rewrites such refs on emit).

### Mounting

A referenced document's `solver` block, including `external`, does not cross a
`subsystems.<k>` or top-level `models.<k>` mount edge. Only the root document's
`solver` block is consulted. This is current behaviour in Rust and Julia (the
hoisted blocks are listed at `ref_loading.rs:894` and `resolve.jl:1302`). The
spec gains one sentence in §4.7 saying so.

### Coexistence with the other fields

- A built-in `solve()` / `init()` in Julia, Rust or Python ignores `external`.
  It applies `abstol` / `reltol` by the §2.2.2 chain exactly as today
  (`pkg/EarthSciAST.jl/src/solver.jl:60-68`,
  `pkg/earthsci-ast-rs/src/solver.rs:63-76`,
  `pkg/earthsci-ast-py/src/earthsci_ast/solver.py:87-121`). It MAY emit an
  informational note that the document names a run config it does not read.
  It MUST NOT fail because of `external`. It MUST NOT change its choice of
  integrator because `external` is present.
- A consumer that acts on `external` SHOULD treat the referenced document as
  more specific than the four hints. The recommended order for such a consumer
  is: call-site argument, then an explicit run-config value, then the document
  hint, then the consumer's default. EarthSciIntegrator adopts this. Its planner
  reads `stiffness` and `splitting` only where the run config says `"auto"`
  (architecture.md §3.5), and per-leaf `control` tolerances fall back to
  `abstol` / `reltol`.
- No combination is invalid. `splitting: "none"` beside a run config that
  splits is not an EarthSciAST error. A consumer MAY warn.

### Normative requirements (amends esm-spec §2.2.4)

1. Parse and validate `external`: a string of length at least 1, and no `${`.
2. Round-trip it verbatim.
3. Change nothing. Presence of `external` MUST NOT alter equations,
   classification, namespacing or the flattened system. Flatten does not read
   it.
4. Gate on version. A document declaring `esm` < 1.3.0 whose `solver` block
   carries `external` is rejected with `solver_external_version_too_old`, before
   schema validation, at pointer `/solver/external`. A 1.1.0 or 1.2.0 document
   with a `solver` block and no `external` stays valid.
5. Never open, read, fetch or validate the referenced document.
6. Resolve on request by §2.2.6.

New diagnostics in §2.2.5:

| Code | Meaning |
|---|---|
| `solver_external_version_too_old` | File declares `esm` < 1.3.0 but `solver` carries `external`. |
| `solver_external_unresolved` | `solver.external` contains `${`. |

### Per-binding API

Each binding adds one field, one gate and one accessor. The accessor returns
`nothing` / `None` when the document has no `external`.

**Julia** (`pkg/EarthSciAST.jl/src/`):

```julia
struct Solver                                # types.jl:1282
    stiffness::Union{String,Nothing}
    abstol::Union{Float64,Nothing}
    reltol::Union{Float64,Nothing}
    splitting::Union{String,Nothing}
    external::Union{String,Nothing}          # new
end
reject_solver_external_pre_v13(raw_data)     # beside reject_solver_pre_v11, resolve.jl:407
solver_external_location(file::EsmFile; base_path=nothing)::Union{String,Nothing}
```

`coerce_solver` (`parse.jl:742-758`) reads the field and includes it in the
empty test. `serialize_solver` (`serialize.jl:410-417`) emits it.

**Rust** (`pkg/earthsci-ast-rs/src/`):

```rust
pub struct Solver { /* … */ pub external: Option<String> }   // types/model.rs:194
pub fn reject_solver_external_pre_v13(view: &serde_json::Value) -> Result<(), DiagnosticError>;
pub fn solver_external_location(file: &EsmFile, base: Option<&std::path::Path>)
    -> Result<Option<String>, DiagnosticError>;
```

`normalize_empty` (`solver.rs:87-95`) includes `external`. The gate runs beside
`reject_solver_pre_v11` (`parse.rs:242-246`). The recorded base should be a
general `document_base` field. Rust records `coupling_import_base` only when the
caller passed a base path (`parse.rs:355-358`), so a path load must record it
for `external` too.

**Python** (`pkg/earthsci-ast-py/src/earthsci_ast/`):

```python
@dataclass
class Solver:                                # esm_types.py:624
    ...
    external: str | None = None
def reject_solver_external_pre_v13(view: Any) -> None: ...      # solver.py
def solver_external_location(esm_file: EsmFile, base: str | None = None) -> str | None: ...
```

Python's `solve()` ignores the field, like Julia's and Rust's.

**TypeScript and Go**: type, schema validation, gate and accessor. Neither has
`solve()`, so neither has anything else to do.

All new symbols are declared in `api-surface.json`.

## What this is not

- **Not a run-config format.** EarthSciAST does not define, parse or validate
  the referenced document. EarthSciIntegrator's run-config schema lives in its
  own `spec/`, versioned by `esm_integrator` (D24).
- **Not an algorithm name in the `.esm`.** §2.2.3's exclusion stands. The
  `.esm` gains a location, not a vocabulary.
- **Not a mount or include.** Nothing is loaded, inlined or merged. Flatten is
  unchanged. The flattened system still has no schedule.
- **Not per-component placement.** The run config places operators by
  selectors over provenance (U2) and locality. The `.esm` stays one block per
  document.
- **Not an obligation on any binding's `solve()`.** CONFORMANCE_SPEC §5.9 still
  governs built-in integration, unchanged.
- **Not a rule about derived documents.** EarthSciIntegrator's planner should
  not write `solver.external` into its derived `.esm` files. That is an
  integrator rule.

## Conformance and tests

Validation corpus (`tests/valid`, `tests/invalid`, `expected_errors.json`):
- `valid/solver_external.esm`: `esm: "1.3.0"`, all five fields. Round-trips
  verbatim.
- `valid/solver_external_only.esm`: `external` alone.
- `invalid/solver_external_version_too_old.esm`: `esm: "1.2.0"`. Pinned
  `resolver_only`, like `solver_version_too_old.esm`
  (`tests/invalid/expected_errors.json:3320-3326`).
- `invalid/solver_external_env_var.esm`: `"${RUNS}/x.json"`. Pinned
  `resolver_only` with `resolver_error_code: solver_external_unresolved`, and
  `is_valid: false` for all five, as `data_source_url_env_var.esm` is
  (`expected_errors.json:3268-3275`).
- `invalid/solver_external_empty.esm` (`""`) and
  `invalid/solver_external_nonstring.esm` (`42`): schema errors.

Resolution suite, `tests/conformance/solver_external/`, modelled on §5.22
(`CONFORMANCE_SPEC.md:3557-3646`):
- One fixture per table row: `./`-relative, bare relative, `../`-climbing,
  dot-segment collapsing, absolute, `file://` URL, `https://` URL. Each is
  loaded from a path and compared against `manifest.json` as
  `{"repo_path": …}` or `{"verbatim": …}`.
- The same document loaded from a string with an explicit base.
- A root document mounting a leaf whose `solver` carries `external`. The
  root's accessor returns the root's value, or none.
- None of the targets exist on disk. That proves no binding reads them.
- `bindings_required` is all five.

Advisory gate: a small simulation fixture whose `external` names a missing
file. Julia, Rust and Python must integrate it with built-in `solve()` and pass
its inline tests.

Invariance: flatten a fixture with and without `external`. The flattened
outputs must be identical.

## Work breakdown

1. **Spec, `esm-spec.md`.** §2 table row (`:89`) updated. §2.2 field table
   (`:176-181`) gains `external`. New §2.2.6 for resolution. §2.2.3 (`:261-276`)
   gains a bullet saying `external` is a location, not an algorithm. §2.2.4
   (`:278-296`) and §2.2.5 (`:298-302`) as above. §4.7 gains the mount sentence.
   Version header to 1.3.0.
2. **Schema.** `$defs.Solver.external`. `$id` → `…/esm/1.3.0/esm.schema.json`.
   Propagate to the vendored copies and pass `scripts/sync-schema.sh`. Rust's
   `SCHEMA_VERSION` (`pkg/earthsci-ast-rs/src/lib.rs:412`) follows.
3. **Version-bump audit.** As in the solver-block RFC
   (`solver-block.md:234-250`): confirm the no-op migration path and find
   fixtures pinned to the current version. If U2 adds format fields, it shares
   this bump.
4. **Bindings.** Field, gate, accessor, recorded base, round-trip, and
   `api-surface.json` entries, in all five.
5. **Conformance.** The fixtures above; `tests/COVERAGE_MATRIX.md` row 2.2;
   a new CONFORMANCE_SPEC section beside §5.22.
6. **Docs.** `MIGRATION.md`: a `new` row and a `format` row for 1.3.0.

The largest item is the five-binding resolution suite. The rest is small.

## Alternatives considered

- **Inline the schedule in the `.esm`** (`solver.schedule: {…}`). Rejected. It
  puts algorithm names and a foreign schema in the format, couples two version
  lines, and is exactly the accretion §2.2.3 guards against.
- **A naming convention** (`model.run.json` beside `model.esm`). Rejected. It is
  invisible, unvalidated, and fails as soon as one model has two run configs.
- **Call-site only.** Rejected as the only mechanism. The run config is
  knowledge about how this model is run. Left downstream, it gets rediscovered,
  which is the solver-block RFC's POLLU argument (`solver-block.md:12-27`). The
  call site still overrides.
- **Resolve at load and emit the resolved form** (the §8.2.1 pattern). Rejected.
  It breaks the verbatim round-trip of `solver`. It puts machine paths into
  emitted documents. And no EarthSciAST code path reads the result.
- **`${VAR}` expansion** (the §4.7 optional capability). Rejected for §8.2.1's
  reasons.
- **A per-component `external` on `Model`.** Rejected. A run config covers a
  whole assembly. Leaf-level run configs would need a merge rule nobody needs.

## Open questions

1. Should a fragment select one of several configs in one file
   (`"runs.json#production"`)? Today a document names one config, and other
   runs go through the call site.
2. Should `validate` offer an opt-in existence check, as a warning only?
3. Should EarthSciIntegrator's run-config `model` key (architecture.md §4.2)
   be required to resolve back to the same document? That is an integrator
   question, but the two links point at each other and someone must own the
   rule.
4. Should Rust generalise `coupling_import_base` into one `document_base`
   shared by every verbatim reference?
