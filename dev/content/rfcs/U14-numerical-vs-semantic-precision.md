---
title: "Declared and unspecified precision: what `element_type` binds, and what an external integrator may choose"
description: "Separate semantic precision from numerical precision (D29). A declared `element_type`, on the document or on a variable, stays binding exactly as esm-spec §11.3.1 says. An omitted one becomes unspecified: every built-in evaluator still runs it in binary64, but a consumer acting on a run configuration may store state and evaluate a group of equations in binary32. It converts only between groups, is held to the integration and test tolerances, and must report what it chose. Jacobian, factor, preconditioner and table storage are the integrator's under any declaration. Format 1.3.0, with U01. A root below 1.3.0 that omits the field keeps its Float64 meaning, so no existing document or fixture changes."
---

Status: draft (proposed by EarthSciIntegrator, 2026-10-13)
Target: EarthSciAST: esm-spec §11.3 (domain table), §11.3.1, a new §11.3.3, the §6.3 `element_type` row, and the §4.7 mount rule of the pending `fix/f59-mounted-leaf-precision` branch; `esm-schema.json` `$defs/Domain/element_type`; CONFORMANCE_SPEC §5.18 (a new §5.18.6); build options on the public RHS entries of U4 (Rust) and U13 (Julia). Format version 1.3.0, released with U01. One annotation is removed from the schema; nothing becomes invalid.
Needed by: EarthSciIntegrator milestone M5g (the mixed GPU configuration, D28) and M7 (EqWeFiC leaves that leave precision unspecified; open question 1).
Integrator refs: architecture.md §4.2 (`precision`), §4.4, §7.8, §8, §13; decisions D13, D18, D24, D28, D29

File citations are relative to the `code/` directory that holds all the repositories.

## Summary

Today `element_type` has one meaning: the precision a document is evaluated in. It defaults to `"Float64"` when omitted, and a runtime must not evaluate in any other precision (`EarthSciAST/esm-spec.md:5373`, `:5399-5402`; `EarthSciAST/CONFORMANCE_SPEC.md:2964-2987`). An integrator that runs reseact's transport with f32 state therefore changes the model by definition, even though reseact never mentions precision.

D29 separates the two things that one field carries:

- **Declared precision is semantic and binding.** If a document writes `element_type` on its domain or on a variable, §11.3.1 and CONFORMANCE_SPEC §5.18 apply unchanged, to every evaluator and every consumer.
- **Omitted precision is unspecified.** The default is still binary64, and every built-in evaluator, golden and test runner keeps using it. A consumer acting on a run configuration may choose binary32 for state storage and/or evaluation, per group of equations (a leaf, for EarthSciIntegrator). It is then held to the integration and test tolerances, and must report what it chose.
- **Mixing across leaves is legal.** Values convert only at group boundaries, never inside an expression. So f32 transport state beside f64 chemistry never produces a `mixed_element_type` operator.
- **Linear-algebra and table storage are always numerical.** Jacobian and W storage, factors, preconditioners and stored tables (f16, bf16, f32) are the integrator's under any declaration, as U05 R3 already says for the first three.

The change ships in format 1.3.0 together with U01's `solver.external`. A root document below 1.3.0 that omits `domain.element_type` is read as declaring `"Float64"`, which is the rule it was written under. Every document and fixture in the repositories today is below 1.3.0 (Motivation), so nothing changes meaning.

## Motivation

**The integrator's precision design has no legal reading today.**

- D18 gives per-leaf precision roles: f32 or f64 state, and RHS evaluation at or above the state precision (`EarthSciIntegrator/docs/architecture.md:462-470`, the role table).
- On FP64-weak GPUs the plan is f32 state for dynamics and transport (`architecture.md:447-451`).
- The M5g gate runs a mixed configuration: f32 transport state, 16-bit W, chemistry on CPU (`architecture.md:615`; D28, `decisions.md:36`).
- reseact, eqice and EqWeFiC declare no `element_type` anywhere. `grep -rn '"element_type"'` over `reseact.esm/`, `eqice/` and `EqWeFiC/` finds nothing. Under today's default all three are Float64 documents, so the D28 configuration evaluates a Float64 document in binary32, which §11.3.1 forbids.
- U05 could not resolve this. Its open question 4 offered either stamping `"Float32"` into the planner's derived documents or a build-time override, and called both "a change of model, not of method" (`EarthSciIntegrator/docs/rfcs/U05-float32-integration.md:192`).

**The default conflates two kinds of author.**

1. **Authors who need binary64 semantics.**
   - MOVES onroad documents omit the field and depend on binary64. `moves.esm/moves-onroad.esm`, `onroad/model.esm` and `runs/*.esm` declare no `element_type`. The process notes say "this is an ONROAD document and evaluates in binary64" (`moves.esm/docs/process-pm-exhaust.md:1654-1655`) and that the `mixed-onroad` fixture "must be evaluated in binary64" because it carries identifier columns binary32 cannot hold (`moves.esm/docs/mixed-onroad.md:1896-1900`).
   - Conformance fixture `tests/fixtures/recurrence/02_recurrence_cancellation_ladder.esm` omits the field and pins `[1e16, 1e16, 0, 1]` at zero tolerance. `1e16` is not representable in binary32.
2. **Authors who never considered precision.** reseact, eqice and, today, EqWeFiC are in this group.

The format cannot tell these two groups apart. Authors who did think about it already write the field:
- the nonroad model declares `"Float32"` on its domain and `"Float64"` on its key columns (`moves.esm/nonroad/model.esm:7-9`, `:592`);
- the binary64 witness fixture declares `"Float64"` explicitly (`EarthSciAST/pkg/earthsci-ast-rs/tests/fixtures/precision/f64_per_op_rounding.esm:7`).

**The format already reads absence as "not declared" elsewhere.**
- For solver hints: "Absence is not a default value. ... A binding MUST NOT read absence as an assertion about the system" (`EarthSciAST/esm-spec.md:196-198`).
- The pending mount-precision branch already says "A mounted document that declares no `domain.element_type` declares no precision" and keeps `"Float64"` only "at a root" (`EarthSciAST/.moves/f59-leaf-domain/esm-spec.md:1381-1385`; branch `fix/f59-mounted-leaf-precision`, 11 commits ahead of `main`, last commit 2026-09-23).
- D29 extends that reading to the root and keeps binary64 as what runs.

**The typed documents keep absence; the evaluators and the schema erase it.**
- All five bindings keep an omitted domain field distinct from `"Float64"`:
  - Rust: `Option<String>`, `EarthSciAST/pkg/earthsci-ast-rs/src/types/components.rs:651-653`;
  - Julia: `Union{String,Nothing}`, `pkg/EarthSciAST.jl/src/types.jl:1966`;
  - Python: `str | None`, `pkg/earthsci-ast-py/src/earthsci_ast/esm_types.py:1171`;
  - Go: `*string`, `pkg/earthsci-ast-go/pkg/esm/types.go:1029`;
  - TypeScript: optional, `pkg/earthsci-ast-ts/src/generated.ts:1662`.
- Rust's evaluator collapses the distinction: `None | Some("Float64") => Ok(Precision::Float64)` (`pkg/earthsci-ast-rs/src/precision.rs:98-100`).
- The schema annotates the domain field `"default": "Float64"` (`EarthSciAST/esm-schema.json:4246`). A validator that fills in defaults would turn an omission into a declaration.
- On `main`, Julia's and Go's `ModelVariable` have no `element_type` field at all (`pkg/EarthSciAST.jl/src/types.jl:1063-1094`; `pkg/earthsci-ast-go/pkg/esm/types.go:167`), so by reading they drop a per-variable declaration on load. The f59 branch adds the field to both (`.moves/f59-leaf-domain/pkg/EarthSciAST.jl/src/types.jl:1088`; `.moves/f59-leaf-domain/pkg/earthsci-ast-go/pkg/esm/types.go:175`). Under D29 a dropped declaration turns a declared variable into an unspecified one, so this RFC depends on f59 landing.

**Two tolerance chains already exist, and the choice belongs under both.**
- `solver.abstol`/`reltol` are what the integrator holds (`esm-spec.md:223-259`).
- The assertion `tolerance` is what a test compares at (`esm-spec.md:2735-2762`).
- A precision choice is a method choice. Nothing says yet that it answers to either chain.

## Decisions of record

All proposed by this RFC; none ruled yet.

| # | Question | Decision | Why |
|---|---|---|---|
| 1 | What is a declaration? | **`element_type` on the variable, on the document's domain, or written onto the variable by a §4.7 mount.** A declaration is binding exactly as today. | This is D29. A written value is an author's statement, and the format already treats it as semantic (§11.3.1). |
| 2 | What does omission mean? | **Unspecified. It evaluates as binary64 unless a run configuration chooses otherwise.** | This is D29. Every current evaluator's output is unchanged. |
| 3 | Who may choose? | **Only a consumer acting on a run configuration**, named by `solver.external` (U01) or passed at its call site. The request must be explicit. An `"auto"` setting resolves to binary64. | Precision decides answers (§11.3.1's 1.0 vs 0.99999994 example), so it must never be inferred. Built-in bindings have no run configuration. |
| 4 | What may be chosen? | **Per evaluation group, one evaluation precision for every unspecified variable in the group. Per variable, a storage precision.** Each is Float32 or Float64. | A single precision per group keeps every expression single-precision. Storage is a separate choice: D18's "state f32, RHS f64" is a storage choice. |
| 5 | Where do conversions happen? | **Only where a value crosses a group boundary, or crosses between storage and evaluation.** Widening is exact; narrowing rounds to nearest even. Inside a group, §11.3.1 runs with the chosen precision substituted for "unspecified". | No expression sees two precisions of one unspecified variable, so `mixed_element_type` cannot arise from the choice. A declared variable that the substitution would mix into an expression is a conflict, and the consumer refuses the choice. |
| 6 | Tolerances | **The choice answers to the resolved integration tolerances**: U05 R5's floor applies to binary32 state storage. **It answers to the test tolerances unchanged.** Neither tolerance may be loosened because of the choice. | A precision is part of the method, like a step size. A test that fails under binary32 has found exactly what it exists to find. |
| 7 | Reporting | **Required.** Every result the consumer produces carries a precision record. | An answer computed in a precision the author did not state must say so (the reason for CONFORMANCE_SPEC §5.18, `:2916-2920`). |
| 8 | Linear algebra and tables | **The integrator's, under any declaration.** This covers Jacobian and W storage, factorizations, preconditioners, and stored tables: forcing buffers and arrays loaded from data sources. Each must be requested explicitly and reported. | These are U05 R3's integrator quantities, and D18 adds tables. Errors in iteration matrices only slow convergence. Table storage changes input data, not arithmetic: the value is still rounded to the reading expression's precision on read. |
| 9 | Keys | **Refuse binary32 evaluation of an unspecified variable in a key position**: a join `on` clause, or an operand of `==` or `!=` (`key_precision_unspecified`). The fix is to declare it. Validators SHOULD warn about any unspecified variable in a key position. | A ten-digit key corrupted in binary32 is 4.1e-8 relative, which no tolerance sees (`CONFORMANCE_SPEC.md:3082-3086`). This is the one silent failure a tolerance cannot catch. A declared key still works inside a binary32 group, because a comparison's operands only have to agree with each other (`CONFORMANCE_SPEC.md:3015-3021`). |
| 10 | Mounts | **f59's "differs from the mounting document's effective precision" counts unspecified as its own value.** A leaf that declares `"Float64"`, mounted into an unspecified root, gets `element_type: "Float64"` written onto its variables. | Otherwise a declaration would be lost at the mount and become eligible for binary32. |
| 11 | Versioning | **esm 1.3.0, with U01.** A root declaring `esm` < 1.3.0 that omits `domain.element_type` is read as declaring `"Float64"`. The schema loses `"default": "Float64"`. No new diagnostic for the gate. | No valid document changes meaning. The gate costs nothing: a consumer needs 1.3.0 for `solver.external` anyway. |

## Design

### 1. Spec: how a document says each

| The document has | Precision of variable `v` |
|---|---|
| `v.element_type: P` | declared `P` |
| no `v.element_type`, and `domain.element_type: P` | declared `P` |
| neither, and a mount wrote `element_type: P` onto `v` (f59) | declared `P` |
| neither, root `esm` ≥ 1.3.0 | **unspecified** |
| neither, root `esm` < 1.3.0 | declared `"Float64"` (the 1.2.0 rule) |

- Reaction-system species and parameters have no per-variable field. They follow the document's row: declared, or unspecified.
- Loop symbols, `t`, metaparameters and literals stay context-adopting (§11.3.1).
- There is no spelling for "unspecified" other than omission (open question 2).

Edits:

- The domain table row at `esm-spec.md:5373` becomes: *The precision the document is evaluated in, `"Float64"` or `"Float32"`, when declared. Omitted: unspecified (§11.3.3), which every built-in evaluator runs in binary64.*
- The variable row at `esm-spec.md:2348` ends: *Omitted means the document's, which may be unspecified (§11.3.3).*
- §11.3.1's "absent, it is the document's" (`esm-spec.md:5406`) gains: *and unspecified when the document declares none (§11.3.3).*

### 2. Spec: new esm-spec §11.3.3 "Declared and unspecified precision"

Normative rules:

- **P1. Declared is binding.** A declared precision follows §11.3.1 and CONFORMANCE_SPEC §5.18 for every evaluator and every consumer. No run configuration may narrow or widen it.
- **P2. Unspecified defaults to binary64.** It evaluates as `"Float64"` unless P3 applies. Every built-in evaluator, conformance golden and inline-test runner of an EarthSciAST binding uses binary64 for it, on every route.
- **P3. Who may choose.** A consumer acting on a run configuration MAY choose, for unspecified variables:
  - an **evaluation precision** per evaluation group;
  - a **storage precision** per variable, inside each group and between groups.

  Each is `Float32` or `Float64`. The run configuration is named by `solver.external` (§2.2.6, U01) or passed at the consumer's call site. The choice MUST be explicit in it, and a consumer MUST NOT infer one.
- **P4. Evaluation groups.** An evaluation group is a set of equations the consumer evaluates together. For EarthSciIntegrator it is a leaf, and output-time observed evaluation is a group too.
  - Within a group, every unspecified variable is read, and every operation over unspecified operands runs, at the group's evaluation precision, by §11.3.1's rules with that precision in place of "unspecified".
  - A declared variable keeps its declaration.
  - If the substitution makes an operator or an equation combine two precisions, the consumer MUST refuse the choice with `precision_choice_conflict`, naming both variables, the declaration and the run-configuration entry. It MUST NOT evaluate the group in binary64 instead.
- **P5. Conversions at boundaries only.** A value is converted when it crosses a group boundary, or crosses between storage and evaluation. This covers state, stage values, iterates, held tendencies, and observed values one group hands to another.
  - Widening is exact. Narrowing is IEEE round-to-nearest-even.
  - No expression sees two precisions of one unspecified variable, so a choice never yields `mixed_element_type`.
  - Unspecified parameters, initial conditions, forcing reads and literals round where they enter the group's evaluation, as §11.3.1 rounds them for a declared precision.
  - `t` crosses as binary64 and is rounded where it is read (U05 R4).
- **P6. Integrator quantities.** These are the consumer's under any declaration, extending U05 R3 to tables:
  - time, step size, error estimates and norms, tolerances, controller state;
  - Jacobian and W storage, factorizations, preconditioners;
  - stored tables (forcing buffers and data-source arrays).

  Storing a table below the precision of the variable it binds is permitted when the run configuration asks for it. A value read from such a table is widened, then rounded to the reading expression's precision.
- **P7. Tolerances.**
  - (a) **Floor.** A group that stores the state of any unspecified variable in binary32 is subject to U05 R5's floor. A resolved relative tolerance below `2^-22` MUST be refused with `tolerance_below_working_precision`, naming the tolerance's source and the run-configuration entry. A group that evaluates in binary32 but stores in binary64 SHOULD warn below the floor.
  - (b) **No loosening.** A consumer MUST NOT loosen an integration tolerance (§2.2.2) or an assertion tolerance (§6.6.4) because of its choice.
  - (c) **Tests.** An inline test run under a choice is judged at its resolved §6.6.4 tolerance. A failure is reported as a failure of that run, with the precision record (P9). A zero-tolerance test therefore effectively requires binary64, which is correct.
- **P8. Keys.** A consumer MUST refuse binary32 evaluation of an unspecified variable in a key position, with `key_precision_unspecified` naming the variable and the expression. A key position is a join `on` clause or an operand of `==` or `!=`. Loop symbols and literals are not variables, so `k == 1` is not a key position. The author's fix is to declare the variable: a declared `"Float64"` key compared inside a binary32 group is legal by §11.3.1's comparison rule. A validator SHOULD emit the same code as a warning for any unspecified variable in a key position, whatever the run. Arithmetic on a key, such as `floor(scc/1000)*1000` (`esm-spec.md:5441-5445`), is not a key position, and this rule does not catch it. That case is covered only by the author declaring the key.
- **P9. Reporting.** A consumer that acts on a run configuration MUST attach a precision record (§5) to every result it produces: trajectories, output files, observed reads and inline-test verdicts.
- **P10. Built-ins never choose.** A binding's own `solve`, `init`, observed and test entry points MUST NOT exercise P3.

New diagnostics (esm-spec diagnostics table): `precision_choice_conflict` and `key_precision_unspecified`. `tolerance_below_working_precision` comes from U05.

### 3. Spec: mounts (amends the f59 branch's §4.7 text)

In `.moves/f59-leaf-domain/esm-spec.md:1381`:

- The effective precision of a root that declares none becomes **unspecified** at esm ≥ 1.3.0, replacing `"Float64"`.
- The comparison at each mount edge treats unspecified as a value of its own:
  - a leaf declaring `P`, mounted under an unspecified effective precision, writes `element_type: P` onto its variables, including `P = "Float64"`;
  - an undeclared leaf adopts the mounting document's effective precision, which may be unspecified.

### 4. Schema

`$defs/Domain/element_type` (`esm-schema.json:4239-4247`):

- Remove `"default": "Float64"`.
- The description becomes: *Declared working precision of the document (esm-spec §11.3.1). Omitted: unspecified (§11.3.3), evaluated in binary64 by every built-in evaluator.*

The variable field's description (`esm-schema.json:1751`) ends: *Omitted means the document's, which may be unspecified.*

Both edits are annotations, so validation is unchanged. Copy them to the embedded schemas in the five packages.

### 5. The precision record

Shape, versioned on its own:

```json
{
  "esm_precision_record": "1.0",
  "variables": {
    "Transport.O3": { "precision": "unspecified", "between_groups": "Float64" },
    "Met.T":        { "precision": "unspecified", "between_groups": "Float32" },
    "Keys.scc":     { "precision": "declared",    "element_type": "Float64" }
  },
  "groups": {
    "transport": { "evaluation": "Float32", "state": { "Transport.O3": "Float32" } },
    "chemistry": { "evaluation": "Float64", "state": { "Transport.O3": "Float64" } },
    "output":    { "evaluation": "Float64" }
  },
  "integrator": {
    "time": "Float64", "step": "Float64", "norms": "Float64",
    "jacobian": { "chemistry": "BFloat16" }, "factorization": { "chemistry": "BFloat16" },
    "tables": { "NEI2016.*": "Float32" }
  }
}
```

Requirements:

- Every variable the run reads or writes appears in `variables`.
- Every group the run evaluated appears in `groups`.
- `integrator` uses U05 R3's quantity classes plus `tables`.
- The record is canonical JSON (sorted keys), so two implementations that made the same choice emit the same bytes.
- Where it goes is the consumer's choice, as long as it travels with the result. EarthSciIntegrator's placement is in §8.

### 6. Binding APIs

**All five bindings.** A pure accessor that applies §1's table, including the version gate and f59's written fields:

- `precision_source(document, variable) -> Declared(Float32 | Float64) | Unspecified`
- Absence MUST survive `load → emit` (it does today for the domain field). The per-variable field MUST survive too, which needs f59 in Julia and Go.

**Rust (with U4 and U5).** U4's `RhsProgram` gains a build option and one accessor. U5's `state_precisions()` now returns resolved precisions:

```rust
pub enum PrecisionSource { Declared(Precision), Unspecified }

pub struct RhsOptions {
    /// Evaluation precision for variables whose precision is unspecified
    /// (esm-spec §11.3.3 P3). Default Float64.
    pub unspecified_evaluation: Precision,
    // ...U3 / U4 / U6 options
}

impl RhsProgram {
    /// Per flat state slot: declared or unspecified.
    pub fn precision_sources(&self) -> &[PrecisionSource];
    /// Per flat state slot: the precision it is evaluated at, after resolution.
    pub fn state_precisions(&self) -> &[Precision];   // U5's accessor, now resolved
}
```

- The build applies P4 through the existing per-variable machinery:
  - the program's document precision becomes `unspecified_evaluation`;
  - declared variables enter the override table (`precision.rs:287-293`) with their declared precision.
- It refuses:
  - a mix that results, with `precision_choice_conflict`;
  - a P8 key, with `key_precision_unspecified`;
  - a request that changes no variable the program reads or writes (`precision_choice_conflict`, naming the declaration that makes it inert). Otherwise a run would ask for binary32 and silently get binary64.
- U5's `eval_into::<f32>` is legal when every slot resolves to Float32, whether declared or by this option. `eval_into::<f64>` is always legal.

**Julia (with U13 and U5).**

- `operator_ir(prob; …, unspecified_eltype = Float64)`. U13's `emit_fragment(ir; eltype)` takes `eltype = Float32` only when every variable the operator reads or writes resolves to Float32 in the same way.
- On the native path, `unspecified_eltype = Float32` needs U5 §4 step 2 (typed literals and const data). Until that lands Julia refuses it with `float32_unsupported`, construct `"binary32 evaluation of unspecified precision"`.
- f32 storage with f64 evaluation needs nothing from EarthSciAST. The consumer widens its stored state exactly before calling `f!`.

**Python, Go and TypeScript.** The accessor and round-trip only.

### 7. Built-in bindings

Unchanged.

- Julia's, Rust's and Python's `solve`, `init`, observed reads and test runners evaluate unspecified precision in binary64 (P2, P10). They gain no knob.
- Rust's `Precision::from_element_type(None)` keeps returning `Float64` (`precision.rs:98-100`); the new accessor sits beside it.
- U5's `SolutionMetadata.integrator_precision` gains no new obligation, because built-ins never choose.

### 8. What EarthSciIntegrator does with it (informative)

- **Request.** The run config's `precision` key (architecture §4.2, §8; D18; JSON per D24) names per-leaf roles: state, RHS evaluation, W, factors, preconditioner, tables. U01's resolution order for a consumer is call site, then run config, then document hint, then default. For precision, a declaration ranks above all of them, because it is not a hint.
- **Plan.** The planner resolves each leaf's roles against `precision_source`.
  - P4 conflicts, P7 floor violations and P8 keys are plan errors.
  - D18 adds one rule of its own: RHS evaluation precision ≥ state precision. P3 allows more than this.
  - The resolved record is written into `plan.json`, so the Julia and Rust planners must agree on it byte for byte (D8, architecture §3.3).
- **Run.**
  - Each leaf's RHS is built with the leaf's evaluation precision (`RhsOptions::unspecified_evaluation`, or U13's `eltype`).
  - Each leaf stores stage vectors at its state precision and converts at entry and exit.
  - The run copies the record into solution metadata and into the global attributes of every EarthSciIO output (`esm_precision_record`, a JSON string).
- **M5g (D28).** reseact declares nothing, so the configuration is legal under P3:
  - transport: f32 state and evaluation through a U13 fragment with `eltype = Float32`;
  - chemistry: f64 on CPU, bf16 W under P6.

  The transport `reltol` of 1e-4 in architecture §4.4 (`architecture.md:214`) is above the `2^-22` floor.
- **M7.** Architecture §8 says EqWeFiC's real*4 ports declare `"Float32"` (`architecture.md:457`). Those leaves are then U5's, not this RFC's, and no EqWeFiC document declares it yet. This RFC matters to M7 for whatever EqWeFiC leaves unspecified, which is where WRF-Chem's double-precision chemistry sits (open question 1).

### 9. Backward compatibility

- **Documents below 1.3.0 keep their meaning** (P2 and the version gate).
  - Every `.esm` under `EarthSciAST/tests/` declares a version below 1.3.0, except a handful of version-compatibility probes (`1.10.0`, `2.5.1`, `12.34.56`); those are evaluated in binary64 by P2 anyway.
  - Every MOVES document is at 1.1.0 or 1.2.0.
  - reseact is at 1.0.0 (`reseact.esm/reseact.esm:2`).
  - eqice and EqWeFiC are at 1.0.0 or 1.1.0.
- **No conformance fixture changes value.** Built-ins run unspecified precision in binary64, and every fixture is below 1.3.0 regardless. The zero-tolerance fixtures that silently rely on binary64 keep passing unchanged. These are the recurrence fixtures 01–05 and 07–09 (`tests/fixtures/recurrence/`, all `esm: 1.1.0`, no `element_type`).
- **When a document moves to 1.3.0,** an author who needs binary64 semantics writes `"element_type": "Float64"`, on the domain or on the variables that need it. `MIGRATION.md` gets a row saying so. MOVES onroad roots are the known case (Motivation). Fixtures are only affected if someone bumps them.
- **Models whose authors meant "don't care"** become eligible for a run-config choice when they move to 1.3.0, which they must do anyway to use `solver.external`. These are reseact, eqice and EqWeFiC.

## What this is not

- **Not a change to declared precision.** A document that writes `"Float64"` cannot be run in binary32 by anyone. U5 still governs integrating a document that declares `"Float32"`.
- **Not an `"auto"` precision.** Nothing infers binary32. P3 requires an explicit request.
- **Not mixing inside an expression.** Mixed precision across leaves comes from boundary conversion. Within a group every unspecified operand has one precision, and `mixed_element_type` still applies to declared mixes.
- **Not a knob on built-in solvers.** Rust's in-crate diffsol solver, Julia's SciML path and Python's scipy path keep binary64 for unspecified precision.
- **Not a run-config format.** That is EarthSciIntegrator's (D24). This RFC fixes only what any consumer may do and must report.
- **Not 16-bit state.** P3 offers Float32 and Float64 only. 16-bit appears only in P6 storage (architecture §8 explains why state cannot be 16-bit).

## Conformance and tests

New CONFORMANCE_SPEC §5.18.6 "Declared and unspecified precision". Fixtures under `tests/conformance/precision_unspecified/`.

1. **Round trip and schema.** A 1.3.0 document that omits `domain.element_type` re-emits without it in all five bindings. A per-variable declaration survives in all five. `esm-schema.json` carries no `default` on the domain field.
2. **Source accessor.** Five documents, with exact agreement across the five bindings:
   - 1.3.0, field omitted → `Unspecified`;
   - 1.3.0 declaring `"Float64"` → `Declared(Float64)`;
   - 1.2.0, field omitted → `Declared(Float64)`;
   - 1.3.0, field omitted, one variable declaring `"Float64"` → that variable `Declared`, the rest `Unspecified`;
   - a leaf declaring `"Float64"` mounted into an unspecified 1.3.0 root → the leaf's variables `Declared(Float64)` (Decision 10).
3. **Built-ins unchanged.** A 1.3.0 twin of `f64_per_op_rounding.esm` with the domain field removed evaluates to exactly `1.0` on every executing binding and route, at zero tolerance.
4. **The build option** (Rust `bindings_required`; Julia once U5 §4 step 2 lands):
   - the same twin built with `unspecified_evaluation = Float32` gives `0.99999994` (`0x3F7FFFFF`) at zero tolerance;
   - the declared-`"Float64"` original built with the option is refused (`precision_choice_conflict`, inert request).
5. **Conflict.** An omitted domain field, `k` declaring `"Float64"`, and `y ~ q * k` with `q` unspecified. With the option the build is refused, naming `q`, `k` and the option. Without it, it builds.
6. **Key guard.** A 1.3.0 twin of the per-variable witness `pkg/earthsci-ast-rs/tests/fixtures/precision/f32_per_variable_element_type.esm`, with the domain field and the key declarations removed. Its `sccExact == sccNeighbour` comparison (`:95`) is a key position.
   - With the option, the build is refused (`key_precision_unspecified`).
   - Validation warns either way.
   - With the keys declared `"Float64"` again and the option on, it builds and reproduces the original's zero-tolerance answers.

Tests that live in EarthSciIntegrator's suite (informative):
- the P7 floor;
- an inline test that fails under a binary32 choice and is reported as failed, not skipped;
- byte-identical precision records from both planners;
- the M5g pair (f64 and mixed) agreeing within the gate's tolerance.

Compliance matrix: one new PREC-11-B row per rule P1–P10.

## Work breakdown

1. **Spec.** §11.3 table row; §11.3.1 cross-reference; new §11.3.3 (P1–P10); §6.3 row; the f59 §4.7 amendment; diagnostics table; CONFORMANCE_SPEC §5.18.6; compliance matrix rows; `MIGRATION.md` row.
2. **Prerequisite.** Land `fix/f59-mounted-leaf-precision`, so Julia and Go keep per-variable `element_type`.
3. **Schema.** Drop the domain `default`; update both descriptions; regenerate the embedded copies.
4. **All bindings.** `precision_source` with the version gate; round-trip tests.
5. **Rust.** `RhsOptions::unspecified_evaluation`, `precision_sources()`, the three refusals, and the record type (shared with U5's `IntegratorPrecision`).
6. **Julia.** The `operator_ir` keyword and U13's `eltype` rule. Refuse native binary32 evaluation until U5 §4 step 2.
7. **Fixtures** 1–6, and `tests/COVERAGE_MATRIX.md`.
8. **EarthSciIntegrator.** Run-config schema for `precision`, planner resolution and record, runtime conversions, output attributes.

## Alternatives considered

- **Keep omission as Float64 and add a run-time narrowing flag.** Any document could then be narrowed, which makes a written `"Float64"` meaningless. U5 rejected caller overrides for the same reason (`U05-float32-integration.md:152`). Rejected.
- **Stamp `"Float32"` into the planner's derived documents** (U5 open question 4). That declares a semantics the author never stated. Derived documents are golden-gated conformance artifacts (architecture §3.3), so they would carry a run choice as if it were model text. Rejected.
- **Add an explicit `"unspecified"` (or `"Any"`) enum member and keep omission as Float64.** Every model would have to edit itself to become eligible: reseact, eqice and every EqWeFiC root. D29 chose omission. An explicit member may still be wanted for variables inside a declared document (open question 2).
- **No version gate: omission is unspecified at every version.** Workable, since built-ins do not change. Rejected because a 1.2.0 document was written under a rule that said Float64, and the gate costs nothing given U01's 1.3.0 requirement.
- **Per-variable evaluation precision inside a group.** Two unspecified operands at different precisions in one expression is exactly `mixed_element_type`. Rejected in favour of one evaluation precision per group.
- **Let the planner's `"auto"` pick binary32 where it estimates the error is small.** Precision decides discrete outcomes (§11.3.1's row-count example), which no error estimate bounds. Rejected.

## Open questions

1. **EqWeFiC's chemistry: declared or unspecified?** Architecture §8 has EqWeFiC's real*4 ports declare `"Float32"`, which reproduces WRF's per-operation rounding; U5's motivation cites the hydrostatic-imbalance and MOSAIC findings. WRF-Chem's KPP chemistry is double precision.
   - If the chemistry components declare `"Float64"`, a rate that reads a declared-Float32 temperature is `mixed_element_type`, and only U5's open question 3 (a widening operator) can fix that.
   - If they leave precision unspecified and the chemistry group evaluates at binary64, P4 refuses the choice. The declared temperature stays binary32 (P1 forbids widening it at the boundary), so `k(T) * O3` combines binary32 and binary64.
   - The split works under this RFC only if the meteorology the chemistry reads is also left unspecified, so the integrator chooses f32 for the dynamics leaves numerically. That gives up WRF's real*4 semantics for those variables.

   The M7 gate should say which document carries which declaration.
2. **Unspecified inside a declared document.** Should a variable be able to opt out of its document's declaration (`"element_type": "unspecified"`)? No current model needs it.
3. **The floor for evaluation-only reduction.** P7(a) only warns when state is binary64 and evaluation binary32. Implicit methods whose Newton residual is evaluated in binary32 may need a hard floor.
4. **Inline `const` tables.** P6 treats data-source arrays and forcing as tables. Should a document's own inline `const` arrays count as tables, or stay literals that follow the evaluation precision?
5. **Where the record's schema lives.** It could be in esm-spec, as proposed, or consumers could only be obliged to report the listed content in a shape of their own.
6. **A later knob for built-ins.** Should Rust's in-crate solver (binary32 diffsol, U5) also accept a run-config-style choice for unspecified documents? P10 says no for now.
