---
title: "Integrating a Float32 document: what the declaration governs, and what the integrator owns"
description: "Replace the blanket refusal to time-integrate a `domain.element_type: \"Float32\"` document with a contract. The declaration governs every document value (right-hand side, observeds, events, and the state itself); the integrator owns its clock, step size, norms, tolerances and linear algebra, and reports their precision. Rust's in-crate solver integrates Float32 documents on a binary32 diffsol; an external integrator may keep an f64 clock and f64 norms. No schema change."
---

Status: draft (proposed by EarthSciIntegrator, 2026-10-13)
Target: EarthSciAST — esm-spec §11.3.1 (new §11.3.2), CONFORMANCE_SPEC §5.18.2 clause 3 and a new §5.18.5, `earthsci-ast-rs` (`solve`, `init`, and the public RHS entry of U4); refusal fixes in Julia and Python. Behavioural only: no schema change and no format version bump.
Needed by: EarthSciIntegrator milestone M7 (EqWeFiC single column in f32). Julia also needs its evaluation rules (R1, R4) for M5g's mixed configuration (D28). reseact declares no `element_type`, so the choice of f32 there is U14's, not this RFC's.
Integrator refs: architecture.md §4.4 (EqWeFiC example), §7.8, §8, §13 (U5); decisions D1, D13, D18, D28, D29; U14

## Summary

Today no binding can integrate a document that declares `"Float32"` as the spec defines it. Rust refuses. Julia and Python integrate it in binary64 and say nothing.

Rust's refusal rests on one sentence: the solver's step-size control, error norms and Newton solve are binary64, "so the trajectory would be binary64 even though the right-hand side is binary32" (`pkg/earthsci-ast-rs/src/problem.rs:2075-2083`). That sentence treats two things as one. The trajectory is the sequence of **state values**. The step size and the error norm are the **integrator's** numbers. The document declares the first and says nothing about the second.

This RFC separates them:

- **The declaration governs document values.** Every evaluation is binary32 (unchanged). Every state value, being a value of a declared variable, is binary32 too. An integrator must store it that way, not carry binary64 and round at the end.
- **The integrator owns its own quantities.** Time, dt, error norms, tolerances, controller state and W factorizations can be any precision. The binding reports which.
- **Time crosses the RHS boundary as f64.** The model reads `t` rounded to the precision of the expression that reads it, which is already the rule.

With that contract, Rust's in-crate `solve`/`init` integrate Float32 documents on diffsol instantiated over `f32`. EarthSciIntegrator integrates them with an f64 clock and f64 norms, per D13 and D18.

This RFC governs **declared** precision. Under D29 a document that omits `element_type` has unspecified precision, and U14 says what an integrator may choose for it.

## Motivation

**What Rust refuses.** `reject_f32_integration` (`pkg/earthsci-ast-rs/src/problem.rs:2086-2108`) returns `Float32Unsupported` for any Float32 document with at least one `D(x, t)` rule (`src/simulate_array/driver.rs:425-433`). It runs at the top of `solve` (`problem.rs:2894`) and `init` (`problem.rs:3183`). Algebraic and observed evaluation are unaffected and already run in binary32.

**The spec makes the refusal normative.** CONFORMANCE_SPEC §5.18.2 clause 3 (`CONFORMANCE_SPEC.md:2978-2983`) says time integration of a Float32 document "is an error naming it", because "an ODE/DAE solver's step-size control, error norms and Newton solve are binary64 in every binding here". The compliance matrix repeats it as PREC-11-A-009 (`ESM_COMPLIANCE_VALIDATION_MATRIX.md:1373`). The reason is a fact about the bindings' solvers in 2026, written as a rule about the format.

**The other bindings do not refuse.** Julia, Python, Go and TypeScript "parse and round-trip `domain.element_type` but do not honour it: a `"Float32"` document evaluates in binary64 there" (`ESM_COMPLIANCE_VALIDATION_MATRIX.md:1387-1390`). In Julia, `element_type` "is read nowhere in Julia's evaluator" (`:886-888`). Julia does refuse a `Float32` state vector at the RHS entry (`pkg/EarthSciAST.jl/src/tree_walk/compile.jl:1449-1466`, called from `tree_walk/acc_merge.jl:655`), because its literals and const data are `Float64` and would promote. Python rounds only inside recurrences (`pkg/earthsci-ast-py/src/earthsci_ast/numpy_interpreter.py:4782-4788`). So a Float32 document is either refused (Rust) or silently integrated in binary64 (Julia, Python). Neither is the declared semantics.

**The motivating model is WRF.** EqWeFiC ports WRF, which is built `REAL*4`, and checks against real32 builds:

- Tests against single-precision WRF references use `rel: 1e-5`; double-precision kernel replays use `rel: 1e-9` (`EqWeFiC/CLAUDE.md:42`). The real64 YSU replay reproduces WRF's real32 tendencies only to about 1e-5 relative, and about 1e-3 for θ because of "single-precision cancellation in (θ_new − θ_old)/dt" (`EqWeFiC/PLAN.md:184-190`).
- Constants are bound to their binary32 roundings, stored as binary64 parameters. New Tiedtke carries `t13 = 0.3333333432674408` and `tmelt = 273.1600036621094` (`components/atmospheric_dynamics/ntiedtke/ntiedtke_parameters.esm:39`, `:45`), and lists ten such roundings. It notes that a 1.3e-8 shift in `tmelt` "moves q_sat by 2.6e-7 relative" (`:5`). YSU and WSM6 have the same defect (`FORTRAN_BUGS.md:10`, note N2; `ysu_parameters.esm:5`).
- Some real*4 behaviour is a property of the state, not of any one expression. "The REAL*4 state at a mid-run step is not in float64 hydrostatic balance, and the imbalance is the same size as the entire physical signal in w" (`FORTRAN_BUGS.md:91`, N70; rounding part in `:86`, N67). MOSAIC's real32 stage increments keep "one to two significant digits" (`FORTRAN_BUGS.md:127`, N87). A binary64 integration cannot show any of this.

No EqWeFiC document declares `element_type` today; `grep -rn '"element_type"'` over the repository finds nothing. The workarounds are real64 kernel builds and loose bands (`EqWeFiC/PLAN.md:2492`). The integrator design plans f32 state for EqWeFiC (`architecture.md:267`), whose real*4 ports are to declare `"Float32"` (architecture §8). It also plans f32 state for GPU transport (D28) on reseact, which declares nothing, so that case is U14's. The survey flags the blocker: "Rust refuses f32 integration today" (`survey.md:39`).

**The integrator's design already splits the roles.** D13 keeps "time, dt, error norms and global reductions" in f64 (`decisions.md:21`). D18 and architecture §8 give per-leaf roles: state in f32 or f64, while time, dt, norms, reductions and adjoint accumulators are f64 only (`architecture.md:462-470`). Under today's §5.18.2 that design is non-conformant by definition for a Float32 document. Under this RFC it is conformant. For a document that declares no `element_type`, D29 makes the state and RHS roles a numerical choice (U14).

**The RHS entry does not carry its precision across threads.** The working precision is a thread-local (`pkg/earthsci-ast-rs/src/precision.rs:135-149`). `solve` arms it (`problem.rs:2893`). The `debug_eval_rhs*` entries do not: "a caller comparing them on a document that declares element types holds this guard around the calls" (`src/simulate_array/driver.rs:583-591`). An external integrator calling the RHS from a worker thread would get binary64 kernels around literals that were rounded to binary32 when the tape was lowered (`tape/lower.rs:919-921`). That answer is neither precision, and nothing in it says so.

## Decisions of record

All proposed by this RFC; none ruled yet.

| # | Question | Decision | Why |
|---|---|---|---|
| 1 | What does `element_type` govern during integration? | **Document values only:** every evaluation, and every state value. Not the integrator's clock, step size, norms, tolerances or linear algebra. This applies to a declared `element_type`. An omitted one is unspecified (U14). | The document carries no algorithm (the solver-block RFC excludes algorithm names), so it cannot declare an algorithm's arithmetic. It does declare its variables, and the state is variables. |
| 2 | Store the state in binary32, or round at the end? | **Store in binary32.** Round on every store of an accepted state. | Rounding only at output gives a *better* answer than the `real*4` reference. CONFORMANCE_SPEC already rejects that for recurrences (§5.19.3a) and compiled emitters (§5.38.4). A binary64 store keeps increments that binary32 loses. |
| 3 | What type is `t` at the RHS boundary? | **`f64`, always.** The model reads `t` at the precision of the expression that reads it. | `t` is context-adopting (esm-spec §11.3.1; `precision_infer.rs:37-38`), and the per-cell evaluator already rounds it on read (`simulate_array/eval.rs:87-88`). D13 keeps the integrator's clock f64. An f32 `t` would lose seconds over a long run. |
| 4 | Rust's in-crate solver | **diffsol instantiated over `f32`.** No f64-with-rounding path. | diffsol 0.11 implements `Scalar` for `f32` (`diffsol-0.11.0/src/scalar/mod.rs:106`). Its faer LU is generic (`linear_solver/faer/lu.rs:31`). Everything it stores is then binary32, so nothing is silently wider. |
| 5 | State with mixed Float32 and Float64 variables | **In-crate solver refuses; external integrators may integrate it** with per-slot rounding. | diffsol has one scalar type for the whole state. A per-slot integrator has no such limit. |
| 6 | A tolerance finer than the state's precision | **Refuse** (`tolerance_below_working_precision`). | Clamping it silently is a fallback. Leaving it alone stalls the controller. |
| 7 | Versioning | **No bump.** | No schema change. A refusal becomes an acceptance. No document that was valid changes meaning. |

## Design

### 1. Spec: new esm-spec §11.3.2 "Integrating a Float32 document"

The text below replaces CONFORMANCE_SPEC §5.18.2 clause 3. It applies per variable: a variable whose `element_type` is `"Float64"` (§5.18.2a) follows it at binary64. It governs declared precision only. A variable whose precision is unspecified follows U14's §11.3.3.

| Quantity | Owner | Precision |
|---|---|---|
| Evaluations: RHS, observeds, event conditions and affects, initial conditions | document | declared (§11.3.1), unchanged |
| State values: accepted states, and stage values and Newton iterates as the RHS reads them | document | declared |
| `t` as read by an expression | document | that expression's precision (context-adopting) |
| Parameter values, including ones the integrator sets at run time (`bind_dt`) | document | declared, rounded on ingress |
| Clock `t`, step size, error estimates and norms, tolerances, controller state | integrator | integrator's choice, reported |
| Jacobian and W storage, factorizations, Newton corrections before they are added to a state | integrator | integrator's choice, reported |

Normative rules:

- **R1. Evaluation.** Every evaluation during integration follows §11.3.1 exactly.
- **R2. State.** Every accepted state is stored rounded to each variable's declared precision, once per store. A binding MUST NOT carry a wider state across steps and round only on output. A stage value or Newton iterate is rounded at the latest when the RHS reads it; the RHS entry's ingress rounding does this.
- **R3. Integrator quantities.** The spec does not fix their precision. A binding MUST report what it used, per quantity class, in its solution metadata.
- **R4. Time.** `t` crosses the RHS boundary as binary64. The model reads it rounded to the reading expression's precision. A binary64 clock is therefore never visible to a binary32 expression.
- **R5. Tolerance floor.** A relative tolerance below `2^-22` (about 2.4e-7, two binary32 ulps at 1.0) for a Float32 state cannot be met. A binding MUST refuse it with `tolerance_below_working_precision`, naming the tolerance's source (call site, `solver.reltol`, or default). Validation SHOULD warn when a Float32 document's `solver.reltol` is below the floor.
- **R6. Refusal.** A binding that cannot meet R1–R5 MUST refuse integration with `float32_unsupported`, construct `"time integration"`. It MUST NOT integrate in another precision.

### 2. Rust: the RHS entry (with U4)

U4 makes the RHS a public, `Send + Sync` entry point, `RhsProgram`, and owns its full signature (U04 §2). This RFC adds only its precision behaviour.

```rust
/// A scalar a compiled RHS exchanges with an integrator. Sealed: f32 and f64.
pub trait StateScalar: Copy + Send + Sync + 'static + sealed::Sealed {
    const PRECISION: Precision;
    fn widen(self) -> f64;      // exact
    fn narrow(v: f64) -> Self;  // round to nearest, ties to even
}

impl RhsProgram {
    /// The document's `domain.element_type`.
    pub fn precision(&self) -> Precision;
    /// One entry per flat state slot: the precision its variable declares.
    pub fn state_precisions(&self) -> &[Precision];
    /// Round `u` in place, each slot to its declared precision (R2).
    pub fn round_state(&self, u: &mut [f64]);
    /// dy = f(u, p, t).
    pub fn eval_into<T: StateScalar>(
        &self, u: &[T], p: &[f64], t: f64, forcing: &ForcingSet,
        dy: &mut [T], ws: &mut RhsWorkspace,
    ) -> Result<(), RhsError>;
}
```

Semantics:

- `eval_into` arms the problem's precision environment (`precision::Env::enter`, `precision.rs:305`) on the calling thread for the duration of the call. Callers never hold a guard. This fixes the thread-local gap above.
- `T = f32` is legal only when every slot is Float32: declared, or unspecified and built with U14's `unspecified_evaluation = Float32`. Otherwise it returns `RhsError::StatePrecision { variable, declared }`, naming the first Float64 variable.
- `T = f64` is always legal. Slots declared Float32 are narrowed on ingress into the workspace, never in the caller's buffer. Their `dy` entries come out binary32-representable.
- `p` is positional, as in `debug_resolve_params` (`driver.rs:596-607`). Slots of Float32 parameters are rounded on read. Today only the build-time values are rounded (`problem.rs:1634-1637`), so a run-time write such as `bind_dt` would arrive unrounded.
- Forcing buffers are array data, and array reads are not rounded (`eval.rs:79-83`). Whatever API writes a forcing buffer at run time (U4, U11) MUST round on write to the bound variable's precision.

### 3. Rust: in-crate `solve` and `init`

- Delete `reject_f32_integration` and its two calls. Make the diffsol drivers generic over `T: StateScalar`. They are typed `FaerVec<f64>` today (`simulate_array/driver.rs:1984-2026`, `:2138`; `simulate/driver.rs:146`, `:186`).
- A Float32 document whose state is all Float32 runs on `FaerVec<f32>`, `FaerMat<f32>`, `FaerLU<f32>`. Every stored state is binary32 (R2). diffsol's time is the same scalar as its state (`diffsol-0.11.0/src/op/mod.rs:36`, `:186`), so the clock, step and norms are binary32 too. That is legal under R3.
- `SolutionMetadata` (`simulate/api.rs:488`) gains `integrator_precision: IntegratorPrecision { state, time, norms, linear_algebra }`. Here all four are `Float32`. `Solution.time` stays `Vec<f64>` and holds binary32 values exactly.
- A binary32 clock cannot separate times closer than one ulp. `solve`/`init` MUST refuse (`float32_unsupported`, construct `"binary32 clock over tspan"`) when `t0` and `t_end`, or two consecutive requested output times, round to the same binary32 value.
- A state mixing Float32 and Float64 variables still refuses, with construct `"time integration of a state mixing Float32 and Float64 variables"`.
- R5 runs after the tolerance resolution order (esm-spec §2.2.2, `esm-spec.md:233`), so a document's `solver.reltol` can trip it.

### 4. Julia

- **Now, as a conformance fix:** refuse to integrate a Float32 document (R6) until Julia honours `element_type`. Today it integrates it in binary64, which §5.18.2 already forbids.
- **Then, as PREC-11-A for Julia:** type the compiled RHS's literals and const data with the document's element type. Julia's `Float32` then rounds per operation natively, and `exp(::Float32)` is the binary32 entry point §5.18.1 asks for. The `E_TREEWALK_FLOAT32_STATE` guard (`compile.jl:1460-1466`) is lifted exactly for Float32 documents; its reason, Float64 literals, no longer holds. The entry has the parity signature `f!(du::AbstractVector{T}, u::AbstractVector{T}, p, t::Float64)` with `T ∈ {Float32, Float64}`, and `Float32(t)` at each binary32 read of `t`. It also gets accessors `element_type(prob)` and `state_element_types(prob)`.
- The SciML `solve` path may pass `t` as `Float32` when `tspan` is `Float32`. That is allowed; R4 is about the RHS boundary, and `Float32(::Float32)` is the identity.

EarthSciIntegrator's Julia implementation needs the second step for M7, not only the first.

### 5. Python

Python refuses integration of a Float32 document with `float32_unsupported` (R6) in `simulate`/`solve`. It has no binary32 integrator, and scipy's are binary64. Its other `element_type` gaps (PREC-11-A-001…017) are tracked in the compliance matrix and are not this RFC's. A later native Python integrator (D1) would adopt §11.3.2 like the others.

### 6. Go and TypeScript

No simulator, so nothing changes.

### 7. What EarthSciIntegrator does with it (informative)

- It keeps the clock, dt, norms and reductions in f64 (D13). It stores state and stage vectors in f32 for f32 leaves (D18), rounding each accepted state on store. An f32 leaf is one whose variables declare Float32, or one whose unspecified precision the run config sets to f32 (U14).
- It calls `eval_into::<f32>` for all-Float32 leaves and `eval_into::<f64>` with `round_state` for mixed ones. It reports its precision roles in `plan.json`, which satisfies R3.
- Its W and LU storage may be f16/bf16 (D18, architecture §8). These are integrator quantities under R3.

## What this is not

- **Not a storage hint.** For a declared precision, "store as f32, compute in f64" is still wrong (§11.3.1). This RFC adds the converse rule: "compute in f32, store in f64" is wrong too (R2). Where precision is unspecified, U14 allows both as numerical choices, reported.
- **Not a caller override of `element_type`.** A run cannot narrow a document that declares Float64 to binary32, and cannot widen one that declares Float32. A document that declares nothing has unspecified precision (D29). An integrator's run config may then choose f32 per leaf (U14). That is how D28's f32 transport state on reseact is legal.
- **Not a widening operator.** Coupling declared Float32 meteorology to Float64 chemistry inside one document is still `mixed_element_type` (open question 3).
- **Not compiled-emitter support.** Rust's XLA emitter refuses Float32 (`tape/xla_emit.rs:242-248`). Compiled-RHS conformance excludes precision-changing fixtures (CONFORMANCE_SPEC §5.38.4). Lowering Float32 through either emitter belongs to U13.
- **Not U4.** The thread-safe public entry point is U4's. This RFC only fixes its precision semantics.

## Conformance and tests

New CONFORMANCE_SPEC §5.18.5 "Integration gate". Fixtures live under `tests/conformance/precision_integration/`. `bindings_required` is `["rust"]` at first, and grows to Julia when it implements step 2 of §4.

1. **Vanishing increment (R2 non-vacuity).** `D(x) = 1e-8` with `x(0) = 1`, and a fast companion `D(y) = cos(2πt)` that holds the step under 1 s. Integrate 100 s. A conforming binary32 integration gives `x(100) == 1.0f0` **exactly**: each step's increment is below half an ulp of 1.0 (`2^-24 ≈ 5.96e-8`). A binary64 store gives `1.000001`, eight ulps above. The description names that value, as §5.19.3a requires. Zero tolerance.
2. **Clock read (R4).** An observed `z ~ t - 100000000` read at `t = 100000001` (binary64). Under Float32, `t` rounds to `1.0e8` (binary32 ulp 8), so `z == 0` exactly on every route: oracle, vectorized, tape, and `eval_into` with each `T`. The tape reads `t` unrounded (`tape/exec/resolve.rs:66`) and relies on the kernels to narrow it. A second observed `w ~ t` checks the bare read. Zero tolerance.
3. **Entry consistency.** On the §5.18.4 witness document, `eval_into::<f32>` and `eval_into::<f64>` are bitwise equal to the guarded oracle. One call is made from a fresh thread with no guard held.
4. **Trajectory band.** Each §5.9 PDE fixture gets a Float32 twin. Its trajectory is compared to the Float64 golden at `rtol 1e-4`, `atol` scaled per fixture. This band checks accuracy, not semantics; fixtures 1 and 2 check semantics.
5. **Refusals.** Float32 under Python (`float32_unsupported`). A mixed-state document under Rust's in-crate solver. `reltol: 1e-9` on a Float32 document (`tolerance_below_working_precision`). A `tspan` of `[1e8, 1e8 + 4]` under the binary32 clock. `eval_into::<f32>` on a Float64 document (`RhsError::StatePrecision`).
6. **Metadata.** `integrator_precision` is present and correct on every Float32 solve.

Compliance matrix: rewrite PREC-11-A-009 to R6, and add rows for R2, R4, R5 and the metadata (R3).

## Work breakdown

1. **Spec.** esm-spec §11.3.2 (§1 above); §11.3.1 cross-reference; diagnostics table gains `tolerance_below_working_precision`. CONFORMANCE_SPEC §5.18.2 clause 3 replaced, new §5.18.5. Compliance matrix rows.
2. **Rust.** Remove `reject_f32_integration` (`problem.rs:2075-2108`, calls at `:2894`, `:3183`). Make the drivers generic over `T`. Add the clock and tolerance refusals. Add `IntegratorPrecision` to `SolutionMetadata`. Add the precision parts of U4's entry: arm per call, `state_precisions`, `round_state`, parameter ingress rounding. A short spike should first confirm that diffsol 0.11's BDF, SDIRK and ERK compile and converge over `f32` with faer (assumption: not yet built here).
3. **Julia.** The R6 refusal now. Then PREC-11-A typing and the lifted state guard.
4. **Python.** The R6 refusal.
5. **Fixtures** 1–6 above, plus `tests/COVERAGE_MATRIX.md`.
6. **Docs.** `MIGRATION.md` row ("Float32 documents now integrate in Rust; Julia and Python now refuse them").

## Alternatives considered

- **Keep refusing; the integrator evaluates Float32 itself.** That duplicates §5.18 in a second code base and outside conformance. Rejected.
- **diffsol over f64, rounding the accepted state after each step.** diffsol's internal stage vectors and history (BDF's difference array) stay f64 and are not reachable for rounding (assumption from diffsol's public API, not verified in its source). Resetting the state each step would also restart BDF's history. Rejected: it violates R2.
- **An opt-in widening flag** (`allow_f64_integration`). It gives the better-than-reference answer that §5.19.3a and §5.38.4 call the hardest wrong to notice. An author who wants binary64 declares `"Float64"`. Rejected.
- **`t: f32` at the RHS boundary.** That would force every integrator's clock to binary32 and break D13. At `t = 1e8 s` a binary32 clock has an 8 s ulp. The model already rounds `t` where it reads it. Rejected.
- **Pinning the stage-combination arithmetic** (binary32 per operation, or binary64 then round). It is a property of the method, not the document. R3 reports it instead. Deferred (open question 5).

## Open questions

1. **The trajectory band.** `rtol 1e-4` is a guess: binary32 epsilon (1.2e-7) times about 1e3 steps of growth. It should be measured on the §5.9 fixtures.
2. **The tolerance floor.** Is `2^-22` right, or should it scale with the method's order?
3. **Mixed-precision coupling.** WRF-Chem feeds real*4 meteorology into double-precision KPP. Inside one document, a Float64 species rate that reads a Float32 temperature is `mixed_element_type`. An equation cannot widen either, because its right-hand side must match its left (`precision_infer.rs:46-49`), and there is no authored conversion operator. A widening operator needs its own RFC. Where precision is unspecified rather than declared, U14 permits f32 meteorology leaves beside f64 chemistry leaves by converting at leaf boundaries. The question remains whenever the meteorology the chemistry reads is declared (U14 open question 1).
4. **Narrowing a Float64 document for a run (D28).** Resolved by D29 and U14. reseact, like every model the integrator targets, declares no `element_type`, so its precision is unspecified. The run config chooses f32 per leaf without stamping derived documents, and reports the choice. Narrowing a document that declares `"Float64"` stays illegal, and that is a change of model, not of method.
5. **Should stage arithmetic be pinned** for Julia–Rust bitwise agreement inside EarthSciIntegrator? That is the integrator spec's call, not EarthSciAST's.
6. **Clock resolution.** The refusal in §3 catches only coincident times. A stricter bound, for example a minimum number of binary32 ulps per output interval, may be wanted.
