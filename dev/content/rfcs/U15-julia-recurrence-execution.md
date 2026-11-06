---
title: "Executing causal recurrences in EarthSciAST.jl: one ordered sweep on every route, native and StableHLO"
description: "EarthSciAST.jl validates causal self-reference recurrences but refuses to run them. Specify how its native compiler executes them, matching Rust and Python bit for bit. A recurrence becomes a materialized buffer that one sweep function fills in recurrence-axis order. Self-reads are fail-closed and guards are lazy. Every Julia route calls the same sweep: in-place and out-of-place right-hand sides, the output-time observed program, build-time fills, and the interpreter. A spec clarification lets cells that share one axis position be evaluated together, since they are independent. The native kernels need this, and so does a level-major StableHLO lowering with a fault latch. No schema change."
---

Status: draft (proposed by EarthSciIntegrator, 2026-10-13)
Target: `EarthSciAST.jl`: `src/tree_walk/` (build, compile, observed program, interpreter route) and `ext/reactant_direct/` (the direct StableHLO emitter). A clarification of esm-spec §4.3.1.1 point 4 and CONFORMANCE_SPEC §5.19.2. Fixture manifests under `tests/conformance/recurrence/` and `tests/conformance/compiled_rhs/`. Pays the binding debt in `docs/content/rfcs/causal-self-reference-recurrence.md` §6.1. No schema change and no format version bump.
Needed by: EarthSciIntegrator milestone M7 (EqWeFiC column physics in Julia, the Julia planner's validation pass on EqWeFiC, and U9's Julia value goldens). Not needed by M5g or M6: reseact's `Mz` and eqice's column integrals are filtered prefix scans, not recurrences (Motivation). The StableHLO half is needed only when EqWeFiC runs on the Julia GPU path, which no v1 milestone gates.
Integrator refs: architecture.md §3.2 (passes 2, 4, 8), §7.2, §7.8, §10, §13 (U9, U13), §14 (M7); decisions D1, D4, D8, D10, D15

File citations are relative to the `code/` directory that holds all the repositories.

## Summary

A causal recurrence is an equation whose `faq` body reads the array it defines at an earlier position along one output axis (`EarthSciAST/esm-spec.md:879-1012`). Two bindings execute it today:

- **Rust:** `AlgebraicRule::Recurrence` and one `sweep_recurrence` function shared by both of its routes.
- **Python:** the bare-LHS form.

Julia implements the static half and then declines at build with `E_TREEWALK_UNSUPPORTED_RECURRENCE` (`EarthSciAST/pkg/EarthSciAST.jl/src/tree_walk/build.jl:1799`, `:1817-1841`). All nine recurrence conformance fixtures skip Julia.

This RFC specifies what Julia must do:

1. **Materialize every recurrence definition, on every compiler,** as a buffer block filled by a new fill-level member, `_RecurSweep`. This includes `:interpreter`, which inlines all other array observeds.
2. **Use one sweep function, `_recur_sweep!`, that owns the order:** recurrence axis outermost and ascending, each slice complete before the next. Each route supplies only a body evaluator: compiled kernels, the tree walker, or StableHLO emission.
3. **Make self-reads fail closed.** A dedicated read node checks `lo_d ≤ p_d < k_d` and raises `E_TREEWALK_RECUR_UNAVAILABLE` otherwise. It never returns the zero ghost.
4. **Keep guards lazy.** `ifelse` evaluates only the selected branch in every body evaluator.
5. **Batch within a slice.** Cells that share a recurrence coordinate are independent, so merged kernels may run them together. Classes still span slices, but their lanes are sorted slice-major and run one slice at a time. This needs a one-paragraph clarification of the spec's "no vectorized path" rule.
6. **Lower to StableHLO level-major, like the prefix scan.** One step per slice, vectorized over the slice's cells. In-body folds stay left folds. Unavailable reads are counted in a fault latch that the fragment returns.
7. **Report recurrences** in `BuildInspection` and in the compiler report, as Rust does.

## Motivation

**Julia refuses the construct, and the integrator needs Julia to run it.**

- The decline sits before observed resolution (`build.jl:1792-1799`). Its reason is that the array backend "class-merges per-cell kernels and so reorders cells" (`build.jl:1829-1836`; the comment at `:1803-1816` says the same).
- A test pins the refusal (`pkg/EarthSciAST.jl/test/recurrence_validation_test.jl:280-291`).
- The conformance manifest gives Julia a `skip_bindings` entry on every recurrence fixture, 01–09 (`EarthSciAST/tests/conformance/recurrence/manifest.json:35-36` and the eight entries after it).
- EqWeFiC recorded the refusal (`EqWeFiC/PLAN.md:2070-2073`) and runs its tests on the Rust CLI only (`EqWeFiC/CLAUDE.md:76`).
- EarthSciIntegrator needs Julia to run them for four reasons:
  - It has two native implementations (D1).
  - Its planner runs in both languages (D8). Pass 8 checks that the operator right-hand sides sum to the flat one at sampled states (`EarthSciIntegrator/docs/architecture.md:134-135`), which requires evaluating them.
  - U9's Julia reference evaluates band coefficients through the stock evaluator and lists this as a prerequisite (`EarthSciIntegrator/docs/rfcs/U09-faq-filters-semirings-recurrences.md:7`, `:72`).
  - The tangent and adjoint models U9 emits are themselves recurrence definitions (U9 Decision 6). The Julia adjoint (D4) must execute them.

**Where recurrences occur.** I scanned every `.esm` under `EqWeFiC/components/`, `eqice/` and `reseact.esm/` for equations whose RHS reads `index(V, …)` of their own LHS. There are 16, all in EqWeFiC, all in bare-LHS form:

| Component (under `EqWeFiC/components/`) | Definitions | Frame (recurrence axis first) | Lags | Body |
|---|---|---|---|---|
| Madronich two-stream, `gaschem/madronich/two_stream.esm` | `bet` :3546, `uf` :3734, `yrev` :3843 | `row`; `wlbin` | 1 | Thomas forward elimination and back substitution. `yrev` runs on the reversed row axis. Guards; `max`/`min` clamps. |
| RRTM LW, `atmospheric_radiation/rrtm_lw/rtrn_sweep.esm` | `rad_dn_top` :2167, `rad_dn_clr_top` :2586, `rad_up` :2745, `rad_up_clr` :3044 | `rlev`; `gpts` | 1 | Expression templates; `min`/`max` |
| Dudhia SW, `atmospheric_radiation/dudhia_sw/dudhia_sw.esm` | `S_dn` :1659 | `lev_down_nodes` | 1 and a contracted symbol `a` | Inner filtered `faq`; `max` |
| GWDO, `atmospheric_dynamics/gwdo/wave_saturation.esm` | `taup_wave` :671 | `lev_nodes` | 1 | Expression templates |
| New Tiedtke, `atmospheric_dynamics/ntiedtke/precipitation_flux.esm` | `tot_p` :1082, `rain_p` :1402, `snow_p` :1543, `tot` :1795, `rain` :1917, `snow` :2124 | `lev_nodes` | 1 | Expression templates; `max` |
| sfclayrev, `atmospheric_dynamics/sfclayrev/sfclayrev.esm` | `secant` :190 | `secant_steps` | 1–5 | `floor`; expression templates |

U9 gives the sizes: Madronich is 138 rows over 101 wavelength bins, and RRTM is 140 g-points over 104 levels. A single-column model has the frames shown. A 3-D assembly would add horizontal inner axes by lifting (esm-spec §10.5). No 3-D EqWeFiC assembly exists yet, so that part is an assumption.

**These recurrences are genuine.** EqWeFiC runs three tests before it accepts one (`EqWeFiC/CLAUDE.md:37`): unroll it, rearrange it, look for a conserved sum. Only sweeps that fail all three survive. Unrolling them is ruled out (U9 Alternatives).

**The scans are not recurrences, and Julia already runs them.**

- **reseact `Mz`.** A filtered `aggregate` with `filter: gk < gke`, an exclusive scan over levels. Its body reads `dB`, `dPSdt` and `divh_fix`, never `Mz` (`reseact.esm/reseact.esm:4459-4495`). It is reseact's only `filter` (`grep -c '"filter"'` gives 1).
- **eqice `I`, `D`, `w3`.** Filtered `faq` cumulative trapezoids (`eqice/stage2/sia.esm:274-300`, `:306-312`, `:891-901`); the stress-balance assembly has five more (`eqice/stage2/stress_balance.esm:669-796`).
  - `w3`'s description says "w[k] = w[k-1] - 0.5*dz*(div[k]+div[k-1])" (`sia.esm:885`). It is still written as a scan, because its terms do not read the answer.
- **How Julia runs them.** It folds these in O(N) as a post-pass over stateless kernels (`pkg/EarthSciAST.jl/src/tree_walk/scan.jl:1-40`). The emitter lowers them one level per step (`ext/reactant_direct/emit.jl:347-376`).
- **Consequence.** M5g (reseact) and M6 (eqice) do not depend on this RFC.

**What Rust does, which Julia must match bit for bit** (CONFORMANCE_SPEC §5.19.1, `EarthSciAST/CONFORMANCE_SPEC.md:3104-3121`):

- `AlgebraicRule::Recurrence` is a separate rule kind, "so every `match` over the rule kinds has to say what it does with a recurrence" (`pkg/earthsci-ast-rs/src/simulate_array/mod.rs:401-428`). It is built for both LHS forms and for eliminated states (`simulate_array/compile.rs:2410-2427`, `:2452-2461`).
- `RecurScope` publishes each cell rounded to the active precision. Its reads return `None` for out-of-frame or unwritten cells, which then fault (`mod.rs:430-509`).
- `sweep_recurrence` is "THE single implementation of the construct's evaluation order, shared by the per-step observed materialization and by the build pipeline" (`simulate_array/rhs.rs:282-344`).
  - It runs the recurrence axis outermost and the other axes inside it in `output_idx` order.
  - It notes that "cells sharing a recurrence coordinate cannot read one another ... their relative order is immaterial to the values" (`rhs.rs:318-322`).
- The tape refuses the rule and leaves a per-rule fallback to that sweep; the rest of the program stays taped (`simulate_array/tape/lower.rs:5179-5193`, `:4740-4748`).
- Build inspection reports `(var, axis, max_lag, lag_proven)` (`simulate_array/driver.rs:2818-2830`).

Python has `_RecurScope` and `sweep_recurrence` for the bare-LHS form (`pkg/earthsci-ast-py/src/earthsci_ast/numpy_interpreter.py:4687`, `:4840`; called at `simulation_array.py:1011`).

**Why Julia cannot copy Rust's shape directly.**

- Rust keeps an interpreter fallback inside a native build. Julia's strict `:native` plan refuses per-call tree walks, and refuses per-cell builds in the in-place form (`pkg/EarthSciAST.jl/src/compiler.jl:251-253`). So Julia's native route needs a compiled body.
- Julia's kernels are stateless per cell by design, and the runners (scalar walk, codegen loop nest, chunked, the `:oop` emitter) rely on that (`scan.jl:31-36`). The design below keeps them stateless and moves the order into the runner.

## Decisions of record

Proposed by the integrator, for the maintainers to rule on.

| # | Question | Proposal | Why |
|---|---|---|---|
| 1 | Where a recurrence lives | **A materialized buffer block, filled by a new fifth member of a fill level (`_RecurSweep`)**, on every compiler including `:interpreter` | Fill levels already exist in both build forms and in the observed program (`build.jl:3443-3478`; `observed_program.jl:216-240`). A self-referential body cannot be inlined, so `:interpreter`'s inline-everything rule (`build.jl:434`, `:441`) must exempt it. |
| 2 | One implementation | **`_recur_sweep!` alone owns the order.** Routes differ only in how they evaluate the cells of one slice. | CONFORMANCE_SPEC §5.19.3b (`:3182-3214`), which was written after Rust shipped two routes that disagreed. |
| 3 | Availability of a self-read | **Positional: `lo_d ≤ p_d < k_d`.** No `written` bitmap. | Under well-foundedness rule 2 (lag exactly 0 on every other axis) and the fixed order, this is exactly Rust's bitmap test. It costs nothing and lowers to StableHLO. |
| 4 | Unavailable read | **`E_TREEWALK_RECUR_UNAVAILABLE`.** Never the zero ghost and never a bare NaN. | esm-spec §4.3.1.1 point 6; CONFORMANCE_SPEC §5.19.4. A `max(x, 0)` clamp launders a NaN. |
| 5 | Guards | **Lazy `ifelse` in every body evaluator.** A batched evaluator either masks the guarded read or evaluates it with a clamped index and latches a fault per lane where the guard selects it. | `esm-spec.md:1004-1009`. The scalar walker is already lazy (`compile.jl:745-746`), and the kernels keep its short-circuit set (`codegen_kernel.jl:507`). |
| 6 | Batching | **Allowed within a slice** (all cells with one recurrence coordinate), with each cell's operation sequence unchanged. Slices run in ascending order. | Rule 2 means no cell reads a sibling in its own slice. Values are identical. Without batching, Julia's native kernels and any GPU lowering are excluded. Needs the §8 spec clarification. |
| 7 | Kernel classes | **Merge across slices, run slice-major.** Lanes are sorted by recurrence coordinate, and each class runs its lanes for slice `k` before any lane of slice `k+1`. | The class count stays what it is today. The kernels stay stateless. |
| 8 | Precision | **The buffer is typed with the RHS value type `T`.** Float32 documents are refused until U5 §4 step 2. | Each store is a publish, so a typed buffer rounds at every cell (§5.19.3a). Python's partial rounding (`numpy_interpreter.py:4782-4788`) is what CONFORMANCE_SPEC §5.18.2 forbids. |
| 9 | StableHLO | **Level-major unrolled sweep.** In-body folds are left folds regardless of `fold`. A fault latch is returned as a program output. | `_de_scan!` sets the precedent (program size O(axis length)). A `stablehlo.reduce` may reassociate (`ext/reactant_direct/api.jl:92-96`). U13 forbids host callbacks (U13 §2 rule 3). |
| 10 | Compiled-tier tolerance | **§5.38's classes, plus a new `exact` class** (rtol 0, atol 0) for order pins with no multiply and no in-body fold | The `reduction` class's scaled floor at magnitude 1e16 is about 100, which would pass a reordered ladder. XLA may contract multiplies into FMAs (assumption). |
| 11 | Targets | **Refuse a recurrence whose axis is a decomposed (horizontal) axis** under U3 tiles and U13 `Tile` targets, with `compiler_refused_rule` | The sweep would need positions outside the tile. No current model has such a recurrence: all 16 run vertically or along non-spatial axes. |

## Design

### 1. Recognition (replaces `_decline_recurrence_definitions`)

- Keep the candidacy predicate exactly as it is (`build.jl:1820-1828`, which mirrors `validate.jl:2005-2011`): an array-shaped variable with `recurrence_self_reference_kind(name, rhs) === :indexed`.
- For each candidate, run the existing well-foundedness walk (`src/recurrence.jl:121-253`, `:269`) to get the axis, `max_lag` and `lag_proven`. A candidate that is not well founded keeps its `recurrence_*` diagnosis.
- Accept both LHS forms, bare and indexed-`faq`, and eliminated-state definitions, as Rust does (`compile.rs:2410-2427`, `:2452-2461`).
- Force the variable into the materialized set, ahead of `_collect_materialized_array_obs` and its interpreter early return (`build.jl:439-441`).
- `_materialized_obs_levels` refuses any fill that reaches its own buffer (`build.jl:2726-2736`). Exempt a recurrence's self-edge from that check, gated on candidacy and never on the verdict (CONFORMANCE_SPEC §5.19.5). A path back to `V` through another observed is still a cycle (causal RFC §3).
- Chains between recurrences, such as `uf` reading `bet` and `yrev` reading `uf`, are ordinary edges, and the level order sequences them.

### 2. IR and the sweep

```julia
struct _RecurSweep
    var::String
    slots::Vector{Int}       # V's block in the extended vector, column-major over the frame
    lo::Vector{Int}; hi::Vector{Int}
    axis::Int                # recurrence axis, swept outermost ascending
    classes::Vector{Any}     # merged kernels (or tree bodies under :interpreter)
    lanes::Vector{Vector{UnitRange{Int}}}  # [class][slice]: that class's lanes in that slice
    max_lag::Int; lag_proven::Bool
end

function _recur_sweep!(ue, S::_RecurSweep, p, t, ::Type{T}) where {T}
    for (s, k) in enumerate(S.lo[S.axis]:S.hi[S.axis])
        for c in eachindex(S.classes)
            _run_slice!(S.classes[c], _slice_lanes(S, c, s), ue, p, t, T)  # route-specific
        end
    end
end
```

- `_fill_obs_levels!` (`build.jl:2779-2798`) gains a fifth member per level and calls `_recur_sweep!` after the level's kernel section and scan folds. This works because a recurrence reads only lower levels and itself.
- The out-of-place runner does the same over `mat_levels_oop`. After each slice it writes that slice's values into the value vector, so the next slice reads them.
- The output-time observed program builds its levels the same way (`observed_program.jl:216-240`).
- Build-time (setup) and refresh-time (discrete) fills call the same function.
- `:interpreter` walks each cell's tree in the slice's inner order.
- The sweep allocates nothing per call, because the buffer slots are part of the extended vector (`build.jl:2757-2772`). That keeps `f!` allocation-free in steady state.
- `T` comes from `_rhs_value_type(u, p, t)` (`build.jl:2762`), so ForwardDiff duals flow through unchanged.
- Enzyme reverse mode through in-place buffer writes inside a loop is expected to work. That is an assumption, and test 7 checks it.

### 3. Self-reads and faults

- Each `index(V, …)` inside `V`'s own body compiles to a dedicated read node, `_NK_RECUR_READ`. It never goes through the ordinary materialized-array gather, whose out-of-range value is the zero ghost (esm-spec §4.3.3).
- The node computes the read position `p` from its index arguments.
  - If `p` is in the frame and `p[axis] < k[axis]`, it reads `ue[slot(p)]`.
  - Otherwise it throws `TreeWalkError("E_TREEWALK_RECUR_UNAVAILABLE", …)` naming `V`, the cell, and `p`. The message follows Rust's (`simulate_array/eval.rs:1982`).
- Rule 2 already proves that the other coordinates equal the cell's own.
- For a literal lag, availability is a build-time fact per lane:
  - an available read compiles to a plain slot read;
  - an unavailable read compiles to a fault leaf, which runs only if the lazy guard selects its branch.
- For a lag that is runtime-valued (a parameter, as in fixture 08, or a runtime-indexed symbol), the read is checked at run time.
- A bare `V` in its own body is already `recurrence_not_wellfounded` (causal RFC §3 rule 4). If one reached the evaluator anyway, it would raise the same code, as Rust's `eval.rs:99` does.

### 4. Routes

Every route calls `_recur_sweep!`. The table says what supplies the per-slice evaluation.

| Route | Entry | Slice evaluator |
|---|---|---|
| In-place RHS | `f!` → `_fill_obs_levels!` (`build.jl:2757-2798`) | merged access kernels over the slice's lanes |
| Out-of-place RHS | `mat_levels_oop` runners (`build.jl:3469-3473`) | the same kernels through `_OopAccPlan`, written back per slice |
| Output-time observed | `_ObservedProgram` (`observed_program.jl:1-24`, `:216-240`) | the same kernels |
| Build-time and refresh fills | setup and discrete tiers | the same kernels, run once or per refresh |
| `compiler = :interpreter` | `evaluate_cellwise` path (`src/inline_tests.jl:536`) | the tree walk per cell, in the slice's inner order |
| `compiler = :xla` and U13 fragments | `ext/reactant_direct` | §6 |
| `compiler = :mtk` | — | refused, `compiler_refused_rule` (no ordered sweep in a ModelingToolkit system) |

The interpreter route is the oracle that every other route is checked against bit for bit, as it already is for the observed program (`observed_program.jl:23-24`).

### 5. Native kernels: slice-major classes

- Build `V`'s cells as the fill equation's access kernels, as for any materialized observed (`build.jl:3450-3460`), with three changes:
  - lanes are ordered by recurrence coordinate first;
  - `_merge_acc_kernel_classes` records each class's lane range per slice;
  - the self-read sites are §3's nodes.
- Each class runs once per slice over a contiguous lane range. The chunked runners iterate lane ranges already. I assume, without having verified it, that a sub-range entry exists or is a small change.
- Threads may split a slice's lanes. The slices themselves stay sequential.
- No reduction spans lanes, so D10's bitwise determinism holds for any thread count.
- **Codegen constraints.** No `@fastmath`, no `muladd` and no `@simd` reduction in a recurrence body. Julia does not contract `a*b + c` into an FMA without them, so each lane performs exactly the scalar walker's operations.

### 6. StableHLO lowering (U13)

- **Steps.** `_de_recur!` mirrors `_de_scan!` (`emit.jl:351-376`). For `k` from `lo` to `hi`, unrolled at trace time, it:
  - emits each class's kernel over that slice's lanes (`_de_acck`);
  - reads earlier slices from the `_DEMap`;
  - writes the slice back (`_de_write!`).

  Program size is O(axis length × classes) and independent of the number of inner cells.
- **Write plan.** `_de_plan_writes!` (`emit.jl:386-419`) treats a slot that an earlier step of the same sweep writes as written. A slot of the current or a later step is not written, and a read of it is a fault. The "zero this section is entitled to" rule (`values.jl:28-46`) never applies to a recurrence slot.
- **Guards.**
  - If a guard folds at trace time (`_de_try_fold`, typical for base cases like `k <= 1`), only the selected branch is emitted.
  - If a read is unavailable at a lane and no guard covers it, the fault is certain, and the emitter raises `E_TREEWALK_RECUR_UNAVAILABLE` at trace time.
  - Otherwise the read uses a clamped index, the result passes through a `select`, and the fragment adds a fault count `Σ (branch selected ∧ unavailable)` to an `i32` latch.
- **Fault reporting.** A fragment whose operator contains a recurrence returns `(du, fault)`. This amends U13 §2 rule 4. The caller must test `fault == 0` at every host boundary and must not report a solution otherwise. `compiler = :xla` tests it after each call and throws `E_TREEWALK_RECUR_UNAVAILABLE`. This is CONFORMANCE_SPEC §5.19.4's "latched diagnostic drained at every exit".
- **In-body folds.** A `faq` contraction inside a recurrence body is emitted as `_de_fold_terms`'s seeded ascending left fold (`ops.jl:133-147`), never as `_de_reduce_terms` (`ops.jl:110-123`). Today a fold of 32 or more terms switches to the reduce (`ops.jl:137-139`); inside a recurrence body it must not, whatever `reduce_min` or U13's `fold` setting says. The sweep feeds each cell's value into the next, so reassociation differences would grow along the axis.
- **Precision.** Float64 today; the emitter refuses other element types (`api.jl:217-231`). Float32 arrives with U13's `eltype`. The buffer is typed, so a slice write is the publish.
- **Targets.** `Lanes` (column shards) puts the inner axes, including the horizontal ones, in the lane dimension. `Tile` refuses a recurrence along a decomposed axis (Decision 11).

### 7. Targets on the native path (U3)

- A range-restricted RHS over a box sweeps only the box's cells.
- A recurrence along the vertical, or along a non-spatial axis, needs no halo.
- A recurrence along a horizontal axis is refused for tiles (Decision 11) and runs only in a `Whole` build.

### 8. Spec clarification (esm-spec §4.3.1.1 point 4; CONFORMANCE_SPEC §5.19.2)

Append to point 4 (`esm-spec.md:987-990`) and to §5.19.2 (`CONFORMANCE_SPEC.md:3123-3139`):

> **Slices.** The cells of `V` that share one coordinate on the recurrence axis form a *slice*. Rule 2 gives every self-read lag 0 on the other axes, so no cell reads another cell of its own slice. A binding MAY evaluate a slice's cells together (vectorized, kernel-merged, or as one tensor operation) provided that:
> (a) slices are completed in ascending order, each before any cell of the next is evaluated;
> (b) each cell's value is computed by the same sequence of operations the sequential sweep performs, with no reassociation, no fused multiply-add the sequential sweep does not perform, and §4.3.1's ascending in-body accumulation order;
> (c) a self-read of the cell's own slice, or of a later one, faults as in point 6.
> Everything else in point 4 stands.

The clarification is pinned by new fixture 10, below.

### 9. Observability

- `BuildInspection` (`build.jl:61`) gains `recurrences::Vector{RecurrenceInfo}` with `(var, axis, max_lag, lag_proven)`, as in Rust (`driver.rs:2818-2830`).
- `CompilerReport` gains the tier `:recurrence_sweep`, alongside `:scan` (`compiler.jl:230`).
- The report also tallies slices and classes per sweep. That makes a silent promotion onto a reordering path visible, as §5.19.2 asks.

### 10. What EarthSciIntegrator does with it (informative)

- **Locality.** The planner's pass 2 classes a recurrence along the vertical as `column` (architecture §3.2). A recurrence along a non-spatial axis, such as sfclayrev's iterations, leaves the operator's locality unchanged (U9 §3).
- **Kernels.** Column leaves use U3 lanes, one column per lane, and the sweep runs inside each lane program.
- **Adjoint.** The adjoint uses U9's VJP models, which are reverse sweeps written as ascending recurrences on a reversed axis. Julia executes them through this same path. Enzyme on the RHS remains Julia's fallback and oracle (D4).
- **Precision.** Under U14, a column leaf may evaluate an unspecified-precision recurrence in binary32. The buffer then has `T = Float32`, and every publish rounds.

## What this is not

- **Not a change to recurrence semantics.** Order, faults and bit-identity stay as esm-spec §4.3.1.1 and CONFORMANCE_SPEC §5.19 state them. §8 only states that values are unchanged when a slice is batched.
- **Not derivatives.** Tangent and adjoint rules are U9's. This RFC only makes Julia able to evaluate what U9 emits.
- **Not Rust performance.** Rust tapes everything except the recurrence rule, which falls back to its interpreter sweep. A slice-batched tape lowering for Rust is a separate follow-up (open question 3).
- **Not Float32 in Julia.** That is U5 §4 step 2. Until it lands, fixture 06 is a refusal in Julia, not a skip.
- **Not a GPU gate.** No v1 milestone runs a recurrence on the GPU.

## Conformance and tests

1. **Unskip Julia** on recurrence fixtures 01–05 and 07–09 (`tests/conformance/recurrence/manifest.json`), with the same pinned values at zero tolerance. Fixture 06 keeps a Julia entry, changed to an expected refusal (`float32_unsupported`) until U5 §4 step 2.
2. **Every route (§5.19.3b).** Each fixture is re-checked against the same pinned value on every route:
   - `observed_field` (the output-time program);
   - the inline-test runner;
   - `f!` in place, through a probe variant per fixture that adds `D(x[k]) ~ V[k]`;
   - the out-of-place build;
   - a state-free variant that lands in the setup tier;
   - `compiler = :interpreter`;
   - `compiler = :xla` (opt-in, with `ESM_TEST_REACTANT=1`).

   The check is agreement across routes, not just completion.
3. **New fixture `10_recurrence_slice_batch`** (all executing bindings, written first for Rust and Python).
   - Frame: 64 inner cells by 6 axis positions.
   - Body: `ifelse(k <= 1, c[j], a[j]*V[j,k-1] + b[j])`, with `a`, `b` and `c` chosen so that a fused multiply-add changes the last bit.
   - Zero tolerance. The description names the FMA value.
   - Julia's slice-batched kernels must equal Rust's sequential sweep bit for bit.
4. **New fixture `11_recurrence_fault_under_clamp`.** The body is `max(index(V, k-1), 0)` with no guard. Every executing binding and every Julia route must raise `E_TREEWALK_RECUR_UNAVAILABLE`. The `:xla` route raises it at trace time (§6). A variant whose guard depends on a parameter exercises the latch. A finite answer fails the test.
5. **Compiled-RHS tier** (`tests/conformance/compiled_rhs/manifest.json`).
   - Add a tolerance class `exact` (rtol 0, atol 0).
   - `recurrence_ladder_rhs`: fixture 02's ladder feeding `D(x[k])`, in class `exact`.
   - `recurrence_thomas_rhs`: a 6-level Thomas solve, forward and reversed sweeps, feeding `D(x)`, in class `algebraic`.

   Julia's compiled lane is `bindings_optional` until this lands, then joins `compiled_required`.
6. **Inspection parity.** Julia's `recurrences` entries equal Rust's on fixtures 01–09.
7. **Differentiation.** On `recurrence_thomas_rhs`:
   - ForwardDiff's Jacobian matches central differences to 1e-8 relative;
   - an Enzyme reverse VJP matches ForwardDiff to 1e-12.

   U9's value goldens and D4's Julia oracle both depend on these.
8. **Update the decline test.** `test/recurrence_validation_test.jl:280-291` asserts execution and inspection instead of the decline.
9. **EqWeFiC smoke (informative).** Julia's inline-test runner on `two_stream.esm`, `rtrn_sweep.esm`, `dudhia_sw.esm`, `wave_saturation.esm`, `precipitation_flux.esm` and `sfclayrev.esm` reports the same verdicts as the Rust CLI. Where EqWeFiC relies on constructs Julia refuses for other reasons, such as the non-const `interp.linear` axis noted at `EqWeFiC/INVENTORY.md:43`, the test reports that refusal by name.

## Work breakdown

0. **Spec.** The §8 clarification in esm-spec and CONFORMANCE_SPEC. Fixtures 10 and 11 in Rust and Python first, since they already execute recurrences.
1. **Julia recognition** (§1): materialize always; the level-cycle exemption; both LHS forms.
2. **The read node and fault** (§3).
3. **`_RecurSweep` and `_recur_sweep!`** (§2), wired into every route (§4). The interpreter route comes first, as the oracle.
4. **Slice-major classes and the sub-range runner** (§5).
5. **Observability** (§9).
6. **Emitter** (§6): `_de_recur!`, left folds in bodies, the fault latch, and the U13 fragment amendment.
7. **Manifests and tests** (above). Update the causal RFC's §6 table to "Julia: yes" and rewrite its §6.1 (`docs/content/rfcs/causal-self-reference-recurrence.md:274-339`).

Steps 0–5 are what M7 needs. Step 6 waits until EqWeFiC is run on the GPU.

## Alternatives considered

- **Port Rust's shape: an interpreter sweep inside a native build.** Strict `:native` refuses per-call tree walks (`compiler.jl:251-253`). A native build would then have to refuse every EqWeFiC radiation column. That shape is kept only for `:interpreter`.
- **Per-cell scalarized build of the body.** Strict in-place `:native` refuses it, and its build cost grows with cell count: about 14,000 cells per column for RRTM alone. Rejected.
- **A stateful kernel that carries an accumulator through the lane loop.** `scan.jl:31-36` explains why every runner relies on stateless kernels. A recurrence with up to five lags (sfclayrev) or a contracted lag (Dudhia) has no single accumulator anyway (causal RFC §5). Rejected.
- **`stablehlo.while` over the axis.** The program would be O(1) in axis length. But the body would need index arithmetic traced from the loop counter, and the emitter bakes indices as constants today (U13 rule 7 introduces traced `i32` only for `origin` and `cells`). Deferred (open question 1).
- **A host callback per slice on the XLA path.** It breaks U13's contract (rule 3) and pays a dispatch per slice. Rejected.
- **Keep the spec's "no vectorized path" literally.** Julia's native kernels and every GPU lowering would then be non-conformant for a construct whose values batching does not change. Rejected in favour of §8.
- **Solve the sweep as a triangular system.** Rejected by causal RFC §5: forward substitution is exact and a solve is not.

## Open questions

1. **Unrolled versus `while` in StableHLO.** Madronich has 138 rows and RRTM 104 levels. What compile time does an unrolled sweep cost at tile size against U13's proposed 120 s budget?
2. **FMA on XLA.** Is there an XLA flag that stops multiply-add contraction on CPU and GPU? If so, the emitter could join bit-identity for all recurrence fixtures, not only `exact`-class ones.
3. **Rust.** Should Rust lower recurrences to a slice-batched tape as well? EqWeFiC at 3-D in Rust would otherwise run every radiation sweep on the interpreter.
4. **Horizontal recurrences.** Decision 11 refuses them under tiles. Should a later version gather the whole line to one rank instead? No current model needs it.
5. **Thread granularity.** In a single column an RRTM slice has 140 cells and a Madronich slice 101. Threading inside a slice may cost more than it saves at that size. Should the runner thread across columns only, and run each slice of one column serially?
