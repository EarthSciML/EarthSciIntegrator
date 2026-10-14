---
title: "Traceable per-operator StableHLO: RHS, tile, lane and band-model fragments for an integrator's own XLA program"
description: "Extend EarthSciAST.jl's direct StableHLO emitter so an external integrator can get, for one operator and one tile or lane capacity, a pure traceable fragment that emits StableHLO into the integrator's own Reactant trace — inside its @trace while loop — with every input explicit: state, read-only state, a dense parameter vector, time, forcing buffers and, for lanes, the cell indices that define shard composition. Forcing refreshes and re-sharding never recompile. Adds Float32, deterministic folds, observed-output (band-model) fragments, and trace hygiene the integrator can rely on."
---

Status: draft (proposed by EarthSciIntegrator, 2026-10-13)
Target: `EarthSciAST.jl` `EarthSciASTReactantExt` (`ext/reactant_direct/`), plus a public IR entry in the package. Answers API_SPEC §8 item 23's open items (1), (2) and (4). Julia only: Rust's `tape/xla_emit.rs` is out of scope until after v1 (D15). No schema change.
Needed by: EarthSciIntegrator milestone M5g (Julia XLA GPU path).
Integrator refs: architecture.md §7.2, §7.6, §7.8, §8, §10, §13 (U13), §14 (M5g), §15 risk 7; decisions D4, D10, D15, D18, D28, decisions.md §4

## Summary

On the GPU path, one leaf step is one XLA program (D15). That covers the
stages, the linear solves, the error norm and the controller, all traced by
Reactant from the integrator's own kernel code. The right-hand side and the band
models inside that program must come from EarthSciAST's direct emitter
(architecture §7.8).

The emitter already builds StableHLO at trace time, and reseact already calls it
inside a `@trace while`. But its public surface was designed for a whole model
compiled on its own: a whole-domain flat state, parameters as a NamedTuple of
scalars, Float64 only, index data baked in as constants, and a 1-D cell-axis
shard that does not match a horizontal decomposition.

This RFC asks for an **emission fragment**: a host-only callable, built once
per (operator, target), that the integrator calls any number of times inside its
own trace, including inside `@trace while` and `@trace for`. It emits ops into
the current block, takes every traced value as an argument, and returns fresh
tensors.

- There are three targets, matching U3: `Whole`, `Tile` and `Lanes`. Outputs
  are `du`, or observed outputs: U6's band models, residuals and maps.
- Forcing buffers and the lane index table are program inputs, so a forcing
  refresh or a re-sharding is an input swap, never a recompile.
- Program size depends on the operator and the tile or lane capacity, never on
  the domain.

## Motivation

**The surface is not composable today.**

- API_SPEC §8 item 23 lists what is still missing (`EarthSciAST/API_SPEC.md:1281`):
  (1) "a public way to obtain the out-of-place compiled intermediate
  representation itself, since `compiler = :xla` hands back a Problem whose `f!`
  is already compiled and host-in/host-out and cannot be composed into a larger
  traced program"; (2) live forcing under `:xla`, which is refused
  (`src/compiler_xla.jl:166-185`); and (4) "a caller that builds SEVERAL
  right-hand sides from one document and drives them in one time loop".
- The only route today is the private `_build_evaluator(doc; form = :oop)`,
  which the device guide documents for exactly this purpose
  (`docs/src/compiled-backend-devices.md:14-20`).

**The emitter already works as a fragment.**

- `DirectRHS`'s call takes `u::TracedRArray{Float64,1}` and emits ops directly.
  It never calls `@compile` itself (`ext/reactant_direct/api.jl:144-164`, `:191`).
- reseact calls the four-argument form inside its traced SSPRK43 and ROS23
  loops (`reseact.esm/run_reseact_reactant.jl:256`, `:325-332`).
- Forcing buffers are arguments, so "`copyto!`-ing new values into that same
  `ConcreteRArray` between calls IS seen by the already-compiled program"
  (`ext/EarthSciASTReactantExt.jl:22-33`).
- Agreement with the reference is good: within 1e-13 on both of reseact's
  halves (`reseact.esm/AGREEMENT.md:8-16`).

**Rules learned in reseact, which the emitter should own.**

1. **Carry everything traced explicitly.**
   - "Any traced value captured inside a closure the body calls becomes an extra
     region block arg with NO matching operand and the stablehlo.while verifier
     rejects the module" (`tools/reactant_handoff/rx_traced_integrator.jl:453-460`).
   - Hence `aux = aux` (`:493`), `f = f` and `track_numbers=false` (`:28-36`),
     and "`bodyf` must close over HOST values only" (`:806-811`).
   - A host parameter is "a compile-time constant" (`ext/reactant_direct/emit.jl:25-29`).
     A captured forcing buffer is "silently stale forcing"
     (`EarthSciASTReactantExt.jl:22-33`).
2. **The IR must be an opaque leaf.**
   - Reactant's `make_tracer` walks the whole object graph of every callable.
     "On the merged transport RHS (1.06M nodes) it costs tens of minutes."
   - reseact registers `_Node`, `_AccKernel`, `_OopAccPlan` and `_AccScratch` as
     leaves itself (`tools/rx_rhs.jl:167-189`).
3. **Read forms and index data decide compile time.**
   - Slices-plus-concatenate put reseact's transport step at 4.1M ops, a 399 s
     pass pipeline, and a reverse program that did not finish in five hours.
   - The gather default took `@compile` from 894 s to 207 s and the per-call
     time from 3.3 ms to 1.9 ms (`reseact.esm/COMPILE_COST.md:9-27`).
   - Gather positions are host constants interned by content
     (`ext/reactant_direct/values.jl:355-401`). Their size scales with the
     domain, and the emitter's own notes say the slice population "grows without
     bound" with the grid (`values.jl:461-466`).
   - What remains is "the per-level emission": the horizontal stencil is "a
     kernel per level" (`COMPILE_COST.md:1531-1540`).
4. **Buffer donation.** "`RX.@compile` donates the input buffers a program does
   not return, so reusing one across two calls silently corrupts the second"
   (`rx_traced_integrator.jl:875-880`). `donated_args=:none` exists
   (`tools/diag/jobs_src/j06_xlaflags.jl:49`).
5. **Determinism.**
   - XLA:CPU computes reseact's chemistry wrongly on about 1 call in 2,000 with
     more than one intra-op thread (`tools/diag/README-nondet.md:6-9`). The
     workaround is `xla_cpu_prefer_vector_width=128` (`run_reseact_reactant.jl:391-400`).
   - With it, the traced pipeline is bit-reproducible run to run
     (`DIFFERENTIABILITY_PLAN.md:756-763`).
   - Folds of 32 or more terms become one `stablehlo.reduce`, "which XLA may
     reassociate" (`ext/reactant_direct/ops.jl:110-119`; `api.jl:92-96`).
6. **Compile-time budgets.**
   - The whole-domain transport half takes 343.5 s to compile at 6×6×8, the
     chemistry half 191.5 s (`AGREEMENT.md:134-152`).
   - `@compile ssp_vjp` takes 195.5 s at CONUS (`COMPILE_COST.md:1516-1520`).
   - reseact's own fix is "compiled at *tile* size", turning an 8.3 h projected
     compile into a fixed ~15 min (`SCALE_025.md:163-174`).

**What the integrator needs that the emitter refuses or lacks.**

- **A parameter vector.** "A vector `p` ... is not wired here"
  (`emit.jl:12-16`). reseact passes 162 runtime parameters as 162 scalar inputs
  (`AGREEMENT.md:137`), each carried through every loop.
- **Float32.** It is refused by name (`api.jl:218-231`). D28's mixed
  configuration needs an f32 transport state.
- **Run-time indices.** None exist: every gather index is a constant. Lane
  composition must be a run-time input (architecture §7.8).
- **A horizontal shard.**
  - `sharding = :cells` cuts the flat state into contiguous slabs
    (`compiled-backend-devices.md:130-147`).
  - Blocks are column-major with the first index fastest
    (`src/tree_walk/state_layout.jl:1-8`). For reseact's `[lon, lat, lev]`
    fields, a contiguous slab is a band of levels. That is a vertical
    decomposition, which D20 rules out.
  - Reads are constant-index gathers over a rank-1 value, so XLA's SPMD
    partitioner cannot see a halo in them. The guide says as much for forcing:
    "a sharded copy would turn every read into a collective"
    (`compiled-backend-devices.md:168-170`).
- **Concurrent tracing.** Emission state lives in module-level refs, on the
  assumption that "one emission runs to completion inside one
  `Reactant.@compile` trace on one task" (`values.jl:72-79`). The integrator
  wants to trace several leaf programs concurrently in one process. reseact
  ran four compiles at a time, but in separate worker processes
  (`tools/shard_kernel.jl:76-79`).

## Decisions of record

Proposed by the integrator, for the maintainers to rule on.

| # | Question | Proposal | Why |
|---|---|---|---|
| 1 | Unit of emission | **A fragment** (a host-only callable that emits into the caller's trace), not a compiled program | The integrator compiles one program per leaf step (D15). The fragment must sit inside its loop. |
| 2 | Inputs | **All explicit**: `u`, `aux`, `p::tensor<P>`, `t`, a forcing tuple, and `cells` for lanes | Rule 1 above. A fragment adds no carry obligations of its own. |
| 3 | Targets | **`Whole`, `Tile(class)`, `Lanes(capacity)`**, with U3's semantics | Program size is bounded by the target, not by the domain. |
| 4 | Re-sharding | **`cells` is a traced `i32` input** | No recompile at macro-step re-sharding (architecture §7.6). |
| 5 | Tile programs | **One per tile *class***, with the tile's origin a traced input | Equal tiles share one program (`SCALE_025.md:163-174`). At most 3^D classes when tiles are wider than the boundary band. |
| 6 | Multi-GPU | **The emitter does not shard.** The integrator composes per-device tile fragments and emits the halo exchange. | Flat-axis shards are vertical slabs, and gathers defeat SPMD halo detection (above). |
| 7 | Precision | **`eltype ∈ {Float64, Float32}`** for state and evaluation. `t` stays f64. Observed outputs may be cast to f16 or bf16. | D18, D28. |
| 8 | Determinism | **`fold = :tree`** emits long folds as fixed pairwise trees. No scatter in forward fragments. | D10 across tile decompositions. One-device run-to-run reproducibility is necessary but not enough. |
| 9 | Numerical contract | **Tolerance against native, bitwise run to run on one device** | Already the emitter's stance (`api.jl:92-96`) and architecture §7.8's. |

## Design

### 1. Public IR

```julia
ir = operator_ir(prob; select = RhsSelection(writes = W, terms = tags),   # U3 §2
                       target = Whole() | Tile(dec, cls) | Lanes(kind, capacity),
                       outputs = :du | KernelSpec(...))                    # U6 observed outputs
```

- `operator_ir` replaces the private `_build_evaluator(form = :oop)` as the
  documented way in.
- It returns the compiled out-of-place IR for one operator and one target, plus
  its layouts:
  - `ir_layout(ir)`: U3's `u`/`aux`/`du` blocks;
  - `parameter_names(ir)`;
  - `forcing_slots(ir)`;
  - for tiles, `halo(ir)` and `forcing_reach(ir)`.
- `outputs = KernelSpec(...)` builds an observed-output kernel. That covers the
  planner's band models (J, VJP, HVP, ∂f/∂p, ∂f/∂forcing: architecture §3.2,
  pass 6; U7, U8) and U6's residuals, maps and conditions. All are derived
  documents with observed outputs, so the emitter needs no derivative logic.

### 2. The fragment

```julia
ext  = Base.get_extension(EarthSciAST, :EarthSciASTReactantExt)
frag = ext.emit_fragment(ir; eltype = Float64, fold = :tree)   # host-only, immutable
```

**Calling forms:**

```julia
du  = frag(u, aux, p, t, forcing)                    # Whole
du  = frag(u, aux, p, t, forcing, origin)            # Tile
du  = frag(u, aux, p, t, forcing, cells)             # Lanes
out = frag(...)                                      # KernelSpec: tuple of observed tensors
```

**Arguments:** `u` and `aux` are `TracedRArray{T,1}` in U3's layouts (a tile
uses its storage box; lanes are species-major, lane index fastest). `p` is a
`TracedRArray{Float64,1}` in `parameter_names(ir)` order, read through the
existing indexed seam (`src/tree_walk/compile.jl:1402-1447`). `t` is a
`TracedRNumber{Float64}`. `forcing` is a tuple aligned with `forcing_slots(ir)`:
whole fields for `Whole` and `Lanes`, the tile's windows for `Tile`. `origin` is
a `TracedRArray{Int32,1}`, the tile's global low corner. `cells` is a
`TracedRArray{Int32,2}` of size `D × capacity`.

**Contract** (normative):

1. `frag` holds only host data. Its IR types are registered as `make_tracer`
   leaves by the extension itself (rule 2). Capturing `frag` in a loop body
   costs nothing and needs no `frag = frag`.
2. It reads no traced value it was not passed. It may be called any number of
   times in one trace, inside `@trace while`, `@trace for` or `@trace if`
   bodies, with default `track_numbers`.
3. It emits only into the current block. It never calls `@compile`, never adds
   a sharding constraint, and never creates host callbacks.
4. It returns fresh tensors and never aliases an input. So it imposes no
   donation constraint. The integrator compiles with `donated_args=:none` for
   `p` and forcing, and may donate state deliberately.
5. Emission state lives in a per-call context. Two traces may run concurrently
   on different tasks. The module-level `_DE_RULE`/`_DE_SECTIONS`
   (`values.jl:72-79`, `:185`) become context fields.
6. **Shapes.**
   - `Whole`: `du` has `u`'s shape.
   - `Tile`: `du` has the storage shape, with structural zeros outside the
     owned rows. Stage algebra is then plain elementwise over storage tensors,
     and the halo exchange overwrites halo rows.
   - `Lanes`: `du` is `[n_W · capacity]`.
7. **Index data is O(target), not O(domain).**
   - A tile's reads are constant-offset reads into storage.
   - Index-derived values are computed from `origin` (tiles) or `cells` (lanes)
     by traced `i32` arithmetic.
   - Lane reads of forcing and other index-dependent inputs are
     `stablehlo.gather`s whose start indices are computed from `cells`.

### 3. Tile classes

A tile program depends on three things: the tile shape, the storage shape, and
which `makearray` regions intersect the need boxes at which positions relative
to the origin (U3 §4, rule 1).

`tile_classes(op, dec, tiles)` groups a decomposition by that key and returns
one `Tile(dec, cls)` per class.

- For equal tiles at least as wide as the boundary band, there are at most 3^D
  classes: 9 in 2-D (corners, edges, interior).
- For a two-region PPM rule set, every interior tile shares one program.
- Compile cost becomes O(classes × operator), independent of the domain and of
  the tile count.

### 4. Forcing epochs

- Forcing tensors are inputs. A refresh at an epoch boundary is the host
  `copyto!` into the same device arrays (`sync_forcing!`,
  `src/data_refresh.jl:447-455`), between leaf-step programs. Epoch boundaries
  are tstops (architecture §7.5), so no program ever spans one.
- Within an epoch, the model's own time blend uses traced `t`, so nothing
  recompiles as `t` advances.
- Tiles receive windows, not replicated fields: `forcing_reach` bounds them.
  Each device holds only its tiles' windows.
- Lanes read forcing either:
  - by gather from the rank's window, or
  - from per-shard buffers pre-gathered once per epoch (U3 §6). This is the
    better choice on GPU.

### 5. Shard composition and lane pools

- The integrator's leaf program gathers the shard's lanes from its
  species-major state with traced `cells`, integrates, and scatters back.
- `ext.gather_lanes(ir, u_rank, cells)` and `ext.scatter_lanes(ir, u_rank, du_L, cells)`
  are provided as fragments because the emitter owns the layout. Their indices
  are unique, so the scatter is deterministic.
- Fixed capacity with masked chunk refill is the integrator's.

The emitter guarantees two things:

- **Lane independence.** A lane's outputs depend only on that lane's inputs
  (U3 §6).
- **Padding.** A lane duplicating a real cell computes finite values. reseact's
  masked loop wasted 20–28% on dead iterations
  (`DIFFERENTIABILITY_PLAN.md:1020-1027`); that is the integrator's to manage.

### 6. Multi-GPU in one node

The emitter's obligations are only §2–§5. The integrator composes the fragments
in one of two ways:

- **A. Manual SPMD (preferred).**
  - Each device runs the `Tile` fragment of its own tile.
  - Between stages, the integrator emits `collective_permute` for halo rows,
    and an all-reduce (fixed tree, D10) for norms and Krylov dot products.
  - *Assumption, unverified here:* Reactant exposes a manual-partitioning
    region (a `shard_map` analogue) that admits `collective_permute`. An M5g
    spike must confirm this.
- **B. One program per device**, launched from host threads.
  - Halos move device to device through PJRT between stage programs.
  - This breaks one-program-per-leaf-step for stencil leaves only.
  - It is the fallback if A is unavailable. reseact warned that host round
    trips "would DOMINATE" (decisions.md §4). Device-to-device copies avoid the
    host but still cost one dispatch per stage.

The existing `sharding = :cells` stays for whole-model users. The integrator
does not use it.

### 7. Precision and determinism

- **Precision.**
  - `eltype = Float32` emits `tensor<…xf32>` throughout.
  - Literals are rounded once at emission, per the document's
    `element_type` rules (U5).
  - `t`, `p` and forcing reads are cast on read.
  - `KernelSpec(out_eltype = BFloat16)` casts observed outputs at the end, for
    16-bit W storage (§8 of the architecture).
- **Determinism.**
  - `fold = :tree` emits every fold of n terms as a fixed balanced tree of
    binary ops.
  - `fold = :reduce` keeps today's behaviour.
  - Forward fragments never emit `scatter` except §5's unique-index lane
    scatter. The emitter already emits none (no `_hlo.scatter` in
    `ext/reactant_direct/`).
- **Compile options** are the integrator's:
  - XLA's deterministic-ops mode on GPU *(flag name to be pinned in the spike)*;
  - `xla_cpu_prefer_vector_width=128` for any XLA:CPU test run.

## What this is not

- **Not an integrator.** Stages, controllers and solves stay in
  EarthSciIntegrator's kernel layer (D15).
- **Not reverse mode.** Adjoints are method-level, from band-model fragments
  (D4). Nothing here needs Enzyme through a `while`.
- **Not Rust.** `XlaBuilder` emission is after v1 (D15).
- **Not native CPU tiles or lanes.** Those are U3. This RFC lowers the same
  semantics to StableHLO.
- **Not multi-node.** MPI between programs, or PJRT's distributed runtime, is
  the integrator's (architecture §7.8).

## Conformance and tests

All Reactant tests are opt-in (`ESM_TEST_REACTANT=1`), like the existing
compiled tier.

1. **In-loop trace.** An RK4 leaf with four fragment call sites inside a
   data-dependent `@trace while`, with default `track_numbers` and no
   `frag = frag`, compiles. It agrees with four host-driven calls within the
   compiled-RHS tolerance class, and two fresh processes agree bitwise.
2. **No recompile.** Change the forcing contents, `p`, `t` and `cells` between
   calls: exactly one compile (counted), and the outputs change accordingly.
3. **Tiles.** For U3's decompositions, owned rows assembled from per-class
   programs match the `Whole` fragment within tolerance, and the class count is
   as expected.
4. **Lanes.** Permuted, padded and duplicated `cells`. Each lane equals the
   `Whole` value at its cell, within tolerance. Swapping inputs must mismatch.
5. **Size census.** Emitted op count is equal for two tile sizes and for two
   lane capacities. Index constants scale with the target. Emission is
   level-batched: the op count is independent of `nlev`.
6. **Float32.** An f32 fragment agrees with f64 to the f32 class.
7. **Concurrent tracing.** Two tasks trace two fragments at once and produce
   the same modules as serial tracing.
8. **Leaves.** Trace time of a program capturing a 1M-node fragment is within
   a constant of one capturing a 1k-node fragment.
9. **Multi-device (opt-in twice, as today).** Four mock host devices
   (`--xla_force_host_platform_device_count=4`) run approach A or B against
   one device, within tolerance.

## Work breakdown

1. `operator_ir` with U3 selection and targets; `ir_layout`, `parameter_names`,
   `forcing_slots`.
2. Emitter:
   - a vector `p` read;
   - an eltype parameter;
   - `Tile` (storage-local reads, region clipping, zero-outside-owned `du`,
     origin-derived index values);
   - `Lanes` (`cells`-derived gathers);
   - `KernelSpec` outputs.
3. `fold = :tree`.
4. Trace hygiene: `make_tracer` leaves in the extension; per-context emission
   state.
5. Level-batched kernels in `src/tree_walk/` (the per-level emission,
   `COMPILE_COST.md:1531-1540`).
6. `tile_classes`, `gather_lanes`, `scatter_lanes`.
7. `docs/src/compiled-backend-devices.md`: a section on composing into an
   integrator's loop.
8. Tests above.

## Alternatives considered

- **`esm_problem(…; compiler = :xla)`.** Its `f!` is host-in/host-out and
  cannot be composed into a larger traced program (API_SPEC §8 item 23).
- **Tracing generic Julia `f!` through Reactant.** That is the traced lane, and
  it has been retired upstream: `rhs_with_buffers` is gone, and the `:oop`
  build is not callable on the host (`reseact.esm/tools/rx_rhs.jl:14-20`).
- **A whole-domain fragment plus `sharding = :cells`.** It gives vertical slabs,
  and all-gathers where halos were meant.
- **An N-D value model** (one tensor per variable, stencil reads as static
  slices). XLA's SPMD partitioner could then turn horizontal shards into
  `collective_permute` halos itself, and index data would disappear. But it
  rewrites the emitter's rank-1 value model (`values.jl:8-16`), and slices are
  where compile time went quadratic (`COMPILE_COST.md:9-27`). Revisit after
  M5g.
- **Compiling each fragment separately and calling it from the host per
  stage.** That pays the per-call floor every stage: about 3 ms on XLA:CPU
  (`reseact.esm/SCALE_025.md:8`).

## Open questions

1. Does Reactant support a manual-SPMD region with `collective_permute` (§6 A)?
   This decides one program per leaf step for multi-GPU stencil leaves.
2. Should the fragment emit the operator once as a private `func.func` and
   `call` it per stage? Trace time is linear in call sites
   (`rx_traced_integrator.jl:25-28`), and interning across emissions was
   rejected as a use-after-free hazard (`COMPILE_COST.md:977-984`).
3. What compile-time target should gate M5g? A proposal is ≤ 120 s per
   tile-class program on the host CPU, against 207–344 s for today's
   whole-domain halves at 6×6×8.
4. Should pre-gathered lane inputs (§4) be emitted by EarthSciAST, or written by
   the integrator from `lane_inputs`?
