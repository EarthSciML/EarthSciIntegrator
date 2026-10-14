---
title: "Forcing-buffer and array-parameter columns: ∂f/∂(forcing) bands"
description: "Add a `forcing` differentiation axis to EarthSciASTDiff whose columns are the cells of live forcing buffers, including their record axis, and make array-valued parameter columns real columns on the `parameters` axis. Because EarthSciIO hands the model a two-record bracket as an ordinary leading length-2 axis and the model writes its own time blend, the chain rule into both bracketing records falls out of the existing band calculus. The integrator turns the transposed bands into vjp_forcing and accumulates per provider record."
---

Status: draft (proposed by EarthSciIntegrator, 2026-10-13)
Target: EarthSciASTDiff (`esm-jacobian-spec.md` §1, §4, §5, §7; draft spec, no version number yet). Julia reference, then Python and Rust.
Needed by: EarthSciIntegrator milestone M4 (forcing and gridded-field gradients)
Integrator refs: architecture.md §3.2 (passes 4, 6), §7.5, §10 (primitives, accumulation), §13 (U7); decisions D4, D5, D8, D10, D18

File citations are relative to the `code/` directory that holds all the repositories.

## Summary

Decision D5 requires gradients with respect to forcing time series and gridded fields. EarthSciASTDiff produces `∂f/∂u`, `∂f/∂p` for scalar `p`, and `∂f/∂t`. It does not produce `∂f/∂(forcing)`; its README leaves "Forcing-buffer columns" unchecked (`EarthSciASTDiff/README.md:144-145`). Its draft spec lists "derivatives with respect to forcing buffers / const arrays" as out of scope (`EarthSciASTDiff/esm-jacobian-spec.md:285-288`).

This RFC proposes:

1. A fourth axis, `wrt: "forcing"`. Its columns are the cells of the forcing buffers the runtime binds, identified by `(buffer name, multi-index)` with the record axis included.
2. Array-valued parameters become real columns on the existing `wrt: "parameters"` axis. Today they are silently dropped at assembly (Motivation 2).
3. No special treatment of the time blend. The record bracket is a leading length-2 axis of an ordinary array parameter, and the blend `(1−w)·F[1,…] + w·F[2,…]` is model AST. So the band coefficients are `(1−w)·∂f/∂F̃` for record 1 and `w·∂f/∂F̃` for record 2. They come out of the existing `*` and `+` rules with no new derivative rule.
4. A canonical column index space for forcing, and a spec-level statement of how the integrator turns transposed bands into `vjp_forcing` and accumulates them per provider record. The accumulation itself stays in the integrator.

## Motivation

**1. How forcing reaches the right-hand side today (verified).**

- **EarthSciIO.** With `records_per_sample = 2`, a provider returns "the TWO bracketing records (floor + successor) with `time_dim` RETAINED at length 2". At the last tick the bracket degenerates to `[last, last]`, and "the provider does pure I/O" (`EarthSciIO/julia/src/provider.jl:55-63`).
- **The document.** reseact declares the buffer as one array parameter with a leading record axis: `F_PS` with shape `["gf_t", "gf_lat", "gf_lon"]` (`reseact.esm/reseact.esm:304-310`). The surface-pressure equation (`reseact.esm/reseact.esm:4043`) reads `100*((1-w_I3)*index(F_PS,1,LAT0+gj,LON0+gi) + w_I3*index(F_PS,2,…))`. The weight is `w = (t − t_ref)/dt − floor((t − t_ref)/dt)` (`EarthSciModels/components/earthsci_data/geosfp.esm:1269`).
- **Both records without the blend.** The same buffer feeds `dPSdt = (PS_2 − PS_1)/dt_I3` (`reseact.esm/reseact.esm:247-254`).
- **Blend after regrid.** `wildlandfire.esm` regrids each record separately and blends afterwards (`wildlandfire.esm/wildlandfire.esm:1263-1272`, `:1651`, `:1741`, and near `:1960`).
- **Spec.** It has no record construct. A `data` update seeds the DISCRETE cadence class and "MUST declare its `shape`: it is a buffer whose extent is fixed at setup and refilled on a discrete cadence" (`EarthSciAST/esm-spec.md:2049-2054`).
- **Runtimes.**
  - Julia binds forcing as name-sorted flat buffers (`EarthSciAST/pkg/EarthSciAST.jl/src/tree_walk/oop.jl:738-750`; `src/simulate.jl:712-722`).
  - Rust today keys an `ArrayD` per name (`EarthSciAST/pkg/earthsci-ast-rs/src/simulate_array/mod.rs:696`). U4 proposes an immutable, column-major `ForcingSet` ordered by `forcing_slots()`.

So the derivative needs no knowledge of brackets. Sites `index(F_PS, 1, …)` and `index(F_PS, 2, …)` are distinct sites (spec §3, `esm-jacobian-spec.md:57-63`), and the calculus already differentiates through `*`, `+` and `floor` (whose derivative is 0, spec §3.2). Any pattern-matching of the blend would break on `dPSdt` and on wildlandfire's blend-after-regrid.

**2. Array-parameter columns are silently dropped (by reading; not executed).**

- `jacobian_bands(wrt = :parameters)` targets every parameter, array-valued ones included (`EarthSciASTDiff/pkg/EarthSciASTDiff.jl/src/jacobian.jl:41-47`).
- Assembly maps parameter columns through the scalar `p` NamedTuple (`assemble.jl:53-64`). That NamedTuple holds scalars only: "Array-shaped parameters must be array-backed … the scalar `p` NamedTuple stays homogeneous Float64" (`EarthSciAST/pkg/EarthSciAST.jl/src/tree_walk/build.jl:786-792`).
- An array-parameter column therefore resolves to slot 0 and is skipped as a "ghost / out-of-layout column" (`assemble.jl:159-160`, `:265-266`).

Architecture §10 sources gridded-field gradients from "`vjp_p` into array parameters". With today's assembly, that gradient is zero with no error, which is the same class of failure as the empty-plan incident (`reseact.esm/tools/reactant_handoff/rx_sym_block_jac.jl:277-286`).

**3. reseact has no forcing gradient, and the Enzyme route was expensive.** `tools/adjoint_gradient.jl:80-89` reads: "FORCING HAS NO DERIVATIVE … only `p` is ACTIVE in the VJP (`RESEACT_ADJ_BUFGRAD=0`, the default)". The old Enzyme path "was materialising and discarding ~1.5 GB of forcing-buffer gradient per VJP at CONUS". Bands give the same gradient as a structured scatter-add into one accumulator per live buffer.

**4. Transport makes this depend on U9.**

- reseact's vertical mass flux `Mz` is a filtered prefix scan over levels (`filter: gk < gke`; `reseact.esm/reseact.esm:4460-4495`). It reads only forcing-derived fields (`dPSdt`, `divh_fix`).
- Under `wrt = :states`, pruning leaves `Mz` opaque, because it is target-independent (`system.jl:168-202`). Under `wrt = :forcing` the calculus must descend into it, and it fails on the filter (`bands.jl:204-206`).
- Forcing gradients through reseact's transport therefore need U9's filtered-faq rule at M4, not M6/M7.

## Decisions of record

| # | Question | Decision | Why |
|---|---|---|---|
| 1 | One axis or two | **`"parameters"` keeps scalar and array parameters, now both real columns. New `"forcing"` axis for live buffers.** | Accumulation differs: a gridded field is summed over the run, a forcing slot is summed per provider record. The pruning seed must follow the target set (`system.jl:191-197`). |
| 2 | Which variables are forcing | **An input to the producer (`forcing = [names]`). Default: every array parameter with an `update` of kind `data`.** | reseact's `F_PS` declares no `update`; its providers are built by hand (`reseact.esm/prototypes/reseact_3d_chem/split_common.jl:277-282`). The planner knows the providers, so it passes the set. |
| 3 | The record axis | **An ordinary column dimension, written as an integer literal in `col_idx`** | The blend is model AST. Brackets get no special case, so no blend shape is privileged. |
| 4 | Column identity | **`(buffer name, multi-index over the declared shape)`. Canonical ordinal: buffers sorted by name, then column-major (first index fastest).** | Matches the Julia buffer order (`oop.jl:738-744`), U4's column-major `ForcingSet`, and the species-major state layout. Each binding maps the ordinal to its own memory. |
| 5 | Unresolvable column | **Error on the `parameters` and `forcing` axes. Never skip.** | The ghost-column skip is legitimate only for state stencils reading past the domain (spec §4, `esm-jacobian-spec.md:188-189`). |
| 6 | Several blocks per model | **`jacobians.<Model>` MAY be an array of blocks with distinct `wrt`. A single object stays legal.** | Additive. Derived documents (architecture §3.2 pass 6) carry one block each, but a planner writing several axes for one operator should not need several files. U8 reuses this. |
| 7 | Accumulation | **The integrator owns it. The block describes `∂f/∂F` at one evaluation point.** | Which bracket a stage read is runtime state (tape), not model structure. |

## Design

### 1. The axis and the block

`jacobian_bands(sv; wrt = :forcing, forcing = names)` differentiates every differential equation with respect to every site of every name in `forcing`. Inlining is pruned with `inlinable_observed` seeded by `forcing` (`system.jl:263-293`). That is the same mechanism, for the same reason, as `wrt = :parameters`: emissions regridding is forcing-dependent but state-independent.

The block gains two things. Illustrative, with reseact's names; the shapes and the row are invented:

```json
"jacobians": { "Transport3D": [
  { "wrt": "states", "entries": [ … ] },
  { "wrt": "forcing",
    "columns": [ { "name": "F_PS", "shape": [2, 361, 576] } ],
    "entries": [
      { "row": "PS_consumer", "col": "F_PS",
        "rows": [[1, 13], [1, 7], [1, 72]], "row_idx": ["i", "j", "k"],
        "col_idx": [1, {"op": "+", "args": ["LAT0", "j"]}, {"op": "+", "args": ["LON0", "i"]}],
        "coef": {"op": "*", "args": [100, {"op": "-", "args": [1, "w_I3"]}, "…∂f/∂PS…"]} },
      { "…": "the same entry with col_idx[1] = 2 and coef factor w_I3" }
    ] } ] }
```

- `columns` is required when `wrt` is `"forcing"`, and on `"parameters"` whenever an array parameter appears. It lists each column variable with its integer shape in canonical order, so a consumer can build the column space without the source document.
- `col_idx` is full rank, including the record index.
- Column-index expressions must be closed at build time: literals, metaparameters, `const` gathers, and row or contracted names. This is the rule state columns already follow (`system.jl:250-257`).
- In the example, `LAT0` and `LON0` must therefore be folded constants. A producer that cannot fold them fails. It does not emit an entry it cannot place.

**Meaning** (unchanged from spec §4): for each row point `r` and contracted point `c`,

```
G_F[ row[r], (col, col_idx(r, c)) ] += coef(r, c)
```

`structure` is omitted for `wrt ≠ "states"`. The three classes describe square matrices (`structure.jl:32-67`).

### 2. The math of the bracket

Let `F` be a buffer with shape `[2, m…]`, and let row `r` read the blended value `F̃(t)[m] = (1 − w(t))·F[1,m] + w(t)·F[2,m]` through any chain of observed. With `φ_r = ∂f_r/∂F̃[m]` from the ordinary rules:

```
∂f_r/∂F[1,m] = (1 − w(t))·φ_r        ∂f_r/∂F[2,m] = w(t)·φ_r
```

A non-blend read such as `dPSdt = (F[2,m] − F[1,m])/Δ` adds `±φ'_r/Δ` to the same two columns. Entries are additive (spec §4), so both contributions land in one column. A regrid between the buffer and the blend (wildlandfire) is a `reduce: "+"` contraction with `const` weights. It yields `contracted` bands over the native grid, which spec §4 already defines.

### 3. vjp_forcing and per-record accumulation

**Per stage.** At each stage evaluation `(U_i, t_i)`, the method adjoint (architecture §5) produces a stage cotangent `λ_i`, already scaled by the method weights. The contribution to the gradient of the buffer set `B_i` that the stage actually read is the transposed scatter-add over band points:

```
ḡ_B[col(r,c)] += coef(U_i, t_i; r, c) · λ_i[row(r)]
```

No transposed matrix is materialized. The accumulators are f64 (D18) and have the buffers' shapes.

**Per record.** A buffer slot is not a dataset record. The tape records each provider's bracket per epoch: the dataset record ids of slots 1 and 2. When the reverse sweep crosses a refresh of provider `P` at `t_k`, slot `s`'s accumulator is added into the gradient of record `rec_P(e, s)`, and the accumulator is zeroed. The gradient of record `R` is

```
∂J/∂R = Σ over epochs e of P, Σ over slots s with rec_P(e, s) = R, of ḡ_{P,e}[s, …]
```

**Consequences.**

- **Accumulation is per provider, not per integrator epoch.** GEOS-FP's collections are not co-phased: I3 refreshes 3-hourly at 00:00, A3 at 01:30, and A1 hourly at 00:30 (`geosfp.esm:601`, `:618`, `:635`). One I3 bracket therefore spans several integrator epochs.
- **The degenerate final bracket `[last, last]` (`provider.jl:61-62`)** maps both slots to one record, and the two gradients sum.
- **When a record is final.** Record `R` can stream to an EarthSciIO sink (U12) once the reverse sweep has passed the earliest epoch whose bracket contains it.

### 4. Bracket boundaries (a runtime precondition, recorded here)

The frac weight is 0 at every record time. Under architecture §7.5's half-open `(t0, tf]` epochs, and in reseact's loop, the step ending at a boundary is integrated before the refresh (`reseact.esm/tools/adjoint_gradient.jl:1046-1056`).

So a stage evaluated at exactly `t = tf` reads the old bracket with `w = 0`. That is record `k−1`, where the field's value is record `k`. Every `c_i = 1` stage of the last step before a refresh has this O(ΔF) forcing error. Examples are the third SSPRK43 stage and the last Rosenbrock23 stage. This is by reading, not measured. It is exact in floating point for GEOS-FP's anchors: `(10800 − 0)/10800` and `(16200 − 5400)/10800` are exactly 1.

The bands are faithful to whatever the right-hand side computed, so the discrete adjoint stays consistent with the computed trajectory. Attribution requires only that the tape record which buffer set each stage read.

The fix belongs to the integrator. Evaluate a stage with the buffers of the epoch that contains `t` under `[t0, tf)`; the prefetched next epoch is already resident (§7.5). The alternative is a clamp weight with injected bracket times, which `geosfp.esm` deliberately avoids. This is Open question 2.

### 5. Bindings

| Binding | Work |
|---|---|
| **Julia (reference)** | `jacobian_bands(…; wrt = :forcing, forcing)`. `_colmap` (`assemble.jl:53-64`) gains canonical array-parameter and forcing column spaces; the strict mode becomes mandatory on these axes. `prepare_jacobian(…; wrt = :forcing)` builds the derived band model with the buffers bound through `build_kwargs` (`param_arrays`). New `vjp_forcing!(ḡ, λ, u, p, t)` does the transposed scatter on the prepared evaluator, without a CSC transpose. Emission writes `columns` and the block array. |
| **Python** | The same axis in the band calculus; `columns` emission; the numpy numeric oracle extended to forcing arrays. |
| **Rust** | `Wrt::Forcing` (`EarthSciASTDiff/pkg/earthsci-astdiff-rs/src/jacobian.rs:10-15`). The oracle `resolve` already gathers N-d arrays from `env.params` (`numeric.rs:58-72`); it gains `env.forcing`. Production assembly comes from U10. Prerequisite: U10 W0 (the Rust binding currently lags EarthSciAST). |

## What this is not

- **Not a forcing representation in EarthSciAST.** No record construct, no new op. The bracket stays an ordinary axis.
- **Not the adjoint of EarthSciIO's own processing.** If a provider regrids outside the AST, the gradient is with respect to the buffer the model sees. Mapping it back is EarthSciIO's business.
- **Not the bookkeeping.** Slot-to-record maps, accumulators and streaming live in the integrator (architecture §10).
- **Not `∂f/∂t`.** The weight's time dependence stays on `wrt: "time"`. Mixed second derivatives such as `∂J/∂F` are U8.

## Conformance and tests

1. **Fixture `tests/valid/forcing_bracket.esm`.** It contains:
   - scalar and array states;
   - a forcing parameter with a leading size-2 axis;
   - a frac-blend weight;
   - a `dPSdt`-style difference of the two records;
   - a gather at a folded constant offset;
   - a regrid-like `reduce: "+"` contraction with `const` weights over one record;
   - one CONST array parameter, standing in for an emission field.
2. **Goldens (byte-identical in Julia, Python and Rust):**
   - `forcing_bracket.forcing.bands.json` and `.forcing.block.json`;
   - `forcing_bracket.parameters.block.json`, now carrying array-parameter columns and `columns`.
3. **Value goldens** `forcing_bracket.forcing.jvals.json` and `.parameters.jvals.json`, with the exact key set and 1e-10 relative. Julia produces them, with ForwardDiff through the tree-walk right-hand side with respect to buffer contents as the oracle. They are evaluated at `w = 0.37` and at `w = 0` exactly, and never at the default point (architecture §3.2 pass 8).
4. **Regressions for the silent paths:**
   - `prepare_jacobian(wrt = :parameters)` on the fixture must give the array parameter nonzero columns.
   - A forcing name that resolves to no buffer must throw.
   - An empty forcing block for a document that reads forcing must throw. This is the `validate_plan` lesson (`rx_sym_block_jac.jl:288-302`).
5. **Integrator side (not gated here):**
   - a dot-product test of `vjp_forcing` against the forward forcing tangent per leaf step;
   - finite differences on the frozen tape with respect to one dataset record, across an I3 boundary.

## Work breakdown

1. **Spec.**
   - §1: the `forcing` axis, and array parameters on `parameters`.
   - §4: `columns`, the block array, and the error rule for unresolvable columns.
   - §5: the fixture.
   - §7: delete "forcing buffers / const arrays" from the out-of-scope list. That list is already stale: `tgrad`, contracted bands and templates are implemented (README:133-143).
2. **Julia:** `jacobian.jl`, `system.jl` (the seed), `assemble.jl` (column spaces and strictness), `emit.jl`, and `vjp_forcing!`. Goldens regenerated in the same commit, per the repository rule (`EarthSciASTDiff/CLAUDE.md`, "Conventions").
3. **Python:** the band calculus, emission and oracle.
4. **Rust:** after U10 W0.
5. **Downstream:** reseact's `RESEACT_ADJ_BUFGRAD` path retires once the integrator gates M4.

## Alternatives considered

- **Differentiate with respect to the blended field, and let the integrator split by `(1−w, w)`.** Rejected. It needs blend recognition, which fails for `dPSdt` and for blend-after-regrid. It also duplicates what the `*` rule already does.
- **Separate variables per record (`F0`, `F1`).** Rejected. The documents and EarthSciIO both use a length-2 axis, and renaming would be document surgery.
- **Enzyme on the buffers, as reseact's `BUFGRAD` path did.** It is Julia only (D4 needs Rust parity) and cost about 1.5 GB per VJP at CONUS (`adjoint_gradient.jl:84-86`). It remains the Julia oracle.
- **Fold forcing into `"parameters"`.** Rejected, by Decision 1. Array parameters are fixed there anyway.

## Open questions

1. **Inferring the forcing set from the document.** Should the forcing set be inferred (`update.kind == "data"`) once documents declare it, so the planner's list becomes a check rather than an input?
2. **Boundary semantics (§4).** Should the integrator adopt `[t0, tf)` stage-buffer selection, or should documents switch to a clamp weight with injected bracket times? Either way, architecture §7.5's `(t0, tf]` needs revisiting.
3. **Out-of-range forcing reads.** Should these follow the evaluator's rule for that read (fault or zero), with producers mirroring it? This RFC only forbids a silent skip at assembly.
4. **f16/bf16 forcing storage (D18).** The band coefficients read the upcast values. Is the gradient with respect to the stored low-precision record, or the f64 upcast? The proposal is the upcast, with the cast's derivative 1.
