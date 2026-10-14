---
title: "Production-rate Jacobian assembly in Rust"
description: "Give earthsci-astdiff-rs a production numeric path: band coefficients compiled once, through earthsci-ast-rs's own lane and tile programs, from a derived evaluation document that is byte-identical across Julia and Rust. Values are scattered by precomputed, golden-gated plans straight into the integrator's storage: lane-SoA per-cell blocks, KPP-style sparse lane blocks, column bands and CSR. Matrix-free J·v and Jᵀ·λ are also provided. The program is Send + Sync with one workspace per thread and is bitwise deterministic. The target is at least as fast as Julia's jac! per call. First, the binding catches up with EarthSciAST: the faq tag and the 1.0.0 variable model. Not diffsol integration."
---

Status: draft (proposed by EarthSciIntegrator, 2026-10-13)
Target: EarthSciASTDiff Rust binding (`pkg/earthsci-astdiff-rs`, next minor). New conformance artifacts in `tests/goldens/`: evaluation documents and scatter plans. No change to the `"jacobians"` block format. Julia gains the same plan artifacts (D8).
Needed by: EarthSciIntegrator milestone M3 (reseact forward in Rust: Rosenbrock with `block_lu`). M4 uses it for U7 and U8 values.
Integrator refs: architecture.md §3.2 (passes 4, 6, 8), §6.1, §7.3, §7.6, §8, §13 (U10), §15 (risk 4); decisions D1, D8, D10, D11, D13, D18; sibling RFCs U3 (lanes, tiles), U4 (`RhsProgram`, `RhsWorkspace`, `ForcingSet`)

File citations are relative to the `code/` directory that holds all the repositories.

## Summary

The Rust binding matches the Julia reference byte for byte on bands, blocks and factorization plans, and to 1e-10 on values (`EarthSciASTDiff/README.md:164-168`). Its numeric assembly, however, is "a correctness surface" (`EarthSciASTDiff/pkg/earthsci-astdiff-rs/src/numeric.rs:1-2`). Each call:

- re-derives the bands;
- substitutes and evaluates an expression tree per entry per cell;
- returns `BTreeMap<(String, String), f64>`.

The README's next step is "the diffsol solver integration (production-rate `jac!` over the bands)". The integrator does not need diffsol. It needs fast, deterministic, thread-safe assembly into its own storage (architecture §6.1), plus matrix-free products for adjoints (architecture §10).

**Proposal.**

1. **W0, catch-up.** Bring the Rust binding level with EarthSciAST and the Julia reference. By reading, it no longer compiles against the sibling checkout, and if it did it would return empty band lists (Motivation 2).
2. **A trimmed derived evaluation document,** emitted identically by Julia and Rust and compiled with earthsci-ast-rs's lane and tile programs (U3, U4).
3. **`ScatterPlan`s** per storage layout: computed once, layout-checked, serialized for goldens.
4. **`JacobianProgram::eval_into`, `apply`, `apply_transpose` and `contract`** (the last for U8), all `&self` with one `JacWorkspace` per thread.
5. **A measured performance target** against Julia's `jac!`.

## Motivation

**1. The numeric path is an oracle, not an engine.**

- `assemble_jacobian` calls `jacobian_bands` on every invocation (`numeric.rs:232`), so it differentiates symbolically each time.
- For every band, row cell and contracted point it builds a `HashMap`, substitutes into the coefficient tree, resolves every read by cloning `Expr` nodes, and calls the generic `evaluate` (`numeric.rs:43-147`, `:236-273`).
- The result is keyed by strings (`numeric.rs:270-271`).

That is the right shape for an oracle and the wrong shape for a Rosenbrock leaf that assembles a 13×13 block for every lane at every step.

**2. The binding has fallen behind EarthSciAST (by reading; not built, because the repositories are read-only for this work).**

- **The faq tag.** It matches the pre-1.1.0 tag `"aggregate"` in `lhs_state` (`src/system.rs:80`), in the array-expression test and dispatcher (`src/bands.rs:102`, `:236`), in the inliner (`src/inline.rs:76`, `:126`, `:141`) and in the oracle (`src/numeric.rs:84`). The Rust loader rewrites every `"aggregate"` to `"faq"` at load (`EarthSciAST/pkg/earthsci-ast-rs/src/parse.rs:1789-1815`).
- **The variable model.**
  - It selects targets with `VariableType::State` (`src/jacobian.rs:42`), and observed definitions with `VariableType::Observed` and `ModelVariable.expression` (`src/system.rs:106-108`).
  - The current enum has only `Unknown` and `Parameter`, and says there is "no `expression` field on a variable" (`EarthSciAST/pkg/earthsci-ast-rs/src/types/model.rs:959-976`).
  - The dependency is a sibling path (`pkg/earthsci-astdiff-rs/Cargo.toml`), so the crate builds against that checkout.
- **History.** The last Rust change is 2026-08-24 (`295c728`). The Julia fix for the same tag landed on 2026-09-14 (`33d2b7c`). Its message describes the failure: every array state equation "classified as non-differential", "zero entries and zero scatter", and "a downstream adjoint driver … built an all-zero Jacobian and reported success". reseact recorded the same incident (`reseact.esm/tools/reactant_handoff/rx_sym_block_jac.jl:277-286`; `reseact.esm/DIRECT_RHS_CENSUS.md:340-349`).
- **Python is in the same state** (`pkg/earthsci-astdiff-py/src/earthsci_astdiff/system.py:77`; `bands.py:83`, `:162`), and its loader normalizes too (`EarthSciAST/pkg/earthsci-ast-py/src/earthsci_ast/parse.py:2937-2943`). By reading, Python would silently produce empty bands for array states. This is out of scope for this RFC and is reported here.

**3. The Rust calculus lacks the Julia scaling work.**

- **Pruning.** Rust inlines every observed (`src/jacobian.rs:53-58`). Julia prunes to target-dependent observed (`EarthSciASTDiff/pkg/EarthSciASTDiff.jl/src/system.jl:263-293`), which "is what lets a coupled document be differentiated at all" (`system.jl:183-189`).
- **The target scan.** Rust scans every target per equation (`src/jacobian.rs:61-73`), which Julia replaced with a set intersection because the scan is quadratic on scalar-lowered documents (`jacobian.jl:59-71`).
- **Missing pieces.** Rust has no evaluation-model CSE (`hoist_observed`, `eval_cse.jl:53`), which cut PPM's `jac!` 8× (`README.md:119-127`), and no `source_layout` check.

**4. Julia's `jac!` sets the bar, and leaves room.**

- **How it works.** It evaluates a derived document in which each band's coefficient field is a state, compiled by the stock evaluator (`emit.jl:184-322`). It copies `u` into the derived state (`assemble.jl:300-302`), then scatters into CSC `nzval` (`assemble.jl:306-310`).
- **Wasted work.** `_base_document` copies every equation of the source, the `D` equations included (`emit.jl:213-216`). So each call also recomputes `f`.
- **reseact's measurements.** `prepare_jacobian` takes 33.9 s at CONUS 13×7×72 (`reseact.esm/COMPILE_COST.md:1421`) and a projected 30 min at 0.25° (`reseact.esm/SCALE_025.md:93-95`). The plan-versus-evaluator check takes 336–338 s (`COMPILE_COST.md:1422`).
- **What the consumer had to do.** It reverse-engineered the evaluator's scatter into gathers, and had to check state ordering because "two builds and two var maps" could "put every band in the wrong cell with no visible symptom" (`rx_sym_block_jac.jl:37-44`). A scatter plan should be a first-class, checked artifact.

**5. What the integrator consumes.**

- **Storage.** Architecture §6.1 lists:
  - `block_lu`: lane-SoA, vectorized across lanes;
  - `sparse_block_lu`: KPP-style, from the Markowitz plan (`factorization.jl:32-146`);
  - `banded` / `block_tridiagonal`: per column;
  - `krylov`: global CSR;
  - `matrix_free`.
- **Executor contract.** Stateless executors, scratch with no driver state (§7.3).
- **Determinism.** Bitwise results per decomposition (D10).
- **Precision.** Precision roles, including f32 RHS and Jacobian storage (D18).

## Decisions of record

| # | Question | Decision | Why |
|---|---|---|---|
| 1 | Scope | **Assembly and matrix-free products only.** No solver, and no diffsol. | The integrator owns its linear algebra (architecture §6). diffsol stays the README's separate future item. |
| 2 | How coefficients are compiled | **Through earthsci-ast-rs's `RhsProgram` and its lane and tile programs (U4, U3), applied to a derived evaluation document**, as Julia v1 does (`emit.jl:184-195`) | One compiler, one set of numerics (f32 included, U5), tape performance work shared with the RHS. No second expression compiler to keep in step. |
| 3 | The derived document | **Trimmed:** the source's observed definitions and the band fields, with **no source `D` equations**. CSE'd by a port of `hoist_observed`. Byte-identical in Julia and Rust: a new golden. | It removes the hidden RHS evaluation from every call. A golden makes it a conformance artifact (D8). |
| 4 | Scatter | **`ScatterPlan` data per (band list, source layout, storage layout).** Serializable and golden-gated. Built in strict mode only. | It makes the reseact class of ordering bug impossible to miss. Every row and column name must resolve; an empty plan for a document with array states is an error (`rx_sym_block_jac.jl:288-302`). |
| 5 | Layouts | **`LaneBlocks`, `LaneSparseBlocks`, `ColumnBanded`, `Csr`.** Matrix-free `Apply` and `TransposeApply`; `Contract` for U8. | Exactly architecture §6.1's list. |
| 6 | Concurrency | **`JacobianProgram` and `ScatterPlan` are `Send + Sync` and immutable. `JacWorkspace` is `Send`, `!Sync`, one per thread. Forcing comes in as U4's `ForcingSet`.** | Matches U4's split. No `Rc` or `RefCell`. |
| 7 | Determinism | **Fixed accumulation order: plan order within a lane, row or column. Each output element is owned by one thread.** No atomics, no thread-local partial sums. | D10: bitwise invariance to thread count. |
| 8 | Plan size for per-cell operators | **One per-cell template, O(nnz), plus a uniformity check across lanes** | A per-point plan for 0.25° chemistry would hold 1.35M cells × ~89 entries. reseact's plan already assumes and checks a uniform per-cell pattern (`rx_sym_block_jac.jl:168-174`). |
| 9 | Performance gate | **Rust `eval_into` at most 1.0× Julia `jac!` per call** on the reference benchmark (Design §5) | A conservative floor. Decision 3 alone removes one RHS evaluation per call. The benchmark also answers architecture §15 risk 4. |

## Design

### 1. Build pipeline

```
bands (existing calculus, after W0)
  → evaluation document (Decision 3; golden <fixture>.<wrt>.evaldoc.json)
  → RhsProgram::from_document → .select(band-field DOFs)          (U4, U3 §2)
  → .lanes(kind, capacity) | .tile(dec, box)                       (U3 §6 / §5)
  → ScatterPlan::build(bands, source_layout, layout)               (golden <fixture>.<wrt>.<layout>.plan.json)
```

**The evaluation document** has:

- **States:** the source's states, as inputs only. Their `D` equations are dropped, so their `du` slots are never computed.
- **Equations:** one band-field unknown `J_k` per band, with `D(J_k) = makearray(0, band region ← faq(coef))` (the same shape as `emit.jl:274-320`).
- **Observed:** the source's observed definitions plus the hoisted CSE observed (`eval_cse.jl`).
- **Contracted bands** get the extra axes they already get in Julia (`emit.jl:285-306`).

**The band slab.** For pointwise operators, evaluating with `LaneKind::Cell` fills `du` with species-major band fields, lane fastest: `B[k][ℓ]`. For column operators the lanes are columns, `B[k][lev][ℓ]`. Stencil operators use tile programs.

**What the pass checks.**

- `source_layout` is always strict: every source name resolves in the `RhsProgram` the integrator built for the same operator.
- It refuses a layout from a different build (`assemble.jl:115-136` checks only the cheap half).
- It refuses an empty band list when the operator writes array states.

### 2. Storage layouts

Here `L` is the lane capacity (padded, architecture §7.6), `n` the per-cell block size, and `T ∈ {f32, f64}`. Positions are in elements of `T`, and all indices (`r`, `c`, `k`, `ℓ`, `q`) are 0-based.

| Layout | Holds | Position of `J[r, c]` for lane `ℓ` | Notes |
|---|---|---|---|
| `LaneBlocks { n, L }` | dense n×n per cell | `(r + n·c)·L + ℓ` | Column-major block, lane fastest: the batched LU vectorizes across lanes (§6.1). The block variable order is the factorization plan's `block_vars` (spec §6), or sorted names when there is no plan. |
| `LaneSparseBlocks { lu_pattern, L }` | KPP-style sparse block | `q·L + ℓ`, `q` = index of `(r, c)` in the plan's `lu_pattern` list (`factorization.jl:136-145`) | Fill-in slots are zeroed on every call. The elimination is a short op tape over lanes (§6.1). |
| `ColumnBanded { axis, nvar, kl, ku, N, L }` | block-banded per column | `((((c·nvar + r)·(kl+ku+1) + (d+kl))·N + k)·L + ℓ`, with `d = col_level − k` | `nvar = 1` is plain banded (batched Thomas). `kl = ku = 1` with `nvar > 1` is block tridiagonal. Bands come from `column_block` / `banded` structure (U9). |
| `Csr { rows, ncols }` | general sparse | row pointer, sorted column indices, merged duplicates | For stencil and global operators. Rows are the operator's DOF set (U3 `select`), columns the state slots it reads, halos included. Krylov preconditioners (bjacobi-ILU(0), ASM) read it. |

**Matrix-free targets.**

- `Apply` (`y += J·v`) and `TransposeApply` (`ḡ += Jᵀ·λ`) walk the band slab with the plan's (slot, row, col) triples. Nothing is assembled. `TransposeApply` is `vjp_u` for explicit and Rosenbrock stages, and with the U7 axis it is `vjp_forcing`.
- `Contract` is U8's second-order kernel, over a second-order slab.

**Accumulation.** It is always `+=` in plan order, because several bands may land on one entry (`rx_sym_block_jac.jl:22-27`; spec §4 additivity). Output elements not written by the plan are zeroed first, per lane, by the owning thread.

### 3. API

```rust
pub struct JacobianProgram { /* Arc<Inner>: RhsProgram of the evaluation document, band table */ } // Send + Sync + Clone
pub struct ScatterPlan     { /* layout, per-cell template or per-point items, checks */ }          // Send + Sync, serde
pub struct JacWorkspace    { /* RhsWorkspace, band slab */ }                                         // Send, !Sync

pub enum Wrt { States, Parameters, Forcing, Time }        // Forcing per U7
pub enum Layout { LaneBlocks{..}, LaneSparseBlocks{..}, ColumnBanded{..}, Csr{..} }

impl JacobianProgram {
    pub fn build(doc: &EsmFile, model: Option<&str>, wrt: Wrt, opts: &BuildOptions)
        -> Result<JacobianProgram, JacError>;
    pub fn bands(&self) -> &[JacEntry];
    pub fn plan(&self, layout: &Layout, source: &RhsProgram) -> Result<ScatterPlan, JacError>;
    pub fn new_workspace(&self, lanes: usize) -> JacWorkspace;

    /// Evaluate band fields for the given lanes and scatter them into `out` per `plan`.
    pub fn eval_into<T: StateScalar>(&self, plan: &ScatterPlan, u: &[T], p: &[f64], t: f64,
        forcing: &ForcingSet, out: &mut [T], ws: &mut JacWorkspace) -> Result<(), JacError>;
    pub fn apply<T: StateScalar>(&self, plan: &ScatterPlan, ws: &JacWorkspace, v: &[T], y: &mut [T]);
    pub fn apply_transpose(&self, plan: &ScatterPlan, ws: &JacWorkspace, lambda: &[f64], gbar: &mut [f64]);
}
```

- `apply` and `apply_transpose` reuse the band slab left in `ws` by the last `eval_into`. That is the common case in a Rosenbrock adjoint, which evaluates once per stage point and then applies several times.
- `u` is species-major with lanes fastest, U3's lane layout. `p` is U4's positional parameter vector. `forcing` is U4's immutable `ForcingSet`.
- `apply_transpose` accumulates in f64 whatever `T` is (D18: adjoint accumulators are f64).

### 4. Concurrency and determinism

- A thread owns whole lanes, for blocks and bands, or whole rows, for CSR. Lane pools refill lanes from a queue (architecture §7.6), so a lane's block depends only on that cell's inputs and the plan order.
- With U3's bitwise lane guarantee, assembled values are bitwise invariant to shard composition, thread count and lane position, provided padding lanes duplicate a real cell (U3).
- `apply_transpose` for CSR and stencil operators writes columns owned by other threads. It runs as a gather over a precomputed transposed plan, column-owned, never as a scatter with atomics.

### 5. Performance target and measurement

**Benchmarks.** Each runs single node, warm, median of 50 calls, f64 and f32.

| ID | Operator | Layout |
|---|---|---|
| B1 | `bd_chem.esm` (fixture) | `LaneBlocks` |
| B2 | reseact SuperFast chemistry operator, CONUS 4×5° (13×7×72 = 6,552 cells, n = 13; survey §1) | `LaneBlocks` and `LaneSparseBlocks` |
| B3 | CBM-Z box replicated over 4,096 lanes (n = 68, from `EqWeFiC/components/gaschem/cbmz/cbmz.esm`) | `LaneSparseBlocks` |
| B4 | `flux_form_adv.esm` and `minmod_adv.esm` scaled to 10⁵ cells | `Csr`, `TransposeApply` |
| B5 | reseact PBL mixing operator (a column contraction, U9) | `ColumnBanded` with dense lines, or `Csr` until U9 lands |

**Targets.**

- **T1 (gate).** On B2, Rust `eval_into` with a single thread takes at most 1.0× Julia's `jac!(J, u, p, t)` from `prepare_jacobian(…; eval_cse = true)` on the same operator and point. The Julia figure includes its CSC scatter.
- **T2.** On B2 and B3, `eval_into` costs at most 2× one evaluation of the same operator's RHS on the same lanes (U3 `LaneProgram`). The exact AD block Jacobian costs about one RHS evaluation carried on `n`-partial duals (`reseact.esm/prototypes/reseact_3d_chem/block_jac.jl:99-103`), so an analytical one should be in the same range.
- **T3.** On B2 at 0.25° shard sizes (architecture §7.6), parallel efficiency is at least 80% at 16 threads.
- **T4.** On B4, `TransposeApply` costs at most 1.5× `Apply`.
- **T5.** Building a plan is O(nnz) for lane layouts and O(points) for CSR, and is never on the per-step path.

The results go into the integrator's risk register (architecture §15 risk 4).

### 6. W0: catch-up, in order

1. **The faq tag:** `"aggregate"` becomes `"faq"` at every site listed in Motivation 2, with the Julia regression test ported (`pkg/EarthSciASTDiff.jl/test/faq_tag_test.jl`). Python gets the same fix.
2. **The esm 1.0.0 variable model:** states are the `D`-targeted unknowns, observed are bare- or faq-LHS definitions per EarthSciAST's classification (`EarthSciAST/pkg/earthsci-ast-rs/src/classification.rs`), and parameters are `VariableType::Parameter`.
3. **Pruning:** `inlinable_observed` with a per-`wrt` seed, and the target-set intersection.
4. **A non-empty gate.** Every band golden must have a non-empty entry list wherever the fixture has array states. `scripts/test-conformance.sh` runs all three bindings (`scripts/test-conformance.sh:13-16`), so CI catches the next tag rename.

## What this is not

- **Not diffsol integration, and not a linear solver, preconditioner or factorization.**
- **Not new derivative rules.** The rules are U7, U8 and U9.
- **Not GPU.** Rust has no GPU in v1 (D15). EarthSciAST's XLA emitter is not used here.
- **Not f16 or bf16 arithmetic.** Low-precision W storage (D18) is a conversion the integrator applies after assembly.
- **Not a replacement for the oracle.** `numeric.rs` stays as the reference that the production path is tested against.

## Conformance and tests

1. **Existing goldens** (`*.bands.json`, `*.block.json`, `*.jvals.json`) pass after W0 on all five fixtures.
2. **New `*.evaldoc.json` goldens:** the trimmed evaluation document, byte-identical in Julia and Rust. Hoisting order and naming are deterministic (`jcax<size>`, `emit.jl:230-241`).
3. **New `*.<layout>.plan.json` goldens:** byte-identical plans for each applicable layout. Julia implements the same `ScatterPlan` for its own runtime (D8).
4. **Values:** production against the oracle on every fixture at three non-default random points (architecture §3.2 pass 8), within 1e-13 relative in f64. In f32 the tolerance is set by the f32 RHS gate (U5).
5. **Determinism:** bitwise-identical output for 1, 2, 7 and 16 threads, for shuffled lane orders, and for padded and unpadded capacities.
6. **Refusals:**
   - an empty band list for an array-state operator;
   - a source layout from another build;
   - a non-uniform per-cell pattern under `LaneBlocks`;
   - an unresolvable parameter or forcing column (U7).
7. **Thread-safety assertions:** `assert_impl_all!(JacobianProgram: Send, Sync)`, `assert_impl_all!(ScatterPlan: Send, Sync)` and `assert_impl_all!(JacWorkspace: Send)` in CI.

## Work breakdown

1. **W0, catch-up (§6).** This blocks everything else and also unblocks U7, U8 and U9 in Rust.
2. **W1:** port the trimmed evaluation document and `hoist_observed`, and add the `evaldoc` goldens. Julia switches `_evaluation_document` to the trimmed form in the same change.
3. **W2:** compile through `RhsProgram` and lanes and tiles. Depends on U4 and U3.
4. **W3:** `ScatterPlan` and the four layouts; plan goldens; the Julia counterparts.
5. **W4:** `Apply`, `TransposeApply`, and U8's `Contract`.
6. **W5:** benchmarks B1–B5 and targets T1–T5; results recorded in EarthSciIntegrator.
7. **README:** replace "the diffsol solver integration" with this RFC, and leave diffsol as an unrelated item.

## Alternatives considered

- **diffsol integration, as the README plans.** It does not serve the integrator, which owns its solvers. It may still be built on top of `Csr`.
- **A coefficient compiler private to earthsci-astdiff-rs** (expression to tape). It duplicates earthsci-ast-rs's tape, numerics and f32 handling. It is the fallback if U3 or U4 slip, with T1 still the gate.
- **Optimize the oracle.** Per-cell tree substitution and string keys cannot reach production rates, and the oracle is more valuable left simple.
- **Generate Rust source.** It needs a compile step per model, is hard to golden, and diverges from Julia's evaluator path.
- **A CSC prototype and scatter, as Julia does.** It forces lane-SoA consumers to invert the scatter, which is what reseact had to do (`rx_sym_block_jac.jl:9-27`).

## Open questions

1. **U3 and U4 timing.** If `RhsProgram` lanes are late for M3, is the private coefficient compiler acceptable as a stopgap behind the same API?
2. **Block variable order.** Should `LaneBlocks` use the Markowitz `block_vars` order always, or the natural sorted order with the permutation applied in the LU? The first saves a gather. The second keeps dense and sparse layouts interchangeable.
3. **Shared plans across leaves.** One `ScatterPlan` per (operator, layout) is shared by every leaf using that operator. Do re-sharding events (architecture §7.6) ever need a plan rebuild? With per-cell templates (Decision 8) they should not.
4. **Python.** The production-rate item is also open there (`README.md:162-163`). Python comes later (architecture §0). The W0 tag fix for Python is urgent regardless.
