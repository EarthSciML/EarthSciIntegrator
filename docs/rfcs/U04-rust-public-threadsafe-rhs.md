---
title: "A public, Send + Sync right-hand side for earthsci-ast-rs"
description: "Make the compiled right-hand side a stable public object that is built once, shared by reference across threads, and evaluated concurrently with one caller-owned workspace per thread. Forcing becomes an immutable, epoch-stamped input set rather than a model-owned Rc<RefCell<…>>. The parameter vector ABI is the positional order the tape already uses. Julia's f! has the same non-reentrancy and gets the same split. No schema change."
---

Status: draft (proposed by EarthSciIntegrator, 2026-10-13); in review upstream as EarthSciAST PR #561
Target: EarthSciAST API_SPEC §5.8 (a new stable entry beside `esm_problem`), `earthsci-ast-rs` next minor (0.3.0); a parallel reentrant entry in `EarthSciAST.jl`. No schema or format-version change.
Needed by: EarthSciIntegrator milestone M3 (shared-memory shards and tiles). Julia's half is also needed by M3.
Integrator refs: architecture.md §7.2, §7.3, §7.5, §10 (accumulation), §13 (U4); decisions D1, D5, D6, D7, D10, D11, D13

## Summary

The integrator runs many right-hand-side evaluations at once: one per shard,
lane pool or tile, on a thread pool it owns (D6, architecture §7.3). Today
`earthsci-ast-rs` has no public way to do that.

- The RHS is reachable only through `#[doc(hidden)]` debug seams.
- Every object on the path is `!Send` and `!Sync`.
- Forcing lives in a model-owned `Rc<RefCell<…>>` that callers mutate in place.

This RFC adds one public type, `RhsProgram`, with these properties:

- It is built once, immutable, `Send + Sync + Clone`, and shared through an `Arc`.
- It is evaluated with `eval_into`, writing into a caller-owned output with a
  caller-owned `RhsWorkspace`, one per thread.
- Forcing arrives as an immutable `ForcingSet` stamped with a unique epoch, so
  there is no invalidation call for a caller to forget.
- The parameter vector is the positional `f64` slice the tape already reads.

U3 (tiles and lanes) and U6 (stripped constructs and observed kernels) build
their programs on this type, and U5 fixes its precision behaviour. Julia's
in-place `f!` is not reentrant either, so the same split is proposed for Julia.

## Motivation

**The only route to the RHS is hidden.**

- `ArrayCompiled::debug_eval_rhs` (`EarthSciAST/pkg/earthsci-ast-rs/src/simulate_array/driver.rs:526-561`)
  allocates a fresh scratch on every call and never runs the tape. The tape's
  own module docs call it "the oracle" (`tape/mod.rs:16`).
- The allocation-free path is `debug_eval_rhs_into` with a scratch from
  `debug_new_scratch_taped` (`driver.rs:574-582`, `:612-645`). Both are
  `#[doc(hidden)]`.
- So is `EsmProblem::debug_array_compiled` (`problem.rs:835-841`), the only way
  to get an `ArrayCompiled` from a Problem.
- The scaling adapter is the one external user. It strings these four seams
  together (`src/bin/earthsci-scaling-adapter-rust/main.rs:519-582`). Its header
  records "Native Rust does not thread its right-hand side" (`main.rs:29-31`).

**Nothing on the path can cross a thread.**

- `EsmProblem` holds `Rc<JsonValue>`, `Rc<Backend>`, `Rc<BuildProducts>`,
  `RefCell<BuildInspection>` and `Option<RefCell<RefreshExecutor>>`
  (`problem.rs:690`, `:707`, `:709`, `:712`, `:734`). The backend is
  `Backend::Array(Rc<ArrayCompiled>)` (`problem.rs:658-665`).
- `ArrayCompiled` (`simulate_array/mod.rs:621`) holds the following:
  - forcing as `Rc<RefCell<HashMap<String, ArrayD<f64>>>>` (`:696`);
  - `forcing_generation: Cell<u64>` (`:700`);
  - `const_scope: Rc<ConstArrayScope>` (`:735`);
  - `field_ic_memo: RefCell` (`:762`);
  - `tape_cache` (`:777`), whose entries are a `RefCell<Vec<CachedTape>>` holding `Rc<TapeProgram>` (`tape/mod.rs:295-306`);
  - `OnceCell<Rc<…>>`s for the XLA executable and shared observeds (`mod.rs:638`, `:780`).
- The scratch is no better.
  - `RhsScratch` holds `Rc<ConstArrayScope>` and `Option<TapeCtx>` (`rhs.rs:64`, `:71`).
  - `TapeCtx` holds `Rc<TapeProgram>` (`tape/exec/mod.rs:306-320`).
  - The fused executor's scratch stores raw `*const f64` / `*mut f64` (`tape/exec/fused.rs:349-354`).
- The interior mutability exists because diffsol's RHS is `Fn`. The code says so:
  "guarded by a `RefCell` because diffsol's RHS is an `Fn`, not `FnMut`"
  (`rhs.rs:27-28`). The forcing buffer is `Rc<RefCell<…>>` "for the same reason"
  (`mod.rs:675-696`).

**Results depend on thread-local state.**

- The working precision is three `thread_local!`s (`precision.rs:135-149`), read
  by `precision::active()` (`:207-209`). `precision::Env` holds an `Rc<VarPrecisions>`
  (`:287-293`).
- The debug entries do not arm it; the caller must (`driver.rs:583-591`).
- Fail-closed faults latch into a thread-local `CONST_OOB` (`eval.rs:1944`). The
  tape's `Instr::Fault` writes there too (`eval.rs:1960-1964`).

**Forcing can go stale silently.**

- `debug_new_scratch_taped` builds the tape with an empty discrete-forcing set
  (`driver.rs:578`). A forcing load then lands in the CONST section
  (`tape/lower.rs:743-748`).
- CONST re-runs only when the parameter hash changes (`tape/exec/mod.rs:523-529`,
  `:572-591`).
- So a caller that refreshes forcing through `forcing_handle()`
  (`driver.rs:408-412`) and calls again with the same parameters reads the old
  forcing.
- The in-crate solver avoids this by bumping a forcing epoch on its own scratch
  (`driver.rs:3047-3069`). That bump is `pub(super)` (`rhs.rs:120-126`).

**Gridded parameters are baked in.**

- Shaped parameters with defaults are lowered into const observeds
  (`mod.rs:763-768`).
- `remake` refuses to change them (`baked_parameters`, `problem.rs:678-680`).
- D5 needs gradients with respect to gridded fields, so the integrator needs
  them as run-time inputs.

**Julia has the same shape of problem.**

- `f!` is "non-reentrant (one instance per integrator)"
  (`EarthSciAST.jl/src/tree_walk/acc_merge.jl:654-661`). It shares `_CSECache`
  buffers and the materialized-observed vector `_ObsExtVec` across calls
  (`build.jl:2757-2772`).
- The forcing epoch is process-global (`compile.jl:1218`).
- Forcing buffers are captured by alias at build time, not passed in
  (`build.jl:2027-2041`).
- With Polyester loaded, `f!` threads internally (`ext/EarthSciASTPolyesterExt.jl`;
  `access_kernel.jl:1132-1163`). That nests inside the integrator's own threads.
- Rebuilding `f!` per thread costs a whole build per thread: 650.2 s for
  reseact's CONUS chemistry reference build (`reseact.esm/tools/capacity_chem.jl:41-44`).

**Precedent in the crate.** `Expr` was moved from `Rc` to `Arc` "so `Expr` stays
`Send + Sync`" (`src/types/expression.rs:26-41`). `CallbackFn` is already
`Arc<dyn Fn + Send + Sync>` (`problem.rs:69`). `TapeProgram` is plain data
(`tape/ir.rs:1306-1363`), so it is `Send + Sync` once it sits behind an `Arc`
*(assumption: not compiled to check)*.

## Decisions of record

Proposed by the integrator, for the maintainers to rule on.

| # | Question | Proposal | Why |
|---|---|---|---|
| 1 | New type or make `ArrayCompiled` `Sync` | **A new immutable `RhsProgram`**, extracted from `ArrayCompiled` | `ArrayCompiled` also serves construction, field ICs, observed passes and diffsol. Making all of that `Sync` is a larger refactor than this ask needs. |
| 2 | Where mutable state lives | **Only in `RhsWorkspace`**, owned by the caller, one per thread | Matches the executor contract (architecture §7.3): stateless executors, scratch with no driver state. |
| 3 | Forcing | **An immutable `ForcingSet` with a process-unique epoch**; every forcing slot is refreshable | Makes the stale-forcing hazard above impossible. Swapping sets needs no lock. |
| 4 | Which compilers | **`native` only in v1** (every rule taped) | Fallback rules evaluate through `EvalCtx` over `Rc<Expr>` rule bodies and thread-locals. XLA in Rust is after v1 (D15). |
| 5 | Internal threading | **None.** The program never spawns or joins. | The integrator owns parallelism (D6), and D10 needs fixed reduction orders. |
| 6 | Error channel | **`Result<(), RhsError>`**: no thread-local latch, no panic, no NaN-fill | Concurrent callers must not see each other's faults. |
| 7 | Gridded parameters | **Run-time `ArrayParameter` slots in the forcing set**, opt-in per problem | D5 gradients with respect to gridded fields need a live input to differentiate against. |
| 8 | Name | **`RhsProgram`** (canonical `rhs_program`) | `xla_runtime::CompiledRhs` is already a `pub struct` (`src/xla_runtime.rs:232`). U5's `CompiledRhs` placeholder becomes this type. |

## Design

### 1. Rust surface

```rust
pub struct RhsProgram { /* Arc<Inner>: tape, layout, slots */ }   // Send + Sync + Clone
pub struct RhsWorkspace { /* slab, state mirror, fused regs, epochs, fault */ }  // Send, !Sync
pub struct ForcingSet { /* epoch: u64, bufs: Arc<[Arc<[f64]>]> */ }  // Send + Sync

pub struct ParamSlot   { pub name: String, pub default: Option<f64> }
pub struct ForcingSlot { pub name: String, pub shape: Vec<usize>, pub kind: SlotKind }
pub enum   SlotKind    { Forcing, ArrayParameter }
pub struct StateBlock  { pub name: String, pub flat_offset: usize, pub shape: Vec<usize> }

impl RhsProgram {
    /// Build once. Refuses with `compiler_refused_rule` if any rule falls back.
    pub fn from_problem(prob: &EsmProblem) -> Result<RhsProgram, SimulateError>;

    pub fn n_states(&self) -> usize;
    pub fn state_layout(&self) -> &[StateBlock];      // column-major blocks, flat order
    pub fn parameters(&self) -> &[ParamSlot];         // the order of `p`
    pub fn parameter_vector(&self, overrides: &HashMap<String, f64>)
        -> Result<Vec<f64>, SimulateError>;           // strict: missing → error
    pub fn forcing_slots(&self) -> &[ForcingSlot];    // the order of a ForcingSet
    pub fn new_workspace(&self) -> RhsWorkspace;

    /// du = f(u, p, t; forcing). Overwrites all of `du`.
    pub fn eval_into<T: StateScalar>(
        &self, u: &[T], p: &[f64], t: f64, forcing: &ForcingSet,
        du: &mut [T], ws: &mut RhsWorkspace,
    ) -> Result<(), RhsError>;
}

impl ForcingSet {
    pub fn new(prog: &RhsProgram, bufs: Vec<Arc<[f64]>>) -> Result<ForcingSet, RhsError>;
    /// A new set, with a new epoch, sharing every buffer but `slot`.
    pub fn replace(&self, slot: usize, buf: Arc<[f64]>) -> Result<ForcingSet, RhsError>;
    pub fn epoch(&self) -> u64;
}

pub enum RhsError {
    Fault { code: &'static str, message: String },   // the §5.5.5 / §5.23 fail-closed faults
    Shape { what: &'static str, expected: usize, got: usize },
    StatePrecision { variable: String, declared: Precision },   // from U5
}
```

`StateScalar` and the precision behaviour of `eval_into` are U5's (§2 there).
This RFC only requires that the result never depends on the calling thread's
ambient precision.

### 2. Semantics

1. **Purity.** The output is a function of `(program, u, p, t, forcing contents)`
   and nothing else. It is bitwise identical across threads, across workspaces,
   whatever a workspace evaluated before, and under any interleaving of
   concurrent calls.
2. **Reference.** On the same inputs, `eval_into` equals today's
   `debug_eval_rhs_into` bitwise, with a taped scratch whose forcing holds the
   same data. Same kernels, same order: this is an API move, not a numerics
   change.
3. **Output.** `du` is zeroed and then written, as `debug_eval_rhs_into` does
   (`driver.rs:621-623`). Slots of states with no `D(·)` equation come back 0.
4. **Workspace.** A workspace may be used with only one program, and with one
   call at a time (`&mut`). It may move between threads (`Send`). Using a
   workspace with a different program is `RhsError::Shape`, not undefined
   behaviour.
5. **Section caching.** The tape's CONST/SEGMENT/CONTINUOUS split
   (`tape/exec/mod.rs:572-591`) is kept and keyed per workspace:
   - CONST re-runs when the bitwise hash of `p` changes (`params_gen`, `:482`),
     as today.
   - SEGMENT re-runs when `forcing.epoch()` differs from the epoch the
     workspace last primed.
   - Every forcing and array-parameter load is placed in SEGMENT: the program is
     built with every forcing name in the discrete set.
   - Epochs come from a process-wide `AtomicU64`. Two sets never share one, so
     a stale read is impossible by construction.
6. **No allocation** in steady state, preserving the zero-allocation property
   (ess-mro) that the counting-allocator test pins.
7. **Errors.** A fault latches into the workspace and is returned as
   `RhsError::Fault`, with `du` unspecified. The thread-local latch
   (`eval.rs:1944`) is not touched on this path.
8. **No internal threads.** No call spawns, joins or uses a global pool.

### 3. Forcing data and refresh

- A `ForcingSet` holds one buffer per `forcing_slots()` entry, column-major
  (first index fastest), the same convention as the state. Today's buffer is an
  `ndarray::ArrayD` read row-major (`tape/ir.rs` `LoadForcing` docs). The load
  already copies into the slab, so it can transpose on the way at no extra cost.
- Refresh is a swap, not a mutation:
  - At an epoch boundary (architecture §7.5) the integrator builds the next set
    from the provider's records, sharing unchanged buffers.
  - It publishes the new set, for example through an `ArcSwap` it owns.
  - Readers that hold the old set finish on it, and it is freed when the last
    one drops.
  - Async prefetch (U11) fills the next set's buffers in the background.
- Precision of forcing values is U5's rule: round on write.
- `ArrayParameter` slots carry gridded parameters (D5). They are opt-in at
  construction (`ProblemOptions::runtime_array_parameters`). Values change only
  by building a new set.

### 4. Parameter vector ABI

- `p: &[f64]` has length `parameters().len()`.
- It is in the order the tape's `Operand::Param(u32)` already indexes
  (`tape/ir.rs:117`; `params_len`, `:1358`): sorted variable-name order
  (`compile.rs:1548-1549`). Names are namespaced on the flattened path.
- `parameter_vector` is the strict resolver (`build_param_vec`,
  `driver.rs:1011`). The lenient `debug_resolve_params`, which substitutes `0.0`
  for a missing value (`driver.rs:595-609`), stays a debug seam.
- The same order is what `vjp_p` accumulates into (architecture §10).

### 5. Ownership of the tape and the slab

- **The tape.**
  - An `RhsProgram` holds exactly one `Arc<TapeProgram>`, built at construction
    for a frozen forcing signature (slot names and shapes).
  - A different signature is a different `RhsProgram`.
  - `ArrayCompiled`'s `TapeCache` keeps serving diffsol. Its entries become
    `Arc` so an `RhsProgram` can share one rather than rebuild.
- **The slab.**
  - The slab and its companions move from `RhsScratch`/`TapeCtx` into
    `RhsWorkspace`: `TapeExec`'s slab, row-major state mirror, fused registers,
    parked exports, epochs and fault latch (`tape/exec/mod.rs:151-205`).
  - The fused scratch's raw pointers are per-call addresses, rewritten before
    every use. They become offsets, or the workspace gets a documented
    `unsafe impl Send`.
- **Shared data.** `ConstArrayScope`, the precision table and the observed rule
  list become `Arc`. With every rule taped, the program does not need the rule
  bodies at run time at all.

### 6. Composition with U3 and U6

U3's operator, tile and lane variants are built from an `RhsProgram`. Each is
its own `Send + Sync` program that takes an `RhsWorkspace`. Each follows §2
except for the output rule: a tile writes only its owned rows of `du` and
leaves every other element untouched, so tiles can share one `du` (U3 §4).

```rust
impl RhsProgram {
    pub fn select(&self, sel: &RhsSelection) -> Result<RhsProgram, SimulateError>;   // U3 §2
    pub fn tile(&self, dec: &Decomposition, b: &TileBox)
        -> Result<TileProgram, SimulateError>;                                       // U3 §5
    pub fn lanes(&self, kind: LaneKind, capacity: usize)
        -> Result<LaneProgram, SimulateError>;                                       // U3 §6
}
```

U6's `Kernel` adopts §2 and the `RhsWorkspace` type for its scratch. The
integrator holds one workspace per (program, executor thread) pair, a few MB
each (the slab size is in `TapeBuildReport`).

### 7. Julia

The same split, for D1 parity:

```julia
prog = rhs_program(prob)                 # immutable; shares the generated code
ws   = rhs_workspace(prog)               # _CSECache, _AccScratch, _ObsExtVec, epochs
fs   = forcing_set(prog, bufs)           # NamedTuple of arrays + a unique epoch
rhs!(du, prog, ws, u, p, t, fs)          # reentrant across distinct `ws`
```

- The generated kernels already touch only locals ("GENERATED CODE NEVER
  TOUCHES INTERPRETER STATE", `codegen_kernel.jl:48-50`). Only their caches move.
- The global `_FORCING_EPOCH` becomes the set's epoch.
- Polyester chunking is off inside `rhs!`.
- `p` follows the existing vector seam `_read_param` (`compile.jl:1402-1447`),
  in `param_names` order.

**Python** follows the same contract when its native implementation lands
(D1), as a third implementation, not bindings.

## What this is not

- **Not internal parallelism.** A parallel RHS for the `global` operators may
  come later. It is not this ask.
- **Not XLA.** A Rust `xla` program is not `Send + Sync` by this RFC.
- **Not a change to `solve`.** The in-crate diffsol path is re-hosted on
  `RhsProgram`, with one workspace inside its `RefCell`, so there is one code
  path. Its behaviour and conformance are unchanged.
- **Not precision semantics** (U5), **not tiles** (U3), **not observed
  kernels** (U6).

## Conformance and tests

1. **Static.** `assert_impl_all!(RhsProgram: Send, Sync, Clone)`,
   `assert_impl_all!(ForcingSet: Send, Sync)` and
   `assert_impl_all!(RhsWorkspace: Send)` compile in CI.
2. **Equivalence.** On every compiled-RHS fixture, `eval_into` equals
   `debug_eval_rhs_into` with a taped scratch, bitwise.
3. **Concurrency.**
   - 16 threads × 1,000 calls at random `(u, p, t)` drawn from a fixed seed,
     with randomized thread assignment, equal the serial results bitwise.
   - The same run under ThreadSanitizer in a nightly CI leg.
4. **History independence.** A workspace that has evaluated other `p` and
   other forcing sets gives the same bits as a fresh one.
5. **Stale-forcing negative control.**
   - Change one buffer, keep `p`, call again: the result must change.
   - The same test against `debug_new_scratch_taped` must reproduce today's
     stale result, which proves the test has teeth.
6. **Faults.** A fixture that faults returns `RhsError::Fault` on its thread.
   A concurrent call on another thread is unaffected.
7. **Allocation.** The counting-allocator test passes on `eval_into` after
   warm-up.
8. **Surface.** `api-surface.json` gains the types and functions. The
   surface tests check both directions.

## Work breakdown

1. `Arc` for `TapeProgram`, `ConstArrayScope`, `VarPrecisions` and the
   observed rule list. `TapeCache` entries hold `Arc`.
2. Extract `RhsProgram` from `ArrayCompiled` for `RuntimeMode::Native`; refuse
   programs with fallbacks.
3. `RhsWorkspace` from `TapeExec` plus the fault latch; per-workspace epochs.
4. `ForcingSet` and column-major `LoadForcing`; route every forcing name to
   SEGMENT.
5. `ArrayParameter` slots behind the opt-in.
6. Re-host `run_one_segment`'s closures (`driver.rs:1984`, `:2022`) on the new
   type.
7. Julia: `rhs_program`, `rhs_workspace`, `forcing_set`, `rhs!`.
8. Tests above; API_SPEC §5.8 text.

## Alternatives considered

- **`unsafe impl Send + Sync for ArrayCompiled`.** Unsound: `RefCell`, `Cell`
  and `Rc` inside it are mutated through `&self`.
- **A `Mutex` around the RHS.** Sound, but it serializes the evaluations the
  integrator exists to parallelize.
- **One `ArrayCompiled` per thread.** Its `Rc`s still cannot be sent. Building
  one per thread repeats the build (minutes at CONUS).
- **Make `ArrayCompiled` itself `Sync`** (`OnceLock`, `Mutex`, `Arc`
  throughout). Possible, but it touches construction, field ICs and diffsol for
  no gain on the hot path.
- **A forcing `RwLock` with in-place refresh.** It keeps the "who bumps the
  epoch" problem, and it blocks readers during a refresh that is 98% of 0.25°
  forward wall time (architecture §7.5).

## Open questions

1. Is `from_problem` right, or should the entry be `rhs_program(input, opts)`
   with no Problem in between?
2. Column-major forcing (proposed) or row-major? The tape reads row-major
   `ArrayD` today. Julia's buffers are column-major Julia arrays.
3. Should `compiler = interpreter` be allowed, as a thread-safe oracle for the
   conformance tier? That needs `Arc` rule bodies and per-workspace overlay
   state (`vectorized.rs:41`, `eval.rs:1785-1935`).
4. Should observed outputs (`Export`) be readable from a workspace after a
   call, or only through U6's kernels?
