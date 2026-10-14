# EarthSciIntegrator architecture (draft v0.1, 2026-10-13)

**Status:** first full draft, for review. Decisions D1–D28 and their tradeoff analyses are in [`decisions.md`](decisions.md); the capability survey is in [`survey.md`](survey.md); upstream RFC drafts are in [`rfcs/`](rfcs/).

## 0. Scope

**Goal.** One integrator library that runs the full 3-D `.esm` models: reseact (CTM), eqice (PISM) and EqWeFiC (WRF, WRF-Chem, fire). It must support:
- explicit, implicit, IMEX, legacy fixed-step and split-explicit schemes;
- nested nonlinear solves and index-1 DAEs;
- per-locality partitioning over threads and MPI;
- full-simulation discrete adjoints with respect to parameters, initial state, gridded fields and forcing.

It is specified once and implemented natively in Julia and in Rust (D1).

**Not in v1:**
- GPU in Rust (Julia has an XLA GPU path in v1, §7.8; D15);
- PETSc (D12);
- continuous adjoints (D4);
- continuous root-finding events (§9);
- multi-domain nesting (not planned for now; D23);
- Python, which comes later as a third native implementation (D1).

**Inspiration from DifferentialEquations.jl:**
- the problem / algorithm / solve separation;
- tableau-generic method families;
- the `init` / `step!` / `solve!` integrator interface;
- PI controllers;
- callbacks;
- retcodes;
- `remake`.

**What we do differently:**
- structure-aware linear algebra by default, never a dense Jacobian;
- operator splitting as a first-class schedule tree whose callbacks and tolerances actually propagate;
- discrete adjoints of the step sequence actually realized;
- partitioning owned by the integrator.

## 1. Glossary

| Term | Meaning |
|---|---|
| **operator** | A named group of additive RHS terms, or an algebraic residual, map, step limit or condition, built by the planner from the flattened AST. |
| **leaf** | A schedule node that advances one or more operators with one algorithm. |
| **macro step** | One step of the outermost schedule, `[t, t+Δ]`. |
| **locality** | Which cells an operator reads relative to the cell it writes: `pointwise`, `column`, `stencil(r)` or `global`. |
| **rank / tile / shard** | The partition hierarchy: an MPI process, a geographic sub-block with halos (thread level), and a set of cells integrated together by a pointwise or column leaf. |
| **lane** | One cell (or column) slot in a shard's struct-of-arrays buffer. Batched kernels vectorize across lanes. |
| **epoch** | An interval over which forcing buffers are constant, bounded by provider refresh times. |
| **tape** | The record of a forward run that the adjoint replays: accepted `(t, dt)` per leaf per shard, discrete decisions, epochs. |
| **plan** | The planner's output artifact: a resolved run config plus derived `.esm` documents. |

## 2. Layers

```
 model.esm ──(EarthSciAST flatten, with per-term provenance)──► flattened AST
     │ solver: {"external": "model.run.json"}
     ▼
 run config ──────────────────────────────────────────────┐
                                                          ▼
 ┌──────────────────────────── PLANNER (Julia | Rust) ──────────────────────────┐
 │ operators ← provenance + selectors   locality ← offset analysis              │
 │ structure ← EarthSciASTDiff bands    strategy ← rules (§3.5) where "auto"    │
 │ derived docs: per-operator RHS, J/VJP/HVP band models, residuals, maps,      │
 │               limits, conditions     partition plan                          │
 └───────────────────────────────┬──────────────────────────────────────────────┘
                                 ▼  plan.json + derived/*.esm   ◄── conformance goldens
 ┌──────────────────────────── RUNTIME (Julia | Rust) ──────────────────────────┐
 │ Kernels      native CPU (Julia f!, Rust tape) · XLA programs on GPU (Julia)   │
 │ Partition    rank → tile → shard; halo exchange; executors (threads, MPI)    │
 │ Schedule     interpreter for combinators (§4)                                │
 │ Methods      ERK/SSP · Rosenbrock · SDIRK/ESDIRK · BDF · ARK · legacy        │
 │ Solvers      batched block LU · column banded · Krylov+bjacobi/ASM · Newton  │
 │ Control      PI/PID · step limits · tstops · domain checks · retcodes        │
 │ Events       preset/periodic/condition · maps · forcing epochs · callbacks   │
 │ Adjoint      tape · checkpoints · method-level adjoint steps · IFT           │
 │ I/O          EarthSciIO sinks · restart                                      │
 └──────────────────────────────────────────────────────────────────────────────┘
```

The planner and the runtime are separate on purpose:
- The plan is an inspectable, editable, cacheable artifact.
- It is the seam where the Julia and Rust implementations are checked against each other (D8).
- A user can pin a plan, overriding inference, by pointing `solver.external` at a resolved run config.

## 3. Planner

### 3.1 Inputs

- **The flattened AST from EarthSciAST,** with:
  - per-term provenance (upstream ask U2);
  - variable classification (`ode_states`, `observed_unknowns`, `algebraic_unknowns`);
  - parameter classes and the cadence partition (CONST / DISCRETE / CONTINUOUS).
- **The run config** (§4), which may leave any choice as `"auto"`.

### 3.2 Passes

1. **Operator formation.**
   - Split each `D(x,t)` right-hand side into additive terms; the EarthSciASTSplitter algorithm, generalized.
   - Assign each term to an operator by provenance (component path) and the run config's selectors.
   - Unselected terms fall into default operators grouped by locality.
   - Algebraic equations become `algebraic` operators. Event conditions, affects and step maps become `condition` and `map` operators. Run-config step-limit expressions become `limit` operators.
2. **Locality analysis.** For every state read in a term, find its index offset relative to the written cell:
   - all offsets zero → `pointwise`;
   - nonzero only on the vertical axis → `column`;
   - bounded horizontal offsets → `stencil(r)`, with radius per axis;
   - an unbounded horizontal `faq` contraction, or a global map → `global`.

   Column contractions such as reseact's PBL mean are `column`, not `pointwise`. That fixes the `contracts_space` special case in reseact's `split_common.jl`.
3. **DOF sets.** Each operator records the variables it writes and reads, so its kernel outputs only its own rows. This removes the Splitter's full-length parts.
4. **Structure.** Run EarthSciASTDiff on each operator to get its Jacobian bands. From the bands, derive:
   - the block pattern per cell (n×n, plus sparsity and a Markowitz LU plan);
   - column bandwidth;
   - the stencil footprint;
   - a Gershgorin spectral-radius bound, as an expression usable as a step limit.
5. **Strategy inference** (§3.5), only where the run config says `"auto"`.
6. **Derived documents**, one `.esm` per artifact:
   - per-operator RHS;
   - Jacobian band model (it also serves `Jᵀ·λ`, because bands transpose at assembly);
   - HVP band model, needed by Rosenbrock adjoints (U8);
   - ∂f/∂p and ∂f/∂(forcing) band models (U7);
   - algebraic residual models;
   - map, limit and condition models.

   All are compiled by the stock EarthSciAST compiler.
7. **Partition plan.** Rank and tile decomposition, halo widths per operator (the radius from pass 2), and shard sizing for pointwise and column leaves.
8. **Validation and emission.** Checks:
   - Σ (operator RHS) = flat RHS at sampled states, within tolerance;
   - band models against finite differences at sampled non-default states (reseact: "never FD-validate at the default IC");
   - a non-empty plan is required, because reseact's band model once went silently empty.

   Then write `plan.json` (canonical JSON) and `derived/*.esm`.

### 3.3 Determinism of the plan

The same `.esm`, run config and planner spec version must produce byte-identical `plan.json` and derived documents in Julia and Rust. These are golden-gated, like EarthSciASTDiff's byte-identical goldens.

### 3.4 Picard is a frozen-observed Jacobian

A run-config `freeze: ["ssa.nuH"]` makes pass 4 treat those observeds as constants when differentiating. This gives PISM's Picard operator (frozen effective viscosity) as an ordinary band model, so no model-specific Picard code is needed.

### 3.5 Strategy inference rules (when "auto")

| Operator structure | Default leaf |
|---|---|
| pointwise; stiff (from `reaction_systems`, `solver.stiffness: high`, or the Gershgorin bound × Δ ≫ 1) | Rosenbrock (`Rodas4`, or `Rosenbrock23` at loose tolerance); batched block LU (sparse-unrolled if n > 16); per-shard dt |
| pointwise; non-stiff | SSPRK43 / Tsit5 |
| column; Gershgorin × Δ ≫ 1 | SDIRK / ESDIRK with batched banded or block-tridiagonal solves |
| stencil | SSPRK43; step limit = the Gershgorin bound (CFL-like); global dt per leaf |
| stencil + column with stiff vertical terms | ARK IMEX: horizontal explicit, vertical implicit |
| algebraic | nested Newton–Krylov (`solve` node); bjacobi-ILU(0); warm start from the previous value |
| default composition | `sequence` in locality order stencil → column → pointwise; the `solver.splitting` hint upgrades it to Strang |

## 4. Run config and schedule language

### 4.1 Hook

The `.esm` gets `"solver": {"external": "reseact.run.json"}` (U1). The run config is JSON with its own JSON Schema, versioned independently of esm, under `spec/`.

### 4.2 Top-level keys

| Key | Contents |
|---|---|
| `esm_integrator` | spec version |
| `model` | path to the `.esm` |
| `operators` | name → selector (`components`, `variables`, `locality`, `terms`) plus options (`freeze`, `bind_dt`) |
| `schedule` | combinator tree (§4.3) |
| `step` | macro-step controller: fixed `dt`, or `limits` (operator names, expressions, `{"max": …}`), `hit` (tstops sources), `resolution` |
| `partition` | ranks (grid of horizontal blocks), tiles per rank, shards per leaf |
| `precision` | precision roles per leaf (§8) |
| `adjoint` | functional, `wrt`, checkpoint policy |
| `output`, `restart` | sinks, saveat, accumulators, checkpoint cadence |

### 4.3 Combinators

| Node | Semantics on `[t, t+Δ]` |
|---|---|
| `leaf` | Advance the sum of its operators with `alg`. Adaptive substeps are allowed inside. Carries `control`, `linsolve`, `jacobian`, `domain`, `partition`, `bind_dt`. |
| `sequence` | Children in order, each over `Δ`. This is Lie–Trotter, and PISM's ordered step when children include maps and solves. |
| `strang` | `a(Δ/2) b(Δ) a(Δ/2)`, generalized symmetrically for n children. |
| `subcycle` | The child `n` times with `Δ/n`, or adaptively under its own `limits` (hydrology, fire `tbound`). |
| `every` | The child runs only when `t` crosses multiples of `period` (+ `phase`). With `hold: "tendency"`, its tendency is evaluated at the crossing and added as a constant to sibling leaves until the next crossing (WRF `radt`). |
| `coupled` | Monolithic integration of the sum of its operators (no splitting error). `imex: {explicit: [...], implicit: [...]}` selects ARK. If any algebraic operators are included, it becomes an index-1 DAE (§6.4). |
| `frozen` | The operator's tendency is evaluated once at the start of the enclosing step and held across all stages (WRF `rk_addtend_dry`). |
| `split_explicit` | WRF RK3 (Wicker–Skamarock: Δ/3, Δ/2, Δ) on `slow`, with `fast` integrated by forward–backward substeps (vertically implicit, `epssm` off-centring) inside each stage. *Spec in v1, implementation in M7.* |
| `map` | `u ← M(u, p, t, Δ)`: calving, mask consistency, level-set reinit, iceberg removal. |
| `solve` | Nested nonlinear solve for algebraic unknowns with everything else frozen (§6.3). |

**`bind_dt`** sets a model parameter to the step actually taken, e.g. `{"wsm6.dtcld": "leaf"}`. A `leaf` with `alg: "ForwardEuler"`, `dt: "step"` and a bound dt reproduces a legacy WRF/PISM increment *exactly*, because the `.esm` already writes these as `(x_out − x_in)/dt` tendencies.

**Step control** is `Δ = min(error control, limits…, distance to the next tstop)`, then `hit_multiples` and a floor at `resolution`, with a deterministic reason string on ties (PISM bugs.md #11).

### 4.4 Examples (illustrative)

**reseact, the prototype re-hosted (M3).** Use `"dt_scope": "global"` to reproduce the prototype bit for bit, and `"shard"` for speed.

```json
{
  "esm_integrator": "0.1.0",
  "model": "reseact.esm",
  "operators": {
    "transport": {"select": {"locality": ["stencil", "column"]}},
    "chemistry": {"select": {"locality": "pointwise"}}
  },
  "schedule": {"sequence": [
    {"leaf": "transport", "alg": "SSPRK43",
     "control": {"reltol": 1e-4, "abstol": 1e-6, "dt0": 15}},
    {"leaf": "chemistry", "alg": "Rosenbrock23", "jacobian": "symbolic",
     "linsolve": {"block_lu": {"pivot": false}},
     "control": {"reltol": 1e-4, "abstol": 1e-9, "dt0": 0.5, "dt_scope": "global"},
     "domain": {"nonnegative": "all", "on_violation": "reject"}}
  ]},
  "step": {"dt": 300, "hit": ["forcing_refresh", "output"]},
  "partition": {"ranks": [1, 1], "shards": {"chemistry": 8}},
  "adjoint": {
    "functional": {"mean": "O3", "where": {"lev": 1}, "over": "time"},
    "wrt": ["params:numeric", "u0", "fields:NEI2016.*", "forcing:*"],
    "checkpoint": {"every": "macro_step", "store": ["memory", "disk"]}
  }
}
```

**eqice, PISM's `IceModel::step` (M6):**

```json
{"schedule": {"sequence": [
   {"solve": "ssa", "unknowns": ["ssa.u", "ssa.v"],
    "method": {"picard": {"freeze": ["ssa.nuH"]},
               "linsolve": {"gmres": {"precond": "bjacobi_ilu0", "fallback": "asm_lu"}},
               "converged": {"rel_change": "ssa.nuH", "tol": 1e-4}, "maxiter": 300,
               "recover": [{"underrelax": 0.8}, {"scale_param": "ssa.epsilon", "by": 4, "max": 3}]},
    "warm_start": "previous"},
   {"map": "yield_stress"},
   {"leaf": ["age", "energy"], "alg": "IMEXEuler", "imex": {"implicit": "column", "explicit": "stencil"},
    "dt": "accumulated"},
   {"leaf": "flow", "alg": "ForwardEuler"},
   {"map": "enforce_consistency"},
   {"map": "front_retreat"},
   {"leaf": "source", "alg": "ForwardEuler"},
   {"map": "iceberg_removal"},
   {"subcycle": {"leaf": "hydrology", "alg": "ForwardEuler"}, "limits": ["hydrology.cfl", "hydrology.diffusive"]},
   {"every": {"map": "bed_deformation"}, "period": "10 yr"}
 ]},
 "step": {"limits": ["sb.cfl_2d", "energy.cfl_3d", "geometry.diffusivity", {"max": "60 yr"}],
          "hit": ["output", "forcing_records"], "resolution": "1 s"}}
```

**EqWeFiC, the WRF step (M7, sketch):**

```json
{"schedule": {"sequence": [
   {"split_explicit": {"slow": "dynamics", "fast": "acoustic", "alg": "WRF_RK3", "substeps": 6,
                       "fast_alg": {"forward_backward": {"vertical": "implicit", "epssm": 0.1}},
                       "frozen": [{"every": "radiation", "period": "10 min", "hold": "tendency"},
                                  "surface", "pbl", "cumulus"]}},
   {"leaf": "microphysics", "alg": "ForwardEuler", "dt": "step", "bind_dt": {"wsm6.dtcld": "leaf"}},
   {"every": {"leaf": "chemistry", "alg": "Rodas3", "linsolve": "sparse_block_lu"}, "period": "chemdt"}
 ]},
 "step": {"dt": 40.5},
 "precision": {"default": {"state": "f32"}, "chemistry": {"state": "f64"}}}
```

## 5. Methods

Every family is generic over its tableau. Each provides:
- a step;
- an embedded error estimate;
- optional dense output;
- a declared requirement set (needs J? W? HVP for its adjoint?);
- a **discrete adjoint step** (D4).

| Family | Methods (v1 first, others later) | Adjoint needs |
|---|---|---|
| Explicit RK | Euler, Heun, **SSPRK33, SSPRK43**, RK4, **Tsit5**, BS3, SSPRK104, **WRF_RK3** | `vjp_u`, `vjp_p` at stored or recomputed stages |
| Rosenbrock / ROW | **Rosenbrock23**, ROS2, ROS3P, **Rodas3, Rodas4**, Rodas5P; KPP's Ros2/3/4 and Rodas3/4 for WRF-Chem parity | plus `Wᵀ` solves (reused LU) and `hvp_u` for ∂J/∂u |
| SDIRK / ESDIRK | **BackwardEuler**, TRBDF2, SDIRK2, Kvaerno3/4 | `Wᵀ` solves at converged stages (IFT) |
| BDF | variable order 1–5, fixed-leading-coefficient form; index-1 DAE | tape-fixed coefficients → linear recursion backward |
| ARK IMEX | **IMEXEuler** (PISM's "explicit horizontal at old time"), ARS222, KenCarp3/4 | ERK and SDIRK parts combined |
| Legacy fixed | θ-method (WRF chem mixing), single-step FE/BE | as above |
| Split-explicit | WRF RK3 + forward–backward acoustic (M7) | composition of the above |

### 5.1 Controllers

- PI with OrdinaryDiffEq's defaults per method, I, and PID (Söderlind).
- Fixed dt.
- A non-finite error estimate counts as a reject.
- A `dt_min` guard returns a retcode instead of hanging (the Rust driver lacks this today).
- `maxiters` is per leaf.

### 5.2 Error norms

- Hairer RMS over the leaf's DOFs, always accumulated in f64 (D13), in a fixed order (D10).
- `dt_scope` is one of `global`, `shard` or `cell`. With `shard` or `cell` the norm is per shard or per cell.
- `cell` runs on lane pools (§7.6). Its results do not depend on how cells are sharded.

### 5.3 Domain enforcement

`domain: {nonnegative: …, on_violation: reject | clip}`. The default is reject, checked on accepted states and optionally on Rosenbrock stage values (the pre-dawn NO failure). `clip` is still supported; its adjoint is a mask.

## 6. Linear and nonlinear solvers

### 6.1 W matrices

W is kept structured and never assembled dense:

| `linsolve` | Structure | Implementation |
|---|---|---|
| `block_lu` | per-cell n×n | Lane-SoA batched LU. Each LU operation is vectorized across all lanes of a shard. Pivoting is optional (reseact uses none). |
| `sparse_block_lu` | per-cell sparse n×n | KPP-style unrolled elimination from EarthSciASTDiff's Markowitz plan. It executes as a short op tape over lanes in both languages (a generated function in Julia). |
| `banded` / `block_tridiagonal` | per column | Batched Thomas / block-Thomas across the columns of a shard. |
| `krylov` | global sparse | GMRES(m) and BiCGStab, distributed (D12). Preconditioners: block-Jacobi ILU(0); ASM (overlap 1, LU subdomains); column-block-Jacobi for 3-D implicit operators. |
| `matrix_free` | any | `J·v` from band models, with FD as a fallback. |

**Transposed solves for the adjoint** reuse the forward factorizations: LUᵀ, and Krylov on `Jᵀ` with the transposed preconditioner.

### 6.2 Nonlinear solvers

- Newton with the structured J; inexact Newton–Krylov (Eisenstat–Walker); Picard (§3.4).
- Pseudo-transient continuation (PTC), which replaces eqice's hand-written pseudo-time relaxation.
- Backtracking line search; Anderson acceleration.
- Convergence criteria on the residual or on the relative change of a named observed (PISM's ‖ΔνH‖/‖νH‖).

### 6.3 `solve` nodes (nested algebraic, D3a)

- Warm start from the previous solution.
- `recover` ladder: under-relaxation, scaling a regularization parameter, reverting to the warm start.
- The converged z is stored on the tape for the adjoint.
- **Adjoint (IFT):** one transposed solve, `g_zᵀ μ = λ_z`, per step.

### 6.4 Index-1 DAE (D3b)

- Semi-explicit form: `u' = f(u,z)`, `0 = g(u,z)`. `M = diag(I,0)` is implicit and never assembled.
- Methods: BDF, stiffly accurate Rodas, stiffly accurate SDIRK.
- Consistent initialization solves `g(u0, z) = 0` at t0.
- The Newton system is block 2×2. It is solved by Krylov with a field-split preconditioner: the column or block preconditioner for u, and the SSA preconditioner for z.

## 7. Partitioning and execution (D6, D10, D11)

### 7.1 Decomposition

Horizontal only, so columns stay whole:
- Ranks form a 2-D block grid.
- Tiles subdivide a rank for threads.
- Staggered face variables are owned by the low-side cell.
- Halo width is the maximum stencil radius per axis over the operators that rank runs.

### 7.2 Per leaf

| Leaf locality | Execution |
|---|---|
| stencil | Exchange halos of the variables the operator reads (width = its radius) **before each stage**, then run the range-restricted RHS per tile (U3). In shared memory the exchange is a barrier. |
| column | Shards of whole columns gathered into lanes. No communication. |
| pointwise | reseact-style shards: lanes gathered from any of the rank's cells, across levels and columns (§7.7), integrated with per-cell (lane pool) or per-shard dt, then scattered back. No communication until the leaf ends. Re-sharded as stiffness changes (§7.6). |
| global | Distributed Krylov / Newton and deterministic reductions. |

### 7.3 Executor contract

Taken from reseact's `shard_exec.jl`:
- Executors are stateless.
- Inputs: lane state, forcing slice, parameters.
- Outputs: new state, partial error sums, per-shard `dJ/dp`.
- Reductions run in shard order, then rank order.

v1 executors are threads (dynamic assignment of fixed-content shards to threads, which keeps D10 intact) and MPI ranks (Julia MPI.jl, Rust rsmpi).

### 7.4 Deterministic reductions (D10)

- Partial sums are gathered in rank order, reduced at the root, then broadcast. `MPI_Allreduce` gives no ordering guarantee.
- Max norms and minimum dt are exact under any order.

### 7.5 Forcing

- Each rank reads its own window (EarthSciIO index-range and projection pushdown).
- Refresh happens at epoch boundaries with half-open `(t0, tf]` semantics.
- The next epoch is prefetched asynchronously, because refresh was 98% of the 0.25° forward wall time (U11).

### 7.6 Dynamic partitioning as stiffness changes

Yes. Partitions can change during a run. What may change, and on what signal, depends on whether the change can alter results:

| What changes | Alters results? | Allowed signal | When |
|---|---|---|---|
| Shard **assignment**: which executor runs which shard | No | anything, including wall-clock time and work stealing | any time |
| Shard **composition**: which cells share one controller | Yes, under `dt_scope: shard` | deterministic signals only: each cell's accepted + rejected steps and Newton iterations in the previous macro step, the Gershgorin stiffness estimate, solar zenith angle, level | macro-step boundaries |
| Rank and tile **geometry** | Yes (halo and reduction order) | a deterministic cost model, e.g. ice-free cells in PISM or the active fire band | checkpoint boundaries, by redistribution as on restart (after v1) |

**Lane pools make per-cell dt cheap.**
- Each SIMD lane integrates one cell with its own controller.
- When a lane finishes its interval, it is refilled from a work queue and the batch compacts.
- A cell's step sequence then no longer depends on which cells share its batch. Composition stops mattering, and chemistry becomes bitwise invariant to shard, thread and rank count.
- That holds only if every lane runs the same instruction sequence: pad remainder lanes instead of running a scalar tail, and fix FMA contraction.
- Load balancing can then use any signal, wall-clock included.

reseact's per-cell and per-bucket attempts lost to a roughly 3 ms per-call floor in XLA:CPU. Native CPU (D11) has no such floor, so the measured 6× gap is the expected win.

**Cross-rank migration of pointwise work.** A lane is self-contained: its state, gathered forcing, geometry and parameters. An overloaded rank can ship lanes to an idle one and receive the results back. This addresses day/night imbalance between ranks. Planned after M5 (M5b).

**Adjoint impact (§10).**
- Under D10, replaying a macro step from its checkpoint with the controller *on* reproduces the forward step sequence bitwise. So the tape stores per-cell step counts, or a hash, to verify the replay, not per-cell dt sequences.
- Checkpoints carry each cell's dt and controller history.
- With `dt_scope: shard` and dynamic composition, the composition used in each macro step is recorded when it changes.

### 7.7 Splitting chemistry columns vertically

Yes, and the design already allows it.
- Chemistry is pointwise, so its shards are arbitrary sets of cells, not columns. "Columns stay whole" (§7.1) applies only to the geographic rank and tile decomposition, so that column operators (PBL, enthalpy, radiation) stay local.
- Within a rank, chemistry lanes can mix any levels and columns. Gathering by level band costs the same as gathering by column: one index-list gather from the species-major arrays.
- **Recommended bucketing key:** the cell's cost in the previous macro step. Level band and solar zenith angle predict cost at cold start and at the terminator.
- With per-cell dt (lane pools), grouping only affects SIMD efficiency: lanes with similar step counts finish together. With per-shard dt, grouping by stiffness directly cuts wasted steps.

**Exception.** If a run config couples chemistry with vertical mixing in one leaf (`coupled` over chemistry and PBL diffusion, as some CTMs do), that leaf's locality is `column`. It cannot be split vertically without changing the numerics.

### 7.8 GPU via XLA (Julia, v1; D15)

**Shape of the GPU path.**
- Each leaf step (RHS, stages, linear solves, error norm, controller) is lowered to **one XLA program**, traced by Reactant from the same kernel-layer code the native CPU path runs.
- The RHS and band models come from EarthSciAST's direct StableHLO emitter (U13), so there is no buffer handoff between XLA and CUDA.
- The host is touched only at macro-step and epoch boundaries, for MPI, and for output.

**Rules on the XLA path:**
- **Static shapes.** Lane pools have a fixed capacity and refill in masked chunks. Shard composition is a runtime gather-index input, so re-sharding never recompiles. Programs are per tile, which bounds compile time.
- **Determinism (D10).** XLA's deterministic-ops mode is on. Our reductions use fixed trees. CPU and GPU agree to tolerance, not bitwise, because XLA's `exp`/`log` are not libm's.
- **Parallelism.** Multiple GPUs in one node use XLA collectives (`collective-permute` for halos, all-reduce for norms and Krylov dot products). Across nodes, MPI exchanges halos on the host between stage programs; PJRT's distributed runtime is an alternative to evaluate.
- **Adjoint.** The method-level adjoint kernels lower the same way. Checkpoints live in device memory and spill to host. The Enzyme-on-step oracle stays on CPU.

**Precision.** The hardware target is unknown, so precision is a run-config choice:
- On FP64-strong GPUs (A100/H100), the state stays f64.
- On FP64-weak GPUs, use f32 state for dynamics and transport and f16/bf16 W (§8).
- Executors are per leaf, so chemistry shards can stay on CPU cores while transport runs on the GPU.

## 8. Mixed precision, including float16

Partly. Low precision is safe where its errors are self-correcting (linear algebra inside iterations, preconditioners, stored tables). It is not safe for the state. Adopted as D18, which amends D13 with precision **roles** per leaf:

| Role | Allowed | Default |
|---|---|---|
| time, dt, error norms, reductions, adjoint accumulators | f64 only | f64 |
| state `u` and stage vectors | f32, f64 | the model's `element_type` |
| RHS evaluation | f32, f64 (≥ state) | state |
| Jacobian / W storage | f16, bf16, f32, f64 | RHS |
| LU factors, preconditioners | f16, bf16, f32, f64 | RHS |
| forcing and lookup-table storage | f16, bf16, f32, f64 (upcast on load) | source |

**Why the state cannot be 16-bit.**
- **f16:** ranges from 6e-8 (subnormal) to 65504 with about 3.3 significant digits. reseact's species span 1e-26 to 1e-13 (NO reaches 1e-26 at night).
- **bf16:** has f32's range but only about 2.4 digits, so small increments vanish (u + du rounds back to u).
- **Flux-form mass conservation** needs consistent sums.

**Where 16-bit helps:**
1. **W factorizations inside Newton (SDIRK, BDF) or Rosenbrock-W methods** (Rosenbrock23, ROS34PW). An inexact W only slows Newton or keeps the W-method's order. Classical Rosenbrock methods (Rodas) assume an exact J, so the planner refuses a low-precision W with them unless overridden.
2. **Mixed-precision iterative refinement:** factor in f16/f32, compute residuals at state precision, refine 1–3 times. This recovers full accuracy when W is well conditioned, and `I − γhJ` is near identity at chemistry step sizes.
3. **Krylov preconditioners** (ILU, block Jacobi): their accuracy affects only iteration counts.
4. **Memory:** CBM-Z block Jacobians on d05 drop from about 220 GB at f64 to about 55 GB at f16 *(est.)*.

**Requirements.**
- f16 needs diagonal equilibration (scale W by the error weights `atol + rtol·|u|`) before casting, to avoid overflow and underflow. bf16 can skip it.
- On v1's CPUs, native 16-bit arithmetic exists only on recent hardware (AVX512-FP16, AMX-BF16, ARM SVE). Elsewhere it is storage-only, converted to f32 for arithmetic, so the gain is memory and bandwidth. Large speedups wait for GPUs.
- **Julia:** `Float16` is native; `BFloat16s.jl` adds bf16.
- **Rust:** the `half` crate provides both on stable; the primitive `f16` is not yet stable, as far as I know.

**Adjoint and determinism.**
- The adjoint transposes the low-precision factors and runs the same refinement. Gradients stay consistent with the computed trajectory to the refinement tolerance, not to 1e-11.
- Precision conversions are deterministic, so D10 is unaffected.

## 9. Events, maps and callbacks (v1)

- **Discrete events:** preset times, periodic, and condition-at-step-end. These cover forcing refresh, output, checkpoints, `every`, and `.esm` `discrete_events`.
- **Maps** inside the schedule.
- **Callbacks are propagated into every leaf and substep.** This is the bug that broke OrdinaryDiffEqOperatorSplitting.
- **Continuous root-finding events are not in v1.** No v1 model needs them; the fire ignition time is a per-cell interpolation map.
- **Firing decisions are recorded on the tape** and held fixed in the adjoint.

## 10. Adjoint (D4, D5)

**Functional.**
- `J = Σ_k φ(u(t_k)) + ∫ ψ(u,t) dt`.
- φ and ψ are named observeds or reductions given in the run config.

**Forward pass.**
- Record the tape:
  - accepted `(t, dt)` per leaf per shard (for `dt_scope: cell`, only per-cell step counts or a hash);
  - shard compositions, when they change;
  - converged `z` from `solve` nodes;
  - map masks, event firings and epochs.
- Checkpoint at macro-step boundaries: state plus each cell's dt and controller history. Storage is an in-memory ring plus disk, with an optional revolve schedule for 0.25°-class runs.

**Backward pass, per macro step in reverse:**
1. Restore the checkpoint.
2. Replay forward, storing stage states per shard. Because of D10, the controller can stay on: it reproduces the forward sequence bitwise, which the recorded counts or hash verify. Where the tape holds a `(t, dt)` sequence, the controller can be turned off instead.
3. Run the family adjoint steps in reverse schedule order.

Under D10 the replay is **bitwise** equal to the forward pass, so the replay-versus-next-checkpoint check becomes an equality test (reseact had to use 1e-6).

**Primitives.**

| Primitive | Source |
|---|---|
| `vjp_u`, `vjp_p` (including array parameters), `vjp_forcing`, `hvp_u` | EarthSciASTDiff band models, in both languages |
| `Wᵀ` solves | Reused factorizations (§6.1) |

Julia additionally has Enzyme on the RHS as a fallback for primitives, and Enzyme on a whole step as the oracle.

**Accumulation.**
- Parameters: per-shard partials, reduced in fixed order.
- u0: `λ(t0)`.
- Gridded fields: `vjp_p` into array parameters.
- Forcing: per-epoch buffer gradients. The model's own time-blend expression appears in the forcing band model, so the chain rule into both bracketing records is automatic.
- Forcing and field gradients stream to EarthSciIO sinks.

**Validation gates.**
- Dot-product identity per leaf step.
- FD on the frozen tape.
- Hand adjoint against the Enzyme-on-step oracle (Julia).
- Bitwise replay.
- Julia ↔ Rust gradient agreement at tolerance.

## 11. Output and restart

- **Output:**
  - EarthSciIO streaming sinks with per-rank `region=` writes (U12).
  - `saveat` uses step-hitting or method dense output.
  - Accumulators, such as precipitation totals, are auxiliary integrated states added by the planner.
- **Restart:**
  - State, controller state (dt, PI history), `solve` warm starts (SSA velocity), forcing epoch and tape position.
  - Restart is bitwise for the same decomposition (D10).

## 12. Spec, conformance and repository layout

```
spec/                       integrator spec (markdown), run-config + plan JSON Schemas
julia/EarthSciIntegrator.jl/
rust/                       crates: earthsci-integrator (core), -planner, -solvers, -mpi
tests/conformance/          small .esm + run configs; goldens (plans, trajectories, gradients); harness
docs/                       survey, decisions, architecture
```

**Conformance tiers:**
1. **Planner goldens:** byte-identical `plan.json` and derived documents.
2. **Methods:** convergence order on reference problems; dot-product adjoint tests per family.
3. **Cross-language trajectories:** Julia ↔ Rust at tolerance; bitwise within each language for a given decomposition.
4. **Model gates:**
   - reseact against the prototype;
   - eqice test F against the C++ dumps;
   - the EqWeFiC 24 h single-column run (SCM).

## 13. Upstream asks

| # | Repo | Ask | Needed by |
|---|---|---|---|
| U1 | EarthSciAST | `solver.external` in the spec and schema | M2 |
| U2 | EarthSciAST | Per-term provenance (component path) carried through flatten | M2 |
| U3 | EarthSciAST | Range-restricted, DOF-subset RHS over a box with halo reads (Julia `f!`; Rust slab tape over sub-boxes) | M3 (tiles), M5 (MPI) |
| U4 | EarthSciAST | Public, `Send + Sync` Rust RHS entry point (no `Rc`/`RefCell` in the executor) | M3 |
| U5 | EarthSciAST | Float32 integration not refused | M7 |
| U6 | EarthSciAST | Compile documents with events or algebraic equations stripped, and compile observed sets (residuals, conditions, maps) as standalone kernels | M6 |
| U7 | EarthSciASTDiff | ∂f/∂(forcing buffer) bands | M4 (forcing gradients) |
| U8 | EarthSciASTDiff | Second-order (HVP) bands | M4 (Rust Rosenbrock adjoint) |
| U9 | EarthSciASTDiff | Filtered `faq`, non-`+` semirings, recurrences (column sweeps) | M6/M7 |
| U10 | EarthSciASTDiff | Production-rate Rust assembly | M3 |
| U11 | EarthSciIO | Async prefetch; per-rank index-range reads | M5 |
| U12 | EarthSciIO | Rust streaming writer with region writes | M5 |
| U13 | EarthSciAST (Julia) | Direct StableHLO emitter: per-operator, range-restricted RHS and band models with forcing as program inputs, traceable inside an integrator `while` loop | M5g |

## 14. Milestones

| | Deliverable | Gate |
|---|---|---|
| M0 | Spec skeleton: combinator semantics, run-config and plan schemas, the adjoint math per family. Repo scaffold and CI. | Schemas validate the three §4.4 examples |
| M1 | Core in both languages: ERK/SSP, Rosenbrock, SDIRK; PI control; `leaf`/`sequence`/`strang`/`subcycle`/`map`; batched block LU; tape; method adjoints | Conformance tiers 2–3 on toy problems |
| M2 | Planner v1: provenance, locality, DOF sets, band models, strategy rules, plan goldens | Tier 1 on reseact-shaped fixtures |
| M3 | reseact forward, shared memory: shards, lane pools (per-cell dt), dynamic re-sharding, epochs | Matches the prototype's native reference trajectory (`dt_scope: global`); `dt_scope: cell` bitwise invariant to shard and thread count |
| M4 | reseact adjoint: params, then u0, gridded emissions, forcing | Agreement with the prototype's 48 h gradient (target to set; ≤1e-9 rel suggested); exact dot-product and replay |
| M5 | MPI: ranks, halo exchange, deterministic reductions, distributed I/O | reseact at 2×2.5° and 0.25° |
| M5g | Julia XLA GPU: kernel-layer lowering, one program per leaf step, lane pools at fixed capacity, in-node multi-GPU via XLA collectives, device checkpoints | reseact forward and adjoint on GPU match CPU to tolerance, in two configurations (D28): f64 state, and mixed (f32 transport state, 16-bit W, chemistry on CPU). Bitwise run-to-run on one device |
| M5b | Cross-rank lane migration for pointwise leaves (§7.6); f16/bf16 W and preconditioner storage with iterative refinement (§8) | Day/night load balance across ranks at 0.25°; refinement recovers f64 trajectories to tolerance |
| M6 | eqice: `solve` (Newton–Krylov, Picard, PTC), index-1 DAE, IMEX columns, PISM limits, maps | Test F against the C++ dumps |
| M7 | EqWeFiC: SCM 24 h (f32, `every`/`frozen`/`bind_dt`), then `split_explicit` + HEVI once `grids/wrf_arw_c` exists; fire mesh | PLAN.md acceptance criteria |

## 15. Risks

1. **Two of everything (D1, D8).** The planner and MPI are the largest bodies of code, and both are written twice. Goldens keep them honest but do not halve the work.
2. **The adjoint math for BDF, DAEs and `split_explicit`** is the subtlest per-family work. Rosenbrock adjoints in Rust depend on U8.
3. **The upstream critical path:** U3 blocks tiles and MPI, and U7 blocks forcing gradients.
4. **Rust tape performance at 3-D scale is unmeasured** against Julia native.
5. **Matching reseact's gradient.** The arithmetic order differs from Enzyme-on-step, so agreement will be at the roundoff-amplified level, not necessarily 1e-11.
6. **Lane-pool efficiency.** Compaction overhead and SIMD padding could eat into the per-cell dt gain at small shard sizes. Measure at CONUS before making `cell` the default for everything.
7. **The XLA GPU path inherits reseact's record:** minutes-long compiles (hours in pathological cases), Reactant miscompiles and buffer donation, and an XLA:CPU race. Kernel code shared between the native and traced paths may be slower in one of them.

## 16. Former defaults, now decided

Every default in the v0.1 draft was decided on 2026-10-13, as D16–D28 in [`decisions.md`](decisions.md):
- per-cell dt on lane pools for pointwise leaves;
- precision roles with f16/bf16;
- reject on domain violation;
- horizontal-only decomposition;
- Lie stencil → column → pointwise as the `auto` schedule;
- no continuous events in v1;
- no nesting for now;
- JSON run config with its own schema;
- Picard as a frozen-observed Jacobian;
- planner-added accumulators;
- the EarthSciIntegrator names;
- the M4 and M5g gates.
