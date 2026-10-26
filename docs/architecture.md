# EarthSciIntegrator architecture (draft v0.3, 2026-10-14)

**Status:** second draft, after the 2026-10-14 review (reuse audit of the sibling repos and the Julia/Rust ecosystems; decisions D31–D38). Decisions D1–D38 and their tradeoff analyses are in [`decisions.md`](decisions.md); the capability survey, including the verified library inventory (§6 there), is in [`survey.md`](survey.md); upstream RFC drafts are in [`rfcs/`](rfcs/).

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
- BDF, and ARK methods other than `IMEXEuler` (D35);
- the implementation of monolithic index-1 DAE integration, D3b (specified in v1, implemented after M6; D35);
- forward sensitivities in Rust (Julia has them, D36);
- Python, which comes later as a third native implementation (D1).

**Relationship to DifferentialEquations.jl.** Borrowed: the problem / algorithm / solve separation; tableau-generic method families; the `init` / `step!` / `solve!` integrator interface; PI controllers; callbacks; retcodes; `remake`. In Julia this is more than inspiration: the implementation speaks SciMLBase (D33), events are DiffEqCallbacks, Krylov is Krylov.jl and nonlinear solves are NonlinearSolve.jl (D32), and the tableau coefficients and controller constants are OrdinaryDiffEq's own (D31). What is written in-house, and why, is D31.

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
 │ Methods      ERK/SSP · Rosenbrock · SDIRK/ESDIRK · IMEXEuler · legacy        │
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
  - parameter classes and the cadence partition (CONST / DISCRETE / CONTINUOUS);
  - events, parameter `update` rules and algebraic equations, returned as structured objects by U6's strip mode, not refused.
- **The run config** (§4), which may leave any choice as `"auto"`.
- **Precedence.** The run config's `model` key is authoritative for a run. If the `.esm` also carries `solver.external`, the two must point at each other, or the planner refuses. The planner never writes `solver.external` into derived documents. Solver hints on mounted component files are dropped at assembly (U01), so the run config restates any per-component tolerances.

### 3.2 Passes

1. **Operator formation.**
   - Split each `D(x,t)` right-hand side into additive terms; the EarthSciASTSplitter algorithm, generalized.
   - Assign each term to an operator by provenance (component path) and the run config's selectors.
   - Unselected terms fall into default operators grouped by locality.
   - Provenance (U2) separates `operator_compose` contributions; reseact has three. Where a model instead sums its physics tendencies in observeds of one assembly component (EqWeFiC's `moist_theta_tendency.esm` and a linear factor), the planner inlines those observeds, distributes the factor, and assigns each term to the component that owns the namespaced variable it reads. The `components` selector uses that rule.
   - Algebraic equations become `algebraic` operators. Event conditions, affects and step maps become `condition` and `map` operators. Run-config step-limit expressions become `limit` operators.
   - **Parameter `update` rules** (esm ≥ 1.0 replaced parameter-writing events with them) become `condition`/`map` operators acting on parameters. `schedule` and `data` rules map to tstops and forcing epochs. `crossing` rules are refused in v1 (D22).
2. **Locality analysis.** For every state read in a term, find its index offset relative to the written cell:
   - all offsets zero → `pointwise`;
   - nonzero only on the vertical axis → `column`;
   - bounded horizontal offsets → `stencil(r)`, with radius per axis;
   - an unbounded horizontal `faq` contraction, or a global map → `global`.

   Column contractions such as reseact's PBL mean are `column`, not `pointwise`. That fixes the `contracts_space` special case in reseact's `split_common.jl`.

   Two refinements:
   - **The radius is the maximum over all `makearray` regions,** per axis and per direction. PPM reaches 3 cells upwind and 2 downwind; its mono flux form reaches 3 each side.
   - **Periodic wraps count as small reach across the seam.** EarthSciDiscretizations' periodic rules wrap by explicit index arithmetic in boundary slabs (cell 1 reads `f[NLON-1]`), which a naive offset analysis would classify as `global`.
3. **DOF sets.** Each operator records the variables it writes and reads, so its kernel outputs only its own rows. This removes the Splitter's full-length parts.
4. **Structure.** Run EarthSciASTDiff on each operator to get its Jacobian bands. From the bands, derive:
   - the block pattern per cell (n×n, plus sparsity and a Markowitz LU plan: EarthSciASTDiff's `BlockDiagonalPlan`, `markowitz_ordering` and `lu_fill`, reused as is);
   - column bandwidth, or `column_block` (dense within a column) for column contractions and sweeps, a structure class U9 adds;
   - the stencil footprint, **derived from the band entries directly**, not from `detect_structure`: that function returns `:general` for any literal column index, so every periodic-wrap stencil would classify as general (`structure.jl:21`, `:62-63`);
   - a Gershgorin spectral-radius bound, as an expression usable as a step limit.
5. **Strategy inference** (§3.5), only where the run config says `"auto"`. Stiffness is classed first from structure and hints (`reaction_systems`, `solver.stiffness`); the Gershgorin bound is evaluated at the same sampled non-default states pass 8 uses, never at the default initial condition alone (a night-time chemistry IC would misclassify).
6. **Derived documents**, one `.esm` per artifact:
   - per-operator RHS;
   - Jacobian band model. It also serves `Jᵀ·λ` by transposing at assembly, except for recurrence and scan operators: there a transpose costs O(N²) per line against O(N·L) for a reverse sweep, so the planner emits a dedicated VJP model (U9);
   - second-order band models for Rosenbrock adjoints: `∇_u`, `∇_p` and `∇_F` of the contraction `μᵀJ(u)w`, plus `∂f_t/∂u` where a method uses `f_t` (U8);
   - ∂f/∂p (including array parameters) and ∂f/∂(forcing) band models (U7);
   - algebraic residual models;
   - map, limit and condition models.

   RHS and band models compile with the stock EarthSciAST compiler. Residual, map, limit and condition models need U6: today algebraic unknowns are state slots, and observeds cannot be compiled as standalone kernels.
7. **Partition plan.** Rank and tile decomposition, halo widths per operator (the radius from pass 2), and shard sizing for pointwise and column leaves. The output stores' shard shape is an input: rank block boundaries fall on multiples of the output shard length, so ranks can write concurrently without locks (U12).
8. **Validation and emission.** Checks:
   - Σ (operator RHS) = flat RHS at sampled states, within tolerance;
   - **every** operator's band models against an independent derivative at sampled non-default states (reseact: "never FD-validate at the default IC"). This is mandatory, not sampled: EarthSciASTDiff's inliner has two paths that are silently wrong, found by reading (a contracting observed loses its Σ; a faq-defined observed is treated as a state). The oracle is ForwardDiff on EarthSciAST's Dual-transparent `f!` in Julia, which avoids reseact's "no good FD step across thirty orders of magnitude" problem; the Rust planner, which has no automatic differentiation, checks against central finite differences with steps scaled per state component, at a looser tolerance, and in the conformance suite additionally compares its assembled values to the Julia planner's values at the same states, which keeps the exact cross-language check (spec/plan.md §8.2, D76). No reusable tool for this exists today: reseact's `validate_plan` checks gather plans against index maps, and EarthSciASTDiff's oracle lives only in its tests;
   - a non-empty plan is required, because reseact's band model once went silently empty.

   Then write `plan.json` (canonical JSON) and `derived/*.esm`.

### 3.3 Determinism of the plan

The same `.esm`, run config and planner spec version must produce byte-identical `plan.json` and derived documents in Julia and Rust. These are golden-gated, like EarthSciASTDiff's byte-identical goldens.

### 3.4 Picard is a frozen-observed Jacobian

A `solve` node's `"method": {"picard": {"freeze": ["ssa.nuH"]}}` makes pass 4 treat those observeds as constants when differentiating the residual for the iteration. This is the only place freezing is written (D51). This gives PISM's Picard operator (frozen effective viscosity) as an ordinary band model, so no model-specific Picard code is needed. The adjoint still needs the full, unfrozen residual Jacobian, because the converged solution satisfies the full residual; the planner emits both band models (spec/adjoint.md §4.6, spec/plan.md §1).

### 3.5 Strategy inference rules (when "auto")

| Operator structure | Default leaf |
|---|---|
| pointwise; stiff (from `reaction_systems`, `solver.stiffness: high`, or the Gershgorin bound × Δ ≫ 1) | Rosenbrock (`Rodas4`, or `Rosenbrock23` at loose tolerance); batched block LU (sparse-unrolled if n > 16); per-cell dt on lane pools (D17) |
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
| `operators` | name → selector (`components`, `variables`, `locality`, `terms`) plus `bind_dt`; `maps` entries define step maps as (target variable ← expression or observed) pairs plus a `dt` binding |
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
| `every` | The child runs only when `t` crosses multiples of `period` (+ `phase`), and then advances by the time since its last firing (D48). Under adaptive macro steps the crossings are tstops; under a fixed `step.dt` the child fires on the step that contains the crossing (D50). With `hold: "tendency"`, its tendency is evaluated at the crossing and added as a constant to sibling leaves until the next crossing (WRF `radt`). |
| `coupled` | Monolithic integration of the sum of its operators (no splitting error). `imex: {explicit: [...], implicit: [...]}` selects ARK. If any algebraic operators are included, it becomes an index-1 DAE (§6.4). |
| `frozen` | An option of the `leaf`, `coupled` or `split_explicit` node it feeds, not a standalone node (D47): the listed operators' tendency is evaluated once at the start of each step of that node and held across all stages (WRF `rk_addtend_dry`). |
| `split_explicit` | WRF RK3 (Wicker–Skamarock: Δ/3, Δ/2, Δ) on `slow`, with `fast` integrated by forward–backward substeps (vertically implicit, `epssm` off-centring) inside each stage. *Spec in v1, implementation in M7.* |
| `map` | `u ← M(u, p, t, Δ)`: calving, mask consistency, level-set reinit, iceberg removal. A model writes a map as an observed that reads `dt` (eqice `stage2/geometry.esm`), so the run config names (target ← observed) pairs and binds `dt`. Maps may also act on parameters (parameter `update` rules). |
| `solve` | Nested nonlinear solve for algebraic unknowns with everything else frozen (§6.3). |

**`bind_dt`** sets a model parameter to the step actually taken, e.g. `{"wsm6.dtcld": "leaf"}`. A `leaf` with `alg: "ForwardEuler"`, `dt: "step"` and a bound dt reproduces a legacy WRF/PISM increment *exactly*, because the `.esm` already writes these as `(x_out − x_in)/dt` tendencies.

**Step control** is `Δ = min(limits…, distance to the next tstop)`, then `hit_multiples` and a floor at `resolution`, with a deterministic reason string on ties (PISM bugs.md #11). The macro step has no error control of its own; leaves adapt inside it, and a schedule whose root is a single adaptive leaf takes that leaf's accepted steps as macro steps (D49). The full semantics are in [`../spec/run-config.md`](../spec/run-config.md).

### 4.4 Examples (illustrative)

**reseact, the prototype re-hosted (M3).** Use `"dt_scope": "global"` to reproduce the prototype, and `"cell"` (the default, D17) for speed.

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

**eqice, PISM's `IceModel::step` (M4b):**

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
   {"leaf": "flow", "alg": "ForwardEuler", "dt": "step"},
   {"map": "enforce_consistency"},
   {"map": "front_retreat"},
   {"leaf": "source", "alg": "ForwardEuler", "dt": "step"},
   {"map": "iceberg_removal"},
   {"subcycle": {"leaf": "hydrology", "alg": "ForwardEuler", "dt": "step"}, "limits": ["hydrology.cfl", "hydrology.diffusive"]},
   {"every": {"map": "bed_deformation"}, "period": "3650 day"}
 ]},
 "step": {"limits": ["sb.cfl_2d", "energy.cfl_3d", "geometry.diffusivity", {"max": "21900 day"}],
          "hit": ["output", "forcing_refresh"], "resolution": "1 s"}}
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
- an embedded error estimate, where the method has one (Euler, SSPRK33, SSPRK104, WRF_RK3, IMEXEuler and the θ-method do not, so their leaves give `dt`, D52);
- optional dense output;
- a declared requirement set (needs J? W? HVP for its adjoint?);
- a **discrete adjoint step** (D4).

| Family | Methods (v1 first, others later) | Adjoint needs |
|---|---|---|
| Explicit RK | Euler, Heun, **SSPRK33, SSPRK43**, RK4, **Tsit5**, BS3, SSPRK104, **WRF_RK3** | `vjp_u`, `vjp_p` at stored or recomputed stages |
| Rosenbrock / ROW | **Rosenbrock23**, ROS2, ROS3P, **Rodas3, Rodas4**, Rodas5P; KPP's Ros2/3/4 and Rodas3/4 for WRF-Chem parity | plus `Wᵀ` solves (reused LU) and `hvp_u` for ∂J/∂u |
| SDIRK / ESDIRK | **BackwardEuler**, TRBDF2, SDIRK2, Kvaerno3/4 | `Wᵀ` solves at converged stages (IFT) |
| BDF | *post-v1* (D35): variable order 1–5, fixed-leading-coefficient form | tape-fixed coefficients → linear recursion backward |
| ARK IMEX | **IMEXEuler** (PISM's "explicit horizontal at old time"); ARS222, KenCarp3/4 post-v1 (D35) | ERK and SDIRK parts combined |
| Legacy fixed | θ-method (WRF chem mixing), single-step FE/BE | as above |
| Split-explicit | WRF RK3 + forward–backward acoustic (M7) | composition of the above |

**Why these are written rather than wrapped, and what is borrowed** (D31): the discrete adjoint of each family must read the forward step's stage internals, and lane pools need one vectorized step with a controller per lane; neither OrdinaryDiffEq nor diffsol offers that. Coefficients are not re-derived: tableaus come verbatim from `OrdinaryDiffEqRosenbrockTableaus`, `OrdinaryDiffEqSSPRK` and `OrdinaryDiffEqSDIRK`, cited in the spec. OrdinaryDiffEq (Julia) and diffsol (Rust) are the tier-2 oracles for order and trajectory; `SciMLSensitivity.ReverseDiffAdjoint` through OrdinaryDiffEq's own `Rosenbrock23`, `SSPRK43` and `TRBDF2(autodiff=false)` is the oracle for each hand-written family adjoint on toy problems, alongside Enzyme-on-step.

### 5.1 Controllers

- PI with OrdinaryDiffEq's defaults per method, I, and PID (Söderlind). The constants are `OrdinaryDiffEqCore`'s (`PIController`, `PIDController`), copied with citation; reseact's `pictrl_ros23`/`pictrl_ssprk43` already did this.
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
| `block_lu` | per-cell n×n | Lane-SoA batched LU. Each LU operation is vectorized across all lanes of a shard. Pivoting is optional (reseact uses none). Starts from reseact's `blocksolve` algorithm (`rx_traced_integrator.jl:110-135`: SoA, dense, no pivot, untyped), typed over U10's `LaneBlocks` layout; EarthSciMLBase's `BlockDiagonal` LU (`blockdiagonal.jl:180-217`: AoS `(n,n,ncells)`, generic `T`, optional pivoting, GPU-tested) is the oracle. |
| `sparse_block_lu` | per-cell sparse n×n | KPP-style unrolled elimination from EarthSciASTDiff's Markowitz plan (`factorization.jl:87-134` gives the ordering and fill pattern; nothing anywhere executes it). It executes as a short op tape over lanes in both languages (a generated function in Julia). New in both languages. |
| `banded` / `block_tridiagonal` | per column | Batched Thomas / block-Thomas across the columns of a shard. |
| `krylov` | global sparse | GMRES(m) and BiCGStab, distributed (D12). Julia: Krylov.jl (`gmres`, `fgmres`, `bicgstab`, with `M=`/`N=` `ldiv` preconditioners). Rust: faer's `bicgstab` with its `Precond` traits, plus an in-house GMRES(m). Preconditioners, in-house in both: block-Jacobi ILU(0); ASM (overlap 1, LU subdomains); column-block-Jacobi for 3-D implicit operators. |
| `matrix_free` | any | `J·v` from band models, with DifferentiationInterface pushforwards (Julia) or FD (Rust) as a fallback. |

**Transposed solves for the adjoint** reuse the forward factorizations for Rosenbrock, whose W is formed at the step's start: LUᵀ, and Krylov on `Jᵀ` with the transposed preconditioner. SDIRK needs W at each converged stage instead (D42): per-cell and per-column solvers refactor there, and Krylov leaves use the forward factorization as a preconditioner for the exact transposed system.

### 6.2 Nonlinear solvers

- Newton with the structured J; inexact Newton–Krylov (Eisenstat–Walker); Picard (§3.4).
- Pseudo-transient continuation (PTC), which replaces eqice's hand-written pseudo-time relaxation.
- Backtracking line search; Anderson acceleration.
- Convergence criteria on the residual or on the relative change of a named observed (PISM's ‖ΔνH‖/‖νH‖).
- **Julia:** NonlinearSolve.jl supplies all of the above (`NewtonRaphson`, `PseudoTransient`, `TrustRegion`, Broyden/Klement; LineSearch.jl; Anderson via its FixedPointAcceleration extension), with the structured W and Krylov.jl reached through a `LinearSolveFunction` (D32). The named-observed criterion is a custom termination condition. **Rust:** in-house.

### 6.3 `solve` nodes (nested algebraic, D3a)

- Warm start from the previous solution.
- `recover` ladder: under-relaxation, scaling a regularization parameter, reverting to the warm start.
- The converged z is stored on the tape for the adjoint.
- **Adjoint (IFT):** one transposed solve, `g_zᵀ μ = λ_z`, per step. Integrator code in both languages; in Julia `SciMLSensitivity.SteadyStateAdjoint` is an acceptable implementation of the same thing (D32).

### 6.4 Index-1 DAE (D3b; specified in v1, implemented after M6 unless M4b needs it, D35)

- Semi-explicit form: `u' = f(u,z)`, `0 = g(u,z)`. `M = diag(I,0)` is implicit and never assembled.
- Methods: stiffly accurate Rodas, stiffly accurate SDIRK (BDF post-v1).
- Consistent initialization solves `g(u0, z) = 0` at t0.
- The Newton system is block 2×2. It is solved by Krylov with a field-split preconditioner: the column or block preconditioner for u, and the SSA preconditioner for z.

## 7. Partitioning and execution (D6, D10, D11)

### 7.1 Decomposition

Horizontal only, so columns stay whole:
- Ranks form a 2-D block grid.
- Tiles subdivide a rank for threads.
- Staggered faces: cell i owns face i, its low (west/south) face. The tile that owns the last cell N on an axis also owns face N+1. This matches EarthSciDiscretizations ("face i is the lower (western) edge of cell i").
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

Threads need a reentrant RHS in **both** languages (U4). Julia's `f!` is not reentrant today: it uses a process-global forcing epoch, captures forcing buffers by alias, and threads internally when Polyester is loaded.

### 7.4 Deterministic reductions (D10)

- Partial sums are gathered in rank order, reduced at the root, then broadcast. `MPI_Allreduce` gives no ordering guarantee.
- Max norms and minimum dt are exact under any order.

### 7.5 Forcing

- **Rank windows.** Each rank reads its own block plus halo, using U11's name-keyed `select`.
  - Files chunked by whole global levels (GEOS-FP 0.25° is chunked `[1,1,721,1152]`) gain nothing from windows. N ranks on a node decoding the same file inflate decompression N times.
  - For those files, use one decoder per node with an MPI scatter, or convert to Zarr at fetch time.
- **Refresh stops.** A macro step stops at every provider refresh time in `(t0, tf]`. That interval is half-open on the right, so a boundary landing on `tf` fires (reseact's fencepost bug).
- **Buffer selection by stage time uses `[t_k, t_{k+1})`.** A stage evaluated at exactly `t_{k+1}` must see epoch k+1.
  - The models' `frac`-based blend weight is 0 at record times. With the old bracket still loaded, such a stage would read record k instead of k+1. (Found by reading; reseact refreshes after the step, `adjoint_gradient.jl:1046-1056`.)
  - So steps never cross an epoch boundary, and the next epoch is resident (prefetched) when a final stage lands on it.
  - A blend weight that is continuous at record times, a model-side fix, would also remove the hazard.
- **Prefetch** decodes upcoming epochs on background workers, forward or in reverse for the adjoint (U11).
  - The "98% of 0.25° forward wall time" figure predates EarthSciIO's variable, window and record pushdown. With variable pushdown alone the projection is about 180 s per refresh, which still needs re-measuring.
  - Prefetch hides refresh only up to the compute time per epoch, so decode cost has to come down first.

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
- **Parallelism.** XLA can generate multi-GPU halos. Its SPMD partitioner (Shardy/GSPMD) emits `collective-permute` halo exchanges for slices, pads and windowed ops on a dimension sharded across devices. Today's emitter output just doesn't take that path, for two reasons:
  - **Layout.** The state is one flat rank-1 vector, so the only axis to shard is the flat one. For column-major `[lon, lat, lev]` blocks a contiguous slab is a band of levels: a vertical decomposition (D20 rules it out).
  - **Reads.** Stencil reads are constant-index gathers. The partitioner cannot see that they touch only neighbours, so it all-gathers the whole field instead of exchanging a halo. reseact chose gathers deliberately: slice-form reads made compile time blow up, and gathers cut `@compile` from 894 s to 207 s.

  There are two ways to get XLA halos (U13 §6):
  - **A. Manual SPMD (preferred).** Each device runs its tile fragment (U3/U13), and the integrator emits `collective_permute` between stages inside an `sdy.manual_computation` region. Reactant 0.2.291 has raw MLIR builders for both ops but no high-level wrapper, so this needs a spike.
  - **B. Automatic.** Emit one N-D tensor per variable, sharded on lon/lat, with stencil reads as static slices, and let the partitioner insert the halos. This reopens the slice compile-time problem, and it rewrites the emitter's value model.

  The fallback if A fails is one program per device per stage, which breaks "one program per leaf step" for stencil leaves only. Norms and Krylov dot products use all-reduce in fixed-tree form. Across nodes, MPI exchanges halos on the host.
- **Adjoint.** The method-level adjoint kernels lower the same way. Checkpoints live in device memory and spill to host. The Enzyme-on-step oracle stays on CPU.

**Precision.** The hardware target is unknown, so precision is a run-config choice:
- On FP64-strong GPUs (A100/H100), the state stays f64.
- On FP64-weak GPUs, use f32 state for dynamics and transport (allowed only where the document leaves precision unspecified, D29/U14) and f16/bf16 W (§8).
- The direct emitter accepts only Float64 today, so f32 emission (U5, U13) is needed by M5g.
- Executors are per leaf, so chemistry shards can stay on CPU cores while transport runs on the GPU.

## 8. Mixed precision, including float16

Partly. Low precision is safe where its errors are self-correcting (linear algebra inside iterations, preconditioners, stored tables). It is not safe for the state. Adopted as D18, which amends D13 with precision **roles** per leaf, and qualified by D29:

- A **declared** `element_type` (document or variable) is semantic and binding. The state and RHS evaluation follow it exactly (esm-spec §11.3.1). EqWeFiC's real*4 ports declare Float32.
- Where precision is **unspecified**, the run config chooses the state and RHS roles, and the error is judged against the tolerances (U14).
- Mixed precision across leaves (f32 transport, f64 chemistry) is legal there, because conversions happen at leaf boundaries in the integrator, not inside an expression, so `mixed_element_type` never arises.
- Linear-algebra and storage roles are numerical choices in every case.
- **EqWeFiC (D30).** The WRF meteorology documents leave `element_type` unspecified, so f64 chemistry can read them legally. The run config may still choose f32 for dynamics. Bit-faithful real*4 rounding is given up. EqWeFiC's tests already run in f64 against real32 WRF at rel 1e-5, and its 3-D agreement target is statistical. No widening operator is needed.

| Role | Allowed | Default |
|---|---|---|
| time, dt, error norms, reductions, adjoint accumulators | f64 only | f64 |
| state `u` and stage vectors | f32, f64; must equal a declared `element_type` | declared `element_type`, else f64 |
| RHS evaluation | f32, f64 (≥ state); must equal a declared `element_type` | state |
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

- **Discrete events:** preset times, periodic, and condition-at-step-end. These cover forcing refresh, output, checkpoints, `every`, and `.esm` `discrete_events`. In Julia they *are* DiffEqCallbacks' `PresetTimeCallback`, `PeriodicCallback` and `DiscreteCallback`, so EarthSciAST's existing composed data-refresh, output-sink and checkpoint callbacks plug in unchanged (D33).
- **Maps** inside the schedule.
- **Callbacks are propagated into every leaf and substep.** This is the bug that broke OrdinaryDiffEqOperatorSplitting; the fix is propagation, not a new callback vocabulary.
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
- Checkpoint at macro-step boundaries: state plus each cell's dt and controller history. Storage is an in-memory ring plus disk, with an optional revolve schedule for 0.25°-class runs. The schedule is computed by the planner with Checkpointing.jl and written into `plan.json`; both runtimes execute it and neither implements revolve (D34). At 0.25° (about 188 GB for 5 days) each rank keeps its own store on node-local scratch.

**Backward pass, per macro step in reverse:**
1. Restore the checkpoint.
2. Replay forward, storing stage states per shard. Because of D10, the controller can stay on: it reproduces the forward sequence bitwise, which the recorded counts or hash verify. Where the tape holds a `(t, dt)` sequence, the controller can be turned off instead.
3. Run the family adjoint steps in reverse schedule order.

Under D10 the replay is **bitwise** equal to the forward pass, so the replay-versus-next-checkpoint check becomes an equality test (reseact had to use 1e-6).

**Primitives.**

| Primitive | Source |
|---|---|
| `vjp_u`, `vjp_p` (including array parameters), `vjp_forcing` | EarthSciASTDiff band models in both languages (U7). Recurrence and scan operators get dedicated VJP models (U9). |
| Rosenbrock second order: `∇_u`, `∇_p`, `∇_F` of `μᵀJ(u)w`, and `∂f_t/∂u` where a method uses `f_t` (nonzero inside forcing brackets) | Second-order band models, one contraction per step (U8) |
| `Wᵀ` solves | Reused factorizations for Rosenbrock; the exact system at each converged stage for SDIRK (§6.1, D42) |

Julia additionally has AD on the RHS as a fallback for primitives, through DifferentiationInterface (`AutoEnzyme` pullbacks, `AutoForwardDiff` pushforwards; D38), and two step-level oracles: Enzyme on a whole step, and `SciMLSensitivity.ReverseDiffAdjoint` through OrdinaryDiffEq's implementation of the same method (D31).

**Forward sensitivities (D36, Julia, v1).** `Dual` numbers are pushed through the whole integrator: state, stages, error norm and controller. Because the controller sees the tangents, the sensitivity includes the dt tangents reseact's prototype carried (survey §2). This is the `ForwardDiffSensitivity(convert_tspan=true)` pattern applied to our integrator and costs no integrator code beyond keeping it generic over the scalar type (D13). It is also the tangent side of the dot-product gate below.

**Accumulation.**
- Parameters: per-shard partials, reduced in fixed order.
- u0: `λ(t0)`.
- **Gridded fields:** `vjp_p` into array parameters. EarthSciASTDiff today maps parameter columns only through the scalar `p`, so array-parameter gradients are silently zero until U7 lands.
- **Forcing:** accumulated per **provider record**, not per integrator epoch, since GEOS-FP's collections are not co-phased. A slot→record map comes from the tape. The model's own time-blend expression is in the forcing band model, so the chain rule into both bracketing records is automatic.
- **Writing forcing gradients:**
  - A record's gradient is complete only after both brackets containing it are processed, so records finish in descending time order. Gradient sinks therefore use indexed (positional) writes (U12), with one store per forcing cadence.
  - Halo contributions are summed into the owning rank, in a fixed order, before writing.
- Field and forcing gradients stream to EarthSciIO sinks (U12).

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
  - Restart is bitwise for the same decomposition (D10). It comes from the integrator's own restart store; EarthSciAST's checkpoint sink promises only tolerance-consistent continuation because it holds no integrator internals.
  - Rust's writer handles f64 only today; f32 state (M7) needs U12's types.

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
2. **Methods:** convergence order on reference problems against OrdinaryDiffEq (Julia) and diffsol (Rust); dot-product adjoint tests per family, plus agreement with `ReverseDiffAdjoint` through OrdinaryDiffEq's own method on the same toy problem (D31).
3. **Cross-language trajectories:** Julia ↔ Rust at tolerance; bitwise within each language for a given decomposition.
4. **Model gates:**
   - reseact against the prototype;
   - eqice test F against the C++ dumps;
   - the EqWeFiC 24 h single-column run (SCM).

**Where the tiers run.** Tiers 1–2 and the single-rank parts of 3–4 run in ordinary CI. Multi-rank conformance (M5 onward) runs as a cluster job on 1, 2 and 4 ranks per fixture; GPU conformance (M5g) is opt-in in the same way as EarthSciAST's Reactant tests (`ESM_TEST_REACTANT=1`). Both are part of the M5 and M5g deliverables, not afterthoughts.

## 13. Upstream asks

RFC drafts for every ask are in [`rfcs/`](rfcs/). Needed-by dates reflect what the drafts found.

| # | Repo | Ask | Needed by |
|---|---|---|---|
| U1 | EarthSciAST | `solver.external` in the spec and schema | M2 |
| U2 | EarthSciAST | Per-term provenance (`equation_contributions`) carried through flatten | M2 |
| U3 | EarthSciAST | Range-restricted, DOF-subset RHS over a tile or a lane list, with halo reads and global indices (Julia `f!`; Rust slab tape) | M3 (tiles, lanes), M5 (MPI) |
| U4 | EarthSciAST | Reentrant, thread-safe RHS entry point: Rust `RhsProgram` (`Send + Sync`); Julia `f!` without global forcing state | M3 |
| U5 | EarthSciAST | Honour declared Float32 in integration: Rust (stop refusing), Julia (honour `element_type`), and the Rust XLA emitter | M5g, M7 |
| U6 | EarthSciAST | Strip mode for events, parameter updates and algebraic equations, plus standalone kernels for residuals, conditions, affects, limits and maps | M4b |
| U7 | EarthSciASTDiff | ∂f/∂(forcing buffer) bands, and array parameters as real columns | M4 |
| U8 | EarthSciASTDiff | Second-order bands for Rosenbrock adjoints (`∇_u`, `∇_p`, `∇_F` of `μᵀJw`; `∂f_t/∂u`) | M4 (Rust; Julia can use the D38 fallback) |
| U9 | EarthSciASTDiff | Filtered `faq`, non-`+` semirings, recurrences; `column_block` structure; dedicated VJP models | M4 (reseact's `Mz` filtered prefix scan), M4b (eqice column contractions)/M7 |
| U10 | EarthSciASTDiff | Production-rate Rust assembly, after a catch-up step: Rust and Python lag EarthSciAST (`aggregate`→`faq`, removed `VariableType`), so Rust does not compile against it and Python returns empty bands (both by reading) | M3 |
| U11 | EarthSciIO | Async prefetch (forward/reverse) and per-rank name-keyed windows | M5 |
| U12 | EarthSciIO | Rust streaming writer, multi-rank region writes, indexed time mode, f32/int types | M4 (Rust gradient sinks), M5 |
| U13 | EarthSciAST (Julia) | Traceable per-operator StableHLO fragments (`operator_ir`, `emit_fragment`), f32 emission, fixed-order folds | M5g |
| U14 | EarthSciAST | Numerical vs semantic precision: an omitted `element_type` is unspecified, so the run config may choose (D29) | M5g, M7 |
| U15 | EarthSciAST (Julia) | Execute recurrences in the native compiler and the StableHLO emitter | M7 |
| U16 | EarthSciAST | `solver.time_span`, the run interval a document asks for (D60); released in v0.4.0 | M2 |
| U17 | EarthSciASTDiff | Time-derivative bands (∂f/∂t) specified and golden-gated, for Rosenbrock's `f_t` on time-dependent operators; the code exists on `main` | M3 |

**Prerequisites the drafts uncovered:**
- **esm 1.3.0.** U01 and U14 ship in esm 1.3.0. reseact (esm 1.0.0), eqice and EqWeFiC (1.0.0/1.1.0) must move to 1.3.0 before they can use `solver.external` or an f32 choice.
- **An unmerged branch.** U14 builds on EarthSciAST's unmerged `fix/f59-mounted-leaf-precision` branch. On `main`, Julia's and Go's `ModelVariable` have no `element_type` field, so per-variable declarations are dropped on load (found by reading).
- **A spec clarification for U15.** esm-spec §4.3.1.1 point 4 forbids vectorized or kernel-merged recurrence evaluation. U15 needs the clarification that cells sharing a recurrence position are independent and may be batched, for the Julia native path and any GPU path.

**Bugs found while drafting, to file upstream:**
- Two inliner paths in EarthSciASTDiff are silently wrong (`inline.jl:125-140`; see U09).
- Bindings disagree on which events survive flatten (U06).
- EarthSciIO's `spec/registries.md` still calls the zarr reader v2-only.

## 14. Milestones

| | Deliverable | Gate |
|---|---|---|
| M0 | Spec skeleton: combinator semantics, run-config and plan schemas, the adjoint math per family. Repo scaffold and CI. | Schemas validate the three §4.4 examples |
| M1 | Core in both languages: ERK/SSP, Rosenbrock, SDIRK; PI control; `leaf`/`sequence`/`strang`/`subcycle`/`map`; batched block LU; tape; method adjoints. Julia on the SciMLBase interfaces (D33) with Krylov.jl and NonlinearSolve.jl (D32). | Conformance tiers 2–3 on toy problems, including the `ReverseDiffAdjoint` oracle per family |
| M2 | Planner v1: provenance, locality, DOF sets, band models, strategy rules, plan goldens | Tier 1 on reseact-shaped fixtures |
| M3 | reseact forward, shared memory: shards, lane pools (per-cell dt), dynamic re-sharding, epochs | Matches the prototype's native reference trajectory (`dt_scope: global`) **at no more than the prototype's native-reference wall time**; `dt_scope: cell` bitwise invariant to shard and thread count |
| M4 | reseact adjoint: params, then u0, gridded emissions, forcing (needs U7, U8, U9 and, in Rust, U12); forward sensitivities with dt tangents in Julia (D36) | Agreement with the prototype's 48 h gradient to ≤1e-9 relative (D16); exact dot-product (adjoint against the frozen-tape forward-sensitivity tangent, D40; Julia, while Rust is gated by finite differences and agreement with Julia, D44) and replay |
| M4b | eqice on shared memory (D37): `solve` nodes (Newton–Krylov, Picard, PTC), IMEX columns, PISM limits, maps; needs U6 and U9's `column_block` | Test F (31×31×61) against the C++ dumps |
| M5 | MPI: ranks, halo exchange, deterministic reductions, distributed I/O; multi-rank conformance in CI | reseact at 2×2.5° and 0.25° |
| M5g | Julia XLA GPU: kernel-layer lowering, one program per leaf step, lane pools at fixed capacity, in-node multi-GPU via XLA collectives, device checkpoints | reseact forward and adjoint on GPU match CPU to tolerance, in two configurations (D28): f64 state, and mixed (f32 transport state, 16-bit W, chemistry on CPU). Bitwise run-to-run on one device |
| M5b | Cross-rank lane migration for pointwise leaves (§7.6); f16/bf16 W and preconditioner storage with iterative refinement (§8) | Day/night load balance across ranks at 0.25°; refinement recovers f64 trajectories to tolerance |
| M6 | eqice at ice-sheet scale under MPI; monolithic index-1 DAE (D3b) if test F or the larger runs need it | Larger-domain eqice runs with the M5 determinism gates; D3b dot-product and consistent-initialization tests |
| M7 | EqWeFiC: SCM 24 h (f32, `every`/`frozen`/`bind_dt`), then `split_explicit` + HEVI once `grids/wrf_arw_c` exists; fire mesh | PLAN.md acceptance criteria |

## 15. Risks

1. **Two of everything (D1, D8).** The planner and MPI are the largest bodies of code, and both are written twice. Goldens keep them honest but do not halve the work. Fallback if planner effort threatens M3: the Rust runtime consumes Julia-produced plan artifacts until the Rust planner lands, with D8 as the end state (decisions §6).
2. **The adjoint math for DAEs and `split_explicit`** is the subtlest per-family work (BDF left v1 with D35). Rosenbrock adjoints in Rust depend on U8.
3. **The upstream critical path:**
   - U3 blocks tiles and MPI.
   - U7 and U9 block M4's forcing gradients.
   - U10's catch-up blocks every Rust band model, so it is a risk for M3.
   - Rust's M4 runs through six RFCs (U10 W0 → U7, U8, U9 → U12) while Julia's M4 can use the D38 fallback. If Rust slips, the Julia M4 gate stands on its own and Rust M4 is reported separately rather than holding the milestone (decisions §6).
   - Fifteen asks across three repos is a lot of upstream work. The minimal set for M3 is U4, U10 W0 and U3's lanes; U2 improves reseact's split but the locality-only split is what the prototype already runs.
4. **Rust tape performance at 3-D scale is unmeasured** against Julia native. Rust runs recurrences only on its slow interpreter (the fast tape falls back per rule, `tape/lower.rs:5179-5193`), so EqWeFiC's 3-D radiation sweeps in Rust would run there (U15, open question 3).
5. **Matching reseact's gradient.** The arithmetic order differs from Enzyme-on-step, so agreement will be at the roundoff-amplified level, not necessarily 1e-11.
6. **Lane-pool efficiency.** Compaction overhead and SIMD padding could eat into the per-cell dt gain at small shard sizes. Measure at CONUS before making `cell` the default for everything.
7. **Multi-GPU halos need a manual-SPMD spike.** Reactant has raw `sdy.manual_computation` and `collective_permute` builders but no wrapper. The fallback costs one program per stage for stencil leaves.
8. **The XLA GPU path inherits reseact's record:** minutes-long compiles (hours in pathological cases), Reactant miscompiles and buffer donation, and an XLA:CPU race. Kernel code shared between the native and traced paths may be slower in one of them.

## 16. Former defaults, now decided

The 2026-10-14 review added D31–D38 in [`decisions.md`](decisions.md): why the method families are in-house and what is borrowed; Krylov.jl, NonlinearSolve.jl and the SciMLBase interfaces in Julia; planner-emitted checkpoint schedules; the BDF/ARK/D3b scope cuts; forward sensitivities in v1; M4b before MPI; DifferentiationInterface for AD fallbacks.

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
