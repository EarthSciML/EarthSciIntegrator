---
title: "EarthSciIntegrator: survey of what the integrator must do"
weight: 30
---

**Status:** survey, 2026-10-13; §6 (library inventory) added 2026-10-14 from the reuse audit. These are the facts the design rests on; they are not the design itself.

**Sources:** `../EqWeFiC` (WRF, WRF-Chem, CFBM fire), `../eqice` (PISM), `../reseact.esm` (a CTM that contains the prototype integrator), and the upstream compiler stack (`../EarthSciAST`, `../EarthSciASTDiff`, `../EarthSciASTSplitter.jl`, `../EarthSciMLBase`, `../EarthSciDiscretizations`). Paths are relative to the parent `code/` directory.

## 1. Where each model stands

| | reseact.esm | eqice | EqWeFiC |
|---|---|---|---|
| What runs in 3-D today | Everything. CONUS 4×5° is 13×7×72 cells × 13 vars = 85k states. 48 h forward + gradient in 43 min. | Nothing yet. Stage 2 has components and the 3-D subassemblies; the Stage-3 C++ reference driver is merged; `eqice.esm` does not exist. | Nothing yet. The largest document is a 59-level single column (656 states, 24 components), tested only at t=0. `grids/wrf_arw_c` has not been started. |
| Integrator in use | Hand-written traced SSPRK43 + ROS23 in Julia/Reactant, Lie split, host-lifted adjoint (`reseact.esm/tools/reactant_handoff/rx_traced_integrator.jl`). The native reference uses OrdinaryDiffEqOperatorSplitting. | EarthSciAST Rust harness (diffsol BDF/SDIRK/ERK) with a dense FD Jacobian. | The same Rust harness. 3-D is "solved by a different method", per the user's decision of 2026-09-12 (`EqWeFiC/PLAN.md:1496`). |
| Scale target | 0.25° CONUS: 193×97×72 = 1.35M cells, 17.5M states (`reseact.esm/SCALE_025.md`) | Test F: 31×31×61, 2-D SSA ≈ 2k unknowns, enthalpy and age each ≈ 60k. Larger ice sheets are implied. | Last Chance: 5 one-way nests, d05 301×451×51 at dt=0.5 s, about 8e7 unknowns, plus a 25 m fire mesh of about 1200×1800 |
| Splitting goal | Lie (transport → chemistry). Strang is untried. | Match PISM's `IceModel::step` order exactly (`eqice/PLAN.md:150-212`) | Statistical agreement only. "RK3 with acoustic sub-stepping is the solver's job and is not written in .esm" (`EqWeFiC/PLAN.md:2252`) |

## 2. Capability matrix

Legend: **●** = needed now, **○** = needed later or optional, blank = not needed.

| Capability | reseact | eqice | EqWeFiC | Notes |
|---|---|---|---|---|
| **Explicit RK, adaptive, SSP** (transport) | ● SSPRK43 | ● ERK for geometry, age and hydrology | ● 5th/3rd-order upwind advection, level set | reseact has no CFL logic; accuracy sets dt |
| **Fixed-step legacy schemes, reproducible step for step** | | ● PISM forward Euler (mass), backward Euler columns, explicit horizontal at the *old* time | ● WRF RK3 (dt/3, dt/2, dt), fire RK3, theta=0.75 chem mixing | eqice: "harness cannot force PISM's single backward-Euler step" (`eqice/PROGRESS.md:641`) |
| **Stiff in-cell ODE** (block-diagonal W) | ● 13×13, ROS23 | ○ PDD firn | ● RADM2 61 sp, CBM-Z 64+4, MOSAIC 235 rows/col | KPP Rosenbrock (Ros2/3/4, Rodas3/4) is the WRF-Chem reference; tolerances 1e-9/1e-3 |
| **Column-implicit** (tridiagonal or banded per column) | ○ PBL is a column contraction, a rank-1 relaxation | ● enthalpy and age BE columns, BTU | ● YSU PBL, chem vertical mixing, soil, HEVI acoustics | eqice energy BDF scaled as N³ through the dense Jacobian |
| **IMEX / HEVI** | ○ (KenCarp47 was 1.7× slower than the split) | ● implicit vertical + explicit horizontal | ● vertically implicit acoustic step with `epssm` off-centring | |
| **Global sparse nonlinear algebraic solve** | | ● SSA: 2 comps × 9-point box = 18 nnz/row; Picard+KSP, Newton (SNES), warm start, regularization ladder | (WRF-ARW has no elliptic solve) | The .esm side fakes it with pseudo-transient relaxation; diffsol's FD Newton stalls at the fixed point (`eqice/PROGRESS.md:570`) |
| **DAE time integration** (singular mass matrix) | | ○ only if SSA is integrated monolithically rather than nested | | No mass matrix appears anywhere in the three repos today |
| **Operator splitting** | ● Lie, 300 s macro step | ● sequential, about 12 ordered substeps + maps | ● RK3 with physics frozen across stages, microphysics after the RK loop, chem after `solve_em` | OrdinaryDiffEqOperatorSplitting dropped callbacks and tolerances and crashed on reject (`reseact.esm/HELPERS.md`) |
| **Multirate / subcycling / every-N** | ○ per-cell or bucket dt (forward only, paused) | ● skip counter, hydrology internal subcycling, BedDef every 10 yr, `dt_TempAge` accumulation | ● acoustic `ns` substeps, `radt`, `chemdt`/`photdt`, fuel-moisture dt, nest ratio 3 | |
| **dt from stability limits** (CFL, diffusivity, stop times) | | ● 17 `max_timestep` restrictions, `hit_multiples`, reason string | ● CFL, fire `tbound` | The schema has no CFL/dt vocabulary |
| **Physics step as a tendency parameter** | | ● `H_out_source(dt)` step maps | ● `dtcld`, `dtsec`, Tiedtke CAPE floor, GWDO `dtfac` | dt-dependent tendencies have no dt→0 limit (`EqWeFiC/TENDENCY_ERROR_BUDGET.md:102`) |
| **Discrete step maps / projections** | ○ clamp | ● calving, FloatKill, mask consistency, iceberg removal (global connected components), positivity limiter, part-grid redistribution | ● saturation adjustment, WSM6 rescaling, level-set reinit, ignition times | eqice: "Calving stays step maps (no discrete events)" |
| **Events** (preset, periodic, continuous root-find) | ● forcing refresh at half-open boundaries | ● BedDef interval, coupler records | ○ ignition-time crossing, prescribed ignition window | Every fast EarthSciAST compiler refuses events today |
| **Forcing epochs** (refresh without recompiling, in-model time blend) | ● 15 providers, 64 refreshes/48 h; refresh is 98% of wall at 0.25° | ● coupler NetCDF | ● `wrfbdy` hourly, lateral relax zone | Needs async prefetch at scale; `tgrad` ≠ 0 within a bracket |
| **Positivity** | ● by rejection, not clipping ("reject … retry smaller, do NOT modify values") | ● S–Z limiter, clamps | ● PD advection, clamps | ROS23 trial stages go negative at the pre-dawn NO minimum |
| **Discrete adjoint** | ● ∂J/∂p over 160 scalars, exact to 1e-11 | ○ PISM has SSA adjoints for inversions | ○ not discussed | reseact records the (t, dt) tape, checkpoints each macro step, replays with the controller off |
| **Forward sensitivity** | ● including the dt tangents | | | |
| **Float32 working precision** | | | ● WRF is real*4; constants bound to f32 roundings | Rust refuses f32 integration today |
| **Multiple meshes / nests** | | | ● 4×-refined 2-D fire mesh ↔ 3-D atmosphere; 5 nests | |
| **Parallelism** | ● chemistry sharded over processes (2.5–3×); transport on 1 core; halo-tiled transport proposed | small | ● required at d05 scale (WRF used 40 cores × 38 h) | XLA:CPU parallel efficiency about 9%; GPU ruled out on Quadro FP64 |
| **Output** | streaming Zarr, saveat | snapshots and diagnostics as hard stop times | history per domain, accumulations (precip totals) | |
| **Restart** | per-macro-step checkpoints | must include solver warm-start state (SSA velocity) | from WRF frames | "tolerance-consistent, not bit-identical" (EarthSciAST streaming-output RFC §16.7) |
| **Validation hooks** | step-sequence tape, dot-product tests, determinism | single-step BE/FE reproduction vs C++ dumps | per-process tendency at an injected state, integer branch diagnostics | |

## 3. Input contract today (EarthSciAST)

- **One flat semi-discrete ODE per document.**
  - Julia: `f!(du,u,p,t)`, in place and ForwardDiff-generic.
  - Rust: `debug_eval_rhs(state, t, params: &HashMap<String,f64>, force_scalar)` plus an allocation-free `debug_eval_rhs_into` (`simulate_array/driver.rs:527-533`, `:613-621`), both `#[doc(hidden)]`. Its in-crate `solve` is diffsol 0.11 (`Bdf`, `tr_bdf2`, `tsit45`; `driver.rs:2160-2207`) with a dense finite-difference Jacobian costing 2n RHS evaluations (`:2022-2122`), a fresh solver per forcing segment (`run_segmented`, `:1593-1716`), and no mass matrix, events, sensitivities or `dt_min` guard (`docs/content/rfcs/rust-diffsol-solver-guard-gap.md`).
  - Reactant: an out-of-place `(u, p, t, bufs) -> du`.
- **Layout is variable (species) major**: scalars first, then one column-major block per array variable (`EarthSciAST/pkg/EarthSciAST.jl/src/tree_walk/state_layout.jl`). Per-cell block solves therefore vectorize across cells, struct-of-arrays style; reseact's `blocksolve` exploits this.
- **Flattening sums `operator_compose` terms and does not record which operator each term came from.**
  - Julia tracks one owner per *equation* internally (`flatten.jl:1050-1056`), and Python exposes one `source_system` per equation (`flatten.py:1733-1737`). Neither is per-term provenance.
  - Splitting today happens downstream, by locality, in EarthSciASTSplitter.jl (Julia only).
- **No Jacobian, sparsity or mass matrix** is attached to the compiled problem.
  - EarthSciASTDiff can produce symbolic bands, `detect_structure` (block_diagonal / banded / general), and per-cell Markowitz LU orderings.
  - None of that is wired into a solver.
- **The `solver` block** offers stiffness, abstol/reltol and `splitting` ∈ {none, lie, strang}, all advisory and document-wide. The RFC explicitly excludes algorithm names, DAE declarations and per-component placement.
- **Variable classification** includes `algebraic_unknowns`, but only the `:mtk` compiler runs them, and `:mtk` refuses data and PDEs.
- **Cadence partition** (CONST / DISCRETE / CONTINUOUS) already identifies values that change only at refresh events.
- **API conventions to keep:**
  - `esm_problem` → `init` / `step!` / `solve` with SciML spelling;
  - the tolerance chain (call site → document → default);
  - `tstops` unioned and `callback` replaced;
  - `retcode`, name-indexed solutions, `remake`, ensembles.

## 4. Other prior art

### EarthSciMLBase (MTK-based; does not use .esm)

- `SolverStrangThreads` / `SolverStrangSerial` are really first-order Lie splitting with a fixed Δt.
- They run one integrator per cell via `reinit!`, with static thread chunks and shared mutable parameters (which force a lock in EarthSciData).
- `SolverIMEX` is a `SplitODEProblem` with a block-diagonal stiff part.

Worth keeping:
- One compiled per-cell kernel, `f(du, u_cell, p, t, c1, c2, c3)`, with coordinates passed as arguments, run through a pluggable map backend (broadcast, threads, AcceleratedKernels).
- A fused batched LU over `(n,n,ncells)` blocks (`EarthSciMLBase/src/blockdiagonal.jl`).
- Data refresh as a preset-time event that also fires at init.

Broken or missing:
- the multi-operator sum (it returns only the last operator's `du`);
- `OperatorJacobian`, vjp, sparse Jacobians, Reactant;
- MPI;
- support for anything but exactly 3 spatial dims.

### EarthSciIO

- The provider contract is `materialize`, `refresh(t)`, `refresh_times()` and `prefetch`; `refresh_times()` becomes the tstops.
- With `records_per_sample=2` a provider returns a record bracket, and the model does the time blend.
- `prefetch` only warms the disk cache, so there is no async overlap.
- Output: Julia has a streaming Zarr writer with a reserved `region=` argument for subdomain writes. Rust's Zarr writer writes a whole dataset in one call; a streaming handle is "future work".

### Other

- `earthsciml/aq_demo/strang_problem.jl` is a hand-rolled Strang split with per-thread Rosenbrock23.
- moves.rs and netcdf-rust contain no time integration.

## 5. Lessons from the prototype (reseact.esm), as design rules

1. **The adjoint is of the realized step sequence.** Record the accepted (t, dt) tape and replay with the controller off; re-deriving the sequence "silently takes the adjoint of a different trajectory".
2. **Reverse mode cannot cross a data-dependent trip count.** Keep adaptive control on the host, or bound it on device with a static cap and masking.
3. **Any parameter that is differentiated must be an explicit argument.** A closure over `p` makes `dJ/dp` "silently vanish".
4. **Symbolic block Jacobians beat FD and colored AD.** They are exact, safe under reverse mode, and 4–25× cheaper to compile. FD has "no good h" across 30 orders of magnitude.
5. **Positivity comes from rejection, not modification.** The clamp zeroes gradients where it bites.
6. **Forcing:** refreshes must fire on a set independent of `macro_dt`, using half-open intervals. Refresh cost dominates at high resolution.
7. **A global chemistry dt costs about 6× the per-cell ideal,** but every per-cell scheme tried so far lost to per-call overhead at 4×5°. That will probably flip at 0.25°.
8. **No caches that grow with grid × run length.** Compilation and memory must become per-tile, not per-domain.
9. **Treat non-finite values as rejections,** and design against XLA:CPU races (a 1-in-2000 NaN) and buffer donation.
10. **Upstream SciML pain:**
    - `similar(jac_prototype)` densified BlockDiagonal;
    - the splitting package dropped callbacks and tolerances;
    - a rejected outer step crashed;
    - Reactant cannot trace through the SciML stack.

## 6. What already exists, verified (reuse audit, 2026-10-14)

Read, not built. Paths are relative to `code/`; depot paths are under `~/.julia/packages` and `~/.cargo/registry/src`.

### 6.1 In the sibling repos

| Thing | Where | State | Plan's use |
|---|---|---|---|
| Time-stepping methods | `reseact.esm/tools/reactant_handoff/rx_traced_integrator.jl:286-449` | Two hard-coded methods (Rosenbrock23, SSPRK43), copied from OrdinaryDiffEq's `perform_step!` (header `:6-18`); no tableau abstraction; PI only, constants from OrdinaryDiffEqCore (`:98-99`) | Oracles only; rewritten tableau-generic (D31) |
| Native reference path | `reseact.esm/run_reseact.jl:103,251`; `tools/reactant_handoff/op_split.jl` | OrdinaryDiffEq `SSPRK43` + `Rosenbrock23(linsolve=LU)` on an EarthSciMLBase `BlockDiagonal`, through `LieTrotterGodunov`, with workarounds for the three upstream bugs in §5 item 10 | The M3 trajectory and wall-time reference |
| Batched block LU | reseact `blocksolve` (`rx_traced_integrator.jl:110-135`): SoA, dense, no pivot, `Matrix{Any}`; EarthSciMLBase `src/blockdiagonal.jl:180-217`: AoS `(n,n,ncells)`, generic `T`, optional pivoting, Float32 and Metal tests | Nothing sparse-unrolled exists anywhere | Start from `blocksolve`; EarthSciMLBase as oracle (D32) |
| Jacobian structure | EarthSciASTDiff `factorization.jl:32-146` (`BlockDiagonalPlan`, `markowitz_ordering`, `lu_fill`); `structure.jl:51-67` | Only the block-diagonal plan is implemented; `detect_structure` returns `:general` for any literal column index, so periodic stencils classify as general; the evaluator fills a global Float64 CSC (`assemble.jl:193-298`), never executes an LU | Plan and pattern reused; stencil structure derived from bands (architecture §3.2) |
| Derivatives | EarthSciASTDiff | ∂f/∂u, scalar ∂f/∂p, ∂f/∂t only; no VJP/JVP/Hessian/forcing/array-parameter columns; Rust and Python ports still match `"aggregate"` (`bands.rs:102,236,278`), last commits 2026-08-24/23 | U7–U10 are real gaps |
| Term splitting | `EarthSciASTSplitter.jl` (636 lines): `additive_terms` `:111`, `_reads_nonlocal` `:311` | Full-length parts; `dofs` only as a caller-supplied tuple | Folded into the planner (D14) |
| Tape, checkpoints, replay, gates | `reseact.esm/tools/adjoint_gradient.jl` (`StepRec` `:840`, `Ckpt` `:980`, `replay_fixed` `:1006`, `backward_sweep` `:1205`, gates `:1619-1755`) | Generic pattern tangled in script globals; step VJPs are Enzyme through Reactant (`:642-672`) | Orchestration reused after extraction; VJPs rewritten |
| Executor contract | `reseact.esm/tools/shard_exec.jl:25-91` | `ProcessExecutor` (Distributed) and `InProcessExecutor`; fixed-order partial norms | Adopted (D6) |
| Per-cell dt attempts | `tools/subcycle_chem.jl` (0.023× wall), `tools/bucket_chem.jl` (0.66× wall) | Both forward-only and paused | The XLA:CPU per-call floor they lost to does not exist on native CPU (architecture §7.6) |
| Half-open refresh rule | `op_split.jl:126-145`; `adjoint_gradient.jl:903-907` | A few lines | Reused as is |
| `validate_plan` | `rx_sym_block_jac.jl:288-337` | Gather plan vs EarthSciASTDiff index maps; refuses an empty Jacobian; not a band-vs-derivative check | Pass 8 is new work |
| SciML problem surface | `EarthSciAST.jl/ext/EarthSciASTSimulateExt.jl:128-275` | `ODEProblem` with unioned tstops, composed `PresetTimeCallback` refresh, output sinks, checkpoint callback, `remake`, ensembles, name indexing; no `jac_prototype`, mass matrix or algorithm choice | Kept, by implementing the SciMLBase interfaces (D33) |
| Reactant multi-device | `EarthSciAST.jl/ext/reactant_direct/device.jl:22-43` | `NamedSharding` over the flat cell axis, GSPMD collectives | Not used: flat-axis slabs are vertical (U13) |
| EarthSciIO | `julia/src/zarr_write.jl:411-661`; `provider.jl:187-252`; `rust/src/format/zarr_write.rs:185` | Julia streaming writer (f64/f32/i32/i64/u32/u64, `region` reserved); positional `select` pushdown in all bindings; Rust writer whole-dataset, f64-only; prefetch synchronous cache-warm everywhere | U11, U12 are real gaps |
| EarthSciDiscretizations | 149 rule files | No halo-width metadata (free-form `stencil_width:3` tags only), no decomposition, no CFL expressions, no integrator code; `-integ` worktree is an integration branch, not an integrator | Halo radius comes from the planner's offset analysis |
| eqice, EqWeFiC | — | No solver code of their own (eqice's C++ is PISM's own driver instrumented; EqWeFiC has Python fixture tools only); `eqice.esm` and `grids/wrf_arw_c` do not exist | As the plan states |

### 6.2 Julia packages in the depot

| Package | Relevant content | Plan's use |
|---|---|---|
| OrdinaryDiffEq* (all sub-packages), OrdinaryDiffEqCore | Every family; `PIController`, `PIDController`, `IController`, `PredictiveController`; tableau packages | Coefficients and controller constants copied with citation; tier-2 oracle (D31) |
| Krylov.jl | `gmres`, `fgmres`, `bicgstab`, `block_gmres`, `cg`, `dqgmres`; `M=`/`N=` with `ldiv`; any `AbstractVector` | Julia Krylov (D32) |
| NonlinearSolve.jl, NonlinearSolveFirstOrder, LineSearch.jl | `NewtonRaphson`, `PseudoTransient` (`src/pseudo_transient.jl`), `TrustRegion`, Broyden/Klement; backtracking, strong-Wolfe, Li–Fukushima; Anderson via `FixedPointAccelerationJL` extension | Julia `solve` nodes (D32) |
| LinearSolve.jl | `LinearSolveFunction` for custom algorithms | The seam for structured W |
| DifferentiationInterface.jl, Enzyme, ForwardDiff | Pullbacks and pushforwards over any backend | AD fallbacks (D38) |
| Reactant 0.2.291 | Raw `sdy.manual_computation` and `stablehlo.collective_permute` builders; no wrapper in `Ops.jl` or `Sharding.jl` | U13 open question 1 confirmed |
| BFloat16s, BandedMatrices, KernelAbstractions, Sundials | present | §8; column solvers; GPU kernels; oracle |
| **Not in the depot** | MPI.jl (only the jlls), SciMLSensitivity, Checkpointing.jl, ImplicitGlobalGrid, ILUZero/IncompleteLU, PartitionedArrays | To add as needed; Checkpointing.jl only in the planner (D34) |

### 6.3 Rust crates in the registry

| Crate | Verified content | Plan's use |
|---|---|---|
| diffsol 0.11.0 (36k lines, MIT) | NDF-BDF orders 1–5 (`ode_solver/bdf.rs`); tableau-generic SDIRK (`tr_bdf2`, `esdirk34`) and ERK (`tsit45` only; `check_explicit_rk` requires FSAL, so SSP tableaus are rejected, `runge_kutta.rs:222-274`); elementary I-controller, not PI (`:473-495`); `set_stop_time`, `RootFinder`; mass matrix with Brown–Hindmarsh–Petzold consistent initialization (`op/init.rs`); modified Newton with backtracking; `LinearSolver` trait pluggable but assembles into its `Matrix` trait (`FaerLU`, `FaerSparseLU`, `NalgebraLU`, `KLU`, `CudaLU`; no Krylov); Jacobian from a user JVP with NaN-propagation sparsity and greedy colouring; **adjoint is continuous** (`ode_solver/adjoint.rs`, uniform checkpoints every 500 steps plus Hermite interpolation, `checkpointing.rs`); `Scalar` for f32 and f64; `Rc`/`RefCell` inside, no rayon | Tier-2 oracle; otherwise replaced (D31) |
| faer 0.24.4 | Dense and sparse LU/Cholesky/QR; `bicgstab` with `Precond`/`BiPrecond` traits (`operator/bicgstab.rs:111`), CG, LSMR; **no GMRES, no ILU, no batched small-matrix routines**; f32/f64 only | BiCGStab reused; GMRES, ILU(0), ASM, batched LU in-house (D32) |
| nalgebra-sparse 0.11 | CSC Cholesky only | — |
| half 2.7.1 | `f16`/`bf16` types; f16 arithmetic native only on aarch64 `fp16`, widens to f32 on x86; bf16 always computes in f32 | §8 storage roles, as assumed |
| xla 0.4.4 | `XlaBuilder` with 67 ops, PJRT CPU/GPU clients, F16/Bf16 element types, no collectives | Post-v1 Rust GPU (D15) |
| rayon, ndarray | present | — |
| **Not found anywhere** (web search) | Rosenbrock, ARK-IMEX, PI/PID controllers, Newton–Krylov, a maintained ILU(0) or Schwarz crate, batched block LU, discrete adjoints, revolve, halo exchange, f16 linear algebra; `std::autodiff` is still nightly-only | New work, as the plan assumes. `mpi` (rsmpi) 0.8.2 is active; `kryst` offers GMRES but was not vetted |
