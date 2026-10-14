# EarthSciIntegrator: survey of what the integrator must do

**Status:** survey, 2026-10-13. These are the facts the design rests on; they are not the design itself.

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
  - Rust: `debug_eval_rhs(state, t, params, forcing)`, hidden from the public API.
  - Reactant: an out-of-place `(u, p, t, bufs) -> du`.
- **Layout is variable (species) major**: scalars first, then one column-major block per array variable (`EarthSciAST/pkg/EarthSciAST.jl/src/tree_walk/state_layout.jl`). Per-cell block solves therefore vectorize across cells, struct-of-arrays style; reseact's `blocksolve` exploits this.
- **Flattening sums `operator_compose` terms and does not record which operator each term came from.** Splitting today happens downstream, by locality, in EarthSciASTSplitter.jl (Julia only).
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
