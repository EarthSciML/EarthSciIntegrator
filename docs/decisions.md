# EarthSciIntegrator: decisions and tradeoff analyses

Facts are in [`survey.md`](survey.md) and the design is in [`architecture.md`](architecture.md). This file records each decision and the analysis behind it.

## Decisions

| # | Date | Decision |
|---|---|---|
| D1 | 2026-10-13 | **A written spec plus a conformance suite, with independent native implementations in Julia and Rust.** Python comes later as its own native implementation, not PyO3 bindings to Rust. This is the same pattern as EarthSciAST and EarthSciIO. |
| D2 | 2026-10-13 | **The schedule lives in a separate run-config document.** The schedule is the splitting tree, the per-operator algorithm, tolerances, subcycling and adjoint settings. The `.esm` `solver` block gains an `external` option that points at the run config, so the `.esm` itself stays integrator-agnostic. This needs an EarthSciAST spec change: today the solver-block RFC is advisory only and excludes algorithm names. |
| D3 | 2026-10-13 | **Algebraic constraints get both paths:** (a) a nested, quasi-static nonlinear solve inside a step, done PISM-style with the slow state frozen, and (b) monolithic semi-explicit index-1 DAE integration, `M = diag(I, 0)`, with stiffly accurate Rosenbrock or SDIRK. *Amended 2026-10-14 (D35):* (a) is implemented in v1; (b) is specified in v1 and implemented after M6, unless eqice test F shows the nested path insufficient (survey §2 marks monolithic DAE ○ for eqice). |
| D4 | 2026-10-13 | **Adjoints are method-level discrete adjoints in both languages (option A in §1).** Every nonlinear and algebraic solve gets its adjoint from the implicit function theorem. Julia additionally offers Enzyme-on-step (option B) for leaves with no hand adjoint, and as the conformance oracle. Continuous adjoints are not supported. |
| D5 | 2026-10-13 | **Gradients are taken with respect to scalar parameters, the initial state, gridded fields and forcing time series.** Consequences: `vjp_p` must handle array-valued parameters. Forcing gradients accumulate **per provider record**, not per integrator epoch (U7 Decision 7; GEOS-FP's collections are not co-phased): the in-model time blend chains to both records of the bracket, so the gradient has the size of the forcing dataset over the window. EarthSciASTDiff does not yet produce ∂f/∂(forcing) ("forcing-buffer columns" is unchecked in its README). |
| D6 | 2026-10-13 | **The integrator owns partitioning by operator locality (§2.2), and MPI is in v1.**<br>• pointwise shards (reseact-style sharding, using its executor contract), with per-cell dt on lane pools by default (D17; per-shard dt remains an option)<br>• column sets<br>• stencil tiles with halo exchange<br>• global solves<br>Hierarchy: rank → tile → shard. Executors: threads, processes and MPI ranks. Proposed: horizontal-only decomposition, so columns stay whole and column operators stay local, as in WRF, PISM and GEOS-Chem. |
| D7 | 2026-10-13 | **Compiler contract: option 2 plus small asks (§3).** The integrator infers structure from the flattened AST. Upstream asks:<br>• `solver.external`<br>• a public, thread-safe Rust RHS<br>• Float32 integration not refused<br>• per-term provenance carried through flatten<br>• a range-restricted RHS, needed in v1 because of D6/MPI; in Rust this means running the slab tape (`simulate_array/tape`) over sub-boxes with halo reads |
| D8 | 2026-10-13 | **All code, including the planner (structure inference, splitting, derivative models, strategy choice), is implemented in both Julia and Rust against one shared spec.** The planner's output is a conformance artifact: a resolved run config plus derived documents, gated by goldens as EarthSciASTDiff does. |
| D9 | 2026-10-13 | **First end-to-end milestone: reseact.** Re-host the prototype and match its forward run and 48 h CONUS 4×5° gradient. (The prototype's own sharded-vs-unsharded agreement was 1.3e-11; the gate for the new adjoint is D16's ≤1e-9, since the arithmetic order differs from Enzyme-on-step.) |
| D10 | 2026-10-13 | **Results are bitwise identical for a fixed decomposition** (ranks, tiles, shards). All reductions use a fixed order. Different decompositions agree to tolerance, because per-shard dt depends on the shard. |
| D11 | 2026-10-13 | **v1 runs on native CPU:** Julia's native in-place `f!` and Rust's slab tape, with threads and MPI. Reactant/XLA and GPU come later, behind the same kernel interface. |
| D12 | 2026-10-13 | **Distributed global solves use Krylov methods we control** (GMRES, BiCGStab) with block-Jacobi ILU(0) or ASM preconditioning, in both languages. No PETSc dependency; a PETSc backend may come later. *Amended 2026-10-14 (D32):* "our own" means "not PETSc", not "hand-written". Julia uses Krylov.jl (`gmres`, `fgmres`, `bicgstab`, `block_gmres`; `M=`/`N=` preconditioners with `ldiv`; any `AbstractVector`, so GPU and distributed vectors work). Rust reuses faer's `bicgstab` with its `Precond` traits and writes GMRES(m), ILU(0) and ASM, which no maintained crate provides. Cross-language agreement is at tolerance (conformance tier 3), so the two need not be the same code. |
| D13 | 2026-10-13 | **Precision is generic over the scalar type T (f32/f64), and can be mixed per leaf.** Time, dt, error norms and global reductions are always f64. |
| D14 | 2026-10-13 | **EarthSciASTSplitter is folded into the planner.** Its additive-term splitting and locality rules become planner passes 1–3: specified, with Julia starting from its 636 lines and Rust implementing to the spec. It is retired once reseact, its only user, moves to the integrator after M3. |
| D15 | 2026-10-13 | **Julia has a GPU path in v1, through XLA (amends D11).** The integrator's kernel layer gets an XLA lowering: each leaf step (RHS, stages, linear solves, controller) is one XLA program, traced by Reactant. CPU stays native in both languages. Rust GPU comes after v1 (`XlaBuilder`). GPU is an optional conformance capability, so Rust declares it unsupported in v1. Analysis in §4. |
| D16 | 2026-10-13 | **M4 gate:** the new adjoint matches the prototype's 48 h CONUS gradient to ≤1e-9 relative, plus exact dot-product and replay gates. |
| D17 | 2026-10-13 | **Pointwise leaves default to per-cell dt on lane pools; stencil leaves to global dt.** `shard` and `global` remain options. |
| D18 | 2026-10-13 | **Precision roles per leaf, amending D13** (architecture §8): f16/bf16 for W, LU factors, preconditioners and stored tables, with equilibration and iterative refinement; f32/f64 for the state; f64 for time, norms, reductions and adjoint accumulators. |
| D19 | 2026-10-13 | **A domain violation rejects the step by default.** `clip` must be requested explicitly. |
| D20 | 2026-10-13 | **Rank and tile decomposition is horizontal only**, so columns stay whole. Chemistry shards may still mix levels. |
| D21 | 2026-10-13 | **The `auto` schedule is a Lie sequence in locality order stencil → column → pointwise.** The `solver.splitting` hint upgrades it to Strang. |
| D22 | 2026-10-13 | **No continuous root-finding events in v1.** |
| D23 | 2026-10-13 | **No multi-domain nesting for now.** It is not planned, and no design hooks are reserved for it. |
| D24 | 2026-10-13 | **The run config is JSON** with its own JSON Schema and an `esm_integrator` version, independent of esm. |
| D25 | 2026-10-13 | **Picard is expressed as a frozen-observed Jacobian** (`freeze: [...]` in the run config). |
| D26 | 2026-10-13 | **Accumulated diagnostics are integrated states** that the planner adds. |
| D27 | 2026-10-13 | **Names: `EarthSciIntegrator.jl`** and the `earthsci-integrator` crates (core, -planner, -solvers, -mpi). |
| D28 | 2026-10-13 | **The GPU gate (M5g) runs two configurations:** f64 state, and mixed (f32 transport state, 16-bit W, chemistry on CPU). Either kind of GPU then works when hardware is chosen. |
| — | 2026-10-13 | **RFCs are drafted for all upstream asks U1–U15** in [`rfcs/`](rfcs/), for review before anything is filed upstream. (U14 and U15 were added on 2026-10-13/14 after the first thirteen.) |
| D29 | 2026-10-13 | **Numerical precision is separate from semantic precision.** A declared `element_type` (document or variable) stays semantic and binding (esm-spec §11.3.1). An omitted one means "unspecified": f64 by default, but the run config's precision roles may choose f32 state or RHS per leaf, with the error judged against tolerances, and the integrator reports what it chose. Mixed precision across leaves is legal because conversions happen at leaf boundaries. Spec change: U14. |
| D30 | 2026-10-14 | **EqWeFiC's WRF meteorology leaves `element_type` unspecified** (D29), so WRF-Chem's double-precision chemistry can read it without a widening operator. Bit-faithful real*4 rounding is given up; tests already compare f64 runs against real32 WRF at rel 1e-5, and 3-D agreement is statistical. |
| — | 2026-10-13 | **GPU hardware target: unknown.** Design for both FP64-strong (A100/H100) and FP64-weak GPUs. Precision roles (architecture §8) make that a run-config choice. |
| D31 | 2026-10-14 | **The method families are written from scratch in both languages, and this is why.** Not because nothing exists: OrdinaryDiffEq.jl has every family, reseact's native reference already runs `SSPRK43` + `Rosenbrock23(linsolve=LU)` on an EarthSciMLBase `BlockDiagonal` through OrdinaryDiffEqOperatorSplitting, and diffsol 0.11 (the in-crate Rust solver) has tableau-generic SDIRK/ESDIRK, NDF-BDF and ERK. The reasons are: (i) D4's method-level discrete adjoint needs the forward step's stage internals in the same code as its transpose, which neither library exposes as API; (ii) D17's lane pools need one vectorized step with per-lane controllers, which no library does (EarthSciMLBase's per-cell `reinit!` is the slow form of it); (iii) D8's spec parity needs step semantics the spec owns. diffsol in particular is ruled out by facts, not taste: its adjoint is continuous (uniform checkpoints plus Hermite interpolation), it has no Rosenbrock, IMEX or PI controller, its ERK tableau check is FSAL-only (SSPRK tableaus are rejected), and its solver objects hold `Rc`/`RefCell`. **What is reused:** tableau coefficients and controller constants are taken verbatim, with citation, from `OrdinaryDiffEqRosenbrockTableaus`, `OrdinaryDiffEqSSPRK`, `OrdinaryDiffEqSDIRK` and `OrdinaryDiffEqCore` (reseact's traced steps were already copies of OrdinaryDiffEq's `perform_step!`); OrdinaryDiffEq and diffsol are the conformance tier-2 oracles for order and trajectory; and `SciMLSensitivity.ReverseDiffAdjoint` through OrdinaryDiffEq's own methods is the oracle for each hand-written family adjoint on toy problems (§1.5). |
| D32 | 2026-10-14 | **Julia's linear and nonlinear solvers are library code where the structure allows it.** Krylov: Krylov.jl (amends D12). Nonlinear `solve` nodes: NonlinearSolve.jl (`NewtonRaphson`, `PseudoTransient`, `TrustRegion`, Broyden/Klement; LineSearch.jl backtracking and strong-Wolfe; Anderson through the FixedPointAcceleration extension), driven through a `LinearSolveFunction` that wraps the structured W and Krylov.jl. The implicit-function-theorem adjoint (one transposed solve), the `recover` ladder, the Picard-as-frozen-observed Jacobian (D25) and the convergence-on-a-named-observed criterion stay integrator code; `SciMLSensitivity.SteadyStateAdjoint` is an optional implementation of the first. The batched per-cell solvers (`block_lu`, `sparse_block_lu`, banded) stay in-house in both languages: `block_lu` starts from reseact's SoA `blocksolve` algorithm typed over the U10 `LaneBlocks` layout, with EarthSciMLBase's `BlockDiagonal` LU (AoS, generic `T`, GPU-tested) as its oracle; `sparse_block_lu` is new everywhere (EarthSciASTDiff emits the Markowitz plan and fill pattern but never executes it). Rust writes all of these; argmin and diffsol's Newton are tied to dense or `Matrix`-trait storage. |
| D33 | 2026-10-14 | **The Julia implementation speaks SciMLBase.** It implements `SciMLBase.AbstractODEAlgorithm` and the `DEIntegrator` interface (CommonSolve `init`/`step!`/`solve!`), so `solve(esm_problem(...), Schedule(run_config))` works through EarthSciAST's existing problem surface, and EarthSciAST's composed data-refresh `PresetTimeCallback`, output sinks, checkpoint callback, `remake`, ensembles and name indexing keep working. Architecture §9's discrete events are DiffEqCallbacks' `PresetTimeCallback`, `PeriodicCallback` and `DiscreteCallback`, propagated into every leaf and substep. The OrdinaryDiffEqOperatorSplitting failure was dropping callbacks; the remedy is propagation, not a parallel callback vocabulary. Rust has no equivalent surface and defines its own. |
| D34 | 2026-10-14 | **Checkpoint schedules are a plan artifact.** Because the macro-step count is known in advance (fixed macro dt, tstops from the plan), the planner computes the revolve or multi-level schedule once, with Checkpointing.jl's algorithms, and writes it into `plan.json`. Both runtimes execute the schedule; neither implements revolve. No Rust revolve crate exists. |
| D35 | 2026-10-14 | **v1 scope cuts.** BDF is out of v1 (no gate needs it; stiffly accurate Rodas and SDIRK cover the index-1 DAE path; the variable-order BDF adjoint was the subtlest item in risk 2). The ARK family ships only `IMEXEuler` in v1 (ARS222, KenCarp3/4 later; reseact measured KenCarp47 at 1.7× the split's cost). D3b, monolithic DAE integration, is specified in v1 and implemented after M6 unless eqice test F requires it. |
| D36 | 2026-10-14 | **Forward sensitivities, including dt tangents, are in v1 for Julia.** The integrator is generic over the scalar type (D13), and EarthSciAST's `f!` is ForwardDiff-transparent, so `Dual` numbers flow through the state, stages, error norm and controller, which gives the controller (dt) tangents reseact's prototype carried. This is `SciMLSensitivity.ForwardDiffSensitivity` with `convert_tspan=true`, applied to our integrator. Survey §2 marks this ● for reseact; the architecture had omitted it. Rust forward sensitivities are post-v1. |
| D37 | 2026-10-14 | **The first non-reseact model comes before MPI and GPU.** A new milestone M4b runs eqice test F (31×31×61) on shared memory: `solve` nodes, maps, limits, IMEX columns. It needs no MPI, it is the first science capability that does not exist today, and it is the first test of the planner on a model it was not designed around. M5 (MPI), M5g (GPU) and M5b follow. M6 becomes eqice at ice-sheet scale under MPI, plus D3b if needed. |
| D38 | 2026-10-14 | **Julia's AD fallbacks go through DifferentiationInterface.jl**, which is already in the depot: `prepare_pullback`/`pullback!` with `AutoEnzyme(mode=Reverse)` for `vjp_u`/`vjp_p` where a band model is unavailable, and `AutoForwardDiff` pushforwards for `jvp`. This replaces a dependency on SciMLSensitivity's `EnzymeVJP`/`ReverseDiffVJP` wrappers, which pull in Zygote, ReverseDiff and Tracker. |

## 1. Adjoint strategy: tradeoffs

**Goal:** a full-simulation adjoint, as in reseact. That means the gradient of a scalar functional of the trajectory with respect to parameters, initial state and possibly gridded fields. It must be exact for the trajectory that was actually computed: reseact checks to 1e-11 against a reference, with dot-product and FD gates.

### 1.1 Parts every option needs

The bulk of reseact's `tools/adjoint_gradient.jl` is this machinery, and it is needed whichever way the per-step derivative is produced:

- **A step tape.** Record the accepted `(t, dt)` sequence, the method per leaf, discrete decisions (event times, which maps fired, clamp masks) and forcing epochs. The backward sweep replays this tape with the controller off.
- **Checkpointing.** A per-macro-step state checkpoint costs about 680 kB per step at CONUS 4×5°. At 0.25° a 5-day run would need about 188 GB, so we need multi-level checkpointing (memory plus disk) and revolve-style schedules for long or high-resolution runs.
- **Adjoint of composition.** The adjoint of `Lie(A, B)` is `B*` then `A*`, and Strang, Sequence, Subcycle and Every compose the same way. Splitting is therefore not a problem for discrete adjoints.
- **Classification and functionals.** Parameters are classified as active or inactive, by where each value is consumed. Cost functionals and observation operators are defined over outputs.
- **Validation.** Dot-product identity per step, FD of the frozen-dt map, and replay-versus-checkpoint consistency.
- **Controller derivative.** The derivative of the controller (dt as a function of state) is ignored, which is standard. reseact's forward-sensitivity lane does carry dt tangents.

### 1.2 The choice: what does automatic or symbolic differentiation see?

| | **A. Method-level discrete adjoint** (AD sees one RHS / Jacobian evaluation) | **B. AD through the step** (AD sees one whole step; this is reseact) | **C. Continuous adjoint** (solve the adjoint ODE backward) |
|---|---|---|---|
| Who writes the step adjoint | The integrator, once per method *family*: ERK/SSP, Rosenbrock, SDIRK/ESDIRK, BDF, ARK. It is generic over tableaus. | Enzyme generates it for every method automatically. | Nobody; it is method-agnostic. |
| Model primitives needed | `f`, `J·v`, `vjp_u`, `vjp_p`. Rosenbrock also needs `(∂(J·k)/∂u)ᵀλ`, a second derivative. Implicit methods also need a structured `W` and its transpose solve. | Only `f`. AD differentiates everything else, including the LU and the controller-free step. | `vjp_u`, `vjp_p`. |
| Exact for the computed trajectory? | Yes for explicit and Rosenbrock methods. For SDIRK, BDF and nested solves it is exact up to the nonlinear-solve tolerance, via the implicit function theorem (IFT). | Yes, though differentiating Newton iterations is costly and needs custom IFT rules anyway. | No. Consistent only to tolerance, which fails reseact's gates and is ill-posed with limiters, maps and splitting. |
| Linear algebra in reverse | Reuses the forward factorization: a `Wᵀ` solve, batched block LU transposed, Krylov on `Jᵀ` for the SSA. | AD differentiates the elimination itself. reseact's unrolled `blocksolve` is reversed elementwise. | `Jᵀ` solves from a fresh factorization. |
| Nested SSA and DAE constraints | IFT: one transposed sparse solve with `g_zᵀ` per step. This is the natural fit. | Needs a hand-written IFT rule, so it falls back to A for this piece. | IFT. |
| Rust parity | Full. Primitives come from the compiler and the rest is plain integrator code. | Weak. Rust Enzyme (`std::autodiff`) is nightly-only and experimental, as far as I know, and the Julia path depends on Enzyme+Reactant. | Full. |
| Toolchain risk | Low. AD or symbolic differentiation only ever sees straight-line RHS code. | High, and documented in reseact: reverse-over-forward segfault, silently wrong Reactant checkpointing, `dJ/dp` silently vanishing through closures, `stablehlo.case` with no reverse rule. | Low. |
| Cost evidence | — | reseact: transport VJP is about 19.6× one RHS (XLA recomputes face chains); `ssp_vjp` compile takes 195 s. | Extra adaptive backward solve plus interpolation of the forward trajectory. |
| Maps, clamps, events | Hand rules: the VJP of a step map, which the compiler can emit since maps are .esm expressions; clamp → mask; event times frozen from the tape. | Automatic. | Event-time jump conditions; hard. |
| Engineering cost | Highest per method family. BDF needs an adjoint of a variable-step linear multistep method, but with the step sizes and orders fixed by the tape it is a known-coefficient recursion. | Lowest per method. Orchestration (§1.1) is still hand-written. | Low, but it misses the goal. |

### 1.3 Where the model primitives come from (for A)

1. **EarthSciAST emits them by source-to-source transformation** (recommended): `f`, `jvp`, `vjp_u`, `vjp_p`, `hvp` and the structured Jacobian (EarthSciASTDiff bands) from the same expression tree.
   - This works the same way for Julia, Rust and later Python, and can be conformance-tested.
   - Recurrences (`faq` sweeps such as Thomas solves or RRTM) need reverse-mode code generation for loops. The compiler has the structure to do that.
2. **Julia:** Enzyme on the compiled RHS only. This is much smaller and more robust than Enzyme on a step.
3. **Rust:** `std::autodiff` on the RHS, if it stabilizes.

### 1.4 Recommendation

- **A is the spec'd contract in both languages.** IFT is used for every nonlinear and algebraic solve, and EarthSciAST emits the model primitives.
- **Julia additionally offers B** (Enzyme on a step) for leaves that have no hand adjoint. B also serves as the conformance oracle that checks A's hand adjoints.
- **C is not supported.**
- **Rosenbrock's second-derivative requirement** is the main extra primitive A needs. For per-cell chemistry it is a small block-local HVP. Where it is unavailable, running SDIRK in adjoint mode avoids it, since SDIRK needs only first derivatives plus transposed solves.

### 1.5 SciMLSensitivity.jl (formerly DiffEqSensitivity.jl), considered 2026-10-14

Its catalogue is exactly the B and C columns of §1.2, so it cannot supply A:

| Its algorithms | Column | Why not the contract |
|---|---|---|
| `BacksolveAdjoint`, `InterpolatingAdjoint`, `QuadratureAdjoint`, `GaussAdjoint` | C | Continuous: they integrate the adjoint ODE backward over a checkpointed or dense forward solution. |
| `ReverseDiffAdjoint`, `TrackerAdjoint`, `ZygoteAdjoint`, `EnzymeAdjoint`, `MooncakeAdjoint` | B | AD through the solver's steps; they require "the pure Julia solvers written generically", i.e. OrdinaryDiffEq. The Enzyme and Mooncake variants are documented as experimental, explicit-only, and currently failing on most solvers. |

No method-level discrete adjoint exists in it or elsewhere in SciML. What is reused: `ReverseDiffAdjoint` through OrdinaryDiffEq as the per-family adjoint oracle (D31); `SteadyStateAdjoint`, the IFT adjoint of a nonlinear solve, as an optional implementation for `solve` nodes (D32); the `ForwardDiffSensitivity(convert_tspan=true)` pattern for forward sensitivities with dt tangents (D36). Its callback handling (freeze firing times, differentiate the affect map by AD) matches architecture §9's rule but is not lifted, since our maps are compiled `.esm` kernels with band models (U6, U7). Its VJP wrappers are replaced by DifferentiationInterface (D38).

## 2. Partitioning and parallelism: tradeoffs

### 2.1 What "every stage is global" loses

If the RHS is one whole-domain function and parallelism lives only inside it, every stage of every leaf touches the whole state and ends in a barrier. The costs:

1. **One dt per leaf for the whole domain.** One stiff cell, such as the pre-dawn NO minimum, rejects the step for every cell. reseact measured its global chemistry dt at about 6× the per-cell ideal in cell-steps.
2. **Memory and locality.** An implicit leaf holds stage vectors and Jacobians for all cells at once. On EqAtmChem d05 that is 6.7M cells × 64² × 8 B ≈ 220 GB of CBM-Z block Jacobians *(est.)*; run per shard, it is O(shard). Each explicit stage also streams the full state through memory, so tiles cannot stay in cache across stages.
3. **Parallel efficiency is whatever the kernel achieves.** In reseact, transport runs on about one core and XLA:CPU parallel efficiency is about 9%. There is a barrier per stage. Load imbalance cannot be rebalanced: chemistry costs 9.6 s/step pre-dawn against 35 s/step at midday.
4. **No distributed memory.** State, checkpoints and I/O share one address space. That rules out WRF d05 with chemistry, and the roughly 188 GB of 0.25° adjoint checkpoints.
5. **No heterogeneous executors** (processes, devices, ranks). reseact needed separate processes because XLA:CPU executions in one process do not overlap.
6. **Compile cost grows with the domain on tracing backends.** reseact's lesson was that compile and memory must be per tile.

Gained by staying global:
- simplicity;
- boundary conditions and halos are handled once, by the compiler;
- global operators are natural: SSA, iceberg connected components, global norms;
- a single GPU likes large kernels.

### 2.2 The partition follows each operator's locality

| Locality | Partition | Halo | Own dt? | Examples |
|---|---|---|---|---|
| pointwise | **shards**: any set of cells, bucketed by expected cost | none | per shard or per cell | chemistry, deposition, microphysics rates |
| column | sets of columns | none horizontally | per shard | PBL, enthalpy/age, radiation, HEVI vertical |
| stencil | geographic tiles | stencil radius, exchanged per stage | per tile (multirate, later) | advection, level set, SIA, hydrology |
| global | whole domain or a distributed solver | — | — | SSA, connected components, global reductions |

**reseact-style stiffness sharding is the pointwise row.** We adopt its executor contract from `tools/shard_exec.jl`:
- species-major layout at the boundary;
- per-shard partial error norms reduced in a fixed order, for determinism;
- per-epoch lane refresh;
- per-shard dJ/dp;
- executors read no driver state.

Executors are threads by default, then processes, devices and MPI ranks. reseact used processes only because of XLA:CPU.

With per-shard dt, the adjoint tape is per shard, so tape memory scales with the number of shards, not cells.

The hierarchy is rank → tile → shard. In shared memory, tiles are index ranges over one global array and the halo "exchange" is a barrier. MPI later replaces this with real exchange, with no change to the leaf algorithms.

**What the compiler must provide:**
- **Pointwise and column partitions:** nothing new. reseact's capacity builds re-instantiate the operator on a lane grid with gathered inputs, at a build cost that depends only on the lane count.
- **Exact stencil tiles:** a range-restricted RHS that evaluates cells in a range and reads halo cells from a neighbour buffer. Today's flat `f` cannot do this. A tile re-instantiated as a small model would apply *boundary* stencils at its edges, so it would not match the global run.

## 3. Compiler contract: inferring strategy from the AST, integrator side

"Option 2" means: EarthSciAST changes only by `solver.external`. The integrator takes the flattened AST, infers the structure itself, and compiles each piece with the stock compiler. That is how reseact already works:
1. flatten;
2. EarthSciASTSplitter classifies terms by locality;
3. each part is compiled separately;
4. EarthSciASTDiff's band model gives the Jacobian;
5. capacity builds give the shards.

### 3.1 What option 2 gives

- **No waiting on spec RFCs.**
- **Strategy inference becomes a feature**, like DifferentialEquations' default algorithm choice but structure-aware:
  - block-diagonal → batched Rosenbrock;
  - banded column → block tridiagonal;
  - stencil → SSP-RK with a Gershgorin bound on the band Jacobian as the step limit;
  - algebraic → Newton–Krylov.
- **It fits the .esm philosophy:** solver-relevant classes are derived, not declared.

### 3.2 What it gives up or makes harder

1. **Authorial intent that structure cannot show.**
   - Locality separates pointwise, column and stencil terms. It cannot tell radiation from PBL (both column), acoustic from advective (both stencil), microphysics-after-RK, the `radt` cadence, or PISM's 12-step order.
   - The run config must name these. After flattening, a term's component can only be recovered from the namespaced variables it references. That usually works, but it fails when a term references two components' variables.
2. **Derivatives come from separately compiled "derivative models".** This is the band-model trick, and it works (reseact).
   - It is fragile across versions. reseact's band model went silently empty after an upstream node rename (aggregate → faq), which is why `validate_plan` (about 338 s) exists. Note what that check is: it compares the gather plan against EarthSciASTDiff's own index maps at one point and refuses an empty Jacobian (`rx_sym_block_jac.jl:288-337`). It is not a band-versus-derivative value check; no reusable tool for that exists in any sibling repo (EarthSciASTDiff's ForwardDiff oracle lives only in its test suite), so architecture §3.2 pass 8 is new integrator work.
   - EarthSciASTDiff fails loudly on filtered `faq` nodes and non-`+` semirings. It has no ∂f/∂(forcing) (D5), and does not list recurrence (column-sweep) support.
   - Rust has no AD fallback, so the Rust adjoint depends entirely on EarthSciASTDiff coverage.
3. **No exact stencil tiling and no MPI** (§2.2). Pointwise and column sharding still work.
4. **Events, algebraic equations and step maps** are refused by every fast compiler. The integrator must strip them from the AST and compile the conditions, affects and residuals as separate observed models, in each language.
5. **Duplication.** AST analysis and surgery would be implemented in Julia and in Rust, and later Python, unless it is done once (§3.3).
6. **Split parts are full-length RHSs over the shared `u`.** The Splitter produces no DOF subsets, so work is wasted on zeros. This is fixable on the integrator side.

**Unavoidable even under option 2:**
- `solver.external`;
- a public, thread-safe RHS entry point in Rust: today `debug_eval_rhs` is `#[doc(hidden)]` and `EsmProblem` is built on `Rc`/`RefCell`;
- Float32 integration not being refused (WRF).

**Cheap additions that recover most of the loss:**
- a per-term provenance tag carried through flatten (fixes 1);
- a range-restricted RHS (fixes 3).

### 3.3 Proposal: a planner that writes artifacts

Structure inference runs once, in a **planner**, which writes portable artifacts:
- a *resolved* run config: an explicit schedule, inspectable and editable, and the file `solver.external` can point to;
- derived `.esm` documents: split parts, Jacobian and VJP band models, residual models for algebraic constraints, and event-condition models.

Both runtimes consume these artifacts. Inference is implemented once, and the runtimes stay thin and conformance-testable.

**Superseded by D8:** the planner is implemented in both languages. The artifact boundary stays, as the conformance seam: both planners must produce equivalent plans for the same document.

## 4. GPU via XLA: tradeoffs

**Feasibility.** Writing the integrator's kernels in XLA works, and has been done:
- reseact's `tools/reactant_handoff/rx_traced_integrator.jl` already runs SSPRK43, Rosenbrock23 (with the batched 13×13 block solve), the PI controller and adaptive `while` loops as one XLA program per macro step.
- Both EarthSciAST bindings already emit the RHS as XLA. Julia emits StableHLO through Reactant. Rust uses `XlaBuilder` via the opt-in `xla` 0.4 crate (`simulate_array/tape/xla_emit.rs`, 2,121 lines), which links a 144 MB prebuilt `xla_extension`.

So a whole leaf step, RHS plus stages plus linear solves plus controller, can be one XLA program on CPU or GPU. No XLA↔CUDA buffer interop is needed.

**What fits XLA's static-shape, data-parallel model:**
- explicit RK stages;
- Rosenbrock and SDIRK stages with per-cell LU, unrolled over the n×n entries and vectorized across lanes (reseact's `blocksolve`);
- column Thomas sweeps (a loop over levels);
- error norms, PI control and fixed-tree reductions;
- Newton and Krylov `while` loops. These run forward only, since D4 never reverse-differentiates through them.

**What is awkward:**
- **Lane pools** need a fixed lane capacity and refill in masked chunks; finished lanes idle until the chunk ends. reseact's masked device loop wasted 20–28%.
- **Dynamic re-sharding** works if the composition is a runtime gather-index input rather than a shape, so nothing recompiles.
- **MPI halos and distributed Krylov.** XLA has no MPI. The options are:
  - split programs per stage and exchange halos on the host — reseact warned these round trips "would DOMINATE on an A100/H100";
  - XLA collectives (`collective-permute`, all-reduce) — fine across one node's GPUs; multi-node needs PJRT's distributed runtime.
- **Determinism (D10)** needs XLA's deterministic-ops mode on GPU. reseact also hit a data race on XLA:CPU (one NaN in about 2,000 calls).

**Measured costs (reseact):**
- On CPU, XLA was 2.8× slower than native Julia, had about 9% parallel efficiency, and a roughly 3 ms per-call floor.
- Compiles took minutes per program. Pathological ones ran for hours or exhausted memory (HloCSE, LLVM section memory). Compiling per tile bounds this.
- Upstream bugs: miscompiles, and `@compile` donating input buffers.

**Effort by language:**
- **Julia:** trace generic kernel code with Reactant; reseact's integrator is a working template. One source that is both fast native code and traceable is possible but may be suboptimal for one of the two.
- **Rust:** build programs op by op with `XlaBuilder`. This is more verbose, but EarthSciAST's tape→XLA emitter is precedent.

**Recommendation.**
- Keep native kernels for CPU (D11).
- Give the integrator's kernel-primitive layer an XLA lowering for GPU: each leaf step becomes one XLA program, and the host is touched only at macro-step and epoch boundaries and for MPI.
- Accept these rules on the XLA path: fixed lane capacity, composition as gather indices, per-tile programs, deterministic-ops mode, and XLA collectives within a node.

## 5. Open questions

- The GPU hardware target (D28 keeps both kinds viable).

## 6. Alternatives recorded but not adopted (2026-10-14 review)

- **Sequence the Rust planner after the Rust runtime.** The plan artifact is the seam, so the Rust runtime could consume Julia-produced `plan.json` and derived documents until a Rust planner lands (after M5), with D8 as the end state. Not adopted; D8 and M2 stand as written. Recorded in architecture §15 as the fallback if planner effort threatens M3.
- **A separate, later Rust M4 gate.** Rust's Rosenbrock adjoint depends on U10 W0 → U7, U8, U9 (filtered `faq`) → U12, six upstream RFCs, while Julia's M4 can use the Enzyme-on-RHS fallback. Not adopted as a milestone change; recorded in architecture §15 risk 3 so that a Rust slip does not hold the Julia gate.
- **Wrapping OrdinaryDiffEq per leaf for the forward pass of stencil, column and global leaves on Julia's native path.** Rejected for the same reasons as D31 (the adjoint would have to read OrdinaryDiffEq cache internals), but it remains the fastest route to a forward-only M3 transport leaf if M1 slips.
