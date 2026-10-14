---
title: "Strip and expose: events, parameter updates and algebraic equations handed to the caller, plus standalone observed kernels"
description: "An opt-in build mode in which the fast compilers remove the constructs CONFORMANCE_SPEC §5.39 makes them refuse (events, symbolic parameter updates, implicit equations) and return them as structured objects, and an API that compiles named observeds and expressions (residuals, conditions, affects, step limits, step maps) as standalone kernels sharing the right-hand side's state, parameter and forcing ABI, with algebraic unknowns passed in as inputs. Refusal stays the default."
---

Status: draft (proposed by EarthSciIntegrator, 2026-10-13)
Target: EarthSciAST — esm-libraries-spec §2.5 (new §2.5.11), API_SPEC §5.8, CONFORMANCE_SPEC (new §5.46), esm-spec §9.6.6 (three new codes). No esm format change.
Needed by: EarthSciIntegrator milestone M6
Integrator refs: architecture.md §3.2 (passes 1 and 6), §4.3 (`map`, `solve`, `coupled`), §6.3, §6.4, §9, §13 (U6); decisions D3, D4, D7, D8, D22; decisions.md §3.2 item 4

## Summary

Every fast EarthSciAST compiler refuses a document that carries a continuous
event, a discrete event, an implicit equation or Wiener noise
(CONFORMANCE_SPEC.md:5686-5697). That rule is right. A compiler that ran the
model without the construct would report a number the document does not
describe.

An external integrator needs something different. It wants to run these
constructs itself: the nested SSA solve, the index-1 DAE, the step-end event,
the step map. To do that it needs two things from the compiler:

1. **Strip and expose.** On explicit request, remove the named constructs from
   what is lowered, compile the rest, and return the removed constructs as
   structured, flattened, namespaced objects. Algebraic unknowns `z` leave the
   state vector and become an input.
2. **Standalone kernels.** Compile a named set of observeds and expressions as
   kernels with the right-hand side's ABI: `(u, z, p, t, dt) -> out`. These
   serve residuals `g(u, z)`, event conditions, affects, step limits and step
   maps `u <- M(u, p, t, dt)`.

Refusal stays the default. Stripping is never silent: the caller names each
construct class it takes over, and `solve` on a stripped Problem is refused.

## Motivation

**eqice cannot state its momentum balance.** The full PISM model is a PDAE:
algebraic SSA momentum equilibrium plus PDE transport. Because the harness
refuses implicit equations, eqice realizes PISM's Picard/Newton solve as a
pseudo-transient relaxation `u_t = residual(u; H, E, tauc)` integrated to
steady state (eqice/PLAN.md:152-163). The stage-2 SSA component does exactly
this (eqice/stage2/ssa.esm:5, the relaxation equation at :240, the residual
observed at :394). It costs more than a solve and it is fragile. diffsol's
finite-difference Newton stalls once the state sits at the fixed point, so the
document must stop at pseudo-time 100 to stay short of a wall at about 108
(eqice/PROGRESS.md:570-576). It also gives the wrong adjoint structure. The
integrator's adjoint for a nested solve is one transposed solve by the implicit
function theorem (architecture §6.3, D4). That needs `g(u, z)` and `z` as
first-class objects, not a fake time derivative.

**eqice's substeps are step maps.** "Calving stays step maps (no discrete
events)" (eqice/PLAN.md:205). A map is written today as an observed of a `dt`
parameter. `GeometrySource` is "the algebraic one-step map of the C++ and does
not integrate" (`dt` at eqice/stage2/geometry.esm:13362-13366;
`H_out_source = H_in_src + dH_SMB + dH_BMB` at :13418-13425, :13666-13667).
Nothing can apply such a map to a running state, and the harness "cannot force
PISM's single implicit-Euler step" (eqice/PROGRESS.md:646-648).

**EqWeFiC has maps and step-end updates too.**
- Ignition time is a per-cell interpolation applied after each level-set step,
  reading the step start `ts` and the step `dt`
  (EqWeFiC/components/wildland_fire/fire_behavior/ignition_time_wrffire.esm:5, :23-34).
- Level-set reinitialisation is a pseudo-time loop run between propagation and
  the ignition update. It is "NOT part of d psi/dt"
  (EqWeFiC/components/wildland_fire/fire_behavior/level_set_reinit_wrffire.esm:5).
- WSM6 saturation adjustment applies "once per sub-step, not integrated"; the
  document writes it as an increment over `dtcld`
  (EqWeFiC/components/atmospheric_dynamics/wsm6/saturation_adjustment.esm:239, :319).
- The fire `tbound` CFL reduction is "a solver step limit, not a
  discretisation" (EqWeFiC/PLAN.md:1140).

**The integrator design depends on it.** The planner turns algebraic
equations into `algebraic` operators, and conditions, affects, step maps and
step limits into `condition`, `map` and `limit` operators (architecture §3.2
pass 1). Pass 6 compiles each with the stock compiler. decisions.md §3.2 item 4
assumed the integrator would strip the AST itself. That fails for four reasons:

1. **Algebraic unknowns are state slots.** Flatten puts `algebraic_variables`
   inside `state_variables` (esm-libraries-spec.md:1389-1392;
   pkg/EarthSciAST.jl/src/flatten.jl:203-214), and so does Rust's array build
   (pkg/earthsci-ast-rs/src/simulate_array/compile.rs:1553-1561). No compiler
   takes `z` as an input separate from `u`.
2. **Observeds are not kernels.** Julia evaluates one observed per call at a
   given `u` and `t` (pkg/EarthSciAST.jl/src/simulate.jl:1401-1404). Rust reads
   observeds only as rows appended to a solution
   (pkg/earthsci-ast-rs/src/problem.rs:1246-1252). Neither is batched or
   allocation-free, and neither takes `z` or `dt`.
3. **Event lifting differs by binding.** Julia's flatten lifts subsystem events
   (pkg/EarthSciAST.jl/src/namespacing.jl:331-345). Rust and Python lift only
   top-level components' events and search the raw document instead
   (pkg/earthsci-ast-rs/src/simulate/driver.rs:357-367;
   pkg/earthsci-ast-rs/src/compile_error.rs:410-426;
   pkg/earthsci-ast-py/src/earthsci_ast/problem.py:1337-1351). Stripping from
   the flattened system would lose events in two of three bindings.
4. **Symbolic parameter updates are refused too.** Rust refuses a parameter
   whose `update` recomputes from an `expression`
   (pkg/earthsci-ast-rs/src/simulate_array/compile.rs:1586-1599). Since 1.0.0
   these replace discrete events that wrote parameters (esm-spec.md:1910-1913).

## Decisions of record

| # | Question | Decision | Why |
|---|---|---|---|
| 1 | Default | **Refuse, unchanged.** Strip is opt-in per construct class. | §5.39 exists because silent dropping produced wrong answers. A caller that did not ask still gets `unsupported_construct`. |
| 2 | Granularity | **Four classes:** `continuous_events`, `discrete_events`, `parameter_updates`, `algebraic`. Wiener noise is never strippable. | An integrator without root-finding (D22) asks only for `discrete_events`; a continuous event then still refuses. No consumer integrates SDEs. |
| 3 | Where it lives | **A keyword on `esm_problem`**, plus two accessors. Not a new top-level verb. | Reuses the whole build pipeline: rewrite, value invention, gated fetch, providers, forcing buffers. |
| 4 | Running a stripped Problem | **`solve` and `init` refuse it** with `stripped_problem_not_solvable`. | Running it would be the silent drop §5.39 forbids. |
| 5 | `z` placement | **A separate input vector with its own layout**, same convention as `u`. | Nested solves (D3a) iterate on `z` alone; the DAE path (D3b) can concatenate. A mutable side buffer would break thread safety (U4). |
| 6 | Residual sign | **`g = lhs − rhs`**, exactly. | One deterministic convention, so Julia and Rust plans agree byte for byte (D8). |
| 7 | `dt` | **An explicit `f64` argument.** Parameters named in `dt_params` read it and are never constant-folded. | Implements `bind_dt` (architecture §4.3) for maps, limits and the RHS alike. |
| 8 | Map and affect assignment | **Simultaneous.** All sources are evaluated on the input state, then written. | Matches `Pre` semantics (esm-spec.md:1759-1777) and makes the result independent of output order. |
| 9 | Diagnostics | **Three new codes:** `stripped_problem_not_solvable`, `kernel_output_unresolved`, `affect_reads_post_value`. | Each names a distinct, testable failure. |

## Design

### 1. Strip and expose

`esm_problem` gains `strip`, a set drawn from the four classes. Empty is the
default and changes nothing.

For each requested class the build does the following. It runs after flatten,
template expansion and discretization, on the same flattened system the RHS is
built from.

- **`continuous_events`, `discrete_events`.** Collect every event: top-level
  components, inline subsystems at any depth, reaction systems, and
  coupling-level events (esm-spec.md:2111-2136). Remove them from the build.
  Each binding must lift subsystem and coupling-level events for this; see
  Open question 3.
- **`parameter_updates`.** Collect every `update` rule of kind `schedule`,
  `condition` or `crossing` whose value form is `expression`. The parameter
  becomes a live runtime parameter: it is read from `p` at every call and never
  folded. `from` and `handler` updates stay on the forcing seam as today. A
  `crossing` rule is reported with the continuous-event class, so it is
  stripped only when `continuous_events` is also requested.
- **`algebraic`.** Remove every implicit equation: an expression LHS, a numeric
  LHS (pkg/earthsci-ast-rs/src/classification.rs:552), or `ic` of one. Move
  every `algebraic_unknowns` member from the `u` layout to the `z` layout.
  Observeds that read `z` stay observeds and read it as an input.
  `D(a + b) ~ 3` is a misspelled equation, not a residual; it stays refused
  (CONFORMANCE_SPEC.md:5731-5736).

Classification functions do not change. `ode_states`, `observed_unknowns` and
`algebraic_unknowns` (esm-spec.md:2359-2363) are properties of the model. Only
the compiled layout changes.

What remains is an ODE right-hand side `f(u, z, p, t)` over the `u` layout.
It is exposed as the kernel `rhs` (§2).

`stripped(prob)` returns the removed constructs as one object. Its canonical
JSON (CONFORMANCE_SPEC §5.5.3) is:

```json
{
  "strip": ["algebraic", "discrete_events"],
  "events": [
    {"name": "front_retreat", "origin": "Calving", "kind": "discrete",
     "trigger": {"type": "condition", "expression": {"op": ">", "args": ["Calving.H", 0]}},
     "affects": [{"lhs": "Calving.H", "rhs": {"op": "*", "args": [{"op": "Pre", "args": ["Calving.H"]}, 0]}}],
     "reinitialize": false}
  ],
  "parameter_updates": [{"parameter": "Sfc.scale", "origin": "Sfc", "rules": ["<§5.4 rules, verbatim>"]}],
  "algebraic": {
    "unknowns": ["SSA.u", "SSA.v"],
    "z_layout": {"SSA.u": {"base": 1, "shape": [31, 31]}, "SSA.v": {"base": 962, "shape": [31, 31]}},
    "z0": "<values from ic equations, else defaults, in z_layout order>",
    "residuals": [
      {"id": "SSA.residual[1]", "origin": "SSA", "shape": [31, 31],
       "expr": {"op": "-", "args": ["<lhs>", "<rhs>"]}}
    ]
  }
}
```

- Every name is flattened and fully qualified (esm-spec §10.7).
- `origin` is the authoring component path. When U2 lands it takes U2's
  `Origin` form (`{"component": path}` or `{"coupling": k}`).
- Residual `id`s are `<origin>.residual[k]`, counting that origin's implicit
  equations in flattened order.
- `z0` gives the warm start for consistent initialization (architecture §6.4).
- Continuous events carry `conditions`, `affects`, `affect_neg` and `root_find`
  verbatim (esm-spec.md:1829-1839).
- Order is deterministic: by origin in flatten order, then by declaration order.

### 2. Standalone kernels

`compile_kernels(prob, specs)` compiles a list of kernel specs against the
Problem's build. Every kernel shares one ABI with the stripped RHS:

```
k(out, u, z, p, t, dt)
```

- `u`: the Problem's state layout with `z` removed. Scalars first, then array
  blocks in sorted-name order, column-major
  (pkg/EarthSciAST.jl/src/tree_walk/state_layout.jl:1-16).
- `z`: the `z_layout` from §1. Empty when nothing was stripped.
- `p`: the Problem's resolved parameter carrier.
- forcing: the same carrier the RHS reads. In Rust that is U4's explicit
  `ForcingSet`; in Julia, the Problem's own buffers.
- `t`, `dt`: always `f64` (D13). `dt` is read only by parameters named in the
  spec's `dt_params`.

A spec is JSON, so plans can store it and golden it (architecture §3.3):

```json
{"name": "geometry_source", "dt_params": ["GeometrySource.dt"],
 "outputs": [
   {"assign": "Geometry.H", "from": "GeometrySource.H_out_source"}
 ]}
```

An output is one of:

| Output | Value | Shape |
|---|---|---|
| `{"observed": name}` | the observed's value | its declared shape |
| `{"expr": <AST>}` | an expression over flattened names | inferred; scalar for a full reduction |
| `{"residual": id}` | a stripped residual `g(u, z)` | the residual's shape |
| `{"condition": event, "index": k}` | a continuous condition, or a discrete trigger (true = 1, false = 0) | scalar or per cell |
| `{"affect": event}` | the event's post-event state | full `u` length |
| `{"update": parameter}` | the parameter's new value under its rule set | the parameter's shape |
| `{"assign": unknown, "from": name or AST}` | a step map | full `u` length |

`rhs` is a reserved kernel name for the stripped RHS, output `du` of `u`
length.

Rules:
- **Concatenation.** `out` holds the outputs in spec order, each column-major.
  `kernel_layout(kernel)` returns the name-to-slot map, like `var_map`.
- **Map outputs.** `assign` and `affect` produce a full state. Unassigned slots
  copy `u`. Several `assign` entries in one spec are simultaneous (Decision 8).
  A map spec may not mix map outputs with other kinds.
- **`Pre`.** In an affect or update, `Pre(x)` reads the input `u`, `z` or `p`.
  A bare reference to a variable that the same event also assigns is refused
  with `affect_reads_post_value`. That would make the affect an implicit system.
- **Step limits** are ordinary `observed` or `expr` outputs. A per-cell limit
  stays per cell. The integrator reduces it in fixed order (D10).
- **Unresolved names** fail at compile with `kernel_output_unresolved`, naming
  the output.
- **Strictness.** Kernels obey the Problem's `compiler`. Under `native` a rule
  the compiled tiers cannot express is `compiler_refused_rule`
  (API_SPEC.md §5.8), never a fallback.

### 3. Per binding

**Julia.**

```julia
prob = esm_problem(doc, tspan; strip = (:algebraic, :discrete_events), compiler = :native)
s    = stripped(prob)                       # StrippedConstructs; to_json(s) gives §1
ks   = compile_kernels(prob, specs)         # KernelSet
k    = ks["ssa_residual"]
k(out::AbstractVector{T}, u::AbstractVector{T}, z::AbstractVector{T}, p, t::Float64, dt::Float64)
kernel_layout(k)::StateLayout
ks["rhs"](du, u, z, p, t, dt)
```

`prob.f!` stays `f!(du, u, p, t)` and is only defined when `z` is empty. `T` is
the state element type (U5). Kernels are generated in-place functions on the
same codegen tiers as `f!`. Under `:xla` they SHOULD be emitted by the direct
emitter (U13). `:mtk` and `:sympy` refuse `strip` with `compiler_unavailable`.

**Rust.**

```rust
let prob = esm_problem(&file, tspan, ProblemOptions {
    strip: StripSet::ALGEBRAIC | StripSet::DISCRETE_EVENTS, ..Default::default() })?;
let s: &StrippedConstructs = stripped(&prob);
let ks: KernelSet = compile_kernels(&prob, &specs)?;
let k: &Kernel = ks.get("ssa_residual")?;
k.eval_into(u: &[T], z: &[T], p: &[f64], t: f64, dt: f64, forcing: &ForcingSet,
            out: &mut [T], ws: &mut RhsWorkspace);
k.layout() -> &OutputLayout;
k.param_names() -> &[String];   // order of p
```

`p` is positional `f64`, rounded on read to each parameter's declared precision, as U5 specifies. `eval_into` follows `debug_eval_rhs_into`: caller-owned output and scratch, no
allocation (pkg/earthsci-ast-rs/src/simulate_array/driver.rs:608-621). It is
public. `Kernel` holds no `Rc` or `RefCell`, so U4 can make it `Send + Sync`
without changing this signature. U3 adds a box-and-halo argument the same way
it does for `rhs`. Kernels lower to the same tape as the RHS.

**Python.** Implements `strip` and `stripped()` in full, because the object is
needed for conformance. `compile_kernels` runs on the NumPy `native` tier with
no performance promise. It is the third interpreter in the conformance tier,
as Python is in §5.38 (CONFORMANCE_SPEC.md:5551-5553). It MAY land after Julia
and Rust. Until then it raises `compiler_unavailable` naming `compile_kernels`.

Go and TypeScript do not simulate. They register the three codes only.

### 4. Interaction with other asks

`origin` uses U2's provenance. The kernel ABI is the RHS ABI, so U3 and U4
apply unchanged, and `T` follows U5. EarthSciASTDiff (U7, U8) differentiates
the stripped expressions directly.

## What this is not

- **Not an event runtime or a DAE solver.** The compiler exposes; the caller
  runs. Root-finding, Newton and index analysis stay in the integrator.
- **Not a relaxation of §5.39.** Without `strip`, every refusal case stays a
  refusal, and `solve` never runs a stripped Problem.
- **Not SDE support.** Wiener noise always refuses.
- **Not index reduction.** Strip moves implicit equations aside. It does not
  check that `g_z` is nonsingular; the planner does that from band models.
- **Not a format change.** No document gains a field. Step maps stay
  observeds; the run config names the `assign` pairs.

## Conformance and tests

New CONFORMANCE_SPEC §5.46, "Strip and Expose", plus kernel probes.

1. **Default unchanged.** All sixteen §5.39 refusal documents still refuse
   without `strip`, in every executing binding.
2. **Per-class strip.** Each §5.39 event and implicit document builds with its
   own class stripped, and still refuses with any other class.
3. **Byte-identical `stripped()`.** Canonical JSON (§5.5.3) matches a committed
   golden in Julia, Rust and Python. Fixtures include subsystem,
   reaction-system and coupling-level events (Motivation item 3).
4. **Not solvable.** `solve` and `init` on a stripped Problem raise
   `stripped_problem_not_solvable`. Control: `strip` on a document with nothing
   to strip still solves.
5. **Remaining RHS.** For each event fixture, `rhs` with empty `z` equals the
   RHS of the §5.39 control document at its probes.
6. **Kernel probes** in the §5.38 format: fixed `(u, z, p, t, dt)`, a Julia
   interpreter golden, an analytic anchor, and the §5.38.2 tolerance classes
   (CONFORMANCE_SPEC.md:5578-5592). Cases: a 1-D SSA-shaped residual with `u`
   algebraic, anchored at the exact van der Veen profile; a `GeometrySource`-
   shaped step map at two `dt` values; an ignition-time map reading `t` and
   `dt`; a per-cell CFL limit; a discrete condition and its affect with `Pre`.
7. **Refusals.** One fixture each for `affect_reads_post_value` and
   `kernel_output_unresolved`.
8. **`dt` is live.** A kernel that reads `dt` only through `dt_params` changes
   its output when `dt` changes, with no rebuild.

## Work breakdown

1. **Spec.** esm-libraries-spec new §2.5.11 (classes, stripped object, kernel
   specs, ABI). API_SPEC §5.8 rows and `api-surface.json` entries for `strip`,
   `stripped`, `compile_kernels`, `kernel_layout`. esm-spec §9.6.6 rows for the
   three codes, cross-referenced from `unsupported_construct` (esm-spec.md:4487).
2. **Flatten lifting** (prerequisite, useful alone). Rust and Python lift
   inline-subsystem events as Julia does. All three lift coupling-level events.
3. **Julia.** Skip the refusals per requested class
   (pkg/EarthSciAST.jl/src/tree_walk/helpers.jl:40-44,
   pkg/EarthSciAST.jl/src/tree_walk/build.jl:1777-1785, :3866-3867); add a `z`
   layout; generate kernels on the existing tiers.
4. **Rust.** The same gates in `simulate_array/compile.rs` (:540-551, :690-701,
   :820-829, :1586-1599). `z` becomes an input block on the tape, lowered like a
   forcing name with no defining rule (:1571-1585). Public `Kernel`, `KernelSet`.
5. **Python.** Strip and the stripped object in `problem.py` (:1317-1370);
   interpreter kernels.
6. **Conformance.** §5.46 fixtures and adapters; kernel probes in the
   compiled-RHS manifest.

## Alternatives considered

- **The integrator strips the AST itself** (decisions.md §3.2 item 4). It would
  reproduce flatten's event lifting, `Pre` handling and layout in Julia, Rust
  and later Python, and still could not compile a kernel with `z` as an input.
- **Use the `:mtk` compiler.** It runs events and implicit equations, but it
  refuses loaded data and continuous spatial dimensions, and exists only in
  Julia (API_SPEC.md §5.8, the `mtk` row). All three target models are 3-D and
  forcing-fed.
- **Keep the pseudo-transient rewrite.** It already stalls in eqice
  (eqice/PROGRESS.md:570-576), costs a time integration per solve, and its
  adjoint is not the IFT adjoint.
- **Declare `z` as shaped parameters in derived documents.** Parameters may be
  folded at build (Julia's `:const_folded` class,
  pkg/EarthSciAST.jl/src/simulate.jl:617-621), and Rust's `p` is a map of
  scalars (pkg/earthsci-ast-rs/src/problem.rs:534-536).
- **Pass `z` through a mutable side buffer**, like forcing. Shared mutable state
  breaks parallel evaluation with a different `z` per thread.
- **One `strip = true` switch.** Rejected by Decision 2.

## Open questions

1. **Bare affect-LHS reads.** This draft refuses them. ModelingToolkit reads a
   bare variable as the post-event value, which makes the affect implicit.
   Should strip expose such affects as small algebraic systems instead?
2. **Residual-to-unknown matching.** Should `stripped()` return a structural
   matching of residuals to `z` blocks for field-split preconditioners
   (architecture §6.4), or should the planner compute it from bands?
3. **Coupling-level events today.** Flatten records them only as descriptive
   strings (pkg/EarthSciAST.jl/src/flatten.jl:1162-1163;
   pkg/earthsci-ast-rs/src/flatten.rs:3123-3136), and no §5.39 fixture covers
   one (tests/conformance/unsupported_construct/fixtures/). Assumption, not
   verified by running a fixture: the fast compilers silently drop them now.
   This should be confirmed and fixed independently of this RFC.
4. **Rust's symbolic-update refusal** raises `UnsupportedFeatureError`, not
   `unsupported_construct` (pkg/earthsci-ast-rs/src/simulate_array/compile.rs:1586-1599).
   Julia's behaviour was not checked. Should §5.39 name symbolic parameter
   updates as a fifth construct, so strip and refusal cover the same set?
5. **Derived-document form.** Should a kernel spec also be expressible as a
   derived `.esm` (an `outputs` block), so plan goldens hold one artifact type
   (architecture §3.2 pass 6)?
