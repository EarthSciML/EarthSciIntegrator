---
title: "Second-order bands for Rosenbrock discrete adjoints"
description: "A Rosenbrock step contains the Jacobian J(u_n) explicitly, so its discrete adjoint needs ∇_u(μᵀ J(u) w): one second-derivative contraction per step. Mixed versions are needed for parameters and forcing, plus ∂f_t/∂u when the method uses the time gradient. This RFC compares bands of bands, a dedicated HVP model and directional second derivatives. It proposes second-order bands as the spec'd artifact, obtained by differentiating band coefficients with the existing calculus, and evaluated by a contraction kernel once per adjoint step. Costs are worked for per-cell chemistry (SuperFast, CBM-Z, RADM2); stencil and column operators do not need second order."
---

Status: draft (proposed by EarthSciIntegrator, 2026-10-13)
Target: EarthSciASTDiff (`esm-jacobian-spec.md` §4 block format, a new §4.3 "Second-order entries", §5 fixtures, §7; draft spec, no version number yet). Julia reference, then Python and Rust.
Needed by: EarthSciIntegrator milestone M4 (the Rust Rosenbrock adjoint). Julia's M4 can use the Enzyme-on-RHS fallback (D4) until this lands.
Integrator refs: architecture.md §3.2 (pass 6, "HVP band model"), §5 (Rosenbrock adjoint needs), §8 (precision roles), §10 (primitives), §15 (risk 2); decisions D4, D5, D8, D18, §1.2, §1.4

File citations are relative to the `code/` directory that holds all the repositories.

## Summary

The stage equations of a Rosenbrock method contain `J = ∂f/∂u` evaluated at the step's start state. Differentiating the step with respect to that state differentiates `J` itself. The discrete adjoint (D4) therefore needs, once per step:

```
H_u = ∇_u Σ_i μ_iᵀ J(u_n) w_i
```

with stage cotangents `μ_i` and stage vectors `w_i`. When the method uses the time gradient `f_t`, it also needs `(∂f_t/∂u)ᵀ ν`. The parameter and forcing gradients need the mixed versions, `∇_p` and `∇_F` of the same contraction. Explicit, SDIRK and BDF adjoints need none of this (architecture §5).

**Proposal.**

1. **Second-order bands.** Apply the existing band calculus to the coefficients of the first-order bands. Each second-order entry is `(row, col, col2)` with index expressions and a coefficient `∂c_e/∂(site2)`.
2. **The block.** They go into a `"jacobians"` block with `"wrt": [axis1, axis2]`.
3. **Evaluation.** A contraction kernel `g[col2] += coef2 · M(row, col)`, run once per adjoint step after the stage terms are summed into `M`. The integrator never forms the Hessian tensor.

Of the three options the request lists (Design §4), this is option (a) for the artifact and option (b)'s shape for evaluation. Option (c) is a Julia-only oracle.

## Motivation

**The requirement is already in the design.**

- Decisions §1.2 states that for method-level adjoints, "Rosenbrock also needs `(∂(J·k)/∂u)ᵀλ`, a second derivative".
- Architecture §5 lists "`hvp_u` for ∂J/∂u" as a Rosenbrock adjoint need, and §3.2 pass 6 lists an "HVP band model, needed by Rosenbrock adjoints (U8)".
- Risk 2 (architecture §15) records that "Rosenbrock adjoints in Rust depend on U8". Rust has no AD fallback (decisions §3.2 item 2).

**EarthSciASTDiff stops at first order.**

- Its draft spec lists "second derivatives" as out of scope (`EarthSciASTDiff/esm-jacobian-spec.md:285-288`).
- The calculus works on any scalar ESM expression with site-based rules (`EarthSciASTDiff/pkg/EarthSciASTDiff.jl/src/scalar_rules.jl:59-139`), and band coefficients are scalar ESM expressions over row-index names, states and parameters (spec §4, `esm-jacobian-spec.md:160-165`). So the existing calculus can differentiate them again.

**reseact got this term for free, and showed why the integrator cannot.**

- reseact's Rosenbrock23 adjoint is Enzyme through the whole step, unrolled block solve included (decisions §1.2 option B).
- It moved to symbolic Jacobians precisely because reverse mode over AD-inside-the-step crashed: "reverse-over-forward segfaults inside Enzyme-MLIR on this model. The symbolic Jacobian has NO nested AD in it at all" (`reseact.esm/tools/reactant_handoff/rx_sym_block_jac.jl:29-35`).
- Under D4 the integrator writes the step adjoint by hand. The second-order term must then come from somewhere, and in Rust the only source is EarthSciASTDiff.

**Which leaves need it.**

- **Chemistry does.** Per-cell chemistry runs with Rosenbrock by default (architecture §3.5): reseact's SuperFast (13 states per cell, `Rosenbrock23`), and WRF-Chem's RADM2 and CBM-Z with KPP Rosenbrock for parity (survey §2: "RADM2 61 sp, CBM-Z 64+4").
- **Stencil leaves do not.** They default to SSPRK43, whose adjoint needs only `vjp_u` at the stages.
- **Column leaves do not.** They default to SDIRK, whose implicit-function-theorem adjoint needs only first derivatives and `Wᵀ` solves (architecture §5).

## Decisions of record

| # | Question | Decision | Why |
|---|---|---|---|
| 1 | Artifact | **Second-order bands (option a)**, produced by differentiating first-order band coefficients | Declarative, golden-able, structure-bearing. One calculus, no new rules. Serves every mixed axis with the same machinery. |
| 2 | Evaluation | **A contraction kernel over the second-order bands**, called once per adjoint step with stage contributions summed into `M` | The integrator needs a vector in ℝⁿ per step, never the tensor. Contracting at evaluation keeps the artifact reusable. |
| 3 | Axes | `wrt` is a pair `[a1, a2]`. `a1 ∈ {states, time}` is the first derivative's axis, `a2 ∈ {states, parameters, forcing}` the second's. | `[states, states]` gives `H_u`; `[states, parameters]` and `[states, forcing]` give the mixed terms of `vjp_p` and `vjp_forcing` (U7); `[time, ·]` gives `∂f_t/∂·`. |
| 4 | Symmetry | **Emit `(r, c, s)` and `(r, s, c)` independently, as they fall out of distinct first-order entries.** Consumers MUST NOT assume deduplication. | Derivation stays entry-local and deterministic. The contraction is symmetric a.e., so either orientation is correct. Storage is not persistent, so the cost of the duplicates is small. |
| 5 | Non-smooth policy | **Spec §3.2 applied again** | `ifelse` differentiates branch-wise and the condition's Dirac term is dropped; `max`/`min` keep the earlier argument on ties. The second-order pattern is the structural union over branches, global like the first-order one (spec §2). |
| 6 | Where unavailable | **The planner refuses a Rosenbrock adjoint on that leaf and offers SDIRK in adjoint mode** | Decisions §1.4 already names this fallback. |

## Design

### 1. Where the term comes from

Take the step in the form OrdinaryDiffEq's Rosenbrock/Rodas methods implement (Hairer–Wanner IV.7):

```
J = ∂f/∂u(u_n, p, t_n),   f_t = ∂f/∂t(u_n, p, t_n),   W = I − hγJ
U_i = u_n + Σ_{j<i} α_ij k_j
W k_i = h f(U_i, p, t_n + α_i h) + h J s_i + h² γ_i f_t,     s_i = Σ_{j<i} γ_ij k_j
u_{n+1} = u_n + Σ_i b_i k_i
```

**The reverse sweep.** Given `λ = ∂Φ/∂u_{n+1}`, seed `k̄_i = b_i λ` and sweep `i = s … 1`:

1. `μ_i = W⁻ᵀ k̄_i`. This reuses the forward LU (architecture §6.1).
2. Each earlier stage `j < i` receives `k̄_j += α_ij h J(U_i)ᵀ μ_i + h γ_ij Jᵀ μ_i`.
3. `ū_n` receives four terms:
   - `h J(U_i)ᵀ μ_i`, which is first order (`vjp_u` at the stage point);
   - `∇_u [ μ_iᵀ J(u) w_i ]` with `w_i = h(γ k_i + s_i)`, which is second order, from varying `J` in both `W k_i` and `h J s_i`;
   - `h² γ_i ∇_u [ μ_iᵀ f_t(u) ]`, which is second order in `(t, u)` and present only if the method passes `f_t`.

`λ_n = λ + ū_n`. The parameter and forcing cotangents receive the same three kinds of term with `∇_p` and `∇_F`.

**One contraction per step.** For a first-order entry `e` with coefficient `c_e` and points `(r, c)`, the second-order terms collapse into a single call:

```
H_u = ∇_u Σ_e Σ_{(r,c)∈e} M_e(r,c) · c_e(u; r, c),     M_e(r,c) = Σ_i μ_i[row(r)] · w_i[col(r,c)]
```

Similarly `H_t = ∇_u (νᵀ f_t)` with `ν = h² Σ_i γ_i μ_i`, which is the gradient of the time-gradient coefficients.

**W-methods with a lagged J.** A method that reuses a `J` evaluated at an earlier state `u_m` moves the term to step `m`'s adjoint. The tape must then record where each `J` was formed (architecture §10). v1 Rosenbrock leaves evaluate `J` at `u_n` every step.

### 2. Second-order entries (spec §4.3, new)

For each first-order entry `e`, the producer differentiates `coef_e` with respect to every site of every axis-2 target. It treats `coef_e` as a scalar body broadcast over `e.rows`, as `_scalar_bands!` already does for scalar bodies (`bands.jl:124-133`), with region clipping (`clip.jl:183`) and merging (`clip.jl:250`). This yields:

```json
"jacobians": { "SuperFast": [
  { "wrt": "states", "entries": [ … ] },
  { "wrt": ["states", "states"],
    "entries": [
      { "row": "O3", "col": "NO", "col2": "O3",
        "rows": [[1, 13], [1, 7], [1, 72]], "row_idx": ["i", "j", "k"],
        "col_idx": ["i", "j", "k"], "col2_idx": ["i", "j", "k"],
        "coef": {"op": "-", "args": ["k_NO_O3"]} } ] } ] }
```

The block-array form is U7 Decision 6.

**Meaning.** For each row point `r` and contracted point `c`:

```
T[ row[r], col[col_idx(r,c)], col2[col2_idx(r,c)] ] += coef(r,c)        ( = ∂²f_row / ∂col ∂col2 )
```

- `contracted` is inherited from `e`. If a second-order site lies inside a scalar `reduce: "+"` coefficient, the rule of spec §3.5 applies again and adds contracted dimensions.
- Out-of-extent `col2_idx` contributes nothing on the states axis (spec §4) and is an error on `parameters` and `forcing` (U7 Decision 5).
- `expression_templates` CSE (spec §4.1) applies unchanged.
- `structure` is `"block_diagonal"` when the first-order block is and every `col2_idx` equals the row index. In that case the contraction is lane-local. Otherwise the field is omitted.

**The `[time, a2]` blocks** differentiate the coefficients of the `wrt: "time"` block. That block is already produced, and `ode_components` attaches it as `tgrad!` (`EarthSciASTDiff/pkg/EarthSciASTDiff.jl/src/ode.jl:34-60`). With record-bracket forcing (U7), `f_t` is nonzero inside a bracket (survey §2, "`tgrad` ≠ 0 within a bracket"), so `[time, states]` is not empty for reseact.

### 3. The contraction kernel

```
hvp!(g, u, p, F, t; pairs = [(μ_1, w_1), …, (μ_s, w_s)])  →  g = ∇_{a2} Σ_i μ_iᵀ J(u) w_i
for each second-order entry e2 (child of e), each point (r, c):
    g[col2(r,c)] += coef2(r,c) · Σ_i μ_i[row(r)] · w_i[col(r,c)]
```

- **Julia v1** reuses the derived evaluation-model mechanism (`emit.jl:184-322`). Each second-order coefficient field becomes a state of the derived model, and a precomputed list `(slot, row, col, col2)` drives the contraction, as `jac!` drives its scatter (`assemble.jl:298-311`).
- **Rust** uses U10's assembler with a "contract" target in place of a storage layout.
- **Precision.** `g` and `M` are f64 (D18, adjoint accumulators). The coefficients are evaluated at RHS precision.
- **Symmetric form.** By symmetry of `∇²(μᵀf)`, the kernel may instead compute `g[col] += μ[row]·coef2·w[col2]`. That form needs no `M` and makes `s` passes. Implementations choose by cost. Conformance tests both against the oracle.

### 4. Options compared

Counts below are per cell per adjoint step. Here `n` is the species per cell and `s` the number of stages: 3 for Rosenbrock23, 6 for Rodas4. `nnz(J)` is the first-order block pattern including the diagonal, and `nnz₂` the second-order pattern.

The table's nnz values are **estimates computed from reaction stoichiometry** under mass action, assuming rate coefficients do not depend on states (they read `T` and `M` as parameters or forcing). They come from the reaction lists in `EarthSciModels/components/gaschem/superfast.esm`, `EqWeFiC/components/gaschem/cbmz/cbmz.esm` and `EqWeFiC/components/gaschem/radm2/radm2.esm`. For mass action, each second-order coefficient is a rate constant: bimolecular `∂²(k·a·b)/∂a∂b = k`.

| Mechanism | n | reactions (bimolecular) | nnz(J) | nnz₂ |
|---|---|---|---|---|
| SuperFast (reseact integrates 13 of these per cell) | 15 | 27 (21) | 89 of 225 | 136 |
| CBM-Z (KPP, WRF-Chem) | 68 | 139 (105, plus 1 termolecular) | 583 of 4624 | 999 |
| RADM2 | 61 | 157 (129, plus 2 termolecular) | 606 of 3721 | 1175 |

**(a) Bands of bands, contracted at evaluation (proposed).**

- **Work:** build `M` with `s·nnz(J)` multiply-adds, then `nnz₂` coefficient evaluations and `nnz₂` multiply-adds.
  - SuperFast, Rosenbrock23: 267 + 136 + 136.
  - CBM-Z, Rodas4: 3,498 + 999 + 999.
- **Comparison.** The first-order part of the same adjoint step already evaluates `J(U_i)` at every stage, `s·nnz(J)` coefficients (3,498 for CBM-Z), plus `s` transposed sparse LU solves. Second order therefore adds roughly one more Jacobian evaluation's worth of work per step. For mass action, `coef2` is a per-cell constant, so the coefficient work shares CSE'd subexpressions with `J`.
- **Storage:** none persistent. Transient `M` is `nnz(J)` f64 per lane, and zero with the symmetric form.
- **If materialized** instead: SuperFast `136 × 8 B ≈ 1.1 kB` per cell, about 1.5 GB at 0.25° CONUS (1.35M cells, survey §1). CBM-Z `999 × 8 B ≈ 8 kB` per cell, about 54 GB on EqAtmChem d05 (6.7M cells, decisions §2.1). Hence evaluate on the fly, per shard.

**(b) A dedicated HVP model.** Emit an ESM model whose right-hand side is `∇_u Φ` for `Φ = Σ_e M_e c_e(u)`, with `M`, or `μ` and `w`, as array inputs.

- **Work and code size** are the same as (a), since the same coefficient derivatives are emitted, already contracted.
- **What it loses:**
  - a declarative pattern and golden for the second-order structure;
  - reuse across the mixed axes, which would need one model per axis and input form;
  - the symmetric-form choice.
- **What it gains:** no contraction list. That is why (a)'s Julia v1 evaluation borrows this shape and lets the stock evaluator compute the coefficients.

**(c) Directional second derivatives.**

- **Forward over forward.** To obtain `(∂(Jw)/∂u)ᵀμ ∈ ℝⁿ` from `D²f(u)[w, v]`, one needs `n` directions per block. They can be colored across cells, as reseact's block Jacobian is (`reseact.esm/prototypes/reseact_3d_chem/block_jac.jl:99-103`). That is `n·s` second-directional evaluations per step: 39 for SuperFast with Rosenbrock23, 408 for CBM-Z with Rodas4. Each costs a dual-of-dual RHS evaluation, so this is one to two orders of magnitude above (a).
- **Forward over reverse.** `∇_u(μᵀJw) = ∇²_u(μᵀf)·w` brings the cost down to O(s) RHS-sized sweeps, but needs reverse mode. Rust has none (decisions §1.2, "Rust parity: weak"). Finite differences are not an option either: there is "no good h" across thirty orders of magnitude (`block_jac.jl:88-97`).
- **Verdict:** (c) is a Julia oracle, via ForwardDiff or Enzyme on the tree-walk RHS, not a primitive.

**Stencil and column operators.** Second-order bands of limiter stencils would multiply the coefficient swell that the CSE work already fights (`EarthSciASTDiff/README.md:114-127`) by the stencil width. They are not needed: explicit stencil leaves and SDIRK column leaves have first-order adjoints. The producer computes second-order blocks only for operators the planner assigns to a Rosenbrock leaf, so the cost is never paid elsewhere.

### 5. Bindings

| Binding | Work |
|---|---|
| **Julia (reference)** | `jacobian_bands2(sv; wrt = (:states, :states))` over `jacobian_bands`'s entries; `prepare_hvp(input; wrt)` returning `hvp!`; emission of the block array; tests against ForwardDiff-over-ForwardDiff contractions of the tree-walk RHS. |
| **Python** | Band half and emission (byte-identical goldens); a numpy oracle for values. |
| **Rust** | `dscalar` already exists for `Expr` (`EarthSciASTDiff/pkg/earthsci-astdiff-rs/src/rules.rs`). Add the coefficient-band driver and emission; values through U10's contraction target. Prerequisite: U10 W0. |

## What this is not

- **Not a Hessian assembly API.** No consumer needs the tensor, so the spec defines entries and a contraction, not a matrix type.
- **Not for stencil or column leaves** (Design §4), and not for SDIRK/BDF. Their implicit-function-theorem adjoints are first order.
- **Not the derivative of the step-size controller,** which is ignored as is standard (decisions §1.1).
- **Not third order, and not general symbolic reverse mode.**

## Conformance and tests

1. **Fixtures:**
   - `bd_chem.esm` (existing; block-diagonal);
   - a SuperFast-sized fixture with a temperature-dependent rate read from a parameter, which gives a `[states, parameters]` block;
   - U7's `forcing_bracket.esm`, which gives `[states, forcing]` and `[time, states]`;
   - `minmod_adv.esm`, to pin the `max`/`min` second-order policy on a kink.
2. **Goldens:** `<fixture>.<a1>-<a2>.bands.json` and `.block.json`, byte-identical in Julia, Python and Rust.
3. **Value goldens** `<fixture>.<a1>-<a2>.hvals.json`: the contracted `g` at a pinned non-default `(u, p, t)` and pinned random `(μ_i, w_i)`, 1e-10 relative. The Julia oracle is the ForwardDiff Hessian of `μᵀf` applied to `w`. Both kernel forms (Design §3) must match it.
4. **Integrator side (not gated here):**
   - a per-step dot-product test of the Rosenbrock23 and Rodas4 adjoints;
   - agreement with the Enzyme-on-step oracle (architecture §10);
   - the M4 gate against reseact's 48 h gradient (D16).

## Work breakdown

1. **Spec:** §4.3 second-order entries and `wrt` pairs; §5 fixtures; remove "second derivatives" from §7.
2. **Julia:** the driver, emission, the evaluation model and contraction list, `prepare_hvp`, and goldens.
3. **Python:** band half and emission.
4. **Rust:** after U10 W0; the contraction target in U10's assembler.
5. **Integrator:** the Rosenbrock adjoint step, using `hvp!` once per step.

## Alternatives considered

- **Option (b) as the artifact.** Rejected as the spec'd form (Design §4), kept as the shape of Julia v1 evaluation.
- **Option (c) as the primitive.** Rejected for Rust parity and cost. Kept as the Julia oracle.
- **Freeze `J` in the adjoint and drop the term.** That gives the adjoint of a different map. It would fail the dot-product gate and the 1e-9 M4 target.
- **Switch every adjoint chemistry leaf to SDIRK.** Available as the fallback (Decision 6), but it changes the forward method. The M4 gate requires matching reseact's Rosenbrock23 trajectory.
- **Exploit mass-action structure directly**, with rate constants as the second derivatives. It is a special case of (a) that the CSE already captures. It would break for state-dependent rate laws, such as reseact's `m` if it enters a rate.

## Open questions

1. **KPP parity methods.** Ros2/3/4 and Rodas3/4 for WRF-Chem (architecture §5) run autonomous, with `f_t` taken by finite differences in KPP. Does parity require reproducing that FD `f_t`? If so, its adjoint is not the `[time, states]` block.
2. **Rate laws that read states.** For SuperFast, do any rate laws read a state, such as air density `m` in reseact? That raises `nnz₂` above the stoichiometric estimate. Measure it on the real flattened operator.
3. **Low-precision W.** With a low-precision `W` under D18 (Rosenbrock-W only), is the adjoint exact for the computed trajectory once the `J` used is rounded? Architecture §8 already accepts refinement-tolerance agreement.
4. **Storing the stage vectors.** Should `w_i` and `μ_i` be stored per stage, or `M` accumulated during the reverse sweep? The answer depends on lane-pool memory (architecture §7.6).
