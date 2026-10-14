---
title: "Derivatives through filtered faq, non-\"+\" reductions and causal recurrences"
description: "Define EarthSciASTDiff's derivative rules and band representation for the three faq constructs it refuses today: filtered faq nodes (prefix scans and windowed searches), non-\"+\" reductions (max, min, *), and causal self-reference recurrences (Thomas sweeps, RRTM up and down sweeps, falling-precipitation sweeps). Filters become an exact point predicate on band entries. max and min select the first argmax. Products use exclusion products. Recurrences differentiate into a tangent recurrence and a reverse-axis adjoint recurrence, with a dense lower-triangular Jacobian per line that a new column_block structure class reports. Fixes two silent inliner paths found along the way."
---

Status: draft (proposed by EarthSciIntegrator, 2026-10-13)
Target: EarthSciASTDiff (`esm-jacobian-spec.md` §3.5, §4 entry fields, a new §4.4 "Recurrences", the structure classes and the `factorization` registry, §5 fixtures; draft spec, no version number yet). Prerequisite in EarthSciAST.jl: execute recurrences (causal-self-reference RFC §6.1 binding debt).
Needed by: EarthSciIntegrator milestone M6/M7 (eqice and EqWeFiC column physics). The filtered-sum rule is needed earlier, at M4: forcing gradients through reseact's transport (U7 Motivation 4).
Integrator refs: architecture.md §3.2 (passes 2, 4, 6), §3.5, §6.1 (`banded` / `block_tridiagonal`), §10 (primitives), §13 (U9); decisions D4, D8, §1.3 item 1, §3.2 item 2

File citations are relative to the `code/` directory that holds all the repositories.

## Summary

The esm format has three faq constructs beyond a plain `reduce: "+"` contraction:

- `filter` predicates. These are the normative spelling of cumulative scans (`EarthSciAST/esm-spec.md:765-799`).
- `reduce` values `"*"`, `"max"` and `"min"` (`esm-spec.md:690`).
- Causal self-reference recurrences: an equation defining array `V` reads `index(V, k−a)`, and cells are swept in ascending `k` (`esm-spec.md` §4.3.1.1; `EarthSciAST/docs/content/rfcs/causal-self-reference-recurrence.md`).

EarthSciASTDiff refuses the first two loudly and does not know about the third. This RFC specifies:

1. **Filters.** The filter is a gate whose Dirac term is dropped, like an `ifelse` condition. An index-only filter becomes a new exact entry field, `where`. A value-dependent filter folds into the coefficient as `ifelse`.
2. **max and min.** The derivative of the first argmax in ascending contraction order, matching the spec's left-fold tie rule. Integer-valued level searches used as indices become one-hot contracted bands.
3. **Products.** The product rule with exclusion products, with no division.
4. **Recurrences.**
   - A tangent recurrence on the same axis.
   - An adjoint recurrence written ascending on a reversed axis, because the spec admits no backward lag.
   - Jacobian structure: dense lower triangular within each line of the recurrence axis, and dense per line when a forward and a reversed sweep compose (Thomas).
5. **A new structure class, `column_block`,** with a per-line pattern, so the planner can pick column solvers (architecture §3.5).
6. **Two inliner fixes.** Today a contracting observed and a faq-LHS definition can yield silently wrong bands.

## Motivation

### What EarthSciASTDiff does today

- **Filtered faq.** `BandError("filtered faq nodes not supported …")` in Julia (`EarthSciASTDiff/pkg/EarthSciASTDiff.jl/src/bands.jl:204-206`) and Rust (`pkg/earthsci-astdiff-rs/src/bands.rs:285-288`).
- **Non-`"+"` reduce.** `BandError` in Julia (`bands.jl:209-210`) and Rust (`bands.rs:294-299`). The Rust numeric oracle also refuses one inside a coefficient (`numeric.rs:89-93`). The spec reserves both: "producers MUST fail rather than guess" (`esm-jacobian-spec.md:119-124`).
- **Recurrence with a bare LHS (`V ~ faq{…}`) (by reading).**
  - `V` is an observed, and target-dependent, so it is inlined.
  - `_inline_once` replaces `index(V, …)` by V's body (`inline.jl:132-140`). That body contains `index(V, k−1)` again, so no fixpoint exists.
  - It throws `InlineError` after 64 passes (`inline.jl:20`, `:49-55`). This failure is loud.
- **Recurrence with a faq LHS (`faq{expr: V[k]} ~ …`), which the causal RFC allows (§2, L71-74) (by reading; not executed).**
  - EarthSciASTDiff's `lhs_definition` recognizes only a bare-variable LHS (`system.jl:123`). EarthSciAST's `_lhs_role` unwraps faq and index nodes (`EarthSciAST/pkg/EarthSciAST.jl/src/classification.jl:77-88`, `:110-117`).
  - So `V` is not in `_observed_defs`, and `view_states` counts it as a solver slot (`system.jl:136-166`). Bands `∂f/∂V` are emitted.
  - At assembly `V` resolves to no source slot and its columns are skipped as ghosts (`assemble.jl:159-160`, `:265-266`). The derivative through `V` is silently missing.
- **Contracting observed (by reading; not executed).**
  - `_inline_once` β-reduces `index(S, e…)` for a faq-defined `S` by substituting the body for the output indices (`inline.jl:125-140`). It does not carry contracted ranges, `reduce` or `filter`.
  - reseact's `pbl_sum_O3[gi,gj] = Σ_gk dp·pbl_f·O3` (`reseact.esm/reseact.esm:8223`) is read through `pbl_mean_O3` by the O3 tendency.
  - Inlined, it drops the Σ and leaves `gk` free, or captured by an enclosing faq that binds the same name.
  - reseact builds band models only for its pointwise chemistry split, so this path has not been exercised. The planner's pass 4 runs on every operator, the PBL included.
- **Structure.** A contracted column index classifies as `:other`, hence `:general` (`structure.jl:9-30`, `:61-62`). A column contraction or sweep is therefore reported as general, and the planner cannot choose the column solvers of architecture §3.5 and §6.1.

### Where the constructs occur

Found by search; the paths below are under `EqWeFiC/components/` unless stated.

| Construct | Uses |
|---|---|
| **Recurrences** (16 equations, all in EqWeFiC) | **Madronich two-stream Thomas solve:** `gaschem/madronich/two_stream.esm` `bet` L3546, `uf` L3734, `yrev` L3843. A 138×138 tridiagonal per wavelength bin over 101 bins, swept along `row`; "back substitution … is written on the REVERSED row axis" (L5).<br>**RRTM LW:** `atmospheric_radiation/rrtm_lw/rtrn_sweep.esm` `rad_dn_top` L2167, `rad_up` L2745. Frame of 140 g-points × 104 levels; the downward sweep is top-counted.<br>**Dudhia SW:** `S_dn` (`dudhia_sw/dudhia_sw.esm:1659`), with filtered inner faq and `max` clamps.<br>**GWDO:** `taup_wave` (`gwdo/wave_saturation.esm:671`).<br>**New Tiedtke fall sweeps:** `ntiedtke/precipitation_flux.esm:1082`, `:1402`, `:1543`, `:1795`, `:1917`, `:2124`.<br>**sfclayrev secant:** an iteration axis of 33 cells with lags 1–5 (`sfclayrev/sfclayrev.esm:190`). |
| **Non-`"+"` reduce** | **`"*"` with a filter:** Dudhia `growth_factor` (`dudhia_sw.esm:262-300`), "homogeneous solution of the linear depletion recurrence".<br>**`"max"` with a filter:** New Tiedtke `s_stable` (`ntiedtke/column_initialization.esm:919`).<br>**`"min"` level searches:** YSU `first_stable_level` (`ysu/bulk_richardson.esm:37-46`); GWDO and Tiedtke `wmin` (`column_initialization.esm:1035`).<br>**`"max"`:** YSU `m_li` (`ysu/ysu.esm:3103`), Noah, Madronich, and the SCM coupling. |
| **Filtered `"+"`** | **reseact `Mz`,** an exclusive forward scan over levels (`reseact.esm/reseact.esm:4460-4495`).<br>**eqice:** `I`, `D` and `w3` (`eqice/stage2/sia.esm:276`, `:306`, `:891`).<br>**GWDO `ncrit`,** "a prefix reduction, not a recurrence" (`gwdo/wave_saturation.esm:546`). |

**Level searches feed state-dependent indices.**

- YSU's `kpbl1 = first_stable_level(Ri1, …)` (`ysu/ysu.esm:2062`) is read as `Ri1[kpbl1 − 1]` and `Ri1[kpbl1]` (`ysu.esm:2075`, `:2091`).
- `Ri` depends on the state, so these are reads whose index depends on the state.
- `dscalar` throws on them (`scalar_rules.jl:128-134`). That failure is loud, but YSU needs a rule.

**Authors prefer closed forms.** EqWeFiC's policy runs three tests before accepting a recurrence (`EqWeFiC/CLAUDE.md:37`). So a recurrence that survives into a document is a genuinely sequential sweep, and unrolling it is not an option.

**The Julia evaluator does not execute recurrences.** EarthSciAST.jl implements only the static half; Rust and Python execute them (causal RFC §6, L274-286; §6.1, L320-331). EarthSciASTDiff's Julia reference evaluates band coefficients through the stock Julia evaluator (`emit.jl:184-195`). So the Julia half of this RFC needs that binding debt paid first.

## Decisions of record

| # | Question | Decision | Why |
|---|---|---|---|
| 1 | Filter derivative | **The derivative of the filtered faq is the filtered faq of the derivative.** The filter's Dirac term is dropped. | This is spec §3.2's `ifelse` policy. A scan or window predicate is index-valued, so the drop is exact. |
| 2 | Filter representation | **Index-only filter: a new optional entry field `where`,** decided exactly at assembly and never in the coefficient. **Value-dependent filter: `ifelse(φ, d, 0)` in the coefficient,** with the pattern unioned. | A triangular window is not a rectangle, so `rows` and `contracted` boxes cannot express it. Padding the box violates region exactness (spec §2, `esm-jacobian-spec.md:48-55`). Splitting per row makes the entry count grow with the extent, against spec §4's guidance. |
| 3 | max/min derivative | **Select the first argmax in ascending contraction order.** Emit a contracted band whose coefficient is an indicator built from max/min faq nodes. | It matches the left-fold tie rule (spec §3.2) and the normative ascending accumulation order (`esm-spec.md:813-819`), and uses existing ops only. `argmax` stays reserved (spec §3.5). |
| 4 | Product derivative | **The product rule with exclusion products** `Π_{c'≠c}`, as filtered `"*"` faq nodes. Never divide. | Robust at zero factors. Dudhia's factors `(1+x)` can underflow. |
| 5 | Index-valued reductions | **A read `x[κ(u)]` with an integer-valued `κ` becomes a one-hot contracted band over the static range of that axis.** κ's own derivative is 0. | It is piecewise constant (spec §3.2's `floor` rule). The pattern is the structural union, global as spec §2 requires. |
| 6 | Recurrence values | **Tangent and adjoint recurrences are emitted as ESM recurrence definitions.** The adjoint is written ascending on a reversed axis. | It reuses each binding's executor; the spec admits ascending sweeps only (causal RFC §4). The document authors already reverse axes this way (`two_stream.esm:3843`, RRTM's `m`). |
| 7 | Recurrence pattern | **Derived symbolically:** the transitive closure of the lag graph along the axis, which is dense lower triangular per line in general. | Sparsity stays structural (spec §2). A lag-1 linear recurrence already has a dense inverse. |
| 8 | VJP for sweeps and scans | **Emit a matrix-free VJP model** (reverse sweeps) beside the band block. | Transposing materialized bands costs O(N²) per line where a reverse sweep costs O(N·L). Architecture §3.2 pass 6 assumes the band model "also serves Jᵀ·λ"; that is too slow here. |
| 9 | Structure | **New class `"column_block"`** with an axis and a per-line pattern: `lower`, `upper`, `dense`, or `banded(kl, ku)`. | It is what architecture §3.5 and §6.1 dispatch on. |
| 10 | Inliner | **Inline a contracting observed as a scalar-position faq,** keeping its ranges, `reduce` and `filter` and α-renaming the contracted names. **Recognize definitions by EarthSciAST's `_lhs_role`.** | It closes both silent paths. |

## Design

### 1. Filtered sums

Let `S[o] = Σ_{c ∈ C, φ(o,c)} b(o,c)` be a sum over a filtered window. Then

```
∂S[o]/∂x = Σ_{c ∈ C, φ(o,c)} ∂b(o,c)/∂x
```

A site whose column is independent of the contracted names gives an ordinary band whose coefficient is a filtered scalar faq: the existing rule, now keeping the filter (`bands.jl:228-239`). A gathered column gives a contracted band with the new field:

```json
{ "row": "q", "col": "divh_fix", "rows": [[1,13],[1,7],[1,73]], "row_idx": ["gi","gj","gke"],
  "contracted": [["gk", 1, 72]], "where": {"op": "<", "args": ["gk", "gke"]},
  "col_idx": ["gi","gj","gk"], "coef": "…" }
```

This is reseact's `Mz` on the forcing axis of U7 (the shapes are illustrative).

**`where` (spec §4, new).** It is a boolean expression over the entry's `row_idx` names, its `contracted` names, literals, metaparameters and `const` gathers. A point contributes iff it is true.

- Producers SHOULD split boxes at affine breakpoints first (`clip.jl:183`) and keep `where` only for non-rectangular windows.
- A predicate that reads a state, parameter or forcing value is not index-only. It goes into the coefficient as `ifelse(φ, d, 0)`, and the box is the full window: the structural union.
- **Accumulation order.** Entries and points accumulate in the band's own enumeration order (ascending). Bitwise agreement with the RHS's fold is not required, since the Jacobian is not a scan.

### 2. max, min and products

For `S[o] = max_{c ∈ W(o)} b(o,c)`, with `W` possibly filtered, let

- `M(o) = max_{c∈W(o)} b`;
- `M_<(o,c) = max_{c'∈W(o), c'<c} b(o,c')`, which is `−∞` on an empty window: the semiring identity (`esm-spec.md:796-799`).

Then

```
∂S[o]/∂x = Σ_{c∈W(o)} I(o,c) · ∂b(o,c)/∂x,     I(o,c) = (b(o,c) >= M(o)) and (b(o,c) > M_<(o,c))
```

`I` selects the first index attaining the maximum. This is exactly what spec §3.2's left fold `ifelse(acc >= x, dacc, dx)` gives for an n-ary `max`. `min` mirrors it with `<=` and `<`. The band is contracted, its pattern is the whole window, and its coefficient is `ifelse(I, d, 0)` with `M` and `M_<` as max-faq nodes in the coefficient.

For `P[o] = Π_{c∈W(o)} b(o,c)`:

```
∂P/∂x = Σ_c ∂b(o,c)/∂x · Π_{c'∈W(o), c'≠c} b(o,c')
```

The inner product is a `"*"` faq with filter `c' != c`. Dudhia's `growth_factor` (60 levels) costs O(N²) per column this way. That is acceptable because it is evaluated only when the Jacobian is.

**Index-valued reductions.** A level search `κ = min{k ∈ W : …}` has a body that is an index name, so its derivative is 0. Where `κ` is used as a subscript, `x[κ(u) + d]`, the site is replaced by the one-hot form:

```
∂x[κ+d]/∂x[k] = [k == κ(u) + d]     → contracted band over the static range of x's axis,
                                        coef ifelse(k == κ + d, outer, 0)
```

`inlinable_observed` clause 2 (`system.jl:250-257`) must stop requiring such a subscript to be build-time evaluable. The column is now the contracted `k`; `κ` moves into the coefficient. For YSU's `Ri1[kpbl1 − 1]` and `Ri1[kpbl1]` this costs 2×59 points per column, all but two of them zero.

### 3. Recurrences

**Setting.** `V[o, k] = g(o, k, {V[o, k−a]}_{a∈A}, x)`, where:

- `k` runs over `[lo, hi]` ascending;
- `o` are the non-recurrence frame axes, with lag 0;
- `A` is the set of derived lags;
- `x` are the targets (states, parameters, forcing) read through ordinary sites;
- the base-case guards are `ifelse(k <= …, base, …)` (causal RFC §4).

Consumers read `V` through ordinary sites. Write `G_a(o,k) = ∂g(o,k)/∂V[o,k−a]` and `B(o,k) = ∂g(o,k)/∂x`. These are scalar expressions given by the existing rules, applied branch-wise through the guards. They read the primal `V`.

**Tangent recurrence** (forward mode). It has the same axis, lags and guards, so it is well founded by causal RFC §3:

```
V̇[o,k] = Σ_{a∈A} G_a(o,k) · V̇[o,k−a] + B(o,k)·ẋ
```

**Adjoint recurrence** (reverse mode). Let `V̄[o,k]` be the cotangent arriving from consumers, `(∂f/∂V)ᵀλ`. The accumulated adjoint is

```
V̂[o,k] = V̄[o,k] + Σ_{a∈A} G_a(o,k+a) · V̂[o,k+a],      x̄ += Σ_{o,k} B(o,k)ᵀ · V̂[o,k]
```

This reads later positions, so it is emitted on the reversed axis `k̃ = lo + hi − k` as an ascending recurrence:

```
V̂rev[o,k̃] = V̄[o, lo+hi−k̃] + Σ_a ifelse(k̃−a >= lo, G_a(o, lo+hi−k̃+a) · V̂rev[o, k̃−a], 0)
```

The producer emits the guards explicitly, so `E_TREEWALK_RECUR_UNAVAILABLE` (causal RFC §4 point 5) can never fire in a derived model. A Thomas solve is a forward recurrence (`bet`, `uf`) composed with a reversed one (`yrev`). Its adjoint is the reversed composition, each part emitted by the same rule.

**Jacobian.** Along one line `o`, `dV/dx = (I − G)⁻¹ B` with `G` strictly lower triangular, with nonzeros at the lags `A`.

- `(I − G)⁻¹` is unit lower triangular and dense in general. A forward sweep therefore gives a dense lower-triangular block per line.
- A forward sweep composed with a reversed one gives a dense block. The inverse of a tridiagonal matrix is dense.
- **Pattern.** The producer derives it from the lag graph's transitive closure. It never probes.
- **Materialized values** come from a matrix tangent recurrence with one extra lag-0 output axis `j`, which causal RFC §3 rule 2 admits: `P[o,k,j] = Σ_a G_a(o,k)·P[o,k−a,j] + B(o,k,j)`.
- **Format.** A band through `V` then has a contracted `j` and a coefficient that reads `P`. `P`'s definition goes in a new block field, `auxiliary`: equations the consumer evaluates with the document's own, including recurrence semantics, before evaluating any coefficient.
- **Where the axis is not a state axis** (sfclayrev's iteration cells, Madronich's `wlbin`), the recurrence changes only coefficient evaluation, not the pattern. sfclayrev stays pointwise.

**VJP model** (Decision 8). Beside the block, the producer emits a derived model computing `Jᵀλ` matrix-free: reverse sweeps for recurrences, and reverse-axis forward scans for prefix sums. The transpose of an inclusive forward scan is a suffix scan. Emitted as a forward scan on the reversed axis, it is O(N) and bit-reproducible, which the spec recommends over a reverse filter (`esm-spec.md:827-851`).

### 4. Structure: `column_block`

`detect_structure(entries, sv; column_axis)` returns `:column_block` when every entry satisfies all three conditions:

1. Its `col_idx` equals the row index in every dimension other than `column_axis`.
2. In `column_axis` it is `row ± literal`, a contracted name ranging over that axis (with or without `where`), or a column reached through a recurrence on that axis.
3. No entry couples a scalar state to an array state.

The per-line pattern is the union of `lower`, `upper`, `dense` and `banded(kl, ku)` across entries. A `column_block` whose lines are all banded is the existing `:banded` case restricted to one axis.

The `factorization` registry (spec §6) gains:

```json
"factorization": { "type": "column_block", "axis": "lev", "line_length": 72,
                   "block_vars": ["O3", "NO"], "pattern": "dense" }
```

`column_axis` comes from the planner's locality pass (architecture §3.2 pass 2), since index-set names carry no vertical meaning.

**reseact's PBL mixing** gives `pbl_f·(mean − x)/τ`: a diagonal plus a rank-1 term per column. Its pattern is `dense`. Keeping the rank-1 form is Open question 2.

### 5. Costs (per line, arithmetic estimates)

| Operator | N | Materialized band points per (V, x) pair | VJP-model work |
|---|---|---|---|
| reseact `Mz` exclusive scan | 72 | 2,556 | 72 |
| Thomas forward + reversed, Madronich (per wavelength bin; 101 bins) | 138 | 19,044 (dense) | ≈ 3 × 138 |
| RRTM LW sweeps (per g-point; 140 g-points), contracted to the 59-level tendency | 104 | ≈ 5,400 (lower) | ≈ 2 × 104 |
| Dudhia `growth_factor` (`"*"`) | 60 | 1,830 points, each an O(N) exclusion product | 60 |

The integrator uses the VJP model for explicit and held-tendency adjoints. RRTM runs under `every` with `hold: "tendency"` (architecture §4.3), so it is never inside an implicit solve. It materializes bands only for column-implicit leaves (`column_block` LU) and for finite-difference validation.

### 6. Bindings

| Binding | Work |
|---|---|
| **Julia (reference)** | `where`; the max/min/`*` rules; one-hot index reads; the inliner fixes; recurrence detection (the `_lhs_role` semantics); tangent, adjoint and matrix-tangent emission; `column_block`; the VJP model. Values for recurrence fixtures wait on EarthSciAST.jl executing recurrences. The band and block goldens do not. |
| **Python** | The same calculus. Its numpy interpreter executes bare-LHS recurrences (causal RFC §6), so it can produce value goldens. |
| **Rust** | The same calculus after U10 W0. It executes recurrences (`AlgebraicRule::Recurrence`), so it produces recurrence value goldens until Julia can. |

## What this is not

- **Not the other relational constructs.** Joins (`on`), Skolem keys, `distinct`, `bool_and_or`, value-invention `argmin`/`argmax` and ragged sets beyond what contraction already handles stay reserved, and producers still fail on them.
- **Not implicit-function-theorem treatment of iterations.** sfclayrev's 10 secant updates are a fixed recurrence and are differentiated exactly as written, not as a converged root.
- **Not the `integral` op.** It is a rewrite target that must be lowered first (spec §3.3). EqWeFiC's `p_hyd_w` (`wrf_arw/column_geometry.esm:112`) reaches the calculus only after lowering.
- **Not a change to the recurrence or scan semantics of esm-spec.**

## Conformance and tests

1. **Fixtures:**
   - `filtered_scan.esm`: inclusive, exclusive and reverse scans; the `Mz` shape; a value-dependent filter.
   - `semiring_reduce.esm`: max and min with exact ties, `"*"` with a zero factor, a level search used as an index.
   - `recurrence_sweep.esm`: a 6-level Thomas solve with a reversed back substitution, a lag-2 recurrence with a base guard, and a non-state iteration axis.
   - `contracting_observed.esm`: the `pbl_sum` shape, with the same contracted name as the enclosing faq to catch capture.
   - `faq_lhs_observed.esm`.
2. **Goldens:** `.bands.json` and `.block.json` (with `where`, `auxiliary` and `column_block`), byte-identical in all three bindings.
3. **Value goldens (`.jvals.json`, 1e-10):**
   - non-recurrence fixtures: produced by Julia, with the ForwardDiff oracle;
   - recurrence fixtures: produced by Rust, and checked in Rust and Python against central differences at several step sizes. The fixtures are scaled to O(1), so a good `h` exists.
4. **Tie tests:** exact ties pick the earliest index.
5. **Tangent/adjoint dot products:** `⟨J v, λ⟩ = ⟨v, Jᵀλ⟩` between the tangent and VJP models, to 1e-12.
6. **Regressions for the silent paths:**
   - the faq-LHS observed must yield nonzero bands through `V`, or an error;
   - the contracting observed must not leave a free index name in any coefficient;
   - an assembled plan with an unresolvable column must throw.

## Work breakdown

1. **Spec:**
   - §3.5: the rules, with filters, max/min/`*` and index-valued reads removed from the reserved list;
   - §4: `where` and `auxiliary`;
   - §4.4: recurrences;
   - §6: `column_block`;
   - §5: the fixtures.
2. **EarthSciAST.jl:** recurrence execution (causal RFC §6.1 debt). This RFC files it upstream as a dependency.
3. **Julia:** the inliner fixes first, since they are silent-wrong today; then filters, the reductions, recurrences, structure and the VJP model.
4. **Python and Rust,** each after its catch-up (U10 W0 for Rust).
5. **Planner:** pass `column_axis` to `detect_structure`; emit VJP models for sweep-bearing operators (architecture §3.2 pass 6).

## Alternatives considered

- **Unroll recurrences per level.** Rejected. EqWeFiC's authoring rules forbid unfactored documents, and the causal RFC calls the hand-unrolled form "the mechanically generated, unfactored document this repo's authoring rules forbid" (§2.1, L133-136).
- **Treat a sweep as an implicit system** and apply the implicit function theorem to `(I − G)V = b`. Rejected, for the reason causal RFC §5 gives for the forward problem: forward substitution is exact, a solve is not, and the tangent recurrence is cheaper.
- **Pad triangular windows to rectangles** with `ifelse(j<i)` coefficients. Rejected: it breaks region exactness and doubles pattern and work.
- **Use `argmax`.** It is reserved, and it is a value-invention construct in the semiring RFC.
- **Enzyme on the RHS (decisions §1.3 item 2).** It remains Julia's fallback and oracle, but gives no Rust parity (D4).

## Open questions

1. **`where` predicates.** Should they admit `const`-table gathers (connectivity masks)? A gather is index-only, but harder to split at breakpoints.
2. **Diagonal plus rank-1.** Should `column_block` carry a `diag_plus_rank1` pattern for PBL-mean operators, so a solver can use Sherman–Morrison instead of a dense LU?
3. **Rosenbrock on sweep-bearing operators.** Second order through recurrences is out of scope (U8). Should the planner refuse a Rosenbrock adjoint on any operator with a recurrence?
4. **Clamped sweeps.** `max` and `min` clamps inside a recurrence body (Dudhia `S_dn`) make `G_a` piecewise. Is the a.e. derivative adequate for the M7 gates, or do those gates need FD-on-frozen-tape tolerances at clamp transitions?
