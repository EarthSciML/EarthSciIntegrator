---
title: "Time-derivative bands"
description: "A Rosenbrock step needs f_t = ∂f/∂t of every operator that depends on time explicitly, such as photolysis that follows the sun. EarthSciASTDiff's main already differentiates with respect to t in Julia, Python and Rust (wrt = time), but its spec still lists tgrad as out of scope and no cross-binding golden gates it. This RFC asks to make time-derivative bands a specified, golden-gated artifact, with the band shape, the serialized block, the evaluation document, and a clear statement of what the derivative leaves out (time dependence that enters through forcing data)."
---

Status: draft (proposed by EarthSciIntegrator, 2026-10-25); done upstream on EarthSciASTDiff `main` (EarthSciASTDiff#13, closing #11), to be released with EarthSciASTDiff 0.2
Target: EarthSciASTDiff (`esm-jacobian-spec.md` §1, §4 `"wrt"`, §5 fixtures, §7 scope list, §8 evaluation document; draft spec, no version number yet). Julia reference, Rust and Python ports.
Needed by: EarthSciIntegrator milestone M3 (reseact's chemistry runs `Rosenbrock23` with time-varying photolysis). Within M2 it would restore the Rosenbrock end-to-end cases on time-dependent toy problems that D80 leaves out.
Integrator refs: architecture.md §3.2 (pass 6), §5 (Rosenbrock); spec/operator-interface.md §3 (`ft`), §8.3; spec/plan.md §10; spec/toy-problems.md §4.2; decisions D8, D62, D80

File citations are relative to the `code/` directory that holds all the repositories, at EarthSciASTDiff `main` as of 2026-10-25 (`cbe768b`).

## Summary

Rosenbrock methods add $h\gamma_i f_t$ to every stage when the right-hand side depends on $t$ explicitly (spec/methods.md §4; D62 for Rosenbrock23's estimate). The integrator's operator interface therefore has a call `ft(out, u, p, t)` (spec/operator-interface.md §3), required for Rosenbrock leaves on operators with `depends_on_t`. Operators built by the planner get their derivatives from EarthSciASTDiff band models, so `ft` needs a band model of $\partial f/\partial t$ that the Julia and Rust planners emit byte-identically.

**Proposal.**

1. **Specify `wrt: "time"`** in `esm-jacobian-spec.md` §4: a band list whose single column is the scalar site `t`. Every entry has `col: "t"`, `col_idx: []`, and the usual `row`, `rows`, `row_idx`, `coef`. The meaning is §4's with $J$ an $N \times 1$ matrix.
2. **Move `tgrad` out of §7's out-of-scope list** and state the semantics already implemented: `datetime.*` differentiates to 0, `interp.linear` and `interp.bilinear` of `t` to the active-segment slope, and observeds are not inlined by the time-dependence seed (below).
3. **Golden-gate it.** Add a non-autonomous fixture, with `.bands.json`, `.block.json`, `.evaldoc.json` and `.jvals.json` goldens for `wrt = time`, compared byte for byte across Julia, Rust and Python as the state goldens are.
4. **State what $\partial f/\partial t$ omits.** Time dependence that enters through forcing data (two-record blend weights, data-updated parameters) is not in the band model. Say so normatively, and either (a) leave it to the consumer, or (b) add bands with respect to forcing-blend weights, which U7 (∂f/∂forcing) already covers.

## Motivation

**The need.** reseact's SuperFast chemistry runs `Rosenbrock23` (architecture §4.4). Its photolysis rates follow solar zenith angle computed from `t`, so the chemistry operator depends on `t` explicitly and its Rosenbrock stages need $f_t$. M2 already excludes 10 of the 51 M1 cases for this reason, all Rosenbrock leaves on `forced_decay` and `photostationary_cells` (spec/toy-problems.md §4.2). Without a time-derivative band model the Rust planner cannot provide `ft` at all, because Rust has no automatic differentiation (decisions §3.2 item 2).

**How close upstream already is.** Closer than the M2 spec first assumed:

- `jacobian_bands(sv; wrt = :time)` exists in Julia (`EarthSciASTDiff/pkg/EarthSciASTDiff.jl/src/jacobian.jl:17-57`), with `Wrt::Time` in Rust (`pkg/earthsci-astdiff-rs/src/jacobian.rs:14-66`) and `wrt="time"` in Python (`pkg/earthsci-astdiff-py/src/earthsci_astdiff/jacobian.py:25-44`). The targets are the single site `"t"`, through the same band calculus.
- Julia wires it into `ode_components`/`odefunction` as an analytical `tgrad!`, with a zero fill for autonomous systems (`src/ode.jl:36-60`). It is tested against a central difference in `t` and against `assemble_jacobian(…; wrt = :time)` (`test/ode_test.jl:76-125`). The README marks `tgrad` done.
- **Gaps:** `esm-jacobian-spec.md` §4 defines `"wrt"` as `"states"` or `"parameters"` only, and §7 still lists time gradients as out of scope. No golden pins time bands, so Julia, Rust and Python are not checked against each other. The evaluation document and scatter plans (§8) are specified for `wrt = states` only.

**Decisions of record.**

- D8: both planners emit byte-identical derived documents, so a planner can only rely on a derivative whose serialized bands are cross-binding golden-gated.
- D80: M2 derives no time-derivative document and leaves the time-dependent Rosenbrock cases to this RFC.

## Design

### 1. Bands with respect to time (spec §4)

`"wrt": "time"`. Each entry has `col: "t"`, `col_idx: []`, and no `contracted`. `rows`, `row_idx` and `coef` are as for state bands. The assembled object is a vector $f_t$ with $f_t[\text{row}(r)] \mathrel{+}= \text{coef}(r)$. Entry order follows the state bands' rule: equation order, and within an equation the single column.

### 2. What the derivative covers

- **Explicit `t`** in a right-hand side or in an observed it reads: differentiated.
- **Observeds.** The implementation deliberately does not prune observeds by a `{"t"}` seed (`jacobian.jl:52-57`, "time also enters implicitly, through externally-provided time-varying data"). The spec should say which observeds are inlined for `wrt = time`. The proposal is all observeds the right-hand side reads, transitively, so the explicit dependence is complete.
- **Implicit dependence through data**, such as two-record blend weights `w_time_*` that a provider sets at each refresh, or `data`-updated parameters: not covered. The spec should state that $\partial f/\partial t$ is the partial derivative with parameters held fixed. The integrator then treats a forcing bracket's time dependence separately: within an epoch the bracket's weights are model expressions of `t` (reseact's `frac`) and are covered; across epochs, steps do not cross a refresh (architecture §7.5).

### 3. Evaluation document and scatter (spec §8)

The evaluation document for `wrt = time` is the state one with `J_k` fields of the row's shape (no contracted dimensions arise from the scalar site). The only scatter layout needed is a dense vector in the canonical state order; no new layout is required.

### 4. Golden fixture

`tests/valid/tgrad_photolysis.esm`: a per-cell reaction pair with $j(t) = j_0 \max(0, \cos(\omega t + \varphi_i))$, one `interp.linear` of `t`, and one autonomous row. Goldens: `tgrad_photolysis.time.bands.json`, `.time.block.json`, `.time.evaldoc.json`, and `.time.jvals.json` at a pinned state (Julia-produced, exact key set, 1e-10 relative).

## Alternatives considered

- **Treat `t` as a parameter.** A consumer could rewrite every `t` in a right-hand side into a parameter `τ`, set to $t$ at each evaluation, and take EarthSciASTDiff's existing `wrt = :parameters` bands. Scalar parameter bands exist in all three bindings today (`jacobian.jl:44-47`). The integrator could do this without upstream work. It is listed as the fallback, not the proposal, because (i) the planner would have to rewrite expressions inside `datetime.*` and `interp.*` calls, whose derivative rules are already special-cased for `t`; (ii) parameter bands are not golden-gated either, so the cross-binding gap is the same; and (iii) it duplicates a capability upstream already has.
- **Finite differences in `t`** in each runtime. This needs no upstream work, but in Rust the forward step would then use an approximate $f_t$. The planned operator's results would differ from a ForwardDiff or analytical $f_t$ at about 1e-8 relative, and the M2 `toy_equivalence` tolerances (1e-10 on fixed-step cases) would fail.
- **Augment the state with $\tau' = 1$.** This makes $f_t$ a column of the state Jacobian, but it changes the state vector, error norm and step-size control of every leaf, which spec/methods.md fixes.

## Conformance and tests

- The new goldens are byte-identical across Julia, Rust and Python.
- Julia's existing finite-difference test generalizes to every fixture with explicit `t`.
- The integrator side is a derived kind `time_derivative` (`<operator>.time_derivative.esm`), the planner's `ft` call (spec/operator-interface.md §8.3), and the M2 cases restored in `tests/conformance/m2/make_cases.py`.

## Work breakdown

1. Spec text for §1, §4, §7, §8 (small).
2. Fixture and four goldens, Julia-generated (small).
3. Rust and Python against the goldens; fix whatever differs (unknown, likely small: the code exists in all three).

## Open questions

1. Should the forcing-blend time dependence (§2) be covered here, or left to U7's forcing bands?
2. Should `ode_components`' zero fill for autonomous systems become a spec rule ("an empty time band list means $f_t \equiv 0$"), so consumers can skip the evaluation?
