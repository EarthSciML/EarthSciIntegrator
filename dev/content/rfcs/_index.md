---
title: "Upstream RFC drafts"
weight: 40
---

**Status:** drafts, 2026-10-13, filed upstream as issues on 2026-10-23 (the Issue column). Needed-by dates updated 2026-10-14 for milestone M4b (D37). Upstream progress updated 2026-10-26 (the table below the list). The 2026-10-14 reuse audit (survey §6) verified that each ask names a real gap: none of U1–U15 duplicates existing upstream code.

These are the changes EarthSciIntegrator needs in sibling repos. Each draft follows the EarthSciAST RFC format and cites current upstream code by `file:line`. Anything a draft could not verify is labelled as an assumption. The integrator-side context is in [`../architecture.md`](/architecture/) §13 and [`../decisions.md`](/decisions/).

| # | Target | Draft | Issue | Needed by |
|---|---|---|---|---|
| U1 | EarthSciAST spec | [`solver.external`](/rfcs/U01-solver-external/): a document points at its run config (esm 1.3.0) | [EarthSciAST#538](https://github.com/EarthSciML/EarthSciAST/issues/538) | M2 |
| U2 | EarthSciAST flatten | [Per-term provenance](/rfcs/U02-term-provenance/): `equation_contributions` beside flattened equations | [EarthSciAST#539](https://github.com/EarthSciML/EarthSciAST/issues/539) | M2 |
| U3 | EarthSciAST runtimes | [Range-restricted RHS](/rfcs/U03-range-restricted-rhs/): operator subsets over tiles with halos, and lanes; bitwise assembly contract | [EarthSciAST#540](https://github.com/EarthSciML/EarthSciAST/issues/540) | M3, M5 |
| U4 | EarthSciAST runtimes | [Reentrant RHS](/rfcs/U04-rust-public-threadsafe-rhs/): Rust `RhsProgram` (`Send + Sync`), plus a reentrant Julia entry | [EarthSciAST#541](https://github.com/EarthSciML/EarthSciAST/issues/541) | M3 |
| U5 | EarthSciAST | [Float32 integration](/rfcs/U05-float32-integration/): what a declared Float32 governs, and what the integrator owns | [EarthSciAST#542](https://github.com/EarthSciML/EarthSciAST/issues/542) | M5g, M7 |
| U6 | EarthSciAST | [Strip and expose](/rfcs/U06-strippable-constructs-and-observed-kernels/): events, parameter updates and algebraic equations returned to the caller; standalone observed kernels | [EarthSciAST#543](https://github.com/EarthSciML/EarthSciAST/issues/543) | M4b (eqice on shared memory, D37) |
| U7 | EarthSciASTDiff | [Forcing and array-parameter columns](/rfcs/U07-forcing-jacobian-bands/) | [EarthSciASTDiff#1](https://github.com/EarthSciML/EarthSciASTDiff/issues/1) | M4 |
| U8 | EarthSciASTDiff | [Second-order bands](/rfcs/U08-second-order-bands/) for Rosenbrock discrete adjoints | [EarthSciASTDiff#2](https://github.com/EarthSciML/EarthSciASTDiff/issues/2) | M4 |
| U9 | EarthSciASTDiff | [Filtered faq, non-`+` reductions, recurrences](/rfcs/U09-faq-filters-semirings-recurrences/); `column_block` structure; VJP models | [EarthSciASTDiff#3](https://github.com/EarthSciML/EarthSciASTDiff/issues/3) | M4, M4b, M7 |
| U10 | EarthSciASTDiff | [Production-rate Rust assembly](/rfcs/U10-rust-production-assembly/), after a catch-up step | [EarthSciASTDiff#4](https://github.com/EarthSciML/EarthSciASTDiff/issues/4) | M3 |
| U11 | EarthSciIO | [Async prefetch and rank windows](/rfcs/U11-async-prefetch-and-rank-windows/) | [EarthSciIO#8](https://github.com/EarthSciML/EarthSciIO/issues/8) | M5 (prefetch from M3) |
| U12 | EarthSciIO | [Rust streaming region writer](/rfcs/U12-rust-streaming-region-writer/): commit protocol, indexed records | [EarthSciIO#9](https://github.com/EarthSciML/EarthSciIO/issues/9) | M4, M5 |
| U13 | EarthSciAST.jl (Reactant ext) | [Traceable per-operator StableHLO](/rfcs/U13-traceable-per-operator-stablehlo/) for the Julia GPU path | [EarthSciAST#544](https://github.com/EarthSciML/EarthSciAST/issues/544) | M5g |
| U14 | EarthSciAST spec | [Declared vs unspecified precision](/rfcs/U14-numerical-vs-semantic-precision/) (D29) | [EarthSciAST#545](https://github.com/EarthSciML/EarthSciAST/issues/545) | M5g, M7 |
| U15 | EarthSciAST.jl | [Recurrence execution](/rfcs/U15-julia-recurrence-execution/), native and StableHLO | [EarthSciAST#546](https://github.com/EarthSciML/EarthSciAST/issues/546) | M7 |
| U16 | EarthSciAST spec | [`solver.time_span`](/rfcs/U16-solver-time-span/): the run interval a document asks for (D60) | [EarthSciAST#550](https://github.com/EarthSciML/EarthSciAST/issues/550) | M2 |
| U17 | EarthSciASTDiff | [Time-derivative bands](/rfcs/U17-time-derivative-bands/): ∂f/∂t as a specified, golden-gated band model, for Rosenbrock's `f_t` | [EarthSciASTDiff#11](https://github.com/EarthSciML/EarthSciASTDiff/issues/11) | M3 (restores M2's time-dependent Rosenbrock cases) |

**Upstream progress** (2026-10-26):

| # | State |
|---|---|
| U1 | Done: released in EarthSciAST v0.4.0 (esm 1.3.0), esm-spec §2.2.6, with the location accessor. |
| U2 | Done: released in EarthSciAST v0.4.0, `equation_contributions` (esm-libraries-spec §4.7.5), plus Rust's `equation_lifted`. |
| U3 | Done: released in EarthSciAST v0.5.0 (EarthSciAST#557; esm stays 1.3.0), range-restricted right-hand sides over tiles and lanes in Julia and Rust (API_SPEC §5.8, CONFORMANCE_SPEC §5.50). Published 2026-10-26 to Julia's General registry, crates.io and PyPI. |
| U4 | Done: released in EarthSciAST v0.5.0 (EarthSciAST#561), `rhs_program` with forcing sets in Julia and Rust's `RhsProgram` (`Send + Sync`) (API_SPEC §5.8, CONFORMANCE_SPEC §5.51). Rust's M2 end-to-end runs can use it (spec/toy-problems.md §4). |
| U10 | Merged on EarthSciASTDiff `main`, closing EarthSciASTDiff#4: catch-up (W0, #5), trimmed evaluation document (W1, #8), scatter plans (W3, #10), Rust production assembly (W2, W4, W5, #12). To be released with EarthSciASTDiff 0.2. |
| U16 | Done: released in EarthSciAST v0.4.0 (esm 1.3.0), esm-spec §2.2.7, with `resolve_time_span` in every binding. |
| U17 | Done on EarthSciASTDiff `main` (EarthSciASTDiff#13, closing #11): `esm-jacobian-spec.md` §4.2 specifies `wrt: "time"` bands, §8 their evaluation document, and the `tgrad_photolysis` goldens gate them in Julia, Rust and Python. To be released with EarthSciASTDiff 0.2. |
| others | Open. EarthSciASTDiff 0.2 (its Rust crate is on EarthSciAST 0.6; its Julia package is 0.1.0 with compat EarthSciAST "0.4, 0.5" and SciMLBase "2", so it cannot yet share an environment with EarthSciIntegrator.jl, spec/plan.md §10 item 12) is expected before M2's stage 2; spec/plan.md is written against its `main`. |

**Cross-cutting prerequisites** (architecture §13):
- esm 1.3.0 for the three model repos.
- EarthSciAST's unmerged `fix/f59-mounted-leaf-precision` branch (for U14).
- A spec clarification permitting batched recurrence positions (for U15).
- The EarthSciASTDiff Rust/Python catch-up (U10 step W0).
