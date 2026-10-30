# Upstream RFC drafts

**Status:** drafts, 2026-10-13, filed upstream as issues on 2026-10-23 (the Issue column). Needed-by dates updated 2026-10-14 for milestone M4b (D37). Upstream progress updated 2026-10-26 (the table below the list). The 2026-10-14 reuse audit (survey §6) verified that each ask names a real gap: none of U1–U15 duplicates existing upstream code.

These are the changes EarthSciIntegrator needs in sibling repos. Each draft follows the EarthSciAST RFC format and cites current upstream code by `file:line`. Anything a draft could not verify is labelled as an assumption. The integrator-side context is in [`../architecture.md`](../architecture.md) §13 and [`../decisions.md`](../decisions.md).

| # | Target | Draft | Issue | Needed by |
|---|---|---|---|---|
| U1 | EarthSciAST spec | [`solver.external`](U01-solver-external.md): a document points at its run config (esm 1.3.0) | [EarthSciAST#538](https://github.com/EarthSciML/EarthSciAST/issues/538) | M2 |
| U2 | EarthSciAST flatten | [Per-term provenance](U02-term-provenance.md): `equation_contributions` beside flattened equations | [EarthSciAST#539](https://github.com/EarthSciML/EarthSciAST/issues/539) | M2 |
| U3 | EarthSciAST runtimes | [Range-restricted RHS](U03-range-restricted-rhs.md): operator subsets over tiles with halos, and lanes; bitwise assembly contract | [EarthSciAST#540](https://github.com/EarthSciML/EarthSciAST/issues/540) | M3, M5 |
| U4 | EarthSciAST runtimes | [Reentrant RHS](U04-rust-public-threadsafe-rhs.md): Rust `RhsProgram` (`Send + Sync`), plus a reentrant Julia entry | [EarthSciAST#541](https://github.com/EarthSciML/EarthSciAST/issues/541) | M3 |
| U5 | EarthSciAST | [Float32 integration](U05-float32-integration.md): what a declared Float32 governs, and what the integrator owns | [EarthSciAST#542](https://github.com/EarthSciML/EarthSciAST/issues/542) | M5g, M7 |
| U6 | EarthSciAST | [Strip and expose](U06-strippable-constructs-and-observed-kernels.md): events, parameter updates and algebraic equations returned to the caller; standalone observed kernels | [EarthSciAST#543](https://github.com/EarthSciML/EarthSciAST/issues/543) | M4b (eqice on shared memory, D37) |
| U7 | EarthSciASTDiff | [Forcing and array-parameter columns](U07-forcing-jacobian-bands.md) | [EarthSciASTDiff#1](https://github.com/EarthSciML/EarthSciASTDiff/issues/1) | M4 |
| U8 | EarthSciASTDiff | [Second-order bands](U08-second-order-bands.md) for Rosenbrock discrete adjoints | [EarthSciASTDiff#2](https://github.com/EarthSciML/EarthSciASTDiff/issues/2) | M4 |
| U9 | EarthSciASTDiff | [Filtered faq, non-`+` reductions, recurrences](U09-faq-filters-semirings-recurrences.md); `column_block` structure; VJP models | [EarthSciASTDiff#3](https://github.com/EarthSciML/EarthSciASTDiff/issues/3) | M4, M4b, M7 |
| U10 | EarthSciASTDiff | [Production-rate Rust assembly](U10-rust-production-assembly.md), after a catch-up step | [EarthSciASTDiff#4](https://github.com/EarthSciML/EarthSciASTDiff/issues/4) | M3 |
| U11 | EarthSciIO | [Async prefetch and rank windows](U11-async-prefetch-and-rank-windows.md) | [EarthSciIO#8](https://github.com/EarthSciML/EarthSciIO/issues/8) | M5 (prefetch from M3) |
| U12 | EarthSciIO | [Rust streaming region writer](U12-rust-streaming-region-writer.md): commit protocol, indexed records | [EarthSciIO#9](https://github.com/EarthSciML/EarthSciIO/issues/9) | M4, M5 |
| U13 | EarthSciAST.jl (Reactant ext) | [Traceable per-operator StableHLO](U13-traceable-per-operator-stablehlo.md) for the Julia GPU path | [EarthSciAST#544](https://github.com/EarthSciML/EarthSciAST/issues/544) | M5g |
| U14 | EarthSciAST spec | [Declared vs unspecified precision](U14-numerical-vs-semantic-precision.md) (D29) | [EarthSciAST#545](https://github.com/EarthSciML/EarthSciAST/issues/545) | M5g, M7 |
| U15 | EarthSciAST.jl | [Recurrence execution](U15-julia-recurrence-execution.md), native and StableHLO | [EarthSciAST#546](https://github.com/EarthSciML/EarthSciAST/issues/546) | M7 |
| U16 | EarthSciAST spec | [`solver.time_span`](U16-solver-time-span.md): the run interval a document asks for (D60) | [EarthSciAST#550](https://github.com/EarthSciML/EarthSciAST/issues/550) | M2 |
| U17 | EarthSciASTDiff | [Time-derivative bands](U17-time-derivative-bands.md): ∂f/∂t as a specified, golden-gated band model, for Rosenbrock's `f_t` | [EarthSciASTDiff#11](https://github.com/EarthSciML/EarthSciASTDiff/issues/11) | M3 (restores M2's time-dependent Rosenbrock cases) |

**Upstream progress** (2026-10-26):

| # | State |
|---|---|
| U1 | Done: released in EarthSciAST v0.4.0 (esm 1.3.0), esm-spec §2.2.6, with the location accessor. |
| U2 | Done: released in EarthSciAST v0.4.0, `equation_contributions` (esm-libraries-spec §4.7.5), plus Rust's `equation_lifted`. |
| U3 | Done: released in EarthSciAST v0.5.0 (EarthSciAST#557; esm stays 1.3.0), range-restricted right-hand sides over tiles and lanes in Julia and Rust (API_SPEC §5.8, CONFORMANCE_SPEC §5.50). Published 2026-10-26 to Julia's General registry, crates.io and PyPI. |
| U4 | Done: released in EarthSciAST v0.5.0 (EarthSciAST#561), `rhs_program` with forcing sets in Julia and Rust's `RhsProgram` (`Send + Sync`) (API_SPEC §5.8, CONFORMANCE_SPEC §5.51). Rust's M2 end-to-end runs can use it (spec/toy-problems.md §4). |
| U10 | Merged on EarthSciASTDiff `main`, closing EarthSciASTDiff#4: catch-up (W0, #5), trimmed evaluation document (W1, #8), scatter plans (W3, #10), Rust production assembly (W2, W4, W5, #12). Released in EarthSciASTDiff 0.1.0, its first registered release (commit 743d140; Julia General registration pending as of 2026-10-30, crates.io to follow). |
| U7 | Merged on EarthSciASTDiff `main` (EarthSciASTDiff#28, closing #1), after the 0.1.0 commit: forcing-buffer and array-parameter columns in Julia, Rust and Python (`esm-jacobian-spec.md` §4.3). Not yet released. |
| U16 | Done: released in EarthSciAST v0.4.0 (esm 1.3.0), esm-spec §2.2.7, with `resolve_time_span` in every binding. |
| U17 | Done on EarthSciASTDiff `main` (EarthSciASTDiff#13, closing #11): `esm-jacobian-spec.md` §4.2 specifies `wrt: "time"` bands, §8 their evaluation document, and the `tgrad_photolysis` goldens gate them in Julia, Rust and Python. Released in EarthSciASTDiff 0.1.0, its first registered release (commit 743d140; Julia General registration pending as of 2026-10-30, crates.io to follow). |
| others | Open. EarthSciASTDiff 0.1.0 (commit 743d140), the release M2's stage 2 needs, accepts EarthSciAST 0.6 and SciMLBase 2 and 3 in Julia (EarthSciASTDiff#19, closing #17) and is on EarthSciAST 0.6 in Rust; spec/plan.md is written against it. |

**Cross-cutting prerequisites** (architecture §13):
- esm 1.3.0 for the three model repos.
- EarthSciAST's unmerged `fix/f59-mounted-leaf-precision` branch (for U14).
- A spec clarification permitting batched recurrence positions (for U15).
- The EarthSciASTDiff Rust/Python catch-up (U10 step W0).
