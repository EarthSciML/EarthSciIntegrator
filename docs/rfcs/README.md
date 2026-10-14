# Upstream RFC drafts

**Status:** drafts, 2026-10-13, for review before anything is filed upstream.

These are the changes EarthSciIntegrator needs in sibling repos. Each draft follows the EarthSciAST RFC format and cites current upstream code by `file:line`. Anything a draft could not verify is labelled as an assumption. The integrator-side context is in [`../architecture.md`](../architecture.md) §13 and [`../decisions.md`](../decisions.md).

| # | Target | Draft | Needed by |
|---|---|---|---|
| U1 | EarthSciAST spec | [`solver.external`](U01-solver-external.md): a document points at its run config (esm 1.3.0) | M2 |
| U2 | EarthSciAST flatten | [Per-term provenance](U02-term-provenance.md): `equation_contributions` beside flattened equations | M2 |
| U3 | EarthSciAST runtimes | [Range-restricted RHS](U03-range-restricted-rhs.md): operator subsets over tiles with halos, and lanes; bitwise assembly contract | M3, M5 |
| U4 | EarthSciAST runtimes | [Reentrant RHS](U04-rust-public-threadsafe-rhs.md): Rust `RhsProgram` (`Send + Sync`), plus a reentrant Julia entry | M3 |
| U5 | EarthSciAST | [Float32 integration](U05-float32-integration.md): what a declared Float32 governs, and what the integrator owns | M5g, M7 |
| U6 | EarthSciAST | [Strip and expose](U06-strippable-constructs-and-observed-kernels.md): events, parameter updates and algebraic equations returned to the caller; standalone observed kernels | M6 |
| U7 | EarthSciASTDiff | [Forcing and array-parameter columns](U07-forcing-jacobian-bands.md) | M4 |
| U8 | EarthSciASTDiff | [Second-order bands](U08-second-order-bands.md) for Rosenbrock discrete adjoints | M4 |
| U9 | EarthSciASTDiff | [Filtered faq, non-`+` reductions, recurrences](U09-faq-filters-semirings-recurrences.md); `column_block` structure; VJP models | M4, M6, M7 |
| U10 | EarthSciASTDiff | [Production-rate Rust assembly](U10-rust-production-assembly.md), after a catch-up step | M3 |
| U11 | EarthSciIO | [Async prefetch and rank windows](U11-async-prefetch-and-rank-windows.md) | M5 (prefetch from M3) |
| U12 | EarthSciIO | [Rust streaming region writer](U12-rust-streaming-region-writer.md): commit protocol, indexed records | M4, M5 |
| U13 | EarthSciAST.jl (Reactant ext) | [Traceable per-operator StableHLO](U13-traceable-per-operator-stablehlo.md) for the Julia GPU path | M5g |
| U14 | EarthSciAST spec | [Declared vs unspecified precision](U14-numerical-vs-semantic-precision.md) (D29) | M5g, M7 |
| U15 | EarthSciAST.jl | [Recurrence execution](U15-julia-recurrence-execution.md), native and StableHLO | M7 |

**Cross-cutting prerequisites** (architecture §13):
- esm 1.3.0 for the three model repos.
- EarthSciAST's unmerged `fix/f59-mounted-leaf-precision` branch (for U14).
- A spec clarification permitting batched recurrence positions (for U15).
- The EarthSciASTDiff Rust/Python catch-up (U10 step W0).
