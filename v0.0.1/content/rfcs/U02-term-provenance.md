---
title: "Per-term provenance through flatten: which component wrote each contribution"
description: "Flatten sums operator_compose and couple contributions into one right-hand side and forgets where each summand came from. This adds an ordered contribution list beside every flattened equation, each entry carrying the component (or coupling entry) that authored it. The list lives outside the expression tree, so canonical form, interning, CSE and equality are untouched. It is a FlattenedSystem field, not a document-format change."
---

Status: draft (proposed by EarthSciIntegrator, 2026-10-13); done upstream: released in EarthSciAST v0.4.0 (esm 1.3.0)
Target: EarthSciAST esm-libraries-spec §4.7.5 (canonical `FlattenedSystem` field set) and esm-spec §10.7; no `.esm` format version change
Needed by: EarthSciIntegrator milestone M2
Integrator refs: architecture.md §2, §3.1, §3.2 (passes 1 and 3), §4.2; decisions D7, D8, D14

## Summary

Flatten merges component equations into one equation per state. An
`operator_compose` match builds `rhs_A + factor * rhs_B`
(esm-libraries-spec §4.7.1 step 4). A `couple` additive connector folds its
expression onto the target tendency (§4.7.2). After that, nothing records which
summand came from which component. The integrator's planner (D14) splits each
right-hand side into operators. It needs that record.

This RFC adds one field to the canonical `FlattenedSystem`:
`equation_contributions`, parallel to `equations`. Entry *i* is the ordered
list of contributions whose sum is equation *i*'s right-hand side. Each
contribution holds its expression and its origins: the authoring component's
path, or the index of the coupling entry that authored it. Every additive term
the planner later finds inside a contribution inherits that contribution's
origins.

The record sits beside the expression tree, never on a node. Expression
equality, hashing, interning, canonical JSON and CSE keys do not see it.

## Motivation

**Flatten drops the information today.** Julia keeps a per-equation owner
while it flattens (`EarthSciAST.jl/src/flatten.jl:1050-1056`), but it is a
local vector, used only to tell A's equations from B's, and it is not returned.
The merge itself is `OpExpr("+", [eq.rhs; terms...])`
(`src/coupling_apply.jl:520`). Rust sums with a binary `sum_exprs`
(`earthsci-ast-rs/src/flatten.rs:3457`, `:3924-3932`) and records authorship
only as which per-system block an equation sits in (`flatten.rs:3744-3749`).
`FlattenMetadata` holds `source_systems` and `coupling_rules_applied` as
document-level lists (Julia `flatten.jl:49-55`; Rust `flatten.rs:402-439`).
Python's `FlattenedEquation` carries one `source_system` string per equation
(`earthsci-ast-py/src/earthsci_ast/flatten.py:321-331`). After a merge it keeps
`systems[0]`'s label (`flatten.py:1733-1737`).

**One label per equation is not provenance.** The cross-binding flatten corpus
shows it. For `tests/coupling/chained_operator_compose.esm` the golden records

```
D(Chem.O3)/Dt = Src.e + (-Sink.kd) * Chem.O3 + (-Chem.k1) * Chem.O3 + (-Adv.wind) * grad(Chem.O3)
source_system: "Src"
```

(`tests/conformance/flatten/cases.json:3824-3827`). Four components wrote that
sum. The record names one.

**Names do not recover it.** decisions.md §3.2 item 1 already notes that
name-based recovery fails when a term references two components. That is the
common case for coupled sinks. In the golden above, `(-Sink.kd) * Chem.O3`
reads `Sink` and `Chem`. In reseact the dry-deposition sink is authored by
`SuperFastDepositionSink` as `-k_O3 * O3`
(`EarthSciModels/components/atmospheric_deposition/superfast_deposition_sink.esm:29`).
After `variable_map` substitution it reads `DryDepositionGas.k_O3` and
`SuperFast.O3`. Neither is its author.

**Locality does not recover it either.** reseact composes four sources onto
each species: the `SuperFast` mechanism, `SuperFastDepositionSink`,
`SuperFastEmissionSource` and `Transport3D` (`reseact.esm/reseact.esm:12609`,
`:12701`, `:12709`). The first three are all pointwise. A run config that wants
emissions or deposition in their own operator, instead of with chemistry, can
only say so by component. The prototype shows how much effort locality-only
splitting costs. `prototypes/reseact_3d_chem/split_common.jl` grew three
heuristics on top of EarthSciASTSplitter's rule:

- following observeds, because the air-mass term `divh_fix` landed in the
  chemistry half (`split_common.jl:26-46`);
- detecting column contractions, because the PBL mean went to chemistry and
  silently broke the block-diagonal Jacobian (`split_common.jl:79-111`);
- detecting unlowered spatial `D`, because vertical mixing went to chemistry
  and the build died with `unlowered_operator` (`split_common.jl:124-146`).

Each was a misrouting found after the fact. Provenance does not replace
locality analysis (the planner still needs it, architecture §3.2 pass 2). It
gives the run config a second, authorial key that cannot be fooled by how a
term happens to be spelled.

**What the Splitter does now.** EarthSciASTSplitter splits a right-hand side
through n-ary `+`, binary `-` and unary `-` (`EarthSciASTSplitter.jl:111-127`),
splits a lifted equation's inner `faq` body (`:399-430`), and copies observeds
into every part (`:491-496`). It never guesses a component; its `references`
helper only tests which names a term reads (`:145-169`). D14 folds this into
the planner, which needs one more input per term: its origin.

**Where it matters, and where it does not.** U2 is on the reseact critical
path (M2/M3). eqice has no `operator_compose` at all in `stage2/`; its PISM
substeps are separate models wired by `variable_map`
(`eqice/PLAN.md:194-205`). EqWeFiC's single-column assembly also has none: 333
`variable_map` and 6 `coupling_import` entries
(`EqWeFiC/couplings/eqweather_scm.esm`). There, the assembly component
`EqWeatherScm` sums the tendencies in observed equations, e.g.
`rthraten = dthdt_lw + dthdt_sw` and
`dtheta_dt_phys = rth_phys_sum + mp_dtheta_dt` (`eqweather_scm.esm:2939`,
`:3317`). Every summand is authored by `EqWeatherScm`. Each is a bare reference
whose owner (`Pbl`, `Heat`, `Sw`) survives in its namespaced name. See *Open
questions* 3.

## Decisions of record

| # | Question | Decision | Why |
|---|---|---|---|
| 1 | Where does provenance live? | **A side list parallel to `equations`, never on an expression node** | Both canonical emitters fail closed on any non-pinned node field (Julia `canonicalize.jl:327-370`; Rust `canonicalize.rs:85-113`). Interning treats every emitted field as identity (`intern.jl:26-37`; Rust `intern.rs:257-260`), so a node field would either stop sharing identical subtrees or drop one of two annotations. Bare-name and literal terms are JSON strings and numbers with no slot for a field. |
| 2 | Unit of annotation | **A contribution**: one per (target equation, contributing source). Additive terms inside it inherit its origins. | Flatten knows sources, not terms. Term decomposition is the planner's (D14). One record per source is deterministic across bindings. |
| 3 | Origin value | **Exactly one of** `{"component": "<dotted path>"}` **or** `{"coupling": <index>}` | The authoring component for anything a component wrote, including the B side of `operator_compose`. A coupling index only for expressions a coupling entry wrote itself. Merges add no coupling index, so labels do not depend on `systems` order. |
| 4 | Non-additive merges | **Collapse to one contribution whose `origins` is the union, in first-appearance order** | A multiplicative `couple` gives `(Σ c) * m`. Distributing would change the arithmetic. The planner must treat the result as indivisible. |
| 5 | Format change? | **No.** A `FlattenedSystem` field and its corpus serialization only. | No document carries it. No version gate. Embedding in `.esm` is deferred (*Open questions* 2). |
| 6 | Bindings | **All five** | Flatten is Core tier (§4.7.5). The change is a rewrite, and CONFORMANCE_SPEC §5.33 excludes no binding for a rewrite-only change. Python is the flatten-corpus oracle (`cases.json` `$comment`), so it cannot be skipped. |

## Design

### The field

`FlattenedSystem.equation_contributions` is a list with one entry per entry of
`equations`, in the same order. Each entry is a non-empty ordered list of:

```
Contribution { expr: Expression, origins: [Origin, ...] }      // origins non-empty, deduplicated
Origin       = { "component": "<dotted component path>" } | { "coupling": <non-negative integer> }
```

`component` is the full dotted path of the model, subsystem or reaction system
whose equation contained the expression, as namespaced by step 2
(`Atmosphere.Chemistry`). `coupling` is the 0-based index into the coupling
sequence *after* `coupling_import` expansion (§4.7.5, between steps 2 and 3).
That is the same sequence `metadata.coupling_rules_applied` lists.

### The invariant

Let `body(eq)` be `eq.rhs`, or `eq.rhs.expr_body` when the pointwise lift has
wrapped the equation in a `faq` (the shape the Splitter already recognizes,
`EarthSciASTSplitter.jl:399-409`). Then:

> `body(eq)` evaluates to the left-to-right sum of the contributions'
> expressions, at every state, bit for bit.

The tree shape of `rhs` is not newly constrained. Julia may keep n-ary `+`,
Rust and Python their binary fold. All three already append contributions left
to right, so the invariant holds without changing any right-hand side.
(Assumption, not verified here: every evaluator folds an n-ary `+` left to
right. The invariant test in *Conformance* checks it.) A binding builds `rhs`
from its contribution list, not the other way round.

### How flatten fills it, step by step

| Step (§4.7.5) | Effect on contributions |
|---|---|
| 1. Reaction systems → ODEs | One contribution per derived equation, origin the reaction system. |
| 2. Namespace | One contribution per collected equation, origin the authoring component. Renames apply inside `expr`. |
| 3. `operator_compose` match | Append B's contributions to A's, in order. Origins unchanged. `factor ≠ 1` wraps each appended `expr` as `factor * expr`. `_var` clones carry B's component. |
| 3. `couple` additive | Append `{expr, [{"coupling": k}]}`. A new `D(to)` equation gets that as its only contribution. |
| 3. `couple` multiplicative | Collapse: one contribution, `expr = (Σ c) * m`, origins = union of prior origins, then `{"coupling": k}`. |
| 3. `couple` replacement | Replace the list with `{expr, [{"coupling": k}]}`. |
| 3. `couple` plain `{lhs, rhs}` | New equation, one contribution, `{"coupling": k}`. |
| 3. `variable_map` | Substitution happens inside each `expr`; origins unchanged. `additive` appends `{"coupling": k}`; `multiplicative` collapses as above; an expression transform's new observed gets `{"coupling": k}`. |
| Merged-away rename (§4.7.1 step 4, CONFORMANCE §5.35) | Applies inside each `expr`. |
| 3b. Pointwise lift | Per-cell indexing applies inside each `expr`; the `faq` wrapper goes around the sum. |
| 3a. RHS `D(x, t)` resolution | The resolved tendency replaces the `D` node inside the referencing contribution and keeps that contribution's origins. |
| 4. `ic` classified out | Its entry leaves `equations` and `equation_contributions` together. |

Observed definitions and expression-LHS algebraic equations get entries too.
A planner that inlines a scalar observed into a term may splice the observed's
contributions, with their own origins, in place of a term that is exactly a
reference to it. Elsewhere the inlined expression is part of the referencing
contribution. That choice is the planner's.

### Templates, discretization and canonical form

- **Expression templates (§9.6.4 Option B).** A surviving
  `apply_expression_template` reference sits inside one contribution. Expanding
  it, at build or in the planner, rewrites inside that contribution. If the
  expansion is a sum, every resulting term inherits the origins.
- **Discretization rewrites (§9.6.8).** Rule-based lowering of spatial `D`
  runs at load, before flatten (§9.6.4 rule 3; §10.8 "consumed by the §9.6.3
  fixpoint before flattening"). The stencil is then part of the component's
  contribution from the start. reseact's pipeline once saw a spatial `D` still
  standing at split time (`split_common.jl:124-139`). The rule covers that
  order too: a rewrite inside a contribution keeps its origins.
- **Canonicalization.** `canonicalize` flattens nested `+`, drops zero
  identities and sorts arguments (`canonicalize.jl:101-113`, `:195-205`). It
  does not combine like terms. In Julia it runs only to key CSE
  (`tree_walk/compile.jl:1009-1043`). In Rust it is exported
  (`lib.rs:220`) but no pipeline stage calls it. Neither rewrites a flattened
  equation today.
  Normative: any pass that rewrites flattened equations (canonicalize,
  `simplify`, CSE that rebuilds trees) MUST work per contribution. It MUST NOT
  flatten, sort, combine or cancel across a contribution boundary. CSE MAY share
  a node between contributions; provenance is not on nodes, so sharing loses
  nothing.

### Equality and hashing

Provenance is outside every expression. So, by construction:

- `canonical_json`, structural `==`, `node_eq`/`node_hash`, interning and CSE
  keys are unchanged;
- two contributions with identical expressions and different origins stay two
  contributions, even when interning makes their `expr` the same object.

Comparisons of whole `FlattenedSystem`s (the conformance corpus) compare
`equation_contributions` as its own field. The CONFORMANCE §5.25 build-reuse
key does not include it, because no build reads it.

### Bindings

**Julia.**

```julia
struct TermOrigin
    component::Union{String,Nothing}
    coupling::Union{Int,Nothing}          # exactly one is non-nothing
end
struct Contribution
    expr::ASTExpr
    origins::Vector{TermOrigin}
end
# FlattenedSystem gains: equation_contributions::Vector{Vector{Contribution}}
equation_contributions(flat::FlattenedSystem)::Vector{Vector{Contribution}}
```

`eq_owners` stays; it is attribution for later merges, not provenance. The
keyword copy constructor `FlattenedSystem(flat; equations = eqs)` (used by the
Splitter, `EarthSciASTSplitter.jl:546`) MUST reset contributions to one
contribution per equation with empty `origins` when `equations` is replaced
alone. A stale list must not survive a caller's edit.

**Rust.**

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TermOrigin { Component { component: String }, Coupling { coupling: usize } }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contribution { pub expr: Expr, pub origins: Vec<TermOrigin> }
// FlattenedSystem gains: pub equation_contributions: Vec<Vec<Contribution>>
```

`SystemBlock` (`flatten.rs:2342-2355`) carries a contribution list per
equation, moved with the equation when attribution relocates it
(`flatten.rs:3744-3760`). `sum_exprs` call sites build from that list.

**Python.** Python is the oracle for `tests/conformance/flatten/cases.json`,
so it implements the field in full. `FlattenedEquation` gains
`contributions: list[Contribution]`. The existing `source_system` stays,
documented as "the equation's slot owner, not provenance". `_add_exprs` drops
a literal-zero operand (`flatten.py:2097-2103`); the zero contribution is still
recorded. Python has no integrator, so nothing consumes the field there beyond
the corpus.

**TypeScript, Go.** Types, the flatten rules above, and the corpus adapter.

### Serialized form

Not in `.esm`. Two serializations exist:

1. Rust's serde form of `FlattenedSystem`, which gains the field.
2. The flatten corpus. Each equation gains a `contributions` array:

```json
{"lhs": "D(Chem.O3)/Dt",
 "rhs": "Src.e + (-Sink.kd) * Chem.O3 + (-Chem.k1) * Chem.O3 + (-Adv.wind) * grad(Chem.O3)",
 "source_system": "Src",
 "contributions": [
   {"rhs": "Src.e",                      "origins": [{"component": "Src"}]},
   {"rhs": "(-Sink.kd) * Chem.O3",        "origins": [{"component": "Sink"}]},
   {"rhs": "(-Chem.k1) * Chem.O3",        "origins": [{"component": "Chem"}]},
   {"rhs": "(-Adv.wind) * grad(Chem.O3)", "origins": [{"component": "Adv"}]}]}
```

Julia's `flattened_to_esm` (`flatten.jl:1565-1632`) drops the field. That is a
documented loss until *Open questions* 2 is settled.

### How the planner uses it

Architecture §3.2 pass 1 becomes: for each equation and each contribution,
take `additive_terms(expr)` (the Splitter algorithm, `:111-127`) and tag each
term with the contribution's origins. A run-config selector
`{"components": ["SuperFastEmissionSource"]}` (architecture §4.2) matches a
term when any origin's `component` equals a listed path or lies under it. A
selector that would split a multi-origin contribution is a planner error, not a
silent choice.

## What this is not

- **Not the §4.7.1 attribution.** Attribution decides which equation a later
  `operator_compose` entry can see. It is unchanged.
- **Not an additive-term splitter.** Term decomposition and locality stay in
  the planner (D14).
- **Not data provenance.** A `variable_map` substitution does not add an
  origin. The authoring component wrote the term; the map changed one of its
  inputs. The input's owner is still readable from its namespaced name.
- **Not a document field.** Authors never write it.
- **Not template provenance.** Which rewrite rule produced a stencil is not
  recorded.

## Conformance and tests

1. **Corpus.** `scripts/generate-flatten-corpus.py` emits `contributions` for
   every equation of every case. All five bindings compare them exactly, in
   order.
2. **New fixtures** under `tests/conformance/term_provenance/`:
   - chained composes with the operator as `systems[0]` (reuse
     `tests/coupling/chained_operator_compose.esm`);
   - a pair differing only in `systems` order, compared against each other:
     origins must match, order follows `rhs_A + rhs_B`;
   - `_var` placeholder expansion;
   - `couple` additive onto an existing and onto an absent tendency;
   - `couple` multiplicative (collapse, union of origins);
   - `variable_map` additive and expression transform;
   - pointwise lift (`lifting: "pointwise"`);
   - a surviving Option B template reference inside a contribution;
   - a deposition-sink-shaped term that reads two components.
3. **Non-vacuity.** Two operators contribute structurally identical
   expressions to one state (both read one mapped source). The golden has two
   contributions with different origins. A binding that hangs provenance on
   interned nodes fails it.
4. **Invariant.** Each binding evaluates `body(eq)` and the left-fold sum of
   its contributions at a sampled non-default state and asserts bit equality.

## Work breakdown

1. **Spec.** esm-libraries-spec §4.7.5 (table row, step table, invariant,
   canonicalization rule); one sentence in §4.7.1 step 4 separating attribution
   from provenance; one paragraph in esm-spec §10.7; a CONFORMANCE_SPEC section.
2. **Julia.** `flatten.jl`; `coupling_apply.jl` (`_apply_operator_compose!`,
   `_combine_tendency_term!` at `:1135-1153`, `_apply_variable_map!`);
   `pointwise_lift.jl`; the 3a resolver (`flatten.jl:580`); copy constructor;
   accessor; `api-surface.json`.
3. **Rust.** `flatten.rs` blocks, merge and couple sites (`:3457`, `:4082`),
   `apply_pointwise_lift` (`:4649`), serde.
4. **Python.** `flatten.py` (`:1733`, `:2165-2178`); corpus generator; goldens.
5. **TypeScript, Go.** Types, rules, adapters.
6. **EarthSciIntegrator.** Planner pass 1 reads the field; plan goldens record
   each term's origins.

## Alternatives considered

1. **Recover the component from namespaced names** (status quo, decisions.md
   §3.2). Fails on every term that reads two components, which is every
   coupled sink.
2. **A `provenance` field on expression nodes.** Breaks canonical JSON
   (fail-closed emitters), forces a choice between interning and correctness,
   and cannot annotate a bare name or literal.
3. **A wrapper op** (`{"op": "term", "origin": …, "args": [e]}`). Every
   evaluator, compiler, canonicalizer and EarthSciASTDiff would have to see
   through it. CSE keys would change. Rejected.
4. **Per-equation `source_system`** (Python today). One label for a four-way
   sum (`cases.json:3827`).
5. **Not flattening `operator_compose`.** The flattened system is the API
   boundary (§4.7.5); a lazy sum of systems is too invasive for this need.

## Open questions

1. **Reaction-level origins.** Step 1 builds the mass-action sum reaction by
   reaction. Recording a `reaction` id per term would let a run config split
   photolysis from thermal chemistry. Cheap to add; no current consumer.
2. **Embedding in `.esm`.** An optional `Equation.contributions` would let
   `flattened_to_esm` round-trip provenance. If wanted, it rides esm 1.3.0 with
   U1 `solver.external` (or the next minor version if U1 lands separately),
   gated `term_provenance_version_too_old`.
3. **Assembly-style models.** EqWeFiC sums mapped tendencies inside one
   assembly component, so every summand has the same author. The planner must
   then chase scalar observeds and use the owner of a bare-reference term. The
   physics sum also passes through a linear factor:
   `dtheta_m_dt_phys = (1 + rvovrd·qv)·dtheta_dt_phys + rvovrd·θ·dqv_dt_phys`
   (`EqWeFiC/components/atmospheric_dynamics/wrf_arw/moist_theta_tendency.esm:184`).
   Should flatten record the source of each `variable_map` substitution, or is
   that the planner's job?
4. **Granularity inside one component.** reseact's `Transport3D` writes PBL
   mixing and advection as two summands (`reseact.esm:2619`, `:2732`); locality
   separates them. Horizontal and vertical advection share one summand,
   `[-div(M q) + q div(M)]/m`. Splitting them needs the planner to distribute
   `/` over `+`. Provenance cannot help there.
5. **3a resolution.** Should a resolved `D(x, t)` splice in `x`'s contributions
   with their origins, instead of inheriting the referencing origin?
