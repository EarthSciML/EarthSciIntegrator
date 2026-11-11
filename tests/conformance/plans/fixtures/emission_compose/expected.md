# emission_compose: expected plan contents

The model: a model `Emis`, declared first, whose one equation `D(Chem.B) = Emis.e` writes another component's state by its scoped name, and a model `Chem` with scalar states `A`, `B`, `C` (`D(A) = interp.linear(const [100.0, 200.0, 400.0], const [1.0, 2.0, 3.0], zsel) / 100 * k - k * A`, `D(B) = k * A - k * B`, `D(C) = k * B`), an array state `q` over index set `cell` (3 cells) decaying at `k * (1 + wet[i])`, and an observed `wet = ifelse(w > 1.0, 1, 0)` declared without a `shape`, `w` a parameter of shape `["cell"]`. One coupling, `operator_compose` with `"systems": ["Emis", "Chem"]`. `solver.time_span` is 0–3600 s. Each part reproduces one of the three places where the two planners' derived documents of reseact differed (spec/plan.md §10 items 23–25). What a golden must show, by spec/plan.md section; anything not listed is fixed by the spec's general rules.

## named

One operator `chemistry` selected by component (`Chem`, `Emis`), advanced by `Tsit5`.

**Flattened model (§4).** States `Chem.A`, `Chem.B`, `Chem.C`, `Chem.q` (shape `["cell"]`); canonical order `Chem.A`, `Chem.B`, `Chem.C`, `Chem.q` (6 DOFs). Parameters `Emis.e`, `Chem.k`, `Chem.zsel`, `Chem.w` (shape `["cell"]`). Observed `Chem.wet`, with no declared shape.

**Equation order (§6.1.2, §10 item 25).** The merge of `Emis`'s `D(Chem.B)` with `Chem`'s belongs to `Chem` and comes after `Chem`'s own equations, as the Python reference and the Rust binding of EarthSciAST order it: the flattened equations are `Chem.wet`, `D(Chem.A)`, `D(Chem.C)`, `D(Chem.q)`, `D(Chem.B)`. EarthSciAST 0.9.0's Julia binding put `D(Chem.B)` first, at `Emis`'s position; the golden needs the fixed binding.

**Terms (§5.1).** Seven, in this order (variable, contribution, summand, provenance), all pointwise:

| variable | c | s | provenance | the summand |
|---|---|---|---|---|
| Chem.A | 0 | 0 | Chem | `interp.linear(…) / 100 * Chem.k` |
| Chem.A | 0 | 1 | Chem | `-(Chem.k * Chem.A)` |
| Chem.C | 0 | 0 | Chem | `Chem.k * Chem.B` |
| Chem.q | 0 | 0 | Chem | the `makearray` decay of `q` |
| Chem.B | 0 | 0 | Emis | `Emis.e` |
| Chem.B | 1 | 0 | Chem | `Chem.k * Chem.A` |
| Chem.B | 1 | 1 | Chem | `-(Chem.k * Chem.B)` |

**The operator.** `locality` `{"class": "pointwise"}`; `writes` `["Chem.A", "Chem.B", "Chem.C", "Chem.q"]`; `reads` `["Chem.A", "Chem.B", "Chem.k", "Chem.q", "Chem.w", "Chem.wet", "Chem.zsel", "Emis.e"]`; `stiffness` `{"class": "nonstiff", "source": "run_config"}`; `structure.class` `block_diagonal`, `linear_in_u` true, `depends_on_t` false; `structure.block` `vars` `["Chem.A", "Chem.B", "Chem.C", "Chem.q"]`, `ordering` `[1, 2, 3, 4]`, `pattern` and `lu_pattern` `[[1, 1], [2, 1], [2, 2], [3, 2], [3, 3], [4, 4]]`. The bands are in EarthSciASTDiff's order over the equations above: `(A, A)`, `(C, B)`, `(q, q)` with `rows` `[[1, 3]]`, `(B, A)`, `(B, B)`.

**Derived documents (§6).** `derived/chemistry.rhs.esm` declares models `Chem` and `Emis` (`Emis` with no equations: its one equation was merged into `Chem`'s). In `Chem`, the equations are in the order above, and:
- the `const` tables are written `[100,200,400]` and `[1,2,3]`, integral numbers as integers, as every other integral literal of the model is (§2.2, §10 item 23); the `1.0` of `w > 1.0` is written `1`;
- `wet` is declared `{"type": "unknown"}`, with no `shape`: its declaration has none, whatever shape a binding's `flatten` infers from `w` (§6.1.1, §10 item 24).

`derived/chemistry.jacobian.esm` (§6.2) has `Chem`'s equations in the same order and the same declaration of `wet`, which the band coefficient of `(q, q)` reads; no band coefficient reads the `const` tables, so they appear only in the right-hand-side document.

**Validation (§8).** `[{"check": "rhs_sum", "passed": true}, {"check": "band_model", "operator": "chemistry", "passed": true}, {"check": "non_empty", "operator": "chemistry", "passed": true}]`.

**Settings.** `time` `{"start": 0, "end": 3600}`; the root is the leaf `chemistry`, `Tsit5` with `dt_scope` `global` and the default control; `step` `{"hit": []}`; `partition` `{"halo": {"cell": 0}, "ranks": [1, 1], "shards": {"root": 1}, "tiles": [1, 1]}`; `precision` all `f64`.
