# interp_slope: expected plan contents

The model: a model `Chem` with one scalar state `z` (`default` 0.5) and parameters `k` and `zsel`, and the one equation `D(z) = -(k * (linear(table = const [100.0, 300.0, 700.0], axis = const [0.0, 1.0, 2.0], x = z, n = 3) + z * linear(table = const [2.0, 4.0], axis = const [1.0, 2.0], x = zsel, n = 2)))`, `linear` the template of `esm:std/interp.esm` (the model was written with `fn` `interp.linear` until esm 2.0.0 removed it, EarthSciAST 0.11.0, and was rewritten with EarthSciAST's codemod). `solver.time_span` is 0–3600 s. It covers the numbers of the Jacobian document that do not come from the model as written (spec/plan.md §2.2, §9): EarthSciASTDiff differentiates the first `linear`, expanded into core operators, by its core rules (its literal-array `index` reads have derivative 0), and the band coefficient keeps the second `linear` expanded, with its `const` tables. What a golden must show, by spec/plan.md section; anything not listed is fixed by the spec's general rules.

## named

One operator `chemistry` selected by component (`Chem`), advanced by `Tsit5`.

**Flattened model (§4).** One state, `Chem.z` (1 DOF); parameters `Chem.k`, `Chem.zsel`; no observed.

**Terms (§5.1).** One: `Chem.z`, contribution 0, summand 0, provenance `Chem`, pointwise.

**The operator.** `locality` `{"class": "pointwise"}`; `writes` `["Chem.z"]`; `reads` `["Chem.k", "Chem.z", "Chem.zsel"]`; `stiffness` `{"class": "nonstiff", "source": "run_config"}`; `structure.class` `block_diagonal`, `linear_in_u` false, `depends_on_t` false; `structure.block` `vars` `["Chem.z"]`, `ordering` `[1]`, `pattern` and `lu_pattern` `[[1, 1]]`; one band, `(z, z)`.

**Derived documents (§6).** In `derived/chemistry.rhs.esm` both `linear` applications are expanded into core operators (`index`, `ifelse`, a `faq` count of the knots at or below the query, `min`, `max`), and the `const` tables are written `[100,300,700]`, `[0,1,2]`, `[2,4]` and `[1,2]` (§2.2). In `derived/chemistry.jacobian.esm` (§6.2) the band field `J_1` is `-(k * (S + L))`, where S is the derivative of the first expansion, `ifelse(z <= axis[1], 0, ifelse(z >= axis[3], 0, (table[k+1] - table[k]) / (axis[k+1] - axis[k])))` with `k` the active segment, and L the second expansion as in the right-hand-side document; repeated subtrees (the segment indices) are shared as `JacobianEval.Jcse<n>` observeds (D90). Every `const` table keeps the integer form. The band values equal those of the `fn` form (the template reproduces `interp.linear` bit for bit, and its derivative is the active segment's slope).

**Validation (§8).** `[{"check": "rhs_sum", "passed": true}, {"check": "band_model", "operator": "chemistry", "passed": true}, {"check": "non_empty", "operator": "chemistry", "passed": true}]`.

**Settings.** `time` `{"start": 0, "end": 3600}`; the root is the leaf `chemistry`, `Tsit5` with the default control.
