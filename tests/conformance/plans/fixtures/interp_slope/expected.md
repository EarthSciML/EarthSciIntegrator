# interp_slope: expected plan contents

The model: a model `Chem` with one scalar state `z` (`default` 0.5) and parameters `k` and `zsel`, and the one equation `D(z) = -(k * (interp.linear(const [100.0, 300.0, 700.0], const [0.0, 1.0, 2.0], z) + z * interp.linear(const [2.0, 4.0], const [1.0, 2.0], zsel)))`. `solver.time_span` is 0–3600 s. It covers the numbers of the Jacobian document that do not come from the model as written (spec/plan.md §2.2, §9): EarthSciASTDiff computes the derivative of the first `interp.linear`, whose segment slopes (200 and 400) and knots (0, 1, 2) are all integral, and the band coefficient keeps the second `interp.linear` with its `const` tables. What a golden must show, by spec/plan.md section; anything not listed is fixed by the spec's general rules.

## named

One operator `chemistry` selected by component (`Chem`), advanced by `Tsit5`.

**Flattened model (§4).** One state, `Chem.z` (1 DOF); parameters `Chem.k`, `Chem.zsel`; no observed.

**Terms (§5.1).** One: `Chem.z`, contribution 0, summand 0, provenance `Chem`, pointwise.

**The operator.** `locality` `{"class": "pointwise"}`; `writes` `["Chem.z"]`; `reads` `["Chem.k", "Chem.z", "Chem.zsel"]`; `stiffness` `{"class": "nonstiff", "source": "run_config"}`; `structure.class` `block_diagonal`, `linear_in_u` false, `depends_on_t` false; `structure.block` `vars` `["Chem.z"]`, `ordering` `[1]`, `pattern` and `lu_pattern` `[[1, 1]]`; one band, `(z, z)`.

**Derived documents (§6).** In `derived/chemistry.rhs.esm` the `const` tables are written `[100,300,700]`, `[0,1,2]`, `[2,4]` and `[1,2]` (§2.2). In `derived/chemistry.jacobian.esm` (§6.2) the band field `J_1` is `-(k * (S + interp.linear(const [2,4], const [1,2], zsel)))`, where S is EarthSciASTDiff's segment slope of the first `interp.linear`, `ifelse(z < 0, 0, ifelse(z < 1, 200, ifelse(z < 2, 400, 0)))`: every slope and knot written as an integer, and the kept `const` tables as in the right-hand-side document.

**Validation (§8).** `[{"check": "rhs_sum", "passed": true}, {"check": "band_model", "operator": "chemistry", "passed": true}, {"check": "non_empty", "operator": "chemistry", "passed": true}]`.

**Settings.** `time` `{"start": 0, "end": 3600}`; the root is the leaf `chemistry`, `Tsit5` with the default control.
