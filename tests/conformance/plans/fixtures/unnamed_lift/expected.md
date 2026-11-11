# unnamed_lift: expected plan contents

The model: `pointwise_chemistry`'s, without the parameter `Dep.cellw` and without any `index_sets`. Reaction system `Chem` (NO2 + hν → NO + O3, rate `j`; NO + O3 → NO2, rate `k`) is lifted pointwise by `operator_compose` with model `Dep`, whose `D(_var)` is a per-cell sink `-kd · _var[i]` over the interval `[1, 4]`. No index set is declared, so no variable names the lift's dimension, and EarthSciAST 0.8.0's `flatten` registers the synthetic index set `_liftdim1_4` (spec/plan.md §6.1.3). Until EarthSciAST 0.8.0 this model was the refusal `lift_shape_unresolved`. `solver.stiffness` is `"high"`, `solver.time_span` is 0–3600 s.

## auto

`schedule: "auto"`, no `operators`, no `step`.

**Flattened model (§4).** States `Chem.NO2`, `Chem.NO`, `Chem.O3`, each lifted onto 4 cells; canonical order `Chem.NO`, `Chem.NO2`, `Chem.O3` (12 DOFs). Parameters `Dep.kd`, `Chem.j`, `Chem.k`. The flattened `index_sets` are `{"_liftdim1_4": {"kind": "interval", "size": 4}}`, and the lifted states take the shape `["_liftdim1_4"]` that `flatten` records on them (§6.1.3).

**Axes (§5.2.1).** Default: horizontal `["_liftdim1_4"]`, no vertical axis.

**Terms (§5.1).** Nine, as in `pointwise_chemistry` (three per species, in the order `Chem.NO2`, `Chem.NO`, `Chem.O3`: the two `Chem` summands of contribution 0, then the `Dep` sink, contribution 1), every one pointwise. The sink no longer reads `Dep.cellw`.

**The operator `default_pointwise` holds all nine terms.**
- `locality`: `{"class": "pointwise"}`.
- `writes`: `["Chem.NO", "Chem.NO2", "Chem.O3"]`; `reads`: `["Chem.NO", "Chem.NO2", "Chem.O3", "Chem.j", "Chem.k", "Dep.kd"]`.
- `structure.class`: `block_diagonal`; `linear_in_u` false; `depends_on_t` false.
- `structure.bands`: nine, one per (row, column) pair of the 3 × 3 species block, every band with `rows` `[[1, 4]]`, `row_idx` `["i"]`, `col_idx` `["i"]`.
- `structure.block`: `vars` `["Chem.NO", "Chem.NO2", "Chem.O3"]`, `ordering` `[1, 2, 3]`, `pattern` and `lu_pattern` all nine pairs `[1,1] … [3,3]`.
- `stiffness`: `{"class": "stiff", "source": "reaction_systems"}`.

**Schedule.** One locality class, so the root is the leaf itself (D72):

```json
{"alg": "Rosenbrock23", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "jacobian": "symbolic", "leaf": ["default_pointwise"], "linsolve": {"block_lu": {"pivot": true, "refine": 0}}}
```

- `step`: `{"hit": []}` (adaptive root, D49).
- `partition`: `{"halo": {"_liftdim1_4": 0}, "ranks": [1, 1], "shards": {"root": 1}, "tiles": [1, 1]}`.
- `precision`: `{"root": {"factors": "f64", "jacobian": "f64", "rhs": "f64", "state": "f64", "tables": "f64"}}`.
- No `checkpoint_schedule` (no `adjoint`).

**Derived documents (§6).** Both carry `index_sets` `{"_liftdim1_4": {"kind": "interval", "size": 4}}` and no `domain`. `derived/default_pointwise.rhs.esm` declares models `Chem` (unknowns `NO`, `NO2`, `O3`, shape `["_liftdim1_4"]`, defaults 5.5, 10.5, 40.5; parameters `j`, `k`) and `Dep` (parameter `kd`). Each species equation is `D(Chem.X) = faq{output_idx ["i"], ranges {"i": {"from": "_liftdim1_4"}}, expr: (t1 + t2) + t3}` (§6.1.2). No `ic` equations. `derived/default_pointwise.jacobian.esm` (EarthSciASTDiff's evaluation document, §6.2: `esm` `1.1.0`, `metadata.name` `JacobianEval`) zeroes the species equations and adds model `JacobianEval` with `J_1 … J_9`, each of shape `["_liftdim1_4"]`.

**Validation (§8).** `[{"check": "rhs_sum", "passed": true}, {"check": "band_model", "operator": "default_pointwise", "passed": true}, {"check": "non_empty", "operator": "default_pointwise", "passed": true}]`.

**Time.** `{"start": 0, "end": 3600}` from `solver.time_span`.
