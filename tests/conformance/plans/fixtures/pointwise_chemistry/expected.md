# pointwise_chemistry: expected plan contents

The model: reaction system `Chem` (NO2 + hν → NO + O3, rate `j`; NO + O3 → NO2, rate `k`), lifted pointwise by `operator_compose` with model `Dep`, whose `D(_var)` is a per-cell sink `-kd · cellw[i] · _var[i]` over index set `cell` (4 cells). `solver.stiffness` is `"high"`, `solver.time_span` is 0–3600 s. What a golden must show, by spec/plan.md section; anything not listed is fixed by the spec's general rules.

## Common to both cases

**Flattened model (§4).** States `Chem.NO2`, `Chem.NO`, `Chem.O3`, each lifted onto 4 cells; canonical order `Chem.NO`, `Chem.NO2`, `Chem.O3` (12 DOFs). Parameters `Dep.kd`, `Dep.cellw` (shape `["cell"]`), `Chem.j`, `Chem.k`. The lifted states take the shape `["cell"]` from `Dep.cellw`, the first declared shape with extent 4 (§6.1.3).

**Axes (§5.2.1).** Default: horizontal `["cell"]`, no vertical axis.

**Terms (§5.1).** Nine, in this order (variable, contribution, summand, provenance, locality):

| variable | c | s | provenance | locality | the summand |
|---|---|---|---|---|---|
| Chem.NO2 | 0 | 0 | Chem | pointwise | `-(Chem.j * NO2[i])` |
| Chem.NO2 | 0 | 1 | Chem | pointwise | `Chem.k * NO[i] * O3[i]` |
| Chem.NO2 | 1 | 0 | Dep | pointwise | `index(makearray(…-kd·cellw[i]·NO2[i]…), i)` |
| Chem.NO | 0 | 0 | Chem | pointwise | `Chem.j * NO2[i]` |
| Chem.NO | 0 | 1 | Chem | pointwise | `-(Chem.k * NO[i] * O3[i])` |
| Chem.NO | 1 | 0 | Dep | pointwise | the Dep sink for NO |
| Chem.O3 | 0 | 0 | Chem | pointwise | `Chem.j * NO2[i]` |
| Chem.O3 | 0 | 1 | Chem | pointwise | `-(Chem.k * NO[i] * O3[i])` |
| Chem.O3 | 1 | 0 | Dep | pointwise | the Dep sink for O3 |

(The reaction-system summands are as Julia's EarthSciAST 0.5.0 `flatten` writes them: a loss is a one-argument `-`, `-(Chem.j * NO2[i])`, not `-1 * Chem.j * NO2[i]`. Rust's and Python's write `-1.0 * (…)` and `-1 * (…)`, an upstream difference, spec/plan.md §10 item 10. The golden review checks there are two per species, split at the `+`.)

**One operator holds all nine terms.** Its entry:
- `locality`: `{"class": "pointwise"}`.
- `writes`: `["Chem.NO", "Chem.NO2", "Chem.O3"]`; `reads`: `["Chem.NO", "Chem.NO2", "Chem.O3", "Chem.j", "Chem.k", "Dep.cellw", "Dep.kd"]`.
- `structure.class`: `block_diagonal`; `linear_in_u` false (`k·NO·O3`); `depends_on_t` false.
- `structure.bands`: one or more bands for each of the 9 (row, column) pairs of the 3 × 3 species block, every band with `rows` `[[1, 4]]`, `row_idx` `["i"]`, `col_idx` `["i"]` (EarthSciASTDiff may keep the sink's band apart from the reaction's on the diagonal).
- `structure.block`: `vars` `["Chem.NO", "Chem.NO2", "Chem.O3"]`, `ordering` `[1, 2, 3]` (a full 3 × 3 pattern gives every candidate the same Markowitz cost, and the first wins), `pattern` and `lu_pattern` all nine pairs `[1,1] … [3,3]`.
- `stiffness`: per case below.

**Derived documents (§6).** `derived/<op>.rhs.esm` declares models `Chem` (unknowns `NO`, `NO2`, `O3`, shape `["cell"]`, defaults 5.5, 10.5, 40.5; parameters `j`, `k`) and `Dep` (parameters `cellw`, `kd`); index set `cell` of size 4. Each species equation is `D(Chem.X) = faq{output_idx ["i"], ranges {"i": {"from": "cell"}}, expr: (t1 + t2) + t3}` (ranges built from the shape `["cell"]`, §6.1.2), the fold of that species' three terms. No `ic` equations (none in the model). `derived/<op>.jacobian.esm` adds model `_jacobian` with one `J_k` per band, each of shape `["cell"]`.

**Validation (§8).** `[{"check": "rhs_sum", "passed": true}, {"check": "band_model", "operator": <op>, "passed": true}, {"check": "non_empty", "operator": <op>, "passed": true}]`.

**Time.** `{"start": 0, "end": 3600}` from `solver.time_span`.

## auto

`schedule: "auto"`, no `operators`, no `step`.

- The operator is `default_pointwise`.
- `stiffness`: `{"class": "stiff", "source": "reaction_systems"}` (rule 2; the model's `"high"` hint would also say stiff, but rule 2 comes first).
- One locality class, so the root is the leaf itself (D72):

```json
{"alg": "Rosenbrock23", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "cell", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "jacobian": "symbolic", "leaf": ["default_pointwise"], "linsolve": {"block_lu": {"pivot": true, "refine": 0}}}
```

  `Rosenbrock23` because the resolved `reltol` 1e-3 is ≥ 1e-3; `block_lu` because the block has 3 variables.
- `step`: `{"hit": []}` (adaptive root, D49).
- `partition`: `{"halo": {"cell": 0}, "ranks": [1, 1], "shards": {"root": 1}, "tiles": [1, 1]}`.
- `precision`: `{"root": {"factors": "f64", "jacobian": "f64", "rhs": "f64", "state": "f64", "tables": "f64"}}`.
- `derived`: `default_pointwise.rhs.esm`, `default_pointwise.jacobian.esm`.
- No `checkpoint_schedule` (no `adjoint`).

## named

One named operator `chemistry` over components `Chem` and `Dep`; leaf with `alg: "auto"` and `reltol` 1e-6, `abstol` 1e-9; `partition.shards` `{"chemistry": 2}`.

- The operator is `chemistry`, with the nine terms; no default operator exists.
- `stiffness`: `{"class": "stiff", "source": "reaction_systems"}`.
- The root leaf:

```json
{"alg": "Rodas4", "control": {"abstol": 1e-9, "controller": "PI", "dt_scope": "cell", "maxiters": 1000000, "reltol": 0.000001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "jacobian": "symbolic", "leaf": ["chemistry"], "linsolve": {"block_lu": {"pivot": true, "refine": 0}}}
```

  `Rodas4` because `reltol` 1e-6 is below 1e-3.
- `partition.shards`: `{"root": 2}`, the operator's entry carried to the leaf that advances it.
- `step`, `precision`, `time` as in `auto`; `derived`: `chemistry.rhs.esm`, `chemistry.jacobian.esm`.
