# reseact_shaped: expected plan contents

reseact's shape at toy size (architecture §14 M2 gate "reseact-shaped fixtures"): a reaction system `Chem` (NO2/NO/O3, rates `j`, `k`) joined by three `operator_compose` entries, in this order:

| k | entry | contributes |
|---|---|---|
| 0 | `Chem` + `Dep` | `-vd · X`, a 0-D sink (merged before the lift) |
| 1 | `Chem` + `Transport`, `lifting: "pointwise"` | PPM-like periodic advection along `lon` (as in `periodic_stencil`, here `c` = 0.5, `dx` = 500.5) plus first-order upwind along `lat` with a closed inflow face at `lat` 1 (`v` = 2.5, `dy` = 600.5) |
| 2 | `Chem` + `Mix`, `lifting: "pointwise"` | the column mean of `column_contraction`, `kmix` = 0.002 |

Grid `lon × lat × lev` = 8 × 3 × 4 (96 cells, 288 DOFs). `solver`: `stiffness: "high"`, `splitting: "lie"`, `time_span` 0–3600 s. Both cases give `axes` `{"horizontal": ["lon", "lat"], "vertical": "lev"}` and `step.dt` 300.

## Common to both cases

**Flattened model.** States `Chem.NO2`, `Chem.NO`, `Chem.O3`, lifted; canonical order `Chem.NO`, `Chem.NO2`, `Chem.O3`. Their shape is `["lon", "lat", "lev"]`, from `Transport.c`, the first declared shape with extents (8, 3, 4) (§6.1.3). Each species equation has four contributions, origins `Chem`, `Dep`, `Transport`, `Mix`.

**Terms.** Six per species, eighteen in all. For species X, in order:

| c | s | provenance | locality | summand |
|---|---|---|---|---|
| 0 | 0 | Chem | pointwise | first reaction-system summand |
| 0 | 1 | Chem | pointwise | second reaction-system summand |
| 1 | 0 | Dep | pointwise | `-1 * Dep.vd * X[i,j,k]` |
| 2 | 0 | Transport | stencil, radius `lon` [3, 2], `lat` [0, 0] | the `lon` advection, `index(makearray, i, j, k)` |
| 2 | 1 | Transport | stencil, radius `lon` [0, 0], `lat` [1, 0] | the `lat` advection |
| 3 | 0 | Mix | column | the column mean |

The Transport contribution splits at its `+` into two terms with one provenance (D64): that is what lets a run config select advection by component while each direction keeps its own reach.

**Structures by locality** (the same whether the operators are named or default):
- The stencil operator (six Transport terms): `radius` `{"lat": [1, 0], "lon": [3, 2]}`; `structure.class` `general`; `structure.stencil.footprint` `[[-3, 0], [-2, 0], [-1, 0], [0, -1], [0, 0], [1, 0], [2, 0]]`; `linear_in_u` true; `depends_on_t` false; `reads` `["Chem.NO", "Chem.NO2", "Chem.O3", "Transport.c", "Transport.dx", "Transport.dy", "Transport.v"]`.
- The column operator (three Mix terms): `structure.column` `{"axis": "lev", "kl": 3, "ku": 3}`; `class` `general`; `linear_in_u` true; `reads` `["Chem.NO", "Chem.NO2", "Chem.O3", "Mix.kmix"]`. Gershgorin: diagonal $k_{mix}(1/4 - 1)$, three off-diagonals $k_{mix}/4$, so $\rho = 1.5\,k_{mix} = 0.003$ and $\rho\Delta = 0.9$ at $\Delta$ = 300 s: non-stiff.
- The pointwise operator (six Chem and three Dep terms): `class` `block_diagonal`; `block` `vars` `["Chem.NO", "Chem.NO2", "Chem.O3"]`, `ordering` `[1, 2, 3]`, full 3 × 3 `pattern` and `lu_pattern`; `linear_in_u` false; `reads` `["Chem.NO", "Chem.NO2", "Chem.O3", "Chem.j", "Chem.k", "Dep.vd"]`.
- Every operator `writes` `["Chem.NO", "Chem.NO2", "Chem.O3"]`.

**Derived documents.** Each right-hand-side document declares models `Chem`, `Dep`, `Mix`, `Transport` (in `stable_json` key order), all three species with shape `["lon", "lat", "lev"]`, every parameter, and the species equations `D(Chem.X) = faq{output_idx ["i", "j", "k"], ranges {"i": {"from": "lon"}, "j": {"from": "lat"}, "k": {"from": "lev"}}, expr: <fold of the operator's terms for X>}`, the `faq` wrapper's indices copied from the flattened lifted equation and its ranges built from the shape (§6.1.2).

**Time.** `{"start": 0, "end": 3600}`; `step` per case.

## auto

`schedule: "auto"`, no `operators`.

- Operators `default_column`, `default_pointwise`, `default_stencil`.
- Schedule (D72; the model's `splitting` is `lie`, so a `sequence`):

```json
{"id": "root", "sequence": [
  {"alg": "SSPRK43", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root.0", "leaf": ["default_stencil"]},
  {"alg": "Tsit5", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root.1", "leaf": ["default_column"]},
  {"alg": "Rosenbrock23", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "cell", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root.2", "jacobian": "symbolic", "leaf": ["default_pointwise"], "linsolve": {"block_lu": {"pivot": true, "refine": 0}}}
]}
```

- `stiffness`: `default_stencil` `{"nonstiff", "gershgorin"}` ($\rho\Delta$ about 2.9: `lon` row sum $(c/dx) \cdot 1.375$ plus `lat` $2v/dy$, under 10); `default_column` `{"nonstiff", "gershgorin"}`; `default_pointwise` `{"stiff", "reaction_systems"}`.
- `step` `{"dt": 300, "hit": []}`.
- `partition` `{"halo": {"lat": 1, "lon": 3}, "ranks": [1, 1], "shards": {"root.1": 1, "root.2": 1}, "tiles": [1, 1]}`.
- `precision`: `root.0`, `root.1`, `root.2`, each with all five roles `"f64"`.
- `derived`: six documents; `validation`: `rhs_sum`, then `band_model` and `non_empty` for `default_column`, `default_pointwise`, `default_stencil` in that order.

## named

Operators by provenance, the three `operator_compose` contributions: `transport` = `{"components": "Transport"}`, `mixing` = `{"components": "Mix"}`, `chemistry` = `{"components": ["Chem", "Dep"]}`. The prototype's methods for transport and chemistry (architecture §4.4), the mixing method left `"auto"`.

- No default operator exists.
- Schedule:

```json
{"id": "root", "sequence": [
  {"alg": "SSPRK43", "control": {"abstol": 0.000001, "controller": "PI", "dt0": 15, "dt_scope": "global", "maxiters": 1000000, "reltol": 0.0001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root.0", "leaf": ["transport"]},
  {"alg": "Tsit5", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root.1", "leaf": ["mixing"]},
  {"alg": "Rosenbrock23", "control": {"abstol": 1e-9, "controller": "PI", "dt0": 0.5, "dt_scope": "global", "maxiters": 1000000, "reltol": 0.0001}, "domain": {"check_stages": false, "nonnegative": "all", "on_violation": "reject"}, "id": "chem", "jacobian": "symbolic", "leaf": ["chemistry"], "linsolve": {"block_lu": {"pivot": false, "refine": 0}}}
]}
```

- `stiffness`: `transport` `{"nonstiff", "run_config"}`; `mixing` `{"nonstiff", "gershgorin"}`; `chemistry` `{"stiff", "run_config"}`.
- `step` `{"dt": 300, "hit": ["output"]}`.
- `partition` `{"halo": {"lat": 1, "lon": 3}, "ranks": [2, 1], "shards": {"chem": 4, "root.1": 1}, "tiles": [1, 3]}`: the run config's `{"chemistry": 4}` lands on the leaf `chem`; `lon` 2 × 1 ≤ 8 and `lat` 1 × 3 ≤ 3.
- `precision`: `root.0` and `root.1` all `"f64"`; `chem` `{"factors": "f32", "jacobian": "f64", "rhs": "f64", "state": "f64", "tables": "f64"}` (low-precision factors are allowed on `Rosenbrock23`, a W-method; on `Rodas4` they would be refused).
- `derived`: `chemistry`, `mixing`, `transport` documents; `validation` sorted by those names.
