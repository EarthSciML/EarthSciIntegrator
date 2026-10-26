# toy_robertson: expected plan contents

The `.esm` version of spec/toy-problems.md §1.3 (`tests/conformance/toys/robertson.esm`), planned with the run config of M1 case `robertson.Rodas4.adaptive` plus the `operators` key the M2 case adds (spec/toy-problems.md §4.2).

## rodas4

**Terms.** Six, all provenance `Kinetics`, all `pointwise`:

| variable | c | s | summand |
|---|---|---|---|
| Kinetics.y1 | 0 | 0 | `-1 * k1 * y1` |
| Kinetics.y1 | 0 | 1 | `k3 * y2 * y3` |
| Kinetics.y2 | 0 | 0 | `k1 * y1` |
| Kinetics.y2 | 0 | 1 | `-(k3 * y2 * y3)` |
| Kinetics.y2 | 0 | 2 | `-(k2 * y2^2)` |
| Kinetics.y3 | 0 | 0 | `k2 * y2^2` |

(`y2`'s right-hand side is `(k1 y1 − k3 y2 y3) − k2 y2²`, two nested binary `-`; names in the plan are fully scoped.)

**Operator `kinetics`.** `locality` `{"class": "pointwise"}`; `writes` `["Kinetics.y1", "Kinetics.y2", "Kinetics.y3"]`; `reads` those plus `"Kinetics.k1", "Kinetics.k2", "Kinetics.k3"`, sorted (`Kinetics.k1` … before `Kinetics.y1`). `structure`: `class` `block_diagonal`; scalar bands (`rows`, `row_idx`, `col_idx` all `[]`) for the pairs (y1: y1, y2, y3), (y2: y1, y2, y3), (y3: y2), in that order; `block` `{"vars": ["Kinetics.y1", "Kinetics.y2", "Kinetics.y3"], "ordering": [1, 2, 3], "pattern": [[1,1],[1,2],[1,3],[2,1],[2,2],[2,3],[3,2],[3,3]], "lu_pattern": the same}` (the `[3,3]` diagonal is added because $W = I - \gamma h J$ always has it; Markowitz costs 2, 4, 2 pick `y1` first, then the order stays); `linear_in_u` false; `depends_on_t` false. `stiffness` `{"stiff", "run_config"}`.

**Schedule.** Root leaf:

```json
{"alg": "Rodas4", "control": {"abstol": 1e-10, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 0.000001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "jacobian": "symbolic", "leaf": ["kinetics"], "linsolve": {"dense_lu": {}}}
```

**Other keys.** `time` `{"start": 0, "end": 40}` from the document (the M2 case's harness passes the same interval as the caller's); `step` `{"dt": 40, "hit": []}`; `partition` `{"ranks": [1, 1], "shards": {"root": 1}, "tiles": [1, 1]}` (no index sets, no halo); `precision` `{"root": {…"f64"}}`; `derived` `kinetics.rhs.esm`, `kinetics.jacobian.esm`; `inputs.model.path` `"../../../toys/robertson.esm"`, as the run config writes it.

**Runtime use (spec/operator-interface.md §8).** This plan's operator, built from its two documents, must reproduce the hand-written `kinetics` operator's `rhs` and `jac` to roundoff; the M2 case `robertson.Rodas4.adaptive` checks the whole run (spec/toy-problems.md §4.3).
