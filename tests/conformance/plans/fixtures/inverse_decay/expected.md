# inverse_decay: expected plan contents

The model: one scalar state `M.x` with `D(x) = -(k / x)`, `k` = 0.5. `x` has no `default`: only its `ic` equation starts it, at 2.5. `solver.time_span` is 0–1. The fixture pins the construction value a Jacobian document gives a state without a `default` (spec/plan.md §6.2 item 4, D89): its one band coefficient, $k/x^2$, divides by that state.

## auto

`schedule: "auto"`, no `operators`, no `step`.

**Terms.** One: `(M.x, 0, 0)`, provenance `M`, `pointwise`.

**The operator** `default_pointwise`:
- `locality` `{"class": "pointwise"}`; `writes` `["M.x"]`; `reads` `["M.k", "M.x"]`.
- `structure.class`: `block_diagonal`; one scalar band `(M.x, M.x)` (`rows`, `row_idx`, `col_idx` all `[]`); `block` `{"vars": ["M.x"], "ordering": [1], "pattern": [[1, 1]], "lu_pattern": [[1, 1]]}`; `linear_in_u` false; `depends_on_t` false.
- `stiffness`: `{"class": "nonstiff", "source": "gershgorin"}`: $k/x^2$ is at most about 0.12 at the sampled states.

**Schedule.** One locality class, so the root is the leaf itself (D72):

```json
{"alg": "Tsit5", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "leaf": ["default_pointwise"]}
```

**Other keys.** `step` `{"hit": []}` (adaptive root, D49); `time` `{"start": 0, "end": 1}`; `partition` `{"ranks": [1, 1], "shards": {"root": 1}, "tiles": [1, 1]}` (no index sets, no halo); `precision` `{"root": {…"f64"}}`; `validation` the three entries, all passed.

**Derived documents.** `default_pointwise.rhs.esm` keeps the `ic` equation `ic(M.x) = 2.5` and declares `x` with no `default`. `default_pointwise.jacobian.esm` has no `ic` equation; it declares `M.x` with the `default` `1.0` (§6.2 item 4), `M.k` with its `0.5`, and the band field `JacobianEval.J_1` with `0.0`, and `D(J_1)` is the band coefficient $k/x^2$ (written `-(-(M.k / M.x^2))`). Built as written and evaluated at its own initial state, its right-hand side is finite: `J_1`'s slot holds $0.5 / 1^2 = 0.5$ (both test suites check this with the native right-hand-side program). With the former `default` `0.0` it was infinite.
