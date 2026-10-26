# column_contraction: expected plan contents

The model: one state `Mix.q` over `lon × lat × lev` = 3 × 2 × 5, relaxed toward its column mean, `D(q)[i,j,k] = kmix · (Σ_kk q[i,j,kk] / 5 − q[i,j,k])`, the shape of reseact's PBL mean (a `faq` contraction over `lev`). `kmix` = 0.2. An `ic` gives `q = (10 k + i) / 7`. `solver.time_span` is 0–600 s.

## auto

`schedule: "auto"`, `axes` `{"horizontal": ["lon", "lat"], "vertical": "lev"}`, `step.dt` 600.

**Terms.** One: `(Mix.q, 0, 0)`, provenance `Mix`, the whole `makearray`.

**Locality.** The term's bands are a diagonal band (`col_idx` `["i", "j", "k"]`) and a band whose `lev` column index is the contracted name `kk` (`contracted` `[["kk", 1, 5]]`, or EarthSciASTDiff's renamed-apart name). The contraction's reach covers the vertical axis, so the term is `column` (architecture §3.2 pass 2: "column contractions such as reseact's PBL mean are `column`, not `pointwise`"). Without the `axes` key, `lev` would be horizontal and the term `global` (refusal fixture `auto-stiff-global` shows that path).

**The operator** `default_column`:
- `locality` `{"class": "column"}`; `writes` `["Mix.q"]`; `reads` `["Mix.kmix", "Mix.q"]`.
- `structure.class`: `general` (a contracted column index); `linear_in_u` true; `depends_on_t` false.
- `structure.column`: `{"axis": "lev", "kl": 4, "ku": 4}` (a full-column contraction is dense in the column, $N_v - 1 = 4$).
- `stiffness`: `{"class": "stiff", "source": "gershgorin"}`. Each row: diagonal $k_{mix}(1/5 - 1) = -0.16$, four off-diagonals $k_{mix}/5 = 0.04$; $\rho = 0.32$, $\Delta$ = 600 s, $\rho\Delta = 192 \ge 10$.

**Schedule.** Root leaf (one class):

```json
{"alg": "TRBDF2", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "jacobian": "symbolic", "leaf": ["default_column"], "linsolve": {"banded": {}}}
```

(column, stiff: TRBDF2 with a banded solve, D59, D71).

**Other keys.** `step` `{"dt": 600, "hit": []}`; `time` `{"start": 0, "end": 600}`; `partition` `{"halo": {"lat": 0, "lon": 0}, "ranks": [1, 1], "shards": {"root": 1}, "tiles": [1, 1]}`; `precision` `{"root": {…"f64"}}`; `derived` `default_column.rhs.esm`, `default_column.jacobian.esm`; `validation` the three entries.

**Derived documents.** The right-hand-side document's equation keeps the nested scalar `faq` over `kk` inside the per-cell body; Julia's native compiler builds it (this is the non-lifted form; spec/plan.md §10 item 5). The Jacobian document's contracted band field has shape `["lon", "lat", "lev", <index set of extent 5 for kk>]`, with the extra index set named as EarthSciASTDiff names it.
