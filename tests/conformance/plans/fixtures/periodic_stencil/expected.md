# periodic_stencil: expected plan contents

The model: one state `Transport.q` over `lon × lat × lev` = 8 × 2 × 2, advected along `lon` by a PPM-like six-point upwind-biased flux divergence with weights over offsets −3 … +2, periodic in `lon`. Interior rows `lon` 4–6 read `q[i-3] … q[i+2]`; the five seam rows (`lon` 1, 2, 3, 7, 8) are their own `makearray` regions reading literal, wrapped indices (row 1 reads columns 6, 7, 8, 1, 2, 3; row 8 reads 5, 6, 7, 8, 1, 2), the way EarthSciDiscretizations writes periodic rules. The coefficient is `-(c[i,j,k] / dx)`, `c` = 0.125 and `dx` = 0.125. An `ic` gives `q = 1 + 0.5 sin(π i / 4)`. `solver.time_span` is 0–0.5 s. Both cases give `axes: {"horizontal": ["lon", "lat"], "vertical": "lev"}` (the default would make three horizontal axes; refusal fixture `axes-too-many`).

## Common to both cases

**Terms (§5.1).** One: `(Transport.q, 0, 0)`, provenance `Transport`, the whole `makearray` (a contribution that is not a sum is one summand).

**Locality (§5.2).** Raw `lon` offsets: interior −3 … 2; seam row 1 reads 6, 7, 8 (raw 5, 6, 7, reduced modulo 8 into [−4, 3] to −3, −2, −1) and 1, 2, 3 (0, 1, 2); row 8 reads 1 and 2 (raw −7, −6, reduced to 1, 2). `lat` and `lev` offsets are 0. So the term is `stencil` with radius `{"lat": [0, 0], "lon": [3, 2]}` (3 cells toward lower index, upwind; 2 toward higher). Not `global`: 3 + 2 + 1 = 6 ≤ 8.

**The operator.**
- `locality`: `{"class": "stencil", "radius": {"lat": [0, 0], "lon": [3, 2]}}`.
- `writes` `["Transport.q"]`; `reads` `["Transport.c", "Transport.dx", "Transport.q"]`.
- `structure.class`: `general` (literal column indices at the seam; this is why the footprint is read from the bands, architecture §3.2 pass 4).
- `structure.bands`: 36: six for the interior region (`rows` `[[4, 6], [1, 2], [1, 2]]`, `row_idx` `["i", "j", "k"]`, `col_idx` `[i-3 | i-2 | i-1 | i | i+1 | i+2, "j", "k"]`), and six for each of the five seam regions (`rows` `[[r, r], [1, 2], [1, 2]]`, `row_idx` `[r, "j", "k"]`, literal first column index).
- `structure.stencil.footprint`: `[[-3, 0], [-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0]]` (horizontal axes `lon`, `lat`; `lev` is vertical and not part of the footprint).
- `linear_in_u` true (coefficients read only `c` and `dx`); `depends_on_t` false.
- No `block`, no `column`.

**Derived documents.** `derived/<op>.rhs.esm`: model `Transport` with `q` (shape `["lon", "lat", "lev"]`, default 1.5), `c` (same shape, 0.125), `dx`; its equation `D(Transport.q) = ` the `makearray` as flattened, with every reference fully scoped; its `ic` equation. The Jacobian document has 36 band fields `_jacobian.J_1 … J_36`, each of shape `["lon", "lat", "lev"]`.

**Time.** `{"start": 0, "end": 0.5}`.

## auto

`schedule: "auto"`, no `operators`, no `step`.

- Operator `default_stencil`.
- `stiffness`: `{"class": "nonstiff", "source": "gershgorin"}`. Every row's absolute sum is `(c/dx) · Σ|w|` = 1 × 1.375; $\Delta$ is the interval, 0.5 s (adaptive root, no `step`), so $\rho\Delta$ = 0.69 < 10.
- Root leaf (one locality class):

```json
{"alg": "SSPRK43", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "leaf": ["default_stencil"]}
```

  No `jacobian` or `linsolve`: SSPRK43 forms no Jacobian (D73).
- `step`: `{"hit": []}`.
- `partition`: `{"halo": {"lat": 0, "lon": 3}, "ranks": [1, 1], "shards": {}, "tiles": [1, 1]}` (a stencil leaf has no shards).
- `precision`: `{"root": {…all five roles "f64"}}`.

## partitioned

Operator `transport` selected by `{"locality": "stencil"}`; leaf `SSPRK33` with `dt` 0.01; `time` `{"start": 0, "duration": "0.5 s"}`; `step.dt` 0.1; `ranks` `[2, 1]`, `tiles` `[2, 1]`.

- Operator `transport`; `stiffness` `{"class": "nonstiff", "source": "run_config"}` (SSPRK33 is explicit).
- Root leaf:

```json
{"alg": "SSPRK33", "control": {"abstol": 0.000001, "dt_scope": "global", "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "dt": 0.01, "id": "root", "leaf": ["transport"]}
```

  A fixed-`dt` leaf has no `controller` or `maxiters` (§7.1).
- `time`: `{"start": 0, "end": 0.5}` (the run config's `duration` added to its start).
- `step`: `{"dt": 0.1, "hit": []}`.
- `partition`: `{"halo": {"lat": 0, "lon": 3}, "ranks": [2, 1], "shards": {}, "tiles": [2, 1]}`. `lon` is cut into 2 × 2 = 4 blocks of 2 cells, each needing a 3-cell halo (wider than the block: the halo spans two neighbours).
