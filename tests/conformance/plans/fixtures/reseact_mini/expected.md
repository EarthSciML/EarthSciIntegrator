# reseact_mini: expected plan contents

The model is the M3 toy `reseact_mini` (spec/toy-problems.md §5.1; `../../../toys/reseact_mini.esm`): states `Mini.NO2`, `Mini.NO`, `Mini.O3` over `lon × lat × lev` = 6 × 4 × 5 (120 cells, 360 DOFs), the data-fed parameter `Mini.TS[rec, lon, lat]` (two hourly records, `discrete`), and one equation per species whose right-hand side is the sum of a pointwise chemistry `faq`, a periodic upwind advection `makearray` along `lon`, and a column-mean `faq` along `lev`. Both cases give `axes` `{"horizontal": ["lon", "lat"], "vertical": "lev"}`, operators `transport` = locality `stencil` and `column`, `chemistry` = locality `pointwise`, the schedule `sequence` [`transport` `SSPRK43`, `chem` = `chemistry`], `step` `{"dt": 300, "hit": ["forcing_refresh", "output"]}` and `output.saveat` 1800. These goldens are written by the Julia planner (spec/plan.md §9); this file says what to check them against.

## Common to both cases

**Terms (§5.1).** Three per species, nine in all, provenance `Mini` (the model's own equation): for species X in canonical order (`Mini.NO`, `Mini.NO2`, `Mini.O3`), summand 0 the chemistry `faq` (pointwise), summand 1 the advection `makearray` (stencil, radius `lon` [1, 0], `lat` [0, 0]: the seam row reads cell 6, a periodic wrap, §5.2.3), summand 2 the column mean (column). The Rust planner of M2 plans the model with these localities (checked on a `TRBDF2`, `dt_scope: global` variant).

**Operators.** `chemistry`: pointwise, `writes` the three species, `forcing` `["Mini.TS"]`, `structure.class` `block_diagonal` with a 3-variable `block`, `depends_on_t` true (photolysis reads $t$, and the blend weight `w` reads $t$), `linear_in_u` false. `transport`: locality `stencil` (the coarsest of its terms' localities, stencil and column) with radius `{"lat": [0, 0], "lon": [1, 0]}` (the seam reads cell 6 from cell 1, a periodic wrap of reach 1), `structure.class` `general`, `structure.stencil.footprint` `[[-1, 0], [0, 0]]`, 18 bands, `depends_on_t` false, `linear_in_u` true, no `forcing`. `chemistry`: 9 bands.

**Forcing.** `forcing` lists `Mini.TS`: shape `["rec", "lon", "lat"]`, source `Met`, file variable `TS`, cadence `discrete`, `records_per_sample` 2.

**Partition.** `ranks` and `tiles` `[1, 1]`; `halo` `{"lat": 0, "lon": 1}`.

## cell

The chemistry leaf `chem` gives `Rosenbrock23`, `jacobian` `symbolic`, `linsolve` `{"block_lu": {"pivot": false}}`, `control` without `dt_scope`, and `partition` `{"shards": 4, "lanes": 8}`.

- **`dt_scope`** (§5.5.3, D97): the resolved `chem` control has `"dt_scope": "cell"` (pointwise, `block_diagonal` storage); the transport leaf's is `"global"`.
- **Partition** (§7.4): `shards` `{"chem": 4}`; `lanes` `{"chem": 8}`; no `reshard` key (no leaf has `shard` scope).
- **Derived documents** (§6.4, D102): besides the four `rhs` and `jacobian` documents, `derived/chemistry.time_derivative.esm`, kind `time_derivative`, operator `chemistry`. Its model `JacobianEval` holds one field `J_k` per time band of the chemistry's right-hand-side document, in equation order (the three species, each reading $t$ through the photolysis rate and through `Mini.w`'s blend); no transport time-derivative document (transport does not depend on $t$, and no Rosenbrock leaf advances it).
- **Validation** (§8.1): entries `rhs_sum`; `chemistry` `band_model`, `non_empty`; `transport` `band_model`, `non_empty`, sorted by `(operator, check)`. No `time_band_model` entry: the Julia planner makes that check and reports it in `plan.report.json`, the Rust planner skips it (D107). The chemistry's band-value file carries `time_entries` per sample.

## shard

The chemistry leaf `chem` gives `TRBDF2`, `jacobian` `symbolic` (its `linsolve` resolves to `block_lu`, §5.5.3), `control.dt_scope` `"shard"`, and `partition` `{"shards": 3, "reshard": {"every": 2}}`.

- **`dt_scope`**: `"shard"` as given.
- **Partition**: `shards` `{"chem": 3}`; `reshard` `{"chem": {"every": 2}}`; no `lanes` key.
- **Derived documents**: the four `rhs` and `jacobian` documents only: `TRBDF2` is not a Rosenbrock method, so no time-derivative document.
- **Validation**: as at M2, no `time_band_model`.

## coupled

The chemistry is advanced by a `coupled` node `chem` (`TRBDF2`, `jacobian` `symbolic`, `control` without `dt_scope`, no `partition`) instead of a `leaf`.

- **`dt_scope`** (spec/partitioning.md §2.1, D97): `"global"`. A `coupled` node is never sharded, so the pointwise `cell` default does not apply to it, although its operator's storage is `block_diagonal`.
- **Partition** (§7.4): `shards` `{}` (a `coupled` node gets no entry); no `lanes` and no `reshard` key.
- **Derived documents**: the four `rhs` and `jacobian` documents only (`TRBDF2` is not a Rosenbrock method).
