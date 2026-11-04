# forced_column: expected plan contents

Data-fed parameters (spec/plan.md §4.1, §5.5.1, §6.3; D87). The planner never opens a data file: `met/met_*.nc` and `surface.nc` do not exist.

The model: one state `Col.c` over `x × lev` = 2 × 4 (`default` 1.0), with an `ic` equation `c = Col.c0`. Its equation is the sum of two `makearray` contributions' summands:

- a column relaxation toward the column mean, `kmix · (Tb[i,k] / T0) · (Σ_kk c[i,kk] / 4 − c[i,k])`, whose rate reads the time blend `Tb = T[1,i,k] + w (T[2,i,k] − T[1,i,k])` of a two-record temperature field, `w = t/dtrec − floor(t/dtrec)` (the observed `Col.w`, `dtrec` = 10800 s, the records' spacing), as reseact's GEOS-FP fields are blended;
- a pointwise source and loss, `g · s0 − kloss · c[i,k]`.

`kmix` = 0.002, `T0` = 300, `s0` = 5e-5, `kloss` = 1e-4. `domain.temporal.reference_time` is 2016-01-01T00:00:00Z; `solver.time_span` is 0–21600 s.

Data sources: `MetFields`, `kind` `grid`, a relative `{date:%Y%m%d}` `url_template`, a `temporal` block with `start`, `frequency` `PT3H` and `file_period` `P1D` (so `discrete`), and `metadata.esio_format` `netcdf`; `Surface`, `kind` `static`, no `temporal` (so `const`).

Data-fed parameters (the `loader_fields` entries):

| Parameter | Declared | Source, file variable | Cadence | Records | Used by |
|---|---|---|---|---|---|
| `Met.T` | `Met` declares it with no shape and `default` 280; a `param_to_var` coupling maps it onto `Col.T[rec, x, lev]` (`default` 300, ignored), so it takes that shape (§10 item 16) | `MetFields`, `T` | discrete | 2 (`update.from.records_per_sample`) | the column term |
| `Col.g` | scalar, `default` 2.0 | `Surface`, `G` | const | 1 | the pointwise term |
| `Col.c0` | `[x, lev]`, `default` 1.5 | `Surface`, `C0` | const | 1 | the `ic` equation only |
| `Met.QV` | `[x, lev]`, no `default` | `MetFields`, `QV` | discrete | 1 | nothing: unused, not refused, and absent from the plan, the derived documents and the band-value files |

## named

`axes` `{"horizontal": ["x"], "vertical": "lev"}`; `step` `{"dt": 3600, "hit": ["forcing_refresh"]}`; schedule a `sequence` of a `TRBDF2` leaf (`dt` 600) for `default_column` and an `alg`-less (so `"auto"`) leaf for `default_pointwise`.

**Terms.** Two, both of `Col.c`, contribution 0, provenance `Col`: summand 0 the column relaxation (`column`: its contracted `kk` reach covers the vertical axis), summand 1 the source and loss (`pointwise`). No operator is named, so they go to `default_column` and `default_pointwise`.

**`forcing`** (§3): `Col.c0` `{"cadence": "const", "file_variable": "C0", "records_per_sample": 1, "shape": ["x", "lev"], "source": "Surface"}`, `Col.g` `{"cadence": "const", "file_variable": "G", "records_per_sample": 1, "shape": [], "source": "Surface"}`, `Met.T` `{"cadence": "discrete", "file_variable": "T", "records_per_sample": 2, "shape": ["rec", "x", "lev"], "source": "MetFields"}`. No `Met.QV`.

**`default_column`**: `locality` `column`; `writes` `["Col.c"]`; `reads` `["Col.T0", "Col.c", "Col.dtrec", "Col.kmix", "Col.w", "Met.T"]`; `forcing` `["Met.T"]`. `structure.class` `general`, `linear_in_u` true, `depends_on_t` true (through `w`), `column` `{"axis": "lev", "kl": 3, "ku": 3}`; two bands (the diagonal and the contracted `kk_j` column). `stiffness` `{"class": "stiff", "source": "run_config"}` (the leaf names `TRBDF2`; under `"auto"` its Jacobian, which reads `Met.T`, would be refused: refusal fixture `auto-stiffness-reads-forcing`).

**`default_pointwise`**: `locality` `pointwise`; `writes` `["Col.c"]`; `reads` `["Col.c", "Col.g", "Col.kloss", "Col.s0"]`; `forcing` `["Col.g"]`. `structure` `block_diagonal`, block `{"vars": ["Col.c"], "ordering": [1], "pattern": [[1, 1]], "lu_pattern": [[1, 1]]}`, `linear_in_u` true, `depends_on_t` false. Its Jacobian (`-kloss`) reads no data-fed parameter, so rule 4 applies: `stiffness` `{"class": "nonstiff", "source": "gershgorin"}`, ρ = 1e-4, Δ = 3600 s, ρΔ = 0.36. The `"auto"` leaf resolves to `Tsit5`, `dt_scope` `cell`.

**Other keys.** `time` `{"start": 0, "end": 21600}`; `partition` `{"halo": {"x": 0}, "ranks": [1, 1], "shards": {"root.0": 1, "root.1": 1}, "tiles": [1, 1]}`; `precision` all `f64` for `root.0` and `root.1`; `validation` the five entries.

**Derived documents** (§6.3). Each operator's two documents declare `Met.T` (in `models.Met`), `Col.g` and `Col.c0` and not `Met.QV`:
- `default_column.*`: `Met.T` with `"update": {"kind": "data", "source": "MetFields", "from": {"file_variable": "T"}}` and `shape` `["rec", "x", "lev"]`, `default` 280.0; `Col.g` and `Col.c0` as ordinary parameters (no `update`); `data_sources` `{"MetFields": {"kind": "grid", "source": {"url_template": "{esm_integrator_forcing}"}}}`.
- `default_pointwise.*`: `Col.g` with `"update": {"kind": "data", "source": "Surface", "from": {"file_variable": "G"}}`; `Met.T` and `Col.c0` ordinary; `data_sources` `{"Surface": {"kind": "static", "source": {"url_template": "{esm_integrator_forcing}"}}}`.

The right-hand-side documents keep the `ic` equation `c = Col.c0`, so every document builds the planner's u0 = 1.5 everywhere (`Col.c0` bound to its default).

**Band values** (§8.2). Each sample carries `forcing`: the sampled forcing values of §5.5.1 at every element of `Col.c0` (8), `Col.g` (1, written `Col.g`) and `Met.T` (16), canonical forcing order `Col.c0`, `Col.g`, `Met.T`; no `Met.QV`.
