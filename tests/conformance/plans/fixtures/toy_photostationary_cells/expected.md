# toy_photostationary_cells: expected plan contents

The `.esm` version of spec/toy-problems.md §1.4 (`tests/conformance/toys/photostationary_cells.esm`): 16 cells of NO2/NO/O3 photochemistry, photolysis $j_c(t) = \tfrac12 j_0 (1 + \sin(\omega t + \pi(i-1)/8))$, planned with M1 case `photostationary_cells.Kvaerno4.adaptive`'s run config with `linsolve` left out (the M1 case gives `{"block_lu": {"pivot": false}}`).

## kvaerno4

**Terms.** Three, one per species: each equation's right-hand side is a single `faq` over `cell`, so one summand (`(Photo.NO2, 0, 0)`, `(Photo.NO, 0, 0)`, `(Photo.O3, 0, 0)`), provenance `Photo`, `pointwise`.

**Axes.** Default: horizontal `["cell"]`.

**Operator `photochemistry`.** `writes` `["Photo.NO", "Photo.NO2", "Photo.O3"]`; `reads` adds `"Photo.j0", "Photo.k", "Photo.omega"`. `structure`: `block_diagonal`; bands for all nine species pairs over `rows` `[[1, 16]]`, `row_idx` `["i"]`, `col_idx` `["i"]`; `block.vars` `["Photo.NO", "Photo.NO2", "Photo.O3"]` (canonical order, which differs from §1.4's NO2, NO, O3: the M2 case's `layout` maps results back), `ordering` `[1, 2, 3]`, full 3 × 3 `pattern` and `lu_pattern`; `linear_in_u` false; **`depends_on_t` true** (`sin(omega · t)`). `stiffness` `{"stiff", "run_config"}`.

**Schedule.** Root leaf; `linsolve` resolves by the pointwise row (block of 3):

```json
{"alg": "Kvaerno4", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 0.000001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "jacobian": "symbolic", "leaf": ["photochemistry"], "linsolve": {"block_lu": {"pivot": true, "refine": 0}}}
```

**Other keys.** `time` `{"start": 0, "end": 3600}`; `step` `{"dt": 600, "hit": []}`; `partition` `{"halo": {"cell": 0}, "ranks": [1, 1], "shards": {"root": 1}, "tiles": [1, 1]}`; `precision` `{"root": {…"f64"}}`.

**Runtime use.** An SDIRK leaf needs only `rhs` and `jac`, so this time-dependent problem runs at M2 under Kvaerno4; its Rosenbrock cases wait for a time derivative (spec/toy-problems.md §4.2).
