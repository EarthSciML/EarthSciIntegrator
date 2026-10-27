# cross_component_term: expected plan contents

The case architecture §3.2 item 1 (decisions §3.2) says provenance alone cannot place: a term that reads two components' variables. Model `Chem` has scalar states `O3`, `NO`, parameter `k`, and an observed `ratio = NO / O3`:

- `D(O3) = -k · O3 · NO`
- `D(NO) = k · ratio − k · O3 · NO`

A `couple` entry (coupling 0, systems `Chem`, `Dep`) adds two sinks: `-vd · O3 / Met.h` to `O3` (it reads `Dep.vd` and `Met.h`) and `-kNO · NO` to `NO` (it reads `Dep.kNO`). `solver.time_span` is 0–600 s. There are no index sets.

## Common to both cases

**Terms (§5.1)**, in order:

| variable | c | s | provenance | owner (§5.1.2) | summand |
|---|---|---|---|---|---|
| Chem.O3 | 0 | 0 | Chem | Chem | `-1 * Chem.k * Chem.O3 * Chem.NO` |
| Chem.O3 | 1 | 0 | coupling:0 | none: reads `Dep` and `Met` | `-1 * Dep.vd * Chem.O3 / Met.h` |
| Chem.NO | 0 | 0 | Chem | Chem | `Chem.k * Chem.ratio` |
| Chem.NO | 0 | 1 | Chem | Chem | `-(Chem.k * Chem.O3 * Chem.NO)` (the right operand of a binary `-`, negated) |
| Chem.NO | 1 | 0 | coupling:0 | Dep: reads only `Dep` besides `Chem` | `-1 * Dep.kNO * Chem.NO` |

All are `pointwise` (scalar states). The observed `Chem.ratio` defines no term.

**Structures.** Every operator here has scalar rows and columns: each band has `rows` `[]`, `row_idx` `[]`, `col_idx` `[]`, `class` is `block_diagonal`, and `block.vars` lists the operator's state columns sorted.

**Time** `{"start": 0, "end": 600}`; `step` `{"dt": 60, "hit": []}`. **Partition**: no index sets, so no horizontal axis: `ranks` and `tiles` `[1, 1]`, no `halo` key, and `shards` one per leaf, each 1. **Precision**: every leaf all `"f64"`.

## by-label

`chemistry` = `{"components": "Chem"}`, `deposition` = `{"terms": ["coupling:0"]}`; schedule `sequence[leaf chemistry Rodas4, leaf deposition ForwardEuler dt "step"]`.

- `chemistry`: the three `Chem` terms. `writes` `["Chem.NO", "Chem.O3"]`; `reads` `["Chem.NO", "Chem.O3", "Chem.k", "Chem.ratio"]` (the observed is read, and its definition reads only names already listed). Bands, in order: row `Chem.O3` columns `Chem.NO`, `Chem.O3`; row `Chem.NO` columns `Chem.NO`, `Chem.O3` (the derived document lists `O3`'s equation first, as the flattened model does; `ratio` is inlined into the coefficients). `block` `{"vars": ["Chem.NO", "Chem.O3"], "ordering": [1, 2], "pattern": [[1,1],[1,2],[2,1],[2,2]], "lu_pattern": same}`. `linear_in_u` false. `stiffness` `{"stiff", "run_config"}` (Rodas4).
- `deposition`: both coupling terms: the owned `NO` sink and the owner-less `O3` sink, which only a `terms` (or `variables`) selector can take. `writes` `["Chem.NO", "Chem.O3"]`; `reads` `["Chem.NO", "Chem.O3", "Dep.kNO", "Dep.vd", "Met.h"]`. Bands: row `Chem.O3` column `Chem.O3`; row `Chem.NO` column `Chem.NO`. `block` `{"vars": ["Chem.NO", "Chem.O3"], "ordering": [1, 2], "pattern": [[1,1],[2,2]], "lu_pattern": [[1,1],[2,2]]}`. `linear_in_u` true. `stiffness` `{"nonstiff", "run_config"}`.
- Schedule:

```json
{"id": "root", "sequence": [
  {"alg": "Rodas4", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "cell", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root.0", "jacobian": "symbolic", "leaf": ["chemistry"], "linsolve": {"block_lu": {"pivot": true, "refine": 0}}},
  {"alg": "ForwardEuler", "control": {"abstol": 0.000001, "dt_scope": "cell", "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "dt": "step", "id": "root.1", "leaf": ["deposition"]}
]}
```

  `jacobian` and `linsolve` were omitted on a named implicit method, so they resolve by the pointwise row (§5.5.3); `dt_scope` resolves to `cell` for pointwise leaves whatever the method.
- `partition.shards` `{"root.0": 1, "root.1": 1}`.
- [`by-label.expected-plan.json`](by-label.expected-plan.json) is this case's full `plan.json` written by hand in canonical form, except that the derived documents' `sha256` values are 64 zeros: the golden must equal it after those four values are replaced.

## by-owner

`chemistry` = `{"components": "Chem"}`, `deposition` = `{"components": "Dep"}`, `scavenging` = `{"terms": ["coupling:0"], "variables": ["Chem.O3"]}`; one leaf `["chemistry", "deposition", "scavenging"]` with `Rodas4`.

- `deposition` matches the `NO` sink through its owner `Dep` (D65); `scavenging` matches the `O3` sink, which has no owner. Neither sink matches two selectors. With `scavenging` removed, planning is refused with `ambiguous_term_owner` (refusal fixture `ambiguous-term-owner`), because the `O3` sink would fall to `default_pointwise` while a selector uses `components`.
- `chemistry` as in `by-label`. `deposition`: `writes` `["Chem.NO"]`, `reads` `["Chem.NO", "Dep.kNO"]`, one band (`Chem.NO`, `Chem.NO`), `block.vars` `["Chem.NO"]`, `pattern` `[[1,1]]`. `scavenging`: `writes` `["Chem.O3"]`, `reads` `["Chem.O3", "Dep.vd", "Met.h"]`, one band (`Chem.O3`, `Chem.O3`), `block.vars` `["Chem.O3"]`. All three `{"stiff", "run_config"}`.
- Schedule: the run config writes a one-child `sequence`, which §7.1 keeps, so the root is the sequence and the leaf is `root.0`:

```json
{"id": "root", "sequence": [
  {"alg": "Rodas4", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "cell", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root.0", "jacobian": "symbolic", "leaf": ["chemistry", "deposition", "scavenging"], "linsolve": {"block_lu": {"pivot": true, "refine": 0}}}
]}
```

  (The block size is 2, the union of the operators' block variables.)
- `partition.shards` `{"root.0": 1}`; `precision` keyed `root.0`.
