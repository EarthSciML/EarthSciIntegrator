# toy_advection_reaction: expected plan contents

The `.esm` version of spec/toy-problems.md §1.5 (`tests/conformance/toys/advection_reaction.esm`): model `Reaction` owns `A` and `B` (64 cells, periodic) and their exchange; model `Transport` authors `D(Reaction.A)`, first-order upwind with the seam cell reading cell 64; `operator_compose` joins them. Planned with the run config of M2 case `advection_reaction.strang.order` (M1's, with `operators` and an explicit `dt_scope`).

## strang

**Terms.** Three:

| variable | c | s | provenance | locality |
|---|---|---|---|---|
| Reaction.A | 0 | 0 | Reaction | pointwise |
| Reaction.A | 1 | 0 | Transport | stencil, radius `cell` [1, 0] |
| Reaction.B | 0 | 0 | Reaction | pointwise |

The Transport term's bands: interior rows 2–64 read `A[i]` (offset 0) and `A[i-1]` (−1); the seam row 1 reads `A[1]` and `A[64]` (raw 63, reduced modulo 64 to −1). Default axes: horizontal `["cell"]`.

**Operators.**
- `transport` (`{"components": "Transport"}`): `locality` `{"class": "stencil", "radius": {"cell": [1, 0]}}`; `writes` `["Reaction.A"]`; `reads` `["Reaction.A", "Transport.dx", "Transport.v"]`; `structure.class` `general` (the literal seam column); four bands; `stencil.footprint` `[[-1], [0]]`; `linear_in_u` true; `depends_on_t` false; `stiffness` `{"nonstiff", "run_config"}`.
- `reaction` (`{"components": "Reaction"}`): `pointwise`; `writes` `["Reaction.A", "Reaction.B"]`; `reads` adds `"Reaction.k1", "Reaction.k2"`; `block_diagonal`, `block` `{"vars": ["Reaction.A", "Reaction.B"], "ordering": [1, 2], "pattern": [[1,1],[1,2],[2,1],[2,2]], "lu_pattern": the same}`; `linear_in_u` true; `stiffness` `{"stiff", "run_config"}`.

**Schedule.**

```json
{"id": "root", "strang": [
  {"alg": "SSPRK43", "control": {"abstol": 1e-12, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 1e-10}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root.0", "leaf": ["transport"]},
  {"alg": "Rodas4", "control": {"abstol": 1e-12, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 1e-10}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root.1", "jacobian": "symbolic", "leaf": ["reaction"], "linsolve": {"block_lu": {"pivot": false, "refine": 0}}}
]}
```

**Other keys.** `time` `{"start": 0, "end": 0.5}`; `step` `{"dt": 0.05, "hit": []}`; `partition` `{"halo": {"cell": 1}, "ranks": [1, 1], "shards": {"root.1": 1}, "tiles": [1, 1]}`; `precision` `root.0` and `root.1` all `"f64"`.

**Runtime use.** The `transport` leaf's DOFs are `A`'s 64 cells (its `writes`), as the hand-written operator's are; `B` is a constant of that leaf (spec/operator-interface.md §1).
