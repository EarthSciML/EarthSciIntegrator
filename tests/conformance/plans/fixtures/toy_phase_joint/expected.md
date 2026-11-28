# toy_phase_joint: expected plan contents

spec/toy-problems.md §1.9: two residual blocks of different extents, so the canonical pairing is unique: `Phase.residual[1]` (over `cell`) with `Phase.w`, `Phase.residual[2]` (over `cell` × `salt`) with `Phase.s` (§5.9.5). In the residual document the solved unknowns' equations come in code-point order, `Phase.s` before `Phase.w` (§6.1.2).

## ptc

**Operator `phase`.** kind `algebraic`; locality `global`; `writes` ['Phase.s', 'Phase.w']; structure class `general` (a band whose column has another rank than its row is never diagonal, EarthSciASTDiff#86); residuals `Phase.residual[1]` paired with `Phase.w`, `Phase.residual[2]` paired with `Phase.s`.

**Schedule.**

```json
{"id": "root", "method": {"converged": {"residual": 1e-12}, "linsolve": {"dense_lu": {}}, "maxiter": 2000, "ptc": {"dt0": 0.1, "growth": 2}}, "solve": "phase", "unknowns": ["Phase.s", "Phase.w"], "warm_start": "initial"}
```

**Derived.** phase.jacobian.esm, phase.residual.esm

## per_cell

As `ptc`, with `axes` `{"horizontal": ["cell"]}`, so `salt` lies inside a cell (D123). The sums over salt keep the operator **pointwise**, and its structure is `cell_block`: one block per cell over the cell axis `cell`, holding `Phase.s` at salts 1 and 2 and `Phase.w` (3 × 3), the residual of `w` coupling to both salts and each salt's to `w` (`pattern` and `lu_pattern` `[[1,1],[1,3],[2,2],[2,3],[3,1],[3,2],[3,3]]`). The solve's `linsolve` resolves to `{"block_lu": {"pivot": true, "refine": 0}}`. This is the shape of MESA's per-bin blocks.

## adjoint

As `ptc`, with the run config's `adjoint` (functional `sum` of `Phase.w` at the final time, `wrt` `u0` and `params:numeric`). The operator gains `vjp` (spec/plan.md §6.6), whose only parameter column is the scalar `Phase.kappa`, and `derived` gains phase.vjp.esm; the checkpoint schedule is the placeholder.
