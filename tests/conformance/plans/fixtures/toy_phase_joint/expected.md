# toy_phase_joint: expected plan contents

spec/toy-problems.md §1.9: two residual blocks of different extents, so the canonical pairing is unique: `Phase.residual[1]` (over `cell`) with `Phase.w`, `Phase.residual[2]` (over `cell` × `salt`) with `Phase.s` (§5.9.5). In the residual document the solved unknowns' equations come in code-point order, `Phase.s` before `Phase.w` (§6.1.2).

## ptc

**Operator `phase`.** kind `algebraic`; locality `global`; `writes` ['Phase.s', 'Phase.w']; structure class `block_diagonal`; residuals `Phase.residual[1]` paired with `Phase.w`, `Phase.residual[2]` paired with `Phase.s`.

**Schedule.**

```json
{"id": "root", "method": {"converged": {"residual": 1e-12}, "linsolve": {"dense_lu": {}}, "maxiter": 2000, "ptc": {"dt0": 0.1, "growth": 2}}, "solve": "phase", "unknowns": ["Phase.s", "Phase.w"], "warm_start": "initial"}
```

**Derived.** phase.jacobian.esm, phase.residual.esm
