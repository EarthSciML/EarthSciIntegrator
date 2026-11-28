# toy_box_lcp: expected plan contents

spec/toy-problems.md §1.8: a bounded unknown (`lb` 0, `ub` the parameter `Box.zmax`) whose residual contracts over `k` (`b - A z`), paired with `Box.z` by the contracted diagonal band (§5.9.4). Locality is `global` because the contraction's reach covers `k`, a horizontal axis by default (§5.2.3).

## newton

**Operator `box`.** kind `algebraic`; locality `global`; `writes` ['Box.z']; structure class `general`; residuals `Box.residual[1]` paired with `Box.z`.

**Schedule.**

```json
{"id": "root", "method": {"converged": {"residual": 1e-12}, "linesearch": "none", "linsolve": {"dense_lu": {}}, "maxiter": 50, "newton": {}}, "solve": "box", "unknowns": ["Box.z"], "warm_start": "previous"}
```

**Derived.** box.jacobian.esm, box.residual.esm

## ptc

**Operator `box`.** kind `algebraic`; locality `global`; `writes` ['Box.z']; structure class `general`; residuals `Box.residual[1]` paired with `Box.z`.

**Schedule.**

```json
{"id": "root", "method": {"converged": {"residual": 1e-12}, "linsolve": {"dense_lu": {}}, "maxiter": 2000, "ptc": {"dt0": 0.1, "growth": 2}}, "solve": "box", "unknowns": ["Box.z"], "warm_start": "previous"}
```

**Derived.** box.jacobian.esm, box.residual.esm

## newton_inexact

As `newton`, with `newton` `{"inexact": true}`: an inexact Newton–Krylov step with Eisenstat–Walker forcing (spec/methods.md §15.3, D126). The plan writes `inexact` as given; the derived documents are `newton`'s.
