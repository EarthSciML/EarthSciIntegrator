# toy_cubic_roots: expected plan contents

spec/toy-problems.md §1.7 (`tests/conformance/toys/cubic_roots.esm`): one algebraic equation over `cell`, written with its unknown on the left (`z + z^3 - c ~ 0`, residual `c - z - z^3`), paired with `Cubic.z` by the diagonal band `col_idx = row_idx` (spec/plan.md §5.9.4). The residual document writes it cell by cell (§6.4).

## newton

**Operator `cubic`.** kind `algebraic`; locality `pointwise`; `writes` ['Cubic.z']; structure class `block_diagonal`; residuals `Cubic.residual[1]` paired with `Cubic.z`.

**Schedule.**

```json
{"id": "root", "method": {"converged": {"residual": 1e-12}, "linesearch": "backtracking", "linsolve": {"block_lu": {"pivot": true, "refine": 0}}, "maxiter": 50, "newton": {}}, "solve": "cubic", "unknowns": ["Cubic.z"], "warm_start": "previous"}
```

**Derived.** cubic.jacobian.esm, cubic.residual.esm

## ptc

**Operator `cubic`.** kind `algebraic`; locality `pointwise`; `writes` ['Cubic.z']; structure class `block_diagonal`; residuals `Cubic.residual[1]` paired with `Cubic.z`.

**Schedule.**

```json
{"id": "root", "method": {"converged": {"residual": 1e-10}, "linsolve": {"block_lu": {"pivot": true, "refine": 0}}, "maxiter": 100, "ptc": {"dt0": 0.001, "growth": 2}}, "solve": "cubic", "unknowns": ["Cubic.z"], "warm_start": "initial"}
```

**Derived.** cubic.jacobian.esm, cubic.residual.esm

## newton_inexact

As `newton`, with `newton` `{"inexact": true}`: an inexact Newton–Krylov step with Eisenstat–Walker forcing (spec/methods.md §15.3, D126). The plan writes `inexact` as given; the derived documents are `newton`'s.
