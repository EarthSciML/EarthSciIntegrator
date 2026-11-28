# toy_picard_roots: expected plan contents

spec/toy-problems.md §1.11 (`tests/conformance/toys/picard_roots.esm`): one algebraic equation over `cell`, written with its unknown on the left (`ν z - c ~ 0`, residual `c - ν z`), with the observed `ν = 1 + z^2`, paired with `Picard.z` by the diagonal band `col_idx = row_idx` (spec/plan.md §5.9.4). The residual document writes it cell by cell (§6.4). The frozen documents are spec/plan.md §6.5's, with the one frozen observed `Picard.nu` as `FrozenEval.frozen_1`.

## picard

**Operator `picard`.** kind `algebraic`; locality `pointwise`; `writes` ['Picard.z']; structure class `block_diagonal`, `linear_in_u` false; residuals `Picard.residual[1]` paired with `Picard.z`. `frozen_bands`: the diagonal band of `Picard.z` on `FrozenEval.frozen_1` and on `Picard.z` (coefficients `-z` and `-frozen_1`), both `col_idx = row_idx` over `[1, 4]`.

**Schedule.**

```json
{"id": "root", "method": {"converged": {"residual": 1e-12}, "linesearch": "none", "linsolve": {"block_lu": {"pivot": true, "refine": 0}}, "maxiter": 100, "picard": {"freeze": ["Picard.nu"]}}, "solve": "picard", "unknowns": ["Picard.z"], "warm_start": "previous"}
```

**Derived.** picard.frozen_jacobian.esm, picard.frozen_residual.esm, picard.jacobian.esm, picard.residual.esm

## rel_change

As `picard`, with `converged` `{"rel_change": "Picard.nu", "tol": 1e-12}` and `maxiter` 200. The `rel_change` observed is the frozen one, so the frozen documents are the same.

## newton_rel_change

`newton` with `converged` `{"rel_change": "Picard.nu", "tol": 1e-12}`: the frozen residual document alone (no `frozen_jacobian`, no `frozen_bands`), with `FrozenEval.frozen_1` holding `Picard.nu`.

**Derived.** picard.frozen_residual.esm, picard.jacobian.esm, picard.residual.esm
