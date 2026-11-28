# box_scaled: expected plan contents

spec/toy-problems.md §1.8's bounded box with a state-free observed upper bound `zcap = 1.5 zmax` and a state-free observed scale `zscale = 0.5 + zmax` (D125). The solution is `z = (0.375, 0)`: `z[1]` at its upper bound 0.375, `z[2]` at 0.

**Operator `box`.** kind `algebraic`; locality `pointwise`... as `toy_box_lcp`'s, with `bounds` `{"Box.z": {"lb": 0, "ub": "Box.zcap"}}`.

## scaled_ptc

`frozen`: `["Box.zscale", "Box.zcap"]` (the scale observed, then the bound observed; spec/plan.md §6.5), so the operator has a `frozen_residual` document and no `frozen_jacobian`.

**Schedule.** `ptc` `{"dt0": 1, "growth": "uncapped"}`, `scale` `{"Box.z": "Box.zscale"}` as given.

**Derived.** box.frozen_residual.esm, box.jacobian.esm, box.residual.esm

## recover_method

`frozen`: `["Box.zcap"]` (the bound observed). `ptc` `{"dt0": 1e-6, "growth": 2}` with `maxiter` 2, which cannot converge; `scale` `{"Box.z": 2}`; `recover` `[{"method": {"newton": {}, "linesearch": "backtracking", "maxiter": 50}}]`, the step's options defaulted (spec/plan.md §7.1).

**Derived.** box.frozen_residual.esm, box.jacobian.esm, box.residual.esm
