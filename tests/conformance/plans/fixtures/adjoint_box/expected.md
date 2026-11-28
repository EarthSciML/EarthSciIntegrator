# adjoint_box: expected plan contents

spec/toy-problems.md §1.8's bounded box with the scalar factor `beta` (default 1) on `b`, the scalar parameter lower bound `zlo` (default 0) and the state-free observed upper bound `zcap = cap zmax` with the scalar `cap` (default 1) (D127). The solution is the toy's, `z = (0.25, 0)`, with `z[1]` at its upper bound and `z[2]` at its lower bound, so both bound paths of spec/adjoint.md §4.6 are active.

## newton

As `toy_box_lcp/newton`, with the run config's `adjoint` (functional `sum` of `Box.z` at the final time, `wrt` `u0` and `params:numeric`).

**Operator `box`.** `bounds` `{"Box.z": {"lb": "Box.zlo", "ub": "Box.zcap"}}`; `frozen` `["Box.zcap"]` (the bound observed); `vjp`, inputs `{"Box.z": "_lam_Box_z"}` and an output per parameter of the residual document (`Box.b`, `Box.beta`, `Box.cap`, `Box.zlo`, `Box.zmax`; only `beta` has a nonzero column); `frozen_vjp`, since a bound references a state-free observed, inputs for `Box.z` and `FrozenEval.frozen_1`, with the same outputs (only `cap` has a nonzero column) (spec/plan.md §6.6).

**Derived.** box.frozen_residual.esm, box.frozen_vjp.esm, box.jacobian.esm, box.residual.esm, box.vjp.esm

**Checkpoint schedule.** The placeholder (spec/plan.md §7.6).
