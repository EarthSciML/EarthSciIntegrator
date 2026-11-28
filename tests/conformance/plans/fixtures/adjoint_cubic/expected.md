# adjoint_cubic: expected plan contents

spec/toy-problems.md §1.7's per-cell cubic with the scalar factor `a` (default 1) on `c`, `z + z³ = a c`, so the solve has a runtime parameter (D127). The solution is the toy's.

## newton

As `toy_cubic_roots/newton`, with the run config's `adjoint` (functional `sum` of `Cubic.z` at the final time, `wrt` `u0` and `params:numeric`).

**Operator `cubic`.** As `toy_cubic_roots`'s, plus `vjp`: the parameter product's inputs and outputs (spec/plan.md §6.6), `{"lam": {"Cubic.z": "_lam_Cubic_z"}, "vjp": {"Cubic.a": "_vjp_Cubic_a", "Cubic.c": "_vjp_Cubic_c"}}`: a column per parameter of the residual document, the array `c` included, though a runtime differentiates only the scalar `a` (array parameters are constants of a planned run until M4).

**Derived.** cubic.jacobian.esm, cubic.residual.esm, cubic.vjp.esm

**Checkpoint schedule.** The placeholder (spec/plan.md §7.6).
