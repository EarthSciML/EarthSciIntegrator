# toy_robertson_dae: expected plan contents

spec/toy-problems.md §1.10: a `coupled` node with a term operator (`kinetics`, selected by provenance) and an algebraic operator (`conservation`, selected by residual id), an index-1 DAE (spec/run-config.md §4.3). `Rober.y3` is a solved unknown, so the `kinetics` right-hand-side document declares it as a state with a zero derivative and its Jacobian has the column y3 (D121). The resolved node carries the default `initialize`.

## rodas4

**Operator `conservation`.** kind `algebraic`; locality `pointwise`; `writes` ['Rober.y3']; structure class `block_diagonal`; residuals `Rober.residual[1]` paired with `Rober.y3`.

**Operator `kinetics`.** kind `rhs`; locality `pointwise`; `writes` ['Rober.y1', 'Rober.y2']; structure class `block_diagonal`.

**Schedule.**

```json
{"alg": "Rodas4", "control": {"abstol": 1e-10, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 1e-06}, "coupled": ["kinetics", "conservation"], "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "initialize": {"converged": {"residual": 1e-10}, "linesearch": "backtracking", "linsolve": {"block_lu": {"pivot": true, "refine": 0}}, "maxiter": 100, "newton": {}}, "jacobian": "symbolic", "linsolve": {"dense_lu": {}}}
```

**Derived.** conservation.jacobian.esm, conservation.residual.esm, kinetics.jacobian.esm, kinetics.rhs.esm

## trbdf2

As `rodas4`, with `alg` `TRBDF2`, a stiffly accurate ESDIRK method of spec/run-config.md §4.3; the operators, structure and derived documents are the same.

## kvaerno4

As `rodas4`, with `alg` `Kvaerno4`, a stiffly accurate ESDIRK method of spec/run-config.md §4.3; the operators, structure and derived documents are the same.

## inconsistent

As `rodas4` on `tests/conformance/toys/robertson_dae_inconsistent.esm`, whose `y3` starts at 0.5, an inconsistent value. The plan is `rodas4`'s with the other model and that `default`; the runtime's consistent initialization (spec/methods.md §14.4) returns `y3 = 0` before the first step.

## kvaerno4_adjoint

As `kvaerno4`, with the run config's `adjoint` (functional `sum` of `Rober.y1` at the final time, `wrt` `u0`; spec/adjoint.md §5.10). The algebraic operator `conservation` gains `vjp` (spec/plan.md §6.6), whose columns are the parameters of its residual document (`Rober.k1`, `Rober.k2`, `Rober.k3`, all zero since the conservation residual reads none of them), and `derived` gains conservation.vjp.esm. The right-hand-side operator `kinetics` gets no parameter product (U7, M4). The checkpoint schedule is the placeholder.

## trbdf2_adjoint

As `trbdf2`, with the same `adjoint` and the same additions as `kvaerno4_adjoint`.
