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
