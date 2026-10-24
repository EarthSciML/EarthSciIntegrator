# Toy problems and the M1 conformance cases

**Status:** M1 draft, 2026-10-23. Normative for both implementations.

M1's gate (architecture §14) is conformance tiers 2 and 3 on toy problems. The problems are written by hand in each language behind the operator interface ([`operator-interface.md`](operator-interface.md)), not compiled from `.esm` documents (D53): Rust has no public reentrant right-hand side until U4 lands. `.esm` versions of the same problems join the suite once it does, and must give the same results.

Each problem below fixes its equations, operators, layout, parameters, initial state and interval exactly. Both implementations MUST implement every listed operator call (operator-interface.md §3) with hand-derived derivatives, not automatic differentiation, so the derivatives themselves are under test. The Julia implementation's ForwardDiff Jacobian of `rhs` is a test oracle for them.

Indices are 0-based. "Cell" coordinates are $x_c = (c + \tfrac12)/\text{ncells}$. The names given to state components ($y$, $\mathrm{O_3}$, $A$, …) are the variable names a run config uses for them, for example in `adjoint.functional`; a problem with cells has one index set, `cell`. At M1 every toy leaf runs with `dt_scope: global` (tape.md covers no other scope until M3), whatever D17's default for its locality. A problem's operators exist under the names given here, so a case's run config names them in its schedule and has no `operators` key.

## 1. Problems

### 1.1 `forced_decay`

A linear, non-autonomous scalar problem with an exact solution. It tests convergence order for every method, the stage-time arguments $t_n + c_i h$, and Rosenbrock's $f_t$ term.

- State: $u \in \mathbb{R}^1$, `dense`. Parameters $p = (k, \omega) = (2, 3)$. Interval $[0, 2]$, $u(0) = 1$.
- One operator, `decay`: $u' = -k u + \sin(\omega t)$. `depends_on_t` = true, `linear_in_u` = true.
- Exact solution, with $A(t) = \dfrac{k \sin \omega t - \omega \cos \omega t}{k^2 + \omega^2}$:
  $u(t) = \big(u(0) - A(0)\big) e^{-kt} + A(t)$.

### 1.2 `van_der_pol`

Nonlinear, so the second-order terms of Rosenbrock adjoints are nonzero.

- State: $u = (y, z)$, `dense`. Parameter $p = (\mu)$. $u(0) = (2, 0)$.
- One operator, `oscillator`: $y' = z$, $\ z' = \mu (1 - y^2) z - y$. `depends_on_t` = false, `linear_in_u` = false.
- Variants: **`nonstiff`**, $\mu = 1$ on $[0, 6.3]$; **`stiff`**, $\mu = 1000$ on $[0, 1]$.

### 1.3 `robertson`

Classic stiff kinetics with a conservation law.

- State: $u = (y_1, y_2, y_3)$, `dense`. Parameters $p = (k_1, k_2, k_3) = (0.04,\ 3\times10^{7},\ 10^{4})$. Interval $[0, 40]$, $u(0) = (1, 0, 0)$.
- One operator, `kinetics`:
  $y_1' = -k_1 y_1 + k_3 y_2 y_3$, $\ y_2' = k_1 y_1 - k_3 y_2 y_3 - k_2 y_2^2$, $\ y_3' = k_2 y_2^2$.
- Invariant: $y_1 + y_2 + y_3 = 1$.

### 1.4 `photostationary_cells`

A batch of independent cells with time-varying photolysis: the shape of reseact's chemistry. It tests `block_diagonal` layout and batched block LU, and gives each cell different stiffness over time.

- State: `block_diagonal`, ncells = 16, n = 3; species order $(\mathrm{NO_2}, \mathrm{NO}, \mathrm{O_3})$, species-major (operator-interface.md §2). Parameters $p = (j_0, k, \omega) = (10^{-2},\ 4.4\times10^{-4},\ 2\pi/3600)$. Interval $[0, 3600]$. Initial state in every cell: $(10, 5, 40)$.
- One operator, `photochemistry`. With $\varphi_c = 2\pi c/\text{ncells}$ and $j_c(t) = \tfrac12 j_0 \big(1 + \sin(\omega t + \varphi_c)\big)$, and $r_c = j_c\,\mathrm{NO_2} - k\,\mathrm{NO}\,\mathrm{O_3}$ in cell $c$:
  $\mathrm{NO_2}' = -r_c$, $\ \mathrm{NO}' = r_c$, $\ \mathrm{O_3}' = r_c$.
  `depends_on_t` = true, `linear_in_u` = false.
- Invariants per cell: $\mathrm{NO_2} + \mathrm{NO}$ and $\mathrm{NO_2} + \mathrm{O_3}$.

### 1.5 `advection_reaction`

Two operators that do not commute, for splitting order, `subcycle`, and adjoints through `sequence` and `strang`.

- Periodic 1-D grid, ncells = 64, $\Delta x = 1/64$. Species $(A, B)$, species-major. Parameters $p = (v, k_1, k_2) = (1, 50, 25)$. Interval $[0, 0.5]$. $A(0) = 1 + \tfrac12 \sin(2\pi x_c)$, $B(0) = 0$.
- Operator `transport` (`dense` structure): first-order upwind advection of $A$ only, $A_c' = -v\,(A_c - A_{c-1})/\Delta x$ with $A_{-1} \equiv A_{63}$. $B$ does not move. `linear_in_u` = true.
- Operator `reaction` (`block_diagonal`, n = 2): $A' = -k_1 A + k_2 B$, $\ B' = k_1 A - k_2 B$. `linear_in_u` = true.
- Invariant: $\sum_c (A_c + B_c)$.
- Because $B$ does not advect, the operators do not commute, so Lie splitting has a first-order and Strang a second-order splitting error.

### 1.6 `source_decay_map`

A leaf and a map in sequence, with an exact discrete solution. It tests `map`, `bind_dt` and the map adjoint.

- State: $u \in \mathbb{R}^1$, `dense`. Parameters $p = (a, r) = (1, 0.5)$. Interval $[0, 4]$, $u(0) = 0$.
- Operator `source`: $u' = a$.
- Map operator `decay`: $u \leftarrow u\, e^{-r \Delta}$, with $\Delta$ the length of the map node's interval (spec/run-config.md §4.4).
- With schedule `sequence[leaf source (ForwardEuler, dt "step"), map decay]` and macro step $\Delta$, the exact discrete solution is $u_{n+1} = (u_n + a\Delta)\, e^{-r\Delta}$.

## 2. Cases

[`../tests/conformance/m1/cases.json`](../tests/conformance/m1/cases.json) lists the cases. It is generated by [`make_cases.py`](../tests/conformance/m1/make_cases.py) beside it; CI checks the two agree. A case is a run config (spec/run-config.md) whose `model` is `toy:<problem>` or `toy:<problem>/<variant>`, plus what the run config does not carry: the interval (`tspan`, which the harness passes as the caller's interval, spec/run-config.md §1.1), output times, the functional for gradient cases, and the checks. The file validates against [`schemas/cases.schema.json`](schemas/cases.schema.json).

`model: "toy:..."` is legal only in conformance cases. Implementations accept it only from the test harness.

### 2.1 Checks

| Check | Meaning | Tier |
|---|---|---|
| `order` | Run the case at fixed `dt` for each value in `dts`. The observed order, the least-squares slope of $\log$ error against $\log$ dt for the end-state error in the max norm, is at least `expected − 0.3`. On `forced_decay` every method measures within 0.1 of its order at dts 0.1–0.0125, or above it, except ROS2 (1.71, pre-asymptotic), whose case uses dts 0.025–0.003125 (1.92). The implicit methods measure, in both implementations: BackwardEuler and IMEXEuler 0.985, TRBDF2 2.014, SDIRK2 1.913, Kvaerno3 2.966, Kvaerno4 4.588, Theta at θ = ½ 2.005. The error is against the exact solution where §1 gives one, otherwise against the `reference` result. | 2 |
| `oracle` | Run the same method in OrdinaryDiffEq with the same step sequence (`adaptive = false`, `tstops` from our tape; D31). End states agree to `rtol`. The check fails if OrdinaryDiffEq's solve does not return a successful code. OrdinaryDiffEq's SDIRK methods run with `NLNewton(κ = 1e-6)` instead of the default $\kappa = 1/100$: at the default, its Kvaerno4, which reuses $J$ across steps and takes each explicit first stage from the previous step's last one (spec/methods.md §5.1, §5.3), ends `robertson.Kvaerno4.adaptive`'s imposed step sequence with `ConvergenceFailure` ($y_3 = 0.166$ against 0.284); with $\kappa = 10^{-6}$ it agrees to $3\cdot10^{-8}$. Not reported upstream. Julia only; Rust compares with the Julia result through `cross_language`. | 2 |
| `invariant` | The problem's invariants hold at every output time to `rtol`. | 2 |
| `exact_discrete` | The end state equals §1's exact discrete solution to `rtol`. | 2 |
| `reference` | Output states agree with the `reference` result (§3) to `rtol`, relative to the largest magnitude of each state component over the run. | 2 |
| `dot_product` | spec/adjoint.md §7.1 for every step, with the frozen-tape tangent. Julia only (D44). | 2 |
| `fd_gradient` | spec/adjoint.md §7.2, central differences at the listed `epsilons`. | 2 |
| `reverse_diff` | Agreement with `SciMLSensitivity.ReverseDiffAdjoint` through OrdinaryDiffEq's same method, with the tape's step sequence imposed (spec/adjoint.md §7.3), to `rtol`; on cases too large for a ReverseDiff tape, and for Kvaerno4, through which ReverseDiffAdjoint returns a zero gradient, ForwardDiff through the same solve (§7.3). Julia only. | 2 |
| `replay` | Replaying from the start with the tape reproduces every accepted step and the end state bitwise (D10). | 2 |
| `cross_language` | Julia and Rust results (§3) agree: output states and gradients to `rtol`, and accepted step counts exactly. With `rtol_steps_differ` the step sequences may differ, in count or in any step's $t$ or $h$, since a roundoff-level accept/reject flip changes every later step, and the toy problems call each platform's own `sin`, `cos` and `pow`, which may differ in the last bit (spec/adjoint.md §7.5); an adaptive controller can turn such a bit into a different step size. When they do, states and gradients agree to `rtol_steps_differ` instead, which is set from the case's `reltol` because the two then differ by about the global error. States are compared relative to the largest magnitude of each state component over the run in either result, as for `reference`; the functional relative to its magnitude; each gradient vector (`u0`, `p`) relative to its largest-magnitude entry. When the step counts agree, the tapes must also agree by path and count (spec/tape.md §5): the same records in the same order, with the same accepted counts and return codes. Run in CI by [`compare_results.py`](../tests/conformance/m1/compare_results.py) on the two implementations' result files. | 3 |

A check fails loudly with the measured quantity and the threshold.

## 3. Result files

For `cross_language` and for references, an implementation writes one result file per case, validated by [`schemas/result.schema.json`](schemas/result.schema.json):

The values in this example are illustrative:

```json
{
  "case": "robertson.Rodas4.adaptive",
  "producer": "julia",
  "t": [0, 10, 40],
  "u": [[1, 0, 0], [0.84136, 1.6e-05, 0.15862], [0.71583, 9.2e-06, 0.28416]],
  "stats": {"accepted": 63, "rejected": 2, "rhs_calls": 410},
  "functional": 0.28416,
  "gradient": {"u0": [0.1, 0.2, 0.3], "p": [-1.2, 3.4e-08, -5.6e-05]}
}
```

- `producer` is `julia`, `rust`, or `reference` (a high-accuracy OrdinaryDiffEq solve, for cases without an exact solution).
- `t` is $t_0$ followed by the case's output times (`saveat`, default $t_f$ alone; spec/run-config.md §7.3), and `u` holds the full state at each of them, in the layout of §1. Numbers are written in shortest round-trip form.
- `functional` and `gradient` are present for gradient cases only.
- `stats.accepted` and `stats.rejected` sum over all leaves and macro steps.

Result files under `tests/conformance/m1/reference/` are checked in. They are produced by a Julia script with OrdinaryDiffEq (`Rodas5P` for stiff and `Vern9` for non-stiff problems, `reltol = abstol = 1e-13`); for `advection_reaction` the reference integrates the unsplit sum of both operators, so `order` checks measure the splitting error. The script and the files arrive with the Julia M1 implementation. Julia and Rust results are produced by CI and compared there, not checked in.
