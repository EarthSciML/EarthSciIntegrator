# Discrete adjoints: the math per family

**Status:** M0 draft, 2026-10-23. Normative for both implementations (D1, D8) unless a paragraph is marked *informative* or *open question*.

This chapter defines what the reverse pass computes. It covers:
- the functional;
- the tape;
- the model primitives;
- the forward step and its discrete adjoint for every v1 method family;
- the adjoint of every schedule combinator;
- how gradients accumulate;
- the identities the conformance suite checks.

Orchestration (checkpoints, replay, the revolve schedule) is in architecture §10 and D34. This chapter assumes a replayed step's stage values are available.

**The contract** (D4, decisions §1.4). Adjoints are method-level and discrete. Each family's adjoint is the exact transpose of the linearization of the step the forward pass actually computed, with the tape held fixed (§2). Every nonlinear and algebraic solve is differentiated by the implicit function theorem at its converged solution, never by differentiating its iterations. So:
- explicit and Rosenbrock adjoints are exact to roundoff;
- SDIRK, IMEX, θ-method and `solve`-node adjoints are exact up to the nonlinear-solve tolerance.

Continuous adjoints are not supported.

**Arithmetic form.** "The step the forward computes" means the arithmetic form the spec defines for each method (§4), not an algebraically equivalent rewrite. Where OrdinaryDiffEq implements a method in a particular form (Rodas's transformed form, Shu–Osher form for SSP methods, the ode23s form for `Rosenbrock23`), the spec adopts that form, so the tier-2 oracle (D31) and the forward pass agree in structure. An implementation MUST NOT run a method's forward step in one form and its adjoint in another.

## 1. Notation

| Symbol | Meaning |
|---|---|
| $u \in \mathbb{R}^N$ | The leaf's state: the degrees of freedom (DOFs) its operators write (architecture §3.2 pass 3). Other state is read-only within the leaf. |
| $p$ | Active parameters: scalar parameters and array-valued (gridded) parameters (D5). |
| $F$ | Forcing buffers resident during a step: one bracket of provider records per field (architecture §7.5). |
| $f(u, p, t; F)$ | The leaf's right-hand side: the sum of its operators' derived RHS models. |
| $t_n,\ h_n$ | Start time and size of accepted step $n$, read from the tape. $h$ is short for $h_n$ when the step is clear. |
| $J(u) = \partial f / \partial u$ | The Jacobian at $(u, p, t; F)$. The time and forcing arguments are left implicit when they are the step's own. |
| $f_t = \partial f / \partial t$ | The time gradient. It is nonzero inside forcing brackets, because models blend records by time (architecture §10). |
| $f_p,\ f_F$ | $\partial f/\partial p$ and $\partial f/\partial F$. |
| $\bar x$ | The cotangent of $x$, i.e. $\partial \mathcal{J} / \partial x$ for the functional $\mathcal{J}$, a column vector shaped like $x$. |
| $\lambda_n$ | $\bar u_n$, the state cotangent at the start of step $n$. $\lambda_{n+1}$ is the cotangent of the step's output. |
| $\langle a, b\rangle$ | The Euclidean inner product $a^\top b$ over DOFs. It is unweighted; error weights never enter the adjoint. |
| $\mathrel{+}=$ | Accumulation into a cotangent. Order of accumulation is fixed (§6.4). |

A **step map** is $\Phi_n : (u_n, p, F) \mapsto u_{n+1}$ for fixed $(t_n, h_n)$ and fixed discrete decisions. Its **adjoint step** maps $\lambda_{n+1}$ to $(\lambda_n, \bar p \mathrel{+}= \cdot, \bar F \mathrel{+}= \cdot)$:

```math
\lambda_n = \Big(\frac{\partial \Phi_n}{\partial u_n}\Big)^{\!\top} \lambda_{n+1}, \qquad
\bar p \mathrel{+}= \Big(\frac{\partial \Phi_n}{\partial p}\Big)^{\!\top} \lambda_{n+1}, \qquad
\bar F \mathrel{+}= \Big(\frac{\partial \Phi_n}{\partial F}\Big)^{\!\top} \lambda_{n+1}.
```

Every section below derives one such map. Derivations linearize the forward equations ($d$ denotes a tangent) and then transpose.

## 2. Setting

### 2.1 The functional

The run config's `adjoint.functional` (architecture §4.2, §10) defines

```math
\mathcal{J} = \sum_{k} \varphi_k\big(u(\tau_k), p\big) + \int_{t_0}^{t_f} \psi(u, p, t)\, dt .
```

- **Point terms.** Each $\varphi_k$ is a named observed or reduction evaluated at an observation time $\tau_k$. Every $\tau_k$ is a macro-step boundary: the planner always makes observation times tstops, whether or not `step.hit` names their source (spec/run-config.md §5). Under a fixed `step.dt`, an observation time that is not a multiple of it is refused like any other such tstop (spec/run-config.md §8 rule 9). $u(\tau_k)$ is the full model state after the whole schedule for the macro step ending at $\tau_k$. In the reverse pass, on reaching $\tau_k$:

  ```math
  \lambda \mathrel{+}= \nabla_u \varphi_k\big(u(\tau_k), p\big), \qquad \bar p \mathrel{+}= \nabla_p \varphi_k .
  ```

- **The integral term is discretized as an integrated auxiliary state** (D26). The planner adds a state $q$ with $q' = \psi(u, p, t)$ and $q(t_0) = 0$, and sets $\int \psi \approx q(t_f)$. The integral is therefore the quadrature the forward method induces, and its adjoint is that method's adjoint applied to the augmented state $(u, q)$. Because nothing reads $q$, $\bar q \equiv 1$ throughout the reverse pass. Each stage's contribution reduces to the stage cotangent of $q$ times $\nabla_u \psi$ and $\nabla_p \psi$ at that stage.

- **Which leaf integrates $q$** (D39). Under splitting, $\psi$ reads state that several leaves write, and the leaf that integrates $q$ decides the quadrature. The planner attaches $q$ to the last leaf of the macro-step schedule, so $\psi$ sees the state after every process. `adjoint.functional.integrate_in` names a different leaf by node id.

### 2.2 The tape and what is held fixed

The forward pass records the tape (architecture §10). In the reverse pass every tape entry is a **constant**:
- the accepted step sequence: $(t_n, h_n)$ per leaf per shard, or per-cell step counts and a hash under `dt_scope: cell`;
- shard compositions;
- converged `solve` unknowns $z$;
- every discrete decision: map and clip masks, `ifelse` and `max`/`min` branches taken, event and `every` firings, `recover` actions;
- forcing epochs, and the buffer-slot-to-provider-record map.

Rejected steps leave no trace in the state and contribute nothing to the adjoint. Error estimators, embedded solutions and dense output feed only the controller, so they are not differentiated.

The reverse pass therefore computes the gradient of the **frozen-tape map**

```math
\Phi_{\text{tape}} : (u_0, p, F) \mapsto \mathcal{J},
```

the composition of step maps with every tape entry fixed. The derivative of the step size with respect to the state, through the controller, is ignored (decisions §1.1). This is the standard choice, and it is what reseact's prototype gradient computed.

### 2.3 Reconciling with forward sensitivities that carry dt tangents (D36)

Julia's forward sensitivities push `Dual` numbers through the state, stages, error norm and controller. The controller therefore produces tangents of $h_n$. That differentiates a different map, the **adaptive map**, in which $h_n$ depends on the state. Its derivative differs from $\Phi_{\text{tape}}$'s by the terms $(\partial \Phi_n / \partial h_n)\, dh_n$. Where an accept/reject decision flips, it is not differentiable at all.

**Normative.** The forward-sensitivity implementation has two modes:
- **`dt_tangents = false` (frozen tape).**
  - The controller acts on the primal values only. Julia strips the partials before the error norm, which D13 keeps in f64 in any case.
  - $h_n$, $t_n$ and every discrete decision carry zero tangent.
  - This tangent is the exact linearization of $\Phi_{\text{tape}}$. **The M4 dot-product gate (adjoint against forward-sensitivity tangent) uses this mode.**
- **`dt_tangents = true` (adaptive map).** This mode keeps reseact's forward-sensitivity lane. It is validated against finite differences of the adaptive map. It is never compared with the adjoint by a dot product.

Both modes are decided (D40). A `Dual` whose value type is f64 satisfies D13's "error norms are always f64" (D41), so with `dt_tangents = true` the partials flow through the norm into the controller; with `dt_tangents = false` they are stripped before it.

### 2.4 Steps never straddle discontinuities

Steps never cross a forcing-epoch boundary (architecture §7.5), so $F$ is constant over each step. A stage at exactly $t_{k+1}$ reads epoch $k+1$, consistently in forward and reverse. Maps and event affects happen only between steps. Each step map is therefore smooth in $(u, p, F)$, apart from `ifelse`/`max`/`min` branches inside $f$. Those are differentiated branch-wise with the branch frozen (EarthSciASTDiff spec §3.2).

## 3. Primitives

All primitives evaluate at a given $(u, p, t; F)$. Each returns a vector shaped like its output's cotangent. None forms a matrix unless the leaf's `linsolve` stores one. The sources are EarthSciASTDiff band models (U7, U8, U9), and in Julia also DifferentiationInterface fallbacks (D38).

| Primitive | Definition | Source |
|---|---|---|
| `jvp_u(u; v)` | $J(u)\,v$ | first-order bands; Julia fallback `AutoForwardDiff` pushforward |
| `vjp_u(u; v)` | $J(u)^\top v$ | transposed bands; dedicated VJP model for recurrences and scans (U9); Julia fallback `AutoEnzyme` pullback |
| `vjp_p(u; v)` | $f_p(u)^\top v$, including array parameters | parameter bands (U7) |
| `vjp_F(u; v)` | $f_F(u)^\top v$, over buffer cells including the record axis | forcing bands (U7) |
| `solve_W(v)`, `solve_Wt(v)` | $W^{-1} v$, $W^{-\top} v$ for the leaf's $W$ (§4) | the leaf's `linsolve`. Rosenbrock reuses the forward factorization, whose $W$ is formed at $u_n$. SDIRK, IMEX Euler and the θ-method solve the exact system at the converged stage, refactoring or iterating with the forward factorization as a preconditioner (§4.3, D42; architecture §6.1) |
| `hess_u(u; {μ_i, w_i})` | $\nabla_u \sum_i \mu_i^\top J(u)\, w_i$ | second-order bands `[states, states]`, one contraction per call (U8) |
| `hess_p`, `hess_F` | $\nabla_p$ and $\nabla_F$ of the same sum | second-order bands `[states, parameters]`, `[states, forcing]` (U8) |
| `ft(u)` | $f_t(u)$ | time bands |
| `ft_vjp_u(u; ν)`, `ft_vjp_p`, `ft_vjp_F` | $\nabla_u (\nu^\top f_t(u))$, and likewise for $p$ and $F$ | second-order bands `[time, ·]` (U8) |
| `map_vjp(u; v)` | $(\partial M/\partial u)^\top v$ and $(\partial M/\partial p)^\top v$ for a step map $M$ | map band models (U6, U7) |
| `res_vjp(u, z; μ)` | $g_u^\top \mu$, $g_z^\top \mu$, $g_p^\top \mu$ for an algebraic residual $g$ | residual band models (U6) |

- `hess_*` and `ft_vjp_*` are needed only by the Rosenbrock family (§4.2).
- Explicit, SDIRK, IMEX and θ-method adjoints need only first-order primitives and transposed solves.
- Where a leaf's second-order bands are unavailable, the planner refuses a Rosenbrock adjoint on that leaf and offers SDIRK in adjoint mode (U8 decision 6).

## 4. Method families

Each subsection gives the forward step in its normative form, its linearization, and the adjoint step. Coefficients are never re-derived here. They are taken verbatim, with citation, from OrdinaryDiffEq's tableau packages (D31):
- `OrdinaryDiffEqSSPRK` and `OrdinaryDiffEqLowOrderRK`/`OrdinaryDiffEqTsit5` for explicit methods (*assumption:* the last two are the packages that hold Tsit5, BS3, RK4 and Heun; confirm when the method catalogue is written);
- `OrdinaryDiffEqRosenbrockTableaus` and `OrdinaryDiffEqRosenbrock`;
- `OrdinaryDiffEqSDIRK`.

### 4.1 Explicit Runge–Kutta

**Butcher form** (Tsit5, BS3, RK4, Heun, Euler, and `WRF_RK3` as a plain explicit method). An $s$-stage tableau $(a_{ij}, b_i, c_i)$ with $a_{ij} = 0$ for $j \ge i$:

```math
U_i = u_n + h \sum_{j<i} a_{ij} K_j, \qquad K_i = f(U_i, p, t_n + c_i h; F), \qquad
u_{n+1} = u_n + h \sum_i b_i K_i .
```

`WRF_RK3` (Wicker–Skamarock) is the instance $a_{21} = 1/3$, $a_{32} = 1/2$, $b = (0, 0, 1)$, $c = (0, 1/3, 1/2)$. Its split-explicit use is §5.9.

*Linearization.* $dU_i = du_n + h\sum_{j<i} a_{ij}\, dK_j$, $\ dK_i = J(U_i)\, dU_i + f_p(U_i)\, dp + f_F(U_i)\, dF$, and $du_{n+1} = du_n + h \sum_i b_i\, dK_i$.

*Adjoint.* Let $\kappa_i$ be the cotangent of $K_i$ and $\sigma_i$ that of $U_i$. Sweep $i = s, \dots, 1$:

```math
\kappa_i = h\Big(b_i\, \lambda_{n+1} + \sum_{j>i} a_{ji}\, \sigma_j\Big), \qquad
\sigma_i = J(U_i)^\top \kappa_i ,
```

then

```math
\lambda_n = \lambda_{n+1} + \sum_i \sigma_i, \qquad
\bar p \mathrel{+}= \sum_i f_p(U_i)^\top \kappa_i, \qquad
\bar F \mathrel{+}= \sum_i f_F(U_i)^\top \kappa_i .
```

That is one `vjp_u` per stage with $\kappa_i \ne 0$, plus one `vjp_p` and one `vjp_F` per stage when those gradients are requested. A stage with $\kappa_i = 0$ is skipped. This happens when $b_i = 0$ and no later stage reads $K_i$, which is Tsit5's seventh stage, used only by the error estimate.

**Shu–Osher form** (SSPRK33, SSPRK43, SSPRK104, as OrdinaryDiffEqSSPRK implements them). With $u^{(0)} = u_n$:

```math
u^{(i)} = \sum_{j<i} \big(\alpha_{ij}\, u^{(j)} + h\,\beta_{ij}\, f(u^{(j)}, p, t_n + c_j h; F)\big), \quad i = 1..s, \qquad u_{n+1} = u^{(s)} .
```

*Assumption:* the exact $(\alpha, \beta)$ arrays, including SSPRK43's embedded pair, are taken from `OrdinaryDiffEqSSPRK`'s `perform_step!` bodies, which state them inline rather than in tableau structs.

*Adjoint.* Seed $\bar u^{(s)} = \lambda_{n+1}$ and $\bar u^{(j)} = 0$ for $j < s$. For $i = s, \dots, 1$, and for each $j < i$ with $(\alpha_{ij}, \beta_{ij}) \ne (0, 0)$:

```math
\bar u^{(j)} \mathrel{+}= \alpha_{ij}\, \bar u^{(i)} + h\beta_{ij}\, J(u^{(j)})^\top \bar u^{(i)}, \qquad
\bar p \mathrel{+}= h\beta_{ij}\, f_p(u^{(j)})^\top \bar u^{(i)} , \qquad
\bar F \mathrel{+}= h\beta_{ij}\, f_F(u^{(j)})^\top \bar u^{(i)} ,
```

and $\lambda_n = \bar u^{(0)}$. When several $i$ share a $j$, an implementation SHOULD first form $v_j = \sum_i \beta_{ij} \bar u^{(i)}$ and then call `vjp_u` once on $v_j$. That calls `vjp_u` once per stage, as in the Butcher form. The summation order over $i$ is ascending (§6.4).

**First same as last.** For methods where the first stage of step $n+1$ reuses $f(u_{n+1})$ from step $n$ (Tsit5, BS3):
- In the adjoint, the reused value is treated as a fresh evaluation at $u_{n+1}$ belonging to step $n+1$. Its cotangent flows through $J(u_{n+1})^\top$ into $\lambda_{n+1}$ during step $n+1$'s adjoint, which is the correct chain rule.
- Step $n$'s last stage, when it serves only the next step and the error estimate, has $\kappa_s = 0$.
- Forward rule: a leaf MUST invalidate its stored stage at entry to each of its invocations within the schedule, because maps and other leaves may have changed $u$ in between.

### 4.2 Rosenbrock

v1 leaves form $J$ and $f_t$ at $(u_n, p, t_n; F)$ every step (U8). The mass matrix is the identity; the index-1 DAE case is §5.10.

**Generic form** (Rodas3, Rodas4, Rodas5P, ROS2, ROS3P, and KPP's Ros/Rodas for parity). This is OrdinaryDiffEq's transformed form with tableau $(A, C, \gamma, c, d, b)$ from `OrdinaryDiffEqRosenbrockTableaus` (`RodasTableau`):

```math
\hat W = \tfrac{1}{h\gamma} I - J(u_n), \qquad U_1 = u_n, \qquad U_i = u_n + \sum_{j<i} A_{ij}\, k_j ,
```
```math
\hat W k_i = r_i, \qquad r_i = f(U_i, p, t_n + c_i h; F) + h\, d_i\, f_t(u_n) + \sum_{j<i} \tfrac{C_{ij}}{h}\, k_j, \qquad
u_{n+1} = u_n + \sum_i b_i k_i .
```

This is the arithmetic of `OrdinaryDiffEqRosenbrock`'s generic `perform_step!`. That code stores $W = J - I/(h\gamma)$ and negates the solve, which is $\hat W$ with the sign moved. The optimization that skips an $f$ evaluation when two stages share $(A_{i\cdot}, c_i)$ does not change the math.

*Linearization.* Since $d\hat W = -dJ$, differentiating $\hat W k_i = r_i$ gives

```math
\hat W\, dk_i = dr_i + dJ(u_n)[du_n]\; k_i ,
```

where $dJ(u_n)[du_n] = \partial_u (J(u) k_i)\,du_n$, and

```math
dr_i = J(U_i)\, dU_i + f_p(U_i)\,dp + f_F(U_i)\,dF + h d_i\, df_t + \sum_{j<i} \tfrac{C_{ij}}{h}\, dk_j .
```

*Adjoint.* Seed $\bar k_i = b_i \lambda_{n+1}$. Sweep $i = s, \dots, 1$. When stage $i$ is reached, every later stage has already added into $\bar k_i$, so:

1. $\mu_i = \hat W^{-\top} \bar k_i$ (`solve_Wt`). $\mu_i$ is the cotangent of $r_i$.
2. $\sigma_i = J(U_i)^\top \mu_i$ (`vjp_u` at the stage point).
3. For each $j < i$: $\bar k_j \mathrel{+}= A_{ij}\, \sigma_i + \tfrac{C_{ij}}{h}\, \mu_i$.

Then

```math
\lambda_n = \lambda_{n+1} + \sum_i \sigma_i
\;+\; \underbrace{\nabla_u \sum_i \mu_i^\top J(u)\, k_i \,\Big|_{u_n}}_{\texttt{hess\_u}}
\;+\; \underbrace{\nabla_u \big(\nu^\top f_t(u)\big)\Big|_{u_n}}_{\texttt{ft\_vjp\_u}},
\qquad \nu = h \sum_i d_i\, \mu_i ,
```
```math
\bar p \mathrel{+}= \sum_i f_p(U_i)^\top \mu_i + \nabla_p \sum_i \mu_i^\top J k_i + \nabla_p(\nu^\top f_t), \qquad
\bar F \mathrel{+}= \text{the same with } \nabla_F .
```

The second-order term is a single `hess_u` call per step with the pairs $(\mu_i, w_i = k_i)$ (U8's contraction kernel). The $f_t$ term exists only for methods with some $d_i \ne 0$, and only where $f_t \ne 0$, i.e. inside forcing brackets or for explicitly time-dependent models.

**`Rosenbrock23`** (ode23s; Shampine–Reichelt). `OrdinaryDiffEqRosenbrock` gives it its own `perform_step!`, and the spec adopts that form. With $d = 1/(2+\sqrt 2)$ from `Rosenbrock23Tableau`:

```math
\hat W = I - h d\, J(u_n), \qquad
\hat W k_1 = f(u_n) + h d\, f_t(u_n), \qquad
U_2 = u_n + \tfrac{h}{2} k_1,
```
```math
\hat W (k_2 - k_1) = f(U_2, t_n + \tfrac h2) - k_1, \qquad
u_{n+1} = u_n + h\, k_2 .
```

The third stage and $c_{32}$ feed only the error estimate and are not differentiated.

*Adjoint.* Here $d\hat W = -hd\, dJ$.

```math
\bar k_2 = h \lambda_{n+1}, \qquad \mu_2 = \hat W^{-\top} \bar k_2, \qquad \sigma_2 = J(U_2)^\top \mu_2,
```
```math
\bar k_1 = \bar k_2 - \mu_2 + \tfrac h2\, \sigma_2, \qquad \mu_1 = \hat W^{-\top} \bar k_1, \qquad \sigma_1 = J(u_n)^\top \mu_1,
```
```math
\lambda_n = \lambda_{n+1} + \sigma_1 + \sigma_2
+ \nabla_u\big[h d\,\big(\mu_1^\top J(u)\,k_1 + \mu_2^\top J(u)\,(k_2 - k_1)\big)\big]_{u_n}
+ \nabla_u\big[h d\, \mu_1^\top f_t(u)\big]_{u_n},
```

with $\bar p$ and $\bar F$ as in the generic form: $f_p(u_n)^\top\mu_1 + f_p(U_2)^\top\mu_2$ plus the mixed second-order and $f_t$ terms. So `hess_u` is called once with the pairs $(\mu_1, hd\,k_1)$ and $(\mu_2, hd\,(k_2-k_1))$.

*Informative.* U08 writes the step in the untransformed Hairer–Wanner form, where the second-order vectors are $w_i = h(\gamma k_i + s_i)$. The primitive is the same contraction; only the vectors differ with the form.

**W-methods and inexact $W$.** The derivation above assumes the $J$ in $\hat W$ is $J(u_n)$, evaluated exactly. Two cases differ:
- **Low-precision storage** (architecture §8). $W$ or its factors are stored in f16, bf16 or f32 and solved with iterative refinement to a tolerance. The adjoint solves $\hat W^\top$ with the same stored factors and the same refinement. The gradient is then consistent to the refinement tolerance, not to roundoff (§7.1).
- **A lagged Jacobian** $J(u_m)$, $m < n$, reused across steps. The second-order term belongs to step $m$'s adjoint, and the tape must record where each $J$ was formed. v1 does not lag Jacobians in Rosenbrock leaves, so this is reserved.

Classical Rosenbrock methods (Rodas) lose order with an inexact $J$. The planner refuses a low-precision $W$ with them unless overridden (architecture §8).

### 4.3 SDIRK and ESDIRK

BackwardEuler, TRBDF2, SDIRK2, Kvaerno3/4, with tableau $(a_{ij}, b_i, c_i)$ from `OrdinaryDiffEqSDIRK`. Here $a_{ij} = 0$ for $j > i$ and $a_{ii} = \gamma$ for every implicit stage. For ESDIRK, $a_{11} = 0$, so $Z_1 = u_n$ explicitly.

**Forward.** For each implicit stage $i$, Newton (or Newton–Krylov) converges $Z_i$ to

```math
G_i(Z_i) \equiv Z_i - h\gamma\, f(Z_i, p, t_n + c_i h; F) - \Big(u_n + h \sum_{j<i} a_{ij} K_j\Big) = 0, \qquad K_i = f(Z_i),
```
```math
u_{n+1} = u_n + h\sum_i b_i K_i \quad\text{(or } u_{n+1} = Z_s \text{ when the method is stiffly accurate, } b_i = a_{si}\text{)}.
```

OrdinaryDiffEq iterates on $z_i = h K_i$ rather than on $Z_i$. At the exact solution the two are the same map, so the derivative below holds for either. An implementation MAY use either unknown.

**Implicit function theorem.** The adjoint differentiates the map defined by the *exact* stage solutions at the computed $Z_i$. Newton iterates, the Newton matrix (which may hold a stale or low-precision $J$), line searches and convergence tests are not differentiated. With $W_i = I - h\gamma J(Z_i)$:

```math
W_i\, dZ_i = du_n + h\sum_{j<i} a_{ij}\, dK_j + h\gamma\,\big(f_p(Z_i)dp + f_F(Z_i)dF\big), \qquad dK_i = J(Z_i)dZ_i + f_p(Z_i)dp + f_F(Z_i)dF .
```

**Adjoint.** Let $\omega_i$ be the cotangent of $K_i$. Sweep $i = s, \dots, 1$:

```math
W_i^\top\, \omega_i = h\Big(b_i\, \lambda_{n+1} + \sum_{j>i} a_{ji}\, \sigma_j\Big), \qquad \sigma_i = J(Z_i)^\top \omega_i ,
```

then

```math
\lambda_n = \lambda_{n+1} + \sum_i \sigma_i, \qquad
\bar p \mathrel{+}= \sum_i f_p(Z_i)^\top \omega_i, \qquad \bar F \mathrel{+}= \sum_i f_F(Z_i)^\top \omega_i .
```

*Derivation.* Write $J_i = J(Z_i)$ and let $\rho_i$ be the cotangent of the right-hand side of the linearized stage equation, so $dZ_i = W_i^{-1}(\cdot)$. Let $\bar K_i = h(b_i\lambda_{n+1} + \sum_{j>i} a_{ji}\rho_j)$ be the cotangent that reaches $K_i$ from the output and from later stages. Then $\rho_i = W_i^{-\top} J_i^\top \bar K_i$. $W_i$ is a polynomial in $J_i$, so the two commute, and $\rho_i = J_i^\top \omega_i$ with $W_i^\top \omega_i = \bar K_i$. That gives $\sigma_i = \rho_i$, the system above, and $\lambda_n = \lambda_{n+1} + \sum_i \rho_i$, because $du_n$ enters every stage's right-hand side and the output. For the parameters, $dK_i$ receives $f_p\,dp$ directly and $J_i W_i^{-1} h\gamma f_p\,dp$ through $Z_i$. So the cotangent is $(\bar K_i + h\gamma J_i^\top\omega_i)^\top f_p = (W_i^\top\omega_i + h\gamma J_i^\top\omega_i)^\top f_p = \omega_i^\top f_p$. $\omega_i$ is the total cotangent of $K_i$, its own implicit dependence included.

For an explicit first stage ($a_{11} = 0$), $W_1 = I$ and the solve is a copy. The stiffly accurate update $u_{n+1} = Z_s$ is the same function of $(u_n, p, F)$ as $u_n + h\sum_i a_{si}K_i$ when stages are exact, so the same adjoint applies with $b_i = a_{si}$.

**Which $W^\top$.** The system matrix is $I - h\gamma J(Z_i)^\top$ at the *converged* stage. It differs from the forward Newton matrix whenever that matrix's $J$ was formed elsewhere (at $u_n$, or reused). The adjoint MUST solve the exact system, to a tolerance at least as tight as the forward Newton tolerance. Two routes are allowed:
- **Refactor.** Form $J(Z_i)$ from bands and factor $W_i$. This is the default for `block_lu`, `sparse_block_lu` and `banded`, where a factorization is cheap per cell or per column.
- **Iterate.** Use the forward factorization as a preconditioner for a Krylov solve or iterative refinement on $W_i^\top$, with `vjp_u` supplying $J(Z_i)^\top v$. This is the default for `krylov` leaves.

These defaults are decided (D42). Reusing the forward factorization is exact for Rosenbrock only; for SDIRK it is at most a preconditioner (architecture §6.1). The same holds for the $W$ of IMEX Euler (§4.4) and the θ-method (§4.5), which is formed at the converged $u_{n+1}$. *Informative:* the cost of refactoring per stage at 3-D column scale has not been measured.

### 4.4 IMEX Euler

PISM's "explicit horizontal at old time" (architecture §5). The leaf's operators split into $f = f_E + f_I$ by the run config's `imex`:

```math
u_{n+1} = u_n + h\, f_E(u_n) + h\, f_I(u_{n+1}),
```

with the implicit equation converged by Newton, or by one linear solve when $f_I$ is linear in $u$, as eqice's column diffusion is. With $W = I - hJ_I(u_{n+1})$:

```math
W^\top \omega = \lambda_{n+1}, \qquad
\lambda_n = \omega + h\, J_E(u_n)^\top \omega, \qquad
\bar p \mathrel{+}= h f_{E,p}(u_n)^\top \omega + h f_{I,p}(u_{n+1})^\top \omega ,
```

and $\bar F$ likewise.

The implicit part is solved to convergence by default (D43). The leaf option `implicit_solve: "linearized"` instead takes a single linearized (Rosenbrock–Euler) step: $W$ is formed at $u_n$ and the step is a one-stage Rosenbrock method, whose adjoint is §4.2 with one stage, including `hess_u`. When $f_I$ is linear in $u$ the two coincide, and the planner records `converged`.

### 4.5 Legacy fixed steps: θ-method, forward Euler, backward Euler

```math
u_{n+1} = u_n + h\big[(1-\theta) f(u_n) + \theta f(u_{n+1})\big].
```

Forward Euler is $\theta = 0$ and backward Euler is $\theta = 1$. For $\theta > 0$, with $W = I - h\theta J(u_{n+1})$:

```math
W^\top \omega = \lambda_{n+1}, \qquad
\lambda_n = \omega + h(1-\theta) J(u_n)^\top \omega, \qquad
\bar p \mathrel{+}= h(1-\theta) f_p(u_n)^\top\omega + h\theta f_p(u_{n+1})^\top \omega .
```

For $\theta = 0$, $\omega = \lambda_{n+1}$ and no solve is needed.

Legacy WRF and PISM increments are forward Euler with `dt: "step"` and a bound dt (§5.8). Their adjoint is one `vjp_u` and one `vjp_p` of the model's own $(x_{\text{out}} - x_{\text{in}})/\Delta t$ tendency.

### 4.6 `solve` nodes: the implicit function theorem (D3a)

A `solve` node finds algebraic unknowns $z$ with
- the slow state $u$ and the parameters frozen;
- warm start, `recover` ladder and convergence criterion as configured (architecture §6.3).

$z$ is stored on the tape. Downstream nodes read $z$. In the reverse pass a cotangent $\bar z$ arrives from them, and:

```math
g_z(u, z, p)^\top\, \mu = \bar z, \qquad
\bar u \mathrel{-}= g_u^\top \mu, \qquad \bar p \mathrel{-}= g_p^\top \mu, \qquad \bar F \mathrel{-}= g_F^\top \mu ,
```

from $dz = -g_z^{-1}(g_u\,du + g_p\,dp + g_F\,dF)$. That is one transposed solve per node invocation (architecture §6.3), with `res_vjp` supplying the products. Julia MAY implement it with `SciMLSensitivity.SteadyStateAdjoint` (D32).

Three consequences of differentiating at the converged solution:
- **The warm start is not differentiated.** The converged $z$ does not depend on it, to tolerance.
- **Picard needs the unfrozen Jacobian.** A Picard iteration (D25) uses a frozen-observed Jacobian as its *iteration operator*. Its converged $z$ still solves the full residual $g(u, z, p) = 0$. So the adjoint's $g_z$ is the **unfrozen** Jacobian, including the dependence of the frozen observed on $z$ (PISM's $\nu H$ on the velocity). The planner MUST therefore emit both band models for a `solve` node with `picard.freeze`, the only place freezing is written (D51): the frozen one (`frozen_jacobian`) for the forward pass and the full one (`jacobian`) for the adjoint (spec/plan.md §1, architecture §3.4).
- **`recover` actions are tape decisions.** An under-relaxation or a revert changes only the path, not the converged $z$, and has no effect on the adjoint. A `scale_param` action changes the residual itself, $g(u, z, s\cdot p_k)$ with the recorded scale $s$. The adjoint uses the scaled residual, and the chain rule multiplies $\bar p_k$ by $s$.

## 5. Combinators

The adjoint of a composition is the composition of the adjoints in reverse order (decisions §1.1). Every combinator below states its forward map on $[t, t+\Delta]$ (architecture §4.3) and its reverse.

### 5.1 `leaf` with adaptive substeps

A leaf over $[t, t+\Delta]$ is the composition of its accepted steps $\Phi_{N-1} \circ \dots \circ \Phi_0$, read from the tape. Its adjoint applies the family's adjoint step for $n = N-1, \dots, 0$ to stage values from the replay (architecture §10). Under `dt_scope: shard` or `cell`, each shard or cell has its own step sequence. Pointwise leaves have no coupling between cells, so the per-cell reverse sweeps are independent and may run in any order. Their contributions to $\bar p$ are reduced in the fixed order of §6.4.

### 5.2 `sequence`

$\Phi = \Phi_m \circ \dots \circ \Phi_1$, so the reverse runs $\Phi_1^*, \dots$ in the order $m, \dots, 1$.

### 5.3 `strang`

The forward is $a(\Delta/2)\, b(\Delta)\, a(\Delta/2)$, generalized symmetrically for $n$ children. The reverse runs the second $a$ half, then $b$, then the first $a$ half, each from its own tape segment. The two $a$ halves are distinct invocations, with separate step sequences and stage values.

### 5.4 `subcycle`

The child runs $n$ times with $\Delta/n$, or adaptively under its own `limits` with the substep sequence recorded on the tape. The reverse runs the child's adjoint once per recorded substep, last first.

### 5.5 `every`

The tape records the firings. A non-firing invocation is the identity and has an identity adjoint. A firing invocation is the child's map, so its reverse is the child's adjoint at that point.

**`hold: "tendency"`.** At a firing at $t_c$, the held operator's tendency $\tau = f_H(u(t_c), p, t_c; F)$ is evaluated and stored, with $F$ the forcing buffers of the epoch containing $t_c$. Until the next firing it is added as a constant to the right-hand side of every sibling leaf: $f_{\text{sib}}(u) + \tau$. Treat $\tau$ as a parameter of those leaves with $\partial(f_{\text{sib}} + \tau)/\partial \tau = I$. In the reverse pass:
- Every stage of a sibling leaf that adds $\tau$ contributes its stage cotangent to $\bar\tau$:
  - $\kappa_i$ for explicit methods;
  - $\omega_i$ for SDIRK;
  - $\mu_i$ for Rosenbrock;
  - $\omega$ weighted by $h$ for the θ-method and IMEX Euler, matching where $f$ enters.

  In each case this is the term that would have gone to `vjp_p` for a parameter entering additively.
- $\tau$ has zero time gradient and does not enter $J$, so Rosenbrock's second-order and $f_t$ terms are unaffected.
- On reaching $t_c$, with every derivative at $(u(t_c), p, t_c; F)$:

  ```math
  \bar u(t_c) \mathrel{+}= J_H^\top \bar\tau, \qquad
  \bar p \mathrel{+}= f_{H,p}^\top \bar\tau, \qquad
  \bar F \mathrel{+}= f_{H,F}^\top \bar\tau ,
  ```

  then $\bar\tau \leftarrow 0$. $\bar F$ goes to the slots of $t_c$'s epoch, and from them to provider records as in §6.3.

### 5.6 `frozen`

The operator's tendency is evaluated once at the start of the enclosing step, at $u_n$, and held across all stages (WRF `rk_addtend_dry`). This is §5.5 with the hold interval equal to one step of the enclosing node: $\bar\tau$ collects the stage cotangents of that step, and then, with every derivative at $(u_n, p, t_n; F)$,

```math
\lambda_n \mathrel{+}= J_{\text{fr}}^\top\bar\tau, \qquad
\bar p \mathrel{+}= f_{\text{fr},p}^\top\bar\tau, \qquad
\bar F \mathrel{+}= f_{\text{fr},F}^\top\bar\tau .
```

*Informative:* the per-step dot-product identity (§7.1) would not catch a missing term here, because its tangent side follows this section too.

### 5.7 `map` and domain clipping

A step map $u \leftarrow M(u, p, t, \Delta)$ has the reverse

```math
\bar u_{\text{before}} = \Big(\frac{\partial M}{\partial u}\Big)^{\!\top} \bar u_{\text{after}}, \qquad
\bar p \mathrel{+}= \Big(\frac{\partial M}{\partial p}\Big)^{\!\top} \bar u_{\text{after}} ,
```

evaluated with `map_vjp` at the pre-map state, and with every branch (`ifelse`, `max`, `min`, and masks such as iceberg removal or calving) frozen at the value the tape recorded. A map acting on parameters (parameter `update` rules) is the same with the roles of $u$ and $p$ exchanged.

**Domain clipping** (`on_violation: "clip"`, architecture §5.3) is the map $u_j \leftarrow \max(u_j, 0)$ on the listed DOFs, applied to the accepted state. Its adjoint is the recorded mask $m_j = [u_j \ge 0]$ before the clip: $\bar u_{\text{before}} = m \odot \bar u_{\text{after}}$. Ties at exactly $0$ keep the value and pass the cotangent. `reject` is a controller decision and has no adjoint.

### 5.8 `bind_dt`

`bind_dt` sets a model parameter to the step actually taken. That step is a tape constant (§2.2), so the bound parameter is a constant of the frozen-tape map:
- No cotangent is accumulated for it.
- A `wrt` that names a bound parameter is refused by the planner.

In forward sensitivities with `dt_tangents = true`, the bound parameter carries the tangent of $h$. This is consistent with §2.3.

### 5.9 `split_explicit` (post-v1 implementation, M7)

The forward is Wicker–Skamarock RK3 on the slow operators, with forward–backward acoustic substeps for the fast operators inside each stage (architecture §4.3). Its adjoint is the composition of:
- the RK3 stage adjoints (§4.1, with each stage's output defined by the acoustic integration rather than a single $f$ evaluation);
- the per-substep adjoints of the forward–backward scheme, whose vertically implicit part with `epssm` off-centring is a θ-type step (§4.5);
- the `frozen` tendencies (§5.6).

The exact forward arithmetic, and hence its adjoint, is specified with the M7 method chapter.

### 5.10 `coupled`

Without algebraic operators, `coupled` is a single leaf on the summed operators, and its adjoint is the family's. With `imex`, it is §4.4.

With algebraic operators it is a semi-explicit index-1 DAE (D3b; specified in v1, implemented after M6 unless M4b needs it, D35). What can be stated now: with $M = \mathrm{diag}(I, 0)$, stiffly accurate Rodas and SDIRK, and consistent initialization by solving $g(u_0, z_0) = 0$:
- the stage adjoints of §4.2 and §4.3 hold with $W$ replaced by the block $2\times2$ matrix $\begin{pmatrix} I/(h\gamma) - f_u & -f_z \\ -g_u & -g_z \end{pmatrix}$ (Rosenbrock) or its SDIRK analogue;
- the consistent initialization contributes a §4.6-type solve at $t_0$, mapping $\bar z_0$ into $\bar u_0$ and $\bar p$.

The DAE adjoint is written out when D3b is implemented.

## 6. Accumulation

### 6.1 Parameters

- $\bar p$ accumulates every term above that names $f_p$, $g_p$, a map's $\partial M/\partial p$ or $\nabla_p \varphi$.
- Scalar parameters reduce over shards and ranks in the fixed order of §6.4.
- Inactive parameters (planner classification, decisions §1.1) are never accumulated.

### 6.2 Initial state

$\bar u_0 = \lambda(t_0)$ after the whole reverse pass, including the `solve`-node and consistent-initialization contributions at $t_0$.

### 6.3 Gridded fields and forcing

- **Gridded fields** are array parameters. `vjp_p` writes into an array-shaped $\bar p$, one entry per cell. Until U7 lands, EarthSciASTDiff maps parameter columns only through scalar parameters, and array-parameter gradients are silently zero (architecture §10). The planner MUST refuse a `wrt` on array parameters until the array-parameter columns exist.
- **Halo contributions.** A stencil leaf's reverse pass on a tile writes cotangents into halo cells that another tile or rank owns. These are summed into the owner by a reverse halo exchange, in a fixed order, before the owner's next adjoint stage.
- **Forcing gradients accumulate per provider record** (D5), not per epoch:
  - Each buffer slot holds record $r(\text{slot}, \text{epoch})$, recorded on the tape.
  - `vjp_F` writes into slots: $\bar F_{\text{slot}}$.
  - At an epoch's end in the reverse pass, $\bar F_{\text{record}\ r} \mathrel{+}= \sum_{\text{slot} \mapsto r} \bar F_{\text{slot}}$.
  - The model's own time blend is inside $f$, so the chain rule into both bracketing records comes from `vjp_F` with no integrator logic.
  - A record's gradient is complete once both brackets containing it have been processed, so records complete in descending time order (architecture §10, U12).

### 6.4 Fixed-order reductions and precision

- Every accumulation that sums contributions from several stages, shards, tiles or ranks uses a fixed order (D10):
  - stages ascending within a step;
  - shards in shard order;
  - tiles in tile order;
  - ranks in rank order, gathered at the root and broadcast.
- Adjoint accumulators ($\lambda$ where it crosses leaf boundaries, $\bar p$, $\bar F$, $\bar u_0$) are f64 (D18). Within a leaf, stage cotangents use the leaf's state precision, and are widened exactly on accumulation.
- Under D10 the gradient is bitwise reproducible for a fixed decomposition, and agrees to tolerance across decompositions.

## 7. Verification identities

These are what the conformance suite checks (architecture §10, §12 tier 2; M4 gate). Each is stated per leaf step or per combinator invocation, then for the whole run.

### 7.1 Dot-product identity

For a step map $\Phi_n$, random directions $(\delta u, \delta p, \delta F)$ and a random $\lambda_{n+1}$, let:
- the tangent $\delta u_{n+1}$ come from the frozen-tape tangent-linear step;
- $(\lambda_n, \bar p, \bar F)$ come from the adjoint step with $\bar p = \bar F = 0$ on entry.

Then

```math
\langle \lambda_{n+1}, \delta u_{n+1}\rangle = \langle \lambda_n, \delta u\rangle + \langle \bar p, \delta p\rangle + \langle \bar F, \delta F\rangle ,
```

with the relative mismatch, scaled by the larger of the two sides' magnitudes:
- **≤ 1e-12** for explicit and Rosenbrock steps, and for SDIRK, θ and IMEX steps whose transposed solves are direct;
- **≤ the transposed-solve or refinement tolerance** where those solves iterate (Krylov, low-precision factors with refinement).

The tangent side uses the same stage values and the same exact-stage linearization (§4.3) as the adjoint, so Newton tolerance does not enter.

Sources of the tangent:
- **Julia:** forward sensitivities with `dt_tangents = false` (§2.3).
- **Rust:** none in v1. Rust has no forward sensitivities (D36) and does not write a tangent-linear step per family. Rust adjoints are gated instead by the finite-difference test (§7.2) and by agreement with Julia's gradient at tolerance (§7.5, conformance tier 3) (D44). The exact dot-product gate of M4 is Julia's.

The same identity is checked for the composition over one macro step and over a whole run, through the combinators of §5.

### 7.2 Finite differences on the frozen tape

Check the directional derivative $\langle \nabla \mathcal{J}, (\delta u_0, \delta p, \delta F)\rangle$ against central differences of $\Phi_{\text{tape}}$, at several step sizes $\epsilon$. A pass shows the $O(\epsilon^2)$ trend down to the roundoff floor.

The frozen-tape replay:
- holds $(t_n, h_n)$, every mask, branch and firing, and the epoch map at their recorded values;
- tightens implicit-stage and `solve`-node tolerances well below $\epsilon^2$. The adjoint is the exact implicit-function derivative, while finite differences of a loosely converged Newton carry an error of order tol/$\epsilon$.

Fixtures are scaled so a good $\epsilon$ exists. reseact's "never validate at the default initial condition" applies: perturb from sampled non-default states.

### 7.3 Oracles (D31)

- **`SciMLSensitivity.ReverseDiffAdjoint`** through OrdinaryDiffEq's own implementation of the same method, on toy problems, for every family:
  - `Rosenbrock23`, `SSPRK43`, and `TRBDF2(autodiff=false)` as named in architecture §5;
  - one representative per remaining v1 family.

  Both sides must differentiate the same map. The OrdinaryDiffEq solve therefore runs with `adaptive = false` and the tape's step sequence imposed (`dt` and `tstops` from the tape). *Assumption:* ReverseDiff through an adaptive solve would also differentiate the controller, which this spec excludes. ReverseDiff differentiates TRBDF2's Newton iterations, so agreement for implicit families is at the Newton tolerance.
- **Enzyme on a whole step** (Julia, decisions §1.4 option B), for the same toy problems and as a fallback oracle for any leaf.

### 7.4 Bitwise replay

Under D10, replaying a macro step from its checkpoint with the controller on reproduces the forward step sequence bitwise. Two checks:
- The replayed end state MUST equal the next checkpoint bit for bit (reseact had to use 1e-6).
- Per-cell step counts or hashes MUST match the tape.

A mismatch is an error, not a tolerance failure.

### 7.5 Cross-language

Julia and Rust gradients for the same plan agree to tolerance (conformance tier 3). They are not required to agree bitwise, because the transposed-solve routes (§4.3) and libm functions may differ.

## 8. Open questions and assumptions, collected

Decided on 2026-10-23 and recorded as D39–D44: which leaf integrates the integral term (§2.1), the two forward-sensitivity modes and `Dual` numbers in the f64 norm (§2.3), SDIRK transposed solves (§4.3), IMEX Euler's converged default (§4.4), and Rust's adjoint gate (§7.1).

Remaining:

1. *Assumption:* the packages holding the Tsit5, BS3, RK4 and Heun tableaus, and the inline Shu–Osher coefficients of `OrdinaryDiffEqSSPRK` (§4, §4.1). Settled when the methods chapter is written (M1).
2. *Assumption:* ReverseDiffAdjoint is run with the tape's step sequence imposed, so it does not differentiate the controller (§7.3).
