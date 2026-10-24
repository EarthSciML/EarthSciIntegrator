# The operator interface

**Status:** M1 draft, 2026-10-23. Normative for both implementations (D1, D8).

A leaf (spec/run-config.md §4.2) advances the sum of its operators. This chapter defines what a leaf calls on each operator: the right-hand side, its derivatives, and the structure the linear solvers rely on. From M2 the planner builds operators from the flattened model, with kernels compiled by EarthSciAST and band models from EarthSciASTDiff (architecture §2, §3.2 pass 6). At M1 the toy problems of [`toy-problems.md`](toy-problems.md) implement the interface by hand (D53). Nothing in the methods, controllers, schedule interpreter or adjoint may depend on which of the two supplied an operator.

## 1. What an operator is

An operator $k$ of a leaf contributes an additive term $f_k$ to the leaf's right-hand side:

```math
f(u, p, t) = \sum_k f_k(u, p, t).
```

- $u$ is the leaf's **state vector**: every degree of freedom (DOF) the leaf's operators write (architecture §3.2 pass 3), of length $N$ and element type $T$ (`f32` or `f64`, spec/run-config.md §7.1).
- $p$ is the **parameter vector**: the active scalar parameters, of length $P$, always `f64`. Array-valued (gridded) parameters and forcing buffers join the interface at M3–M4 (U7); §6 reserves their place.
- $t$ is time in seconds, always `f64` (D13).
- State an operator reads but no operator of the leaf writes is a constant of the leaf over its interval. The schedule interpreter supplies it; at M1 no toy problem has any.

An operator never stores state between calls. Every call is a pure function of its arguments, so one operator object may be called from several threads at once (U4). Workspace an operator needs is passed in by the caller.

## 2. Layout

The leaf's state is one flat vector. Its order is part of the operator set's declaration:

- **`dense`**: no structure is promised; DOF order is the operator set's own.
- **`block_diagonal`** with `ncells` cells of `n` DOFs each: every operator's Jacobian is zero between different cells. DOFs are **species-major**: DOF $i$ of cell $c$ (both 0-based) is at index $i \cdot \text{ncells} + c$. This is the struct-of-arrays layout that batched kernels vectorize across (architecture §6.1, the `LaneBlocks` layout of U10), and reseact's.

The leaf's structure is the coarsest that all its operators share: a leaf whose operators are all `block_diagonal` with the same `ncells` and `n` is `block_diagonal`; any other mix is `dense`. Banded and stencil structures are added with the planner (M2).

## 3. Calls

Every call **accumulates** into its output: `out ← out + (value)`. The leaf zeroes an output once, then calls each operator in its declared order. Accumulation in a fixed operator order makes the sum bitwise reproducible (D10).

All arguments are as in §1. $v$ and $w$ are vectors of length $N$, element type $T$; $\mu$ is a list of pairs $(\mu_i, w_i)$; outputs have the shape of the cotangent or tangent they hold. A $u$-shaped output has element type $T$; a $p$-shaped output is `f64`.

| Call | Accumulates | Required |
|---|---|---|
| `rhs(out, u, p, t)` | $f_k(u, p, t)$ | always |
| `jvp_u(out, u, p, t, v)` | $J_k(u)\, v$ | always |
| `vjp_u(out, u, p, t, w)` | $J_k(u)^\top w$ | always |
| `vjp_p(out, u, p, t, w)` | $f_{k,p}(u)^\top w$ | when an adjoint is taken with respect to $p$ |
| `jac(out, u, p, t)` | $J_k(u)$ into the leaf's matrix storage (§4) | for Rosenbrock, SDIRK, IMEX and θ-method leaves with a direct `linsolve`, from the operators whose Jacobian the method forms: for IMEX Euler the implicit part only, for the θ-method at θ = 0 none |
| `ft(out, u, p, t)` | $\partial f_k/\partial t$ | for Rosenbrock leaves whose operator depends on $t$ explicitly |
| `hess_u(out, u, p, t, μ)` | $\nabla_u \sum_i \mu_i^\top J_k(u)\, w_i$ | for Rosenbrock adjoints, and for adjoints of IMEX Euler with `implicit_solve: "linearized"` from its implicit part only (spec/adjoint.md §4.4) |
| `hess_p(out, u, p, t, μ)` | $\nabla_p \sum_i \mu_i^\top J_k(u)\, w_i$ | likewise, with respect to $p$ |
| `ft_vjp_u(out, u, p, t, ν)` | $\nabla_u\, (\nu^\top \partial f_k/\partial t)$ | for Rosenbrock adjoints of $t$-dependent operators |
| `ft_vjp_p(out, u, p, t, ν)` | $\nabla_p\, (\nu^\top \partial f_k/\partial t)$ | likewise, with respect to $p$ |

These are the primitives of spec/adjoint.md §3, per operator. The leaf's primitive is the sum over its operators.

Each operator also declares:
- `depends_on_t`: whether $f_k$ depends on $t$ explicitly. When false, `ft` and `ft_vjp_*` are zero and need not be implemented.
- `linear_in_u`: whether $J_k$ is independent of $u$. When true, `hess_u` is zero, and IMEX Euler's Newton iteration converges its implicit part in one linear solve (spec/adjoint.md §4.4). The other second-order calls need not vanish: `hess_p` is zero only when $J_k$ also does not depend on $p$, and `ft_vjp_u` only when it does not depend on $t$. For $f_k = -k\,u$, $J_k = -k$ and `hess_p` is $-\sum_i \mu_i^\top w_i$.
- Its structure (§2).
- `writes`: the indices of the state vector it writes, those where $f_k$ can be nonzero. The leaf's DOFs, over which the error norm and first step run (spec/methods.md §7, §9), are the union over its operators, in ascending order. At M1 operators take the full state vector; from M2 the plan's `writes` lists give the same set (spec/plan.md §3). *Deviation:* OrdinaryDiffEq norms over the whole state, so a leaf whose operators leave some entries unwritten divides by a smaller $N$ than an OrdinaryDiffEq solve of the same equations, and its step sizes can differ from that solve's.

A leaf whose method or adjoint needs a call its operators do not provide is refused when the leaf is built, naming the operator and the call.

## 4. Matrix storage

`jac` accumulates into storage the leaf's `linsolve` owns:
- **`block_diagonal`:** `ncells` dense `n × n` blocks, stored lane-innermost: entry $(i, j)$ of cell $c$ is at offset $(i \cdot n + j) \cdot \text{ncells} + c$. Batched LU (`block_lu`, architecture §6.1) operates on this layout, one operation across all lanes at a time.
- **`dense`:** an `N × N` column-major matrix. This is for toy problems and tests only; architecture §6.1 never assembles a dense W for a real model.

The leaf, not the operator, forms $W$ from $J$ and factors it (spec/methods.md).

## 5. Bindings

The names above are the spec's. Each binding spells them idiomatically and documents the mapping:
- **Julia:** methods on an operator type, `rhs!(op, out, u, p, t)` and so on, generic over `T`. At M1 the leaf's sum of operators is wrapped as a `SciMLBase.ODEFunction` (D33) so OrdinaryDiffEq can integrate the same problem as an oracle, with `jac` and `vjp` filled from the operator calls.
- **Rust:** a trait `Operator<T>` with `&self` methods taking slices, `Send + Sync` (U4).

## 6. Reserved for later milestones

- **Forcing** (M3–M4): a forcing-buffer argument `F` on every call, and `vjp_F`, `hess_F` and `ft_vjp_F` (U7, U8).
- **Array parameters** (M4): gridded parameters as additional $p$-shaped arguments with their own `vjp_*` (D5, U7).
- **Range restriction** (M3): calls over a tile with halos or a list of lanes rather than the whole state (U3).
- **Maps, limits, conditions and residuals** (M1 maps; M4b the rest): §7 defines maps; the others follow with U6.

## 7. Map operators

A map operator (spec/run-config.md §3.2) provides:
- `apply(out, u, p, t, Δ)`: writes $M(u, p, t, \Delta)$ into `out` (overwrites, unlike §3), and records into a caller-provided mask any discrete decision it takes;
- `map_vjp(out_u, out_p, u, p, t, Δ, mask, w)`: accumulates $(\partial M/\partial u)^\top w$ and $(\partial M/\partial p)^\top w$ at the pre-map state with the recorded decisions held fixed (spec/adjoint.md §5.7);
- `map_jvp(out, u, p, t, Δ, mask, v_u, v_p)`: $(\partial M/\partial u)\, v_u + (\partial M/\partial p)\, v_p$, for tangent checks;
- `apply_frozen(out, u, p, t, Δ, mask)`: like `apply`, but with the decisions in `mask` imposed rather than taken, so finite differences on the frozen tape (spec/adjoint.md §7.2) hold them fixed.

$\Delta$ is the value `bind_dt` gives the map (spec/run-config.md §4.4).
