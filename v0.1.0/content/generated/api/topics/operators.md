---
title: "The operator interface"
description: "What a leaf calls on an operator: the right-hand side, its derivative products and Jacobian, and map operators."
type: "api"
---


| Operation | Kind | Bindings | Summary |
|---|---|---|---|
| [`apply`](/generated/api/ref/apply) | function | Julia | Apply the map operator: `out = M(u, p, t, Δ)`, recording in `mask` (one byte per decision) the discrete decisions taken; the adjoint holds them fixed (operator-interface.md §7). |
| [`ft`](/generated/api/ref/ft) | function | Julia | `out += ∂f_k/∂t`: the explicit time derivative. |
| [`ft_vjp_p`](/generated/api/ref/ft_vjp_p) | function | Julia | `out += ∇_p (νᵀ ∂f_k/∂t)`. |
| [`ft_vjp_u`](/generated/api/ref/ft_vjp_u) | function | Julia | `out += ∇_u (νᵀ ∂f_k/∂t)`. |
| [`hess_p`](/generated/api/ref/hess_p) | function | Julia | `out += ∇_p Σᵢ μᵢᵀ J_k(u) wᵢ`, for `μ` a vector of `(μᵢ, wᵢ)` pairs. |
| [`hess_u`](/generated/api/ref/hess_u) | function | Julia | `out += ∇_u Σᵢ μᵢᵀ J_k(u) wᵢ`, for `μ` a vector of `(μᵢ, wᵢ)` pairs. |
| [`jac`](/generated/api/ref/jac) | function | Julia | `J_k(u)` accumulated into the leaf's matrix storage (operator-interface.md §4). |
| [`jvp_u`](/generated/api/ref/jvp_u) | function | Julia | `out += J_k(u) v`: the Jacobian-vector product. |
| [`map_jvp`](/generated/api/ref/map_jvp) | function | Julia | The Jacobian-vector product `(∂M/∂u) v_u + (∂M/∂p) v_p` of a map operator, with the decisions in `mask` held fixed, written to `out`, for tangent checks (operator-interface.md §7). |
| [`map_vjp`](/generated/api/ref/map_vjp) | function | Julia | Accumulate the vector-Jacobian product of a map operator, with the decisions in `mask` held fixed, into `out_u` and `out_p` (operator-interface.md §7). |
| [`rhs`](/generated/api/ref/rhs) | function | Julia | `out += f_k(u, p, t)`: the operator's contribution to the right-hand side. |
| [`vjp_p`](/generated/api/ref/vjp_p) | function | Julia | `out += f_{k,p}(u)ᵀ w`: the vector-Jacobian product with respect to the parameters. |
| [`vjp_u`](/generated/api/ref/vjp_u) | function | Julia | `out += J_k(u)ᵀ w`: the vector-Jacobian product with respect to the state. |
| [`map_operator`](/generated/api/ref/map_operator) | type | Rust | A map operator `u ← M(u, p, t, Δ)` (§7). |
| [`operator`](/generated/api/ref/operator) | type | Rust | An additive term operator `f_k(u, p, t)` (§1, §3). |

[All topics](/generated/api)
