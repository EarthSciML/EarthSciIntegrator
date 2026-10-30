---
title: "Solvers"
description: "The time-stepping methods (Rust `earthsci-integrator-solvers` crate): explicit Runge-Kutta, Rosenbrock and implicit solvers, the step-size controller and the method tableaus."
type: "api"
---


| Operation | Kind | Bindings | Summary |
|---|---|---|---|
| [`all_tableaus`](/generated/api/ref/all_tableaus) | function | Rust | Every table, by name. |
| [`initial_dt`](/generated/api/ref/initial_dt) | function | Rust | Computes the first step. |
| [`tableau`](/generated/api/ref/tableau-function) | function | Rust | The table for run-config `alg` value `name`, if there is one. |
| [`controller`](/generated/api/ref/controller) | type | Rust | Controller constants plus the method's orders. |
| [`explicit_rk`](/generated/api/ref/explicit_rk) | type | Rust | An explicit Runge–Kutta method built from its table. |
| [`implicit`](/generated/api/ref/implicit) | type | Rust | An implicit method built from its table. |
| [`rhs_fn`](/generated/api/ref/rhs_fn) | type | Rust | The leaf's right-hand side as methods see it: `f(out, u, t)` overwrites `out`. |
| [`rosenbrock`](/generated/api/ref/rosenbrock) | type | Rust | A Rosenbrock method built from its table. |
| [`scalar`](/generated/api/ref/scalar) | type | Rust | A floating-point element type for state vectors: `f32` or `f64`. |
| [`tableau`](/generated/api/ref/tableau-type) | type | Rust | One coefficient table, as `spec/schemas/tableau.schema.json` defines it. |

[All topics](/generated/api)
