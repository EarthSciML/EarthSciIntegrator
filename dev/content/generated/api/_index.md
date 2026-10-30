---
title: "API reference"
description: "Every stable operation of the two EarthSciIntegrator bindings, side by side."
type: "api"
api_tabs: true
---


EarthSciIntegrator has two bindings — Julia and Rust — that implement the one EarthSciIntegrator spec (`spec/`) natively, checked against each other by a shared conformance suite. This reference has one page per **stable** operation, with a tab for each binding showing its spelling, signatures and documentation, taken from the binding's own documentation tooling, along with runnable examples. The examples run in CI, so what is shown here is what the code does.

The surface has two tiers. **Stable** names are harmonized across bindings and break only at a major version. **Extension** names are a binding's own documented seams; they may differ between bindings and may change in a minor release, and each binding lists them on one page.

## Guides

- [Getting started](/generated/api/guides/getting-started) — Build a run from a run configuration, integrate it, differentiate it with the discrete adjoint and replay it, in Julia and Rust.

## Topics

- [Building and running](/generated/api/topics/runs) (10) — Turning a run configuration into a run, integrating it forward over an interval, and reading the result.
- [Replay and adjoint](/generated/api/topics/adjoint) (4) — Replaying a recorded run on its frozen tape and differentiating it with the discrete adjoint.
- [The operator interface](/generated/api/topics/operators) (15) — What a leaf calls on an operator: the right-hand side, its derivative products and Jacobian, and map operators.
- [The planner](/generated/api/topics/planner) (11) — Planning a run from an `.esm` document and a run configuration (Julia, and the Rust `earthsci-integrator-planner` crate).
- [Solvers](/generated/api/topics/solvers) (10) — The time-stepping methods (Rust `earthsci-integrator-solvers` crate): explicit Runge-Kutta, Rosenbrock and implicit solvers, the step-size controller and the method tableaus.
- [Errors](/generated/api/topics/errors) (8) — Refusals of a run configuration or plan, and failed runs.

Or browse [all stable operations A–Z](/generated/api/ref).

## Bindings

| Binding | Package | Version | Docs from | Stable | Extension |
|---|---|---|---|---|---|
| [Julia](/generated/api/bindings/julia) | `EarthSciIntegrator` | 0.0.1 | Base.Docs (Julia 1.13.3) | 33 | [0](/generated/api/extension/julia) |
| [Rust](/generated/api/bindings/rust) |  | 0.0.1 | rustdoc JSON (rustdoc 1.101.0-nightly (21b707e3f 2026-09-30)), format_version 61 | 30 | [0](/generated/api/extension/rust) |

[Documentation coverage](/generated/api/coverage) lists the stable names each binding has not documented yet.
