---
title: "Building and running"
description: "Turning a run configuration into a run, integrating it forward over an interval, and reading the result."
type: "api"
---


| Operation | Kind | Bindings | Summary |
|---|---|---|---|
| [`build_run`](/generated/api/ref/build_run) | function | Julia | Build a run from a run config (a parsed JSON object). |
| [`forward`](/generated/api/ref/forward) | function | Julia | Run the schedule forward over the case's interval with every leaf's controller on, recording the tape, a checkpoint at every macro-step boundary, and the outputs. |
| [`odefunction`](/generated/api/ref/odefunction) | function | Julia | The sum of a leaf's operators as a `SciMLBase.ODEFunction` with `jac`, `tgrad`, `vjp` and `jvp` filled from the operator calls (operator-interface.md §5, D33). |
| [`toy_problem`](/generated/api/ref/toy_problem-function) | function | Julia, Rust | The toy problem a conformance case's `model` names: `toy:<problem>` or `toy:<problem>/<variant>` (toy-problems.md §2). |
| [`forward_run`](/generated/api/ref/forward_run) | type | Rust | The result of a forward run or a replay. |
| [`run_config`](/generated/api/ref/run_config) | type | Rust | A run configuration. |
| [`run_spec`](/generated/api/ref/run_spec) | type | Rust | A configured run: problem, schedule, interval, macro step, outputs and the adjoint settings. |
| [`toy_problem`](/generated/api/ref/toy_problem-type) | type | Rust | A toy problem: its operators, layout, parameters, initial state, interval and the facts the conformance checks use. |
| [`spec_version`](/generated/api/ref/spec_version) | constant | Julia, Rust | Version of the integrator spec (the run config's `esm_integrator` key) this package implements. |
| [`tableaus`](/generated/api/ref/tableaus) | constant | Julia | Every method's `Tableau`, keyed by run-config `alg` name, read from `data/tableaus/*.json` (a copy of `spec/tableaus`) at precompile time. |

[All topics](/generated/api)
