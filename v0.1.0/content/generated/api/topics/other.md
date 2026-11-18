---
title: "Other"
description: "Stable operations not yet placed in a topic."
type: "api"
---


| Operation | Kind | Bindings | Summary |
|---|---|---|---|
| [`build_plan_run`](/generated/api/ref/build_plan_run) | function | Julia | A run of the plan in `dir` (spec/operator-interface.md §8, spec/toy-problems.md §4): its operators built by `plan_problem`, its resolved schedule and step, over the plan's `time`. |
| [`load_plan`](/generated/api/ref/load_plan) | function | Rust | Reads the plan directory `dir` and builds every operator its schedule names (operator-interface.md §8). |
| [`load_plan_with`](/generated/api/ref/load_plan_with) | function | Rust | `load_plan`, fetching the data of a plan with `forcing` as `data` says (operator-interface.md §8.4). |
| [`plan_problem`](/generated/api/ref/plan_problem) | function | Julia | Build every operator the resolved schedule of the plan in `dir` names, from its entry `operators.<name>` and its documents `derived/<name>.rhs.esm` and… |
| [`write_plan`](/generated/api/ref/write_plan) | function | Julia | Write the plan directory (spec/plan.md §1): `plan.json`, `derived/*.esm` and `plan.report.json`, and with `bandvalues = true` the conformance band-value files… |
| [`data_options`](/generated/api/ref/data_options) | type | Rust | The run-time options of the data layer (spec/operator-interface.md §8.4.2 item 1): in neither the run config nor the plan, since none changes what is computed. |
| [`diff_backend`](/generated/api/ref/diff_backend) | type | Julia | The planner's backend on EarthSciASTDiff (M2 stage 2), holding what one planning run derives: per operator its documents, bands and built evaluators, the initial state and the… |
| [`forcing_timeline`](/generated/api/ref/forcing_timeline) | type | Rust | The forcing epochs of a run (§8.4.3) and each epoch's buffers. |
| [`loaded_plan`](/generated/api/ref/loaded_plan) | type | Rust | A plan loaded for running: the problem its operators make and the run config of its resolved schedule and step. |
| [`planned_operator`](/generated/api/ref/planned_operator) | type | Julia, Rust | An operator a plan supplies (spec/operator-interface.md §8): the right-hand side and Jacobian documents built by EarthSciAST, the canonical DOFs it writes, its structure, and… |

[All topics](/generated/api)
