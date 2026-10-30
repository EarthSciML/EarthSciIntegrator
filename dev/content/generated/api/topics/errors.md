---
title: "Errors"
description: "Refusals of a run configuration or plan, and failed runs."
type: "api"
---


| Operation | Kind | Bindings | Summary |
|---|---|---|---|
| [`config_error`](/generated/api/ref/config_error) | error | Julia | Raised when a run config is invalid for a reason the schema cannot check (spec/run-config.md §8). |
| [`erk_error`](/generated/api/ref/erk_error) | error | Rust | The error estimate (spec/methods.md §3.2). |
| [`error`](/generated/api/ref/error) | error | Rust | Errors: refusals of a run config or schedule, and failed runs. |
| [`plan_error`](/generated/api/ref/plan_error) | error | Rust | Why planning stopped. |
| [`planner_refusal`](/generated/api/ref/planner_refusal) | error | Julia, Rust | The planner refuses its inputs: `code` is one of spec/planner-errors.json's codes, `at` the JSON pointer into the run config of the offending key (`/model` for a refusal about… |
| [`planner_stage_error`](/generated/api/ref/planner_stage_error) | error | Julia | A part of the planner that M2 stage 1 does not implement yet: everything that needs EarthSciASTDiff (term and operator bands, structure, stiffness by the Gershgorin bound, the… |
| [`replay_mismatch_error`](/generated/api/ref/replay_mismatch_error) | error | Julia | Raised when a tape does not fit the run it is replayed for (spec/tape.md §3): a record missing or of the wrong kind, an interval other than the one the schedule gives the… |
| [`unsupported_error`](/generated/api/ref/unsupported_error) | error | Julia | Raised when a run config asks for something this implementation does not support yet (spec/run-config.md §8 rule 16): a method family not implemented in this stage, a combinator… |

[All topics](/generated/api)
