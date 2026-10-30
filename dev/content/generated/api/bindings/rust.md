---
title: "Rust API index"
description: "Every stable name Rust exports, and its extension seams."
type: "api"
---


Documentation extracted with rustdoc JSON (rustdoc 1.101.0-nightly (21b707e3f 2026-09-30)), format_version 61.

Extension-tier names are on the [Rust extension page](/generated/api/extension/rust).

## Stable names

| Rust | Canonical | Kind | Summary |
|---|---|---|---|
| `adjoint::Gradient` | [`gradient_result`](/generated/api/ref/gradient_result) | type | The functional and its gradient. |
| `config::RunConfig` | [`run_config`](/generated/api/ref/run_config) | type | A run configuration. |
| `earthsci_integrator_planner::jcs::to_string` | [`jcs`](/generated/api/ref/jcs) | function | The canonical form of `v`. |
| `earthsci_integrator_planner::Options` | [`plan_options`](/generated/api/ref/plan_options) | type | What the caller may give besides the run config. |
| `earthsci_integrator_planner::plan` | [`plan`](/generated/api/ref/plan) | function | Plans the run config at `path` as far as stage 1 goes. |
| `earthsci_integrator_planner::PlanError` | [`plan_error`](/generated/api/ref/plan_error) | error | Why planning stopped. |
| `earthsci_integrator_planner::Planned` | [`planned`](/generated/api/ref/planned) | type | Everything stage 1 computes for a run config. |
| `earthsci_integrator_planner::Refusal` | [`planner_refusal`](/generated/api/ref/planner_refusal) | error | A refusal: a code of spec/planner-errors.json, the JSON pointer into the run config it concerns (`/model` for the model), and a message. |
| `earthsci_integrator_planner::runconfig::read` | [`read_run_config`](/generated/api/ref/read_run_config) | function | Step 1 of spec/plan.md §4: parse the run config, validate it against `schemas/run-config.schema.json` (`run_config_invalid`) and refuse a version this planner does not implement… |
| `earthsci_integrator_planner::runconfig::RunConfig` | [`planner_run_config`](/generated/api/ref/planner_run_config) | type | A run config as read from its file. |
| `earthsci_integrator_solvers::controller::Controller` | [`controller`](/generated/api/ref/controller) | type | Controller constants plus the method's orders. |
| `earthsci_integrator_solvers::erk::ErkError` | [`erk_error`](/generated/api/ref/erk_error) | error | The error estimate (spec/methods.md §3.2). |
| `earthsci_integrator_solvers::erk::ExplicitRk` | [`explicit_rk`](/generated/api/ref/explicit_rk) | type | An explicit Runge–Kutta method built from its table. |
| `earthsci_integrator_solvers::implicit::Implicit` | [`implicit`](/generated/api/ref/implicit) | type | An implicit method built from its table. |
| `earthsci_integrator_solvers::initdt::initial_dt` | [`initial_dt`](/generated/api/ref/initial_dt) | function | Computes the first step. |
| `earthsci_integrator_solvers::RhsFn` | [`rhs_fn`](/generated/api/ref/rhs_fn) | type | The leaf's right-hand side as methods see it: `f(out, u, t)` overwrites `out`. |
| `earthsci_integrator_solvers::rosenbrock::Rosenbrock` | [`rosenbrock`](/generated/api/ref/rosenbrock) | type | A Rosenbrock method built from its table. |
| `earthsci_integrator_solvers::Scalar` | [`scalar`](/generated/api/ref/scalar) | type | A floating-point element type for state vectors: `f32` or `f64`. |
| `earthsci_integrator_solvers::tableau::all_tableaus` | [`all_tableaus`](/generated/api/ref/all_tableaus) | function | Every table, by name. |
| `earthsci_integrator_solvers::tableau::Tableau` | [`tableau`](/generated/api/ref/tableau-type) | type | One coefficient table, as `spec/schemas/tableau.schema.json` defines it. |
| `earthsci_integrator_solvers::tableau::tableau` | [`tableau`](/generated/api/ref/tableau-function) | function | The table for run-config `alg` value `name`, if there is one. |
| `Error` | [`error`](/generated/api/ref/error) | error | Errors: refusals of a run config or schedule, and failed runs. |
| `operator::MapOperator` | [`map_operator`](/generated/api/ref/map_operator) | type | A map operator `u ← M(u, p, t, Δ)` (§7). |
| `operator::Operator` | [`operator`](/generated/api/ref/operator) | type | An additive term operator `f_k(u, p, t)` (§1, §3). |
| `schedule::ForwardRun` | [`forward_run`](/generated/api/ref/forward_run) | type | The result of a forward run or a replay. |
| `schedule::RunSpec` | [`run_spec`](/generated/api/ref/run_spec) | type | A configured run: problem, schedule, interval, macro step, outputs and the adjoint settings. |
| `SPEC_VERSION` | [`spec_version`](/generated/api/ref/spec_version) | constant | Version of the integrator spec (the run config's `esm_integrator` key) this crate implements. |
| `tape::Tape` | [`tape`](/generated/api/ref/tape) | type | The tape of a run. |
| `toy::toy_problem` | [`toy_problem`](/generated/api/ref/toy_problem-function) | function | Builds the toy problem a case's `model: "toy:<problem>[/<variant>]"` names. |
| `toy::ToyProblem` | [`toy_problem`](/generated/api/ref/toy_problem-type) | type | A toy problem: its operators, layout, parameters, initial state, interval and the facts the conformance checks use. |
