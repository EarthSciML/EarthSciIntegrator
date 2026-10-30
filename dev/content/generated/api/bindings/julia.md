---
title: "Julia API index"
description: "Every stable name Julia exports, and its extension seams."
type: "api"
---


**Package:** `EarthSciIntegrator` 0.0.1

Documentation extracted with Base.Docs (Julia 1.13.3).

Extension-tier names are on the [Julia extension page](/generated/api/extension/julia).

## Stable names

| Julia | Canonical | Kind | Summary |
|---|---|---|---|
| `apply!` | [`apply`](/generated/api/ref/apply) | function | Apply the map operator: `out = M(u, p, t, Δ)`, recording in `mask` (one byte per decision) the discrete decisions taken; the adjoint holds them fixed (operator-interface.md §7). |
| `BandEntry` | [`band_entry`](/generated/api/ref/band_entry) | type | One entry of EarthSciASTDiff's band list (esm-jacobian-spec §4), in the form its `serialize_band` writes, without `coef`: |
| `build_run` | [`build_run`](/generated/api/ref/build_run) | function | Build a run from a run config (a parsed JSON object). |
| `ConfigError` | [`config_error`](/generated/api/ref/config_error) | error | Raised when a run config is invalid for a reason the schema cannot check (spec/run-config.md §8). |
| `forward!` | [`forward`](/generated/api/ref/forward) | function | Run the schedule forward over the case's interval with every leaf's controller on, recording the tape, a checkpoint at every macro-step boundary, and the outputs. |
| `ft!` | [`ft`](/generated/api/ref/ft) | function | `out += ∂f_k/∂t`: the explicit time derivative. |
| `ft_vjp_p!` | [`ft_vjp_p`](/generated/api/ref/ft_vjp_p) | function | `out += ∇_p (νᵀ ∂f_k/∂t)`. |
| `ft_vjp_u!` | [`ft_vjp_u`](/generated/api/ref/ft_vjp_u) | function | `out += ∇_u (νᵀ ∂f_k/∂t)`. |
| `gradient` | [`gradient`](/generated/api/ref/gradient) | function | The discrete adjoint gradient of the run's functional on the frozen tape (spec/adjoint.md §2.2): `ū₀ = λ(t₀)` (§6.2) and `p̄` over every parameter, with bind_dt-bound parameters… |
| `hess_p!` | [`hess_p`](/generated/api/ref/hess_p) | function | `out += ∇_p Σᵢ μᵢᵀ J_k(u) wᵢ`, for `μ` a vector of `(μᵢ, wᵢ)` pairs. |
| `hess_u!` | [`hess_u`](/generated/api/ref/hess_u) | function | `out += ∇_u Σᵢ μᵢᵀ J_k(u) wᵢ`, for `μ` a vector of `(μᵢ, wᵢ)` pairs. |
| `jac!` | [`jac`](/generated/api/ref/jac) | function | `J_k(u)` accumulated into the leaf's matrix storage (operator-interface.md §4). |
| `jcs` | [`jcs`](/generated/api/ref/jcs) | function | `x` (dictionaries, vectors, strings, numbers, booleans, `nothing`) serialized as RFC 8785 canonical JSON (spec/plan.md §2.1). |
| `jvp_u!` | [`jvp_u`](/generated/api/ref/jvp_u) | function | `out += J_k(u) v`: the Jacobian-vector product. |
| `map_jvp!` | [`map_jvp`](/generated/api/ref/map_jvp) | function | The Jacobian-vector product `(∂M/∂u) v_u + (∂M/∂p) v_p` of a map operator, with the decisions in `mask` held fixed, written to `out`, for tangent checks (operator-interface.md §7). |
| `map_vjp!` | [`map_vjp`](/generated/api/ref/map_vjp) | function | Accumulate the vector-Jacobian product of a map operator, with the decisions in `mask` held fixed, into `out_u` and `out_p` (operator-interface.md §7). |
| `odefunction` | [`odefunction`](/generated/api/ref/odefunction) | function | The sum of a leaf's operators as a `SciMLBase.ODEFunction` with `jac`, `tgrad`, `vjp` and `jvp` filled from the operator calls (operator-interface.md §5, D33). |
| `PendingBackend` | [`pending_backend`](/generated/api/ref/pending_backend) | type | M2 stage 1's backend: EarthSciASTDiff is not available, so nothing is derived. |
| `plan` | [`plan`](/generated/api/ref/plan) | function | Plan the run config at `config_path` (spec/plan.md): read and flatten its model, run passes 1–4, check and resolve the schedule and the settings, and resolve strategies. |
| `plan_dict` | [`plan_dict`](/generated/api/ref/plan_dict) | function | The whole `plan.json` object (spec/plan.md §3) of a `plan` draft. |
| `plan_json` | [`plan_json`](/generated/api/ref/plan_json) | function | `plan.json`'s bytes: `plan_dict` as RFC 8785 canonical JSON (§2.1). |
| `PlanBackend` | [`plan_backend`](/generated/api/ref/plan_backend) | type | What the planner needs from EarthSciASTDiff (M2 stage 2): a term's bands (§5.2.2), an operator's structure (§5.4), the Gershgorin bound (§5.5.2), and the derived documents with… |
| `PlannerRefusal` | [`planner_refusal`](/generated/api/ref/planner_refusal) | error | The planner refuses its inputs: `code` is one of spec/planner-errors.json's codes, `at` the JSON pointer into the run config of the offending key (`/model` for a refusal about… |
| `PlannerStageError` | [`planner_stage_error`](/generated/api/ref/planner_stage_error) | error | A part of the planner that M2 stage 1 does not implement yet: everything that needs EarthSciASTDiff (term and operator bands, structure, stiffness by the Gershgorin bound, the… |
| `replay` | [`replay`](/generated/api/ref/replay) | function | Replay the whole run on the frozen tape (spec/adjoint.md §2.2), generic over the element types of `u0` and `p`. |
| `ReplayMismatchError` | [`replay_mismatch_error`](/generated/api/ref/replay_mismatch_error) | error | Raised when a tape does not fit the run it is replayed for (spec/tape.md §3): a record missing or of the wrong kind, an interval other than the one the schedule gives the… |
| `rhs!` | [`rhs`](/generated/api/ref/rhs) | function | `out += f_k(u, p, t)`: the operator's contribution to the right-hand side. |
| `SPEC_VERSION` | [`spec_version`](/generated/api/ref/spec_version) | constant | Version of the integrator spec (the run config's `esm_integrator` key) this package implements. |
| `TABLEAUS` | [`tableaus`](/generated/api/ref/tableaus) | constant | Every method's `Tableau`, keyed by run-config `alg` name, read from `data/tableaus/*.json` (a copy of `spec/tableaus`) at precompile time. |
| `toy_problem` | [`toy_problem`](/generated/api/ref/toy_problem-function) | function | The toy problem a conformance case's `model` names: `toy:<problem>` or `toy:<problem>/<variant>` (toy-problems.md §2). |
| `UnsupportedError` | [`unsupported_error`](/generated/api/ref/unsupported_error) | error | Raised when a run config asks for something this implementation does not support yet (spec/run-config.md §8 rule 16): a method family not implemented in this stage, a combinator… |
| `vjp_p!` | [`vjp_p`](/generated/api/ref/vjp_p) | function | `out += f_{k,p}(u)ᵀ w`: the vector-Jacobian product with respect to the parameters. |
| `vjp_u!` | [`vjp_u`](/generated/api/ref/vjp_u) | function | `out += J_k(u)ᵀ w`: the vector-Jacobian product with respect to the state. |
