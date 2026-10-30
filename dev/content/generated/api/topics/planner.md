---
title: "The planner"
description: "Planning a run from an `.esm` document and a run configuration (Julia, and the Rust `earthsci-integrator-planner` crate)."
type: "api"
---


| Operation | Kind | Bindings | Summary |
|---|---|---|---|
| [`jcs`](/generated/api/ref/jcs) | function | Julia, Rust | `x` (dictionaries, vectors, strings, numbers, booleans, `nothing`) serialized as RFC 8785 canonical JSON (spec/plan.md §2.1). |
| [`plan`](/generated/api/ref/plan) | function | Julia, Rust | Plan the run config at `config_path` (spec/plan.md): read and flatten its model, run passes 1–4, check and resolve the schedule and the settings, and resolve strategies. |
| [`plan_dict`](/generated/api/ref/plan_dict) | function | Julia | The whole `plan.json` object (spec/plan.md §3) of a `plan` draft. |
| [`plan_json`](/generated/api/ref/plan_json) | function | Julia | `plan.json`'s bytes: `plan_dict` as RFC 8785 canonical JSON (§2.1). |
| [`read_run_config`](/generated/api/ref/read_run_config) | function | Rust | Step 1 of spec/plan.md §4: parse the run config, validate it against `schemas/run-config.schema.json` (`run_config_invalid`) and refuse a version this planner does not implement… |
| [`band_entry`](/generated/api/ref/band_entry) | type | Julia | One entry of EarthSciASTDiff's band list (esm-jacobian-spec §4), in the form its `serialize_band` writes, without `coef`: |
| [`pending_backend`](/generated/api/ref/pending_backend) | type | Julia | M2 stage 1's backend: EarthSciASTDiff is not available, so nothing is derived. |
| [`plan_backend`](/generated/api/ref/plan_backend) | type | Julia | What the planner needs from EarthSciASTDiff (M2 stage 2): a term's bands (§5.2.2), an operator's structure (§5.4), the Gershgorin bound (§5.5.2), and the derived documents with… |
| [`plan_options`](/generated/api/ref/plan_options) | type | Rust | What the caller may give besides the run config. |
| [`planned`](/generated/api/ref/planned) | type | Rust | Everything stage 1 computes for a run config. |
| [`planner_run_config`](/generated/api/ref/planner_run_config) | type | Rust | A run config as read from its file. |

[All topics](/generated/api)
