# Conformance suite

Shared by the Julia and Rust implementations (architecture §12, spec/README.md).

| Directory | Contents | Checked by |
|---|---|---|
| `run-configs/valid/` | Run configs the schema must accept. `reseact`, `eqice` and `eqwefic` are architecture §4.4's three examples, completed with the `esm_integrator`, `model` and `operators` keys the architecture's fragments omit; their schedules are unchanged. The eqice and EqWeFiC operator selectors use illustrative component names until M4b and M7 bind them to the real documents. | `validate_schemas.py`; the Julia and Rust test suites also check their `esm_integrator` version |
| `run-configs/invalid/` | Run configs the schema must reject, one defect each, named for the defect. | `validate_schemas.py` |
| `plans/valid/`, `plans/invalid/` | The same for `plan.json`. The valid plan is hand-written and reseact-shaped; real goldens arrive with the planner (M2). | `validate_schemas.py` |

Each `invalid/` directory has an `expected-errors.json` naming, per fixture, the schema keyword that must fail and where; a fixture that fails anywhere else has a second defect and is reported. Run tier 0 locally with `python tests/conformance/validate_schemas.py` (needs `jsonschema` ≥ 4.18).
