---
title: "Replay and adjoint"
description: "Replaying a recorded run on its frozen tape and differentiating it with the discrete adjoint."
type: "api"
---


| Operation | Kind | Bindings | Summary |
|---|---|---|---|
| [`gradient`](/generated/api/ref/gradient) | function | Julia | The discrete adjoint gradient of the run's functional on the frozen tape (spec/adjoint.md §2.2): `ū₀ = λ(t₀)` (§6.2) and `p̄` over every parameter, with bind_dt-bound parameters… |
| [`replay`](/generated/api/ref/replay) | function | Julia | Replay the whole run on the frozen tape (spec/adjoint.md §2.2), generic over the element types of `u0` and `p`. |
| [`gradient_result`](/generated/api/ref/gradient_result) | type | Rust | The functional and its gradient. |
| [`tape`](/generated/api/ref/tape) | type | Rust | The tape of a run. |

[All topics](/generated/api)
