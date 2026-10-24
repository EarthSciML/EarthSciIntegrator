# The tape and checkpoints

**Status:** M1 draft, 2026-10-23. Normative for both implementations. M1 covers single-rank, single-shard runs (`dt_scope: global`); shard compositions, per-cell step counts and forcing epochs are added at M3, and the checkpoint schedule's execution at M4 (D34).

The tape is the record of a forward run that the adjoint replays (architecture §10). Every tape entry is a constant of the frozen-tape map the adjoint differentiates (spec/adjoint.md §2.2). This chapter defines what the tape holds, how it is keyed, what a checkpoint holds, and a serialized form for tests.

## 1. Invocations

The schedule interpreter advances the model by **invocations**: one call of one schedule node over one interval. An invocation is identified by its **path**:

```
<macro step index> / <node id> [ #<occurrence> ]
```

- The macro step index counts from 0.
- The node id is the resolved schedule's (spec/plan.md §3); before M2, the interpreter assigns ids by the same rule (`root`, `root.0`, …): children of `sequence` and `strang` are numbered from 0 (a `coupled` node has operators, not children), and the child of a wrapping node is named by the node's key (`root.1.subcycle`, `root.every`).
- `#<occurrence>` counts, from 0, the invocations of the same node within the same macro step: the two halves of a `strang` child, a `subcycle` child's substeps, and a leaf inside either. It is omitted when 0.

Invocation order within a macro step is the execution order. Paths and order are the same in every implementation for a given run config, because the schedule's structure, not the step sizes, determines them, except for adaptive `subcycle`, whose substep count is itself recorded.

## 2. What is recorded

| Node | Recorded per invocation |
|---|---|
| `leaf`, `coupled` | `t`, `Δ`; the accepted steps as a list of `(t_n, h_n)`; counts of accepted and rejected steps; the return code. For SDIRK and IMEX leaves, nothing about Newton iterations: the adjoint differentiates the converged stages (spec/adjoint.md §4.3). |
| `subcycle` | the number of substeps and, under `limits`, each substep's size and the limit that set it. |
| `every` | whether it fired, and Δₑ when it did (spec/run-config.md §4.9). |
| `map` | the mask of discrete decisions the map recorded (operator-interface.md §7). |
| `solve` | the converged unknowns and the `recover` actions taken (M4b). |
| macro step | `t`, `Δ`, and the reason string for Δ (spec/run-config.md §5): under a fixed `step.dt`, `step.dt`, or `tstop` when the step was shortened to a tstop. Every record, a `subcycle`'s, an `every`'s and a macro step's included, follows the records of the invocations inside it. |

Clip masks under `domain.on_violation: "clip"` are recorded with the leaf step that clipped. In the serialized form (§5) a leaf record carries `"clips": [[k, [i, ...]], ...]`: the 0-based index $k$ of the accepted step in the record's `steps`, and the ascending 0-based indices $i$, into the leaf's state vector, of the DOFs that step clipped. Steps that clipped nothing are omitted, and so is `clips` when no step clipped. Rejected steps leave no record beyond the rejected count (spec/adjoint.md §2.2).

Stage values are **not** on the tape. The adjoint recomputes them by replaying each leaf invocation from its starting state with the recorded `(t_n, h_n)` (architecture §10, step 2).

## 3. Replay

Replay re-executes an invocation with the step sequence taken from the tape instead of the controller: the controller is not consulted, and each `h_n` is used as recorded. Under D10 a replay from the same starting state reproduces every stage and every accepted state bitwise. The conformance check `replay` (spec/toy-problems.md §2.1) verifies this.

*Informative.* Architecture §10 also allows replaying with the controller on, verified by step counts, which is what makes per-cell step sizes cheap to tape at M3. At M1 the recorded sequence is used.

## 4. Checkpoints

A checkpoint is taken at a macro-step boundary and holds everything needed to resume the forward run bitwise:
- the full state $u$ of every leaf, in each leaf's element type;
- per leaf (per shard or cell under `dt_scope: shard` or `cell`), the whole persistent state of spec/methods.md §8.6: the proposed next step, the controller's error history ($E_{\text{old}}$ or the PID history), the Newton rate $\theta_{\text{prev}}$ and $\eta_{\text{old}}$ (§5.2), and whether it has accepted a step (which decides between `qmax_first_step` and `qmax`);
- per `every` node, the time of its last firing;
- per `solve` node, the warm start (M4b);
- the macro step index and time.

Restart (architecture §11) uses the same contents. At M1 the adjoint checkpoints every macro step in memory; the planner's checkpoint schedule (D34) replaces this at M4.

## 5. Serialized form

For tests and debugging, an implementation can write its tape as JSON Lines, one object per invocation in execution order:

```json
{"path": "0/root.0", "node": "leaf", "t": 0, "dt": 0.025, "steps": [[0, 0.0125], [0.0125, 0.0125]], "accepted": 2, "rejected": 0, "retcode": "Success"}
{"path": "0/root.1", "node": "map", "t": 0, "dt": 0.025, "mask": []}
{"path": "0", "node": "macro", "t": 0, "dt": 0.025, "reason": "step.dt"}
```

- Numbers are written in shortest round-trip form, so a tape read back is bitwise the tape written. JSON numbers cannot hold NaN or infinities, so these are written as the strings `"NaN"`, `"Infinity"` and `"-Infinity"`; `"NaN"` reads back as the canonical quiet NaN.
- The macro-step record follows the invocations of its step.
- The form is for comparison and diagnosis. Implementations need not hold the tape in this form in memory.
- Cross-language comparison of tapes is by path and count, not by value: step sizes agree across languages only to tolerance (D10).
