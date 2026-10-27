# The run configuration

**Status:** M0 draft, 2026-10-23; M2 additions 2026-10-25 (`axes`, §6.1; time origin, §1.1; selector attribution, §3.1; refusal codes, §8). Normative for both implementations (D1, D8). The structure is fixed by [`schemas/run-config.schema.json`](schemas/run-config.schema.json); this chapter gives the meaning, and the rules a schema cannot check (§8). Background is in architecture §3–§5 and decisions D2, D21, D24.

A run configuration says how to integrate one `.esm` model: how its right-hand side is split into operators, how the operators are scheduled and with which methods, how the macro step is chosen, how the domain is partitioned, at what precision each part runs, and what the adjoint differentiates. The `.esm` stays free of algorithm names (D2).

## 1. The document

- A run config is a JSON document (D24) that validates against the schema for its `esm_integrator` version.
- `esm_integrator` is the version of this spec, independent of the esm format version. An implementation refuses a version it does not implement.
- `model` is the path of the `.esm` document, resolved relative to the run config's own location. In conformance cases only, `toy:<problem>` names a hand-written toy problem instead (spec/toy-problems.md §2).
- **Precedence** (architecture §3.1). The run config's `model` is authoritative for a run. If the `.esm` also carries `solver.external` (U1), the two must name each other, or the planner refuses. The planner never writes `solver.external` into derived documents. Solver hints on mounted component files are dropped at assembly (U1), so per-component tolerances are restated here.
- `time` is the run interval (§1.1).
- Unknown keys are errors at every level. There is no extension mechanism in 0.1.0.

### 1.1 The run interval: `time`

A run integrates over an interval $[t_0, t_f]$. Three sources may give it. In order of preference (D60):

1. **The caller**: an argument of the API call that plans or runs the model, or a command-line option of a tool that does.
2. **The run config**: `time`.
3. **The model**: the `.esm` document's `solver.time_span` (U16, in EarthSciAST 0.4.0 and esm 1.3.0; esm-spec §2.2.7), read with EarthSciAST's `resolve_time_span`.

The most preferred source that gives an interval gives all of it; start and end are never taken from different sources. If none gives one, the planner refuses (§8 rule 14).

`time` is `{"start": T, "end": T}` or `{"start": T, "duration": D}`. `D` is a number of seconds or a `"<number> <unit>"` string (§2.2), but not a parameter name: the interval is fixed when the plan is made, and a parameter can be overridden or differentiated later. `T` is a **time**, either:
- a number of seconds of model time; or
- an RFC 3339 timestamp in UTC (`"2019-07-01T00:00:00Z"`), meaning the seconds from the model's **time origin** to that instant, with no leap seconds.

The time origin is the model's, by esm-spec §2.2.7: `domain.temporal.reference_time`, else `domain.temporal.start`, else `1970-01-01T00:00:00Z` (POSIX time). A timestamp from the caller or the run config is converted exactly as EarthSciAST converts a `solver.time_span` timestamp, so all three sources share one clock (D74; this replaces the M0 rule that fixed run-config timestamps at the POSIX epoch). A model with no temporal anchor whose $t = 0$ is its own start takes numbers. `start` and `end` are both numbers or both timestamps; the schema refuses a mix. A timestamp names a real instant: the schema bounds each field and has no second 60 (no leap seconds), and the planner refuses a day the month does not have (§8 rule 14).

The end must be after the start (§8 rule 14). The caller gives a time the same way. `solver.time_span` numbers are seconds of the document's independent variable (esm-spec §2.2.7); a timestamp anywhere is refused when the model's origin field is present but unreadable (§8 rule 14).

Times derived from the interval count from model time zero, not from $t_0$: `hit_multiples` (§5) and `every` firing times (§4.9). With timestamps they are therefore aligned to UTC, so a `hit_multiples` of `"1 h"` falls on the hour. The one exception is a fixed `step.dt`, whose macro steps start at $t_0$ (§5).

The planner resolves the interval afresh for every run, from the sources that run has, and records it in seconds in the plan (spec/plan.md §3). The plan does not record which source gave it, so the same interval always gives the same plan. A plan is stale when the interval resolved for a run differs from the one it records.

## 2. Conventions

### 2.1 Names

- A **qualified name** is a model variable, parameter or observed, namespaced by component path with dots: `ssa.nuH`, `wsm6.dtcld`. It is resolved against the flattened model.
- An **operator name** is a plain identifier defined under `operators`, or a default operator (§3.3).
- The four locality names `pointwise`, `column`, `stencil` and `global` are reserved. They are never operator names, so wherever a value may be either (§4.5), there is no ambiguity.
- `auto` and `default` are reserved too: `auto` because it is the planner's-choice value (§2.3), and `default` because it is the fallback key of `precision` (§7.1), whose other keys are operator names and node ids. Neither is an operator name or a node id.
- Names beginning `default_` are reserved for the default operators (§3.3).
- A **node id** is an optional identifier on any schedule node, unique within the schedule. Precision, partition and output may refer to a node by id. The planner gives every node an id in the plan: unnamed nodes get their path from the root (`root`, `root.0`, `root.1.every`; spec/plan.md §7.1). A run config may write a path-form id only on the node whose path it is.

### 2.2 Durations

A duration is positive. It is one of:
- a number of seconds;
- a string `"<number> <unit>"`, with the space optional, where the unit is `s`, `min` (60 s), `h` (3600 s) or `day` (86 400 s). The number's digits before any exponent are not all zero, so `"0 s"` and `"0.0 day"` are refused by the schema. Its value in seconds is the number parsed as the nearest binary64, times the unit's seconds (an exact integer), one rounding in all (spec/plan.md §7.1);
- the qualified name of a model parameter whose value is in seconds (WRF's `chemdt`). It is read once, at plan time, from the parameter's default or the value the run supplies, and must be a positive finite number, else the planner refuses (`run_config_invalid`, at the duration's key).

The words `auto`, `step` and `accumulated` are reserved and are never parameter names in a duration.

Years and months are not units, because they are not constant durations (D45). Year-scale settings are written in days (eqice's bed-deformation period is `"3650 day"`). Calendar-aware periods, if a model needs them, would be a separate construct in a later version.

### 2.3 `"auto"`

Where the schema allows `"auto"`, the planner chooses by the rules of architecture §3.5 and records the choice in the plan. Omitting an optional key means the same as `"auto"` unless the key's description gives a different default.

## 3. Operators

An operator is a named group of additive right-hand-side terms, or a step map. The planner forms operators from the flattened model (architecture §3.2 pass 1).

### 3.1 Term operators: `select`

`{"select": {...}, "bind_dt": {...}}`.

- The planner splits every `D(x, t)` right-hand side into additive terms: the summands of each U2 contribution (spec/plan.md §5.1.1, D64). Each term carries its provenance, the component (or coupling entry) that authored it, written as a **provenance label**: the component path, `coupling:<k>` for the coupling entry at 0-based position $k$, or several joined by `+`.
- A selector matches a term when **every** key present matches (conjunction):
  - `components`: the term's **owner** is one of these components or lies under one of them. The owner is the authoring component; for a term a coupling entry wrote, it is the one other component whose variables the term reads, and a term reading variables of two or more other components has none (spec/plan.md §5.1.2, D65);
  - `variables`: the term contributes to one of these state variables;
  - `locality`: the term's locality (architecture §3.2 pass 2) is one of these;
  - `terms`: the term's provenance label is listed exactly.
- **Each term belongs to exactly one operator.** A term matched by two selectors is an error naming both operators (§8). A term with no owner that no selector matches is an error when any selector uses `components`, since its operator would otherwise be chosen by default (§8 rule 4).
- Algebraic equations are selected the same way, by component or variable, and form algebraic operators that a `solve` node (§4.10) names.
- `bind_dt` (§4.4) here applies wherever the operator is advanced.

### 3.2 Map operators: `map`

`{"map": {"<target>": "<observed>", ...}, "bind_dt": {...}}`.

- Each target is a state variable or parameter, and each source is a qualified observed that computes its new value. A model writes a map as an observed that reads `dt` (eqice `stage2/geometry.esm`), and `bind_dt` binds that `dt`.
- All sources are evaluated at the pre-map state, then all targets are assigned at once.
- A map that targets parameters is how esm parameter `update` rules run (architecture §3.2 pass 1).

### 3.3 Default operators

Terms no selector matches fall into **default operators**, one per locality, named `default_pointwise`, `default_column`, `default_stencil` and `default_global`. A default operator exists only if it has terms. It may be named in the schedule like any other.

### 3.4 Step limits

A step limit (in `step.limits` or `subcycle.limits`) is a qualified observed whose value is the largest admissible step in seconds, such as a CFL bound, or `{"max": duration}`. The planner compiles each observed limit as a `limit` operator (architecture §3.2 pass 1).

## 4. The schedule

The schedule is a tree of nodes. Every node advances the model over an interval `[t, t+Δ]` that its parent gives it; the root receives the macro step (§5). Exactly one combinator key identifies a node's kind.

### 4.1 `"auto"`

`"schedule": "auto"` asks for the default composition (D21): a `sequence` of one leaf per locality in the order stencil → column → pointwise, upgraded to `strang` when the model's `solver.splitting` hint asks for it. Each leaf's options are chosen by architecture §3.5.

### 4.2 `leaf`

`{"leaf": <operators>, "alg": ..., ...}` advances the **sum** of its operators over `[t, t+Δ]` with one method.

- `alg` names the method (architecture §5). Tableau coefficients and controller constants are OrdinaryDiffEq's (D31).
- **Substeps.** With `dt` omitted, the leaf steps adaptively under `control` and may take any number of substeps; the last substep ends exactly at `t+Δ`. A numeric `dt` gives fixed substeps, the last shortened to land on `t+Δ`. `"step"` takes exactly one step of size Δ. `"accumulated"` takes one step whose size is the time since this leaf last ran (equal to Δ unless the leaf sits under `every`).
- **Methods without an error estimate.** Adaptive stepping needs an embedded error estimate. `ForwardEuler`, `SSPRK33`, `SSPRK104`, `WRF_RK3`, `IMEXEuler` and `Theta` have none, so a leaf using one of them must give `dt` (D52). Legacy increments use `"step"`. An `alg` of `"auto"` with `dt` omitted resolves to a method that has an estimate, and on a stiff leaf never to SDIRK2, Kvaerno3 or Kvaerno4, whose estimate is unreliable on stiff problems (D59).
- `control`: `reltol` and `abstol` (a number, or per variable); `dt0`, `dtmin`, `dtmax`; `controller` (PI, I or PID, with OrdinaryDiffEqCore's constants per method); `dt_scope` (`global`, `shard` or `cell`; D17 makes `cell` the default for pointwise leaves and `global` for stencil leaves); `maxiters`. Omitted tolerances default to DiffEqBase's, `reltol` $10^{-3}$ and `abstol` $10^{-6}$. An implicit leaf (SDIRK, IMEX Euler, θ-method) uses the tolerances at fixed `dt` too, to scale its Newton increments (spec/methods.md §5.2). A non-finite error estimate is a reject. A step below `dtmin` ends the run with a return code instead of hanging (architecture §5.1).
- `jacobian`: `symbolic` (EarthSciASTDiff band models), `ad` (DifferentiationInterface, Julia only, D38) or `fd`.
- `linsolve`: the W-matrix solver of architecture §6.1, or `dense_lu`, a dense LU of the whole system, for toy problems and tests only (operator-interface.md §4). The planner resolves an omitted or `"auto"` `linsolve` by spec/plan.md §5.5.3; for the hand-written toy problems of M1, which the planner does not plan, the default is `block_lu` for a `block_diagonal` leaf and `dense_lu` otherwise. As an object it carries options: `block_lu.pivot` (default true), `refine` (iterative-refinement sweeps for low-precision factors, architecture §8), Krylov `restart`, `rtol`, `maxiter`, `precond` and `fallback` (the preconditioner tried when the first fails to converge).
- `domain`: `nonnegative` (`"all"` or a list of variables), `on_violation` (`reject`, the default, D19; or `clip`), and `check_stages` (also check Rosenbrock stage values).
- `imex` (§4.5), `implicit_solve` (§4.5), `frozen` (§4.6), `partition` (§6; a `leaf` only, since a `coupled` node is not sharded at M2), `bind_dt` (§4.4) and `theta` (the θ-method's off-centring, in $[0, 1]$). `theta` is required when `alg` is `Theta`, and a leaf without it is refused; it has no default.

### 4.3 `coupled`

`{"coupled": <operators>, ...}` takes the same options as `leaf`. Without algebraic operators it is the same as a `leaf`. With algebraic operators it is a semi-explicit index-1 DAE integrated monolithically (D3b; architecture §6.4), which is specified for v1 and implemented after M6 unless M4b needs it (D35). Until then the planner refuses it.

### 4.4 `bind_dt`

`bind_dt` maps a model parameter to a step size, set before every evaluation that reads it:
- `"leaf"`: the size of the step the leaf's method is actually taking. With `dt: "step"` and `alg: "ForwardEuler"`, this reproduces a legacy `(x_out − x_in)/dt` tendency exactly (architecture §4.3).
- `"node"`: the Δ the enclosing node received.
- `"macro"`: the macro step.

A bound parameter is a constant of the adjoint's frozen tape, so it cannot appear in `adjoint.wrt` (spec/adjoint.md §5.8).

### 4.5 IMEX partitions

`imex: {"explicit": ..., "implicit": ...}` splits a leaf's terms between the explicit and implicit parts of an IMEX method. Entries are operator names, or locality names that select the leaf's terms of that locality. Every term of the leaf must land in exactly one part; `explicit` may be omitted, meaning "everything not implicit". `implicit` is required, and an empty implicit list is refused. In v1 the only IMEX method is `IMEXEuler` (D35). Its implicit part is solved to convergence by Newton with the leaf's `linsolve` unless `implicit_solve: "linearized"` asks for a single linearized step (D43); the two coincide when the implicit terms are linear in the state and independent of $t$, since the linearized step evaluates $f_I$ at $(u_n, t_n)$ and the converged step at $(u_{n+1}, t_n + h)$ (for $f_I = -k u + c(t)$ they differ by $h(I - hJ)^{-1}(c(t_n + h) - c(t_n))$). Implementations run the `implicit_solve` the leaf asks for in every case and never substitute one for the other.

### 4.6 `frozen`

`frozen: [...]` on a `leaf`, `coupled` or `split_explicit` node lists operators whose tendency is evaluated once, at the start of each step of that node, and added as a constant to every stage of that step (WRF `rk_addtend_dry`). An entry `{"every": op, "period": ..., "hold": "tendency"}` is re-evaluated only on steps where `every` fires (§4.9) and held in between (WRF `radt`).

`frozen` is an option of the node whose stages it feeds, not a standalone node (D47). Alone in a `sequence`, a frozen tendency would just be a forward Euler step.

### 4.7 `sequence` and `strang`

- `sequence`: the children in order, each over the whole `[t, t+Δ]` (Lie–Trotter). With maps and solves among the children this is PISM's ordered step.
- `strang`: for children `a, b`: `a(Δ/2) b(Δ) a(Δ/2)`. For `n` children `c₁ … cₙ`: `c₁(Δ/2) … cₙ₋₁(Δ/2) cₙ(Δ) cₙ₋₁(Δ/2) … c₁(Δ/2)`. Each half is a separate invocation with its own step sequence.

### 4.8 `subcycle`

`{"subcycle": node, "n": k}` runs the child `k` times with Δ/k: substep $i$ starts at $t + i\,(Δ/k)$, and the last ends at $t + Δ$ exactly. `{"subcycle": node, "limits": [...]}` runs it adaptively: each substep is the minimum of the limits evaluated at the substep's start state and the time remaining, floored at `step.resolution`. The substep sequence is recorded on the tape.

### 4.9 `every`

`{"every": node, "period": P, "phase": φ, "hold": ...}`.

- The node **fires** in an invocation over `[t, t+Δ]` when some `φ + kP`, `k ≥ 0`, lies in `[t, t+Δ)`.
- Under `step.limits`, firing times are tstops (§5), so a firing time is always the start of a macro step.
- Under a fixed `step.dt`, firing times are not tstops, and `P` and `φ` need not be multiples of `dt` (D50). The node fires on the macro step that contains the firing time, as WRF fires radiation every `radt` and chemistry every `chemdt` on whichever step crosses the period. If one macro step contains several firing times, the node fires once.
- When it fires, the child runs over an interval of length Δₑ ending at `t+Δ`, where Δₑ is the time since the end of the previous firing (or since the start of the run). This is how WRF-Chem advances chemistry by `chemdt` and how PISM's bed deformation accumulates its step (D48). Bindings with `"node"` see Δₑ.
- When it does not fire, it is the identity.
- `hold: "tendency"`: at a firing, the child's tendency is evaluated and then added as a constant to the right-hand side of every leaf among the enclosing node's other descendants, until the next firing. The child itself is not advanced. Its child must then be a single `leaf`.

### 4.10 `map` and `solve`

- `{"map": op}` applies a map operator (§3.2).
- `{"solve": op, "unknowns": [...], "method": {...}, "warm_start": ...}` solves the algebraic operator `op` for the listed unknowns with everything else frozen (architecture §6.3). `method` holds exactly one of `newton` (with `inexact` for Eisenstat–Walker forcing), `picard` (with its `freeze` list, the observeds treated as constants when differentiating the residual for the iteration; this is the only place freezing is written, D25, D51) or `ptc` (pseudo-transient continuation), plus `linsolve`, `linesearch`, `anderson` (memory depth), `converged` (`residual` tolerance, or `rel_change` of a named observed with `tol`), `maxiter` and `recover`. `recover` is a ladder tried in order on failure: under-relaxation, scaling a regularization parameter by a factor up to `max` times, and `revert_warm_start`. `warm_start` is `previous` (the last converged value, the default) or `initial`.

### 4.11 `split_explicit`

WRF's RK3 (Wicker–Skamarock: Δ/3, Δ/2, Δ) on `slow`, with `fast` advanced by `substeps` forward–backward substeps inside each stage, vertically implicit with `epssm` off-centring, and `frozen` tendencies added to every stage (architecture §4.3). Specified in v1, implemented at M7.

## 5. The macro step: `step`

At most one of:
- **`dt`**: a fixed macro step. The macro steps start at $t_0, t_0 + \texttt{dt}, \ldots$, and every tstop must fall on one of them (§8 rule 9), except inside the macro step of a root leaf or `coupled` node that lands on stops itself.
- **`limits`**: Δ is the minimum of the limits evaluated at the macro step's start state, and the distance to the next tstop.
- **Neither**: allowed only when the schedule's root is a single adaptive `leaf` or `coupled` node (D49). That node's accepted steps are the macro steps; it steps adaptively under its own `control` and lands exactly on every tstop. Omitting `step` entirely is the same as `{}`.

Then, for both:
- **tstops** are the times named by `hit`:
  - `forcing_refresh`: the times any forcing provider's two-record bracket changes, which start forcing epochs (architecture §7.5). For a provider that brackets consecutive records these are its record timestamps. This is the only forcing hit source (D46);
  - `output`, `restart`: the output and restart times;
  - `events`: preset and periodic event times of the model.

  The observation times of the adjoint functional (spec/adjoint.md §2.1) are always tstops, whether or not `hit` names their source. Under `limits` or with neither key, `every` firing times (§4.9) are tstops too. The interval is `(t₀, t_f]`, from §1.1, so a boundary at `t_f` is hit (architecture §7.5).
- `hit_multiples` additionally makes every multiple of a duration a tstop.
- `resolution` floors Δ to a multiple of it, and Δ is never less than one `resolution` unless a tstop is closer.
- **Ties.** Each macro step records the reason for its size: the first source achieving the minimum, taking the entries of `limits` in the order listed (a `{"max": ...}` entry counts at its own position in the list), then the next tstop. The reason is deterministic, so two implementations agree on it (PISM bugs.md #11). Adaptive `subcycle` substeps (§4.8) record their reason by the same rule over the subcycle's `limits`, then the time remaining.
- **Fixed boundaries.** Under `dt`, macro step $k$ runs over $[t_0 + k\,\texttt{dt},\ t_0 + (k+1)\,\texttt{dt}]$, each end computed by multiplication, and the last ends at $t_f$. A tstop $\tau$ falls on a boundary $b$ when $|\tau - b| \le 100\,\epsilon(\max(|\tau|, |b|))$, the landing tolerance of spec/methods.md §10; that boundary is then $\tau$ exactly (the earlier, if two fall on it). So with `dt` 0.1 an output at 0.3 falls on $3 \times 0.1 = 0.30000000000000004$, and the state is reported at 0.3. A macro step is the last when, from its start $t_a$, the stop rule of spec/methods.md §10 with $h = \texttt{dt}$ and $t_{\text{stop}} = t_f$ makes it a stop step; the number of macro steps is never computed by dividing. So `dt` 0.1 on $[0, 0.3]$ takes three macro steps, although $(0.3 - 0)/0.1 = 2.9999999999999996$.
- The macro step has no error control of its own (D49). Leaves adapt inside it. A schedule whose root is a single adaptive leaf behaves like an ordinary adaptive integration, because the planner then gives that leaf's accepted steps the role of macro steps.

## 6. Partition

### 6.1 Axes

`axes: {"horizontal": [h_1, h_2], "vertical": v}` names the roles of the model's index sets (D67). Locality analysis (architecture §3.2 pass 2) reads offsets on the vertical axis as column reach and offsets on horizontal axes as stencil reach; `ranks` and `tiles` divide `h_1` by their first entry and `h_2` by their second. Defaults: no vertical axis, and every index set that shapes a state variable, other than the vertical one, horizontal in the model's `index_sets` order. A model without a vertical axis has no column operators. The planner refuses an index set that does not exist, one named twice, or more than two horizontal axes after defaults (spec/plan.md §5.2.1).

### 6.2 Decomposition

- `ranks`: the 2-D grid of MPI ranks over the horizontal axes. Decomposition is horizontal only (D20).
- `tiles`: the 2-D grid of thread tiles within each rank.
- `shards`: shard counts per operator or node id, for pointwise and column leaves. A leaf's own `partition` (`shards`, or `shard_size` in lanes) overrides it. An entry for a `coupled` node, or for an operator a `coupled` node names, is refused (spec/plan.md §7.4).

The planner derives halo widths and shard sizing (architecture §3.2 pass 7). Results are bitwise identical for a fixed decomposition (D10).

## 7. Precision, adjoint, output, restart

### 7.1 `precision`

Precision roles per leaf (architecture §8, D18, D29). Keys are `default`, operator names or node ids (`default` is never either, §2.1); a more specific key overrides `default` role by role.

| Role | Allowed |
|---|---|
| `state`, `rhs` | `f32`, `f64`. Must equal a declared `element_type`; free where the document leaves it unspecified (U14). `rhs` ≥ `state`. |
| `jacobian`, `factors`, `tables` | `f16`, `bf16`, `f32`, `f64`, under any declaration. |

Time, dt, error norms, reductions and adjoint accumulators are always f64 and are not configurable. The planner refuses a low-precision `jacobian` or `factors` with a classical Rosenbrock method (Rodas) unless overridden (architecture §8).

### 7.2 `adjoint`

- `functional`: the reduction (`mean` or `sum`) of one quantity, restricted by `where` (index-set name → 1-based index or inclusive range), `over` the run: `time` (the integral term, spec/adjoint.md §2.1), `final` (the end state), or `saveat` (a point term at each output time after $t_0$). For `time`, `integrate_in` names the leaf, by node id, that integrates the term; the default is the last leaf of the macro-step schedule (D39).
- `wrt`: what to differentiate with respect to. `params:numeric` is every numeric scalar parameter; `params:<pattern>`, `fields:<pattern>` (gridded array parameters) and `forcing:<pattern>` (forcing fields, accumulated per provider record, D5) take a glob over qualified names, where `*` matches any characters including dots; `u0` is the initial state. A parameter bound by `bind_dt` is not differentiated: a pattern such as `params:numeric` or `params:*` that matches one gives it a zero entry, while an entry that names it is refused (§8 rule 11).
- `checkpoint`: `every` (`macro_step` or a count of macro steps), `store` levels, and the `schedule` the planner computes (D34): `all`, `revolve` or `multilevel`, with `snapshots` slots.

The math is in [`adjoint.md`](adjoint.md).

### 7.3 `output` and `restart`

- `output.sinks`: EarthSciIO sinks by `path`, with `variables` and their own `saveat`; `output.saveat` is the default. Output times are tstops, so `saveat` never needs dense output in v1. `saveat` is a duration $d$, and the output times are its multiples $k\,d$ ($k$ an integer, each the binary64 product), counted from model time zero (the model's time origin, §1.1, D74) like `hit_multiples`, that lie in $(t_0, t_f]$; $t_f$ is not added when it is not a multiple (D83). With no `saveat` the output time is $t_f$ alone. The initial state is always reported, and a functional over `saveat` has no term at $t_0$.
- `output.accumulators`: name → observed. The planner adds each as an integrated state (D26).
- `restart`: `path`, cadence `every`, and `from`, a restart store to start from. Restart is bitwise for the same decomposition (architecture §11).

## 8. Rules the planner checks

The schema checks structure. The planner additionally refuses, naming the offending key:

1. a `model` that does not load, or a `solver.external` that does not name this file back (§1);
2. a qualified name that does not resolve in the flattened model, or resolves to the wrong kind (a map target that is not a state variable or parameter, a limit that is not an observed, a `bind_dt` key that is not a parameter, a per-variable `abstol` key or `domain.nonnegative` entry that is not a state), or a `precision` key that is not `default`, an operator name or a node id;
3. an operator name used in the schedule that is neither defined nor an existing default operator;
4. a term selected by two operators, an operator whose selector matches nothing, or a term with no owner that no selector matches while some selector uses `components` (§3.1);
5. an operator that the schedule never advances, or advances twice in one invocation outside `strang`: an operator named by two `leaf` or `coupled` nodes (a `strang` running a child twice is one place). A `frozen` entry counts as advancing for "never advances" but not for "twice" (spec/plan.md §7.1, D84);
6. an `imex` partition that leaves a term in neither part or in both, or an `IMEXEuler` leaf with no `imex`;
7. duplicate node ids, or a path-form id on a node whose path it is not (§2.1);
8. a `hold: "tendency"` `every` whose child is not a single leaf;
9. a tstop that is not $t_0$ plus a multiple of a fixed `step.dt` (`every` firing times are not tstops then, §4.9), unless the schedule's root is a single `leaf` or `coupled` node with `dt` omitted or numeric, which lands on such tstops inside the macro step; a root leaf with `dt` `"step"` or `"accumulated"` takes one step per macro step and cannot;
10. a precision role that contradicts a declared `element_type`, or `rhs` below `state`;
11. a `wrt` entry that matches nothing, or names a parameter bound by `bind_dt`;
12. a `step` with neither `dt` nor `limits` whose schedule root is not a single adaptive `leaf` or `coupled` node (§5);
13. a leaf with `dt` omitted whose method has no error estimate (§4.2);
14. no run interval from any source; one whose end is not after its start; a timestamp naming a day its month does not have; or a timestamp from any source when the model's time-origin field is present but unreadable (§1.1);
15. a leaf with `dt` omitted that has a `bind_dt: "leaf"` parameter but no `control.dt0`, since the first-step algorithm evaluates the right-hand side before any step size exists (spec/methods.md §9);
16. `coupled` with algebraic operators, `split_explicit`, `crossing` parameter-update rules (D22; `unsupported_crossing_update` wins over the general `unsupported_parameter_update`), or any other construct this version specifies but no implementation yet supports, with a refusal that says so.

Each refusal has a code, listed with the rule it implements in [`planner-errors.json`](planner-errors.json) (D79); a refusal reports its code and the run-config pointer, which conformance fixtures check, and a message, which they do not. Where the cause has no run-config key of its own, or the key is not obvious, the pointer is fixed, and spec/planner-errors.json gives it in the code's description; that file is the one list of these pointers. They include `""` for a schema failure and for refusals about the caller's interval, `/model` for refusals about the model alone, `/schedule` for a default operator no node advances, and the precision entry that set the role for precision refusals. At M2, rule 16 also covers algebraic equations, events, every parameter `update` rule, data-loader forcing, `map` operators and nodes, observed step limits and `solve` nodes (spec/plan.md §0.1, D75).
