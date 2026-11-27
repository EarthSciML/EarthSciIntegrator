# Partitioning and executors

**Status:** M3 draft, 2026-11-24. Normative for both implementations (D1, D8). Shared memory only: one rank, its tiles, shards and threads. Ranks, halo exchange between ranks and cross-rank lane migration are M5's and M5b's (architecture §7, §14). Background: architecture §7, D6, D10, D17, D20; this chapter's decisions are D96–D104.

A leaf (spec/run-config.md §4.2) advances the sum of its operators over its interval. This chapter says how a runtime divides that work among threads without changing what is computed, and the one case, `dt_scope: shard`, in which the division is part of the numerics. It defines `dt_scope` (§2), lane pools (§3), executors (§4), the order of every reduction (§5), shard composition and re-sharding (§6), forcing at scale (§7), tiles for stencil leaves (§8), the invariance a runtime must show and the fixtures that check it (§9), the run-config and plan keys (§10), and what the tape records (§11).

## 1. Terms

| Term | Meaning |
|---|---|
| **cell** | Of a pointwise leaf whose storage is `block_diagonal` (spec/operator-interface.md §8.3): one of its `ncells` cells, the `n` DOFs $\{i \cdot \text{ncells} + c : i = 0..n-1\}$ of the leaf's state vector for cell $c$ (0-based, column-major over the leaf's state shape, the **canonical cell order**). Species $i$ follow the leaf's variable order. |
| **shard** | A set of cells of one leaf, listed in increasing cell order. The shards of a leaf partition its cells. |
| **lane** | One slot of a batch: the state, controller state and inputs of one cell, laid out so that one operation runs across all lanes of the batch (struct of arrays, spec/operator-interface.md §2, §4). |
| **lane pool** | A batch of $L$ lanes (the **lane capacity**) that integrates the cells of one shard under `dt_scope: cell`, refilling lanes from the shard's work queue as cells finish (§3.2). |
| **executor** | What runs a shard: a thread of the runtime's pool at M3 (§4). |
| **tile** | A box of a rank's horizontal cells, with halos, that a stencil leaf evaluates on its own (§8). |
| **invocation** | One call of a leaf over one interval $[t_a, t_b]$ (spec/tape.md §1). |

## 2. `dt_scope`

`control.dt_scope` (spec/run-config.md §4.2) says which DOFs share one step-size controller.

### 2.1 Which leaves may use which scope

| Leaf | Allowed | Default (`"auto"` or omitted) |
|---|---|---|
| `leaf`, every operator pointwise, storage `block_diagonal` | `global`, `shard`, `cell` | `cell` (D17, D97) |
| `leaf`, every operator pointwise, storage `dense` | `global` | `global` |
| `leaf` whose class is `column` | `global` (`cell` and `shard` are specified in §2.5 and refused until column lane pools are implemented, `unsupported_dt_scope`) | `global` |
| `leaf` whose class is `stencil` or `global`; any `coupled` node | `global` | `global` |

A leaf's **class** is the coarsest locality of its operators (spec/plan.md §5.5.3). The planner refuses `cell` or `shard` where the table does not allow it: on a stencil or global leaf, on a pointwise leaf with `dense` storage, and on a `coupled` node, `dt_scope_not_local` at the node's `P/control/dt_scope` (spec/run-config.md §8 rule 19); on a column leaf, `unsupported_dt_scope` (rule 16). It also refuses `cell` or `shard` on a leaf with a `bind_dt: "leaf"` parameter, `dt_scope_bind_dt` at the same pointer (rule 19), since one scalar parameter cannot hold the step sizes of several cells at once. A `dt` given on the leaf (fixed, `"step"` or `"accumulated"`) leaves no controller, and `cell` and `shard` then differ from `global` only in their return codes and statistics (§2.3); they are allowed.

### 2.2 `global`

One controller for the leaf, exactly as spec/methods.md specifies and as at M2. The error norm's sum runs over the leaf's DOFs in ascending order, sequentially (spec/methods.md §7). A runtime MAY compute the stages, the Jacobian, the factorization and the per-DOF ratios $e_j/\mathrm{sc}_j$ in parallel, over shards, tiles or lanes, provided each value equals the value the whole-leaf computation gives bit for bit (§3.1, §8); it MUST then sum the squares in the one sequence above (D97). So results under `global` do not depend on the shard count, the tile count, the thread count or the lane capacity, and equal M2's.

### 2.3 `cell`

Each cell is integrated as its own initial-value problem over the invocation's interval, with its own controller:

- **State.** The cell's $n$ DOFs, in species order. Every other DOF of the model, the other cells of the leaf included, is a constant of the cell's integration. For a pointwise operator the rows of a cell depend on no other cell's state, so the constants other cells hold never enter.
- **Operators.** The leaf's operators, restricted to the cell's rows: `rhs`, `jac` (one $n \times n$ block), `ft` and the other calls of spec/operator-interface.md §3 at the cell. A runtime evaluates them through EarthSciAST's lane programs (§3.3) or by any other means that gives the same values bit for bit as the whole-domain program at that cell.
- **Error norm, first step, control, stops.** spec/methods.md §7–§10 with the leaf's DOFs replaced by the cell's: $N = n$, the sum over the cell's species in species order (which is ascending DOF index). Every cell has the same stops: the interval end and the tstops and output times inside it. The first step (spec/methods.md §9) is computed per cell, with norms over the cell (spec/methods.md §13 assumption 8). The controller's persistent state (spec/methods.md §8.6) is kept per cell and persists across invocations and macro steps.
- **Domain, first same as last, `maxiters`.** Checked, reused and counted per cell (spec/methods.md §3.3, §11, §12).
- **Return code.** A cell's code is spec/methods.md §12's for its integration. The invocation's code is the first non-success in canonical cell order, naming the cell, and success when every cell succeeds.
- **Statistics.** The leaf's accepted, rejected and right-hand-side counts are the sums over its cells of each cell's counts, where a cell's right-hand-side count is the number of evaluations its integration makes as if it ran alone. They are integers, so the sums are exact.

The result of each cell (its state at the interval end and at every interior stop, its controller state, its counts and its return code) depends only on the cell's own inputs: its state and controller state at the start of the invocation, the parameters, the constants it reads, the forcing sets (§7), the interval and the stops. It MUST NOT depend on the shard the cell is in, the other cells of its shard or batch, its lane, the lane capacity, the thread that ran it or the order of work (D97). §3 says how lane pools keep this.

### 2.4 `shard`

Each shard is integrated as its own initial-value problem with its own controller: §2.3 with "cell" replaced by "shard", the shard's state being the DOFs of its cells in ascending leaf-vector order (species-major: species 0 of every cell of the shard in cell order, then species 1, …). The error norm sums over those DOFs in that order, $N = n \cdot |\text{shard}|$. The invocation's return code is the first non-success in shard order, the shards in order of their smallest cell. Results depend on the shard composition (§6) and on nothing else of §2.3's list: not on the thread count, the lane capacity or the order of work.

### 2.5 Column leaves (specified, not yet implemented)

For a leaf whose class is `column`, a "cell" of §2.3 is a **column**: every DOF of the leaf in one horizontal position, ordered species-major within the column with the vertical index innermost. `cell` and `shard` scopes are §2.3 and §2.4 with columns in place of cells, columns in canonical order of their horizontal indices. Until a release of both runtimes implements column lane pools, the planner refuses them, `unsupported_dt_scope`, at the node's `P/control/dt_scope`.

### 2.6 A root leaf with no macro-step control

When the schedule's root is a single adaptive `leaf` and `step` has neither `dt` nor `limits` (spec/run-config.md §5), the macro steps are that leaf's accepted steps (D49). Under `cell` or `shard` there is no single step sequence, so the macro steps are instead the intervals between consecutive tstops of $(t_0, t_f]$, starting at $t_0$; each is one invocation of the leaf.

## 3. Lane pools

### 3.1 What a runtime must preserve

Under `cell` the result of a cell is a function of its own inputs (§2.3). Under `global` and `shard` the value of every stage, Jacobian block, factorization and ratio at a cell is the value the whole-leaf computation gives. A runtime keeps both by these rules:

1. **No arithmetic across lanes.** No operation reads two lanes of a batch, except to copy, and no lane's arithmetic depends on another lane's values or on which lanes are active. Reductions across cells (§5) are taken outside the batch, in their specified order.
2. **The same instructions in every lane.** Every operation over a batch runs at the batch's full capacity. A remainder smaller than the capacity is **padded**: the inactive lanes hold a copy of an active lane's inputs (or any finite valid inputs) and their outputs are discarded. There is **no scalar tail**: a runtime never finishes the last lanes of a batch with different code, which could round differently (a scalar `exp` beside a vector one, say).
3. **Fixed contraction.** Whether a multiply and an add are fused is fixed by the code, never chosen per lane or per call: neither implementation contracts into fused multiply-adds unless the spec says so (spec/methods.md §3.1, §7), and neither builds lane kernels with options that let the compiler contract (Julia `@fastmath`, `muladd`; Rust `mul_add` or fast-math intrinsics).
4. **Kernels are lane-exact.** The right-hand-side, Jacobian and time-derivative programs evaluated over lanes give each lane the whole-domain program's value at its cell, bit for bit. EarthSciAST's lane programs promise this for every lane, every capacity and every padding (U3, API_SPEC §5.8); a runtime relies on it and its tests check it (§9.3).

Rules 1–4 make the result of every lane independent of its batch, so lane capacity, padding and grouping never change a value.

### 3.2 The pool (informative, with rule 2 normative)

Under `cell` an executor integrates a shard's cells with a pool of $L$ lanes. Its work queue holds the shard's cells in the order §6.3 gives. It fills every free lane from the queue, gathering the cell's state, its controller state and its inputs into the lane. Then it repeats a **pool step**: every active lane makes one step attempt of its own size $h_\ell$ from its own time $t_\ell$, the method's stages computed across the batch; each lane accepts or rejects by its own error and moves its own time. A lane whose time reaches the interval end, or whose integration ends with a non-success code, **retires**: its state and controller state are scattered back to the cell and its records kept, and the lane is refilled from the queue. **Compaction** moves active lanes together when the pool runs low, so that a pool step runs over as few batches as hold the active lanes. When the queue is empty and fewer than $L$ lanes are active, the batch is padded (rule 2). The pool ends when every cell of the shard has retired.

Under `shard` and `global` every lane of a shard moves together (one $h$, one $t$), and the lanes are simply the shard's cells in batches of $L$.

### 3.3 The time of a lane (D98)

A lane is evaluated at its own time. EarthSciAST's lane programs (Rust `RhsProgram::lanes`, Julia `lanes`; U3, EarthSciAST 0.5.0) take one scalar $t$ per call. Until they take a time per lane (U19), a runtime evaluates in one call only the lanes whose stage times are equal bit for bit, and the other lanes of the batch as padding of that call; it calls once per distinct time. That is always correct, since a lane's value depends only on its own inputs (rule 4). With U19 (EarthSciAST#781: Rust `LaneProgram::eval_times_into`, Julia `rhs!` with a vector of times) a batch is one call whatever its times; EarthSciAST.jl serves such a call by groups of equal time for now. Which lanes share a call never changes a value.

The same holds for forcing: a lane's call takes the forcing set its own step selects (spec/operator-interface.md §8.4.1, §7 below), so lanes of one batch taking different sets are evaluated in separate calls until a lane program takes a set per lane.

### 3.4 Linear algebra per lane

`block_lu` and `sparse_block_lu` already factor and solve every lane of the `block_diagonal` storage with the same instructions (spec/methods.md §4.2; spec/operator-interface.md §4), with no pivot choice or operation shared between lanes. Under `cell` each lane factors its own $W = I/(h_\ell \gamma) - J_\ell$ with its own $h_\ell$. A singular $W$ in one lane rejects that lane's step only.

## 4. Executors (D99)

An executor runs one unit of work: under `cell`, the lane pool of one shard over the invocation; under `shard`, the integration of one shard; under `global`, the evaluation of one shard's or tile's part of one stage, Jacobian or factorization. It is **stateless** (architecture §7.3):

- **Inputs:** the cells' states and controller states (under `cell` and `shard`), the leaf's constants it reads, the parameters, the resident forcing sets (§7), the interval and its stops.
- **Outputs:** the new states and controller states; per cell (under `cell`) or per shard, the accepted and rejected counts, the right-hand-side count, the return code and the tape records of §11; under `shard`, the per-cell error sums of §6.2; under `global`, the per-DOF values of its part. Per-shard parameter gradients are added at M4.
- **Nothing persists in an executor between calls.** Workspaces are pooled and reused, but a workspace's contents at the start of a call never affect its result.

At M3 the executors are the threads of one pool in the runtime, of a size the caller chooses: a run-time option of the runtime's API and command line (Julia keyword `threads`, Rust `RunOptions::threads`), never a run-config or plan key, since no result depends on it (D99). Shards are assigned to threads dynamically: any thread may run any shard, in any order, with work stealing; assignment may use wall-clock time.

Every operator is reentrant (spec/operator-interface.md §1, U4): one operator object, and one compiled program, serves every thread at once, each with its own workspace.

## 5. Ordered reductions (D10)

Every floating-point sum over more than one cell has one order, fixed by the spec, whatever the executors:

| Reduction | Order |
|---|---|
| `global` error norm (spec/methods.md §7) and first-step norms (§9) | ascending DOF index over the leaf, sequential |
| `shard` error norm | ascending DOF index over the shard, sequential |
| `cell` error norm | ascending species index over the cell |
| the functional's integral and other sums the adjoint chapter defines | as spec/adjoint.md gives them, over cells in canonical order |

Integer counts, and the minimum or maximum of floating-point values, are exact under any order. Across ranks (M5) partial sums are gathered in rank order and reduced at the root (architecture §7.4); within a rank the orders above hold.

## 6. Shards

### 6.1 The initial composition

A leaf with $S$ shards (the plan's `partition.shards`, spec/plan.md §7.4) and $C$ cells has shard $s = 0, \ldots, S-1$ holding the cells $\lfloor sC/S \rfloor \le c < \lfloor (s+1)C/S \rfloor$ of the canonical cell order. With $S > C$ the empty shards are dropped. A column leaf's shards hold whole columns the same way, over the canonical column order (§2.5). A stencil or global leaf has no shards; a runtime may still divide its work by tiles (§8).

### 6.2 Re-sharding (D100)

What may change, on which signal and when, depends on whether the change can alter results (architecture §7.6):

| What changes | Under | Allowed signal | When |
|---|---|---|---|
| Which executor runs which shard | every scope | anything, wall-clock time and work stealing included | any time |
| Shard composition and the order of work within a shard | `cell`, `global` | anything | between invocations |
| Shard composition | `shard` | §6.2's deterministic signal only | at a macro-step boundary named by `partition.reshard` |

Under `shard`, a leaf whose plan gives `reshard: {"every": k}` re-composes its shards before the first invocation of every macro step whose index is a positive multiple of $k$ (macro steps counted from 0; never before the first). For each cell $c$ the signal is

```math
R_c = \sum_{\text{attempts}} \sum_{j \in c} \Big(\frac{e_j}{\mathrm{sc}_j}\Big)^2 ,
```

summed over every step attempt (accepted or rejected) of the shard containing $c$ during the leaf's invocations in the previous macro step, the inner sum over the cell's DOFs in species order and the outer in attempt order, in f64. It is formed from the ratios the shard's error norm already computes (spec/methods.md §7): those of the norm whose value the controller accepts or rejects the attempt on, and of no other norm the method computes (not a Newton increment's, spec/methods.md §5.2). A method whose error estimate is the larger of two norms adds the ratios of the norm that gave it (the first on a tie). An attempt that computes no error estimate (a singular iteration matrix, a Newton iteration that fails, or a first step whose estimate is taken without a norm) adds nothing, and neither does an attempt whose error estimate is not finite, so $R_c$ is always finite. $R_c$ is part of the leaf's persistent state, kept with its units' controllers: it is saved in a checkpoint and restored from it (§11), so a resumed run re-shards exactly as the uninterrupted run does, and it is never shared between runs. The cells are sorted by $R_c$ descending, ties (equal bit patterns) by cell index ascending, and the sorted list is cut into $S$ contiguous shards as in §6.1; each new shard lists its cells in increasing cell order. The new shards' controllers start from the state of the old shard that held the new shard's first cell (its smallest), proposal included. `reshard: "none"` (the default) keeps the composition of §6.1 for the whole run. $R_c$ is deterministic within a language; across languages it agrees only to tolerance, so the compositions, and with them the trajectories, may differ to tolerance (D10).

### 6.3 Order of work

Under `cell` and `global` the order in which a shard's cells enter the pool and the order in which shards are handed to threads change no result. The recommended order, for load balance, is decreasing cost in the previous invocation of the leaf (a cell's step attempts), ties by cell index; a cell with no previous invocation counts as cost 0.

## 7. Forcing at scale (D101)

The epochs and the selection of a call's buffers are spec/operator-interface.md §8.4.1 and §8.4.3, unchanged. At scale:

- **One set per epoch, shared.** A rank holds one forcing set per resident epoch, read-only and shared by every thread (Julia `Vector{Float64}` buffers, Rust `Arc<[f64]>`); a thread never copies a whole buffer.
- **Resident before the invocation.** Before an invocation of a leaf over $[t_a, t_b]$ starts, the runtime makes resident every set a call of the invocation can take: the set of the epoch containing $t_a$, and, when $t_b$ is an epoch boundary $B_{k+1}$, the set of epoch $k+1$, which the right-end stages read for two-record parameters. Sampling, decoding and conversion (spec/operator-interface.md §8.4.2) happen only between invocations, on the coordinating thread or on background prefetch workers (U11), never inside an executor.
- **Per lane.** Each lane takes the set its own step selects (§3.3). Lanes of one batch may take different sets in the same pool step (one ending its step on $B_{k+1}$, another mid-step).
- **Forcing slices.** A lane may hold its **slice**: the values its programs read from a set at its own cell, gathered once (EarthSciAST's pre-gathered lane inputs, `gather_lane_inputs` in Rust) and reused while the lane holds that cell and that set. A slice gives the values of an evaluation with the whole buffers bit for bit, so whether, when and how often slices are gathered changes nothing. A slice is never reused for a different set: a new epoch's set, or the right-end set, needs its own.
- A set that no resident invocation can take any more may be dropped.

## 8. Tiles and halos for stencil leaves (D103)

A rank's horizontal cells are divided into `partition.tiles` $= [T_1, T_2]$ tiles. Along horizontal axis $a$ of size $N_a$, tile index $q = 0, \ldots, T_a - 1$ owns the cells $\lfloor q N_a / T_a \rfloor \le i < \lfloor (q+1) N_a / T_a \rfloor$ (0-based), and every vertical level of them. A staggered face belongs to the cell above it on its axis, and the tile owning the last cell owns the last face as well (architecture §7.1).

A stencil leaf evaluates its operators tile by tile: each tile's rows by the operator's tile program (Rust `RhsProgram::tile`, Julia the equivalent; U3), which reads the tile's halo, of the plan's `partition.halo` width, from the stage's input state. In shared memory that state is one array: the **halo exchange** is a barrier, so that every stage's input is complete before any tile reads it. Tiles are handed to threads like shards (§4). The U3 contract makes a tile's rows bitwise equal to the whole-domain program's; with the norm order of §5, a stencil leaf's results do not depend on the tile count.

A column leaf is divided into shards of whole columns (§6.1), which need no halo. A pointwise leaf ignores tiles.

## 9. Invariance and conformance

### 9.1 What each implementation must show

Within one implementation, on one machine:

| Scope | Bitwise invariant to |
|---|---|
| `cell` | shard count, shard composition, thread count, lane capacity, order of work, U19's presence |
| `global` | shard count, tile count, thread count, lane capacity, order of work |
| `shard` | thread count, lane capacity, order of work (not the composition) |

"Bitwise" covers the state at every output time, the per-cell (or per-shard) accepted and rejected counts, and the return code. Across implementations results agree to tolerance (D10; the `cross_language` check).

### 9.2 The fixture model

`reseact_mini` ([`../tests/conformance/toys/reseact_mini.esm`](../tests/conformance/toys/reseact_mini.esm), written with its data by [`make_data.py`](../tests/conformance/toys/make_data.py); spec/toy-problems.md §5): reseact's shape at toy size, 6 × 4 × 5 cells (`lon`, `lat`, `lev`), NO2/NO/O3 chemistry whose photolysis varies with longitude and level and whose thermal rate reads hourly two-record data across a daily file boundary, periodic upwind advection along `lon` and a column mean along `lev`. Its cells differ in stiffness through the run, so their step sequences differ under `cell`.

### 9.3 Checks

Runtime cases are in [`../tests/conformance/m3/cases.json`](../tests/conformance/m3/cases.json), generated by [`make_cases.py`](../tests/conformance/m3/make_cases.py); spec/toy-problems.md §5.2 lists them. Two checks join spec/toy-problems.md §2.1:

| Check | Meaning |
|---|---|
| `invariance` | Run the case once per entry of `variants`, each an override of the leaf partitions (`shards`, `lanes`) and of the run-time thread count, and compare every run with the first: the output states, the per-cell (`cell`) or per-shard (`shard`) accepted and rejected counts and the return codes must be equal bit for bit. Each implementation on its own. |
| `scope_agreement` | The case's output states agree with the same implementation's result for the case named by `against`, which differs only in `dt_scope`, to `rtol` relative to each component's largest magnitude, as `reference` compares. The two integrate the same problem with different controllers, so they differ by about the integration tolerance. |

Each runtime's own tests also check rule 4 of §3.1 on `reseact_mini`'s planned operators: for every cell, the lane-program value of `rhs`, `jac` and `ft` at capacities 1, 3 and 8, with the cell in every lane position and with padding, equals the whole-domain value bit for bit.

### 9.4 Planner fixtures

Planner fixture `reseact_mini` (spec/plan.md §9) covers the plan keys of §10 and the time-derivative document (spec/plan.md §6.4). Refusal fixtures: `dt-scope-stencil` (`dt_scope: cell` on the transport leaf; `dt_scope_not_local`), `dt-scope-bind-dt` (`cell` on a leaf with a `bind_dt: "leaf"` parameter; `dt_scope_bind_dt`), `dt-scope-column` (`cell` on a column leaf; `unsupported_dt_scope`). Goldens are written by the Julia planner (spec/plan.md §9).

### 9.5 The M3 gate's agreement when step sequences differ (D108)

The M3 gate (D96) compares the Julia and Rust runs of reseact at the gate's tolerances: transport `SSPRK43` with `reltol` $10^{-4}$, `abstol` $10^{-6}$; chemistry `Rosenbrock23` with `reltol` $10^{-4}$, `abstol` $10^{-9}$. Where the two runs take the same steps they must agree to $10^{-9}$ relative (the `cross_language` check). Where their step sequences differ, which a rounding difference in an error estimate is enough to cause, they are judged against a **truth run** instead, not against each other:

- **Truth.** Julia on the same plan's model, schedule and step, with every leaf's tolerances $10^{3}$ times tighter than the gate's: chemistry `reltol` $10^{-7}$, `abstol` $10^{-12}$; transport `reltol` $10^{-7}$, `abstol` $10^{-9}$.
- **Error of a run.** For each species, at every output time and cell, the difference $|x - x^st|$ between the run's value $x$ and the truth's $x^st$, measured two ways: relative to the value, $|x - x^st| / |x^st|$, and in the leaf's weighted error norm at the gate's tolerances, $|x - x^st| / (\mathrm{abstol} + \mathrm{reltol}\,\max(|x|, |x^st|))$. A species' error is the worst over output times and cells.
- **Pass.** For every species, Rust's error in the weighted norm is at most twice Julia's (both runs at the gate's tolerances). Both measures are reported per species for both runtimes.

The criterion was chosen by the user (2026-11-27) after the like-for-like comparison showed the two runtimes' step sequences diverging from a $1.6\times10^{-13}$ relative difference in one step size, with OH at a sunrise cell differing by 2.5 % of its value while every other species stayed within $10^{-4}$ of its largest value.

## 10. Run-config and plan keys (D104)

### 10.1 Run config

A `leaf` node's `partition` (spec/run-config.md §6.2) takes:

| Key | Value | Meaning |
|---|---|---|
| `shards` | integer ≥ 1, or `"auto"` | shard count (M2) |
| `shard_size` | integer ≥ 1 | cells per shard, turned into a count (M2) |
| `lanes` | integer ≥ 1 | the lane capacity $L$; omitted, the runtime chooses. Changes no result (§3.1) |
| `reshard` | `"none"` or `{"every": k}`, $k \ge 1$ | §6.2; allowed only with `dt_scope: shard` (else `reshard_without_shard_scope`, rule 19) |

`partition.shards` at the top level keeps its M2 meaning. Threads are not a run-config key (§4).

### 10.2 Plan

`plan.json`'s `partition` (spec/plan.md §7.4) gains, beside `shards`:

- `lanes`: by node id, the `lanes` a pointwise or column leaf gives; absent when no leaf gives one (D73: a default computed at run time is not written).
- `reshard`: by node id, for every leaf whose resolved `dt_scope` is `shard`: `"none"` or `{"every": k}`. Absent when no leaf has `shard` scope.

The resolved `dt_scope` is in each leaf's `control` (spec/plan.md §7.1). The plan schema ([`schemas/plan.schema.json`](schemas/plan.schema.json)) and the run-config schema ([`schemas/run-config.schema.json`](schemas/run-config.schema.json)) carry these keys.

## 11. Tape and checkpoints

spec/tape.md §2 records, per leaf invocation:

- under `global`: as at M2, the accepted steps `(t_n, h_n)`;
- under `shard`: per shard in shard order, its cells (when the composition changed since the shard's last record, or on its first), its accepted steps and its counts;
- under `cell`: per cell, in canonical cell order, its accepted and rejected counts, not its steps (architecture §7.6). A replay re-runs the invocation with the controllers on, from the checkpointed controller states, and checks the counts: under D10 it reproduces every step bitwise.

In the serialized form (spec/tape.md §5) a `cell` leaf's record carries `"cells": {"accepted": [...], "rejected": [...]}` (arrays in canonical cell order) and an empty `"steps": []`, and a `shard` leaf's carries `"shards": [{"cells": [...], "steps": [...], "accepted": a, "rejected": r}]` (`cells` only when §11 says so). A checkpoint holds each cell's (or shard's) controller state of spec/methods.md §8.6, and, under `shard`, the current composition and the re-sharding signal $R_c$ summed so far in the macro step in progress (§6.2).

## 12. Open questions

Recorded with a recommended answer in the M3 notes and the PR that adds this chapter; none changes the design above:

1. Whether `scope_agreement`'s tolerance should be a multiple of `reltol` (recommended: $10\cdot$`reltol`).
2. Whether the default lane capacity should be fixed by the spec for reproducible performance (recommended: no; it changes no result).
