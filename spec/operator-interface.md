# The operator interface

**Status:** M1 draft, 2026-10-23; §8 (operators a plan supplies) added for M2, 2026-10-25; §8.4 (forcing, M2) added 2026-11-02 (D87). Normative for both implementations (D1, D8).

A leaf (spec/run-config.md §4.2) advances the sum of its operators. This chapter defines what a leaf calls on each operator: the right-hand side, its derivatives, and the structure the linear solvers rely on. From M2 the planner builds operators from the flattened model, with kernels compiled by EarthSciAST and band models from EarthSciASTDiff (architecture §2, §3.2 pass 6), and the runtime constructs them from the plan's derived documents (§8). At M1 the toy problems of [`toy-problems.md`](toy-problems.md) implement the interface by hand (D53), and they stay as a second supplier. Nothing in the methods, controllers, schedule interpreter or adjoint may depend on which of the two supplied an operator: both satisfy this chapter's calls with the same meaning, and the M2 gate checks that they give the same results (spec/toy-problems.md §4).

## 1. What an operator is

An operator $k$ of a leaf contributes an additive term $f_k$ to the leaf's right-hand side:

```math
f(u, p, t) = \sum_k f_k(u, p, t).
```

- $u$ is the leaf's **state vector**: every degree of freedom (DOF) the leaf's operators write (architecture §3.2 pass 3), of length $N$ and element type $T$ (`f32` or `f64`, spec/run-config.md §7.1).
- $p$ is the **parameter vector**: the active scalar parameters, of length $P$, always `f64`. Data-fed parameters are not in $p$: they are the forcing set $F$ below (M2). Array-valued (gridded) parameters as gradient targets join at M4 (U7); §6 reserves their place.
- $t$ is time in seconds, always `f64` (D13).
- $F$ is the **forcing set**: the values of the data-fed parameters the operator reads (spec/plan.md §4.1), one `f64` buffer per parameter, constant over a step (§8.4). It joins the calls at M2 (§8.4); an operator that reads no data-fed parameter takes an empty set, and every call below then means exactly what it means without $F$.
- State an operator reads but no operator of the leaf writes is a constant of the leaf over its interval. The schedule interpreter supplies it; at M1 no toy problem has any.

An operator never stores state between calls. Every call is a pure function of its arguments, so one operator object may be called from several threads at once (U4). Workspace an operator needs is passed in by the caller.

## 2. Layout

The leaf's state is one flat vector. Its order is part of the operator set's declaration:

- **`dense`**: no structure is promised; DOF order is the operator set's own.
- **`block_diagonal`** with `ncells` cells of `n` DOFs each: every operator's Jacobian is zero between different cells. DOFs are **species-major**: DOF $i$ of cell $c$ (both 0-based) is at index $i \cdot \text{ncells} + c$. This is the struct-of-arrays layout that batched kernels vectorize across (architecture §6.1, the `LaneBlocks` layout of U10), and reseact's.

The leaf's structure is the coarsest that all its operators share: a leaf whose operators are all `block_diagonal` with the same `ncells` and `n` is `block_diagonal`; any other mix is `dense`. Operators a plan supplies declare one of these two at M2 (§8.3); banded and stencil storage arrive with their solvers (M3, M4b).

## 3. Calls

Every call **accumulates** into its output: `out ← out + (value)`. The leaf zeroes an output once, then calls each operator in its declared order. Accumulation in a fixed operator order makes the sum bitwise reproducible (D10).

All arguments are as in §1. $v$ and $w$ are vectors of length $N$, element type $T$; $\mu$ is a list of pairs $(\mu_i, w_i)$; outputs have the shape of the cotangent or tangent they hold. A $u$-shaped output has element type $T$; a $p$-shaped output is `f64`.

| Call | Accumulates | Required |
|---|---|---|
| `rhs(out, u, p, t)` | $f_k(u, p, t)$ | always |
| `jvp_u(out, u, p, t, v)` | $J_k(u)\, v$ | always |
| `vjp_u(out, u, p, t, w)` | $J_k(u)^\top w$ | always |
| `vjp_p(out, u, p, t, w)` | $f_{k,p}(u)^\top w$ | when an adjoint is taken with respect to $p$ |
| `jac(out, u, p, t)` | $J_k(u)$ into the leaf's matrix storage (§4) | for Rosenbrock, SDIRK, IMEX and θ-method leaves with a direct `linsolve`, from the operators whose Jacobian the method forms: for IMEX Euler the implicit part only, for the θ-method at θ = 0 none |
| `ft(out, u, p, t)` | $\partial f_k/\partial t$ | for Rosenbrock leaves whose operator depends on $t$ explicitly |
| `hess_u(out, u, p, t, μ)` | $\nabla_u \sum_i \mu_i^\top J_k(u)\, w_i$ | for Rosenbrock adjoints, and for adjoints of IMEX Euler with `implicit_solve: "linearized"` from its implicit part only (spec/adjoint.md §4.4) |
| `hess_p(out, u, p, t, μ)` | $\nabla_p \sum_i \mu_i^\top J_k(u)\, w_i$ | likewise, with respect to $p$ |
| `ft_vjp_u(out, u, p, t, ν)` | $\nabla_u\, (\nu^\top \partial f_k/\partial t)$ | for Rosenbrock adjoints of $t$-dependent operators |
| `ft_vjp_p(out, u, p, t, ν)` | $\nabla_p\, (\nu^\top \partial f_k/\partial t)$ | likewise, with respect to $p$ |

These are the primitives of spec/adjoint.md §3, per operator. The leaf's primitive is the sum over its operators.

Each operator also declares:
- `depends_on_t`: whether $f_k$ depends on $t$ explicitly. When false, `ft` and `ft_vjp_*` are zero and need not be implemented.
- `linear_in_u`: whether $J_k$ is independent of $u$. When true, `hess_u` is zero, and IMEX Euler's Newton iteration converges its implicit part in one linear solve (spec/adjoint.md §4.4). The other second-order calls need not vanish: `hess_p` is zero only when $J_k$ also does not depend on $p$, and `ft_vjp_u` only when it does not depend on $t$. For $f_k = -k\,u$, $J_k = -k$ and `hess_p` is $-\sum_i \mu_i^\top w_i$.
- Its structure (§2).
- `writes`: the indices of the state vector it writes, those where $f_k$ can be nonzero. The leaf's DOFs, over which the error norm and first step run (spec/methods.md §7, §9), are the union over its operators, in ascending order. At M1 operators take the full state vector; from M2 the plan's `writes` lists give the same set (spec/plan.md §3). *Deviation:* OrdinaryDiffEq norms over the whole state, so a leaf whose operators leave some entries unwritten divides by a smaller $N$ than an OrdinaryDiffEq solve of the same equations, and its step sizes can differ from that solve's.

A leaf whose method or adjoint needs a call its operators do not provide is refused when the leaf is built, naming the operator and the call.

## 4. Matrix storage

`jac` accumulates into storage the leaf's `linsolve` owns:
- **`block_diagonal`:** `ncells` dense `n × n` blocks, stored lane-innermost: entry $(i, j)$ of cell $c$ is at offset $(i \cdot n + j) \cdot \text{ncells} + c$. Batched LU (`block_lu`, architecture §6.1) operates on this layout, one operation across all lanes at a time.
- **`dense`:** an `N × N` column-major matrix. This is for toy problems and tests only; architecture §6.1 never assembles a dense W for a real model.

The leaf, not the operator, forms $W$ from $J$ and factors it (spec/methods.md).

## 5. Bindings

The names above are the spec's. Each binding spells them idiomatically and documents the mapping:
- **Julia:** methods on an operator type, `rhs!(op, out, u, p, t)` and so on, generic over `T`. At M1 the leaf's sum of operators is wrapped as a `SciMLBase.ODEFunction` (D33) so OrdinaryDiffEq can integrate the same problem as an oracle, with `jac` and `vjp` filled from the operator calls.
- **Rust:** a trait `Operator<T>` with `&self` methods taking slices, `Send + Sync` (U4).

## 6. Reserved for later milestones

- **Forcing derivatives** (M4): `vjp_F`, `hess_F` and `ft_vjp_F` (U7, U8). The forcing-set argument $F$ itself is M2's (§1, §8.4).
- **Array parameters** (M4): gridded parameters as additional $p$-shaped arguments with their own `vjp_*` (D5, U7).
- **Range restriction** (M3): calls over a tile with halos or a list of lanes rather than the whole state (U3).
- **Maps, limits, conditions and residuals** (M1 maps; M4b the rest): §7 defines maps; the others follow with U6.

## 7. Map operators

A map operator (spec/run-config.md §3.2) provides:
- `apply(out, u, p, t, Δ)`: writes $M(u, p, t, \Delta)$ into `out` (overwrites, unlike §3), and records into a caller-provided mask any discrete decision it takes;
- `map_vjp(out_u, out_p, u, p, t, Δ, mask, w)`: accumulates $(\partial M/\partial u)^\top w$ and $(\partial M/\partial p)^\top w$ at the pre-map state with the recorded decisions held fixed (spec/adjoint.md §5.7);
- `map_jvp(out, u, p, t, Δ, mask, v_u, v_p)`: $(\partial M/\partial u)\, v_u + (\partial M/\partial p)\, v_p$, for tangent checks;
- `apply_frozen(out, u, p, t, Δ, mask)`: like `apply`, but with the decisions in `mask` imposed rather than taken, so finite differences on the frozen tape (spec/adjoint.md §7.2) hold them fixed.

$\Delta$ is the value `bind_dt` gives the map (spec/run-config.md §4.4).

## 8. Operators supplied by a plan (M2)

A runtime that executes a plan (spec/plan.md) builds each operator the schedule names from the plan's entry `operators.<name>` and the operator's two derived documents. The result implements §1–§4 exactly as a hand-written operator does.

### 8.1 Building the kernels

- **Right-hand side.** Load `derived/<name>.rhs.esm` with EarthSciAST and build its right-hand side as the public reentrant program of U4 (EarthSciAST 0.5.0), the same way in both languages: `rhs_program(esm_problem(path, tspan))` with the native compiler, which returns an `RhsProgram`. `esm_problem` flattens the document, so both languages build from the flattened document, whose names and parameter order §8.2 refers to. `tspan` is the run's interval; it does not affect the program, which takes $t$ as an argument. Derived documents carry no data and no data location: a data-fed parameter is declared with a stub source (spec/plan.md §6.3), bound at construction to placeholder values that no evaluation reads, and supplied at every call through the forcing set (§8.4). Runtimes do not use the problem's `f!`, which is not reentrant (architecture §7.3), nor `build_evaluator`, deprecated since EarthSciAST 0.4.0 (spec/plan.md §10 item 3). A runtime calls one `RhsProgram` from as many threads as it runs shards on.
- **Jacobian.** Load and build `derived/<name>.jacobian.esm` the same way. Its right-hand side at $(u, p, t)$ holds band field `J_k` in the slots of the state `JacobianEval.J_k` (spec/plan.md §6.2).
- A build failure refuses the leaf, naming the operator and the document.

### 8.2 Layout and arguments

- **State.** The operator acts on the whole model state in the plan's **canonical order** (spec/plan.md §4): state names sorted by code point, each array's cells column-major. Every derived document of a plan declares the same states, so one permutation from the binding's own build order (its name map) to canonical order serves all of them; the runtime computes it once per document at build time. The leaf's state vector $u$ (§1) is the union of its operators' `writes`, in canonical order; every other state is a constant of the leaf over its interval (§1).
- **`writes`.** The operator's `writes` (§3) are the canonical indices of every cell of every variable in the plan's `writes` list. The right-hand side document is zero outside them by construction.
- **Parameters.** $p$ is the flattened model's scalar parameters other than data-fed ones, in code-point order of their names, at their defaults (the order of `rhs_program`'s `parameters`, which leaves externally fed parameters out); array parameters are fixed at their defaults, since array-parameter gradients wait for U7 (§6).
- **Declarations.** `depends_on_t` and `linear_in_u` are the plan's `structure.depends_on_t` and `structure.linear_in_u`.

### 8.3 Calls

- `rhs(out, u, p, t)`: evaluate the right-hand-side document at the full state (the leaf's $u$ scattered into the model state, the constants elsewhere), and add its values at the operator's `writes` into `out`.
- `jac(out, u, p, t)`: evaluate the Jacobian document, then for each band $k$ (1-based), with `b = structure.bands[k-1]`, add `J_k` at each point of the band into the entry $(\text{row}(r), \text{col}(\text{col\_idx}(r, c)))$, exactly as esm-jacobian-spec §4 defines a band's meaning; a column index outside the column variable's extent is a ghost read and contributes nothing. Entries outside the leaf's DOFs are dropped. The storage is `block_diagonal` (§4) when the leaf is: every operator is pointwise with `structure.block`, and the leaf's state variables are all arrays of one shape (`ncells` the number of cells of that shape) or all scalars (`ncells` = 1), with `n` the number of the leaf's state variables; otherwise `dense`. A `block_lu` or `sparse_block_lu` leaf whose storage is not `block_diagonal` is refused when it is built. A `dense_lu` leaf factors the same entries in either storage, so a runtime may store it `dense` (an all-scalar pointwise leaf, such as `robertson`'s, is `block_diagonal` with `ncells` = 1).
- `jvp_u` and `vjp_u`: $J v$ and $J^\top w$ from the same assembled entries, restricted to the leaf's DOFs.
- `ft`, `vjp_p`, `hess_u`, `hess_p`, `ft_vjp_u`, `ft_vjp_p`: **not provided at M2.** They need time-derivative bands (U17), parameter bands (U7) and second-order bands (U8). A leaf whose method or adjoint needs one of them is refused when it is built (§3), naming the operator and the call; so are Rosenbrock leaves on operators with `depends_on_t` true.

### 8.4 Forcing (M2; D87)

An operator whose plan entry has `forcing` (spec/plan.md §5.3) reads data. Both runtimes implement this section at M2: they fetch the data, build forcing sets and run leaves of such operators. The planner reads no data (spec/plan.md §4.1); everything here happens when the runtime builds and runs a plan.

#### 8.4.1 The calls

- Every call of §3 takes the forcing set as a last argument, `rhs(out, u, p, t, F)` and so on. $F$ is an `RhsProgram` forcing set (EarthSciAST API_SPEC §5.8, "Forcing sets"): one buffer per slot of the program's `forcing_slots`, in that order, each column-major over the parameter's declared shape (one value for a scalar). Julia evaluates with `rhs!(du, prog, ws, u, aux, p, t, fs)`, Rust with `RhsProgram::eval_into(&self, u, aux, p, t, &forcing, du, ws)`.
- The runtime holds one buffer per data-fed parameter of the plan's `forcing` (Julia `Vector{Float64}`, Rust `Arc<[f64]>`) and builds each program's set from the buffers its slots name (`forcing_set(prog, bufs)`, `ForcingSet::new(&prog, bufs)`), so one buffer is shared by every operator's right-hand-side and Jacobian programs. A program whose slots name a parameter that is not in the plan's `forcing`, or a slot whose shape differs from it, refuses the leaf.
- `jac`, `jvp_u` and `vjp_u` are with respect to $u$ with $F$ fixed. A step never crosses a forcing epoch (§8.4.3), so $F$ is constant over every step, and $F$ contributes nothing to `ft`: within an epoch the operator depends on $t$ only through the model's own expressions, which `depends_on_t` and the time-derivative bands (U17) describe.
- **Time interpolation is the model's.** With `records` 2 (spec/run-config.md §7.4) a buffer holds the record at or before the epoch's start and its successor along the parameter's first dimension, and the model blends them with a weight it computes from $t$, as reseact's GEOS-FP fields do. With `records` 1 it holds the one record at or before. The runtime never interpolates.

#### 8.4.2 The data layer

The runtime owns the data; an operator and a derived document own none. It builds the data layer from the model that `inputs.model` names (the plan's model, whose hash the plan checks), never from the derived documents:

1. Load the model with EarthSciAST, as the planner does (spec/plan.md §4 steps 3–5). Its `data_sources` are then resolved: a relative `url_template` is an absolute `file://` URL relative to the model's own directory (esm-spec §8.2.1). That is how a run locates its data: by the model's own declaration, so the same plan runs wherever the model and its data sit together. A caller may also pass, as run-time options of the runtime's API that are in neither the run config nor the plan, an EarthSciIO cache directory (default: EarthSciIO's own `datadir`), offline mode, and a map from source name to a replacement URL template (a local mirror; EarthSciAST's `url_overrides`). None of them changes what is computed, so none of them makes a plan stale.
2. For each source a data-fed parameter of the plan names, build one EarthSciIO provider per parameter, keyed by the parameter's flattened name, reading its `file_variable`:
   - **Format** from `data_sources.<source>.metadata.esio_format` (`"netcdf"`, `"zarr"`, …), as EarthSciAST's `providers_from_document` reads it (spec/plan.md §10 item 22). A source without it refuses the run, naming the source.
   - **URL.** The `url_template` with esm-spec §8.2's `{date:<strftime>}` substitutions expanded at each record's anchor time, in UTC (spec/plan.md §10 item 19). At M2 the directives `%Y`, `%m`, `%d`, `%H`, `%M`, `%j` and `%%` are supported; any other substitution refuses the run, naming the source and the substitution. Julia passes a function `t -> url` to EarthSciIO; Rust translates the template to the `time` crate's format description (`%Y` → `[year]`, `%m` → `[month]`, `%d` → `[day]`, `%H` → `[hour]`, `%M` → `[minute]`, `%j` → `[ordinal]`).
   - **Cadence.** `const`: Julia `EarthSciIO.const_provider(cache, url; format, variables = [fv])`; Rust `earthsciio::Provider::new(DataSource::new(name, format, url).variables([fv]), cache, None)`. `discrete`: the source's `temporal` read as EarthSciAST's Rust `to_esio_temporal` reads it (`start`, `frequency`, `file_period`, `end`, `time_variable` as the time dimension, default `"time"`); Rust passes it as `DataSource::temporal(SourceTemporal::new(start, frequency, file_period)…)` with `.records_per_sample(records)`, and the run window $[t_0, t_f]$ converted to UTC as the provider's window; Julia passes `EarthSciIO.discrete_provider(cache, t -> url, times; format, variables = [fv], time_dim, records_per_sample = records)` with `times` the anchors $\text{start} + k\,\text{frequency}$ from the last one at or before $t_0$ to the first at or after $t_f$, in model seconds. A source's `temporal.records_per_file`, when given, must equal `file_period / frequency`, else the run is refused.
   - **Not at M2**, each refusing the run with its name: a custom substitution that load did not bind (a mount's `url_substitutions` binds one at load, esm-spec §8.2), a `frequency` or `file_period` with a year or month part (a calendar duration, which EarthSciIO's fixed-second cadence cannot step; spec/plan.md §10 item 23), a source of kind `points`, a source or parameter `select`, `codes`, `record_filter`, `extent`, a non-numeric `unit_conversion`, `mirrors` (ignored with a warning would hide a failed fetch, so they are refused), and authentication realms.
   These are what EarthSciAST's `providers_from_document` does in Rust, plus what it does not yet do in either language (spec/plan.md §10 items 18–20). When those land upstream, the runtimes call `providers_from_document` (Julia, from the EarthSciIO extension; Rust, feature `esio`) instead, and the rules above become a check that they agree.
3. **Sample to buffer.** A provider sample (Julia: a `NativeDataset` from `materialize(p, t)`; Rust: `Provider::sample_at(t)`) is turned into the parameter's buffer exactly as EarthSciAST's `esm_problem` turns one into the parameter's value: the file variable's dimensions, in the file's order with the record dimension first when `records` is 2, are the parameter's declared dimensions in order and must have the same sizes (else the run is refused, naming the parameter and both shapes); the values are written column-major over the declared shape; a numeric `unit_conversion` multiplies them. EarthSciAST has no public function for this yet (spec/plan.md §10 item 21), so each runtime implements it, and its tests check it against EarthSciAST: for every data-fed parameter of the M2 data case, the buffer at $t_0$ equals, bit for bit, `forcing_buffers(prob)` of `esm_problem(model, (t₀, t_f); providers)` in Julia and the matching forcing array of the Rust problem built with the same providers.
4. **Times.** EarthSciIO's Rust provider counts in Unix seconds and its Julia provider in whatever unit the caller's `times` use. A runtime converts between model time and UTC with the model's time origin (`time_origin`, EarthSciAST 0.7, D74): UTC seconds = origin + $t$.

#### 8.4.3 Epochs

- `const` buffers are filled once, before the run. The forcing epochs are those of architecture §7.5: the sorted union of the `discrete` providers' refresh times inside $(t_0, t_f]$, plus $t_0$; epoch $k$ is $[B_k, B_{k+1})$, and its buffers are each discrete provider's sample at $B_k$.
- At M2 a runtime samples epoch $k$ when the run reaches $B_k$, on the calling thread. EarthSciIO's `PrefetchGroup` (Julia `PrefetchGroup(members; depth)`, `start!`, `take!`; Rust `PrefetchGroup::builder().member(key, provider)…`, `start`, `take`), whose `take(k)` is bitwise equal to `sample_at(boundaries()[k])`, may replace this without changing a result; asynchronous prefetch and rank windows are U11's (M5).
- At each boundary the runtime builds new sets (Julia `replace_forcing`, Rust `ForcingSet::replace`, a new epoch stamp, never an in-place write), so a workspace never sees a stale buffer.
- `forcing_refresh` is a tstop source (spec/run-config.md §5), required in `step.hit` whenever a discrete data-fed parameter is used (spec/plan.md §7.3), so no step crosses $B_k$; a stage evaluated at exactly $B_{k+1}$ belongs to the next step and reads epoch $k+1$. These times are the ones rule 9 is checked against when the runtime builds the run (spec/plan.md §7.3), and the runtime adds them to the macro-step boundaries as it adds every tstop. A data request outside the source's `temporal.start`/`end` refuses the run before its first step, naming the source and the time.
- The tape (spec/tape.md) records each epoch's $B_k$ and, per discrete parameter, the anchor of its sample, so a replay re-reads the same records; the adjoint's per-record accumulation (D5) is M4's.

#### 8.4.4 Ranks and tiles

At M2 (one rank, one tile) every buffer is held whole. A rank's window of a buffer and the halo a stencil operator's forcing reads need are M5's (architecture §7.5, U11); the state bands do not give the reach of a forcing read.
