# The plan and the planner

**Status:** M2 draft, 2026-10-25. Normative for both planners (D1, D8); the M0 skeleton of 2026-10-23 fixed the field set, this draft specifies how every field is computed. Its gate is conformance tier 1 (§9): for each fixture, the Julia and Rust planners write byte-identical `plan.json` and `derived/*.esm`, equal to the checked-in goldens. The checkpoint action vocabulary is specified at M4. Background: architecture §2, §3, D8, D14, D34, D64–D80.

The planner reads a model and a run config and writes a **plan**: `plan.json` plus a directory `derived/` of `.esm` documents. The runtime executes only the plan. The plan is where the Julia and Rust implementations are checked against each other (D8): the same inputs and spec version must give byte-identical files in both (architecture §3.3).

Every choice in this chapter that can change a byte of the output is stated exactly. Where this chapter says "EarthSciAST" or "EarthSciASTDiff" it means the bindings of the pinned versions (§0.2) in the planner's own language, never a call into the other language.

## 0. Scope at M2

### 0.1 What M2 plans

| Construct | M2 | Refusal code (spec/planner-errors.json) |
|---|---|---|
| Ordinary differential equations `D(x, t)`, scalar and arrayed, with `faq`/`makearray` right-hand sides, pointwise lifts, observeds, `ic` equations | planned | — |
| `operator_compose`, `couple`, `variable_map` couplings (EarthSciAST flattens them) | planned | — |
| Term operators (`select`), default operators, `bind_dt` | planned | — |
| The schedule grammar of spec/run-config.md §4 except `map`, `solve`, `coupled` with algebraic operators and `split_explicit` | planned | — |
| Derived documents `rhs` and `jacobian` | emitted | — |
| `time`, `step` (`dt`, `{"max": …}` limits, `hit`, `hit_multiples`, `resolution`), `partition`, `precision` | resolved | — |
| Algebraic equations (an `algebraic_variables` entry in the flattened model) | refused | `unsupported_algebraic_equation` |
| Continuous or discrete events | refused | `unsupported_event` |
| Parameter `update` rules of any kind, `schedule` and `data` included (D75) | refused | `unsupported_parameter_update` |
| Data-loader fields (`loader_fields`) | refused | `unsupported_forcing` |
| `map` operators and `map` nodes | refused | `unsupported_map_operator` |
| Observed step limits in `step.limits` or `subcycle.limits` | refused | `unsupported_step_limit` |
| `solve` nodes | refused | `unsupported_solve_node` |
| `coupled` with algebraic operators, `split_explicit`, `crossing` updates (spec/run-config.md §8 rule 16) | refused | `unsupported_coupled_dae`, `unsupported_split_explicit`, `unsupported_crossing_update` |
| Derived kinds other than `rhs` and `jacobian` (`frozen_jacobian`, `vjp`, `second_order`, `param_jacobian`, `forcing_jacobian` need U7–U9, M4; `residual`, `map`, `limit`, `condition` need U6, M4b) | not emitted | — |
| `column_block` structure (U9) | not emitted | — |

A run config whose `adjoint` key is present is planned; its checkpoint schedule is a placeholder (§7.6) and no adjoint artifact is derived.

### 0.2 Upstream versions

M2 is specified against EarthSciAST **0.4.0** (esm format **1.3.0**: `solver.external`, `solver.time_span` and `resolve_time_span`, per-term provenance `equation_contributions`, and Rust's `equation_lifted`) and EarthSciASTDiff **0.2** (the release of its current `main`, on EarthSciAST 0.4, in both languages). A planner reports the versions it was built against in `plan.report.json`, never in `plan.json`.

## 1. Files

```
<plan dir>/
  plan.json            canonical JSON (§2.1), golden-gated
  derived/*.esm        one document per derived artifact (§6), golden-gated
  plan.report.json     implementation-specific report, not golden-gated
```

- `derived/` documents are named `<operator>.<kind>.esm`, where `kind` is one of the `derived` kinds in the schema: `rhs`, `jacobian`, `frozen_jacobian`, `vjp`, `second_order`, `param_jacobian`, `forcing_jacobian`, `residual`, `map`, `limit`, `condition` (architecture §3.2 pass 6). At M2 every operator has exactly an `rhs` and a `jacobian` document.
- `frozen_jacobian` is emitted for an operator used by a `solve` node with `picard.freeze`, the only place freezing is written (D51). The forward iteration uses the frozen Jacobian; the adjoint needs the full residual Jacobian at the converged solution, emitted as `jacobian` (spec/adjoint.md §4.6). Not at M2.
- `plan.report.json` holds everything that legitimately differs between implementations: the planner's name and version, the upstream versions it was built with, timings, and the measured errors behind each validation check. Its format is not specified.
- **Band-value files** (Julia only, §8.2): `validation/<operator>.bandvalues.json` beside `plan.json`, written by the Julia planner when asked, read by the Rust planner. They are not part of the plan and not golden-gated byte for byte.

## 2. Canonical form

### 2.1 `plan.json`

`plan.json` is serialized as **RFC 8785 JSON Canonicalization Scheme** (JCS): UTF-8, object keys sorted by UTF-16 code units, no insignificant whitespace, no trailing newline, and numbers in ECMAScript's `Number.prototype.toString` form. Both implementations write JCS directly; neither post-processes the other's output. Check values for the number form: 300 → `300`, 0.1 → `0.1`, 1e-6 → `0.000001`, 1e-7 → `1e-7`, 1.5e21 → `1.5e+21`, −0 → `0`.

Beyond JCS, the content is canonical:
- Name lists (`writes`, `reads`, `hit`) are sorted by Unicode code point and duplicate-free.
- `terms` follow flattened-equation order, then contribution order, then summand order (§5.1).
- `structure.bands` follow EarthSciASTDiff's band order (§5.4).
- `validation` is sorted by `(operator, check)`, the operator-less `rhs_sum` entry first (§8).
- Every default is made explicit (§7); nothing is omitted because it equals a default. Options that do not apply, and options whose default is computed at run time, are absent (D73).
- Paths are relative to the plan directory or, for `inputs`, as §7.7 gives them. No absolute paths, hostnames, timestamps or implementation names appear.
- `sha256` is the lowercase hex SHA-256 of a file's bytes.

### 2.2 Derived documents

A derived document is a JSON object built as §6 describes and written as EarthSciASTDiff's **`stable_json`** writes it (Julia `EarthSciASTDiff.stable_json`, Rust `earthsci_astdiff::stable_json`): compact, object keys sorted by code point at every level, arrays in order, no trailing newline. Expressions inside it are EarthSciAST's serialized expression objects (Julia `EarthSciAST.serialize_expression`), whose numbers keep the literal type the flattened model holds: an integer literal is written as an integer (`2`), a float literal in the shortest round-trip form with Julia's layout (`0.5`, `1.0`, `1.0e-9`, `3.0e7`). This is the writer whose output EarthSciASTDiff's evaluation-document goldens already hold byte-identical between Julia and Rust. *Fixture authors:* a float-valued literal should be written non-integral or with a fraction (`1.5`, `2.0`); `1e3` is a float and is written `1000.0`.

## 3. Contents

| Key | Contents |
|---|---|
| `esm_integrator` | The spec version, equal to the run config's. |
| `inputs` | The model and run config, by path and hash (§7.7). A plan is stale when either hash changes. |
| `time` | The run interval in seconds (§7.2, D60, D74). Which source gave it is not recorded. A plan is also stale when the interval resolved for a run differs from this one. |
| `operators` | Per operator: `kind` (`rhs` at M2); `locality` with its per-axis `radius` for stencils (§5.2); sorted `writes` and `reads` (§5.3); `terms` (§5.1); `structure` (§5.4); `stiffness` with the source of the classification (§5.5). |
| `schedule` | The **resolved** schedule (§7.1): the run-config grammar with no `"auto"` anywhere, every leaf's options explicit, every default operator named, and an `id` on every node. |
| `step` | The macro-step control, as resolved (§7.3). |
| `partition` | Rank and tile grids, halo per horizontal axis, and shard counts per leaf (§7.4). |
| `precision` | Every precision role for every leaf, by node id (§7.5). |
| `checkpoint_schedule` | Present exactly when the run config has `adjoint`; a placeholder until M4 (§7.6, D78). |
| `derived` | Every derived document, by path, with its kind, operator and hash. |
| `validation` | The checks of pass 8 that ran (§8). A plan is written only when every check passes, so each entry records `passed: true`; measured errors go to `plan.report.json`. |

## 4. Reading the inputs

The planner runs these steps in order and stops at the first refusal. A refusal names its code (spec/planner-errors.json), the offending run-config key as a JSON pointer (or the model location), and a message. Only the code and pointer are conformance-tested.

1. **The run config.** Parse it; validate it against `schemas/run-config.schema.json` (`run_config_invalid`); refuse an `esm_integrator` version the planner does not implement (`version_unsupported`).
2. **The model path.** `model` is resolved relative to the run config's directory. `toy:<problem>` is the conformance harness's and is refused by the planner (`toy_model_not_plannable`).
3. **Load.** Load the document with EarthSciAST (Julia `EarthSciAST.load_path`). A load or validation failure, including EarthSciAST's own `solver_external_*` and `solver_time_span_*` load-time refusals, is `model_load_failed` with EarthSciAST's code in the message.
4. **`solver.external`** (spec/run-config.md §1, rule 1). When the model carries `solver.external`, its resolved location (EarthSciAST's accessor, esm-spec §2.2.6, with the model's directory as base) must equal the run config's own location written the same way (`file://` + the absolute path with dot segments removed lexically and percent-encoded as esm-spec §2.2.6 encodes), else `solver_external_mismatch`. The comparison is of the two strings. A URL `external` never matches a file. The planner never writes `solver.external` into a derived document.
5. **Flatten.** Flatten with EarthSciAST (`flatten`), which applies the couplings and records `equation_contributions` (esm-libraries-spec §4.7.5). Then expand surviving expression-template references (Julia `EarthSciAST.expand_flattened_refs`), lower `table_lookup` to `fn` and `enum` to integers (EarthSciAST's own lowering), each inside every contribution, as esm-libraries-spec §4.7.5 requires of passes over flattened equations. A flatten failure is `model_flatten_failed`.
6. **M2 scope.** Refuse, in this order, a flattened model with algebraic variables, events, parameter `update` rules, loader fields, or an index set that is not an `interval` with an integer size (`unsupported_index_set`), with the codes of §0.1.
7. **Axes** (§5.2.1) and **time** (§7.2) are resolved next, since passes 2 and 5 use them.

**Order of refusals.** After these steps the planner runs passes 1–4 (§5.1–§5.4), then checks the schedule (§7.1), step, partition, precision and adjoint (§7.3–§7.6), then resolves strategies (§5.5), builds the derived documents (§6) and validates (§8). The first refusal ends planning. Within §7.1, unsupported constructs are refused first, in tree order, then spec/run-config.md §8's rules in rule order.

The **flattened model** below means the result of step 5. Its state variables are the keys of `state_variables` that some equation's left-hand side differentiates; its **canonical state order** is those names sorted by code point, each array's cells in column-major order (EarthSciASTDiff's canonical order, esm-jacobian-spec §8). DOF $n$ (0-based) is position $n$ in that order. The initial state $u_0$ is EarthSciAST's, from building the flattened model (defaults and `field_ics`), permuted into canonical order.

## 5. Passes

### 5.1 Pass 1: terms and operators

#### 5.1.1 Terms

For each equation of the flattened model, in order, whose left-hand side differentiates a state (`D(x, t)`, or the pointwise lift's `faq` over `D(index(x, …), t)`): for each of its contributions $c = 0, 1, \ldots$ (the equation's `equation_contributions` entry, in order), decompose the contribution's expression into **summands** $s = 0, 1, \ldots$ by EarthSciASTSplitter's `additive_terms` (D14, D64):

- `+` with any number of arguments: the concatenation of the decompositions of its arguments, in order;
- `-` with two arguments $a, b$: the decomposition of $a$, then each summand $t$ of $b$ as `{"op": "-", "args": [t]}`;
- `-` with one argument $a$: each summand $t$ of $a$ as `{"op": "-", "args": [t]}`;
- anything else, including `*`, `faq`, `makearray` and `index`: itself, as one summand.

A summand that is a numeric literal equal to zero is dropped (esm-libraries-spec §4.7.5 lets a binding drop a literal-zero contribution's operand). Each remaining summand is a **term**, identified by `(variable, contribution, summand)`: the dependent variable's flattened name and the two 0-based indices. For a lifted equation the contribution expressions are the per-cell bodies EarthSciAST records (the `faq` wrapper is around their sum), so terms are per-cell expressions; Rust identifies lifted equations by `equation_lifted`, Julia by the lift's shape (esm-libraries-spec §4.7.5). Equations defining observeds and `ic` equations are not split into terms.

A term's **provenance label** is built from its contribution's `origins`: `{"component": P}` gives `P`; `{"coupling": k}` gives `coupling:k`; several origins (a collapsed contribution) are joined by `+` in their recorded order, e.g. `SuperFast+coupling:4`. An empty `origins` list is refused (`term_origin_unknown`), since only a caller's edit of a flattened system produces one.

#### 5.1.2 Owner

A term's **owner** is the component a `components` selector attributes it to (D65):

1. If its contribution has exactly one origin and it is a component `P`: `P`.
2. Otherwise, let $R$ be the set of components owning the names the term references (states, parameters, observeds; a name's component is the name without its last dot-separated segment), excluding the dependent variable's own component, `t` and index names. If $R$ is empty, the owner is the dependent variable's component; if $R$ has one element, that one; if two or more, the term has **no owner**.

A component $C$ in a selector **covers** a component $P$ when $P = C$ or $P$ begins with $C$ followed by a dot.

#### 5.1.3 Selection

Every term operator of the run config (spec/run-config.md §3.1) is tested against every term. A selector matches a term when every key present matches:
- `components`: some listed component covers the term's owner (a term with no owner never matches);
- `variables`: the term's dependent variable is listed;
- `locality`: the term's locality class (§5.2) is listed;
- `terms`: the term's provenance label is listed exactly.

Then, for each term in term order:
- matched by one operator: it belongs to that operator;
- matched by two or more: `term_selected_twice`, naming the first two operator names in code-point order;
- matched by none, with no owner, while any operator's selector has a `components` key: `ambiguous_term_owner`, naming the term and the components it reads;
- matched by none otherwise: it belongs to the **default operator** of its locality, `default_pointwise`, `default_column`, `default_stencil` or `default_global` (spec/run-config.md §3.3).

An operator of the run config that no term matches is `selector_matches_nothing`; component paths in a selector are not otherwise checked against the model, while `variables` entries must name states (`name_unresolved`, `name_wrong_kind`). A `map` operator is `unsupported_map_operator`. Architecture §3.2's inlining of an assembly component's observeds, for EqWeFiC's summed tendencies, is not done at M2: such a term belongs to the assembly component by rule 1 (D65).

#### 5.1.4 Operators

An operator is the set of its terms in term order, with the run config's `bind_dt` if it has one. At M2 every operator's `kind` is `rhs`. Default operators exist only when they have terms. The plan lists each term as an object:

```json
{"variable": "Chem.O3", "contribution": 1, "summand": 0, "provenance": "Transport", "locality": "stencil"}
```

### 5.2 Pass 2: locality

#### 5.2.1 Axes

The run config's `axes` (spec/run-config.md §6.1, D67) gives `horizontal`, a list of one or two index-set names, and `vertical`, one name. Defaults: no vertical axis; horizontal is every index set that is a dimension of some state's shape (§6.1.3 for lifted states), other than the vertical one, in the flattened model's `index_sets` order. A named index set that does not exist, a name in both roles, or more than two horizontal axes is `axes_unresolved`. An index set a state uses that is neither horizontal nor vertical is an **other** axis.

#### 5.2.2 A term's bands

The term's **band list** is EarthSciASTDiff's state Jacobian (`jacobian_bands`, `wrt = states`) of the term's own right-hand-side document: §6.1's document for a hypothetical operator whose only term is this one, loaded and flattened by EarthSciAST and viewed with `sysview`. An EarthSciASTDiff failure (`underivable_operator`, a filtered or non-`+` `faq`, any other refusal) is `band_model_failed`, carrying its message.

#### 5.2.3 Offsets

For each band entry, with row variable $x$ and column variable $y$:

- **Scalar row and scalar column:** no reach.
- **One scalar, one array:** the term is **global**.
- **Both arrays:** dimensions are paired by index-set name. For each dimension $d$ of $y$ with index set $X$:
  - if $x$ has no dimension on $X$: the reach covers $X$ (global if $X$ is horizontal or other, column if $X$ is vertical);
  - if `col_idx[d]` references a `contracted` name of the entry: likewise, the reach covers $X$;
  - otherwise, with `row_idx` entry $r$ of $x$'s dimension on $X$ and `col_idx[d]` = $c$, the **raw offsets** are:
    - $c$ equal to $r$: $\{0\}$;
    - $c$ of the form $r + \delta$, $r - \delta$ or $\delta + r$, $\delta$ an integer literal: $\{\pm\delta\}$;
    - $c$ an integer constant (a literal, or `+`, `-`, `*` of integer literals) $k$, and $r$ an integer $\rho$: $\{k - \rho\}$;
    - $c$ an integer constant $k$, and $r$ a name over the band's row range $[lo, hi]$ on that dimension: $\{k - hi, \ldots, k - lo\}$;
    - anything else (another index name, a table gather, a nonlinear index): the reach covers $X$.

On a horizontal axis of size $N$, each raw offset $o$ is **reduced**: $o' = ((o + \lfloor N/2 \rfloor) \bmod N) - \lfloor N/2 \rfloor$, with $\bmod$ giving a value in $[0, N)$. A periodic seam, cell 1 reading cell $N$, has $o = N - 1$ and $o' = -1$ (D66). Vertical offsets are not reduced.

#### 5.2.4 Class and radius

- The term is **global** if any band makes it global, or the reach covers a horizontal or other axis, or, on some horizontal axis, its lower plus upper radius plus one exceeds the axis size.
- Otherwise **stencil** if some reduced horizontal offset is nonzero. Its radius on a horizontal axis is `[lower, upper]`: `lower` the largest $-o'$ over negative offsets on that axis (cells toward lower index), `upper` the largest $o'$ over positive ones, each 0 if there are none.
- Otherwise **column** if some vertical offset is nonzero or the reach covers the vertical axis.
- Otherwise **pointwise**. A term with no bands is pointwise.

An operator's class is the coarsest of its terms' in the order pointwise < column < stencil < global. A stencil operator's `radius` lists **every** horizontal axis, each direction the maximum over its stencil terms (0 where none reaches). Only stencil operators have `radius` (the schema checks this).

### 5.3 Pass 3: DOF sets

- `writes`: the dependent variables of the operator's terms.
- `reads`: every state, parameter and observed name the operator's terms reference, together with, transitively, every name referenced by the definitions of the observeds among them. `t` and index names are not names.

Both are sorted by code point (§2.1).

### 5.4 Pass 4: structure

The operator's **band list** is EarthSciASTDiff's `jacobian_bands` (states) of its right-hand-side document (§6.1), in EarthSciASTDiff's order (equation order, then sorted column variable, then canonical site key; the order its `.bands.json` goldens hold). Band $k$ (1-based) is band field `J_k` of the Jacobian document (§6.2). An EarthSciASTDiff refusal is `band_model_failed`.

`structure` holds:

| Key | Present | Contents |
|---|---|---|
| `class` | always | EarthSciASTDiff's `detect_structure` on the band list: `block_diagonal`, `banded`, `general` or `empty`. A classification only: a periodic stencil is `general` (`structure.jl` marks every literal column index general), which is why the footprint below is read from the bands directly. |
| `linear_in_u` | always | `true` when no band coefficient references a state, or an observed whose definition references a state, transitively. |
| `depends_on_t` | always | `true` when the operator's right-hand-side document references `t` in a state equation or in the definition of an observed it keeps. |
| `bands` | always | One object per band, in field order: `row`, `col`, `rows`, `row_idx`, `col_idx` and, when non-empty, `contracted`, exactly as EarthSciASTDiff's `serialize_band` writes them, without `coef`. |
| `block` | pointwise operators whose `class` is `block_diagonal` | EarthSciASTDiff's `plan_factorization` (`BlockDiagonalPlan`, `markowitz_ordering`, `lu_fill`) serialized by `serialize_plan` without its `type` key: `vars` (its `block_vars`), `ordering`, `pattern`, `lu_pattern` (1-based pairs). |
| `column` | column operators | `axis` (the vertical axis), `kl`, `ku`: the largest sub- and super-diagonal vertical offsets over all bands, $\max(0, -\min o)$ and $\max(0, \max o)$; a band whose reach covers the vertical axis of size $N_v$ makes both $N_v - 1$. |
| `stencil` | stencil operators | `footprint`: the distinct vectors of reduced horizontal offsets over all bands (one integer per horizontal axis, in `axes.horizontal` order, an axis without an offset in that band contributing 0), sorted lexicographically. |

**The Gershgorin bound** of an operator at $(u, p, t)$ is $\rho = \max_r \sum_c |J_{rc}|$ over the rows $r$ of its `writes` DOFs, where $J$ is assembled from the band fields of its Jacobian document at $(u, p, t)$, adding every band's contribution to an entry before taking the absolute value (esm-jacobian-spec §4 additivity). At M2 it is used by pass 5 only; architecture §3.5's step limit from it needs `limit` operators (U6) and is not emitted (D71).

### 5.5 Pass 5: stiffness and strategy

#### 5.5.1 Sampled states

Passes 5 and 8 evaluate at three **sampled states** $s = 1, 2, 3$:
- time $t_s = t_0 + (t_f - t_0)\,q_s$, with $q_s$ = 0.25, 0.5, 0.75 and the interval of §7.2;
- parameters at their defaults;
- state $u_s[n] = b_n \cdot (0.5 + r_{s,n})$ for every DOF $n$ in canonical order, where $b_n = u_0[n]$ if it is nonzero and $1$ otherwise, and $r_{s,n} \in [0, 1)$ is SplitMix64 of the 64-bit unsigned integer $z = 2^{32} s + n$:

```
z = z + 0x9E3779B97F4A7C15
z = (z XOR (z >> 30)) * 0xBF58476D1CE4E5B9
z = (z XOR (z >> 27)) * 0x94D049BB133111EB
z = z XOR (z >> 31)
r = (z >> 11) * 2^-53
```

with wrapping 64-bit arithmetic and `>>` a logical shift. Check values: $r_{1,0}$ = 0.766301757339086, $r_{1,1}$ = 0.12603102042098713, $r_{2,0}$ = 0.905065227878185, $r_{3,2}$ = 0.03750424614895098. Every operation is exact or correctly rounded binary64, so both languages produce the same bits given the same $u_0$.

#### 5.5.2 Stiffness

Each operator gets `stiffness: {class, source}` (D70), by the first rule that applies:

1. **`run_config`**: the leaf that advances the operator gives `alg` (not `"auto"`). `stiff` if the method is a Rosenbrock, SDIRK, `IMEXEuler` or `Theta` method (spec/methods.md §2), `nonstiff` if explicit.
2. **`reaction_systems`**: some term of the operator has a single component origin that is a reaction system of the model. `stiff`.
3. **`solver_hint`**: the operator is pointwise and the model's `solver.stiffness` is `"high"`. `stiff`. (`"low"` and `"moderate"` assert nothing; D70.)
4. **`gershgorin`**: `stiff` if $\max_s \rho(u_s, p, t_s) \cdot \Delta \ge 10$, else `nonstiff`. $\Delta$ is the reference step of the leaf that advances the operator (§5.5.4).

#### 5.5.3 Methods where `alg` is `"auto"`

A leaf's **class** is the coarsest locality of its operators, and it is **stiff** if any of its operators is. Then (D71):

| Leaf | `alg` | `jacobian` | `linsolve` | `dt_scope` |
|---|---|---|---|---|
| pointwise, stiff | `Rosenbrock23` if the leaf's resolved `reltol` ≥ 1e-3, else `Rodas4` | `symbolic` | `{"block_lu": {"pivot": true, "refine": 0}}` if the block has at most 16 variables, else `{"sparse_block_lu": {"refine": 0}}` | `cell` |
| pointwise, non-stiff | `Tsit5` | — | — | `cell` |
| column, stiff | `TRBDF2` | `symbolic` | `{"banded": {}}` | `global` |
| column, non-stiff | `Tsit5` | — | — | `global` |
| stencil | `SSPRK43` | — | — | `global` |
| stencil with a stiff column operator, leaf `dt` given | `IMEXEuler`, `imex` implicit = the stiff column operators, explicit = the rest, `implicit_solve` `converged` | `symbolic` | `{"banded": {}}` | `global` |
| stencil with a stiff column operator, no `dt`; global, stiff | refused: `auto_strategy_unavailable` | | | |
| global, non-stiff | `SSPRK43` | — | — | `global` |

The block size is the number of `vars` in the union of the leaf's operators' `structure.block` variables. A `linsolve`, `jacobian` or `dt_scope` the run config gives explicitly is kept; only `"auto"` or an omitted key is resolved, and an omitted `linsolve` or `jacobian` on an explicitly named implicit method resolves as this table's row for the leaf's class would (pointwise `block_lu`/`sparse_block_lu`, column `banded`, otherwise `auto_strategy_unavailable`). `controller` `"auto"` is `PI`.

#### 5.5.4 Reference step $\Delta$

Computed top down over the resolved schedule. The root receives $\Delta_{\text{root}}$: `step.dt` when fixed; else the smallest `{"max": d}` in `step.limits`; else $t_f - t_0$. A `sequence` passes its $\Delta$ to every child; a `strang` passes $\Delta/2$ to every child but the last, which gets $\Delta$; a `subcycle` with `n` passes $\Delta/n$, and with `limits` passes the smaller of $\Delta$ and its smallest `{"max": d}`; an `every` passes its `period` in seconds. A leaf's $\Delta$ is its numeric `dt` if it has one, else what it receives. Operators advanced by no leaf are refused earlier (§7.1).

#### 5.5.5 The `"auto"` schedule

`"schedule": "auto"` (D21, D72): group every operator, named and default, by its locality class; one leaf per non-empty class, holding that class's operators sorted by name, in the order global, stencil, column, pointwise; every leaf option `"auto"`. One leaf is itself the root. Two or more form a `sequence`, or a `strang` when the model's `solver.splitting` is `"strang"`.

### 5.6 Pass 6: derived documents

§6.

### 5.7 Pass 7: partition

§7.4.

### 5.8 Pass 8: validation

§8.

## 6. Derived documents

### 6.1 The right-hand-side document `<operator>.rhs.esm`

A JSON object with exactly these keys:

- `"esm": "1.3.0"`.
- `"metadata": {"name": "<operator>"}`.
- `"index_sets"`: every index set of the flattened model, `{"kind": "interval", "size": N}` (§4 step 6 refused any other kind).
- `"domain"`: the flattened model's `domain`, as EarthSciAST serializes it, when it has one; otherwise absent.
- `"models"`: the **component tree** (§6.1.1).

No other key appears: no `solver`, `coupling`, `reaction_systems`, `data_sources`, `function_tables`, `enums`, `expression_templates` or `metaparameters`, all of which were consumed by load and flatten.

#### 6.1.1 The component tree

The document declares the flattened model's variables in models named after their components (D68). A variable's flattened name $P.v$ places $v$ in the component at path $P$: the top-level segment of $P$ is a key of `models`, and each further segment a key of the parent's `subsystems`. A reaction system's species and parameters are declared in a model of the reaction system's name. Every model and subsystem has `"variables"` and `"equations"` (possibly empty) and, when it has children, `"subsystems"`.

Declared, in each model:
- every **state** of the flattened model;
- every **parameter** of the flattened model;
- every **observed** that is *live*: referenced by a term of this operator, or by the definition of a live observed.

A variable's object is `{"type": T}` plus `"shape"` and `"default"` when present: `T` is `"parameter"` for parameters and `"unknown"` otherwise; `shape` is the declared shape, or for a lifted state the shape of §6.1.3; `default` is the flattened variable's default, as EarthSciAST holds it.

#### 6.1.2 Equations

In each model, `equations` lists, in the flattened model's equation order, the equations whose subject belongs to that model's component (for a state equation the dependent variable, for an observed definition the observed), followed by the `ic` equations of its states in `field_ics` order. Every reference is written as its flattened (fully scoped) name, on both sides; both EarthSciAST 0.4 bindings resolve such references from any model (D68).

- **State equation, not lifted.** `{"lhs": {"op": "D", "args": ["<x>"], "wrt": "t"}, "rhs": R}`, where R is the left fold `((t_1 + t_2) + t_3) + …` of this operator's terms of that equation in term order, written as nested binary `{"op": "+", "args": [a, b]}` (one term: the term itself). With no term: `0.0` for a scalar state, and `{"op": "makearray", "args": [], "regions": [[[1, n_1], …, [1, n_k]]], "values": [0.0]}` for an array of extents $n_1 \ldots n_k$.
- **State equation, lifted.** The same left-hand side, `D(x)` on the whole array, and R = `{"op": "faq", "args": [], "output_idx": O, "ranges": G, "expr": F}` with `O` and `G` copied from the flattened equation's right-hand-side `faq` node and `F` the left fold of the per-cell terms; with no term, the `makearray` zero above.
- **Observed definition** (a live observed): copied, left-hand side and right-hand side serialized as they stand in the flattened model.
- **Initial condition**: `{"lhs": {"op": "ic", "args": ["<x>"]}, "rhs": E}` for every `field_ics` entry of every state, whether or not the operator writes it, so every document builds the same $u_0$.

#### 6.1.3 Shapes of lifted states

A lifted state has extents but no declared shape (EarthSciAST's `lifted_shapes`). Its shape is the declared shape of the first variable of the flattened model, searching `state_variables`, then `parameters`, then `observed_variables`, each in document order, whose declared shape has exactly those extents, matched index set by index set. If none does: `lift_shape_unresolved` (D68; an upstream gap, §10).

#### 6.1.4 Name collisions

The planner refuses a model with a top-level component named `_jacobian` (`reserved_component_name`), since §6.2 uses that name.

### 6.2 The Jacobian document `<operator>.jacobian.esm`

The operator's right-hand-side document (§6.1), changed as follows (D69):

1. Every state equation's right-hand side is replaced by the zero of §6.1.2 (`0.0`, or the `makearray` zero), whether or not the operator writes the state.
2. Observeds are live when referenced, directly or transitively, by a band coefficient, instead of by a term.
3. A model `_jacobian` is added to `models`, holding, for each band $k$ = 1, 2, … in band order, a variable `J_k` and an equation for it, exactly as EarthSciASTDiff's evaluation document (`_evaluation_document` in Julia, `evaluation_document` in Rust) builds its band field `J_k` with coefficient hoisting off (`cse = false`), except that each reference is written fully scoped and `J_k` itself is referenced as `_jacobian.J_k`: the variable is `{"type": "unknown", "default": 0.0}` plus, for an array row or a band with `contracted` dimensions, `"shape"` (the row variable's index sets followed by one index set per contracted dimension); the equation's left-hand side is `D(_jacobian.J_k)` and its right-hand side the band's coefficient over its region, zero elsewhere. Index sets for contracted dimensions are added to `index_sets` under EarthSciASTDiff's names.

Evaluating the document's right-hand side at $(u, p, t)$ gives every band coefficient in the `J_k` slots and zero in every state's slot. A runtime scatters `J_k` into its matrix storage with `structure.bands[k-1]` (spec/operator-interface.md §8).

## 7. Resolved settings

### 7.1 Schedule

The planner refuses the unsupported nodes of §0.1 (`map`, `solve`, `split_explicit`, `coupled` with algebraic operators) in tree order, then checks the schedule against spec/run-config.md §8 rules 3, 5–9, 12, 13 and 15 in rule order, with the codes of spec/planner-errors.json, and then writes the resolved tree. For every node:

- **`id`**: the given id, else the node's path: `root` for the root; a child of a `sequence` or `strang` at position $i$ (0-based) is `<parent path>.<i>`; the child of a `subcycle` is `<parent path>.subcycle`, of an `every` `<parent path>.every`. Paths are built from positions alone, whatever ids the ancestors have, so the unnamed first child of a node `transport` at `root.0` is `root.0.0`. A run config may write a path-form id only on the node whose path it is (`node_id_reserved` otherwise), so given and assigned ids never collide; the schema accepts both forms.
- **`leaf`, `coupled`**: always a list of operator names, in the order given (accumulation order, spec/operator-interface.md §3).
- **Durations** (`dt` when numeric, `dt0`, `dtmin`, `dtmax`, `period`, `phase`, `{"max": …}`) are written in seconds as numbers; a parameter-name duration is the parameter's default (spec/run-config.md §2.2). `phase` 0 is written `0`.

A leaf or `coupled` node is written with exactly these keys:

| Key | When | Value |
|---|---|---|
| `id`, `leaf`/`coupled`, `alg` | always | resolved (§5.5.3) |
| `dt` | when given | seconds, `"step"` or `"accumulated"` |
| `control` | always | object below |
| `jacobian` | methods that form $J$ (Rosenbrock, SDIRK, `IMEXEuler`, `Theta`) | given, or `symbolic` |
| `linsolve` | methods that form $J$ | object form with every option this spec defaults: `{"block_lu": {"pivot": P, "refine": R}}` (defaults `true`, 0), `{"sparse_block_lu": {"refine": R}}` (0), `{"dense_lu": {}}`, `{"banded": {}}`, `{"block_tridiagonal": {}}`, `{"matrix_free": {}}`; Krylov options as given |
| `domain` | always | `{"on_violation": V, "check_stages": B}` (defaults `reject`, `false`) plus `nonnegative` when given |
| `imex`, `implicit_solve` | `IMEXEuler` | `imex` as lists of operator names in leaf order, a locality name expanded to the leaf's operators of that class; `implicit_solve` given or `converged` |
| `theta` | `Theta` | as given |
| `frozen`, `bind_dt` | when given | as given, durations in seconds |

`control` holds `reltol` (default 1e-3) and `abstol` (default 1e-6; a per-variable object as given) always; `dt_scope` always (§5.5.3; an explicit method's leaf takes the table's `dt_scope` for its class); and, for adaptive leaves only (no `dt`), `controller` (default `PI`) and `maxiters` (default 1000000), plus `dt0`, `dtmin` and `dtmax` when given (`dtmin`'s default, 0, is not a duration the schema admits, and means no minimum). A fixed-`dt` leaf's `maxiters` is written only when given (spec/methods.md §12: unlimited otherwise). A leaf's `partition` key does not appear in the resolved leaf; it lands in `partition.shards` (§7.4).

Other nodes keep their keys, resolved recursively: `sequence`, `strang` (children), `subcycle` (`n` or `limits` with `{"max"}` in seconds), `every` (`period`, `phase` (default 0), `hold` (default `none`)).

### 7.2 Time

The interval comes from the caller, the run config's `time`, or the model's `solver.time_span`, the first that gives one giving all of it (D60). Numbers are seconds. A timestamp, from any source, is converted exactly as esm-spec §2.2.7 converts a `time_span` timestamp: seconds from the model's time origin (`domain.temporal.reference_time`, else `domain.temporal.start`, else the POSIX epoch) as $(W - W_0) + (f - f_0)$ (D74). The model's `solver.time_span` is read with EarthSciAST's `resolve_time_span`. `duration` is added to the start in binary64. Refusals: no interval (`time_interval_missing`), end not after start (`time_interval_empty`), a timestamp naming no calendar day (`time_invalid_date`), a timestamp when the model's origin field is unreadable (`time_origin_invalid`) (spec/run-config.md §8 rule 14). The plan's `time` is `{"start": t_0, "end": t_f}`.

### 7.3 Step

`step` is written with `dt` (seconds) or `limits` (each `{"max": seconds}`; an observed limit is `unsupported_step_limit`) when given, `hit` always (sorted; `[]` by default), and `hit_multiples` and `resolution` (seconds) when given. With neither `dt` nor `limits` the root must be a single adaptive `leaf` or `coupled` node (`step_control_missing`, rule 12). Rule 9 (`tstop_off_fixed_step`) is checked for the tstops M2 can know: `hit_multiples` and the adjoint's observation times.

### 7.4 Partition

`partition` (D77):
- `ranks`, `tiles`: as given, default `[1, 1]`. Entry 0 divides `axes.horizontal[0]`, entry 1 `axes.horizontal[1]`. With fewer horizontal axes, the entries for missing axes must be 1 (`partition_axis_missing`); `ranks[a] · tiles[a]` above the axis size is `partition_too_fine`.
- `halo`: for each horizontal axis, the largest `lower` or `upper` radius over all stencil operators, 0 when none; absent when there is no horizontal axis.
- `shards`: for every pointwise or column leaf, by node id: the leaf's own `partition.shards`, else its `shard_size` turned into $\lceil \text{cells} / \text{shard\_size} \rceil$ (cells: the leaf's array cells for pointwise, columns for column), else the run config's `partition.shards` entry for the leaf's node id, else for one of its operators (all of a leaf's operator entries must agree, `partition_conflict`), else 1. `"auto"` is 1 at M2.
- `output_shard`: absent at M2.

### 7.5 Precision

For every leaf and `coupled` node, by id, all five roles (spec/run-config.md §7.1): the node-id entry overrides the leaf's operators' entries, which override `default`, role by role; operator entries of one leaf that disagree on a role are `precision_conflict`. Unset roles: `state` is the declared `element_type` that governs the leaf's states (esm-spec §11.3.1), else `f64`; `rhs` is `state`; `jacobian` and `factors` are `rhs`; `tables` is `f64` (no forcing tables at M2). Refusals: rule 10 (`precision_contradicts_declaration`, `precision_rhs_below_state`), and a `jacobian` or `factors` role below `rhs` on a Rosenbrock method other than `Rosenbrock23` (architecture §8; `precision_low_with_classical_rosenbrock`).

### 7.6 Checkpoint schedule

Present exactly when the run config has `adjoint`: `{"kind": "all", "macro_steps": 0}`, a placeholder that M4 replaces (D34, D78). The adjoint rules 11 (`wrt_matches_nothing`, `wrt_bound_parameter`) are checked.

### 7.7 Inputs and derived

- `inputs.model.path` is the run config's `model` string as written; `inputs.run_config.path` is the run config's file name without directories. Hashes are of the files' bytes.
- `derived` lists `derived/<operator>.rhs.esm` and `derived/<operator>.jacobian.esm` for every operator, each with `kind`, `operator` and the hash of the bytes written.

## 8. Pass 8: validation

The checks run after the derived documents are written to memory and before anything is written to disk. A failing check is a refusal (`validation_rhs_sum_failed`, `validation_band_model_failed`, `validation_empty_operator`) carrying the measured quantity.

### 8.1 Checks

- **`rhs_sum`** (one entry, no operator): at each sampled state, every operator's right-hand-side document and the flattened model are built with EarthSciAST and evaluated; for every DOF, $|\sum_k f_k - f| \le 10^{-12} \max(|f|, \sum_k |f_k|)$, an exact zero passing (D76).
- **`band_model`** (per operator): at each sampled state, the band Jacobian $J^B$ assembled from the Jacobian document's fields and `structure.bands` is compared with an independent Jacobian $J^I$ (§8.2). For every entry, $|J^B_{rc} - J^I_{rc}| \le 10^{-10} \max_{c'} |J^I_{rc'}|$, and every $(r, c)$ with $J^I_{rc} \ne 0$ is in the band pattern.
- **`non_empty`** (per operator): the operator has at least one term, and its band list is non-empty when any of its terms references a state.

`validation` lists `rhs_sum` first, then `band_model` and `non_empty` for each operator in code-point order of the operator name.

### 8.2 The independent Jacobian

- **Julia:** ForwardDiff's Jacobian of EarthSciAST's `f!` built from the operator's right-hand-side document, at the sampled state, restricted to the rows of `writes`. This is the oracle architecture §3.2 pass 8 names; EarthSciASTDiff's own oracle lives only in its tests.
- **Rust:** the values of a **band-value file** written by the Julia planner for the same plan: `validation/<operator>.bandvalues.json`, `{"samples": [{"t": t_s, "u": {"<name>[i,…]": value, …}, "entries": [["<row element>", "<column element>", value], …]}, …]}`, in the style of EarthSciASTDiff's `.jvals.json` (element names as EarthSciAST writes them, entries sorted by row then column name). The Rust planner evaluates its band Jacobian at the file's `u` and `t`, requires the same structural key set, and applies §8.1's tolerance. Without the file it refuses with `band_values_missing` (D76). For conformance fixtures the files are checked in beside the goldens (§9).

## 9. Conformance

Tier 1 (architecture §12): for each fixture case under `tests/conformance/plans/fixtures/` (layout in `tests/conformance/README.md`), both planners write `plan.json` and `derived/*.esm`, and both are compared byte for byte with the checked-in golden. Goldens are written by the Julia planner and reviewed by hand against the case's `expected.md` before they are checked in; the Rust planner must then match them. Refusal fixtures under `tests/conformance/plans/refusals/` must be refused with the listed code at the listed pointer.

Tier 0 (spec/README.md) checks the fixtures' structure: every `.esm` loads and validates with EarthSciAST 0.4.0, every run config validates, every refusal code is listed in spec/planner-errors.json, and every case has an `expected.md` entry. Until M2's goldens exist, it also checks that the schema accepts `tests/conformance/plans/valid/` and rejects `tests/conformance/plans/invalid/`.

## 10. Upstream gaps found while specifying M2

Recorded here, not filed (the RFC process files them):

1. **Lifted states have no index-set names.** EarthSciAST's `lifted_shapes` records a lifted state's extents but not which index sets they are; §6.1.3's search for a matching declared shape works around it and fails for a lift onto a grid no variable declares. Asking: record the lift's index sets.
2. **EarthSciASTDiff's evaluation document for a flattened system does not load.** It declares dotted names (`SimpleOzone.O3`) in one model, which EarthSciAST 0.4.0 reads as references to an undefined system (checked in Python and Julia). §6.2 builds its own tree for this reason. The Julia builder is also internal (`_evaluation_document`).
3. **Julia's `build_evaluator(::EsmFile)` refuses a model's scoped references to its own variables** (`E_TREEWALK_UNBOUND_VARIABLE` for `Decay.k` inside model `Decay`) where flattening and Python accept them; runtimes build derived documents through `flatten` (spec/operator-interface.md §8).
4. **No time derivative from EarthSciASTDiff** (`tgrad` is a planned extension, esm-jacobian-spec §7), so planned operators cannot provide `ft` and Rosenbrock leaves on time-dependent planned operators wait (D80).
5. **Julia's native compiler declines a contracting `faq` inside a lifted equation** (`compiler_refused_rule`; the interpreter evaluates it). Derived documents rewrite lifted equations as plain arrays (§6.1.2), which the native compiler accepts in the column fixture.
6. **A `makearray` region with singleton axes** must give a value whose rank is the number of non-singleton axes in Julia, while Python accepts a value over all axes; fixtures avoid size-1 axes.
