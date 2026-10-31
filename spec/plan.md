# The plan and the planner

**Status:** M2 draft, 2026-10-25. Normative for both planners (D1, D8); the M0 skeleton of 2026-10-23 fixed the field set, this draft specifies how every field is computed. Its gate is conformance tier 1 (§9): for each fixture, the Julia and Rust planners write byte-identical `plan.json` and `derived/*.esm`, equal to the checked-in goldens. The checkpoint action vocabulary is specified at M4. Background: architecture §2, §3, D8, D14, D34, D64–D84. Stage 1's findings (Julia PR #21, Rust PR #20) were resolved on 2026-10-26 (D81–D84).

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

M2 is specified against EarthSciAST **0.6.0** (esm format **1.3.0**: `solver.external`, `solver.time_span` and `resolve_time_span`, per-term provenance `equation_contributions`, and Rust's `equation_lifted`, all since 0.4.0; 0.5.0 adds the public reentrant right-hand side `rhs_program` in Julia and Rust, U4; 0.6.0 lowers reaction systems to one form in every binding, esm-libraries-spec §4.6.2, and its Julia binding accepts SciMLBase 3) and EarthSciASTDiff **0.1.0** (commit 743d140, its first registered release; on EarthSciAST 0.6 in both languages, its Julia package since EarthSciASTDiff#19, §10 item 12). Conformance tier 0 runs on EarthSciAST's Python binding, 0.6.0. A planner reports the versions it was built against in `plan.report.json`, never in `plan.json`.

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
- **Band-value files** (conformance only, §8.2): `validation/<operator>.bandvalues.json` beside a golden's `plan.json`, written by the Julia planner when asked, read by the conformance harness's Rust comparison. They are not part of the plan and not golden-gated byte for byte.

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

The planner runs these steps in order and stops at the first refusal. A refusal names its code (spec/planner-errors.json), the offending run-config key as a JSON pointer (or the model location), and a message. Only the code and pointer are conformance-tested. For a refusal whose cause has no run-config key of its own, spec/planner-errors.json gives the pointer in the code's description (D79).

1. **The run config.** Parse it; validate it against `schemas/run-config.schema.json` (`run_config_invalid`, at the empty pointer `""`, since a failing `oneOf` has no single location); refuse an `esm_integrator` version the planner does not implement (`version_unsupported`).
2. **The model path.** `model` is resolved relative to the run config's directory. `toy:<problem>` is the conformance harness's and is refused by the planner (`toy_model_not_plannable`).
3. **Load.** Load the document with EarthSciAST (Julia `EarthSciAST.load_path`). A load or validation failure, including EarthSciAST's own `solver_external_*` load-time refusals, is `model_load_failed` with EarthSciAST's code in the message. A `solver.time_span` that EarthSciAST rejects, here or later in `resolve_time_span` (§7.2), is refused with the rule-14 code of its cause, at `/model`: an end not after the start is `time_interval_empty`, a day its month does not have `time_invalid_date`, an unreadable time origin `time_origin_invalid`; any other cause is `model_load_failed`.
4. **`solver.external`** (spec/run-config.md §1, rule 1). When the model carries `solver.external`, its resolved location (EarthSciAST's accessor, esm-spec §2.2.6, with the model's directory as base) must equal the run config's own location written the same way (`file://` + the absolute path with dot segments removed lexically and percent-encoded as esm-spec §2.2.6 encodes), else `solver_external_mismatch`. The comparison is of the two strings. A URL `external` never matches a file. The planner never writes `solver.external` into a derived document.
5. **Flatten.** Flatten with EarthSciAST (`flatten`), which applies the couplings and records `equation_contributions` (esm-libraries-spec §4.7.5). Then expand surviving expression-template references (Julia `EarthSciAST.expand_flattened_refs`), lower `table_lookup` to `fn` and `enum` to integers (EarthSciAST's own lowering), each inside every contribution, as esm-libraries-spec §4.7.5 requires of passes over flattened equations. A flatten failure is `model_flatten_failed`.
6. **M2 scope.** Refuse, in this order, a flattened model with algebraic variables, events, parameter `update` rules, loader fields, or an index set that is not an `interval` with an integer size (`unsupported_index_set`), with the codes of §0.1, each at `/model`. The order holds whatever the parameter rules are: a model with events and a parameter `update` rule is `unsupported_event`. At the parameter-update place, the code is `unsupported_crossing_update` when any parameter has a `crossing` update rule, whatever the order of the parameters, and `unsupported_parameter_update` otherwise: the more specific code wins over the general one only.
7. **Reserved names.** Refuse a model with a top-level component named `_jacobian` (`reserved_component_name`, §6.1.4).
8. **Axes** (§5.2.1) and **time** (§7.2) are resolved next, since passes 2 and 5 use them. The shapes of lifted states (§6.1.3) are resolved here, before the axes and whether or not the run config gives `axes`, since the default axes read every state's shape; a lifted state with no matching declared shape is `lift_shape_unresolved`. Then the axes (`axes_unresolved`), then the time.

**Order of refusals** (D81). After these steps the planner runs, in order:

1. pass 1's terms and owners (§5.1.1–§5.1.2; `term_origin_unknown`);
2. each term's locality (§5.2.2–§5.2.4 applied to the term; `band_model_failed`), since selection reads it;
3. selection (§5.1.3), with its refusals in the order §5.1.3 gives;
4. each operator's class and radius (§5.2.4), pass 3 (§5.3) and pass 4 (§5.4; `band_model_failed`);
5. the schedule (§7.1), the step (§7.3), the partition (§7.4), the precision (§7.5) and the adjoint (§7.6);
6. the strategies (§5.5), then the precision check of §7.5 that waits for a resolved method (`precision_low_with_classical_rosenbrock` on a leaf whose `alg` is `"auto"`);
7. the derived documents (§6) and validation (§8).

So `band_model_failed` for a term precedes every selection refusal. The first refusal ends planning. Within §7.1, unsupported constructs are refused first, in tree order, then spec/run-config.md §8's rules in rule order (§7.1).

The **flattened model** below means the result of step 5. Its state variables are the keys of `state_variables` that some equation's left-hand side differentiates; its **canonical state order** is those names sorted by code point, each array's cells in column-major order (EarthSciASTDiff's canonical order, esm-jacobian-spec §8). DOF $n$ (0-based) is position $n$ in that order. The initial state $u_0$ is EarthSciAST's, from building the flattened model (defaults and `field_ics`), permuted into canonical order.

## 5. Passes

### 5.1 Pass 1: terms and operators

#### 5.1.1 Terms

For each equation of the flattened model, in order, whose left-hand side differentiates a state (`D(x, t)`, or the pointwise lift's `faq` over `D(index(x, …), t)`): for each of its contributions $c = 0, 1, \ldots$ (the equation's `equation_contributions` entry, in order), decompose the contribution's expression into **summands** $s = 0, 1, \ldots$ by EarthSciASTSplitter's `additive_terms` (D14, D64):

- `+` with any number of arguments: the concatenation of the decompositions of its arguments, in order;
- `-` with two arguments $a, b$: the decomposition of $a$, then each summand $t$ of $b$ as `{"op": "-", "args": [t]}`;
- `-` with one argument $a$: each summand $t$ of $a$ as `{"op": "-", "args": [t]}`;
- anything else, including `*`, `faq`, `makearray` and `index`: itself, as one summand.

A summand that is a numeric literal equal to zero is dropped (esm-libraries-spec §4.7.5 lets a binding drop a literal-zero contribution's operand). Only a literal is dropped: in `a - 0` the second summand is `{"op": "-", "args": [0]}`, an operation and not a numeric literal, so it is kept as a term. Each remaining summand is a **term**, identified by `(variable, contribution, summand)`: the dependent variable's flattened name and the two 0-based indices. For a lifted equation the contribution expressions are the per-cell bodies EarthSciAST records (the `faq` wrapper is around their sum), so terms are per-cell expressions; Rust identifies lifted equations by `equation_lifted`, Julia by the lift's shape (esm-libraries-spec §4.7.5). Equations defining observeds and `ic` equations are not split into terms.

A term's **provenance label** is built from its contribution's `origins`: `{"component": P}` gives `P`; `{"coupling": k}` gives `coupling:k`; several origins (a collapsed contribution) are joined by `+` in their recorded order, e.g. `SuperFast+coupling:4`. An empty `origins` list is refused (`term_origin_unknown`), since only a caller's edit of a flattened system produces one.

#### 5.1.2 Owner

A term's **owner** is the component a `components` selector attributes it to (D65):

1. If its contribution has exactly one origin and it is a component `P`: `P`.
2. Otherwise, let $R$ be the set of components owning the names the term references (states, parameters, observeds; a name's component is the name without its last dot-separated segment), excluding the dependent variable's own component, `t` and index names. If $R$ is empty, the owner is the dependent variable's component; if $R$ has one element, that one; if two or more, the term has **no owner**.

A component $C$ in a selector **covers** a component $P$ when $P = C$ or $P$ begins with $C$ followed by a dot.

#### 5.1.3 Selection

Selection runs after every term's locality is known (§5.2, §4 "Order of refusals"; D81). Every term operator of the run config (spec/run-config.md §3.1) is tested against every term. A selector matches a term when every key present matches:
- `components`: some listed component covers the term's owner (a term with no owner never matches);
- `variables`: the term's dependent variable is listed;
- `locality`: the term's locality class (§5.2) is listed;
- `terms`: the term's provenance label is listed exactly.

Then, for each term in term order:
- matched by one operator: it belongs to that operator;
- matched by two or more: `term_selected_twice`, naming the first two operator names in code-point order;
- matched by none, with no owner, while any operator's selector has a `components` key: `ambiguous_term_owner`, naming the term and the components it reads;
- matched by none otherwise: it belongs to the **default operator** of its locality, `default_pointwise`, `default_column`, `default_stencil` or `default_global` (spec/run-config.md §3.3).

An operator of the run config that no term matches is `selector_matches_nothing`; component paths in a selector are not otherwise checked against the model, while `variables` entries must name states (`name_unresolved`, `name_wrong_kind`). A `map` operator is `unsupported_map_operator`.

Selection's refusals come in this order, the first one found ending planning:
1. `unsupported_map_operator`, for the `map` operators in code-point order of operator name;
2. `variables` entries (`name_unresolved`, `name_wrong_kind`), operators in code-point order, each operator's entries in written order;
3. per term, in term order, `term_selected_twice` or `ambiguous_term_owner` as above;
4. `selector_matches_nothing`, for the first such operator in code-point order.

Architecture §3.2's inlining of an assembly component's observeds, for EqWeFiC's summed tendencies, is not done at M2: such a term belongs to the assembly component by rule 1 (D65).

#### 5.1.4 Operators

An operator is the set of its terms in term order, with the run config's `bind_dt` if it has one. At M2 every operator's `kind` is `rhs`. Default operators exist only when they have terms. The plan lists each term as an object:

```json
{"variable": "Chem.O3", "contribution": 1, "summand": 0, "provenance": "Transport", "locality": "stencil"}
```

### 5.2 Pass 2: locality

#### 5.2.1 Axes

The run config's `axes` (spec/run-config.md §6.1, D67) gives `horizontal`, a list of one or two index-set names, and `vertical`, one name. Defaults: no vertical axis; horizontal is every index set that is a dimension of some state's shape (§6.1.3 for lifted states, whose shapes §4 step 8 resolves first), other than the vertical one, in the flattened model's `index_sets` order. A named index set that does not exist, a name in both roles, or more than two horizontal axes is `axes_unresolved`. An index set a state uses that is neither horizontal nor vertical is an **other** axis (possible only when the run config gives `axes`). Offsets on an other axis do not affect a term's class, while a reach that covers one makes the term global (§5.2.3–§5.2.4, D82).

Pass 2 runs in two parts (D81): each term's class (§5.2.2–§5.2.4) before selection (§5.1.3), which reads it, and each operator's class and radius (the end of §5.2.4) after.

#### 5.2.2 A term's bands

The term's **band list** is EarthSciASTDiff's state Jacobian (`jacobian_bands`, `wrt = states`) of the term's own right-hand-side document: §6.1's document for a hypothetical operator whose only term is this one, loaded and flattened by EarthSciAST and viewed with `sysview`. An EarthSciASTDiff failure (`underivable_operator`, a filtered or non-`+` `faq`, any other refusal) is `band_model_failed` at `/model`, carrying its message.

This defines the band list; it does not prescribe one document per term, which at reseact's size (thousands of terms) would mean thousands of EarthSciAST loads and flattens. Pass 2 uses the bands only for the locality and radius they imply (§5.2.3), so a planner may compute them any way that gives the same offsets. One way: a single probe document in which each term is the whole right-hand side of its own new state, of the shape of the term's equation's state, and one `jacobian_bands` call, whose rows for that state are the term's bands. Whatever the method, an EarthSciASTDiff failure is reported against the term it concerns.

#### 5.2.3 Offsets

For each band entry, with row variable $x$ and column variable $y$:

- **Scalar row and scalar column:** no reach.
- **One scalar, one array:** the term is **global**.
- **Both arrays:** dimensions are paired by index-set name: the $j$-th dimension of $y$ on index set $X$ (counting only $y$'s dimensions on $X$, from the left) pairs with the $j$-th dimension of $x$ on $X$. For each dimension $d$ of $y$ with index set $X$:
  - if $x$ has no paired dimension on $X$ (no dimension on $X$, or fewer than $j$): the reach covers $X$ (global if $X$ is horizontal or other, column if $X$ is vertical);
  - if `col_idx[d]` references a `contracted` name of the entry: likewise, the reach covers $X$;
  - otherwise, with `row_idx` entry $r$ of $x$'s paired dimension and `col_idx[d]` = $c$, the **raw offsets** are:
    - $c$ equal to $r$: $\{0\}$;
    - $c$ of the form $r + \delta$, $r - \delta$ or $\delta + r$, $\delta$ an integer literal: $\{\pm\delta\}$;
    - $c$ an **integer constant** $k$, and $r$ an integer $\rho$: $\{k - \rho\}$. An integer constant is, recursively, an integer literal, a `-` of one or two integer constants, or a `+` or `*` of one or more integer constants. Its value is evaluated exactly in 64-bit signed integers; an expression whose value, or the value of any of its sub-expressions, falls outside that range is not an integer constant (the "anything else" case below). An **integer literal** is a number the flattened model holds as an integer (§2.2: the literal type is kept); a float literal such as `1.0` is not one, even when its value is integral, so `+(i, 1.0)` is "anything else";
    - $c$ an integer constant $k$, and $r$ a name over the band's row range $[lo, hi]$ on that dimension: $\{k - hi, \ldots, k - lo\}$;
    - one of the three cases above whose offset $-\delta$, $k - \rho$, $k - hi$ or $k - lo$ falls outside 64-bit signed integers: as "anything else" below, the reach covers $X$, even when $k$ itself is a valid integer constant;
    - anything else (another index name, a table gather, a nonlinear index): the reach covers $X$.

On a horizontal axis of size $N$, each raw offset $o$ is **reduced**: $o' = ((o + \lfloor N/2 \rfloor) \bmod N) - \lfloor N/2 \rfloor$, with $\bmod$ giving a value in $[0, N)$. A periodic seam, cell 1 reading cell $N$, has $o = N - 1$ and $o' = -1$ (D66). Vertical offsets are not reduced. Offsets on an other axis are computed by the same rules but not used: they do not affect the class (§5.2.4, D82).

#### 5.2.4 Class and radius

- The term is **global** if any band makes it global, or the reach covers a horizontal or other axis. (No size check on the radius is needed: reduction keeps every horizontal offset in $[-\lfloor N/2 \rfloor, \lceil N/2 \rceil - 1]$, so `lower` + `upper` + 1 never exceeds the axis size $N$.)
- Otherwise **stencil** if some reduced horizontal offset is nonzero. Its radius on a horizontal axis is `[lower, upper]`: `lower` the largest $-o'$ over negative offsets on that axis (cells toward lower index), `upper` the largest $o'$ over positive ones, each 0 if there are none.
- Otherwise **column** if some vertical offset is nonzero or the reach covers the vertical axis.
- Otherwise **pointwise**. A term with no bands is pointwise.

A nonzero offset on an other axis therefore leaves the class as the horizontal and vertical offsets give it: such a coupling stays inside one cell's block, since an other axis is never divided among shards, tiles or ranks (D82).

An operator's class is the coarsest of its terms' in the order pointwise < column < stencil < global. A stencil operator's `radius` lists **every** horizontal axis, each direction the maximum over its stencil terms (0 where none reaches). Only stencil operators have `radius` (the schema checks this).

### 5.3 Pass 3: DOF sets

- `writes`: the dependent variables of the operator's terms.
- `reads`: every state, parameter and observed name the operator's terms reference, together with, transitively, every name referenced by the definitions of the observeds among them. `t` and index names are not names.

Both are sorted by code point (§2.1).

### 5.4 Pass 4: structure

The operator's **band list** is EarthSciASTDiff's `jacobian_bands` (states) of its right-hand-side document (§6.1), in EarthSciASTDiff's order (equation order, then sorted column variable, then canonical site key; the order its `.bands.json` goldens hold). Band $k$ (1-based) is band field `J_k` of the Jacobian document (§6.2). An EarthSciASTDiff refusal is `band_model_failed` at `/model`.

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

For a stencil leaf, only the stiffness of its column operators enters the choice: the stencil rows depend on whether a column operator is stiff, never on the stiffness of its stencil, pointwise or global operators. The block size is the number of `vars` in the union of the leaf's operators' `structure.block` variables. The pointwise stiff row applies only when the leaf's Jacobian storage is `block_diagonal`, whether or not the run config gives `linsolve` (spec/operator-interface.md §8.3: every operator has `structure.block`, and the leaf's state variables, the states its operators write, are all arrays of one shape or all scalars), since `block_lu` and `sparse_block_lu` need it; otherwise the leaf is refused, `auto_strategy_unavailable`. A `linsolve`, `jacobian` or `dt_scope` the run config gives explicitly is kept; only `"auto"` or an omitted key is resolved, and an omitted `linsolve` or `jacobian` on an explicitly named implicit method resolves as this table's row for the leaf's class would (pointwise `block_lu`/`sparse_block_lu`, column `banded`, otherwise `auto_strategy_unavailable`). `controller` `"auto"` is `PI`. Once a leaf's `"auto"` method is resolved here, its `precision_low_with_classical_rosenbrock` check (§7.5) runs.

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

In each model, `equations` lists, in the flattened model's equation order, the equations whose subject belongs to that model's component (for a state equation the dependent variable, for an observed definition the observed), followed by the `ic` equations of its states in `field_ics` order. Every reference is written as its flattened (fully scoped) name, on both sides; both EarthSciAST bindings (0.4 and 0.5) resolve such references from any model (D68).

- **State equation, not lifted.** `{"lhs": {"op": "D", "args": ["<x>"], "wrt": "t"}, "rhs": R}`, where R is the left fold `((t_1 + t_2) + t_3) + …` of this operator's terms of that equation in term order, written as nested binary `{"op": "+", "args": [a, b]}` (one term: the term itself). With no term: `0.0` for a scalar state, and `{"op": "makearray", "args": [], "regions": [[[1, n_1], …, [1, n_k]]], "values": [0.0]}` for an array of extents $n_1 \ldots n_k$.
- **State equation, lifted.** The same left-hand side, `D(x)` on the whole array, and R = `{"op": "faq", "args": [], "output_idx": O, "ranges": G, "expr": F}` with `O` copied from the flattened equation's right-hand-side `faq` node, `G` mapping the $d$-th name of `O` to `{"from": X_d}`, $X_d$ the index set of dimension $d$ of the state's shape (§6.1.3), and `F` the left fold of the per-cell terms; with no term, the `makearray` zero above. `G` is built, not copied, because the bindings write the flattened ranges differently: Julia's `flatten` writes `{"from": "cell"}`, the form above, while Rust's (and Python's) write the interval `[1, 4]`.
- **Observed definition** (a live observed): copied, left-hand side and right-hand side serialized as they stand in the flattened model.
- **Initial condition**: `{"lhs": {"op": "ic", "args": ["<x>"]}, "rhs": E}` for every `field_ics` entry of every state, whether or not the operator writes it, so every document builds the same $u_0$.

#### 6.1.3 Shapes of lifted states

A lifted state has extents but no declared shape (EarthSciAST's `lifted_shapes`). Its shape is the declared shape of the first non-lifted variable of the flattened model, searching `state_variables`, then `parameters`, then `observed_variables`, each in document order, whose declared shape has exactly those extents, matched index set by index set. A `shape` that `flatten` records on a lifted state itself is ignored: Julia's `flatten` writes one (`["cell"]`, even when no variable declares that shape), Rust's does not, so it cannot be relied on. If no variable matches: `lift_shape_unresolved` (D68; an upstream gap, §10), raised at §4 step 8, where the shapes are resolved.

#### 6.1.4 Name collisions

A model with a top-level component named `_jacobian` is refused at §4 step 7 (`reserved_component_name`), since §6.2 uses that name.

### 6.2 The Jacobian document `<operator>.jacobian.esm`

The operator's right-hand-side document (§6.1), changed as follows (D69):

1. Every state equation's right-hand side is replaced by the zero of §6.1.2 (`0.0`, or the `makearray` zero), whether or not the operator writes the state.
2. Observeds are live when referenced, directly or transitively, by a band coefficient, instead of by a term.
3. A model `_jacobian` is added to `models`, holding, for each band $k$ = 1, 2, … in band order, a variable `J_k` and an equation for it, exactly as EarthSciASTDiff's evaluation document (`_evaluation_document` in Julia, `evaluation_document` in Rust) builds its band field `J_k` with coefficient hoisting off (`cse = false`), except that each reference is written fully scoped and `J_k` itself is referenced as `_jacobian.J_k`: the variable is `{"type": "unknown", "default": 0.0}` plus, for an array row or a band with `contracted` dimensions, `"shape"` (the row variable's index sets followed by one index set per contracted dimension); the equation's left-hand side is `D(_jacobian.J_k)` and its right-hand side the band's coefficient over its region, zero elsewhere. Index sets for contracted dimensions are added to `index_sets` under EarthSciASTDiff's names.

Evaluating the document's right-hand side at $(u, p, t)$ gives every band coefficient in the `J_k` slots and zero in every state's slot. A runtime scatters `J_k` into its matrix storage with `structure.bands[k-1]` (spec/operator-interface.md §8).

## 7. Resolved settings

### 7.1 Schedule

The planner refuses the unsupported constructs of §0.1 in the schedule (`map`, `solve`, `split_explicit` and `coupled` with algebraic operators, and an observed in a `subcycle`'s `limits`) in tree order, then an observed in `step.limits`, entries in written order, then checks the run config against spec/run-config.md §8 rules 2, 3, 5–9, 12, 13 and 15 in rule order, with the codes of spec/planner-errors.json, and then writes the resolved tree. **Tree order** is depth first, a node before its children, children in written order. A name in a `subcycle`'s or `step`'s `limits` is first resolved (`name_unresolved`; `name_wrong_kind` when it is not an observed) and then refused, `unsupported_step_limit`, at its pointer (`/step/limits/<i>` for `step`'s).

A node's **pointer** is its JSON pointer in the run config: `/schedule` for the root, and below a node at pointer $P$, `P/sequence/<i>` or `P/strang/<i>` for the child at position $i$ (0-based), `P/subcycle` and `P/every` for the child of a `subcycle` or `every`. A refusal "at the node" or "at the leaf's pointer" uses it. Every node of an `"auto"` schedule (§5.5.5) is reported at `/schedule`, with no key appended, since the run config has no node there to point at.

**Rule 2 here** covers the names §5.1.3 has not already checked, in this order:
1. each operator's `bind_dt` keys (model parameters), operators in code-point order, keys in code-point order;
2. the parameter-name durations outside the schedule: `step.dt`, each `step.limits` entry's `max` in written order, `step.hit_multiples`, `step.resolution`, then `output.saveat`, then each `output.sinks` entry's `saveat`, sinks in written order;
3. per schedule node in tree order: its parameter-name durations, in the order `dt`, `control.dt0`, `dtmin`, `dtmax`, `period`, `phase`, then each `limits[i].max` in order, then each `frozen[i]`'s `period` and `phase` in order; then its `bind_dt` keys, its per-variable `control.abstol` keys and its `domain.nonnegative` entries (states), object keys in code-point order and list entries in written order;
4. the `precision` keys, in code-point order: each must be `default`, an operator name (named or an existing default operator) or a node id, else `name_unresolved` at `/precision/<key>`.

**Rule 3** (`operator_undefined`) covers every operator name the schedule names: per node in tree order, its `leaf` or `coupled` list, then its `frozen` entries, then its `imex` entries, `explicit` before `implicit`, each in written order (a locality name in `imex` is not an operator name). The pointer is the entry's: `P/leaf/<i>`, `P/coupled/<i>`, `P/frozen/<i>` (or `P/frozen/<i>/every`), and `P/imex/<part>/<i>`, or `P/leaf` and `P/imex/<part>` for a single string. Rule 6 therefore checks only partitions whose operators are defined.

**Rule 5** (D84): an operator is **advanced** at a place when a `leaf` or `coupled` node names it or a `frozen` entry of such a node names it. An operator advanced nowhere is `operator_not_advanced`, at `/operators/<op>`, or at `/schedule` for a default operator, which has no run-config key. An operator named by two `leaf` or `coupled` nodes is `operator_advanced_twice`, at the second such node's operator entry in tree order (`P/leaf/<i>` or `P/coupled/<i>`, `P/leaf` when the leaf names one operator as a string); a `strang` that runs a child twice is still one place, and a `frozen` entry does not count toward "twice".

**Rule 15's pointer** is the binding's: the leaf's `/…/bind_dt` when the leaf gives the `"leaf"` binding, and `/operators/<op>/bind_dt` when it comes from an operator's `bind_dt`.

**Rule 6** (`imex_partition_invalid`) is checked here, in rule order with the others, not when the leaf's settings are resolved, and only on leaves whose written `alg` is `IMEXEuler`. A bad partition is refused at the leaf's `P/imex`. **An `IMEXEuler` leaf without `imex`** is refused with the same code at the leaf's pointer: the schema does not require the key, and there is no default partition. A leaf whose `alg` is `"auto"` and resolves to `IMEXEuler` gets its partition from §5.5.3 and is not checked here. An `imex` key on a leaf whose written `alg` is not `IMEXEuler` (`"auto"` included) is ignored: it is not checked by rule 6 and does not appear in the resolved leaf.

**Rule 8** (`hold_tendency_child_not_leaf`) is reported at the `every` node's `P/every`.

For every node:

- **`id`**: the given id, else the node's path: `root` for the root; a child of a `sequence` or `strang` at position $i$ (0-based) is `<parent path>.<i>`; the child of a `subcycle` is `<parent path>.subcycle`, of an `every` `<parent path>.every`. Paths are built from positions alone, whatever ids the ancestors have, so the unnamed first child of a node `transport` at `root.0` is `root.0.0`. A run config may write a path-form id only on the node whose path it is (`node_id_reserved` otherwise), so given and assigned ids never collide; the schema accepts both forms.
- **`leaf`, `coupled`**: always a list of operator names, in the order given (accumulation order, spec/operator-interface.md §3).
- **Durations** (`dt` when numeric, `dt0`, `dtmin`, `dtmax`, `period`, `phase`, `{"max": …}`) are written in seconds as numbers; a parameter-name duration is the parameter's default (spec/run-config.md §2.2). A `"<number> <unit>"` string is the number parsed as the nearest binary64, times the unit's seconds (an exact integer: 1, 60, 3600, 86400), one rounding in all; so `"1.1 h"` is the binary64 product of 1.1 and 3600, `3960.0000000000005`, not 3960. `phase` 0 is written `0`. A parameter-name duration whose default is not a positive finite number is `run_config_invalid`, at the pointer of the duration's key (spec/run-config.md §2.2: a duration is positive, which the schema can check only for literals).

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

`control` holds `reltol` (default 1e-3) and `abstol` (default 1e-6; a per-variable object as given) always; `dt_scope` always (§5.5.3; an explicit method's leaf takes the table's `dt_scope` for its class); and, for adaptive leaves only (no `dt`), `controller` (default `PI`) and `maxiters` (default 1000000), plus `dt0`, `dtmin` and `dtmax` when given (`dtmin`'s default, 0, is not a duration the schema admits, and means no minimum). A leaf with a `dt` (numeric, `"step"` or `"accumulated"`) has no `controller`, `dt0`, `dtmin` or `dtmax` in the resolved leaf, even when the run config gives them. A fixed-`dt` leaf's `maxiters` is written only when given (spec/methods.md §12: unlimited otherwise). A leaf's `partition` key does not appear in the resolved leaf; it lands in `partition.shards` (§7.4).

Other nodes keep their keys, resolved recursively: `sequence`, `strang` (children), `subcycle` (`n` or `limits` with `{"max"}` in seconds), `every` (`period`, `phase` (default 0), `hold` (default `none`)).

### 7.2 Time

The interval comes from the caller, the run config's `time`, or the model's `solver.time_span`, the first that gives one giving all of it (D60). Numbers are seconds. A timestamp, from any source, is converted exactly as esm-spec §2.2.7 converts a `time_span` timestamp: seconds from the model's time origin (`domain.temporal.reference_time`, else `domain.temporal.start`, else the POSIX epoch) as $(W - W_0) + (f - f_0)$ (D74). The model's `solver.time_span` is read with EarthSciAST's `resolve_time_span`. `duration` is added to the start in binary64. Refusals: no interval (`time_interval_missing`), end not after start (`time_interval_empty`), a timestamp naming no calendar day (`time_invalid_date`), a timestamp when the model's origin field is unreadable (`time_origin_invalid`, at the key of the timestamp that needs the origin, such as `/time/start`; `""` for the caller's, `/model` for the model's `solver.time_span`) (spec/run-config.md §8 rule 14). A refusal about an interval the caller gave has no run-config key and is reported at the empty pointer `""`. A model `solver.time_span` that EarthSciAST rejects is refused at `/model` with the code of its cause (§4 step 3). The plan's `time` is `{"start": t_0, "end": t_f}`.

### 7.3 Step

`step` is written with `dt` (seconds) or `limits` (each `{"max": seconds}`; an observed limit was refused in §7.1, `unsupported_step_limit`) when given, `hit` always (sorted; `[]` by default), and `hit_multiples` and `resolution` (seconds) when given. With neither `dt` nor `limits` the root must be a single adaptive `leaf` or `coupled` node (`step_control_missing`, rule 12). Rule 9 (`tstop_off_fixed_step`) is checked for the tstops M2 can know: `hit_multiples` (its multiples $k\,m$ from model time zero in $(t_0, t_f]$, each the binary64 product) and the adjoint's observation times. A tstop $\tau$ is **on the fixed step** when it falls on a macro-step boundary of spec/run-config.md §5, tested exactly as follows in binary64, with $\lambda(a, b)$ meaning $|a - b| \le 100\,\epsilon(\max(|a|, |b|))$ (spec/methods.md §10's landing tolerance, $\epsilon(x)$ the spacing of binary64 at $x$): $\lambda(\tau, t_f)$; or, with $r$ the integer nearest $(\tau - t_0)/\mathtt{dt}$ (either on a tie), some $k \in \{r - 1, r, r + 1\}$ with $k \ge 1$ has $b = t_0 + k \cdot \mathtt{dt}$ (product, then sum) with $\lambda(\tau, b)$, and the macro step from $t_0 + (k - 1) \cdot \mathtt{dt}$ (just $t_0$ for $k = 1$) is not a stop step by spec/methods.md §10 with $h = \mathtt{dt}$ and $t_{\text{stop}} = t_f$. Otherwise it is refused, at the key that gave it: `/step/hit_multiples`, or `/adjoint/functional/over` for an adjoint observation time. The observation times are $t_f$ for `over: "final"`, none for `over: "time"`, and for `over: "saveat"` the output times of spec/run-config.md §7.3: with `output.saveat` a duration $d$, the multiples $k\,d$ ($k$ an integer, counted from model time zero, the time origin of D74, each the binary64 product) taken as follows: a multiple with $\lambda(k\,d, t_f)$ is the output time $t_f$, written as $t_f$; a multiple with $\lambda(k\,d, t_0)$ is excluded, like $t_0$ itself; any other multiple is kept when it lies in $(t_0, t_f]$; $t_f$ is not added when no multiple lands on it. Without `saveat`, $t_f$ alone (D83). So $d = 0.1$ on $[0, 0.3]$ gives 0.1, 0.2 and 0.3: $3 \times 0.1$ is `0.30000000000000004`, which lands on $t_f$ and is written 0.3.

### 7.4 Partition

`partition` (D77):
- `ranks`, `tiles`: as given, default `[1, 1]`. Entry 0 divides `axes.horizontal[0]`, entry 1 `axes.horizontal[1]`. With fewer horizontal axes, the entries for missing axes must be 1 (`partition_axis_missing`, at `/partition`); `ranks[a] · tiles[a]` above the axis size is `partition_too_fine`.
- `halo`: for each horizontal axis, the largest `lower` or `upper` radius over all stencil operators, 0 when none; absent when there is no horizontal axis.
- `shards`: for every pointwise or column leaf, by node id: the leaf's own `partition.shards`, else its `shard_size` turned into $\lceil \text{cells} / \text{shard\_size} \rceil$, else the run config's `partition.shards` entry for the leaf's node id, else for one of its operators (all of a leaf's operator entries must agree, `partition_conflict`, reported at the first operator entry, in the leaf's operator order, that disagrees with the leaf's first entry, `/partition/shards/<op>`), else 1. `"auto"` is 1 at M2. With no pointwise or column leaf, `shards` is `{}`. **Cells** is the largest, over the states the leaf's operators write, of the product of the sizes of the state's dimensions on horizontal axes and, for a pointwise leaf only, on the vertical axis (a column leaf's shards hold whole columns); a dimension on an other axis is not counted, since it stays inside a cell (D82), and a state with no such dimension counts 1. So a column leaf writing `A[lon, lat, lev]` and `B[lon, lat, species]` has $|\text{lon}| \cdot |\text{lat}|$ cells. "Leaf" here means a `leaf` node only: a `coupled` node gets no `shards` entry, unlike §7.5, which names `coupled` nodes explicitly. The schema admits no `partition` key on a `coupled` node, and a `partition.shards` entry whose key is a `coupled` node's id, or an operator that a `coupled` node names, is `run_config_invalid` at `/partition/shards/<key>`, since it could shard nothing; these entries are checked in code-point order of the key, before the rest of `shards`.
- `output_shard`: absent at M2.

### 7.5 Precision

For every leaf and `coupled` node, by id, all five roles (spec/run-config.md §7.1): the node-id entry overrides the leaf's operators' entries, which override `default`, role by role; operator entries of one leaf that disagree on a role are `precision_conflict`, reported at the first operator entry, in the leaf's operator order, that disagrees on that role with the leaf's first entry setting it, `/precision/<op>` (the reading of `partition_conflict`, §7.4). Unset roles: `state` is the declared `element_type` that governs the leaf's states, the states its operators write (esm-spec §11.3.1), else `f64`; `rhs` is `state`; `jacobian` and `factors` are `rhs`; `tables` is `f64` (no forcing tables at M2). Refusals: rule 10 (`precision_contradicts_declaration`, `precision_rhs_below_state`), and a `jacobian` or `factors` role below `rhs` on a Rosenbrock method other than `Rosenbrock23` (architecture §8; `precision_low_with_classical_rosenbrock`). A leaf whose states declare different `element_type`s is `precision_contradicts_declaration`, whatever the roles, since no `state` role satisfies them all. A precision key that names no operator, node or `default` is `name_unresolved`, checked with rule 2 in §7.1, not ignored.

A precision refusal is reported at the precision entry that set the offending role, `/precision/<key>` (for a role left unset, the entry that set the role it defaults from; for mixed declarations with no entry setting `state`, the leaf's pointer, §7.1). For a leaf whose `alg` is given, the classical-Rosenbrock check runs here; for a leaf whose `alg` is `"auto"`, it runs after §5.5.3 resolves the method (§4 "Order of refusals").

### 7.6 Checkpoint schedule

Present exactly when the run config has `adjoint`: `{"kind": "all", "macro_steps": 0}`, a placeholder that M4 replaces (D34, D78). The adjoint rules 11 (`wrt_matches_nothing`, `wrt_bound_parameter`) are checked. At M2 the `wrt` prefixes range over: `params:` the scalar parameters (`params:numeric`: those with a numeric default); `fields:` the array parameters; `forcing:` nothing, since M2 plans no forcing, so a `forcing:` entry is `wrt_matches_nothing`; `u0` the state.

### 7.7 Inputs and derived

- `inputs.model.path` is the run config's `model` string as written; `inputs.run_config.path` is the run config's file name without directories. Hashes are of the files' bytes.
- `derived` lists `derived/<operator>.rhs.esm` and `derived/<operator>.jacobian.esm` for every operator, each with `kind`, `operator` and the hash of the bytes written.

## 8. Pass 8: validation

The checks run after the derived documents are written to memory and before anything is written to disk. A failing check is a refusal (`validation_rhs_sum_failed`, `validation_band_model_failed`, `validation_empty_operator`) carrying the measured quantity.

### 8.1 Checks

- **`rhs_sum`** (one entry, no operator): at each sampled state, every operator's right-hand-side document and the flattened model are built with EarthSciAST and evaluated; for every DOF, $|\sum_k f_k - f| \le 10^{-12} \max(|f|, \sum_k |f_k|)$, an exact zero passing (D76).
- **`band_model`** (per operator): at each sampled state, the band Jacobian $J^B$ assembled from the Jacobian document's fields and `structure.bands` is compared with an independent Jacobian of the operator's right-hand-side document, rows of `writes` only: ForwardDiff in Julia, central finite differences in Rust (§8.2). The check also confirms the band pattern is a structural superset of the true dependence.
- **`non_empty`** (per operator): the operator has at least one term, and its band list is non-empty when any of its terms references a state.

`validation` lists `rhs_sum` first, then `band_model` and `non_empty` for each operator in code-point order of the operator name.

### 8.2 The independent Jacobian

Each planner validates on its own, in production and in conformance runs alike (D8, D76). $M_r$ below is the largest magnitude in row $r$ of either Jacobian.

**Julia: ForwardDiff.** $J^I$ is ForwardDiff's Jacobian of EarthSciAST's `f!` built from the operator's right-hand-side document. For every entry, $|J^B_{rc} - J^I_{rc}| \le 10^{-10} M_r$, and every $(r, c)$ with $J^I_{rc} \ne 0$ is in the band pattern. This is the oracle architecture §3.2 pass 8 names; EarthSciASTDiff's own oracle lives only in its tests.

**Rust: central finite differences.** Rust has no automatic differentiation, so it differences the operator's right-hand side, evaluated in f64 through EarthSciAST's public reentrant Rust right-hand side (`RhsProgram`, U4, EarthSciAST 0.5.0) built from the right-hand-side document:

- **Step.** For state component $n$ at sampled state $s$, $h_n = \varepsilon^{1/3} \max(|u_s[n]|, |b_n|)$, with $\varepsilon = 2^{-52}$ ($\varepsilon^{1/3} \approx 6.06 \times 10^{-6}$) and $b_n$ the sampling base of §5.5.1, so the step scales with the component's own magnitude and a species near $10^{-20}$ gets a step near $10^{-25}$, not $10^{-6}$. The step actually taken is $\tilde h_n = (u_s[n] + h_n) - u_s[n]$ in binary64, and the backward point is $u_s[n] - \tilde h_n$, so the difference quotient divides by the exact distance.
- **Grouping.** Columns are differenced together when no row has band-pattern entries in two of them: the planner colours the columns greedily in canonical order, each taking the lowest colour none of whose columns shares a pattern row with it (Curtis, Powell and Reid). For each colour, every column $c$ in it is moved by $+\tilde h_c$, then by $-\tilde h_c$; for a row $r$ whose pattern has column $c$ in that colour, $J^I_{rc} = (f_r^+ - f_r^-)/(2\tilde h_c)$. A block-diagonal operator needs one colour per species, a stencil one per species per footprint offset, so the cost is two right-hand sides per colour, not per DOF.
- **Tolerance.** For every pattern entry, $|J^B_{rc} - J^I_{rc}| \le \tau_{rc}$ with $\tau_{rc} = 10^{-6} M_r + 10^{3}\,\varepsilon\, F_r / \tilde h_c$, where $F_r = \max(|f_r^+|, |f_r^-|)$ for $c$'s colour. The central difference's truncation error is about $h^2 |f'''|/6$ and its rounding error about $\varepsilon |f| / h$; with $h \propto \varepsilon^{1/3}$ both are near $\varepsilon^{2/3} \approx 4 \times 10^{-11}$ relative to a row whose terms scale with the state, so $10^{-6}$ leaves five orders of magnitude for curvature and for cancellation inside a right-hand side (chemistry production and loss terms that nearly balance). The second term is the floor for a row whose entries are all near zero, where only rounding in $f_r^\pm$ remains; $10^3 \varepsilon$ allows for the operation count of a long right-hand side.
- **Pattern.** For a row $r$ with no pattern column in a colour, the difference $(f_r^+ - f_r^-)/2$ must stay within the floor $10^{3} \varepsilon F_r$ (no step to divide by); larger means the operator depends on a column the bands omit, and the check fails. A dependence outside the pattern in a row that does have a pattern column in that colour corrupts that entry's $J^I$ and fails its tolerance instead. Either way an incomplete pattern is caught, as the ForwardDiff check catches it in Julia.

**Rust in the conformance suite.** In tier 1 the Rust planner additionally compares its band Jacobian with the Julia planner's values, so the cross-language check stays exact: the golden directory carries `validation/<operator>.bandvalues.json` for every operator (§9), `{"samples": [{"t": t_s, "u": {"<name>[i,…]": value, …}, "entries": [["<row element>", "<column element>", value], …]}, …]}` in the style of EarthSciASTDiff's `.jvals.json` (element names as EarthSciAST writes them, entries sorted by row then column name), written by the Julia planner from its ForwardDiff values. Rust evaluates its band Jacobian at the file's `u` and `t`, requires the same structural key set, and applies the ForwardDiff tolerance $10^{-10} M_r$. This comparison belongs to the conformance harness, not to the planner: a missing file is a harness error, never a planner refusal, and production plans need no file.

## 9. Conformance

Tier 1 (architecture §12): for each fixture case under `tests/conformance/plans/fixtures/` (layout in `tests/conformance/README.md`), both planners write `plan.json` and `derived/*.esm`, and both are compared byte for byte with the checked-in golden, and the harness compares the Rust planner's band values with the golden's `validation/*.bandvalues.json` (§8.2). Goldens are written by the Julia planner and reviewed by hand against the case's `expected.md` before they are checked in; the Rust planner must then match them. Refusal fixtures under `tests/conformance/plans/refusals/` must be refused with the listed code at the listed pointer.

Tier 0 (spec/README.md) checks the fixtures' structure: every `.esm` loads, validates and flattens with EarthSciAST (the Python binding, 0.6.0; §0.2), every run config validates, every refusal code is listed in spec/planner-errors.json, and every case has an `expected.md` entry. Until M2's goldens exist, it also checks that the schema accepts `tests/conformance/plans/valid/` and rejects `tests/conformance/plans/invalid/`.

## 10. Upstream gaps found while specifying M2

Items 1–6 are recorded here, not filed (the RFC process files them); items 7, 8, 11 and 12 are filed as issues. Each item's status was rechecked on 2026-10-26 against EarthSciAST 0.5.0 and EarthSciASTDiff `main` (after EarthSciASTDiff#13); items 4, 7, 8, 11 and 12 again on 2026-10-30, against EarthSciAST `main` and EarthSciASTDiff 0.1.0:

1. **Lifted states have no index-set names.** EarthSciAST's `lifted_shapes` records a lifted state's extents but not which index sets they are; §6.1.3's search for a matching declared shape works around it and fails for a lift onto a grid no variable declares. Asking: record the lift's index sets. *Still open in 0.5.0:* esm-libraries-spec §4.7.5 still defines `lifted_shapes` as post-lift extents only. The bindings also differ beyond it: Julia's `flatten` writes index-set names as a `shape` on the lifted state (`["cell"]`, even where no variable declares that shape) and writes the lifted `faq`'s ranges as `{"from": "cell"}`; Rust's (checked in the 0.5.0 crate source, `apply_pointwise_lift`) and Python's write no `shape` and write the ranges as `[1, 4]`. §6.1.3 ignores the recorded shape and §6.1.2 builds the ranges, so neither difference reaches a plan.
2. **EarthSciASTDiff's evaluation document for a flattened system does not load.** It declares dotted names (`SimpleOzone.O3`) in one model, which EarthSciAST 0.4.0 reads as references to an undefined system (checked in Python and Julia). §6.2 builds its own tree for this reason. The Julia builder is also internal (`_evaluation_document`). *Still open:* the trimmed evaluation document (EarthSciASTDiff#8) still declares every name in one model `JacobianEval`; for the flattened `coupled_chem_advection` fixture, EarthSciAST 0.5.0's Python binding refuses it (`reference 'SimpleOzone.O3' to undefined system 'SimpleOzone'`), and Julia loads and flattens it but refuses to build its right-hand side (`E_TREEWALK_UNKNOWN_STATE: SimpleOzone.O3`).
3. **Julia's `build_evaluator(::EsmFile)` refuses a model's scoped references to its own variables** (`E_TREEWALK_UNBOUND_VARIABLE` for `Decay.k` inside model `Decay`) where flattening and Python accept them. *Not blocking:* still open in 0.5.0, but `build_evaluator` is deprecated since 0.4.0 and runtimes do not use it. They build derived documents with `rhs_program(esm_problem(...))` (spec/operator-interface.md §8.1), and `esm_problem` accepts such references in 0.4.0 and 0.5.0. Nothing to ask upstream.
4. **No specified time derivative from EarthSciASTDiff.** Its `main` computes bands with respect to `t` (`wrt = :time`) in Julia, Rust and Python, but `esm-jacobian-spec.md` still lists time gradients as out of scope and no golden gates them across bindings. So M2 emits no time-derivative document, planned operators cannot provide `ft`, and Rosenbrock leaves on time-dependent planned operators wait (D80). Promoted to upstream ask U17 (docs/rfcs/U17-time-derivative-bands.md), needed by M3. *Resolved upstream* on EarthSciASTDiff `main` by EarthSciASTDiff#13 (`esm-jacobian-spec.md` §4.2 and §8, `tgrad_photolysis` goldens in Julia, Rust and Python), released in EarthSciASTDiff 0.1.0 (commit 743d140). M2's scope is unchanged: it still emits no time-derivative document (D80). M3 adds it, with the planner's numerical check of time bands (D76 covers state Jacobians only).
5. **Julia's native compiler declines a contracting `faq` inside a lifted equation** (`compiler_refused_rule`; the interpreter evaluates it). Derived documents write a lifted equation as a whole-array `faq` whose body is the per-cell terms (§6.1.2, D68), so a lifted equation whose terms contract keeps the contracting `faq` inside it, which Julia's native compiler declines. The column fixture states its equation explicitly (a `makearray` over a `faq`, not lifted), which the native compiler accepts; no fixture yet contracts inside a lifted equation. Asking: compile a contracting `faq` inside a lifted equation natively. *Not reproduced:* 0.5.0's notes list no change here, and a small hand-written whole-array `faq` holding a contracting `faq` builds natively in both 0.4.0 and 0.5.0, so the refusal needs the original lifted case to recheck.
6. **A `makearray` region with singleton axes** must give a value whose rank is the number of non-singleton axes in Julia, while Python accepts a value over all axes; fixtures avoid size-1 axes. *Not reproduced:* 0.5.0's notes list no change here, and a small `makearray` over a size-1 axis builds in Julia 0.4.0 and 0.5.0 with a value of either rank, so the failure needs the original case to recheck.

Found while implementing M2 stage 1 (Julia PR #21, Rust PR #20), rechecked against EarthSciAST 0.6.0 on 2026-10-27:

7. **Rust EarthSciAST has no public `table_lookup` lowering.** §4 step 5 lowers `table_lookup` to `fn` with EarthSciAST's own lowering, which Julia exports and Rust does not. The Rust planner stops with an internal not-implemented error on a model that keeps a `table_lookup` (not a refusal code). No M2 fixture has one. Asking: a public lowering in Rust. *Still open in 0.6.0:* the crate's `lower_table_lookup` module is crate-private. Filed as EarthSciAST#580; still open on 2026-10-30.
8. **EarthSciAST (Rust) does not compile without its `solve` feature**: its compiler-agreement adapter module is not gated on it. The Rust planner turns `solve` on although it uses no ODE solver. Asking: gate the module. *Still open in 0.6.0:* 0.6.0 adds a `conformance-adapters` feature that turns `solve` on, but `compiler_agreement_adapter` is still declared without a feature gate, so a build with `default-features = false` and no `solve` fails in that module (unresolved imports `problem::solve` and `problem::observed_trajectories`). Filed as EarthSciAST#581; still open on 2026-10-30.
9. **EarthSciAST.jl's weak-dependency compat `SciMLBase = "2"`** held SciMLBase, and with it OrdinaryDiffEqSDIRK, at old versions whenever both were in an environment, which changed an M1 result. So EarthSciIntegrator.jl loaded EarthSciAST as a package extension, not a dependency. *Resolved in 0.6.0:* EarthSciAST.jl accepts SciMLBase 2 and 3. EarthSciIntegrator.jl now depends on EarthSciAST directly, and its test environment, with EarthSciAST in it, resolves the same SciMLBase (3.58) and OrdinaryDiffEqSDIRK (2.9.8) as without it. The test environment's compat `OrdinaryDiffEqSDIRK = "2"` and `SciMLBase = "3"` keeps it so: a package that holds SciMLBase at 2 makes the environment fail to resolve rather than quietly change the M1 oracle.
10. **Reaction-system summands differ between the bindings.** For a reaction whose net stoichiometry for a species is $-1$, Julia's `flatten` writes the summand as a one-argument `-`, `{"op": "-", "args": [{"op": "*", "args": ["Chem.j", NO2[i]]}]}`; Rust's (`lower_reactions_to_equations` in the 0.5.0 crate source) writes `{"op": "*", "args": [-1.0, …]}` with a float literal, and Python's `{"op": "*", "args": [-1, …]}`. The term count and classes agree, but derived documents copy terms (§6.1.2), so for a model with a reaction system the two planners' `derived/*.esm` cannot be byte-identical until the bindings agree. Asking: one form in every binding. *Resolved in 0.6.0:* esm-libraries-spec §4.6.2 fixes the form and every binding writes it. A species' term is the bare rate law for a net coefficient of 1 and otherwise the two-operand product `{"op": "*", "args": [c, rate]}`, `c` an integer literal when integral, so a loss of 1 is `-1 * rate`; a species' terms are one n-ary `+` in reaction order. Checked by flattening `pointwise_chemistry` and `reseact_shaped` with the Python binding and the Julia binding, and in the Rust crate source (`lower_reactions_to_equations`). Both fixtures' `expected.md` show this form.
11. **No public accessor for timestamp conversion in Julia.** §7.2 converts a caller's or run config's timestamp exactly as `resolve_time_span` converts a `solver.time_span` one, from the model's time origin. EarthSciAST.jl exposes no public function for either half, so the Julia planner calls its internals `_datetime_parts` and `_time_origin`, which may change in any release. Asking: a public accessor for a model's time origin and for converting a timestamp to model seconds. *Still open in 0.6.0:* the two functions and the pattern `_TIME_SPAN_TIMESTAMP_RE` are unchanged from 0.5.0 and still internal. Filed as EarthSciAST#582. *Partly resolved on EarthSciAST `main`, not yet released* (EarthSciAST#587, after 0.6.0): `time_origin(file)` and `timestamp_to_seconds(file, timestamp)` are public in every binding. They do not yet replace the internals: both raise the one code `solver_time_span_invalid` for a timestamp that names no calendar day and for an unreadable origin, where §7.2 refuses the first with `time_invalid_date` and the second with `time_origin_invalid`, and both take a loaded `EsmFile`, where the planner also checks a model's `solver.time_span` in the raw document before it loads (rule 14). Asking: a way to tell the two causes apart, and the same conversion for a raw `domain`.

Found while moving to EarthSciAST 0.6.0, on 2026-10-27:

12. **EarthSciASTDiff.jl does not accept EarthSciAST 0.6.** Its `main` has compat `EarthSciAST = "0.4, 0.5"` and, for its SciMLBase extension, `SciMLBase = "2"`, so it cannot share an environment with EarthSciIntegrator.jl, which needs EarthSciAST 0.6, and wherever it is loaded with SciMLBase it holds SciMLBase at 2, item 9's problem. Its Rust crate is on `earthsci-ast = "0.6"`. Stage 1 does not use EarthSciASTDiff, so nothing fails yet; the Julia planner's stage 2 backend needs it. Asking: compat EarthSciAST 0.6 and SciMLBase 2 and 3. Filed as EarthSciASTDiff#17. *Resolved* by EarthSciASTDiff#19 (compat `EarthSciAST = "0.4, 0.5, 0.6"`, `SciMLBase = "2, 3"`), released in EarthSciASTDiff 0.1.0.
