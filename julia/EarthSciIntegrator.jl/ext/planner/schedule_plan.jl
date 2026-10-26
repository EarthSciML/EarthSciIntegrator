# The schedule (spec/plan.md §7.1, §5.5.3–§5.5.5): the "auto" schedule,
# the checks of spec/run-config.md §8, the reference step, stiffness, the
# methods where alg is "auto", and the resolved tree.

const _NODE_KINDS = ("leaf", "coupled", "sequence", "strang", "subcycle", "every", "split_explicit", "map", "solve")
const _NODE_PATH_RE = r"^root(\.([0-9]+|subcycle|every))*\z"

"""
    SNode

A schedule node as the planner walks it: its run-config object, its kind,
its position path (`root`, `root.0`, `root.1.every`, spec/plan.md §7.1), the
JSON pointer of its object in the run config, its id, and its children. The
nodes of an `"auto"` schedule are the planner's (§5.5.5); every refusal about
them is reported at `/schedule`.
"""
mutable struct SNode
    spec::AbstractDict
    kind::String
    path::String
    ptr::String
    auto::Bool
    id::String
    children::Vector{SNode}
end

"""The pointer of a key of a node (`/schedule` itself for every node of an auto schedule)."""
nptr(n::SNode, toks...) = n.auto ? "/schedule" : jptr(n.ptr, toks...)

is_leaflike(n::SNode) = n.kind == "leaf" || n.kind == "coupled"
leaf_ops(n::SNode) = _as_list(n.spec[n.kind])

function build_snode(spec::AbstractDict, path::String, ptr::String, auto::Bool)
    kind = only(k for k in _NODE_KINDS if haskey(spec, k))
    kids = SNode[]
    if kind == "sequence" || kind == "strang"
        for (i, c) in enumerate(spec[kind])
            push!(kids, build_snode(c, "$path.$(i - 1)", jptr(ptr, kind, i - 1), auto))
        end
    elseif kind == "subcycle" || kind == "every"
        push!(kids, build_snode(spec[kind], "$path.$kind", jptr(ptr, kind), auto))
    end
    return SNode(spec, kind, path, ptr, auto, haskey(spec, "id") ? String(spec["id"]) : path, kids)
end

"""All nodes of the tree, parents before children (tree order)."""
function allnodes(n::SNode, acc = SNode[])
    push!(acc, n)
    foreach(c -> allnodes(c, acc), n.children)
    return acc
end

"""
    auto_schedule(ops, splitting) -> Dict

spec/plan.md §5.5.5 (D21, D72): one leaf per non-empty locality class, its
operators sorted by name, in the order global, stencil, column, pointwise,
every option `"auto"` (omitted); one leaf is the root, two or more a
`sequence`, or a `strang` when the model's `solver.splitting` is `"strang"`.
"""
function auto_schedule(ops::Dict{String, PlanOperator}, splitting)
    leaves = Any[]
    for c in (:global, :stencil, :column, :pointwise)
        names = sort!([n for (n, op) in ops if need_locality(op) === c])
        isempty(names) || push!(leaves, Dict{String, Any}("leaf" => names, "alg" => "auto"))
    end
    length(leaves) == 1 && return leaves[1]
    return Dict{String, Any}((splitting == "strang" ? "strang" : "sequence") => leaves)
end

# ---------------------------------------------------------------------------
# Checks (spec/plan.md §7.1)

"""Operator references of a node, with their pointers: the leaf list, `frozen` entries, and `imex` entries that are not locality names."""
function operator_refs(n::SNode)
    refs = Tuple{String, String}[]
    is_leaflike(n) || return refs
    v = n.spec[n.kind]
    if v isa AbstractString
        push!(refs, (String(v), nptr(n, n.kind)))
    else
        for (i, x) in enumerate(v)
            push!(refs, (String(x), nptr(n, n.kind, i - 1)))
        end
    end
    return refs
end

function frozen_refs(n::SNode)
    refs = Tuple{String, String}[]
    haskey(n.spec, "frozen") || return refs
    for (i, f) in enumerate(n.spec["frozen"])
        f isa AbstractString ? push!(refs, (String(f), nptr(n, "frozen", i - 1))) :
                               push!(refs, (String(f["every"]), nptr(n, "frozen", i - 1, "every")))
    end
    return refs
end

function imex_refs(n::SNode)
    refs = Tuple{String, String}[]
    haskey(n.spec, "imex") || return refs
    for part in ("explicit", "implicit")
        haskey(n.spec["imex"], part) || continue
        v = n.spec["imex"][part]
        items = v isa AbstractString ? [(String(v), nptr(n, "imex", part))] :
                [(String(x), nptr(n, "imex", part, i - 1)) for (i, x) in enumerate(v)]
        for (x, p) in items
            Symbol(x) in LOCALITY_ORDER || push!(refs, (x, p))
        end
    end
    return refs
end

"""The bound parameters of a leaf: its own `bind_dt` and its operators'."""
function leaf_bindings(n::SNode, ops)
    b = Tuple{String, String, String}[]       # (parameter, kind, pointer)
    if haskey(n.spec, "bind_dt")
        for (p, k) in n.spec["bind_dt"]
            push!(b, (String(p), String(k), nptr(n, "bind_dt")))
        end
    end
    for o in leaf_ops(n)
        op = get(ops, o, nothing)
        (op === nothing || op.bind_dt === nothing) && continue
        for (p, k) in op.bind_dt
            push!(b, (String(p), String(k), jptr(op.pointer, "bind_dt")))
        end
    end
    return b
end

"""
    expand_imex(n, ops) -> (explicit, implicit)

A leaf's `imex` partition as operator-name lists in leaf order, locality
names expanded to the leaf's operators of that class (spec/plan.md §7.1),
`explicit` omitted meaning every other operator of the leaf;
`imex_partition_invalid` (rule 6) when an operator of the leaf is in neither
part or in both, an entry names an operator outside the leaf, or the
implicit part is empty.
"""
function expand_imex(n::SNode, ops)
    names = leaf_ops(n)
    spec = n.spec["imex"]
    function part(key)
        haskey(spec, key) || return nothing
        out = String[]
        for x in _as_list(spec[key])
            if Symbol(x) in LOCALITY_ORDER
                append!(out, [o for o in names if need_locality(ops[o]) === Symbol(x)])
            else
                x in names || refuse("imex_partition_invalid", nptr(n, "imex"), "imex names $x, which is not an operator of the leaf")
                push!(out, x)
            end
        end
        return out
    end
    impl = part("implicit")
    expl = part("explicit")
    isempty(impl) && refuse("imex_partition_invalid", nptr(n, "imex"), "the imex implicit part is empty")
    expl === nothing && (expl = [o for o in names if !(o in impl)])
    for o in names
        k = count(==(o), impl) + count(==(o), expl)
        k == 1 || refuse("imex_partition_invalid", nptr(n, "imex"), "imex puts operator $o in $k parts; every operator of the leaf must be in exactly one")
    end
    return [o for o in names if o in expl], [o for o in names if o in impl]
end

"""
    tstops_known(t0, tf, step, m, config) -> Vector{Float64}

The tstops M2 can know (spec/plan.md §7.3): the multiples of
`hit_multiples` in `(t₀, t_f]`, counted from model time zero
(spec/run-config.md §1.1), and the adjoint's observation times.

Spec finding: an adjoint over `saveat` observes at the output times, which
come from `output.saveat`, a duration whose times the specs do not
enumerate (from zero or from `t₀`); they are not checked. `final` observes at
`t_f`, always a boundary.
"""
function tstops_known(m::PlanModel, t0, tf, config)
    step = get(config, "step", Dict{String, Any}())
    out = Float64[]
    if haskey(step, "hit_multiples")
        h = plan_duration(m, step["hit_multiples"], "/step/hit_multiples")
        j = floor(Int, t0 / h)
        while true
            τ = j * h
            τ > tf && break
            τ > t0 && push!(out, τ)
            j += 1
        end
    end
    return out
end

"""
    check_schedule(ctx, root)

spec/plan.md §7.1: unsupported nodes in tree order, then spec/run-config.md
§8 rules 3, 5–9, 12, 13 and 15 in rule order.
"""
function check_schedule(ctx, root::SNode)
    m, ops, config = ctx.model, ctx.ops, ctx.config
    nodes = allnodes(root)
    # Unsupported constructs, tree order (§0.1).
    for n in nodes
        n.kind == "map" && refuse("unsupported_map_operator", n.ptr, "map nodes need U6 (M4b)")
        n.kind == "solve" && refuse("unsupported_solve_node", n.ptr, "solve nodes need U6 (M4b)")
        n.kind == "split_explicit" && refuse("unsupported_split_explicit", n.ptr, "split_explicit arrives at M7")
        # (coupled with algebraic operators cannot occur: §4 step 6 refused algebraic equations.)
        if n.kind == "subcycle" && haskey(n.spec, "limits")
            # Spec finding: §7.1 lists map, solve, split_explicit and coupled DAE as
            # its unsupported nodes; an observed subcycle limit (§0.1) is refused
            # here too, in tree order, after its name is resolved (rule 2).
            for (i, l) in enumerate(n.spec["limits"])
                l isa AbstractString || continue
                at = nptr(n, "limits", i - 1)
                k = name_kind(m, l)
                k === nothing && refuse("name_unresolved", at, "step limit $l is not a name of the model")
                k === :observed || refuse("name_wrong_kind", at, "step limit $l is a $k, not an observed")
                refuse("unsupported_step_limit", at, "observed step limits need limit operators, U6 (M4b)")
            end
        end
    end
    # Rule 3: operator names.
    for n in nodes, (o, p) in vcat(operator_refs(n), frozen_refs(n), imex_refs(n))
        haskey(ops, o) || refuse("operator_undefined", p,
            "the schedule names operator $o, which is neither defined in the run config nor an existing default operator")
    end
    # Rule 5: every operator advanced exactly once. Spec finding: "advances twice
    # in one invocation outside strang" is read as "named by two places in the
    # tree" (a strang's repeated invocation of a child is one place); a frozen
    # entry advances its operator too.
    places = Dict{String, Vector{String}}()
    for n in nodes, (o, p) in vcat(operator_refs(n), frozen_refs(n))
        push!(get!(places, o, String[]), p)
    end
    for o in sort!(collect(keys(ops)))
        haskey(places, o) || refuse("operator_not_advanced", isempty(ops[o].pointer) ? "/schedule" : ops[o].pointer,
            "operator $o is advanced by no node of the schedule")
    end
    for n in nodes, (o, p) in vcat(operator_refs(n), frozen_refs(n))
        p != places[o][1] && refuse("operator_advanced_twice", p, "operator $o is advanced twice in one invocation")
    end
    # Rule 6: imex partitions.
    for n in nodes
        is_leaflike(n) && haskey(n.spec, "imex") && expand_imex(n, ops)
    end
    # Rule 7: node ids.
    seen = Set{String}()
    for n in nodes
        if haskey(n.spec, "id")
            id = String(n.spec["id"])
            occursin(_NODE_PATH_RE, id) && id != n.path && refuse("node_id_reserved", nptr(n, "id"),
                "id $id has the form of a node path, and this node's path is $(n.path)")
            id in seen && refuse("duplicate_node_id", nptr(n, "id"), "two nodes have the id $id")
        end
        push!(seen, n.id)
    end
    # Rule 8: hold: tendency needs a single leaf child.
    for n in nodes
        n.kind == "every" && get(n.spec, "hold", "none") == "tendency" && n.children[1].kind != "leaf" &&
            refuse("hold_tendency_child_not_leaf", nptr(n, "every"), "an every with hold: tendency needs a single leaf as its child")
    end
    # Rule 9: tstops on a fixed step.dt.
    step = get(config, "step", Dict{String, Any}())
    if haskey(step, "dt")
        dt = plan_duration(m, step["dt"], "/step/dt")
        lands_inside = is_leaflike(root) && (!haskey(root.spec, "dt") || !(root.spec["dt"] in ("step", "accumulated")))
        if !lands_inside
            t0, tf = ctx.time
            τs = sort!(unique!(tstops_known(m, t0, tf, config)))
            if !isempty(τs)
                bounds, _ = macro_bounds(t0, tf, dt, τs)
                for τ in τs
                    τ in bounds || refuse("tstop_off_fixed_step", "/step/hit_multiples",
                        "tstop $τ s is not t₀ plus a multiple of step.dt = $dt s, and the schedule root cannot land on it")
                end
            end
        end
    end
    # Rule 12: step control.
    if !haskey(step, "dt") && !haskey(step, "limits")
        (is_leaflike(root) && !haskey(root.spec, "dt")) || refuse("step_control_missing", "/step",
            "step has neither dt nor limits, and the schedule root is not a single adaptive leaf or coupled node")
    end
    # Rule 13: adaptive leaves need an error estimate.
    for n in nodes
        is_leaflike(n) || continue
        alg = get(n.spec, "alg", "auto")
        (alg == "auto" || haskey(n.spec, "dt")) && continue
        has_error_estimate(TABLEAUS[alg]) || refuse("adaptive_without_error_estimate", nptr(n, "alg"),
            "$alg has no embedded error estimate, so the leaf must give dt (D52)")
    end
    # Rule 15: a "leaf" binding on an adaptive leaf needs control.dt0.
    for n in nodes
        is_leaflike(n) && !haskey(n.spec, "dt") || continue
        haskey(get(n.spec, "control", Dict{String, Any}()), "dt0") && continue
        for (p, k, at) in leaf_bindings(n, ops)
            k == "leaf" && refuse("bind_dt_leaf_without_dt0", at,
                "parameter $p is bound to the leaf step on an adaptive leaf without control.dt0 (spec/methods.md §9)")
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# The reference step (§5.5.4)

"""
    reference_steps(ctx, root) -> Dict{String, Float64}

Δ for every leaf and coupled node, by node id (spec/plan.md §5.5.4): the
root receives `step.dt` when fixed, else the smallest `{"max": d}` of
`step.limits`, else `t_f − t₀`; a `sequence` passes Δ on; a `strang` Δ/2 to
all children but the last, which gets Δ; a `subcycle` Δ/n, or with `limits`
the smaller of Δ and its smallest `{"max": d}`; an `every` its `period`. A
leaf's Δ is its numeric `dt` if it has one.
"""
function reference_steps(ctx, root::SNode)
    m = ctx.model
    step = get(ctx.config, "step", Dict{String, Any}())
    t0, tf = ctx.time
    maxes(ls, base) = [plan_duration(m, l["max"], jptr(base, i - 1, "max")) for (i, l) in enumerate(ls) if l isa AbstractDict]
    Δroot = if haskey(step, "dt")
        plan_duration(m, step["dt"], "/step/dt")
    elseif haskey(step, "limits") && !isempty(maxes(step["limits"], "/step/limits"))
        minimum(maxes(step["limits"], "/step/limits"))
    else
        tf - t0
    end
    out = Dict{String, Float64}()
    function walk(n::SNode, Δ)
        if is_leaflike(n)
            dt = get(n.spec, "dt", nothing)
            out[n.id] = (dt === nothing || dt in ("step", "accumulated")) ? Δ : plan_duration(m, dt, nptr(n, "dt"))
        elseif n.kind == "sequence"
            foreach(c -> walk(c, Δ), n.children)
        elseif n.kind == "strang"
            for (i, c) in enumerate(n.children)
                walk(c, i == length(n.children) ? Δ : Δ / 2)
            end
        elseif n.kind == "subcycle"
            if haskey(n.spec, "n")
                walk(n.children[1], Δ / n.spec["n"])
            else
                ms = maxes(n.spec["limits"], nptr(n, "limits"))
                walk(n.children[1], isempty(ms) ? Δ : min(Δ, minimum(ms)))
            end
        elseif n.kind == "every"
            walk(n.children[1], plan_duration(m, n.spec["period"], nptr(n, "period")))
        end
    end
    walk(root, Δroot)
    return out
end

# ---------------------------------------------------------------------------
# Stiffness (§5.5.2) and methods (§5.5.3)

const _FORMS_J = (:rosenbrock, :sdirk, :imex, :theta)

"""Whether a method forms a Jacobian (Rosenbrock, SDIRK, IMEXEuler, Theta): the stiff methods of §5.5.2 rule 1."""
forms_jacobian(alg::AbstractString) = TABLEAUS[alg].family in _FORMS_J

"""The leaf or coupled node that advances each operator (its leaf list or a frozen entry)."""
function advancing_leaf(root::SNode)
    out = Dict{String, SNode}()
    for n in allnodes(root)
        is_leaflike(n) || continue
        for (o, _) in vcat(operator_refs(n), frozen_refs(n))
            out[o] = n
        end
    end
    return out
end

"""
    operator_stiffness!(ctx, root, Δ)

spec/plan.md §5.5.2: per operator, `{class, source}` by the first rule that
applies: `run_config` (its leaf names a method), `reaction_systems` (a term
with a single component origin that is a reaction system), `solver_hint`
(pointwise and `solver.stiffness` `"high"`), `gershgorin` (`ρΔ ≥ 10`, the
bound from the backend).
"""
function operator_stiffness!(ctx, root::SNode, Δ::Dict{String, Float64})
    leaf_of = advancing_leaf(root)
    hint = ctx.model.file.solver === nothing ? nothing : ctx.model.file.solver.stiffness
    stiff = Dict{String, Tuple{Symbol, String}}()
    for o in sort!(collect(keys(ctx.ops)))
        op = ctx.ops[o]
        n = leaf_of[o]
        alg = get(n.spec, "alg", "auto")
        stiff[o] = if alg != "auto"
            (forms_jacobian(alg) ? :stiff : :nonstiff, "run_config")
        elseif any(t -> length(t.origins) == 1 && t.origins[1].component !== nothing &&
                        t.origins[1].component in ctx.model.reaction_systems, op.terms)
            (:stiff, "reaction_systems")
        elseif need_locality(op) === :pointwise && hint == "high"
            (:stiff, "solver_hint")
        else
            ρ = gershgorin_bound(ctx.backend, ctx, op)
            ρ === nothing && throw(PlannerStageError("the Gershgorin bound of operator $o needs its EarthSciASTDiff Jacobian document"))
            (ρ * Δ[n.id] >= 10 ? :stiff : :nonstiff, "gershgorin")
        end
    end
    return stiff
end

"""
    leaf_block_size(ctx, n, at) -> Int

The number of variables in the union of the leaf's operators' `structure.block`
variables (§5.5.3), when the leaf's Jacobian storage is `block_diagonal`:
every operator has `structure.block`, and the leaf's states (the states its
operators write; spec finding: §5.5.3 says "the leaf's state variables") are
all arrays of one shape or all scalars. Otherwise `auto_strategy_unavailable`.
"""
function leaf_block_size(ctx, n::SNode, at)
    vars = Set{String}()
    for o in leaf_ops(n)
        s = ctx.ops[o].structure
        s === nothing && throw(PlannerStageError("the block size of leaf $(n.id) needs operator $o's structure (EarthSciASTDiff)"))
        haskey(s, "block") || refuse("auto_strategy_unavailable", at,
            "operator $o has no structure.block, so leaf $(n.id) has no block_diagonal storage for block_lu")
        union!(vars, String.(s["block"]["vars"]))
    end
    written = sort!(unique!(reduce(vcat, (ctx.ops[o].writes for o in leaf_ops(n)); init = String[])))
    shapes = unique([get(ctx.model.shapes, s, String[]) for s in written])
    length(shapes) == 1 || refuse("auto_strategy_unavailable", at,
        "leaf $(n.id)'s states are neither all scalars nor all arrays of one shape, so it has no block_diagonal storage for block_lu")
    return length(vars)
end

_pointwise_linsolve(size::Int) = size <= 16 ? Dict{String, Any}("block_lu" => Dict{String, Any}("pivot" => true, "refine" => 0)) :
    Dict{String, Any}("sparse_block_lu" => Dict{String, Any}("refine" => 0))

"""The object form of a given `linsolve` with every option this spec defaults (§7.1)."""
function linsolve_object(ls)
    ls isa AbstractString && (ls = Dict{String, Any}(ls => Dict{String, Any}()))
    (k, v), = collect(ls)
    o = Dict{String, Any}(String(kk) => vv for (kk, vv) in v)
    if k == "block_lu"
        haskey(o, "pivot") || (o["pivot"] = true)
        haskey(o, "refine") || (o["refine"] = 0)
    elseif k == "sparse_block_lu"
        haskey(o, "refine") || (o["refine"] = 0)
    end
    return Dict{String, Any}(String(k) => o)
end

"""
    resolve_leaf(ctx, n, stiff) -> Dict

A leaf or coupled node written with exactly the keys of spec/plan.md §7.1,
`"auto"` resolved by §5.5.3.
"""
function resolve_leaf(ctx, n::SNode, stiff)
    m = ctx.model
    spec = n.spec
    names = leaf_ops(n)
    cls = reduce(coarser, (need_locality(ctx.ops[o]) for o in names))
    isstiff = any(o -> stiff[o][1] === :stiff, names)
    ctrl = get(spec, "control", Dict{String, Any}())
    given(d, k) = haskey(d, k) && d[k] != "auto"
    reltol = Float64(get(ctrl, "reltol", 1.0e-3))
    alg = get(spec, "alg", "auto")
    jac = given(spec, "jacobian") ? String(spec["jacobian"]) : nothing
    ls = given(spec, "linsolve") ? linsolve_object(spec["linsolve"]) : nothing
    imex = nothing
    at = nptr(n)
    if alg == "auto"
        if cls === :pointwise
            if isstiff
                alg = reltol >= 1e-3 ? "Rosenbrock23" : "Rodas4"
                jac === nothing && (jac = "symbolic")
                ls === nothing && (ls = _pointwise_linsolve(leaf_block_size(ctx, n, at)))
            else
                alg = "Tsit5"
            end
        elseif cls === :column
            if isstiff
                alg = "TRBDF2"
                jac === nothing && (jac = "symbolic")
                ls === nothing && (ls = Dict{String, Any}("banded" => Dict{String, Any}()))
            else
                alg = "Tsit5"
            end
        elseif cls === :stencil
            stiffcol = [o for o in names if need_locality(ctx.ops[o]) === :column && stiff[o][1] === :stiff]
            if isempty(stiffcol)
                alg = "SSPRK43"
            elseif haskey(spec, "dt")
                alg = "IMEXEuler"
                imex = ([o for o in names if !(o in stiffcol)], stiffcol)
                jac === nothing && (jac = "symbolic")
                ls === nothing && (ls = Dict{String, Any}("banded" => Dict{String, Any}()))
            else
                refuse("auto_strategy_unavailable", at,
                    "a stencil leaf with a stiff column operator and no dt has no M2 method (D71)")
            end
        else
            isstiff && refuse("auto_strategy_unavailable", at, "a stiff global leaf has no M2 method (D71)")
            alg = "SSPRK43"
        end
    end
    formsJ = forms_jacobian(alg)
    if formsJ
        jac === nothing && (jac = "symbolic")
        if ls === nothing
            ls = cls === :pointwise ? _pointwise_linsolve(leaf_block_size(ctx, n, at)) :
                 cls === :column ? Dict{String, Any}("banded" => Dict{String, Any}()) :
                 refuse("auto_strategy_unavailable", at,
                     "an implicit method on a $cls leaf with linsolve omitted has no M2 linear solver (D71)")
        end
    end
    adaptive = !haskey(spec, "dt")
    out = Dict{String, Any}("id" => n.id, n.kind => names, "alg" => alg)
    if haskey(spec, "dt")
        out["dt"] = spec["dt"] in ("step", "accumulated") ? spec["dt"] : plan_duration(m, spec["dt"], nptr(n, "dt"))
    end
    c = Dict{String, Any}("reltol" => reltol)
    at_ = get(ctrl, "abstol", 1.0e-6)
    c["abstol"] = at_ isa AbstractDict ? Dict{String, Any}(String(k) => v for (k, v) in at_) : at_
    c["dt_scope"] = given(ctrl, "dt_scope") ? ctrl["dt_scope"] : (cls === :pointwise ? "cell" : "global")
    if adaptive
        c["controller"] = given(ctrl, "controller") ? ctrl["controller"] : "PI"
        c["maxiters"] = get(ctrl, "maxiters", 1000000)
        for k in ("dt0", "dtmin", "dtmax")
            haskey(ctrl, k) && (c[k] = plan_duration(m, ctrl[k], nptr(n, "control", k)))
        end
    else
        # Spec finding: §7.1 lists controller, dt0, dtmin and dtmax for adaptive
        # leaves only; given on a fixed-dt leaf they are not written.
        haskey(ctrl, "maxiters") && (c["maxiters"] = ctrl["maxiters"])
    end
    out["control"] = c
    if formsJ
        out["jacobian"] = jac
        out["linsolve"] = ls
    end
    dom = get(spec, "domain", Dict{String, Any}())
    d = Dict{String, Any}("on_violation" => get(dom, "on_violation", "reject"), "check_stages" => get(dom, "check_stages", false))
    haskey(dom, "nonnegative") && (d["nonnegative"] = dom["nonnegative"])
    out["domain"] = d
    if alg == "IMEXEuler"
        # Spec finding: the schema does not require imex on an IMEXEuler leaf and
        # §7.1 gives no default partition; without one the implicit part is empty.
        imex === nothing && !haskey(spec, "imex") &&
            refuse("imex_partition_invalid", nptr(n), "IMEXEuler needs an imex partition with a non-empty implicit part")
        expl, impl = imex !== nothing ? imex : expand_imex(n, ctx.ops)
        out["imex"] = Dict{String, Any}("explicit" => expl, "implicit" => impl)
        out["implicit_solve"] = get(spec, "implicit_solve", "converged")
    end
    alg == "Theta" && (out["theta"] = spec["theta"])
    if haskey(spec, "frozen")
        out["frozen"] = Any[f isa AbstractString ? f : _frozen_entry(m, f, nptr(n, "frozen", i - 1))
                            for (i, f) in enumerate(spec["frozen"])]
    end
    haskey(spec, "bind_dt") && (out["bind_dt"] = Dict{String, Any}(String(k) => v for (k, v) in spec["bind_dt"]))
    return out
end

function _frozen_entry(m, f, at)
    e = Dict{String, Any}(String(k) => v for (k, v) in f)
    e["period"] = plan_duration(m, f["period"], jptr(at, "period"))
    if haskey(f, "phase")
        e["phase"] = f["phase"] == 0 ? 0 : plan_duration(m, f["phase"], jptr(at, "phase"))
    end
    return e
end

"""
    resolve_tree(ctx, n, stiff) -> Dict

The resolved schedule (spec/plan.md §7.1): every node with its `id`, leaves by
[`resolve_leaf`](@ref), `subcycle` with `n` or `limits` in seconds, `every`
with `period`, `phase` (default 0) and `hold` (default `none`).
"""
function resolve_tree(ctx, n::SNode, stiff)
    is_leaflike(n) && return resolve_leaf(ctx, n, stiff)
    out = Dict{String, Any}("id" => n.id)
    if n.kind == "sequence" || n.kind == "strang"
        out[n.kind] = Any[resolve_tree(ctx, c, stiff) for c in n.children]
    elseif n.kind == "subcycle"
        out["subcycle"] = resolve_tree(ctx, n.children[1], stiff)
        if haskey(n.spec, "n")
            out["n"] = n.spec["n"]
        else
            out["limits"] = Any[Dict{String, Any}("max" => plan_duration(ctx.model, l["max"], nptr(n, "limits", i - 1, "max")))
                                for (i, l) in enumerate(n.spec["limits"])]
        end
    elseif n.kind == "every"
        out["every"] = resolve_tree(ctx, n.children[1], stiff)
        out["period"] = plan_duration(ctx.model, n.spec["period"], nptr(n, "period"))
        ph = get(n.spec, "phase", 0)
        out["phase"] = ph == 0 ? 0 : plan_duration(ctx.model, ph, nptr(n, "phase"))
        out["hold"] = get(n.spec, "hold", "none")
    end
    return out
end
