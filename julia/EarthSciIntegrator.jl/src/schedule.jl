# Schedule nodes and their construction from a run config (spec/run-config.md §4).

"""A schedule node (spec/run-config.md §4)."""
abstract type Node end

"""How a leaf binds parameters to step sizes (spec/run-config.md §4.4): parameter index => :leaf, :node or :macro."""
const Bindings = Vector{Pair{Int, Symbol}}

"""
    LeafNode

A `leaf` (spec/run-config.md §4.2): the sum of its operators advanced by one
method, adaptively (`dt === nothing`) or with fixed substeps (`dt` a number,
`:step` or `:accumulated`).
"""
struct LeafNode{O <: OperatorSum, M <: AbstractMethod} <: Node
    id::String
    opnames::Vector{String}
    ops::O
    tab::Tableau
    method::M
    dt::Union{Nothing, Float64, Symbol}
    tol::Union{Nothing, Tolerances}
    dt0::Union{Nothing, Float64}
    dtmin::Float64
    dtmax::Union{Nothing, Float64}
    controller::ControllerKind
    maxiters::Int
    nonnegative::Vector{Int}
    on_violation::Symbol
    bindings::Bindings
end
is_adaptive(l::LeafNode) = l.dt === nothing

"""`sequence`: the children in order over the whole interval (Lie–Trotter)."""
struct SequenceNode <: Node
    id::String
    children::Vector{Node}
end

"""`strang`: `c₁(Δ/2) … cₙ₋₁(Δ/2) cₙ(Δ) cₙ₋₁(Δ/2) … c₁(Δ/2)`."""
struct StrangNode <: Node
    id::String
    children::Vector{Node}
end

"""`subcycle` with a fixed count: the child `n` times with `Δ/n`."""
struct SubcycleNode <: Node
    id::String
    child::Node
    n::Int
end

"""`map`: applies a map operator (spec/run-config.md §4.10, operator-interface.md §7)."""
struct MapNode{O <: AbstractMapOperator} <: Node
    id::String
    name::String
    op::O
    bindings::Bindings
end

children(::LeafNode) = Node[]
children(::MapNode) = Node[]
children(n::Union{SequenceNode, StrangNode}) = n.children
children(n::SubcycleNode) = Node[n.child]

"""All nodes of a tree, depth first, parents before children."""
function allnodes(n::Node, acc = Node[])
    push!(acc, n)
    for c in children(n)
        allnodes(c, acc)
    end
    return acc
end

nodekind(::LeafNode) = "leaf"
nodekind(::MapNode) = "map"
nodekind(::SequenceNode) = "sequence"
nodekind(::StrangNode) = "strang"
nodekind(::SubcycleNode) = "subcycle"

# ---------------------------------------------------------------------------
# Durations (spec/run-config.md §2.2)

const _UNITS = Dict("s" => 1.0, "min" => 60.0, "h" => 3600.0, "day" => 86400.0)

"""Seconds of a duration: a number, `"<number> <unit>"`, or a parameter name."""
function parse_duration(x, prob)
    x isa Real && return Float64(x)
    x isa AbstractString || throw(ConfigError("not a duration: $(repr(x))"))
    m = match(r"^([0-9.]+(?:[eE][-+]?[0-9]+)?) ?(s|min|h|day)$", x)
    if m !== nothing
        return parse(Float64, m.captures[1]) * _UNITS[m.captures[2]]
    end
    i = findfirst(==(x), prob.pnames)
    i === nothing && throw(ConfigError("duration '$x' is neither '<number> <unit>' nor a parameter"))
    return prob.p[i]
end

# ---------------------------------------------------------------------------
# Building nodes

const _LEAF_KEYS = Set(["leaf", "id", "alg", "theta", "implicit_solve", "dt", "control", "jacobian", "linsolve",
    "imex", "domain", "frozen", "partition", "bind_dt"])
const _COMBINATORS = ("leaf", "coupled", "sequence", "strang", "subcycle", "every", "split_explicit", "map", "solve")

function _bindings(spec, prob)
    b = Bindings()
    haskey(spec, "bind_dt") || return b
    for (name, kind) in spec["bind_dt"]
        i = findfirst(==(name), prob.pnames)
        i === nothing && throw(ConfigError("bind_dt names '$name', which is not a parameter of $(prob.name)"))
        kind in ("leaf", "node", "macro") || throw(ConfigError("bind_dt kind '$kind' (leaf, node or macro)"))
        push!(b, i => Symbol(kind))
    end
    return b
end

"""DOF indices named by a variable list (or `"all"`)."""
function _dofs(prob, names)
    names == "all" && return collect(1:prob.N)
    return sort!(unique!(reduce(vcat, (variable_dofs(prob, n) for n in names); init = Int[])))
end

"""
    build_node(spec, prob, path) -> Node

Build a schedule node from its run-config object; `path` is its position path
from the root. Node ids the run config gives are kept; the others are named by
the path (spec/plan.md §3): `root`, then `<path>.<i>` for the 0-based `i`th
child of a `sequence` or `strang` and `<path>.subcycle` for a `subcycle`'s
child. The path is built from positions alone, whatever ids the ancestors
have: the unnamed first child of a node `transport` at `root.0` is `root.0.0`.
"""
function build_node(spec::AbstractDict, prob::ToyProblem, path::String)
    kinds = [k for k in _COMBINATORS if haskey(spec, k)]
    length(kinds) == 1 || throw(ConfigError("schedule node at $path has $(length(kinds)) combinator keys"))
    kind = only(kinds)
    id = haskey(spec, "id") ? String(spec["id"]) : path
    if kind == "leaf"
        return build_leaf(spec, prob, id)
    elseif kind == "sequence" || kind == "strang"
        for k in keys(spec)
            k in ("id", kind) || throw(ConfigError("unknown key '$k' on $kind node $id"))
        end
        kids = Node[build_node(c, prob, "$path.$(i - 1)") for (i, c) in enumerate(spec[kind])]
        kind == "strang" && length(kids) < 2 && throw(ConfigError("strang node $id needs at least two children"))
        return kind == "sequence" ? SequenceNode(id, kids) : StrangNode(id, kids)
    elseif kind == "subcycle"
        haskey(spec, "limits") && throw(UnsupportedError("subcycle with limits (node $id): step limits need observed limit operators, which arrive with the planner"))
        haskey(spec, "n") || throw(ConfigError("subcycle node $id needs n or limits"))
        return SubcycleNode(id, build_node(spec["subcycle"], prob, "$path.subcycle"), Int(spec["n"]))
    elseif kind == "map"
        name = String(spec["map"])
        haskey(prob.maps, name) || throw(ConfigError("map node $id names '$name', which is not a map operator of $(prob.name)"))
        return MapNode(id, name, prob.maps[name], Bindings())
    else
        throw(UnsupportedError("schedule node '$kind' (node $id) is not implemented in this stage"))
    end
end

function build_leaf(spec::AbstractDict, prob::ToyProblem, id::String)
    for k in keys(spec)
        k in _LEAF_KEYS || throw(ConfigError("unknown key '$k' on leaf $id"))
    end
    for k in ("imex", "frozen", "implicit_solve")
        haskey(spec, k) && throw(UnsupportedError("leaf option '$k' (leaf $id) is not implemented in this stage"))
    end
    names = spec["leaf"] isa AbstractString ? [String(spec["leaf"])] : String.(spec["leaf"])
    ops = AbstractOperator[]
    for n in names
        haskey(prob.operators, n) || throw(ConfigError("leaf $id names operator '$n', which $(prob.name) does not define"))
        push!(ops, prob.operators[n])
    end
    alg = String(get(spec, "alg", "auto"))
    alg == "auto" && throw(UnsupportedError("alg \"auto\" (leaf $id) needs the planner's strategy rules (M2)"))
    haskey(TABLEAUS, alg) || throw(ConfigError("unknown alg '$alg' on leaf $id"))
    tab = TABLEAUS[alg]
    method = method_for(tab)   # refuses families not implemented yet
    opsum = OperatorSum(ops)
    for op in ops, call in (:rhs, :jvp_u, :vjp_u)
        provides(op, call) || throw(ConfigError("operator '$(opname(op))' of leaf $id does not provide $call"))
    end
    dt = if !haskey(spec, "dt")
        nothing
    elseif spec["dt"] == "step"
        :step
    elseif spec["dt"] == "accumulated"
        :accumulated
    else
        parse_duration(spec["dt"], prob)
    end
    if dt === nothing && !has_error_estimate(tab)
        throw(ConfigError("leaf $id: $alg has no error estimate, so the leaf must give dt (D52)"))
    end
    ctrl = get(spec, "control", Dict{String, Any}())
    for k in keys(ctrl)
        k in ("reltol", "abstol", "dt0", "dtmin", "dtmax", "controller", "dt_scope", "maxiters") ||
            throw(ConfigError("unknown control key '$k' on leaf $id"))
    end
    scope = get(ctrl, "dt_scope", "auto")
    scope in ("auto", "global") ||
        throw(UnsupportedError("dt_scope '$scope' (leaf $id): M1 runs dt_scope global only (spec/tape.md)"))
    # Defaults when control omits them are OrdinaryDiffEq's (reltol 1e-3, abstol 1e-6).
    tol = nothing
    if dt === nothing
        reltol = Float64(get(ctrl, "reltol", 1.0e-3))
        at = get(ctrl, "abstol", 1.0e-6)
        abstol = fill(at isa Real ? Float64(at) : 1.0e-6, prob.N)
        if at isa AbstractDict
            for (name, v) in at
                abstol[variable_dofs(prob, name)] .= Float64(v)
            end
        end
        tol = Tolerances(reltol, abstol, writes(opsum, prob.N))
    end
    controller = controller_kind(let c = get(ctrl, "controller", "auto"); c == "auto" ? "PI" : c end)
    maxiters = haskey(ctrl, "maxiters") ? Int(ctrl["maxiters"]) : (dt === nothing ? 1_000_000 : typemax(Int))
    dom = get(spec, "domain", nothing)
    nonneg = Int[]
    onv = :reject
    if dom !== nothing
        get(dom, "check_stages", false) && throw(UnsupportedError("domain.check_stages applies to Rosenbrock leaves only (leaf $id)"))
        haskey(dom, "nonnegative") && (nonneg = _dofs(prob, dom["nonnegative"]))
        onv = Symbol(get(dom, "on_violation", "reject"))
    end
    getd(k) = haskey(ctrl, k) ? parse_duration(ctrl[k], prob) : nothing
    return LeafNode(id, names, opsum, tab, method, dt, tol, getd("dt0"), something(getd("dtmin"), 0.0), getd("dtmax"),
        controller, maxiters, nonneg, onv, _bindings(spec, prob))
end

"""
    apply_bindings(p, bindings, Δleaf, Δnode, Δmacro) -> p′

`p` with every bound parameter set to its step (spec/run-config.md §4.4).
Returns `p` itself when nothing is bound.
"""
function apply_bindings(p, bindings::Bindings, Δleaf, Δnode, Δmacro)
    isempty(bindings) && return p
    q = copy(p)
    for (i, kind) in bindings
        q[i] = kind === :leaf ? Δleaf : kind === :node ? Δnode : Δmacro
    end
    return q
end
