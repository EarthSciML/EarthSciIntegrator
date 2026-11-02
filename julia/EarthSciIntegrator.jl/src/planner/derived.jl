# The derived documents (spec/plan.md §6, D68, D69): the right-hand-side
# document of a set of terms and the Jacobian document of an operator
# (EarthSciASTDiff's evaluation document), as JSON objects; their bytes are
# EarthSciASTDiff's `stable_json` (§2.2).

_ser(e::ASTExpr) = EarthSciAST.serialize_expression(e)

"""The zero right-hand side of a state (§6.1.2): `0.0` for a scalar, a `makearray` of `0.0` over the state's extents for an array."""
function zero_rhs(m::PlanModel, x::AbstractString)
    haskey(m.shapes, x) || return 0.0
    return Dict{String, Any}(
        "op" => "makearray", "args" => Any[],
        "regions" => Any[Any[Any[1, m.sizes[X]] for X in m.shapes[x]]], "values" => Any[0.0]
    )
end

"""The left fold `((t₁ + t₂) + t₃) + …` of serialized terms as nested binary `+` (§6.1.2)."""
fold_sum(xs::AbstractVector) = foldl((a, b) -> Dict{String, Any}("op" => "+", "args" => Any[a, b]), xs)

"""
The observed a flattened equation defines, or `nothing`: a bare left-hand
side `y`, or an indexed one, `index(y, …)` or `faq{index(y, …)}` (an arrayed
observed).
"""
defined_observed(m::PlanModel, eq::Equation) = defined_observed(m.flat, eq)
function defined_observed(flat::EarthSciAST.FlattenedSystem, eq::Equation)
    isobs(n) = haskey(flat.observed_variables, n)
    l = eq.lhs
    l isa VarExpr && return isobs(l.name) ? l.name : nothing
    l isa OpExpr || return nothing
    l.op == "faq" && l.expr_body isa OpExpr && (l = l.expr_body)
    (l.op == "index" && !isempty(l.args) && l.args[1] isa VarExpr && isobs(l.args[1].name)) || return nothing
    return l.args[1].name
end

"""Every right-hand side defining each observed of the flattened model, in equation order (`PlanModel.observed_rhs`)."""
function observed_definitions(flat::EarthSciAST.FlattenedSystem)
    defs = Dict{String, Vector{ASTExpr}}()
    for eq in flat.equations
        o = defined_observed(flat, eq)
        o === nothing || push!(get!(defs, o, ASTExpr[]), eq.rhs)
    end
    return defs
end

"""
    live_observeds(m, names) -> Set{String}

The observeds among `names`, with every observed the definitions of those
reference, transitively (§6.1.1 "live", §6.2 item 2).
"""
function live_observeds(m::PlanModel, names)
    defs = m.observed_rhs
    live = Set{String}()
    stack = String[n for n in names if is_observed(m, n)]
    while !isempty(stack)
        o = pop!(stack)
        o in live && continue
        push!(live, o)
        for r in get(defs, o, ASTExpr[]), n in EarthSciAST.free_variables(r)
            is_observed(m, n) && !(n in live) && push!(stack, n)
        end
    end
    return live
end

"""The object of a declared variable (§6.1.1): `type`, plus `shape` and `default` when present."""
function variable_object(m::PlanModel, name::AbstractString, v)
    d = Dict{String, Any}("type" => is_param(m, name) ? "parameter" : "unknown")
    haskey(m.shapes, name) && (d["shape"] = Any[m.shapes[name]...])
    v.default === nothing || (d["default"] = v.default)
    return d
end

"""The model of component path `path` in the tree `models`, created (with its parents) when missing."""
function component_model!(models::AbstractDict, path::AbstractString)
    segs = split(path, '.')
    node = get!(
        models, String(segs[1]),
        Dict{String, Any}("variables" => Dict{String, Any}(), "equations" => Any[])
    )
    for s in segs[2:end]
        subs = get!(node, "subsystems", Dict{String, Any}())
        node = get!(subs, String(s), Dict{String, Any}("variables" => Dict{String, Any}(), "equations" => Any[]))
    end
    return node
end

"""
    rhs_document(m, name, terms) -> Dict

§6.1: the right-hand-side document named `name` of the terms `terms` (in
term order): every index set, the domain, and the component tree declaring
every state and parameter and the live observeds, with each state's
equation the fold of its terms (zero when it has none), the live observeds'
definitions, and every `ic` equation.
"""
function rhs_document(m::PlanModel, name::AbstractString, terms::AbstractVector)
    flat = m.flat
    live = live_observeds(m, reduce(vcat, (t.refs for t in terms); init = String[]))
    models = Dict{String, Any}()
    for dict in (flat.state_variables, flat.parameters, flat.observed_variables), (vname, v) in dict
        (is_state(m, vname) || is_param(m, vname) || vname in live) || continue
        component_model!(models, component_of(vname))["variables"][last_segment(vname)] = variable_object(m, vname, v)
    end
    by_eq = Dict{Int, Vector{Term}}()
    for t in terms
        push!(get!(by_eq, t.equation, Term[]), t)
    end
    eqinfo = Dict(ei => (var, lifted) for (ei, var, lifted) in m.state_eqs)
    for (ei, eq) in enumerate(flat.equations)
        if haskey(eqinfo, ei)
            var, lifted = eqinfo[ei]
            ts = get(by_eq, ei, Term[])
            rhs = if isempty(ts)
                zero_rhs(m, var)
            elseif lifted
                O, G = lifted_ranges(m, ei, var)
                Dict{String, Any}(
                    "op" => "faq", "args" => Any[], "output_idx" => Any[O...], "ranges" => G,
                    "expr" => fold_sum(Any[_ser(t.expr) for t in ts])
                )
            else
                fold_sum(Any[_ser(t.expr) for t in ts])
            end
            lhs = Dict{String, Any}("op" => "D", "args" => Any[var], "wrt" => "t")
            push!(component_model!(models, component_of(var))["equations"], Dict{String, Any}("lhs" => lhs, "rhs" => rhs))
        else
            o = defined_observed(m, eq)
            (o !== nothing && o in live) || continue
            push!(
                component_model!(models, component_of(o))["equations"],
                Dict{String, Any}("lhs" => _ser(eq.lhs), "rhs" => _ser(eq.rhs))
            )
        end
    end
    for (x, e) in flat.field_ics
        push!(
            component_model!(models, component_of(x))["equations"],
            Dict{String, Any}("lhs" => Dict{String, Any}("op" => "ic", "args" => Any[x]), "rhs" => _ser(e))
        )
    end
    isets = Dict{String, Any}(k => Dict{String, Any}("kind" => "interval", "size" => n) for (k, n) in m.sizes)
    doc = Dict{String, Any}(
        "esm" => "1.3.0", "metadata" => Dict{String, Any}("name" => String(name)),
        "index_sets" => isets, "models" => models
    )
    dom = derived_domain(flat.domain)
    dom === nothing || (doc["domain"] = dom)
    return doc
end

last_segment(name::AbstractString) = (i = findlast('.', name); i === nothing ? String(name) : name[nextind(name, i):end])

"""
The `domain` of a derived document, or `nothing`: of the flattened model's
`domain` as EarthSciAST serializes it, `temporal` with only its `start`,
`end` and `reference_time`, then `element_type` and `array_type`, each when
given, and no `independent_variable`; `nothing` when none of these is given.
EarthSciASTDiff builds the Jacobian document's `domain` so (`_domain!`), so
an operator's two documents carry the same one.
"""
function derived_domain(domain)
    domain === nothing && return nothing
    src = EarthSciAST.serialize_domain(domain)
    out = Dict{String, Any}()
    tmp = get(src, "temporal", nothing)
    if tmp isa AbstractDict
        t = Dict{String, Any}(k => tmp[k] for k in ("start", "end", "reference_time") if get(tmp, k, nothing) !== nothing)
        isempty(t) || (out["temporal"] = t)
    end
    for k in ("element_type", "array_type")
        get(src, k, nothing) === nothing || (out[k] = src[k])
    end
    return isempty(out) ? nothing : out
end

"""
    jacobian_document(rhs) -> Dict

§6.2: the Jacobian document of an operator whose right-hand-side document,
loaded and flattened, is `rhs`: EarthSciASTDiff's evaluation document of
`rhs` with coefficient hoisting off (`cse = false`), its band fields `J_k`
in the model `JacobianEval`, without its `ic` equations.
"""
function jacobian_document(rhs::EarthSciAST.FlattenedSystem)
    doc, _ = EarthSciASTDiff.evaluation_document(rhs; cse = false)
    drop_ic!(doc["models"])
    return doc
end

"""Remove the `ic` equations of every model and subsystem in `models`."""
function drop_ic!(models::AbstractDict)
    for (_, node) in models
        filter!(eq -> !(eq["lhs"] isa AbstractDict && get(eq["lhs"], "op", nothing) == "ic"), node["equations"])
        haskey(node, "subsystems") && drop_ic!(node["subsystems"])
    end
    return models
end

"""The bytes of a derived document: EarthSciASTDiff's `stable_json` (§2.2)."""
document_bytes(doc::AbstractDict) = EarthSciASTDiff.stable_json(doc)
