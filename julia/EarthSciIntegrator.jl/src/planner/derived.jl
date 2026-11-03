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
    return _rhs_document(m, name, terms, false)
end

"""
The component of the probe document's new states (§5.2.2): the name
§6.1.4 reserves for the Jacobian document, so no name of the model can
clash with them.
"""
const PROBE_COMPONENT = "JacobianEval"

"""The probe document's new state for the `k`-th term (§5.2.2)."""
probe_state(k::Integer) = "$PROBE_COMPONENT.term_$k"

"""
    probe_document(m, terms) -> Dict

§5.2.2's probe document: §6.1's document of the terms `terms` in which every
state of the model has the zero right-hand side (§6.1.2), and the `k`-th term
is instead the whole right-hand side of a new state [`probe_state`](@ref)`(k)`
of the shape of its equation's state (the same index-set names), written as
that state's equation would hold it alone (a lifted term inside the same
`faq`). Its bands with row `probe_state(k)` are the `k`-th term's bands,
row renamed.
"""
probe_document(m::PlanModel, terms::AbstractVector; min_nodes::Int = PROBE_TEMPLATE_MIN_NODES) =
    _rhs_document(m, "probe", terms, true; min_nodes)

"""
The fewest operator nodes, every occurrence counted, of a subexpression the
probe document names as a template (§5.2.2).
"""
const PROBE_TEMPLATE_MIN_NODES = 8

"""
EarthSciAST's `MAX_TEMPLATE_EXPANSION_DEPTH` (esm-spec §9.7.3): the most
templates on a chain of template bodies referencing templates.
"""
const MAX_TEMPLATE_CHAIN = 32

"""
    shared_templates(exprs; min_nodes = PROBE_TEMPLATE_MIN_NODES) -> (templates, rewritten)

§5.2.2's shared subexpressions of `exprs`: every subexpression that occurs
at least twice in them, counting every occurrence, and has at least
`min_nodes` operator nodes, is named as a zero-parameter template, `shared_k`
(`k` from 1, children before parents), and each of its occurrences, in
`exprs` and in other templates' bodies, is replaced by an
`apply_expression_template` reference to it, except that a subexpression
whose template would end a chain of more than [`MAX_TEMPLATE_CHAIN`](@ref)
templates is left in place. Expanding every reference gives `exprs` back,
structurally. Subexpressions are compared structurally; the walk is
memoized on node identity, so it costs time in proportion to the distinct
nodes of `exprs` rather than to their size as trees.

`templates` is a vector of `name => body` pairs; `rewritten` is `exprs` with
the references.
"""
function shared_templates(exprs::AbstractVector; min_nodes::Int = PROBE_TEMPLATE_MIN_NODES)
    sat(a, b) = a > typemax(Int) - b ? typemax(Int) : a + b
    idof = IdDict{EarthSciAST.OpExpr, Int}()
    ids = Dict{String, Int}()
    rep = EarthSciAST.OpExpr[]
    nodes = Int[]
    kids = Vector{Int}[]
    # Structural ids, children before parents: a node's key is its JCS form
    # with each operator child replaced by a name holding the child's id.
    function intern(e)
        e isa EarthSciAST.OpExpr || return 0
        id = get(idof, e, 0)
        id == 0 || return id
        cs = Int[]
        shell = EarthSciAST.map_children(e) do c
            cid = intern(c)
            cid == 0 && return c
            push!(cs, cid)
            return EarthSciAST.VarExpr("⟦$(cid)⟧")
        end
        key = jcs(_ser(shell))
        id = get!(ids, key) do
            push!(rep, e)
            push!(nodes, foldl((a, c) -> sat(a, nodes[c]), cs; init = 1))
            push!(kids, cs)
            length(rep)
        end
        return idof[e] = id
    end
    roots = [intern(e) for e in exprs]
    n = length(rep)
    # Occurrences, every one counted: parents have larger ids than children.
    occ = zeros(Int, n)
    for r in roots
        r == 0 || (occ[r] = sat(occ[r], 1))
    end
    for id in n:-1:1, c in kids[id]
        occ[c] = sat(occ[c], occ[id])
    end
    # Templates, children first, within the chain bound.
    name = Vector{Union{Nothing, String}}(nothing, n)
    chain = zeros(Int, n)
    k = 0
    for id in 1:n
        d = maximum((chain[c] for c in kids[id]); init = 0)
        if occ[id] >= 2 && nodes[id] >= min_nodes && d < MAX_TEMPLATE_CHAIN
            name[id] = "shared_$(k += 1)"
            d += 1
        end
        chain[id] = d
    end
    # Rewrite, memoized on node identity.
    memo = IdDict{EarthSciAST.OpExpr, EarthSciAST.ASTExpr}()
    ref(nm) = EarthSciAST.OpExpr("apply_expression_template", EarthSciAST.ASTExpr[]; name = nm, bindings = Dict{String, EarthSciAST.ASTExpr}())
    body(e) = EarthSciAST.map_children(rw, e)
    function rw(e)
        e isa EarthSciAST.OpExpr || return e
        return get!(memo, e) do
            nm = name[idof[e]]
            nm === nothing ? body(e) : ref(nm)
        end
    end
    rewritten = EarthSciAST.ASTExpr[rw(e) for e in exprs]
    templates = Pair{String, EarthSciAST.ASTExpr}[name[id] => body(rep[id]) for id in 1:n if name[id] !== nothing]
    return templates, rewritten
end

function _rhs_document(
        m::PlanModel, name::AbstractString, terms::AbstractVector, probe::Bool;
        min_nodes::Int = PROBE_TEMPLATE_MIN_NODES
    )
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
            ts = probe ? Term[] : get(by_eq, ei, Term[])
            push!(component_model!(models, component_of(var))["equations"], state_equation(m, ei, var, lifted, ts))
        else
            o = defined_observed(m, eq)
            (o !== nothing && o in live) || continue
            push!(
                component_model!(models, component_of(o))["equations"],
                Dict{String, Any}("lhs" => _ser(eq.lhs), "rhs" => _ser(eq.rhs))
            )
        end
    end
    if probe
        v = component_model!(models, PROBE_COMPONENT)
        # Subexpressions the terms share, written once each (§5.2.2).
        templates, exprs = shared_templates(ASTExpr[t.expr for t in terms]; min_nodes)
        isempty(templates) || (
            v["expression_templates"] = Dict{String, Any}(
                nm => Dict{String, Any}("params" => Any[], "body" => _ser(b)) for (nm, b) in templates
            )
        )
        for (k, t) in enumerate(terms)
            x = probe_state(k)
            # The term's state's declaration (its shape and `default`) and its
            # `ic` equation, if any, so the new state is initialized as it is.
            v["variables"][last_segment(x)] = variable_object(m, t.variable, flat.state_variables[t.variable])
            eq = state_equation(m, t.equation, t.variable, t.lifted, [t]; exprs = exprs[k:k])
            eq["lhs"]["args"] = Any[x]
            push!(v["equations"], eq)
            for (y, e) in flat.field_ics
                y == t.variable || continue
                push!(v["equations"], Dict{String, Any}("lhs" => Dict{String, Any}("op" => "ic", "args" => Any[x]), "rhs" => _ser(e)))
            end
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

"""The equation `D(var) = …` of state `var`, flattened equation `ei`, holding the terms `ts` (§6.1.2): their left fold, inside the lift's `faq` when `lifted`, or the zero right-hand side when there are none."""
function state_equation(
        m::PlanModel, ei::Int, var::AbstractString, lifted::Bool, ts::AbstractVector;
        exprs::AbstractVector = ASTExpr[t.expr for t in ts]
    )
    rhs = if isempty(ts)
        zero_rhs(m, var)
    elseif lifted
        O, G = lifted_ranges(m, ei, var)
        Dict{String, Any}(
            "op" => "faq", "args" => Any[], "output_idx" => Any[O...], "ranges" => G,
            "expr" => fold_sum(Any[_ser(e) for e in exprs])
        )
    else
        fold_sum(Any[_ser(e) for e in exprs])
    end
    lhs = Dict{String, Any}("op" => "D", "args" => Any[var], "wrt" => "t")
    return Dict{String, Any}("lhs" => lhs, "rhs" => rhs)
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
    jacobian_document(rhs, entries) -> Dict

§6.2: the Jacobian document of an operator whose right-hand-side document,
loaded and flattened, is `rhs` and whose bands are `entries` (§5.4):
EarthSciASTDiff's evaluation document of `rhs` with coefficient hoisting off
(`cse = false`), its band fields `J_k` in the model `JacobianEval`, without
its `ic` equations.

Band `k` must be field `J_k`. EarthSciASTDiff's public `evaluation_document`
computes its own bands, with the same `jacobian_bands(sysview(rhs); wrt =
:states)` call as [`document_bands`](@ref), so its fields are checked
against `entries`: their names, and each field's row region against the
band's `rows`. (The `band_model` check of §8.1 compares the values.)
"""
function jacobian_document(rhs::EarthSciAST.FlattenedSystem, entries::AbstractVector)
    doc, fields = EarthSciASTDiff.evaluation_document(rhs; cse = false)
    fields == ["J_$k" for k in eachindex(entries)] || error(
        "EarthSciASTDiff's evaluation document has band fields $fields, " *
            "but the operator has $(length(entries)) bands (spec/plan.md §6.2)"
    )
    eqs = Dict(
        eq["lhs"]["args"][1] => eq["rhs"] for eq in doc["models"]["JacobianEval"]["equations"]
            if eq["lhs"] isa AbstractDict && get(eq["lhs"], "op", nothing) == "D"
    )
    for (k, en) in enumerate(entries)
        r = eqs["J_$k"]
        r isa AbstractDict && get(r, "op", nothing) == "makearray" || continue
        rows = [Int.(x) for x in r["regions"][2][1:length(en.band.rows)]]
        rows == [Int[lo, hi] for (lo, hi) in en.band.rows] || error(
            "band field J_$k of EarthSciASTDiff's evaluation document covers rows $rows, " *
                "but band $k has rows $(en.band.rows) (spec/plan.md §6.2)"
        )
    end
    drop_ic!(doc["models"])
    return doc
end

"""Remove the `ic` equations of every model and subsystem in `models`."""
function drop_ic!(models::AbstractDict)
    for (_, node) in models
        eqs = get(node, "equations", nothing)
        eqs === nothing || filter!(eq -> !(eq["lhs"] isa AbstractDict && get(eq["lhs"], "op", nothing) == "ic"), eqs)
        haskey(node, "subsystems") && drop_ic!(node["subsystems"])
    end
    return models
end

"""The bytes of a derived document: EarthSciASTDiff's `stable_json` (§2.2)."""
document_bytes(doc::AbstractDict) = EarthSciASTDiff.stable_json(doc)
