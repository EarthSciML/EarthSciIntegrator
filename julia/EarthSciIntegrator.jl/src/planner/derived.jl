# The derived documents (spec/plan.md §6, D68, D69): the right-hand-side
# document of a set of terms and the Jacobian document of an operator, as
# JSON objects; their bytes are EarthSciASTDiff's `stable_json` (§2.2).

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
function defined_observed(m::PlanModel, eq::Equation)
    l = eq.lhs
    l isa VarExpr && return is_observed(m, l.name) ? l.name : nothing
    l isa OpExpr || return nothing
    l.op == "faq" && l.expr_body isa OpExpr && (l = l.expr_body)
    (l.op == "index" && !isempty(l.args) && l.args[1] isa VarExpr && is_observed(m, l.args[1].name)) || return nothing
    return l.args[1].name
end

"""
    live_observeds(m, names) -> Set{String}

The observeds among `names`, with every observed the definitions of those
reference, transitively (§6.1.1 "live", §6.2 item 2).
"""
function live_observeds(m::PlanModel, names)
    defs = Dict{String, Vector{ASTExpr}}()
    for eq in m.flat.equations
        o = defined_observed(m, eq)
        o === nothing || push!(get!(defs, o, ASTExpr[]), eq.rhs)
    end
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
    rhs_document(m, name, terms; jacobian = nothing) -> Dict

§6.1: the right-hand-side document named `name` of the terms `terms` (in
term order): every index set, the domain, and the component tree declaring
every state and parameter and the live observeds, with each state's
equation the fold of its terms (zero when it has none), the live observeds'
definitions, and every `ic` equation.

With `jacobian = (bands, coefs)` (`bands` as EarthSciASTDiff's `JacEntry`s,
in band order), §6.2's Jacobian document instead: every state's right-hand
side is zero, observeds are live when a band coefficient references them,
and a model `_jacobian` holds one band field per band.
"""
function rhs_document(m::PlanModel, name::AbstractString, terms::AbstractVector; jacobian = nothing)
    flat = m.flat
    refs = jacobian === nothing ? reduce(vcat, (t.refs for t in terms); init = String[]) :
        String[n for en in jacobian for n in EarthSciAST.free_variables(en.band.coef)]
    live = live_observeds(m, refs)
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
            rhs = if jacobian !== nothing || isempty(ts)
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
    flat.domain === nothing || (doc["domain"] = EarthSciAST.serialize_domain(flat.domain))
    jacobian === nothing || add_band_fields!(doc, m, jacobian)
    rename_captured_indices!(doc)
    return doc
end

# ---------------------------------------------------------------------------
# Index names a model's own variables would capture

"""Every string in a JSON tree, keys included."""
function _strings!(acc::Set{String}, x)
    if x isa AbstractString
        push!(acc, String(x))
    elseif x isa AbstractDict
        for (k, v) in x
            push!(acc, String(k))
            _strings!(acc, v)
        end
    elseif x isa AbstractVector
        foreach(v -> _strings!(acc, v), x)
    end
    return acc
end

_is_faq(x) = x isa AbstractDict && get(x, "op", nothing) == "faq"
_binders(x) = Set{String}(vcat(String[String(n) for n in get(x, "output_idx", Any[]) if n isa AbstractString], String.(collect(keys(get(x, "ranges", Dict{String, Any}()))))))

# The expression-bearing fields of a serialized node.
const _EXPR_FIELDS = ("args", "expr", "values", "filter")

"""`x` with the free occurrences of index name `b` written `nb`; a `faq` that binds `b` again shadows it."""
function _subst_free(x, b::String, nb::String)
    x isa AbstractString && return x == b ? nb : x
    x isa AbstractVector && return Any[_subst_free(v, b, nb) for v in x]
    x isa AbstractDict || return x
    _is_faq(x) && b in _binders(x) && return x
    y = Dict{String, Any}(String(k) => v for (k, v) in x)
    for k in _EXPR_FIELDS
        haskey(y, k) && (y[k] = _subst_free(y[k], b, nb))
    end
    return y
end

"""
`x` with every `faq` binder that names a variable of `vars` renamed: the
binder (in `output_idx` and `ranges`) and its free occurrences in the body.
"""
function _rename_binders(x, vars::Set{String}, fresh)
    x isa AbstractVector && return Any[_rename_binders(v, vars, fresh) for v in x]
    x isa AbstractDict || return x
    y = Dict{String, Any}(String(k) => v for (k, v) in x)
    if _is_faq(y)
        for b in sort!(collect(intersect(_binders(y), vars)))
            nb = fresh(b)
            haskey(y, "output_idx") && (y["output_idx"] = Any[n == b ? nb : n for n in y["output_idx"]])
            haskey(y, "ranges") && (y["ranges"] = Dict{String, Any}((k == b ? nb : String(k)) => v for (k, v) in y["ranges"]))
            for k in ("expr", "filter")
                haskey(y, k) && (y[k] = _subst_free(y[k], b, nb))
            end
        end
    end
    for k in _EXPR_FIELDS
        haskey(y, k) && (y[k] = _rename_binders(y[k], vars, fresh))
    end
    return y
end

"""
    rename_captured_indices!(doc)

Rename apart every index name a `faq` binds that equals the name of a
variable declared in the model whose equations hold it, to `_c<n>_<name>`
with `n` the smallest positive integer for which that name occurs nowhere in
the document (EarthSciASTDiff's convention for renaming binders apart).
EarthSciAST.jl 0.6.0's `flatten` reads such an index as a reference to the
model's variable (an index `j` of a lifted equation in a reaction system with
a rate `j` becomes `Chem.j`), so without this the document would not mean
what §6.1 writes. A deviation from spec/plan.md §6.1.2's copied expressions,
reported as a stage-2 finding; it changes nothing in a document without such
a collision.
"""
function rename_captured_indices!(doc::AbstractDict)
    taken = _strings!(Set{String}(), doc)
    names = Dict{String, String}()
    fresh(b) = get!(names, b) do
        n = 1
        while "_c$(n)_$b" in taken
            n += 1
        end
        "_c$(n)_$b"
    end
    function model!(node)
        vars = Set{String}(String(k) for k in keys(node["variables"]))
        node["equations"] = Any[
            Dict{String, Any}("lhs" => _rename_binders(eq["lhs"], vars, fresh), "rhs" => _rename_binders(eq["rhs"], vars, fresh))
                for eq in node["equations"]
        ]
        foreach(model!, values(get(node, "subsystems", Dict{String, Any}())))
        return
    end
    foreach(model!, values(doc["models"]))
    return doc
end

last_segment(name::AbstractString) = (i = findlast('.', name); i === nothing ? String(name) : name[nextind(name, i):end])

"""
EarthSciASTDiff's name for the index set of a contracted dimension of extent
`hi` (`_jcax_name`): `jcax<hi>`, with `x` appended while an index set of that
name has another size; added to the document's `index_sets` when missing.
"""
function contracted_axis!(isets::AbstractDict, hi::Int)
    nm = "jcax$hi"
    while haskey(isets, nm) && isets[nm]["size"] != hi
        nm *= "x"
    end
    haskey(isets, nm) || (isets[nm] = Dict{String, Any}("kind" => "interval", "size" => hi))
    return nm
end

"""
§6.2 item 3: the model `_jacobian` with band field `J_k` per band, built as
EarthSciASTDiff's `_evaluation_document` builds them with `cse = false`,
in the component tree: every reference fully scoped and `J_k` written
`_jacobian.J_k`.
"""
function add_band_fields!(doc::AbstractDict, m::PlanModel, entries)
    jm = Dict{String, Any}("variables" => Dict{String, Any}(), "equations" => Any[])
    for (k, en) in enumerate(entries)
        b = en.band
        nm = "J_$k"
        cr = b.contracted
        rowsets = get(m.shapes, en.u, String[])
        if isempty(rowsets) && isempty(cr)
            jm["variables"][nm] = Dict{String, Any}("type" => "unknown", "default" => 0.0)
            rhs = _ser(b.coef)
        else
            jm["variables"][nm] = Dict{String, Any}(
                "type" => "unknown", "default" => 0.0,
                "shape" => Any[rowsets..., (contracted_axis!(doc["index_sets"], hi) for (_, _, hi) in cr)...]
            )
            free = [(d, r) for (d, r) in enumerate(b.ridx) if r isa String]
            aggnames = Any[(r for (_, r) in free)..., (n for (n, _, _) in cr)...]
            val = if isempty(aggnames)
                _ser(b.coef)
            else
                rngs = Dict{String, Any}(r => Any[b.rows[d][1], b.rows[d][2]] for (d, r) in free)
                for (n, lo, hi) in cr
                    rngs[n] = Any[lo, hi]
                end
                Dict{String, Any}("op" => "faq", "args" => Any[], "output_idx" => aggnames, "ranges" => rngs, "expr" => _ser(b.coef))
            end
            rhs = Dict{String, Any}(
                "op" => "makearray", "args" => Any[],
                "regions" => Any[
                    Any[Any[Any[1, m.sizes[X]] for X in rowsets]..., Any[Any[1, hi] for (_, _, hi) in cr]...],
                    Any[Any[Any[lo, hi] for (lo, hi) in b.rows]..., Any[Any[lo, hi] for (_, lo, hi) in cr]...],
                ],
                "values" => Any[0.0, val]
            )
        end
        lhs = Dict{String, Any}("op" => "D", "args" => Any["_jacobian." * nm], "wrt" => "t")
        push!(jm["equations"], Dict{String, Any}("lhs" => lhs, "rhs" => rhs))
    end
    doc["models"]["_jacobian"] = jm
    return doc
end

"""The bytes of a derived document: EarthSciASTDiff's `stable_json` (§2.2)."""
document_bytes(doc::AbstractDict) = EarthSciASTDiff.stable_json(doc)
