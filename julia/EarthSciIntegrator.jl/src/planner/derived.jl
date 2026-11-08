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
    probe_document(m, terms; min_nodes = PROBE_TEMPLATE_MIN_NODES) -> Dict

§5.2.2's probe document: §6.1's document of the terms `terms` in which every
state of the model has the zero right-hand side (§6.1.2), and the `k`-th term
is instead the whole right-hand side of a new state [`probe_state`](@ref)`(k)`
of the shape of its equation's state (the same index-set names), written as
that state's equation would hold it alone (a lifted term inside the same
`faq`). Its bands with row `probe_state(k)` are the `k`-th term's bands,
row renamed.

The subexpressions the terms share are written once each, as the
zero-parameter templates `shared_k` in `expression_templates` of the
component `JacobianEval`, and every place they occur, in the terms and in
other templates' bodies, as an `apply_expression_template` reference to one
([`shared_templates`](@ref), whose `min_nodes` this passes on; the planner
always uses the default). The references expand to exactly the terms, so
the templates change no band.
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

§5.2.2's shared subexpressions of `exprs`, named as zero-parameter templates
`shared_k` and replaced, where they occur, by `apply_expression_template`
references to them. Expanding every reference gives `exprs` back,
structurally.

Subexpressions are compared structurally and chosen outermost first, so a
subexpression inside a template counts once however often the template is
referenced. A subexpression becomes a template when it has at least
`min_nodes` operator nodes, every occurrence counted, and occurs at least
twice in what is written: in `exprs` (written in place, as they are) and in
the templates' bodies, each body written once. A subexpression of a
template's body that occurs nowhere else therefore stays in that body. A
subexpression below [`MAX_TEMPLATE_CHAIN`](@ref) templates, on some path
from an expression of `exprs` down to it, is written in place: the chain of
templates each referencing the next is then never longer than EarthSciAST
loads.

The walk is memoized on node identity, so it costs time in proportion to
the distinct nodes of `exprs` rather than to their size as trees.
`templates` is a vector of `name => body` pairs (`k` from 1, outermost
first); `rewritten` is `exprs` with the references.
"""
function shared_templates(exprs::AbstractVector; min_nodes::Int = PROBE_TEMPLATE_MIN_NODES)
    sat(a, b) = a > typemax(Int) - b ? typemax(Int) : a + b
    idof = IdDict{EarthSciAST.OpExpr, Int}()
    ids = Dict{String, Int}()
    rep = EarthSciAST.OpExpr[]
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
            push!(kids, cs)
            length(rep)
        end
        return idof[e] = id
    end
    roots = [intern(e) for e in exprs]
    n = length(rep)
    # Whether a subexpression has at least `min_nodes` operator nodes, by the
    # walk `operator_nodes` makes, its count capped at `cap` (low enough that
    # the walk's sums cannot overflow).
    sizes = IdDict{EarthSciAST.OpExpr, Int}()
    cap = min(min_nodes, typemax(Int) >> 2)
    large(id) = _operator_nodes(rep[id], sizes, cap - 1) >= min_nodes
    # Templates, outermost first: parents have larger ids than their
    # children, so a node's references and the templates above it are all
    # known when it is reached. `refs` counts the node's references in what
    # is written: a template's body is written once, any other node's once
    # for each of its own references. `above` is the most templates on a
    # path from a root down to the node.
    refs = zeros(Int, n)
    above = zeros(Int, n)
    for r in roots
        r == 0 || (refs[r] = sat(refs[r], 1))
    end
    name = Vector{Union{Nothing, String}}(nothing, n)
    k = 0
    for id in n:-1:1
        t = refs[id] >= 2 && above[id] < MAX_TEMPLATE_CHAIN && large(id)
        t && (name[id] = "shared_$(k += 1)")
        each = t ? 1 : refs[id]
        for c in kids[id]
            refs[c] = sat(refs[c], each)
            above[c] = max(above[c], above[id] + t)
        end
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
    templates = Pair{String, EarthSciAST.ASTExpr}[name[id] => body(rep[id]) for id in n:-1:1 if name[id] !== nothing]
    return templates, rewritten
end

function _rhs_document(
        m::PlanModel, name::AbstractString, terms::AbstractVector, probe::Bool;
        min_nodes::Int = PROBE_TEMPLATE_MIN_NODES
    )
    flat = m.flat
    live = live_observeds(m, reduce(vcat, (t.refs for t in terms); init = String[]))
    models = Dict{String, Any}()
    # §6.3: the data-fed parameters this document's terms read, directly or
    # through live observeds, are its forcing slots.
    fed = forcing_among(m, observed_closure(m, reduce(vcat, (t.refs for t in terms); init = String[])))
    for dict in (flat.state_variables, flat.parameters, flat.observed_variables), (vname, v) in dict
        (is_state(m, vname) || is_param(m, vname) || vname in live) || continue
        is_forcing(m, vname) && !(vname in m.used_forcing) && continue        # unused: not declared
        obj = variable_object(m, vname, v)
        vname in fed && (obj["update"] = forcing_update(m, vname))
        component_model!(models, component_of(vname))["variables"][last_segment(vname)] = obj
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
    ds = forcing_stubs(m, fed)
    ds === nothing || (doc["data_sources"] = ds)
    return canonical_const_values!(doc)
end

"""
    canonical_const_values!(x) -> x

Writes every number of every `const` node's `value` in `x` (a derived
document, or any part of one) in the esm canonical number form (§2.2;
EarthSciAST CONFORMANCE_SPEC §5.5.3.1 rule 1): an integral value in
[-2^63, 2^63) as an integer, so `[100.0, 0.5]` is written `[100,0.5]`.
EarthSciAST.jl already holds an authored `const` value that way from load,
but a `const` node that reaches a derived document by another road (a
`table_lookup` lowering, or one EarthSciASTDiff writes into the Jacobian
document) may hold floats. The Rust planner needs no rewrite: from
EarthSciAST#670 its binding writes every `const` value in this form, and it
refuses a `table_lookup` (§4 step 5), so the two write every `const` value
alike wherever it came from.
"""
function canonical_const_values!(x)
    if x isa AbstractDict
        get(x, "op", nothing) == "const" && haskey(x, "value") && (x["value"] = canonical_number(x["value"]))
        foreach(canonical_const_values!, values(x))
    elseif x isa AbstractVector
        foreach(canonical_const_values!, x)
    end
    return x
end

"""A `const` value's number in the canonical form (§2.2): an integral float in [-2^63, 2^63) as an `Int64`; arrays element by element."""
canonical_number(v::AbstractVector) = Any[canonical_number(y) for y in v]
canonical_number(v::AbstractFloat) = isfinite(v) && isinteger(v) && -2.0^63 <= v < 2.0^63 ? Int64(v) : v
canonical_number(v) = v

"""
The `update` a data-fed parameter in a document's forcing is declared with
(§6.3): its `data` rule naming its source and file variable, nothing else
of the original.
"""
forcing_update(m::PlanModel, n::AbstractString) = Dict{String, Any}(
    "kind" => "data", "source" => m.forcing[n].source,
    "from" => Dict{String, Any}("file_variable" => m.forcing[n].file_variable)
)

"""The placeholder location of every forcing stub (§6.3): substitution-led, so EarthSciAST keeps it as written and it names no file."""
const FORCING_STUB_URL = "{esm_integrator_forcing}"

"""
§6.3's `data_sources`: one stub per source the data-fed parameters `fed`
name, `{"kind": <the source's kind>, "source": {"url_template":
"{esm_integrator_forcing}"}}`; `nothing` when `fed` is empty.
"""
function forcing_stubs(m::PlanModel, fed)
    isempty(fed) && return nothing
    return Dict{String, Any}(
        m.forcing[n].source => Dict{String, Any}(
            "kind" => m.forcing[n].source_kind,
            "source" => Dict{String, Any}("url_template" => FORCING_STUB_URL)
        ) for n in fed
    )
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
loaded and flattened, is `rhs` and whose bands are `entries` (pass 4's,
§5.4): EarthSciASTDiff's evaluation document of `rhs` for those bands, with
subtree sharing on (`cse = true`, `cse_min_nodes = 12`; D90), its band
fields `J_k` and the shared subtrees' observeds in the model `JacobianEval`,
without its `ic` equations.

The bands are passed to `evaluation_document`, which uses them as given
rather than computing them again (EarthSciASTDiff#64), as the Rust planner
does; band `k` must be field `J_k`.
Every state without a `default` gets the `default` `1.0` (§6.2 item 4; see
[`default_states!`](@ref)), and every `const` value is written in the
canonical number form (§2.2; [`canonical_const_values!`](@ref)).
"""
function jacobian_document(rhs::EarthSciAST.FlattenedSystem, entries::AbstractVector; data_sources = nothing)
    doc, fields = EarthSciASTDiff.evaluation_document(
        rhs, collect(EarthSciASTDiff.JacEntry, entries); cse = true, cse_min_nodes = 12, data_sources
    )
    fields == ["J_$k" for k in eachindex(entries)] || error(
        "EarthSciASTDiff's evaluation document has band fields $fields, " *
            "but the operator has $(length(entries)) bands (spec/plan.md §6.2)"
    )
    drop_ic!(doc["models"])
    default_states!(doc["models"])
    # §2.2: every `const` value, whether EarthSciASTDiff copied it from the
    # right-hand-side document or wrote it, in the canonical number form.
    return canonical_const_values!(doc)
end

"""
Give every state of `models` without a `default` the `default` `1.0`
(spec/plan.md §6.2 item 4). With its `ic` equations dropped, such a state (one
only an `ic` equation or `field_ics` entry starts) has no starting value, which
EarthSciAST refuses when it builds the document
(`E_TREEWALK_MISSING_INITIAL_VALUE`). A `"default": null` in the built
document counts as no `default` and is replaced (a model's own `default` is
never null: the schema refuses it at load). A state with a `default` keeps
it, `0.0` included. One, not zero, so that no coefficient divides by a
placeholder zero the planner wrote (D89). A state is a variable a `D`
equation defines: by its fully scoped name (§6.2 item 2), or by its name in
the equation's own model or subsystem.
"""
function default_states!(models::AbstractDict)
    found = Vector{Vector{String}}()
    function states(node, path)
        for eq in get(node, "equations", [])
            l = eq["lhs"]
            l isa AbstractDict && get(l, "op", nothing) == "D" && push!(found, vcat(path, String(l["args"][1])))
        end
        for (k, sub) in get(node, "subsystems", Dict())
            states(sub, vcat(path, String(k)))
        end
        return
    end
    for (k, m) in models
        states(m, [String(k)])
    end
    # The declaration of the variable `name` (its components) under `models`.
    function variable(name)
        length(name) >= 2 || return nothing
        node = get(models, name[1], nothing)
        for s in name[2:(end - 1)]
            node === nothing && return nothing
            node = get(get(node, "subsystems", Dict()), s, nothing)
        end
        node === nothing && return nothing
        return get(get(node, "variables", Dict()), name[end], nothing)
    end
    for local_name in found
        x = local_name[end]
        decl = variable(String.(split(x, '.')))
        decl === nothing && (decl = variable(vcat(local_name[1:(end - 1)], x)))
        decl isa AbstractDict || error("the Jacobian document declares no state $x (spec/plan.md §6.2)")
        get(decl, "default", nothing) === nothing && (decl["default"] = 1.0)
    end
    return models
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
