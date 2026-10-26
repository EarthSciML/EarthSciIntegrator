# Pass 1: terms, provenance, owners, selection and operators (spec/plan.md §5.1).

"""
    additive_terms(e) -> Vector{ASTExpr}

The summands of `e` by the rule of spec/plan.md §5.1.1 (EarthSciASTSplitter's
`additive_terms`, D64), implemented here as written rather than by depending
on EarthSciASTSplitter:

- `+` with any number of arguments: the concatenation of its arguments' summands;
- `-` with two arguments `a, b`: the summands of `a`, then each summand `t`
  of `b` as `-(t)`;
- `-` with one argument `a`: each summand `t` of `a` as `-(t)`;
- anything else (`*`, `faq`, `makearray`, `index`, a `-` with three or more
  arguments, a literal, a name): itself.
"""
function additive_terms(e::ASTExpr)
    e isa OpExpr || return ASTExpr[e]
    if e.op == "+"
        return reduce(vcat, (additive_terms(a) for a in e.args); init = ASTExpr[])
    elseif e.op == "-" && length(e.args) == 2
        return vcat(additive_terms(e.args[1]), ASTExpr[_negate(t) for t in additive_terms(e.args[2])])
    elseif e.op == "-" && length(e.args) == 1
        return ASTExpr[_negate(t) for t in additive_terms(e.args[1])]
    end
    return ASTExpr[e]
end

_negate(t::ASTExpr) = OpExpr("-", ASTExpr[t])

"""A numeric literal equal to zero (§5.1.1: such a summand is dropped). A negated literal, `-(0)`, is an operation, not a literal."""
is_zero_literal(e::ASTExpr) = (e isa NumExpr && e.value == 0) || (e isa IntExpr && e.value == 0)

"""
    Term

One additive summand of one contribution of a state equation (§5.1.1),
identified by `(variable, contribution, summand)`.

`locality` and `radius` are pass 2's (§5.2); `nothing` until computed, and
left `nothing` in M2 stage 1 when the band source cannot give the term's
bands yet.
"""
mutable struct Term
    variable::String
    contribution::Int
    summand::Int
    expr::ASTExpr
    origins::Vector{TermOrigin}
    provenance::String
    owner::Union{String, Nothing}
    lifted::Bool
    equation::Int
    refs::Vector{String}
    locality::Union{Symbol, Nothing}
    radius::Union{Dict{String, Tuple{Int, Int}}, Nothing}
end

"""The provenance label of a contribution's origins (§5.1.1): `P`, `coupling:k`, joined by `+`."""
provenance_label(origins) = join((o.component !== nothing ? o.component : "coupling:$(o.coupling)" for o in origins), "+")

"""
    term_owner(m, variable, origins, refs) -> Union{String, Nothing}

spec/plan.md §5.1.2 (D65): the single component origin, else the one
component other than the dependent variable's whose names the term
references; the dependent variable's component when it references none;
`nothing` (no owner) when it references two or more.
"""
function term_owner(m::PlanModel, variable::AbstractString, origins, refs)
    length(origins) == 1 && origins[1].component !== nothing && return origins[1].component
    own = component_of(variable)
    R = sort!(unique!([component_of(n) for n in refs if component_of(n) != own]))
    isempty(R) && return own
    length(R) == 1 && return R[1]
    return nothing
end

"""
    build_terms(m) -> Vector{Term}

Every term of the flattened model in term order (§5.1.1): flattened-equation
order, then contribution, then summand. Equations defining observeds and
`ic` equations are not split.
"""
function build_terms(m::PlanModel)
    terms = Term[]
    for (ei, var, lifted) in m.state_eqs
        for (c, contrib) in enumerate(m.contributions[ei])
            isempty(contrib.origins) && refuse("term_origin_unknown", "/model",
                "contribution $(c - 1) of the equation for $var has no origins (only a caller's edit of a flattened system produces one)")
            label = provenance_label(contrib.origins)
            s = 0
            for t in additive_terms(contrib.expr)
                is_zero_literal(t) && continue
                refs = referenced_names(m, t)
                push!(terms, Term(var, c - 1, s, t, contrib.origins, label, term_owner(m, var, contrib.origins, refs),
                    lifted, ei, refs, nothing, nothing))
                s += 1
            end
        end
    end
    return terms
end

# ---------------------------------------------------------------------------
# Selection (§5.1.3)

_as_list(x) = x isa AbstractString ? [String(x)] : String[String(y) for y in x]

"""A component `C` covers `P` when `P == C` or `P` begins with `C` followed by a dot (§5.1.2)."""
covers(C::AbstractString, P::AbstractString) = P == C || startswith(P, C * ".")

"""
    selector_matches(sel, term) -> Bool

Every key present matches (§5.1.3). A `locality` key needs the term's
locality, which pass 2 gives; in stage 1 it may be pending, and then this
raises [`PlannerStageError`](@ref).
"""
function selector_matches(sel::AbstractDict, t::Term)
    if haskey(sel, "components")
        t.owner === nothing && return false
        any(C -> covers(C, t.owner), _as_list(sel["components"])) || return false
    end
    if haskey(sel, "variables")
        t.variable in _as_list(sel["variables"]) || return false
    end
    if haskey(sel, "terms")
        t.provenance in _as_list(sel["terms"]) || return false
    end
    if haskey(sel, "locality")
        String(need_locality(t)) in _as_list(sel["locality"]) || return false
    end
    return true
end

"""The term's locality class, or a [`PlannerStageError`](@ref) when stage 1 cannot compute it."""
function need_locality(t::Term)
    t.locality === nothing && throw(PlannerStageError(
        "the locality of term ($(t.variable), $(t.contribution), $(t.summand)) needs its EarthSciASTDiff bands"))
    return t.locality
end

"""
    PlanOperator

An operator (§5.1.4): its terms in term order, its `bind_dt` from the run
config, pass 2's class and radius, pass 3's writes and reads.
"""
mutable struct PlanOperator
    name::String
    terms::Vector{Term}
    bind_dt::Union{AbstractDict, Nothing}
    pointer::String                     # "/operators/<name>", or "" for a default operator
    locality::Union{Symbol, Nothing}
    radius::Union{Dict{String, Tuple{Int, Int}}, Nothing}
    writes::Vector{String}
    reads::Vector{String}
    structure::Any                      # pass 4 (stage 2); `nothing` while pending
end

is_default_operator(op::PlanOperator) = startswith(op.name, "default_")

"""
    select_operators(m, terms, config) -> Dict{String, PlanOperator}

§5.1.3 and §5.1.4: map operators are refused, `variables` entries must name
states, then each term goes to the one operator whose selector matches it,
or to its locality's default operator; a term matched twice, an owner-less
term left unmatched while some selector uses `components`, and an operator
that matches nothing are refused.

Spec finding: §5.1.3 does not order its refusals. They are raised in the
order written: map operators (operators in code-point order), unresolved or
non-state `variables` entries, then per term in term order, then operators
that match nothing (code-point order).
"""
function select_operators(m::PlanModel, terms::Vector{Term}, config::AbstractDict)
    opdefs = get(config, "operators", Dict{String, Any}())
    names = sort!(String.(collect(keys(opdefs))))
    for n in names
        haskey(opdefs[n], "map") && refuse("unsupported_map_operator", jptr("/operators", n),
            "operator $n is a map operator; map operators need U6 (M4b)")
    end
    for n in names
        sel = opdefs[n]["select"]
        haskey(sel, "variables") || continue
        vs = sel["variables"]
        for (i, v) in enumerate(_as_list(vs))
            at = vs isa AbstractString ? jptr("/operators", n, "select", "variables") :
                 jptr("/operators", n, "select", "variables", i - 1)
            k = name_kind(m, v)
            k === nothing && refuse("name_unresolved", at, "selector variable $v is not a name of the flattened model")
            k === :state || refuse("name_wrong_kind", at, "selector variable $v is a $k, not a state")
        end
    end
    uses_components = any(n -> haskey(opdefs[n]["select"], "components"), names)
    assigned = Dict{String, Vector{Term}}()
    for t in terms
        matched = [n for n in names if selector_matches(opdefs[n]["select"], t)]
        if length(matched) >= 2
            refuse("term_selected_twice", jptr("/operators", matched[1]),
                "term ($(t.variable), $(t.contribution), $(t.summand)) is matched by operators $(matched[1]) and $(matched[2])")
        elseif length(matched) == 1
            push!(get!(assigned, matched[1], Term[]), t)
        elseif t.owner === nothing && uses_components
            others = sort!(unique!([component_of(r) for r in t.refs if component_of(r) != component_of(t.variable)]))
            refuse("ambiguous_term_owner", "/operators",
                "term ($(t.variable), $(t.contribution), $(t.summand)), provenance $(t.provenance), reads components " *
                "$(join(others, ", ")) and so has no owner; no selector matches it while some selector uses components. " *
                "Select it with a terms or variables selector (D65)")
        else
            push!(get!(assigned, "default_" * String(need_locality(t)), Term[]), t)
        end
    end
    for n in names
        haskey(assigned, n) || refuse("selector_matches_nothing", jptr("/operators", n, "select"),
            "operator $n's selector matches no term")
    end
    ops = Dict{String, PlanOperator}()
    for (n, ts) in assigned
        def = get(opdefs, n, nothing)
        bind = def === nothing ? nothing : get(def, "bind_dt", nothing)
        ops[n] = PlanOperator(n, ts, bind, def === nothing ? "" : jptr("/operators", n), nothing, nothing,
            String[], String[], nothing)
    end
    return ops
end

"""The plan's object for a term (§5.1.4)."""
term_json(t::Term) = Dict{String, Any}("variable" => t.variable, "contribution" => t.contribution,
    "summand" => t.summand, "provenance" => t.provenance, "locality" => String(need_locality(t)))
