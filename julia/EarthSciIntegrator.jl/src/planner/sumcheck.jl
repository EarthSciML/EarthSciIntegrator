# Pass 8's `rhs_sum` (spec/plan.md §8.1, D91): the operators' right-hand-side
# documents, loaded and flattened from their bytes, sum to the flattened model,
# compared structurally. Nothing is built or evaluated.

"""An expression's canonical form (§8.1): EarthSciAST's serialization as `stable_json` writes it (§2.2)."""
canonical_expr(e) = EarthSciASTDiff.stable_json(EarthSciAST.serialize_expression(e))

"""A JSON value with every number a `Float64`, as §8.1 compares declared `default`s."""
_float_form(x::Real) = Float64(x)
_float_form(x::AbstractVector) = Any[_float_form(v) for v in x]
_float_form(x::AbstractDict) = Dict{String, Any}(String(k) => _float_form(v) for (k, v) in x)
_float_form(x) = x

"""A declared `default` as §8.1 compares it: its `stable_json` text read back with every number a binary64 value, or `nothing`."""
default_form(d) = d === nothing ? nothing : EarthSciASTDiff.stable_json(_float_form(JSON.parse(EarthSciASTDiff.stable_json(d))))

"""A declared shape: its index-set names, `[]` for none."""
shape_form(v) = v.shape === nothing ? String[] : String[String(x) for x in v.shape]

"""
    state_summands(rhs, zero, lifted) -> Union{Vector{String}, Nothing}

§8.1's summands of a state equation's right-hand side `rhs`, canonical: none
when `rhs` is the state's zero right-hand side `zero` (§6.1.2, canonical);
else the summands of §5.1.1 (`additive_terms`, literal zeros dropped) of
`rhs`, or of the body of its `faq` when `lifted`. `nothing` when `lifted` and
`rhs` is no `faq`.
"""
function state_summands(rhs, zero::AbstractString, lifted::Bool)
    canonical_expr(rhs) == zero && return String[]
    body = rhs
    if lifted
        (rhs isa OpExpr && rhs.op == "faq" && rhs.expr_body !== nothing) || return nothing
        body = rhs.expr_body
    end
    return String[canonical_expr(t) for t in additive_terms(body) if !is_zero_literal(t)]
end

"""
The variable an equation's left-hand side defines, at any spelling: a bare
name, `index(y, …)` or `faq{index(y, …)}`; `nothing` for any other.
A derived document declares the observeds it keeps as unknowns (§6.1.1), so
its flattened form need not list them as observeds; the model says which
names are observeds.
"""
function defined_name(eq::Equation)
    l = eq.lhs
    l isa VarExpr && return l.name
    l isa OpExpr || return nothing
    l.op == "faq" && l.expr_body isa OpExpr && (l = l.expr_body)
    (l.op == "index" && !isempty(l.args) && l.args[1] isa VarExpr) || return nothing
    return l.args[1].name
end

"""The definitions of the model's observeds `observeds` in a flattened system, canonical, by observed: `(lhs, rhs)` in equation order."""
function observed_forms(flat::EarthSciAST.FlattenedSystem, observeds)
    out = Dict{String, Vector{Tuple{String, String}}}()
    for eq in flat.equations
        o = defined_name(eq)
        (o !== nothing && o in observeds) || continue
        push!(get!(out, o, Tuple{String, String}[]), (canonical_expr(eq.lhs), canonical_expr(eq.rhs)))
    end
    return out
end

"""The `ic` equations of a flattened system, canonical, sorted."""
ic_forms(flat::EarthSciAST.FlattenedSystem) = sort!([(String(x), canonical_expr(e)) for (x, e) in flat.field_ics])

"""The `faq` node's `output_idx`, or `nothing` when `e` is no `faq`."""
faq_output_idx(e) = (e isa OpExpr && e.op == "faq" && e.output_idx !== nothing) ? String[String(x) for x in e.output_idx] : nothing

"""
    rhs_sum_check(m, docs) -> Int

§8.1's `rhs_sum`: the flattened right-hand-side documents `docs`
(`operator => flattened document`, in code-point order of the operator) sum
to the flattened model `m`. Refuses `validation_rhs_sum_failed` at the first
difference; returns the number of summands compared, for the report.
"""
function rhs_sum_check(m::PlanModel, docs::AbstractVector)
    fail(msg) = refuse("validation_rhs_sum_failed", "/model", msg)
    flat = m.flat
    zeros_ = Dict{String, String}(x => canonical_expr(EarthSciAST.expression_from_json(wire_form(zero_rhs(m, x)))) for x in m.states)
    # The flattened model: each state's summands, lifted or not.
    want = Dict{String, Vector{String}}()
    lifted = Dict{String, Bool}()
    output_idx = Dict{String, Union{Nothing, Vector{String}}}()
    for (ei, x, l) in m.state_eqs
        haskey(want, x) && error("the flattened model has two equations for $x")
        s = state_summands(flat.equations[ei].rhs, zeros_[x], l)
        s === nothing && error("the flattened model's lifted equation for $x has no faq right-hand side")
        want[x] = s
        lifted[x] = l
        output_idx[x] = l ? faq_output_idx(flat.equations[ei].rhs) : nothing
    end
    obs = Set{String}(m.observeds)
    observeds = observed_forms(flat, obs)
    ics = ic_forms(flat)
    declared_params = Set{String}(p for p in m.params if !(is_forcing(m, p) && !(p in m.used_forcing)))
    got = Dict{String, Vector{String}}(x => String[] for x in m.states)
    for (o, d) in docs
        seen = Set{String}()
        for eq in d.equations
            x = derivative_variable(eq)
            x === nothing && (x = lifted_variable(eq, d.lifted_shapes))
            if x !== nothing
                haskey(want, x) || fail("operator $o's right-hand-side document has an equation for $x, which is not a state of the model")
                x in seen && fail("operator $o's right-hand-side document has two equations for $x")
                push!(seen, x)
                s = state_summands(eq.rhs, zeros_[x], lifted[x])
                s === nothing && fail("operator $o's right-hand-side document writes the lifted state $x without the lift's faq")
                if lifted[x] && !isempty(s) && faq_output_idx(eq.rhs) != output_idx[x]
                    fail("operator $o's right-hand-side document writes $x over the indices $(faq_output_idx(eq.rhs)), the model over $(output_idx[x])")
                end
                append!(got[x], s)
                continue
            end
            y = defined_name(eq)
            (y !== nothing && y in obs) || fail("operator $o's right-hand-side document has an equation that defines neither a state nor an observed: $(canonical_expr(eq.lhs))")
        end
        for x in m.states
            x in seen || fail("operator $o's right-hand-side document has no equation for the state $x")
        end
        # Observeds: the definitions the model gives them.
        for (y, defs) in observed_forms(d, obs)
            get(observeds, y, nothing) == defs ||
                fail("operator $o's right-hand-side document defines the observed $y otherwise than the model")
        end
        # Declarations: every state and parameter, with the model's shape and default.
        dstates = Set{String}(n for n in keys(d.state_variables) if !(n in obs))
        dstates == m.state_set ||
            fail("operator $o's right-hand-side document declares the states $(sort!(collect(dstates))), the model has $(m.canonical)")
        Set(keys(d.parameters)) == declared_params ||
            fail("operator $o's right-hand-side document declares the parameters $(sort!(collect(keys(d.parameters)))), not the model's $(sort!(collect(declared_params)))")
        for (dict, mdict) in ((d.state_variables, flat.state_variables), (d.parameters, flat.parameters)), (n, v) in dict
            n in obs && continue
            w = mdict[n]
            shape_form(v) == shape_form(w) ||
                fail("operator $o's right-hand-side document declares $n with the shape $(shape_form(v)), the model with $(shape_form(w))")
            default_form(v.default) == default_form(w.default) ||
                fail("operator $o's right-hand-side document declares $n with a default other than the model's")
        end
        ic_forms(d) == ics || fail("operator $o's right-hand-side document's ic equations are not the model's")
    end
    n = 0
    for x in m.canonical
        a, b = sort(want[x]), sort(got[x])
        if a != b
            missing_ = setdiff(a, b)
            extra = setdiff(b, a)
            fail(
                "the operators' right-hand sides of $x do not sum to the model's: " *
                    (isempty(missing_) ? "" : "the model's summand $(first(missing_)) is in no operator's ") *
                    (isempty(extra) ? "" : "the operators' summand $(first(extra)) is not the model's ") *
                    "($(length(a)) summands in the model, $(length(b)) in the operators)"
            )
        end
        n += length(a)
    end
    return n
end
