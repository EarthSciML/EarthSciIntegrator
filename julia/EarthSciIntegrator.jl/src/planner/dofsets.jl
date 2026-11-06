# Pass 3: DOF sets (spec/plan.md §5.3).

"""
    observed_closure(m, names) -> Vector{String}

`names` together with, transitively, every name referenced by the
definitions of the observeds among them; sorted by code point.
"""
function observed_closure(m::PlanModel, names)
    out = Set{String}(names)
    queue = [n for n in out if is_observed(m, n)]
    while !isempty(queue)
        o = pop!(queue)
        def = get(m.observed_defs, o, nothing)
        def === nothing && continue
        for n in referenced_names(m, def)
            n in out && continue
            push!(out, n)
            is_observed(m, n) && push!(queue, n)
        end
    end
    return sort!(collect(out))
end

"""
    dof_sets!(m, op)

§5.3: `writes`, the dependent variables of the operator's terms; `reads`,
every state, parameter and observed name its terms reference, with the
names the referenced observeds' definitions reference, transitively. Both
sorted by code point (Julia's `String` order is code-point order).
"""
function dof_sets!(m::PlanModel, op::PlanOperator)
    op.writes = sort!(unique!([t.variable for t in op.terms]))
    op.reads = observed_closure(m, reduce(vcat, (t.refs for t in op.terms); init = String[]))
    return op
end

"""
    mark_used_forcing!(m, terms)

§4.1: a data-fed parameter is **used** when some operator's `reads` lists
it (every term belongs to an operator, so: when a term references it,
directly or through observed definitions) or an `ic` equation of the
flattened model references it, directly or through observed definitions.
"""
function mark_used_forcing!(m::PlanModel, terms)
    empty!(m.used_forcing)
    isempty(m.forcing) && return m
    names = String[]
    for t in terms
        append!(names, t.refs)
    end
    for (_, e) in m.flat.field_ics
        append!(names, referenced_names(m, e))
    end
    for n in observed_closure(m, unique!(names))
        is_forcing(m, n) && push!(m.used_forcing, n)
    end
    return m
end

"""The used data-fed parameters (§4.1), in code-point order: the plan's `forcing` (§3) and the canonical forcing order's names (§5.5.1)."""
used_forcing(m::PlanModel) = sort!(collect(m.used_forcing))

"""
    check_forcing_scale(m)

§4.1: every used data-fed parameter needs a `default` that is nonzero at
every element (a scalar default broadcast over the shape counts at every
element); otherwise `forcing_without_scale` at `/model`, naming the first
such parameter in code-point order.
"""
function check_forcing_scale(m::PlanModel)
    for n in used_forcing(m)
        d = m.flat.parameters[n].default
        why = d === nothing ? "has no default" :
            any(x -> !(x isa Real) || x == 0, _default_elements(d)) ? "has a default with a zero element" : nothing
        why === nothing && continue
        refuse(
            "forcing_without_scale", "/model",
            "data-fed parameter $n $why, so the planner's checks have no scale to sample it around (spec/plan.md §4.1)"
        )
    end
    return nothing
end

_default_elements(d::Real) = (d,)
_default_elements(d::AbstractArray) = Iterators.flatten(_default_elements(x) for x in d)
_default_elements(d) = (d,)
