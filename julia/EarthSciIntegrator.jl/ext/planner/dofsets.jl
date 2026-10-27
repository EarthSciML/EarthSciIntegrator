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
