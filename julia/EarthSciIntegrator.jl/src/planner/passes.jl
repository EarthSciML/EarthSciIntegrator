# Pass 2 over the model (spec/plan.md §5.2): axes, the terms' bands and
# classes, operator classes and radii.

"""
    resolve_axes(m, config) -> Axes

§5.2.1: the run config's `axes`, else no vertical axis and every index set
that is a dimension of some state's shape (lifted states by §6.1.3), other
than the vertical one, horizontal, in `index_sets` order. A named index set
that does not exist, a name in both roles, or more than two horizontal axes
is `axes_unresolved` at `/axes`.
"""
function resolve_axes(m::PlanModel, config::AbstractDict)
    ax = get(config, "axes", Dict{String, Any}())
    vertical = haskey(ax, "vertical") ? String(ax["vertical"]) : nothing
    if vertical !== nothing
        haskey(m.sizes, vertical) || refuse("axes_unresolved", "/axes", "the vertical axis $vertical is not an index set of the model")
    end
    horizontal = if haskey(ax, "horizontal")
        h = String.(ax["horizontal"])
        for n in h
            haskey(m.sizes, n) || refuse("axes_unresolved", "/axes", "the horizontal axis $n is not an index set of the model")
            n == vertical && refuse("axes_unresolved", "/axes", "index set $n is named both horizontal and vertical")
        end
        h
    else
        used = Set{String}()
        for s in m.states
            haskey(m.shapes, s) && union!(used, m.shapes[s])
        end
        [k for k in keys(m.sizes) if k in used && k != vertical]
    end
    length(horizontal) > 2 && refuse("axes_unresolved", "/axes",
        "$(length(horizontal)) horizontal axes ($(join(horizontal, ", "))); at most two (give axes in the run config)")
    return Axes(horizontal, vertical)
end

"""
    term_localities!(ctx, terms)

§5.2.2–§5.2.4 for every term: its bands from the backend, then class and
radius. Left `nothing` when the backend has no bands for it yet.
"""
function term_localities!(ctx, terms::Vector{Term})
    for t in terms
        bands = term_bands(ctx.backend, ctx, t)
        bands === nothing && continue
        t.locality, t.radius = classify_bands(bands, ctx.model.shapes, ctx.axes, ctx.model.sizes)
    end
end

"""
    operator_locality!(op, axes)

§5.2.4: the coarsest class of the operator's terms; a stencil operator's
radius lists every horizontal axis, each direction the maximum over its
stencil terms.
"""
function operator_locality!(op::PlanOperator, axes::Axes)
    any(t -> t.locality === nothing, op.terms) && return op
    op.locality = reduce(coarser, (t.locality for t in op.terms))
    if op.locality === :stencil
        op.radius = Dict{String, Tuple{Int, Int}}(X => (0, 0) for X in axes.horizontal)
        for t in op.terms
            t.locality === :stencil || continue
            for (X, (l, u)) in t.radius
                L, U = op.radius[X]
                op.radius[X] = (max(L, l), max(U, u))
            end
        end
    end
    return op
end

"""The operator's class, or a [`PlannerStageError`](@ref) while its terms' bands are pending."""
function need_locality(op::PlanOperator)
    op.locality === nothing && throw(PlannerStageError("the locality of operator $(op.name) needs its terms' EarthSciASTDiff bands"))
    return op.locality
end

"""The plan's `locality` object (§5.2.4)."""
function locality_json(op::PlanOperator)
    d = Dict{String, Any}("class" => String(need_locality(op)))
    op.locality === :stencil && (d["radius"] = Dict{String, Any}(X => [l, u] for (X, (l, u)) in op.radius))
    return d
end
