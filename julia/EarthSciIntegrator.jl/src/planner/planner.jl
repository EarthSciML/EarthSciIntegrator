# The planner's driver (spec/plan.md §4 "Order of refusals"): read the
# inputs, run passes 1–4, check and resolve the schedule and the settings,
# resolve strategies, and assemble plan.json.

"""
    PlanContext

The state of one planning run, passed to the passes and to the
[`PlanBackend`](@ref).
"""
mutable struct PlanContext
    inputs::PlanInputs
    config::AbstractDict
    model::PlanModel
    backend::PlanBackend
    axes::Axes
    time::Tuple{Float64, Float64}
    terms::Vector{Term}
    ops::Dict{String, PlanOperator}
end

"""
    PlanDraft

Everything a plan holds, as the planner computed it. `structure`, the
derived documents and `validation` come from the backend; with
[`PendingBackend`](@ref) (M2 stage 1) they are absent and
[`plan_dict`](@ref) raises [`PlannerStageError`](@ref).
"""
struct PlanDraft
    ctx::PlanContext
    stiffness::Dict{String, Tuple{Symbol, String}}
    schedule::Dict{String, Any}
    step::Dict{String, Any}
    partition::Dict{String, Any}
    precision::Dict{String, Any}
    derived::Union{Nothing, Tuple{AbstractDict, AbstractVector}}
end

# `EarthSciIntegrator.plan` (documented with its stub in src/planner/bands.jl).
function plan(config_path::AbstractString; tspan = nothing, backend::PlanBackend = PendingBackend())
    inp = read_inputs(config_path)                       # §4 steps 1–5
    check_scope(inp.flat)                                # step 6
    check_reserved_names(inp.file)                       # step 7
    m = plan_model(inp)                                  # step 8: lifted shapes (§6.1.3), then axes and time
    config = inp.config
    axes = resolve_axes(m, config)
    time = resolve_time(m, config; tspan)
    ctx = PlanContext(inp, config, m, backend, axes, time, Term[], Dict{String, PlanOperator}())
    # Passes 1 and 2, interleaved (§4 "Order of refusals", D81): terms and
    # owners, each term's locality, selection, then operator classes.
    ctx.terms = build_terms(m)
    term_localities!(ctx, ctx.terms)
    ctx.ops = select_operators(m, ctx.terms, config)
    opnames = sort!(collect(keys(ctx.ops)))
    for o in opnames
        operator_locality!(ctx.ops[o], axes)
    end
    for o in opnames                                   # pass 3
        dof_sets!(m, ctx.ops[o])
    end
    for o in opnames                                   # pass 4
        ctx.ops[o].structure = operator_structure(backend, ctx, ctx.ops[o])
    end
    # §7.1: the schedule.
    sched = config["schedule"]
    root = if sched == "auto"
        splitting = m.file.solver === nothing ? nothing : m.file.solver.splitting
        build_snode(auto_schedule(ctx.ops, splitting), "root", "/schedule", true)
    else
        build_snode(sched, "root", "/schedule", false)
    end
    check_schedule(ctx, root)
    # §7.3–§7.6.
    step = resolve_step(m, config)
    partition = resolve_partition(ctx, root)
    precision, psources = resolve_precision(ctx, root)
    haskey(config, "adjoint") && check_adjoint(ctx, root)
    # §5.5: stiffness and strategies.
    Δ = reference_steps(ctx, root)
    stiff = operator_stiffness!(ctx, root, Δ)
    schedule = resolve_tree(ctx, root, stiff)
    for n in allnodes(root)
        is_leaflike(n) && get(n.spec, "alg", "auto") == "auto" || continue
        check_rosenbrock_precision(n, _resolved_alg(schedule, n.id), precision[n.id], psources[n.id])
    end
    # §6: derived documents (stage 2), then §8 validation.
    derived = derived_and_validation(backend, ctx)
    return PlanDraft(ctx, stiff, schedule, step, partition, precision, derived)
end

function _resolved_alg(tree, id)
    tree["id"] == id && haskey(tree, "alg") && return tree["alg"]
    for k in ("sequence", "strang"), c in get(tree, k, Any[])
        a = _resolved_alg(c, id)
        a === nothing || return a
    end
    for k in ("subcycle", "every")
        haskey(tree, k) || continue
        a = _resolved_alg(tree[k], id)
        a === nothing || return a
    end
    return nothing
end

"""§4 step 7 (§6.1.4): a top-level component named `_jacobian` is refused, since the Jacobian document uses that name."""
function check_reserved_names(file::EarthSciAST.EsmFile)
    for d in (file.models, file.reaction_systems)
        d !== nothing && haskey(d, "_jacobian") && refuse("reserved_component_name", "/model",
            "the model has a top-level component named _jacobian, which the Jacobian document reserves (spec/plan.md §6.1.4)")
    end
end

"""The plan's object for an operator (§3, §5)."""
function operator_json(op::PlanOperator, stiff)
    op.structure === nothing && throw(PlannerStageError("operator $(op.name)'s structure (§5.4) needs EarthSciASTDiff"))
    return Dict{String, Any}(
        "kind" => "rhs",
        "locality" => locality_json(op),
        "writes" => op.writes,
        "reads" => op.reads,
        "terms" => Any[term_json(t) for t in op.terms],
        "structure" => op.structure,
        "stiffness" => Dict{String, Any}("class" => String(stiff[1]), "source" => stiff[2]))
end

function plan_dict(d::PlanDraft)
    ctx = d.ctx
    d.derived === nothing && throw(PlannerStageError("the derived documents (§6) and validation (§8) need EarthSciASTDiff"))
    derived, validation = d.derived
    out = Dict{String, Any}(
        "esm_integrator" => String(ctx.config["esm_integrator"]),
        "inputs" => inputs_json(ctx.inputs),
        "time" => Dict{String, Any}("start" => ctx.time[1], "end" => ctx.time[2]),
        "operators" => Dict{String, Any}(n => operator_json(op, d.stiffness[n]) for (n, op) in ctx.ops),
        "schedule" => d.schedule,
        "step" => d.step,
        "partition" => d.partition,
        "precision" => d.precision,
        "derived" => derived,
        "validation" => validation)
    haskey(ctx.config, "adjoint") && (out["checkpoint_schedule"] = Dict{String, Any}("kind" => "all", "macro_steps" => 0))
    return out
end

plan_json(d::PlanDraft) = jcs(plan_dict(d))
