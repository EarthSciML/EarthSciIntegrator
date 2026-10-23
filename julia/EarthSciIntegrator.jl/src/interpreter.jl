# The schedule interpreter and the macro step (spec/run-config.md §4–§5,
# spec/tape.md).
#
# Time arithmetic. Every invocation is given its interval by its two end
# points, and its Δ is `tb - ta`:
# - macro step k runs over [t₀ + k·dt, t₀ + (k+1)·dt] (multiplied, not
#   accumulated), the last one ending at t_f exactly (by the stop rule of
#   spec/methods.md §10);
# - a `strang` node's halves run over [ta, tm] and [tm, tb] with
#   tm = ta + (tb - ta)/2, its last child over [ta, tb];
# - a `subcycle` node's substep i (0-based) runs over [ta + i·δ, ta + (i+1)·δ]
#   with δ = (tb - ta)/n, the last one ending at tb exactly.

"""
    Functional

`adjoint.functional` (spec/run-config.md §7.2): the `sum` or `mean` of one
variable over `dofs`, at the final time or at every output time.
"""
struct Functional
    reduction::Symbol
    dofs::Vector{Int}
    over::Symbol
end

phi(F::Functional, u) = F.reduction === :sum ? sum(u[j] for j in F.dofs) : sum(u[j] for j in F.dofs) / length(F.dofs)
function phi_gradient(F::Functional, N)
    g = zeros(N)
    w = F.reduction === :sum ? 1.0 : 1.0 / length(F.dofs)
    for j in F.dofs
        g[j] += w
    end
    return g
end

"""
    Run

A run of a schedule on a toy problem: the configuration, and after
[`forward!`](@ref) the tape, the checkpoints (one per macro-step boundary, in
memory, spec/tape.md §4), the outputs and the statistics.
"""
mutable struct Run
    prob::ToyProblem
    p::Vector{Float64}
    u0::Vector{Float64}
    root::Node
    leaves::Dict{String, LeafNode}
    tspan::Tuple{Float64, Float64}
    macro_dt::Float64
    bounds::Vector{Float64}           # macro-step boundaries t₀ … t_f
    reasons::Vector{String}
    output_times::Vector{Float64}     # sorted; always includes t_f
    functional::Union{Nothing, Functional}
    wrt_u0::Bool
    wrt_p::Vector{Int}                # parameter indices differentiated
    bound_params::Vector{Int}
    # Filled by forward!
    controllers::Dict{String, ControllerState}
    tape::Tape
    checkpoints::Vector{Checkpoint}
    outputs::Vector{Tuple{Float64, Vector{Float64}}}
    u::Vector{Float64}
    accepted::Int
    rejected::Int
    nf::Int
    retcode::String
    failure::String
end

const _TOP_KEYS = Set(["esm_integrator", "model", "operators", "schedule", "step", "partition", "precision",
    "adjoint", "output", "restart"])

"""
    build_run(config; tspan, saveat=nothing, prob=toy_problem(config["model"]), p=prob.p, u0=prob.u0) -> Run

Build a run from a run config (a parsed JSON object). `tspan` and `saveat`
come from the conformance case (spec/toy-problems.md §2). Raises
[`UnsupportedError`](@ref) for constructs not implemented in this stage, and
[`ConfigError`](@ref) for invalid ones.
"""
function build_run(config::AbstractDict; tspan, saveat = nothing, prob::ToyProblem = toy_problem(config["model"]),
        p::AbstractVector = prob.p, u0::AbstractVector = prob.u0)
    for k in keys(config)
        k in _TOP_KEYS || throw(ConfigError("unknown top-level key '$k'"))
    end
    VersionNumber(config["esm_integrator"]) == SPEC_VERSION ||
        throw(ConfigError("run config is for spec $(config["esm_integrator"]); this implementation implements $SPEC_VERSION"))
    haskey(config, "operators") && throw(ConfigError("a toy problem defines its own operators; the run config must not"))
    for k in ("partition", "precision", "restart", "output")
        haskey(config, k) && throw(UnsupportedError("run-config key '$k' is not implemented in this stage"))
    end
    config["schedule"] == "auto" && throw(UnsupportedError("schedule \"auto\" needs the planner (M2)"))
    root = build_node(config["schedule"], prob, "root")
    nodes = allnodes(root)
    ids = [n.id for n in nodes]
    length(unique(ids)) == length(ids) || throw(ConfigError("duplicate node ids in the schedule"))
    leaves = Dict{String, LeafNode}(n.id => n for n in nodes if n isa LeafNode)

    step = get(config, "step", Dict{String, Any}())
    for k in keys(step)
        k in ("dt", "hit") || throw(UnsupportedError("step.$k is not implemented in this stage"))
    end
    haskey(step, "dt") || throw(UnsupportedError("a step without dt (macro steps from the root leaf's accepted steps, D49) is not implemented in this stage"))
    dt = parse_duration(step["dt"], prob)
    t0, tf = Float64(tspan[1]), Float64(tspan[2])
    bounds, reasons = macro_bounds(t0, tf, dt)

    outs = saveat === nothing ? [tf] : sort!(unique!(Float64[Float64(x) for x in saveat]))
    tf in outs || push!(outs, tf)
    for τ in outs
        (τ > t0 && τ <= tf) || throw(ConfigError("output time $τ outside ($t0, $tf]"))
        # Output times are tstops. Under a fixed step.dt they must be macro-step
        # boundaries (spec/run-config.md §8 rule 9), except that when the root is a
        # single leaf they are stops inside its interval (spec/methods.md §10).
        τ in bounds || root isa LeafNode ||
            throw(ConfigError("output time $τ is not a multiple of step.dt = $dt"))
    end

    functional = nothing
    wrt_u0 = false
    wrt_p = Int[]
    bound = Int[]
    for n in nodes
        (n isa LeafNode || n isa MapNode) || continue
        append!(bound, first.(n.bindings))
    end
    sort!(unique!(bound))
    if haskey(config, "adjoint")
        adj = config["adjoint"]
        haskey(adj, "checkpoint") && throw(UnsupportedError("adjoint.checkpoint: M1 checkpoints every macro step in memory"))
        functional = build_functional(adj["functional"], prob)
        for w in adj["wrt"]
            if w == "u0"
                wrt_u0 = true
            elseif startswith(w, "params:")
                pat = w[8:end]
                sel = pat == "numeric" ? collect(1:length(prob.pnames)) :
                      findall(n -> occursin(_glob(pat), n), prob.pnames)
                isempty(sel) && throw(ConfigError("wrt '$w' matches nothing"))
                any(in(bound), sel) && pat != "numeric" && throw(ConfigError("wrt '$w' names a parameter bound by bind_dt"))
                append!(wrt_p, setdiff(sel, bound))
            else
                throw(UnsupportedError("wrt '$w': only u0 and params are implemented in this stage"))
            end
        end
        sort!(unique!(wrt_p))
        # operator-interface.md §3: vjp_p is required when an adjoint is taken with respect to p.
        if !isempty(wrt_p)
            for leaf in values(leaves), op in leaf.ops.ops
                provides(op, :vjp_p) ||
                    throw(ConfigError("operator '$(opname(op))' of leaf $(leaf.id) does not provide vjp_p, which the adjoint with respect to p needs"))
            end
        end
    end
    return Run(prob, collect(Float64, p), collect(Float64, u0), root, leaves, (t0, tf), dt, bounds, reasons, outs,
        functional, wrt_u0, wrt_p, bound, Dict{String, ControllerState}(), Tape(), Checkpoint[],
        Tuple{Float64, Vector{Float64}}[], Float64[], 0, 0, 0, "Default", "")
end

_glob(pat) = Regex("^" * replace(pat, "." => "\\.", "*" => ".*") * "\$")

function build_functional(f::AbstractDict, prob::ToyProblem)
    red = haskey(f, "sum") ? :sum : :mean
    var = String(f[String(red)])
    over = Symbol(f["over"])
    over === :time && throw(UnsupportedError("functional over \"time\" (the integrated auxiliary state of spec/adjoint.md §2.1) is not implemented in this stage"))
    dofs = copy(variable_dofs(prob, var))
    if haskey(f, "where")
        for (set, sel) in f["where"]
            set == "cell" || throw(ConfigError("$(prob.name) has no index set '$set'"))
            lo, hi = sel isa Integer ? (sel, sel) : (sel[1], sel[2])
            (1 <= lo <= hi <= length(dofs)) || throw(ConfigError("where $set = $sel is out of range"))
            dofs = dofs[lo:hi]
        end
    end
    return Functional(red, dofs, over)
end

"""
    macro_bounds(t0, tf, dt) -> (bounds, reasons)

The macro-step boundaries under a fixed `step.dt`: `t₀ + k·dt`, the last one
`t_f` exactly, decided by the stop rule of spec/methods.md §10. The reason is
`"step.dt"`, or `"tstop"` for a last step shortened to land on `t_f`.
"""
function macro_bounds(t0::Float64, tf::Float64, dt::Float64)
    bounds = [t0]
    reasons = String[]
    k = 1
    while true
        ta = bounds[end]
        h, isstop = stop_rule(ta, dt, tf)
        if isstop
            push!(bounds, tf)
            push!(reasons, h < dt - 100 * eps(max(abs(ta), abs(tf))) ? "tstop" : "step.dt")
            break
        end
        push!(bounds, t0 + k * dt)
        push!(reasons, "step.dt")
        k += 1
    end
    return bounds, reasons
end

# ---------------------------------------------------------------------------
# Execution

"""Per-macro-step execution context: the macro index and Δ, occurrence counters, and the mode."""
mutable struct ExecCtx{M}
    k::Int
    Δmacro::Float64
    occ::Dict{String, Int}
    mode::M
end

"""Forward mode: controllers on, records written to `tape`."""
struct ForwardMode
    tape::Tape
    controllers::Dict{String, ControllerState}
end

"""
Replay mode: step sequences and masks from `tape`, parameters `p` (any
element type); with `trace`, every leaf and map invocation is recorded for the
reverse pass; `visit(t, u)` is called after every replayed leaf step.
"""
struct ReplayMode{P, V}
    tape::Tape
    p::P
    trace::Union{Nothing, Vector{Any}}
    visit::V
end

function next_path!(ctx::ExecCtx, id::String)
    occ = get(ctx.occ, id, 0)
    ctx.occ[id] = occ + 1
    return invocation_path(ctx.k, id, occ)
end

_params(run::Run, ::ForwardMode) = run.p
_params(run::Run, m::ReplayMode) = m.p

struct RunFailure <: Exception
    retcode::String
    msg::String
end

function _stops(run::Run, node::LeafNode, ta, tb)
    stops = Float64[]
    if node === run.root
        for τ in run.output_times
            ta < τ < tb && push!(stops, τ)
        end
    end
    push!(stops, tb)
    return stops
end

function advance!(run::Run, ctx::ExecCtx, node::LeafNode, u, ta::Float64, tb::Float64)
    path = next_path!(ctx, node.id)
    nb = filter(b -> b.second !== :leaf, node.bindings)
    pinv = apply_bindings(_params(run, ctx.mode), nb, 0.0, tb - ta, ctx.Δmacro)
    stops = _stops(run, node, ta, tb)
    mode = ctx.mode
    if mode isa ForwardMode
        cs = get!(() -> ControllerState(node.tab.controller), mode.controllers, node.id)
        out = leaf_forward(node, cs, u, pinv, ta, tb, path, ctx.Δmacro, stops)
        push!(mode.tape, out.record)
        if mode.tape === run.tape
            run.accepted += out.record.accepted
            run.rejected += out.record.rejected
            run.nf += out.nf
            append!(run.outputs, out.interior)
        end
        out.record.retcode == "Success" ||
            throw(RunFailure(out.record.retcode, "leaf $(node.id) ($path) ended with $(out.record.retcode)"))
        return out.u
    else
        rec = mode.tape[path]::LeafRecord
        mode.trace === nothing || push!(mode.trace, (:leaf, node, rec, copy(u), pinv, stops))
        return leaf_replay(node, u, pinv, rec, stops; visit = mode.visit)
    end
end

"""
    apply_frozen(op, u, p, t, Δ, mask) -> u′

Apply a map with its recorded decisions (spec/adjoint.md §5.7). The map is
re-applied and its decisions checked against the record; a flip is an error,
since the frozen-tape map is then not the one recorded.
"""
function apply_frozen(op::AbstractMapOperator, u, p, t, Δ, mask)
    out = similar(u, promote_type(eltype(u), eltype(p)))
    m = Vector{Bool}(undef, length(mask))
    apply!(op, out, m, u, p, t, Δ)
    m == mask || error("map $(opname(op)) took a different discrete decision on replay")
    return out
end

function advance!(run::Run, ctx::ExecCtx, node::MapNode, u, ta::Float64, tb::Float64)
    path = next_path!(ctx, node.id)
    Δ = tb - ta
    p = apply_bindings(_params(run, ctx.mode), node.bindings, Δ, Δ, ctx.Δmacro)
    mode = ctx.mode
    if mode isa ForwardMode
        mask = Vector{Bool}(undef, mask_length(node.op, length(u)))
        out = similar(u)
        apply!(node.op, out, mask, u, p, ta, Δ)
        push!(mode.tape, MapRecord(path, ta, Δ, mask))
        return out
    else
        rec = mode.tape[path]::MapRecord
        mode.trace === nothing || push!(mode.trace, (:map, node, rec, copy(u), p, ta))
        return apply_frozen(node.op, u, p, ta, Δ, rec.mask)
    end
end

function advance!(run::Run, ctx::ExecCtx, node::SequenceNode, u, ta::Float64, tb::Float64)
    for c in node.children
        u = advance!(run, ctx, c, u, ta, tb)
    end
    return u
end

function advance!(run::Run, ctx::ExecCtx, node::StrangNode, u, ta::Float64, tb::Float64)
    tm = ta + (tb - ta) / 2
    n = length(node.children)
    for i in 1:(n - 1)
        u = advance!(run, ctx, node.children[i], u, ta, tm)
    end
    u = advance!(run, ctx, node.children[n], u, ta, tb)
    for i in (n - 1):-1:1
        u = advance!(run, ctx, node.children[i], u, tm, tb)
    end
    return u
end

function advance!(run::Run, ctx::ExecCtx, node::SubcycleNode, u, ta::Float64, tb::Float64)
    path = next_path!(ctx, node.id)
    δ = (tb - ta) / node.n
    for i in 0:(node.n - 1)
        a = ta + i * δ
        b = i == node.n - 1 ? tb : ta + (i + 1) * δ
        u = advance!(run, ctx, node.child, u, a, b)
    end
    mode = ctx.mode
    if mode isa ForwardMode
        push!(mode.tape, SubcycleRecord(path, ta, tb - ta, node.n))
    else
        rec = mode.tape[path]::SubcycleRecord
        rec.n == node.n || error("subcycle $path: tape has $(rec.n) substeps, schedule $(node.n)")
    end
    return u
end

"""
    forward!(run) -> run

Run the schedule forward over the case's interval with every leaf's
controller on, recording the tape, a checkpoint at every macro-step boundary,
and the outputs. A non-success return code from any leaf ends the run with
that code (spec/methods.md §12), stored in `run.retcode` and `run.failure`.
"""
function forward!(run::Run)
    run.controllers = Dict{String, ControllerState}(id => ControllerState(l.tab.controller) for (id, l) in run.leaves)
    run.tape = Tape()
    run.checkpoints = Checkpoint[]
    run.outputs = [(run.tspan[1], copy(run.u0))]
    run.accepted = run.rejected = run.nf = 0
    u = copy(run.u0)
    mode = ForwardMode(run.tape, run.controllers)
    try
        for k in 0:(length(run.bounds) - 2)
            ta, tb = run.bounds[k + 1], run.bounds[k + 2]
            push!(run.checkpoints, Checkpoint(k, ta, copy(u), Dict(id => copy(c) for (id, c) in run.controllers)))
            ctx = ExecCtx(k, tb - ta, Dict{String, Int}(), mode)
            u = advance!(run, ctx, run.root, u, ta, tb)
            push!(run.tape, MacroRecord(string(k), ta, tb - ta, run.reasons[k + 1]))
            tb in run.output_times && push!(run.outputs, (tb, copy(u)))
        end
        K = length(run.bounds) - 1
        push!(run.checkpoints, Checkpoint(K, run.bounds[end], copy(u), Dict(id => copy(c) for (id, c) in run.controllers)))
        run.retcode = "Success"
    catch e
        e isa RunFailure || rethrow()
        run.retcode = e.retcode
        run.failure = e.msg
    end
    sort!(run.outputs; by = first)
    run.u = u
    return run
end

"""
    rerun_macro(run, k) -> (u, tape)

Re-run macro step `k` forward from its checkpoint with the controllers on
(restored from the checkpoint), into a fresh tape. Under D10 this reproduces
the original step sequence and end state bitwise (spec/adjoint.md §7.4).
"""
function rerun_macro(run::Run, k::Integer)
    cp = run.checkpoints[k + 1]
    ctrls = Dict(id => copy(c) for (id, c) in cp.controllers)
    tape = Tape()
    ta, tb = run.bounds[k + 1], run.bounds[k + 2]
    ctx = ExecCtx(k, tb - ta, Dict{String, Int}(), ForwardMode(tape, ctrls))
    u = advance!(run, ctx, run.root, copy(cp.u), ta, tb)
    push!(tape, MacroRecord(string(k), ta, tb - ta, run.reasons[k + 1]))
    return u, tape, ctrls
end

"""
    replay_macro(run, k, u, p; tape=run.tape, trace=nothing, visit=nothing) -> u

Replay macro step `k` from state `u` with parameters `p` (any element types)
using the tape's step sequences and masks (spec/tape.md §3).
"""
function replay_macro(run::Run, k::Integer, u, p; tape::Tape = run.tape, trace = nothing, visit = nothing)
    ta, tb = run.bounds[k + 1], run.bounds[k + 2]
    ctx = ExecCtx(k, tb - ta, Dict{String, Int}(), ReplayMode(tape, p, trace, visit))
    return advance!(run, ctx, run.root, u, ta, tb)
end

"""
    replay(run, u0=run.u0, p=run.p; tape=run.tape) -> (u_final, outputs)

Replay the whole run on the frozen tape (spec/adjoint.md §2.2), generic over
the element types of `u0` and `p`. `outputs` holds the state at every output
time, in order.
"""
function replay(run::Run, u0 = run.u0, p = run.p; tape::Tape = run.tape)
    T = promote_type(eltype(u0), eltype(p))
    u = convert(Vector{T}, copy(u0))
    pp = convert(Vector{T}, copy(p))
    outputs = Tuple{Float64, Vector{T}}[]
    visit = (t, x) -> (t in run.output_times && !(t in run.bounds) && push!(outputs, (t, copy(x))))
    for k in 0:(length(run.bounds) - 2)
        u = replay_macro(run, k, u, pp; tape, visit)
        tb = run.bounds[k + 2]
        tb in run.output_times && push!(outputs, (tb, copy(u)))
    end
    sort!(outputs; by = first)
    return u, outputs
end

"""The functional `J` of a run's outputs (spec/adjoint.md §2.1)."""
function functional_value(run::Run, outputs)
    F = run.functional
    F === nothing && error("the run config has no adjoint.functional")
    if F.over === :final
        return phi(F, last(outputs)[2])
    else
        return sum(phi(F, u) for (t, u) in outputs if t > run.tspan[1])
    end
end
