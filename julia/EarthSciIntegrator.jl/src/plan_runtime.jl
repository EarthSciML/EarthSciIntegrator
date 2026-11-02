# Executing a plan (spec/operator-interface.md §8, spec/toy-problems.md §4):
# the operators the schedule names, built from the plan's entries and their
# derived documents, and a run of the plan's resolved schedule on them.
#
# Every operator acts on the whole model state in canonical order (§8.2,
# spec/plan.md §4): state names sorted by code point, each array's cells
# column-major. A leaf advances that whole vector, as at M1, with its error
# norm and first step over its DOFs, the union of its operators' `writes`
# (§3); the states no operator of the leaf writes have a zero right-hand side
# there, so they stay constant over the leaf's interval, as §8.2 requires.

"""
    PlanDocument

One derived document built as spec/operator-interface.md §8.1 says:
`rhs_program(esm_problem(path, tspan))` with the native compiler. `perm`
maps each canonical DOF to its slot in the program's `u`, `pidx` each of the
program's parameters to its index in the model's parameter vector (§8.2),
and `nslots` is the length of the program's `u`. Workspaces are kept in a
pool, one per concurrent caller (EarthSciAST's `rhs_workspace`), so one
document serves every thread.
"""
struct PlanDocument
    path::String
    prog::EarthSciAST.RhsProgram
    fs::EarthSciAST.ForcingSet
    aux::Vector{Float64}
    nslots::Int
    perm::Vector{Int}
    pidx::Vector{Int}
    var_map::Dict{String, Int}
    u0::Vector{Float64}
    pool::Vector{Any}
    lock::ReentrantLock
end

"""A workspace of a [`PlanDocument`](@ref): EarthSciAST's, and the slot, output and parameter buffers of one call."""
struct DocWorkspace
    ws::Any
    U::Vector{Float64}
    du::Vector{Float64}
    P::Vector{Float64}
end

"""Load and build the derived document at `path` (§8.1). A build failure is a [`ConfigError`](@ref) naming the operator and the document."""
function build_plan_document(path::AbstractString, opname::AbstractString, tspan)
    prob, prog = try
        prob = EarthSciAST.esm_problem(path, tspan; compiler = :native, base_path = dirname(path))
        prob, EarthSciAST.rhs_program(prob)
    catch e
        e isa InterruptException && rethrow()
        throw(ConfigError("operator '$opname': building $(basename(path)) failed: " * sprint(showerror, e)))
    end
    vm = Dict{String, Int}(String(k) => Int(v) for (k, v) in prob.var_map)
    nslots = EarthSciAST.n_states(prog)
    nslots == length(prob.u0) ||
        throw(ConfigError("operator '$opname': $(basename(path)) builds a program of $nslots states and a problem of $(length(prob.u0))"))
    return PlanDocument(
        String(path), prog, EarthSciAST.default_forcing_set(prog), zeros(EarthSciAST.tile_layout(prog).n_aux),
        nslots, Int[], Int[], vm, Float64.(prob.u0), Any[], ReentrantLock()
    )
end

"""`d` with its canonical permutation and parameter map for the layout `L` and parameter names `pnames`."""
function bind_document(d::PlanDocument, L::DofLayout, pnames::Vector{String}, opname)
    perm = Int[
        haskey(d.var_map, n) ? d.var_map[n] :
            throw(ConfigError("operator '$opname': $(basename(d.path)) does not declare state element $n"))
            for n in L.names
    ]
    pi = Dict(n => i for (i, n) in enumerate(pnames))
    pidx = Int[
        haskey(pi, s.name) ? pi[s.name] :
            throw(ConfigError("operator '$opname': $(basename(d.path)) has parameter $(s.name), which the model does not declare"))
            for s in EarthSciAST.parameters(d.prog)
    ]
    return PlanDocument(d.path, d.prog, d.fs, d.aux, d.nslots, perm, pidx, d.var_map, d.u0, d.pool, d.lock)
end

function _acquire(d::PlanDocument)
    w = lock(d.lock) do
        isempty(d.pool) ? nothing : pop!(d.pool)
    end
    w === nothing || return w
    return DocWorkspace(EarthSciAST.rhs_workspace(d.prog), zeros(d.nslots), zeros(d.nslots), zeros(length(d.pidx)))
end
_release(d::PlanDocument, w::DocWorkspace) = lock(() -> push!(d.pool, w), d.lock)

"""
    evaluate(f, d, u, p, t)

Evaluate document `d` at the canonical state `u` (every other slot, band
fields included, zero) and the model parameters `p`, then call `f(du)` on
the program's output in its own slot order, and return what `f` returns.
"""
function evaluate(f, d::PlanDocument, u, p, t)
    (eltype(u) <: Float64 && eltype(p) <: Float64) || throw(
        UnsupportedError(
            "operators built from a plan evaluate Float64 states and parameters only " *
                "(EarthSciAST's program takes no dual numbers; spec/operator-interface.md §8.3)"
        )
    )
    w = _acquire(d)
    try
        U, P = w.U, w.P
        fill!(U, 0.0)
        @inbounds for i in eachindex(d.perm)
            U[d.perm[i]] = u[i]
        end
        @inbounds for j in eachindex(d.pidx)
            P[j] = p[d.pidx[j]]
        end
        EarthSciAST.rhs!(w.du, d.prog, w.ws, U, d.aux, P, Float64(t), d.fs)
        return f(w.du)
    finally
        _release(d, w)
    end
end

"""
    PlannedOperator

An operator a plan supplies (spec/operator-interface.md §8): the right-hand
side and Jacobian documents built by EarthSciAST, the canonical DOFs it
writes, its structure, and every point of its bands as `(slot, row, col)`,
the slot of the band field `J_k` cell in the Jacobian document and the
canonical row and column DOFs it adds into. It provides `rhs`, `jvp_u`,
`vjp_u` and `jac` (§8.3); the other calls are not provided at M2.
"""
struct PlannedOperator <: AbstractOperator
    name::String
    writes::Vector{Int}
    structure::Structure
    depends_on_t::Bool
    linear_in_u::Bool
    rhs::PlanDocument
    jac::PlanDocument
    points::Vector{NTuple{3, Int}}
end

opname(op::PlannedOperator) = op.name
structure(op::PlannedOperator) = op.structure
depends_on_t(op::PlannedOperator) = op.depends_on_t
linear_in_u(op::PlannedOperator) = op.linear_in_u
writes(op::PlannedOperator, N) = op.writes

function rhs!(op::PlannedOperator, out, u, p, t)
    d = op.rhs
    return evaluate(d, u, p, t) do du
        @inbounds for i in op.writes
            out[i] += du[d.perm[i]]
        end
        out
    end
end

function jac!(op::PlannedOperator, out, u, p, t)
    return evaluate(op.jac, u, p, t) do du
        for (s, r, c) in op.points
            addentry!(out, r, c, du[s])
        end
        out
    end
end

function jvp_u!(op::PlannedOperator, out, u, p, t, v)
    return evaluate(op.jac, u, p, t) do du
        @inbounds for (s, r, c) in op.points
            out[r] += du[s] * v[c]
        end
        out
    end
end

function vjp_u!(op::PlannedOperator, out, u, p, t, w)
    return evaluate(op.jac, u, p, t) do du
        @inbounds for (s, r, c) in op.points
            out[c] += du[s] * w[r]
        end
        out
    end
end

# ---------------------------------------------------------------------------
# The plan's problem

"""Operator names a resolved schedule node and its descendants advance (`leaf`, `coupled` and `frozen` entries)."""
function schedule_operators(node, acc = String[])
    for k in ("leaf", "coupled")
        haskey(node, k) && append!(acc, node[k] isa AbstractString ? [node[k]] : node[k])
    end
    for f in get(node, "frozen", Any[])
        push!(acc, f isa AbstractString ? f : f["every"])
    end
    for k in ("sequence", "strang"), c in get(node, k, Any[])
        schedule_operators(c, acc)
    end
    for k in ("subcycle", "every")
        haskey(node, k) && schedule_operators(node[k], acc)
    end
    return unique!(acc)
end

"""
The canonical layout (spec/plan.md §4) of a built right-hand-side document:
its states sorted by code point, each array's cells column-major, with the
extents its program's state layout gives.
"""
function canonical_layout(d::PlanDocument)
    blocks = EarthSciAST.state_layout(d.prog)
    ext = Dict{String, Vector{Int}}(b.name => collect(Int, b.shape) for b in blocks)
    offset = Dict{String, Int}()
    names = String[]
    for x in sort!(collect(keys(ext)))
        offset[x] = length(names)
        append!(names, element_name(x, c) for c in cells(ext[x]))
    end
    return DofLayout(offset, ext, names)
end

"""
    PlanProblem

What a run of a plan needs besides its schedule: the problem (a
[`ToyProblem`](@ref) container holding the planned operators, the canonical
initial state and the model's parameters), the parsed `plan.json`, and the
canonical layout.
"""
struct PlanProblem
    prob::ToyProblem
    plan::AbstractDict
    layout::DofLayout
end

"""
    plan_problem(dir; cache = nothing) -> PlanProblem

Build every operator the resolved schedule of the plan in `dir` names, from
its entry `operators.<name>` and its documents `derived/<name>.rhs.esm` and
`derived/<name>.jacobian.esm` (spec/operator-interface.md §8). A derived
document whose bytes do not hash to the plan's `derived` entry is refused:
the plan is stale. `cache`, a dictionary, keeps built documents by their hash
across plans, so a document many plans share is built once.
"""
function plan_problem(dir::AbstractString; cache::Union{Nothing, AbstractDict} = nothing)
    planfile = joinpath(dir, "plan.json")
    plan = JSON.parsefile(planfile)
    VersionNumber(plan["esm_integrator"]) == SPEC_VERSION ||
        throw(ConfigError("plan is for spec $(plan["esm_integrator"]); this implementation implements $SPEC_VERSION"))
    tspan = (Float64(plan["time"]["start"]), Float64(plan["time"]["end"]))
    for (node, roles) in plan["precision"], (role, v) in roles
        v == "f64" || throw(UnsupportedError("precision $role = $v on node $node: plans are executed in f64 only in this stage"))
    end
    names = schedule_operators(plan["schedule"])
    isempty(names) && throw(ConfigError("the plan's schedule advances no operator"))
    function doc(op, kind)
        rel = "derived/$op.$kind.esm"
        entry = get(plan["derived"], rel, nothing)
        entry === nothing && throw(ConfigError("operator '$op': the plan lists no derived document $rel"))
        path = joinpath(dir, rel)
        isfile(path) || throw(ConfigError("operator '$op': $rel is missing from the plan directory"))
        h = bytes2hex(sha256(read(path)))
        h == entry["sha256"] || throw(ConfigError("operator '$op': $rel does not match the plan's hash (the plan is stale)"))
        cache === nothing && return build_plan_document(path, op, tspan)
        key = (h, tspan)
        return get!(() -> build_plan_document(path, op, tspan), cache, key)
    end
    built = Dict{String, Tuple{PlanDocument, PlanDocument}}(n => (doc(n, "rhs"), doc(n, "jacobian")) for n in names)
    # The layout and the parameters come from the first operator's right-hand
    # side; every other document must declare the same states (§8.2).
    first_rhs = built[first(names)][1]
    L = canonical_layout(first_rhs)
    params = EarthSciAST.parameters(first_rhs.prog)
    pnames = String[s.name for s in params]
    p = Float64[
        s.default === nothing ? throw(ConfigError("parameter $(s.name) has no default (spec/operator-interface.md §8.2)")) :
            s.default for s in params
    ]
    issorted(pnames) || (perm = sortperm(pnames); pnames = pnames[perm]; p = p[perm])
    ops = Dict{String, AbstractOperator}()
    u0 = nothing
    for n in names
        r, j = (bind_document(d, L, pnames, n) for d in built[n])
        rs = canonical_layout(r)
        rs.names == L.names ||
            throw(ConfigError("operator '$n': derived/$n.rhs.esm declares other states than derived/$(first(names)).rhs.esm"))
        x0 = r.u0[r.perm]
        u0 === nothing && (u0 = x0)
        bitwise_same(u0, x0) ||
            throw(ConfigError("operator '$n': derived/$n.rhs.esm gives another initial state than derived/$(first(names)).rhs.esm"))
        push!(ops, n => planned_operator(n, plan["operators"][n], L, r, j))
    end
    variables = Pair{String, Vector{Int}}[
        x => collect((L.offset[x] + 1):(L.offset[x] + prod(L.extents[x]; init = 1))) for x in sort!(collect(keys(L.offset)))
    ]
    name = String(get(get(plan["inputs"], "model", Dict()), "path", "plan"))
    prob = ToyProblem(
        name, length(L.names), u0, p, pnames, tspan, ops, Dict{String, AbstractMapOperator}(), variables,
        Function[], nothing, false
    )
    return PlanProblem(prob, plan, L)
end

bitwise_same(a, b) = length(a) == length(b) && all(i -> reinterpret(UInt64, a[i]) == reinterpret(UInt64, b[i]), eachindex(a))

"""The [`PlannedOperator`](@ref) of plan entry `entry` (spec/plan.md §3) on its built documents."""
function planned_operator(name, entry, L::DofLayout, r::PlanDocument, j::PlanDocument)
    st = entry["structure"]
    bands = BandEntry[BandEntry(b) for b in st["bands"]]
    w = state_dofs(L, String.(entry["writes"]))
    # The band fields: JacobianEval.J_k over the row's extents and the
    # contracted dimensions (spec/plan.md §6.2), column-major.
    fields = map(enumerate(bands)) do (k, b)
        ext = vcat(L.extents[b.row], Int[hi for (_, _, hi) in b.contracted])
        Int[
            get(j.var_map, element_name("JacobianEval.J_$k", c)) do
                throw(ConfigError("operator '$name': derived/$name.jacobian.esm has no band field element $(element_name("JacobianEval.J_$k", c))"))
            end for c in cells(ext)
        ]
    end
    points = NTuple{3, Int}[(fields[k][f], row, col) for (k, f, row, col) in band_points(L, bands)]
    return PlannedOperator(
        String(name), w, planned_structure(entry, L), Bool(st["depends_on_t"]), Bool(st["linear_in_u"]), r, j, points
    )
end

"""
The structure of a planned operator (spec/operator-interface.md §8.3):
`block_diagonal` when it is pointwise with `structure.block` and the model's
states are all arrays of one shape (`ncells` its cells) or all scalars
(`ncells` = 1), with `n` the number of states; otherwise `dense`.
"""
function planned_structure(entry, L::DofLayout)
    st = entry["structure"]
    entry["locality"]["class"] == "pointwise" && haskey(st, "block") || return DenseStructure()
    exts = collect(values(L.extents))
    all(==(first(exts)), exts) || return DenseStructure()
    return BlockDiagonalStructure(prod(first(exts); init = 1), length(exts))
end

"""
    build_plan_run(dir; saveat = nothing, tspan = nothing, cache = nothing) -> (Run, PlanProblem)

A run of the plan in `dir` (spec/operator-interface.md §8, spec/toy-problems.md
§4): its operators built by [`plan_problem`](@ref), its resolved schedule and
step, over the plan's `time`. A caller's `tspan` other than the plan's is
refused (the plan is stale, spec/plan.md §3). Run it with [`forward!`](@ref).
"""
function build_plan_run(dir::AbstractString; saveat = nothing, tspan = nothing, cache = nothing)
    pp = plan_problem(dir; cache)
    prob, plan = pp.prob, pp.plan
    if tspan !== nothing && (Float64(tspan[1]), Float64(tspan[2])) != prob.tspan
        throw(ConfigError("run interval $(tspan) differs from the plan's time $(prob.tspan): the plan is stale (spec/plan.md §3)"))
    end
    config = Dict{String, Any}(
        "esm_integrator" => plan["esm_integrator"], "schedule" => plan["schedule"], "step" => plan["step"]
    )
    return build_run(config; tspan = prob.tspan, saveat, prob), pp
end

"""
    layout_dofs(pp::PlanProblem, layout) -> Vector{Int}

The canonical DOFs of the variables `layout` lists, each variable's cells
column-major, in the listed order (spec/toy-problems.md §4.3).
"""
function layout_dofs(pp::PlanProblem, layout)
    return reduce(vcat, (variable_dofs(pp.prob, String(x)) for x in layout); init = Int[])
end
