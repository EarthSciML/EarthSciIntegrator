# Executing a plan (spec/operator-interface.md §8, spec/toy-problems.md §4):
# the operators the schedule names, built from the plan's entries and their
# derived documents, and a run of the plan's resolved schedule on them.
#
# A `PlannedOperator` acts on the whole model state in canonical order (§8.2,
# spec/plan.md §4): state names sorted by code point, each array's cells
# column-major. A leaf advances only its own state, the union of its
# operators' `writes` in canonical order (§8.2): its operators are bound to
# it as `LeafOperator`s, which the interpreter binds to the model state at
# the start of each invocation (`with_background`) and which fill every
# other state from it, so those are constants of the leaf.

"""
A workspace of a [`PlanDocument`](@ref) for one caller: EarthSciAST's, the
slot, output and parameter buffers, and the auxiliary buffer and forcing set
the program's `rhs!` is given. Nothing a call writes is shared between
callers.
"""
struct DocWorkspace{W}
    ws::W
    U::Vector{Float64}
    du::Vector{Float64}
    P::Vector{Float64}
    aux::Vector{Float64}
    fs::EarthSciAST.ForcingSet
end

"""A new workspace for program `prog`, with `nslots` slots and `npar` parameters."""
DocWorkspace(prog, nslots::Integer, npar::Integer) = DocWorkspace(
    EarthSciAST.rhs_workspace(prog), zeros(nslots), zeros(nslots), zeros(npar),
    zeros(EarthSciAST.tile_layout(prog).n_aux), EarthSciAST.default_forcing_set(prog)
)

"""
    PlanDocument

One derived document built as spec/operator-interface.md §8.1 says:
`rhs_program(esm_problem(path, tspan))` with the native compiler. `perm`
maps each canonical DOF to its slot in the program's `u`, `pidx` each of the
program's parameters to its index in the model's parameter vector (§8.2),
and `nslots` is the length of the program's `u`. `slots` maps each state
element name to its slot, from the program's state layout, which the build
checks the problem's `var_map` and `u0` agree with. Workspaces
([`DocWorkspace`](@ref), everything a call writes) are kept in a pool, one per
concurrent caller, so one document serves every thread, and every plan that
shares it through a cache.
"""
struct PlanDocument{P, W}
    path::String
    prog::P
    nslots::Int
    perm::Vector{Int}
    pidx::Vector{Int}
    slots::Dict{String, Int}
    u0::Vector{Float64}
    pool::Vector{DocWorkspace{W}}
    lock::ReentrantLock
end

"""`d` with another `path` (the plan directory's own, for a document a cache built for another plan), `perm` or `pidx`."""
_with(d::PlanDocument; path = d.path, perm = d.perm, pidx = d.pidx) =
    PlanDocument(String(path), d.prog, d.nslots, perm, pidx, d.slots, d.u0, d.pool, d.lock)

"""
The slot of every state element of program `prog` in its `u`, from its state
layout: each block's cells column-major from the block's offset.
"""
function program_slots(prog)
    slots = Dict{String, Int}()
    for b in EarthSciAST.state_layout(prog)
        for (k, c) in enumerate(cells(collect(Int, b.shape)))
            slots[element_name(b.name, c)] = b.offset + k - 1
        end
    end
    return slots
end

"""Load and build the derived document at `path` (§8.1). A build failure is a [`ConfigError`](@ref) naming the operator and the document."""
function build_plan_document(path::AbstractString, opname::AbstractString, tspan)
    # A document whose `ic` equations read one of its data-fed parameters
    # (spec/plan.md §6.3) is built twice: EarthSciAST's `rhs_program` cannot
    # yet rebuild such an `ic` equation, so the program is built without the
    # `ic` equations and the initial state from the document as written.
    prob, prog, u0 = try
        prob = EarthSciAST.esm_problem(path, tspan; compiler = :native, base_path = dirname(path))
        flat = EarthSciAST.flatten(EarthSciAST.load_path(path))
        if ic_reads(flat, Set(fed_parameters(flat)))
            pp = EarthSciAST.esm_problem(
                EarthSciAST.FlattenedSystem(flat; field_ics = empty(flat.field_ics)), tspan; compiler = :native
            )
            Dict(String(k) => Int(v) for (k, v) in pp.var_map) == Dict(String(k) => Int(v) for (k, v) in prob.var_map) ||
                error("its two builds lay out the states differently")
            pp, EarthSciAST.rhs_program(pp), prob.u0
        else
            prob, EarthSciAST.rhs_program(prob), prob.u0
        end
    catch e
        e isa InterruptException && rethrow()
        throw(ConfigError("operator '$opname': building $(basename(path)) failed: " * sprint(showerror, e)))
    end
    nslots = EarthSciAST.n_states(prog)
    nslots == length(prob.u0) ||
        throw(ConfigError("operator '$opname': $(basename(path)) builds a program of $nslots states and a problem of $(length(prob.u0))"))
    # The program's `u` is laid out as its state layout says; the problem's
    # `var_map` indexes `u0`. Both must name the same slot for every element,
    # or the initial state and every evaluation would put states in the wrong
    # slots.
    slots = program_slots(prog)
    length(slots) == nslots ||
        throw(ConfigError("operator '$opname': $(basename(path)): the program's state layout covers $(length(slots)) of its $nslots states"))
    for (k, v) in prob.var_map
        n = String(k)
        get(slots, n, 0) == Int(v) || throw(
            ConfigError(
                "operator '$opname': $(basename(path)): the problem puts state element $n at $v, " *
                    "the program's state layout at $(get(slots, n, "no slot"))"
            )
        )
    end
    length(prob.var_map) == nslots ||
        throw(ConfigError("operator '$opname': $(basename(path)): the problem names $(length(prob.var_map)) of the program's $nslots states"))
    w = DocWorkspace(prog, nslots, length(EarthSciAST.parameters(prog)))
    return PlanDocument(
        String(path), prog, nslots, Int[], Int[], slots, Float64.(u0), [w], ReentrantLock()
    )
end

"""`d` with its canonical permutation and parameter map for the layout `L` and parameter names `pnames`."""
function bind_document(d::PlanDocument, L::DofLayout, pnames::Vector{String}, opname)
    perm = Int[
        haskey(d.slots, n) ? d.slots[n] :
            throw(ConfigError("operator '$opname': $(basename(d.path)) does not declare state element $n"))
            for n in L.names
    ]
    pindex = Dict(n => i for (i, n) in enumerate(pnames))
    pidx = Int[
        haskey(pindex, s.name) ? pindex[s.name] :
            throw(ConfigError("operator '$opname': $(basename(d.path)) has parameter $(s.name), which the model does not declare"))
            for s in EarthSciAST.parameters(d.prog)
    ]
    return _with(d; perm, pidx)
end

function _acquire(d::PlanDocument{P, W}) where {P, W}
    w = lock(d.lock) do
        isempty(d.pool) ? nothing : pop!(d.pool)
    end
    w === nothing || return w
    return DocWorkspace(d.prog, d.nslots, length(d.pidx))::DocWorkspace{W}
end
_release(d::PlanDocument, w::DocWorkspace) = lock(() -> push!(d.pool, w), d.lock)

function _check_f64(u, p)
    (eltype(u) <: Float64 && eltype(p) <: Float64) || throw(
        UnsupportedError(
            "operators built from a plan evaluate Float64 states and parameters only " *
                "(EarthSciAST's program takes no dual numbers; spec/operator-interface.md §8.3)"
        )
    )
    return nothing
end

"""
    evaluate(f, d, u, p, t)

Evaluate document `d` at the canonical state `u` (every other slot, band
fields included, zero) and the model parameters `p`, with the forcing set
`fs` (the program's default set when `nothing`, for a document that reads no
data), then call `f(du)` on the program's output in its own slot order, and
return what `f` returns.
"""
function evaluate(f, d::PlanDocument, u, p, t, fs = nothing)
    _check_f64(u, p)
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
        EarthSciAST.rhs!(w.du, d.prog, w.ws, U, w.aux, P, Float64(t), fs === nothing ? w.fs : fs)
        return f(w.du)
    finally
        _release(d, w)
    end
end

"""
    BandCache

The values of a planned operator's band points at the last linearization
point `(u, p, t)` it was evaluated at, so `jac`, `jvp_u` and `vjp_u` at one
point evaluate the Jacobian document once. `vals` is replaced, never written
in place.
"""
mutable struct BandCache
    valid::Bool
    t::Float64
    u::Vector{Float64}
    p::Vector{Float64}
    vals::Vector{Float64}
    const lock::ReentrantLock
end
BandCache() = BandCache(false, NaN, Float64[], Float64[], Float64[], ReentrantLock())

"""
    PlannedOperator

An operator a plan supplies (spec/operator-interface.md §8): the right-hand
side and Jacobian documents built by EarthSciAST, the canonical DOFs it
writes, its structure, and every point of its bands as `(slot, row, col)`,
the slot of the band field `J_k` cell in the Jacobian document and the
canonical row and column DOFs it adds into. It provides `rhs`, `jvp_u`,
`vjp_u` and `jac` (§8.3); the other calls are not provided at M2.
"""
struct PlannedOperator{R <: PlanDocument, J <: PlanDocument} <: AbstractOperator
    name::String
    writes::Vector{Int}
    structure::Structure
    depends_on_t::Bool
    linear_in_u::Bool
    rhs::R
    jac::J
    points::Vector{NTuple{3, Int}}
    cache::BandCache
    layout::DofLayout            # the model's canonical layout
    blockable::Bool              # pointwise, with a block plan (`structure.block`)
    # The run's data (spec/operator-interface.md §8.4), when the operator reads
    # any, and each document's set of the epoch it last evaluated in.
    forcing::Union{Nothing, ForcingRuntime}
    rhs_fs::Base.RefValue{Any}
    jac_fs::Base.RefValue{Any}
end

forcing_boundary(op::PlannedOperator, t) = op.forcing !== nothing && is_epoch_start(op.forcing, t)

"""The forcing set of `op`'s right-hand-side (`jac = false`) or Jacobian program at `t` (§8.4.1), or `nothing` without data."""
op_forcing(op::PlannedOperator, t, jac::Bool) = op.forcing === nothing ? nothing :
    forcing_set_for(op.forcing, (jac ? op.jac : op.rhs).prog, t, jac ? op.jac_fs : op.rhs_fs)

opname(op::PlannedOperator) = op.name
structure(op::PlannedOperator) = op.structure
depends_on_t(op::PlannedOperator) = op.depends_on_t
linear_in_u(op::PlannedOperator) = op.linear_in_u
writes(op::PlannedOperator, N) = op.writes
# The columns of the band points that land on a DOF: the DOFs the right-hand
# side depends on (spec/operator-interface.md §8.2). Ghost reads are left out.
# They are as complete as the bands, which the operator's jac, jvp_u and vjp_u
# also rest on: a dependence the planner's bands miss would make vjp_u wrong
# for a leaf over every state too, so the bands' completeness is the planner's
# to guarantee (esm-jacobian-spec), not something this check can add.
reads(op::PlannedOperator, N) = sort!(unique!(Int[c for (_, _, c) in op.points]))
# §8.3: a Rosenbrock leaf on a planned operator that depends on t is refused,
# whatever the method's dᵢ.
ft_refused_on_rosenbrock(::PlannedOperator) = true

function rhs!(op::PlannedOperator, out, u, p, t)
    d = op.rhs
    return evaluate(d, u, p, t, op_forcing(op, t, false)) do du
        @inbounds for i in op.writes
            out[i] += du[d.perm[i]]
        end
        out
    end
end

"""
    with_band_values(f, op, u, p, t)

`f(vals)`, with `vals[k]` the value of the Jacobian document at band point
`op.points[k]` at `(u, p, t)`. The document is evaluated once per point
`(u, p, t)`, compared bitwise; the calls at the point it last saw reuse it.
"""
function with_band_values(f, op::PlannedOperator, u, p, t)
    _check_f64(u, p)
    c, tt = op.cache, Float64(t)
    hit, res = lock(c.lock) do
        c.valid && c.t === tt && bitwise_same(c.u, u) && bitwise_same(c.p, p) ? (true, f(c.vals)) : (false, nothing)
    end
    hit && return res
    vals = evaluate(op.jac, u, p, tt, op_forcing(op, tt, true)) do du
        Float64[du[s] for (s, _, _) in op.points]
    end
    lock(c.lock) do
        resize!(c.u, length(u)) .= u
        resize!(c.p, length(p)) .= p
        c.t, c.vals, c.valid = tt, vals, true
    end
    return f(vals)
end

function jac!(op::PlannedOperator, out, u, p, t)
    return with_band_values(op, u, p, t) do vals
        for (k, (_, r, c)) in enumerate(op.points)
            addentry!(out, r, c, vals[k])
        end
        out
    end
end

function jvp_u!(op::PlannedOperator, out, u, p, t, v)
    return with_band_values(op, u, p, t) do vals
        @inbounds for (k, (_, r, c)) in enumerate(op.points)
            out[r] += vals[k] * v[c]
        end
        out
    end
end

function vjp_u!(op::PlannedOperator, out, u, p, t, w)
    return with_band_values(op, u, p, t) do vals
        @inbounds for (k, (_, r, c)) in enumerate(op.points)
            out[c] += vals[k] * w[r]
        end
        out
    end
end

# ---------------------------------------------------------------------------
# A leaf's view of its planned operators (spec/operator-interface.md §8.2)

"""
    LeafOperator

A [`PlannedOperator`](@ref) bound to a leaf's state ([`LeafState`](@ref)):
its calls take and return leaf-sized vectors. `writes` are leaf indices,
`points` the operator's band points whose column lies in the leaf, as `(k,
row, col)` with `k` the point's index in the operator's band points and the
leaf's row and column (an entry in a column outside the leaf's DOFs is
dropped, §8.3), and `structure` the leaf's storage (§8.3): its block size is
the number of the leaf's state variables. `background` is the model state of
the invocation it is bound to ([`with_background`](@ref)). `pool` holds the
model-sized buffers of concurrent calls, shared by every invocation's binding.
"""
struct LeafOperator{O <: PlannedOperator} <: AbstractOperator
    op::O
    leaf::LeafState
    writes::Vector{Int}
    points::Vector{NTuple{3, Int}}
    structure::Structure
    background::Vector{Float64}
    pool::Vector{Vector{Float64}}
    lock::ReentrantLock
end

opname(o::LeafOperator) = o.op.name
structure(o::LeafOperator) = o.structure
depends_on_t(o::LeafOperator) = o.op.depends_on_t
linear_in_u(o::LeafOperator) = o.op.linear_in_u
writes(o::LeafOperator, N) = o.writes
# Model-state DOFs, as the adjoint's check compares them with the leaf's (§1).
reads(o::LeafOperator, N) = reads(o.op, N)
ft_refused_on_rosenbrock(::LeafOperator) = true

"""
`op` bound to a leaf invocation that starts from the model state `u`
([`with_background`](@ref)), which it holds as its background (not a copy:
the leaf's `with_background` copies `u` once for all of its operators). The
buffer pool is kept.
"""
function with_background(o::LeafOperator, u)
    eltype(u) === Float64 ||
        throw(UnsupportedError("operator $(o.op.name): replaying a leaf of planned operators with dual numbers is not implemented"))
    return LeafOperator(o.op, o.leaf, o.writes, o.points, o.structure, u, o.pool, o.lock)
end

"""`f(full)`, with `full` the model state: the leaf's `x` at its DOFs, the background elsewhere."""
function with_full_state(f, o::LeafOperator, x)
    _check_f64(x, o.background)
    full = lock(() -> isempty(o.pool) ? nothing : pop!(o.pool), o.lock)
    full === nothing && (full = similar(o.background))
    try
        place!(copyto!(full, o.background), o.leaf.dofs, x)
        return f(full)
    finally
        lock(() -> push!(o.pool, full), o.lock)
    end
end

function rhs!(o::LeafOperator, out, x, p, t)
    d = o.op.rhs
    return with_full_state(o, x) do u
        evaluate(d, u, p, t, op_forcing(o.op, t, false)) do du
            @inbounds for (i, c) in zip(o.writes, o.op.writes)
                out[i] += du[d.perm[c]]
            end
            out
        end
    end
end

function jac!(o::LeafOperator, out, x, p, t)
    return with_full_state(o, x) do u
        with_band_values(o.op, u, p, t) do vals
            for (k, r, c) in o.points
                addentry!(out, r, c, vals[k])
            end
            out
        end
    end
end

function jvp_u!(o::LeafOperator, out, x, p, t, v)
    return with_full_state(o, x) do u
        with_band_values(o.op, u, p, t) do vals
            @inbounds for (k, r, c) in o.points
                out[r] += vals[k] * v[c]
            end
            out
        end
    end
end

function vjp_u!(o::LeafOperator, out, x, p, t, w)
    return with_full_state(o, x) do u
        with_band_values(o.op, u, p, t) do vals
            @inbounds for (k, r, c) in o.points
                out[c] += vals[k] * w[r]
            end
            out
        end
    end
end

forcing_boundary(o::LeafOperator, t) = forcing_boundary(o.op, t)

binds_to_leaf(::PlannedOperator) = true

"""
    bind_planned_leaf(ops, prob, id) -> (ops′, LeafState)

The planned operators `ops` of leaf `id` bound to the leaf's state, the
union of their `writes` in canonical order (spec/operator-interface.md §8.2).
The leaf's storage is `block_diagonal` when every operator is pointwise with
`structure.block` and the leaf's state variables are all arrays of one shape
or all scalars: `ncells` that shape's cells (1 for scalars), `n` the number
of the leaf's state variables; otherwise `dense` (§8.3).
"""
function bind_planned_leaf(ops, prob, id)
    all(o -> o isa PlannedOperator, ops) ||
        throw(ConfigError("leaf $id mixes operators a plan supplies with others"))
    leaf = leaf_state(ops, prob.N, id)
    loc = leaf.loc
    L = first(ops).layout
    vars = [x for x in sort!(collect(keys(L.offset))) if loc[L.offset[x] + 1] > 0]
    st = all(o -> o.blockable, ops) ? block_structure(L, vars) : DenseStructure()
    bg = copy(prob.u0)
    bound = map(ops) do o
        pts = NTuple{3, Int}[(k, loc[r], loc[c]) for (k, (_, r, c)) in enumerate(o.points) if loc[c] > 0]
        LeafOperator(o, leaf, loc[o.writes], pts, st, bg, Vector{Float64}[], ReentrantLock())
    end
    return AbstractOperator[bound...], leaf
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
    forcing::Union{Nothing, ForcingRuntime}
end

"""
    plan_problem(dir; cache = nothing) -> PlanProblem

Build every operator the resolved schedule of the plan in `dir` names, from
its entry `operators.<name>` and its documents `derived/<name>.rhs.esm` and
`derived/<name>.jacobian.esm` (spec/operator-interface.md §8). A derived
document whose bytes do not hash to the plan's `derived` entry is refused:
the plan is stale. `cache`, a dictionary, keeps built documents by their hash
across plans, so a document many plans share is built once.

A plan with `forcing` reads data (§8.4): its model, at `model` (default: its
`inputs.model.path` relative to `dir`) and checked against the plan's hash,
gives the providers, through EarthSciAST's `providers_from_document` with
EarthSciIO's cache under `cache_root` (default `dir/data-cache`) and the
source-name-to-URL-template map `url_overrides`. These three are run-time
options, in neither the run config nor the plan.
"""
function plan_problem(
        dir::AbstractString; cache::Union{Nothing, AbstractDict} = nothing, model::Union{Nothing, AbstractString} = nothing,
        cache_root::Union{Nothing, AbstractString} = nothing, url_overrides::AbstractDict = Dict{String, String}()
    )
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
        # A derived document is self-contained (spec/plan.md §6.1: no data
        # sources, function tables or other files), so its bytes and the
        # interval determine its build; a document another plan built is
        # reused, under this plan's path.
        d = get!(() -> build_plan_document(path, op, tspan), cache, (h, tspan))
        return _with(d; path)
    end
    built = Dict{String, Tuple{PlanDocument, PlanDocument}}(n => (doc(n, "rhs"), doc(n, "jacobian")) for n in names)
    # The layout comes from the first operator's right-hand side; every other
    # document must declare the same states (§8.2). The parameters are the
    # model's: every scalar parameter any of the documents reads, at its
    # default, in code-point order of their names (§8.2).
    first_rhs = built[first(names)][1]
    L = canonical_layout(first_rhs)
    defaults = Dict{String, Float64}()
    for n in names, d in built[n], s in EarthSciAST.parameters(d.prog)
        s.default === nothing &&
            throw(ConfigError("operator '$n': parameter $(s.name) of $(basename(d.path)) has no default (spec/operator-interface.md §8.2)"))
        # Defaults are compared as the Float64 the parameter vector holds, so
        # an integer default (1) agrees with itself and with 1.0.
        dv = Float64(s.default)
        v = get!(defaults, s.name, dv)
        v === dv ||
            throw(ConfigError("operator '$n': parameter $(s.name) has default $(s.default) in $(basename(d.path)) and $v in another derived document"))
    end
    pnames = sort!(collect(keys(defaults)))
    p = Float64[defaults[x] for x in pnames]
    # The data (spec/operator-interface.md §8.4): the plan's model, located by
    # its `inputs` entry and checked against its hash, gives the providers.
    F, mfile = nothing, nothing
    if haskey(plan, "forcing")
        mpath = model === nothing ? joinpath(dir, String(plan["inputs"]["model"]["path"])) : String(model)
        isfile(mpath) || throw(ConfigError("the plan's model $(plan["inputs"]["model"]["path"]) is not at $mpath; pass its path as `model`"))
        bytes2hex(sha256(read(mpath))) == plan["inputs"]["model"]["sha256"] ||
            throw(ConfigError("the model at $mpath does not match the plan's hash (the plan is stale)"))
        mfile = EarthSciAST.load_path(mpath)
        readers = sort!(unique!(String[x for n in names for x in get(plan["operators"][n], "forcing", Any[])]))
        root = cache_root === nothing ? joinpath(dir, "data-cache") : String(cache_root)
        F = forcing_runtime(mfile, plan, tspan, readers; cache_root = root, url_overrides)
        # Every program's forcing slots must be data-fed parameters of the
        # plan's `forcing`, of their planned shapes (§8.4.1).
        for n in names, d in built[n], s in EarthSciAST.forcing_slots(d.prog)
            haskey(F.shapes, s.name) || throw(
                ConfigError("operator '$n': $(basename(d.path)) has a forcing slot $(s.name), which no operator's forcing in the plan lists")
            )
            prod(s.shape; init = 1) == prod(F.shapes[s.name]; init = 1) || throw(
                ConfigError("operator '$n': $(basename(d.path))'s forcing slot $(s.name) has shape $(s.shape), the plan's $(F.shapes[s.name])")
            )
        end
    end
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
        op = planned_operator(n, plan["operators"][n], L, r, j)
        haskey(plan["operators"][n], "forcing") && (op = _with_forcing(op, F))
        push!(ops, n => op)
    end
    # §8.4.5: when an `ic` equation of the plan's model reads a data-fed
    # parameter, the run starts from EarthSciAST's initial state of the model
    # with its providers, not from the derived documents' (bound to defaults).
    if mfile !== nothing && ic_reads(EarthSciAST.flatten(mfile), Set(keys(plan["forcing"])))
        u0 = data_initial_state(mfile, F, tspan, L)
    end
    variables = Pair{String, Vector{Int}}[
        x => collect((L.offset[x] + 1):(L.offset[x] + prod(L.extents[x]; init = 1))) for x in sort!(collect(keys(L.offset)))
    ]
    name = String(get(get(plan["inputs"], "model", Dict()), "path", "plan"))
    prob = ToyProblem(
        name, length(L.names), u0, p, pnames, tspan, ops, Dict{String, AbstractMapOperator}(), variables,
        Function[], nothing, false
    )
    return PlanProblem(prob, plan, L, F)
end

_with_forcing(op::PlannedOperator, F) = PlannedOperator(
    op.name, op.writes, op.structure, op.depends_on_t, op.linear_in_u, op.rhs, op.jac, op.points, op.cache,
    op.layout, op.blockable, F, Ref{Any}(nothing), Ref{Any}(nothing)
)

"""
    data_problem(file, F, tspan) -> EsmProblem

EarthSciAST's problem of the plan's model with the run's providers (the
plan's `forcing` only), the one whose own solve a planned run reproduces
(spec/operator-interface.md §8.4.2 step 3, §8.4.5).
"""
data_problem(file::EarthSciAST.EsmFile, F::ForcingRuntime, tspan) =
    EarthSciAST.esm_problem(file, tspan; compiler = :native, providers = F.providers)

"""The initial state of [`data_problem`](@ref), in the canonical order of `L` (§8.4.5)."""
function data_initial_state(file, F, tspan, L::DofLayout)
    prob = try
        data_problem(file, F, tspan)
    catch e
        e isa InterruptException && rethrow()
        throw(ConfigError("building the plan's model with its data, for the initial state, failed: " * sprint(showerror, e)))
    end
    vm = Dict{String, Int}(String(k) => Int(v) for (k, v) in prob.var_map)
    return Float64[
        prob.u0[
            get(vm, n) do
                throw(ConfigError("the plan's model built with its data has no state element $n"))
            end,
        ] for n in L.names
    ]
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
            get(j.slots, element_name("JacobianEval.J_$k", c)) do
                throw(ConfigError("operator '$name': derived/$name.jacobian.esm has no band field element $(element_name("JacobianEval.J_$k", c))"))
            end for c in cells(ext)
        ]
    end
    points = NTuple{3, Int}[(fields[k][f], row, col) for (k, f, row, col) in band_points(L, bands)]
    return PlannedOperator(
        String(name), w, planned_structure(entry, L), Bool(st["depends_on_t"]), Bool(st["linear_in_u"]), r, j, points,
        BandCache(), L, entry["locality"]["class"] == "pointwise" && haskey(st, "block"), nothing, Ref{Any}(nothing), Ref{Any}(nothing)
    )
end

"""
The structure of a planned operator: the structure a leaf of this operator
alone has (spec/operator-interface.md §8.3). It is `block_diagonal` when the
operator is pointwise with `structure.block` and the state variables it
writes are all arrays of one shape (`ncells` its cells) or all scalars
(`ncells` = 1), with `n` the number of those variables; otherwise `dense`.
The model's other states do not matter. A leaf of several planned operators
takes its structure from all of its state variables ([`bind_planned_leaf`](@ref)).
"""
function planned_structure(entry, L::DofLayout)
    st = entry["structure"]
    entry["locality"]["class"] == "pointwise" && haskey(st, "block") || return DenseStructure()
    return block_structure(L, unique(String.(entry["writes"])))
end

"""
The storage of a pointwise state over the state variables `vars` of layout
`L` (spec/operator-interface.md §8.3): `block_diagonal` when they are all
arrays of one shape (`ncells` its cells) or all scalars (`ncells` = 1), with
`n` the number of variables; otherwise (or with none) `dense`.
"""
function block_structure(L::DofLayout, vars)
    isempty(vars) && return DenseStructure()
    exts = [L.extents[x] for x in vars]
    all(==(first(exts)), exts) || return DenseStructure()
    return BlockDiagonalStructure(prod(first(exts); init = 1), length(vars))
end

"""
    build_plan_run(dir; saveat = nothing, tspan = nothing, cache = nothing) -> (Run, PlanProblem)

A run of the plan in `dir` (spec/operator-interface.md §8, spec/toy-problems.md
§4): its operators built by [`plan_problem`](@ref), its resolved schedule and
step, over the plan's `time`. A caller's `tspan` other than the plan's is
refused (the plan is stale, spec/plan.md §3). Run it with [`forward!`](@ref).
"""
function build_plan_run(
        dir::AbstractString; saveat = nothing, tspan = nothing, cache = nothing, model = nothing,
        cache_root = nothing, url_overrides = Dict{String, String}()
    )
    pp = plan_problem(dir; cache, model, cache_root, url_overrides)
    prob, plan = pp.prob, pp.plan
    if tspan !== nothing && (Float64(tspan[1]), Float64(tspan[2])) != prob.tspan
        throw(ConfigError("run interval $(tspan) differs from the plan's time $(prob.tspan): the plan is stale (spec/plan.md §3)"))
    end
    config = Dict{String, Any}(
        "esm_integrator" => plan["esm_integrator"], "schedule" => plan["schedule"], "step" => plan["step"]
    )
    # The forcing epochs' starts inside the run are tstops (§8.4.3), when the
    # plan's step.hit names forcing_refresh (spec/plan.md §7.3 requires it
    # whenever an operator reads a discrete data-fed parameter).
    stops = Float64[]
    if pp.forcing !== nothing && "forcing_refresh" in get(plan["step"], "hit", Any[])
        stops = Float64[B for B in pp.forcing.epochs if prob.tspan[1] < B < prob.tspan[2]]
    end
    return build_run(config; tspan = prob.tspan, saveat, prob, tstops = stops), pp
end

"""
    layout_dofs(pp::PlanProblem, layout) -> Vector{Int}

The canonical DOFs of the variables `layout` lists, each variable's cells
column-major, in the listed order (spec/toy-problems.md §4.3).
"""
function layout_dofs(pp::PlanProblem, layout)
    return reduce(vcat, (variable_dofs(pp.prob, String(x)) for x in layout); init = Int[])
end
