# Executing a plan (spec/operator-interface.md §8, spec/toy-problems.md §4):
# the operators the schedule names, built from the plan's entries and their
# derived documents, and a run of the plan's resolved schedule on them.
#
# A `PlannedOperator` acts on the whole model state in canonical order (§8.2,
# spec/plan.md §4): state names sorted by code point, each array's cells
# column-major. A leaf advances only its own state, the union of its
# operators' `writes` in canonical order (§8.2): its operators are bound to
# it as `LeafOperator`s, which fill every other state from the model state
# at the start of the invocation, so those are constants of the leaf.

"""A workspace of a [`PlanDocument`](@ref): EarthSciAST's, and the slot, output and parameter buffers of one call."""
struct DocWorkspace{W}
    ws::W
    U::Vector{Float64}
    du::Vector{Float64}
    P::Vector{Float64}
end

"""
    PlanDocument

One derived document built as spec/operator-interface.md §8.1 says:
`rhs_program(esm_problem(path, tspan))` with the native compiler. `perm`
maps each canonical DOF to its slot in the program's `u`, `pidx` each of the
program's parameters to its index in the model's parameter vector (§8.2),
and `nslots` is the length of the program's `u`. `slots` maps each state
element name to its slot, from the program's state layout, which the build
checks the problem's `var_map` and `u0` agree with. Workspaces are kept in a
pool, one per concurrent caller (EarthSciAST's `rhs_workspace`), so one
document serves every thread.
"""
struct PlanDocument{P, W}
    path::String
    prog::P
    fs::EarthSciAST.ForcingSet
    aux::Vector{Float64}
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
    PlanDocument(String(path), d.prog, d.fs, d.aux, d.nslots, perm, pidx, d.slots, d.u0, d.pool, d.lock)

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
    prob, prog = try
        prob = EarthSciAST.esm_problem(path, tspan; compiler = :native, base_path = dirname(path))
        prob, EarthSciAST.rhs_program(prob)
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
    ws = EarthSciAST.rhs_workspace(prog)
    npar = length(EarthSciAST.parameters(prog))
    w = DocWorkspace(ws, zeros(nslots), zeros(nslots), zeros(npar))
    return PlanDocument(
        String(path), prog, EarthSciAST.default_forcing_set(prog), zeros(EarthSciAST.tile_layout(prog).n_aux),
        nslots, Int[], Int[], slots, Float64.(prob.u0), [w], ReentrantLock()
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
    return DocWorkspace{W}(EarthSciAST.rhs_workspace(d.prog), zeros(d.nslots), zeros(d.nslots), zeros(length(d.pidx)))
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
fields included, zero) and the model parameters `p`, then call `f(du)` on
the program's output in its own slot order, and return what `f` returns.
"""
function evaluate(f, d::PlanDocument, u, p, t)
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
        EarthSciAST.rhs!(w.du, d.prog, w.ws, U, d.aux, P, Float64(t), d.fs)
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
end

opname(op::PlannedOperator) = op.name
structure(op::PlannedOperator) = op.structure
depends_on_t(op::PlannedOperator) = op.depends_on_t
linear_in_u(op::PlannedOperator) = op.linear_in_u
writes(op::PlannedOperator, N) = op.writes
# §8.3: a Rosenbrock leaf on a planned operator that depends on t is refused,
# whatever the method's dᵢ.
ft_refused_on_rosenbrock(::PlannedOperator) = true

function rhs!(op::PlannedOperator, out, u, p, t)
    d = op.rhs
    return evaluate(d, u, p, t) do du
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
    vals = evaluate(op.jac, u, p, tt) do du
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
the number of the leaf's state variables.
"""
struct LeafOperator{O <: PlannedOperator} <: AbstractOperator
    op::O
    leaf::LeafState
    writes::Vector{Int}
    points::Vector{NTuple{3, Int}}
    structure::Structure
    pool::Vector{Vector{Float64}}
    lock::ReentrantLock
end

opname(o::LeafOperator) = o.op.name
structure(o::LeafOperator) = o.structure
depends_on_t(o::LeafOperator) = o.op.depends_on_t
linear_in_u(o::LeafOperator) = o.op.linear_in_u
writes(o::LeafOperator, N) = o.writes
ft_refused_on_rosenbrock(::LeafOperator) = true

"""`f(full)`, with `full` the model state: the leaf's `x` at its DOFs, the background elsewhere."""
function with_full_state(f, o::LeafOperator, x)
    full = lock(() -> isempty(o.pool) ? nothing : pop!(o.pool), o.lock)
    full === nothing && (full = similar(o.leaf.background))
    try
        _check_f64(x, o.leaf.background)
        copyto!(full, o.leaf.background)
        @inbounds for (i, d) in enumerate(o.leaf.dofs)
            full[d] = x[i]
        end
        return f(full)
    finally
        lock(() -> push!(o.pool, full), o.lock)
    end
end

function rhs!(o::LeafOperator, out, x, p, t)
    d = o.op.rhs
    return with_full_state(o, x) do u
        evaluate(d, u, p, t) do du
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
    N = prob.N
    dofs = sort!(unique!(reduce(vcat, (o.writes for o in ops); init = Int[])))
    isempty(dofs) && throw(ConfigError("leaf $id: its operators write nothing"))
    loc = zeros(Int, N)
    loc[dofs] = 1:length(dofs)
    leaf = LeafState(dofs, loc, copy(prob.u0))
    L = first(ops).layout
    vars = [x for x in sort!(collect(keys(L.offset))) if loc[L.offset[x] + 1] > 0]
    exts = [L.extents[x] for x in vars]
    st = all(o -> o.blockable, ops) && all(==(first(exts)), exts) ?
        BlockDiagonalStructure(prod(first(exts); init = 1), length(vars)) : DenseStructure()
    bound = map(ops) do o
        pts = NTuple{3, Int}[(k, loc[r], loc[c]) for (k, (_, r, c)) in enumerate(o.points) if loc[c] > 0]
        LeafOperator(o, leaf, loc[o.writes], pts, st, Vector{Float64}[], ReentrantLock())
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
        v = get!(defaults, s.name, s.default)
        v === s.default ||
            throw(ConfigError("operator '$n': parameter $(s.name) has default $(s.default) in $(basename(d.path)) and $v in another derived document"))
    end
    pnames = sort!(collect(keys(defaults)))
    p = Float64[defaults[x] for x in pnames]
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
            get(j.slots, element_name("JacobianEval.J_$k", c)) do
                throw(ConfigError("operator '$name': derived/$name.jacobian.esm has no band field element $(element_name("JacobianEval.J_$k", c))"))
            end for c in cells(ext)
        ]
    end
    points = NTuple{3, Int}[(fields[k][f], row, col) for (k, f, row, col) in band_points(L, bands)]
    return PlannedOperator(
        String(name), w, planned_structure(entry, L), Bool(st["depends_on_t"]), Bool(st["linear_in_u"]), r, j, points,
        BandCache(), L, entry["locality"]["class"] == "pointwise" && haskey(st, "block")
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
