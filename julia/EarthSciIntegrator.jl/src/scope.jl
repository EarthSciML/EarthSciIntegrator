# `dt_scope: cell` and `shard` (spec/partitioning.md §2.3, §2.4).
#
# A scoped leaf integrates **units**, each an independent initial-value
# problem with its own controller: one cell under `cell`, one shard (a set of
# cells) under `shard`. A unit is integrated by an ordinary leaf
# ([`leaf_forward`](@ref)) over the unit's own state vector (species-major
# over its cells), whose operators ([`UnitOperator`](@ref)) evaluate the
# leaf's operators at the unit's cells only, with the rest of the model state
# the leaf's constants. So every method, controller, stop rule and domain
# check is the one a whole leaf uses (spec/methods.md), applied per unit.
#
# Executors (§4) are tasks on Julia's threads: each takes units from a shared
# counter, in unit order, and integrates each with a unit leaf it owns; units
# share nothing while they run, so the thread count, the order in which units
# are taken and the lane capacity change no result (§9.1). Under `cell` the
# shards only group cells into work; under `shard` they are the units, and
# re-sharding (§6.2) changes them at macro-step boundaries from each cell's
# share of its shard's squared error ratios.
#
# Lane pools. A unit's cells are evaluated together through the operators'
# lane programs (EarthSciAST's `lanes`, U3), `capacity` lanes per call; a
# remainder is padded by the program (§3.1 rule 2). Cells of different units
# are never batched together: lane programs take one time per call (D98), and
# cells of different units are at different times after their first step.
# When lane programs take a time per lane (U19), one pool may batch the cells
# of many units without changing a value.

# ---------------------------------------------------------------------------
# Lane programs (spec/partitioning.md §3)

"""The lane programs built so far, by derived document (its path and bytes' hash) and lane capacity."""
const _LANE_PROGRAMS = Dict{Tuple{String, String, Int}, Any}()
const _LANE_LOCK = ReentrantLock()

"""
    lane_program(d::PlanDocument, sets, capacity, tspan) -> LaneProgram

EarthSciAST's lane program (`lanes(op, :cells, sets; chunk = capacity)`) of
the derived document `d`, compiled natively.
Built once per document and capacity, and shared by every thread and every
run that uses that document (it is reentrant across workspaces).
"""
function lane_program(d::PlanDocument, sets::Vector{String}, capacity::Int, tspan)
    bytes = read(d.path)
    key = (d.path, bytes2hex(sha256(bytes)), capacity)
    return lock(_LANE_LOCK) do
        get!(_LANE_PROGRAMS, key) do
            doc = JSON.parse(String(bytes))
            file = EarthSciAST.load_document(wire_form(doc))
            prob = EarthSciAST.esm_problem(file, tspan; compiler = :native, base_path = dirname(d.path))
            EarthSciAST.lanes(EarthSciAST.rhs_program(prob), :cells, sets; chunk = capacity)
        end
    end
end

"""A workspace of a [`LaneDoc`](@ref): EarthSciAST's, and the lane buffers of one call."""
struct LaneWork{W}
    ws::W
    u::Vector{Float64}
    aux::Vector{Float64}
    du::Vector{Float64}
    P::Vector{Float64}
    cells::Matrix{Int32}
end

"""
    LaneDoc

One derived document of a planned operator over the cells of one leaf: its
lane program `lp` of capacity `L`, the model parameter each of the program's
parameters takes (`pidx`), and where each lane block of `u` takes its value
at a cell (`fill`, `(block offset, species, start)`: the unit's state of leaf
species `species` when it is positive, else the model state at canonical DOF
`start + cell - 1`, a constant of the leaf). A block with no entry (a band
field) is zero, as in the whole-domain evaluation. Workspaces are pooled.
"""
struct LaneDoc{P}
    lp::P
    L::Int
    pidx::Vector{Int}
    fill::Vector{NTuple{3, Int}}
    nu::Int
    na::Int
    offsets::Dict{String, Int}
    pool::Vector{Any}
    sets::Dict{NTuple{2, Int}, EarthSciAST.ForcingSet}
    lock::ReentrantLock
end

function LaneDoc(lp, L::Int, pnames::Vector{String}, species::Vector{String}, layout::DofLayout, ncells::Int, what)
    lay = EarthSciAST.lane_layout(lp)
    isempty(lay.aux) || throw(ConfigError("$what: its lane program has auxiliary blocks $(join((b.name for b in lay.aux), ", ")), which a leaf's lane pool does not fill"))
    pindex = Dict(n => i for (i, n) in enumerate(pnames))
    pidx = Int[
        get(pindex, s.name) do
            throw(ConfigError("$what: its lane program has parameter $(s.name), which the model does not declare"))
        end for s in EarthSciAST.parameters(lp)
    ]
    fill = NTuple{3, Int}[]
    offsets = Dict{String, Int}()
    for b in lay.u
        isempty(b.sets) || throw(ConfigError("$what: its lane block $(b.name) has index sets $(b.sets) beside the cell's; only per-cell blocks are supported (spec/partitioning.md §2.3)"))
        offsets[b.name] = b.offset
        i = findfirst(==(b.name), species)
        if i !== nothing
            push!(fill, (b.offset, i, 0))
        elseif haskey(layout.offset, b.name)
            prod(layout.extents[b.name]; init = 1) == ncells ||
                throw(ConfigError("$what: it reads state $(b.name), which is not over the leaf's cells"))
            push!(fill, (b.offset, 0, layout.offset[b.name] + 1))
        end
    end
    return LaneDoc(lp, L, pidx, fill, lay.n_u, lay.n_aux, offsets, Any[], Dict{NTuple{2, Int}, EarthSciAST.ForcingSet}(), ReentrantLock())
end

function _acquire(d::LaneDoc)
    w = lock(() -> isempty(d.pool) ? nothing : pop!(d.pool), d.lock)
    w === nothing || return w
    return LaneWork(
        EarthSciAST.rhs_workspace(d.lp), zeros(d.nu * d.L), zeros(d.na * d.L), zeros(d.nu * d.L),
        zeros(length(d.pidx)), ones(Int32, length(EarthSciAST.lane_layout(d.lp).sets), d.L)
    )
end
_release(d::LaneDoc, w) = lock(() -> push!(d.pool, w), d.lock)

"""
The lane program's forcing set for a call at `t` (spec/operator-interface.md
§8.4.1; spec/partitioning.md §7): the buffers of the epochs
[`forcing_key`](@ref) selects, which are resident before the invocation;
one set per key, built once.
"""
function lane_forcing(d::LaneDoc, F::Union{Nothing, ForcingRuntime}, t)
    slots = EarthSciAST.forcing_slots(d.lp)
    if F === nothing
        isempty(slots) || throw(ConfigError("a lane program reads data, but the operator has no forcing"))
        return EarthSciAST.forcing_set(d.lp, Any[])
    end
    key = forcing_key(F, t)
    return lock(d.lock) do
        get!(d.sets, key) do
            ks, kt = key
            bt = epoch_buffers(F, kt)
            bs = ks == kt ? bt : epoch_buffers(F, ks)
            EarthSciAST.forcing_set(d.lp, [F.records[s.name] == 2 ? bt[s.name] : bs[s.name] for s in slots])
        end
    end
end

"""
    lane_eval(g, d, F, cells, at, nb, m, x, bg, p, t, coords)

Evaluate document `d` at the unit's cells `cells[at + 1 : at + nb]` (`nb ≤
d.L`), `x` the unit's state (`m` cells, species-major), `bg` the model state
and `coords[:, c]` cell `c`'s index on each lane set; then `g(du, L)` on the
lane-layout output. Lanes past `nb` are padding (rule 2).
"""
function lane_eval(g, d::LaneDoc, F, cells, at, nb, m, x, bg, p, t, coords)
    w = _acquire(d)
    try
        L = d.L
        U = w.u
        fill!(U, 0.0)
        @inbounds for (off, i, start) in d.fill
            for l in 1:nb
                k = at + l
                U[off * L + l] = i > 0 ? Float64(x[(i - 1) * m + k]) : bg[start + cells[k] - 1]
            end
        end
        @inbounds for l in 1:nb, r in axes(coords, 1)
            w.cells[r, l] = coords[r, cells[at + l]]
        end
        @inbounds for j in eachindex(d.pidx)
            w.P[j] = p[d.pidx[j]]
        end
        fs = lane_forcing(d, F, t)
        EarthSciAST.rhs!(w.du, d.lp, w.ws, U, w.aux, w.P, Float64(t), fs, w.cells, nb)
        return g(w.du, L)
    finally
        _release(d, w)
    end
end

"""
    LaneOps

A plan's pointwise operator over lists of cells of one leaf (spec/partitioning.md
§3): the lane programs of its right-hand-side, Jacobian and (when the plan
gives one) time-derivative documents, and how their outputs map to a unit's
rows: `rhs_out` and `time_out` as `(block offset, leaf species)`, `jac_out`
per canonical cell the entries `(block offset of J_k, row species, column
species)` in the whole-domain operator's point order (points outside the
leaf dropped, as the leaf's `jac!` drops them).
"""
struct LaneOps{R, J}
    op::PlannedOperator
    rhs::R
    jac::J
    time::Union{Nothing, LaneDoc}
    rhs_out::Vector{NTuple{2, Int}}
    jac_out::Vector{Vector{NTuple{3, Int}}}
    time_out::Vector{NTuple{2, Int}}
    coords::Matrix{Int32}
end

"""
    lane_ops(op, species, capacity, pnames) -> LaneOps

The lane programs of planned operator `op` for a leaf whose state variables
are `species` (canonical order, each over the same cells), at lane capacity
`capacity`.
"""
function lane_ops(op::PlannedOperator, species::Vector{String}, capacity::Int, pnames::Vector{String}, tspan)
    L = op.layout
    ext = L.extents[species[1]]
    ncells = prod(ext; init = 1)
    flat = EarthSciAST.flatten(EarthSciAST.load_path(op.rhs.path))
    sets = flat.state_variables[species[1]].shape
    (sets === nothing || length(sets) != length(ext)) &&
        throw(ConfigError("operator '$(op.name)': its state $(species[1]) has no index sets for its extents $ext"))
    sets = String[String(s) for s in sets]
    what(kind) = "operator '$(op.name)': derived/$(op.name).$kind.esm"
    build(d, kind) = try
        LaneDoc(lane_program(d, sets, capacity, tspan), capacity, pnames, species, L, ncells, what(kind))
    catch e
        e isa InterruptException && rethrow()
        e isa ConfigError && rethrow()
        throw(ConfigError("$(what(kind)) over lanes: " * sprint(showerror, e)))
    end
    rhs = build(op.rhs, "rhs")
    jac = build(op.jac, "jacobian")
    time = op.time === nothing ? nothing : build(op.time, "time_derivative")
    spec_of(dof) = begin
        for (i, s) in enumerate(species)
            o = L.offset[s]
            o < dof <= o + ncells && return (i, dof - o)
        end
        (0, 0)
    end
    rhs_out = NTuple{2, Int}[]
    for (i, s) in enumerate(species)
        any(w -> L.offset[s] < w <= L.offset[s] + ncells, op.writes) || continue
        haskey(rhs.offsets, s) || throw(ConfigError("$(what("rhs")): its lane program has no block for $s"))
        push!(rhs_out, (rhs.offsets[s], i))
    end
    jac_out = [NTuple{3, Int}[] for _ in 1:ncells]
    for (k, f, r, c) in op.bandpts
        (i, cr), (j, cc) = spec_of(r), spec_of(c)
        (i == 0 || j == 0) && continue
        cr == cc || throw(ConfigError("operator '$(op.name)': its band entry ($(L.names[r]), $(L.names[c])) joins two cells"))
        f == cr || throw(ConfigError("operator '$(op.name)': band field J_$k is not over the leaf's cells"))
        name = "JacobianEval.J_$k"
        haskey(jac.offsets, name) || throw(ConfigError("$(what("jacobian")): its lane program has no block for $name"))
        push!(jac_out[cr], (jac.offsets[name], i, j))
    end
    time_out = NTuple{2, Int}[]
    if time !== nothing
        for (k, row) in enumerate(op.time_rows)
            i = findfirst(==(row), species)
            # A time band of a state outside the leaf adds nothing to its f_t.
            i === nothing && continue
            name = "JacobianEval.J_$k"
            haskey(time.offsets, name) || throw(ConfigError("$(what("time_derivative")): its lane program has no block for $name"))
            push!(time_out, (time.offsets[name], i))
        end
    end
    coords = Matrix{Int32}(undef, length(ext), ncells)
    for (c, I) in enumerate(CartesianIndices(Tuple(ext)))
        for d in eachindex(ext)
            coords[d, c] = I[d]
        end
    end
    return LaneOps(op, rhs, jac, time, rhs_out, jac_out, time_out, coords)
end

function _lane_loop(f, d::LaneDoc, cells)
    m = length(cells)
    at = 0
    while at < m
        nb = min(d.L, m - at)
        f(at, nb)
        at += nb
    end
    return nothing
end

"""`out += f` at the unit's cells: `x` the unit's state (species-major over `cells`), `bg` the model state."""
function lane_rhs!(lo::LaneOps, out, cells, x, bg, p, t)
    m = length(cells)
    d = lo.rhs
    _lane_loop(d, cells) do at, nb
        lane_eval(d, lo.op.forcing, cells, at, nb, m, x, bg, p, t, lo.coords) do du, L
            @inbounds for (off, i) in lo.rhs_out, l in 1:nb
                out[(i - 1) * m + at + l] += du[off * L + l]
            end
        end
    end
    return out
end

"""`out += f_t` at the unit's cells (spec/plan.md §6.4)."""
function lane_ft!(lo::LaneOps, out, cells, x, bg, p, t)
    d = lo.time
    d === nothing && return out
    m = length(cells)
    _lane_loop(d, cells) do at, nb
        lane_eval(d, lo.op.forcing, cells, at, nb, m, x, bg, p, t, lo.coords) do du, L
            @inbounds for (off, i) in lo.time_out, l in 1:nb
                out[(i - 1) * m + at + l] += du[off * L + l]
            end
        end
    end
    return out
end

"""The operator's Jacobian blocks at the unit's cells, added into the unit's storage `out`."""
function lane_jac!(lo::LaneOps, out, cells, x, bg, p, t)
    m = length(cells)
    d = lo.jac
    _lane_loop(d, cells) do at, nb
        lane_eval(d, lo.op.forcing, cells, at, nb, m, x, bg, p, t, lo.coords) do du, L
            for l in 1:nb
                k = at + l
                for (off, i, j) in lo.jac_out[cells[k]]
                    addentry!(out, (i - 1) * m + k, (j - 1) * m + k, du[off * L + l])
                end
            end
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# Units

"""
    UnitTarget

The unit a unit leaf's operators currently evaluate: its cells (canonical,
1-based, ascending), the leaf's state at the invocation's start `xbg` and the
model state at its start `bg` (the leaf's constants).
"""
mutable struct UnitTarget
    cells::Vector{Int}
    xbg::Vector{Float64}
    bg::Vector{Float64}
end

"""
    UnitOperator

One of a scoped leaf's operators over a unit's state vector (species-major
over the unit's `m` cells). With `lanes`, it evaluates the operator's lane
programs; without, the operator `base` over the leaf's whole state with the
unit's cells set and reads the unit's rows: exact for a pointwise operator,
whose rows at a cell read that cell only. That second form is the lane
programs' oracle in the tests (spec/partitioning.md §9.3) and serves
operators that have no lane programs (hand-written ones).
"""
struct UnitOperator{O, LO} <: AbstractOperator
    base::O
    lanes::LO
    target::UnitTarget
    n::Int
    m::Int
    ncells::Int
end

opname(o::UnitOperator) = opname(o.base)
structure(o::UnitOperator) = BlockDiagonalStructure(o.m, o.n)
depends_on_t(o::UnitOperator) = depends_on_t(o.base)
linear_in_u(o::UnitOperator) = linear_in_u(o.base)
writes(o::UnitOperator, N) = collect(1:N)
forcing_boundary(o::UnitOperator, t) = forcing_boundary(o.base, t)
forcing_epochs(o::UnitOperator) = forcing_epochs(o.base)
ft_refused_on_rosenbrock(o::UnitOperator) = ft_refused_on_rosenbrock(o.base)
provides(o::UnitOperator, call::Symbol) = call in (:rhs, :jac) || (call === :ft && provides(o.base, :ft))

"""The leaf's state at the invocation's start with the unit's state `x` at the unit's cells."""
function _leaf_vector(o::UnitOperator, x)
    tg = o.target
    y = copy(tg.xbg)
    m, nc = o.m, o.ncells
    @inbounds for i in 1:o.n, (k, c) in enumerate(tg.cells)
        y[(i - 1) * nc + c] = x[(i - 1) * m + k]
    end
    return y
end

"""`out += g(leaf vector)` at the unit's rows, for a call `g(out, y)` over the leaf's whole state."""
function _whole!(g, o::UnitOperator, out, x)
    y = _leaf_vector(o, x)
    fo = zeros(length(y))
    g(fo, y)
    tg = o.target
    m, nc = o.m, o.ncells
    @inbounds for i in 1:o.n, (k, c) in enumerate(tg.cells)
        out[(i - 1) * m + k] += fo[(i - 1) * nc + c]
    end
    return out
end

function rhs!(o::UnitOperator, out, x, p, t)
    o.lanes === nothing || return lane_rhs!(o.lanes, out, o.target.cells, x, o.target.bg, p, t)
    return _whole!((fo, y) -> rhs!(o.base, fo, y, p, t), o, out, x)
end

function ft!(o::UnitOperator, out, x, p, t)
    depends_on_t(o) || return out
    o.lanes === nothing || return lane_ft!(o.lanes, out, o.target.cells, x, o.target.bg, p, t)
    return _whole!((fo, y) -> ft!(o.base, fo, y, p, t), o, out, x)
end

function jac!(o::UnitOperator, out, x, p, t)
    o.lanes === nothing || return lane_jac!(o.lanes, out, o.target.cells, x, o.target.bg, p, t)
    y = _leaf_vector(o, x)
    S = BlockJacobian{Float64}(o.ncells, o.n)
    jac!(o.base, S, y, p, t)
    m = o.m
    for i in 1:o.n, j in 1:o.n, (k, c) in enumerate(o.target.cells)
        v = _entry(S, (i - 1) * o.ncells + c, (j - 1) * o.ncells + c)
        # Only the entries the operator wrote: an entry it did not write is
        # never added, as the lane path adds only the band points.
        v === 0.0 && continue
        addentry!(out, (i - 1) * m + k, (j - 1) * m + k, v)
    end
    return out
end

# ---------------------------------------------------------------------------
# The scope of a leaf

"""
    ScopeState

What a leaf with `dt_scope` `cell` or `shard` keeps across its invocations
(spec/partitioning.md §2.3, §2.4, §6): the kind, its `ncells` cells and `n`
species, the plan's shard count, `reshard: {"every": k}` (under `shard`),
the lane capacity, the run-time thread count, whether the tests' whole-state
oracle replaces the lane programs; per unit (cell, or shard of the current
composition) its controller state; the re-sharding signal of the current and
the previous macro step; and the lane programs, built on first use.
"""
mutable struct ScopeState
    kind::Symbol
    ncells::Int
    n::Int
    nshards::Int
    reshard_every::Union{Nothing, Int}
    capacity::Int
    threads::Int
    whole_state::Bool
    units::Vector{ControllerState}
    shards::Vector{Vector{Int}}
    macro_step::Int
    signal::Vector{Float64}
    previous::Vector{Float64}
    lanes::Any
    lock::ReentrantLock
end

"""The default lane capacity when the plan gives none (spec/partitioning.md §10.1: the runtime chooses; it changes no result)."""
const DEFAULT_LANES = 8

ScopeState(kind::Symbol, ncells, n, nshards, reshard_every, capacity) = ScopeState(
    kind, ncells, n, max(nshards, 1), reshard_every, max(capacity, 1), 1, false, ControllerState[], Vector{Int}[], -1,
    Float64[], Float64[], nothing, ReentrantLock()
)

"""Forget every unit's controller state, the composition and the signal: a new run starts."""
function reset!(s::ScopeState)
    empty!(s.units)
    empty!(s.shards)
    s.macro_step = -1
    s.signal = zeros(s.ncells)
    s.previous = zeros(s.ncells)
    return s
end

"""The initial composition (§6.1): shard `s` holds cells `⌊sC/S⌋ ≤ c < ⌊(s+1)C/S⌋` (0-based) of the canonical cell order; empty shards dropped. Cells are 1-based here."""
function initial_shards(C::Int, S::Int)
    out = Vector{Int}[]
    for s in 0:(S - 1)
        r = collect((fld(s * C, S) + 1):fld((s + 1) * C, S))
        isempty(r) || push!(out, r)
    end
    return out
end

"""
    reshard!(s, R)

Re-compose a `shard` leaf's shards from the signal `R` of the previous macro
step (§6.2): cells sorted by `R_c` descending, ties (equal bits) by cell
index, cut into as many contiguous shards as before, each listing its cells
ascending; each new shard's controller starts from the old shard that held
its smallest cell.
"""
function reshard!(s::ScopeState, R::Vector{Float64})
    C, S = s.ncells, length(s.shards)
    order = sort(collect(1:C); lt = (a, b) -> reinterpret(UInt64, R[a]) == reinterpret(UInt64, R[b]) ? a < b :
        (isless(R[b], R[a]) || (!isless(R[a], R[b]) && a < b)))
    shards = Vector{Int}[]
    for k in 0:(S - 1)
        v = sort!(order[(fld(k * C, S) + 1):fld((k + 1) * C, S)])
        isempty(v) || push!(shards, v)
    end
    owner(c) = something(findfirst(sh -> insorted(c, sh), s.shards), 1)
    s.units = ControllerState[copy(s.units[owner(first(sh))]) for sh in shards]
    s.shards = shards
    return s
end

"""
    unit_leaf(leaf, s, m, ops) -> (LeafNode, UnitTarget, cellsq, cellacc)

A leaf over a unit of `m` cells of the scoped leaf `leaf`: its operators
([`UnitOperator`](@ref)) over the unit's state, `block_diagonal` storage of
`m` cells, the leaf's tolerances, domain and method at the unit's DOFs.
Under `shard`, its tolerances carry the buffers the error norm writes each
cell's squared ratios into (§6.2).
"""
function unit_leaf(leaf::LeafNode, s::ScopeState, m::Int, lanes)
    n, nc = s.n, s.ncells
    target = UnitTarget(Int[], Float64[], Float64[])
    ops = AbstractOperator[
        UnitOperator(op, lanes === nothing ? nothing : lanes[k], target, n, m, nc) for (k, op) in enumerate(leaf.ops.ops)
    ]
    tol = nothing
    if leaf.tol !== nothing
        # Every unit DOF (i, k) takes the leaf's abstol at species i; a
        # leaf's tolerance per variable is equal over its cells.
        abstol = Float64[leaf.tol.abstol[(i - 1) * nc + 1] for i in 1:n for _ in 1:m]
        tol = s.kind === :shard ? Tolerances(leaf.tol.reltol, abstol, collect(1:(n * m)), zeros(m), zeros(m)) :
            Tolerances(leaf.tol.reltol, abstol, collect(1:(n * m)))
    end
    nonneg = sort!(unique!(Int[(div(j - 1, nc)) * m + k for j in leaf.nonnegative for k in 1:m]))
    ul = LeafNode(
        leaf.id, leaf.opnames, OperatorSum(ops), leaf.tab, leaf.method, leaf.dt, tol, leaf.dt0, leaf.dtmin, leaf.dtmax,
        leaf.controller, leaf.maxiters, nonneg, leaf.on_violation, leaf.bindings, leaf.linsolve, leaf.check_stages, nothing,
        nothing, nothing
    )
    return ul, target
end

"""The planned operator under a leaf's operator (`LeafOperator` or `PlannedOperator`), or `nothing`."""
_planned(o::PlannedOperator) = o
_planned(o::LeafOperator) = o.op
_planned(o) = nothing

"""The lane programs of a scoped leaf's operators, built on first use; `nothing` when one operator is not a plan's."""
function scope_lanes!(leaf::LeafNode, s::ScopeState, pnames, tspan)
    s.whole_state && return nothing
    return lock(s.lock) do
        s.lanes === nothing || return s.lanes
        planned = [_planned(o) for o in leaf.ops.ops]
        any(isnothing, planned) && return (s.lanes = false)
        L = first(planned).layout
        dofs = leaf.state === nothing ? collect(1:length(L.names)) : leaf.state.dofs
        species = String[]
        for x in sort!(collect(keys(L.offset)))
            o = L.offset[x]
            any(d -> o < d <= o + prod(L.extents[x]; init = 1), dofs) && push!(species, x)
        end
        s.lanes = [lane_ops(op, species, s.capacity, pnames, tspan) for op in planned]
    end
end

"""
    ScopedOutcome

A scoped leaf invocation's result: the leaf's state at its end, its tape
record, the right-hand-side evaluations and the leaf's states at the
interior stops.
"""
const ScopedOutcome = LeafOutcome

"""Make resident every forcing set the invocation over `[ta, tb]` can take (spec/partitioning.md §7, D101)."""
function make_resident!(leaf::LeafNode, ta, tb)
    for o in leaf.ops.ops
        op = _planned(o)
        (op === nothing || op.forcing === nothing) && continue
        F = op.forcing
        epoch_buffers(F, epoch_of(F, ta))
        is_epoch_start(F, tb) && epoch_buffers(F, epoch_of(F, tb))
    end
    return nothing
end

"""
    scoped_forward(leaf, s, x, bg, p, ta, tb, path, Δmacro, stops, k, pnames, tspan) -> LeafOutcome

One invocation of the scoped leaf `leaf` over `[ta, tb]` from its state `x`,
the model state at its start being `bg`, in macro step `k`: every unit
integrated by [`leaf_forward`](@ref) on the executor threads, then merged
(spec/partitioning.md §2.3, §2.4): the state, the counts summed, the first
non-success in unit order, the per-cell or per-shard records of §11.
"""
function scoped_forward(
        leaf::LeafNode, s::ScopeState, x::Vector{Float64}, bg::Vector{Float64}, p, ta::Float64, tb::Float64,
        path::String, Δmacro::Float64, stops::Vector{Float64}, k::Int, pnames, tspan
    )
    n, nc = s.n, s.ncells
    length(s.signal) == nc || (s.signal = zeros(nc); s.previous = zeros(nc))
    if s.kind === :shard && isempty(s.shards)
        s.shards = initial_shards(nc, s.nshards)
    end
    # The signal of the macro step (§6.2), and re-sharding at its start.
    if s.macro_step != k
        if s.macro_step >= 0
            s.previous = s.signal
            s.signal = zeros(nc)
        end
        s.macro_step = k
        if s.kind === :shard && s.reshard_every !== nothing && k > 0 && k % s.reshard_every == 0 && !isempty(s.units)
            reshard!(s, s.previous)
        end
    end
    units = s.kind === :cell ? [[c] for c in 1:nc] : s.shards
    if length(s.units) != length(units)
        s.units = [ControllerState(leaf.tab.controller) for _ in units]
    end
    lanes = scope_lanes!(leaf, s, pnames, tspan)
    lanes === false && (lanes = nothing)
    make_resident!(leaf, ta, tb)
    results = Vector{Any}(undef, length(units))
    next = Threads.Atomic{Int}(1)
    function work()
        held = Dict{Int, Any}()
        while true
            i = Threads.atomic_add!(next, 1)
            i > length(units) && break
            cells = units[i]
            m = length(cells)
            ul, target = get!(() -> unit_leaf(leaf, s, m, lanes), held, m)
            target.cells = cells
            target.xbg = x
            target.bg = bg
            xu = Float64[x[(sp - 1) * nc + c] for sp in 1:n for c in cells]
            cs = copy(s.units[i])
            ul.tol !== nothing && ul.tol.cellacc !== nothing && fill!(ul.tol.cellacc, 0.0)
            out = leaf_forward(ul, cs, xu, p, ta, tb, path, Δmacro, copy(stops))
            sig = ul.tol === nothing || ul.tol.cellacc === nothing ? nothing : copy(ul.tol.cellacc)
            results[i] = (out, cs, sig)
        end
        return nothing
    end
    nt = clamp(s.threads, 1, max(length(units), 1))
    if nt == 1
        work()
    else
        tasks = [Threads.@spawn work() for _ in 1:nt]
        foreach(wait, tasks)
    end
    # Merge, units in order.
    xnew = copy(x)
    rec = LeafRecord(path, ta, tb - ta, NTuple{2, Float64}[], 0, 0, "Success", Pair{Int, Vector{Int}}[])
    nf = 0
    interior = Tuple{Float64, Vector{Float64}}[]
    acc, rej = Int[], Int[]
    shard_recs = ShardRecord[]
    failed = nothing
    for (i, cells) in enumerate(units)
        out, cs, sig = results[i]
        m = length(cells)
        for sp in 1:n, (kk, c) in enumerate(cells)
            xnew[(sp - 1) * nc + c] = out.u[(sp - 1) * m + kk]
        end
        for (j, (ts, xs)) in enumerate(out.interior)
            length(interior) < j && push!(interior, (ts, copy(x)))
            for sp in 1:n, (kk, c) in enumerate(cells)
                interior[j][2][(sp - 1) * nc + c] = xs[(sp - 1) * m + kk]
            end
        end
        s.units[i] = cs
        r = out.record
        rec.accepted += r.accepted
        rec.rejected += r.rejected
        nf += out.nf
        if rec.retcode == "Success" && r.retcode != "Success"
            rec.retcode = r.retcode
            failed = s.kind === :cell ? "cell $(cells[1] - 1)" : "shard $(i - 1)"
        end
        if s.kind === :cell
            push!(acc, r.accepted)
            push!(rej, r.rejected)
        else
            for (kk, c) in enumerate(cells)
                s.signal[c] += sig === nothing ? 0.0 : sig[kk]
            end
            push!(shard_recs, ShardRecord(Int[c - 1 for c in cells], r.steps, r.accepted, r.rejected))
        end
    end
    if s.kind === :cell
        rec.cells = (acc, rej)
    else
        rec.shards = shard_recs
    end
    failed === nothing || (rec.failed_unit = failed)
    return LeafOutcome(xnew, rec, nf, interior, 0.0)
end
