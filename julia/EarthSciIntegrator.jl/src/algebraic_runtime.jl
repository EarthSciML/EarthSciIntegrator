# The runtime of algebraic equations: `solve` nodes and the consistent
# initialization of index-1 DAEs (spec/methods.md §14, §15; spec/run-config.md
# §4.3, §4.10; D95, D120–D122).

"""
    AlgebraicInfo

What the runtime knows of an algebraic operator beyond its documents: its
unknowns' DOFs (canonical order) and, per DOF, its lower and upper bound
sources (spec/plan.md §5.9.7): `(:const, v)` or `(:param, i)` (index into the
parameter vector), `nothing` for an unbounded side; its frozen documents
(`FrozenDocs`), when the plan has them; and its blocks, when the plan gives
its structure a `block` (spec/plan.md §5.4): per cell, the positions among
the unknowns of the block's elements in the block's order.
"""
struct AlgebraicInfo
    dofs::Vector{Int}
    names::Vector{String}            # the unknown variable of each DOF
    lower::Vector{Any}
    upper::Vector{Any}
    frozen::Any                      # FrozenDocs or nothing
    blocks::Union{Nothing, Vector{Vector{Int}}}
end
AlgebraicInfo(dofs, names, lower, upper) = AlgebraicInfo(dofs, names, lower, upper, nothing, nothing)
AlgebraicInfo(dofs, names, lower, upper, frozen) = AlgebraicInfo(dofs, names, lower, upper, frozen, nothing)

"""
    FrozenDocs

An algebraic operator's frozen documents (spec/plan.md §6.5): the frozen
residual document `resid` and, for a `picard` node, the frozen Jacobian
document `jac` (else `nothing`), both bound to their own layout of `nf`
DOFs, the model's states and the frozen states `FrozenEval.frozen_m`. `tolf`
maps each model DOF to its DOF there, `fdofs[m]` lists the DOFs of
`FrozenEval.frozen_m`, and `points` the frozen Jacobian's band points whose
row and column are unknowns, as `(slot, row, col)`: the slot of the band
field in `jac`'s output, and the positions of the row and column among the
operator's unknowns.
"""
struct FrozenDocs
    resid::Any
    jac::Any
    nf::Int
    tolf::Vector{Int}
    fdofs::Vector{Vector{Int}}
    points::Vector{NTuple{3, Int}}
end

"""The frozen layout's state at the model state `x`, its frozen states zero."""
function frozen_state_vector(F::FrozenDocs, x)
    uf = zeros(F.nf)
    uf[F.tolf] = x
    return uf
end

"""The values of the frozen observeds `O_m` at the model state `x` (spec/plan.md §6.5), each column-major."""
function frozen_values(F::FrozenDocs, x, p, t)
    return evaluate(F.resid, frozen_state_vector(F, x), p, t) do du
        [Float64[du[F.resid.perm[k]] for k in ds] for ds in F.fdofs]
    end
end

"""The Jacobian over the unknowns with the frozen observeds held at their values at `x` (spec/methods.md §15.5), dense."""
function frozen_jacobian(F::FrozenDocs, x, nz, p, t)
    uf = frozen_state_vector(F, x)
    for (ds, v) in zip(F.fdofs, frozen_values(F, x, p, t))
        uf[ds] = v
    end
    A = Dict{Tuple{Int, Int}, Float64}()
    evaluate(F.jac, uf, p, t) do du
        for (s, r, c) in F.points
            A[(r, c)] = get(A, (r, c), 0.0) + du[s]
        end
    end
    return A
end

"""The bounds of `info` at the parameter vector `p`: `(l, u)` with `-Inf`/`Inf` for an unbounded side."""
function bound_values(info::AlgebraicInfo, p)
    val(b, inf) = b === nothing ? inf : b[1] === :const ? Float64(b[2]) : Float64(p[b[2]])
    return Float64[val(b, -Inf) for b in info.lower], Float64[val(b, Inf) for b in info.upper]
end

"""
    algebraic_info(entry, L, pnames, docjson) -> AlgebraicInfo

From an algebraic operator's plan entry (`writes`, `bounds`), the layout, the
parameter names and its residual document as JSON (whose declarations give
an array parameter's elements, outermost index first).
"""
function algebraic_info(name, entry, L::DofLayout, pnames::Vector{String}, docjson)
    zs = String.(entry["writes"])
    dofs = Int[]
    names = String[]
    for z in zs, i in 1:prod(L.extents[z]; init = 1)
        push!(dofs, L.offset[z] + i)
        push!(names, z)
    end
    lower = Any[nothing for _ in dofs]
    upper = Any[nothing for _ in dofs]
    bounds = get(entry, "bounds", Dict{String, Any}())
    k = 0
    for z in zs
        n = prod(L.extents[z]; init = 1)
        b = get(bounds, z, Dict{String, Any}())
        for (key, dest) in (("lb", lower), ("ub", upper))
            haskey(b, key) || continue
            v = b[key]
            src = if v isa Real
                fill((:const, Float64(v)), n)
            else
                pn = String(v)
                i = findfirst(==(pn), pnames)
                i !== nothing ? fill((:param, i), n) :
                    [(:const, x) for x in array_default(docjson, pn, L.extents[z], name)]
            end
            dest[(k + 1):(k + n)] = src
        end
        k += n
    end
    return AlgebraicInfo(dofs, names, lower, upper)
end

"""The `default` of the parameter `name` declared in a derived document's JSON, at each element of extents `ext`, column-major."""
function array_default(doc, name::AbstractString, ext, opname)
    segs = split(name, '.')
    node = get(doc["models"], segs[1], nothing)
    for s in segs[2:(end - 1)]
        node === nothing && break
        node = get(get(node, "subsystems", Dict()), s, nothing)
    end
    v = node === nothing ? nothing : get(get(node, "variables", Dict()), segs[end], nothing)
    (v === nothing || !haskey(v, "default")) &&
        throw(ConfigError("operator '$opname': the bound parameter $name has no default in its residual document"))
    d = v["default"]
    d isa Real && return fill(Float64(d), prod(ext; init = 1))
    out = Float64[]
    for c in cells(ext)
        x = d
        for i in c
            x = x[i]
        end
        push!(out, Float64(x))
    end
    return out
end

# ---------------------------------------------------------------------------
# The solve (spec/methods.md §15)

"""A solve method as the resolved plan gives it (spec/plan.md §7.1)."""
struct SolveMethod
    kind::Symbol                     # :newton, :picard, :ptc
    dt0::Float64
    growth::Float64
    linesearch::Symbol
    tol::Float64                     # converged.residual; Inf when only rel_change is given
    relchange::Int                   # the rel_change observed's m among the frozen observeds (spec/plan.md §6.5); 0 for none
    rctol::Float64
    maxiter::Int
    recover::Vector{Any}
    pivot::Bool
    block::Bool                      # linsolve block_lu or sparse_block_lu: one LU per block of the plan's structure
end

function SolveMethod(m::AbstractDict, id)
    kind = haskey(m, "ptc") ? :ptc : haskey(m, "picard") ? :picard : :newton
    conv = get(m, "converged", Dict{String, Any}("residual" => 1.0e-10))
    # The frozen observeds, in the frozen documents' order (spec/plan.md §6.5):
    # picard's freeze list, then the rel_change observed if not among them.
    frozen = kind === :picard ? String.(m["picard"]["freeze"]) : String[]
    rc = 0
    if haskey(conv, "rel_change")
        o = String(conv["rel_change"])
        o in frozen || push!(frozen, o)
        rc = findfirst(==(o), frozen)
    end
    ls = get(m, "linsolve", Dict{String, Any}("dense_lu" => Dict{String, Any}()))
    # block_lu and sparse_block_lu factor each block of the plan's structure
    # on its own; within a block both factor densely here.
    block = ls isa AbstractDict && (haskey(ls, "block_lu") || haskey(ls, "sparse_block_lu"))
    pivot = ls isa AbstractDict && haskey(ls, "block_lu") ? Bool(get(ls["block_lu"], "pivot", true)) : true
    p = get(m, "ptc", Dict{String, Any}())
    return SolveMethod(
        kind, Float64(get(p, "dt0", 1.0e-3)), Float64(get(p, "growth", 2.0)),
        Symbol(get(m, "linesearch", "none")), Float64(get(conv, "residual", Inf)), rc,
        Float64(get(conv, "tol", 0.0)), Int(get(m, "maxiter", 100)), collect(Any, get(m, "recover", Any[])), pivot, block
    )
end

"""The outcome of a solve: the converged unknowns, iterations, the recover actions taken, and the active set per bounded DOF."""
struct SolveOutcome
    z::Vector{Float64}
    ok::Bool
    iterations::Int
    recover::Vector{String}
    lower::Vector{Int}               # positions (1-based) into the unknowns at their lower bound
    upper::Vector{Int}
    degenerate::Vector{Int}
end

"""`G(z)`: the residuals of `ops` at the model state `u` with its unknowns `dofs` set to `z` (§15.1)."""
function residual_at(ops, dofs, u, z, p, t)
    x = copy(u)
    x[dofs] = z
    out = zeros(length(u))
    for op in ops
        rhs!(op, out, x, p, t)
    end
    return out[dofs]
end

"""
Jacobian storage that keeps the entries whose row and column are both
unknowns, by their positions among the unknowns (`pos`, model DOF to
position), so a solve never forms a matrix over the model state.
"""
struct EntryJacobian <: JacobianStorage
    pos::Dict{Int, Int}
    A::Dict{Tuple{Int, Int}, Float64}
end
@inline function addentry!(S::EntryJacobian, i::Integer, j::Integer, v)
    pi, pj = get(S.pos, i, 0), get(S.pos, j, 0)
    (pi > 0 && pj > 0) && (S.A[(pi, pj)] = get(S.A, (pi, pj), 0.0) + v)
    return S
end

"""`G_z` over the unknowns, as entries `(row, col) => value` by position."""
function residual_jacobian(ops, dofs, u, z, p, t)
    x = copy(u)
    x[dofs] = z
    J = EntryJacobian(Dict(d => i for (i, d) in enumerate(dofs)), Dict{Tuple{Int, Int}, Float64}())
    for op in ops
        jac!(op, J, x, p, t)
    end
    return J.A
end

"""
The blocks of a solve's linear systems (spec/methods.md §15.3): with
`block_lu`, the plan's blocks (`info.blocks`), else one block of every
unknown. Each block is a list of positions among the unknowns, `of[i]` the
block and place of position `i`.
"""
struct SolveBlocks
    blocks::Vector{Vector{Int}}
    of::Vector{Tuple{Int, Int}}
end
function SolveBlocks(info::AlgebraicInfo, m::SolveMethod)
    nz = length(info.dofs)
    blocks = if m.block
        info.blocks === nothing && throw(
            UnsupportedError("solve: a block linsolve, but the operator's plan entry gives no block structure (spec/plan.md §5.4)")
        )
        info.blocks
    else
        [collect(1:nz)]
    end
    of = fill((0, 0), nz)
    for (k, P) in enumerate(blocks), (i, q) in enumerate(P)
        of[q] = (k, i)
    end
    any(==((0, 0)), of) && throw(ConfigError("solve: the plan's blocks do not cover every unknown"))
    return SolveBlocks(blocks, of)
end

"""
Solve `(σI − A) x = b`, with the rows `held` replaced by the identity's,
block by block: one LU per block (spec/methods.md §15.3, §15.4). `nothing`
when a block is singular.
"""
function _linsolve(A::AbstractDict, σ, held, b, B::SolveBlocks, pivot::Bool)
    Ms = [zeros(length(P), length(P)) for P in B.blocks]
    for ((r, c), v) in A
        (kr, ir), (kc, ic) = B.of[r], B.of[c]
        kr == kc || throw(ConfigError("solve: the Jacobian couples unknowns $r and $c of two blocks of the plan's block structure"))
        Ms[kr][ir, ic] -= v
    end
    x = zeros(length(b))
    for (k, P) in enumerate(B.blocks)
        M = Ms[k]
        for (i, q) in enumerate(P)
            M[i, i] += σ
            if held[q]
                M[i, :] .= 0.0
                M[i, i] = 1.0
            end
        end
        F = lu_factor(M, pivot)
        issuccess_lu(F) || return nothing
        x[P] = lu_solve(F, b[P])
    end
    return x
end

_mid(l, x, h) = min(max(x, l), h)
_proj(z, l, h) = Float64[_mid(l[i], z[i], h[i]) for i in eachindex(z)]
_R(z, G, l, h) = Float64[z[i] - _mid(l[i], z[i] + G[i], h[i]) for i in eachindex(z)]
_ninf(v) = isempty(v) ? 0.0 : maximum(abs, v)


"""The model state `u` with the unknowns `dofs` set to `z`."""
function _with_z(u, dofs, z)
    x = Float64.(u)
    x[dofs] = z
    return x
end

"""The frozen documents of `info`, which a picard method or `converged.rel_change` needs (spec/plan.md §6.5)."""
function _frozen(info::AlgebraicInfo, m::SolveMethod)
    (m.kind === :picard || m.relchange > 0) || return nothing
    F = info.frozen
    F === nothing && throw(ConfigError("solve: a picard method or converged.rel_change needs the operator's frozen documents (spec/plan.md §6.5), which the plan does not have"))
    m.kind === :picard && F.jac === nothing &&
        throw(ConfigError("solve: a picard method needs the operator's frozen Jacobian document (spec/plan.md §6.5), which the plan does not have"))
    return F
end

"""`converged.rel_change` (§15.2): `‖O_k − O_{k-1}‖₂ ≤ tol ‖O_k‖₂`."""
_rel_met(o, oprev, tol) = sqrt(sum(abs2, o .- oprev)) <= tol * sqrt(sum(abs2, o))

"""
One attempt of a solve from `z0` (§15.2–§15.5); `ω` the under-relaxation,
`p` the parameters (possibly scaled). With `converged.rel_change` the
attempt converges when every criterion given is met; the change is measured
from the previous iterate, so the starting iterate cannot meet it.
"""
function solve_attempt(ops, info::AlgebraicInfo, u, z0, p, t, m::SolveMethod, l, h, ω)
    dofs = info.dofs
    F = _frozen(info, m)
    G(z) = residual_at(ops, dofs, u, z, p, t)
    O(z) = frozen_values(F, _with_z(u, dofs, z), p, t)[m.relchange]
    z = _proj(z0, l, h)
    g = G(z)
    r = _R(z, g, l, h)
    n = _ninf(r)
    isfinite(n) || return z, false, 0
    m.relchange == 0 && n <= m.tol && return z, true, 0
    o = m.relchange == 0 ? nothing : O(z)
    B = SolveBlocks(info, m)
    δ = m.dt0
    for k in 1:m.maxiter
        A = m.kind === :picard ? frozen_jacobian(F, _with_z(u, dofs, z), length(z), p, t) :
            residual_jacobian(ops, dofs, u, z, p, t)
        if m.kind === :ptc
            # A component held at a bound by an outward rate does not move (§15.4).
            held = BitVector([(z[i] <= l[i] && g[i] <= 0) || (z[i] >= h[i] && g[i] >= 0) for i in eachindex(z)])
            b = copy(g)
            b[held] .= 0.0
            s = _linsolve(A, inv(δ), held, b, B, m.pivot)
            s === nothing && return z, false, k
            znew = _proj(z .+ ω .* s, l, h)
            gnew = G(znew)
            rnew = _R(znew, gnew, l, h)
            nnew = _ninf(rnew)
            isfinite(nnew) || return z, false, k
            δ *= nnew == 0 ? m.growth : min(m.growth, n / nnew)
            z, g, r, n = znew, gnew, rnew, nnew
        else
            # The generalized Jacobian of R (§15.3): an identity row where
            # `mid` takes a bound, else -G_z's row.
            held = BitVector([(z[i] + g[i] <= l[i] || z[i] + g[i] >= h[i]) for i in eachindex(z)])
            s = _linsolve(A, 0.0, held, -r, B, m.pivot)
            s === nothing && return z, false, k
            s .*= ω
            α = 1.0
            if m.linesearch === :backtracking
                φ = 0.5 * sum(abs2, r)
                α = 2.0^-10
                for a in (2.0^-j for j in 0:10)
                    zt = _proj(z .+ a .* s, l, h)
                    rt = _R(zt, G(zt), l, h)
                    if 0.5 * sum(abs2, rt) <= (1 - 1.0e-4 * a) * φ
                        α = a
                        break
                    end
                end
            end
            z = _proj(z .+ α .* s, l, h)
            g = G(z)
            r = _R(z, g, l, h)
            n = _ninf(r)
            isfinite(n) || return z, false, k
        end
        if m.relchange == 0
            n <= m.tol && return z, true, k
        else
            onew = O(z)
            met = n <= m.tol && _rel_met(onew, o, m.rctol)
            o = onew
            met && return z, true, k
        end
    end
    return z, false, m.maxiter
end

"""
    run_solve(ops, info, u, p, t, m, warm, initial, pnames) -> SolveOutcome

A solve with its `recover` ladder (§15.7) and, at convergence, the active set
(spec/run-config.md §4.10): a component at a bound is set to it exactly.
`warm` is the starting iterate; `initial` the model's initial value of the
unknowns (`revert_warm_start`).
"""
function run_solve(ops, info::AlgebraicInfo, u, p, t, m::SolveMethod, warm, initial, pnames)
    l, h = bound_values(info, p)
    any(i -> l[i] > h[i], eachindex(l)) && throw(
        ConfigError("solve: the lower bound of $(info.names[findfirst(i -> l[i] > h[i], eachindex(l))]) exceeds its upper bound")
    )
    actions = String[]
    z, ok, it = solve_attempt(ops, info, u, warm, p, t, m, l, h, 1.0)
    pp = copy(p)
    scaled = Dict{Int, Int}()
    for a in m.recover
        ok && break
        if haskey(a, "underrelax")
            push!(actions, "underrelax")
            z, ok, it = solve_attempt(ops, info, u, warm, pp, t, m, l, h, Float64(a["underrelax"]))
        elseif haskey(a, "scale_param")
            i = findfirst(==(String(a["scale_param"])), pnames)
            i === nothing && throw(ConfigError("solve: recover scale_param $(a["scale_param"]) is not a scalar parameter"))
            for _ in 1:Int(get(a, "max", 1))
                ok && break
                pp[i] *= Float64(a["by"])
                push!(actions, "scale_param")
                z, ok, it = solve_attempt(ops, info, u, warm, pp, t, m, l, h, 1.0)
            end
        elseif haskey(a, "revert_warm_start") || a == "revert_warm_start"
            push!(actions, "revert_warm_start")
            z, ok, it = solve_attempt(ops, info, u, initial, pp, t, m, l, h, 1.0)
        end
    end
    lower, upper, degenerate = Int[], Int[], Int[]
    if ok
        g = residual_at(ops, info.dofs, u, z, pp, t)
        ε = isfinite(m.tol) ? m.tol : 0.0
        for i in eachindex(z)
            (isfinite(l[i]) || isfinite(h[i])) || continue
            atl = isfinite(l[i]) && (z[i] + g[i] <= l[i] || (z[i] - l[i] <= ε && abs(g[i]) <= ε))
            atu = !atl && isfinite(h[i]) && (z[i] + g[i] >= h[i] || (h[i] - z[i] <= ε && abs(g[i]) <= ε))
            if atl
                z[i] = l[i]; push!(lower, i)
            elseif atu
                z[i] = h[i]; push!(upper, i)
            end
            (atl || atu) && abs(g[i]) <= ε && push!(degenerate, i)
        end
    end
    return SolveOutcome(z, ok, it, actions, lower, upper, degenerate)
end

# ---------------------------------------------------------------------------
# The schedule node and its tape record (spec/tape.md §2)

"""A `solve` node (spec/run-config.md §4.10)."""
struct SolveNode <: Node
    id::String
    opname::String
    ops::Vector{AbstractOperator}
    info::AlgebraicInfo
    method::SolveMethod
    warm_start::Symbol
end
children(::SolveNode) = Node[]
nodekind(::SolveNode) = "solve"

"""A `solve` invocation: the converged unknowns, iterations, recover actions and the active set per bounded unknown (positions into its elements)."""
struct SolveRecord <: TapeRecord
    path::String
    t::Float64
    dt::Float64
    z::Vector{Float64}
    iterations::Int
    recover::Vector{String}
    active::Dict{String, NTuple{3, Vector{Int}}}
end

function build_solve(spec::AbstractDict, prob::ToyProblem, id::String)
    name = String(spec["solve"])
    haskey(prob.operators, name) || throw(ConfigError("solve node $id names '$name', which $(prob.name) does not define"))
    info = get(prob.algebraic, name, nothing)
    info === nothing && throw(ConfigError("solve node $id: '$name' is not an algebraic operator"))
    ws = Symbol(get(spec, "warm_start", "previous"))
    ws in (:previous, :initial) || throw(ConfigError("solve node $id: warm_start $ws"))
    return SolveNode(id, name, AbstractOperator[prob.operators[name]], info, SolveMethod(spec["method"], id), ws)
end

"""The active set of an outcome per bounded unknown variable, element positions 0-based within the variable (spec/tape.md §2)."""
function active_sets(info::AlgebraicInfo, o::SolveOutcome)
    out = Dict{String, NTuple{3, Vector{Int}}}()
    first = Dict{String, Int}()
    for (i, z) in enumerate(info.names)
        haskey(first, z) || (first[z] = i)
    end
    for (set, k) in ((o.lower, 1), (o.upper, 2), (o.degenerate, 3)), i in set
        z = info.names[i]
        e = get!(out, z, (Int[], Int[], Int[]))
        push!(e[k], i - first[z])
    end
    return out
end

_jline(io::IO, r::SolveRecord) = begin
    print(
        io, "{\"path\": ", _jstr(r.path), ", \"node\": \"solve\", \"t\": ", _jnum(r.t), ", \"dt\": ", _jnum(r.dt),
        ", \"z\": [", join(map(_jnum, r.z), ", "), "], \"iterations\": ", r.iterations,
        ", \"recover\": [", join(map(_jstr, r.recover), ", "), "]"
    )
    if !isempty(r.active)
        parts = String[]
        for z in sort!(collect(keys(r.active)))
            lo, up, dg = r.active[z]
            fields = String[]
            isempty(lo) || push!(fields, "\"lower\": [" * join(lo, ", ") * "]")
            isempty(up) || push!(fields, "\"upper\": [" * join(up, ", ") * "]")
            isempty(dg) || push!(fields, "\"degenerate\": [" * join(dg, ", ") * "]")
            isempty(fields) || push!(parts, _jstr(z) * ": {" * join(fields, ", ") * "}")
        end
        isempty(parts) || print(io, ", \"active\": {", join(parts, ", "), "}")
    end
    print(io, "}")
end

"""
A `solve` node over `[ta, tb]` (spec/methods.md §15): forward, the solve from
its warm start, recorded on the tape; on replay, the tape's converged
unknowns. A solve that fails after its `recover` ladder ends the run with
`SolveFailure` (§12).
"""
function advance!(run::Run, ctx::ExecCtx, node::SolveNode, u, ta::Float64, tb::Float64)
    path = next_path!(ctx, node.id)
    mode = ctx.mode
    info = node.info
    out = copy(u)
    if mode isa ForwardMode
        p = _params(run, mode)
        init = run.u0[info.dofs]
        warm = node.warm_start === :initial ? init : Float64.(u[info.dofs])
        o = run_solve(node.ops, info, u, p, ta, node.method, warm, init, run.prob.pnames)
        o.ok || throw(RunFailure("SolveFailure", "solve node $(node.id) ($path) did not converge at t = $ta"))
        push!(mode.tape, SolveRecord(path, ta, tb - ta, o.z, o.iterations, o.recover, active_sets(info, o)))
        out[info.dofs] = o.z
        return out
    end
    mode.trace === nothing || throw(UnsupportedError("the adjoint of solve node $(node.id) (spec/adjoint.md §4.6) is not implemented in this stage"))
    rec = tape_record(mode.tape, path, SolveRecord, ta, tb)
    out[info.dofs] = rec.z
    return out
end

# ---------------------------------------------------------------------------
# Index-1 DAEs (spec/methods.md §14)

"""
    DAEInfo

A `coupled` node with algebraic operators: `mass`, its algebraic rows over
the leaf's state (true where `M` has a zero diagonal); `ops`, its algebraic
operators on the model state; `info`, their unknowns (no bounds: a bounded
unknown in a DAE is refused); `init`, the consistent-initialization method.
"""
struct DAEInfo
    mass::BitVector
    ops::Vector{AbstractOperator}
    info::AlgebraicInfo
    init::SolveMethod
end

const _DEFAULT_INITIALIZE = Dict{String, Any}(
    "newton" => Dict{String, Any}(), "linesearch" => "backtracking",
    "converged" => Dict{String, Any}("residual" => 1.0e-10), "maxiter" => 100
)

"""The DAEInfo of a `coupled` node with algebraic operators, else `nothing`."""
function dae_info(spec::AbstractDict, prob::ToyProblem, id, names, leafstate)
    haskey(spec, "coupled") || return nothing
    alg = [n for n in names if haskey(prob.algebraic, n)]
    isempty(alg) && return nothing
    infos = AlgebraicInfo[prob.algebraic[n] for n in alg]
    dofs = reduce(vcat, (i.dofs for i in infos))
    znames = reduce(vcat, (i.names for i in infos))
    any(i -> any(!isnothing, i.lower) || any(!isnothing, i.upper), infos) &&
        throw(ConfigError("coupled node $id: a bounded unknown in a DAE is not specified (unsupported_dae_bounds)"))
    # One algebraic operator's frozen documents serve a picard or rel_change
    # initialization (spec/plan.md §6.5), and its blocks a block_lu one.
    one = length(infos) == 1
    info = AlgebraicInfo(dofs, znames, Any[nothing for _ in dofs], Any[nothing for _ in dofs], one ? only(infos).frozen : nothing, one ? only(infos).blocks : nothing)
    model_dofs = leafstate === nothing ? collect(1:prob.N) : leafstate.dofs
    zset = Set(dofs)
    mass = BitVector([d in zset for d in model_dofs])
    init = SolveMethod(get(spec, "initialize", _DEFAULT_INITIALIZE), id)
    return DAEInfo(mass, AbstractOperator[prob.operators[n] for n in alg], info, init)
end

"""
Consistent initialization (§14.4): `g(u, z) = 0` solved for `z` with `u`
held, from the current `z`, at the start of every invocation. A failure ends
the run with `InitialFailure`.
"""
function dae_initialize(run::Run, ctx::ExecCtx, node::LeafNode, u, ta::Float64)
    d = node.dae
    ctx.mode isa ReplayMode && ctx.mode.trace !== nothing &&
        throw(UnsupportedError("the adjoint of the index-1 DAE leaf $(node.id) (spec/adjoint.md §5.10) is not implemented in this stage"))
    p = _params(run, ctx.mode)
    uf = Float64.(u)
    z0 = uf[d.info.dofs]
    o = run_solve(d.ops, d.info, uf, p, ta, d.init, z0, run.u0[d.info.dofs], run.prob.pnames)
    o.ok || throw(RunFailure("InitialFailure", "the consistent initialization of $(node.id) did not converge at t = $ta"))
    out = copy(u)
    out[d.info.dofs] = o.z
    return out
end
