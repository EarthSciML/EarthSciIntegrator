# The runtime of algebraic equations: `solve` nodes and the consistent
# initialization of index-1 DAEs (spec/methods.md §14, §15; spec/run-config.md
# §4.3, §4.10; D95, D120–D122).

"""
    AlgebraicInfo

What the runtime knows of an algebraic operator beyond its documents: its
unknowns' DOFs (canonical order) and, per DOF, its lower and upper bound
sources (spec/plan.md §5.9.7): `(:const, v)` or `(:param, i)` (index into the
parameter vector), `nothing` for an unbounded side.
"""
struct AlgebraicInfo
    dofs::Vector{Int}
    names::Vector{String}            # the unknown variable of each DOF
    lower::Vector{Any}
    upper::Vector{Any}
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
    tol::Float64
    relchange::Union{Nothing, String}
    maxiter::Int
    recover::Vector{Any}
    pivot::Bool
end

function SolveMethod(m::AbstractDict, id)
    kind = haskey(m, "ptc") ? :ptc : haskey(m, "picard") ? :picard : :newton
    kind === :picard && throw(UnsupportedError("solve node $id: picard needs the frozen Jacobian document (spec/plan.md §6.5), which the planner does not emit yet"))
    conv = get(m, "converged", Dict{String, Any}("residual" => 1.0e-10))
    haskey(conv, "rel_change") && throw(UnsupportedError("solve node $id: converged.rel_change needs an observed's value, which the runtime does not evaluate yet"))
    ls = get(m, "linsolve", Dict{String, Any}("dense_lu" => Dict{String, Any}()))
    pivot = ls isa AbstractDict && haskey(ls, "block_lu") ? Bool(get(ls["block_lu"], "pivot", true)) : true
    p = get(m, "ptc", Dict{String, Any}())
    return SolveMethod(
        kind, Float64(get(p, "dt0", 1.0e-3)), Float64(get(p, "growth", 2.0)),
        Symbol(get(m, "linesearch", "none")), Float64(conv["residual"]), nothing,
        Int(get(m, "maxiter", 100)), collect(Any, get(m, "recover", Any[])), pivot
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

"""`G_z`, dense over the unknowns."""
function residual_jacobian(ops, dofs, u, z, p, t)
    x = copy(u)
    x[dofs] = z
    J = DenseJacobian{Float64}(length(u))
    for op in ops
        jac!(op, J, x, p, t)
    end
    return J.A[dofs, dofs]
end

_mid(l, x, h) = min(max(x, l), h)
_proj(z, l, h) = Float64[_mid(l[i], z[i], h[i]) for i in eachindex(z)]
_R(z, G, l, h) = Float64[z[i] - _mid(l[i], z[i] + G[i], h[i]) for i in eachindex(z)]
_ninf(v) = isempty(v) ? 0.0 : maximum(abs, v)

function _linsolve(A::Matrix{Float64}, b::Vector{Float64}, pivot::Bool)
    F = lu_factor(copy(A), pivot)
    issuccess_lu(F) || return nothing
    return lu_solve(F, b)
end

"""One attempt of a solve from `z0` (§15.2–§15.5); `ω` the under-relaxation, `p` the parameters (possibly scaled)."""
function solve_attempt(ops, dofs, u, z0, p, t, m::SolveMethod, l, h, ω)
    G(z) = residual_at(ops, dofs, u, z, p, t)
    z = _proj(z0, l, h)
    g = G(z)
    r = _R(z, g, l, h)
    n = _ninf(r)
    isfinite(n) || return z, false, 0
    n <= m.tol && return z, true, 0
    δ = m.dt0
    for k in 1:m.maxiter
        A = residual_jacobian(ops, dofs, u, z, p, t)
        if m.kind === :ptc
            M = -A
            b = copy(g)
            for i in eachindex(z)
                M[i, i] += inv(δ)
                # A component held at a bound by an outward rate does not move (§15.4).
                if (z[i] <= l[i] && g[i] <= 0) || (z[i] >= h[i] && g[i] >= 0)
                    M[i, :] .= 0.0
                    M[i, i] = 1.0
                    b[i] = 0.0
                end
            end
            s = _linsolve(M, b, m.pivot)
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
            D = -A
            for i in eachindex(z)
                x = z[i] + g[i]
                if x <= l[i] || x >= h[i]
                    D[i, :] .= 0.0
                    D[i, i] = 1.0
                end
            end
            s = _linsolve(D, -r, m.pivot)
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
        n <= m.tol && return z, true, k
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
    z, ok, it = solve_attempt(ops, info.dofs, u, warm, p, t, m, l, h, 1.0)
    pp = copy(p)
    scaled = Dict{Int, Int}()
    for a in m.recover
        ok && break
        if haskey(a, "underrelax")
            push!(actions, "underrelax")
            z, ok, it = solve_attempt(ops, info.dofs, u, warm, pp, t, m, l, h, Float64(a["underrelax"]))
        elseif haskey(a, "scale_param")
            i = findfirst(==(String(a["scale_param"])), pnames)
            i === nothing && throw(ConfigError("solve: recover scale_param $(a["scale_param"]) is not a scalar parameter"))
            for _ in 1:Int(get(a, "max", 1))
                ok && break
                pp[i] *= Float64(a["by"])
                push!(actions, "scale_param")
                z, ok, it = solve_attempt(ops, info.dofs, u, warm, pp, t, m, l, h, 1.0)
            end
        elseif haskey(a, "revert_warm_start") || a == "revert_warm_start"
            push!(actions, "revert_warm_start")
            z, ok, it = solve_attempt(ops, info.dofs, u, initial, pp, t, m, l, h, 1.0)
        end
    end
    lower, upper, degenerate = Int[], Int[], Int[]
    if ok
        g = residual_at(ops, info.dofs, u, z, pp, t)
        ε = m.tol
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
    info = AlgebraicInfo(dofs, znames, Any[nothing for _ in dofs], Any[nothing for _ in dofs])
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
