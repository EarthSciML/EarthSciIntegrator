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
    doc::Any                         # the residual document's JSON (an array parameter's default), or nothing
    extents::Dict{String, Vector{Int}}   # each unknown variable's extents
    vjp::Vector{Any}                 # the parameter products of the residual documents (spec/plan.md §6.6), ParamProduct
    fvjp::Vector{Any}                # the parameter products of the frozen residual document
end
AlgebraicInfo(dofs, names, lower, upper) = AlgebraicInfo(dofs, names, lower, upper, nothing, nothing)
AlgebraicInfo(dofs, names, lower, upper, frozen) = AlgebraicInfo(dofs, names, lower, upper, frozen, nothing)
AlgebraicInfo(dofs, names, lower, upper, frozen, blocks) =
    AlgebraicInfo(dofs, names, lower, upper, frozen, blocks, nothing, Dict{String, Vector{Int}}())
AlgebraicInfo(dofs, names, lower, upper, frozen, blocks, doc, extents) =
    AlgebraicInfo(dofs, names, lower, upper, frozen, blocks, doc, extents, Any[], Any[])

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
    names::Vector{String}            # the frozen observeds O_m, in order (the plan entry's `frozen`)
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

"""
The values of the frozen observeds of `info` at the model state `x` (D125), by
name: what a referenced bound or scale reads, once per invocation.
"""
function observed_values(info::AlgebraicInfo, x, p, t)
    F = info.frozen
    F === nothing && return Dict{String, Vector{Float64}}()
    return Dict(zip(F.names, frozen_values(F, x, p, t)))
end

"""
The bounds of `info` at the parameter vector `p` and the frozen observeds'
values `obs`: `(l, u)` with `-Inf`/`Inf` for an unbounded side.
"""
function bound_values(info::AlgebraicInfo, p, obs = Dict{String, Vector{Float64}}())
    function val(b, inf)
        b === nothing && return inf
        b[1] === :const && return Float64(b[2])
        b[1] === :param && return Float64(p[b[2]])
        v = obs[b[2]]
        return length(v) == 1 ? v[1] : v[b[3]]
    end
    return Float64[val(b, -Inf) for b in info.lower], Float64[val(b, Inf) for b in info.upper]
end

"""
The scale of each unknown DOF of `info` under method `m` (spec/methods.md
§15.1, D125): 1 where `scale` names no unknown, else a number, a parameter,
an array parameter's default, or a frozen observed (scalar or one per
element). A scale that is not positive and finite fails the invocation.
"""
function scale_values(info::AlgebraicInfo, m, p, pnames, obs, id)
    d = ones(length(info.dofs))
    isempty(m.scale) && return d
    first = Dict{String, Int}()
    for (i, z) in enumerate(info.names)
        haskey(first, z) || (first[z] = i)
    end
    for (z, v) in m.scale
        haskey(first, z) || continue
        n = count(==(z), info.names)
        vals = if v isa Real
            fill(Float64(v), n)
        else
            name = String(v)
            k = findfirst(==(name), pnames)
            if k !== nothing
                fill(Float64(p[k]), n)
            elseif haskey(obs, name)
                o = obs[name]
                length(o) == 1 ? fill(o[1], n) : o
            else
                array_default(info.doc, name, info.extents[z], id)
            end
        end
        length(vals) == n || throw(ConfigError("solve node $id: the scale $v of $z has $(length(vals)) elements, $z $n"))
        d[first[z]:(first[z] + n - 1)] = vals
    end
    bad = findfirst(x -> !(isfinite(x) && x > 0), d)
    bad === nothing || throw(
        RunFailure("SolveFailure", "solve node $id: the scale of $(info.names[bad]) element $(bad - first[info.names[bad]] + 1) is $(d[bad]), not positive and finite (spec/methods.md §15.1)")
    )
    return d
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
            frozen = String.(get(entry, "frozen", Any[]))
            src = if v isa Real
                fill((:const, Float64(v)), n)
            else
                pn = String(v)
                i = findfirst(==(pn), pnames)
                # A state-free observed bound reads the frozen residual
                # document once per invocation (D125).
                i !== nothing ? fill((:param, i), n) :
                    pn in frozen ? [(:obs, pn, e) for e in 1:n] :
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
    scale::Dict{String, Any}         # unknown => number or name (D125)
    inexact::Bool                    # newton.inexact: Newton–Krylov with Eisenstat–Walker forcing (D126)
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
        kind, Float64(get(p, "dt0", 1.0e-3)), _growth(get(p, "growth", 2.0)),
        Symbol(get(m, "linesearch", "none")), Float64(get(conv, "residual", Inf)), rc,
        Float64(get(conv, "tol", 0.0)), Int(get(m, "maxiter", 100)), collect(Any, get(m, "recover", Any[])), pivot, block,
        Dict{String, Any}(String(k) => v for (k, v) in get(m, "scale", Dict{String, Any}())),
        Bool(get(get(m, "newton", Dict{String, Any}()), "inexact", false))
    )
end

"""`ptc.growth`: a number, or `Inf` for "uncapped" (switched evolution relaxation, D125)."""
_growth(g) = g == "uncapped" ? Inf : Float64(g)

"""
The method of a `recover` step `{"method": M}` (spec/methods.md §15.7): `M`'s
kind, options, `linesearch` and `maxiter`, with the node's `converged`,
`linsolve` and `scale`.
"""
function recover_method(m::SolveMethod, mm::AbstractDict)
    kind = haskey(mm, "ptc") ? :ptc : :newton
    p = get(mm, "ptc", Dict{String, Any}())
    return SolveMethod(
        kind, Float64(get(p, "dt0", 1.0e-3)), _growth(get(p, "growth", 2.0)),
        Symbol(get(mm, "linesearch", "none")), m.tol, m.relchange, m.rctol, Int(get(mm, "maxiter", 100)),
        Any[], m.pivot, m.block, m.scale, Bool(get(get(mm, "newton", Dict{String, Any}()), "inexact", false))
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
    Ms = _block_matrices(A, σ, held, B)
    x = zeros(length(b))
    for (k, P) in enumerate(B.blocks)
        F = lu_factor(Ms[k], pivot)
        issuccess_lu(F) || return nothing
        x[P] = lu_solve(F, b[P])
    end
    return x
end

"""The matrix `σI − A`, with the rows `held` replaced by the identity's, block by block (`B`)."""
function _block_matrices(A::AbstractDict, σ, held, B::SolveBlocks)
    Ms = [zeros(length(P), length(P)) for P in B.blocks]
    for ((r, c), v) in A
        (kr, ir), (kc, ic) = B.of[r], B.of[c]
        kr == kc || throw(ConfigError("solve: the Jacobian couples unknowns $r and $c of two blocks of the plan's block structure"))
        Ms[kr][ir, ic] -= v
    end
    for (k, P) in enumerate(B.blocks)
        M = Ms[k]
        for (i, q) in enumerate(P)
            M[i, i] += σ
            if held[q]
                M[i, :] .= 0.0
                M[i, i] = 1.0
            end
        end
    end
    return Ms
end

_norm2(v) = sqrt(sum(abs2, v))

"""
    _krylov_step(A, held, b, B, m, η) -> s or nothing

The inexact Newton–Krylov step of spec/methods.md §15.3 (D126): GMRES on
`D P⁻¹ y = b`, `s = P⁻¹ y`, with `D = −A` and the rows `held` the identity's;
right-preconditioned, unrestarted, at most N iterations, zero initial guess,
modified Gram–Schmidt with Givens rotations, stopping at `‖D s − b‖₂ ≤ η ‖b‖₂`.
`P` is the per-cell block LU with a block linsolve, else the diagonal of `D`.
`nothing` when `P` is singular.
"""
function _krylov_step(A::AbstractDict, held, b, B::SolveBlocks, m::SolveMethod, η)
    n = length(b)
    Dv(v) = begin
        out = zeros(n)
        for ((r, c), a) in A
            out[r] -= a * v[c]
        end
        for i in 1:n
            held[i] && (out[i] = v[i])
        end
        out
    end
    prec = if m.block
        Fs = [lu_factor(M, m.pivot) for M in _block_matrices(A, 0.0, held, B)]
        all(issuccess_lu, Fs) || return nothing
        v -> begin
            out = zeros(n)
            for (k, P) in enumerate(B.blocks)
                out[P] = lu_solve(Fs[k], v[P])
            end
            out
        end
    else
        # Jacobi: the diagonal of D, −A_ii (1 on a held row; a zero taken as 1).
        d = zeros(n)
        for ((r, c), a) in A
            r == c && (d[r] -= a)
        end
        for i in 1:n
            (held[i] || d[i] == 0) && (d[i] = 1.0)
        end
        v -> v ./ d
    end
    β = _norm2(b)
    β == 0 && return zeros(n)
    V = Vector{Float64}[b ./ β]
    Z = Vector{Float64}[]
    H = zeros(n + 1, n)
    cs, sn = zeros(n), zeros(n)
    g = zeros(n + 1)
    g[1] = β
    k = 0
    for j in 1:n
        z = prec(V[j])
        push!(Z, z)
        w = Dv(z)
        for i in 1:j
            H[i, j] = sum(w .* V[i])
            w .-= H[i, j] .* V[i]
        end
        hnext = _norm2(w)
        H[j + 1, j] = hnext
        for i in 1:(j - 1)
            t = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
            H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
            H[i, j] = t
        end
        dj = hypot(H[j, j], H[j + 1, j])
        cs[j], sn[j] = dj == 0 ? (1.0, 0.0) : (H[j, j] / dj, H[j + 1, j] / dj)
        H[j, j] = dj
        H[j + 1, j] = 0.0
        g[j + 1] = -sn[j] * g[j]
        g[j] = cs[j] * g[j]
        k = j
        (abs(g[j + 1]) <= η * β || hnext == 0) && break
        push!(V, w ./ hnext)
    end
    y = zeros(k)
    for i in k:-1:1
        acc = g[i]
        for c in (i + 1):k
            acc -= H[i, c] * y[c]
        end
        y[i] = H[i, i] == 0 ? 0.0 : acc / H[i, i]
    end
    s = zeros(n)
    for i in 1:k
        s .+= y[i] .* Z[i]
    end
    return s
end

"""Eisenstat–Walker choice 2 (spec/methods.md §15.3, D126): the forcing term from the last two residual 2-norms and the previous term."""
function _forcing(nr, nr_prev, η_prev)
    nr_prev === nothing && return 0.5
    η = 0.9 * (nr / nr_prev)^2
    s = 0.9 * η_prev^2
    s > 0.1 && (η = max(η, s))
    return min(η, 0.9)
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
function solve_attempt(ops, info::AlgebraicInfo, u, z0, p, t, m::SolveMethod, l, h, ω, d = ones(length(info.dofs)))
    dofs = info.dofs
    F = _frozen(info, m)
    # The iteration works on the scaled residual D·G (§15.1, D125); the
    # convergence test on the unscaled one.
    G(z) = residual_at(ops, dofs, u, z, p, t) .* d
    conv(z, g) = _ninf(_R(z, g ./ d, l, h))
    O(z) = frozen_values(F, _with_z(u, dofs, z), p, t)[m.relchange]
    z = _proj(z0, l, h)
    g = G(z)
    r = _R(z, g, l, h)
    n = _ninf(r)
    isfinite(n) || return z, false, 0
    m.relchange == 0 && conv(z, g) <= m.tol && return z, true, 0
    o = m.relchange == 0 ? nothing : O(z)
    B = SolveBlocks(info, m)
    δ = m.dt0
    nr_prev, η_prev = nothing, 0.5
    for k in 1:m.maxiter
        A = m.kind === :picard ? frozen_jacobian(F, _with_z(u, dofs, z), length(z), p, t) :
            residual_jacobian(ops, dofs, u, z, p, t)
        for (rc, v) in A
            A[rc] = v * d[rc[1]]
        end
        if m.kind === :ptc
            # A component held at a bound by an outward rate does not move (§15.4).
            held = BitVector([(z[i] <= l[i] && g[i] <= 0) || (z[i] >= h[i] && g[i] >= 0) for i in eachindex(z)])
            b = copy(g)
            b[held] .= 0.0
            s = _linsolve(A, inv(δ), held, b, B, m.pivot)
            s === nothing && return z, false, k
            # A step that is not finite (a NaN or infinite Jacobian entry)
            # fails the attempt (§15.7).
            all(isfinite, s) || return z, false, k
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
            s = if m.inexact
                # Newton–Krylov with Eisenstat–Walker forcing (§15.3, D126).
                nr = _norm2(r)
                η = _forcing(nr, nr_prev, η_prev)
                nr_prev, η_prev = nr, η
                _krylov_step(A, held, -r, B, m, η)
            else
                _linsolve(A, 0.0, held, -r, B, m.pivot)
            end
            s === nothing && return z, false, k
            all(isfinite, s) || return z, false, k
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
            conv(z, g) <= m.tol && return z, true, k
        else
            onew = O(z)
            met = conv(z, g) <= m.tol && _rel_met(onew, o, m.rctol)
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
function run_solve(ops, info::AlgebraicInfo, u, p, t, m::SolveMethod, warm, initial, pnames; id = "")
    # Referenced bounds and scales are evaluated once per invocation (D125).
    obs = observed_values(info, _with_z(u, info.dofs, warm), p, t)
    l, h = bound_values(info, p, obs)
    d = scale_values(info, m, p, pnames, obs, id)
    any(i -> l[i] > h[i], eachindex(l)) && throw(
        ConfigError("solve: the lower bound of $(info.names[findfirst(i -> l[i] > h[i], eachindex(l))]) exceeds its upper bound")
    )
    actions = String[]
    z, ok, it = solve_attempt(ops, info, u, warm, p, t, m, l, h, 1.0, d)
    pp = copy(p)
    scaled = Dict{Int, Int}()
    for a in m.recover
        ok && break
        if a isa AbstractDict && haskey(a, "underrelax")
            push!(actions, "underrelax")
            z, ok, it = solve_attempt(ops, info, u, warm, pp, t, m, l, h, Float64(a["underrelax"]), d)
        elseif a isa AbstractDict && haskey(a, "scale_param")
            i = findfirst(==(String(a["scale_param"])), pnames)
            i === nothing && throw(ConfigError("solve: recover scale_param $(a["scale_param"]) is not a scalar parameter"))
            for _ in 1:Int(get(a, "max", 1))
                ok && break
                pp[i] *= Float64(a["by"])
                push!(actions, "scale_param")
                z, ok, it = solve_attempt(ops, info, u, warm, pp, t, m, l, h, 1.0, d)
            end
        elseif a == "revert_warm_start" || (a isa AbstractDict && haskey(a, "revert_warm_start"))
            push!(actions, "revert_warm_start")
            z, ok, it = solve_attempt(ops, info, u, initial, pp, t, m, l, h, 1.0, d)
        elseif a isa AbstractDict && haskey(a, "method")
            # An attempt with another method from the warm start (D125).
            push!(actions, "method")
            z, ok, it = solve_attempt(ops, info, u, warm, pp, t, recover_method(m, a["method"]), l, h, 1.0, d)
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
        o = run_solve(node.ops, info, u, p, ta, node.method, warm, init, run.prob.pnames; id = node.id)
        o.ok || throw(RunFailure("SolveFailure", "solve node $(node.id) ($path) did not converge at t = $ta"))
        push!(mode.tape, SolveRecord(path, ta, tb - ta, o.z, o.iterations, o.recover, active_sets(info, o)))
        out[info.dofs] = o.z
        return out
    end
    rec = tape_record(mode.tape, path, SolveRecord, ta, tb)
    z = rec.z
    if mode.trace !== nothing
        # The replay for the reverse pass: the tape's z, bit for bit (spec/adjoint.md §4.6).
        push!(mode.trace, (:solve, node, rec, Float64.(u), Float64.(_params(run, mode)), ta))
    elseif NEWTON_TIGHT[]
        # At perturbed inputs: the tape's active set held, the free rows
        # converged to roundoff (spec/adjoint.md §4.6, §7.2; D127).
        held, up = held_set(info, rec.active)
        pp, _ = recorded_scaling(node.method, rec.recover, run.prob.pnames, Float64.(_params(run, mode)))
        z = replay_held(node.ops, info, u, rec.z, pp, ta, held, up)
    end
    out[info.dofs] = z
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
    info = AlgebraicInfo(
        dofs, znames, Any[nothing for _ in dofs], Any[nothing for _ in dofs], one ? only(infos).frozen : nothing,
        one ? only(infos).blocks : nothing, nothing, Dict{String, Vector{Int}}(), reduce(vcat, (i.vjp for i in infos)), Any[]
    )
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
    p = _params(run, ctx.mode)
    uf = Float64.(u)
    z0 = uf[d.info.dofs]
    o = run_solve(d.ops, d.info, uf, p, ta, d.init, z0, run.u0[d.info.dofs], run.prob.pnames; id = node.id)
    o.ok || throw(RunFailure("InitialFailure", "the consistent initialization of $(node.id) did not converge at t = $ta"))
    z = o.z
    mode = ctx.mode
    if mode isa ReplayMode
        nz = length(z)
        if mode.trace !== nothing
            # The initialization's reverse is a §4.6 solve at the invocation's
            # start, after the leaf's steps (spec/adjoint.md §5.10, D127).
            push!(mode.trace, (:dae_init, node, z, uf, Float64.(p), ta))
        elseif NEWTON_TIGHT[]
            z = replay_held(d.ops, d.info, uf, z, Float64.(p), ta, falses(nz), falses(nz))
        end
    end
    out = copy(u)
    out[d.info.dofs] = z
    return out
end

# ---------------------------------------------------------------------------
# Adjoints (spec/adjoint.md §4.6, §5.10; D127)

"""
    ParamProduct

A parameter product document (spec/plan.md §6.6) bound to its own layout of
`n` DOFs: `src[j]` is the DOF of source-layout DOF `j` (the model's layout
for a residual document, the frozen layout for a frozen residual document),
`lam[j]` the DOF of its cotangent input (0 for a state with none), and `out`
the `(parameter index, output DOF)` of each scalar parameter of the run.
"""
struct ParamProduct
    doc::Any
    n::Int
    src::Vector{Int}
    lam::Vector{Int}
    out::Vector{Tuple{Int, Int}}
end

"""Adds `g_pᵀ w` (or the frozen residual's) to `p̄`: product `P` at the source state `x` with cotangent `w` per source DOF."""
function product!(P::ParamProduct, p̄, x, p, t, w)
    isempty(P.out) && return p̄
    U = zeros(P.n)
    U[P.src] = x
    for j in eachindex(w)
        (P.lam[j] > 0 && w[j] != 0) && (U[P.lam[j]] = w[j])
    end
    evaluate(P.doc, U, p, t) do du
        for (k, o) in P.out
            p̄[k] += du[P.doc.perm[o]]
        end
    end
    return p̄
end

"""The held positions of a solve record's active set (`active`, per unknown variable), and which of them are at their upper bound."""
function held_set(info::AlgebraicInfo, active)
    nz = length(info.dofs)
    held, up = falses(nz), falses(nz)
    first = Dict{String, Int}()
    for (i, z) in enumerate(info.names)
        haskey(first, z) || (first[z] = i)
    end
    for (z, (lo, upr, _)) in active, (set, isup) in ((lo, false), (upr, true)), e in set
        i = first[z] + e
        held[i] = true
        up[i] = isup
    end
    return held, up
end

"""The blocks of the adjoint's transposed solve: the plan's, else one block of every unknown (spec/adjoint.md §4.6)."""
function adjoint_blocks(info::AlgebraicInfo)
    nz = length(info.dofs)
    blocks = info.blocks === nothing ? [collect(1:nz)] : info.blocks
    of = fill((0, 0), nz)
    for (k, P) in enumerate(blocks), (i, q) in enumerate(P)
        of[q] = (k, i)
    end
    return SolveBlocks(blocks, of)
end

"""
The parameters a solve's residual was evaluated with: `p` with each
`scale_param` action the tape records applied (spec/adjoint.md §4.6), and
the factor per parameter.
"""
function recorded_scaling(m::SolveMethod, actions, pnames, p)
    s = ones(length(p))
    k = 1
    for a in m.recover
        k > length(actions) && break
        if a isa AbstractDict && haskey(a, "scale_param")
            i = findfirst(==(String(a["scale_param"])), pnames)
            for _ in 1:Int(get(a, "max", 1))
                (k <= length(actions) && actions[k] == "scale_param") || break
                s[i] *= Float64(a["by"])
                k += 1
            end
        else
            k += 1
        end
    end
    return p .* s, s
end

"""
    solve_adjoint(ops, info, u, z, p, t, held, up, λ, p̄; s) -> λₙ

The reverse of a solve (spec/adjoint.md §4.6, D127) at its converged `z`
with the active set `held` (`up`: at the upper bound): one transposed solve
with `g_z`'s held rows the identity's gives `−μ_I` and `b̄_A`; `λₙ` is `λ`
with the unknowns' cotangents zeroed (the warm start is not differentiated)
and `−g_uᵀμ` added on the other states, `p̄` gains `−g_pᵀμ` (times `s`, the
recorded scale of each parameter) and each referenced bound's cotangent.
"""
function solve_adjoint(ops, info::AlgebraicInfo, u, z, p, t, held, up, λ, p̄; s = nothing)
    dofs = info.dofs
    nz, N = length(dofs), length(u)
    x = _with_z(u, dofs, z)
    A = residual_jacobian(ops, dofs, x, z, p, t)
    B = adjoint_blocks(info)
    Ms = _block_matrices(A, 0.0, held, B)
    zb = Float64[λ[d] for d in dofs]
    ν = zeros(nz)
    for (k, P) in enumerate(B.blocks)
        F = lu_factor(Ms[k], true)
        issuccess_lu(F) || throw(ReplayMismatchError("solve adjoint: the free block of the residual's Jacobian is singular at t = $t"))
        ν[P] = lu_solve_transpose(F, zb[P])
    end
    w = zeros(N)
    for i in 1:nz
        held[i] || (w[dofs[i]] = ν[i])
    end
    gx = zeros(N)
    for op in ops
        vjp_u!(op, gx, x, p, t, w)
    end
    λn = Float64.(λ)
    zset = Set(dofs)
    for j in 1:N
        j in zset ? (λn[j] = 0.0) : (λn[j] += gx[j])
    end
    pc = zeros(length(p̄))
    for P in info.vjp
        product!(P, pc, x, p, t, w)
    end
    s === nothing ? (p̄ .+= pc) : (p̄ .+= pc .* s)
    fl = nothing
    for i in 1:nz
        held[i] || continue
        b = up[i] ? info.upper[i] : info.lower[i]
        b === nothing && throw(ReplayMismatchError("solve adjoint: $(info.names[i]) is recorded at a bound it does not have"))
        if b[1] === :param
            p̄[b[2]] += ν[i]
        elseif b[1] === :obs
            Fz = info.frozen
            fl === nothing && (fl = zeros(Fz.nf))
            ds = Fz.fdofs[findfirst(==(b[2]), Fz.names)]
            fl[length(ds) == 1 ? ds[1] : ds[b[3]]] += ν[i]
        end
    end
    if fl !== nothing
        xf = frozen_state_vector(info.frozen, x)
        for P in info.fvjp
            product!(P, p̄, xf, p, t, fl)
        end
    end
    return λn
end

"""
    solve_tangent(ops, info, u, z, p, t, held, up, du, dp) -> dz

The implicit-function tangent of a solve (spec/adjoint.md §4.6, D127):
`dz_A = db_A`, `g_{I,z_I} dz_I = −(g_{I,u} du + g_{I,z_A} dz_A + g_{I,p} dp)`.
`g_u du` is the residual's Jacobian applied to `du`; `g_p dp` is formed from
the parameter products, one evaluation per free unknown (and per element
at an observed bound), which suits the tests it is for.
"""
function solve_tangent(ops, info::AlgebraicInfo, u, z, p, t, held, up, du, dp)
    dofs = info.dofs
    nz, N = length(dofs), length(u)
    x = _with_z(u, dofs, z)
    dz = zeros(nz)
    for i in 1:nz
        held[i] || continue
        b = up[i] ? info.upper[i] : info.lower[i]
        b[1] === :param && (dz[i] = dp[b[2]])
        if b[1] === :obs
            # db = (∂O/∂p) dp at the element, from the frozen residual's product.
            Fz = info.frozen
            ds = Fz.fdofs[findfirst(==(b[2]), Fz.names)]
            e = zeros(Fz.nf)
            e[length(ds) == 1 ? ds[1] : ds[b[3]]] = 1.0
            row = zeros(length(p))
            for P in info.fvjp
                product!(P, row, frozen_state_vector(Fz, x), p, t, e)
            end
            dz[i] = dot(row, dp)
        end
    end
    # (−g_z with the held rows the identity's) dz = (g_u du + g_p dp on the
    # free rows, dz_A on the held ones): the free rows are
    # −g_{I,z_I} dz_I − g_{I,z_A} dz_A = g_{I,u} du + g_{I,p} dp.
    v = Float64.(du)
    v[dofs] .= 0.0
    jv = zeros(N)
    for op in ops
        jvp_u!(op, jv, x, p, t, v)
    end
    rr = Float64[jv[d] for d in dofs]
    for i in 1:nz
        held[i] && (rr[i] = dz[i]; continue)
        e = zeros(N)
        e[dofs[i]] = 1.0
        row = zeros(length(p))
        for P in info.vjp
            product!(P, row, x, p, t, e)
        end
        rr[i] += dot(row, dp)
    end
    A = residual_jacobian(ops, dofs, x, z, p, t)
    s = _linsolve(A, 0.0, held, rr, adjoint_blocks(info), true)
    s === nothing && throw(ReplayMismatchError("solve_tangent: the free block is singular"))
    return s
end

"""
The frozen-tape replay of a solve at perturbed inputs (spec/adjoint.md §4.6,
D127): the active set held (`z_A = b_A` at `p`), Newton on the free rows from
`z0`, increments applied while they shrink.
"""
function replay_held(ops, info::AlgebraicInfo, u, z0, p, t, held, up)
    dofs = info.dofs
    uf = Float64.(u)
    obs = observed_values(info, _with_z(uf, dofs, z0), p, t)
    l, h = bound_values(info, p, obs)
    z = copy(z0)
    for i in eachindex(z)
        held[i] && (z[i] = up[i] ? h[i] : l[i])
    end
    B = adjoint_blocks(info)
    prev = Inf
    for _ in 1:100
        g = residual_at(ops, dofs, uf, z, p, t)
        g[held] .= 0.0
        A = residual_jacobian(ops, dofs, uf, z, p, t)
        s = _linsolve(A, 0.0, held, g, B, true)
        s === nothing && throw(ReplayMismatchError("solve replay: the free block is singular at t = $t"))
        ns = maximum(abs, s; init = 0.0)
        ns < prev || break
        z .+= s
        prev = ns
        ns == 0 && break
    end
    return z
end
