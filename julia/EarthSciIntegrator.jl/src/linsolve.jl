# Linear solves with W (spec/methods.md §4.2, operator-interface.md §4,
# run-config.md §4.2 `linsolve`): `dense_lu`, an LU of the whole N × N system
# (toy problems and tests only), and `block_lu`, a batched LU of the
# block-diagonal layout, every operation applied across all lanes (cells) at
# a time with the lane index innermost.
#
# Arithmetic, fixed so the two implementations can reproduce it:
# - W is formed from J as `W_ii = α - s·J_ii`, `W_ij = -(s·J_ij)` (i ≠ j), with
#   `(α, s) = (1/(hγ), 1)` for the transformed Rosenbrock form (Ŵ = I/(hγ) - J)
#   and `(1, h·d)` for Rosenbrock23 (Ŵ = I - h d J). For the transformed form
#   this is OrdinaryDiffEq's `W = J - I/(hγ)` negated, which is exact.
# - Factorization: right-looking (k, then rows i > k, then columns j > k)
#   Gaussian elimination, `l_ik = a_ik / a_kk`, `a_ij ← a_ij - l_ik·a_kj`. Each
#   entry's update sequence over k is fixed, so the order of the i, j loops
#   does not change the result. With pivoting, row k is swapped with the first
#   row p ≥ k whose |a_pk| is largest (ties to the lowest index) before the
#   elimination of column k. A pivot that is zero or NaN makes W singular
#   (spec/methods.md §4.2); for `block_lu` one singular lane makes W singular.
# - Solves: the row swaps applied in order k = 1..N, then forward substitution
#   `y_i = b_i - Σ_{j<i} l_ij y_j` and back substitution
#   `x_i = (y_i - Σ_{j>i} u_ij x_j) / u_ii`, each sum subtracted term by term
#   in ascending j from the leading b_i or y_i. The transposed solve
#   (`solve_Wt`, spec/adjoint.md §3) uses the same factors:
#   `w_i = (b_i - Σ_{j<i} u_ji w_j) / u_ii` ascending, then
#   `z_i = w_i - Σ_{j>i} l_ji z_j` descending i, ascending j, then the swaps
#   undone in order k = N..1.
# - No fused multiply-adds. Generic over the element type, so `Dual` numbers
#   pass through for the frozen-tape tangent; pivots are chosen on primal values.

"""Which linear solver a leaf uses for W, and its options."""
struct LinsolveSpec
    kind::Symbol      # :dense_lu or :block_lu
    pivot::Bool
end

"""
    parse_linsolve(x, structure, id) -> LinsolveSpec

The leaf's `linsolve` option. `"auto"` (or absent) is `block_lu` on a
block-diagonal leaf and `dense_lu` otherwise. `block_lu.pivot` defaults to
`true` (spec/run-config.md does not give a default; see the stage report).
"""
function parse_linsolve(x, s::Structure, id)
    kind, opts = if x === nothing || x == "auto"
        (s isa BlockDiagonalStructure ? "block_lu" : "dense_lu"), Dict{String, Any}()
    elseif x isa AbstractString
        String(x), Dict{String, Any}()
    elseif x isa AbstractDict && length(x) == 1
        k, v = only(x)
        String(k), v
    else
        throw(ConfigError("leaf $id: linsolve $(repr(x))"))
    end
    if kind == "dense_lu"
        isempty(opts) || throw(ConfigError("leaf $id: dense_lu takes no options"))
        return LinsolveSpec(:dense_lu, true)
    elseif kind == "block_lu"
        s isa BlockDiagonalStructure ||
            throw(ConfigError("leaf $id: linsolve block_lu needs a block_diagonal leaf (operator-interface.md §2); its operators' structure is dense"))
        for k in keys(opts)
            k == "pivot" || throw(UnsupportedError("leaf $id: block_lu option '$k' is not implemented in this stage"))
        end
        return LinsolveSpec(:block_lu, Bool(get(opts, "pivot", true)))
    else
        throw(UnsupportedError("leaf $id: linsolve '$kind' is not implemented in this stage (dense_lu and block_lu are)"))
    end
end

"""The Jacobian storage the leaf's linear solver owns (operator-interface.md §4)."""
jacobian_storage(ls::LinsolveSpec, s::Structure, ::Type{T}, N) where {T} =
    ls.kind === :block_lu ? BlockJacobian{T}(s.ncells, s.n) : DenseJacobian{T}(N)

_singular_pivot(x) = (v = primal(x); iszero(v) || isnan(v))

# ---------------------------------------------------------------------------
# Dense

"""LU factors of a dense W with row swaps `perm` (row k was swapped with row perm[k])."""
struct DenseLU{T}
    F::Matrix{T}
    perm::Vector{Int}
    ok::Bool
end

"""`W = α I - s J` as a new dense matrix (see the file header for the arithmetic)."""
function form_W(J::DenseJacobian, α, s)
    A = J.A
    N = size(A, 1)
    T = promote_type(eltype(A), typeof(α), typeof(s))
    W = Matrix{T}(undef, N, N)
    @inbounds for j in 1:N, i in 1:N
        sj = s * A[i, j]
        W[i, j] = i == j ? α - sj : -sj
    end
    return W
end

function lu_factor(W::Matrix{T}, pivot::Bool) where {T}
    N = size(W, 1)
    perm = collect(1:N)
    @inbounds for k in 1:N
        if pivot
            p = k
            best = abs(primal(W[k, k]))
            for i in (k + 1):N
                a = abs(primal(W[i, k]))
                if a > best
                    best, p = a, i
                end
            end
            perm[k] = p
            if p != k
                for j in 1:N
                    W[k, j], W[p, j] = W[p, j], W[k, j]
                end
            end
        end
        piv = W[k, k]
        _singular_pivot(piv) && return DenseLU{T}(W, perm, false)
        for i in (k + 1):N
            W[i, k] = W[i, k] / piv
        end
        for j in (k + 1):N
            akj = W[k, j]
            for i in (k + 1):N
                W[i, j] = W[i, j] - W[i, k] * akj
            end
        end
    end
    return DenseLU{T}(W, perm, true)
end

"""`W⁻¹ b` (`solve_W`)."""
function lu_solve(F::DenseLU, b::AbstractVector)
    A = F.F
    N = length(b)
    T = promote_type(eltype(A), eltype(b))
    x = Vector{T}(undef, N)
    copyto!(x, b)
    @inbounds for k in 1:N
        p = F.perm[k]
        p != k && ((x[k], x[p]) = (x[p], x[k]))
    end
    @inbounds for i in 2:N
        acc = x[i]
        for j in 1:(i - 1)
            acc = acc - A[i, j] * x[j]
        end
        x[i] = acc
    end
    @inbounds for i in N:-1:1
        acc = x[i]
        for j in (i + 1):N
            acc = acc - A[i, j] * x[j]
        end
        x[i] = acc / A[i, i]
    end
    return x
end

"""`W⁻ᵀ b` (`solve_Wt`) with the same factors."""
function lu_solve_transpose(F::DenseLU, b::AbstractVector)
    A = F.F
    N = length(b)
    T = promote_type(eltype(A), eltype(b))
    x = Vector{T}(undef, N)
    copyto!(x, b)
    @inbounds for i in 1:N
        acc = x[i]
        for j in 1:(i - 1)
            acc = acc - A[j, i] * x[j]
        end
        x[i] = acc / A[i, i]
    end
    @inbounds for i in (N - 1):-1:1
        acc = x[i]
        for j in (i + 1):N
            acc = acc - A[j, i] * x[j]
        end
        x[i] = acc
    end
    @inbounds for k in N:-1:1
        p = F.perm[k]
        p != k && ((x[k], x[p]) = (x[p], x[k]))
    end
    return x
end

# ---------------------------------------------------------------------------
# Block diagonal, lane-innermost: entry (i, j) (0-based) of cell c is at
# 0-based offset (i*n + j)*ncells + c; DOF i of cell c is at i*ncells + c.

"""Batched LU factors of a block-diagonal W, `perm[k*ncells + c + 1]` the row swapped with row k in cell c."""
struct BlockLU{T}
    ncells::Int
    n::Int
    F::Vector{T}
    perm::Vector{Int}
    ok::Bool
end

function form_W(J::BlockJacobian, α, s)
    nc, n = J.ncells, J.n
    T = promote_type(eltype(J.data), typeof(α), typeof(s))
    W = Vector{T}(undef, length(J.data))
    @inbounds for i in 0:(n - 1), j in 0:(n - 1)
        base = (i * n + j) * nc
        for c in 1:nc
            sj = s * J.data[base + c]
            W[base + c] = i == j ? α - sj : -sj
        end
    end
    return BlockJacobian{T}(nc, n, W)
end

function lu_factor(W::BlockJacobian{T}, pivot::Bool) where {T}
    nc, n = W.ncells, W.n
    a = W.data
    at(i, j) = (i * n + j) * nc          # 0-based (i, j) → offset of lane 0, minus 1
    perm = Vector{Int}(undef, n * nc)
    ok = true
    @inbounds for k in 0:(n - 1)
        if pivot
            for c in 1:nc
                p = k
                best = abs(primal(a[at(k, k) + c]))
                for i in (k + 1):(n - 1)
                    v = abs(primal(a[at(i, k) + c]))
                    if v > best
                        best, p = v, i
                    end
                end
                perm[k * nc + c] = p
                if p != k
                    for j in 0:(n - 1)
                        a[at(k, j) + c], a[at(p, j) + c] = a[at(p, j) + c], a[at(k, j) + c]
                    end
                end
            end
        else
            for c in 1:nc
                perm[k * nc + c] = k
            end
        end
        for c in 1:nc
            _singular_pivot(a[at(k, k) + c]) && (ok = false)
        end
        ok || return BlockLU{T}(nc, n, a, perm, false)
        for i in (k + 1):(n - 1)
            for c in 1:nc
                a[at(i, k) + c] = a[at(i, k) + c] / a[at(k, k) + c]
            end
            for j in (k + 1):(n - 1)
                for c in 1:nc
                    a[at(i, j) + c] = a[at(i, j) + c] - a[at(i, k) + c] * a[at(k, j) + c]
                end
            end
        end
    end
    return BlockLU{T}(nc, n, a, perm, true)
end

function lu_solve(F::BlockLU, b::AbstractVector)
    nc, n, a = F.ncells, F.n, F.F
    at(i, j) = (i * n + j) * nc
    T = promote_type(eltype(a), eltype(b))
    x = Vector{T}(undef, length(b))
    copyto!(x, b)
    @inbounds for k in 0:(n - 1), c in 1:nc
        p = F.perm[k * nc + c]
        p != k && ((x[k * nc + c], x[p * nc + c]) = (x[p * nc + c], x[k * nc + c]))
    end
    @inbounds for i in 1:(n - 1), j in 0:(i - 1), c in 1:nc
        x[i * nc + c] = x[i * nc + c] - a[at(i, j) + c] * x[j * nc + c]
    end
    @inbounds for i in (n - 1):-1:0
        for j in (i + 1):(n - 1), c in 1:nc
            x[i * nc + c] = x[i * nc + c] - a[at(i, j) + c] * x[j * nc + c]
        end
        for c in 1:nc
            x[i * nc + c] = x[i * nc + c] / a[at(i, i) + c]
        end
    end
    return x
end

function lu_solve_transpose(F::BlockLU, b::AbstractVector)
    nc, n, a = F.ncells, F.n, F.F
    at(i, j) = (i * n + j) * nc
    T = promote_type(eltype(a), eltype(b))
    x = Vector{T}(undef, length(b))
    copyto!(x, b)
    @inbounds for i in 0:(n - 1)
        for j in 0:(i - 1), c in 1:nc
            x[i * nc + c] = x[i * nc + c] - a[at(j, i) + c] * x[j * nc + c]
        end
        for c in 1:nc
            x[i * nc + c] = x[i * nc + c] / a[at(i, i) + c]
        end
    end
    @inbounds for i in (n - 2):-1:0, j in (i + 1):(n - 1), c in 1:nc
        x[i * nc + c] = x[i * nc + c] - a[at(j, i) + c] * x[j * nc + c]
    end
    @inbounds for k in (n - 1):-1:0, c in 1:nc
        p = F.perm[k * nc + c]
        p != k && ((x[k * nc + c], x[p * nc + c]) = (x[p * nc + c], x[k * nc + c]))
    end
    return x
end

issuccess_lu(F::Union{DenseLU, BlockLU}) = F.ok

"""
    factor_W(J, α, s, ls) -> factors

Form `W = α I - s J` from the Jacobian storage and factor it with the leaf's
linear solver. Check `issuccess_lu` before solving.
"""
function factor_W(J::DenseJacobian, α, s, ls::LinsolveSpec)
    return lu_factor(form_W(J, α, s), ls.pivot)
end
function factor_W(J::BlockJacobian, α, s, ls::LinsolveSpec)
    return lu_factor(form_W(J, α, s), ls.pivot)
end
