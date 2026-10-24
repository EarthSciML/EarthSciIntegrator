# Linear solves with W (spec/methods.md §4.2, operator-interface.md §4,
# run-config.md §4.2 `linsolve`): `dense_lu`, an LU of the whole N × N system
# (toy problems and tests only), and `block_lu`, a batched LU of the
# block-diagonal layout, every operation applied across all lanes (cells) at
# a time with the lane index innermost.
#
# Arithmetic, fixed so the two implementations can reproduce it:
# - W is formed from J as `W_ii = α - J_ii`, `W_ij = -J_ij` (i ≠ j), with
#   α = 1/(hγ) for the transformed Rosenbrock form (Ŵ = I/(hγ) - J) and
#   α = 1/(h d) for Rosenbrock23 (whose solves are then scaled by 1/(h d),
#   rosenbrock.jl). This is OrdinaryDiffEq's `W = J - I/(hγ)` negated, which
#   is exact.
# - Factorization: Julia's `LinearAlgebra.generic_lufact!` (the same
#   arithmetic as the Rust implementation): right-looking (k, then rows i > k,
#   then columns j > k) Gaussian elimination, `l_ik = a_ik · (1/a_kk)`,
#   `a_ij ← a_ij - l_ik·a_kj`. Each
#   entry's update sequence over k is fixed, so the order of the i, j loops
#   does not change the result. With pivoting, row k is swapped with the first
#   row p ≥ k whose |a_pk| is largest (ties to the lowest index) before the
#   elimination of column k. A pivot that is zero or NaN makes W singular
#   (spec/methods.md §4.2); for `block_lu` one singular lane makes W singular.
# - Solves are column oriented: the row swaps applied in order k = 1..N; then
#   forward substitution over ascending columns j, `y_i ← y_i - l_ij y_j` for
#   i > j; then back substitution over descending columns j, `x_j = y_j / u_jj`
#   and `y_i ← y_i - u_ij x_j` for i < j. So `y_i` subtracts its terms in
#   ascending j and `x_i` in descending j. The transposed solve (`solve_Wt`,
#   spec/adjoint.md §3) with the same factors: Uᵀ by ascending columns,
#   `w_j = b_j / u_jj` and `b_i ← b_i - u_ji w_j` for i > j; Lᵀ by
#   descending columns, `z_i ← z_i - l_ji z_j` for i < j; then the swaps
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

"""`W = α I - J` as a new dense matrix (see the file header for the arithmetic)."""
function form_W(J::DenseJacobian, α)
    A = J.A
    N = size(A, 1)
    T = promote_type(eltype(A), typeof(α))
    W = Matrix{T}(undef, N, N)
    @inbounds for j in 1:N, i in 1:N
        W[i, j] = i == j ? α - A[i, j] : -A[i, j]
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
        pinv = inv(piv)
        for i in (k + 1):N
            W[i, k] = W[i, k] * pinv
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
    @inbounds for j in 1:N, i in (j + 1):N
        x[i] = x[i] - A[i, j] * x[j]
    end
    @inbounds for j in N:-1:1
        x[j] = x[j] / A[j, j]
        for i in 1:(j - 1)
            x[i] = x[i] - A[i, j] * x[j]
        end
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
    @inbounds for j in 1:N
        x[j] = x[j] / A[j, j]
        for i in (j + 1):N
            x[i] = x[i] - A[j, i] * x[j]
        end
    end
    @inbounds for j in N:-1:2, i in 1:(j - 1)
        x[i] = x[i] - A[j, i] * x[j]
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

function form_W(J::BlockJacobian, α)
    nc, n = J.ncells, J.n
    T = promote_type(eltype(J.data), typeof(α))
    W = Vector{T}(undef, length(J.data))
    @inbounds for i in 0:(n - 1), j in 0:(n - 1)
        base = (i * n + j) * nc
        for c in 1:nc
            W[base + c] = i == j ? α - J.data[base + c] : -J.data[base + c]
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
        pinv = [inv(a[at(k, k) + c]) for c in 1:nc]
        for i in (k + 1):(n - 1)
            for c in 1:nc
                a[at(i, k) + c] = a[at(i, k) + c] * pinv[c]
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
    @inbounds for j in 0:(n - 2), i in (j + 1):(n - 1), c in 1:nc
        x[i * nc + c] = x[i * nc + c] - a[at(i, j) + c] * x[j * nc + c]
    end
    @inbounds for j in (n - 1):-1:0
        for c in 1:nc
            x[j * nc + c] = x[j * nc + c] / a[at(j, j) + c]
        end
        for i in 0:(j - 1), c in 1:nc
            x[i * nc + c] = x[i * nc + c] - a[at(i, j) + c] * x[j * nc + c]
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
    @inbounds for j in 0:(n - 1)
        for c in 1:nc
            x[j * nc + c] = x[j * nc + c] / a[at(j, j) + c]
        end
        for i in (j + 1):(n - 1), c in 1:nc
            x[i * nc + c] = x[i * nc + c] - a[at(j, i) + c] * x[j * nc + c]
        end
    end
    @inbounds for j in (n - 1):-1:1, i in 0:(j - 1), c in 1:nc
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
    factor_W(J, α, ls) -> factors

Form `W = α I - J` from the Jacobian storage and factor it with the leaf's
linear solver. Check `issuccess_lu` before solving.
"""
factor_W(J::Union{DenseJacobian, BlockJacobian}, α, ls::LinsolveSpec) = lu_factor(form_W(J, α), ls.pivot)
