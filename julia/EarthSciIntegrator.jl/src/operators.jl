# The operator interface (spec/operator-interface.md).
#
# Binding (operator-interface.md §5): the spec's calls are Julia methods on an
# operator type, with a `!` because they write into `out`:
#
#   spec name                  Julia method
#   rhs(out, u, p, t)          rhs!(op, out, u, p, t)
#   jvp_u(out, u, p, t, v)     jvp_u!(op, out, u, p, t, v)
#   vjp_u(out, u, p, t, w)     vjp_u!(op, out, u, p, t, w)
#   vjp_p(out, u, p, t, w)     vjp_p!(op, out, u, p, t, w)
#   jac(out, u, p, t)          jac!(op, out::JacobianStorage, u, p, t)
#   ft(out, u, p, t)           ft!(op, out, u, p, t)
#   hess_u(out, u, p, t, μ)    hess_u!(op, out, u, p, t, μ)   μ: a vector of (μᵢ, wᵢ) pairs
#   hess_p(out, u, p, t, μ)    hess_p!(op, out, u, p, t, μ)
#   ft_vjp_u(out, u, p, t, ν)  ft_vjp_u!(op, out, u, p, t, ν)
#   ft_vjp_p(out, u, p, t, ν)  ft_vjp_p!(op, out, u, p, t, ν)
#   apply(out, u, p, t, Δ)     apply!(op, out, mask, u, p, t, Δ)        (maps, §7)
#   map_vjp(...)               map_vjp!(op, out_u, out_p, u, p, t, Δ, mask, w)
#   map_jvp(...)               map_jvp!(op, out, u, p, t, Δ, mask, v_u, v_p)
#
# Every call except `apply!` accumulates into `out` (§3). The methods are
# generic over the element types of `u` and `p`, so `ForwardDiff.Dual` numbers
# can be pushed through them for the frozen-tape tangent (spec/adjoint.md §2.3).

"""Supertype of term operators: an additive contribution `f_k(u, p, t)` to a leaf's right-hand side."""
abstract type AbstractOperator end

"""Supertype of map operators (operator-interface.md §7)."""
abstract type AbstractMapOperator end

"""Layout and Jacobian structure of an operator or a leaf (operator-interface.md §2)."""
abstract type Structure end

"""No structure promised; DOF order is the operator set's own."""
struct DenseStructure <: Structure end

"""
    BlockDiagonalStructure(ncells, n)

`ncells` independent cells of `n` DOFs each, species-major: DOF `i` of cell
`c` (both 0-based) is at 0-based index `i*ncells + c`.
"""
struct BlockDiagonalStructure <: Structure
    ncells::Int
    n::Int
end

Base.:(==)(a::BlockDiagonalStructure, b::BlockDiagonalStructure) = a.ncells == b.ncells && a.n == b.n

"""The coarsest structure every structure in `ss` shares (operator-interface.md §2)."""
function leaf_structure(ss)
    s1 = first(ss)
    s1 isa BlockDiagonalStructure && all(s -> s == s1, ss) && return s1
    return DenseStructure()
end

# Declarations every operator makes (§3). Defaults are the conservative ones.
"""`structure(op)`: the operator's [`Structure`](@ref)."""
function structure end
"""`depends_on_t(op)`: whether `f_k` depends on `t` explicitly."""
depends_on_t(::AbstractOperator) = true
"""`linear_in_u(op)`: whether `J_k` is independent of `u`."""
linear_in_u(::AbstractOperator) = false
"""
`writes(op, N)`: the 1-based indices of the DOFs of an `N`-vector state that the
operator writes (its `f_k` can be nonzero there). Default: all of them. A leaf's
error norm and first step sum over the union of its operators' (spec/methods.md §7).
"""
writes(::AbstractOperator, N) = collect(1:N)
"""`opname(op)`: the operator's name, as a run config's schedule names it."""
function opname end

function rhs! end
function jvp_u! end
function vjp_u! end
function vjp_p! end
function jac! end
function ft! end
function hess_u! end
function hess_p! end
function ft_vjp_u! end
function ft_vjp_p! end
function apply! end
function map_vjp! end
function map_jvp! end

"""
    provides(op, call::Symbol) -> Bool

Whether `op` implements `call` (one of `:rhs, :jvp_u, :vjp_u, :vjp_p, :jac,
:ft, :hess_u, :hess_p, :ft_vjp_u, :ft_vjp_p`). A leaf whose method or adjoint
needs a call its operators do not provide is refused when the leaf is built
(operator-interface.md §3).
"""
function provides(op::AbstractOperator, call::Symbol)
    f = getfield(@__MODULE__, Symbol(call, "!"))
    nargs = call in (:rhs, :jac, :ft) ? 4 : 5
    return hasmethod(f, Tuple{typeof(op), ntuple(_ -> Any, nargs)...})
end

# ---------------------------------------------------------------------------
# Matrix storage (operator-interface.md §4)

"""Storage that `jac!` accumulates into; the leaf's linear solver owns it."""
abstract type JacobianStorage end

"""An `N × N` column-major matrix (toy problems and tests only)."""
struct DenseJacobian{T} <: JacobianStorage
    A::Matrix{T}
end
DenseJacobian{T}(N::Integer) where {T} = DenseJacobian{T}(zeros(T, N, N))

"""
    BlockJacobian(ncells, n)

`ncells` dense `n × n` blocks stored lane-innermost: entry `(i, j)` (0-based) of
cell `c` is at 0-based offset `(i*n + j)*ncells + c`.
"""
struct BlockJacobian{T} <: JacobianStorage
    ncells::Int
    n::Int
    data::Vector{T}
end
BlockJacobian{T}(ncells::Integer, n::Integer) where {T} = BlockJacobian{T}(ncells, n, zeros(T, n * n * ncells))

"""
    addentry!(S, i, j, v)

Accumulate `v` into Jacobian entry `(i, j)`, with `i` and `j` 1-based indices
into the leaf's state vector. For [`BlockJacobian`](@ref) both must lie in the
same cell.
"""
@inline addentry!(S::DenseJacobian, i::Integer, j::Integer, v) = (S.A[i, j] += v; S)
@inline function addentry!(S::BlockJacobian, i::Integer, j::Integer, v)
    si, ci = divrem(i - 1, S.ncells)
    sj, cj = divrem(j - 1, S.ncells)
    ci == cj || throw(ArgumentError("entry ($i, $j) couples cells $ci and $cj of a block-diagonal Jacobian"))
    S.data[(si * S.n + sj) * S.ncells + ci + 1] += v
    return S
end

"""Dense `N × N` copy of a Jacobian storage, for tests."""
todense(S::DenseJacobian) = copy(S.A)
function todense(S::BlockJacobian{T}) where {T}
    N = S.ncells * S.n
    A = zeros(T, N, N)
    for c in 0:(S.ncells - 1), i in 0:(S.n - 1), j in 0:(S.n - 1)
        A[i * S.ncells + c + 1, j * S.ncells + c + 1] = S.data[(i * S.n + j) * S.ncells + c + 1]
    end
    return A
end

"""The matrix storage a leaf of this structure owns (operator-interface.md §4)."""
jacobian_storage(::DenseStructure, ::Type{T}, N) where {T} = DenseJacobian{T}(N)
jacobian_storage(s::BlockDiagonalStructure, ::Type{T}, N) where {T} = BlockJacobian{T}(s.ncells, s.n)

# ---------------------------------------------------------------------------
# A leaf's sum of operators. The leaf zeroes an output once and calls each
# operator in its declared order (operator-interface.md §3), which makes the sum
# bitwise reproducible (D10).

"""
    OperatorSum(ops)

The sum `f = Σ_k f_k` of a leaf's operators, in declared order.
"""
struct OperatorSum{O <: Tuple}
    ops::O
end
OperatorSum(ops::AbstractVector) = OperatorSum(Tuple(ops))

structure(s::OperatorSum) = leaf_structure(map(structure, s.ops))
writes(s::OperatorSum, N) = sort!(unique!(reduce(vcat, (writes(op, N) for op in s.ops); init = Int[])))
depends_on_t(s::OperatorSum) = any(depends_on_t, s.ops)
linear_in_u(s::OperatorSum) = all(linear_in_u, s.ops)

for call in (:rhs!, :ft!, :jac!)
    @eval function $call(s::OperatorSum, out, u, p, t)
        for op in s.ops
            $call(op, out, u, p, t)
        end
        return out
    end
end
for call in (:jvp_u!, :vjp_u!, :vjp_p!, :hess_u!, :hess_p!, :ft_vjp_u!, :ft_vjp_p!)
    @eval function $call(s::OperatorSum, out, u, p, t, x)
        for op in s.ops
            $call(op, out, u, p, t, x)
        end
        return out
    end
end

"""`f!(out, s, u, p, t)`: overwrite `out` with the leaf's right-hand side."""
function eval_rhs!(out, s::OperatorSum, u, p, t)
    fill!(out, zero(eltype(out)))
    return rhs!(s, out, u, p, t)
end
