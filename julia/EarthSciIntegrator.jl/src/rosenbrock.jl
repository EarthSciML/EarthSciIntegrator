# The Rosenbrock family (spec/methods.md §4, spec/adjoint.md §4.2).
#
# Arithmetic of the transformed form (ROS2, ROS3P, Rodas3/4/5P, the KPP
# methods), following OrdinaryDiffEqRosenbrock 2.7.5's in-place generic step
# (`src/rosenbrock_perform_step.jl:734` onward, `_stage_accum!` and
# `_stage_rhs!` at :616-664):
# - J and f_t at (uₙ, tₙ); Ŵ = I/(hγ) - J with 1/(hγ) = `inv(h*γ)` (linsolve.jl).
# - Uᵢ = uₙ + A_i1 k₁ + A_i2 k₂ + …, a flat left-to-right sum from uₙ.
# - rᵢ = f(Uᵢ, tₙ + cᵢh) + (h dᵢ) f_t + (C_i1/h) k₁ + (C_i2/h) k₂ + …, a flat
#   left-to-right sum; the coefficient of kⱼ is `C[i,j] / h` (OrdinaryDiffEq's
#   `dtC = C ./ dt`), that of f_t is `h * d[i]` (`dtd = dt .* d`). The f_t
#   term is present only when f_t is formed (spec/methods.md §4.2).
# - kᵢ = Ŵ⁻¹ rᵢ; u_{n+1} = uₙ + b₁k₁ + b₂k₂ + …; e = b̃₁k₁ + b̃₂k₂ + … starting
#   from the first non-zero term.
# - Zero coefficients skip their term everywhere (as spec/methods.md §3.1 says
#   for explicit methods). OrdinaryDiffEq multiplies zeros through in its stage
#   sums; the values agree except for the sign of a zero.
# - A stage whose row of A and cᵢ equal the previous stage's reuses its f value.
# *Deviation:* OrdinaryDiffEq's out-of-place step groups rᵢ as
# `(f + dtd*dT) + (Σⱼ dtC*kⱼ)` instead (`:487-497`), and it writes everything
# under `@muladd`; both agree with this form to roundoff only.
#
# Rosenbrock23 (the ode23s step, spec/adjoint.md §4.2 and spec/methods.md §4.3)
# with dtγ = h·d:
#   Ŵ = I - dtγ J,  k₁ = Ŵ⁻¹(f₀ + dtγ f_t),  U₂ = uₙ + (h/2) k₁,
#   f₁ = f(U₂, tₙ + h/2),  k₂ = Ŵ⁻¹(f₁ - k₁) + k₁,  u_{n+1} = uₙ + h k₂,
#   f₂ = f(u_{n+1}, tₙ + h),
#   k₃ = Ŵ⁻¹(((f₂ - c₃₂(k₂ - f₁)) - 2(k₁ - f₀)) + h f_t),
#   e = (h/6)((k₁ - 2k₂) + k₃),
# with h/2 and h/6 formed as `h / 2` and `h / 6` (OrdinaryDiffEq's `dto2`, `dto6`).
# *Deviation:* OrdinaryDiffEq solves with W = J - I/dtγ and multiplies the
# result by -1/dtγ (`:34-145`), which equals Ŵ⁻¹ to roundoff.

"""A Rosenbrock method: the transformed form or Rosenbrock23's ode23s form."""
struct RosenbrockMethod <: AbstractMethod
    tab::Tableau
    form::Symbol                 # :transformed or :rosenbrock23
    s::Int
    A::Matrix{Float64}
    C::Matrix{Float64}
    b::Vector{Float64}
    btilde::Vector{Float64}
    c::Vector{Float64}
    d::Vector{Float64}           # for Rosenbrock23, d[1] is the scalar d
    gamma::Float64
    c32::Float64
    reuse_f::Vector{Bool}        # stage i reuses stage i-1's f value
    uses_ft::Bool                # some dᵢ ≠ 0
end

function RosenbrockMethod(tab::Tableau)
    if tab.form === :rosenbrock23
        d = something(tab.d)::Float64
        return RosenbrockMethod(tab, :rosenbrock23, 2, zeros(0, 0), zeros(0, 0), Float64[], Float64[],
            [0.0, 0.5], [d], d, something(tab.c32), [false, false], d != 0)
    end
    A, C, b, c = something(tab.A), something(tab.C), something(tab.b), something(tab.c)
    d = something(tab.d)::Vector{Float64}
    bt = tab.btilde === nothing ? zeros(length(b)) : tab.btilde
    s = length(b)
    reuse = [i > 1 && c[i] == c[i - 1] && all(j -> A[i, j] == A[i - 1, j], 1:(i - 1)) for i in 1:s]
    return RosenbrockMethod(tab, :transformed, s, A, C, b, bt, c, d, something(tab.gamma), NaN, reuse,
        any(!iszero, d))
end

# `acc .= acc + coef * x` with the coefficient converted to the element type.
@inline function _addto!(acc, coef, x)
    ct = convert(eltype(acc), coef)
    @inbounds for k in eachindex(acc)
        acc[k] = acc[k] + ct * x[k]
    end
    return acc
end

"""The context a Rosenbrock step needs: the leaf's operators, linear solver, and whether f_t is formed."""
struct RosenbrockContext{O}
    ops::O
    ls::LinsolveSpec
    structure::Structure
    depends_on_t::Bool
    check_stages::Vector{Int}    # DOFs whose stage values must be ≥ 0 (domain.check_stages), else empty
end

"""Whether f_t is formed for this method and leaf (spec/methods.md §4.2)."""
forms_ft(m::RosenbrockMethod, ctx::RosenbrockContext) = m.uses_ft && ctx.depends_on_t

"""
    rosenbrock_linearization(m, ctx, u, p, t, h) -> (J, ft, F)

J(uₙ) and f_t(uₙ) at `(u, p, t)` and the factors of Ŵ (spec/methods.md §4.2).
`ft` is `nothing` when it is not formed.
"""
function rosenbrock_linearization(m::RosenbrockMethod, ctx::RosenbrockContext, u, p, t, h)
    T = promote_type(eltype(u), eltype(p))
    N = length(u)
    J = jacobian_storage(ctx.ls, ctx.structure, T, N)
    jac!(ctx.ops, J, u, p, t)
    ft = nothing
    if forms_ft(m, ctx)
        ft = zeros(T, N)
        ft!(ctx.ops, ft, u, p, t)
    end
    if m.form === :rosenbrock23
        F = factor_W(J, one(T), h * m.d[1], ctx.ls)
    else
        F = factor_W(J, inv(h * m.gamma), one(T), ctx.ls)
    end
    return J, ft, F
end

_stage_violation(ctx::RosenbrockContext, U) = any(j -> U[j] < 0, ctx.check_stages)

"""
    rosenbrock_stages(m, ctx, u, p, t, h, k1, tk1) -> NamedTuple

Every stage of one step from `(u, t)` (generic over element types): stage
states `U`, values `f`, increments `k`, the end state, the linearization, the
number of f evaluations, and whether W was singular or a stage state violated
`domain.check_stages`. For Rosenbrock23 `k1` is a valid stored f(uₙ) (first
same as last, evaluated at `tk1`) or `nothing`.
"""
function rosenbrock_stages(m::RosenbrockMethod, ctx::RosenbrockContext, u, p, t, h, k1 = nothing; tk1 = t)
    T = promote_type(eltype(u), eltype(p))
    J, ft, F = rosenbrock_linearization(m, ctx, u, p, t, h)
    N = length(u)
    nf = 0
    if !issuccess_lu(F)
        return (; singular = true, violation = false, nf, J, ft, F, U = nothing, fv = nothing, k = nothing, u1 = nothing)
    end
    feval(x, τ) = (out = zeros(T, N); rhs!(ctx.ops, out, x, p, τ); out)
    violation = false
    if m.form === :rosenbrock23
        dtγ = h * m.d[1]
        f0 = k1 === nothing ? (nf += 1; feval(u, tk1)) : k1
        r1 = copy(f0)
        ft === nothing || _addto!(r1, dtγ, ft)
        kk1 = lu_solve(F, r1)
        U2 = convert(Vector{T}, copy(u))
        _addto!(U2, h / 2, kk1)
        violation |= _stage_violation(ctx, u) || _stage_violation(ctx, U2)
        f1 = feval(U2, t + h / 2)
        nf += 1
        r2 = f1 .- kk1
        kk2 = lu_solve(F, r2)
        @inbounds for j in eachindex(kk2)
            kk2[j] = kk2[j] + kk1[j]
        end
        u1 = convert(Vector{T}, copy(u))
        _addto!(u1, h, kk2)
        return (; singular = false, violation, nf, J, ft, F, U = [convert(Vector{T}, u), U2], fv = [f0, f1],
            k = [kk1, kk2], u1)
    end
    s = m.s
    U = Vector{Vector{T}}(undef, s)
    fv = Vector{Vector{T}}(undef, s)
    k = Vector{Vector{T}}(undef, s)
    for i in 1:s
        Ui = convert(Vector{T}, copy(u))
        for j in 1:(i - 1)
            a = m.A[i, j]
            a == 0 || _addto!(Ui, a, k[j])
        end
        U[i] = Ui
        violation |= _stage_violation(ctx, Ui)
        if m.reuse_f[i]
            fv[i] = fv[i - 1]
        else
            fv[i] = feval(Ui, t + m.c[i] * h)
            nf += 1
        end
        r = copy(fv[i])
        ft === nothing || m.d[i] == 0 || _addto!(r, h * m.d[i], ft)
        for j in 1:(i - 1)
            cij = m.C[i, j]
            cij == 0 || _addto!(r, cij / h, k[j])
        end
        k[i] = lu_solve(F, r)
    end
    u1 = convert(Vector{T}, copy(u))
    for i in 1:s
        m.b[i] == 0 || _addto!(u1, m.b[i], k[i])
    end
    return (; singular = false, violation, nf, J, ft, F, U, fv, k, u1)
end

"""
    rosenbrock_step(m, ctx, u, p, t, h, k1, adaptive, tol; tk1) -> StepResult

One forward step attempt. A singular W gives `E = 2` (spec/methods.md §4.2)
and the unchanged state; the caller distinguishes it by `singular`.
"""
function rosenbrock_step(m::RosenbrockMethod, ctx::RosenbrockContext, u, p, t, h, k1, adaptive::Bool, tol; tk1 = t)
    st = rosenbrock_stages(m, ctx, u, p, t, h, k1; tk1)
    if st.singular
        return StepResult(u, 2.0, nothing, st.nf, true, false)
    end
    nf = st.nf
    u1 = st.u1
    E = NaN
    fsal = nothing
    if m.form === :rosenbrock23
        # f(u_{n+1}, tₙ + h): the error estimate's f₂ and the next step's f₀ (§3.3).
        f2 = zeros(eltype(u1), length(u1))
        rhs!(ctx.ops, f2, u1, p, t + h)
        nf += 1
        fsal = f2
        if adaptive
            kk1, kk2 = st.k
            f0, f1 = st.fv
            c32 = m.c32
            r3 = similar(f2)
            @inbounds for j in eachindex(r3)
                r3[j] = (f2[j] - c32 * (kk2[j] - f1[j])) - 2 * (kk1[j] - f0[j])
            end
            st.ft === nothing || _addto!(r3, h, st.ft)
            kk3 = lu_solve(st.F, r3)
            dto6 = h / 6
            e = similar(f2)
            @inbounds for j in eachindex(e)
                e[j] = dto6 * ((kk1[j] - 2 * kk2[j]) + kk3[j])
            end
            E = error_norm(e, u, u1, tol)
        end
    elseif adaptive
        m.tab.error_estimate === :embedded ||
            throw(ConfigError("method $(m.tab.name) has no error estimate; give the leaf a dt (D52)"))
        e = nothing
        for i in 1:m.s
            bt = m.btilde[i]
            bt == 0 && continue
            if e === nothing
                e = convert(eltype(u1), bt) .* st.k[i]
            else
                _addto!(e, bt, st.k[i])
            end
        end
        E = e === nothing ? 0.0 : error_norm(e, u, u1, tol)
    end
    return StepResult(u1, E, fsal, nf, false, st.violation)
end

"""
    rosenbrock_adjoint_step(m, ctx, u, p, t, h, λ; tk1) -> (λₙ, p̄, nf)

The discrete adjoint of one step (spec/adjoint.md §4.2), with the stages
recomputed from `u` (replay) and the forward factorization of Ŵ reused for
the transposed solves. Transformed form: k̄ᵢ = bᵢλ; for i = s..1,
μᵢ = Ŵ⁻ᵀk̄ᵢ, σᵢ = J(Uᵢ)ᵀμᵢ, k̄ⱼ += Aᵢⱼσᵢ + (Cᵢⱼ/h)μᵢ for j < i; then
λₙ = λ + Σσᵢ + ∇ᵤΣμᵢᵀJ(u)kᵢ + ∇ᵤ(νᵀf_t), ν = Σ (h dᵢ) μᵢ, and
p̄ = Σ f_p(Uᵢ)ᵀμᵢ + ∇ₚΣμᵢᵀJkᵢ + ∇ₚ(νᵀf_t). Rosenbrock23: the pairs are
(μ₁, dtγ k₁), (μ₂, dtγ (k₂ - k₁)) and ν = dtγ μ₁. Sums run over ascending
stages (spec/adjoint.md §6.4).
"""
function rosenbrock_adjoint_step(m::RosenbrockMethod, ctx::RosenbrockContext, u, p, t, h, λ; tk1 = t)
    st = rosenbrock_stages(m, ctx, u, p, t, h, nothing; tk1)
    st.singular && error("rosenbrock adjoint: W is singular on replay at t = $t, h = $h")
    N, P = length(u), length(p)
    ops = ctx.ops
    F = st.F
    if m.form === :rosenbrock23
        kk1, kk2 = st.k
        U2 = st.U[2]
        dtγ = h * m.d[1]
        kb2 = h .* λ
        μ2 = lu_solve_transpose(F, kb2)
        σ2 = zeros(N)
        vjp_u!(ops, σ2, U2, p, t + h / 2, μ2)
        kb1 = similar(kb2)
        @inbounds for j in 1:N
            kb1[j] = (kb2[j] - μ2[j]) + (h / 2) * σ2[j]
        end
        μ1 = lu_solve_transpose(F, kb1)
        σ1 = zeros(N)
        vjp_u!(ops, σ1, u, p, tk1, μ1)
        pairs = [(μ1, dtγ .* kk1), (μ2, dtγ .* (kk2 .- kk1))]
        λn = copy(λ)
        λn .+= σ1
        λn .+= σ2
        hu = zeros(N)
        hess_u!(ops, hu, u, p, t, pairs)
        λn .+= hu
        p̄ = zeros(P)
        vjp_p!(ops, p̄, u, p, tk1, μ1)
        vjp_p!(ops, p̄, U2, p, t + h / 2, μ2)
        hess_p!(ops, p̄, u, p, t, pairs)
        if st.ft !== nothing
            ν = dtγ .* μ1
            fu = zeros(N)
            ft_vjp_u!(ops, fu, u, p, t, ν)
            λn .+= fu
            ft_vjp_p!(ops, p̄, u, p, t, ν)
        end
        return λn, p̄, st.nf
    end
    s = m.s
    kb = Vector{Union{Nothing, Vector{Float64}}}(nothing, s)
    for i in 1:s
        m.b[i] == 0 || (kb[i] = m.b[i] .* λ)
    end
    μ = Vector{Union{Nothing, Vector{Float64}}}(nothing, s)
    σ = Vector{Union{Nothing, Vector{Float64}}}(nothing, s)
    pc = Vector{Union{Nothing, Vector{Float64}}}(nothing, s)
    for i in s:-1:1
        kb[i] === nothing && continue
        μ[i] = lu_solve_transpose(F, kb[i])
        τ = t + m.c[i] * h
        σ[i] = zeros(N)
        vjp_u!(ops, σ[i], st.U[i], p, τ, μ[i])
        pc[i] = zeros(P)
        vjp_p!(ops, pc[i], st.U[i], p, τ, μ[i])
        for j in 1:(i - 1)
            a, cij = m.A[i, j], m.C[i, j]
            (a == 0 && cij == 0) && continue
            kb[j] === nothing && (kb[j] = zeros(N))
            a == 0 || _addto!(kb[j], a, σ[i])
            cij == 0 || _addto!(kb[j], cij / h, μ[i])
        end
    end
    λn = copy(λ)
    p̄ = zeros(P)
    pairs = Tuple{Vector{Float64}, Vector{Float64}}[]
    ν = nothing
    for i in 1:s
        μ[i] === nothing && continue
        λn .+= σ[i]
        p̄ .+= pc[i]
        push!(pairs, (μ[i], st.k[i]))
        if st.ft !== nothing && m.d[i] != 0
            ν === nothing ? (ν = (h * m.d[i]) .* μ[i]) : _addto!(ν, h * m.d[i], μ[i])
        end
    end
    hu = zeros(N)
    hess_u!(ops, hu, u, p, t, pairs)
    λn .+= hu
    hess_p!(ops, p̄, u, p, t, pairs)
    if ν !== nothing
        fu = zeros(N)
        ft_vjp_u!(ops, fu, u, p, t, ν)
        λn .+= fu
        ft_vjp_p!(ops, p̄, u, p, t, ν)
    end
    return λn, p̄, st.nf
end
