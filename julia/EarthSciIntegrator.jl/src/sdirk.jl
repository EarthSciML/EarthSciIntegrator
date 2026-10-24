# The implicit families: SDIRK and ESDIRK (spec/methods.md §5, spec/adjoint.md
# §4.3), IMEX Euler and the θ-method (spec/methods.md §6, spec/adjoint.md
# §4.4–§4.5), and the Newton iteration they share (spec/methods.md §5.2).
#
# Arithmetic, agreed with the Rust implementation:
# - W = I - hγJ is factored as M = σI - J with σ = `inv(h*γ)` (the same
#   `factor_W` as Rosenbrock), J formed at (uₙ, tₙ) once per step attempt and
#   shared by every implicit stage (§5.3). A solve W⁻¹r is computed as
#   M⁻¹(r·σ) in the Newton iteration, which is OrdinaryDiffEqNonlinearSolve's
#   `ztmp = (dt*f(ustep) - z)*invγdt`, `dz = W \ ztmp` (`src/newton.jl:874-893`,
#   with its W = J - σI negated), and likewise as M⁻¹(e·σ) for the smoothed
#   error estimate and M⁻ᵀ(R·σ) for the adjoint's transposed solves.
# - Newton iteration m (§5.2): Zᵢ = τᵢ + γ z (elementwise `τ[j] + γ*z[j]`),
#   r = (h f(Zᵢ, tₙ + cᵢh) - z)·σ elementwise as `(h*f[j] - z[j])*σ`,
#   Δz = M⁻¹r, δₘ = ‖Δz‖ in §7's norm with uₙ and Zᵢ, z ← z + Δz.
# - τᵢ = uₙ + a_i1 z₁ + a_i2 z₂ + … left to right from uₙ; u_{n+1} = uₙ +
#   b₁z₁ + b₂z₂ + … (spec/methods.md §5.1, also for the stiffly accurate
#   methods, where OrdinaryDiffEq takes τ_s + γz_s instead; the two agree to
#   roundoff); e_raw = b̃₁z₁ + … starting from its first non-zero term. Zero
#   coefficients skip their term. The explicit first stage is z₁ = h·f(uₙ, tₙ)
#   elementwise.
#
# The Newton iteration follows OrdinaryDiffEqNonlinearSolve 2.9.9's `nlsolve!`
# (`src/nlsolve.jl:105-275`) with two readings that differ from the wording of
# spec/methods.md §5.2 (both reported as spec findings, and agreed with the
# Rust implementation):
# 1. η on iteration 1 is θ₁/(1 - θ₁), as `nlsolve!` computes it before its
#    convergence test (`:233`); the value it sets from η_old before the loop
#    (`:140`) is overwritten and never tested. So η_old has no effect, and with
#    θ₁ = 1 iteration 1 converges only through δ₁ < 1e-5. For a fixed-dt leaf
#    iteration 1 converges only through δ₁ < 1e-5 whatever θ₁ is
#    (`check_η_convergence`, `:235-246`, "don't trust θ for non-adaptive on
#    first iter").
# 2. θ_prev is 1 at the start of every step attempt and carries across the
#    attempt's stages. OrdinaryDiffEq carries it across steps as well, resetting
#    it only for the attempts that follow a rejection; the tape does not record
#    where rejections fell (spec/tape.md §2), so the step would not be a
#    function of (uₙ, h) alone and replay could not reproduce it bitwise
#    (spec/methods.md §5.3 states that intent; spec/tape.md §3 requires replay
#    to be bitwise).
# Also as in `nlsolve!`: the floating-point-limit and divergence tests return
# before the iteration's Δz is applied and before θ_prev is updated; running
# out of iterations is a failure.
#
# Forward sensitivities (spec/adjoint.md §2.3, §7.1): with `Dual` numbers in
# the state or parameters, the Newton iteration runs on the primal values (so
# the primal stages are bitwise those of a Float64 run), and each converged
# stage then gets the tangent of the exact stage equation by the implicit
# function theorem, (I - hγJ(Zᵢ)) dzᵢ = h (J(Zᵢ) dτᵢ + f_p dp), solved with W
# refactored at the converged stage, as the adjoint does (D42).

"""
    NEWTON_TIGHT

When set, every implicit stage, once §5.2's iteration has converged, is
polished by full Newton iterations (J refactored at each iterate) until the
increment stops shrinking: the finite-difference check on the frozen tape
replays with implicit solves converged to roundoff (spec/adjoint.md §7.2).
Use [`with_tight_newton`](@ref).
"""
const NEWTON_TIGHT = Ref(false)

"""`with_tight_newton(f)`: call `f()` with implicit stages converged to roundoff (spec/adjoint.md §7.2)."""
function with_tight_newton(f)
    old = NEWTON_TIGHT[]
    NEWTON_TIGHT[] = true
    try
        return f()
    finally
        NEWTON_TIGHT[] = old
    end
end

"""
    with_value(x, v)

`x` with its primal value replaced by `v` and any tangent kept (the
ForwardDiff extension handles `Dual`); for a plain number, `v`.
"""
with_value(x::Real, v) = convert(typeof(x), v)

_is_plain(::Type{T}) where {T} = T <: AbstractFloat

"""An SDIRK or ESDIRK method in z form (spec/methods.md §5)."""
struct SDIRKMethod <: AbstractMethod
    tab::Tableau
    s::Int
    A::Matrix{Float64}
    b::Vector{Float64}
    btilde::Vector{Float64}
    c::Vector{Float64}
    gamma::Float64
    explicit_first::Bool
end

function SDIRKMethod(tab::Tableau)
    A, b, c = something(tab.A), something(tab.b), something(tab.c)
    s = length(b)
    ef = something(tab.explicit_first_stage, false)
    γ = A[s, s]
    for i in (ef ? 2 : 1):s
        A[i, i] == γ || error("table $(tab.name): implicit stage $i has a_ii = $(A[i, i]), not γ = $γ")
    end
    bt = tab.btilde === nothing ? zeros(s) : tab.btilde
    return SDIRKMethod(tab, s, A, b, bt, c, γ, ef)
end

"""The θ-method (spec/methods.md §6) with the leaf's `theta`."""
struct ThetaMethod <: AbstractMethod
    tab::Tableau
    theta::Float64
end

"""IMEX Euler (spec/methods.md §6); `linearized` for `implicit_solve: "linearized"` (D43)."""
struct IMEXEulerMethod <: AbstractMethod
    tab::Tableau
    linearized::Bool
end

const ImplicitMethod = Union{SDIRKMethod, ThetaMethod, IMEXEulerMethod}

"""
The context an implicit step needs: the leaf's operators (`ops`, all of them),
the explicit and implicit parts (`opsE` is `nothing` when empty; for SDIRK and
θ, `opsI` is `ops`), the linear solver and the tolerances of the Newton norm.
"""
struct ImplicitContext{O, OE, OI}
    ops::O
    opsE::OE
    opsI::OI
    ls::LinsolveSpec
    structure::Structure
    tol::Tolerances
end

_feval(ops, x, p, τ, ::Type{T}) where {T} = (out = zeros(T, length(x)); eval_rhs!(out, ops, x, p, τ); out)

"""`J` of `ops` at `(u, p, t)` on primal values, and the factors of `σI - J`."""
function _factor_at(ctx::ImplicitContext, ops, u, p, t, σ)
    up, pp = map(_f64, u), map(_f64, p)
    J = jacobian_storage(ctx.ls, ctx.structure, Float64, length(u))
    jac!(ops, J, up, pp, t)
    return factor_W(J, σ, ctx.ls)
end

"""
    newton_stage(fI, F, σ, τ, γ, h, ti, z0, un, tol, θ, adaptive; refactor) -> (z, ok, nf)

One implicit stage by modified Newton on primal (Float64) values
(spec/methods.md §5.2, with the readings in the file header): `fI(x, τ)`
returns the implicit right-hand side, `F` the factors of `σI - J`, `τ` the
stage's explicit part, `z0` the initial guess, `θ` a `Ref` holding θ_prev
(updated). Under [`NEWTON_TIGHT`](@ref) the converged stage is then polished
with `refactor(Z)`, the factors of `σI - J(Z)`.
"""
function newton_stage(fI, F, σ, τ::Vector{Float64}, γ, h, ti, z0::Vector{Float64}, un, tol::Tolerances,
        θ::Base.RefValue{Float64}, adaptive::Bool; refactor = nothing)
    z, ok, nf = _newton(fI, F, σ, τ, γ, h, ti, z0, un, tol, θ, adaptive)
    (ok && NEWTON_TIGHT[] && refactor !== nothing) || return z, ok, nf
    # Full Newton from the converged stage until the increment stops shrinking.
    N = length(z)
    Z = similar(z)
    ndzprev = Inf
    for _ in 1:30
        @inbounds for j in 1:N
            Z[j] = τ[j] + γ * z[j]
        end
        fz = fI(Z, ti)
        nf += 1
        r = [(h * fz[j] - z[j]) * σ for j in 1:N]
        dz = lu_solve(refactor(Z), r)
        ndz = error_norm(dz, un, Z, tol)
        (isfinite(ndz) && ndz < ndzprev) || break
        z .= z .+ dz
        iszero(ndz) && break
        ndzprev = ndz
    end
    return z, true, nf
end

function _newton(fI, F, σ, τ::Vector{Float64}, γ, h, ti, z0::Vector{Float64}, un, tol::Tolerances,
        θ::Base.RefValue{Float64}, adaptive::Bool)
    N = length(τ)
    z = copy(z0)
    Z = similar(z)
    r = similar(z)
    κ = 0.01
    maxiters = 10
    ndz = 1.0
    nf = 0
    for iter in 1:maxiters
        @inbounds for j in 1:N
            Z[j] = τ[j] + γ * z[j]
        end
        fz = fI(Z, ti)
        nf += 1
        @inbounds for j in 1:N
            r[j] = (h * fz[j] - z[j]) * σ
        end
        dz = lu_solve(F, r)
        ndzprev = ndz
        ndz = error_norm(dz, un, Z, tol)
        isfinite(ndz) || return z, false, nf
        if iter > 1
            θm = max(0.3 * θ[], ndz / ndzprev)
            if abs(θm - 1) <= 100 * sqrt(eps(1.0))
                # The floating-point limit (§5.2 step 6), tested before Δz is applied.
                return z, ndz <= 1, nf
            end
            θm > 2 && return z, false, nf
            # θ_prev is stored only after both tests, as `nlsolve!` does (`:228-230`).
            θ[] = θm
        else
            θm = θ[]
        end
        @inbounds for j in 1:N
            z[j] = z[j] + dz[j]
        end
        η = θm / (1 - θm)
        if (iter == 1 && ndz < 1.0e-5) || ((iter > 1 || adaptive) && η >= 0 && η * ndz < κ)
            return z, true, nf
        end
    end
    return z, false, nf
end

"""
    stage_tangent(ctx, opsI, τ, zp, γ, h, ti, p) -> z

The converged stage `zp` (primal) with the tangent of the exact stage
equation z = h f(τ + γz) (implicit function theorem, spec/adjoint.md §4.3):
`dz = M⁻¹((h J dτ + h f_p dp)·σ)` with `M = σI - J(Z)` refactored at the
converged stage `Z = τ + γ zp`. The values stay `zp` bitwise.
"""
function stage_tangent(ctx::ImplicitContext, opsI, τ, zp, γ, h, ti, p)
    T = promote_type(eltype(τ), eltype(p))
    _is_plain(T) && return zp
    σ = inv(h * γ)
    N = length(zp)
    Zp = [_f64(τ[j]) + γ * zp[j] for j in 1:N]
    Fex = _factor_at(ctx, opsI, Zp, p, ti, σ)
    issuccess_lu(Fex) || throw(ReplayMismatchError("W is singular at a converged implicit stage (t = $ti)"))
    Zd = [τ[j] + γ * zp[j] for j in 1:N]
    fz = _feval(opsI, Zd, p, ti, T)
    r = [(h * fz[j] - zp[j]) * σ for j in 1:N]
    dz = lu_solve(Fex, r)
    return [with_value(dz[j], zp[j]) for j in 1:N]
end

# ---------------------------------------------------------------------------
# SDIRK / ESDIRK

"""
    sdirk_stages(m, ctx, u, p, t, h, adaptive) -> NamedTuple

Every stage of one step attempt from `(u, t)`: `τ`, `z`, the stage states
`Z = τ + γz` (uₙ for an explicit first stage), the end state, the factors of
the step's W, the f count, and whether W was singular or a Newton solve failed.
Generic over element types (see the file header for tangents).
"""
function sdirk_stages(m::SDIRKMethod, ctx::ImplicitContext, u, p, t, h, adaptive::Bool)
    T = promote_type(eltype(u), eltype(p))
    N = length(u)
    s, γ = m.s, m.gamma
    σ = inv(h * γ)
    F = _factor_at(ctx, ctx.ops, u, p, t, σ)
    nf = 0
    fail(singular) = (; singular, newton_fail = !singular, nf, F, σ, τ = nothing, z = nothing, Z = nothing, u1 = nothing)
    issuccess_lu(F) || return fail(true)
    up, pp = map(_f64, u), map(_f64, p)
    fI = (x, τ) -> _feval(ctx.ops, x, pp, τ, Float64)
    θ = Ref(1.0)      # θ_prev starts at 1 in every step attempt (file header, reading 2)
    τs = Vector{Vector{T}}(undef, s)
    zs = Vector{Vector{T}}(undef, s)
    Zs = Vector{Vector{T}}(undef, s)
    zprev = nothing
    for i in 1:s
        τi = convert(Vector{T}, copy(u))
        for j in 1:(i - 1)
            a = m.A[i, j]
            a == 0 || _addto!(τi, a, zs[j])
        end
        τs[i] = τi
        ti = t + m.c[i] * h
        if i == 1 && m.explicit_first
            fu = _feval(ctx.ops, u, p, t, T)
            nf += 1
            zs[1] = [h * fu[j] for j in 1:N]
            Zs[1] = convert(Vector{T}, copy(u))
            continue
        end
        z0 = zprev === nothing ? zeros(N) : zprev
        zp, ok, n = newton_stage(fI, F, σ, map(_f64, τi), γ, h, ti, z0, up, ctx.tol, θ, adaptive;
            refactor = Z -> _factor_at(ctx, ctx.ops, Z, pp, ti, σ))
        nf += n
        ok || return fail(false)
        zprev = zp
        zs[i] = stage_tangent(ctx, ctx.ops, τi, zp, γ, h, ti, p)
        Zs[i] = [τi[j] + γ * zs[i][j] for j in 1:N]
    end
    u1 = convert(Vector{T}, copy(u))
    for i in 1:s
        m.b[i] == 0 || _addto!(u1, m.b[i], zs[i])
    end
    return (; singular = false, newton_fail = false, nf, F, σ, τ = τs, z = zs, Z = Zs, u1)
end

"""The stage times tₙ + cᵢh (spec/methods.md §5.4)."""
_stage_time(m::SDIRKMethod, t, h, i) = (i == 1 && m.explicit_first) ? t : t + m.c[i] * h

"""
Δq of the integrated functional (q' = ψ(u), spec/adjoint.md §2.1): the method
on the augmented state at the exact stages, Δq = Σ bᵢ (h ψ(Zᵢ)).
"""
function sdirk_dq(m::SDIRKMethod, st, ψ, h)
    dq = nothing
    for i in 1:m.s
        m.b[i] == 0 && continue
        v = m.b[i] * (h * ψ(st.Z[i]))
        dq = dq === nothing ? v : dq + v
    end
    return dq
end

"""
    sdirk_step(m, ctx, u, p, t, h, adaptive, hist; ψ) -> StepResult

One step attempt. `hist` is `nothing` or `(u_{n-1}, t_{n-1})`, the previous
accepted step of this invocation, for BackwardEuler's divided-difference
estimate (spec/methods.md §5.5).
"""
function sdirk_step(m::SDIRKMethod, ctx::ImplicitContext, u, p, t, h, adaptive::Bool, hist; ψ = nothing)
    st = sdirk_stages(m, ctx, u, p, t, h, adaptive)
    st.singular && return StepResult(u, 2.0, nothing, st.nf, true, false, nothing, false)
    st.newton_fail && return StepResult(u, NaN, nothing, st.nf, false, false, nothing, true)
    u1 = st.u1
    E = NaN
    if adaptive
        est = m.tab.error_estimate
        if est === :embedded_smoothed
            # e = W⁻¹ e_raw with the step's W (Hosea and Shampine; §5.5, D56).
            e = nothing
            for i in 1:m.s
                bt = m.btilde[i]
                bt == 0 && continue
                e === nothing ? (e = convert(eltype(u1), bt) .* st.z[i]) : _addto!(e, bt, st.z[i])
            end
            E = e === nothing ? 0.0 : error_norm(lu_solve(st.F, e .* st.σ), u, u1, ctx.tol)
        elseif est === :divided_difference
            # OrdinaryDiffEqSDIRK `src/generic_imex_perform_step.jl:2401-2420`, in its
            # arithmetic: dt1 = h(t + h - t_{n-1}), dt2 = (t - t_{n-1})(t + h - t_{n-1}),
            # e = (7/12 h²)|(u_{n+1} - uₙ)/dt1 - (uₙ - u_{n-1})/dt2|; E = 1 on the first
            # step of an invocation.
            if hist === nothing
                E = 1.0
            else
                uprev2, tprev = hist
                dt1 = h * ((t + h) - tprev)
                dt2 = (t - tprev) * ((t + h) - tprev)
                rr = something(m.tab.divided_difference) * h^2
                e = [rr * abs((u1[j] - u[j]) / dt1 - (u[j] - uprev2[j]) / dt2) for j in eachindex(u1)]
                E = error_norm(e, u, u1, ctx.tol)
            end
        else
            throw(ConfigError("method $(m.tab.name) has no error estimate; give the leaf a dt (D52)"))
        end
    end
    dq = ψ === nothing ? nothing : sdirk_dq(m, st, ψ, h)
    return StepResult(u1, E, nothing, st.nf, false, false, dq, false)
end

"""
    sdirk_adjoint_step(m, ctx, u, p, t, h, λ; ψ) -> (λₙ, p̄, nf)

spec/adjoint.md §4.3: stages recomputed from `u`; for i = s..1,
Wᵢᵀωᵢ = h(bᵢλ + Σ_{j>i} a_ji σⱼ) with Wᵢ = I - hγJ(Zᵢ) refactored at the
converged stage (D42; Wᵢ = I for an explicit first stage), σᵢ = J(Zᵢ)ᵀωᵢ;
`adaptive` is the leaf's, which the replayed Newton iteration needs;
λₙ = λ + Σσᵢ, p̄ = Σ f_p(Zᵢ)ᵀωᵢ, sums over ascending stages. The transposed
solve is Mᵢ⁻ᵀ(rhs·σ) with Mᵢ = σI - J(Zᵢ). With the integrated functional
(q̄ = 1) the q stage's cotangent is h bᵢ, which adds hγ·h bᵢ ∇ψ to an
implicit stage's right-hand side and h bᵢ ∇ψ to σᵢ.
"""
function sdirk_adjoint_step(m::SDIRKMethod, ctx::ImplicitContext, u, p, t, h, λ; ψ = nothing, adaptive::Bool = false)
    st = sdirk_stages(m, ctx, u, p, t, h, adaptive)
    (st.singular || st.newton_fail) &&
        throw(ReplayMismatchError("implicit stages fail on replay of the step at t = $t, h = $h"))
    N, P = length(u), length(p)
    ops = ctx.ops
    γ, σW = m.gamma, st.σ
    σ = Vector{Union{Nothing, Vector{Float64}}}(nothing, m.s)
    pc = Vector{Union{Nothing, Vector{Float64}}}(nothing, m.s)
    for i in m.s:-1:1
        ρ = nothing
        m.b[i] == 0 || (ρ = m.b[i] .* λ)
        for j in (i + 1):m.s
            a = m.A[j, i]
            (a == 0 || σ[j] === nothing) && continue
            ρ === nothing ? (ρ = a .* σ[j]) : _addto!(ρ, a, σ[j])
        end
        ωq = ψ === nothing ? 0.0 : h * m.b[i]
        (ρ === nothing && ωq == 0) && continue
        rhs = ρ === nothing ? zeros(N) : h .* ρ
        ti = _stage_time(m, t, h, i)
        Zi = st.Z[i]
        ω = if i == 1 && m.explicit_first
            rhs
        else
            ωq == 0 || _addto!(rhs, h * γ * ωq, ψ.w)
            Fi = _factor_at(ctx, ops, Zi, p, ti, σW)
            issuccess_lu(Fi) || throw(ReplayMismatchError("W is singular at the converged stage $i (t = $t, h = $h)"))
            lu_solve_transpose(Fi, rhs .* σW)
        end
        σ[i] = zeros(N)
        vjp_u!(ops, σ[i], Zi, p, ti, ω)
        ωq == 0 || _addto!(σ[i], ωq, ψ.w)
        pc[i] = zeros(P)
        vjp_p!(ops, pc[i], Zi, p, ti, ω)
    end
    λn = copy(λ)
    p̄ = zeros(P)
    for i in 1:m.s
        σ[i] === nothing && continue
        λn .+= σ[i]
        p̄ .+= pc[i]
    end
    return λn, p̄, st.nf
end

# ---------------------------------------------------------------------------
# θ-method and IMEX Euler (converged): one implicit stage after an explicit part
#
#   τ = uₙ + c_E g(uₙ, tₙ),  z = h f_I(τ + γz, tₙ + h),  u_{n+1} = τ + γz,
#
# θ-method: g = f, c_E = h(1 - θ) (formed as `h * (1 - θ)`), γ = θ, f_I = f;
# IMEX Euler: g = f_E, c_E = h, γ = 1, f_I the implicit part. A zero c_E or an
# empty explicit part skips g (θ = 1, and IMEX Euler with every operator
# implicit, take τ = uₙ); θ = 0 is forward Euler, with no solve.

"""`(g, c_E, γ, fI)` of a θ or IMEX Euler step: the explicit operators (or nothing), their coefficient, γ, the implicit operators."""
function _one_stage_parts(m::ThetaMethod, ctx::ImplicitContext, h)
    θ = m.theta
    cE = h * (1 - θ)
    return (cE == 0 ? nothing : ctx.ops), cE, θ, (θ == 0 ? nothing : ctx.ops)
end
_one_stage_parts(m::IMEXEulerMethod, ctx::ImplicitContext, h) = ctx.opsE, h, 1.0, ctx.opsI

function one_stage(m::Union{ThetaMethod, IMEXEulerMethod}, ctx::ImplicitContext, u, p, t, h, adaptive::Bool)
    T = promote_type(eltype(u), eltype(p))
    N = length(u)
    g, cE, γ, opsI = _one_stage_parts(m, ctx, h)
    nf = 0
    τ = convert(Vector{T}, copy(u))
    if g !== nothing
        g0 = _feval(g, u, p, t, T)
        nf += 1
        _addto!(τ, cE, g0)
    end
    nothing_out = (; singular = false, newton_fail = false, nf, τ, z = nothing, Z = τ, u1 = τ, F = nothing, σ = NaN)
    opsI === nothing && return nothing_out
    σ = inv(h * γ)
    F = _factor_at(ctx, opsI, u, p, t, σ)
    issuccess_lu(F) || return (; nothing_out..., singular = true, u1 = nothing)
    pp = map(_f64, p)
    fI = (x, τ) -> _feval(opsI, x, pp, τ, Float64)
    zp, ok, n = newton_stage(fI, F, σ, map(_f64, τ), γ, h, t + h, zeros(N), map(_f64, u), ctx.tol, Ref(1.0), adaptive;
        refactor = Z -> _factor_at(ctx, opsI, Z, pp, t + h, σ))
    nf += n
    ok || return (; nothing_out..., newton_fail = true, nf, u1 = nothing)
    z = stage_tangent(ctx, opsI, τ, zp, γ, h, t + h, p)
    Z = [τ[j] + γ * z[j] for j in 1:N]
    return (; singular = false, newton_fail = false, nf, τ, z, Z, u1 = Z, F, σ)
end

"""
Δq of the θ-method: h(1 - θ) ψ(uₙ) + θ (h ψ(u_{n+1})), the method applied to
q' = ψ(u) at the exact stage.
"""
function theta_dq(m::ThetaMethod, u, u1, ψ, h)
    θ = m.theta
    cE = h * (1 - θ)
    dq = cE == 0 ? nothing : cE * ψ(u)
    θ == 0 && return dq
    v = θ * (h * ψ(u1))
    return dq === nothing ? v : dq + v
end

function one_stage_step(m::Union{ThetaMethod, IMEXEulerMethod}, ctx::ImplicitContext, u, p, t, h; ψ = nothing)
    m isa IMEXEulerMethod && m.linearized && return imex_linearized_step(ctx, u, p, t, h)
    st = one_stage(m, ctx, u, p, t, h, false)
    st.singular && return StepResult(u, 2.0, nothing, st.nf, true, false, nothing, false)
    st.newton_fail && return StepResult(u, NaN, nothing, st.nf, false, false, nothing, true)
    dq = ψ === nothing ? nothing : theta_dq(m, u, st.u1, ψ, h)
    return StepResult(st.u1, NaN, nothing, st.nf, false, false, dq, false)
end

"""
spec/adjoint.md §4.4–§4.5: Wᵀω = λ with W = I - hγJ_I(u_{n+1}) refactored at
the converged state, solved as M⁻ᵀ(λ·σ); λₙ = ω + c_E J_g(uₙ)ᵀω and
p̄ = c_E g_p(uₙ)ᵀω + hγ f_{I,p}(u_{n+1})ᵀω (θ: c_E = h(1-θ), γ = θ; IMEX
Euler: c_E = h, γ = 1). With the integrated functional (θ-method only) the
q stage's cotangent is 1, which adds hγ∇ψ to the right-hand side and c_E∇ψ
to λₙ.
"""
function one_stage_adjoint(m::Union{ThetaMethod, IMEXEulerMethod}, ctx::ImplicitContext, u, p, t, h, λ; ψ = nothing)
    m isa IMEXEulerMethod && m.linearized && return imex_linearized_adjoint(ctx, u, p, t, h, λ)
    st = one_stage(m, ctx, u, p, t, h, false)
    (st.singular || st.newton_fail) &&
        throw(ReplayMismatchError("implicit stage fails on replay of the step at t = $t, h = $h"))
    N, P = length(u), length(p)
    g, cE, γ, opsI = _one_stage_parts(m, ctx, h)
    p̄ = zeros(P)
    ω = if opsI === nothing
        copy(λ)
    else
        rhs = copy(λ)
        ψ === nothing || _addto!(rhs, h * γ, ψ.w)
        Fx = _factor_at(ctx, opsI, st.Z, p, t + h, st.σ)
        issuccess_lu(Fx) || throw(ReplayMismatchError("W is singular at the converged state (t = $t, h = $h)"))
        lu_solve_transpose(Fx, rhs .* st.σ)
    end
    λn = copy(ω)
    if g !== nothing
        s = zeros(N)
        vjp_u!(g, s, u, p, t, ω)
        ψ === nothing || _addto!(s, 1.0, ψ.w)
        _addto!(λn, cE, s)
        pg = zeros(P)
        vjp_p!(g, pg, u, p, t, ω)
        _addto!(p̄, cE, pg)
    end
    if opsI !== nothing
        pI = zeros(P)
        vjp_p!(opsI, pI, st.Z, p, t + h, ω)
        _addto!(p̄, h * γ, pI)
    end
    return λn, p̄, st.nf
end

# ---------------------------------------------------------------------------
# IMEX Euler, `implicit_solve: "linearized"` (D43): one Rosenbrock–Euler step,
# Ŵ k = f_E(uₙ) + f_I(uₙ) with Ŵ = I/h - J_I(uₙ) (σ = `inv(h)`), u_{n+1} = uₙ + k.
# Its adjoint is spec/adjoint.md §4.2 with one stage: μ = Ŵ⁻ᵀλ,
# λₙ = λ + J(uₙ)ᵀμ + ∇ᵤ(μᵀJ_I(u)k), p̄ = f_p(uₙ)ᵀμ + ∇ₚ(μᵀJ_I k).

function imex_linearized_stage(ctx::ImplicitContext, u, p, t, h)
    T = promote_type(eltype(u), eltype(p))
    J = jacobian_storage(ctx.ls, ctx.structure, T, length(u))
    jac!(ctx.opsI, J, u, p, t)
    F = factor_W(J, inv(h), ctx.ls)
    issuccess_lu(F) || return nothing, F
    # f_E + f_I, each part one right-hand-side evaluation (as in the Rust implementation).
    r = _feval(ctx.opsI, u, p, t, T)
    ctx.opsE === nothing || (r = _feval(ctx.opsE, u, p, t, T) .+ r)
    return lu_solve(F, r), F
end

function imex_linearized_step(ctx::ImplicitContext, u, p, t, h)
    k, _ = imex_linearized_stage(ctx, u, p, t, h)
    nf = ctx.opsE === nothing ? 1 : 2
    k === nothing && return StepResult(u, 2.0, nothing, 0, true, false, nothing, false)
    u1 = convert(Vector{eltype(k)}, copy(u))
    _addto!(u1, 1.0, k)
    return StepResult(u1, NaN, nothing, nf, false, false, nothing, false)
end

function imex_linearized_adjoint(ctx::ImplicitContext, u, p, t, h, λ)
    k, F = imex_linearized_stage(ctx, u, p, t, h)
    k === nothing && throw(ReplayMismatchError("W is singular on replay of the step at t = $t, h = $h"))
    N, P = length(u), length(p)
    μ = lu_solve_transpose(F, λ)
    λn = copy(λ)
    s = zeros(N)
    vjp_u!(ctx.ops, s, u, p, t, μ)
    λn .+= s
    hu = zeros(N)
    hess_u!(ctx.opsI, hu, u, p, t, [(μ, k)])
    λn .+= hu
    p̄ = zeros(P)
    vjp_p!(ctx.ops, p̄, u, p, t, μ)
    hess_p!(ctx.opsI, p̄, u, p, t, [(μ, k)])
    return λn, p̄, 1
end
