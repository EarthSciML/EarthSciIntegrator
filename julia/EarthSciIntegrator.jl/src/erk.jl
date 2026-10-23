# The explicit Runge–Kutta family (spec/methods.md §3, spec/adjoint.md §4.1).
#
# Arithmetic (spec/methods.md §3.1). Every linear combination is accumulated
# left to right in ascending index order, starting from its leading term (the
# base state `u_n` for a Butcher stage and for `u_{n+1}`, the first non-skipped
# term otherwise), each term formed as `(h * a) * K` with the f64 scalar product
# first and converted to the state type. A zero coefficient skips its term.
# Shu–Osher sums run flat over ascending `j`, the α term before the β term for
# each `j`. No fused multiply-adds.

"""A method family's step. Families implemented later (Rosenbrock, SDIRK, IMEX, θ) add subtypes."""
abstract type AbstractMethod end

"""Explicit Runge–Kutta in Butcher form (`A`, `b`, `c`)."""
struct ButcherERK <: AbstractMethod
    tab::Tableau
    s::Int
    A::Matrix{Float64}
    b::Vector{Float64}
    c::Vector{Float64}
    # Stages that u_{n+1} depends on; the rest feed only the error estimate
    # and first-same-as-last (Tsit5's and BS3's last stage).
    active::Vector{Bool}
    # The last stage evaluates f(u_{n+1}, t_n + h): its row of A equals b.
    last_stage_is_end::Bool
end

"""Explicit Runge–Kutta in Shu–Osher form (`alpha`, `beta`, `c`)."""
struct ShuOsherERK <: AbstractMethod
    tab::Tableau
    s::Int
    alpha::Matrix{Float64}   # alpha[i, j+1] = αᵢⱼ, i = 1..s, j = 0..i-1
    beta::Matrix{Float64}
    c::Vector{Float64}       # c[j+1]: time fraction of f(u⁽ʲ⁾)
    needs_f::Vector{Bool}    # needs_f[j+1]: some βᵢⱼ ≠ 0 (including the embedded pair)
end

function ButcherERK(tab::Tableau)
    A, b, c = something(tab.A), something(tab.b), something(tab.c)
    s = length(b)
    active = falses(s)
    for i in s:-1:1
        active[i] = b[i] != 0 || any(j -> active[j] && A[j, i] != 0, (i + 1):s)
    end
    last_end = s > 1 && all(j -> A[s, j] == b[j], 1:s) && b[s] == 0 && c[s] == 1
    return ButcherERK(tab, s, A, b, c, collect(active), last_end)
end

function ShuOsherERK(tab::Tableau)
    α, β, c = something(tab.alpha), something(tab.beta), something(tab.c)
    s = size(α, 1)
    needs = [any(i -> β[i, j + 1] != 0, (j + 1):s) for j in 0:(s - 1)]
    if tab.embedded_beta !== nothing
        for (j, bt) in enumerate(tab.embedded_beta)
            bt != 0 && (needs[j] = true)
        end
    end
    return ShuOsherERK(tab, s, α, β, c, needs)
end

"""
    method_for(tab) -> AbstractMethod

The family dispatch: the step implementation for a method's table. Families
not implemented in this stage are refused with an [`UnsupportedError`](@ref).
"""
function method_for(tab::Tableau)
    if tab.family === :explicit_rk
        tab.form === :butcher && return ButcherERK(tab)
        tab.form === :shu_osher && return ShuOsherERK(tab)
    end
    throw(UnsupportedError("method $(tab.name) (family $(tab.family), form $(tab.form)) is not implemented yet; " *
                           "this stage implements the explicit Runge–Kutta family only"))
end

# `y .= x + coef * K`, elementwise, with coef converted to the element type.
@inline function _axpy!(y, x, coef::Float64, K)
    T = eltype(y)
    ct = convert(T, coef)
    @inbounds for k in eachindex(y)
        y[k] = x[k] + ct * K[k]
    end
    return y
end

"""
    StepResult

What one forward step attempt produced: the new state, the scaled error
(`NaN` when not estimated), the value of `f(u_{n+1}, t_n + h)` when the step
computed it (for first-same-as-last reuse, spec/methods.md §3.3), and the
number of right-hand-side evaluations.
"""
struct StepResult{V}
    u::V
    E::Float64
    fsal::Union{Nothing, V}
    nf::Int
end

# ---------------------------------------------------------------------------
# Butcher form

"""
    butcher_stages(m, f!, u, t, h, k1, nstages) -> (U, K, nf)

Stage states `U[i]` and values `K[i] = f(U[i], t + c_i h)` for stages
`1..nstages`. `k1`, when given, is a valid stored first-same-as-last value;
otherwise stage 1 is evaluated at `tk1` (`tₙ` normally, `tₙ₋₁ + hₙ₋₁` when
replaying a step whose first stage was reused, spec/methods.md §3.3).
"""
function butcher_stages(m::ButcherERK, f!, u, t, h, k1, nstages; tk1 = t + m.c[1] * h)
    U = Vector{typeof(u)}(undef, nstages)
    K = Vector{typeof(u)}(undef, nstages)
    nf = 0
    U[1] = u
    if k1 === nothing
        K[1] = similar(u)
        f!(K[1], u, tk1)
        nf += 1
    else
        K[1] = k1
    end
    for i in 2:nstages
        Ui = copy(u)
        for j in 1:(i - 1)
            a = m.A[i, j]
            a == 0 && continue
            _axpy!(Ui, Ui, h * a, K[j])
        end
        U[i] = Ui
        K[i] = similar(u)
        f!(K[i], Ui, t + m.c[i] * h)
        nf += 1
    end
    return U, K, nf
end

function butcher_update(m::ButcherERK, u, h, K)
    unew = copy(u)
    for i in 1:m.s
        bi = m.b[i]
        bi == 0 && continue
        _axpy!(unew, unew, h * bi, K[i])
    end
    return unew
end

"""
    erk_step(m, f!, u, t, h, k1, adaptive, tol) -> StepResult

One forward step attempt of size `h` from `(u, t)`. `f!(out, u, t)` overwrites
`out` with the leaf's right-hand side. With `adaptive` the error estimate of
spec/methods.md §3.2 is formed and normed.
"""
function erk_step(m::ButcherERK, f!, u, t, h, k1, adaptive::Bool, tol; tk1 = t + m.c[1] * h)
    est = m.tab.error_estimate
    fsal_method = m.tab.first_same_as_last
    nstages = (adaptive || (fsal_method && m.last_stage_is_end)) ? m.s : findlast(m.active)
    U, K, nf = butcher_stages(m, f!, u, t, h, k1, nstages; tk1)
    unew = butcher_update(m, u, h, K)
    E = NaN
    fsal = nothing
    if m.last_stage_is_end && nstages == m.s
        fsal = K[m.s]
    end
    if adaptive
        if est === :embedded
            bt = something(m.tab.btilde)
            e = nothing
            for i in 1:m.s
                bt[i] == 0 && continue
                if e === nothing
                    e = similar(u)
                    ct = convert(eltype(u), h * bt[i])
                    @inbounds for k in eachindex(e)
                        e[k] = ct * K[i][k]
                    end
                else
                    _axpy!(e, e, h * bt[i], K[i])
                end
            end
            E = error_norm(e, u, unew, tol)
        elseif est === :residual_control
            E, k5, nfr = rk4_residual_error(m, f!, u, unew, t, h, K[1], tol)
            nf += nfr
            fsal = k5                       # f(u_{n+1}, tₙ + h)
        else
            throw(ConfigError("method $(m.tab.name) has no error estimate; give the leaf a dt (D52)"))
        end
    end
    # First same as last (spec/methods.md §3.3): the end-of-step value
    # f(u_{n+1}, tₙ + h), evaluated at tₙ + h even when the step's end time is
    # snapped to a stop (OrdinaryDiffEq's `fsallast`).
    if fsal_method && fsal === nothing
        fsal = similar(u)
        f!(fsal, unew, t + h)
        nf += 1
    end
    return StepResult(unew, E, fsal, nf)
end

"""
Shampine's residual estimate for RK4 (spec/methods.md §3.2), in the arithmetic
of OrdinaryDiffEqLowOrderRK `src/fixed_timestep_perform_step.jl:262-306`
evaluated left to right without fused multiply-adds. Returns `(E, k5, nf)`.
"""
function rk4_residual_error(m::ButcherERK, f!, un, u1, t, h, k1, tol)
    k5 = similar(un)
    f!(k5, u1, t + h)
    nf = 1
    σs = something(m.tab.residual_sigma)
    Es = Float64[]
    for σ in σs
        pσ = similar(un)
        ppσ = similar(un)
        @inbounds for j in eachindex(un)
            pσ[j] = (1 - σ) * un[j] + σ * u1[j] +
                σ * (σ - 1) * ((1 - 2σ) * (u1[j] - un[j]) + (σ - 1) * h * k1[j] + σ * h * k5[j])
            ppσ[j] = k1[j] +
                σ * (-4 * h * k1[j] - 2 * h * k5[j] - 6 * un[j] +
                 σ * (3 * h * k1[j] + 3 * h * k5[j] + 6 * un[j] - 6 * u1[j]) + 6 * u1[j]) / h
        end
        fp = similar(un)
        f!(fp, pσ, t + σ * h)
        nf += 1
        r = similar(un)
        @inbounds for j in eachindex(r)
            r[j] = h * (fp[j] - ppσ[j])
        end
        push!(Es, error_norm(r, un, u1, tol))
    end
    return something(m.tab.residual_scale) * maximum(Es), k5, nf
end

"""
    erk_adjoint_step(m, ops, u, p, t, h, λ) -> (λn, p̄, nf)

The discrete adjoint of one Butcher-form step (spec/adjoint.md §4.1): stages
are recomputed from `u` (replay), then swept `i = s..1` with
`κᵢ = h bᵢ λ + Σ_{j>i} h a_{ji} σⱼ`, `σᵢ = J(Uᵢ)ᵀκᵢ`; `λₙ = λ + Σᵢ σᵢ` and
`p̄ = Σᵢ f_p(Uᵢ)ᵀκᵢ`, both summed over ascending stages (§6.4). Stages with
structurally zero `κᵢ` are skipped.
"""
function erk_adjoint_step(m::ButcherERK, ops, u, p, t, h, λ; tk1 = t + m.c[1] * h)
    f! = (out, x, τ) -> eval_rhs!(out, ops, x, p, τ)
    nact = findlast(m.active)
    U, _, nf = butcher_stages(m, f!, u, t, h, nothing, nact; tk1)
    N, P = length(u), length(p)
    σ = Vector{Union{Nothing, Vector{Float64}}}(nothing, m.s)
    pc = Vector{Union{Nothing, Vector{Float64}}}(nothing, m.s)
    for i in nact:-1:1
        m.active[i] || continue
        κ = nothing
        if m.b[i] != 0
            κ = (h * m.b[i]) .* λ
        end
        for j in (i + 1):nact
            a = m.A[j, i]
            (a == 0 || σ[j] === nothing) && continue
            κ === nothing ? (κ = (h * a) .* σ[j]) : _axpy!(κ, κ, h * a, σ[j])
        end
        κ === nothing && continue
        τ = i == 1 ? tk1 : t + m.c[i] * h
        σ[i] = zeros(N)
        vjp_u!(ops, σ[i], U[i], p, τ, κ)
        pc[i] = zeros(P)
        vjp_p!(ops, pc[i], U[i], p, τ, κ)
    end
    λn = copy(λ)
    p̄ = zeros(P)
    for i in 1:m.s
        σ[i] === nothing && continue
        λn .+= σ[i]
        p̄ .+= pc[i]
    end
    return λn, p̄, nf
end

# ---------------------------------------------------------------------------
# Shu–Osher form

"""
    shu_osher_stages(m, f!, u, t, h) -> (us, F, nf)

`us[j+1] = u⁽ʲ⁾` for `j = 0..s` and `F[j+1] = f(u⁽ʲ⁾, t + c_j h)` where needed.
"""
function shu_osher_stages(m::ShuOsherERK, f!, u, t, h)
    s = m.s
    us = Vector{typeof(u)}(undef, s + 1)
    F = Vector{Union{Nothing, typeof(u)}}(nothing, s)
    us[1] = u
    nf = 0
    for i in 1:s
        j_need = i - 1
        if m.needs_f[j_need + 1] && F[j_need + 1] === nothing
            Fj = similar(u)
            f!(Fj, us[j_need + 1], t + m.c[j_need + 1] * h)
            F[j_need + 1] = Fj
            nf += 1
        end
        us[i + 1] = shu_osher_combine(u, us, F, @view(m.alpha[i, 1:i]), @view(m.beta[i, 1:i]), h)
    end
    return us, F, nf
end

"""
`Σⱼ (αⱼ u⁽ʲ⁾ + (h βⱼ) F⁽ʲ⁾)` as one flat sum over ascending `j`, the α term
before the β term for each `j`, zero coefficients skipped, starting from the
first non-skipped term: `((α₀u⁽⁰⁾ + (hβ₀)F⁽⁰⁾) + α₁u⁽¹⁾) + …`.
"""
function shu_osher_combine(u, us, F, α, β, h)
    T = eltype(u)
    acc = nothing
    function add!(coef, x)
        ct = convert(T, coef)
        if acc === nothing
            acc = similar(u)
            @inbounds for k in eachindex(acc)
                acc[k] = ct * x[k]
            end
        else
            @inbounds for k in eachindex(acc)
                acc[k] = acc[k] + ct * x[k]
            end
        end
    end
    for j in eachindex(α)
        a = α[j]
        b = j <= length(β) ? β[j] : 0.0
        a != 0 && add!(a, us[j])
        b != 0 && add!(h * b, F[j])
    end
    return acc === nothing ? zero(u) : acc
end

function erk_step(m::ShuOsherERK, f!, u, t, h, k1, adaptive::Bool, tol; tk1 = t)
    us, F, nf = shu_osher_stages(m, f!, u, t, h)
    unew = us[m.s + 1]
    E = NaN
    if adaptive
        m.tab.error_estimate === :shu_osher_embedded ||
            throw(ConfigError("method $(m.tab.name) has no error estimate; give the leaf a dt (D52)"))
        ũ = shu_osher_combine(u, us, F, something(m.tab.embedded_alpha), something(m.tab.embedded_beta), h)
        σ = something(m.tab.embedded_scale)
        e = similar(u)
        @inbounds for k in eachindex(e)
            e[k] = σ * (ũ[k] - unew[k])
        end
        E = error_norm(e, u, unew, tol)
    end
    return StepResult(unew, E, nothing, nf)
end

"""
    erk_adjoint_step(m::ShuOsherERK, ops, u, p, t, h, λ) -> (λn, p̄, nf)

spec/adjoint.md §4.1, Shu–Osher form: with `ū⁽ˢ⁾ = λ`, for `j = s-1..0`,
`wⱼ = Σ_{i>j} (h βᵢⱼ) ū⁽ⁱ⁾` and `ū⁽ʲ⁾ = Σ_{i>j} αᵢⱼ ū⁽ⁱ⁾ + J(u⁽ʲ⁾)ᵀ wⱼ`, both
over ascending `i`; one `vjp_u` and one `vjp_p` per evaluated stage.
"""
function erk_adjoint_step(m::ShuOsherERK, ops, u, p, t, h, λ; tk1 = t)
    f! = (out, x, τ) -> eval_rhs!(out, ops, x, p, τ)
    us, _, nf = shu_osher_stages(m, f!, u, t, h)
    s = m.s
    N, P = length(u), length(p)
    ū = Vector{Vector{Float64}}(undef, s + 1)
    ū[s + 1] = copy(λ)
    pc = Vector{Union{Nothing, Vector{Float64}}}(nothing, s)
    for j in (s - 1):-1:0
        acc = zeros(N)
        w = nothing
        for i in (j + 1):s
            a = m.alpha[i, j + 1]
            b = m.beta[i, j + 1]
            a != 0 && _axpy!(acc, acc, a, ū[i + 1])
            if b != 0
                w === nothing ? (w = (h * b) .* ū[i + 1]) : _axpy!(w, w, h * b, ū[i + 1])
            end
        end
        if w !== nothing
            τ = t + m.c[j + 1] * h
            vjp_u!(ops, acc, us[j + 1], p, τ, w)
            pc[j + 1] = zeros(P)
            vjp_p!(ops, pc[j + 1], us[j + 1], p, τ, w)
        end
        ū[j + 1] = acc
    end
    p̄ = zeros(P)
    for j in 1:s
        pc[j] === nothing || (p̄ .+= pc[j])
    end
    return ū[1], p̄, nf
end
