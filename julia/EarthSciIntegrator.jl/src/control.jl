# Error norm, step-size controllers and the first step (spec/methods.md §7–§9).

"""
    primal(x)

The value of `x` with any tangent stripped. Error norms and controllers act on
primal values only (spec/methods.md §7, spec/adjoint.md §2.3 with
`dt_tangents = false`); a `ForwardDiff.Dual`'s partials are stripped.
"""
primal(x::Real) = x
primal(x::ForwardDiff.Dual) = primal(ForwardDiff.value(x))
_f64(x) = Float64(primal(x))

"""
    Tolerances(reltol, abstol, dofs=eachindex(abstol))

`reltol` is a scalar; `abstol` holds one absolute tolerance per entry of the
state vector; `dofs` are the (1-based, ascending) DOFs the leaf's operators
write, over which the error norm and the first step sum (spec/methods.md §7, §9).
"""
struct Tolerances
    reltol::Float64
    abstol::Vector{Float64}
    dofs::Vector{Int}
end
Tolerances(reltol, abstol) = Tolerances(reltol, abstol, collect(eachindex(abstol)))

"""
    error_norm(e, un, un1, tol) -> Float64

spec/methods.md §7: `sqrt(Σⱼ (eⱼ/scⱼ)² / N)` with `scⱼ = abstolⱼ + reltol·max(|u_{n,j}|, |u_{n+1,j}|)`,
over the `N` DOFs the leaf writes (`tol.dofs`), in f64 whatever the state type,
summed sequentially in ascending DOF order with no fused multiply-add.
"""
function error_norm(e, un, un1, tol::Tolerances)
    N = length(tol.dofs)
    s = 0.0
    @inbounds for j in tol.dofs
        sc = tol.abstol[j] + tol.reltol * max(abs(_f64(un[j])), abs(_f64(un1[j])))
        r = _f64(e[j]) / sc
        s = s + r * r
    end
    return sqrt(s / N)
end

"""The controllers of spec/methods.md §8."""
@enum ControllerKind PI_CONTROLLER I_CONTROLLER PID_CONTROLLER

function controller_kind(s::AbstractString)
    s == "PI" && return PI_CONTROLLER
    s == "I" && return I_CONTROLLER
    s == "PID" && return PID_CONTROLLER
    throw(ConfigError("unknown controller '$s' (PI, I or PID)"))
end

"""
    ControllerState

The persistent state of one leaf's controller (spec/methods.md §8.6): the
proposed next step, the error history (`Eold` for PI, the PID `ε₂`, `ε₃`),
and whether the leaf has accepted a step. The Newton iteration keeps no state
across step attempts (spec/methods.md §5.2, D63). It persists
across the leaf's invocations and macro steps.
"""
mutable struct ControllerState
    h::Float64              # proposal; NaN before the first step is chosen
    Eold::Float64           # PI error history
    q11::Float64            # PI: E^β₁ of the last attempt (used on rejection)
    q::Float64              # I: the last attempt's factor (used on rejection)
    pid_eps::NTuple{3, Float64}
    dt_factor::Float64      # PID: the last attempt's factor
    accepted_any::Bool
end

function ControllerState(cc::Union{Nothing, ControllerConstants})
    qoldinit = cc === nothing ? 1.0e-4 : cc.qoldinit
    return ControllerState(NaN, qoldinit, 1.0, 1.0, (1.0, 1.0, 1.0), 1.0, false)
end

Base.copy(s::ControllerState) = ControllerState(s.h, s.Eold, s.q11, s.q, s.pid_eps, s.dt_factor, s.accepted_any)
function Base.:(==)(a::ControllerState, b::ControllerState)
    return all(f -> isequal(getfield(a, f), getfield(b, f)), fieldnames(ControllerState))
end

_qmax(cs::ControllerState, cc::ControllerConstants) = cs.accepted_any ? cc.qmax : cc.qmax_first_step

"""
    controller_factor!(cs, kind, cc, E, tab) -> q

The controller's factor for scaled error `E` (finite): for PI and I the `q`
that divides `h` (§8.2, §8.3), for PID the `φ` that multiplies it (§8.4).
Arithmetic follows OrdinaryDiffEqCore `src/integrators/controllers.jl`.
"""
function controller_factor!(cs::ControllerState, kind::ControllerKind, cc::ControllerConstants, E::Float64, tab::Tableau)
    if kind == PI_CONTROLLER
        qmax = _qmax(cs, cc)
        if iszero(E)
            q = inv(qmax)
        else
            q11 = E^cc.beta1
            q = q11 / cs.Eold^cc.beta2
            cs.q11 = q11
            q = clamp(q / cc.gamma, inv(qmax), inv(cc.qmin))
        end
        return q
    elseif kind == I_CONTROLLER
        qmax = _qmax(cs, cc)
        if iszero(E)
            q = inv(qmax)
        else
            expo = 1 / (something(tab.embedded_order) + 1)
            qtmp = E^expo / cc.gamma
            q = max(inv(qmax), min(inv(cc.qmin), qtmp))
            cs.q = q
        end
        return q
    else
        Ep = max(E, eps(Float64))
        ε1 = inv(Ep)
        _, ε2, ε3 = cs.pid_eps
        cs.pid_eps = (ε1, ε2, ε3)
        k = min(tab.order, something(tab.embedded_order)) + 1
        φ = ε1^(cc.beta1 / k) * ε2^(cc.beta2 / k) * ε3^(cc.pid_beta3 / k)
        φ = 1 + atan(φ - 1)
        cs.dt_factor = φ
        return φ
    end
end

"""Whether the controller accepts a step with scaled error `E` and factor `q` (§8.1)."""
accepts(kind::ControllerKind, cc::ControllerConstants, E, q) =
    kind == PID_CONTROLLER ? q >= cc.pid_accept_safety : E <= 1

"""
    controller_accept!(cs, kind, cc, E, q, h) -> ĥ

On acceptance: the dead band, the error history update, and the raw proposal
(before §8.6's rounding and bounds). `h` is the controller's own size, the
unshortened one for a stop step (§8.6).
"""
function controller_accept!(cs::ControllerState, kind::ControllerKind, cc::ControllerConstants, E, q, h)
    if kind == PID_CONTROLLER
        φ = q
        if cc.qsteady_min <= inv(φ) <= cc.qsteady_max
            φ = one(φ)
        end
        ε1, ε2, _ = cs.pid_eps
        cs.pid_eps = (ε1, ε1, ε2)
        return h * φ
    else
        if cc.qsteady_min <= q <= cc.qsteady_max
            q = one(q)
        end
        if kind == PI_CONTROLLER
            cs.Eold = max(E, cc.qoldinit)
        end
        return h / q
    end
end

"""On an error-test rejection: the next attempt's size (§8.2–§8.4)."""
function controller_reject(cs::ControllerState, kind::ControllerKind, cc::ControllerConstants, h)
    if kind == PI_CONTROLLER
        return h / min(inv(cc.qmin), cs.q11 / cc.gamma)
    elseif kind == I_CONTROLLER
        return h / cs.q
    else
        return h * cs.dt_factor
    end
end

"""
    propose(ĥ, t1, dtmin, dtmax) -> h

§8.6: `ĥ ← (t₁ + ĥ) - t₁`, then clamped to `[max(dtmin, eps(t₁)), dtmax]`
(OrdinaryDiffEqCore `calc_dt_propose!`).
"""
function propose(hhat, t1, dtmin, dtmax)
    hhat = abs(hhat)
    hhat = (t1 + hhat) - t1
    h = min(dtmax, hhat)
    return max(h, max(eps(t1), dtmin))
end

"""Clamp an attempt's size to `[max(dtmin, eps(t)), dtmax]` (OrdinaryDiffEqCore `fix_dt_at_bounds!`)."""
fix_dt_at_bounds(h, t, dtmin, dtmax) = max(min(dtmax, h), max(eps(t), dtmin))

"""
    initial_step(f!, u0, p, t0, order, tol, dtmin, dtmax) -> h

spec/methods.md §9, OrdinaryDiffEqCore `_ode_initdt_oop` in f64. `f!(out, u, t)`
overwrites `out` with the leaf's right-hand side. Returns `(h, nf)` with `nf`
the number of right-hand-side evaluations made.
"""
function initial_step(f!, u0, t0::Float64, order::Integer, tol::Tolerances, dtmin::Float64, dtmax::Float64)
    N = length(u0)
    D = tol.dofs                 # the DOFs the leaf writes (spec/methods.md §9 with §7)
    dtmin_ = nextfloat(max(dtmin, eps(t0)))
    smalldt = max(dtmin_, 1.0e-6)
    sk = zeros(N)
    for j in D
        sk[j] = tol.abstol[j] + abs(_f64(u0[j])) * tol.reltol
    end
    rms(x) = (
        s = 0.0; for v in x
            s = s + v * v
        end; sqrt(s / length(x))
    )
    d0 = rms([_f64(u0[j]) / sk[j] for j in D])
    f0 = similar(u0)
    f!(f0, u0, t0)
    nf = 1
    any(j -> isnan(_f64(f0[j])), D) && return dtmin_, nf
    d1 = rms([_f64(f0[j]) / sk[j] for j in D])
    isnan(d1) && return dtmin_, nf
    if d0 < 1.0e-5 || d1 < 1.0e-5
        h0 = smalldt
    else
        h0 = (d0 / d1) / 100
    end
    h0 = min(h0, dtmax)
    u1 = [u0[j] + h0 * f0[j] for j in 1:N]
    f1 = similar(u0)
    f!(f1, u1, t0 + h0)
    nf += 1
    f1[D] == f0[D] && return max(dtmin_, 100 * h0), nf
    d2 = rms([(_f64(f1[j]) - _f64(f0[j])) / sk[j] for j in D]) / h0
    m = max(d1, d2)
    if m <= 1.0e-15
        h1 = max(smalldt, h0 * 0.001)
    else
        h1 = 10.0^(-(2 + log10(m)) / order)
    end
    return max(dtmin_, min(100 * h0, h1, dtmax)), nf
end
