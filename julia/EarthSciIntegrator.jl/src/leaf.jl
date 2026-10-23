# A leaf invocation: the step loop of spec/methods.md §8–§12 (forward), and
# its replay from the tape (spec/tape.md §3).

"""
    stop_rule(t, h, ts) -> (h′, isstop)

spec/methods.md §10: with `d = ts - t` and `tol = 100 eps(max(|t|, |ts|))`, the
step is ordinary when `h + tol < d`; otherwise it is a stop step of size
`min(h, d)` whose end time is `ts` exactly.
"""
@inline function stop_rule(t::Float64, h::Float64, ts::Float64)
    d = ts - t
    tol = 100 * eps(max(abs(t), abs(ts)))
    h + tol < d && return h, false
    return min(h, d), true
end

"""The first stop strictly after `t`; `stops` is sorted and ends with the interval end."""
@inline function next_stop(stops::Vector{Float64}, t::Float64)
    for s in stops
        s > t && return s
    end
    return last(stops)
end

"""Fixed substep size of a non-adaptive leaf over an interval of length `Δ`."""
fixed_dt(leaf::LeafNode, Δ) = leaf.dt isa Float64 ? leaf.dt : Δ

"""
    proposal_base(hc, h, isstop) -> size

The step size the controller's proposal after an accepted step is computed
from. For a step shortened to land on a stop or the interval end this is the
shortened step `h` actually taken, as OrdinaryDiffEqCore 4.19 does in effect
(its `loopheader!` calls `modify_dt_for_tstops!` again after the accept). This
follows the user's decision of 2026-10-23 to follow OrdinaryDiffEq here; the
merged spec/methods.md §8.6 still says the unshortened size `hc`, and is being
updated. This is the single place the rule lives: return `isstop ? hc : h`
for the other reading.
"""
proposal_base(hc, h, isstop::Bool) = h

"""
    LeafOutcome

A forward leaf invocation's result: end state, its tape record, the number of
right-hand-side evaluations, and the states at interior stops.
"""
struct LeafOutcome
    u::Vector{Float64}
    record::LeafRecord
    nf::Int
    interior::Vector{Tuple{Float64, Vector{Float64}}}
end

_domain_violation(leaf::LeafNode, u) = any(j -> u[j] < 0, leaf.nonnegative)

"""Clip the listed DOFs at zero; returns the ascending 0-based indices of the DOFs clipped."""
function _clip!(leaf::LeafNode, u)
    clipped = Int[]
    for j in leaf.nonnegative
        if u[j] < 0
            push!(clipped, j - 1)
            u[j] = zero(eltype(u))
        end
    end
    return sort!(clipped)
end

"""
    leaf_forward(leaf, cs, u, p, ta, tb, path, Δmacro, stops) -> LeafOutcome

Advance `leaf` over `[ta, tb]` from `u` with its controller (state `cs`,
persistent across invocations). `stops` are the stops inside the interval,
sorted, ending with `tb` (spec/methods.md §10). `p` already has the `node` and
`macro` bindings applied; the `leaf` binding is applied per step.
"""
function leaf_forward(leaf::LeafNode, cs::ControllerState, u0::Vector{Float64}, p, ta::Float64, tb::Float64,
        path::String, Δmacro::Float64, stops::Vector{Float64})
    tab = leaf.tab
    adaptive = is_adaptive(leaf)
    cc = tab.controller
    Δ = tb - ta
    dtmax = something(leaf.dtmax, Δ)
    dtmin = leaf.dtmin
    rec = LeafRecord(path, ta, Δ, NTuple{2, Float64}[], 0, 0, "Success", Pair{Int, Vector{Int}}[])
    interior = Tuple{Float64, Vector{Float64}}[]
    u = copy(u0)
    t = ta
    nf = 0
    fsal = nothing                          # invalid at entry (spec/methods.md §3.3)
    pstep(h) = apply_bindings(p, filter(b -> b.second === :leaf, leaf.bindings), h, 0.0, 0.0)
    fp(pp) = (out, x, τ) -> eval_rhs!(out, leaf.ops, x, pp, τ)
    if adaptive && isnan(cs.h)
        if leaf.dt0 !== nothing
            cs.h = fix_dt_at_bounds(leaf.dt0, t, dtmin, dtmax)
        else
            cs.h, n0 = initial_step(fp(p), u, t, tab.order, leaf.tol, dtmin, dtmax)
            nf += n0
        end
    end
    hctrl = adaptive ? cs.h : NaN
    iter = 0
    last_rejected = false
    istop = 1
    while true
        # Record states at interior stops already reached.
        while istop < length(stops) && stops[istop] <= t
            stops[istop] == t && push!(interior, (t, copy(u)))
            istop += 1
        end
        t >= tb && break
        ts = stops[istop]
        hc = adaptive ? fix_dt_at_bounds(hctrl, t, dtmin, dtmax) : fixed_dt(leaf, Δ)
        h, isstop = stop_rule(t, hc, ts)
        iter += 1
        # Limits and return codes (spec/methods.md §12), checked before the attempt
        # in OrdinaryDiffEq's order (SciMLBase `check_error`).
        if isnan(h)
            rec.retcode = "DtNaN"
            break
        elseif iter > leaf.maxiters
            rec.retcode = "MaxIters"
            break
        elseif adaptive && dtmin > 0 && h <= dtmin && (last_rejected || t + h < ts)
            rec.retcode = "DtLessThanMin"
            break
        elseif adaptive && last_rejected && h <= eps(abs(t))
            rec.retcode = "Unstable"
            break
        end
        if isstop && h < eps(abs(t))
            # A stop step shorter than eps(|t|) is accepted without computing it.
            push!(rec.steps, (t, h))
            rec.accepted += 1
            t = ts
            fsal = nothing
            last_rejected = false
            continue
        end
        pp = pstep(h)
        res = erk_step(leaf.method, fp(pp), u, t, h, fsal, adaptive, leaf.tol)
        nf += res.nf
        unew = res.u
        violation = leaf.on_violation === :reject && _domain_violation(leaf, unew)
        if adaptive
            E = res.E
            if !isfinite(E) || violation
                # spec/methods.md §7 and §8.5: h ← h·qmin.
                hctrl = h * cc.qmin
                rec.rejected += 1
                last_rejected = true
                continue
            end
            q = controller_factor!(cs, leaf.controller, cc, E, tab)
            if accepts(leaf.controller, cc, E, q)
                tnew = isstop ? ts : t + h
                hhat = controller_accept!(cs, leaf.controller, cc, E, q, proposal_base(hc, h, isstop))
                cs.accepted_any = true
                hctrl = propose(hhat, tnew, dtmin, dtmax)
            else
                hctrl = controller_reject(cs, leaf.controller, cc, h)
                rec.rejected += 1
                last_rejected = true
                continue
            end
        else
            if !all(x -> isfinite(x), unew)
                rec.retcode = "Unstable"
                break
            end
            if violation
                rec.retcode = "DomainViolation"
                break
            end
        end
        # Accept.
        tnew = isstop ? ts : t + h
        push!(rec.steps, (t, h))
        rec.accepted += 1
        clipped = false
        if leaf.on_violation === :clip && _domain_violation(leaf, unew)
            unew = copy(unew)
            push!(rec.clips, (length(rec.steps) - 1) => _clip!(leaf, unew))
            clipped = true
        end
        # First same as last: reuse f(u_{n+1}) only if it was evaluated at the
        # step's end time and the state was not clipped (spec/methods.md §3.3).
        # First same as last: reuse f(u_{n+1}, tₙ + h) unless the state was clipped
        # (spec/methods.md §3.3). It is reused even when tₙ₊₁ was snapped to a
        # stop, as OrdinaryDiffEq does; replay reproduces this (leaf_replay).
        fsal = (res.fsal !== nothing && !clipped) ? res.fsal : nothing
        u = unew
        t = tnew
        last_rejected = false
    end
    adaptive && (cs.h = hctrl)
    return LeafOutcome(u, rec, nf, interior)
end

"""
    leaf_replay(leaf, u, p, rec, stops; states=nothing, seeds=nothing) -> u

Re-execute a leaf invocation with the step sequence of its tape record
(spec/tape.md §3): no controller, each `hₙ` as recorded, recorded clip masks
applied. Generic over the element types of `u` and `p` (the frozen-tape
tangent pushes `Dual` numbers through it). With `states`, pushes
`(uₙ, tₙ, hₙ, pₙ, maskₙ, tₙ₊₁, t_K1)` for every step, `t_K1` the time
stage 1 was evaluated at (`tₙ₋₁ + hₙ₋₁` when it was reused, spec/methods.md §3.3); `visit(tₙ₊₁, uₙ₊₁)` is called
after every step.
"""
function leaf_replay(leaf::LeafNode, u0, p, rec::LeafRecord, stops::Vector{Float64}; states = nothing, visit = nothing)
    u = copy(u0)
    clipmap = Dict(rec.clips)
    lb = filter(b -> b.second === :leaf, leaf.bindings)
    fsal_method = leaf.tab.first_same_as_last
    tk1_next = nothing      # time of a reused first stage, as in leaf_forward
    for (n, (t, h)) in enumerate(rec.steps)
        ts = next_stop(stops, t)
        _, isstop = stop_rule(t, h, ts)
        tnew = isstop ? ts : t + h
        pp = apply_bindings(p, lb, h, 0.0, 0.0)
        mask = get(clipmap, n - 1, nothing)    # 0-based clipped DOFs of this step
        tk1 = something(tk1_next, t)
        states === nothing || push!(states, (copy(u), t, h, pp, mask, tnew, tk1))
        if isstop && h < eps(abs(t))
            tk1_next = nothing
        else
            u = leaf_step(leaf, u, pp, t, h, mask, tk1)
            tk1_next = fsal_method && mask === nothing ? t + h : nothing
        end
        visit === nothing || visit(tnew, u)
    end
    return u
end

"""
    leaf_step(leaf, u, p, t, h, mask) -> u′

One replayed step (stages from `u`; `mask` is `nothing` or the 0-based
indices of the DOFs the step clipped, as recorded on the tape),
generic over element types; the frozen-tape step map `Φₙ` of spec/adjoint.md §1.
"""
function leaf_step(leaf::LeafNode, u, p, t, h, mask, tk1 = t)
    f! = (out, x, τ) -> eval_rhs!(out, leaf.ops, x, p, τ)
    u1 = erk_step(leaf.method, f!, u, t, h, nothing, false, nothing; tk1).u
    if mask !== nothing
        u1 = copy(u1)
        for i in mask
            u1[i + 1] = zero(eltype(u1))
        end
    end
    return u1
end

"""
    leaf_step_adjoint(leaf, u, p, t, h, mask, λ) -> (λₙ, p̄)

The adjoint of [`leaf_step`](@ref): the clip mask first (spec/adjoint.md §5.7),
then the method's adjoint step. A skipped (sub-eps) stop step is the identity.
"""
function leaf_step_adjoint(leaf::LeafNode, u, p, t, h, mask, λ, skipped::Bool = false, tk1 = t)
    if mask !== nothing
        λ = copy(λ)
        for i in mask
            λ[i + 1] = 0.0
        end
    end
    skipped && return copy(λ), zeros(length(p))
    λn, p̄, _ = erk_adjoint_step(leaf.method, leaf.ops, u, p, t, h, λ; tk1)
    return λn, p̄
end
