# OrdinaryDiffEq as the tier-2 oracle (D31): the same method run with our step
# sequence imposed (`adaptive = false`, `tstops` from our tape), and
# SciMLSensitivity's ReverseDiffAdjoint through it (spec/adjoint.md §7.3).

import SciMLBase
using SciMLBase: ODEProblem, solve, remake
using OrdinaryDiffEqLowOrderRK: Euler, Heun, RK4, BS3
using OrdinaryDiffEqTsit5: Tsit5
using OrdinaryDiffEqSSPRK: SSPRK33, SSPRK43, SSPRK104
using OrdinaryDiffEqRosenbrock: Rosenbrock23, ROS2, ROS3P, Rodas3, Rodas4, Rodas5P
using SciMLSensitivity: ReverseDiffAdjoint
import Zygote
import ReverseDiff

"""OrdinaryDiffEq's algorithm for a method (the table's `ordinarydiffeq_algorithm`)."""
function ode_alg(alg::AbstractString)
    name = TABLEAUS[alg].ordinarydiffeq_algorithm
    name === nothing && error("$alg has no OrdinaryDiffEq counterpart")
    return Dict("Euler" => Euler(), "Heun" => Heun(), "RK4" => RK4(), "BS3" => BS3(), "Tsit5" => Tsit5(),
        "SSPRK33" => SSPRK33(), "SSPRK43" => SSPRK43(), "SSPRK104" => SSPRK104(),
        # The ODEFunction carries `jac` and `tgrad` from the operators, which
        # OrdinaryDiffEq's Rosenbrock methods use instead of differentiating f.
        "Rosenbrock23" => Rosenbrock23(), "ROS2" => ROS2(), "ROS3P" => ROS3P(), "Rodas3" => Rodas3(),
        "Rodas4" => Rodas4(), "Rodas5P" => Rodas5P())[name]
end

"""
The accepted steps of every invocation of a single-leaf run, and the step end
times OrdinaryDiffEq must stop at to reproduce them.
"""
function step_sequence(run)
    run.root isa EarthSciIntegrator.LeafNode || error("the oracle needs a single-leaf schedule")
    steps = NTuple{2, Float64}[]
    for r in run.tape.records
        r isa EarthSciIntegrator.LeafRecord && append!(steps, r.steps)
    end
    ends = [s[1] for s in steps[2:end]]
    push!(ends, run.tspan[2])
    return steps, ends
end

"""
    oracle_problem(run; inplace=true) -> (prob, solve keywords)

The run's problem in OrdinaryDiffEq with our step sequence imposed: one step to
each recorded step end (`adaptive = false`, `dt` the whole interval, `tstops`
the step end times).
"""
function oracle_problem(run; inplace = true)
    _, ends = step_sequence(run)
    f = odefunction(run.root; inplace)
    prob = ODEProblem(f, copy(run.u0), run.tspan, copy(run.p))
    kw = (; adaptive = false, dt = run.tspan[2] - run.tspan[1], tstops = ends, saveat = copy(run.output_times),
        save_start = false, save_end = true)
    return prob, kw
end

"""Solve the run's problem in OrdinaryDiffEq with our step sequence imposed."""
function oracle_solve(run, alg)
    prob, kw = oracle_problem(run)
    return solve(prob, alg; kw...)
end

# ReverseDiffAdjoint through OrdinaryDiffEq's out-of-place Rosenbrock methods.
# OrdinaryDiffEqRosenbrock 2.7.5 stores the time derivative in a slot typed
# like the state (`JacReuseState.cached_dT`, `src/rosenbrock_caches.jl:51`),
# which under ReverseDiff is a `TrackedArray`; a `tgrad` (ours, or its own
# ForwardDiff one) that returns a `Vector` of `TrackedReal`s fails to convert.
# So the reverse_diff problem's `tgrad` is a ReverseDiff primitive: its value
# is the operators' `ft`, and its pullback takes ForwardDiff Jacobians of that
# `ft` in u and p, which keeps the oracle independent of the hand-derived
# `ft_vjp_u` and `ft_vjp_p` the adjoint uses. (The in-place path fails too, in
# LinearSolve's typed cache.)
#
# A functional over time is integrated, as in our leaf, as the auxiliary state
# q' = ψ(u) appended to the state (spec/adjoint.md §2.1); with the step
# sequence imposed, q's presence does not change OrdinaryDiffEq's steps.
_ftvec(ops, u, p, t) = (d = zeros(promote_type(eltype(u), eltype(p)), length(u)); EarthSciIntegrator.ft!(ops, d, u, p, t); d)
"""f_t of the state, with a zero appended for q when `aug`."""
struct FtOf{O}
    ops::O
    aug::Bool
end
(g::FtOf)(u, p, t) = g.aug ? vcat(_ftvec(g.ops, u[1:(end - 1)], p, t), zero(promote_type(eltype(u), eltype(p)))) :
                      _ftvec(g.ops, u, p, t)
tracked_ft(g, u, p, t) = g(u, p, t)
tracked_ft(g, u::ReverseDiff.TrackedArray, p::ReverseDiff.TrackedArray, t) = ReverseDiff.track(tracked_ft, g, u, p, t)
ReverseDiff.@grad function tracked_ft(g, u, p, t)
    uv, pv = ReverseDiff.value(u), ReverseDiff.value(p)
    Ju = ForwardDiff.jacobian(x -> g(x, pv, t), uv)
    Jp = ForwardDiff.jacobian(q -> g(uv, q, t), pv)
    return g(uv, pv, t), Δ -> (nothing, Ju' * Δ, Jp' * Δ, nothing)
end

"""
The run's out-of-place problem for ReverseDiffAdjoint, with our step sequence
imposed; for a functional over time, on the state (u, q).
"""
function reverse_diff_problem(run)
    prob, kw = oracle_problem(run; inplace = false)
    ops = run.root.ops
    ψ = run.root.ψ
    if ψ === nothing
        f = SciMLBase.ODEFunction{false}(prob.f.f; jac = prob.f.jac, tgrad = (u, p, t) -> tracked_ft(FtOf(ops, false), u, p, t))
        return remake(prob; f), kw
    end
    f0, J0 = prob.f.f, prob.f.jac
    faug(x, p, t) = (u = x[1:(end - 1)]; vcat(f0(u, p, t), ψ(u)))
    jaug = function (x, p, t)
        N = length(x) - 1
        Ju = J0(x[1:N], p, t)
        J = zeros(eltype(Ju), N + 1, N + 1)
        J[1:N, 1:N] .= Ju
        J[N + 1, 1:N] .= ψ.w
        return J
    end
    f = SciMLBase.ODEFunction{false}(faug; jac = jaug, tgrad = (u, p, t) -> tracked_ft(FtOf(ops, true), u, p, t))
    return remake(prob; f, u0 = vcat(prob.u0, 0.0)), kw
end
