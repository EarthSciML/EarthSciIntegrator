# OrdinaryDiffEq as the tier-2 oracle (D31): the same method run with our step
# sequence imposed (`adaptive = false`, `tstops` from our tape), and
# SciMLSensitivity's ReverseDiffAdjoint through it (spec/adjoint.md §7.3).

using SciMLBase: ODEProblem, solve, remake
using OrdinaryDiffEqLowOrderRK: Euler, Heun, RK4, BS3
using OrdinaryDiffEqTsit5: Tsit5
using OrdinaryDiffEqSSPRK: SSPRK33, SSPRK43, SSPRK104
using SciMLSensitivity: ReverseDiffAdjoint
import Zygote

"""OrdinaryDiffEq's algorithm for a method (the table's `ordinarydiffeq_algorithm`)."""
function ode_alg(alg::AbstractString)
    name = TABLEAUS[alg].ordinarydiffeq_algorithm
    name === nothing && error("$alg has no OrdinaryDiffEq counterpart")
    return Dict("Euler" => Euler(), "Heun" => Heun(), "RK4" => RK4(), "BS3" => BS3(), "Tsit5" => Tsit5(),
        "SSPRK33" => SSPRK33(), "SSPRK43" => SSPRK43(), "SSPRK104" => SSPRK104())[name]
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
