# Reference results for the M1 conformance cases (spec/toy-problems.md §3).
#
# For every case with a `reference` check, or an `order` check against the
# reference, this writes <case id>.json beside this script: a high-accuracy
# OrdinaryDiffEq solve of the case's problem (Rodas5P for stiff problems, Vern9
# for non-stiff ones, reltol = abstol = 1e-13), at t₀ and every output time,
# with `producer` "reference". Every operator the schedule names is summed into
# one right-hand side, so for advection_reaction the reference integrates the
# unsplit sum of transport and reaction and `order` checks measure the
# splitting error. The right-hand side and its Jacobian come from the Julia
# implementation's toy problems (EarthSciIntegrator.odefunction).
#
# Stiff: van_der_pol/stiff, robertson, photostationary_cells.
# Non-stiff: van_der_pol/nonstiff, advection_reaction.
#
# Usage, from the repository root:
#   julia --project=julia/EarthSciIntegrator.jl/test tests/conformance/m1/reference/make_reference.jl

using EarthSciIntegrator
using EarthSciIntegrator: OperatorSum, result_json, toy_problem
using JSON
using SciMLBase: ODEProblem, solve, ReturnCode
using OrdinaryDiffEqRosenbrock: Rodas5P
using OrdinaryDiffEqVerner: Vern9

const HERE = @__DIR__
const CASES = joinpath(HERE, "..", "cases.json")

"""Operator names a schedule node advances, in order of first appearance."""
function operator_names(node, acc = String[])
    if haskey(node, "leaf")
        for n in (node["leaf"] isa AbstractString ? [node["leaf"]] : node["leaf"])
            n in acc || push!(acc, n)
        end
    end
    for k in ("sequence", "strang")
        haskey(node, k) && foreach(c -> operator_names(c, acc), node[k])
    end
    haskey(node, "subcycle") && operator_names(node["subcycle"], acc)
    return acc
end

needs_reference(case) = any(c -> c["check"] == "reference" || (c["check"] == "order" && c["against"] == "reference"),
    case["checks"])

function reference(case)
    prob = toy_problem(case["run_config"]["model"])
    ops = OperatorSum([prob.operators[n] for n in operator_names(case["run_config"]["schedule"])])
    t0, tf = Float64.(case["tspan"])
    saveat = haskey(case, "saveat") ? Float64.(case["saveat"]) : [tf]
    tf in saveat || push!(saveat, tf)
    f = odefunction(ops)
    odeprob = ODEProblem(f, copy(prob.u0), (t0, tf), copy(prob.p))
    alg = prob.stiff ? Rodas5P() : Vern9()
    sol = solve(odeprob, alg; reltol = 1e-13, abstol = 1e-13, saveat, tstops = saveat, save_start = false, maxiters = 10^8)
    sol.retcode == ReturnCode.Success || error("$(case["id"]): reference solve ended with $(sol.retcode)")
    t = vcat(t0, sol.t)
    u = vcat([copy(prob.u0)], [copy(x) for x in sol.u])
    return result_json(case["id"], "reference", t, u; accepted = sol.stats.naccept, rejected = sol.stats.nreject,
        rhs_calls = sol.stats.nf)
end

function main()
    cases = JSON.parsefile(CASES)["cases"]
    n = 0
    for case in cases
        needs_reference(case) || continue
        path = joinpath(HERE, case["id"] * ".json")
        write(path, reference(case))
        println("wrote ", basename(path))
        n += 1
    end
    println("$n reference results")
end

main()
