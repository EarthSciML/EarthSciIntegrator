# Measures the global error OrdinaryDiffEq's SDIRK methods achieve against the
# requested tolerance with the smoothed error estimate (smooth_est = true, its
# default) and without it. Evidence for D56 (spec/methods.md §5.5).
# Run: julia --project=tools/tableaus tools/tableaus/sdirk_error_estimate_check.jl
using OrdinaryDiffEqSDIRK, OrdinaryDiffEqRosenbrock
# forced decay (non-stiff) and a stiff Prothero-Robinson problem, both with exact solutions
fd(u,p,t) = -2u + sin(3t)
A(t) = (2sin(3t) - 3cos(3t))/13
fd_exact(t) = (1 - A(0))*exp(-2t) + A(t)
pr(u,p,t) = -1e4*(u - cos(t)) - sin(t)
pr_exact(t) = cos(t)
probs = [("forced_decay", ODEProblem(fd, 1.0, (0.0, 2.0)), fd_exact(2.0)),
         ("prothero_robinson", ODEProblem(pr, 1.0, (0.0, 10.0)), pr_exact(10.0))]
println(rpad("problem",18), rpad("alg",10), rpad("tol",8), rpad("smooth",8), rpad("err/tol",12), "steps")
for (name, prob, ex) in probs, alg in (TRBDF2, SDIRK2, Kvaerno3, Kvaerno4), tol in (1e-4, 1e-6, 1e-8), smooth in (true, false)
    sol = solve(prob, alg(smooth_est=smooth); reltol=tol, abstol=tol, save_everystep=false)
    err = abs(sol.u[end] - ex)
    println(rpad(name,18), rpad(string(alg),10), rpad(tol,8), rpad(smooth,8), rpad(round(err/tol, sigdigits=3),12), sol.stats.naccept)
end
