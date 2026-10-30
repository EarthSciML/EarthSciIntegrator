# Measures the global error SDIRK methods achieve against the requested
# tolerance under three error estimates: OrdinaryDiffEq's smoothed estimate
# (smooth_est = true, its default), OrdinaryDiffEq's unsmoothed one
# (smooth_est = false), and the estimate the spec adopts,
# (I - hγJ)⁻¹ e_raw (Hosea and Shampine). Evidence for D56 (spec/methods.md §5.5).
# Run: julia --project=tools/tableaus tools/tableaus/sdirk_error_estimate_check.jl
using OrdinaryDiffEqSDIRK, OrdinaryDiffEqRosenbrock, JSON
# Both problems are u' = λu + g(t) with exact solutions: forced decay (non-stiff)
# and a stiff Prothero-Robinson problem.
A(t) = (2sin(3t) - 3cos(3t)) / 13
probs = [
    ("forced_decay", -2.0, t -> sin(3t), 1.0, 2.0, (1 - A(0)) * exp(-4.0) + A(2.0)),
    ("prothero_robinson", -1.0e4, t -> 1.0e4cos(t) - sin(t), 1.0, 10.0, cos(10.0)),
]

# The spec's SDIRK step (spec/methods.md §5.1, §5.4) and PI controller (§8.2,
# §8.6) for scalar u' = λu + g(t). The stage equations are linear, so each is
# solved exactly; Newton (§5.2) would converge to the same stage.
function spec_solve(name, λ, g, u0, tf, tol)
    d = JSON.parsefile(joinpath(@__DIR__, "..", "..", "spec", "tableaus", "$name.json"))
    s = d["stages"]
    Am = [Float64(d["A"][i][j]) for i in 1:s, j in 1:s]
    b, bt, c = Float64.(d["b"]), Float64.(d["btilde"]), Float64.(d["c"])
    γ = Am[s, s]
    k = d["controller"]
    β1, β2, safety, qmin = k["beta1"], k["beta2"], k["gamma"], k["qmin"]
    t, u, h, Eold, accepted, prev_accepted = 0.0, u0, 1.0e-6, k["qoldinit"], 0, false
    while t < tf
        hh = min(h, tf - t)
        z = zeros(s)
        for i in 1:s
            τ = u + sum(Am[i, j] * z[j] for j in 1:(i - 1); init = 0.0)
            z[i] = i == 1 && d["explicit_first_stage"] ? hh * (λ * u + g(t)) :
                hh * (λ * τ + g(t + c[i] * hh)) / (1 - hh * γ * λ)
        end
        unew = u + sum(b .* z)
        e = sum(bt .* z) / (1 - hh * γ * λ)
        E = abs(e) / (tol + tol * max(abs(u), abs(unew)))
        qmax = accepted == 0 ? k["qmax_first_step"] : k["qmax"]
        q11 = E^β1
        q = E == 0 ? 1 / qmax : clamp(q11 / (Eold^β2 * safety), 1 / qmax, 1 / qmin)
        if E <= 1
            t, u, accepted = t + hh, unew, accepted + 1
            k["qsteady_min"] <= q <= k["qsteady_max"] && (q = 1.0)
            Eold = max(E, k["qoldinit"])
            h = (prev_accepted ? hh : h) / q  # shortened size after an accepted attempt (§8.6, D61)
            prev_accepted = true
        else
            h = hh / min(1 / qmin, q11 / safety)
            prev_accepted = false
        end
    end
    return u, accepted
end

println(rpad("problem", 18), rpad("alg", 10), rpad("tol", 8), rpad("estimate", 22), rpad("err/tol", 12), "steps")
row(name, alg, tol, est, err, n) =
    println(rpad(name, 18), rpad(alg, 10), rpad(tol, 8), rpad(est, 22), rpad(round(err / tol, sigdigits = 3), 12), n)
for (name, λ, g, u0, tf, ex) in probs, alg in (TRBDF2, SDIRK2, Kvaerno3, Kvaerno4), tol in (1.0e-4, 1.0e-6, 1.0e-8)
    prob = ODEProblem((u, p, t) -> λ * u + g(t), u0, (0.0, tf))
    for smooth in (true, false)
        sol = solve(prob, alg(smooth_est = smooth); reltol = tol, abstol = tol, save_everystep = false)
        row(name, string(alg), tol, smooth ? "ODE smoothed" : "ODE unsmoothed", abs(sol.u[end] - ex), sol.stats.naccept)
    end
    u, n = spec_solve(string(alg), λ, g, u0, tf, tol)
    row(name, string(alg), tol, "spec (Hosea-Shampine)", abs(u - ex), n)
end
