using EarthSciIntegrator
using Test

# A run config is a JSON object. This one runs the `forced_decay` toy problem
# (u' = -k u + sin(ωt)) with RK4 at a fixed step of 0.1 over t = 0 … 2.
config = Dict{String, Any}(
    "esm_integrator" => "0.1.0",
    "model" => "toy:forced_decay",
    "schedule" => Dict{String, Any}("leaf" => "decay", "alg" => "RK4", "dt" => 0.1),
    "step" => Dict{String, Any}("dt" => 2),
    "adjoint" => Dict{String, Any}(
        "functional" => Dict{String, Any}("sum" => "u", "over" => "final"),
        "wrt" => ["u0", "params:numeric"],
    ),
)

# region: toy ops=toy_problem
prob = toy_problem("toy:forced_decay")
@test prob.p == [2.0, 3.0]
@test prob.u0 == [1.0]
# endregion

# region: forward ops=build_run,forward
run = build_run(config; tspan = (0.0, 2.0))
forward!(run)
@test run.retcode == "Success" # hide
u_final = run.outputs[end][2]
# Exact solution of u' = -k u + sin(ωt), u(0) = 1.
k, ω = prob.p
A(t) = (k * sin(ω * t) - ω * cos(ω * t)) / (k^2 + ω^2)
u_exact = (1 - A(0)) * exp(-k * 2) + A(2)
@test isapprox(only(u_final), u_exact; rtol = 1.0e-4)
# endregion

# region: gradient ops=gradient
J, ū0, p̄ = gradient(run)
# The functional is u(2), so ū0 is d u(2) / d u(0) = exp(-k t).
@test isapprox(only(ū0), exp(-k * 2); rtol = 1.0e-4)
@test length(p̄) == length(prob.p)
# endregion

# region: replay ops=replay
u_replayed, outputs, q = replay(run)
@test u_replayed == u_final
# endregion

# region: refusal ops=build_run,config_error
bad = deepcopy(config)
bad["schedule"]["alg"] = "Bogus"
@test_throws ConfigError build_run(bad; tspan = (0.0, 2.0))
# endregion

println("forward_and_gradient: ok")
