# Checks beyond cases.json that this stage's code paths need: every explicit
# method's adjoint on nonlinear and block-diagonal problems, the combinator
# adjoints with explicit leaves (cases.json exercises them only with Rodas4
# leaves, which arrive in a later stage), the I and PID controllers, domain
# clipping and its adjoint, and the return codes.

xcase(id, model, schedule, step, tspan; adjoint = nothing, saveat = nothing, checks = Dict{String, Any}[]) = begin
    cfg = Dict{String, Any}("esm_integrator" => "0.1.0", "model" => model, "schedule" => schedule, "step" => step)
    adjoint === nothing || (cfg["adjoint"] = adjoint)
    c = Dict{String, Any}("id" => id, "run_config" => cfg, "tspan" => tspan, "checks" => checks)
    saveat === nothing || (c["saveat"] = saveat)
    c
end
const XGRAD = [Dict{String, Any}("check" => "dot_product", "rtol" => 1e-12, "scale" => "terms"),
    Dict{String, Any}("check" => "fd_gradient", "rtol" => 1e-6, "epsilons" => [1e-3, 1e-4, 1e-5]),
    Dict{String, Any}("check" => "replay")]
xgrad(q, over = "final", how = "sum") = Dict{String, Any}("functional" => Dict{String, Any}(how => q, "over" => over),
    "wrt" => ["u0", "params:numeric"])

function run_xcase(c)
    @testset "$(c["id"])" begin
        run = forward!(build_case(c))
        @test run.retcode == "Success"
        for chk in c["checks"]
            CHECKS[chk["check"]](c, run, chk)
        end
    end
end

const EXPLICIT = ["ForwardEuler", "Heun", "RK4", "BS3", "Tsit5", "WRF_RK3", "SSPRK33", "SSPRK43", "SSPRK104"]
const ADAPTIVE_EXPLICIT = ["Heun", "RK4", "BS3", "Tsit5", "SSPRK43"]

@testset "explicit methods beyond cases.json" begin
    n0 = length(OUTCOMES)
    for alg in EXPLICIT
        # Nonlinear, autonomous; gradient of the final y and of y at every output time.
        run_xcase(xcase("x.van_der_pol.$alg.fixed", "toy:van_der_pol/nonstiff",
            Dict("leaf" => "oscillator", "alg" => alg, "dt" => 0.05), Dict("dt" => 0.5), [0, 2.0];
            adjoint = xgrad("y", "saveat"), saveat = [1.0, 2.0], checks = XGRAD))
        # Block-diagonal, time-dependent; gradient of the mean O3.
        run_xcase(xcase("x.photostationary.$alg.fixed", "toy:photostationary_cells",
            Dict("leaf" => "photochemistry", "alg" => alg, "dt" => 10.0), Dict("dt" => 600), [0, 1200.0];
            adjoint = xgrad("O3", "final", "mean"), checks = XGRAD))
    end
    for alg in ADAPTIVE_EXPLICIT
        # Adaptive, with interior output times as stops inside the root leaf.
        tol = Dict("reltol" => 1e-7, "abstol" => 1e-9)
        run_xcase(xcase("x.van_der_pol.$alg.adaptive", "toy:van_der_pol/nonstiff",
            Dict("leaf" => "oscillator", "alg" => alg, "control" => tol), Dict("dt" => 6.3), [0, 6.3];
            adjoint = xgrad("y", "saveat"), saveat = [1.0, 3.0, 6.3],
            checks = vcat(XGRAD, [Dict{String, Any}("check" => "oracle", "rtol" => 1e-12)])))
    end
    for ctrl in ("I", "PID")
        run_xcase(xcase("x.van_der_pol.Tsit5.$ctrl", "toy:van_der_pol/nonstiff",
            Dict("leaf" => "oscillator", "alg" => "Tsit5", "control" => Dict("reltol" => 1e-8, "abstol" => 1e-10, "controller" => ctrl)),
            Dict("dt" => 6.3), [0, 6.3]; saveat = [1.0, 3.0, 6.3],
            checks = [Dict{String, Any}("check" => "oracle", "rtol" => 1e-12), Dict{String, Any}("check" => "replay")]))
    end
    # Combinators with explicit leaves: sequence, strang, subcycle, adaptive leaves inside.
    transport = Dict("leaf" => "transport", "alg" => "SSPRK43", "control" => Dict("reltol" => 1e-8, "abstol" => 1e-10))
    reaction = Dict("leaf" => "reaction", "alg" => "Tsit5", "control" => Dict("reltol" => 1e-8, "abstol" => 1e-10))
    for (name, node) in [("lie", Dict("sequence" => [transport, reaction])), ("strang", Dict("strang" => [transport, reaction])),
        ("subcycle", Dict("sequence" => [Dict("subcycle" => Dict("leaf" => "transport", "alg" => "SSPRK33", "dt" => "step"), "n" => 4),
            Dict("leaf" => "reaction", "alg" => "RK4", "dt" => 2e-3)]))]
        run_xcase(xcase("x.advection_reaction.$name", "toy:advection_reaction", node, Dict("dt" => 0.025), [0, 0.1];
            adjoint = xgrad("B"), checks = vcat(XGRAD, [Dict{String, Any}("check" => "invariant", "rtol" => 1e-12)])))
    end
    # Splitting order with explicit leaves against the unsplit reference.
    for (name, node, order) in [("lie", Dict("sequence" => [transport, reaction]), 1), ("strang", Dict("strang" => [transport, reaction]), 2)]
        c = xcase("advection_reaction.$name.order", "toy:advection_reaction", node, Dict("dt" => 0.05), [0, 0.5];
            checks = [Dict{String, Any}("check" => "order", "vary" => "macro", "dts" => [0.05, 0.025, 0.0125, 0.00625],
                "expected" => order, "against" => "reference")])
        c["id"] = "advection_reaction.$name.order"       # reads that case's reference
        run_xcase(c)
    end
    # Domain clipping (forced_decay crosses zero) and its adjoint.
    run_xcase(xcase("x.forced_decay.clip", "toy:forced_decay",
        Dict("leaf" => "decay", "alg" => "SSPRK43", "control" => Dict("reltol" => 1e-6, "abstol" => 1e-8),
            "domain" => Dict("nonnegative" => "all", "on_violation" => "clip")), Dict("dt" => 2), [0, 2.0];
        adjoint = xgrad("u"), checks = XGRAD))
    # bind_dt: the leaf's step bound to a parameter, which gets no cotangent.
    run_xcase(xcase("x.forced_decay.bind_dt", "toy:forced_decay",
        Dict("leaf" => "decay", "alg" => "RK4", "dt" => 0.1, "bind_dt" => Dict("omega" => "node")), Dict("dt" => 0.5), [0, 2.0];
        adjoint = xgrad("u"), checks = XGRAD))
    for o in OUTCOMES[(n0 + 1):end]
        println(rpad(o.passed ? "  pass" : "  FAIL", 7), rpad(o.case, 42), rpad(o.check, 15), o.detail)
    end
end

@testset "return codes" begin
    base(leaf) = xcase("rc", "toy:forced_decay", leaf, Dict("dt" => 2), [0, 2.0])
    clipped = run -> any(r -> r isa LeafRecord && !isempty(r.clips), run.tape.records)
    r = forward!(build_case(base(Dict("leaf" => "decay", "alg" => "Tsit5", "dt" => 0.1,
        "domain" => Dict("nonnegative" => ["u"])))))
    @test r.retcode == "DomainViolation"
    @test occursin("root", r.failure)
    r = forward!(build_case(base(Dict("leaf" => "decay", "alg" => "Tsit5", "control" => Dict("reltol" => 1e-6, "abstol" => 1e-8),
        "domain" => Dict("nonnegative" => "all")))))
    @test r.retcode in ("Unstable", "DtLessThanMin")
    r = forward!(build_case(base(Dict("leaf" => "decay", "alg" => "Tsit5", "control" => Dict("reltol" => 1e-6, "abstol" => 1e-8,
        "dtmin" => 1e-3), "domain" => Dict("nonnegative" => "all")))))
    @test r.retcode == "DtLessThanMin"
    r = forward!(build_case(base(Dict("leaf" => "decay", "alg" => "Tsit5", "control" => Dict("reltol" => 1e-10, "maxiters" => 5)))))
    @test r.retcode == "MaxIters"
    r = forward!(build_case(base(Dict("leaf" => "decay", "alg" => "ForwardEuler", "dt" => 2.0))))
    @test r.retcode == "Success"
    # Explicit Euler far beyond its stability limit on stiff van der Pol: non-finite state.
    c = xcase("rc", "toy:van_der_pol/stiff", Dict("leaf" => "oscillator", "alg" => "ForwardEuler", "dt" => 0.1), Dict("dt" => 1), [0, 1.0])
    @test forward!(build_case(c)).retcode == "Unstable"
end
