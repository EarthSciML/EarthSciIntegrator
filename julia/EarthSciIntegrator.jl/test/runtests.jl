# Run with `julia --project=julia/EarthSciIntegrator.jl -e 'using Pkg; Pkg.test()'`
# from the repository root. Test-only dependencies are in test/Project.toml.
# Result files and tapes of the conformance cases are written to test/results/
# (or $EARTHSCIINTEGRATOR_RESULTS).

using EarthSciIntegrator
using EarthSciIntegrator: TABLEAUS, ControllerState, Tolerances, error_norm, controller_factor!, controller_accept!,
    controller_reject, accepts, propose, PI_CONTROLLER, I_CONTROLLER, PID_CONTROLLER, stop_rule, macro_bounds,
    UnsupportedError, ConfigError
using ForwardDiff
using JSON
using LinearAlgebra: dot
using Random
using Test

const FIXTURES = joinpath(@__DIR__, "..", "..", "..", "tests", "conformance", "run-configs", "valid")

@testset "EarthSciIntegrator" begin
    @testset "run-config fixtures target this spec version" begin
        files = filter(endswith(".json"), readdir(FIXTURES; join = true))
        @test !isempty(files)
        for f in files
            cfg = JSON.parsefile(f)
            @test VersionNumber(cfg["esm_integrator"]) == EarthSciIntegrator.SPEC_VERSION
        end
    end

    @testset "coefficient tables" begin
        dir = joinpath(@__DIR__, "..", "..", "..", "spec", "tableaus")
        names = sort([splitext(f)[1] for f in readdir(dir) if endswith(f, ".json")])
        @test length(names) == 27
        @test sort(collect(keys(TABLEAUS))) == names
        for (name, tab) in TABLEAUS
            if tab.form === :butcher
                A, b, c = tab.A, tab.b, tab.c
                s = length(b)
                @test size(A) == (s, s)
                @test all(A[i, j] == 0 for i in 1:s for j in i:s)          # explicit
                @test sum(b) ≈ 1
            elseif tab.form === :shu_osher
                s = size(tab.alpha, 1)
                @test all(sum(tab.alpha[i, 1:i]) ≈ 1 for i in 1:s)         # consistency
            end
        end
        # Numbers are parsed exactly (to the nearest double): every number of every
        # table file, as JSON parses it, is bitwise Base.parse(Float64, text).
        flat(x, acc) = x isa AbstractDict ? (foreach(v -> flat(v, acc), values(x)); acc) :
                       x isa AbstractVector ? (foreach(v -> flat(v, acc), x); acc) :
                       x isa Real && !(x isa Bool) ? push!(acc, Float64(x)) : acc
        for f in readdir(dir; join = true)
            endswith(f, ".json") || continue
            text = replace(read(f, String), r"\"(?:[^\"\\\\]|\\\\.)*\"" => "\"\"")
            tokens = [parse(Float64, m.match) for m in eachmatch(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)]
            parsed = flat(JSON.parsefile(f), Float64[])
            @test sort(reinterpret(UInt64, tokens)) == sort(reinterpret(UInt64, parsed))
        end
        @test TABLEAUS["Tsit5"].A[3, 1] === parse(Float64, "-0.008480655492356989")
        @test TABLEAUS["Tsit5"].controller.beta1 == 0.14
        @test TABLEAUS["SSPRK43"].embedded_scale == 0.5
        @test TABLEAUS["ForwardEuler"].controller === nothing
        @test TABLEAUS["Rodas4"].family === :rosenbrock
        @test_throws UnsupportedError EarthSciIntegrator.method_for(TABLEAUS["Rodas4"])
        @test_throws UnsupportedError EarthSciIntegrator.method_for(TABLEAUS["TRBDF2"])
    end

    @testset "error norm, controllers and stops" begin
        tol = Tolerances(1e-3, [1e-6, 1e-6])
        e, u0, u1 = [1e-4, -2e-4], [1.0, -2.0], [1.5, 0.5]
        sc = [1e-6 + 1e-3 * 1.5, 1e-6 + 1e-3 * 2.0]
        @test error_norm(e, u0, u1, tol) == sqrt(((e[1] / sc[1])^2 + (e[2] / sc[2])^2) / 2)
        # The norm sums over the DOFs the leaf writes only.
        sub = Tolerances(1e-3, [1e-6, 1e-6], [2])
        @test error_norm(e, u0, u1, sub) == sqrt((e[2] / sc[2])^2 / 1)
        # Dual numbers: the norm sees primal values only.
        d = ForwardDiff.Dual(1e-4, 3.0)
        @test error_norm([d, -2e-4], u0, u1, tol) == error_norm(e, u0, u1, tol)
        cc = TABLEAUS["Tsit5"].controller
        tab = TABLEAUS["Tsit5"]
        cs = ControllerState(cc)
        q = controller_factor!(cs, PI_CONTROLLER, cc, 0.5, tab)
        @test q == clamp(0.5^0.14 / 1e-4^0.08 / 0.9, 1 / 10000, 1 / 0.2)   # qmax_first_step before the first acceptance
        @test accepts(PI_CONTROLLER, cc, 0.5, q)
        @test controller_accept!(cs, PI_CONTROLLER, cc, 0.5, q, 0.1) == 0.1 / q
        @test cs.Eold == 0.5
        cs.accepted_any = true
        q = controller_factor!(cs, PI_CONTROLLER, cc, 4.0, tab)
        @test !accepts(PI_CONTROLLER, cc, 4.0, q)
        @test controller_reject(cs, PI_CONTROLLER, cc, 0.1) == 0.1 / min(5.0, 4.0^0.14 / 0.9)
        @test controller_factor!(cs, PI_CONTROLLER, cc, 0.0, tab) == 1 / 10
        qi = controller_factor!(cs, I_CONTROLLER, cc, 2.0, tab)
        @test qi == max(1 / 10, min(5.0, 2.0^(1 / 5) / 0.9))
        φ = controller_factor!(cs, PID_CONTROLLER, cc, 0.5, tab)
        @test φ ≈ 1 + atan(2.0^(0.14 / 5) - 1)
        @test propose(0.3, 1.0, 0.0, 10.0) == (1.0 + 0.3) - 1.0
        @test propose(0.3, 1.0, 0.0, 0.2) == 0.2
        # Landing on stops (spec/methods.md §10).
        @test stop_rule(0.0, 0.1, 1.0) == (0.1, false)
        @test stop_rule(0.9, 0.2, 1.0)[2] && stop_rule(0.9, 0.2, 1.0)[1] == 1.0 - 0.9
        @test stop_rule(0.9, 0.1 - 1e-15, 1.0)[2]     # within 100 eps: a stop step
        b, r = macro_bounds(0.0, 4.0, 0.5)
        @test b == collect(0.0:0.5:4.0) && all(==("step.dt"), r)
        b, r = macro_bounds(0.0, 1.0, 0.3)
        @test b[end] == 1.0 && r[end] == "tstop"
    end

    @testset "first step" begin
        # spec/methods.md §9 against OrdinaryDiffEq's own choice for Tsit5.
        prob = toy_problem("toy:van_der_pol/nonstiff")
        ops = EarthSciIntegrator.OperatorSum([prob.operators["oscillator"]])
        f! = (out, u, t) -> EarthSciIntegrator.eval_rhs!(out, ops, u, prob.p, t)
        h, nf = EarthSciIntegrator.initial_step(f!, prob.u0, 0.0, 5, Tolerances(1e-8, fill(1e-10, 2)), 0.0, 6.3)
        @test nf == 2
        @test 1e-6 < h < 1e-1
    end

    @testset "run configs refused with clear errors" begin
        base = Dict("esm_integrator" => "0.1.0", "model" => "toy:forced_decay",
            "schedule" => Dict("leaf" => "decay", "alg" => "Tsit5", "dt" => 0.1), "step" => Dict("dt" => 2))
        @test build_run(base; tspan = (0, 2)) isa EarthSciIntegrator.Run
        bad = deepcopy(base); bad["esm_integrator"] = "0.2.0"
        @test_throws ConfigError build_run(bad; tspan = (0, 2))
        bad = deepcopy(base); bad["schedule"]["alg"] = "Rodas4"
        err = try build_run(bad; tspan = (0, 2)); nothing; catch e; e; end
        @test err isa UnsupportedError && occursin("Rodas4", err.msg)
        bad = deepcopy(base); bad["schedule"]["alg"] = "SSPRK33"; delete!(bad["schedule"], "dt")
        @test_throws ConfigError build_run(bad; tspan = (0, 2))      # no error estimate: dt required (D52)
        bad = deepcopy(base); bad["schedule"]["leaf"] = "nonexistent"
        @test_throws ConfigError build_run(bad; tspan = (0, 2))
        bad = deepcopy(base); bad["schedule"] = Dict("every" => base["schedule"], "period" => 1)
        @test_throws UnsupportedError build_run(bad; tspan = (0, 2))
    end

    @testset "tape: paths, JSON Lines" begin
        cfg = Dict("esm_integrator" => "0.1.0", "model" => "toy:source_decay_map",
            "schedule" => Dict("sequence" => [Dict("leaf" => "source", "alg" => "ForwardEuler", "dt" => "step"),
                Dict("map" => "decay")]), "step" => Dict("dt" => 0.5))
        run = forward!(build_run(cfg; tspan = (0, 4)))
        lines = split(strip(EarthSciIntegrator.tape_string(run.tape)), '\n')
        @test length(lines) == 3 * 8
        @test JSON.parse(lines[1])["path"] == "0/root.0"
        @test JSON.parse(lines[2])["path"] == "0/root.1"
        @test JSON.parse(lines[3]) == Dict("path" => "0", "node" => "macro", "t" => 0.0, "dt" => 0.5, "reason" => "step.dt")
        @test JSON.parse(lines[4])["steps"] == [[0.5, 0.5]]
        # strang and subcycle occurrences
        cfg = Dict("esm_integrator" => "0.1.0", "model" => "toy:advection_reaction",
            "schedule" => Dict("strang" => [Dict("subcycle" => Dict("leaf" => "transport", "alg" => "SSPRK33", "dt" => "step"), "n" => 2),
                Dict("leaf" => "reaction", "alg" => "RK4", "dt" => 1e-3)]), "step" => Dict("dt" => 0.05))
        run = forward!(build_run(cfg; tspan = (0, 0.1)))
        paths = [r.path for r in run.tape.records]
        @test paths[1:9] == ["0/root.0.subcycle", "0/root.0.subcycle#1", "0/root.0", "0/root.1",
            "0/root.0.subcycle#2", "0/root.0.subcycle#3", "0/root.0#1", "0", "1/root.0.subcycle"]
        u, _ = replay(run)
        @test u == run.u
    end

    include("test_operators.jl")
    include("oracle.jl")
    include("conformance.jl")
    include("extra.jl")
end
