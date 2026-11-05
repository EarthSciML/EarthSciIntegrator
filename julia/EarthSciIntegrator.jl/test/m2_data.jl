# The runtime's data layer beyond the M2 data-input cases' checks
# (spec/operator-interface.md §8.4, spec/toy-problems.md §4.4–§4.6): the
# buffers at t₀ against EarthSciAST's own problem, a run whose initial
# condition reads data, and an extent the data contradicts.

using EarthSciIntegrator: plan, write_plan, build_plan_run, plan_problem, forward!, ConfigError
import EarthSciAST

@isdefined(M2_PLANS) || include("m2.jl")

"""Plan the run config `rc` (a Dict) as case `id` under M2_PLANS, beside the copied toys."""
function plan_data_case(id, rc, tspan)
    mkpath(joinpath(M2_PLANS, "m2"))
    isdir(joinpath(M2_PLANS, "toys")) || cp(joinpath(REPO, "tests", "conformance", "toys"), joinpath(M2_PLANS, "toys"))
    cfg = joinpath(M2_PLANS, "m2", "$id.json")
    write(cfg, JSON.json(rc))
    dir = joinpath(M2_PLANS, id)
    write_plan(plan(cfg; tspan), dir)
    return dir
end

decay_config(model) = Dict(
    "esm_integrator" => "0.1.0", "model" => "../toys/$model",
    "operators" => Dict("decay" => Dict("select" => Dict("components" => "Decay"))),
    "schedule" => Dict("leaf" => "decay", "alg" => "Tsit5", "control" => Dict("reltol" => 1.0e-8, "abstol" => 1.0e-12, "dt_scope" => "global")),
    "step" => Dict("dt" => 14400, "hit" => ["forcing_refresh", "output"])
)

@testset "M2 data layer (spec/operator-interface.md §8.4)" begin
    tspan = [79200, 93600]
    saveat = [82800.0, 86400.0, 90000.0, 93600.0]

    @testset "buffers at t₀ are esm_problem's, bit for bit (§8.4.2 step 3)" begin
        dir = plan_data_case("data.decay.buffers", decay_config("data_decay.esm"), tspan)
        pp = plan_problem(dir)
        F = pp.forcing
        @test F !== nothing && sort(F.names) == ["Decay.F_k", "Decay.s"]
        @test F.epochs == [79200.0, 82800.0, 86400.0, 90000.0, 93600.0]
        file = EarthSciAST.load_path(joinpath(M2_PLANS, "toys", "data_decay.esm"))
        prob = EarthSciIntegrator.data_problem(file, F, (79200.0, 93600.0))
        fs = EarthSciAST.default_forcing_set(EarthSciAST.rhs_program(prob))
        mine = EarthSciIntegrator.epoch_buffers(F, 1)
        for (s, b) in zip(fs.slots, fs.bufs)
            @test reinterpret(UInt64, mine[s.name]) == reinterpret(UInt64, b)
        end
        # The bracket at t₀ = 22:00 holds hours 22 and 23 of 1 January.
        K(h, i) = 1.0e-4 * (1 + i / 2 + (h % 5) / 4)
        @test mine["Decay.F_k"] == [K(h, i) for h in (22, 23) for i in 0:2][[1, 4, 2, 5, 3, 6]]
        @test mine["Decay.s"] == [1.0, 1.25, 1.5]
    end

    @testset "an initial condition that reads data starts from the data (§8.4.5)" begin
        dir = plan_data_case("data.decay_ic", decay_config("data_decay_ic.esm"), tspan)
        run, pp = build_plan_run(dir; saveat, tspan)
        @test run.u0 == [1.0, 1.25, 1.5]          # S_i, not the default 1.0
        forward!(run)
        @test run.retcode == "Success"
        ref = JSON.parsefile(joinpath(REPO, "tests", "conformance", "m2", "reference", "data_decay_ic.json"))
        for (t, u) in run.outputs
            i = findfirst(==(t), Float64.(ref["t"]))
            @test i !== nothing && maximum(abs.(u .- Float64.(ref["u"][i])) ./ [1.0, 1.25, 1.5]) <= 1.0e-6
        end
    end

    @testset "an extent the data contradicts refuses the run before its first step (§8.4.2 step 4)" begin
        doc = JSON.parsefile(joinpath(REPO, "tests", "conformance", "toys", "data_points.esm"))
        doc["metaparameters"]["N_STN"]["default"] = 5
        isdir(joinpath(M2_PLANS, "toys")) || cp(joinpath(REPO, "tests", "conformance", "toys"), joinpath(M2_PLANS, "toys"))
        write(joinpath(M2_PLANS, "toys", "data_points_n5.esm"), JSON.json(doc))
        rc = Dict(
            "esm_integrator" => "0.1.0", "model" => "../toys/data_points_n5.esm",
            "operators" => Dict("sources" => Dict("select" => Dict("components" => "Src"))),
            "schedule" => Dict("leaf" => "sources", "alg" => "Tsit5"),
            "step" => Dict("dt" => 3600, "hit" => ["output"])
        )
        dir = plan_data_case("data.points_n5", rc, [0, 3600])
        e = try
            build_plan_run(dir; tspan = [0, 3600]); nothing
        catch err
            err
        end
        @test e isa ConfigError
    end
end
