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

"""Write the toy document `doc` (a Dict) as `name` beside the copied toys, so its data paths resolve as the toys' do."""
function write_toy(name, doc)
    isdir(joinpath(M2_PLANS, "toys")) || cp(joinpath(REPO, "tests", "conformance", "toys"), joinpath(M2_PLANS, "toys"))
    write(joinpath(M2_PLANS, "toys", name), JSON.json(doc))
    return name
end

toy_doc(name) = JSON.parsefile(joinpath(REPO, "tests", "conformance", "toys", name))

"""
data_decay with one record per sample: u' = -s F u, with F the hourly record
at or before the step's start, constant on each hour (D88: a stage at the
right end of a step that ends on the hour takes the hour that ends there).
"""
function decay_step_doc()
    doc = toy_doc("data_decay.esm")
    delete!(doc["index_sets"], "rec")
    m = doc["models"]["Decay"]
    m["variables"]["F_k"]["shape"] = ["x"]
    delete!(m["variables"]["F_k"]["update"]["from"], "records_per_sample")
    delete!(m["variables"], "w")
    delete!(m["variables"], "dtrec")
    filter!(e -> e["lhs"] != "w", m["equations"])
    ix(n) = Dict("op" => "index", "args" => [n, "i"])
    m["equations"][1]["rhs"]["values"][1]["expr"] =
        Dict("op" => "-", "args" => [Dict("op" => "*", "args" => [ix("Decay.s"), ix("Decay.F_k"), ix("Decay.u")])])
    return doc
end

"""Every refresh time strictly inside some accepted step of `run`'s tape: none when no step crosses an epoch boundary."""
crossed(run, epochs) = [B for r in run.tape.records if r isa EarthSciIntegrator.LeafRecord for (t, h) in r.steps for B in epochs if t < B < t + h]

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

    # The same initial state in a fresh session that loads EarthSciIntegrator
    # alone, not the test environment's solver packages: building the model
    # with its providers needs EarthSciAST's data-refresh extension
    # (DiffEqCallbacks + SciMLBase), which EarthSciIntegrator itself must load.
    @testset "an initial condition that reads data, in a session with EarthSciIntegrator alone" begin
        dir = joinpath(M2_PLANS, "data.decay_ic")
        code = """
        using EarthSciIntegrator
        run, _ = EarthSciIntegrator.build_plan_run($(repr(dir)); tspan = $(repr(tspan)))
        print(join(run.u0, ","))
        """
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`
        out = read(pipeline(cmd; stderr), String)
        @test parse.(Float64, split(out, ",")) == [1.0, 1.25, 1.5]
    end

    # spec/plan.md §6.2 item 4: the Jacobian document of an operator that
    # reads data, whose state has no `default` and an `ic` equation that reads
    # data, gets the `default` 1.0; planner and runtime both build it.
    @testset "an initial condition that reads data, with no default (§6.2 item 4)" begin
        doc = toy_doc("data_decay_ic.esm")
        delete!(doc["models"]["Decay"]["variables"]["u"], "default")
        dir = plan_data_case("data.decay_ic_nodefault", decay_config(write_toy("data_decay_ic_nodefault.esm", doc)), tspan)
        jac = JSON.parsefile(joinpath(dir, "derived", "decay.jacobian.esm"))
        @test jac["models"]["Decay"]["variables"]["u"]["default"] === 1.0
        run, pp = build_plan_run(dir; saveat, tspan)
        @test run.u0 == [1.0, 1.25, 1.5]
        forward!(run)
        @test run.retcode == "Success"
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

    # §8.4.1 (D88): a stage at the right end of a step that ends on a refresh
    # time takes the old hour's buffer of a one-record parameter. The adjoint
    # re-evaluates the step's stages and must take the same buffers as the
    # forward pass; the step's start, not the stage's time, decides.
    @testset "the adjoint takes the forward pass's forcing at a step's right end (§8.4.1, D88)" begin
        model = write_toy("data_decay_step.esm", decay_step_doc())
        rc = Dict(
            "esm_integrator" => "0.1.0", "model" => "../toys/$model",
            "operators" => Dict("decay" => Dict("select" => Dict("components" => "Decay"))),
            "schedule" => Dict("leaf" => "decay", "alg" => "RK4", "dt" => 1800, "control" => Dict("dt_scope" => "global")),
            "step" => Dict("dt" => 3600, "hit" => ["forcing_refresh", "output"])
        )
        dir = plan_data_case("data.decay_step", rc, tspan)
        pp = plan_problem(dir)
        @test pp.forcing.records["Decay.F_k"] == 1
        adj = Dict("functional" => Dict("sum" => "Decay.u", "over" => "final"), "wrt" => ["u0"])
        for (what, sch, step) in (
                ("RK4", pp.plan["schedule"], pp.plan["step"]),
                # Two steps of each macro step meet at a refresh time: the
                # implicit stage of the first and the explicit stage of the
                # second are at the same (u, t) and take different hours.
                ("Trapezoid", Dict("leaf" => "decay", "alg" => "Theta", "theta" => 0.5, "dt" => 3600), Dict("dt" => 7200, "hit" => ["forcing_refresh", "output"])),
            )
            cfg = Dict("esm_integrator" => "0.1.0", "step" => step, "schedule" => sch, "adjoint" => adj)
            run = forward!(build_run(cfg; tspan = pp.prob.tspan, prob = pp.prob))
            @test run.retcode == "Success"
            @test isempty(crossed(run, pp.forcing.epochs))
            case = Dict("id" => "m2.data_decay_step.adjoint.$what")
            check_fd_gradient(case, run, Dict("rtol" => 1.0e-6, "epsilons" => [1.0e-3, 1.0e-4, 1.0e-5]))
            check_replay(case, run, Dict())
        end
        # The Jacobian's band values at one (u, t) are cached; the cache is
        # keyed by the forcing buffers too, so the right-end stage of the step
        # that ends at 23:00 and the first stage of the step that starts there
        # read different hours (§8.4.1).
        op = pp.prob.operators["decay"]
        u, p, B = copy(pp.prob.u0), pp.prob.p, 82800.0
        Jat(start) = EarthSciIntegrator.with_step_start(start) do
            Jm = EarthSciIntegrator.DenseJacobian{Float64}(3)
            EarthSciIntegrator.jac!(op, Jm, u, p, B)
            [Jm.A[i, i] for i in 1:3]
        end
        K(h, i) = 1.0e-4 * (1 + i / 2 + (h % 5) / 4)
        S = [1.0, 1.25, 1.5]
        left, right = Jat(B - 1800), Jat(B)
        @test left == [-S[i + 1] * K(22, i) for i in 0:2]
        @test right == [-S[i + 1] * K(23, i) for i in 0:2]
        @test Jat(B - 1800) == left
    end

    # §8.4.2 step 2: providers count time from the model's time origin (D74),
    # the POSIX epoch for a model that declares none, while EarthSciAST's
    # providers count from the run window's start then. A run of data_decay
    # without its `domain`, at the same instants (model time = Unix seconds),
    # refreshes on the hours and reads the same records.
    @testset "a model without a time origin reads its data at the right instants (§8.4.2)" begin
        doc = toy_doc("data_decay.esm")
        delete!(doc, "domain")
        model = write_toy("data_decay_unix.esm", doc)
        T = 1451606400.0                        # 2016-01-01T00:00:00Z
        rc = decay_config(model)
        dir = plan_data_case("data.decay_unix", rc, T .+ tspan)
        run, pp = build_plan_run(dir; saveat = T .+ saveat, tspan = T .+ tspan)
        F = pp.forcing
        @test F.epochs == T .+ [79200.0, 82800.0, 86400.0, 90000.0, 93600.0]
        K(h, i) = 1.0e-4 * (1 + i / 2 + (h % 5) / 4)
        @test EarthSciIntegrator.epoch_buffers(F, 1)["Decay.F_k"] == [K(h, i) for h in (22, 23) for i in 0:2][[1, 4, 2, 5, 3, 6]]
        @test EarthSciIntegrator.epoch_buffers(F, 3)["Decay.F_k"] == [K(h, i) for h in (24, 25) for i in 0:2][[1, 4, 2, 5, 3, 6]]
        forward!(run)
        @test run.retcode == "Success"
        ref = JSON.parsefile(joinpath(M2_REFERENCE_DIR, "m2.data_decay.tsit5.json"))
        for (t, u) in run.outputs
            i = findfirst(==(t - T), Float64.(ref["t"]))
            @test i !== nothing && maximum(abs.(u .- Float64.(ref["u"][i]))) <= 1.0e-6
        end
    end

    # §8.4.2 item 2: a source that only unused data-fed parameters read is
    # never built, fetched or checked against the run window.
    @testset "a source nothing reads is never built (§8.4.2)" begin
        doc = toy_doc("data_decay.esm")
        doc["data_sources"]["Absent"] = Dict(
            "kind" => "grid", "source" => Dict("url_template" => "absent/none_{date:%Y}.nc"),
            "temporal" => Dict(
                "start" => "2030-01-01T00:00:00Z", "frequency" => "P1Y", "file_period" => "P1Y",
                "records_per_file" => 1, "time_variable" => "time"
            ),
            "metadata" => Dict("esio_format" => "netcdf")
        )
        doc["models"]["Decay"]["variables"]["junk"] = Dict(
            "type" => "parameter", "units" => "1", "default" => 1.0,
            "update" => Dict("kind" => "data", "source" => "Absent", "from" => Dict("file_variable" => "J"))
        )
        model = write_toy("data_decay_unused.esm", doc)
        dir = plan_data_case("data.decay_unused", decay_config(model), tspan)
        @test !haskey(JSON.parsefile(joinpath(dir, "plan.json"))["forcing"], "Decay.junk")
        run, pp = build_plan_run(dir; saveat, tspan)
        @test sort(collect(keys(pp.forcing.providers))) == ["Decay.F_k", "Decay.s"]
        forward!(run)
        @test run.retcode == "Success"
    end

    # §8.4.3: no step crosses a refresh time, whichever way the run is built.
    @testset "every run of a plan with discrete data hits its refresh times (§8.4.3)" begin
        dir = plan_data_case("data.decay.buffers", decay_config("data_decay.esm"), tspan)
        pp = plan_problem(dir)
        B = pp.forcing.epochs
        cfg = Dict(
            "esm_integrator" => "0.1.0", "schedule" => pp.plan["schedule"],
            "step" => Dict("dt" => 14400, "hit" => ["forcing_refresh"])
        )
        run = build_run(cfg; tspan = pp.prob.tspan, prob = pp.prob)
        @test run.tstops == B[2:end]
        forward!(run)
        @test run.retcode == "Success"
        @test isempty(crossed(run, B))
        # Without forcing_refresh in step.hit the run is refused, as the
        # planner refuses such a plan (spec/plan.md §7.3).
        cfg["step"] = Dict("dt" => 14400, "hit" => Any[])
        e = try
            build_run(cfg; tspan = pp.prob.tspan, prob = pp.prob); nothing
        catch err
            err
        end
        @test e isa ConfigError && occursin("forcing_refresh", e.msg)
    end

    # §8.4.1: a program's forcing slot must have the parameter's planned
    # shape, not only as many elements.
    @testset "a forcing slot of another shape refuses the leaf (§8.4.1)" begin
        dir = plan_data_case("data.decay.buffers", decay_config("data_decay.esm"), tspan)
        bad = joinpath(M2_PLANS, "data.decay.transposed")
        rm(bad; force = true, recursive = true)
        cp(dir, bad)
        pj = JSON.parsefile(joinpath(bad, "plan.json"))
        @test pj["forcing"]["Decay.F_k"]["shape"] == ["rec", "x"]
        pj["forcing"]["Decay.F_k"]["shape"] = ["x", "rec"]
        write(joinpath(bad, "plan.json"), JSON.json(pj))
        e = try
            plan_problem(bad; model = joinpath(M2_PLANS, "toys", "data_decay.esm")); nothing
        catch err
            err
        end
        @test e isa ConfigError && occursin("forcing slot Decay.F_k has shape [2, 3], the plan's [3, 2]", e.msg)
    end
end
