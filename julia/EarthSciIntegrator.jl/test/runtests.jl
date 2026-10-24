# Run with `julia --project=julia/EarthSciIntegrator.jl -e 'using Pkg; Pkg.test()'`
# from the repository root. Test-only dependencies are in test/Project.toml.
# Result files and tapes of the conformance cases are written to test/results/
# (or $EARTHSCIINTEGRATOR_RESULTS).

using EarthSciIntegrator
using EarthSciIntegrator: TABLEAUS, ControllerState, Tolerances, error_norm, controller_factor!, controller_accept!,
    controller_reject, accepts, propose, PI_CONTROLLER, I_CONTROLLER, PID_CONTROLLER, stop_rule, macro_bounds,
    UnsupportedError, ConfigError, LeafRecord, read_tape, tape_string
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
        @test EarthSciIntegrator.method_for(TABLEAUS["Rodas4"]) isa EarthSciIntegrator.RosenbrockMethod
        @test EarthSciIntegrator.method_for(TABLEAUS["Rosenbrock23"]).form === :rosenbrock23
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
        bad = deepcopy(base); bad["schedule"]["alg"] = "TRBDF2"
        err = try build_run(bad; tspan = (0, 2)); nothing; catch e; e; end
        @test err isa UnsupportedError && occursin("TRBDF2", err.msg)
        bad = deepcopy(base); bad["schedule"]["alg"] = "SSPRK33"; delete!(bad["schedule"], "dt")
        @test_throws ConfigError build_run(bad; tspan = (0, 2))      # no error estimate: dt required (D52)
        bad = deepcopy(base); bad["schedule"]["leaf"] = "nonexistent"
        @test_throws ConfigError build_run(bad; tspan = (0, 2))
        bad = deepcopy(base); bad["schedule"] = Dict("every" => base["schedule"], "period" => 1)
        @test_throws UnsupportedError build_run(bad; tspan = (0, 2))
        # Durations are positive and finite (spec/run-config.md §2.2): step.dt,
        # a leaf's dt, and control.dt0, dtmin, dtmax.
        refused(cfg; kw...) = try
            build_run(cfg; tspan = (0, 2), kw...); nothing
        catch e
            e isa ConfigError ? e : rethrow()
        end
        for x in (0, 0.0, -1, "0 s", "0.0 min", Inf, NaN)
            bad = deepcopy(base); bad["step"] = Dict{String, Any}("dt" => x)
            @test refused(bad) isa ConfigError
            bad = deepcopy(base); bad["schedule"]["dt"] = x
            @test refused(bad) isa ConfigError
            for k in ("dt0", "dtmin", "dtmax")
                bad = deepcopy(base); delete!(bad["schedule"], "dt"); bad["schedule"]["control"] = Dict(k => x)
                @test refused(bad) isa ConfigError
            end
        end
        let e = (bad = deepcopy(base); bad["step"] = Dict{String, Any}("dt" => 0); refused(bad))
            @test occursin("step.dt", e.msg) && occursin("positive", e.msg)
        end
        # A subcycle count below 1 (run-config §4.8).
        for n in (0, -2, 1.5)
            bad = deepcopy(base); bad["schedule"] = Dict("subcycle" => base["schedule"], "n" => n)
            @test refused(bad) isa ConfigError
        end
        ok = deepcopy(base); ok["schedule"] = Dict("subcycle" => base["schedule"], "n" => 1)
        @test build_run(ok; tspan = (0, 2)) isa EarthSciIntegrator.Run
        # domain.on_violation is reject or clip.
        for (ov, good) in (("reject", true), ("clip", true), ("ignore", false), ("Clip", false))
            cfg = deepcopy(base); cfg["schedule"]["domain"] = Dict("nonnegative" => "all", "on_violation" => ov)
            @test (refused(cfg) === nothing) == good
        end
        # run-config §8 rule 15: an adaptive leaf with a "leaf" binding needs control.dt0.
        adaptive_bound = Dict("leaf" => "decay", "alg" => "Tsit5", "bind_dt" => Dict("omega" => "leaf"))
        bad = merge(base, Dict("schedule" => adaptive_bound))
        e = refused(bad)
        @test e isa ConfigError && occursin("dt0", e.msg) && occursin("rule 15", e.msg)
        ok = merge(base, Dict("schedule" => merge(adaptive_bound, Dict("control" => Dict("dt0" => 0.1)))))
        @test build_run(ok; tspan = (0, 2)) isa EarthSciIntegrator.Run
        ok = merge(base, Dict("schedule" => merge(adaptive_bound, Dict("bind_dt" => Dict("omega" => "node")))))
        @test build_run(ok; tspan = (0, 2)) isa EarthSciIntegrator.Run
        # run-config §8 rule 11: a wrt naming a bound parameter is refused; a
        # pattern skips bound parameters, and is refused if it then matches nothing.
        bound = merge(base, Dict("schedule" => Dict("leaf" => "decay", "alg" => "Heun", "dt" => 0.1,
            "bind_dt" => Dict("omega" => "leaf"))))
        adj(w) = merge(bound, Dict("adjoint" => Dict("functional" => Dict("sum" => "u", "over" => "final"), "wrt" => w)))
        @test refused(adj(["params:omega"])) isa ConfigError
        @test refused(adj(["params:om*"])) isa ConfigError
        @test build_run(adj(["params:*"]); tspan = (0, 2)).wrt_p == [1]
        @test build_run(adj(["u0", "params:numeric"]); tspan = (0, 2)).wrt_p == [1]
        # An output time inside a macro step needs a root leaf that substeps: a
        # leaf with dt "step" or "accumulated" takes one step of the whole interval.
        for dt in ("step", "accumulated")
            cfg = merge(base, Dict("schedule" => Dict("leaf" => "decay", "alg" => "ForwardEuler", "dt" => dt), "step" => Dict("dt" => 1)))
            e = refused(cfg; saveat = [0.5, 2.0])
            @test e isa ConfigError && occursin("substeps", e.msg)
            @test refused(cfg; saveat = [1.0, 2.0]) === nothing
        end
        cfg = merge(base, Dict("step" => Dict("dt" => 1)))
        @test refused(cfg; saveat = [0.5, 2.0]) === nothing
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

    @testset "output times and fixed macro boundaries" begin
        # spec/run-config.md §5 "Fixed boundaries": a tstop within the landing
        # tolerance of t₀ + k·dt is that boundary, exactly.
        @test 3 * 0.1 != 0.3
        b, _ = macro_bounds(0.0, 1.0, 0.1)
        @test b[4] == 3 * 0.1
        b, r = macro_bounds(0.0, 1.0, 0.1, [0.3, 0.7, 1.0])
        @test b[4] === 0.3 && b[8] === 0.7 && b[end] === 1.0 && length(b) == 11 && all(==("step.dt"), r)
        @test b[5] == 4 * 0.1 && b[9] == 8 * 0.1      # the others are still multiplied
        smap = Dict("esm_integrator" => "0.1.0", "model" => "toy:source_decay_map",
            "schedule" => Dict("sequence" => [Dict("leaf" => "source", "alg" => "RK4", "dt" => "step"), Dict("map" => "decay")]),
            "step" => Dict("dt" => 0.1))
        # A non-leaf root with an output at 0.3: accepted, the state reported at 0.3
        # exactly, and t_f not added to the list.
        run = forward!(build_run(smap; tspan = (0, 1), saveat = [0.3]))
        @test run.retcode == "Success"
        @test run.output_times == [0.3]
        @test first.(run.outputs) == [0.0, 0.3]
        @test run.outputs[2][2] == run.checkpoints[4].u
        @test run.u == run.checkpoints[end].u && run.u != run.outputs[2][2]
        m = [r for r in run.tape.records if r isa EarthSciIntegrator.MacroRecord]
        @test m[4].t === 0.3 && m[3].dt == 0.3 - m[3].t
        @test replay(run)[2] == run.outputs[2:end]
        rf = JSON.parse(EarthSciIntegrator.result_json("x", run))
        @test rf["t"] == [0.0, 0.3] && length(rf["u"]) == 2
        # A listed t₀ adds nothing; no saveat means t_f alone.
        run = forward!(build_run(smap; tspan = (0, 1), saveat = [0.0, 0.5, 1.0]))
        @test run.output_times == [0.5, 1.0] && first.(run.outputs) == [0.0, 0.5, 1.0]
        @test build_run(smap; tspan = (0, 1)).output_times == [1.0]
        # Times outside [t₀, t_f], and off the boundaries with a non-leaf root, are refused.
        @test_throws ConfigError build_run(smap; tspan = (0, 1), saveat = [-0.1, 0.5])
        @test_throws ConfigError build_run(smap; tspan = (0, 1), saveat = [1.1])
        @test_throws ConfigError build_run(smap; tspan = (0, 1), saveat = [0.35])
        # A single-leaf root: the boundary near 0.3 snaps too; 0.35 is a stop inside a step.
        leafroot = merge(smap, Dict("schedule" => Dict("leaf" => "source", "alg" => "RK4", "dt" => 0.05)))
        run = forward!(build_run(leafroot; tspan = (0, 1), saveat = [0.3, 0.35]))
        @test 0.3 in run.bounds && !(0.35 in run.bounds)
        @test first.(run.outputs) == [0.0, 0.3, 0.35]
        # Functionals: over saveat, one point term per output time after t₀; final, the
        # end state even when t_f is not an output time.
        sum_over(over) = Dict("functional" => Dict("sum" => "u", "over" => over), "wrt" => ["u0", "params:numeric"])
        for (over, saveat) in (("saveat", [0.0, 0.3, 0.7]), ("final", [0.3]))
            cfg = merge(smap, Dict("adjoint" => sum_over(over)))
            run = forward!(build_run(cfg; tspan = (0, 1), saveat))
            J, ū0, p̄ = gradient(run)
            us = Dict(run.outputs)
            @test J == (over == "saveat" ? us[0.3][1] + us[0.7][1] : run.u[1])
            Jx(x) = EarthSciIntegrator.functional_value(run, replay(run, x[1:1], x[2:3])...)
            g = ForwardDiff.gradient(Jx, vcat(run.u0, run.p))
            @test Jx(vcat(run.u0, run.p)) == J
            @test isapprox(vcat(ū0, p̄), g; rtol = 1e-12)
        end
    end

    @testset "a map writing some entries leaves the others" begin
        # source_decay_map with a second variable v that neither the leaf nor
        # the map writes: v keeps its initial value, forward and on replay.
        sdm = toy_problem("toy:source_decay_map")
        prob = EarthSciIntegrator.ToyProblem("source_decay_map_v", 2, [0.0, 7.0], sdm.p, sdm.pnames, sdm.tspan,
            sdm.operators, sdm.maps, ["u" => [1], "v" => [2]], Function[], nothing, false)
        cfg = Dict("esm_integrator" => "0.1.0", "model" => "toy:source_decay_map",
            "schedule" => Dict("sequence" => [Dict("leaf" => "source", "alg" => "RK4", "dt" => "step"), Dict("map" => "decay")]),
            "step" => Dict("dt" => 0.5))
        run = forward!(build_run(cfg; tspan = (0, 2), prob))
        @test run.retcode == "Success"
        @test run.u[2] === 7.0 && all(o -> o[2][2] === 7.0, run.outputs)
        @test replay(run)[1] == run.u
        ud, _ = replay(run, ForwardDiff.Dual.(run.u0, 1.0), run.p)
        @test ForwardDiff.value.(ud) == run.u
        out = EarthSciIntegrator.apply_frozen(sdm.maps["decay"], [1.0, 7.0], ForwardDiff.Dual.(sdm.p, 1.0), 0.0, 0.5, Bool[])
        @test eltype(out) <: ForwardDiff.Dual && ForwardDiff.value(out[2]) === 7.0
    end

    @testset "node ids by position" begin
        cfg = Dict("esm_integrator" => "0.1.0", "model" => "toy:advection_reaction",
            "schedule" => Dict("sequence" => [
                Dict("id" => "transport", "strang" => [
                    Dict("subcycle" => Dict("leaf" => "transport", "alg" => "SSPRK33", "dt" => "step"), "n" => 2),
                    Dict("leaf" => "reaction", "alg" => "RK4", "dt" => 1e-3)]),
                Dict("id" => "chem", "leaf" => "reaction", "alg" => "RK4", "dt" => 1e-3)]),
            "step" => Dict("dt" => 0.05))
        run = build_run(cfg; tspan = (0, 0.1))
        @test [n.id for n in EarthSciIntegrator.allnodes(run.root)] ==
              ["root", "transport", "root.0.0", "root.0.0.subcycle", "root.0.1", "chem"]
    end

    @testset "replay refuses a tape that does not fit the run" begin
        Mismatch = EarthSciIntegrator.ReplayMismatchError
        MacroRecord = EarthSciIntegrator.MacroRecord
        clipcfg = Dict("esm_integrator" => "0.1.0", "model" => "toy:advection_reaction",
            "schedule" => Dict("leaf" => ["transport", "reaction"], "alg" => "RK4", "dt" => 0.01,
                "domain" => Dict("nonnegative" => ["A"], "on_violation" => "clip")), "step" => Dict("dt" => 0.05))
        run = forward!(build_run(clipcfg; tspan = (0, 0.1)))
        @test run.retcode == "Success"
        @test replay(run)[1] == run.u
        fresh() = read_tape(tape_string(run.tape))
        rec1(t) = t.records[1]::LeafRecord
        # Valid clips replay; each kind of invalid entry is a mismatch, not a crash
        # or a silent zeroing: a DOF outside the clip domain (B), out of range,
        # not ascending, repeated, an empty list, a step the record does not have,
        # and two entries for one step.
        t = fresh(); push!(rec1(t).clips, 2 => [0, 5])
        @test replay(run; tape = t)[1] isa Vector{Float64}
        for clip in (2 => [64], 2 => [200], 2 => [-1], 2 => [5, 0], 2 => [3, 3], 2 => Int[], 5 => [0], -1 => [0])
            t = fresh(); push!(rec1(t).clips, clip)
            @test_throws Mismatch replay(run; tape = t)
        end
        t = fresh(); append!(rec1(t).clips, [2 => [0], 2 => [1]])
        @test_throws Mismatch replay(run; tape = t)
        # Clips on a leaf that does not clip.
        noclip = deepcopy(clipcfg); delete!(noclip["schedule"], "domain")
        r2 = forward!(build_run(noclip; tspan = (0, 0.1)))
        t = read_tape(tape_string(r2.tape)); push!(rec1(t).clips, 2 => [0])
        @test_throws Mismatch replay(r2; tape = t)
        # Macro steps: t and dt must be the run's boundaries, bitwise; the record must
        # exist, and there must be no more of them than the run has.
        i = findfirst(r -> r isa MacroRecord && r.path == "1", run.tape.records)
        for (dt, tt) in ((nextfloat(0.05), 0.05), (0.05, nextfloat(0.05)))
            t = fresh(); t.records[i] = MacroRecord("1", tt, dt, "step.dt")
            @test_throws Mismatch replay(run; tape = t)
        end
        t = EarthSciIntegrator.Tape()
        foreach(r -> r isa MacroRecord || push!(t, r), run.tape.records)
        @test_throws Mismatch replay(run; tape = t)
        t = fresh(); push!(t, MacroRecord("2", 0.1, 0.05, "step.dt"))
        @test_throws Mismatch replay(run; tape = t)
        # The same for an invocation's interval, and a missing invocation.
        t = fresh(); r = rec1(t)
        t.records[1] = LeafRecord(r.path, r.t, prevfloat(r.dt), r.steps, r.accepted, r.rejected, r.retcode, r.clips)
        @test_throws Mismatch replay(run; tape = t)
        t = EarthSciIntegrator.Tape()
        foreach(r -> r.path == "1/root" || push!(t, r), run.tape.records)
        @test_throws Mismatch replay(run; tape = t)
        # A tape from a run with another step.dt.
        other = deepcopy(clipcfg); other["step"]["dt"] = 0.1
        r3 = forward!(build_run(other; tspan = (0, 0.1)))
        @test_throws Mismatch replay(run; tape = r3.tape)
    end

    include("test_operators.jl")
    include("oracle.jl")
    include("conformance.jl")
    include("rosenbrock.jl")
    include("extra.jl")
end
