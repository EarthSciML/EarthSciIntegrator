# The planner's passes, M2 stage 1 (spec/plan.md): terms and selection,
# locality, the resolved settings, and the refusal fixtures that need no
# EarthSciASTDiff.

using EarthSciIntegrator: PlannerRefusal, PlannerStageError, BandEntry, jcs, Axes, PlanBackend, PendingBackend
import EarthSciAST

@isdefined(REPO) || include("paths.jl")

"""Refusal fixtures that need term or operator bands, structure or stiffness from EarthSciASTDiff (stage 2)."""
const STAGE2_REFUSALS = Dict(
    "auto-stiff-global" => "the column mean's locality needs its bands",
    "auto-stiff-not-block-diagonal" => "the array state's locality needs its bands",
    "operator-undefined" => "the default operator's existence needs the lifted terms' localities",
    "partition-too-fine" => "the auto schedule needs the stencil term's locality",
    "selector-matches-nothing" => "the locality selector needs the lifted terms' localities",
    "split-explicit" => "default operators need the reseact-shaped terms' localities",
    "step-control-missing" => "the auto schedule needs the reseact-shaped terms' localities",
    "term-selected-twice" => "the locality selector needs the reseact-shaped terms' localities",
    "tstop-off-fixed-step" => "the auto schedule needs the reseact-shaped terms' localities")

"""
    HandBackend(bands, structures, derived)

A test backend standing in for EarthSciASTDiff: term bands by
`(variable, contribution, summand)` (any other term falls back to
[`PendingBackend`](@ref)), operator structures by name, and the derived
documents and validation as given.
"""
struct HandBackend <: PlanBackend
    bands::Dict{Tuple{String, Int, Int}, Vector{BandEntry}}
    structures::Dict{String, Any}
    derived::Any
end
HandBackend(; bands = Dict{Tuple{String, Int, Int}, Vector{BandEntry}}(), structures = Dict{String, Any}(), derived = nothing) =
    HandBackend(bands, structures, derived)
EarthSciIntegrator.term_bands(b::HandBackend, ctx, t) =
    get(b.bands, (t.variable, t.contribution, t.summand), EarthSciIntegrator.term_bands(PendingBackend(), ctx, t))
EarthSciIntegrator.operator_structure(b::HandBackend, ctx, op) = get(b.structures, op.name, nothing)
EarthSciIntegrator.gershgorin_bound(::HandBackend, ctx, op) = nothing
EarthSciIntegrator.derived_and_validation(b::HandBackend, ctx) = b.derived

refusal(f) = try
    f(); nothing
catch e
    e isa PlannerRefusal ? (e.code, e.at) : rethrow()
end

@testset "planner (M2 stage 1)" begin
    @testset "additive terms (§5.1.1)" begin
        E(x) = EarthSciAST.expression_from_json(x)
        S(e) = EarthSciAST.serialize_expression(e)
        a, b, c = "a", Dict("op" => "*", "args" => ["k", "b"]), "c"
        @test S.(EarthSciIntegrator.additive_terms(E(Dict("op" => "+", "args" => [a, b, c])))) == [a, b, c]
        @test S.(EarthSciIntegrator.additive_terms(E(Dict("op" => "-", "args" => [a, Dict("op" => "+", "args" => [b, c])])))) ==
              [a, Dict("op" => "-", "args" => [b]), Dict("op" => "-", "args" => [c])]
        @test S.(EarthSciIntegrator.additive_terms(E(Dict("op" => "-", "args" => [Dict("op" => "-", "args" => [a, b])])))) ==
              [Dict("op" => "-", "args" => [a]), Dict("op" => "-", "args" => [Dict("op" => "-", "args" => [b])])]
        @test S.(EarthSciIntegrator.additive_terms(E(Dict("op" => "*", "args" => [a, Dict("op" => "+", "args" => [b, c])])))) ==
              [Dict("op" => "*", "args" => [a, Dict("op" => "+", "args" => [b, c])])]
        # A three-argument `-` is "anything else".
        @test length(EarthSciIntegrator.additive_terms(E(Dict("op" => "-", "args" => [a, b, c])))) == 1
        # a - 0 keeps its second summand, -(0): an operation, not a literal (§5.1.1).
        @test S.(EarthSciIntegrator.additive_terms(E(Dict("op" => "-", "args" => [a, 0])))) == [a, Dict("op" => "-", "args" => [0])]
        @test !EarthSciIntegrator.is_zero_literal(EarthSciIntegrator.additive_terms(E(Dict("op" => "-", "args" => [a, 0])))[2])
        @test EarthSciIntegrator.is_zero_literal(E(0)) && EarthSciIntegrator.is_zero_literal(E(0.0)) && !EarthSciIntegrator.is_zero_literal(E(Dict("op" => "-", "args" => [0])))
    end

    @testset "operator class and radius (§5.2.4)" begin
        ax = Axes(["lon", "lat"], nothing)
        # Operators: the coarsest class; a stencil radius lists every horizontal axis.
        mk(loc, rad) = (t = EarthSciIntegrator.Term("q", 0, 0, EarthSciAST.expression_from_json(0), EarthSciAST.TermOrigin[], "P", "P", false, 1,
                                 String[], loc, rad); t)
        o = EarthSciIntegrator.PlanOperator("o", [mk(:pointwise, nothing), mk(:stencil, Dict("lon" => (1, 0), "lat" => (0, 0))),
                              mk(:stencil, Dict("lon" => (0, 2), "lat" => (0, 0)))], nothing, "", nothing, nothing,
                         String[], String[], nothing)
        EarthSciIntegrator.operator_locality!(o, ax)
        @test o.locality === :stencil && o.radius == Dict("lon" => (1, 2), "lat" => (0, 0))
        @test EarthSciIntegrator.locality_json(o) == Dict("class" => "stencil", "radius" => Dict("lon" => [1, 2], "lat" => [0, 0]))
        o2 = EarthSciIntegrator.PlanOperator("o2", [mk(:column, nothing), mk(:pointwise, nothing)], nothing, "", nothing, nothing, String[], String[], nothing)
        @test EarthSciIntegrator.operator_locality!(o2, ax).locality === :column && o2.radius === nothing
        o3 = EarthSciIntegrator.PlanOperator("o3", [mk(:stencil, Dict("lon" => (1, 0), "lat" => (0, 0))), mk(:global, nothing)], nothing, "",
                          nothing, nothing, String[], String[], nothing)
        @test EarthSciIntegrator.operator_locality!(o3, ax).locality === :global
        # Pending terms leave the operator pending.
        o4 = EarthSciIntegrator.PlanOperator("o4", [mk(nothing, nothing)], nothing, "", nothing, nothing, String[], String[], nothing)
        @test EarthSciIntegrator.operator_locality!(o4, ax).locality === nothing
        @test_throws PlannerStageError EarthSciIntegrator.need_locality(o4)
    end

    @testset "refusal fixtures" begin
        dir = joinpath(CONF, "plans", "refusals")
        expected = JSON.parsefile(joinpath(dir, "expected-codes.json"))
        codes = Set(c["code"] for c in JSON.parsefile(joinpath(REPO, "spec", "planner-errors.json"))["codes"])
        for name in sort!(collect(keys(expected)))
            want = expected[name]
            @test want["code"] in codes
            if haskey(STAGE2_REFUSALS, name)
                # Not decidable without EarthSciASTDiff: planning stops at the stage boundary.
                @test_throws PlannerStageError EarthSciIntegrator.plan(joinpath(dir, name, "run.json"))
                continue
            end
            got = refusal(() -> EarthSciIntegrator.plan(joinpath(dir, name, "run.json")))
            @test got == (want["code"], want["at"])
            got == (want["code"], want["at"]) || @info "refusal fixture $name" got want
        end
        # Every code the planner raises is one spec/planner-errors.json lists.
        srcdir = joinpath(@__DIR__, "..", "src", "planner")
        src = join((read(joinpath(srcdir, f), String) for f in readdir(srcdir)), "\n")
        raised = Set(m.captures[1] for m in eachmatch(r"refuse\(\s*\"([a-z_]+)\"", src))
        @test !isempty(raised) && issubset(raised, codes)
    end

    @testset "cross_component_term by-label: the whole plan" begin
        # Structure, derived documents and validation are stage 2's; they are taken
        # from the hand-written plan, and every other byte must match it.
        dir = joinpath(PFIX, "cross_component_term")
        text = read(joinpath(dir, "by-label.expected-plan.json"), String)
        exp = JSON.parse(text)
        backend = HandBackend(structures = Dict{String, Any}(k => v["structure"] for (k, v) in exp["operators"]),
            derived = (exp["derived"], exp["validation"]))
        d = EarthSciIntegrator.plan(joinpath(dir, "by-label.run.json"); backend)
        @test EarthSciIntegrator.plan_json(d) == text
        # With the stage-1 backend, the plan stops where structure is needed (the
        # pointwise linsolve of the Rodas4 leaf reads structure.block).
        @test_throws PlannerStageError EarthSciIntegrator.plan(joinpath(dir, "by-label.run.json"))
    end

    @testset "cross_component_term by-owner" begin
        dir = joinpath(PFIX, "cross_component_term")
        blk(vars) = Dict{String, Any}("class" => "block_diagonal", "block" => Dict{String, Any}("vars" => vars))
        backend = HandBackend(structures = Dict{String, Any}("chemistry" => blk(["Chem.NO", "Chem.O3"]),
            "deposition" => blk(["Chem.NO"]), "scavenging" => blk(["Chem.O3"])))
        d = EarthSciIntegrator.plan(joinpath(dir, "by-owner.run.json"); backend)
        ops = d.ctx.ops
        @test sort(collect(keys(ops))) == ["chemistry", "deposition", "scavenging"]
        @test [(t.variable, t.contribution, t.summand, t.provenance) for t in ops["chemistry"].terms] ==
              [("Chem.O3", 0, 0, "Chem"), ("Chem.NO", 0, 0, "Chem"), ("Chem.NO", 0, 1, "Chem")]
        @test [t.owner for t in d.ctx.terms] == ["Chem", nothing, "Chem", "Chem", "Dep"]
        @test ops["deposition"].writes == ["Chem.NO"] && ops["deposition"].reads == ["Chem.NO", "Dep.kNO"]
        @test ops["scavenging"].writes == ["Chem.O3"] && ops["scavenging"].reads == ["Chem.O3", "Dep.vd", "Met.h"]
        @test ops["chemistry"].reads == ["Chem.NO", "Chem.O3", "Chem.k", "Chem.ratio"]
        @test all(==((:stiff, "run_config")), values(d.stiffness))
        # The run config writes a one-child sequence, which §7.1 keeps (expected.md).
        @test jcs(d.schedule) == jcs(JSON.parse("""{"id": "root", "sequence": [{"alg": "Rodas4", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "cell", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root.0", "jacobian": "symbolic", "leaf": ["chemistry", "deposition", "scavenging"], "linsolve": {"block_lu": {"pivot": true, "refine": 0}}}]}"""))
        @test d.partition == Dict("ranks" => [1, 1], "tiles" => [1, 1], "shards" => Dict("root.0" => 1))
        @test d.step == Dict("dt" => 60.0, "hit" => String[])
        @test d.ctx.time == (0.0, 600.0)
    end

    @testset "toy_robertson rodas4" begin
        backend = HandBackend(structures = Dict{String, Any}("kinetics" => Dict{String, Any}("class" => "block_diagonal")))
        d = EarthSciIntegrator.plan(joinpath(PFIX, "toy_robertson", "rodas4.run.json"); backend)
        op = d.ctx.ops["kinetics"]
        @test [(t.variable, t.contribution, t.summand) for t in op.terms] ==
              [("Kinetics.y1", 0, 0), ("Kinetics.y1", 0, 1), ("Kinetics.y2", 0, 0), ("Kinetics.y2", 0, 1),
               ("Kinetics.y2", 0, 2), ("Kinetics.y3", 0, 0)]
        @test EarthSciAST.serialize_expression(op.terms[4].expr) ==
              Dict("op" => "-", "args" => [Dict("op" => "*", "args" => ["Kinetics.k3", "Kinetics.y2", "Kinetics.y3"])])
        @test op.reads == ["Kinetics.k1", "Kinetics.k2", "Kinetics.k3", "Kinetics.y1", "Kinetics.y2", "Kinetics.y3"]
        @test jcs(d.schedule) == jcs(JSON.parse("""{"alg": "Rodas4", "control": {"abstol": 1e-10, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 0.000001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "jacobian": "symbolic", "leaf": ["kinetics"], "linsolve": {"dense_lu": {}}}"""))
        @test jcs(d.partition) == """{"ranks":[1,1],"shards":{"root":1},"tiles":[1,1]}"""
        @test jcs(d.step) == """{"dt":40,"hit":[]}"""
        @test jcs(d.precision) == """{"root":{"factors":"f64","jacobian":"f64","rhs":"f64","state":"f64","tables":"f64"}}"""
        @test d.ctx.time == (0.0, 40.0)
        @test EarthSciIntegrator.inputs_json(d.ctx.inputs)["model"]["path"] == "../../../toys/robertson.esm"
        @test EarthSciIntegrator.inputs_json(d.ctx.inputs)["run_config"]["path"] == "rodas4.run.json"
        # The caller's interval wins (§7.2).
        @test EarthSciIntegrator.plan(joinpath(PFIX, "toy_robertson", "rodas4.run.json"); backend, tspan = (5, 10)).ctx.time == (5.0, 10.0)
    end

    # Bands written from the fixtures' expected.md, standing in for EarthSciASTDiff.
    diag1(row, col; n) = BandEntry(row, col, [(1, n)], Union{String, Int}["i"], Any["i"])

    @testset "toy_advection_reaction strang" begin
        A = "Reaction.A"
        tbands = [BandEntry(A, A, [(2, 64)], Union{String, Int}["i"], Any["i"]),
            BandEntry(A, A, [(2, 64)], Union{String, Int}["i"], Any[Dict{String, Any}("op" => "-", "args" => Any["i", 1])]),
            BandEntry(A, A, [(1, 1)], Union{String, Int}[1], Any[1]),
            BandEntry(A, A, [(1, 1)], Union{String, Int}[1], Any[64])]
        pw = [diag1(A, A; n = 64), diag1(A, "Reaction.B"; n = 64)]
        backend = HandBackend(bands = Dict(("Reaction.A", 0, 0) => pw, ("Reaction.A", 1, 0) => tbands,
            ("Reaction.B", 0, 0) => pw))
        d = EarthSciIntegrator.plan(joinpath(PFIX, "toy_advection_reaction", "strang.run.json"); backend)
        ops = d.ctx.ops
        @test [(t.variable, t.contribution, t.summand, t.provenance, t.locality) for t in d.ctx.terms] ==
              [("Reaction.A", 0, 0, "Reaction", :pointwise), ("Reaction.A", 1, 0, "Transport", :stencil),
               ("Reaction.B", 0, 0, "Reaction", :pointwise)]
        @test EarthSciIntegrator.locality_json(ops["transport"]) == Dict("class" => "stencil", "radius" => Dict("cell" => [1, 0]))
        @test ops["transport"].reads == ["Reaction.A", "Transport.dx", "Transport.v"]
        @test ops["reaction"].reads == ["Reaction.A", "Reaction.B", "Reaction.k1", "Reaction.k2"]
        @test jcs(d.schedule) == jcs(JSON.parse("""{"id": "root", "strang": [
          {"alg": "SSPRK43", "control": {"abstol": 1e-12, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 1e-10}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root.0", "leaf": ["transport"]},
          {"alg": "Rodas4", "control": {"abstol": 1e-12, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 1e-10}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root.1", "jacobian": "symbolic", "leaf": ["reaction"], "linsolve": {"block_lu": {"pivot": false, "refine": 0}}}]}"""))
        @test jcs(d.partition) == """{"halo":{"cell":1},"ranks":[1,1],"shards":{"root.1":1},"tiles":[1,1]}"""
        @test jcs(d.step) == """{"dt":0.05,"hit":[]}"""
        @test d.ctx.time == (0.0, 0.5)
        @test d.stiffness == Dict("transport" => (:nonstiff, "run_config"), "reaction" => (:stiff, "run_config"))
    end

    @testset "toy_photostationary_cells kvaerno4" begin
        sp = ["Photo.NO", "Photo.NO2", "Photo.O3"]
        bands = Dict((s, 0, 0) => [diag1(s, c; n = 16) for c in sp] for s in sp)
        blk = Dict{String, Any}("class" => "block_diagonal", "block" => Dict{String, Any}("vars" => sp))
        d = EarthSciIntegrator.plan(joinpath(PFIX, "toy_photostationary_cells", "kvaerno4.run.json");
            backend = HandBackend(bands = bands, structures = Dict{String, Any}("photochemistry" => blk)))
        @test jcs(d.schedule) == jcs(JSON.parse("""{"alg": "Kvaerno4", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "global", "maxiters": 1000000, "reltol": 0.000001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "jacobian": "symbolic", "leaf": ["photochemistry"], "linsolve": {"block_lu": {"pivot": true, "refine": 0}}}"""))
        @test jcs(d.partition) == """{"halo":{"cell":0},"ranks":[1,1],"shards":{"root":1},"tiles":[1,1]}"""
        @test d.ctx.ops["photochemistry"].reads == ["Photo.NO", "Photo.NO2", "Photo.O3", "Photo.j0", "Photo.k", "Photo.omega"]
    end

    @testset "pointwise_chemistry" begin
        sp = ["Chem.NO", "Chem.NO2", "Chem.O3"]
        bands = Dict((s, c, k) => [diag1(s, x; n = 4) for x in sp] for s in sp, c in 0:1, k in 0:1)
        blk = Dict{String, Any}("class" => "block_diagonal", "block" => Dict{String, Any}("vars" => sp))
        for (case, opname, json) in [
            ("auto", "default_pointwise", """{"alg": "Rosenbrock23", "control": {"abstol": 0.000001, "controller": "PI", "dt_scope": "cell", "maxiters": 1000000, "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "jacobian": "symbolic", "leaf": ["default_pointwise"], "linsolve": {"block_lu": {"pivot": true, "refine": 0}}}"""),
            ("named", "chemistry", """{"alg": "Rodas4", "control": {"abstol": 1e-9, "controller": "PI", "dt_scope": "cell", "maxiters": 1000000, "reltol": 0.000001}, "domain": {"check_stages": false, "on_violation": "reject"}, "id": "root", "jacobian": "symbolic", "leaf": ["chemistry"], "linsolve": {"block_lu": {"pivot": true, "refine": 0}}}""")]
            d = EarthSciIntegrator.plan(joinpath(PFIX, "pointwise_chemistry", "$case.run.json");
                backend = HandBackend(bands = bands, structures = Dict{String, Any}(opname => blk)))
            @test collect(keys(d.ctx.ops)) == [opname]
            op = d.ctx.ops[opname]
            @test [(t.variable, t.contribution, t.summand, t.provenance) for t in op.terms] ==
                  [(v, c, s, c == 0 ? "Chem" : "Dep") for v in ["Chem.NO2", "Chem.NO", "Chem.O3"] for (c, s) in ((0, 0), (0, 1), (1, 0))]
            @test op.writes == sp
            @test op.reads == ["Chem.NO", "Chem.NO2", "Chem.O3", "Chem.j", "Chem.k", "Dep.cellw", "Dep.kd"]
            @test d.ctx.model.shapes["Chem.NO"] == ["cell"]
            @test d.stiffness[opname] == (:stiff, "reaction_systems")
            @test jcs(d.schedule) == jcs(JSON.parse(json))
            @test jcs(d.step) == """{"hit":[]}"""
            @test d.partition["shards"] == Dict("root" => (case == "named" ? 2 : 1)) && d.partition["halo"] == Dict("cell" => 0)
            @test d.ctx.time == (0.0, 3600.0)
        end
    end

    @testset "periodic_stencil partitioned" begin
        jk = ("j", "k")
        bands = BandEntry[]
        for d in -3:2
            c = d == 0 ? "i" : Dict{String, Any}("op" => d < 0 ? "-" : "+", "args" => Any["i", abs(d)])
            push!(bands, BandEntry("Transport.q", "Transport.q", [(4, 6), (1, 2), (1, 2)], Union{String, Int}["i", jk...], Any[c, jk...]))
        end
        for r in (1, 2, 3, 7, 8), d in -3:2
            push!(bands, BandEntry("Transport.q", "Transport.q", [(r, r), (1, 2), (1, 2)], Union{String, Int}[r, jk...],
                Any[mod1(r + d, 8), jk...]))
        end
        d = EarthSciIntegrator.plan(joinpath(PFIX, "periodic_stencil", "partitioned.run.json");
            backend = HandBackend(bands = Dict(("Transport.q", 0, 0) => bands)))
        op = d.ctx.ops["transport"]
        @test EarthSciIntegrator.locality_json(op) == Dict("class" => "stencil", "radius" => Dict("lat" => [0, 0], "lon" => [3, 2]))
        @test op.reads == ["Transport.c", "Transport.dx", "Transport.q"]
        @test d.stiffness["transport"] == (:nonstiff, "run_config")
        @test jcs(d.schedule) == jcs(JSON.parse("""{"alg": "SSPRK33", "control": {"abstol": 0.000001, "dt_scope": "global", "reltol": 0.001}, "domain": {"check_stages": false, "on_violation": "reject"}, "dt": 0.01, "id": "root", "leaf": ["transport"]}"""))
        @test jcs(d.partition) == """{"halo":{"lat":0,"lon":3},"ranks":[2,1],"shards":{},"tiles":[2,1]}"""
        @test jcs(d.step) == """{"dt":0.1,"hit":[]}"""
        @test d.ctx.time == (0.0, 0.5)
        @test d.ctx.axes.horizontal == ["lon", "lat"] && d.ctx.axes.vertical == "lev"
    end

    @testset "refusals on hand-written run configs" begin
        tmp = mktempdir()
        rob = abspath(joinpath(CONF, "toys", "robertson.esm"))
        cross = abspath(joinpath(PFIX, "cross_component_term", "model.esm"))
        leaf(ops; kw...) = merge(Dict{String, Any}("leaf" => ops, "alg" => "Rodas4"), Dict{String, Any}(String(k) => v for (k, v) in kw))
        function try_plan(cfg; kw...)
            path = joinpath(tmp, "run.json")
            write(path, JSON.json(merge(Dict{String, Any}("esm_integrator" => "0.1.0", "model" => rob), cfg)))
            return refusal(() -> EarthSciIntegrator.plan(path; backend = HandBackend(structures = Dict{String, Any}(
                "kinetics" => Dict{String, Any}("class" => "block_diagonal", "block" => Dict{String, Any}("vars" => ["Kinetics.y1"])),
                "default_pointwise" => Dict{String, Any}("class" => "block_diagonal", "block" => Dict{String, Any}("vars" => ["Kinetics.y1"])))), kw...))
        end
        kin = Dict("kinetics" => Dict("select" => Dict("components" => "Kinetics")))
        @test try_plan(Dict("schedule" => Dict("leaf" => 3))) == ("run_config_invalid", "")
        @test try_plan(Dict("esm_integrator" => "0.2.0", "schedule" => leaf("x"))) == ("version_unsupported", "/esm_integrator")
        @test try_plan(Dict("model" => "nowhere.esm", "schedule" => leaf("x"))) == ("model_load_failed", "/model")
        @test try_plan(Dict("schedule" => leaf("x"))) == ("operator_undefined", "/schedule/leaf")
        @test try_plan(Dict("schedule" => leaf(["default_pointwise", "x"]))) == ("operator_undefined", "/schedule/leaf/1")
        @test try_plan(Dict("operators" => Dict("a" => Dict("select" => Dict("variables" => "Kinetics.k1"))), "schedule" => leaf("a"))) ==
              ("name_wrong_kind", "/operators/a/select/variables")
        @test try_plan(Dict("operators" => Dict("a" => Dict("select" => Dict("variables" => ["Kinetics.y1", "Nope.x"]))), "schedule" => leaf("a"))) ==
              ("name_unresolved", "/operators/a/select/variables/1")
        @test try_plan(Dict("operators" => Dict("b" => Dict("select" => Dict("components" => "Kinetics")),
                                                "a" => Dict("select" => Dict("variables" => "Kinetics.y1"))), "schedule" => leaf(["a", "b"]))) ==
              ("term_selected_twice", "/operators/a")
        @test try_plan(Dict("operators" => Dict("a" => Dict("select" => Dict("components" => "Other"))), "schedule" => leaf("a"))) ==
              ("selector_matches_nothing", "/operators/a/select")
        @test try_plan(Dict("operators" => Dict("a" => Dict("select" => Dict("terms" => ["Kinetics"]), "bind_dt" => Dict("Kinetics.k1" => "leaf"))),
            "schedule" => leaf("a"))) == ("bind_dt_leaf_without_dt0", "/operators/a/bind_dt")
        @test try_plan(Dict("operators" => kin, "schedule" => Dict("sequence" => [leaf("kinetics"), leaf("kinetics")]), "step" => Dict("dt" => 1))) ==
              ("operator_advanced_twice", "/schedule/sequence/1/leaf")
        @test try_plan(Dict("operators" => kin, "schedule" => Dict("leaf" => "kinetics", "alg" => "IMEXEuler", "dt" => 0.1,
            "imex" => Dict("implicit" => "stencil")), "step" => Dict("dt" => 1))) == ("imex_partition_invalid", "/schedule/imex")
        @test try_plan(Dict("operators" => kin, "schedule" => Dict("leaf" => "kinetics", "alg" => "IMEXEuler", "dt" => 0.1,
            "imex" => Dict("implicit" => "kinetics", "explicit" => "kinetics")), "step" => Dict("dt" => 1))) == ("imex_partition_invalid", "/schedule/imex")
        @test try_plan(Dict("operators" => kin, "schedule" => Dict("every" => Dict("sequence" => [leaf("kinetics")]), "period" => 1,
            "hold" => "tendency"), "step" => Dict("dt" => 1))) == ("hold_tendency_child_not_leaf", "/schedule/every")
        seqroot = Dict("sequence" => [leaf("kinetics")])
        @test try_plan(Dict("operators" => kin, "schedule" => seqroot)) == ("step_control_missing", "/step")
        @test try_plan(Dict("operators" => kin, "schedule" => seqroot, "step" => Dict("dt" => 10, "hit_multiples" => "3 s"))) ==
              ("tstop_off_fixed_step", "/step/hit_multiples")
        @test try_plan(Dict("operators" => kin, "schedule" => seqroot, "step" => Dict("dt" => 10, "hit_multiples" => "20 s"))) === nothing
        # A root leaf with a numeric or omitted dt lands on tstops inside the macro step.
        @test try_plan(Dict("operators" => kin, "schedule" => leaf("kinetics"), "step" => Dict("dt" => 10, "hit_multiples" => "3 s"))) === nothing
        @test try_plan(Dict("operators" => kin, "schedule" => leaf("kinetics"; dt = "step"), "step" => Dict("dt" => 10, "hit_multiples" => "3 s"))) ==
              ("tstop_off_fixed_step", "/step/hit_multiples")
        @test try_plan(Dict("operators" => kin, "schedule" => leaf("kinetics"), "partition" => Dict("ranks" => [2, 1]))) ==
              ("partition_axis_missing", "/partition")
        @test try_plan(Dict("operators" => kin, "schedule" => leaf("kinetics"), "precision" => Dict("default" => Dict("factors" => "bf16")))) ==
              ("precision_low_with_classical_rosenbrock", "/precision/default")
        @test try_plan(Dict("operators" => kin, "schedule" => leaf("kinetics"; alg = "Rosenbrock23"),
            "precision" => Dict("root" => Dict("factors" => "f32")))) === nothing
        @test try_plan(Dict("operators" => kin, "schedule" => leaf("kinetics"; alg = "auto"),
            "precision" => Dict("kinetics" => Dict("jacobian" => "f32")))) === nothing             # auto → Rosenbrock23
        @test try_plan(Dict("operators" => kin, "schedule" => leaf("kinetics"; alg = "auto", control = Dict("reltol" => 1e-6)),
            "precision" => Dict("kinetics" => Dict("jacobian" => "f32")))) == ("precision_low_with_classical_rosenbrock", "/precision/kinetics")
        adj(w) = Dict("operators" => Dict("kinetics" => Dict("select" => Dict("components" => "Kinetics"), "bind_dt" => Dict("Kinetics.k1" => "node"))),
            "schedule" => leaf("kinetics"), "adjoint" => Dict("functional" => Dict("sum" => "Kinetics.y3", "over" => "final"), "wrt" => w))
        @test try_plan(adj(["params:Kinetics.k1"])) == ("wrt_bound_parameter", "/adjoint/wrt/0")
        @test try_plan(adj(["u0", "params:Kinetics.k1*"])) == ("wrt_matches_nothing", "/adjoint/wrt/1")
        @test try_plan(adj(["params:*", "params:numeric", "u0"])) === nothing
        @test try_plan(adj(["fields:*"])) == ("wrt_matches_nothing", "/adjoint/wrt/0")
        @test try_plan(Dict("operators" => kin, "schedule" => leaf("kinetics"), "time" => Dict("start" => "2019-07-01T00:00:00Z",
            "end" => "2019-07-01T01:00:00Z"))) === nothing
        # The caller's interval is not in the run config: its refusals are at "".
        @test try_plan(Dict("operators" => kin, "schedule" => leaf("kinetics")); tspan = (3, 3)) == ("time_interval_empty", "")
        @test try_plan(Dict("operators" => kin, "schedule" => leaf("kinetics"));
            tspan = ("2021-02-30T00:00:00Z", "2021-03-02T00:00:00Z")) == ("time_invalid_date", "")
        # A model solver.time_span EarthSciAST rejects takes its cause's code, at /model.
        backwards = joinpath(tmp, "backwards.esm")
        doc = JSON.parsefile(rob)
        doc["solver"]["time_span"] = Dict("start" => 40, "end" => 0)
        write(backwards, JSON.json(doc))
        @test try_plan(Dict("model" => backwards, "operators" => kin, "schedule" => leaf("kinetics"))) == ("time_interval_empty", "/model")
        # The cross-component model: a schedule naming a default operator that does not exist.
        @test try_plan(Dict("model" => cross, "schedule" => leaf("default_column"), "step" => Dict("dt" => 60))) ==
              ("operator_undefined", "/schedule/leaf")
        @test try_plan(Dict("model" => cross, "schedule" => Dict("sequence" => [leaf("default_pointwise"),
            Dict("map" => "x")]), "step" => Dict("dt" => 60))) == ("unsupported_map_operator", "/schedule/sequence/1")

        # Rule 5 (D84): a frozen entry advances its operator, but does not count toward "twice".
        two = Dict("a" => Dict("select" => Dict("variables" => "Kinetics.y1")), "b" => Dict("select" => Dict("variables" => ["Kinetics.y2", "Kinetics.y3"])))
        @test try_plan(Dict("operators" => two, "schedule" => Dict("sequence" => [leaf("a"; dt = 0.1, frozen = ["b"], linsolve = "dense_lu"), leaf("b"; dt = 0.1, linsolve = "dense_lu")]),
            "step" => Dict("dt" => 1))) === nothing
        @test try_plan(Dict("operators" => two, "schedule" => leaf("a"; dt = 0.1, frozen = ["b"], linsolve = "dense_lu"), "step" => Dict("dt" => 1))) === nothing
        @test try_plan(Dict("operators" => two, "schedule" => leaf("a"; dt = 0.1), "step" => Dict("dt" => 1))) ==
              ("operator_not_advanced", "/operators/b")
        # An IMEXEuler leaf without imex is refused at the leaf's pointer.
        @test try_plan(Dict("operators" => kin, "schedule" => Dict("sequence" => [Dict("leaf" => "kinetics", "alg" => "IMEXEuler", "dt" => 0.1)]),
            "step" => Dict("dt" => 1))) == ("imex_partition_invalid", "/schedule/sequence/0")
        # Rule 6 runs in §7.1's rule order, for leaves whose written alg is IMEXEuler.
        imexleaf(imex; kw...) = merge(Dict{String, Any}("leaf" => "kinetics", "alg" => "IMEXEuler", "dt" => 0.1, "imex" => imex),
            Dict{String, Any}(String(k) => v for (k, v) in kw))
        # Rule 3 reads imex names (explicit before implicit; after frozen), before rule 6.
        @test try_plan(Dict("operators" => kin, "schedule" => imexleaf(Dict("implicit" => "ghost")), "step" => Dict("dt" => 1))) ==
              ("operator_undefined", "/schedule/imex/implicit")
        @test try_plan(Dict("operators" => kin, "schedule" => imexleaf(Dict("implicit" => ["kinetics", "ghost"], "explicit" => ["kinetics"])),
            "step" => Dict("dt" => 1))) == ("operator_undefined", "/schedule/imex/implicit/1")      # undefined name wins over a bad partition
        @test try_plan(Dict("operators" => kin, "schedule" => imexleaf(Dict("implicit" => ["ghost1"], "explicit" => ["ghost2"])),
            "step" => Dict("dt" => 1))) == ("operator_undefined", "/schedule/imex/explicit/0")      # explicit before implicit
        @test try_plan(Dict("operators" => kin, "schedule" => imexleaf(Dict("implicit" => "ghost1"); frozen = ["ghost2"]),
            "step" => Dict("dt" => 1))) == ("operator_undefined", "/schedule/frozen/0")             # frozen before imex
        @test try_plan(Dict("operators" => kin, "schedule" => imexleaf(Dict("implicit" => "column")), "step" => Dict("dt" => 1))) ==
              ("imex_partition_invalid", "/schedule/imex")            # a locality name is not an operator name
        @test try_plan(Dict("operators" => kin, "schedule" => leaf("kinetics"; imex = Dict("implicit" => "ghost")))) ==
              ("operator_undefined", "/schedule/imex/implicit")       # rule 3 checks every leaf
        @test try_plan(Dict("operators" => kin, "schedule" => leaf("kinetics"; imex = Dict("implicit" => "kinetics", "explicit" => "kinetics")))) ===
              nothing                                                 # rule 6 only on written IMEXEuler leaves
        # Earlier rules win: rule 5 (advanced twice) before rule 6, rule 3 before a missing imex.
        @test try_plan(Dict("operators" => kin, "schedule" => Dict("sequence" => [imexleaf(Dict("implicit" => "stencil")),
            leaf("kinetics")]), "step" => Dict("dt" => 1))) == ("operator_advanced_twice", "/schedule/sequence/1/leaf")
        @test try_plan(Dict("operators" => kin, "schedule" => Dict("sequence" => [Dict("leaf" => "kinetics", "alg" => "IMEXEuler", "dt" => 0.1),
            leaf("ghost")]), "step" => Dict("dt" => 1))) == ("operator_undefined", "/schedule/sequence/1/leaf")
        # ... and rule 6 before rule 7 (duplicate ids).
        @test try_plan(Dict("operators" => two, "schedule" => Dict("sequence" => [
            Dict("id" => "x", "leaf" => "a", "alg" => "IMEXEuler", "dt" => 0.1), Dict("id" => "x", "leaf" => "b", "alg" => "Rodas4", "linsolve" => "dense_lu")]),
            "step" => Dict("dt" => 1))) == ("imex_partition_invalid", "/schedule/sequence/0")
        # Rule 2 (§7.1), in its order.
        bound(k) = Dict("kinetics" => Dict("select" => Dict("components" => "Kinetics"), "bind_dt" => Dict(k => "node")))
        @test try_plan(Dict("operators" => bound("Nope.k"), "schedule" => leaf("kinetics"))) ==
              ("name_unresolved", "/operators/kinetics/bind_dt/Nope.k")
        @test try_plan(Dict("operators" => bound("Kinetics.y1"), "schedule" => leaf("kinetics"))) ==
              ("name_wrong_kind", "/operators/kinetics/bind_dt/Kinetics.y1")
        @test try_plan(Dict("operators" => kin, "schedule" => leaf("kinetics"; bind_dt = Dict("Kinetics.y2" => "node")))) ==
              ("name_wrong_kind", "/schedule/bind_dt/Kinetics.y2")
        @test try_plan(Dict("operators" => kin, "schedule" => leaf("kinetics"; control = Dict("abstol" => Dict("Kinetics.k1" => 1e-9))))) ==
              ("name_wrong_kind", "/schedule/control/abstol/Kinetics.k1")
        @test try_plan(Dict("operators" => kin, "schedule" => leaf("kinetics"; domain = Dict("nonnegative" => ["Kinetics.y1", "Nope.y"])))) ==
              ("name_unresolved", "/schedule/domain/nonnegative/1")
        @test try_plan(Dict("operators" => kin, "schedule" => leaf("kinetics"; dt = "Nope.dt"), "step" => Dict("dt" => 40))) ==
              ("name_unresolved", "/schedule/dt")
        # One node's durations: dt, control.dt0/dtmin/dtmax, ..., then frozen entries.
        @test try_plan(Dict("operators" => two, "schedule" => leaf("a"; control = Dict("dtmax" => "Nope.c"), linsolve = "dense_lu",
            frozen = [Dict("every" => "b", "period" => "Nope.b")]), "step" => Dict("dt" => 40))) == ("name_unresolved", "/schedule/control/dtmax")
        @test try_plan(Dict("operators" => two, "schedule" => leaf("a"; dt = "Nope.a", control = Dict("dt0" => "Nope.c"), linsolve = "dense_lu",
            frozen = [Dict("every" => "b", "period" => "Nope.b")]), "step" => Dict("dt" => 40))) == ("name_unresolved", "/schedule/dt")
        @test try_plan(Dict("operators" => two, "schedule" => leaf("a"; linsolve = "dense_lu",
            frozen = [Dict("every" => "b", "period" => "Nope.b")]), "step" => Dict("dt" => 40))) == ("name_unresolved", "/schedule/frozen/0/period")
        sub = Dict("every" => Dict("subcycle" => leaf("kinetics"; linsolve = "dense_lu"), "limits" => [Dict("max" => "Nope.m")]),
            "period" => 40, "phase" => "Nope.p")
        @test try_plan(Dict("operators" => kin, "schedule" => sub, "step" => Dict("dt" => 40))) == ("name_unresolved", "/schedule/phase")
        sub["phase"] = 0
        @test try_plan(Dict("operators" => kin, "schedule" => sub, "step" => Dict("dt" => 40))) ==
              ("name_unresolved", "/schedule/every/limits/0/max")
        # step's durations come before the schedule's; output's too.
        @test try_plan(Dict("operators" => kin, "schedule" => leaf("kinetics"; dt = "Nope.dt"), "step" => Dict("dt" => "Kinetics.y1"))) ==
              ("name_wrong_kind", "/step/dt")
        @test try_plan(Dict("operators" => kin, "schedule" => leaf("kinetics"; dt = "Nope.dt"), "step" => Dict("dt" => 40),
            "output" => Dict("saveat" => "Nope.s"))) == ("name_unresolved", "/output/saveat")
        # A parameter-name duration must default to a positive number.
        zero = joinpath(tmp, "zero.esm")
        zdoc = JSON.parsefile(rob)
        zdoc["models"]["Kinetics"]["variables"]["dtz"] = Dict("type" => "parameter", "default" => 0)
        write(zero, JSON.json(zdoc))
        @test try_plan(Dict("model" => zero, "operators" => kin, "schedule" => leaf("kinetics"), "step" => Dict("dt" => "Kinetics.dtz"))) ==
              ("run_config_invalid", "/step/dt")
        # Precision keys: default, an operator (named or default) or a node id.
        @test try_plan(Dict("operators" => kin, "schedule" => leaf("kinetics"), "precision" => Dict("chem" => Dict("state" => "f64")))) ==
              ("name_unresolved", "/precision/chem")
        @test try_plan(Dict("operators" => kin, "schedule" => leaf("kinetics"; id = "chem"), "precision" => Dict("chem" => Dict("state" => "f64"),
            "kinetics" => Dict("rhs" => "f64"), "default" => Dict("tables" => "f64")))) === nothing
        # Observed step.limits come after the schedule's unsupported constructs.
        @test try_plan(Dict("model" => cross, "schedule" => Dict("sequence" => [Dict("solve" => "default_pointwise",
            "unknowns" => ["Chem.O3"], "method" => Dict("newton" => Dict()))]), "step" => Dict("limits" => ["Chem.ratio"]))) ==
              ("unsupported_solve_node", "/schedule/sequence/0")
        @test try_plan(Dict("model" => cross, "schedule" => leaf("default_pointwise"), "step" => Dict("limits" => [Dict("max" => 60), "Chem.k"]))) ==
              ("name_wrong_kind", "/step/limits/1")
        # Rule 9 for the adjoint's observation times (D83): output.saveat multiples from time zero.
        sv(save; over = "saveat") = Dict("operators" => kin, "schedule" => seqroot, "step" => Dict("dt" => 10),
            "output" => Dict("saveat" => save), "adjoint" => Dict("functional" => Dict("sum" => "Kinetics.y3", "over" => over), "wrt" => ["u0"]))
        @test try_plan(sv(20)) === nothing
        @test try_plan(sv(15)) == ("tstop_off_fixed_step", "/adjoint/functional/over")
        @test try_plan(sv(15; over = "time")) === nothing
        @test try_plan(sv(15; over = "final")) === nothing
        @test EarthSciIntegrator.multiples_in(15.0, 0.0, 40.0) == [15.0, 30.0]
        @test EarthSciIntegrator.multiples_in(15.0, 20.0, 40.0) == [30.0]
        # hit_multiples: plain products in (t₀, t_f].
        @test EarthSciIntegrator.multiples_in(0.1, 0.0, 0.3) == [0.1, 0.2]
        # Output times: a product within the landing tolerance of t_f is t_f; one near t₀ is excluded.
        @test EarthSciIntegrator.output_times(0.1, 0.0, 0.3) == [0.1, 0.2, 0.3]
        @test EarthSciIntegrator.output_times(15.0, 0.0, 40.0) == [15.0, 30.0]
        @test EarthSciIntegrator.output_times(0.1, 3 * 0.1, 0.5) == [0.4, 0.5]     # 0.30000000000000004 ≈ t₀, excluded
        @test EarthSciIntegrator.output_times(0.1, 0.0, 0.7) == [0.1, 0.2, 3 * 0.1, 0.4, 0.5, 0.6000000000000001, 0.7]
        # Mixed declared element types in one leaf.
        mixed = joinpath(tmp, "mixed.esm")
        decay(x) = Dict("lhs" => Dict("op" => "D", "args" => [x], "wrt" => "t"), "rhs" => Dict("op" => "*", "args" => [-0.5, x]))
        write(mixed, JSON.json(Dict("esm" => "1.3.0", "metadata" => Dict("name" => "mixed"),
            "models" => Dict("Kinetics" => Dict("variables" => Dict(
                "u" => Dict("type" => "unknown", "default" => 1.5, "element_type" => "Float32"),
                "v" => Dict("type" => "unknown", "default" => 1.5, "element_type" => "Float64")),
                "equations" => [decay("u"), decay("v")])),
            "solver" => Dict("time_span" => Dict("start" => 0, "end" => 40)))))
        @test try_plan(Dict("model" => mixed, "operators" => kin, "schedule" => Dict("sequence" => [leaf("kinetics")]), "step" => Dict("dt" => 40))) ==
              ("precision_contradicts_declaration", "/schedule/sequence/0")
        @test try_plan(Dict("model" => mixed, "operators" => kin, "schedule" => leaf("kinetics"),
            "precision" => Dict("default" => Dict("state" => "f32")))) == ("precision_contradicts_declaration", "/precision/default")
        # precision_conflict: the first entry in the leaf's operator order that disagrees with the first.
        three = Dict("a" => Dict("select" => Dict("variables" => "Kinetics.y1")), "b" => Dict("select" => Dict("variables" => "Kinetics.y2")),
            "c" => Dict("select" => Dict("variables" => "Kinetics.y3")))
        @test try_plan(Dict("operators" => three, "schedule" => leaf(["a", "b", "c"]; linsolve = "dense_lu"),
            "precision" => Dict("a" => Dict("factors" => "f64"), "b" => Dict("factors" => "f64"), "c" => Dict("factors" => "f32")))) ==
              ("precision_conflict", "/precision/c")
        @test try_plan(Dict("operators" => three, "schedule" => leaf(["c", "a", "b"]; linsolve = "dense_lu"),
            "precision" => Dict("a" => Dict("factors" => "f64"), "b" => Dict("factors" => "f32"), "c" => Dict("rhs" => "f64")))) ==
              ("precision_conflict", "/precision/b")
        # Rule 9 with output times near t_f (D83): saveat 0.1 on [0, 0.3] with step.dt 0.1 is on the step.
        @test try_plan(Dict("operators" => kin, "schedule" => seqroot, "step" => Dict("dt" => 0.1), "time" => Dict("start" => 0, "end" => 0.3),
            "output" => Dict("saveat" => 0.1), "adjoint" => Dict("functional" => Dict("sum" => "Kinetics.y3", "over" => "saveat"), "wrt" => ["u0"]))) === nothing
        # Model-scope order (§4 step 6): events before parameter updates, crossing replacing the general code.
        evdoc = JSON.parsefile(joinpath(CONF, "plans", "refusals", "discrete-event", "model.esm"))
        evdoc["models"]["Box"]["variables"]["k"]["shape"] = []
        evdoc["models"]["Box"]["variables"]["k"]["update"] = Dict("kind" => "schedule", "times" => [0.5], "expression" => 0.25)
        evpath = joinpath(tmp, "event_update.esm")
        write(evpath, JSON.json(evdoc))
        @test try_plan(Dict("model" => evpath, "schedule" => leaf("default_pointwise"))) == ("unsupported_event", "/model")
        # Reserved component name at load, without bands.
        @test refusal(() -> EarthSciIntegrator.plan(joinpath(CONF, "plans", "refusals", "reserved-component-name", "run.json"))) ==
              ("reserved_component_name", "/model")
    end

    @testset "the auto schedule on a scalar model" begin
        tmp = mktempdir()
        path = joinpath(tmp, "run.json")
        write(path, JSON.json(Dict("esm_integrator" => "0.1.0", "model" => abspath(joinpath(PFIX, "cross_component_term", "model.esm")),
            "schedule" => "auto")))
        blk = Dict{String, Any}("class" => "block_diagonal", "block" => Dict{String, Any}("vars" => ["Chem.NO", "Chem.O3"]))
        # cross_component_term has no reaction system and no stiffness hint: the
        # Gershgorin bound decides, which stage 1 cannot compute.
        @test_throws PlannerStageError EarthSciIntegrator.plan(path; backend = HandBackend(structures = Dict{String, Any}("default_pointwise" => blk)))
        rob = joinpath(tmp, "rob.json")
        write(rob, JSON.json(Dict("esm_integrator" => "0.1.0", "model" => abspath(joinpath(CONF, "toys", "robertson.esm")),
            "schedule" => "auto")))
        d = EarthSciIntegrator.plan(rob; backend = HandBackend(structures = Dict{String, Any}("default_pointwise" =>
            Dict{String, Any}("class" => "block_diagonal", "block" => Dict{String, Any}("vars" => ["Kinetics.y1", "Kinetics.y2", "Kinetics.y3"])))))
        @test d.stiffness["default_pointwise"] == (:stiff, "solver_hint")
        @test d.schedule["alg"] == "Rosenbrock23" && d.schedule["id"] == "root" && d.schedule["control"]["dt_scope"] == "cell"
        @test d.step == Dict("hit" => String[])
        @test EarthSciIntegrator.reference_steps(d.ctx, EarthSciIntegrator.build_snode(Dict("leaf" => "default_pointwise"), "root", "/schedule", false)) ==
              Dict("root" => 40.0)
    end

    @testset "time (§7.2, D74)" begin
        m = EarthSciIntegrator.plan_model(EarthSciIntegrator.read_inputs(joinpath(PFIX, "toy_robertson", "rodas4.run.json")))
        @test EarthSciIntegrator.time_seconds(m, "1970-01-02T00:00:00Z", "/x") == 86400.0
        @test EarthSciIntegrator.time_seconds(m, "2019-07-01T00:00:00.25Z", "/x") == 1561939200.25
        @test refusal(() -> EarthSciIntegrator.time_seconds(m, "2019-02-29T00:00:00Z", "/time/end")) == ("time_invalid_date", "/time/end")
        @test EarthSciIntegrator.time_seconds(m, "2020-02-29T00:00:00Z", "/x") == 1582934400.0
        @test EarthSciIntegrator.resolve_time(m, Dict("time" => Dict("start" => 10, "duration" => "1.5 min"))) == (10.0, 100.0)
        @test EarthSciIntegrator.resolve_time(m, Dict("time" => Dict("start" => 0.1, "duration" => 0.2)))[2] === 0.1 + 0.2
        @test EarthSciIntegrator.resolve_time(m, Dict{String, Any}()) == (0.0, 40.0)
        @test EarthSciIntegrator.plan_duration(m, "2 day", "/x") == 172800.0
        @test EarthSciIntegrator.plan_duration(m, "1.1 h", "/x") === 3960.0000000000005      # §7.1: one rounding, 1.1 × 3600
        @test EarthSciIntegrator.plan_duration(m, "Kinetics.k2", "/x") == 3.0e7
        @test refusal(() -> EarthSciIntegrator.plan_duration(m, "Kinetics.y1", "/x")) == ("name_wrong_kind", "/x")
        @test refusal(() -> EarthSciIntegrator.plan_duration(m, "Nope.x", "/x")) == ("name_unresolved", "/x")
        # A duration parameter must default to a positive number of seconds.
        v = m.flat.parameters["Kinetics.k2"]
        kw = (f => getfield(v, f) for f in fieldnames(typeof(v)) if !(f in (:type, :default)))
        for d in (0.0, -1.0)
            m.flat.parameters["Kinetics.k2"] = EarthSciAST.ModelVariable(v.type; default = d, kw...)
            @test refusal(() -> EarthSciIntegrator.plan_duration(m, "Kinetics.k2", "/x")) == ("run_config_invalid", "/x")
        end
        m.flat.parameters["Kinetics.k2"] = v
    end

    @testset "the model's solver.time_span refusals (§7.2 rule 14)" begin
        code(doc) = (r = EarthSciIntegrator.model_span_refusal(doc); r === nothing ? nothing : (r.code, r.at))
        @test code(Dict("solver" => Dict("time_span" => Dict("start" => 100, "end" => 50)))) == ("time_interval_empty", "/model")
        @test code(Dict("solver" => Dict("time_span" => Dict("start" => "2021-02-30T00:00:00Z", "end" => "2021-03-02T00:00:00Z")))) ==
              ("time_invalid_date", "/model")
        @test code(Dict("domain" => Dict("temporal" => Dict("reference_time" => "yesterday")),
            "solver" => Dict("time_span" => Dict("start" => "2021-03-01T00:00:00Z", "end" => "2021-03-02T00:00:00Z")))) ==
              ("time_origin_invalid", "/model")
        @test code(Dict("solver" => Dict("time_span" => Dict("start" => 0, "end" => 1)))) === nothing
        @test code(Dict("solver" => Dict())) === nothing
    end

    @testset "tstops on a fixed step.dt (rule 9)" begin
        @test EarthSciIntegrator.on_fixed_step(0.0, 1.0e7, 1.0e-3, 3600.0)
        @test !EarthSciIntegrator.on_fixed_step(0.0, 1.0e7, 7.0, 3600.0)
        @test EarthSciIntegrator.on_fixed_step(0.0, 1.0, 0.1, 0.3)              # 0.1 + 0.1 + 0.1 ≠ 0.3, within tolerance
        @test EarthSciIntegrator.on_fixed_step(0.0, 1.05, 0.1, 1.05)            # t_f always is one
        @test EarthSciIntegrator.on_fixed_step(0.0, 1.05, 0.1, 1.0)
        @test !EarthSciIntegrator.on_fixed_step(0.5, 2.0, 1.0, 0.75)
        for (t0, tf, dt, τ) in ((0.0, 1.0, 0.1, 0.3), (0.0, 1.05, 0.1, 1.0), (0.25, 3.0, 0.5, 1.75), (0.0, 1.0, 0.25, 0.6))
            bounds, _ = EarthSciIntegrator.macro_bounds(t0, tf, dt, [τ])
            @test EarthSciIntegrator.on_fixed_step(t0, tf, dt, τ) == (τ in bounds)
        end
    end

    @testset "shards (§7.4)" begin
        tmp = mktempdir()
        sp = ["Chem.NO", "Chem.NO2", "Chem.O3"]
        bands = Dict((s, c, k) => [BandEntry(s, x, [(1, 4)], Union{String, Int}["i"], Any["i"]) for x in sp] for s in sp, c in 0:1, k in 0:1)
        blk = Dict{String, Any}("class" => "block_diagonal", "block" => Dict{String, Any}("vars" => sp))
        model = abspath(joinpath(PFIX, "pointwise_chemistry", "model.esm"))
        function draft(sched; partition = Dict{String, Any}())
            path = joinpath(tmp, "run.json")
            write(path, JSON.json(Dict("esm_integrator" => "0.1.0", "model" => model, "schedule" => sched, "partition" => partition,
                "operators" => Dict("chem" => Dict("select" => Dict("components" => ["Chem", "Dep"]))))))
            return EarthSciIntegrator.plan(path; backend = HandBackend(bands = bands, structures = Dict{String, Any}("chem" => blk)))
        end
        # 4 cells, 3 per shard: 2 shards.
        d = draft(Dict("leaf" => "chem", "alg" => "Rodas4", "partition" => Dict("shard_size" => 3)))
        @test d.partition["shards"] == Dict("root" => 2)
        # A coupled node gets no shards entry; an entry naming it or its operator is refused.
        d = draft(Dict("coupled" => "chem", "alg" => "Rodas4"))
        @test d.partition["shards"] == Dict()
        @test refusal(() -> draft(Dict("coupled" => "chem", "alg" => "Rodas4", "id" => "c"); partition = Dict("shards" => Dict("chem" => 2, "c" => 3)))) ==
              ("run_config_invalid", "/partition/shards/c")
        # leaf_cells: horizontal dimensions, and the vertical one for a pointwise leaf only.
        ctx = (axes = Axes(["lon", "lat"], "lev"),
               model = (sizes = Dict("lon" => 8, "lat" => 3, "lev" => 4, "spc" => 5),
                        shapes = Dict("A" => ["lon", "lat", "lev"], "B" => ["lon", "lat", "spc"])),
               ops = Dict("o" => (writes = ["A", "B"],)))
        @test EarthSciIntegrator.leaf_cells(ctx, ["o"], :column) == 24
        @test EarthSciIntegrator.leaf_cells(ctx, ["o"], :pointwise) == 96
        # The lifted faq's ranges are built from the state's shape (§6.1.2).
        m = d.ctx.model
        ei, var, lifted = m.state_eqs[1]
        @test lifted && EarthSciIntegrator.lifted_ranges(m, ei, var) == (["i"], Dict("i" => Dict("from" => "cell")))
    end

    @testset "schedule helpers" begin
        # Every node of an "auto" schedule is reported at /schedule.
        auto = EarthSciIntegrator.build_snode(Dict("sequence" => Any[Dict("leaf" => ["a"], "alg" => "auto"), Dict("leaf" => ["b"], "alg" => "auto")]),
            "root", "/schedule", true)
        @test all(n -> EarthSciIntegrator.nptr(n) == "/schedule" && EarthSciIntegrator.nptr(n, "alg") == "/schedule", EarthSciIntegrator.allnodes(auto))
        mkop(name, loc) = (o = EarthSciIntegrator.PlanOperator(name, EarthSciIntegrator.Term[], nothing, "", loc, nothing, String[], String[], nothing); o)
        ops = Dict("a" => mkop("a", :pointwise), "b" => mkop("b", :stencil), "c" => mkop("c", :pointwise))
        @test EarthSciIntegrator.auto_schedule(ops, nothing) == Dict("sequence" => Any[
            Dict("leaf" => ["b"], "alg" => "auto"), Dict("leaf" => ["a", "c"], "alg" => "auto")])
        @test haskey(EarthSciIntegrator.auto_schedule(ops, "strang"), "strang")
        @test EarthSciIntegrator.auto_schedule(Dict("a" => ops["a"]), "strang") == Dict("leaf" => ["a"], "alg" => "auto")
        @test EarthSciIntegrator.linsolve_object("block_lu") == Dict("block_lu" => Dict("pivot" => true, "refine" => 0))
        @test EarthSciIntegrator.linsolve_object(Dict("sparse_block_lu" => Dict())) == Dict("sparse_block_lu" => Dict("refine" => 0))
        @test EarthSciIntegrator.linsolve_object("gmres") == Dict("gmres" => Dict())
        @test EarthSciIntegrator.glob_match("Chem.*", "Chem.a.b") && !EarthSciIntegrator.glob_match("Chem.*", "Dep.k") && EarthSciIntegrator.glob_match("k", "k")
        # Node paths and pointers.
        spec = Dict("sequence" => Any[Dict("id" => "t", "strang" => Any[Dict("subcycle" => Dict("leaf" => "x"), "n" => 2),
                                                                         Dict("leaf" => "y")]),
                                      Dict("every" => Dict("leaf" => "z"), "period" => 5)])
        root = EarthSciIntegrator.build_snode(spec, "root", "/schedule", false)
        @test [(n.id, n.path, n.ptr) for n in EarthSciIntegrator.allnodes(root)] == [
            ("root", "root", "/schedule"), ("t", "root.0", "/schedule/sequence/0"),
            ("root.0.0", "root.0.0", "/schedule/sequence/0/strang/0"),
            ("root.0.0.subcycle", "root.0.0.subcycle", "/schedule/sequence/0/strang/0/subcycle"),
            ("root.0.1", "root.0.1", "/schedule/sequence/0/strang/1"), ("root.1", "root.1", "/schedule/sequence/1"),
            ("root.1.every", "root.1.every", "/schedule/sequence/1/every")]
    end
end
