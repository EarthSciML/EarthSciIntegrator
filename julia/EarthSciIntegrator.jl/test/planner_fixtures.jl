# The planner on EarthSciASTDiff (spec/plan.md §5.2.2, §5.4, §5.5, §6, §8):
# every fixture case of tests/conformance/plans/fixtures/ planned with the
# production backend, checked against its expected.md (the contents it
# names) and, where there is one, its hand-written expected plan; the derived
# documents; the band-value files; and the parts the backend is built from.

using EarthSciIntegrator: PlannerRefusal, DiffBackend, jcs, write_plan
import EarthSciAST, EarthSciASTDiff

@isdefined(REPO) || include("paths.jl")

"""
Whether plan files validate against spec/schemas/plan.schema.json, by
Python's jsonschema (which resolves its reference into the run-config
schema); `nothing` when python3 with jsonschema ≥ 4.18 is not available.
"""
function plans_validate(files)
    validator = """
    import json, pathlib, sys
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    schemas = [json.load(open(p)) for p in pathlib.Path(sys.argv[1]).glob("*.schema.json")]
    registry = Registry().with_resources((s["\$id"], Resource.from_contents(s)) for s in schemas)
    schema = json.load(open(pathlib.Path(sys.argv[1]) / "plan.schema.json"))
    bad = 0
    for path in sys.argv[2:]:
        errs = list(Draft202012Validator(schema, registry=registry).iter_errors(json.load(open(path))))
        if errs:
            bad += 1
            print(path, errs[0].message)
    sys.exit(1 if bad else 0)
    """
    python = Sys.which("python3")
    (python !== nothing && success(`$python -c "import jsonschema, referencing"`)) || return nothing
    return success(pipeline(`$python -c $validator $(joinpath(REPO, "spec", "schemas")) $files`; stdout, stderr))
end

"""Every fixture case, `(fixture, case)`, in code-point order."""
fixture_cases() = [
    (f, c[1:(end - length(".run.json"))]) for f in sort!(readdir(PFIX)) for c in sort!(readdir(joinpath(PFIX, f)))
        if endswith(c, ".run.json")
]

# What each case's expected.md says of its operators: (locality, stiffness,
# structure class, the structure key of its locality or nothing).
const FOOTPRINT_PPM = [[-3, 0], [-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0]]
const FOOTPRINT_RESEACT = [[-3, 0], [-2, 0], [-1, 0], [0, -1], [0, 0], [1, 0], [2, 0]]
const FULL3 = [[i, j] for i in 1:3 for j in 1:3]
block(vars, pattern, lu = pattern; ordering = collect(1:length(vars))) =
    Dict("block" => Dict("vars" => vars, "ordering" => ordering, "pattern" => pattern, "lu_pattern" => lu))
const CHEM3 = ["Chem.NO", "Chem.NO2", "Chem.O3"]
const EXPECTED_OPERATORS = Dict(
    "column_contraction/auto" => Dict(
        "default_column" => ("column", ("stiff", "gershgorin"), "general", Dict("column" => Dict("axis" => "lev", "kl" => 4, "ku" => 4))),
    ),
    "cross_component_term/by-owner" => Dict(
        "chemistry" => ("pointwise", ("stiff", "run_config"), "block_diagonal", block(["Chem.NO", "Chem.O3"], [[1, 1], [1, 2], [2, 1], [2, 2]])),
        "deposition" => ("pointwise", ("stiff", "run_config"), "block_diagonal", block(["Chem.NO"], [[1, 1]])),
        "scavenging" => ("pointwise", ("stiff", "run_config"), "block_diagonal", block(["Chem.O3"], [[1, 1]])),
    ),
    "periodic_stencil/auto" => Dict(
        "default_stencil" => ("stencil", ("nonstiff", "gershgorin"), "general", Dict("stencil" => Dict("footprint" => FOOTPRINT_PPM))),
    ),
    "periodic_stencil/partitioned" => Dict(
        "transport" => ("stencil", ("nonstiff", "run_config"), "general", Dict("stencil" => Dict("footprint" => FOOTPRINT_PPM))),
    ),
    "pointwise_chemistry/auto" => Dict(
        "default_pointwise" => ("pointwise", ("stiff", "reaction_systems"), "block_diagonal", block(CHEM3, FULL3)),
    ),
    "pointwise_chemistry/named" => Dict(
        "chemistry" => ("pointwise", ("stiff", "reaction_systems"), "block_diagonal", block(CHEM3, FULL3)),
    ),
    "reseact_shaped/auto" => Dict(
        "default_stencil" => ("stencil", ("nonstiff", "gershgorin"), "general", Dict("stencil" => Dict("footprint" => FOOTPRINT_RESEACT))),
        "default_column" => ("column", ("nonstiff", "gershgorin"), "general", Dict("column" => Dict("axis" => "lev", "kl" => 3, "ku" => 3))),
        "default_pointwise" => ("pointwise", ("stiff", "reaction_systems"), "block_diagonal", block(CHEM3, FULL3)),
    ),
    "reseact_shaped/named" => Dict(
        "transport" => ("stencil", ("nonstiff", "run_config"), "general", Dict("stencil" => Dict("footprint" => FOOTPRINT_RESEACT))),
        "mixing" => ("column", ("nonstiff", "gershgorin"), "general", Dict("column" => Dict("axis" => "lev", "kl" => 3, "ku" => 3))),
        "chemistry" => ("pointwise", ("stiff", "run_config"), "block_diagonal", block(CHEM3, FULL3)),
    ),
    "toy_advection_reaction/strang" => Dict(
        "transport" => ("stencil", ("nonstiff", "run_config"), "general", Dict("stencil" => Dict("footprint" => [[-1], [0]]))),
        "reaction" => ("pointwise", ("stiff", "run_config"), "block_diagonal", block(["Reaction.A", "Reaction.B"], [[1, 1], [1, 2], [2, 1], [2, 2]])),
    ),
    "toy_photostationary_cells/kvaerno4" => Dict(
        "photochemistry" => ("pointwise", ("stiff", "run_config"), "block_diagonal", block(["Photo.NO", "Photo.NO2", "Photo.O3"], FULL3)),
    ),
    "toy_robertson/rodas4" => Dict(
        "kinetics" => (
            "pointwise", ("stiff", "run_config"), "block_diagonal",
            block(["Kinetics.y1", "Kinetics.y2", "Kinetics.y3"], [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 2], [3, 3]]),
        ),
    ),
)
# `linear_in_u` and `depends_on_t` where expected.md names them.
const EXPECTED_FLAGS = Dict(
    "toy_photostationary_cells/kvaerno4" => Dict("photochemistry" => (false, true)),
    "toy_robertson/rodas4" => Dict("kinetics" => (false, false)),
    "toy_advection_reaction/strang" => Dict("transport" => (true, false), "reaction" => (true, false)),
    "periodic_stencil/auto" => Dict("default_stencil" => (true, false)),
    "column_contraction/auto" => Dict("default_column" => (true, false)),
    "pointwise_chemistry/auto" => Dict("default_pointwise" => (false, false)),
    "reseact_shaped/named" => Dict("transport" => (true, false), "mixing" => (true, false), "chemistry" => (false, false)),
)

const PLAN_FILES = String[]

@testset "planner on EarthSciASTDiff" begin
    @testset "sampled states (§5.5.1)" begin
        r = EarthSciIntegrator.splitmix_unit
        @test r(1, 0) == 0.766301757339086
        @test r(1, 1) == 0.12603102042098713
        @test r(2, 0) == 0.905065227878185
        @test r(3, 2) == 0.03750424614895098
        S = EarthSciIntegrator.sampled_states([2.0, 0.0, -1.0], (10.0, 30.0))
        @test [t for (t, _) in S] == [15.0, 20.0, 25.0]
        @test S[1][2] == [2.0 * (0.5 + r(1, 0)), 1.0 * (0.5 + r(1, 1)), -1.0 * (0.5 + r(1, 2))]
    end

    @testset "index names a model's variables would capture (§6.1.2)" begin
        faq(idx, expr) = Dict{String, Any}("op" => "faq", "args" => Any[], "output_idx" => Any[idx...], "ranges" => Dict{String, Any}(i => Dict("from" => "cell") for i in idx), "expr" => expr)
        ix(args...) = Dict{String, Any}("op" => "index", "args" => Any[args...])
        inner = faq(["j"], ix("M.x", "j"))
        doc = Dict{String, Any}(
            "models" => Dict{String, Any}(
                "M" => Dict{String, Any}(
                    "variables" => Dict{String, Any}("j" => Dict(), "x" => Dict(), "_c1_j" => Dict()),
                    "equations" => Any[Dict{String, Any}("lhs" => "M.y", "rhs" => faq(["i", "j"], Dict{String, Any}("op" => "+", "args" => Any[ix("M.x", "i", "j"), "M.j", inner])))]
                ),
                "N" => Dict{String, Any}("variables" => Dict{String, Any}("x" => Dict()), "equations" => Any[Dict{String, Any}("lhs" => "N.y", "rhs" => faq(["j"], ix("N.x", "j")))])
            )
        )
        EarthSciIntegrator.rename_captured_indices!(doc)
        rhs = doc["models"]["M"]["equations"][1]["rhs"]
        # `_c1_j` is taken (a variable name), so the fresh name is `_c2_j`; `i`
        # names no variable of M and is kept; the scoped reference M.j is not an index.
        @test rhs["output_idx"] == ["i", "_c2_j"] && sort(collect(keys(rhs["ranges"]))) == ["_c2_j", "i"]
        @test rhs["expr"]["args"][1] == ix("M.x", "i", "_c2_j") && rhs["expr"]["args"][2] == "M.j"
        @test rhs["expr"]["args"][3] == faq(["_c2_j"], ix("M.x", "_c2_j"))
        # Model N declares no `j`: nothing to rename.
        @test doc["models"]["N"]["equations"][1]["rhs"] == faq(["j"], ix("N.x", "j"))
    end

    @testset "band points and the Gershgorin bound" begin
        L = EarthSciIntegrator.DofLayout(Dict("a" => 0, "b" => 3), Dict("a" => [3], "b" => Int[]), ["a[1]", "a[2]", "a[3]", "b"])
        bands = [
            BandEntry("a", "a", [(1, 3)], Union{String, Int}["i"], Any[Dict{String, Any}("op" => "+", "args" => Any["i", 1])]),
            BandEntry("a", "a", [(1, 3)], Union{String, Int}["i"], Any["k"], [("k", 1, 3)]),
            BandEntry("b", "a", Tuple{Int, Int}[], Union{String, Int}[], Any[2]),
        ]
        pts = EarthSciIntegrator.band_points(L, bands)
        # Band 1's row 3 reads a[4], a ghost; band 2 has a point per (row, contracted) cell.
        @test pts[1:2] == [(1, 1, 1, 2), (1, 2, 2, 3)]
        @test length(pts) == 2 + 9 + 1 && pts[end] == (3, 1, 4, 2)
        @test (2, 4, 1, 2) in pts              # field cell (1, 2) of the 3 × 3 field: row a[1], column a[2]
        J = Dict((1, 2) => -2.0, (1, 1) => 1.5, (2, 1) => 0.5, (4, 2) => 3.0)
        @test EarthSciIntegrator.gershgorin(J, [1, 2, 3]) == 3.5
        @test EarthSciIntegrator.gershgorin(J, [3]) == 0.0
    end

    @testset "fixture $fixture/$case" for (fixture, case) in fixture_cases()
        key = "$fixture/$case"
        d = EarthSciIntegrator.plan(joinpath(PFIX, fixture, "$case.run.json"))
        dir = mktempdir()
        write_plan(d, dir; bandvalues = true)
        text = read(joinpath(dir, "plan.json"), String)
        p = JSON.parse(text)
        @test jcs(p) == text
        push!(PLAN_FILES, joinpath(dir, "plan.json"))
        exp = joinpath(PFIX, fixture, "$case.expected-plan.json")
        if isfile(exp)
            e = JSON.parse(read(exp, String))
            for (path, x) in e["derived"]
                x["sha256"] = p["derived"][path]["sha256"]
            end
            @test text == jcs(e)
        end
        ops = sort!(collect(keys(p["operators"])))
        # Derived documents (§6, §7.7): one rhs and one jacobian per operator, hashed as written.
        @test sort!(collect(keys(p["derived"]))) == sort!(["derived/$o.$k.esm" for o in ops for k in ("rhs", "jacobian")])
        for (path, x) in p["derived"]
            bytes = read(joinpath(dir, path))
            @test x["sha256"] == bytes2hex(EarthSciIntegrator.sha256(bytes))
            doc = JSON.parse(String(copy(bytes)))
            @test EarthSciASTDiff.stable_json(doc) == String(bytes)
            @test doc["esm"] == "1.3.0" && doc["metadata"] == Dict("name" => x["operator"])
            if x["kind"] == "jacobian"
                nb = length(p["operators"][x["operator"]]["structure"]["bands"])
                @test sort(collect(keys(doc["models"]["_jacobian"]["variables"]))) == sort(["J_$k" for k in 1:nb])
            end
        end
        # §8: the validation entries, and the band-value files (§8.2).
        @test p["validation"] == vcat(
            Any[Dict("check" => "rhs_sum", "passed" => true)],
            Any[Dict("check" => c, "operator" => o, "passed" => true) for o in ops for c in ("band_model", "non_empty")]
        )
        for o in ops
            bv = JSON.parsefile(joinpath(dir, "validation", "$o.bandvalues.json"))
            @test length(bv["samples"]) == 3
            s = bv["samples"][1]
            @test issorted(s["entries"]; by = e -> (e[1], e[2]))
            @test s["t"] == p["time"]["start"] + (p["time"]["end"] - p["time"]["start"]) * 0.25
        end
        # expected.md's contents.
        for (o, (loc, stiff, cls, extra)) in get(EXPECTED_OPERATORS, key, Dict())
            op = p["operators"][o]
            @test op["locality"]["class"] == loc
            @test (op["stiffness"]["class"], op["stiffness"]["source"]) == stiff
            @test op["structure"]["class"] == cls
            for (k, v) in extra
                @test op["structure"][k] == v
            end
            @test count(k -> haskey(op["structure"], k), ("block", "column", "stencil")) == 1
        end
        for (o, (lin, dt)) in get(EXPECTED_FLAGS, key, Dict())
            @test p["operators"][o]["structure"]["linear_in_u"] == lin
            @test p["operators"][o]["structure"]["depends_on_t"] == dt
        end
        haskey(EXPECTED_OPERATORS, key) && @test Set(ops) == Set(keys(EXPECTED_OPERATORS[key]))
    end

    @testset "the plans validate against plan.schema.json" begin
        ok = plans_validate(PLAN_FILES)
        ok === nothing ? (@test_skip "python3 with jsonschema not available" == "") : (@test ok)
    end

    @testset "derived documents load, flatten and keep their names" begin
        d = EarthSciIntegrator.plan(joinpath(PFIX, "reseact_shaped", "named.run.json"))
        m = d.ctx.model
        for (o, a) in d.ctx.backend.ops
            # Flattening a derived document gives back the flattened model's names (D68).
            flat = a.rhs.flat
            @test sort!(collect(keys(flat.state_variables))) == sort(m.states)
            @test Set(keys(flat.parameters)) == Set(m.params)
            # The lifted equations' indices a reaction rate would capture are renamed apart.
            @test occursin("\"_c1_j\"", a.rhs.bytes) && !occursin("\"Chem.j\",\"Chem.k\"", a.rhs.bytes)
        end
        # The same plan twice is the same bytes.
        d2 = EarthSciIntegrator.plan(joinpath(PFIX, "reseact_shaped", "named.run.json"))
        @test EarthSciIntegrator.plan_json(d2) == EarthSciIntegrator.plan_json(d)
        @test EarthSciIntegrator.derived_files(d2.ctx.backend) == EarthSciIntegrator.derived_files(d.ctx.backend)
    end

    @testset "validation refuses" begin
        d = EarthSciIntegrator.plan(joinpath(PFIX, "toy_robertson", "rodas4.run.json"))
        b, ctx = d.ctx.backend, d.ctx
        op = ctx.ops["kinetics"]
        # A band list missing a dependence fails band_model (§8.1): drop band 3 (y1 on y3).
        a = b.ops["kinetics"]
        a.points = filter(p -> p[1] != 3, a.points)
        err = try
            EarthSciIntegrator.band_model_check!(b, ctx, op); nothing
        catch e
            e
        end
        @test err isa PlannerRefusal && err.code == "validation_band_model_failed" && err.at == "/model"
        # rhs_sum's measure (§8.1): an exact zero passes.
        @test EarthSciIntegrator._rhs_sum_error(0.0, [0.0, 0.0]) == 0.0
        @test EarthSciIntegrator._rhs_sum_error(1.0, [0.5, 0.5]) == 0.0
        @test EarthSciIntegrator._rhs_sum_error(1.0, [1.0, 1.0e-3]) ≈ 1.0e-3 / 1.001
    end
end
