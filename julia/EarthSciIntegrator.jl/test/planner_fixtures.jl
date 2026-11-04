# The planner on EarthSciASTDiff (spec/plan.md §5.2.2, §5.4, §5.5, §6, §8):
# every fixture case of tests/conformance/plans/fixtures/ planned with the
# production backend, checked against its expected.md (the contents it
# names) and, where there is one, its hand-written expected plan; against its
# golden `<case>.golden/` (§9: plan.json and derived/*.esm byte for byte, the
# band values as §8.2 compares them; a missing golden fails); the derived
# documents; the band-value files; and the parts the backend is built from.
# tools/plan-goldens.jl rewrites the goldens.

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

"""
Why the golden's band-value file `want` for operator `o` of plan draft `d`
does not match the Julia planner's bands (spec/plan.md §8.2), or `nothing`
when it does. As the Rust harness does, the band Jacobian is evaluated at
each sample's own `t` and `u`, so this run need not reproduce the golden's
sampled states bit for bit; it must have the file's structural (row, column)
key set, and each value must be within ``10^{-10} M_r`` of the file's,
``M_r`` the largest magnitude in row ``r`` of either. The file itself must
have the form §8.2 gives it: one sample per quantile of §5.5.1 at that
quantile's `t`, a `u` naming every DOF and nothing else, and no (row,
column) listed twice.
"""
function golden_bandvalues_mismatch(d, o, want)
    b, ctx = d.ctx.backend, d.ctx
    L = EarthSciIntegrator.layout!(b, ctx)
    B = EarthSciIntegrator.jac_built!(b, ctx, ctx.ops[o])
    t0, tf = ctx.time
    quantiles = (0.25, 0.5, 0.75)
    samples = want["samples"]
    length(samples) == length(quantiles) || return "$(length(samples)) samples, §5.5.1 has $(length(quantiles))"
    for (k, (s, q)) in enumerate(zip(samples, quantiles))
        s["t"] == t0 + (tf - t0) * q || return "sample $k: t is $(s["t"]), quantile $q of the interval is $(t0 + (tf - t0) * q)"
        Set(keys(s["u"])) == Set(L.names) || return "sample $k: u does not name exactly the model's DOFs"
        u = Float64[s["u"][n] for n in L.names]
        JB = EarthSciIntegrator.band_jacobian(B, b.ops[o].points, u, Float64(s["t"]))
        G = Dict((L.names[r], L.names[c]) => v for ((r, c), v) in JB)
        W = Dict{Tuple{String, String}, Float64}()
        for e in s["entries"]
            haskey(W, (e[1], e[2])) && return "sample $k: ($(e[1]), $(e[2])) is listed twice"
            W[(e[1], e[2])] = Float64(e[3])
        end
        extra = setdiff(keys(G), keys(W))
        isempty(extra) || return "sample $k: $(first(extra)) is in the band pattern, not in the golden"
        absent = setdiff(keys(W), keys(G))
        isempty(absent) || return "sample $k: $(first(absent)) is in the golden, not in the band pattern"
        M = Dict{String, Float64}()
        for ((r, _), v) in Iterators.flatten((G, W))
            M[r] = max(get(M, r, 0.0), abs(v))
        end
        for (key, v) in G
            abs(v - W[key]) <= 1.0e-10 * M[key[1]] || return "sample $k: $key is $v, $(W[key]) in the golden"
        end
    end
    return nothing
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
    "unnamed_lift/auto" => Dict(
        "default_pointwise" => ("pointwise", ("stiff", "reaction_systems"), "block_diagonal", block(CHEM3, FULL3)),
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
    "unnamed_lift/auto" => Dict("default_pointwise" => (false, false)),
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
        # A `where` guard drops the cells it excludes (EarthSciASTDiff's `_all_points`).
        g = BandEntry(
            "a", "a", [(1, 3)], Union{String, Int}["i"], Any["i"], Tuple{String, Int, Int}[],
            Dict{String, Any}("op" => ">", "args" => Any["i", 1])
        )
        @test EarthSciIntegrator.band_points(L, [g]) == [(1, 2, 2, 2), (1, 3, 3, 3)]
        @test BandEntry(Dict{String, Any}("row" => "a", "col" => "a", "rows" => [[1, 3]], "row_idx" => ["i"], "col_idx" => ["i"], "where" => g.guard)).guard == g.guard
        # A column index gathering from a constant table is folded first (`_fold_const_gathers`).
        nbr = Dict{String, Any}("op" => "index", "args" => Any[Dict{String, Any}("op" => "const", "args" => Any[], "value" => Any[3, 1, 2]), "i"])
        @test EarthSciIntegrator.band_points(L, [BandEntry("a", "a", [(1, 3)], Union{String, Int}["i"], Any[nbr])]) ==
            [(1, 1, 1, 3), (1, 2, 2, 1), (1, 3, 3, 2)]
        # A state over an empty index set has no DOF.
        L0 = EarthSciIntegrator.DofLayout(Dict("a" => 0, "e" => 3, "z" => 3), Dict("a" => [3], "e" => [0], "z" => Int[]), ["a[1]", "a[2]", "a[3]", "z"])
        @test EarthSciIntegrator.state_dofs(L0, ["e"]) == Int[]
        @test EarthSciIntegrator.state_dofs(L0, ["e", "z"]) == [4]
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
        # A stencil operator's footprint reads the pattern its radius reads
        # (§5.2.2, §5.4): on each horizontal axis its offsets reach exactly
        # the radius, -l to u.
        if any(o -> haskey(p["operators"][o]["structure"], "stencil"), ops)
            horizontal = EarthSciIntegrator.plan_context(
                EarthSciIntegrator.read_inputs(joinpath(PFIX, fixture, "$case.run.json")), DiffBackend()
            ).axes.horizontal
            for o in ops
                st = p["operators"][o]["structure"]
                haskey(st, "stencil") || continue
                rad = p["operators"][o]["locality"]["radius"]
                for (i, X) in enumerate(horizontal)
                    offs = [v[i] for v in st["stencil"]["footprint"]]
                    @test (-minimum(offs), maximum(offs)) == Tuple(rad[X])
                end
            end
        end
        # Derived documents (§6, §7.7): one rhs and one jacobian per operator, hashed as written.
        @test sort!(collect(keys(p["derived"]))) == sort!(["derived/$o.$k.esm" for o in ops for k in ("rhs", "jacobian")])
        for (path, x) in p["derived"]
            bytes = read(joinpath(dir, path))
            @test x["sha256"] == bytes2hex(EarthSciIntegrator.sha256(bytes))
            doc = JSON.parse(String(copy(bytes)))
            @test EarthSciASTDiff.stable_json(doc) == String(bytes)
            if x["kind"] == "rhs"
                @test doc["esm"] == "1.3.0" && doc["metadata"] == Dict("name" => x["operator"])
            else
                # EarthSciASTDiff's evaluation document, as it writes it (§6.2).
                @test doc["esm"] == "1.1.0" && doc["metadata"] == Dict("name" => "JacobianEval")
                @test !occursin("\"op\":\"ic\"", String(copy(bytes)))
                nb = length(p["operators"][x["operator"]]["structure"]["bands"])
                @test sort(collect(keys(doc["models"]["JacobianEval"]["variables"]))) == sort(["J_$k" for k in 1:nb])
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
        # The golden (§9): a missing one is a failure, not a skip.
        golden = joinpath(PFIX, fixture, "$case.golden")
        @test isfile(joinpath(golden, "plan.json"))
        if isfile(joinpath(golden, "plan.json"))
            @test text == read(joinpath(golden, "plan.json"), String)
            gd = isdir(joinpath(golden, "derived")) ? sort!(readdir(joinpath(golden, "derived"))) : String[]
            @test gd == sort!(readdir(joinpath(dir, "derived")))
            for f in gd
                @test read(joinpath(dir, "derived", f)) == read(joinpath(golden, "derived", f))
            end
            for o in ops
                path = joinpath(golden, "validation", "$o.bandvalues.json")
                @test isfile(path)
                isfile(path) || continue
                @test golden_bandvalues_mismatch(d, o, JSON.parsefile(path)) === nothing
            end
            # The comparison itself, on one case: a state one ulp off still
            # passes; a changed value, a missing or doubled entry, a dropped
            # sample, a shifted t and a u without every DOF fail.
            if key == "pointwise_chemistry/auto"
                o = only(ops)
                want = JSON.parsefile(joinpath(golden, "validation", "$o.bandvalues.json"))
                bad(f) = (w = deepcopy(want); f(w); golden_bandvalues_mismatch(d, o, w) !== nothing)
                function nudge!(w)
                    u = w["samples"][1]["u"]
                    for n in keys(u)
                        u[n] = nextfloat(Float64(u[n]))
                    end
                    return
                end
                @test !bad(nudge!)
                @test bad(w -> (e = w["samples"][2]["entries"][1]; e[3] = Float64(e[3]) * (1 + 1.0e-6) + 1.0e-12))
                @test bad(w -> pop!(w["samples"][1]["entries"]))
                @test bad(w -> push!(w["samples"][1]["entries"], w["samples"][1]["entries"][1]))
                @test bad(w -> pop!(w["samples"]))
                @test bad(w -> empty!(w["samples"]))
                @test bad(w -> (w["samples"][1]["t"] += 1.0))
                @test bad(w -> delete!(w["samples"][3]["u"], first(keys(w["samples"][3]["u"]))))
            end
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
        if key == "unnamed_lift/auto"
            # No index set is declared, so `flatten` names the lift's
            # dimension `_liftdim1_4` (§6.1.3): it is the default axis and in
            # both derived documents' `index_sets`, and the states and band
            # fields are shaped on it.
            @test p["partition"]["halo"] == Dict("_liftdim1_4" => 0)
            want = Dict("_liftdim1_4" => Dict("kind" => "interval", "size" => 4))
            for k in ("rhs", "jacobian")
                doc = JSON.parsefile(joinpath(dir, "derived", "default_pointwise.$k.esm"))
                @test doc["index_sets"] == want
                @test all(x -> x == ["_liftdim1_4"], (doc["models"]["Chem"]["variables"][v]["shape"] for v in ("NO", "NO2", "O3")))
                k == "jacobian" && @test all(v -> v["shape"] == ["_liftdim1_4"], values(doc["models"]["JacobianEval"]["variables"]))
            end
        end
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
            # The lifted equations keep the indices the model wrote, though a
            # reaction rate shares the name `j` (EarthSciAST#610).
            @test !occursin("_c1_", a.rhs.bytes) && !occursin("\"Chem.j\",\"Chem.k\"", a.rhs.bytes)
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
        # An operator that reads a state but has no band is `non_empty`'s
        # refusal, although `band_model` would fail on it too: `non_empty`
        # runs first (§8.1).
        a.entries = EarthSciASTDiff.JacEntry[]
        a.points = NTuple{4, Int}[]
        err = try
            EarthSciIntegrator.derived_and_validation(b, ctx); nothing
        catch e
            e
        end
        @test err isa PlannerRefusal && err.code == "validation_empty_operator" && err.at == "/model"
        # rhs_sum's measure (§8.1): an exact zero passes.
        @test EarthSciIntegrator._rhs_sum_error(0.0, [0.0, 0.0]) == 0.0
        @test EarthSciIntegrator._rhs_sum_error(1.0, [0.5, 0.5]) == 0.0
        @test EarthSciIntegrator._rhs_sum_error(1.0, [1.0, 1.0e-3]) ≈ 1.0e-3 / 1.001
    end

    @testset "the derived documents' domain" begin
        # No fixture has a domain: robertson with one. Both documents carry
        # `temporal`'s start, end and reference_time and `element_type` as
        # given, and not `independent_variable`.
        dir = mktempdir()
        doc = JSON.parsefile(joinpath(REPO, "tests", "conformance", "toys", "robertson.esm"))
        doc["domain"] = Dict(
            "independent_variable" => "t", "element_type" => "Float64",
            "temporal" => Dict("start" => "2020-01-01T00:00:00Z", "reference_time" => "2020-01-01T00:00:00Z")
        )
        write(joinpath(dir, "model.esm"), JSON.json(doc))
        cfg = JSON.parsefile(joinpath(PFIX, "toy_robertson", "rodas4.run.json"))
        cfg["model"] = "model.esm"
        write(joinpath(dir, "run.json"), JSON.json(cfg))
        d = EarthSciIntegrator.plan(joinpath(dir, "run.json"))
        a = d.ctx.backend.ops["kinetics"]
        want = Dict(
            "element_type" => "Float64",
            "temporal" => Dict("start" => "2020-01-01T00:00:00Z", "reference_time" => "2020-01-01T00:00:00Z")
        )
        @test JSON.parse(a.rhs.bytes)["domain"] == want
        @test JSON.parse(a.jac.bytes)["domain"] == want
    end

    @testset "the Jacobian document's band fields are the operator's bands" begin
        d = EarthSciIntegrator.plan(joinpath(PFIX, "toy_robertson", "rodas4.run.json"))
        a = d.ctx.backend.ops["kinetics"]
        @test EarthSciIntegrator.jacobian_document(a.rhs.flat, a.entries) isa AbstractDict
        # A band count that differs from the fields EarthSciASTDiff writes is an error.
        @test_throws ErrorException EarthSciIntegrator.jacobian_document(a.rhs.flat, a.entries[1:(end - 1)])
        # `drop_ic!` passes over a node with no `equations`.
        models = Dict{String, Any}(
            "A" => Dict{String, Any}(
                "variables" => Dict{String, Any}(), "subsystems" => Dict{String, Any}(
                    "B" => Dict{String, Any}(
                        "equations" => Any[
                            Dict{String, Any}("lhs" => Dict{String, Any}("op" => "ic", "args" => Any["A.B.x"]), "rhs" => 1.0),
                            Dict{String, Any}("lhs" => Dict{String, Any}("op" => "D", "args" => Any["A.B.x"], "wrt" => "t"), "rhs" => 0.0),
                        ]
                    )
                )
            )
        )
        EarthSciIntegrator.drop_ic!(models)
        @test length(models["A"]["subsystems"]["B"]["equations"]) == 1
        @test !haskey(models["A"], "equations")
    end

    @testset "a backend planned again starts afresh" begin
        cfg = joinpath(PFIX, "toy_robertson", "rodas4.run.json")
        b = DiffBackend()
        d1 = EarthSciIntegrator.plan(cfg; backend = b)
        a1 = b.ops["kinetics"]
        d2 = EarthSciIntegrator.plan(cfg; backend = b)
        @test b.ctx === d2.ctx && b.ops["kinetics"] !== a1
        @test EarthSciIntegrator.plan_json(d2) == EarthSciIntegrator.plan_json(d1)
    end

    @testset "the probe document gives each term's own bands (§5.2.2): $fixture/$case" for (fixture, case) in fixture_cases()
        inp = EarthSciIntegrator.read_inputs(joinpath(PFIX, fixture, "$case.run.json"))
        b = DiffBackend()
        ctx = EarthSciIntegrator.plan_context(inp, b)
        m = ctx.model
        terms = EarthSciIntegrator.build_terms(m)
        probe = EarthSciIntegrator.probe_band_dicts(b, ctx, terms)
        # The probe itself succeeds, so planning does not split it.
        @test probe !== nothing
        # Band for band, in order, every field `serialize_band` writes (but
        # `coef`) equals each term's bands from its own document.
        @test probe == [EarthSciIntegrator.term_band_dicts(b, ctx, t) for t in terms]
        # all_term_bands gives the same, every field of every BandEntry.
        fields(x) = ntuple(i -> getfield(x, i), fieldcount(typeof(x)))
        @test [map(fields, x) for x in EarthSciIntegrator.all_term_bands(b, ctx, terms)] ==
            [map(fields, EarthSciIntegrator.term_bands(b, ctx, t)) for t in terms]
        # A BandEntry read from EarthSciASTDiff's entry equals one read from
        # `serialize_band`'s dictionary.
        @test [map(fields, EarthSciIntegrator.term_bands(b, ctx, t)) for t in terms] ==
            [map(d -> fields(EarthSciIntegrator.BandEntry(d)), EarthSciIntegrator.term_band_dicts(b, ctx, t)) for t in terms]
        # The probe document's bands are the same, in order, with none of its
        # shared subexpressions written as templates and with more of them,
        # down to every repeated one (§5.2.2).
        doc_bands(min_nodes) = map(
            EarthSciIntegrator.band_json,
            EarthSciIntegrator.document_bands(
                EarthSciAST.flatten(EarthSciAST.load_document(EarthSciIntegrator.probe_document(m, terms; min_nodes))), "the probe document"
            )
        )
        plain = doc_bands(typemax(Int))
        for min_nodes in (1, 2, EarthSciIntegrator.PROBE_TEMPLATE_MIN_NODES)
            @test doc_bands(min_nodes) == plain
        end
        # So do smaller batches (§5.2.2): with a lower bound, each batch is a
        # consecutive run within the bound in all, or a single term, and it
        # ends only where the next term would take it over the bound.
        sizes = [EarthSciIntegrator.operator_nodes(t.expr, 10^9) for t in terms]
        median = isempty(sizes) ? 0 : sort(sizes)[cld(length(sizes), 2)]
        for large in unique!([0, 1, median, sum(sizes; init = 0)])
            batches = EarthSciIntegrator.probe_batches(terms; large)
            @test reduce(vcat, collect.(batches); init = Int[]) == collect(eachindex(terms))
            @test all(r -> length(r) == 1 || sum(sizes[r]) <= large, batches)
            @test all(i -> sum(sizes[batches[i]]) + sizes[first(batches[i + 1])] > large, 1:(length(batches) - 1))
            @test [map(fields, x) for x in EarthSciIntegrator.all_term_bands(b, ctx, terms; large)] ==
                [map(fields, EarthSciIntegrator.term_bands(b, ctx, t)) for t in terms]
        end
        # The probe states are declared with their equation state's index sets.
        doc = EarthSciIntegrator.probe_document(m, terms)
        vars = doc["models"]["JacobianEval"]["variables"]
        @test length(vars) == length(terms)
        for (k, t) in enumerate(terms)
            @test get(vars["term_$k"], "shape", nothing) == (haskey(m.shapes, t.variable) ? m.shapes[t.variable] : nothing)
        end
        # Each has its term's state's `ic` equation, when the state has one.
        ics = Set(only(e["lhs"]["args"]) for e in doc["models"]["JacobianEval"]["equations"] if e["lhs"]["op"] == "ic")
        @test ics == Set(EarthSciIntegrator.probe_state(k) for (k, t) in enumerate(terms) if any(p -> first(p) == t.variable, m.flat.field_ics))
    end

    @testset "shared subexpressions are templates that expand to the terms (§5.2.2)" begin
        templated = 0
        for (fixture, case) in fixture_cases()
            ctx = EarthSciIntegrator.plan_context(EarthSciIntegrator.read_inputs(joinpath(PFIX, fixture, "$case.run.json")), DiffBackend())
            exprs = EarthSciAST.ASTExpr[t.expr for t in EarthSciIntegrator.build_terms(ctx.model)]
            templates, rewritten = EarthSciIntegrator.shared_templates(exprs; min_nodes = 1)
            templated += !isempty(templates)
            reg = Dict{String, EarthSciAST.ASTExpr}(templates)
            @test length(reg) == length(templates)
            key(e) = jcs(EarthSciAST.serialize_expression(e))
            @test key.(EarthSciAST.ASTExpr[EarthSciASTDiff.expand_templates(e, reg) for e in rewritten]) == key.(exprs)
        end
        @test templated > 0
    end

    @testset "a subexpression inside a template is a template only when it is shared outside it (§5.2.2)" begin
        E(x) = EarthSciAST.expression_from_json(x)
        key(e) = jcs(EarthSciAST.serialize_expression(e))
        isref(e) = e isa EarthSciAST.OpExpr && e.op == "apply_expression_template"
        inner = Dict("op" => "*", "args" => ["a", Dict("op" => "+", "args" => ["b", "c"])])
        outer = Dict("op" => "-", "args" => [inner, "d"])
        # `outer` occurs twice (two distinct but equal nodes); `inner` and its
        # parts only inside it, so only `outer` is a template.
        exprs = EarthSciAST.ASTExpr[E(outer), E(Dict("op" => "exp", "args" => [outer]))]
        templates, rewritten = EarthSciIntegrator.shared_templates(exprs; min_nodes = 1)
        @test first.(templates) == ["shared_1"]
        @test key(last(only(templates))) == key(E(outer))
        @test isref(rewritten[1]) && isref(only(rewritten[2].args))
        # When `inner` also occurs on its own, it is a template too, and
        # `outer`'s body references it.
        exprs = vcat(exprs, E(Dict("op" => "sin", "args" => [inner])))
        templates, rewritten = EarthSciIntegrator.shared_templates(exprs; min_nodes = 1)
        reg = Dict{String, EarthSciAST.ASTExpr}(templates)
        @test length(templates) == 2
        @test isref(reg["shared_1"].args[1])
        @test key.(EarthSciAST.ASTExpr[EarthSciASTDiff.expand_templates(e, reg) for e in rewritten]) == key.(exprs)
        # Below `min_nodes` operator nodes, nothing is a template.
        @test isempty(first(EarthSciIntegrator.shared_templates(exprs; min_nodes = 4)))
    end

    @testset "the chain of templates stops at EarthSciAST's limit, outermost first (§5.2.2)" begin
        # 40 nested levels, each the sum of two copies of the one below: every
        # level below the top occurs twice in what is written.
        x = EarthSciAST.VarExpr("JacobianEval.chain")
        e = x
        for _ in 1:40
            e = EarthSciAST.OpExpr("+", EarthSciAST.ASTExpr[e, e])
        end
        templates, rewritten = EarthSciIntegrator.shared_templates(EarthSciAST.ASTExpr[e]; min_nodes = 1)
        reg = Dict{String, EarthSciAST.ASTExpr}(templates)
        isref(y) = y isa EarthSciAST.OpExpr && y.op == "apply_expression_template"
        function refs(y, out = String[])
            y isa EarthSciAST.OpExpr || return out
            isref(y) ? push!(out, y.name) : EarthSciAST.map_children(c -> (refs(c, out); c), y)
            return out
        end
        depth = Dict{String, Int}()
        chain(nm) = get!(() -> 1 + maximum(chain.(refs(reg[nm])); init = 0), depth, nm)
        @test length(templates) == EarthSciIntegrator.MAX_TEMPLATE_CHAIN == 32
        @test maximum(chain(nm) for (nm, _) in templates) == 32
        # The outermost levels are the templates: the top's two arguments are
        # references, and the innermost levels are written in place.
        @test all(isref, only(rewritten).args)
        @test isempty(refs(reg["shared_32"]))
        # (Not compared with `e` by serializing: as a tree it has 2^41 nodes.)
        # EarthSciAST loads a probe document holding them, and refuses one
        # more template on the chain.
        ctx = EarthSciIntegrator.plan_context(EarthSciIntegrator.read_inputs(joinpath(PFIX, first(fixture_cases())[1], "$(first(fixture_cases())[2]).run.json")), DiffBackend())
        terms = EarthSciIntegrator.build_terms(ctx.model)
        function with_chain(templates, rhs)
            doc = EarthSciIntegrator.probe_document(ctx.model, terms)
            v = doc["models"]["JacobianEval"]
            v["variables"]["chain"] = Dict{String, Any}("type" => v["variables"]["term_1"]["type"], "default" => 0.0)
            ts = get!(v, "expression_templates", Dict{String, Any}())
            for (nm, body) in templates
                ts[nm] = Dict{String, Any}("params" => Any[], "body" => EarthSciAST.serialize_expression(body))
            end
            lhs = Dict{String, Any}("op" => "D", "args" => Any["JacobianEval.chain"], "wrt" => "t")
            push!(v["equations"], Dict{String, Any}("lhs" => lhs, "rhs" => EarthSciAST.serialize_expression(rhs)))
            return doc
        end
        @test EarthSciAST.flatten(EarthSciAST.load_document(with_chain(templates, only(rewritten)))) !== nothing
        ref(nm) = EarthSciAST.OpExpr("apply_expression_template", EarthSciAST.ASTExpr[]; name = nm, bindings = Dict{String, EarthSciAST.ASTExpr}())
        longer = vcat(templates, ["one_more" => EarthSciAST.OpExpr("+", EarthSciAST.ASTExpr[ref("shared_1"), x])])
        err = try
            EarthSciAST.load_document(with_chain(longer, ref("one_more")))
            nothing
        catch ex
            ex
        end
        @test err !== nothing && occursin("MAX_TEMPLATE_EXPANSION_DEPTH", sprint(showerror, err))
    end

    @testset "band_model_failed names the term whose derivative fails, in pass 4" begin
        cfg = joinpath(CONF, "plans", "refusals", "band-model-failed", "run.json")
        err = try
            EarthSciIntegrator.plan(cfg); nothing
        catch e
            e
        end
        @test err isa PlannerRefusal && err.code == "band_model_failed" && err.at == "/model"
        # Pass 2 reads the terms' structure only, which takes no derivative:
        # the probe succeeds, in batches of any size, with every term's own
        # structure.
        b = DiffBackend()
        ctx = EarthSciIntegrator.plan_context(EarthSciIntegrator.read_inputs(cfg), b)
        terms = EarthSciIntegrator.build_terms(ctx.model)
        each = [EarthSciIntegrator.term_band_dicts(b, ctx, t) for t in terms]
        @test EarthSciIntegrator.probe_band_dicts(b, ctx, terms) == each
        fields(x) = ntuple(i -> getfield(x, i), fieldcount(typeof(x)))
        for large in (0, 1, 3)
            @test [map(fields, x) for x in EarthSciIntegrator.all_term_bands(b, ctx, terms; large)] ==
                [map(d -> fields(EarthSciIntegrator.BandEntry(d)), x) for x in each]
        end
        # The operator's document fails in pass 4, and the refusal names its
        # third term, erf(x), with that term's own document's message.
        @test occursin("term (K.y, 0, 1): ", err.msg) && occursin("erf", err.msg)
        # In the full mode the probe fails, and the halving search finds the
        # same term with the same message, in batches of any size.
        @test EarthSciIntegrator.probe_band_dicts(b, ctx, terms; coefficients = true) === nothing
        for large in (0, 1, 3)
            e2 = try
                EarthSciIntegrator.all_term_bands(b, ctx, terms; large, coefficients = true); nothing
            catch e
                e
            end
            @test e2 isa PlannerRefusal && e2.code == err.code && e2.msg == err.msg
        end
        # Pass 4's search, alone, refuses the same way.
        e3 = try
            EarthSciIntegrator.refuse_failing_term(b, ctx, terms); nothing
        catch e
            e
        end
        @test e3 isa PlannerRefusal && e3.msg == err.msg
        # Without the failing term, the search refuses nothing.
        @test EarthSciIntegrator.refuse_failing_term(b, ctx, terms[1:2]) === nothing
    end

    @testset "pass 4 refuses the operator's first term, in term order, whose derivative fails (§5.2.2)" begin
        # Several underivable terms, among derivable ones, in one operator:
        # the halving search names the first, (K.y, 0, 3), wherever the
        # halves fall.
        src = joinpath(CONF, "plans", "refusals", "band-model-failed")
        doc = JSON.parsefile(joinpath(src, "model.esm"))
        x, y = "x", "y"
        doc["models"]["K"]["equations"][2]["rhs"] = Dict(
            "op" => "+", "args" => Any[
                Dict("op" => "*", "args" => Any[-1, "k", y]), Dict("op" => "*", "args" => Any[2, y]),
                Dict("op" => "*", "args" => Any[3, "k", x]), Dict("op" => "erf", "args" => Any[y]),
                Dict("op" => "*", "args" => Any[5, x]), Dict("op" => "erf", "args" => Any[x]),
                Dict("op" => "*", "args" => Any[7, y]),
            ]
        )
        dir = mktempdir()
        write(joinpath(dir, "model.esm"), JSON.json(doc))
        cp(joinpath(src, "run.json"), joinpath(dir, "run.json"))
        err = try
            EarthSciIntegrator.plan(joinpath(dir, "run.json")); nothing
        catch e
            e
        end
        @test err isa PlannerRefusal && err.code == "band_model_failed" && err.at == "/model"
        @test occursin("term (K.y, 0, 3): ", err.msg) && occursin("erf", err.msg)
        b = DiffBackend()
        ctx = EarthSciIntegrator.plan_context(EarthSciIntegrator.read_inputs(joinpath(dir, "run.json")), b)
        terms = EarthSciIntegrator.build_terms(ctx.model)
        for r in (1:length(terms), 2:length(terms), 4:length(terms))
            e2 = try
                EarthSciIntegrator.refuse_failing_term(b, ctx, terms[r]); nothing
            catch e
                e
            end
            @test e2 isa PlannerRefusal && e2.msg == err.msg
        end
        e4 = try
            EarthSciIntegrator.refuse_failing_term(b, ctx, terms[6:end]); nothing
        catch e
            e
        end
        @test e4 isa PlannerRefusal && occursin("term (K.y, 0, 5): ", e4.msg)
    end

    @testset "the probe document fails exactly when a term's own document fails (§5.2.2)" begin
        dir = joinpath(CONF, "plans", "refusals")
        checked = failed = 0
        for name in sort(collect(keys(JSON.parsefile(joinpath(dir, "expected-codes.json")))))
            b = DiffBackend()
            ctx, terms = try
                inp = EarthSciIntegrator.read_inputs(joinpath(dir, name, "run.json"))
                EarthSciIntegrator.check_scope(inp.flat)
                EarthSciIntegrator.check_reserved_names(inp.file)
                c = EarthSciIntegrator.plan_context(inp, b)
                c, EarthSciIntegrator.build_terms(c.model)
            catch e
                e isa PlannerRefusal || rethrow()
                continue        # refused before pass 2
            end
            isempty(terms) && continue
            # Both modes: pass 2's structure-only one and the full one, in
            # which band-model-failed's probe and its third term fail.
            for coefficients in (false, true)
                probe = EarthSciIntegrator.probe_band_dicts(b, ctx, terms; coefficients)
                each = try
                    [EarthSciIntegrator.term_band_dicts(b, ctx, t; coefficients) for t in terms]
                catch e
                    (e isa PlannerRefusal && e.code == "band_model_failed") || rethrow()
                    nothing
                end
                # A probe that succeeds hides no term's failure, and gives its bands.
                @test (probe === nothing) == (each === nothing)
                probe === nothing || @test probe == each
                failed += probe === nothing
                checked += 1
            end
        end
        @test checked > 0 && failed > 0
    end

    @testset "write_plan removes an earlier plan's files" begin
        d = EarthSciIntegrator.plan(joinpath(PFIX, "toy_robertson", "rodas4.run.json"))
        dir = mktempdir()
        mkpath(joinpath(dir, "derived")); mkpath(joinpath(dir, "validation"))
        write(joinpath(dir, "derived", "old.rhs.esm"), "{}")
        write(joinpath(dir, "validation", "old.bandvalues.json"), "{}")
        write(joinpath(dir, "notes.txt"), "kept")
        write_plan(d, dir; bandvalues = true)
        @test !isfile(joinpath(dir, "derived", "old.rhs.esm"))
        @test !isfile(joinpath(dir, "validation", "old.bandvalues.json"))
        @test read(joinpath(dir, "notes.txt"), String) == "kept"
        @test sort!(readdir(joinpath(dir, "derived"))) == ["kinetics.jacobian.esm", "kinetics.rhs.esm"]
    end
end
