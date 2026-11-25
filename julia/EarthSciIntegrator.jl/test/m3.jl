# The M3 cases (spec/toy-problems.md §5, spec/partitioning.md §9,
# tests/conformance/m3/cases.json).
#
# Each case is planned once per `invariance` variant (an override of the
# leaf partitions, `shards` and `lanes`) and run on the variant's thread
# count; the runs must be the same bit for bit: the output states, every
# scoped leaf record's per-cell or per-shard counts and steps, and the return
# codes. `scope_agreement` compares a case with the one that differs only in
# `dt_scope`. The base run's result file and tape go to RESULTS_DIR for the
# `cross_language` comparison, which CI runs on both implementations' files.
# Plans are written to RESULTS_DIR/m3-plans/. Threads run in parallel up to
# Julia's own thread count (`julia -t 4`); with fewer, the executor tasks
# interleave, which must not change a value either.

using EarthSciIntegrator: build_plan_run, layout_dofs, result_json, write_tape, LeafRecord, plan_problem, LeafOperator,
    PlannedOperator, BlockJacobian, lane_ops, lane_rhs!, lane_jac!, lane_ft!, rhs!, jac!, ft!, CellRatios, estimate_norm,
    begin_attempt!, add_cell_ratios!
const sha256 = EarthSciIntegrator.sha256

@isdefined(RESULTS_DIR) || include("conformance.jl")
const M3_CASES_FILE = joinpath(REPO, "tests", "conformance", "m3", "cases.json")
const M3_PLANS = joinpath(RESULTS_DIR, "m3-plans")
const M3_CACHE = Dict{Any, Any}()

"""The case's run config with every leaf that has a `partition` given the variant's `shards` and `lanes`, and its model as an absolute path."""
function m3_run_config(case, variant)
    rc = deepcopy(case["run_config"])
    rc["model"] = abspath(joinpath(REPO, "tests", "conformance", "m3", rc["model"]))
    function walk(x)
        if x isa AbstractDict
            if haskey(x, "leaf") && haskey(x, "partition")
                for k in ("shards", "lanes")
                    haskey(variant, k) && (x["partition"][k] = variant[k])
                end
            end
            foreach(walk, values(x))
        elseif x isa AbstractVector
            foreach(walk, x)
        end
        return
    end
    walk(rc)
    return rc
end

"""Plan the case's run config for `variant`, build it on the variant's threads and run it."""
function m3_run(case, variant; whole_state = false)
    id = case["id"]
    rc = m3_run_config(case, variant)
    text = JSON.json(rc)
    tag = bytes2hex(sha256(text))[1:12]
    mkpath(M3_PLANS)
    cfg = joinpath(M3_PLANS, "$id.$tag.run.json")
    write(cfg, text)
    plandir = joinpath(M3_PLANS, "$id.$tag")
    isfile(joinpath(plandir, "plan.json")) || write_plan(plan(cfg; tspan = case["tspan"]), plandir)
    run, pp = build_plan_run(
        plandir; saveat = case["saveat"], tspan = case["tspan"], model = rc["model"], cache = M3_CACHE,
        cache_root = joinpath(M3_PLANS, "data-cache"), threads = get(variant, "threads", 1), whole_state
    )
    forward!(run)
    return run, pp
end

"""Every leaf record of the run's tape, as what the invariance check compares: steps, counts, per-cell counts, per-shard records, return code."""
unit_records(run) = [
    (
        r.path, reinterpret(UInt64, Float64[x for s in r.steps for x in s]), r.accepted, r.rejected, r.retcode, r.cells,
        r.shards === nothing ? nothing :
            [(s.cells, reinterpret(UInt64, Float64[x for st in s.steps for x in st]), s.accepted, s.rejected) for s in r.shards],
    ) for r in run.tape.records if r isa LeafRecord
]

"""Whether two runs are the same bit for bit (spec/partitioning.md §9.1); a reason when not."""
function bitwise_diff(a, b)
    a.retcode == b.retcode || return "return codes $(a.retcode) and $(b.retcode)"
    first.(a.outputs) == first.(b.outputs) || return "output times differ"
    for ((t, x), (_, y)) in zip(a.outputs, b.outputs)
        i = findfirst(j -> reinterpret(UInt64, x[j]) != reinterpret(UInt64, y[j]), eachindex(x))
        i === nothing || return "state at t = $t, DOF $i: $(x[i]) against $(y[i])"
    end
    ra, rb = unit_records(a), unit_records(b)
    length(ra) == length(rb) || return "$(length(ra)) leaf records against $(length(rb))"
    for (x, y) in zip(ra, rb)
        x == y || return "leaf record $(x[1]) differs"
    end
    return nothing
end

"""`scope_agreement`: the output states agree to `rtol` relative to each component's largest magnitude over the run in either result."""
function scope_agreement(a, b, dofs, rtol)
    length(a.outputs) == length(b.outputs) || return false, "$(length(a.outputs)) outputs against $(length(b.outputs))"
    worst = 0.0
    for j in dofs
        scale = max(maximum(abs(x[j]) for (_, x) in a.outputs), maximum(abs(x[j]) for (_, x) in b.outputs), floatmin())
        for ((_, x), (_, y)) in zip(a.outputs, b.outputs)
            worst = max(worst, abs(x[j] - y[j]) / scale)
        end
    end
    return worst <= rtol, "worst relative difference $worst, rtol $rtol"
end

@testset "M3 cases" begin
    cases = JSON.parsefile(M3_CASES_FILE)["cases"]
    pat = get(ENV, "EARTHSCIINTEGRATOR_CASES", "")
    only_case(cs, id) = cs[findfirst(c -> c["id"] == id, cs)]
    mkpath(RESULTS_DIR)
    bases = Dict{String, Any}()
    base_of(c) = get!(() -> m3_run(c, Dict{String, Any}()), bases, c["id"])
    for case in cases
        id = case["id"]
        isempty(pat) || occursin(Regex(pat), id) || continue
        @testset "$id" begin
            run, pp = base_of(case)
            @test run.retcode == "Success"
            dofs = layout_dofs(pp, case["layout"])
            write(joinpath(RESULTS_DIR, "$id.json"), result_json(id, run; dofs))
            open(io -> write_tape(io, run.tape), joinpath(RESULTS_DIR, "$id.tape.jsonl"), "w")
            for chk in case["checks"]
                name = chk["check"]
                if name == "invariance"
                    @testset "invariance" begin
                        EarthSciIntegrator.EXECUTOR_THREADS[] = 0
                        runs = [m3_run(case, v)[1] for v in chk["variants"]]
                        # With several Julia threads, the variants with
                        # several executor threads run on several (CI runs
                        # the tests on four).
                        if Threads.nthreads() > 1 && any(v -> get(v, "threads", 1) > 1, chk["variants"]) && !occursin("global", id)
                            @test EarthSciIntegrator.EXECUTOR_THREADS[] > 1
                        end
                        @info "$id: executor tasks on at most $(EarthSciIntegrator.EXECUTOR_THREADS[]) threads"
                        @info "$id: invariance over $(length(runs)) variants"
                        for (k, r) in enumerate(runs)
                            k == 1 && continue
                            why = bitwise_diff(runs[1], r)
                            ok = why === nothing
                            @test ok
                            record!(id, "invariance", ok, ok ? "variant $(chk["variants"][k]) equals $(chk["variants"][1]) bit for bit" : "variant $(chk["variants"][k]): $why")
                        end
                    end
                elseif name == "scope_agreement"
                    @testset "scope_agreement" begin
                        other = only_case(cases, chk["against"])
                        orun, opp = base_of(other)
                        ok, detail = scope_agreement(run, orun, dofs, chk["rtol"])
                        @test ok
                        record!(id, "scope_agreement", ok, "with $(chk["against"]): $detail")
                        @info "$id: scope_agreement with $(chk["against"]): $detail"
                    end
                else
                    @test name == "cross_language"   # compared in CI
                end
            end
        end
    end

    # spec/partitioning.md §3.1 rule 4, §9.3: a cell run through the lane
    # programs equals the same run evaluating every operator over the whole
    # leaf state; the Rosenbrock case calls rhs, jac and ft, the shard case
    # runs several cells per lane call.
    @testset "lane programs equal the whole-domain program in a run" begin
        for (id, v) in (("m3.reseact_mini.cell", Dict{String, Any}()), ("m3.reseact_mini.shard", Dict{String, Any}("lanes" => 3)))
            isempty(pat) || occursin(Regex(pat), id) || continue
            case = only_case(cases, id)
            lanes, _ = m3_run(case, v)
            whole, _ = m3_run(case, v; whole_state = true)
            why = bitwise_diff(lanes, whole)
            @test why === nothing
            why === nothing || @info "$id: lanes against the whole state: $why"
        end
    end

    # §3.2, U19: under `cell` a lane pool serves cells at different times in
    # one lane-program call (the run's values are the invariance checks').
    @testset "a lane pool serves cells at different times in one call" begin
        id = "m3.reseact_mini.cell"
        if isempty(pat) || occursin(Regex(pat), id)
            EarthSciIntegrator.MIXED_TIME_LANES[] = 0
            run, _ = m3_run(only_case(cases, id), Dict{String, Any}("lanes" => 8, "threads" => 2))
            @test run.retcode == "Success"
            @test EarthSciIntegrator.MIXED_TIME_LANES[] > 1
            @info "$id: at most $(EarthSciIntegrator.MIXED_TIME_LANES[]) distinct times in one lane-program call"
        end
    end

    # §9.3: for every cell, the lane-program values of rhs, jac and ft at
    # capacities 1, 3 and 8, with the cell in every lane position and with
    # padding, equal the whole-domain values bit for bit.
    @testset "lane programs are lane-exact" begin
        case = only_case(cases, "m3.reseact_mini.cell")
        if isempty(pat) || occursin(Regex(pat), case["id"])
            m3_run(case, Dict{String, Any}())   # plans it
            rc = m3_run_config(case, Dict{String, Any}())
            tag = bytes2hex(sha256(JSON.json(rc)))[1:12]
            pp = plan_problem(
                joinpath(M3_PLANS, "$(case["id"]).$tag"); cache = M3_CACHE, model = rc["model"],
                cache_root = joinpath(M3_PLANS, "data-cache")
            )
            op = pp.prob.operators["chemistry"]
            L = pp.layout
            species = ["Mini.NO", "Mini.NO2", "Mini.O3"]
            nc = 120
            rng = Random.MersenneTwister(3)
            u = Float64[(startswith(L.names[j], "Mini.O3") ? 40.0 : 10.0) * (0.5 + rand(rng)) for j in eachindex(L.names)]
            p, t = pp.prob.p, 80400.0
            F = op.forcing
            EarthSciIntegrator.epoch_buffers(F, EarthSciIntegrator.epoch_of(F, t))
            whole_f = rhs!(op, zeros(length(u)), u, p, t)
            whole_ft = ft!(op, zeros(length(u)), u, p, t)
            whole_J = BlockJacobian{Float64}(nc, 3)
            jac!(op, whole_J, u, p, t)
            bad = 0
            for cap in (1, 3, 8)
                lo = lane_ops(op, species, cap, pp.prob.pnames, pp.prob.tspan)
                for c in 1:nc, pos in 1:cap, nb in (cap, max(pos, cap - 1))
                    pos <= nb || continue
                    # nb cells, the cell at position pos, the others drawn at
                    # random; nb < cap pads the call.
                    cells = [k == pos ? c : rand(rng, 1:nc) for k in 1:nb]
                    x = Float64[u[L.offset[s] + cc] for s in species for cc in cells]
                    f = lane_rhs!(lo, zeros(3 * nb), cells, x, u, p, t)
                    g = lane_ft!(lo, zeros(3 * nb), cells, x, u, p, t)
                    J = lane_jac!(lo, BlockJacobian{Float64}(nb, 3), cells, x, u, p, t)
                    for i in 1:3
                        d = L.offset[species[i]] + c
                        bad += reinterpret(UInt64, f[(i - 1) * nb + pos]) != reinterpret(UInt64, whole_f[d])
                        bad += reinterpret(UInt64, g[(i - 1) * nb + pos]) != reinterpret(UInt64, whole_ft[d])
                        for j in 1:3
                            a = EarthSciIntegrator._entry(J, (i - 1) * nb + pos, (j - 1) * nb + pos)
                            w = EarthSciIntegrator._entry(whole_J, (i - 1) * nc + c, (j - 1) * nc + c)
                            bad += reinterpret(UInt64, a) != reinterpret(UInt64, w)
                        end
                    end
                end
            end
            @test bad == 0
        end
    end
end

# spec/partitioning.md §2.3, §2.4: an IMEX Euler leaf of pointwise operators
# runs per cell and per shard, each unit's implicit part solved alone (as the
# Rust runtime's units do). Its NO2 terms are the explicit part, the rest the
# implicit part. With `dt_scope` omitted the planner resolves the leaf to
# `cell` (spec/plan.md §7.4). Every scope agrees with `global` to the Newton
# iteration's tolerance, and each is the same bit for bit on any threads and
# lanes.
@testset "M3: IMEX Euler leaves per cell and per shard" begin
    pat = get(ENV, "EARTHSCIINTEGRATOR_CASES", "")
    if isempty(pat) || occursin(Regex(pat), "m3.imex")
        function imex_case(scope)
            control = Dict{String, Any}("reltol" => 1.0e-8, "abstol" => 1.0e-12)
            scope === nothing || (control["dt_scope"] = scope)
            leaf = Dict{String, Any}(
                "leaf" => ["photo_no2", "photo_rest"], "alg" => "IMEXEuler", "dt" => 60,
                "imex" => Dict("explicit" => ["photo_no2"], "implicit" => ["photo_rest"]),
                "control" => control, "partition" => Dict{String, Any}("shards" => 3),
            )
            rc = Dict{String, Any}(
                "esm_integrator" => "0.1.0", "model" => "../toys/photostationary_cells.esm",
                "operators" => Dict(
                    "photo_no2" => Dict("select" => Dict("variables" => ["Photo.NO2"])),
                    "photo_rest" => Dict("select" => Dict("variables" => ["Photo.NO", "Photo.O3"])),
                ),
                "schedule" => leaf, "step" => Dict("dt" => 600, "hit" => ["output"]), "output" => Dict("saveat" => 600),
            )
            return Dict{String, Any}(
                "id" => "m3.imex.$(something(scope, "auto"))", "run_config" => rc, "tspan" => [0.0, 3600.0],
                "saveat" => [600.0 * k for k in 1:6],
            )
        end
        layout = ["Photo.NO2", "Photo.NO", "Photo.O3"]
        global_run, gpp = m3_run(imex_case("global"), Dict{String, Any}())
        @test global_run.retcode == "Success"
        dofs = layout_dofs(gpp, layout)
        for scope in ("cell", "shard", nothing)
            case = imex_case(scope)
            base, pp = m3_run(case, Dict{String, Any}())
            @test base.retcode == "Success"
            recs = [r for r in base.tape.records if r isa LeafRecord]
            @test !isempty(recs)
            # The leaf ran per cell (per shard): its records carry per-unit counts.
            @test all(r -> (scope == "shard" ? r.shards !== nothing : r.cells !== nothing), recs)
            ok, detail = scope_agreement(base, global_run, dofs, 1.0e-6)
            @test ok
            ok || @info "$(case["id"]) against global: $detail"
            # Under `shard` the shards are the units, so only lanes and
            # threads vary (§9.1).
            variants = scope == "shard" ? (Dict("lanes" => 4, "threads" => 2), Dict("lanes" => 5, "threads" => 4)) :
                (Dict("lanes" => 4, "threads" => 2), Dict("shards" => 7, "lanes" => 5, "threads" => 4))
            for v in variants
                why = bitwise_diff(base, m3_run(case, Dict{String, Any}(v))[1])
                @test why === nothing
                why === nothing || @info "$(case["id"]) $v: $why"
            end
        end
    end
end

# spec/partitioning.md §6.2: which attempts add to a shard unit's R_c. Only
# the ratios of the norm the controller decides on (`estimate_norm`), never
# another norm's (`error_norm`, a Newton increment's); an attempt with no
# estimate, or a non-finite one, adds nothing; attempts add in order.
@testset "M3: the re-sharding signal's attempts" begin
    m = 2
    tol = Tolerances(0.0, ones(2m), collect(1:(2m)), CellRatios(m))
    u = zeros(2m)
    e = [1.0, 2.0, 3.0, 4.0]        # species-major: cell 1 has DOFs 1 and 3
    # An attempt whose only norm is not the estimate adds nothing.
    begin_attempt!(tol)
    error_norm(e, u, u, tol)
    add_cell_ratios!(tol, 0.5)
    @test tol.cells.acc == [0.0, 0.0]
    # The estimate's ratios, in species order per cell.
    begin_attempt!(tol)
    E = estimate_norm(e, u, u, tol)
    error_norm(100 .* e, u, u, tol)  # a later norm of another kind changes nothing
    add_cell_ratios!(tol, E)
    @test tol.cells.acc == [1.0 + 9.0, 4.0 + 16.0]
    # A non-finite estimate adds nothing, and neither does the next attempt
    # when it computes no estimate (the stale ratios stay unused).
    begin_attempt!(tol)
    E = estimate_norm([Inf, 0.0, 0.0, 0.0], u, u, tol)
    add_cell_ratios!(tol, E)
    begin_attempt!(tol)
    add_cell_ratios!(tol, 2.0)
    @test tol.cells.acc == [10.0, 20.0]

    # The shard case: the tape writes a shard's cells on the first record and
    # after the composition changed only (§11), and every checkpoint holds the
    # signal summed so far.
    cases = JSON.parsefile(M3_CASES_FILE)["cases"]
    case = cases[findfirst(c -> c["id"] == "m3.reseact_mini.shard", cases)]
    pat = get(ENV, "EARTHSCIINTEGRATOR_CASES", "")
    if isempty(pat) || occursin(Regex(pat), case["id"])
        run, _ = m3_run(case, Dict{String, Any}())
        recs = [r for r in run.tape.records if r isa LeafRecord && r.shards !== nothing]
        @test !isempty(recs)
        @test all(sh -> sh.cells !== nothing, recs[1].shards)
        # Each shard's cells as last listed; a record lists them exactly when
        # they changed. Shards are in order of their smallest cell (§2.4).
        prev = [sh.cells for sh in recs[1].shards]
        listed = 0
        sorted = issorted(first.(prev))
        for r in recs[2:end]
            for (i, sh) in enumerate(r.shards)
                sh.cells === nothing && continue
                @test sh.cells != prev[i]
                prev[i] = sh.cells
                listed += 1
            end
            sorted &= issorted(first.(prev))
        end
        @test sorted
        @info "m3.reseact_mini.shard: $(length(recs)) records, $listed later shard entries list their cells"
        @test listed < (length(recs) - 1) * length(recs[1].shards)
        saved = [cp.units["chem"] for cp in run.checkpoints]
        @test all(sv -> length(sv.signal) == 120 && all(isfinite, sv.signal), saved)
        @test any(sv -> any(>(0), sv.signal), saved)
        s = EarthSciIntegrator.ScopeState(:shard, 120, 3, 4, 2, 8)
        EarthSciIntegrator.restore!(s, saved[3])
        @test s.signal == saved[3].signal && s.shards == saved[3].shards && s.macro_step == saved[3].macro_step
    end
end
