# The M2 end-to-end cases (spec/toy-problems.md §4, tests/conformance/m2/cases.json).
#
# Each case is planned (spec/plan.md), its plan is built into operators and run
# (spec/operator-interface.md §8, `build_plan_run`), and its result file is
# checked against this implementation's result file for the M1 case it names
# (`toy_equivalence`, §4.3). The M1 result files are the ones conformance.jl
# writes to RESULTS_DIR; a missing one (when EARTHSCIINTEGRATOR_CASES filters
# the M1 case out) is produced here by running the M1 case. Plans are written
# to RESULTS_DIR/m2-plans/<case>/, their run configs to RESULTS_DIR/m2-plans/m2/.

using EarthSciIntegrator: build_plan_run, layout_dofs, result_json, write_tape, read_tape, LeafRecord, UnsupportedError
using Random

@isdefined(RESULTS_DIR) || include("conformance.jl")
const M2_CASES_FILE = joinpath(REPO, "tests", "conformance", "m2", "cases.json")
const M2_PLANS = joinpath(RESULTS_DIR, "m2-plans")

"""Every leaf record's accepted steps `(t, h)`, in tape order."""
tape_steps(tape) = [r.steps for r in tape.records if r isa LeafRecord]

"""
    toy_equivalence(m2, m1, steps2, steps1, chk) -> (passed, detail)

spec/toy-problems.md §4.3 by the comparison rule of the `cross_language`
check (§2.1): the two result files' states agree to `rtol`, relative to each
component's largest magnitude over the run in either result, when the step
sequences agree (the same accepted step count and every step's `t` and `h`),
and to `rtol_steps_differ` when they differ. Only states and step sequences
are compared. `steps2` and `steps1` are the runs' leaf steps from their tapes.
"""
function toy_equivalence(m2, m1, steps2, steps1, chk)
    t2, t1 = Float64.(m2["t"]), Float64.(m1["t"])
    t2 == t1 || return false, "output times differ: M2 $t2, M1 $t1"
    u2 = [Float64.(x) for x in m2["u"]]
    u1 = [Float64.(x) for x in m1["u"]]
    ncomp = length(u1[1])
    all(x -> length(x) == ncomp, vcat(u2, u1)) || return false, "state lengths differ"
    a2, a1 = m2["stats"]["accepted"], m1["stats"]["accepted"]
    differ = a2 != a1 || steps2 != steps1
    rtol = chk["rtol"]
    note = a2 == a1 ? "accepted $a2" : "accepted M2 $a2, M1 $a1"
    if differ
        note *= ", step sequences differ"
        haskey(chk, "rtol_steps_differ") ||
            return false, "step sequences differ and the case has no rtol_steps_differ [$note]"
        rtol = chk["rtol_steps_differ"]
    end
    worst, where = 0.0, ""
    for j in 1:ncomp
        scale = max(maximum(abs(x[j]) for x in u2), maximum(abs(x[j]) for x in u1))
        for k in eachindex(t1)
            d = abs(u2[k][j] - u1[k][j])
            d = scale > 0 ? d / scale : d
            if d > worst
                worst, where = d, "component $(j - 1) at t = $(t1[k])"
            end
        end
    end
    return worst <= rtol, "largest scaled state difference $worst$(isempty(where) ? "" : " ($where)") [$note; rtol $rtol]"
end

"""The M1 case `id`'s result file and tape in RESULTS_DIR, running the case first when they are missing."""
function m1_result(id, m1cases)
    res, tp = joinpath(RESULTS_DIR, "$id.json"), joinpath(RESULTS_DIR, "$id.tape.jsonl")
    if !(isfile(res) && isfile(tp))
        case = only(filter(c -> c["id"] == id, m1cases))
        run = forward!(build_case(case))
        run.retcode == "Success" || error("M1 case $id ended with $(run.retcode)")
        grad = run.functional === nothing ? nothing : gradient(run)
        mkpath(RESULTS_DIR)
        write(res, result_json(id, run; gradient = grad))
        open(io -> write_tape(io, run.tape), tp, "w")
    end
    return JSON.parsefile(res), open(read_tape, tp)
end

"""Plan, build and run one M2 case, write its result file and tape, and run its checks."""
function run_m2_case(case, m1cases, cache)
    id = case["id"]
    return @testset "$id" begin
        # The case's model is relative to the cases file (spec/toy-problems.md
        # §4.2). Its run config is written to M2_PLANS/m2/, beside a copy of
        # the toy documents in M2_PLANS/toys/, so the model string resolves as
        # written and the plan's inputs keep it (spec/plan.md §7.7).
        cfgpath = joinpath(M2_PLANS, "m2", "$id.json")
        write(cfgpath, JSON.json(case["run_config"]))
        draft = plan(cfgpath; tspan = case["tspan"])
        plandir = joinpath(M2_PLANS, id)
        write_plan(draft, plandir)
        run, pp = build_plan_run(plandir; saveat = get(case, "saveat", nothing), tspan = case["tspan"], cache)
        forward!(run)
        @test run.retcode == "Success"
        run.retcode == "Success" || return
        write(joinpath(RESULTS_DIR, "$id.json"), result_json(id, run; dofs = layout_dofs(pp, case["layout"])))
        open(io -> write_tape(io, run.tape), joinpath(RESULTS_DIR, "$id.tape.jsonl"), "w")
        m2res = JSON.parsefile(joinpath(RESULTS_DIR, "$id.json"))
        for chk in case["checks"]
            name = chk["check"]
            @test name == "toy_equivalence"
            @testset "$name" begin
                m1res, m1tape = m1_result(chk["against"], m1cases)
                ok, detail = toy_equivalence(m2res, m1res, tape_steps(run.tape), tape_steps(m1tape), chk)
                record!(id, name, ok, detail)
            end
        end
    end
end

@testset "M2 end-to-end cases" begin
    file = JSON.parsefile(M2_CASES_FILE)
    cases = file["cases"]
    @test length(cases) == 41
    @test file["esm_integrator"] == string(SPEC_VERSION)
    m1cases = JSON.parsefile(CASES_FILE)["cases"]
    only = get(ENV, "EARTHSCIINTEGRATOR_CASES", "")
    cache = Dict{Any, Any}()     # built derived documents, by hash, shared across plans
    rm(M2_PLANS; force = true, recursive = true)
    mkpath(joinpath(M2_PLANS, "m2"))
    cp(joinpath(REPO, "tests", "conformance", "toys"), joinpath(M2_PLANS, "toys"))
    n0 = length(OUTCOMES)
    elapsed = @elapsed for case in cases
        isempty(only) || occursin(Regex(only), case["id"]) || continue
        run_m2_case(case, m1cases, cache)
    end
    outs = OUTCOMES[(n0 + 1):end]
    println(
        "\nM2 end-to-end: ", length(unique(o.case for o in outs)), " cases run in ", round(elapsed; digits = 1), " s, ",
        count(o -> o.passed, outs), "/", length(outs), " checks passed"
    )
    for o in outs
        println(rpad(o.passed ? "  pass" : "  FAIL", 7), rpad(o.case, 46), rpad(o.check, 16), o.detail)
    end
end

# The planned operators against the hand-written toy operators of the same
# problem, at a perturbed initial state: rhs, jac, jvp_u and vjp_u, through the
# case's layout (the toy's DOF order). Uses the plans the cases above wrote.
@testset "planned operators match the hand-written ones" begin
    file = JSON.parsefile(M2_CASES_FILE)
    m1cases = JSON.parsefile(CASES_FILE)["cases"]
    cache = Dict{Any, Any}()
    rng = Random.MersenneTwister(7)
    for id in (
            "m2.forced_decay.BackwardEuler.fixed", "m2.van_der_pol.stiff.Rodas4.adaptive", "m2.robertson.Rodas4.adaptive",
            "m2.photostationary_cells.Kvaerno4.adaptive", "m2.advection_reaction.strang.order",
        )
        plandir = joinpath(M2_PLANS, id)
        isfile(joinpath(plandir, "plan.json")) || continue      # case filtered out above
        case = only(filter(c -> c["id"] == id, file["cases"]))
        m1 = only(filter(c -> c["id"] == case["checks"][1]["against"], m1cases))
        toy = toy_problem(m1["run_config"]["model"])
        pp = EarthSciIntegrator.plan_problem(plandir; cache)
        dofs = layout_dofs(pp, case["layout"])
        @test pp.prob.u0[dofs] == toy.u0
        N = pp.prob.N
        @test N == toy.N && sort(dofs) == 1:N
        x = toy.u0 .* (1 .+ 0.1 .* rand(rng, N)) .+ 0.01 .* rand(rng, N)
        u = zeros(N)
        u[dofs] = x
        t = toy.tspan[1] + 0.3 * (toy.tspan[2] - toy.tspan[1])
        v, w = randn(rng, N), randn(rng, N)
        for (name, op) in pp.prob.operators
            ref = toy.operators[name]
            @test EarthSciIntegrator.writes(op, N) == sort(dofs[EarthSciIntegrator.writes(ref, N)])
            @test EarthSciIntegrator.depends_on_t(op) == EarthSciIntegrator.depends_on_t(ref)
            @test EarthSciIntegrator.linear_in_u(op) == EarthSciIntegrator.linear_in_u(ref)
            f, fr = rhs!(op, zeros(N), u, pp.prob.p, t), rhs!(ref, zeros(N), x, toy.p, t)
            @test isapprox(f[dofs], fr; rtol = 1.0e-12, atol = 1.0e-12 * maximum(abs, fr))
            J = EarthSciIntegrator.DenseJacobian{Float64}(N)
            Jr = EarthSciIntegrator.DenseJacobian{Float64}(N)
            jac!(op, J, u, pp.prob.p, t)
            jac!(ref, Jr, x, toy.p, t)
            @test isapprox(J.A[dofs, dofs], Jr.A; rtol = 1.0e-12, atol = 1.0e-12 * maximum(abs, Jr.A))
            @test isapprox(jvp_u!(op, zeros(N), u, pp.prob.p, t, v), J.A * v; rtol = 1.0e-12)
            @test isapprox(vjp_u!(op, zeros(N), u, pp.prob.p, t, w), J.A' * w; rtol = 1.0e-12)
            # The band values are reused only at the point they were evaluated at.
            u2 = u .* 1.07 .+ 0.003
            J2 = EarthSciIntegrator.DenseJacobian{Float64}(N)
            Jr2 = EarthSciIntegrator.DenseJacobian{Float64}(N)
            @test isapprox(jvp_u!(op, zeros(N), u2, pp.prob.p, t, v), (jac!(op, J2, u2, pp.prob.p, t); J2.A) * v; rtol = 1.0e-12)
            jac!(ref, Jr2, u2[dofs], toy.p, t)
            @test isapprox(J2.A[dofs, dofs], Jr2.A; rtol = 1.0e-12, atol = 1.0e-12 * maximum(abs, Jr2.A))
            @test isapprox(vjp_u!(op, zeros(N), u, pp.prob.p, t, w), J.A' * w; rtol = 1.0e-12)
            # §8.3: the calls planned operators do not provide at M2.
            for call in (:ft, :vjp_p, :hess_u, :hess_p, :ft_vjp_u, :ft_vjp_p)
                @test !EarthSciIntegrator.provides(op, call)
            end
        end
        if haskey(pp.prob.operators, "photochemistry")
            # Pointwise with a block plan, over arrays of one shape: block_diagonal.
            @test EarthSciIntegrator.structure(pp.prob.operators["photochemistry"]) ==
                EarthSciIntegrator.BlockDiagonalStructure(16, 3)
        end
        if haskey(pp.prob.operators, "transport")
            @test EarthSciIntegrator.structure(pp.prob.operators["transport"]) isa EarthSciIntegrator.DenseStructure
        end
    end
end

"""An operator written by hand that depends on t and provides no ft."""
struct M2TimeDependentOp <: EarthSciIntegrator.AbstractOperator end
EarthSciIntegrator.depends_on_t(::M2TimeDependentOp) = true

@testset "plans the runtime refuses" begin
    plandir = joinpath(M2_PLANS, "m2.forced_decay.BackwardEuler.fixed")
    if isfile(joinpath(plandir, "plan.json"))
        pp = EarthSciIntegrator.plan_problem(plandir)
        # A Rosenbrock leaf on an operator that depends on t needs ft, which a
        # planned operator does not provide at M2 (spec/operator-interface.md §8.3).
        cfg = Dict{String, Any}(
            "esm_integrator" => "0.1.0", "step" => Dict("dt" => 2),
            "schedule" => Dict("leaf" => "decay", "alg" => "Rodas4", "dt" => 0.1),
        )
        err = try
            build_run(cfg; tspan = (0, 2), prob = pp.prob); nothing
        catch e
            e
        end
        @test err isa ConfigError && occursin("decay", err.msg) && occursin("ft", err.msg)
        # §8.3 refuses that leaf under any Rosenbrock method; an operator
        # written by hand needs ft only when the method forms f_t (§3).
        rodas4 = EarthSciIntegrator.RosenbrockMethod(EarthSciIntegrator.TABLEAUS["Rodas4"])
        fields = [getfield(rodas4, f) for f in fieldnames(EarthSciIntegrator.RosenbrockMethod)]
        no_ft = EarthSciIntegrator.RosenbrockMethod(fields[1:(end - 1)]..., false)
        decay = pp.prob.operators["decay"]
        hand = M2TimeDependentOp()
        @test EarthSciIntegrator.depends_on_t(decay) && EarthSciIntegrator.depends_on_t(hand)
        @test EarthSciIntegrator.rosenbrock_needs_ft(no_ft, decay)
        @test !EarthSciIntegrator.rosenbrock_needs_ft(no_ft, hand)
        @test EarthSciIntegrator.rosenbrock_needs_ft(rodas4, hand)
        # A plan's step writes `hit` as a list; null is a config error, not a crash.
        cfg = Dict{String, Any}(
            "esm_integrator" => "0.1.0", "step" => Dict("dt" => 2, "hit" => nothing),
            "schedule" => Dict("leaf" => "decay", "alg" => "BackwardEuler", "dt" => 0.1),
        )
        err = try
            build_run(cfg; tspan = (0, 2), prob = pp.prob); nothing
        catch e
            e
        end
        @test err isa ConfigError && occursin("hit", err.msg)
        # A derived document that does not match the plan's hash: the plan is stale.
        stale = mktempdir(RESULTS_DIR)
        cp(plandir, joinpath(stale, "plan"))
        open(io -> write(io, " "), joinpath(stale, "plan", "derived", "decay.rhs.esm"), "a")
        err = try
            build_plan_run(joinpath(stale, "plan")); nothing
        catch e
            e
        end
        @test err isa ConfigError && occursin("stale", err.msg)
        # A run interval other than the plan's.
        err = try
            build_plan_run(plandir; tspan = (0, 3)); nothing
        catch e
            e
        end
        @test err isa ConfigError && occursin("stale", err.msg)
        rm(stale; recursive = true)
        # An adjoint with respect to parameters needs vjp_p, which a planned
        # operator does not provide at M2 (§8.3). With respect to u0 it is
        # accepted: the leaf writes every state.
        cfg = Dict{String, Any}(
            "esm_integrator" => "0.1.0", "step" => Dict("dt" => 2),
            "schedule" => Dict("leaf" => "decay", "alg" => "Heun", "dt" => 0.1),
            "adjoint" => Dict("functional" => Dict("sum" => "Decay.u", "over" => "final"), "wrt" => ["params:numeric"]),
        )
        err = try
            build_run(cfg; tspan = (0, 2), prob = pp.prob); nothing
        catch e
            e
        end
        @test err isa ConfigError && occursin("decay", err.msg) && occursin("vjp_p", err.msg)
        cfg["adjoint"]["wrt"] = ["u0"]
        @test build_run(cfg; tspan = (0, 2), prob = pp.prob).wrt_u0
    end
end

@testset "planned documents: defaults and per-caller buffers" begin
    plandir = joinpath(M2_PLANS, "m2.forced_decay.BackwardEuler.fixed")
    if isfile(joinpath(plandir, "plan.json"))
        # A parameter default written as an integer agrees with itself: the
        # consistency check compares the Float64 the parameter vector holds.
        dir = mktempdir(RESULTS_DIR)
        cp(plandir, joinpath(dir, "plan"))
        plan = JSON.parsefile(joinpath(dir, "plan", "plan.json"))
        for kind in ("rhs", "jacobian")
            rel = "derived/decay.$kind.esm"
            f = joinpath(dir, "plan", rel)
            doc = JSON.parsefile(f)
            doc["models"]["Decay"]["variables"]["k"]["default"] = 2
            write(f, JSON.json(doc))
            plan["derived"][rel]["sha256"] = bytes2hex(EarthSciIntegrator.sha256(read(f)))
        end
        write(joinpath(dir, "plan", "plan.json"), JSON.json(plan))
        pp = EarthSciIntegrator.plan_problem(joinpath(dir, "plan"))
        @test pp.prob.p[findfirst(==("Decay.k"), pp.prob.pnames)] === 2.0
        rm(dir; recursive = true)
        # Two callers holding workspaces at once get their own auxiliary
        # buffers, as well as their slot buffers.
        d = EarthSciIntegrator.plan_problem(plandir).prob.operators["decay"].rhs
        w1, w2 = EarthSciIntegrator._acquire(d), EarthSciIntegrator._acquire(d)
        @test w1.aux !== w2.aux && w1.U !== w2.U && w1.du !== w2.du
        EarthSciIntegrator._release(d, w1)
        EarthSciIntegrator._release(d, w2)
    end
end

# spec/operator-interface.md §8.2: a leaf of planned operators advances the
# union of its operators' writes; every other state is a constant of the leaf,
# and a block-diagonal leaf's block size is the number of its own states.
@testset "a leaf advances only its operators' writes" begin
    # advection_reaction's transport leaf writes Reaction.A only.
    plandir = joinpath(M2_PLANS, "m2.advection_reaction.strang.order")
    if isfile(joinpath(plandir, "plan.json"))
        run, pp = build_plan_run(plandir)
        A, B = EarthSciIntegrator.variable_dofs(pp.prob, "Reaction.A"), EarthSciIntegrator.variable_dofs(pp.prob, "Reaction.B")
        transport, reaction = run.leaves["root.0"], run.leaves["root.1"]
        @test transport.opnames == ["transport"] && reaction.opnames == ["reaction"]
        @test transport.state.dofs == A && length(transport.tol.abstol) == 64
        @test EarthSciIntegrator.structure(transport.ops) isa EarthSciIntegrator.DenseStructure
        @test reaction.state.dofs == vcat(A, B)
        @test EarthSciIntegrator.structure(reaction.ops) == EarthSciIntegrator.BlockDiagonalStructure(64, 2)
        # The transport leaf alone, from a state with B nonzero: B is unchanged
        # bit for bit, A is advected.
        cfg = Dict{String, Any}(
            "esm_integrator" => "0.1.0", "step" => pp.plan["step"], "schedule" => pp.plan["schedule"]["strang"][1],
        )
        u0 = copy(pp.prob.u0)
        u0[B] .= 0.25 .+ 0.5 .* rand(Random.MersenneTwister(3), 64)
        r = forward!(build_run(cfg; tspan = pp.prob.tspan, prob = pp.prob, u0))
        @test r.retcode == "Success"
        @test reinterpret(UInt64, r.u[B]) == reinterpret(UInt64, u0[B])
        @test r.u[A] != u0[A]
    end
    # A model of three states over 4 cells: chem, pointwise, writes X and Y
    # and reads Z; decay writes Z. Each leaf is block-diagonal over its own
    # states only (n = 2 and n = 1, not 3).
    dir = joinpath(M2_PLANS, "partial_leaves")
    mkpath(dir)
    fa(e) = Dict("op" => "faq", "args" => [], "output_idx" => ["i"], "ranges" => Dict("i" => Dict("from" => "cell")), "expr" => e)
    ix(v) = Dict("op" => "index", "args" => [v, "i"])
    D(x) = Dict("op" => "D", "args" => [x], "wrt" => "t")
    doc = Dict(
        "esm" => "1.3.0", "metadata" => Dict("name" => "partial_leaves"),
        "index_sets" => Dict("cell" => Dict("kind" => "interval", "size" => 4)),
        "models" => Dict(
            "Box" => Dict(
                "variables" => Dict(
                    "X" => Dict("type" => "unknown", "shape" => ["cell"], "default" => 1),
                    "Y" => Dict("type" => "unknown", "shape" => ["cell"], "default" => 0.5),
                    "Z" => Dict("type" => "unknown", "shape" => ["cell"], "default" => 2),
                    "k" => Dict("type" => "parameter", "default" => 3.5),
                    "c" => Dict("type" => "parameter", "default" => 0.25),
                ),
                "equations" => [
                    Dict("lhs" => D("X"), "rhs" => fa(Dict("op" => "+", "args" => [Dict("op" => "*", "args" => [-1, "k", ix("X")]), ix("Z")]))),
                    Dict("lhs" => D("Y"), "rhs" => fa(Dict("op" => "-", "args" => [Dict("op" => "*", "args" => ["k", ix("X")]), ix("Y")]))),
                    Dict("lhs" => D("Z"), "rhs" => fa(Dict("op" => "*", "args" => [-1, "c", ix("Z")]))),
                ],
            ),
        ),
    )
    write(joinpath(dir, "partial_leaves.esm"), JSON.json(doc))
    ctl = Dict("dt_scope" => "global", "reltol" => 1.0e-8, "abstol" => 1.0e-10)
    leaf(op, alg) = Dict("leaf" => op, "alg" => alg, "linsolve" => Dict("block_lu" => Dict("pivot" => false)), "control" => ctl)
    cfg = Dict(
        "esm_integrator" => "0.1.0", "model" => "partial_leaves.esm",
        "operators" => Dict(
            "chem" => Dict("select" => Dict("variables" => ["Box.X", "Box.Y"])),
            "decay" => Dict("select" => Dict("variables" => ["Box.Z"])),
        ),
        "schedule" => Dict("sequence" => [leaf("chem", "Rodas4"), leaf("decay", "Kvaerno4")]),
        "step" => Dict("dt" => 0.25),
    )
    write(joinpath(dir, "run-config.json"), JSON.json(cfg))
    write_plan(plan(joinpath(dir, "run-config.json"); tspan = [0, 1]), joinpath(dir, "plan"))
    run, pp = build_plan_run(joinpath(dir, "plan"))
    X, Y, Z = (EarthSciIntegrator.variable_dofs(pp.prob, "Box.$v") for v in ("X", "Y", "Z"))
    chem, decay = run.leaves["root.0"], run.leaves["root.1"]
    @test chem.state.dofs == vcat(X, Y) && decay.state.dofs == Z
    @test EarthSciIntegrator.structure(chem.ops) == EarthSciIntegrator.BlockDiagonalStructure(4, 2)
    @test EarthSciIntegrator.structure(decay.ops) == EarthSciIntegrator.BlockDiagonalStructure(4, 1)
    forward!(run)
    @test run.retcode == "Success"
    @test all(z -> isapprox(z, 2 * exp(-0.25); rtol = 1.0e-7), run.u[Z])
    # The chem leaf alone reads Z as a constant and leaves it unchanged.
    alone = Dict{String, Any}("esm_integrator" => "0.1.0", "step" => pp.plan["step"], "schedule" => pp.plan["schedule"]["sequence"][1])
    r = forward!(build_run(alone; tspan = pp.prob.tspan, prob = pp.prob))
    @test r.retcode == "Success"
    @test reinterpret(UInt64, r.u[Z]) == reinterpret(UInt64, pp.prob.u0[Z])
    @test r.u[X] != pp.prob.u0[X]
    # The leaf's Jacobian has no column for Z: dropped, not stored (§8.3).
    op = only(chem.ops.ops)
    @test all(((_, r, c),) -> 1 <= r <= 8 && 1 <= c <= 8, op.points)
end

# A model whose states do not share one shape: X, Y and W over 4 cells, Z a
# scalar. chem writes X and Y and reads only them, decay writes Z, and push
# writes W and reads Z (W' = -W + Z). The Rust tests plan the same document
# (rust/earthsci-integrator/tests/leaf_state_m2.rs).
function mixed_shapes_plan()
    dir = joinpath(M2_PLANS, "mixed_shapes")
    rm(dir; force = true, recursive = true)
    mkpath(dir)
    fa(e) = Dict("op" => "faq", "args" => [], "output_idx" => ["i"], "ranges" => Dict("i" => Dict("from" => "cell")), "expr" => e)
    ix(v) = Dict("op" => "index", "args" => [v, "i"])
    D(x) = Dict("op" => "D", "args" => [x], "wrt" => "t")
    doc = Dict(
        "esm" => "1.3.0", "metadata" => Dict("name" => "mixed_shapes"),
        "index_sets" => Dict("cell" => Dict("kind" => "interval", "size" => 4)),
        "models" => Dict(
            "Box" => Dict(
                "variables" => Dict(
                    "X" => Dict("type" => "unknown", "shape" => ["cell"], "default" => 1),
                    "Y" => Dict("type" => "unknown", "shape" => ["cell"], "default" => 0.5),
                    "W" => Dict("type" => "unknown", "shape" => ["cell"], "default" => 0.75),
                    "Z" => Dict("type" => "unknown", "default" => 2),
                    "k" => Dict("type" => "parameter", "default" => 3.5),
                    "c" => Dict("type" => "parameter", "default" => 0.25),
                ),
                "equations" => [
                    Dict("lhs" => D("X"), "rhs" => fa(Dict("op" => "*", "args" => [-1, "k", ix("X")]))),
                    Dict("lhs" => D("Y"), "rhs" => fa(Dict("op" => "-", "args" => [Dict("op" => "*", "args" => ["k", ix("X")]), ix("Y")]))),
                    Dict("lhs" => D("W"), "rhs" => fa(Dict("op" => "+", "args" => [Dict("op" => "*", "args" => [-1, ix("W")]), "Z"]))),
                    Dict("lhs" => D("Z"), "rhs" => Dict("op" => "*", "args" => [-1, "c", "Z"])),
                ],
            ),
        ),
    )
    write(joinpath(dir, "mixed_shapes.esm"), JSON.json(doc))
    ctl = Dict("dt_scope" => "global", "reltol" => 1.0e-8, "abstol" => 1.0e-10)
    block = Dict("block_lu" => Dict("pivot" => false))
    cfg = Dict(
        "esm_integrator" => "0.1.0", "model" => "mixed_shapes.esm",
        "operators" => Dict(
            "chem" => Dict("select" => Dict("variables" => ["Box.X", "Box.Y"])),
            "decay" => Dict("select" => Dict("variables" => ["Box.Z"])),
            "push" => Dict("select" => Dict("variables" => ["Box.W"])),
        ),
        "schedule" => Dict(
            "sequence" => [
                Dict("leaf" => "chem", "alg" => "Kvaerno4", "linsolve" => block, "control" => ctl),
                Dict("leaf" => "decay", "alg" => "Kvaerno4", "linsolve" => block, "control" => ctl),
                Dict("leaf" => "push", "alg" => "RK4", "dt" => 0.05),
            ]
        ),
        "step" => Dict("dt" => 0.25),
    )
    write(joinpath(dir, "run-config.json"), JSON.json(cfg))
    write_plan(plan(joinpath(dir, "run-config.json"); tspan = [0, 1]), joinpath(dir, "plan"))
    return joinpath(dir, "plan")
end

# The mixed-shapes plan, planned once and shared by the testsets below.
const MIXED_SHAPES = Ref{Any}(nothing)
function mixed_shapes_problem()
    MIXED_SHAPES[] === nothing && (MIXED_SHAPES[] = EarthSciIntegrator.plan_problem(mixed_shapes_plan()))
    return MIXED_SHAPES[]
end

"""A run config of `schedule` and `adjoint` against the mixed-shapes plan `pp`."""
function mixed_cfg(pp, schedule; adjoint = nothing)
    cfg = Dict{String, Any}("esm_integrator" => "0.1.0", "step" => pp.plan["step"], "schedule" => schedule)
    adjoint === nothing || (cfg["adjoint"] = adjoint)
    return cfg
end

# spec/operator-interface.md §8.3: a leaf's storage is block_diagonal when its
# operators are pointwise with structure.block and its own state variables
# share one shape, whatever the shapes of the model's other states.
@testset "block_diagonal over the leaf's own states" begin
    pp = mixed_shapes_problem()
    # Canonical order: Box.W, Box.X, Box.Y (4 cells each), Box.Z.
    @test pp.prob.N == 13
    # An operator's own structure is that of a leaf of it alone: chem writes
    # X and Y over 4 cells, decay the scalar Z.
    @test EarthSciIntegrator.structure(pp.prob.operators["chem"]) == EarthSciIntegrator.BlockDiagonalStructure(4, 2)
    @test EarthSciIntegrator.structure(pp.prob.operators["decay"]) == EarthSciIntegrator.BlockDiagonalStructure(1, 1)
    # §8.2: an operator's reads are its bands' columns. The adjoint's check
    # below rests on them: push reads W and Z.
    reads(o) = EarthSciIntegrator.reads(pp.prob.operators[o], pp.prob.N)
    @test reads("chem") == 5:12
    @test reads("decay") == [13]
    @test reads("push") == [1, 2, 3, 4, 13]
    leaves = pp.plan["schedule"]["sequence"]
    run = build_run(mixed_cfg(pp, Dict("sequence" => leaves)); tspan = pp.prob.tspan, prob = pp.prob)
    chem, decay, push = run.leaves["root.0"], run.leaves["root.1"], run.leaves["root.2"]
    @test chem.state.dofs == 5:12
    @test EarthSciIntegrator.structure(chem.ops) == EarthSciIntegrator.BlockDiagonalStructure(4, 2)
    @test decay.state.dofs == [13]
    @test EarthSciIntegrator.structure(decay.ops) == EarthSciIntegrator.BlockDiagonalStructure(1, 1)
    @test push.state.dofs == 1:4
    forward!(run)
    @test run.retcode == "Success"
    # X(1) = e^{-k}, Z(1) = 2 e^{-c}.
    @test all(x -> isapprox(x, exp(-3.5); rtol = 1.0e-6), run.u[5:8])
    @test isapprox(run.u[13], 2 * exp(-0.25); atol = 1.0e-7)
end

# §1, §8.3: an adjoint with respect to the initial state through a leaf of
# planned operators that read only the leaf's own states is accepted and agrees
# with finite differences of the frozen tape; one through a leaf whose operator
# reads a state the leaf does not advance is refused, and so is one with
# respect to parameters (planned operators provide no vjp_p at M2).
@testset "the adjoint through a planned leaf" begin
    pp = mixed_shapes_problem()
    leaves = pp.plan["schedule"]["sequence"]
    wrt_u0 = Dict("functional" => Dict("sum" => "Box.Y", "over" => "final"), "wrt" => ["u0"])
    ctl = Dict("dt_scope" => "global", "reltol" => 1.0e-8, "abstol" => 1.0e-10)
    erk = Dict("leaf" => "chem", "alg" => "Tsit5", "control" => ctl)
    for (what, sch) in ("Kvaerno4" => [leaves[1], leaves[2]], "Tsit5" => [erk, leaves[2]])
        run = forward!(build_run(mixed_cfg(pp, Dict("sequence" => sch); adjoint = wrt_u0); tspan = pp.prob.tspan, prob = pp.prob))
        @test run.retcode == "Success"
        _, ū0, p̄ = gradient(run)
        # Σ Y(1) does not depend on W or Z: their cotangents pass through the
        # chem leaf unchanged, and no leaf sends them anything.
        @test all(==(0.0), ū0[1:4]) && ū0[13] == 0.0
        # dY(1)/dY₀ = e^{-1}.
        @test all(y -> isapprox(y, exp(-1); atol = 1.0e-6), ū0[9:12])
        # Planned operators provide no vjp_p, so the parameter cotangent is
        # not formed: NaN, never a partial sum.
        @test all(isnan, p̄)
        case = Dict("id" => "m2.mixed_shapes.adjoint.$what")
        check_fd_gradient(case, run, Dict("rtol" => 1.0e-6, "epsilons" => [1.0e-3, 1.0e-4, 1.0e-5]))
        check_replay(case, run, Dict())
    end
    # push reads Z, which its leaf does not advance.
    err = try
        build_run(
            mixed_cfg(pp, leaves[3]; adjoint = Dict("functional" => Dict("sum" => "Box.W", "over" => "final"), "wrt" => ["u0"]));
            tspan = pp.prob.tspan, prob = pp.prob
        ); nothing
    catch e
        e
    end
    @test err isa UnsupportedError &&
        err.msg == "leaf root.2: operator push may read states the leaf does not advance; the adjoint of a leaf through its constants is not implemented (operator-interface.md §1)"
    # Parameter gradients need vjp_p. Both chem and decay lack it; the
    # refusal names the first in schedule order.
    err = try
        build_run(
            mixed_cfg(pp, Dict("sequence" => [leaves[1], leaves[2]]); adjoint = Dict("functional" => Dict("sum" => "Box.Y", "over" => "final"), "wrt" => ["params:numeric"]));
            tspan = pp.prob.tspan, prob = pp.prob
        ); nothing
    catch e
        e
    end
    @test err isa ConfigError && err.msg == "operator 'chem' of leaf root.0 does not provide vjp_p, which the adjoint needs"
end

@testset "M2 result files validate against result.schema.json" begin
    validate_result_files(filter(f -> startswith(basename(f), "m2.") && endswith(f, ".json"), readdir(RESULTS_DIR; join = true)))
end
