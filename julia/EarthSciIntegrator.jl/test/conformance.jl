# The M1 conformance cases (spec/toy-problems.md §2, tests/conformance/m1/cases.json).
#
# Every case whose leaves all use implemented methods runs with all its checks.
# Cases that need an unimplemented method or construct are reported as skipped
# by name (`@test_skip`). The `cross_language` check is not run here: CI compares
# the result files and tapes written to RESULTS_DIR with Rust's
# (tests/conformance/m1/compare_results.py).

using EarthSciIntegrator: Run, LeafNode, LeafRecord, MapRecord, leaf_replay, leaf_step, leaf_step_adjoint,
    macro_adjoint, replay_macro, rerun_macro, functional_value, write_tape, read_tape, tape_string, stop_rule,
    next_stop, result_json, map_jvp!, map_vjp!

const REPO = normpath(joinpath(@__DIR__, "..", "..", ".."))
const CASES_FILE = joinpath(REPO, "tests", "conformance", "m1", "cases.json")
const REFERENCE_DIR = joinpath(REPO, "tests", "conformance", "m1", "reference")
const RESULTS_DIR = get(ENV, "EARTHSCIINTEGRATOR_RESULTS", joinpath(@__DIR__, "results"))

"""A check's outcome, printed in the summary: the measured quantity and the threshold."""
struct CheckOutcome
    case::String
    check::String
    passed::Bool
    detail::String
end
const OUTCOMES = CheckOutcome[]
const SKIPPED = Tuple{String, String}[]

function record!(case, check, passed, detail)
    push!(OUTCOMES, CheckOutcome(case, check, passed, detail))
    passed || @error "check $check of $case failed: $detail"
    @test passed
end

maxabs(x) = maximum(abs, x; init = 0.0)
relerr(a, b) = maxabs(a .- b) / max(maxabs(b), floatmin())

function build_case(case; modify = identity)
    cfg = modify(deepcopy(case["run_config"]))
    return build_run(cfg; tspan = case["tspan"], saveat = get(case, "saveat", nothing))
end

function load_reference(id)
    path = joinpath(REFERENCE_DIR, "$id.json")
    isfile(path) || return nothing
    return JSON.parsefile(path)
end

# ---------------------------------------------------------------------------
# Checks

function check_order(case, run, chk)
    id = case["id"]
    errs = Float64[]
    dts = Float64.(chk["dts"])
    target = if chk["against"] == "exact"
        run.prob.exact(run.tspan[2])
    else
        ref = load_reference(id)
        ref === nothing && return record!(id, "order", false, "no reference result for $id")
        Float64.(ref["u"][end])
    end
    for dt in dts
        modify = if chk["vary"] == "leaf"
            cfg -> (for n in leaf_specs(cfg["schedule"]); haskey(n, "dt") && n["dt"] isa Real && (n["dt"] = dt); end; cfg)
        else
            cfg -> (cfg["step"]["dt"] = dt; cfg)
        end
        r = forward!(build_case(case; modify))
        r.retcode == "Success" || return record!(id, "order", false, "run at dt = $dt ended with $(r.retcode)")
        push!(errs, maxabs(r.u .- target))
    end
    x, y = log.(dts), log.(errs)
    slope = sum((x .- mean(x)) .* (y .- mean(y))) / sum((x .- mean(x)) .^ 2)
    record!(id, "order", slope >= chk["expected"] - 0.3,
        "observed order $(round(slope; digits = 3)) (errors $(errs)), expected ≥ $(chk["expected"] - 0.3)")
end

mean(x) = sum(x) / length(x)

function leaf_specs(node, acc = Any[])
    if haskey(node, "leaf")
        push!(acc, node)
    end
    for k in ("sequence", "strang")
        haskey(node, k) && foreach(c -> leaf_specs(c, acc), node[k])
    end
    haskey(node, "subcycle") && leaf_specs(node["subcycle"], acc)
    return acc
end

function check_oracle(case, run, chk)
    id = case["id"]
    sol = oracle_solve(run, ode_alg(run.root.tab.name))
    err = relerr(run.u, sol.u[end])
    record!(id, "oracle", err <= chk["rtol"], "end-state relative difference $err, rtol $(chk["rtol"])")
end

function check_invariant(case, run, chk)
    id = case["id"]
    worst = 0.0
    for inv in run.prob.invariants
        I0 = inv(run.u0)
        for (t, u) in run.outputs
            worst = max(worst, relerr(inv(u), I0))
        end
    end
    record!(id, "invariant", worst <= chk["rtol"], "largest relative invariant drift $worst, rtol $(chk["rtol"])")
end

function check_exact_discrete(case, run, chk)
    id = case["id"]
    run.prob.name == "source_decay_map" || return record!(id, "exact_discrete", false, "no exact discrete solution for $(run.prob.name)")
    a, r = run.p
    u = run.u0[1]
    for k in 1:(length(run.bounds) - 1)
        Δ = run.bounds[k + 1] - run.bounds[k]
        u = (u + a * Δ) * exp(-r * Δ)
    end
    err = abs(run.u[1] - u) / abs(u)
    record!(id, "exact_discrete", err <= chk["rtol"], "relative difference from the exact discrete solution $err, rtol $(chk["rtol"])")
end

function check_reference(case, run, chk)
    id = case["id"]
    ref = load_reference(id)
    ref === nothing && return record!(id, "reference", false, "no reference result for $id")
    rt = Float64.(ref["t"])
    ru = [Float64.(x) for x in ref["u"]]
    scale = [maximum(abs(ru[i][j]) for i in eachindex(ru)) for j in eachindex(ru[1])]
    worst = 0.0
    for (t, u) in run.outputs
        i = findfirst(==(t), rt)
        i === nothing && return record!(id, "reference", false, "reference has no output at t = $t")
        worst = max(worst, maximum(abs(u[j] - ru[i][j]) / max(scale[j], floatmin()) for j in eachindex(u)))
    end
    record!(id, "reference", worst <= chk["rtol"], "largest scaled difference from the reference $worst, rtol $(chk["rtol"])")
end

"""Random direction for (u, p): normal entries scaled by each value's magnitude (1 where it is zero)."""
direction(rng, x) = [randn(rng) * (iszero(v) ? 1.0 : abs(v)) for v in x]

dual(x, dx) = [ForwardDiff.Dual{:tangent}(a, b) for (a, b) in zip(x, dx)]
partials(x) = [ForwardDiff.partials(v)[1] for v in x]

# spec/adjoint.md §7.1 (as amended in PR #9): the mismatch of the two sides
# divided by the larger of the summed magnitudes of their individual terms,
# max(Σ|λ_{n+1,i} δu_{n+1,i}|, Σ|λ_{n,i} δu_i| + Σ|p̄_k δp_k|).
function dot_mismatch(λ1, δu1, λn, δu, p̄, δp)
    lhs = dot(λ1, δu1)
    rhs = dot(λn, δu) + dot(p̄, δp)
    scale = max(sum(abs, λ1 .* δu1), sum(abs, λn .* δu) + sum(abs, p̄ .* δp), floatmin())
    return abs(lhs - rhs) / scale
end

function check_dot_product(case, run, chk)
    id = case["id"]
    rng = Random.MersenneTwister(hash(id))
    worst_step, worst_macro, nsteps = 0.0, 0.0, 0
    worst_info = ""
    P = length(run.p)
    for k in 0:(length(run.bounds) - 2)
        trace = Any[]
        replay_macro(run, k, copy(run.checkpoints[k + 1].u), run.p; trace)
        # Every step of every leaf invocation, and every map invocation.
        for entry in trace
            if entry[1] === :leaf
                _, leaf, rec, uin, pinv, stops = entry
                states = []
                leaf_replay(leaf, uin, pinv, rec, stops; states)
                for (u, t, h, pp, mask, tnew, tk1) in states
                    _, isstop = stop_rule(t, h, next_stop(stops, t))
                    skipped = isstop && h < eps(abs(t))
                    δu, δp, λ = direction(rng, u), direction(rng, pp), randn(rng, length(u))
                    λn, p̄ = leaf_step_adjoint(leaf, u, pp, t, h, mask, λ, skipped, tk1)
                    if leaf.ψ === nothing
                        u1 = skipped ? dual(u, δu) : leaf_step(leaf, dual(u, δu), dual(pp, δp), t, h, mask, tk1)
                        r = dot_mismatch(λ, partials(u1), λn, δu, p̄, δp)
                    else
                        # The integrating leaf's step also maps to Δq, with q̄ = 1.
                        u1, dq = skipped ? (dual(u, δu), 0.0) :
                                 leaf_step(leaf, dual(u, δu), dual(pp, δp), t, h, mask, tk1; withq = true)
                        δdq = dq isa ForwardDiff.Dual ? ForwardDiff.partials(dq)[1] : 0.0
                        r = dot_mismatch(vcat(λ, 1.0), vcat(partials(u1), δdq), λn, δu, p̄, δp)
                    end
                    if r > worst_step
                        worst_step = r
                        worst_info = "worst step at t = $t"
                    end
                    nsteps += 1
                end
            else
                _, node, rec, uin, pm, t = entry
                δu, δp, λ = direction(rng, uin), direction(rng, pm), randn(rng, length(uin))
                jv = zeros(length(uin))
                map_jvp!(node.op, jv, uin, pm, t, rec.dt, rec.mask, δu, δp)
                λb, p̄ = zeros(length(uin)), zeros(P)
                map_vjp!(node.op, λb, p̄, uin, pm, t, rec.dt, rec.mask, λ)
                worst_step = max(worst_step, dot_mismatch(λ, jv, λb, δu, p̄, δp))
                nsteps += 1
            end
        end
        # The whole macro step.
        u = run.checkpoints[k + 1].u
        δu, δp, λ = direction(rng, u), direction(rng, run.p), randn(rng, length(u))
        timefun = run.functional.over === :time
        qacc = timefun ? Ref{Any}(0.0) : nothing
        u1 = replay_macro(run, k, dual(u, δu), dual(run.p, δp); qacc)
        λn, p̄ = macro_adjoint(run, k, λ)
        p̄[run.bound_params] .= 0
        δp[run.bound_params] .= 0
        λ1, δu1 = λ, partials(u1)
        if timefun
            λ1, δu1 = vcat(λ, 1.0), vcat(δu1, qacc[] isa ForwardDiff.Dual ? ForwardDiff.partials(qacc[])[1] : 0.0)
        end
        worst_macro = max(worst_macro, dot_mismatch(λ1, δu1, λn, δu, p̄, δp))
    end
    # The whole run: the frozen-tape tangent of J against the gradient.
    δu, δp = direction(rng, run.u0), direction(rng, run.p)
    δp[run.bound_params] .= 0
    _, outs, qd = replay(run, dual(run.u0, δu), dual(run.p, δp))
    _, ū0, p̄ = gradient(run)
    # The left side's terms: the tangent of each point term's DOFs times its weight.
    F = run.functional
    if F.over === :time
        λ1, δu1 = [1.0], [ForwardDiff.partials(qd)[1]]
    else
        w = EarthSciIntegrator.phi_gradient(F, length(run.u0))
        finals = F.over === :final ? [last(outs)] : [o for o in outs if o[1] > run.tspan[1]]
        λ1 = reduce(vcat, [w for _ in finals])
        δu1 = reduce(vcat, [partials(u) for (_, u) in finals])
    end
    whole = dot_mismatch(λ1, δu1, ū0, δu, p̄, δp)
    worst = max(worst_step, worst_macro, whole)
    record!(id, "dot_product", worst <= chk["rtol"],
        "largest relative mismatch $worst over $nsteps steps (per step $worst_step, per macro step $worst_macro, whole run $whole; $worst_info), rtol $(chk["rtol"])")
end

function check_fd_gradient(case, run, chk)
    id = case["id"]
    rng = Random.MersenneTwister(hash(id) + 1)
    _, ū0, p̄ = gradient(run)
    # A state component that starts at zero is perturbed on the scale of its
    # largest magnitude over the run's checkpoints and outputs, not 1 (as the
    # Rust harness does): robertson's y₂ peaks near 4e-5, and a unit
    # perturbation at ε = 1e-3 leaves the stable region of the frozen steps.
    scale = [maximum(abs(x[j]) for x in vcat([c.u for c in run.checkpoints], [o[2] for o in run.outputs]))
             for j in eachindex(run.u0)]
    u0dir = [iszero(v) && !iszero(scale[j]) ? scale[j] : v for (j, v) in enumerate(run.u0)]
    δu = run.wrt_u0 ? direction(rng, u0dir) : zeros(length(run.u0))
    δp = zeros(length(run.p))
    δp[run.wrt_p] .= direction(rng, run.p[run.wrt_p])
    D = dot(ū0, δu) + dot(p̄, δp)
    J(ε) = functional_value(run, replay(run, run.u0 .+ ε .* δu, run.p .+ ε .* δp)...)
    errs = Float64[]
    for ε in chk["epsilons"]
        fd = (J(ε) - J(-ε)) / (2ε)
        push!(errs, abs(fd - D) / max(abs(D), floatmin()))
    end
    best = minimum(errs)
    record!(id, "fd_gradient", best <= chk["rtol"],
        "relative error of central differences at ε = $(chk["epsilons"]): $errs (directional derivative $D), rtol $(chk["rtol"])")
end

function check_reverse_diff(case, run, chk)
    id = case["id"]
    alg = ode_alg(run.root.tab.name)
    F = run.functional
    F.over in (:final, :time) || error("reverse_diff check implemented for final-time and time-integrated functionals")
    prob, kw = reverse_diff_problem(run)
    N = length(run.u0)
    w = F.over === :time ? vcat(zeros(N), 1.0) : EarthSciIntegrator.phi_gradient(F, N)
    # ReverseDiff tapes every scalar operation of OrdinaryDiffEq's dense LU,
    # about N³ per step: photostationary_cells' 49 states over 6354 steps need
    # more than 140 GiB. Above a budget the same OrdinaryDiffEq solve (same
    # method, same imposed steps, same map) is differentiated in forward mode
    # with ForwardDiff instead, and the check says so.
    nsteps = sum(length(r.steps) for r in run.tape.records if r isa LeafRecord)
    forward_mode = length(prob.u0)^3 * nsteps > 10^7
    if forward_mode
        fprob, _ = oracle_problem(run; inplace = false)
        if run.root.ψ !== nothing
            ψ, f0, J0 = run.root.ψ, fprob.f.f, fprob.f.jac
            faug(x, p, t) = (u = x[1:(end - 1)]; vcat(f0(u, p, t), ψ(u)))
            jaug = function (x, p, t)
                Ju = J0(x[1:N], p, t)
                J = zeros(eltype(Ju), N + 1, N + 1)
                J[1:N, 1:N] .= Ju
                J[N + 1, 1:N] .= ψ.w
                return J
            end
            fprob = remake(fprob; f = SciMLBase.ODEFunction{false}(faug; jac = jaug, tgrad = FtOf(run.root.ops, true)),
                u0 = vcat(fprob.u0, 0.0))
        end
        M = length(fprob.u0)
        lossx = x -> dot(w, solve(remake(fprob; u0 = x[1:M], p = x[(M + 1):end]), alg; kw...).u[end])
        gx = ForwardDiff.gradient(lossx, vcat(fprob.u0, run.p))
        gu, gp = gx[1:N], gx[(M + 1):end]
    else
        loss = function (u0, p)
            sol = solve(remake(prob; u0, p), alg; sensealg = ReverseDiffAdjoint(), kw...)
            return dot(w, sol.u[end])
        end
        gu, gp = Zygote.gradient(loss, copy(prob.u0), copy(run.p))
        gu = gu[1:N]
    end
    _, ū0, p̄ = gradient(run)
    eu = run.wrt_u0 ? relerr(ū0, gu) : 0.0
    ep = isempty(run.wrt_p) ? 0.0 : relerr(p̄[run.wrt_p], gp[run.wrt_p])
    err = max(eu, ep)
    how = forward_mode ? "ForwardDiff through the same OrdinaryDiffEq solve (too large for ReverseDiffAdjoint's tape)" :
          "ReverseDiffAdjoint"
    record!(id, "reverse_diff", err <= chk["rtol"],
        "relative difference from $how: u0 $eu, p $ep; rtol $(chk["rtol"])")
end

bitwise_equal(a, b) = length(a) == length(b) && all(i -> reinterpret(UInt64, Float64(a[i])) == reinterpret(UInt64, Float64(b[i])), eachindex(a))

function steps_equal(t1, t2)
    r1 = [r for r in t1.records if r isa LeafRecord]
    r2 = [r for r in t2.records if r isa LeafRecord]
    length(r1) == length(r2) || return false
    for (a, b) in zip(r1, r2)
        (a.path == b.path && a.accepted == b.accepted && a.rejected == b.rejected && length(a.steps) == length(b.steps)) || return false
        all(bitwise_equal(collect(x), collect(y)) for (x, y) in zip(a.steps, b.steps)) || return false
    end
    return true
end

function check_replay(case, run, chk)
    id = case["id"]
    problems = String[]
    # 1. Replay from the start with the tape: end state bitwise.
    u, _, q = replay(run)
    bitwise_equal(u, run.u) || push!(problems, "replayed end state differs (max |Δ| = $(maxabs(u .- run.u)))")
    q === nothing || bitwise_equal([q], [run.q]) || push!(problems, "replayed integrated functional differs ($q, $(run.q))")
    # 2. Every macro step replayed from its checkpoint lands bitwise on the next one,
    #    and every replayed step ends where the tape's next step starts.
    for k in 0:(length(run.bounds) - 2)
        trace = Any[]
        uk = replay_macro(run, k, copy(run.checkpoints[k + 1].u), run.p; trace)
        bitwise_equal(uk, run.checkpoints[k + 2].u) || push!(problems, "macro step $k: replay differs from checkpoint $(k + 1)")
        for entry in trace
            entry[1] === :leaf || continue
            _, leaf, rec, uin, pinv, stops = entry
            states = []
            leaf_replay(leaf, uin, pinv, rec, stops; states)
            ends = [s[6] for s in states]
            starts = [s[2] for s in states]
            bitwise_equal(ends[1:(end - 1)], starts[2:end]) || push!(problems, "$(rec.path): step end times do not chain")
            isempty(ends) || ends[end] == last(stops) || push!(problems, "$(rec.path): last step does not end at the interval end")
        end
        # 3. Re-running the macro step with the controllers on (restored from the
        #    checkpoint) reproduces the steps, the end state and the controller state.
        ur, tape_r, ctrls = rerun_macro(run, k)
        bitwise_equal(ur, run.checkpoints[k + 2].u) || push!(problems, "macro step $k: rerun with controller differs")
        sub = EarthSciIntegrator.Tape()
        for r in run.tape.records
            startswith(r.path, "$k/") && push!(sub, r)
        end
        steps_equal(sub, tape_r) || push!(problems, "macro step $k: rerun took different steps")
        all(ctrls[id] == run.checkpoints[k + 2].controllers[id] for id in keys(ctrls)) ||
            push!(problems, "macro step $k: controller state differs from checkpoint $(k + 1)")
    end
    # 4. The JSON Lines tape reads back bitwise and replays to the same end state.
    s = tape_string(run.tape)
    t2 = read_tape(s)
    tape_string(t2) == s || push!(problems, "tape does not round-trip through JSON Lines")
    u2, _ = replay(run; tape = t2)
    bitwise_equal(u2, run.u) || push!(problems, "replay from the serialized tape differs")
    record!(id, "replay", isempty(problems), isempty(problems) ? "bitwise ($(run.accepted) accepted steps)" : join(problems, "; "))
end

# ---------------------------------------------------------------------------

const CHECKS = Dict(
    "order" => check_order, "oracle" => check_oracle, "invariant" => check_invariant,
    "exact_discrete" => check_exact_discrete, "reference" => check_reference, "dot_product" => check_dot_product,
    "fd_gradient" => check_fd_gradient, "reverse_diff" => check_reverse_diff, "replay" => check_replay,
)

function run_case(case)
    id = case["id"]
    run = try
        build_case(case)
    catch e
        e isa UnsupportedError || rethrow()
        push!(SKIPPED, (id, e.msg))
        @testset "$id (skipped)" begin
            @test_skip "needs: $(e.msg)" == ""
        end
        return
    end
    @testset "$id" begin
        forward!(run)
        @test run.retcode == "Success"
        run.retcode == "Success" || return
        grad = run.functional === nothing ? nothing : gradient(run)
        mkpath(RESULTS_DIR)
        write(joinpath(RESULTS_DIR, "$id.json"), result_json(id, run; gradient = grad))
        open(io -> write_tape(io, run.tape), joinpath(RESULTS_DIR, "$id.tape.jsonl"), "w")
        for chk in case["checks"]
            name = chk["check"]
            name == "cross_language" && continue  # compared in CI, see the header
            @testset "$name" begin
                CHECKS[name](case, run, chk)
            end
        end
    end
end

@testset "M1 conformance cases" begin
    cases = JSON.parsefile(CASES_FILE)["cases"]
    @test length(cases) == 51
    @test JSON.parsefile(CASES_FILE)["esm_integrator"] == string(SPEC_VERSION)
    # EARTHSCIINTEGRATOR_CASES, a regular expression, runs only the matching
    # cases (for development; CI runs them all).
    only = get(ENV, "EARTHSCIINTEGRATOR_CASES", "")
    for case in cases
        isempty(only) || occursin(Regex(only), case["id"]) || continue
        run_case(case)
    end
    ran = unique(o.case for o in OUTCOMES)
    println("\nM1 conformance: $(length(ran)) cases run, $(length(SKIPPED)) skipped, ",
        count(o -> o.passed, OUTCOMES), "/", length(OUTCOMES), " checks passed (cross_language is compared in CI)")
    for o in OUTCOMES
        println(rpad(o.passed ? "  pass" : "  FAIL", 7), rpad(o.case, 42), rpad(o.check, 15), o.detail)
    end
    println("Skipped cases:")
    for (id, why) in SKIPPED
        println("  ", rpad(id, 46), why)
    end
end

@testset "result files validate against result.schema.json" begin
    files = filter(f -> endswith(f, ".json"), readdir(RESULTS_DIR; join = true))
    @test !isempty(files)
    validator = """
import json, sys
from jsonschema import Draft202012Validator
schema = json.load(open(sys.argv[1]))
bad = 0
for path in sys.argv[2:]:
    errs = list(Draft202012Validator(schema).iter_errors(json.load(open(path))))
    if errs:
        bad += 1
        print(path, errs[0].message)
sys.exit(1 if bad else 0)
"""
    python = Sys.which("python3")
    has_jsonschema = python !== nothing && success(`$python -c "import jsonschema"`)
    if has_jsonschema
        schema = joinpath(REPO, "spec", "schemas", "result.schema.json")
        @test success(pipeline(`$python -c $validator $schema $files`; stdout, stderr))
    else
        @test_skip "python3 with jsonschema not available" == ""
    end
end
