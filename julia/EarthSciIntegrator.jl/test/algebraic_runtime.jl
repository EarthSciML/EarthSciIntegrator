# The runtime of algebraic equations on the toy problems with exact answers
# (spec/toy-problems.md §1.7–§1.11; spec/methods.md §14, §15).

using EarthSciIntegrator
using Test
using Random: MersenneTwister
using LinearAlgebra: dot
using JSON
@isdefined(REPO) || include("paths.jl")

const AR = EarthSciIntegrator

"""Plan the fixture case, run it forward, and return the run and its problem."""
function _run_case(fixture, case)
    out = mktempdir()
    AR.write_plan(AR.plan(joinpath(PFIX, fixture, "$case.run.json")), out)
    run, pp = AR.build_plan_run(out)
    AR.forward!(run)
    return run, pp
end

_end(run, pp, v) = run.outputs[end][2][AR.layout_dofs(pp, [v])]
_solve_record(run) = only(r for r in run.tape.records if r isa AR.SolveRecord)


"""Plan the fixture case and run it forward with its run config's `adjoint`."""
function _adjoint_run(fixture, case)
    out = mktempdir()
    rcpath = joinpath(PFIX, fixture, "$case.run.json")
    AR.write_plan(AR.plan(rcpath), out)
    pp = AR.plan_problem(out)
    cfg = Dict{String, Any}(
        "esm_integrator" => pp.plan["esm_integrator"], "schedule" => pp.plan["schedule"], "step" => pp.plan["step"],
        "adjoint" => JSON.parsefile(rcpath)["adjoint"]
    )
    run = AR.forward!(AR.build_run(cfg; tspan = pp.prob.tspan, prob = pp.prob))
    return run, pp
end

"""
The relative error of central differences of the frozen-tape functional
(spec/adjoint.md §7.2) against the gradient's directional derivative, the
smallest over `eps`. A state that starts at zero is perturbed on the scale
of its largest magnitude over the run's checkpoints and outputs (as the
conformance check does: robertson's y₂ perturbed by 1e-3 diverges), a
parameter that is zero on the scale 1.
"""
function _fd_error(run; eps = (1.0e-3, 1.0e-4, 1.0e-5), seed = 1)
    rng = MersenneTwister(seed)
    _, ū0, p̄ = AR.gradient(run)
    mag(v) = [x == 0 ? 1.0 : abs(x) for x in v]
    states = vcat([c.u for c in run.checkpoints], [o[2] for o in run.outputs])
    umag = [x != 0 ? abs(x) : max(maximum(abs(s[j]) for s in states), floatmin()) for (j, x) in enumerate(run.u0)]
    δu = run.wrt_u0 ? randn(rng, length(run.u0)) .* umag : zeros(length(run.u0))
    δp = zeros(length(run.p))
    δp[run.wrt_p] = randn(rng, length(run.wrt_p)) .* mag(run.p[run.wrt_p])
    D = dot(ū0, δu) + dot(p̄[run.wrt_p], δp[run.wrt_p])
    J(ε) = AR.with_tight_newton() do
        AR.functional_value(run, AR.replay(run, run.u0 .+ ε .* δu, run.p .+ ε .* δp)...)
    end
    errs = [abs((J(ε) - J(-ε)) / (2ε) - D) / max(abs(D), 1.0e-14) for ε in eps]
    return minimum(errs), D
end

"""
The per-invocation dot-product identity of a solve (spec/adjoint.md §7.1,
§4.6): ⟨λ, dz⟩ against ⟨λₙ, δu⟩ + ⟨p̄, δp⟩ for the implicit-function tangent
and the adjoint at the same converged `z`, relative to the terms' magnitude.
"""
function _node_dot(ops, info, u, z, p, t, held, up, wrt_p; seed = 2)
    rng = MersenneTwister(seed)
    N, P = length(u), length(p)
    δu = randn(rng, N)
    δp = zeros(P)
    δp[wrt_p] = randn(rng, length(wrt_p))
    dz = AR.solve_tangent(ops, info, u, z, p, t, held, up, δu, δp)
    λ = zeros(N)
    λ[info.dofs] = randn(rng, length(info.dofs))
    p̄ = zeros(P)
    λn = AR.solve_adjoint(ops, info, u, z, p, t, held, up, λ, p̄)
    lhs = dot(λ[info.dofs], dz)
    rhs = dot(λn, δu) + dot(p̄, δp)
    scale = max(sum(abs, λ[info.dofs] .* dz), sum(abs, λn .* δu) + sum(abs, p̄ .* δp))
    return abs(lhs - rhs) / max(scale, floatmin()), p̄
end

@testset "algebraic runtime" begin
    @testset "cubic_roots $case" for case in ("newton", "newton_inexact", "ptc")
        run, pp = _run_case("toy_cubic_roots", case)
        @test run.outputs[end][1] == 1.0
        @test _end(run, pp, "Cubic.z") ≈ [0.6823278038280193, 1.0, 1.2134116627622296, 1.378796700129551] atol = 1.0e-11
        @test isempty(_solve_record(run).active)
        # block_diagonal: one 1 × 1 block per cell.
        @test pp.prob.algebraic["cubic"].blocks == [[1], [2], [3], [4]]
    end
    @testset "box_lcp $case" for case in ("newton", "newton_inexact", "ptc")
        run, pp = _run_case("toy_box_lcp", case)
        # Both bounds active: z₁ at its referenced upper bound, z₂ at 0 (§1.8).
        @test _end(run, pp, "Box.z") == [0.25, 0.0]
        @test _solve_record(run).active == Dict("Box.z" => ([1], [0], Int[]))
        # The tape's JSON Lines form reads back the same record (spec/tape.md §5).
        r2 = AR.read_tape(AR.tape_string(run.tape))
        @test only(r for r in r2.records if r isa AR.SolveRecord).active == _solve_record(run).active
    end
    @testset "phase_joint $case" for case in ("ptc", "per_cell")
        run, pp = _run_case("toy_phase_joint", case)
        # s column-major over cell × salt; cell 1 saturates salt 1 (§1.9).
        @test _end(run, pp, "Phase.s") ≈ [1.0, 0.0, 0.0, 0.0] atol = 1.0e-11
        @test _end(run, pp, "Phase.w") ≈ [1.0, 0.75] atol = 1.0e-11
        # per_cell's block_lu factors one 3 × 3 block per cell: s[c, 1],
        # s[c, 2], w[c] (positions among the unknowns s then w, column-major).
        case == "per_cell" && @test pp.prob.algebraic["phase"].blocks == [[1, 3, 5], [2, 4, 6]]
    end
    @testset "picard_roots $case" for (case, tol) in (("picard", 1.0e-12), ("rel_change", 1.0e-11), ("newton_rel_change", 1.0e-11))
        run, pp = _run_case("toy_picard_roots", case)
        # The roots of z + z³ = c, c = i/4 (§1.11).
        @test _end(run, pp, "Picard.z") ≈ [0.2367329038645632, 0.4238537990697833, 0.5673642266809228, 0.6823278038280195] atol = tol
        # Picard with ν frozen is the fixed point z ← c/ν(z): linear
        # convergence, at rate 0.64 in the slowest cell; Newton is quadratic.
        it = _solve_record(run).iterations
        @test case == "newton_rel_change" ? it <= 8 : 30 <= it <= 100
    end
    @testset "box_scaled $case" for case in ("scaled_ptc", "recover_method")
        run, pp = _run_case("box_scaled", case)
        # z[1] at the state-free observed upper bound 1.5 zmax[1] = 0.375,
        # z[2] at 0 (D125).
        @test _end(run, pp, "Box.z") == [0.375, 0.0]
        rec = _solve_record(run)
        @test rec.active == Dict("Box.z" => ([1], [0], Int[]))
        @test rec.recover == (case == "recover_method" ? ["method"] : String[])
    end
    @testset "a scale that is not positive fails the invocation" begin
        out = mktempdir()
        AR.write_plan(AR.plan(joinpath(PFIX, "box_scaled", "recover_method.run.json")), out)
        f = joinpath(out, "plan.json")
        pj = JSON.parsefile(f)
        pj["schedule"]["method"]["scale"]["Box.z"] = -1.0
        write(f, AR.JSON.json(pj))
        run, pp = AR.build_plan_run(out)
        # forward! records a run failure rather than throwing it.
        AR.forward!(run)
        @test run.retcode == "SolveFailure"
        @test occursin("not positive", run.failure)
    end
    @testset "robertson_dae $case" for (case, rtol) in (("rodas4", 1.0e-8), ("kvaerno4", 1.0e-6), ("trbdf2", 1.0e-4), ("inconsistent", 1.0e-8))
        ode, opp = _run_case("toy_robertson", "rodas4")
        run, pp = _run_case("toy_robertson_dae", case)
        @test run.outputs[end][1] == 40.0
        y = [_end(run, pp, "Rober.y1")[1], _end(run, pp, "Rober.y2")[1], _end(run, pp, "Rober.y3")[1]]
        @test isapprox(y[1], _end(ode, opp, "Kinetics.y1")[1]; rtol)
        @test isapprox(y[2], _end(ode, opp, "Kinetics.y2")[1]; rtol)
        # The algebraic equation holds exactly at the step ends (stiffly accurate methods).
        @test abs(1 - sum(y)) <= 1.0e-14
    end
    # The adjoint of a solve (spec/adjoint.md §4.6, D127): the parameter
    # products, a parameter and an observed bound, the per-invocation
    # dot-product identity, finite differences of the frozen-tape map
    # (the active set held), and the exact replay.
    @testset "solve adjoint $fixture/$case" for (fixture, case) in (("adjoint_cubic", "newton"), ("adjoint_box", "newton"), ("toy_phase_joint", "adjoint"))
        # Julia's EarthSciAST does not build box.frozen_vjp.esm: its program
        # build refuses EarthSciASTDiff's hoisted observed `_dp_zcap_1`, an
        # array-valued observed of parameters declared without a shape
        # (E_TREEWALK_GEOMETRY_SETUP), though esm-spec §4.3.4 gives it its
        # operands' index sets; Rust's builds it and passes these checks.
        # Broken until EarthSciAST builds it; then this fails as a reminder.
        if fixture == "adjoint_box"
            built = try
                _adjoint_run(fixture, case)
                true
            catch e
                occursin("E_TREEWALK_GEOMETRY_SETUP", sprint(showerror, e)) || rethrow()
                false
            end
            @test_broken built
            built || continue
        end
        run, pp = _adjoint_run(fixture, case)
        @test run.retcode == "Success"
        rec = _solve_record(run)
        node = run.root
        @test node isa AR.SolveNode
        held, up = AR.held_set(node.info, rec.active)
        u = run.checkpoints[1].u
        mis, p̄ = _node_dot(node.ops, node.info, u, rec.z, run.p, rec.t, held, up, run.wrt_p)
        @test mis <= 1.0e-12
        err, D = _fd_error(run)
        @test err <= 1.0e-6
        @test D != 0
        # The tape replays exactly (spec/adjoint.md §7.4).
        ur, _, _ = AR.replay(run)
        @test ur == run.u
        if fixture == "adjoint_box"
            # Both bounds active: dz₁/dcap = zmax₁ through the frozen residual's
            # product, dz₂/dzlo = 1 through the parameter bound, and b does not
            # move an unknown held at a bound.
            _, _, g = AR.gradient(run)
            pn = pp.prob.pnames
            @test g[findfirst(==("Box.cap"), pn)] ≈ 0.25 atol = 1.0e-14
            @test g[findfirst(==("Box.zlo"), pn)] ≈ 1.0 atol = 1.0e-14
            @test g[findfirst(==("Box.beta"), pn)] == 0.0
        elseif fixture == "adjoint_cubic"
            # Σ dz/da = Σ cᵢ / (1 + 3zᵢ²).
            _, _, g = AR.gradient(run)
            z = _end(run, pp, "Cubic.z")
            @test only(g) ≈ sum([1.0, 2.0, 3.0, 4.0] ./ (1 .+ 3 .* z .^ 2)) rtol = 1.0e-12
        end
    end
    # The adjoint of an index-1 DAE (spec/adjoint.md §5.10, D127) with respect
    # to the initial state: ESDIRK steps with σM − J and the consistent
    # initialization as a §4.6 solve; finite differences of the frozen-tape
    # map, the initialization's per-invocation identity, the exact replay.
    @testset "dae adjoint $case" for (case, rtol) in (("kvaerno4_adjoint", 1.0e-5), ("trbdf2_adjoint", 1.0e-5))
        run, pp = _adjoint_run("toy_robertson_dae", case)
        @test run.retcode == "Success"
        err, D = _fd_error(run)
        @test err <= rtol
        @test D != 0
        _, ū0, _ = AR.gradient(run)
        # y₃ is algebraic: its initial value only starts the initialization.
        @test ū0[only(AR.layout_dofs(pp, ["Rober.y3"]))] == 0.0
        d = run.root.dae
        u = run.checkpoints[1].u
        o = AR.run_solve(d.ops, d.info, u, run.p, 0.0, d.init, u[d.info.dofs], u[d.info.dofs], pp.prob.pnames)
        nz = length(o.z)
        mis, _ = _node_dot(d.ops, d.info, u, o.z, run.p, 0.0, falses(nz), falses(nz), Int[])
        @test mis <= 1.0e-12
        ur, _, _ = AR.replay(run)
        @test ur == run.u
    end
end
