# The runtime of algebraic equations on the toy problems with exact answers
# (spec/toy-problems.md §1.7–§1.11; spec/methods.md §14, §15).

using EarthSciIntegrator
using Test
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
        pj = AR.JSON.parsefile(f)
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
end
