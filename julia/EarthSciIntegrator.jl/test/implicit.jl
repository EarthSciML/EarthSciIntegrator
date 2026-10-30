# The implicit families beyond what cases.json exercises: adjoints of every
# SDIRK method on nonlinear and block-diagonal problems, IMEX Euler with an
# explicit part and its linearized form, the θ-method's special cases,
# BackwardEuler's divided-difference estimate, Newton failures, singular W, a
# functional over time, and the refusals. Uses `xcase`, `run_xcase` and `XGRAD`
# from extra.jl.

using EarthSciIntegrator: SDIRKMethod, ThetaMethod, IMEXEulerMethod, newton_stage, Tolerances

const SDIRK = ["BackwardEuler", "TRBDF2", "SDIRK2", "Kvaerno3", "Kvaerno4"]

# u' = -u without `jac`: usable where no Jacobian of it is formed.
struct NoJacDecay <: EarthSciIntegrator.AbstractOperator end
EarthSciIntegrator.opname(::NoJacDecay) = "nojac"
EarthSciIntegrator.structure(::NoJacDecay) = EarthSciIntegrator.DenseStructure()
EarthSciIntegrator.depends_on_t(::NoJacDecay) = false
EarthSciIntegrator.rhs!(::NoJacDecay, out, u, p, t) = (out[1] += -u[1]; out)
EarthSciIntegrator.jvp_u!(::NoJacDecay, out, u, p, t, v) = (out[1] += -v[1]; out)
EarthSciIntegrator.vjp_u!(::NoJacDecay, out, u, p, t, w) = (out[1] += -w[1]; out)
EarthSciIntegrator.vjp_p!(::NoJacDecay, out, u, p, t, w) = out

@testset "implicit families beyond cases.json" begin
    implicit(alg; kw...) = merge(Dict{String, Any}("alg" => alg, "jacobian" => "symbolic"), Dict{String, Any}(string(k) => v for (k, v) in kw))

    @testset "SDIRK adjoints" begin
        for alg in SDIRK
            # Nonlinear and stiff-ish, fixed steps, point terms at every output time.
            run_xcase(
                xcase(
                    "x.van_der_pol.$alg.fixed", "toy:van_der_pol/nonstiff",
                    merge(Dict("leaf" => "oscillator"), implicit(alg; dt = 0.05)), Dict("dt" => 0.5), [0, 2.0];
                    adjoint = xgrad("y", "saveat"), saveat = [1.0, 2.0], checks = XGRAD
                )
            )
            # Block-diagonal and time-dependent, batched block LU.
            run_xcase(
                xcase(
                    "x.photostationary.$alg.fixed", "toy:photostationary_cells",
                    merge(Dict("leaf" => "photochemistry"), implicit(alg; dt = 20.0, linsolve = "block_lu")), Dict("dt" => 600),
                    [0, 1200.0]; adjoint = xgrad("O3", "final", "mean"), checks = XGRAD
                )
            )
            # Adaptive with interior stops; every adaptive method's estimate, including
            # BackwardEuler's divided difference. The adjoint differentiates the exact
            # stage equations at the stages the forward Newton iteration converged to,
            # so it matches finite differences of the tightly converged map only to
            # about the Newton tolerance relative to the stages (spec/adjoint.md
            # "exact up to the nonlinear-solve tolerance"): at reltol 1e-5 the
            # finite-difference check gets rtol 1e-5 (measured up to 3e-6 on a
            # direction whose derivative is small).
            run_xcase(
                xcase(
                    "x.van_der_pol.$alg.adaptive", "toy:van_der_pol/nonstiff",
                    merge(Dict("leaf" => "oscillator"), implicit(alg; control = Dict("reltol" => 1.0e-5, "abstol" => 1.0e-7))),
                    Dict("dt" => 6.3), [0, 6.3]; adjoint = xgrad("y", "saveat"), saveat = [1.0, 3.0, 6.3],
                    checks = [XGRAD[1], merge(XGRAD[2], Dict("rtol" => 1.0e-5)), XGRAD[3]]
                )
            )
        end
        # Splitting: SDIRK leaves inside strang, with a functional of B.
        reaction = merge(Dict("leaf" => "reaction"), implicit("TRBDF2"; control = Dict("reltol" => 1.0e-6, "abstol" => 1.0e-8)))
        transport = Dict("leaf" => "transport", "alg" => "SSPRK33", "dt" => 2.0e-3)
        run_xcase(
            xcase(
                "x.advection_reaction.strang.TRBDF2", "toy:advection_reaction",
                Dict("strang" => [transport, reaction]), Dict("dt" => 0.025), [0, 0.1];
                adjoint = xgrad("B"), checks = vcat(XGRAD, [Dict{String, Any}("check" => "invariant", "rtol" => 1.0e-12)])
            )
        )
    end

    @testset "oracle at fixed steps on an autonomous problem" begin
        # Kvaerno3/4 match OrdinaryDiffEq's on autonomous problems only (D55).
        for alg in SDIRK
            run_xcase(
                xcase(
                    "x.van_der_pol.$alg.oracle", "toy:van_der_pol/nonstiff",
                    merge(Dict("leaf" => "oscillator"), implicit(alg; dt = 0.01)), Dict("dt" => 2.0), [0, 2.0];
                    checks = [Dict{String, Any}("check" => "oracle", "rtol" => 1.0e-6)]
                )
            )
        end
    end

    @testset "IMEX Euler" begin
        # Transport explicit, reaction implicit: order 1 against the unsplit reference,
        # and the adjoint with an explicit part.
        ar = Dict(
            "leaf" => ["transport", "reaction"], "alg" => "IMEXEuler", "jacobian" => "symbolic", "dt" => 2.0e-3,
            "imex" => Dict("implicit" => "reaction")
        )
        run_xcase(
            xcase(
                "x.advection_reaction.IMEXEuler", "toy:advection_reaction", ar, Dict("dt" => 0.05), [0, 0.1];
                adjoint = xgrad("B"), checks = vcat(XGRAD, [Dict{String, Any}("check" => "invariant", "rtol" => 1.0e-12)])
            )
        )
        run = forward!(build_run(xcase("", "toy:advection_reaction", ar, Dict("dt" => 0.05), [0, 0.1])["run_config"]; tspan = (0, 0.1)))
        @test run.root.imex[1].ops == (run.prob.operators["transport"],)
        @test run.root.imex[2].ops == (run.prob.operators["reaction"],)
        # "linearized" runs as asked, also with a linear implicit part: on forced_decay,
        # whose implicit part depends on t, it evaluates f_I at (uₙ, tₙ) instead of
        # (u_{n+1}, tₙ + h), so the two differ; reaction (linear, autonomous) they agree to roundoff.
        lin = merge(ar, Dict("implicit_solve" => "linearized"))
        rl = forward!(build_run(xcase("", "toy:advection_reaction", lin, Dict("dt" => 0.05), [0, 0.1])["run_config"]; tspan = (0, 0.1)))
        rc = forward!(build_run(xcase("", "toy:advection_reaction", ar, Dict("dt" => 0.05), [0, 0.1])["run_config"]; tspan = (0, 0.1)))
        @test rl.root.method.linearized && isapprox(rl.u, rc.u; rtol = 1.0e-12)
        fd(solve) = Dict{String, Any}(
            "esm_integrator" => "0.1.0", "model" => "toy:forced_decay",
            "schedule" => Dict(
                "leaf" => "decay", "alg" => "IMEXEuler", "dt" => 0.1, "imex" => Dict("implicit" => "decay"),
                "implicit_solve" => solve
            ), "step" => Dict("dt" => 2)
        )
        ul, uc = forward!(build_run(fd("linearized"); tspan = (0, 2))).u, forward!(build_run(fd("converged"); tspan = (0, 2))).u
        @test abs(ul[1] - uc[1]) > 1.0e-3
        run_xcase(
            xcase(
                "x.forced_decay.IMEXEuler.linearized", "toy:forced_decay", fd("linearized")["schedule"], Dict("dt" => 2),
                [0, 2.0]; adjoint = xgrad("u"), checks = XGRAD
            )
        )
        # A nonlinear implicit part: the linearized step (a one-stage Rosenbrock
        # method, D43) and its adjoint, which includes hess_u.
        for solve in ("converged", "linearized")
            vdp = Dict(
                "leaf" => "oscillator", "alg" => "IMEXEuler", "jacobian" => "symbolic", "dt" => 0.01,
                "imex" => Dict("implicit" => "oscillator"), "implicit_solve" => solve
            )
            c = xcase(
                "x.van_der_pol.IMEXEuler.$solve", "toy:van_der_pol/nonstiff", vdp, Dict("dt" => 0.5), [0, 1.0];
                adjoint = xgrad("y", "saveat"), saveat = [0.5, 1.0], checks = XGRAD
            )
            run_xcase(c)
            r = forward!(build_case(c))
            @test r.root.method.linearized == (solve == "linearized")
        end
        # Partitions: every term in exactly one part; explicit omitted means "the rest".
        bad(imex) = try
            build_run(
                xcase("", "toy:advection_reaction", merge(ar, Dict("imex" => imex)), Dict("dt" => 0.05), [0, 0.1])["run_config"];
                tspan = (0, 0.1)
            ); nothing
        catch e
            e
        end
        @test bad(Dict("implicit" => "reaction", "explicit" => ["transport", "reaction"])) isa ConfigError
        @test bad(Dict("implicit" => "reaction", "explicit" => String[])) isa ConfigError
        @test bad(Dict("implicit" => "chemistry")) isa ConfigError
        @test occursin("locality names are not supported", bad(Dict("implicit" => "pointwise")).msg)
        @test bad(Dict("implicit" => String[])) isa ConfigError          # no implicit part: refused, not forward Euler
        @test bad(Dict("implicit" => 3)) isa ConfigError
        @test bad(["reaction"]) isa ConfigError
        @test bad(Dict("explicit" => "transport")) isa ConfigError
        @test bad(Dict("implicit" => ["transport", "reaction"])) === nothing
    end

    @testset "θ-method" begin
        dec(alg; kw...) = Dict{String, Any}(
            "esm_integrator" => "0.1.0", "model" => "toy:forced_decay",
            "schedule" => merge(Dict{String, Any}("leaf" => "decay"), implicit(alg; kw...)), "step" => Dict("dt" => 2)
        )
        # θ = 1 is backward Euler, bitwise; θ = 0 forward Euler, bitwise.
        be = forward!(build_run(dec("BackwardEuler"; dt = 0.1); tspan = (0, 2)))
        t1 = forward!(build_run(dec("Theta"; dt = 0.1, theta = 1.0); tspan = (0, 2)))
        @test t1.u == be.u
        fe = forward!(
            build_run(
                Dict{String, Any}(
                    "esm_integrator" => "0.1.0", "model" => "toy:forced_decay",
                    "schedule" => Dict("leaf" => "decay", "alg" => "ForwardEuler", "dt" => 0.1), "step" => Dict("dt" => 2)
                ); tspan = (0, 2)
            )
        )
        t0 = forward!(build_run(dec("Theta"; dt = 0.1, theta = 0.0); tspan = (0, 2)))
        @test t0.u == fe.u
        # First order away from θ = 1/2.
        errs = [
            abs(forward!(build_run(dec("Theta"; dt = h, theta = 0.7); tspan = (0, 2))).u[1] - t1.prob.exact(2.0)[1])
                for h in (0.1, 0.05, 0.025)
        ]
        @test 0.9 < log2(errs[1] / errs[2]) < 1.1 && 0.9 < log2(errs[2] / errs[3]) < 1.1
        # Adjoints at θ = 0, 0.3 and 1 on a nonlinear problem.
        for θ in (0.0, 0.3, 1.0)
            run_xcase(
                xcase(
                    "x.van_der_pol.Theta.$θ", "toy:van_der_pol/nonstiff",
                    merge(Dict("leaf" => "oscillator"), implicit("Theta"; dt = 0.02, theta = θ)), Dict("dt" => 0.5), [0, 1.0];
                    adjoint = xgrad("z"), checks = XGRAD
                )
            )
        end
        @test_throws ConfigError build_run(dec("Theta"; dt = 0.1, theta = 1.5); tspan = (0, 2))
        @test_throws ConfigError build_run(dec("Theta"; dt = 0.1, theta = "half"); tspan = (0, 2))
        @test_throws ConfigError build_run(dec("Theta"; dt = 0.1, theta = true); tspan = (0, 2))
        @test_throws ConfigError build_run(dec("TRBDF2"; dt = 0.1, implicit_solve = 1); tspan = (0, 2))
        @test_throws ConfigError build_run(dec("TRBDF2"; dt = 0.1, implicit_solve = "exact"); tspan = (0, 2))
        @test_throws ConfigError build_run(dec("Theta"; dt = 0.1); tspan = (0, 2))
        @test_throws ConfigError build_run(dec("TRBDF2"; dt = 0.1, theta = 0.5); tspan = (0, 2))
        @test_throws UnsupportedError build_run(dec("TRBDF2"; dt = 0.1, implicit_solve = "linearized"); tspan = (0, 2))
        @test build_run(dec("TRBDF2"; dt = 0.1, implicit_solve = "converged"); tspan = (0, 2)) isa EarthSciIntegrator.Run
    end

    @testset "functional over time" begin
        # ∫u dt on forced_decay: q against the exact integral, the gradient against
        # ForwardDiff through the frozen-tape replay (whose tangents are the exact
        # stage linearization, as the adjoint's).
        prob = toy_problem("toy:forced_decay")
        k, ω = prob.p
        A(s) = (k * sin(ω * s) - ω * cos(ω * s)) / (k^2 + ω^2)
        Aint(s) = (-k * cos(ω * s) / ω - sin(ω * s)) / (k^2 + ω^2)
        exact = (prob.u0[1] - A(0.0)) * (1 - exp(-2k)) / k + Aint(2.0) - Aint(0.0)
        for (alg, extra, rtol) in (
                ("Kvaerno4", (;), 1.0e-7), ("TRBDF2", (;), 1.0e-4), ("Theta", (; theta = 0.5), 1.0e-4),
                ("Theta", (; theta = 0.0), 1.0e-1), ("BackwardEuler", (;), 1.0e-1),
            )
            cfg = Dict{String, Any}(
                "esm_integrator" => "0.1.0", "model" => "toy:forced_decay",
                "schedule" => merge(Dict{String, Any}("leaf" => "decay"), implicit(alg; dt = 0.01, extra...)), "step" => Dict("dt" => 2),
                "adjoint" => Dict("functional" => Dict("sum" => "u", "over" => "time"), "wrt" => ["u0", "params:numeric"])
            )
            run = forward!(build_run(cfg; tspan = (0, 2)))
            @test isapprox(run.q, exact; rtol)
            J, ū0, p̄ = gradient(run)
            @test replay(run)[3] == run.q
            g = ForwardDiff.gradient(x -> replay(run, x[1:1], x[2:3])[3], vcat(run.u0, run.p))
            @test isapprox(vcat(ū0, p̄), g; rtol = 1.0e-12)
        end
        # And on a nonlinear problem with the dot-product and finite-difference checks.
        run_xcase(
            xcase(
                "x.van_der_pol.Kvaerno3.time", "toy:van_der_pol/nonstiff",
                merge(Dict("leaf" => "oscillator"), implicit("Kvaerno3"; dt = 0.05)), Dict("dt" => 1.0), [0, 1.0];
                adjoint = xgrad("y", "time"), checks = XGRAD
            )
        )
        imex = Dict(
            "esm_integrator" => "0.1.0", "model" => "toy:forced_decay",
            "schedule" => Dict("leaf" => "decay", "alg" => "IMEXEuler", "dt" => 0.1, "imex" => Dict("implicit" => "decay")),
            "step" => Dict("dt" => 2), "adjoint" => Dict("functional" => Dict("sum" => "u", "over" => "time"), "wrt" => ["u0"])
        )
        @test_throws UnsupportedError build_run(imex; tspan = (0, 2))
    end

    @testset "jac only from operators whose Jacobian is formed" begin
        sdm = toy_problem("toy:source_decay_map")
        ops = Dict{String, EarthSciIntegrator.AbstractOperator}("source" => sdm.operators["source"], "nojac" => NoJacDecay())
        prob = EarthSciIntegrator.ToyProblem(
            "nojac", 1, [1.0], sdm.p, sdm.pnames, (0.0, 1.0), ops, sdm.maps, ["u" => [1]],
            Function[], nothing, false
        )
        cfg(sched) = Dict{String, Any}(
            "esm_integrator" => "0.1.0", "model" => "toy:source_decay_map",
            "schedule" => merge(Dict{String, Any}("leaf" => ["source", "nojac"], "dt" => 0.1), sched), "step" => Dict("dt" => 1)
        )
        build(sched) = build_run(cfg(sched); tspan = (0, 1), prob)
        # IMEX Euler forms J of its implicit part only; θ = 0 forms none.
        r = forward!(build(Dict("alg" => "IMEXEuler", "imex" => Dict("implicit" => "source"))))
        @test r.retcode == "Success"
        @test forward!(build(Dict("alg" => "Theta", "theta" => 0.0))).retcode == "Success"
        @test_throws ConfigError build(Dict("alg" => "IMEXEuler", "imex" => Dict("implicit" => "nojac")))
        @test_throws ConfigError build(Dict("alg" => "Theta", "theta" => 0.5))
        @test_throws ConfigError build(Dict("alg" => "BackwardEuler"))
    end

    @testset "Newton iteration" begin
        # A linear stage converges in two iterations: the first is exact, the second's
        # increment is roundoff. The first converges alone only with δ₁ < 1e-5.
        tol = Tolerances(1.0e-3, [1.0e-6])
        F = EarthSciIntegrator.factor_W(EarthSciIntegrator.DenseJacobian(fill(-2.0, 1, 1)), inv(0.1), EarthSciIntegrator.LinsolveSpec(:dense_lu, true))
        fI = (x, τ) -> [-2.0 * x[1]]
        θ = Ref(1.0)
        z, ok, nf = newton_stage(fI, F, inv(0.1), [1.0], 1.0, 0.1, 0.1, [0.0], [1.0], tol, θ, true)
        @test ok && nf == 2 && isapprox(z[1], -0.2 / 1.2; rtol = 1.0e-15)
        @test θ[] == 0.3                 # max(0.3 θ_prev, δ₂/δ₁) with θ_prev = 1
        # The rate carries across stages: with a small θ_prev an adaptive solve stops after one iteration…
        θ = Ref(1.0e-6)
        _, ok, nf = newton_stage(fI, F, inv(0.1), [1.0], 1.0, 0.1, 0.1, [0.0], [1.0], tol, θ, true)
        @test ok && nf == 1
        # …a fixed-step one does not (OrdinaryDiffEq does not trust θ on its first iteration).
        θ = Ref(1.0e-6)
        _, ok, nf = newton_stage(fI, F, inv(0.1), [1.0], 1.0, 0.1, 0.1, [0.0], [1.0], tol, θ, false)
        @test ok && nf == 2
        # Divergence: W far from the true Jacobian.
        Fbad = EarthSciIntegrator.factor_W(EarthSciIntegrator.DenseJacobian(fill(0.0, 1, 1)), inv(0.1), EarthSciIntegrator.LinsolveSpec(:dense_lu, true))
        fstiff = (x, τ) -> [-1000.0 * x[1]]
        _, ok, _ = newton_stage(fstiff, Fbad, inv(0.1), [1.0], 1.0, 0.1, 0.1, [0.0], [1.0], tol, Ref(1.0), true)
        @test !ok
    end

    @testset "Newton failure and singular W" begin
        # Robertson at a fixed step far beyond the Newton iteration's reach: the
        # leaf ends with ConvergenceFailure; adaptively the step is halved instead.
        rob(sched) = Dict{String, Any}(
            "esm_integrator" => "0.1.0", "model" => "toy:robertson",
            "schedule" => merge(Dict{String, Any}("leaf" => "kinetics"), sched), "step" => Dict("dt" => 40)
        )
        run = forward!(build_run(rob(implicit("SDIRK2"; dt = 40.0)); tspan = (0, 40)))
        @test run.retcode == "ConvergenceFailure"
        run = forward!(
            build_run(
                rob(implicit("SDIRK2"; control = Dict("dt0" => 40.0, "reltol" => 1.0e-6, "abstol" => 1.0e-10)));
                tspan = (0, 40)
            )
        )
        @test run.retcode == "Success" && run.rejected >= 1
        # The first attempt (h = 40) fails in Newton, not in the error test, and the
        # Newton-failure branch halves the step (h/failfactor = 20): an error-test
        # rejection would give 40/min(5, E^β₁/γ) and a non-finite one 40·qmin = 8.
        adaptive_rob = build_run(rob(implicit("SDIRK2"; control = Dict("dt0" => 40.0, "reltol" => 1.0e-6, "abstol" => 1.0e-10))); tspan = (0, 40))
        @test EarthSciIntegrator.method_step(adaptive_rob.root, nothing, adaptive_rob.p, adaptive_rob.u0, 0.0, 40.0, nothing, true).newton_fail
        one_attempt = build_run(
            rob(implicit("SDIRK2"; control = Dict("dt0" => 40.0, "reltol" => 1.0e-6, "abstol" => 1.0e-10, "maxiters" => 1)));
            tspan = (0, 40)
        )
        forward!(one_attempt)
        @test one_attempt.retcode == "MaxIters" && one_attempt.rejected == 1
        @test one_attempt.controllers["root"].h == 20.0
        # forced_decay with k = 1/(hγ) makes W = I - hγJ = 1 + hγk... exactly zero for k = -1/(hγ).
        h = 0.1
        p = [-1 / h, 3.0]
        dec = Dict{String, Any}(
            "esm_integrator" => "0.1.0", "model" => "toy:forced_decay",
            "schedule" => merge(Dict{String, Any}("leaf" => "decay"), implicit("BackwardEuler"; dt = h)), "step" => Dict("dt" => 2)
        )
        @test forward!(build_run(dec; tspan = (0, 2), p)).retcode == "Unstable"
    end
end
