# Rosenbrock family and linear solvers beyond what cases.json exercises: the
# LU arithmetic against LinearAlgebra, block_lu against dense_lu, the KPP
# methods' orders and adjoints, singular W, and the refusals.

using EarthSciIntegrator: LinsolveSpec, DenseJacobian, BlockJacobian, form_W, lu_factor, lu_solve, lu_solve_transpose,
    issuccess_lu, todense, RosenbrockMethod
import LinearAlgebra
using LinearAlgebra: I

@testset "Rosenbrock family" begin
    rng = Random.MersenneTwister(4242)

    @testset "dense LU against LinearAlgebra" begin
        N = 7
        A = randn(rng, N, N)
        α = 3.5
        for pivot in (true, false)
            W = form_W(DenseJacobian(copy(A)), α)
            @test W == α * I - A
            F = lu_factor(copy(W), pivot)
            @test issuccess_lu(F)
            b = randn(rng, N)
            @test lu_solve(F, b) ≈ W \ b rtol = 1.0e-12
            @test lu_solve_transpose(F, b) ≈ W' \ b rtol = 1.0e-12
        end
        # Julia's generic_lufact! with RowMaximum gives the same factors bitwise
        # (LAPACK's blocked getrf, which `lu` calls for Float64, does not).
        W = form_W(DenseJacobian(copy(A)), α)
        F = lu_factor(copy(W), true)
        G = LinearAlgebra.generic_lufact!(copy(W), LinearAlgebra.RowMaximum())
        @test F.F == G.factors
        @test F.perm == G.ipiv
        # A zero pivot makes W singular.
        @test !issuccess_lu(lu_factor(zeros(3, 3), true))
        @test !issuccess_lu(lu_factor([0.0 1.0; 1.0 0.0], false))
        @test issuccess_lu(lu_factor([0.0 1.0; 1.0 0.0], true))
        # Dual numbers pass through.
        Wd = ForwardDiff.Dual.(W, 1.0)
        Fd = lu_factor(Wd, true)
        @test ForwardDiff.value.(lu_solve(Fd, ones(N))) ≈ W \ ones(N)
    end

    @testset "block LU over the lane-innermost layout" begin
        nc, n = 5, 3
        B = BlockJacobian{Float64}(nc, n, randn(rng, n * n * nc))
        α = 2.0
        for pivot in (true, false)
            Wb = form_W(B, α)
            D = todense(Wb)
            Fb = lu_factor(Wb, pivot)
            @test issuccess_lu(Fb)
            b = randn(rng, nc * n)
            @test lu_solve(Fb, b) ≈ D \ b rtol = 1.0e-12
            @test lu_solve_transpose(Fb, b) ≈ D' \ b rtol = 1.0e-12
            # The same arithmetic per cell as the dense LU of that cell's block, bitwise.
            Fd = lu_factor(form_W(DenseJacobian(todense(B)), α), pivot)
            @test lu_solve(Fb, b) == lu_solve(Fd, b)
        end
        # One singular lane makes W singular.
        data = randn(rng, n * n * nc)
        for i in 0:(n - 1), j in 0:(n - 1)
            data[(i * n + j) * nc + 3] = 0.0
        end
        @test !issuccess_lu(lu_factor(form_W(BlockJacobian{Float64}(nc, n, data), 0.0), false))
    end

    base(alg; kw...) = Dict{String, Any}(
        "esm_integrator" => "0.1.0", "model" => "toy:forced_decay",
        "schedule" => Dict{String, Any}("leaf" => "decay", "alg" => alg, (String(k) => v for (k, v) in kw)...), "step" => Dict("dt" => 2)
    )

    @testset "KPP methods: order and adjoint on forced_decay" begin
        prob = toy_problem("toy:forced_decay")
        exact = prob.exact(2.0)[1]
        for (alg, p) in (("KPP_Ros2", 2), ("KPP_Ros3", 3), ("KPP_Ros4", 4), ("KPP_Rodas3", 3), ("KPP_Rodas4", 4))
            dts = alg == "KPP_Ros2" ? [0.025, 0.0125, 0.00625, 0.003125] : [0.1, 0.05, 0.025, 0.0125]
            errs = [abs(forward!(build_run(base(alg; dt = dt); tspan = (0, 2))).u[1] - exact) for dt in dts]
            x, y = log.(dts), log.(errs)
            slope = sum((x .- mean(x)) .* (y .- mean(y))) / sum((x .- mean(x)) .^ 2)
            @test slope >= p - 0.3
            # Adaptive with a gradient: the adjoint against ForwardDiff on the frozen tape.
            cfg = base(alg; control = Dict("reltol" => 1.0e-6, "abstol" => 1.0e-8))
            cfg["adjoint"] = Dict("functional" => Dict("sum" => "u", "over" => "final"), "wrt" => ["u0", "params:numeric"])
            run = forward!(build_run(cfg; tspan = (0, 2)))
            @test run.retcode == "Success"
            J, ū0, p̄ = gradient(run)
            g = ForwardDiff.gradient(x -> replay(run, x[1:1], x[2:3])[1][1], vcat(run.u0, run.p))
            @test isapprox(vcat(ū0, p̄), g; rtol = 1.0e-12)
        end
    end

    @testset "f reuse follows OrdinaryDiffEq's rule (third stage on)" begin
        @test RosenbrockMethod(TABLEAUS["ROS3P"]).reuse_f == [false, false, true]
        @test !any(RosenbrockMethod(TABLEAUS["Rodas4"]).reuse_f)
        @test !any(RosenbrockMethod(TABLEAUS["Rodas3"]).reuse_f)
        @test RosenbrockMethod(TABLEAUS["KPP_Ros4"]).reuse_f == [false, false, false, true]
        # One f per stage except reused ones: ROS3P at fixed dt takes 2 per step.
        run = forward!(build_run(base("ROS3P"; dt = 0.5); tspan = (0, 2)))
        @test run.nf == 2 * 4
        # Rosenbrock23 evaluates f(u_{n+1}) every step, fixed dt included (first same as last).
        run = forward!(build_run(base("Rosenbrock23"; dt = 0.5); tspan = (0, 2)))
        @test run.nf == 1 + 2 * 4
    end

    @testset "block_lu and dense_lu give the same run" begin
        cfg(ls) = Dict(
            "esm_integrator" => "0.1.0", "model" => "toy:photostationary_cells",
            "schedule" => Dict(
                "leaf" => "photochemistry", "alg" => "Rodas4", "linsolve" => ls,
                "control" => Dict("reltol" => 1.0e-6, "abstol" => 1.0e-6)
            ), "step" => Dict("dt" => 600)
        )
        rb = forward!(build_run(cfg(Dict("block_lu" => Dict("pivot" => false))); tspan = (0, 1200)))
        rd = forward!(build_run(cfg("dense_lu"); tspan = (0, 1200)))
        @test rb.accepted == rd.accepted
        @test isapprox(rb.u, rd.u; rtol = 1.0e-12)
    end

    @testset "singular W" begin
        # forced_decay has J = -k, so k = -1/(hγ) makes Ŵ = 1/(hγ) - J exactly zero.
        γ = TABLEAUS["Rodas4"].gamma
        h = 0.1
        prob = toy_problem("toy:forced_decay")
        p = [-1 / (h * γ), 3.0]
        run = forward!(build_run(base("Rodas4"; dt = h); tspan = (0, 2), p))
        @test run.retcode == "Unstable"
        # Adaptive: the step is rejected and the controller shrinks it.
        cfg = base("Rodas4"; control = Dict("dt0" => h, "reltol" => 1.0e-3))
        run = forward!(build_run(cfg; tspan = (0, 0.5), p))
        @test run.rejected >= 1
        @test first(r for r in run.tape.records if r isa LeafRecord).steps[1][2] < h
    end

    @testset "refusals" begin
        refused(cfg) = try
            build_run(cfg; tspan = (0, 2)); nothing
        catch e
            e
        end
        @test refused(base("Rodas4"; dt = 0.1, linsolve = "block_lu")) isa ConfigError   # forced_decay is dense
        @test refused(base("Rodas4"; dt = 0.1, linsolve = "gmres")) isa UnsupportedError
        @test refused(base("Rodas4"; dt = 0.1, jacobian = "fd")) isa UnsupportedError
        @test refused(base("Rodas4"; dt = 0.1, jacobian = "symbolic", linsolve = "dense_lu")) === nothing
        @test refused(base("Tsit5"; dt = 0.1, domain = Dict("check_stages" => true))) isa UnsupportedError
        # A functional over time needs a single-leaf schedule.
        smap = Dict(
            "esm_integrator" => "0.1.0", "model" => "toy:source_decay_map",
            "schedule" => Dict("sequence" => [Dict("leaf" => "source", "alg" => "RK4", "dt" => "step"), Dict("map" => "decay")]),
            "step" => Dict("dt" => 0.5), "adjoint" => Dict("functional" => Dict("sum" => "u", "over" => "time"), "wrt" => ["u0"])
        )
        @test refused(smap) isa UnsupportedError
    end

    @testset "functional over time" begin
        # ∫u dt on forced_decay with explicit and Rosenbrock leaves: q against
        # the exact integral, and the gradient against ForwardDiff on the frozen tape.
        prob = toy_problem("toy:forced_decay")
        k, ω = prob.p
        A(s) = (k * sin(ω * s) - ω * cos(ω * s)) / (k^2 + ω^2)
        Aint(s) = (-k * cos(ω * s) / ω - sin(ω * s)) / (k^2 + ω^2)
        c = prob.u0[1] - A(0.0)
        exact = c * (1 - exp(-2k)) / k + Aint(2.0) - Aint(0.0)
        for alg in ("RK4", "Tsit5", "Rosenbrock23", "Rodas4", "Rodas5P")
            cfg = base(alg; dt = 0.01)
            cfg["adjoint"] = Dict("functional" => Dict("sum" => "u", "over" => "time"), "wrt" => ["u0", "params:numeric"])
            run = forward!(build_run(cfg; tspan = (0, 2)))
            # Second order for Rosenbrock23, whose state error at dt 0.01 is about 1e-5.
            @test isapprox(run.q, exact; rtol = alg == "Rosenbrock23" ? 1.0e-4 : 1.0e-6)
            J, ū0, p̄ = gradient(run)
            @test J == run.q
            @test replay(run)[3] == run.q
            g = ForwardDiff.gradient(x -> replay(run, x[1:1], x[2:3])[3], vcat(run.u0, run.p))
            @test isapprox(vcat(ū0, p̄), g; rtol = 1.0e-12)
        end
    end
end
