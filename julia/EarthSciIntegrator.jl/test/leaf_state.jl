# A leaf advances only its own state, the union of its operators' writes
# (spec/operator-interface.md §1, §8.2): every other state is a constant of
# the leaf over its interval and comes out unchanged bit for bit.
# advection_reaction's transport leaf writes A only; B is set nonzero here,
# with negative values and signed zeros, so that a leaf carrying B through its
# stage combinations would round it (Shu–Osher forms) or turn -0.0 into +0.0.

using EarthSciIntegrator: build_run, forward!, toy_problem, gradient, PartOperator, DenseStructure,
    BlockDiagonalStructure, ToyProblem, AbstractOperator, addentry!

const LS_NC = 64

function ls_u0(prob)
    u0 = copy(prob.u0)
    for c in 0:(LS_NC - 1)
        u0[LS_NC + c + 1] = c % 8 == 0 ? 0.0 : c % 8 == 1 ? -0.0 : c % 8 == 2 ? -0.3 - 1.0e-3 * c :
            0.1 + 0.37 * abs(sin(c * 0.71)) + 1.0e-17 * c
    end
    return u0
end

ls_cfg(sch; dt = 0.01, model = "toy:advection_reaction") =
    Dict{String, Any}("esm_integrator" => "0.1.0", "model" => model, "schedule" => sch, "step" => Dict("dt" => dt))

@testset "a leaf advances only its own state" begin
    prob = toy_problem("toy:advection_reaction")
    u0 = ls_u0(prob)
    A, B = 1:LS_NC, (LS_NC + 1):(2 * LS_NC)
    ctl = Dict("reltol" => 1.0e-8, "abstol" => 1.0e-10)
    leaves = Pair{String, Any}[]
    for a in ("ForwardEuler", "Heun", "SSPRK33", "SSPRK43", "RK4", "Tsit5", "BS3", "SSPRK104", "WRF_RK3")
        push!(leaves, "$a fixed" => Dict("leaf" => "transport", "alg" => a, "dt" => 0.001))
    end
    for a in ("Heun", "SSPRK43", "RK4", "Tsit5", "BS3")
        push!(leaves, "$a adaptive" => Dict("leaf" => "transport", "alg" => a, "control" => ctl))
    end
    for a in (
            "Rosenbrock23", "Rodas3", "Rodas4", "Rodas5P", "ROS2", "ROS3P", "KPP_Ros4", "KPP_Rodas3",
            "TRBDF2", "SDIRK2", "Kvaerno3", "Kvaerno4", "BackwardEuler",
        )
        push!(leaves, "$a adaptive" => Dict("leaf" => "transport", "alg" => a, "control" => ctl))
        push!(leaves, "$a fixed" => Dict("leaf" => "transport", "alg" => a, "dt" => 0.005))
    end
    for th in (0.5, 0.7)
        push!(leaves, "Theta $th" => Dict("leaf" => "transport", "alg" => "Theta", "theta" => th, "dt" => 0.005))
    end
    for s in ("converged", "linearized")
        push!(
            leaves, "IMEXEuler $s" => Dict(
                "leaf" => "transport", "alg" => "IMEXEuler", "imex" => Dict("implicit" => "transport"),
                "implicit_solve" => s, "dt" => 0.005
            )
        )
    end
    push!(leaves, "subcycle" => Dict("subcycle" => Dict("leaf" => "transport", "alg" => "SSPRK33", "dt" => "step"), "n" => 4))
    push!(
        leaves, "sequence" => Dict(
            "sequence" => [
                Dict("leaf" => "transport", "alg" => "SSPRK43", "control" => ctl),
                Dict("leaf" => "transport", "alg" => "Rodas4", "dt" => 0.005),
            ]
        )
    )
    push!(
        leaves, "strang" => Dict(
            "strang" => [
                Dict("leaf" => "transport", "alg" => "SSPRK43", "control" => ctl),
                Dict("leaf" => "transport", "alg" => "Kvaerno4", "control" => ctl),
            ]
        )
    )
    for (what, sch) in leaves
        @testset "$what" begin
            r = forward!(build_run(ls_cfg(sch); tspan = (0.0, 0.02), u0))
            @test r.retcode == "Success"
            @test reinterpret(UInt64, r.u[B]) == reinterpret(UInt64, u0[B])
            @test r.u[A] != u0[A]
        end
    end

    # A root leaf lands on the output times inside its interval and reports
    # the model state there, B included.
    r = forward!(
        build_run(
            ls_cfg(Dict("leaf" => "transport", "alg" => "SSPRK43", "control" => ctl); dt = 0.02);
            tspan = (0.0, 0.02), saveat = [0.005, 0.0125, 0.02], u0
        )
    )
    @test [t for (t, _) in r.outputs] == [0.0, 0.005, 0.0125, 0.02]
    @test all(x -> length(x) == 2 * LS_NC && reinterpret(UInt64, x[B]) == reinterpret(UInt64, u0[B]), last.(r.outputs))

    # The transport leaf's state vector, tolerances and storage are A's 64
    # DOFs; the reaction leaf, which writes everything, is unbound.
    run = build_run(
        ls_cfg(
            Dict(
                "sequence" => [
                    Dict("leaf" => "transport", "alg" => "Rodas4", "control" => ctl),
                    Dict("leaf" => "reaction", "alg" => "Rodas4", "control" => ctl),
                ]
            )
        ); tspan = (0.0, 0.02)
    )
    tr, re = run.leaves["root.0"], run.leaves["root.1"]
    @test tr.state.dofs == collect(A) && length(tr.tol.abstol) == LS_NC && tr.tol.dofs == collect(1:LS_NC)
    @test EarthSciIntegrator.structure(tr.ops) isa DenseStructure
    @test all(o -> o isa PartOperator, tr.ops.ops)
    @test re.state === nothing
    @test EarthSciIntegrator.structure(re.ops) == BlockDiagonalStructure(LS_NC, 2)
end

@testset "the adjoint through a leaf that advances part of the state" begin
    ctl = Dict("reltol" => 1.0e-8, "abstol" => 1.0e-10)
    for (what, sch) in (
            "SSPRK43" => Dict("leaf" => "transport", "alg" => "SSPRK43", "control" => ctl),
            "Rodas4" => Dict("leaf" => "transport", "alg" => "Rodas4", "dt" => 0.005),
            "Kvaerno4" => Dict("leaf" => "transport", "alg" => "Kvaerno4", "dt" => 0.005),
            "IMEXEuler" => Dict(
                "leaf" => "transport", "alg" => "IMEXEuler", "imex" => Dict("implicit" => "transport"),
                "implicit_solve" => "linearized", "dt" => 0.005
            ),
            "strang" => Dict(
                "strang" => [
                    Dict("leaf" => "transport", "alg" => "SSPRK43", "control" => ctl),
                    Dict("leaf" => "reaction", "alg" => "Rodas4", "control" => ctl),
                ]
            ),
        )
        c = xcase("leaf_state.$what", "toy:advection_reaction", sch, Dict("dt" => 0.01), [0.0, 0.02]; adjoint = xgrad("B"), checks = XGRAD)
        run_xcase(c)
        if what != "strang"
            # Σ B is untouched by transport: dJ/dB₀ = 1 and dJ/dA₀ = 0, exactly.
            run = forward!(build_case(c))
            _, ū0, _ = gradient(run)
            @test all(==(0.0), ū0[1:LS_NC]) && all(==(1.0), ū0[(LS_NC + 1):end])
        end
    end
end

# A two-state problem: `push` writes x and reads y (x' = -y), `relax` writes y
# (y' = -y). A leaf of push alone reads y as a constant of the leaf.
struct LSPush <: AbstractOperator end
EarthSciIntegrator.opname(::LSPush) = "push"
EarthSciIntegrator.structure(::LSPush) = DenseStructure()
EarthSciIntegrator.writes(::LSPush, N) = [1]
EarthSciIntegrator.depends_on_t(::LSPush) = false
EarthSciIntegrator.linear_in_u(::LSPush) = true
EarthSciIntegrator.reads(::LSPush, N) = [2]
EarthSciIntegrator.rhs!(::LSPush, out, u, p, t) = (out[1] += -u[2]; out)
EarthSciIntegrator.jvp_u!(::LSPush, out, u, p, t, v) = (out[1] += -v[2]; out)
EarthSciIntegrator.vjp_u!(::LSPush, out, u, p, t, w) = (out[2] += -w[1]; out)
EarthSciIntegrator.jac!(::LSPush, out, u, p, t) = (addentry!(out, 1, 2, -1.0); out)
struct LSRelax <: AbstractOperator end
EarthSciIntegrator.opname(::LSRelax) = "relax"
EarthSciIntegrator.structure(::LSRelax) = DenseStructure()
EarthSciIntegrator.writes(::LSRelax, N) = [2]
EarthSciIntegrator.depends_on_t(::LSRelax) = false
EarthSciIntegrator.linear_in_u(::LSRelax) = true
EarthSciIntegrator.reads(::LSRelax, N) = [2]
EarthSciIntegrator.rhs!(::LSRelax, out, u, p, t) = (out[2] += -u[2]; out)
EarthSciIntegrator.jvp_u!(::LSRelax, out, u, p, t, v) = (out[2] += -v[2]; out)
EarthSciIntegrator.vjp_u!(::LSRelax, out, u, p, t, w) = (out[2] += -w[2]; out)

ls_push_relax() = ToyProblem(
    "push_relax", 2, [1.0, 0.75], Float64[], String[], (0.0, 1.0),
    Dict{String, AbstractOperator}("push" => LSPush(), "relax" => LSRelax()),
    Dict{String, EarthSciIntegrator.AbstractMapOperator}(), ["x" => [1], "y" => [2]], Function[], nothing, false
)

@testset "a leaf reads the states it does not advance as constants" begin
    for alg in ("RK4", "Rodas4", "BackwardEuler")
        r = forward!(build_run(ls_cfg(Dict("leaf" => "push", "alg" => alg, "dt" => 0.125); dt = 0.5); tspan = (0.0, 1.0), prob = ls_push_relax()))
        @test r.retcode == "Success"
        @test reinterpret(UInt64, r.u[2]) == reinterpret(UInt64, 0.75)
        @test isapprox(r.u[1], 0.25; atol = 1.0e-14)
    end
    # After relax, push reads the y relax left at the start of its invocation.
    sch = Dict(
        "sequence" => [
            Dict("leaf" => "relax", "alg" => "RK4", "dt" => 0.5), Dict("leaf" => "push", "alg" => "RK4", "dt" => 0.5),
        ]
    )
    r = forward!(build_run(ls_cfg(sch; dt = 0.5); tspan = (0.0, 0.5), prob = ls_push_relax()))
    @test r.u[2] < 0.75 && isapprox(r.u[1], 1 - 0.5 * r.u[2]; atol = 1.0e-15)
    # The cotangent of a leaf's constants is not implemented: its adjoint is refused.
    cfg = ls_cfg(Dict("leaf" => "push", "alg" => "RK4", "dt" => 0.125); dt = 0.5)
    cfg["adjoint"] = Dict("functional" => Dict("sum" => "x", "over" => "final"), "wrt" => ["u0"])
    err = try
        build_run(cfg; tspan = (0.0, 1.0), prob = ls_push_relax()); nothing
    catch e
        e
    end
    @test err isa UnsupportedError && occursin("push", err.msg)
end
