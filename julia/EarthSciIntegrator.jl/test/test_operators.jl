# Every operator call of every toy problem against ForwardDiff
# (spec/toy-problems.md: "The Julia implementation's ForwardDiff Jacobian of
# `rhs` is a test oracle for them").

using EarthSciIntegrator: ToyProblem, AbstractOperator, OperatorSum, DenseJacobian, BlockJacobian, BlockDiagonalStructure,
    DenseStructure, structure, todense, depends_on_t, linear_in_u, eval_rhs!, mask_length, provides

const CALLS = (:rhs, :jvp_u, :vjp_u, :vjp_p, :jac, :ft, :hess_u, :hess_p, :ft_vjp_u, :ft_vjp_p)

f_of(op, N) = (u, p, t) -> (out = zeros(promote_type(eltype(u), eltype(p), typeof(t)), N); rhs!(op, out, u, p, t); out)

function check_operator(prob::ToyProblem, name, op::AbstractOperator; rng)
    N, P = prob.N, length(prob.p)
    f = f_of(op, N)
    # A sampled non-default state, parameters and time (not the defaults).
    u = prob.u0 .* (1 .+ 0.3 .* rand(rng, N)) .+ 0.1 .* rand(rng, N)
    p = prob.p .* (1 .+ 0.2 .* rand(rng, P))
    t = prob.tspan[1] + 0.37 * (prob.tspan[2] - prob.tspan[1])
    J = ForwardDiff.jacobian(x -> f(x, p, t), u)
    Jp = ForwardDiff.jacobian(q -> f(u, q, t), p)
    ftref = ForwardDiff.derivative(τ -> f(u, p, τ), t)
    v, w = randn(rng, N), randn(rng, N)
    pairs = [(randn(rng, N), randn(rng, N)) for _ in 1:3]
    ν = randn(rng, N)
    tol(x) = 1e-12 * max(1.0, maximum(abs, x))
    for c in CALLS
        @test provides(op, c)
    end
    # Every call accumulates: start from a nonzero output.
    base_u, base_p = randn(rng, N), randn(rng, P)
    acc(call, base, args...) = (out = copy(base); call(op, out, args...); out .- base)
    @test isapprox(acc(rhs!, base_u, u, p, t), f(u, p, t); atol = tol(f(u, p, t)))
    @test isapprox(acc(jvp_u!, base_u, u, p, t, v), J * v; atol = tol(J * v))
    @test isapprox(acc(vjp_u!, base_u, u, p, t, w), J' * w; atol = tol(J' * w))
    @test isapprox(acc(vjp_p!, base_p, u, p, t, w), Jp' * w; atol = tol(Jp' * w))
    @test isapprox(acc(ft!, base_u, u, p, t), ftref; atol = tol(ftref))
    depends_on_t(op) || @test all(iszero, ftref)
    linear_in_u(op) && @test isapprox(ForwardDiff.jacobian(x -> f(x, p, t), u .+ randn(rng, N)), J; atol = tol(J))
    # jac into the dense storage, and into the block storage when the operator is block diagonal.
    S = DenseJacobian{Float64}(N)
    jac!(op, S, u, p, t)
    @test isapprox(todense(S), J; atol = tol(J))
    s = structure(op)
    if s isa BlockDiagonalStructure
        B = BlockJacobian{Float64}(s.ncells, s.n)
        jac!(op, B, u, p, t)
        @test isapprox(todense(B), J; atol = tol(J))
        # Block-diagonal: no coupling between cells.
        cell(i) = (i - 1) % s.ncells
        @test all(J[i, j] == 0 for i in 1:N, j in 1:N if cell(i) != cell(j))
    end
    # Second-order calls.
    S2(x, q) = sum(dot(m, ForwardDiff.jacobian(y -> f(y, q, t), x) * ww) for (m, ww) in pairs)
    hu = ForwardDiff.gradient(x -> S2(x, p), u)
    hp = ForwardDiff.gradient(q -> S2(u, q), p)
    @test isapprox(acc(hess_u!, base_u, u, p, t, pairs), hu; atol = tol(hu))
    @test isapprox(acc(hess_p!, base_p, u, p, t, pairs), hp; atol = tol(hp))
    linear_in_u(op) && @test all(x -> abs(x) < 1e-14, hu)
    FT(x, q) = dot(ν, ForwardDiff.derivative(τ -> f(x, q, τ), t))
    fu = ForwardDiff.gradient(x -> FT(x, p), u)
    fp = ForwardDiff.gradient(q -> FT(u, q), p)
    @test isapprox(acc(ft_vjp_u!, base_u, u, p, t, ν), fu; atol = tol(fu))
    @test isapprox(acc(ft_vjp_p!, base_p, u, p, t, ν), fp; atol = tol(fp))
    # Generic over the element type: Float32 state.
    out32 = zeros(Float32, N)
    rhs!(op, out32, Float32.(u), p, t)
    @test eltype(out32) == Float32
    @test isapprox(out32, f(u, p, t); rtol = 1e-4, atol = 1e-4 * maximum(abs, f(u, p, t)))
end

function check_map(prob::ToyProblem, name, op; rng)
    N, P = prob.N, length(prob.p)
    u = prob.u0 .+ 1 .+ rand(rng, N)
    p = prob.p .* (1 .+ 0.2 .* rand(rng, P))
    t, Δ = 0.3, 0.7
    M(x, q) = (out = zeros(promote_type(eltype(x), eltype(q)), N); apply!(op, out, Vector{Bool}(undef, mask_length(op, N)), x, q, t, Δ); out)
    mask = Vector{Bool}(undef, mask_length(op, N))
    out = fill(NaN, N)
    apply!(op, out, mask, u, p, t, Δ)                     # overwrites
    @test out ≈ u .* exp(-p[2] * Δ)
    Ju = ForwardDiff.jacobian(x -> M(x, p), u)
    Jp = ForwardDiff.jacobian(q -> M(u, q), p)
    w, vu, vp = randn(rng, N), randn(rng, N), randn(rng, P)
    bu, bp = randn(rng, N), randn(rng, P)
    ou, op_ = copy(bu), copy(bp)
    map_vjp!(op, ou, op_, u, p, t, Δ, mask, w)
    @test ou .- bu ≈ Ju' * w
    @test op_ .- bp ≈ Jp' * w
    jv = zeros(N)
    map_jvp!(op, jv, u, p, t, Δ, mask, vu, vp)
    @test jv ≈ Ju * vu + Jp * vp
end

@testset "operators against ForwardDiff" begin
    rng = Random.MersenneTwister(20261023)
    for model in EarthSciIntegrator.TOY_MODELS
        prob = toy_problem(model)
        @testset "$model" begin
            for (name, op) in sort!(collect(prob.operators); by = first)
                @testset "$name" begin
                    check_operator(prob, name, op; rng)
                end
            end
            for (name, op) in prob.maps
                @testset "map $name" begin
                    check_map(prob, name, op; rng)
                end
            end
        end
    end
    @testset "structures and layout" begin
        ar = toy_problem("toy:advection_reaction")
        @test structure(ar.operators["transport"]) isa DenseStructure
        @test structure(ar.operators["reaction"]) == BlockDiagonalStructure(64, 2)
        @test structure(OperatorSum([ar.operators["transport"], ar.operators["reaction"]])) isa DenseStructure
        @test structure(OperatorSum([ar.operators["reaction"]])) == BlockDiagonalStructure(64, 2)
        pc = toy_problem("toy:photostationary_cells")
        # Species-major: DOF i of cell c at i*ncells + c (0-based).
        @test pc.variables[2] == ("NO" => collect(17:32))
        # Lane-innermost block storage: entry (i, j) of cell c at (i*n + j)*ncells + c.
        B = BlockJacobian{Float64}(16, 3)
        EarthSciIntegrator.addentry!(B, 1 * 16 + 5 + 1, 2 * 16 + 5 + 1, 7.0)
        @test B.data[(1 * 3 + 2) * 16 + 5 + 1] == 7.0
        @test_throws ArgumentError EarthSciIntegrator.addentry!(B, 1, 2, 1.0)
        # The exact solution of forced_decay solves the ODE.
        fd = toy_problem("toy:forced_decay")
        k, ω = fd.p
        for t in (0.0, 0.5, 1.7)
            du = ForwardDiff.derivative(τ -> fd.exact(τ)[1], t)
            @test du ≈ -k * fd.exact(t)[1] + sin(ω * t)
        end
        @test fd.exact(0.0) ≈ fd.u0
    end
end
