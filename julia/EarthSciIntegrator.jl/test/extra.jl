# Checks beyond cases.json that this stage's code paths need: every explicit
# method's adjoint on nonlinear and block-diagonal problems, the combinator
# adjoints with explicit leaves (cases.json exercises them only with Rodas4
# leaves, which arrive in a later stage), the I and PID controllers, domain
# clipping and its adjoint, and the return codes.

xcase(id, model, schedule, step, tspan; adjoint = nothing, saveat = nothing, checks = Dict{String, Any}[]) = begin
    cfg = Dict{String, Any}("esm_integrator" => "0.1.0", "model" => model, "schedule" => schedule, "step" => step)
    adjoint === nothing || (cfg["adjoint"] = adjoint)
    c = Dict{String, Any}("id" => id, "run_config" => cfg, "tspan" => tspan, "checks" => checks)
    saveat === nothing || (c["saveat"] = saveat)
    c
end
const XGRAD = [Dict{String, Any}("check" => "dot_product", "rtol" => 1e-12),
    Dict{String, Any}("check" => "fd_gradient", "rtol" => 1e-6, "epsilons" => [1e-3, 1e-4, 1e-5]),
    Dict{String, Any}("check" => "replay")]
xgrad(q, over = "final", how = "sum") = Dict{String, Any}("functional" => Dict{String, Any}(how => q, "over" => over),
    "wrt" => ["u0", "params:numeric"])

function run_xcase(c)
    @testset "$(c["id"])" begin
        run = forward!(build_case(c))
        @test run.retcode == "Success"
        for chk in c["checks"]
            CHECKS[chk["check"]](c, run, chk)
        end
    end
end

const EXPLICIT = ["ForwardEuler", "Heun", "RK4", "BS3", "Tsit5", "WRF_RK3", "SSPRK33", "SSPRK43", "SSPRK104"]
const ADAPTIVE_EXPLICIT = ["Heun", "RK4", "BS3", "Tsit5", "SSPRK43"]

@testset "explicit methods beyond cases.json" begin
    n0 = length(OUTCOMES)
    for alg in EXPLICIT
        # Nonlinear, autonomous; gradient of the final y and of y at every output time.
        run_xcase(xcase("x.van_der_pol.$alg.fixed", "toy:van_der_pol/nonstiff",
            Dict("leaf" => "oscillator", "alg" => alg, "dt" => 0.05), Dict("dt" => 0.5), [0, 2.0];
            adjoint = xgrad("y", "saveat"), saveat = [1.0, 2.0], checks = XGRAD))
        # Block-diagonal, time-dependent; gradient of the mean O3.
        run_xcase(xcase("x.photostationary.$alg.fixed", "toy:photostationary_cells",
            Dict("leaf" => "photochemistry", "alg" => alg, "dt" => 10.0), Dict("dt" => 600), [0, 1200.0];
            adjoint = xgrad("O3", "final", "mean"), checks = XGRAD))
    end
    for alg in ADAPTIVE_EXPLICIT
        # Adaptive, with interior output times as stops inside the root leaf.
        tol = Dict("reltol" => 1e-7, "abstol" => 1e-9)
        run_xcase(xcase("x.van_der_pol.$alg.adaptive", "toy:van_der_pol/nonstiff",
            Dict("leaf" => "oscillator", "alg" => alg, "control" => tol), Dict("dt" => 6.3), [0, 6.3];
            adjoint = xgrad("y", "saveat"), saveat = [1.0, 3.0, 6.3],
            checks = vcat(XGRAD, [Dict{String, Any}("check" => "oracle", "rtol" => 1e-12)])))
    end
    for ctrl in ("I", "PID")
        run_xcase(xcase("x.van_der_pol.Tsit5.$ctrl", "toy:van_der_pol/nonstiff",
            Dict("leaf" => "oscillator", "alg" => "Tsit5", "control" => Dict("reltol" => 1e-8, "abstol" => 1e-10, "controller" => ctrl)),
            Dict("dt" => 6.3), [0, 6.3]; saveat = [1.0, 3.0, 6.3],
            checks = [Dict{String, Any}("check" => "oracle", "rtol" => 1e-12), Dict{String, Any}("check" => "replay")]))
    end
    # Combinators with explicit leaves: sequence, strang, subcycle, adaptive leaves inside.
    transport = Dict("leaf" => "transport", "alg" => "SSPRK43", "control" => Dict("reltol" => 1e-8, "abstol" => 1e-10))
    reaction = Dict("leaf" => "reaction", "alg" => "Tsit5", "control" => Dict("reltol" => 1e-8, "abstol" => 1e-10))
    for (name, node) in [("lie", Dict("sequence" => [transport, reaction])), ("strang", Dict("strang" => [transport, reaction])),
        ("subcycle", Dict("sequence" => [Dict("subcycle" => Dict("leaf" => "transport", "alg" => "SSPRK33", "dt" => "step"), "n" => 4),
            Dict("leaf" => "reaction", "alg" => "RK4", "dt" => 2e-3)]))]
        run_xcase(xcase("x.advection_reaction.$name", "toy:advection_reaction", node, Dict("dt" => 0.025), [0, 0.1];
            adjoint = xgrad("B"), checks = vcat(XGRAD, [Dict{String, Any}("check" => "invariant", "rtol" => 1e-12)])))
    end
    # Splitting order with explicit leaves against the unsplit reference.
    for (name, node, order) in [("lie", Dict("sequence" => [transport, reaction]), 1), ("strang", Dict("strang" => [transport, reaction]), 2)]
        c = xcase("advection_reaction.$name.order", "toy:advection_reaction", node, Dict("dt" => 0.05), [0, 0.5];
            checks = [Dict{String, Any}("check" => "order", "vary" => "macro", "dts" => [0.05, 0.025, 0.0125, 0.00625],
                "expected" => order, "against" => "reference")])
        c["id"] = "advection_reaction.$name.order"       # reads that case's reference
        run_xcase(c)
    end
    # Domain clipping (forced_decay crosses zero) and its adjoint.
    run_xcase(xcase("x.forced_decay.clip", "toy:forced_decay",
        Dict("leaf" => "decay", "alg" => "SSPRK43", "control" => Dict("reltol" => 1e-6, "abstol" => 1e-8),
            "domain" => Dict("nonnegative" => "all", "on_violation" => "clip")), Dict("dt" => 2), [0, 2.0];
        adjoint = xgrad("u"), checks = XGRAD))
    # A leaf whose operators write a subset of the state: transport writes A only,
    # so its error norm and first step sum over A's 64 DOFs (spec/methods.md §7, §9).
    tr = xcase("x.advection_reaction.transport_only", "toy:advection_reaction",
        Dict("leaf" => "transport", "alg" => "Tsit5", "control" => Dict("reltol" => 1e-8, "abstol" => 1e-10)),
        Dict("dt" => 0.1), [0, 0.1]; adjoint = xgrad("A"),
        checks = vcat(XGRAD, [Dict{String, Any}("check" => "oracle", "rtol" => 1e-12)]))
    run_xcase(tr)
    let run = build_case(tr)
        @test run.root.tol.dofs == collect(1:64)
        forward!(run)
        @test run.u[65:128] == run.u0[65:128]
        # The same run with the norm over the whole state takes different steps.
        whole = EarthSciIntegrator.Tolerances(run.root.tol.reltol, run.root.tol.abstol)
        @test whole.dofs == collect(1:128)
        l = run.root
        lw = EarthSciIntegrator.LeafNode(l.id, l.opnames, l.ops, l.tab, l.method, l.dt, whole, l.dt0, l.dtmin, l.dtmax,
            l.controller, l.maxiters, l.nonnegative, l.on_violation, l.bindings)
        cs = EarthSciIntegrator.ControllerState(l.tab.controller)
        ub = EarthSciIntegrator.leaf_forward(lw, cs, copy(run.u0), run.p, 0.0, 0.1, "0/root", 0.1, [0.1])
        @test ub.record.steps != [s for r in run.tape.records if r isa LeafRecord for s in r.steps]
    end
    # bind_dt: the leaf's step bound to a parameter, which gets no cotangent.
    run_xcase(xcase("x.forced_decay.bind_dt", "toy:forced_decay",
        Dict("leaf" => "decay", "alg" => "RK4", "dt" => 0.1, "bind_dt" => Dict("omega" => "node")), Dict("dt" => 0.5), [0, 2.0];
        adjoint = xgrad("u"), checks = XGRAD))
    # Clips on the tape (spec/tape.md §2): [[k, [i, ...]], ...], k the 0-based step
    # index within `steps`, i the 0-based clipped DOFs; absent when nothing clipped.
    let run = forward!(build_case(xcase("x.forced_decay.clip", "toy:forced_decay",
            Dict("leaf" => "decay", "alg" => "SSPRK43", "control" => Dict("reltol" => 1e-6, "abstol" => 1e-8),
                "domain" => Dict("nonnegative" => "all", "on_violation" => "clip")), Dict("dt" => 2), [0, 2.0])))
        line = first(split(EarthSciIntegrator.tape_string(run.tape), '\n'))
        d = JSON.parse(line)
        @test haskey(d, "clips") && !isempty(d["clips"])
        @test all(c -> c[2] == [0] && 0 <= c[1] < length(d["steps"]), d["clips"])
        rec = run.tape.records[1]
        @test all(k -> run.tape.records[1].steps[k + 1][1] >= 0, first.(rec.clips))
        # A run that clips nothing writes no clips key.
        r2 = forward!(build_case(xcase("x.nc", "toy:forced_decay", Dict("leaf" => "decay", "alg" => "RK4", "dt" => 0.1,
            "domain" => Dict("nonnegative" => "all", "on_violation" => "clip")), Dict("dt" => 0.5), [0, 0.5])))
        @test !occursin("clips", EarthSciIntegrator.tape_string(r2.tape))
    end
    for o in OUTCOMES[(n0 + 1):end]
        println(rpad(o.passed ? "  pass" : "  FAIL", 7), rpad(o.case, 42), rpad(o.check, 15), o.detail)
    end
end

@testset "return codes" begin
    base(leaf) = xcase("rc", "toy:forced_decay", leaf, Dict("dt" => 2), [0, 2.0])
    clipped = run -> any(r -> r isa LeafRecord && !isempty(r.clips), run.tape.records)
    r = forward!(build_case(base(Dict("leaf" => "decay", "alg" => "Tsit5", "dt" => 0.1,
        "domain" => Dict("nonnegative" => ["u"])))))
    @test r.retcode == "DomainViolation"
    @test occursin("root", r.failure)
    r = forward!(build_case(base(Dict("leaf" => "decay", "alg" => "Tsit5", "control" => Dict("reltol" => 1e-6, "abstol" => 1e-8),
        "domain" => Dict("nonnegative" => "all")))))
    @test r.retcode in ("Unstable", "DtLessThanMin")
    r = forward!(build_case(base(Dict("leaf" => "decay", "alg" => "Tsit5", "control" => Dict("reltol" => 1e-6, "abstol" => 1e-8,
        "dtmin" => 1e-3), "domain" => Dict("nonnegative" => "all")))))
    @test r.retcode == "DtLessThanMin"
    r = forward!(build_case(base(Dict("leaf" => "decay", "alg" => "Tsit5", "control" => Dict("reltol" => 1e-10, "maxiters" => 5)))))
    @test r.retcode == "MaxIters"
    r = forward!(build_case(base(Dict("leaf" => "decay", "alg" => "ForwardEuler", "dt" => 2.0))))
    @test r.retcode == "Success"
    # Explicit Euler far beyond its stability limit on stiff van der Pol: non-finite state.
    c = xcase("rc", "toy:van_der_pol/stiff", Dict("leaf" => "oscillator", "alg" => "ForwardEuler", "dt" => 0.1), Dict("dt" => 1), [0, 1.0])
    @test forward!(build_case(c)).retcode == "Unstable"
end

# Stops near the start, against OrdinaryDiffEqCore 4.19.0's Tsit5 with these
# tstops (van der Pol, nonstiff, reltol 1e-8, abstol 1e-10). With dt0 the first
# step lands on the first stop and the next proposal comes from the
# unshortened size (spec/methods.md §8.6, D61); without dt0 the automatic first
# step is capped at the first stop (§9).
@testset "proposal after a stop and the first-step cap" begin
    vdp(control, saveat) = forward!(build_case(xcase("x.vdp.stops", "toy:van_der_pol/nonstiff",
        Dict("leaf" => "oscillator", "alg" => "Tsit5", "control" => control), Dict("dt" => 6.3), [0, 6.3]; saveat)))
    tol = Dict{String, Any}("reltol" => 1e-8, "abstol" => 1e-10)
    for (dt0, saveat, counts, ode) in [
            (0.1, [0.002, 1.0, 3.0, 6.3], (161, 6, 1003), [1.832135188412958, 1.161899074957998]),
            (0.05, [0.01, 1.0, 3.0, 6.3], (162, 4, 997), [1.8321351883177492, 1.161899075326617]),
            (0.5, [0.02, 1.0, 3.0, 6.3], (159, 5, 985), [1.832135188369107, 1.161899075127413]),
            (nothing, [0.002, 1.0, 3.0, 6.3], (165, 3, 1011), [1.8321351883562347, 1.1618990751839322]),
            (nothing, [0.01, 1.0, 3.0, 6.3], (165, 2, 1005), [1.832135188414688, 1.1618990749463005])]
        control = dt0 === nothing ? tol : merge(tol, Dict{String, Any}("dt0" => dt0))
        run = vdp(control, saveat)
        @test run.retcode == "Success"
        @test (run.accepted, run.rejected, run.nf) == counts
        @test maximum(abs.(last(run.outputs)[2] .- ode)) < 1e-12
    end
    # spec/methods.md §12: a step at or below dtmin ends the leaf unless the
    # previous attempt was accepted and the step lands on a stop; the first
    # attempt of the run follows no accepted step (OrdinaryDiffEq: DtLessThanMin
    # with the stop at 1e-4, Success with 12 accepted steps with it at 0.1001).
    decay(saveat) = forward!(build_case(xcase("x.decay.dtmin", "toy:forced_decay",
        Dict("leaf" => "decay", "alg" => "Tsit5",
            "control" => Dict("reltol" => 1e-3, "abstol" => 1e-6, "dt0" => 0.1, "dtmin" => 1e-3)),
        Dict("dt" => 2.0), [0, 2.0]; saveat)))
    @test decay([1e-4, 2.0]).retcode == "DtLessThanMin"
    let run = decay([0.1001, 2.0])
        @test run.retcode == "Success"
        @test run.accepted == 12
    end
end
