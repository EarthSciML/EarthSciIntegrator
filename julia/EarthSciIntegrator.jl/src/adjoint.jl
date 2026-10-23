# The reverse pass (spec/adjoint.md §5, §6; architecture §10).
#
# Per macro step in reverse: restore the checkpoint, replay the macro step on
# the tape recording the input state of every leaf and map invocation, then
# run the adjoints of those invocations in reverse execution order. Within a
# leaf invocation the steps are replayed again from its input state to recover
# each step's start state, and the method's adjoint step is applied for
# n = N-1 … 0 (§5.1). The reverse of `sequence`, `strang` and `subcycle` is the
# reverse execution order (§5.2–§5.4); `bind_dt` parameters get no cotangent
# (§5.8).

"""
    leaf_adjoint!(entry, λ, p̄, seeds) -> λ

Reverse one leaf invocation. `seeds` maps interior output times to
`∇φ` (point terms of the functional at stops inside the invocation).
"""
function leaf_adjoint!(entry, λ, p̄, seeds)
    _, leaf, rec, uin, pinv, stops = entry
    states = []
    leaf_replay(leaf, uin, pinv, rec, stops; states)
    tb = last(stops)
    for (u, t, h, pp, mask, tnew, tk1) in Iterators.reverse(states)
        if tnew != tb && haskey(seeds, tnew)
            λ = λ .+ seeds[tnew]
        end
        ts = next_stop(stops, t)
        _, isstop = stop_rule(t, h, ts)
        λ, pc = leaf_step_adjoint(leaf, u, pp, t, h, mask, λ, isstop && h < eps(abs(t)), tk1)
        p̄ .+= pc
    end
    return λ
end

"""Reverse one map invocation (spec/adjoint.md §5.7)."""
function map_adjoint!(entry, λ, p̄)
    _, node, rec, uin, p, t = entry
    λb = zeros(length(λ))
    map_vjp!(node.op, λb, p̄, uin, p, t, rec.dt, rec.mask, λ)
    return λb
end

"""
    macro_adjoint(run, k, λ; seeds=Dict()) -> (λₖ, p̄ₖ)

The adjoint of macro step `k`: `λ` is the cotangent of the state at its end;
returns the cotangent at its start and its contribution to `p̄` (all
parameters, bound ones included; [`gradient`](@ref) drops those).
"""
function macro_adjoint(run::Run, k::Integer, λ::Vector{Float64}; seeds = Dict{Float64, Vector{Float64}}())
    cp = run.checkpoints[k + 1]
    trace = Any[]
    replay_macro(run, k, copy(cp.u), run.p; trace)
    p̄ = zeros(length(run.p))
    λ = copy(λ)
    for entry in Iterators.reverse(trace)
        λ = entry[1] === :leaf ? leaf_adjoint!(entry, λ, p̄, seeds) : map_adjoint!(entry, λ, p̄)
    end
    return λ, p̄
end

"""
    gradient(run) -> (J, ū₀, p̄)

The discrete adjoint gradient of the run's functional on the frozen tape
(spec/adjoint.md §2.2): `ū₀ = λ(t₀)` (§6.2) and `p̄` over every parameter,
with bind_dt-bound parameters set to zero (§5.8). Needs [`forward!`](@ref)
first. Point terms are added on reaching their output times (§2.1).
"""
function gradient(run::Run)
    run.retcode == "Success" || error("gradient needs a successful forward run (retcode $(run.retcode))")
    F = run.functional
    F === nothing && error("the run config has no adjoint.functional")
    N, P = length(run.u0), length(run.p)
    g = phi_gradient(F, N)
    seedtimes = F.over === :final ? [run.tspan[2]] : run.output_times
    seeds = Dict{Float64, Vector{Float64}}(τ => g for τ in seedtimes)
    λ = zeros(N)
    p̄ = zeros(P)
    K = length(run.bounds) - 1
    for k in (K - 1):-1:0
        tb = run.bounds[k + 2]
        haskey(seeds, tb) && (λ = λ .+ seeds[tb])
        λ, pk = macro_adjoint(run, k, λ; seeds)
        p̄ .+= pk
    end
    for i in run.bound_params
        p̄[i] = 0.0
    end
    J = functional_value(run, run.outputs)
    return J, λ, p̄
end
