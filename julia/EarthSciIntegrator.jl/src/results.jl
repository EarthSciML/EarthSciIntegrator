# Result files (spec/toy-problems.md §3, spec/schemas/result.schema.json).

_jarr(v) = "[" * join((_jnum(Float64(x)) for x in v), ", ") * "]"

"""
    result_json(case_id, producer, t, u; accepted, rejected, rhs_calls=nothing, functional=nothing, gradient=nothing) -> String

One case's result file. Numbers are written in shortest round-trip form.
`gradient` is a named tuple or dict with optional `u0` and `p` vectors.
"""
function result_json(
        case_id::AbstractString, producer::AbstractString, t, u; accepted::Integer, rejected::Integer,
        rhs_calls = nothing, functional = nothing, gradient = nothing
    )
    io = IOBuffer()
    print(
        io, "{\n \"case\": ", _jstr(case_id), ",\n \"producer\": ", _jstr(producer), ",\n \"t\": ", _jarr(t),
        ",\n \"u\": [", join((_jarr(x) for x in u), ",\n  "), "],\n \"stats\": {\"accepted\": ", accepted,
        ", \"rejected\": ", rejected
    )
    rhs_calls === nothing || print(io, ", \"rhs_calls\": ", rhs_calls)
    print(io, "}")
    if functional !== nothing
        print(io, ",\n \"functional\": ", _jnum(Float64(functional)), ",\n \"gradient\": {")
        parts = String[]
        haskey(gradient, :u0) && push!(parts, "\"u0\": " * _jarr(gradient[:u0]))
        haskey(gradient, :p) && push!(parts, "\"p\": " * _jarr(gradient[:p]))
        print(io, join(parts, ", "), "}")
    end
    print(io, "\n}\n")
    return String(take!(io))
end

"""
    result_json(case_id, run; gradient=nothing) -> String

The result file of a forward run (`producer` `julia`): the state at `t₀` and
every output time, the step statistics summed over leaves and macro steps, and
the functional and gradient for gradient cases (`gradient = (J, ū₀, p̄)` from
[`gradient`](@ref)).
"""
function result_json(case_id::AbstractString, run::Run; gradient = nothing)
    ts = [t for (t, _) in run.outputs]
    us = [u for (_, u) in run.outputs]
    if gradient === nothing
        return result_json(case_id, "julia", ts, us; accepted = run.accepted, rejected = run.rejected, rhs_calls = run.nf)
    end
    J, ū0, p̄ = gradient
    g = Dict{Symbol, Vector{Float64}}()
    run.wrt_u0 && (g[:u0] = ū0)
    isempty(run.wrt_p) || (g[:p] = p̄[run.wrt_p])
    return result_json(
        case_id, "julia", ts, us; accepted = run.accepted, rejected = run.rejected, rhs_calls = run.nf,
        functional = J, gradient = g
    )
end
