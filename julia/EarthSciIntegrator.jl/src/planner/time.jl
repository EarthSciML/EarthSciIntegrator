# Durations (spec/run-config.md §2.2) and the run interval (spec/plan.md §7.2,
# D60, D74).

const _DURATION_RE = r"^([0-9]*\.?[0-9]*(?:[eE][-+]?[0-9]+)?) ?(s|min|h|day)\z"
const _UNIT_SECONDS = Dict("s" => 1.0, "min" => 60.0, "h" => 3600.0, "day" => 86400.0)

"""
    plan_duration(m, x, at) -> Float64

Seconds of a duration (spec/run-config.md §2.2): a number; `"<number> <unit>"`,
the number parsed as a binary64 and multiplied by the unit's seconds (spec
finding: §2.2 does not say how the product is formed, so this is the one
rounding a planner can make); or the qualified name of a parameter, read from
its default (spec/plan.md §7.1). A name that is not in the flattened model is
`name_unresolved`, one that is not a parameter with a scalar numeric default
`name_wrong_kind` (rule 2), at `at`.
"""
function plan_duration(m::PlanModel, x, at::AbstractString)
    x isa Real && !(x isa Bool) && return Float64(x)
    x isa AbstractString || refuse("run_config_invalid", at, "not a duration: $(repr(x))")
    mt = match(_DURATION_RE, x)
    if mt !== nothing
        v = tryparse(Float64, mt.captures[1])
        v === nothing || return v * _UNIT_SECONDS[mt.captures[2]]
    end
    k = name_kind(m, x)
    k === nothing && refuse("name_unresolved", at, "duration $(repr(x)) is neither '<number> <unit>' nor a name of the model")
    k === :parameter || refuse("name_wrong_kind", at, "duration $(repr(x)) names a $k, not a parameter")
    d = m.flat.parameters[x].default
    d isa Real || refuse("name_wrong_kind", at, "duration parameter $x has no scalar numeric default")
    # §7.1: a parameter-name duration must default to a positive finite number.
    (isfinite(d) && d > 0) || refuse("run_config_invalid", at, "duration parameter $x has default $d, not a positive number of seconds")
    return Float64(d)
end

# ---------------------------------------------------------------------------
# Timestamps, converted as esm-spec §2.2.7 converts a time_span bound (D74).
# EarthSciAST's own reading is used (its internal `_datetime_parts` and
# `_time_origin`), so a planner time and a solver.time_span bound cannot
# disagree; both return the parts rather than a refusal, so the codes of
# spec/plan.md §7.2 stay the planner's.

"""
    time_seconds(m, T, at) -> Float64

A time of spec/run-config.md §1.1 in seconds: a number, or a UTC timestamp
counted from the model's time origin as `(W − W₀) + (f − f₀)` (D74).
`time_invalid_date` for a day its month does not have (at `at`),
`time_origin_invalid` when the origin field cannot be read.
"""
time_seconds(m::PlanModel, T, at::AbstractString) = _bound_seconds(T, m.file.domain, at)

"""`time_seconds` against a model `domain`, typed or raw JSON (EarthSciAST's `_time_origin` reads both)."""
function _bound_seconds(T, domain, at::AbstractString)
    T isa Real && !(T isa Bool) && return Float64(T)
    parts = EarthSciAST._datetime_parts(String(T), EarthSciAST._TIME_SPAN_TIMESTAMP_RE)
    parts === nothing && refuse("time_invalid_date", at, "timestamp $(repr(T)) names no calendar day")
    origin = EarthSciAST._time_origin(domain)
    # §7.2: time_origin_invalid is at the key of the timestamp that needs the origin.
    origin isa Tuple || refuse(
        "time_origin_invalid", at,
        "timestamp $(repr(T)) needs the model's time origin, and its domain.temporal field is not an RFC 3339 date-time"
    )
    (w, f), (w0, f0) = parts, origin
    return Float64(w - w0) + (f - f0)
end

const _MODEL_SPAN_CODES = ("time_interval_empty", "time_invalid_date", "time_origin_invalid")

"""
    model_span_refusal(start, stop, domain) -> Union{PlannerRefusal, Nothing}

The rule-14 refusal, at `/model`, for a model whose `solver.time_span`
(`start`, `stop`) EarthSciAST rejects; `domain` is the model's, typed or raw
JSON. EarthSciAST gives every such rejection the one code
`solver_time_span_invalid`; spec/plan.md §7.2 asks for the code of the
cause. `nothing` when the cause is not one of rule 14's (a malformed bound,
which the schema reports).
"""
function model_span_refusal(start, stop, domain)
    try
        s = _bound_seconds(start, domain, "/model")
        e = _bound_seconds(stop, domain, "/model")
        e > s || refuse("time_interval_empty", "/model", "the model's solver.time_span ends ($e s) no later than it starts ($s s)")
    catch err
        err isa PlannerRefusal && err.code in _MODEL_SPAN_CODES && return err
        err isa PlannerRefusal || err isa ArgumentError || err isa MethodError || rethrow()
    end
    return nothing
end

"""The `model_span_refusal` of a raw model document, or `nothing`."""
function model_span_refusal(doc)
    doc isa AbstractDict || return nothing
    solver = get(doc, "solver", nothing)
    span = solver isa AbstractDict ? get(solver, "time_span", nothing) : nothing
    (span isa AbstractDict && haskey(span, "start") && haskey(span, "end")) || return nothing
    return model_span_refusal(span["start"], span["end"], get(doc, "domain", nothing))
end

"""
    resolve_time(m, config; tspan=nothing) -> (t0, tf)

spec/plan.md §7.2: the caller's interval (`tspan`, a pair of numbers or of
timestamps), else the run config's `time`, else the model's
`solver.time_span` through EarthSciAST's `resolve_time_span`. `duration` is
added to the start in binary64. Refusals at `/time`: no interval
(`time_interval_missing`), end not after start (`time_interval_empty`). A
refusal about the caller's interval is at the empty pointer, since that
interval is not in the run config; one about the model's `solver.time_span`
takes its cause's code at `/model` ([`model_span_refusal`](@ref)).
"""
function resolve_time(m::PlanModel, config::AbstractDict; tspan = nothing)
    at = tspan === nothing ? "/time" : ""
    t0, tf = if tspan !== nothing
        time_seconds(m, tspan[1], at), time_seconds(m, tspan[2], at)
    elseif haskey(config, "time")
        tm = config["time"]
        s = time_seconds(m, tm["start"], "/time/start")
        e = haskey(tm, "end") ? time_seconds(m, tm["end"], "/time/end") :
            s + plan_duration(m, tm["duration"], "/time/duration")
        (s, e)
    else
        r = try
            EarthSciAST.resolve_time_span(m.file)
        catch e
            span = m.file.solver.time_span
            r = model_span_refusal(span.start, span.stop, m.file.domain)
            r === nothing || throw(r)
            refuse("model_load_failed", "/model", "EarthSciAST refused the model's solver.time_span: " * sprint(showerror, e))
        end
        r === nothing && refuse(
            "time_interval_missing", "/time",
            "no run interval: the caller gives none, the run config has no time, the model no solver.time_span"
        )
        (Float64(r[1]), Float64(r[2]))
    end
    tf > t0 || refuse("time_interval_empty", at, "the interval's end ($tf s) is not after its start ($t0 s)")
    return (t0, tf)
end
