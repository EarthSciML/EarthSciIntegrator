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
    # Spec finding: §2.2 makes a duration positive, which the schema checks
    # only for a literal; a parameter's default is checked here, and §7.2 names
    # no code for it.
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
function time_seconds(m::PlanModel, T, at::AbstractString)
    T isa Real && !(T isa Bool) && return Float64(T)
    parts = EarthSciAST._datetime_parts(String(T), EarthSciAST._TIME_SPAN_TIMESTAMP_RE)
    parts === nothing && refuse("time_invalid_date", at, "timestamp $(repr(T)) names no calendar day")
    origin = EarthSciAST._time_origin(m.file.domain)
    # Spec finding: spec/plan.md §7.2 gives time_origin_invalid no pointer; the
    # timestamp that needs the origin is named.
    origin isa Tuple || refuse("time_origin_invalid", at,
        "timestamp $(repr(T)) needs the model's time origin, and its domain.temporal field is not an RFC 3339 date-time")
    (w, f), (w0, f0) = parts, origin
    return Float64(w - w0) + (f - f0)
end

"""
    resolve_time(m, config; tspan=nothing) -> (t0, tf)

spec/plan.md §7.2: the caller's interval (`tspan`, a pair of numbers or of
timestamps), else the run config's `time`, else the model's
`solver.time_span` through EarthSciAST's `resolve_time_span`. `duration` is
added to the start in binary64. Refusals at `/time`: no interval
(`time_interval_missing`), end not after start (`time_interval_empty`).

Spec finding: a refusal about the caller's interval has no run-config key;
it is reported at `/time` too.
"""
function resolve_time(m::PlanModel, config::AbstractDict; tspan = nothing)
    t0, tf = if tspan !== nothing
        time_seconds(m, tspan[1], "/time"), time_seconds(m, tspan[2], "/time")
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
            refuse("model_load_failed", "/model", "EarthSciAST refused the model's solver.time_span: " * sprint(showerror, e))
        end
        r === nothing && refuse("time_interval_missing", "/time",
            "no run interval: the caller gives none, the run config has no time, the model no solver.time_span")
        (Float64(r[1]), Float64(r[2]))
    end
    tf > t0 || refuse("time_interval_empty", "/time", "the interval's end ($tf s) is not after its start ($t0 s)")
    return (t0, tf)
end
