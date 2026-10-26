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
    return Float64(d)
end

# ---------------------------------------------------------------------------
# Timestamps, converted as esm-spec §2.2.7 converts a time_span bound (D74).

const _UTC_TIMESTAMP_RE =
    r"^([0-9]{4})-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])T([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])(\.[0-9]+)?Z\z"
const _RFC3339_RE =
    r"^([0-9]{4})-([0-9]{2})-([0-9]{2})[Tt]([0-9]{2}):([0-9]{2}):([0-9]{2})(\.[0-9]+)?([Zz]|([+-])([0-9]{2}):([0-9]{2}))\z"

_leap(y::Int) = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
_days_in_month(y::Int, m::Int) = m == 2 ? (_leap(y) ? 29 : 28) : (m in (4, 6, 9, 11) ? 30 : 31)

"""Days from 1970-01-01 to `y-m-d`, proleptic Gregorian (Hinnant's `days_from_civil`), exact integers."""
function _epoch_days(y::Int, m::Int, d::Int)
    y -= m <= 2
    era = fld(y, 400)
    yoe = y - era * 400
    doy = div(153 * (m + (m > 2 ? -3 : 9)) + 2, 5) + d - 1
    doe = yoe * 365 + div(yoe, 4) - div(yoe, 100) + doy
    return era * 146097 + doe - 719468
end

"""`(W, f)`: whole seconds from the POSIX epoch, offset subtracted, and the fraction in [0, 1); `nothing` when `s` names no instant."""
function _datetime_parts(s::AbstractString, re::Regex)
    mt = match(re, s)
    mt === nothing && return nothing
    y, mo, d, h, mi, sec = (parse(Int, mt.captures[i]) for i in 1:6)
    (1 <= mo <= 12 && 1 <= d <= _days_in_month(y, mo) && h <= 23 && mi <= 59 && sec <= 59) || return nothing
    offset = 0
    if length(mt.captures) >= 9 && mt.captures[9] !== nothing
        oh, om = parse(Int, mt.captures[10]), parse(Int, mt.captures[11])
        (oh <= 23 && om <= 59) || return nothing
        offset = (mt.captures[9] == "-" ? -1 : 1) * (oh * 3600 + om * 60)
    end
    whole = _epoch_days(y, mo, d) * 86400 + h * 3600 + mi * 60 + sec - offset
    frac = mt.captures[7] === nothing ? 0.0 : parse(Float64, "0" * mt.captures[7])
    return (whole, frac)
end

"""
The model's time origin `(W₀, f₀)` (esm-spec §2.2.7): `domain.temporal.reference_time`,
else `domain.temporal.start`, else the POSIX epoch; `nothing` when the
selected field is present but not an RFC 3339 date-time.
"""
function model_time_origin(file::EarthSciAST.EsmFile)
    dom = file.domain
    temporal = dom === nothing ? nothing : dom.temporal
    temporal isa AbstractDict || return (0, 0.0)
    for key in ("reference_time", "start")
        haskey(temporal, key) || continue
        v = temporal[key]
        return v isa AbstractString ? _datetime_parts(v, _RFC3339_RE) : nothing
    end
    return (0, 0.0)
end

"""
    time_seconds(m, T, at) -> Float64

A time of spec/run-config.md §1.1 in seconds: a number, or a UTC timestamp
counted from the model's time origin as `(W − W₀) + (f − f₀)` (D74).
`time_invalid_date` for a day its month does not have (at `at`),
`time_origin_invalid` when the origin field cannot be read.
"""
function time_seconds(m::PlanModel, T, at::AbstractString)
    T isa Real && !(T isa Bool) && return Float64(T)
    parts = _datetime_parts(String(T), _UTC_TIMESTAMP_RE)
    parts === nothing && refuse("time_invalid_date", at, "timestamp $(repr(T)) names no calendar day")
    origin = model_time_origin(m.file)
    # Spec finding: spec/plan.md §7.2 gives time_origin_invalid no pointer; the
    # timestamp that needs the origin is named.
    origin === nothing && refuse("time_origin_invalid", at,
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
