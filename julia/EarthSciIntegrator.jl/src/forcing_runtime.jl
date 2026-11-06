# The runtime's data layer (spec/operator-interface.md §8.4): providers built by
# EarthSciAST from the plan's model (through EarthSciIO), the forcing epochs,
# and one buffer per data-fed parameter of the plan's `forcing`, shared by every
# operator's programs. The runtime writes no data layer of its own: locating,
# reading, unit conversion and record selection are EarthSciAST's and
# EarthSciIO's; the conversion of a sample to a slot's buffer is EarthSciAST's
# `forcing_from_sample`.

"""
    ForcingRuntime

The data of one run (spec/operator-interface.md §8.4). `names` are the plan's
used data-fed parameters that some operator reads (their buffers are what
programs' forcing slots take), `shapes` their extents, `providers` one per
name (from `providers_from_document`), `discrete` the names whose providers
refresh, `records` each one's records per sample, and `epochs` the forcing
epochs' starts `B_1 = t₀ < B_2 < …` (§8.4.3), in model seconds. A call at `t`
in a step that starts at `tₙ` takes, for a one-record parameter, the buffer
of the epoch `tₙ` is in, and for a two-record one the buffer of the epoch `t`
is in, so a stage at the step's right end takes each parameter's limit from
the left (§8.4.1, D88). Each epoch's buffers are sampled once, on the
calling thread, when a call first needs them; the most recent few are kept.
"""
mutable struct ForcingRuntime
    names::Vector{String}
    shapes::Dict{String, Vector{Int}}
    providers::Dict{String, Any}
    discrete::Set{String}
    records::Dict{String, Int}     # each name's records per sample (1 or 2)
    epochs::Vector{Float64}
    const_bufs::Dict{String, Vector{Float64}}
    cache::Dict{Int, Dict{String, Vector{Float64}}}
    order::Vector{Int}
    lock::ReentrantLock
    report::Vector{Tuple{Float64, String}}   # what was read: (model time sampled at, parameter)
end

"""The starts of the forcing epochs other than the first: the refresh times no step may cross (§8.4.3)."""
discrete_epochs(F::ForcingRuntime) = F.epochs[2:end]

"""The epoch a call at model time `t` takes (§8.4.1): the last `k` with `B_k ≤ t`, the first for `t` before it."""
epoch_of(F::ForcingRuntime, t::Real) = max(1, searchsortedlast(F.epochs, Float64(t)))

const _EPOCHS_KEPT = 4

"""The buffers of epoch `k`, by name: each discrete provider's sample at `B_k`, converted by `forcing_from_sample`; const buffers once for the run."""
function epoch_buffers(F::ForcingRuntime, k::Int)
    return lock(F.lock) do
        b = get(F.cache, k, nothing)
        b === nothing || return b
        b = Dict{String, Vector{Float64}}()
        for n in F.names
            b[n] = n in F.discrete ? sample_buffer(F, n, F.epochs[k]) : F.const_bufs[n]
        end
        F.cache[k] = b
        push!(F.order, k)
        while length(F.order) > _EPOCHS_KEPT
            delete!(F.cache, popfirst!(F.order))
        end
        b
    end
end

"""Provider `n`'s sample at model time `t` as its slot's column-major buffer (§8.4.2 step 3)."""
function sample_buffer(F::ForcingRuntime, n::AbstractString, t::Float64)
    p = F.providers[n]
    slot = EarthSciAST.ForcingSlot(String(n), F.shapes[n], :forcing)
    s = try
        EarthSciAST.provider_sample(p, t)
    catch e
        e isa InterruptException && rethrow()
        throw(ConfigError("data-fed parameter $n: reading its data at model time $t s failed: " * sprint(showerror, e)))
    end
    buf = try
        EarthSciAST.forcing_from_sample(s, slot)
    catch e
        e isa InterruptException && rethrow()
        mp = EarthSciAST.provider_extent_metaparameter(p)
        why = mp === nothing ? "" : " (its source's extent binds metaparameter $mp, which the plan sized to $(F.shapes[n]))"
        throw(ConfigError("data-fed parameter $n: the data at model time $t s does not fit its planned shape$why: " * sprint(showerror, e)))
    end
    push!(F.report, (t, String(n)))
    return buf
end

"""
    forcing_key(F, t) -> (ks, kt)

The epochs whose buffers a call at `t` takes (§8.4.1, D88): `ks`, the epoch of
the start of the step being evaluated, for one-record parameters, and `kt`,
the epoch of `t`, for two-record ones. Two calls with the same key take the
same buffers.
"""
forcing_key(F::ForcingRuntime, t::Real) = (epoch_of(F, step_start(t)), epoch_of(F, t))

"""A program's forcing set and the [`forcing_key`](@ref) it was built for."""
struct EpochSet
    key::NTuple{2, Int}
    fs::EarthSciAST.ForcingSet
end

"""The holder of a program's last [`EpochSet`](@ref)."""
const EpochSetRef = Base.RefValue{Union{Nothing, EpochSet}}
EpochSetRef() = EpochSetRef(nothing)

"""
    forcing_set_for(F, prog, t, cache) -> ForcingSet

The forcing set of program `prog` for a call at `t` (§8.4.1): the buffers its
slots name, from the epochs [`forcing_key`](@ref) selects. `cache` holds the
program's set of the last key it was asked for, so a set is built once per
epoch and program (a new set at each boundary, never an in-place write, §8.4.3).
"""
function forcing_set_for(F::ForcingRuntime, prog, t::Real, cache::EpochSetRef)
    key = forcing_key(F, t)
    c = cache[]
    c !== nothing && c.key == key && return c.fs
    ks, kt = key
    bt = epoch_buffers(F, kt)
    bs = ks == kt ? bt : epoch_buffers(F, ks)
    fs = EarthSciAST.forcing_set(
        prog, [F.records[s.name] == 2 ? bt[s.name] : bs[s.name] for s in EarthSciAST.forcing_slots(prog)]
    )
    cache[] = EpochSet(key, fs)
    return fs
end

"""Whether `t` starts a forcing epoch other than the first (§8.4.1, D88)."""
is_epoch_start(F::ForcingRuntime, t::Real) = length(F.epochs) > 1 && insorted(Float64(t), @view F.epochs[2:end])

"""
    ShiftedProvider(inner, shift)

A provider built by `providers_from_document` whose clock starts `shift`
seconds after the model's time origin, on the model's clock: a sample at model
time `t` is the inner provider's at `t - shift`, and each of its refresh times
`τ` is `τ + shift` in model time. Every capability of the provider interface
is forwarded, as EarthSciAST's own wrappers do.
"""
struct ShiftedProvider
    inner::Any
    shift::Float64
end
EarthSciAST.provider_refresh_times(p::ShiftedProvider) =
    Float64[τ + p.shift for τ in EarthSciAST.provider_refresh_times(p.inner)]
EarthSciAST.provider_is_const(p::ShiftedProvider) = EarthSciAST.provider_is_const(p.inner)
EarthSciAST.provider_supports_selection(p::ShiftedProvider) = EarthSciAST.provider_supports_selection(p.inner)
EarthSciAST.provider_gate_spec(p::ShiftedProvider) = EarthSciAST.provider_gate_spec(p.inner)
EarthSciAST.provider_extent_metaparameter(p::ShiftedProvider) = EarthSciAST.provider_extent_metaparameter(p.inner)
EarthSciAST.provider_sample(p::ShiftedProvider, t::Real; selection = nothing) = selection === nothing ?
    EarthSciAST.provider_sample(p.inner, t - p.shift) :
    EarthSciAST.provider_sample(p.inner, t - p.shift; selection)

"""
    model_time_origin(file) -> Float64

The UTC instant of model time zero, in Unix seconds (D74): EarthSciAST's
`time_origin`, the POSIX epoch when the model declares none. An origin that
cannot be read refuses the run.
"""
function model_time_origin(file::EarthSciAST.EsmFile)
    return try
        Float64(EarthSciAST.time_origin(file))
    catch e
        e isa InterruptException && rethrow()
        throw(ConfigError("the plan's model has no readable time origin (time_origin_invalid): " * sprint(showerror, e)))
    end
end

"""
    provider_clock_origin(file, origin, window_start) -> Float64

The UTC instant, in Unix seconds, that the providers `providers_from_document`
builds count their time from: by EarthSciAST's rule, the model's
`domain.temporal.reference_time`, else its `start`, else the start of the run
window. It is the model's time origin `origin` when the model declares one,
and `window_start` when it does not.
"""
function provider_clock_origin(file::EarthSciAST.EsmFile, origin::Float64, window_start::Float64)
    d = file.domain
    temporal = d === nothing ? nothing : d.temporal
    if temporal isa AbstractDict
        for key in ("reference_time", "start")
            v = get(temporal, key, nothing)
            (v === nothing || v == "") || return origin
        end
    end
    return window_start
end

"""
    forcing_shapes(file, forcing) -> Dict{String, Vector{Int}}

The extents of each data-fed parameter of the plan's `forcing`, from its
index-set names and the flattened model `file`'s index sets (synthetic lift
sets included).
"""
function forcing_shapes(file::EarthSciAST.EsmFile, pf::AbstractDict)
    sizes = Dict{String, Int}(k => Int(s.size) for (k, s) in EarthSciAST.flatten(file).index_sets)
    return Dict{String, Vector{Int}}(
        String(n) => Int[
            get(sizes, X) do
                throw(ConfigError("data-fed parameter $n: its shape names index set $X, which the plan's model does not declare"))
            end for X in v["shape"]
        ] for (n, v) in pf
    )
end

"""
    forcing_runtime(file, plan, tspan, readers; cache_root, url_overrides) -> ForcingRuntime

The data layer of the plan whose model, loaded, is `file` (spec/operator-interface.md
§8.4.2): providers from EarthSciAST's `providers_from_document`, built for the
sources of the parameters of the plan's `forcing` only; `readers` the names
some operator reads (the union of the operators' `forcing`). The const
buffers and the first and last epochs' are read here, so missing data refuses
the run before its first step.
"""
function forcing_runtime(
        file::EarthSciAST.EsmFile, plan::AbstractDict, tspan::Tuple{Float64, Float64}, readers::Vector{String};
        cache_root::AbstractString, url_overrides::AbstractDict = Dict{String, String}()
    )
    pf = plan["forcing"]
    # Model time t is UTC origin + t (D74). EarthSciIO's window is half-open,
    # [start, stop), and a refresh time at t_f starts an epoch (§8.4.3), so
    # the window ends one second after t_f (EarthSciIO reads it as a
    # DateTime, to the millisecond); the epochs are the refresh times in
    # (t₀, t_f].
    origin = model_time_origin(file)
    window = (origin + tspan[1], origin + tspan[2] + 1.0)
    # The providers count time from their own clock origin, which is not the
    # model's when the model declares none (it is then the window's start):
    # every time passed to or read from them is shifted by the difference.
    shift = provider_clock_origin(file, origin, window[1]) - origin
    # Only the sources of the plan's data-fed parameters are built (§8.4.2
    # item 2): a source that only unused parameters read is never built,
    # fetched or checked against the run window.
    loaders = sort!(unique!(String[String(v["source"]) for v in values(pf)]))
    all_provs = try
        EarthSciAST.providers_from_document(file; cache_root = String(cache_root), loaders, url_overrides, window)
    catch e
        e isa InterruptException && rethrow()
        throw(ConfigError("building the providers of the plan's model failed: " * sprint(showerror, e)))
    end
    shapes = forcing_shapes(file, pf)
    providers = Dict{String, Any}()
    for n in keys(pf)
        haskey(all_provs, n) || throw(ConfigError("data-fed parameter $n: EarthSciAST built no provider for it from the plan's model"))
        providers[n] = shift == 0 ? all_provs[n] : ShiftedProvider(all_provs[n], shift)
    end
    discrete = Set{String}(n for n in readers if !isempty(EarthSciAST.provider_refresh_times(providers[n])))
    t0, tf = tspan
    epochs = Float64[t0]
    for n in discrete, τ in EarthSciAST.provider_refresh_times(providers[n])
        t0 < τ <= tf && push!(epochs, Float64(τ))
    end
    sort!(unique!(epochs))
    records = Dict{String, Int}(n => Int(pf[n]["records_per_sample"]) for n in readers)
    F = ForcingRuntime(
        sort(readers), Dict(n => shapes[n] for n in readers), providers, discrete, records, epochs,
        Dict{String, Vector{Float64}}(), Dict{Int, Dict{String, Vector{Float64}}}(), Int[], ReentrantLock(), Tuple{Float64, String}[]
    )
    for n in readers
        n in discrete || (F.const_bufs[n] = sample_buffer(F, n, t0))
    end
    epoch_buffers(F, 1)
    length(epochs) > 1 && epoch_buffers(F, length(epochs))
    return F
end
