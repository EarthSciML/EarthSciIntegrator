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
    report::Vector{Any}            # what was read: (epoch start, parameter, provider record time)
end

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
    forcing_set_for(F, prog, t, cache) -> ForcingSet

The forcing set of program `prog` for a call at `t` (§8.4.1): the buffers its
slots name, from the epoch `t` selects. `cache` (a `Ref`) holds the program's
set of the last epoch it was asked for, so a set is built once per epoch and
program (a new set at each boundary, never an in-place write, §8.4.3).
"""
function forcing_set_for(F::ForcingRuntime, prog, t::Real, cache::Base.RefValue)
    kt = epoch_of(F, t)                    # two-record parameters
    ks = epoch_of(F, step_start(t))        # one-record parameters: the step's epoch
    key = (ks, kt)
    c = cache[]
    c !== nothing && c[1] == key && return c[2]
    bt = epoch_buffers(F, kt)
    bs = ks == kt ? bt : epoch_buffers(F, ks)
    fs = EarthSciAST.forcing_set(
        prog, [F.records[s.name] == 2 ? bt[s.name] : bs[s.name] for s in EarthSciAST.forcing_slots(prog)]
    )
    cache[] = (key, fs)
    return fs
end

"""Whether `t` starts a forcing epoch other than the first (§8.4.1, D88)."""
is_epoch_start(F::ForcingRuntime, t::Real) = length(F.epochs) > 1 && insorted(Float64(t), @view F.epochs[2:end])

"""
    model_time_origin(file) -> Union{Float64, Nothing}

The UTC instant of model time zero, in Unix seconds (D74), when the model's
`domain.temporal` names one; `nothing` otherwise.
"""
function model_time_origin(file::EarthSciAST.EsmFile)
    d = file.domain
    d === nothing && return nothing
    return try
        EarthSciAST.time_origin(file)
    catch
        nothing
    end
end

"""
    forcing_runtime(file, plan, tspan, readers; cache_root, url_overrides) -> ForcingRuntime

The data layer of the plan whose model, loaded, is `file` (spec/operator-interface.md
§8.4.2): providers from EarthSciAST's `providers_from_document`, kept for the
parameters of the plan's `forcing`; `readers` the names some operator reads
(the union of the operators' `forcing`). The const buffers and the first
and last epochs' are read here, so missing data refuses the run before its
first step.
"""
function forcing_runtime(
        file::EarthSciAST.EsmFile, plan::AbstractDict, tspan::Tuple{Float64, Float64}, readers::Vector{String};
        cache_root::AbstractString, url_overrides::AbstractDict = Dict{String, String}()
    )
    pf = plan["forcing"]
    # Shapes are the flattened model's index sets, synthetic lift sets included.
    sizes = Dict{String, Int}(k => Int(s.size) for (k, s) in EarthSciAST.flatten(file).index_sets)
    origin = model_time_origin(file)
    # EarthSciIO's window is half-open, [start, stop), and a refresh time at
    # t_f starts an epoch (§8.4.3), so the window ends one second after t_f
    # (EarthSciIO reads it as a DateTime, to the millisecond); the epochs are
    # the refresh times in (t₀, t_f].
    window = origin === nothing ? nothing : (origin + tspan[1], origin + tspan[2] + 1.0)
    all_provs = try
        EarthSciAST.providers_from_document(file; cache_root = String(cache_root), url_overrides, window)
    catch e
        e isa InterruptException && rethrow()
        throw(ConfigError("building the providers of the plan's model failed: " * sprint(showerror, e)))
    end
    shapes = Dict{String, Vector{Int}}()
    for (n, v) in pf
        shapes[n] = Int[
            get(sizes, X) do
                throw(ConfigError("data-fed parameter $n: its shape names index set $X, which the plan's model does not declare"))
            end for X in v["shape"]
        ]
    end
    providers = Dict{String, Any}()
    for n in keys(pf)
        haskey(all_provs, n) || throw(ConfigError("data-fed parameter $n: EarthSciAST built no provider for it from the plan's model"))
        providers[n] = all_provs[n]
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
        Dict{String, Vector{Float64}}(), Dict{Int, Dict{String, Vector{Float64}}}(), Int[], ReentrantLock(), Any[]
    )
    for n in readers
        n in discrete || (F.const_bufs[n] = sample_buffer(F, n, t0))
    end
    epoch_buffers(F, 1)
    length(epochs) > 1 && epoch_buffers(F, length(epochs))
    return F
end
