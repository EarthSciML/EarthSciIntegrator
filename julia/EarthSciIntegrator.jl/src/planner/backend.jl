# The planner's EarthSciASTDiff backend (M2 stage 2): term bands (spec/plan.md
# §5.2.2), operator structure (§5.4), the Gershgorin bound (§5.4, §5.5.2), the
# derived documents (§6) and validation (§8).

"""
    DerivedDocument

A derived document as the planner holds it: its JSON object, its bytes
(`stable_json`, §2.2), and the document loaded and flattened by EarthSciAST.
"""
struct DerivedDocument
    doc::AbstractDict{String, Any}
    bytes::String
    file::EarthSciAST.EsmFile
    flat::EarthSciAST.FlattenedSystem
end

function DerivedDocument(doc::AbstractDict{String, Any})
    bytes = document_bytes(doc)
    file = EarthSciAST.load_string(bytes)
    return DerivedDocument(doc, bytes, file, EarthSciAST.flatten(file))
end

"""
    document_bands(flat, what; coefficients = true) -> Vector{EarthSciASTDiff.JacEntry}

EarthSciASTDiff's state Jacobian of a right-hand-side document, loaded,
flattened (`flat`) and viewed with `sysview` (§5.2.2, §5.4). With
`coefficients = false`, its structure-only entries (esm-jacobian-spec §2.1),
which carry no coefficient: the pattern pass 2 reads (§5.2.2). An
EarthSciASTDiff refusal is `band_model_failed` at `/model`, naming `what`.
"""
function document_bands(flat::EarthSciAST.FlattenedSystem, what::AbstractString; coefficients::Bool = true)
    try
        return EarthSciASTDiff.jacobian_bands(EarthSciASTDiff.sysview(flat); wrt = :states, coefficients)
    catch e
        e isa PlannerRefusal && rethrow()
        refuse("band_model_failed", "/model", "EarthSciASTDiff refused to differentiate $what: " * sprint(showerror, e))
    end
end

"""A band as `serialize_band` writes it, without `coef` (§5.4)."""
function band_json(en)
    d = EarthSciASTDiff.serialize_band(en)
    return Dict{String, Any}(String(k) => v for (k, v) in d if k != "coef")
end

"""
    BandEntry(en::EarthSciASTDiff.JacEntry, row = en.u)

The [`BandEntry`](@ref) of `BandEntry(band_json(en))`, with row `row`. Its
coefficient, which pass 2 never reads (§5.2.2; pass 2's entries are
structure-only and have none), is replaced by the literal 0 before
`serialize_band` writes the entry, so a full entry's coefficient is never
serialized; every other field is read through `serialize_band`, as for the
operator's bands.
"""
function BandEntry(en::EarthSciASTDiff.JacEntry, row::AbstractString = en.u)
    # `_rebuild` copies every field of the band but the ones it is given, so a
    # change to `Band`'s fields fails here rather than diverging silently.
    band = EarthSciASTDiff._rebuild(en.band; coef = EarthSciAST.IntExpr(0))
    d = band_json(EarthSciASTDiff.JacEntry(en.u, en.v, band))
    d["row"] = String(row)
    return BandEntry(d)
end

"""An operator's artifacts: its right-hand-side document, its bands (as EarthSciASTDiff's entries, `serialize_band`'s dictionaries without `coef`, and [`BandEntry`](@ref)s), and, once needed, its Jacobian document and the built documents."""
mutable struct OperatorArtifacts
    rhs::DerivedDocument
    entries::Vector{EarthSciASTDiff.JacEntry}
    band_dicts::Vector{Dict{String, Any}}
    bands::Vector{BandEntry}
    sv::EarthSciASTDiff.SysView
    jac::Union{Nothing, DerivedDocument}
    rhs_built::Union{Nothing, BuiltDocument}
    jac_built::Union{Nothing, BuiltDocument}
    points::Union{Nothing, Vector{NTuple{4, Int}}}
end

"""
    DiffBackend()

The planner's backend on EarthSciASTDiff (M2 stage 2), holding what one
planning run derives: per operator its documents, bands and built
evaluators, the flattened model's evaluator and the sampled states, and,
after validation, the band values (§8.2) and measured errors for
`plan.report.json`. All of it belongs to the planning context `ctx` it was
derived from; a backend passed to `plan` again starts afresh.
"""
mutable struct DiffBackend <: PlanBackend
    ctx::Any
    ops::Dict{String, OperatorArtifacts}
    layout::Union{Nothing, DofLayout}
    model::Union{Nothing, BuiltDocument}
    samples::Union{Nothing, Vector{Tuple{Float64, Vector{Float64}}}}
    forcing::Union{Nothing, Vector{Dict{String, Vector{Float64}}}}
    bandvalues::Dict{String, Any}
    report::Dict{String, Any}
end
DiffBackend() = DiffBackend(nothing, Dict{String, OperatorArtifacts}(), nothing, nothing, nothing, nothing, Dict{String, Any}(), Dict{String, Any}())

"""Bind `b` to the planning context `ctx`, dropping everything derived from another one."""
function bind!(b::DiffBackend, ctx)
    b.ctx === ctx && return b
    b.ctx = ctx
    empty!(b.ops)
    b.layout = b.model = b.samples = b.forcing = nothing
    empty!(b.bandvalues)
    empty!(b.report)
    return b
end

"""
    term_entries(b::DiffBackend, ctx, t; coefficients = false) -> Vector{EarthSciASTDiff.JacEntry}

Term `t`'s EarthSciASTDiff entries from its own right-hand-side document
(§5.2.2): structure-only by default, the full mode with `coefficients =
true`. An EarthSciASTDiff failure is `band_model_failed`, naming the term.
"""
function term_entries(b::DiffBackend, ctx, t::Term; coefficients::Bool = false)
    bind!(b, ctx)
    # A term's document is never written, so it is loaded from memory rather
    # than through its bytes.
    flat = EarthSciAST.flatten(EarthSciAST.load_document(rhs_document(ctx.model, "term", [t])))
    return document_bands(flat, "term ($(t.variable), $(t.contribution), $(t.summand))"; coefficients)
end

"""
    term_band_dicts(b::DiffBackend, ctx, t; coefficients = false) -> Vector{Dict{String, Any}}

[`term_entries`](@ref) as [`band_json`](@ref) gives them.
"""
term_band_dicts(b::DiffBackend, ctx, t::Term; coefficients::Bool = false) =
    Dict{String, Any}[band_json(en) for en in term_entries(b, ctx, t; coefficients)]

term_bands(b::DiffBackend, ctx, t::Term; coefficients::Bool = false) =
    BandEntry[BandEntry(en) for en in term_entries(b, ctx, t; coefficients)]

"""
    probe_band_dicts(b::DiffBackend, ctx, terms) -> Union{Vector{Vector{Dict{String, Any}}}, Nothing}

Every term's bands, as [`band_json`](@ref) gives them, from one probe
document (§5.2.2, [`probe_document`](@ref)): one EarthSciAST load and
flatten and one `jacobian_bands` call, whose bands with row
`probe_state(k)` are term `k`'s, row renamed to its variable. `nothing` when
EarthSciAST or EarthSciASTDiff fails on the probe document, a failure that
names no term. Only those calls are guarded: an error in building the probe
document or in reading its bands is the planner's own and propagates.
"""
function probe_band_dicts(b::DiffBackend, ctx, terms::AbstractVector{Term}; coefficients::Bool = false)
    return _probe_bands(b, ctx, terms, Dict{String, Any}; coefficients) do en, row
        d = band_json(en)
        d["row"] = row
        d
    end
end

"""
    probe_term_bands(b::DiffBackend, ctx, terms) -> Union{Vector{Vector{BandEntry}}, Nothing}

[`probe_band_dicts`](@ref) as [`BandEntry`](@ref)s, read from EarthSciASTDiff's
structure-only entries. Nothing of the probe is kept once it returns: its
flattened document and its entries are garbage.
"""
probe_term_bands(b::DiffBackend, ctx, terms::AbstractVector{Term}; coefficients::Bool = false) =
    _probe_bands(BandEntry, b, ctx, terms, BandEntry; coefficients)

"""
    _probe_bands(convert, b, ctx, terms, T; coefficients = false) -> Union{Vector{Vector{T}}, Nothing}

Every term's bands from the probe document of `terms`, each entry turned into
a `T` by `convert(entry, row)`, `row` the term's variable; `nothing` when
EarthSciAST or EarthSciASTDiff fails on the probe document.
"""
function _probe_bands(
        convert, b::DiffBackend, ctx, terms::AbstractVector{Term}, ::Type{T}; coefficients::Bool = false
    ) where {T}
    entries = _probe_entries(b, ctx, terms; coefficients)
    entries === nothing && return nothing
    out = [T[] for _ in terms]
    index = Dict(probe_state(k) => k for k in eachindex(terms))
    for en in entries
        # Only the probe states have nonzero right-hand sides.
        k = get(index, en.u, nothing)
        k === nothing && error("probe document: a band whose row $(en.u) is not a probe state")
        push!(out[k], convert(en, terms[k].variable))
    end
    return out
end

"""
The probe document's entries (EarthSciASTDiff's `JacEntry`s), or `nothing`
when EarthSciAST or EarthSciASTDiff fails on it. Structure-only entries by
default, the full mode's with `coefficients = true`.
"""
function _probe_entries(b::DiffBackend, ctx, terms::AbstractVector{Term}; coefficients::Bool = false)
    bind!(b, ctx)
    doc = probe_document(ctx.model, terms)
    return try
        document_bands(EarthSciAST.flatten(EarthSciAST.load_document(doc)), "the probe document"; coefficients)
    catch e
        e isa InterruptException && rethrow()
        e isa PlannerRefusal && e.code != "band_model_failed" && rethrow()
        @debug "The probe document of $(length(terms)) terms failed; splitting it (§5.2.2)" exception = e
        nothing
    end
end

"""
    LARGE_TERM_NODES

The most operator nodes, every occurrence counted, that a probe batch's terms
hold in all, unless the batch is a single term (§5.2.2). A term with more
(a **large** term) is therefore a batch by itself.
"""
const LARGE_TERM_NODES = 10_000

"""
    operator_nodes(e, limit) -> Int

The number of operator nodes of `e`, every occurrence counted, in every
Expression-valued child, or `limit + 1` when it exceeds `limit`. A node
reached through several parents is counted once per path but walked once.
"""
function operator_nodes(e, limit::Int)
    memo = IdDict{EarthSciAST.OpExpr, Int}()
    return min(_operator_nodes(e, memo, limit), limit + 1)
end

function _operator_nodes(e, memo::IdDict{EarthSciAST.OpExpr, Int}, limit::Int)
    e isa EarthSciAST.OpExpr || return 0
    n = get(memo, e, nothing)
    n === nothing || return n
    c = 1
    # `foreach_child` visits every Expression-valued field without allocating;
    # once the count passes `limit`, the remaining children are not walked.
    EarthSciAST.foreach_child(e) do x
        c > limit || (c = min(c + _operator_nodes(x, memo, limit), limit + 1))
        nothing
    end
    return memo[e] = c
end

"""
    probe_batches(terms; large = LARGE_TERM_NODES) -> Vector{UnitRange{Int}}

§5.2.2's batches: consecutive runs of `terms`, in term order, whose terms hold
at most `large` operator nodes in all, unless the run is a single term. A
batch ends just before a term that would take its total over `large`, so a
large term (more than `large` operator nodes) is a batch by itself.
"""
function probe_batches(terms::AbstractVector{Term}; large::Int = LARGE_TERM_NODES)
    out = UnitRange{Int}[]
    start, total = 1, 0
    for (k, t) in enumerate(terms)
        n = operator_nodes(t.expr, large)
        if k > start && total + n > large
            push!(out, start:(k - 1))
            start, total = k, 0
        end
        total += n
    end
    isempty(terms) || push!(out, start:length(terms))
    return out
end

"""
    all_term_bands(b::DiffBackend, ctx, terms; large = LARGE_TERM_NODES, coefficients = false) -> Vector{Vector{BandEntry}}

Every term's bands, EarthSciASTDiff's structure-only entries (§5.2.2; the
full mode's with `coefficients = true`, which only tests use), from one probe
document per batch ([`probe_batches`](@ref), [`probe_term_bands`](@ref)), in
batch order. Nothing of a batch's probe is kept once its bands are read, so
it is garbage before the next batch is built.
A failure on a batch's probe names no term, so that batch is then split in
halves and each half probed in turn, first half first, down to single terms,
whose own documents are differentiated ([`term_bands`](@ref)). The first term
whose own document fails, in term order, refuses with `band_model_failed`
and that document's message (§5.2.2): the batches are consecutive and taken
in order, so it is the first such term of the whole list.
"""
function all_term_bands(
        b::DiffBackend, ctx, terms::AbstractVector{Term}; large::Int = LARGE_TERM_NODES, coefficients::Bool = false
    )
    bind!(b, ctx)
    out = Vector{Vector{BandEntry}}(undef, length(terms))
    for r in probe_batches(terms; large)
        _term_bands!(out, b, ctx, terms, r; coefficients)
    end
    return out
end

"""
    _term_bands!(out, b, ctx, terms, r; coefficients = false)

The bands of `terms[r]` into `out[r]`: one probe of the range, and when it
fails, the first half then the second, recursively, down to single terms,
whose own documents are differentiated ([`term_bands`](@ref)), so the first
term in `r` whose own document fails refuses. `coefficients` picks the mode.
"""
function _term_bands!(out, b::DiffBackend, ctx, terms, r::UnitRange{Int}; coefficients::Bool = false)
    isempty(r) && return out
    if length(r) == 1
        out[only(r)] = term_bands(b, ctx, terms[only(r)]; coefficients)
        return out
    end
    probe = probe_term_bands(b, ctx, terms[r]; coefficients)
    if probe !== nothing
        out[r] = probe
        return out
    end
    mid = first(r) + length(r) ÷ 2 - 1
    _term_bands!(out, b, ctx, terms, first(r):mid; coefficients)
    _term_bands!(out, b, ctx, terms, (mid + 1):last(r); coefficients)
    return out
end

"""
    refuse_failing_term(b::DiffBackend, ctx, terms)

Pass 4's search (§5.2.2), when an operator's own document fails in the full
mode: pass 2's search ([`all_term_bands`](@ref): the same batches, each
probed whole and, when it fails, split in halves, first half first, down to
single terms) in the full mode over the operator's `terms`, in term order.
The first term whose own document fails refuses, with that document's
message; that takes about `2 log2(n)` probe documents rather than one per
term. Returns `nothing` when no term's own document fails, or when the
search meets an error that is no `band_model_failed` refusal (an EarthSciAST
failure on a term's own document), so the caller refuses against the
operator.
"""
function refuse_failing_term(b::DiffBackend, ctx, terms::AbstractVector{Term})
    try
        all_term_bands(b, ctx, terms; coefficients = true)
    catch e
        e isa InterruptException && rethrow()
        e isa PlannerRefusal && e.code == "band_model_failed" && rethrow()
        @debug "Pass 4's search for the failing term met an error; refusing against the operator (§5.2.2)" exception = e
    end
    return nothing
end

"""The operator's artifacts, building its right-hand-side document and bands on first use."""
function artifacts!(b::DiffBackend, ctx, op::PlanOperator)
    bind!(b, ctx)
    haskey(b.ops, op.name) && return b.ops[op.name]
    d = DerivedDocument(rhs_document(ctx.model, op.name, op.terms))
    entries = try
        document_bands(d.flat, "operator $(op.name)")
    catch e
        (e isa PlannerRefusal && e.code == "band_model_failed") || rethrow()
        # Pass 2 read the terms' structure only; a term whose full mode
        # fails where its structure did not is reported here, against the first
        # such term of the operator, with its own document's message, and
        # against the operator when there is none (§5.2.2).
        refuse_failing_term(b, ctx, op.terms)
        throw(e)
    end
    dicts = Dict{String, Any}[band_json(en) for en in entries]
    a = OperatorArtifacts(
        d, entries, dicts, BandEntry[BandEntry(x) for x in dicts], EarthSciASTDiff.sysview(d.flat),
        nothing, nothing, nothing, nothing
    )
    return b.ops[op.name] = a
end

"""The operator's Jacobian document (§6.2), built on first use."""
function jacobian_document!(b::DiffBackend, ctx, op::PlanOperator)
    a = artifacts!(b, ctx, op)
    # §6.2 item 5: the right-hand-side document's stubs, so the data-fed
    # parameters keep their data rules (EarthSciASTDiff#55).
    a.jac === nothing && (a.jac = DerivedDocument(jacobian_document(a.rhs.flat, a.entries; data_sources = a.rhs.file.data_sources)))
    return a.jac
end

layout!(b::DiffBackend, ctx) = (bind!(b, ctx); b.layout === nothing ? (b.layout = DofLayout(ctx.model)) : b.layout)

"""The flattened model built with EarthSciAST, whose `u0` is §4's initial state; built on first use."""
function model_built!(b::DiffBackend, ctx)
    bind!(b, ctx)
    b.model === nothing || return b.model
    return b.model = build_document(model_with_forcing_defaults(ctx.model), layout!(b, ctx), ctx.time)
end

"""
The flattened model with every unused data-fed parameter's `default` set to
1 (§4: the planner's build binds each used one to its `default`, nonzero by
§4.1, and each unused one to 1, which nothing reads).
"""
function model_with_forcing_defaults(m::PlanModel)
    flat = m.flat
    any(n -> !(n in m.used_forcing), keys(m.forcing)) || return flat
    ps = copy(flat.parameters)
    for n in keys(m.forcing)
        n in m.used_forcing && continue
        v = ps[n]
        ps[n] = EarthSciAST.ModelVariable(
            v.type; default = 1.0, description = v.description, units = v.units, default_units = v.default_units,
            shape = v.shape, location = v.location, distribution = v.distribution, update = v.update,
            element_type = v.element_type
        )
    end
    return EarthSciAST.FlattenedSystem(flat; parameters = ps)
end

"""
§5.5.1's sampled forcing values, one dictionary per sampled state: every used
data-fed parameter's values by name, and every unused one's, all 1 (the
flattened model's program has a slot for each; nothing reads an unused one).
Empty dictionaries for a model without data-fed parameters.
"""
function forcing_samples!(b::DiffBackend, ctx)
    bind!(b, ctx)
    b.forcing === nothing || return b.forcing
    m = ctx.model
    used = used_forcing(m)
    F = sampled_forcing(used, Dict(n => default_array(m.flat, n) for n in used))
    for n in keys(m.forcing), f in F
        n in m.used_forcing || (f[n] = ones(length(default_array(model_with_forcing_defaults(m), n))))
    end
    return b.forcing = F
end

"""The forcing `F` of sampled state `s` a document is given (§8): `nothing` when the model has no data-fed parameters."""
sample_forcing(b::DiffBackend, ctx, s::Int) = isempty(ctx.model.forcing) ? nothing : forcing_samples!(b, ctx)[s]

"""The three sampled states of §5.5.1, from the flattened model's initial state."""
function samples!(b::DiffBackend, ctx)
    bind!(b, ctx)
    b.samples === nothing || return b.samples
    B = model_built!(b, ctx)
    return b.samples = sampled_states(B.u0[B.perm], ctx.time)
end

function rhs_built!(b::DiffBackend, ctx, op::PlanOperator)
    a = artifacts!(b, ctx, op)
    a.rhs_built === nothing && (a.rhs_built = build_document(a.rhs.file, layout!(b, ctx), ctx.time))
    return a.rhs_built
end

function jac_built!(b::DiffBackend, ctx, op::PlanOperator)
    a = artifacts!(b, ctx, op)
    if a.jac_built === nothing
        L = layout!(b, ctx)
        fields = [vcat(L.extents[x.row], Int[hi for (_, _, hi) in x.contracted]) for x in a.bands]
        a.jac_built = build_document(jacobian_document!(b, ctx, op).file, L, ctx.time; fields, u0 = jacobian_u0(L.names))
        a.points = band_points(L, a.bands)
    end
    return a.jac_built
end

# ---------------------------------------------------------------------------
# Pass 4: structure (§5.4)

"""`true` when no band coefficient references a state, or an observed whose definition references a state, transitively (§5.4)."""
function linear_in_u(m::PlanModel, entries)
    names = Set{String}(n for en in entries for n in EarthSciAST.free_variables(en.band.coef))
    any(n -> is_state(m, n), names) && return false
    for o in live_observeds(m, names), r in get(m.observed_rhs, o, ASTExpr[])
        any(n -> is_state(m, n), EarthSciAST.free_variables(r)) && return false
    end
    return true
end

"""`true` when the operator's right-hand-side document references `t` in a state equation or in the definition of an observed it keeps (§5.4)."""
function depends_on_t(m::PlanModel, op::PlanOperator)
    any(t -> "t" in EarthSciAST.free_variables(t.expr), op.terms) && return true
    for o in live_observeds(m, reduce(vcat, (t.refs for t in op.terms); init = String[])), r in get(m.observed_rhs, o, ASTExpr[])
        "t" in EarthSciAST.free_variables(r) && return true
    end
    return false
end

"""`column`'s `kl` and `ku` (§5.4): the largest sub- and super-diagonal vertical offsets over all bands; a band whose reach covers the vertical axis makes both `N_v − 1`."""
function column_bandwidth(bands, m::PlanModel, axes::Axes)
    v = axes.vertical
    kl = ku = 0
    for b in bands
        r = band_reach(b, m.shapes, axes, m.sizes)
        if v in r.covers
            kl = max(kl, m.sizes[v] - 1)
            ku = max(ku, m.sizes[v] - 1)
        end
        for o in get(r.offsets, v, Int[])
            kl = max(kl, -o)
            ku = max(ku, o)
        end
    end
    return kl, ku
end

"""`stencil.footprint` (§5.4): the distinct vectors of reduced horizontal offsets over all bands, one integer per horizontal axis (0 for an axis without an offset in the band), sorted lexicographically."""
function stencil_footprint(bands, m::PlanModel, axes::Axes)
    vecs = Set{Vector{Int}}()
    for b in bands
        r = band_reach(b, m.shapes, axes, m.sizes)
        per = [sort!(unique(get(r.offsets, X, [0]))) for X in axes.horizontal]
        any(isempty, per) && continue
        for c in Iterators.product(per...)
            push!(vecs, collect(c))
        end
    end
    return Any[Any[v...] for v in sort!(collect(vecs))]
end

"""
    check_full_locality(op, bands, m, axes)

The class and radius the operator's full band list implies must be no wider
than its locality, read from its terms' structure-only band lists (§5.2.2:
the full mode's points are a subset of the structure-only mode's). A wider
one is an internal planner error, never a refusal; a narrower one is allowed.
"""
function check_full_locality(op::PlanOperator, bands, m::PlanModel, axes::Axes)
    cls, rad = classify_bands(bands, m.shapes, axes, m.sizes)
    wider = coarser(op.locality, cls) !== op.locality
    if !wider && cls === :stencil
        wider = any(X -> rad[X][1] > op.radius[X][1] || rad[X][2] > op.radius[X][2], axes.horizontal)
    end
    wider && error(
        "operator $(op.name): its full band list implies locality $cls" * (rad === nothing ? "" : " with radius $rad") *
            ", wider than $(op.locality)" * (op.radius === nothing ? "" : " with radius $(op.radius)") *
            " from its terms' structure-only band lists (spec/plan.md §5.2.2: the full mode's points are a subset of the structure-only mode's)"
    )
    return nothing
end

function operator_structure(b::DiffBackend, ctx, op::PlanOperator)
    a = artifacts!(b, ctx, op)
    m, axes = ctx.model, ctx.axes
    cls = EarthSciASTDiff.detect_structure(a.entries, a.sv)
    s = Dict{String, Any}(
        "class" => String(cls),
        "linear_in_u" => linear_in_u(m, a.entries),
        "depends_on_t" => depends_on_t(m, op),
        "bands" => Any[a.band_dicts...]
    )
    loc = need_locality(op)
    check_full_locality(op, a.bands, m, axes)
    # The footprint and the bandwidth read the pattern the locality and radius
    # were read from: the operator's terms' band lists (§5.2.2, §5.4).
    pattern() = BandEntry[x for t in op.terms for x in something(t.bands, BandEntry[])]
    if loc === :pointwise && cls === :block_diagonal
        p = EarthSciASTDiff.serialize_plan(EarthSciASTDiff.plan_factorization(a.entries, a.sv; structure = cls))
        s["block"] = Dict{String, Any}(
            "vars" => Any[p["block_vars"]...], "ordering" => Any[p["ordering"]...],
            "pattern" => p["pattern"], "lu_pattern" => p["lu_pattern"]
        )
    elseif loc === :column
        kl, ku = column_bandwidth(pattern(), m, axes)
        s["column"] = Dict{String, Any}("axis" => axes.vertical, "kl" => kl, "ku" => ku)
    elseif loc === :stencil
        s["stencil"] = Dict{String, Any}("footprint" => stencil_footprint(pattern(), m, axes))
    end
    return s
end

# ---------------------------------------------------------------------------
# Pass 5: the Gershgorin bound (§5.4, §5.5.2)

"""Whether a band coefficient of the operator references a data-fed parameter, directly or through an observed's definition, transitively (§5.5.2 rule 4)."""
function jacobian_reads_forcing(b::DiffBackend, ctx, op::PlanOperator)
    m = ctx.model
    isempty(m.forcing) && return false
    a = artifacts!(b, ctx, op)
    names = Set{String}(n for en in a.entries for n in EarthSciAST.free_variables(en.band.coef))
    any(n -> is_forcing(m, n), names) && return true
    for o in live_observeds(m, names), r in get(m.observed_rhs, o, ASTExpr[])
        any(n -> is_forcing(m, n), EarthSciAST.free_variables(r)) && return true
    end
    return false
end

function gershgorin_bound(b::DiffBackend, ctx, op::PlanOperator)
    B = jac_built!(b, ctx, op)
    a = b.ops[op.name]
    rows = state_dofs(layout!(b, ctx), op.writes)
    ρ = maximum(
        gershgorin(band_jacobian(B, a.points, u, t, sample_forcing(b, ctx, s)), rows) for (s, (t, u)) in enumerate(samples!(b, ctx))
    )
    get!(b.report, "gershgorin", Dict{String, Any}())[op.name] = ρ
    return ρ
end

# ---------------------------------------------------------------------------
# Passes 6 and 8: derived documents and validation (§6, §7.7, §8)

"""The relative error measure of `rhs_sum` at one DOF (§8.1), 0 for an exact zero."""
function _rhs_sum_error(f, fs)
    s = sum(fs)
    scale = max(abs(f), sum(abs, fs))
    d = abs(s - f)
    return d == 0 ? 0.0 : scale == 0 ? Inf : d / scale
end

function derived_and_validation(b::DiffBackend, ctx)
    m = ctx.model
    names = sort!(collect(keys(ctx.ops)))
    derived = Dict{String, Any}()
    for o in names
        op = ctx.ops[o]
        for (kind, d) in (("rhs", artifacts!(b, ctx, op).rhs), ("jacobian", jacobian_document!(b, ctx, op)))
            # SHA's `sha256(::String)` re-hashes the whole string for every
            # block (its copy's alias check takes the string's `objectid`), so a
            # document of megabytes takes minutes; a byte vector does not.
            derived["derived/$o.$kind.esm"] = Dict{String, Any}(
                "kind" => kind, "operator" => o, "sha256" => bytes2hex(sha256(Vector{UInt8}(d.bytes)))
            )
        end
    end
    # §8.1, in the order validation lists the checks; the first failure refuses.
    L = layout!(b, ctx)
    S = samples!(b, ctx)
    M = model_built!(b, ctx)
    worst = 0.0
    for (s, (t, u)) in enumerate(S)
        F = sample_forcing(b, ctx, s)
        f = eval_rhs(M, u, t, F)
        fk = [eval_rhs(rhs_built!(b, ctx, ctx.ops[o]), u, t, F) for o in names]
        for n in eachindex(f)
            e = _rhs_sum_error(f[n], [x[n] for x in fk])
            worst = max(worst, e)
            e <= 1.0e-12 || refuse(
                "validation_rhs_sum_failed", "/model",
                "at t = $t the operators' right-hand sides sum to $(sum(x[n] for x in fk)) at $(L.names[n]), " *
                    "the model's is $(f[n]) (relative error $e > 1e-12)"
            )
        end
    end
    b.report["rhs_sum"] = worst
    validation = Any[Dict{String, Any}("check" => "rhs_sum", "passed" => true)]
    for o in names
        op = ctx.ops[o]
        # `non_empty` runs first, so an operator that reads a state but has
        # no band is `validation_empty_operator` rather than a `band_model`
        # failure; `validation` still lists `band_model` first.
        non_empty_check(b, ctx, op)
        band_model_check!(b, ctx, op)
        push!(validation, Dict{String, Any}("check" => "band_model", "operator" => o, "passed" => true))
        push!(validation, Dict{String, Any}("check" => "non_empty", "operator" => o, "passed" => true))
    end
    return derived, validation
end

"""
    band_model_check!(b, ctx, op)

§8.1 `band_model`, Julia's form (§8.2): at each sampled state, the band
Jacobian from the Jacobian document's fields and the bands against
ForwardDiff's Jacobian of the right-hand-side document, rows of `writes`
only: `|J^B − J^I| ≤ 1e-10 M_r` per entry, and every nonzero of `J^I` in the
band pattern. Keeps the ForwardDiff values on the band pattern for the
band-value file (§8.2). Both Jacobians are held sparsely, row by row; an
entry absent from both is zero in both and passes.
"""
function band_model_check!(b::DiffBackend, ctx, op::PlanOperator)
    L = layout!(b, ctx)
    R = rhs_built!(b, ctx, op)
    B = jac_built!(b, ctx, op)
    a = b.ops[op.name]
    pattern = Set{Tuple{Int, Int}}((r, c) for (_, _, r, c) in a.points)
    rows = state_dofs(L, op.writes)
    worst = 0.0
    samples = Any[]
    jrows = sort!(collect(union(Set(rows), (r for (r, _) in pattern))))
    for (s, (t, u)) in enumerate(samples!(b, ctx))
        F = sample_forcing(b, ctx, s)
        JB = band_jacobian(B, a.points, u, t, F)
        JI = forwarddiff_jacobian(R, u, t, jrows; keep = pattern, F)
        cols = Dict{Int, Set{Int}}(r => Set{Int}() for r in rows)
        for J in (JB, JI), (r, c) in keys(J)
            haskey(cols, r) && push!(cols[r], c)
        end
        for r in rows
            Mr = 0.0
            for c in cols[r]
                Mr = max(Mr, abs(get(JB, (r, c), 0.0)), abs(get(JI, (r, c), 0.0)))
            end
            for c in sort!(collect(cols[r]))
                jb, ji = get(JB, (r, c), 0.0), get(JI, (r, c), 0.0)
                d = abs(jb - ji)
                worst = max(worst, d == 0 ? 0.0 : d / Mr)
                d <= 1.0e-10 * Mr || refuse(
                    "validation_band_model_failed", "/model",
                    "operator $(op.name) at t = $t: band Jacobian entry ($(L.names[r]), $(L.names[c])) is $jb, " *
                        "ForwardDiff's is $ji (row scale $Mr)"
                )
                ji != 0 && !((r, c) in pattern) && refuse(
                    "validation_band_model_failed", "/model",
                    "operator $(op.name) at t = $t: its right-hand side depends on $(L.names[c]) in row $(L.names[r]), " *
                        "which no band covers"
                )
            end
        end
        entries = sort!([(L.names[r], L.names[c], JI[(r, c)]) for (r, c) in pattern]; by = e -> (e[1], e[2]))
        sample = Dict{String, Any}(
            "t" => t, "u" => Dict{String, Any}(L.names[n] => u[n] for n in eachindex(u)),
            "entries" => Any[Any[e...] for e in entries]
        )
        # §8.2: the sampled forcing values at every element of the canonical
        # forcing order (the used data-fed parameters only).
        F === nothing || (sample["forcing"] = forcing_element_values(ctx.model, F))
        push!(samples, sample)
    end
    get!(b.report, "band_model", Dict{String, Any}())[op.name] = worst
    b.bandvalues[op.name] = Dict{String, Any}("samples" => samples)
    return
end

"""§8.1 `non_empty`: the operator has a term, and a band when any of its terms references a state."""
function non_empty_check(b::DiffBackend, ctx, op::PlanOperator)
    isempty(op.terms) && refuse("validation_empty_operator", "/model", "operator $(op.name) has no term")
    reads_state = any(t -> any(n -> is_state(ctx.model, n), t.refs), op.terms)
    reads_state && isempty(b.ops[op.name].entries) && refuse(
        "validation_empty_operator", "/model",
        "operator $(op.name) has no band although its terms reference a state"
    )
    return
end

"""The values of `F` at every element of the used data-fed parameters, by element name (`x[i,j,…]`, a scalar's bare name; §8.2)."""
function forcing_element_values(m::PlanModel, F)
    out = Dict{String, Any}()
    for n in used_forcing(m)
        ext = Int[m.sizes[X] for X in m.forcing[n].shape]
        for (i, c) in enumerate(cells(ext))
            out[element_name(n, c)] = F[n][i]
        end
    end
    return out
end

"""The derived documents' bytes by plan-relative path (`derived/<op>.<kind>.esm`)."""
function derived_files(b::DiffBackend)
    out = Dict{String, String}()
    for (o, a) in b.ops
        out["derived/$o.rhs.esm"] = a.rhs.bytes
        a.jac === nothing || (out["derived/$o.jacobian.esm"] = a.jac.bytes)
    end
    return out
end
derived_files(::PlanBackend) = nothing
