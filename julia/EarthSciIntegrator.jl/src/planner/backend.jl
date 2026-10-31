# The planner's EarthSciASTDiff backend (M2 stage 2): term bands (spec/plan.md
# §5.2.2), operator structure (§5.4), the Gershgorin bound (§5.4, §5.5.2), the
# derived documents (§6) and validation (§8).

"""
    DerivedDocument

A derived document as the planner holds it: its JSON object, its bytes
(`stable_json`, §2.2), and the document loaded and flattened by EarthSciAST.
"""
struct DerivedDocument
    doc::Dict{String, Any}
    bytes::String
    file::EarthSciAST.EsmFile
    flat::EarthSciAST.FlattenedSystem
end

function DerivedDocument(doc::Dict{String, Any})
    bytes = document_bytes(doc)
    file = EarthSciAST.load_string(bytes)
    return DerivedDocument(doc, bytes, file, EarthSciAST.flatten(file))
end

"""
    document_bands(d, what) -> Vector{EarthSciASTDiff.JacEntry}

EarthSciASTDiff's state Jacobian of a right-hand-side document, loaded,
flattened and viewed with `sysview` (§5.2.2, §5.4). An EarthSciASTDiff
refusal is `band_model_failed` at `/model`, naming `what`.
"""
function document_bands(d::DerivedDocument, what::AbstractString)
    try
        return EarthSciASTDiff.jacobian_bands(EarthSciASTDiff.sysview(d.flat); wrt = :states)
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

"""An operator's artifacts: its right-hand-side document and bands, and, once needed, its Jacobian document and the built documents."""
mutable struct OperatorArtifacts
    rhs::DerivedDocument
    entries::Vector{EarthSciASTDiff.JacEntry}
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
`plan.report.json`.
"""
mutable struct DiffBackend <: PlanBackend
    ops::Dict{String, OperatorArtifacts}
    layout::Union{Nothing, DofLayout}
    model::Union{Nothing, BuiltDocument}
    samples::Union{Nothing, Vector{Tuple{Float64, Vector{Float64}}}}
    bandvalues::Dict{String, Any}
    report::Dict{String, Any}
end
DiffBackend() = DiffBackend(Dict{String, OperatorArtifacts}(), nothing, nothing, nothing, Dict{String, Any}(), Dict{String, Any}())

function term_bands(::DiffBackend, ctx, t::Term)
    doc = rhs_document(ctx.model, "term", [t])
    what = "term ($(t.variable), $(t.contribution), $(t.summand))"
    return BandEntry[BandEntry(band_json(en)) for en in document_bands(DerivedDocument(doc), what)]
end

"""The operator's artifacts, building its right-hand-side document and bands on first use."""
function artifacts!(b::DiffBackend, ctx, op::PlanOperator)
    haskey(b.ops, op.name) && return b.ops[op.name]
    d = DerivedDocument(rhs_document(ctx.model, op.name, op.terms))
    entries = document_bands(d, "operator $(op.name)")
    a = OperatorArtifacts(
        d, entries, BandEntry[BandEntry(band_json(en)) for en in entries], EarthSciASTDiff.sysview(d.flat),
        nothing, nothing, nothing, nothing
    )
    return b.ops[op.name] = a
end

"""The operator's Jacobian document (§6.2), built on first use."""
function jacobian_document!(b::DiffBackend, ctx, op::PlanOperator)
    a = artifacts!(b, ctx, op)
    a.jac === nothing && (a.jac = DerivedDocument(rhs_document(ctx.model, op.name, op.terms; jacobian = a.entries)))
    return a.jac
end

layout!(b::DiffBackend, ctx) = b.layout === nothing ? (b.layout = DofLayout(ctx.model)) : b.layout

"""The flattened model built with EarthSciAST, whose `u0` is §4's initial state; built on first use."""
model_built!(b::DiffBackend, ctx) =
    b.model === nothing ? (b.model = build_document(ctx.model.flat, layout!(b, ctx), ctx.time)) : b.model

"""The three sampled states of §5.5.1, from the flattened model's initial state."""
function samples!(b::DiffBackend, ctx)
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
        a.jac_built = build_document(jacobian_document!(b, ctx, op).file, L, ctx.time; fields)
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
    live = live_observeds(m, names)
    for eq in m.flat.equations
        o = defined_observed(m, eq)
        (o !== nothing && o in live) || continue
        any(n -> is_state(m, n), EarthSciAST.free_variables(eq.rhs)) && return false
    end
    return true
end

"""`true` when the operator's right-hand-side document references `t` in a state equation or in the definition of an observed it keeps (§5.4)."""
function depends_on_t(m::PlanModel, op::PlanOperator)
    any(t -> "t" in EarthSciAST.free_variables(t.expr), op.terms) && return true
    live = live_observeds(m, reduce(vcat, (t.refs for t in op.terms); init = String[]))
    for eq in m.flat.equations
        o = defined_observed(m, eq)
        (o !== nothing && o in live) || continue
        "t" in EarthSciAST.free_variables(eq.rhs) && return true
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

function operator_structure(b::DiffBackend, ctx, op::PlanOperator)
    a = artifacts!(b, ctx, op)
    m, axes = ctx.model, ctx.axes
    cls = EarthSciASTDiff.detect_structure(a.entries, a.sv)
    s = Dict{String, Any}(
        "class" => String(cls),
        "linear_in_u" => linear_in_u(m, a.entries),
        "depends_on_t" => depends_on_t(m, op),
        "bands" => Any[band_json(en) for en in a.entries]
    )
    loc = need_locality(op)
    if loc === :pointwise && cls === :block_diagonal
        p = EarthSciASTDiff.serialize_plan(EarthSciASTDiff.plan_factorization(a.entries, a.sv; structure = cls))
        s["block"] = Dict{String, Any}(
            "vars" => Any[p["block_vars"]...], "ordering" => Any[p["ordering"]...],
            "pattern" => p["pattern"], "lu_pattern" => p["lu_pattern"]
        )
    elseif loc === :column
        kl, ku = column_bandwidth(a.bands, m, axes)
        s["column"] = Dict{String, Any}("axis" => axes.vertical, "kl" => kl, "ku" => ku)
    elseif loc === :stencil
        s["stencil"] = Dict{String, Any}("footprint" => stencil_footprint(a.bands, m, axes))
    end
    return s
end

# ---------------------------------------------------------------------------
# Pass 5: the Gershgorin bound (§5.4, §5.5.2)

function gershgorin_bound(b::DiffBackend, ctx, op::PlanOperator)
    B = jac_built!(b, ctx, op)
    a = b.ops[op.name]
    rows = state_dofs(layout!(b, ctx), op.writes)
    ρ = maximum(gershgorin(band_jacobian(B, a.points, u, t), rows) for (t, u) in samples!(b, ctx))
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
            derived["derived/$o.$kind.esm"] = Dict{String, Any}(
                "kind" => kind, "operator" => o, "sha256" => bytes2hex(sha256(d.bytes))
            )
        end
    end
    # §8.1, in the order validation lists the checks; the first failure refuses.
    L = layout!(b, ctx)
    S = samples!(b, ctx)
    M = model_built!(b, ctx)
    worst = 0.0
    for (t, u) in S
        f = eval_rhs(M, u, t)
        fk = [eval_rhs(rhs_built!(b, ctx, ctx.ops[o]), u, t) for o in names]
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
        band_model_check!(b, ctx, op)
        push!(validation, Dict{String, Any}("check" => "band_model", "operator" => o, "passed" => true))
        non_empty_check(b, ctx, op)
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
band-value file (§8.2).
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
    for (t, u) in samples!(b, ctx)
        JB = band_jacobian(B, a.points, u, t)
        JI = forwarddiff_jacobian(R, u, t)
        for r in rows
            Mr = maximum(abs, view(JI, r, :); init = 0.0)
            for c in axes(JI, 2)
                Mr = max(Mr, abs(get(JB, (r, c), 0.0)))
            end
            for c in axes(JI, 2)
                jb, ji = get(JB, (r, c), 0.0), JI[r, c]
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
        entries = sort!([(L.names[r], L.names[c], JI[r, c]) for (r, c) in pattern]; by = e -> (e[1], e[2]))
        push!(
            samples, Dict{String, Any}(
                "t" => t, "u" => Dict{String, Any}(L.names[n] => u[n] for n in eachindex(u)),
                "entries" => Any[Any[e...] for e in entries]
            )
        )
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
