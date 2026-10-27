# Resolved settings other than the schedule (spec/plan.md §7.3–§7.7).

"""
    resolve_step(m, config) -> Dict

spec/plan.md §7.3: `dt` (seconds) or `limits` (each `{"max": seconds}`; an
observed limit is `unsupported_step_limit`, after its name resolves to an
observed, rule 2) when given; `hit` always, sorted; `hit_multiples` and
`resolution` in seconds when given.
"""
function resolve_step(m::PlanModel, config::AbstractDict)
    step = get(config, "step", Dict{String, Any}())
    out = Dict{String, Any}()
    haskey(step, "dt") && (out["dt"] = plan_duration(m, step["dt"], "/step/dt"))
    if haskey(step, "limits")
        lims = Any[]
        for (i, l) in enumerate(step["limits"])
            at = jptr("/step/limits", i - 1)
            if l isa AbstractString
                k = name_kind(m, l)
                k === nothing && refuse("name_unresolved", at, "step limit $l is not a name of the model")
                k === :observed || refuse("name_wrong_kind", at, "step limit $l is a $k, not an observed")
                refuse("unsupported_step_limit", at, "observed step limits need limit operators, U6 (M4b)")
            end
            push!(lims, Dict{String, Any}("max" => plan_duration(m, l["max"], jptr(at, "max"))))
        end
        out["limits"] = lims
    end
    out["hit"] = sort!(unique!(String.(get(step, "hit", String[]))))
    haskey(step, "hit_multiples") && (out["hit_multiples"] = plan_duration(m, step["hit_multiples"], "/step/hit_multiples"))
    haskey(step, "resolution") && (out["resolution"] = plan_duration(m, step["resolution"], "/step/resolution"))
    return out
end

"""The number of cells of a leaf's states (spec finding: §7.4 says "the leaf's array cells"; this is the largest cell count among the states its operators write, 1 for scalars)."""
function leaf_cells(ctx, names)
    written = unique!(reduce(vcat, (ctx.ops[o].writes for o in names); init = String[]))
    return maximum((prod(ctx.model.sizes[X] for X in get(ctx.model.shapes, s, String[]); init = 1) for s in written); init = 1)
end

"""
    resolve_partition(ctx, root) -> Dict

spec/plan.md §7.4 (D77): `ranks` and `tiles` (default `[1, 1]`), checked
against the horizontal axes; `halo` per horizontal axis, the largest stencil
radius in either direction, absent without horizontal axes; `shards` per
pointwise or column leaf by node id.
"""
function resolve_partition(ctx, root::SNode)
    config, axes, m = ctx.config, ctx.axes, ctx.model
    p = get(config, "partition", Dict{String, Any}())
    ranks = Int.(get(p, "ranks", [1, 1]))
    tiles = Int.(get(p, "tiles", [1, 1]))
    for a in 1:2
        if a > length(axes.horizontal)
            (ranks[a] == 1 && tiles[a] == 1) || refuse("partition_axis_missing", "/partition",
                "ranks or tiles divide horizontal axis $a, which the model does not have")
        else
            X = axes.horizontal[a]
            ranks[a] * tiles[a] > m.sizes[X] && refuse("partition_too_fine", "/partition",
                "ranks × tiles = $(ranks[a] * tiles[a]) exceeds axis $X's size $(m.sizes[X])")
        end
    end
    out = Dict{String, Any}("ranks" => ranks, "tiles" => tiles)
    if !isempty(axes.horizontal)
        halo = Dict{String, Any}(X => 0 for X in axes.horizontal)
        for (_, op) in ctx.ops
            need_locality(op) === :stencil || continue
            for (X, (l, u)) in op.radius
                halo[X] = max(halo[X], l, u)
            end
        end
        out["halo"] = halo
    end
    rshards = get(p, "shards", Dict{String, Any}())
    shards = Dict{String, Any}()
    for n in allnodes(root)
        is_leaflike(n) || continue
        names = leaf_ops(n)
        cls = reduce(coarser, (need_locality(ctx.ops[o]) for o in names))
        cls in (:pointwise, :column) || continue
        lp = get(n.spec, "partition", Dict{String, Any}())
        v = if haskey(lp, "shards")
            lp["shards"]
        elseif haskey(lp, "shard_size")
            cells = leaf_cells(ctx, names)
            if cls === :column && axes.vertical !== nothing
                cells = cells ÷ m.sizes[axes.vertical]
            end
            cld(cells, Int(lp["shard_size"]))
        elseif haskey(rshards, n.id)
            rshards[n.id]
        else
            entries = [(o, rshards[o]) for o in names if haskey(rshards, o)]
            for (o, s) in entries
                s == entries[1][2] || refuse("partition_conflict", jptr("/partition/shards", o),
                    "partition.shards gives operators $(entries[1][1]) and $o of leaf $(n.id) different counts")
            end
            isempty(entries) ? 1 : entries[1][2]
        end
        shards[n.id] = v == "auto" ? 1 : Int(v)
    end
    out["shards"] = shards
    return out
end

const _PRECISION_ROLES = ("state", "rhs", "jacobian", "factors", "tables")
const _PRECISION_BITS = Dict("f16" => 16, "bf16" => 16, "f32" => 32, "f64" => 64)
_below(a, b) = _PRECISION_BITS[a] < _PRECISION_BITS[b]

"""The declared precision governing a state: its own `element_type`, else the domain's; `nothing` when unspecified (esm-spec §11.3.1)."""
function declared_precision(m::PlanModel, s)
    et = m.flat.state_variables[s].element_type
    et === nothing && m.flat.domain !== nothing && (et = m.flat.domain.element_type)
    et === nothing && return nothing
    return et == "Float32" ? "f32" : et == "Float64" ? "f64" : et
end

"""
    resolve_precision(ctx, root; check_rosenbrock) -> Dict

spec/plan.md §7.5: all five roles for every leaf and coupled node, by id: the
node-id entry over the leaf's operators' entries over `default`, role by
role; unset roles from the declared `element_type`. Refusals:
`precision_conflict`, `precision_contradicts_declaration`,
`precision_rhs_below_state` and, for leaves whose method is given,
`precision_low_with_classical_rosenbrock`, each at the `/precision` key the
offending role came from.

Spec finding: §7.5 runs before the strategies (§5.5), so a leaf whose `alg`
is `"auto"` has no method yet; its classical-Rosenbrock check runs after
§5.5.3 resolves it ([`check_rosenbrock_precision`](@ref)).

Spec finding: with states of different declared element types in one leaf,
"the declared element_type that governs the leaf's states" is not one value;
the first written state's in canonical order is used, and a role that differs
from any of them contradicts a declaration.
"""
function resolve_precision(ctx, root::SNode)
    pc = get(ctx.config, "precision", Dict{String, Any}())
    out = Dict{String, Any}()
    sources = Dict{String, Dict{String, String}}()
    for n in allnodes(root)
        is_leaflike(n) || continue
        names = leaf_ops(n)
        roles = Dict{String, String}()
        src = Dict{String, String}()
        for r in _PRECISION_ROLES
            if haskey(pc, n.id) && haskey(pc[n.id], r)
                roles[r] = pc[n.id][r]; src[r] = n.id
                continue
            end
            ents = [(o, pc[o][r]) for o in names if haskey(pc, o) && haskey(pc[o], r)]
            for (o, v) in ents
                v == ents[1][2] || refuse("precision_conflict", jptr("/precision", o),
                    "precision entries for operators $(ents[1][1]) and $o of leaf $(n.id) disagree on $r")
            end
            if !isempty(ents)
                roles[r] = ents[1][2]; src[r] = ents[1][1]
            elseif haskey(pc, "default") && haskey(pc["default"], r)
                roles[r] = pc["default"][r]; src[r] = "default"
            end
        end
        written = sort!(unique!(reduce(vcat, (ctx.ops[o].writes for o in names); init = String[])))
        decl = filter(!isnothing, [declared_precision(ctx.model, s) for s in written])
        haskey(roles, "state") || (roles["state"] = isempty(decl) ? "f64" : decl[1])
        haskey(roles, "rhs") || (roles["rhs"] = roles["state"])
        haskey(roles, "jacobian") || (roles["jacobian"] = roles["rhs"])
        haskey(roles, "factors") || (roles["factors"] = roles["rhs"])
        haskey(roles, "tables") || (roles["tables"] = "f64")
        for r in ("state", "rhs")
            haskey(src, r) || continue
            any(!=(roles[r]), decl) && refuse("precision_contradicts_declaration", jptr("/precision", src[r]),
                "precision $r = $(roles[r]) on leaf $(n.id) contradicts a declared element_type ($(join(unique(decl), ", ")))")
        end
        _below(roles["rhs"], roles["state"]) && refuse("precision_rhs_below_state",
            jptr("/precision", get(src, "rhs", get(src, "state", "default"))),
            "precision rhs = $(roles["rhs"]) is below state = $(roles["state"]) on leaf $(n.id)")
        alg = get(n.spec, "alg", "auto")
        alg == "auto" || check_rosenbrock_precision(n, alg, roles, src)
        out[n.id] = Dict{String, Any}(roles)
        sources[n.id] = src
    end
    return out, sources
end

"""`precision_low_with_classical_rosenbrock`: a `jacobian` or `factors` role below `rhs` on a Rosenbrock method other than `Rosenbrock23` (architecture §8)."""
function check_rosenbrock_precision(n::SNode, alg, roles, src)
    (TABLEAUS[alg].family === :rosenbrock && alg != "Rosenbrock23") || return
    for r in ("jacobian", "factors")
        _below(roles[r], roles["rhs"]) && refuse("precision_low_with_classical_rosenbrock",
            jptr("/precision", get(src, r, "default")),
            "precision $r = $(roles[r]) is below rhs = $(roles["rhs"]) on $alg, a classical Rosenbrock method (leaf $(n.id))")
    end
end

"""Glob over qualified names: `*` matches any characters, dots included (spec/run-config.md §7.2)."""
glob_match(pat::AbstractString, s::AbstractString) =
    occursin(Regex("^" * join(map(p -> replace(p, r"[\\^$.|?+()\[\]{}]" => s"\\\0"), split(pat, '*')), ".*") * "\\z"), s)

"""
    check_adjoint(ctx, root)

spec/plan.md §7.6, spec/run-config.md §8 rule 11: each `adjoint.wrt` entry
must match something once parameters bound by `bind_dt` are left out
(`wrt_matches_nothing`), and must not name a bound parameter
(`wrt_bound_parameter`).

Spec finding: what each prefix ranges over is read as: `u0` the state;
`params:` the scalar parameters (`numeric`: those with a numeric default);
`fields:` the array parameters; `forcing:` the forcing fields, of which a
model at M2 has none, so a `forcing:` entry matches nothing.
"""
function check_adjoint(ctx, root::SNode)
    adj = ctx.config["adjoint"]
    m = ctx.model
    bound = Set{String}()
    for n in allnodes(root)
        is_leaflike(n) && haskey(n.spec, "bind_dt") && union!(bound, String.(keys(n.spec["bind_dt"])))
    end
    for (_, op) in ctx.ops
        op.bind_dt === nothing || union!(bound, String.(keys(op.bind_dt)))
    end
    scalars = [p for p in m.params if !haskey(m.shapes, p)]
    arrays = [p for p in m.params if haskey(m.shapes, p)]
    for (i, w) in enumerate(adj["wrt"])
        at = jptr("/adjoint/wrt", i - 1)
        if w == "u0"
            isempty(m.states) && refuse("wrt_matches_nothing", at, "u0: the model has no state")
            continue
        end
        prefix, pat = split(w, ':'; limit = 2)
        pool = prefix == "params" ? scalars : prefix == "fields" ? arrays : String[]
        if prefix == "params" && pat == "numeric"
            matched = [p for p in pool if m.flat.parameters[p].default isa Real]
        else
            !occursin('*', pat) && pat in bound && refuse("wrt_bound_parameter", at,
                "$w names a parameter bound by bind_dt, a constant of the adjoint's tape (spec/adjoint.md §5.8)")
            matched = [p for p in pool if glob_match(pat, p)]
        end
        isempty(setdiff(matched, bound)) && refuse("wrt_matches_nothing", at, "$w matches nothing (bound parameters are left out)")
    end
    return nothing
end

"""spec/plan.md §7.7: the model by the run config's `model` string as written and the run config by its file name, each with the SHA-256 of its bytes."""
function inputs_json(inp::PlanInputs)
    Dict{String, Any}(
        "model" => Dict{String, Any}("path" => String(inp.config["model"]), "sha256" => bytes2hex(sha256(inp.model_bytes))),
        "run_config" => Dict{String, Any}("path" => basename(inp.config_path), "sha256" => bytes2hex(sha256(inp.config_bytes))))
end
