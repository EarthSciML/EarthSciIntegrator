# Algebraic equations, solving nodes and their pairing (spec/plan.md §5.9,
# §6.4; D120, D121, D122).

"""
    Residual

One algebraic equation of the flattened model (spec/plan.md §5.9.1): its id
and origin as EarthSciAST's stripped object gives them (`<prefix>.residual[j]`),
its owner for `components` selection, its flattened equation, its extents (the
shape of rhs − lhs), the names it references and the names it reads (those
with the observed definitions' names, transitively).
"""
struct Residual
    id::String
    origin::String
    owner::Union{String, Nothing}
    equation::Int
    lhs::ASTExpr
    rhs::ASTExpr
    extents::Vector{Int}
    refs::Vector{String}
    reads::Vector{String}
end

"""Whether flattened equation `eq` is an algebraic (implicit) equation: neither a state equation nor an observed's definition."""
function is_implicit_equation(flat, eq, lifted_shapes)
    lifted_variable(eq, lifted_shapes) === nothing || return false
    differentiated_variable(eq.lhs) === nothing || return false
    defined_observed(flat, eq) === nothing || return false
    return true
end

"""The names `e` references that are variables of `flat` (states, parameters, observeds)."""
_flat_names(flat, e) = [n for n in EarthSciAST.free_variables(e) if haskey(flat.state_variables, n) || haskey(flat.parameters, n) || haskey(flat.observed_variables, n)]

"""`names` with, transitively, the names the definitions of the observeds among them reference."""
function _flat_closure(flat, defs, names)
    out = Set{String}(names)
    queue = [n for n in out if haskey(flat.observed_variables, n)]
    while !isempty(queue)
        o = pop!(queue)
        for r in get(defs, o, ASTExpr[]), n in _flat_names(flat, r)
            n in out && continue
            push!(out, n)
            haskey(flat.observed_variables, n) && push!(queue, n)
        end
    end
    return out
end

"""
The solved unknowns (spec/plan.md §4): the algebraic unknowns of `flat` that
some algebraic equation reads, directly or through observeds.
"""
function solved_unknowns(flat, contributions, observed_defs)
    defs = Dict{String, Vector{ASTExpr}}(k => ASTExpr[v] for (k, v) in observed_defs)
    out = Set{String}()
    for eq in flat.equations
        is_implicit_equation(flat, eq, flat.lifted_shapes) || continue
        names = _flat_closure(flat, defs, vcat(_flat_names(flat, eq.lhs), _flat_names(flat, eq.rhs)))
        for n in names
            haskey(flat.algebraic_variables, n) && push!(out, n)
        end
    end
    return out
end

"""The extents of an expression side (spec/plan.md §5.9.1): `[]` for a scalar."""
function side_extents(m::PlanModel, e)
    if e isa VarExpr
        return haskey(m.shapes, e.name) ? Int[m.sizes[X] for X in m.shapes[e.name]] : Int[]
    elseif e isa OpExpr
        if e.op == "faq"
            return Int[_range_size(m, e.ranges[String(o)]) for o in something(e.output_idx, Any[]) if o isa AbstractString]
        elseif e.op == "index"
            return Int[]
        elseif e.op == "makearray"
            return Int[r[2] - r[1] + 1 for r in e.regions[1]]
        end
        for a in e.args
            x = side_extents(m, a)
            isempty(x) || return x
        end
    end
    return Int[]
end

function _range_size(m::PlanModel, r)
    r isa EarthSciAST.IndexSetRef && isempty(r.of) && return m.sizes[r.from]
    r isa AbstractDict && haskey(r, "from") && return m.sizes[String(r["from"])]
    r isa AbstractVector && length(r) == 2 && all(x -> x isa Integer || x isa IntExpr, r) &&
        return _int(r[2]) - _int(r[1]) + 1
    error("a faq range $r the planner does not read")
end

_int(x::Integer) = Int(x)
_int(x::IntExpr) = Int(x.value)

"""The algebraic equations of the model as [`Residual`](@ref)s, in flattened equation order (§5.9.1)."""
function build_residuals(m::PlanModel)
    flat = m.flat
    out = Residual[]
    count = Dict{String, Int}()
    for (i, eq) in enumerate(flat.equations)
        is_implicit_equation(flat, eq, flat.lifted_shapes) || continue
        contribs = m.contributions[i]
        (isempty(contribs) || isempty(contribs[1].origins)) && refuse(
            "term_origin_unknown", "/model", "algebraic equation $i of the flattened model has no origin"
        )
        o = contribs[1].origins[1]
        prefix = o.component !== nothing ? o.component : "coupling[$(o.coupling)]"
        j = count[prefix] = get(count, prefix, 0) + 1
        refs = sort!(unique!(vcat(referenced_names(m, eq.lhs), referenced_names(m, eq.rhs))))
        owner = if o.component !== nothing
            o.component
        else
            R = sort!(unique!([component_of(n) for n in refs]))
            length(R) == 1 ? R[1] : nothing
        end
        ext = side_extents(m, eq.rhs)
        isempty(ext) && (ext = side_extents(m, eq.lhs))
        push!(out, Residual("$prefix.residual[$j]", prefix, owner, i, eq.lhs, eq.rhs, ext, refs, observed_closure(m, refs)))
    end
    return out
end

"""The extents of variable `x` (`[]` for a scalar)."""
var_extents(m::PlanModel, x) = haskey(m.shapes, x) ? Int[m.sizes[X] for X in m.shapes[x]] : Int[]

# ---------------------------------------------------------------------------
# The residual written cell by cell (spec/plan.md §6.4)

"""The index names `_r1`, `_r2`, … of the cell-by-cell form."""
cell_names(d::Int) = ["_r$k" for k in 1:d]

"""
    residual_rhs(m, r, shape) -> JSON

The right-hand side of the state equation of the unknown paired with
residual `r`, whose shape (index-set names) is `shape` (§6.4): for a scalar,
`rhs − lhs` as they stand; for an array, a `faq` over `_r1, …` of each side
read at the cell. A side the planner cannot read at a cell is refused,
`unsupported_solve_pairing`, at `at`.
"""
function residual_rhs(m::PlanModel, r::Residual, shape::AbstractVector, at::AbstractString)
    isempty(shape) && return Dict{String, Any}("op" => "-", "args" => Any[_ser(r.rhs), _ser(r.lhs)])
    names = cell_names(length(shape))
    cell(e) = _cell_side(m, e, names, r, at)
    return Dict{String, Any}(
        "op" => "faq", "args" => Any[], "output_idx" => Any[names...],
        "ranges" => Dict{String, Any}(names[d] => Dict{String, Any}("from" => shape[d]) for d in eachindex(names)),
        "expr" => Dict{String, Any}("op" => "-", "args" => Any[cell(r.rhs), cell(r.lhs)])
    )
end

function _cell_side(m::PlanModel, e, names, r, at)
    side = _ser(e)
    any(n -> _json_mentions(side, n), names) && refuse(
        "unsupported_solve_pairing", at,
        "residual $(r.id) already uses an index name $(join(names, ", ")), which the cell-by-cell form binds (spec/plan.md §6.4)"
    )
    (e isa NumExpr || e isa IntExpr) && return side
    if e isa VarExpr
        isempty(var_extents(m, e.name)) && return side
        return Dict{String, Any}("op" => "index", "args" => Any[e.name, names...])
    end
    if e isa OpExpr && e.op == "faq"
        O = something(e.output_idx, Any[])
        all(o -> o isa AbstractString, O) && length(O) == length(names) || refuse(
            "unsupported_solve_pairing", at,
            "residual $(r.id): a faq side whose output indices are not one name per dimension of its unknown (spec/plan.md §6.4)"
        )
        ren = Dict{String, String}(String(O[d]) => names[d] for d in eachindex(O))
        body = _rename(side["expr"], ren)
        rest = Dict{String, Any}(k => v for (k, v) in side["ranges"] if !haskey(ren, k))
        isempty(rest) && return body
        haskey(side, "filter") && refuse("unsupported_solve_pairing", at, "residual $(r.id): a filtered reducing faq side (spec/plan.md §6.4)")
        return Dict{String, Any}("op" => "faq", "args" => Any[], "output_idx" => Any[], "ranges" => rest, "expr" => body)
    end
    isempty(side_extents(m, e)) && return side
    return refuse(
        "unsupported_solve_pairing", at,
        "residual $(r.id) has an array-valued side that is neither a name nor a faq; the planner does not read it at a cell (spec/plan.md §6.4)"
    )
end

_json_mentions(x::AbstractString, n) = x == n
_json_mentions(x::AbstractDict, n) = any(v -> _json_mentions(v, n), values(x)) || any(k -> k == n, keys(x))
_json_mentions(x::AbstractVector, n) = any(v -> _json_mentions(v, n), x)
_json_mentions(x, n) = false

"""Rename the index names `ren` in a serialized expression; a nested `faq` binding one of them shadows it."""
function _rename(x, ren::Dict{String, String})
    isempty(ren) && return x
    if x isa AbstractString
        return get(ren, x, x)
    elseif x isa AbstractVector
        return Any[_rename(v, ren) for v in x]
    elseif x isa AbstractDict
        if get(x, "op", nothing) == "faq"
            bound = Set{String}(String.(keys(something(get(x, "ranges", nothing), Dict()))))
            for o in something(get(x, "output_idx", nothing), Any[])
                o isa AbstractString && push!(bound, o)
            end
            ren = Dict{String, String}(k => v for (k, v) in ren if !(k in bound))
        end
        out = Dict{String, Any}()
        for (k, v) in x
            out[k] = k in ("op", "ranges", "output_idx", "units") ? v : _rename(v, ren)
        end
        return out
    end
    return x
end

# ---------------------------------------------------------------------------
# Solving nodes and the pairing (spec/plan.md §5.9.2–§5.9.6)

"""
    SolvingNode

A `solve` node, or a `coupled` node with algebraic operators (§5.9.2): its
schedule node, its algebraic operators in node order, its residual blocks in
order, its unknown blocks in code-point order, and, once paired, the unknown
of each residual block.
"""
mutable struct SolvingNode
    node::Any
    ops::Vector{String}
    residuals::Vector{Residual}
    unknowns::Vector{String}
    pairing::Vector{String}
end

is_bounded(m::PlanModel, z) = (v = m.flat.state_variables[z]; v.lb !== nothing || v.ub !== nothing)

"""The maximum flow of §5.9.3's network: source → residual k (its element count) → unknown j it reads → sink (its element count)."""
function block_flow(rcap::Vector{Int}, zcap::Vector{Int}, edges::Vector{Vector{Int}})
    nr, nz = length(rcap), length(zcap)
    n = nr + nz + 2
    S, T = n - 1, n
    cap = zeros(Int, n, n)
    big = sum(rcap; init = 0) + 1
    for k in 1:nr
        cap[S, k] = rcap[k]
        for j in edges[k]
            cap[k, nr + j] = big
        end
    end
    for j in 1:nz
        cap[nr + j, T] = zcap[j]
    end
    flow = 0
    while true
        prev = zeros(Int, n)
        prev[S] = S
        q = [S]
        while !isempty(q) && prev[T] == 0
            u = popfirst!(q)
            for v in 1:n
                (prev[v] == 0 && cap[u, v] > 0) || continue
                prev[v] = u
                push!(q, v)
            end
        end
        prev[T] == 0 && return flow
        f = typemax(Int)
        v = T
        while v != S
            f = min(f, cap[prev[v], v]); v = prev[v]
        end
        v = T
        while v != S
            cap[prev[v], v] -= f; cap[v, prev[v]] += f; v = prev[v]
        end
        flow += f
    end
    return flow
end

"""
    check_singular(m, sn, at)

§5.9.3: refuse `solve_structurally_singular` when the element counts differ
or the block network carries less than every unknown element.
"""
function check_singular(m::PlanModel, sn::SolvingNode, at)
    rcap = Int[prod(r.extents; init = 1) for r in sn.residuals]
    zcap = Int[prod(var_extents(m, z); init = 1) for z in sn.unknowns]
    NE, NZ = sum(rcap; init = 0), sum(zcap; init = 0)
    NE == NZ || refuse(
        "solve_structurally_singular", at,
        "node $(sn.node.id) has $NE residual elements and $NZ unknown elements (spec/plan.md §5.9.3)"
    )
    edges = [Int[j for (j, z) in enumerate(sn.unknowns) if z in r.reads] for r in sn.residuals]
    f = block_flow(rcap, zcap, edges)
    f < NZ && refuse(
        "solve_structurally_singular", at,
        "node $(sn.node.id): no matching of its residuals with its unknowns can cover every unknown element (a maximum flow of $f of $NZ, spec/plan.md §5.9.3)"
    )
    return
end

"""
A band's diagonal box (§5.9.4): per dimension the `(lo, hi)` of the rows at
which its column equals its row, or `nothing` when it is empty.
"""
function diagonal_box(b::AbstractDict)
    rows = b["rows"]
    ri, ci = b["row_idx"], b["col_idx"]
    contracted = Dict{String, Tuple{Int, Int}}(String(c[1]) => (Int(c[2]), Int(c[3])) for c in get(b, "contracted", Any[]))
    length(ri) == length(ci) == length(rows) || return nothing
    box = Tuple{Int, Int}[]
    for d in eachindex(rows)
        lo, hi = Int(rows[d][1]), Int(rows[d][2])
        r, c = ri[d], ci[d]
        if c isa AbstractString && r isa AbstractString && c == r
            # the whole range
        elseif c isa AbstractString && haskey(contracted, c)
            clo, chi = contracted[c]
            lo, hi = max(lo, clo), min(hi, chi)
        else
            k = integer_constant(c)
            k === nothing && return nothing
            if r isa AbstractString
                lo, hi = max(lo, k), min(hi, k)
            else
                ρ = integer_constant(r)
                (ρ !== nothing && ρ == k) || return nothing
            end
        end
        lo <= hi || return nothing
        push!(box, (lo, hi))
    end
    return box
end

"""An integer constant of a band index (§5.2.3), or `nothing`."""
function integer_constant(x)
    x isa Integer && return Int(x)
    x isa AbstractDict || return nothing
    op = get(x, "op", nothing)
    args = get(x, "args", Any[])
    vals = Int[]
    for a in args
        v = integer_constant(a)
        v === nothing && return nothing
        push!(vals, v)
    end
    try
        op == "+" && !isempty(vals) && return reduce(Base.checked_add, vals)
        op == "*" && !isempty(vals) && return reduce(Base.checked_mul, vals)
        op == "-" && length(vals) == 1 && return Base.checked_neg(vals[1])
        op == "-" && length(vals) == 2 && return Base.checked_sub(vals[1], vals[2])
    catch e
        e isa OverflowError && return nothing
        rethrow()
    end
    return nothing
end

"""Whether the union of `boxes` covers the block of extents `ext`, decided on breakpoints (§5.9.4)."""
function covers_block(boxes::Vector{Vector{Tuple{Int, Int}}}, ext::Vector{Int})
    isempty(ext) && return !isempty(boxes)
    isempty(boxes) && return false
    D = length(ext)
    cuts = [sort!(unique!(vcat(1, ext[d] + 1, [b[d][1] for b in boxes], [b[d][2] + 1 for b in boxes]))) for d in 1:D]
    cuts = [filter(c -> 1 <= c <= ext[d] + 1, cuts[d]) for d in 1:D]
    for cell in Iterators.product((1:(length(cuts[d]) - 1) for d in 1:D)...)
        p = [cuts[d][cell[d]] for d in 1:D]
        any(b -> all(b[d][1] <= p[d] <= b[d][2] for d in 1:D), boxes) || return false
    end
    return true
end

"""The probe's state for residual block `k` (§5.9.4)."""
residual_probe_state(k::Integer) = "$PROBE_COMPONENT.res_$k"

"""
    diagonal_pairs(b, ctx, sn) -> Vector{Vector{Int}}

§5.9.4: for each residual block, the unknown blocks (indices into
`sn.unknowns`) it forms a diagonal pair with, from one probe document of the
node's residuals.
"""
function diagonal_pairs(b, ctx, sn::SolvingNode, at)
    m = ctx.model
    cand = Vector{Vector{Int}}(undef, length(sn.residuals))
    probe = Tuple{Int, String}[]
    for (k, r) in enumerate(sn.residuals)
        cand[k] = Int[j for (j, z) in enumerate(sn.unknowns) if z in r.reads && var_extents(m, z) == r.extents]
        isempty(cand[k]) || push!(probe, (k, sn.unknowns[cand[k][1]]))
    end
    out = [Int[] for _ in sn.residuals]
    isempty(probe) && return out
    entries = residual_probe_bands(b, ctx, sn, probe, at)
    for (k, _) in probe
        row = residual_probe_state(k)
        for j in cand[k]
            boxes = Vector{Tuple{Int, Int}}[]
            for d in entries
                (d["row"] == row && d["col"] == sn.unknowns[j]) || continue
                box = diagonal_box(d)
                box === nothing || push!(boxes, box)
            end
            covers_block(boxes, sn.residuals[k].extents) && push!(out[k], j)
        end
    end
    return out
end

"""
The canonical pairing (§5.9.5): Kuhn's augmenting paths over the diagonal
pairs, residual blocks in order, candidates in the owner's component first
and then in code-point order. `nothing` when some residual block stays
unmatched.
"""
function canonical_pairing(m::PlanModel, sn::SolvingNode, diag::Vector{Vector{Int}})
    order = map(eachindex(sn.residuals)) do k
        own = sn.residuals[k].owner
        mine(j) = own !== nothing && component_of(sn.unknowns[j]) == own
        vcat(sort([j for j in diag[k] if mine(j)]; by = j -> sn.unknowns[j]), sort([j for j in diag[k] if !mine(j)]; by = j -> sn.unknowns[j]))
    end
    partner = zeros(Int, length(sn.unknowns))   # unknown → residual
    function augment(k, seen)
        for j in order[k]
            j in seen && continue
            push!(seen, j)
            if partner[j] == 0 || augment(partner[j], seen)
                partner[j] = k
                return true
            end
        end
        return false
    end
    for k in eachindex(sn.residuals)
        augment(k, Set{Int}()) || return nothing
    end
    pair = zeros(Int, length(sn.residuals))
    for (j, k) in enumerate(partner)
        k == 0 || (pair[k] = j)
    end
    return pair
end

"""
The first alternating cycle (§5.9.6), as the residual and unknown blocks on
it, or `nothing` when the pairing is unique: arcs residual → each diagonal
partner it is not paired with, unknown → its residual; depth first from the
residual blocks in order.
"""
function alternating_cycle(sn::SolvingNode, diag, pair)
    nr = length(sn.residuals)
    partner = zeros(Int, length(sn.unknowns))
    for (k, j) in enumerate(pair)
        partner[j] = k
    end
    succ(v) = v <= nr ? [nr + j for j in diag[v] if j != pair[v]] : [partner[v - nr]]
    state = zeros(Int, nr + length(sn.unknowns))   # 0 new, 1 on stack, 2 done
    stack = Int[]
    function dfs(v)
        state[v] = 1
        push!(stack, v)
        for w in succ(v)
            state[w] == 1 && return stack[findfirst(==(w), stack):end]
            if state[w] == 0
                c = dfs(w)
                c === nothing || return c
            end
        end
        pop!(stack)
        state[v] = 2
        return nothing
    end
    for k in 1:nr
        state[k] == 0 || continue
        c = dfs(k)
        c === nothing || return c
    end
    return nothing
end

"""
    pair_node!(b, ctx, sn)

§5.9.3–§5.9.6 for one solving node: the singularity test, the diagonal
pairs, the canonical pairing and its uniqueness for bounded unknowns.
"""
function pair_node!(b, ctx, sn::SolvingNode)
    m = ctx.model
    at = nptr(sn.node)
    check_singular(m, sn, at)
    diag = diagonal_pairs(b, ctx, sn, at)
    pair = canonical_pairing(m, sn, diag)
    pair === nothing && refuse(
        "unsupported_solve_pairing", at,
        "node $(sn.node.id) has no diagonal pairing of its residuals with its unknowns; an element-level matching is not yet specified (spec/plan.md §5.9.5)"
    )
    cyc = alternating_cycle(sn, diag, pair)
    nr = length(sn.residuals)
    if cyc !== nothing
        blocks = [v <= nr ? sn.residuals[v].id : sn.unknowns[v - nr] for v in cyc]
        any(v -> v > nr && is_bounded(m, sn.unknowns[v - nr]), cyc) && refuse(
            "solve_pairing_ambiguous", at,
            "node $(sn.node.id) can pair its residuals with its bounded unknowns more than one way (around $(join(blocks, ", "))), and the pairing decides the complementarity conditions (spec/plan.md §5.9.6)"
        )
    end
    sn.pairing = String[sn.unknowns[j] for j in pair]
    return sn
end

# ---------------------------------------------------------------------------
# The schedule's algebraic checks (spec/run-config.md §8 rules 18, 19;
# spec/plan.md §7.1)

"""The stiffly accurate methods a DAE may use (spec/run-config.md §4.3, D122)."""
const DAE_METHODS = ("Rodas3", "Rodas4", "Rodas5P", "TRBDF2", "Kvaerno3", "Kvaerno4")

"""Whether schedule node `n` is a solving node: a `solve` node, or a `coupled` node naming an algebraic operator."""
is_solving(ctx, n) = n.kind == "solve" || (n.kind == "coupled" && any(o -> haskey(ctx.ops, o) && is_algebraic(ctx.ops[o]), leaf_ops(n)))

"""The algebraic operators a solving node names, in its order."""
node_algebraic_ops(ctx, n) = n.kind == "solve" ? [String(n.spec["solve"])] : [o for o in leaf_ops(n) if is_algebraic(ctx.ops[o])]

"""A solving node's unknowns (§5.9.2), in code-point order: a `solve` node's `unknowns`; a DAE's, the solved unknowns its residuals read."""
function node_unknowns(ctx, n)
    n.kind == "solve" && return sort!(String.(collect(n.spec["unknowns"])))
    m = ctx.model
    return sort!(unique!([z for o in node_algebraic_ops(ctx, n) for r in ctx.ops[o].residuals for z in r.reads if is_solved(m, z)]))
end

"""The method objects of a solving node with their pointers: a `solve` node's `method`, a DAE's `initialize` when given."""
function node_methods(n)
    n.kind == "solve" && return [(n.spec["method"], "method")]
    return haskey(n.spec, "initialize") ? [(n.spec["initialize"], "initialize")] : Tuple{Any, String}[]
end

"""
    check_algebraic_schedule(ctx, nodes)

Rule 19 in §7.1's order (operator kinds, the unknowns, the DAE methods and
bounds, the unspecified solve options), then rule 2 on the names a solve
method gives and rule 18's checks that need no derivative.
"""
function check_algebraic_schedule(ctx, nodes)
    m, ops = ctx.model, ctx.ops
    for n in nodes
        if n.kind == "solve"
            o = String(n.spec["solve"])
            is_algebraic(ops[o]) || refuse(
                "operator_kind_mismatch", nptr(n, "solve"),
                "solve node $(n.id) names operator $o, which holds right-hand-side terms, not algebraic equations"
            )
        elseif n.kind == "leaf"
            for (o, p) in operator_refs(n)
                is_algebraic(ops[o]) && refuse(
                    "operator_kind_mismatch", p,
                    "leaf $(n.id) names algebraic operator $o; a solve node or a coupled node advances it"
                )
            end
        end
    end
    for n in nodes
        n.kind == "solve" || continue
        for (i, z) in enumerate(n.spec["unknowns"])
            at = nptr(n, "unknowns", i - 1)
            k = name_kind(m, String(z))
            k === nothing && refuse("name_unresolved", at, "solve unknown $z is not a name of the flattened model")
            is_solved(m, String(z)) || refuse("name_wrong_kind", at, "solve unknown $z is a $k, not an algebraic unknown an algebraic equation reads")
        end
    end
    solver = Dict{String, String}()
    for n in nodes
        is_solving(ctx, n) || continue
        for z in node_unknowns(ctx, n)
            haskey(solver, z) && refuse(
                "algebraic_unknown_solved_twice", nptr(n),
                "algebraic unknown $z is solved by node $(solver[z]) and by node $(n.id)"
            )
            solver[z] = n.id
        end
    end
    for z in m.solved
        haskey(solver, z) || refuse(
            "algebraic_unknown_unsolved", "/model",
            "algebraic unknown $z is read by an algebraic equation, and no solve or coupled node solves it"
        )
    end
    for n in nodes
        (n.kind == "coupled" && is_solving(ctx, n)) || continue
        alg = get(n.spec, "alg", "auto")
        alg == "auto" || alg in DAE_METHODS || refuse(
            "dae_method_invalid", nptr(n, "alg"),
            "coupled node $(n.id) holds algebraic operators, an index-1 DAE, and $alg is not one of the stiffly accurate methods $(join(DAE_METHODS, ", "))"
        )
        haskey(n.spec, "imex") && refuse("dae_method_invalid", nptr(n, "imex"), "coupled node $(n.id) is an index-1 DAE; imex is not specified for it")
        for z in node_unknowns(ctx, n)
            is_bounded(m, z) && refuse(
                "unsupported_dae_bounds", nptr(n),
                "coupled node $(n.id) is an index-1 DAE with the bounded unknown $z; a complementarity DAE is not specified"
            )
        end
    end
    for n in nodes, (meth, key) in node_methods(n)
        get(meth, "linesearch", "none") == "strong_wolfe" && refuse(
            "unsupported_solve_option", nptr(n, key, "linesearch"), "linesearch strong_wolfe is not specified in v1 (spec/methods.md §15.6)"
        )
        get(meth, "anderson", 0) > 0 && refuse(
            "unsupported_solve_option", nptr(n, key, "anderson"), "Anderson acceleration is not specified in v1 (spec/methods.md §15.6)"
        )
    end
    for n in nodes, (meth, key) in node_methods(n)
        rd = Set{String}(z for o in node_algebraic_ops(ctx, n) for r in ops[o].residuals for z in r.reads)
        function frozen_name(o, at, what)
            need_kind(m, o, :observed, at, what)
            return String(o) in rd || refuse("name_wrong_kind", at, "$what $o is not read by the residuals of node $(n.id) (spec/plan.md §6.5)")
        end
        if haskey(meth, "picard")
            for (i, o) in enumerate(meth["picard"]["freeze"])
                frozen_name(o, nptr(n, key, "picard", "freeze", i - 1), "frozen observed")
            end
        end
        # Scale keys are unknowns of the node; values parameters or observeds (D125).
        zs = Set(node_unknowns(ctx, n))
        sc = get(meth, "scale", Dict{String, Any}())
        for z in sort!(String.(collect(keys(sc))))
            at = nptr(n, key, "scale", z)
            name_kind(m, z) === nothing && refuse("name_unresolved", at, "scale key $z is not a name of the flattened model")
            z in zs || refuse("name_wrong_kind", at, "scale key $z is not an unknown of node $(n.id)")
            v = sc[z]
            v isa AbstractString || continue
            k = name_kind(m, String(v))
            k === nothing && refuse("name_unresolved", at, "the scale $v of $z is not a name of the flattened model")
            k in (:parameter, :observed) || refuse("name_wrong_kind", at, "the scale $v of $z is a $k, not a parameter or an observed")
        end
        conv = get(meth, "converged", Dict{String, Any}())
        # A rel_change observed need only be an observed (§6.5).
        haskey(conv, "rel_change") && need_kind(m, conv["rel_change"], :observed, nptr(n, key, "converged", "rel_change"), "rel_change observed")
    end
    for n in nodes
        n.kind == "solve" || continue
        zs = node_unknowns(ctx, n)
        any(z -> is_bounded(m, z), zs) || continue
        conv = get(n.spec["method"], "converged", Dict{String, Any}())
        haskey(conv, "rel_change") && refuse(
            "solve_bounds_invalid", nptr(n, "method", "converged", "rel_change"),
            "solve node $(n.id) has a bounded unknown, which converges on its reformulated residual only (spec/run-config.md §4.10)"
        )
        for (i, z) in enumerate(n.spec["unknowns"])
            v = m.flat.state_variables[String(z)]
            for bnd in (v.lb, v.ub)
                bnd isa AbstractString || continue
                reads = observed_closure(m, [String(bnd)])
                any(y -> y in zs, reads) && refuse(
                    "solve_bounds_invalid", nptr(n, "unknowns", i - 1),
                    "the bound $bnd of $z is, or depends on, an unknown of solve node $(n.id), so it would move during the solve"
                )
            end
        end
    end
    # §5.9.7: a bound is a number, a parameter or a state-free observed (D125).
    for n in nodes
        n.kind == "solve" || continue
        for (i, z) in enumerate(n.spec["unknowns"])
            v = m.flat.state_variables[String(z)]
            for bnd in (v.lb, v.ub)
                bnd isa AbstractString || continue
                state_free(m, String(bnd)) || refuse(
                    "unsupported_bound_reference", nptr(n, "unknowns", i - 1),
                    "the bound $bnd of $z is a $(something(name_kind(m, String(bnd)), "name")) that depends on a state; a bound that moves with the state is a later addition (spec/plan.md §5.9.7)"
                )
            end
        end
    end
    # Rule 18 (D125): a scale names a state-free observed or a parameter,
    # scalar or of the unknown's shape.
    for n in nodes, (meth, key) in node_methods(n)
        sc = get(meth, "scale", Dict{String, Any}())
        for z in sort!(String.(collect(keys(sc))))
            v = sc[z]
            v isa AbstractString || continue
            at = nptr(n, key, "scale", z)
            name_kind(m, String(v)) === :observed && !state_free(m, String(v)) && refuse(
                "solve_scale_invalid", at, "the scale $v of $z is an observed that depends on a state; a scale is evaluated once per invocation (spec/run-config.md §4.10)"
            )
            sh = get(m.shapes, String(v), String[])
            isempty(sh) || sh == get(m.shapes, z, String[]) || refuse(
                "solve_scale_invalid", at, "the scale $v of $z has the shape $(sh), neither scalar nor $z's $(get(m.shapes, z, String[]))"
            )
        end
    end
    return nothing
end

"""Whether `name` is a parameter, or an observed that depends on no state directly or through other observeds (D125)."""
state_free(m::PlanModel, name::String) = is_param(m, name) ||
    (name_kind(m, name) === :observed && !any(y -> is_state(m, y), observed_closure(m, [name])))

"""The plan's `bounds` of an algebraic operator (§5.9.7): per bounded paired unknown its `lb` and `ub`; `nothing` when none is bounded."""
function bounds_json(m::PlanModel, op)
    out = Dict{String, Any}()
    for z in unique(op.pairing)
        v = m.flat.state_variables[z]
        b = Dict{String, Any}()
        v.lb === nothing || (b["lb"] = v.lb)
        v.ub === nothing || (b["ub"] = v.ub)
        isempty(b) || (out[z] = b)
    end
    return isempty(out) ? nothing : out
end

# ---------------------------------------------------------------------------
# The algebraic structure (spec/plan.md §4 "Order of refusals" item 5)

"""
    algebraic_structure!(ctx, root)

For each solving node in tree order its pairing (§5.9.3–§5.9.6), then each
algebraic operator's class, pass 3 and pass 4 on its residual document, in
code-point order, then rule 18's sign check (§5.9.7).
"""
function algebraic_structure!(ctx, root)
    b = ctx.backend
    nodes = SolvingNode[]
    for n in allnodes(root)
        is_solving(ctx, n) || continue
        names = node_algebraic_ops(ctx, n)
        sn = SolvingNode(n, names, Residual[r for o in names for r in ctx.ops[o].residuals], node_unknowns(ctx, n), String[])
        pair_node!(b, ctx, sn)
        frozen, nfreeze = node_frozen(ctx, n)
        i = 0
        for o in names
            op = ctx.ops[o]
            op.pairing = sn.pairing[(i + 1):(i + length(op.residuals))]
            i += length(op.residuals)
            op.frozen, op.nfreeze = frozen, nfreeze
        end
        push!(nodes, sn)
    end
    m = ctx.model
    for o in sort!([k for (k, op) in ctx.ops if is_algebraic(op)])
        op = ctx.ops[o]
        isempty(op.pairing) && continue          # rule 5 refused an operator no node advances
        op.writes = sort!(unique(op.pairing))
        op.reads = observed_closure(m, reduce(vcat, (r.refs for r in op.residuals); init = String[]))
        op.structure = operator_structure(b, ctx, op)
    end
    for sn in nodes
        sn.node.kind == "solve" || continue
        check_bounds_sign(b, ctx, sn)
    end
    return nodes
end

"""
§5.9.7: at each sampled state, every element of a bounded unknown has a
diagonal Jacobian entry of its paired residual that is not positive; else
`solve_bounds_sign` at the unknown's entry of the node.
"""
function check_bounds_sign(b, ctx, sn::SolvingNode)
    m = ctx.model
    bounded = [z for z in sn.unknowns if is_bounded(m, z)]
    isempty(bounded) && return
    written = String.(collect(sn.node.spec["unknowns"]))
    L = layout!(b, ctx)
    for o in sn.ops
        op = ctx.ops[o]
        B = jac_built!(b, ctx, op)
        a = b.ops[o]
        for (s, (t, u)) in enumerate(samples!(b, ctx))
            J = band_jacobian(B, a.points, u, t, sample_forcing(b, ctx, s))
            for z in op.pairing
                z in bounded || continue
                n = prod(L.extents[z]; init = 1)
                for i in 1:n
                    d = L.offset[z] + i
                    v = get(J, (d, d), 0.0)
                    v > 0 && refuse(
                        "solve_bounds_sign", nptr(sn.node, "unknowns", findfirst(==(z), written) - 1),
                        "the residual paired with $(L.names[d]) has the positive diagonal derivative $v at t = $t; the residual of lhs ~ rhs is rhs - lhs, the pseudo-time rate of its unknown (spec/run-config.md §4.10)"
                    )
                end
            end
        end
    end
    return
end

"""The probe bands of §5.9.4 for the residual blocks `probe` (`(k, shape-giving unknown)` pairs) of node `sn`, as `serialize_band` dictionaries without `coef`."""
function residual_probe_bands(b, ctx, sn::SolvingNode, probe, at)
    m = ctx.model
    res = [(k, sn.residuals[k]) for (k, _) in probe]
    shapes = [get(m.shapes, z, String[]) for (_, z) in probe]
    d = DerivedDocument(residual_probe_document(m, res, shapes, at))
    return Dict{String, Any}[band_json(en) for en in document_bands(d.flat, "the residuals of node $(sn.node.id)"; coefficients = false)]
end

# ---------------------------------------------------------------------------
# Resolved solve nodes (spec/plan.md §7.1)

"""A solve method (or a DAE's `initialize`) with every option this spec defaults (§7.1)."""
function resolve_method(ctx, meth::AbstractDict, op)
    out = Dict{String, Any}()
    if haskey(meth, "ptc")
        p = meth["ptc"]
        g = get(p, "growth", 2.0)
        out["ptc"] = Dict{String, Any}("dt0" => Float64(get(p, "dt0", 1.0e-3)), "growth" => g == "uncapped" ? "uncapped" : Float64(g))
    elseif haskey(meth, "picard")
        out["picard"] = Dict{String, Any}("freeze" => Any[meth["picard"]["freeze"]...])
    else
        nw = get(meth, "newton", Dict{String, Any}())
        out["newton"] = Dict{String, Any}(k => v for (k, v) in nw)
    end
    out["linsolve"] = solve_linsolve(ctx, meth, op)
    haskey(meth, "ptc") || (out["linesearch"] = get(meth, "linesearch", "none"))
    c = get(meth, "converged", Dict{String, Any}("residual" => 1.0e-10))
    out["converged"] = Dict{String, Any}(k => v for (k, v) in c)
    out["maxiter"] = get(meth, "maxiter", 100)
    haskey(meth, "scale") && (out["scale"] = Dict{String, Any}(String(k) => v for (k, v) in meth["scale"]))
    haskey(meth, "recover") && (out["recover"] = Any[resolve_recover(a) for a in meth["recover"]])
    return out
end

"""A `recover` action as written, but a `{"method": M}` step with `M`'s method key, `linesearch` and `maxiter` defaulted (§7.1, D125)."""
function resolve_recover(a)
    a isa AbstractDict && haskey(a, "method") || return a
    mm = a["method"]
    out = Dict{String, Any}()
    if haskey(mm, "ptc")
        p = mm["ptc"]
        g = get(p, "growth", 2.0)
        out["ptc"] = Dict{String, Any}("dt0" => Float64(get(p, "dt0", 1.0e-3)), "growth" => g == "uncapped" ? "uncapped" : Float64(g))
    else
        out["newton"] = Dict{String, Any}(k => v for (k, v) in mm["newton"])
        out["linesearch"] = get(mm, "linesearch", "none")
    end
    out["maxiter"] = get(mm, "maxiter", 100)
    return Dict{String, Any}("method" => out)
end

"""A solve's `linsolve`: given (in object form), else by §5.5.3's table for the operator's class and structure, as if stiff."""
function solve_linsolve(ctx, meth, op)
    ls = get(meth, "linsolve", "auto")
    ls == "auto" || return linsolve_object(ls)
    s = op.structure
    if need_locality(op) === :pointwise && haskey(s, "block")
        # A cell block's size is its element count (D123).
        return _pointwise_linsolve(haskey(s["block"], "elements") ? length(s["block"]["elements"]) : length(s["block"]["vars"]))
    end
    return Dict{String, Any}("dense_lu" => Dict{String, Any}())
end

"""The resolved `solve` node (§7.1)."""
function resolve_solve(ctx, n)
    op = ctx.ops[String(n.spec["solve"])]
    return Dict{String, Any}(
        "id" => n.id, "solve" => String(n.spec["solve"]),
        "unknowns" => Any[sort!(String.(collect(n.spec["unknowns"])))...],
        "warm_start" => get(n.spec, "warm_start", "previous"),
        "method" => resolve_method(ctx, n.spec["method"], op)
    )
end

"""The default consistent initialization of a DAE (spec/run-config.md §4.3)."""
const DEFAULT_INITIALIZE = Dict{String, Any}(
    "newton" => Dict{String, Any}(), "linesearch" => "backtracking",
    "converged" => Dict{String, Any}("residual" => 1.0e-10), "maxiter" => 100
)

# ---------------------------------------------------------------------------
# The frozen documents (spec/plan.md §6.5, D124)

"""
A solving node's frozen observeds (§6.5, D124, D125), each once at its first
occurrence: picard's freeze (in written order), the rel_change observed, the
observeds its method's scale names (keys in code-point order), and the
observeds its unknowns' bounds name (unknowns in code-point order, lb before
ub); and how many of the first are picard's freeze.
"""
function node_frozen(ctx, n)
    meth = n.kind == "solve" ? n.spec["method"] : get(n.spec, "initialize", nothing)
    meth === nothing && return String[], 0
    m = ctx.model
    out = String[]
    add(o) = (o = String(o); o in out || push!(out, o); nothing)
    haskey(meth, "picard") && foreach(add, meth["picard"]["freeze"])
    nfreeze = length(out)
    rc = get(get(meth, "converged", Dict{String, Any}()), "rel_change", nothing)
    rc === nothing || add(rc)
    sc = get(meth, "scale", Dict{String, Any}())
    for z in sort!(String.(collect(keys(sc))))
        v = sc[z]
        v isa AbstractString && name_kind(m, String(v)) === :observed && add(v)
    end
    if n.kind == "solve"
        for z in sort!(String.(collect(n.spec["unknowns"])))
            sv = get(m.flat.state_variables, z, nothing)
            sv === nothing && continue
            for bnd in (sv.lb, sv.ub)
                bnd isa AbstractString && name_kind(m, String(bnd)) === :observed && add(bnd)
            end
        end
    end
    return out, nfreeze
end

frozen_state(k::Integer) = "$FROZEN_COMPONENT.frozen_$k"

"""The declaration of a frozen state for observed `o` (§6.5): its shape, `default` 0.0."""
function frozen_declaration(m::PlanModel, o)
    d = Dict{String, Any}("type" => "unknown", "default" => 0.0)
    haskey(m.shapes, o) && (d["shape"] = Any[m.shapes[o]...])
    return d
end

"""`D(frozen_k) = O_k`, an array observed read cell by cell (§6.5)."""
function frozen_rhs(m::PlanModel, o)
    haskey(m.shapes, o) || return o
    shape = m.shapes[o]
    names = cell_names(length(shape))
    return Dict{String, Any}(
        "op" => "faq", "args" => Any[], "output_idx" => Any[names...],
        "ranges" => Dict{String, Any}(names[d] => Dict{String, Any}("from" => shape[d]) for d in eachindex(names)),
        "expr" => Dict{String, Any}("op" => "index", "args" => Any[o, names...])
    )
end

"""Rename every reference to the names of `ren` in the right-hand sides and operator bindings of a document's equations."""
function _rename_refs!(models::AbstractDict, ren::Dict{String, String})
    rn(x) = x isa AbstractString ? get(ren, x, x) : x isa AbstractVector ? Any[rn(v) for v in x] :
        x isa AbstractDict ? Dict{String, Any}(k => (k in ("op", "ranges", "output_idx", "units") ? v : rn(v)) for (k, v) in x) : x
    for (_, node) in models
        for eq in get(node, "equations", Any[])
            eq["rhs"] = rn(eq["rhs"])
        end
        haskey(node, "subsystems") && _rename_refs!(node["subsystems"], ren)
    end
    return models
end

"""
    frozen_documents!(b, ctx, op) -> Vector{(kind, DerivedDocument)}

An algebraic operator's frozen documents (§6.5): the frozen residual
document when its solving node names observeds, and the frozen Jacobian
document when that node is a picard solve; its band list is kept for the
plan entry's `frozen_bands`.
"""
function frozen_documents!(b, ctx, op)
    isempty(op.frozen) && return Any[]
    haskey(b.frozen, op.name) && return b.frozen[op.name]
    m = ctx.model
    extra = Any[(frozen_state(k), frozen_declaration(m, o), frozen_rhs(m, o)) for (k, o) in enumerate(op.frozen)]
    fr = DerivedDocument(_rhs_document(m, op.name, Term[], false; extra, extra_refs = op.frozen))
    docs = Any[("frozen_residual", fr)]
    if op.nfreeze > 0
        doc = residual_document(m, op.name, op)
        # Only picard's freeze observeds are held (§6.5).
        ren = Dict{String, String}(o => frozen_state(k) for (k, o) in enumerate(op.frozen) if k <= op.nfreeze)
        _rename_refs!(doc["models"], ren)
        v = component_model!(doc["models"], FROZEN_COMPONENT)
        for (k, o) in enumerate(op.frozen)
            x = frozen_state(k)
            v["variables"][last_segment(x)] = frozen_declaration(m, o)
            push!(v["equations"], Dict{String, Any}("lhs" => Dict{String, Any}("op" => "D", "args" => Any[x], "wrt" => "t"), "rhs" => zero_rhs(m, o)))
        end
        src = DerivedDocument(doc)
        entries = document_bands(src.flat, "the frozen Jacobian of operator $(op.name)")
        b.frozen_bands[op.name] = Dict{String, Any}[band_json(en) for en in entries]
        push!(docs, ("frozen_jacobian", DerivedDocument(jacobian_document(src.flat, entries; data_sources = src.file.data_sources))))
    end
    return b.frozen[op.name] = docs
end

# ---------------------------------------------------------------------------
# The parameter products (spec/plan.md §6.6, D127)

"""The state-free observeds that bound the unknowns of `op` (D125), among its frozen observeds."""
function bound_observeds(ctx, op)
    m = ctx.model
    out = String[]
    for z in op.writes
        sv = get(m.flat.state_variables, z, nothing)
        sv === nothing && continue
        for bnd in (sv.lb, sv.ub)
            bnd isa AbstractString && String(bnd) in op.frozen && !(String(bnd) in out) && push!(out, String(bnd))
        end
    end
    return out
end

"""
§6.6: EarthSciASTDiff's `vjp` product on the parameters axis of the derived
document `src`, written as §6.2 writes the Jacobian document (no `ic`
equations, every state a `default`), and its io for the plan entry.
"""
function parameter_product(src)
    doc, io = EarthSciASTDiff.vjp_document(src.flat; wrt = :parameters, data_sources = src.file.data_sources)
    drop_ic!(doc["models"])
    default_states!(doc["models"])
    js = Dict{String, Any}(
        "lam" => Dict{String, Any}(String(k) => String(v) for (k, v) in io.lam),
        "vjp" => Dict{String, Any}(String(k) => String(v) for (k, v) in io.vjp),
    )
    return DerivedDocument(doc), js
end

"""
    product_documents!(b, ctx, op) -> (Vector{(kind, DerivedDocument)}, io)

An algebraic operator's parameter products when the run config has
`adjoint` (§6.6): `vjp` of its residual document, and `frozen_vjp` of its
frozen residual document when a bound of its unknowns references a
state-free observed.
"""
function product_documents!(b, ctx, op)
    haskey(ctx.config, "adjoint") || return (Tuple{String, Any}[], Dict{String, Any}())
    haskey(b.products, op.name) && return b.products[op.name]
    docs = Tuple{String, Any}[]
    ios = Dict{String, Any}()
    d, io = parameter_product(artifacts!(b, ctx, op).rhs)
    push!(docs, ("vjp", d))
    ios["vjp"] = io
    if !isempty(bound_observeds(ctx, op))
        fr = only(d for (k, d) in frozen_documents!(b, ctx, op) if k == "frozen_residual")
        d2, io2 = parameter_product(fr)
        push!(docs, ("frozen_vjp", d2))
        ios["frozen_vjp"] = io2
    end
    return b.products[op.name] = (docs, ios)
end
