# Pass 2's band calculus (spec/plan.md §5.2.3–§5.2.4) as pure functions over
# band lists, and the interface through which the planner gets what needs
# EarthSciASTDiff (term bands, structure, the Gershgorin bound, derived
# documents and validation). Nothing here needs EarthSciAST; the passes over
# the model are in the other files of src/planner/.

const LOCALITY_ORDER = (:pointwise, :column, :stencil, :global)
locality_rank(c::Symbol) = findfirst(==(c), LOCALITY_ORDER)
"""The coarser of two classes in the order pointwise < column < stencil < global (§5.2.4)."""
coarser(a::Symbol, b::Symbol) = locality_rank(a) >= locality_rank(b) ? a : b

"""
    Axes(horizontal, vertical)

The roles of the index sets (§5.2.1, D67): one or two horizontal axes in
`partition` order, and at most one vertical axis.
"""
struct Axes
    horizontal::Vector{String}
    vertical::Union{String, Nothing}
end

"""The role of index set `X`: `:horizontal`, `:vertical` or `:other`."""
axis_role(a::Axes, X::AbstractString) = X in a.horizontal ? :horizontal : X == a.vertical ? :vertical : :other

# ---------------------------------------------------------------------------
# Band entries

"""
    BandEntry

One entry of EarthSciASTDiff's band list (esm-jacobian-spec §4), in the form
its `serialize_band` writes, without `coef`:

- `row`, `col`: the row and column variables (flattened names);
- `rows`: per dimension of `row`, the inclusive range `(lo, hi)`; empty for a scalar row;
- `row_idx`: per dimension of `row`, an index name (`String`) or an integer;
- `col_idx`: per dimension of `col`, an expression in EarthSciAST's serialized
  form (a name `String`, an `Integer` literal, a `Float64` literal, or a
  `Dict` with `"op"` and `"args"`); empty for a scalar column;
- `contracted`: free contracted dimensions `(name, lo, hi)`, possibly empty.

Stage 2 builds these from `EarthSciASTDiff.serialize_band`'s output.
"""
struct BandEntry
    row::String
    col::String
    rows::Vector{Tuple{Int, Int}}
    row_idx::Vector{Union{String, Int}}
    col_idx::Vector{Any}
    contracted::Vector{Tuple{String, Int, Int}}
end
BandEntry(row, col, rows, row_idx, col_idx) = BandEntry(row, col, rows, row_idx, col_idx, Tuple{String, Int, Int}[])

"""A `BandEntry` from `serialize_band`'s dictionary (its `coef`, if present, is ignored)."""
function BandEntry(d::AbstractDict)
    return BandEntry(
        String(d["row"]), String(d["col"]), Tuple{Int, Int}[(Int(r[1]), Int(r[2])) for r in d["rows"]],
        Union{String, Int}[x isa AbstractString ? String(x) : Int(x) for x in d["row_idx"]], Any[d["col_idx"]...],
        haskey(d, "contracted") ? Tuple{String, Int, Int}[(String(c[1]), Int(c[2]), Int(c[3])) for c in d["contracted"]] :
            Tuple{String, Int, Int}[]
    )
end

"""An integer literal (§5.2.3): a number the flattened model holds as an integer (serialized as a JSON integer), within 64-bit signed range. A float such as `1.0` is not one."""
_is_int_literal(x) = x isa Integer && !(x isa Bool) && typemin(Int) <= x <= typemax(Int)

"""Whether a serialized index expression mentions `name`."""
_mentions(x, name) = x isa AbstractString ? x == name :
    x isa AbstractDict ? any(a -> _mentions(a, name), get(x, "args", Any[])) ||
    any(v -> _mentions(v, name), (v for (k, v) in x if k != "args" && k != "op")) :
    x isa AbstractVector ? any(a -> _mentions(a, name), x) : false

"""
The value of an integer constant (§5.2.3), or `nothing`: recursively, an
integer literal, a `-` of one or two integer constants, or a `+` or `*` of
one or more. The value is computed exactly in 64-bit signed integers; an
expression whose value or any sub-expression's value overflows is not a
constant.
"""
function _int_constant(x)
    _is_int_literal(x) && return Int(x)
    x isa AbstractDict || return nothing
    op = get(x, "op", nothing)
    op in ("+", "-", "*") || return nothing
    length(keys(x)) == 2 || return nothing          # exactly "op" and "args"
    args = x["args"]
    (args isa AbstractVector && !isempty(args)) || return nothing
    op == "-" && length(args) > 2 && return nothing
    vals = Int[]
    for a in args
        v = _int_constant(a)
        v === nothing && return nothing
        push!(vals, v)
    end
    try
        op == "+" && return foldl(Base.Checked.checked_add, vals)
        op == "*" && return foldl(Base.Checked.checked_mul, vals)
        length(vals) == 1 && return Base.Checked.checked_neg(vals[1])
        return Base.Checked.checked_sub(vals[1], vals[2])
    catch e
        e isa OverflowError && return nothing
        rethrow()
    end
end

"""
    raw_offsets(c, r, lo, hi) -> Union{Vector{Int}, Nothing}

The raw offsets of a column index `c` against the row index `r` over the
row range `[lo, hi]` (§5.2.3), or `nothing` when the reach covers the axis.
"""
function raw_offsets(c, r, lo::Int, hi::Int)
    c isa AbstractString && r isa AbstractString && c == r && return [0]
    _is_int_literal(c) && _is_int_literal(r) && c == r && return [0]
    if r isa AbstractString && c isa AbstractDict && get(c, "op", nothing) in ("+", "-") &&
            length(keys(c)) == 2 && length(c["args"]) == 2
        a, b = c["args"]
        if c["op"] == "+" && a == r && _is_int_literal(b)
            return [Int(b)]                  # r + δ
        elseif c["op"] == "+" && _is_int_literal(a) && b == r
            return [Int(a)]                  # δ + r
        elseif c["op"] == "-" && a == r && _is_int_literal(b)
            b == typemin(Int) && return nothing   # −δ overflows: anything else
            return [-Int(b)]                 # r − δ
        end
    end
    k = _int_constant(c)
    if k !== nothing
        # An offset that overflows 64-bit integers, although k is a constant, is
        # "anything else": the reach covers the axis (as the Rust planner reads it).
        sub(x, y) = try
            Base.Checked.checked_sub(x, y)
        catch e
            e isa OverflowError ? nothing : rethrow()
        end
        if _is_int_literal(r)
            o = sub(k, Int(r))
            return o === nothing ? nothing : [o]
        elseif r isa AbstractString
            a, b = sub(k, hi), sub(k, lo)
            (a === nothing || b === nothing) && return nothing
            return collect(a:b)
        end
    end
    return nothing
end

"""
`o` reduced on a horizontal axis of size `N` (§5.2.3, D66):
`((o + ⌊N/2⌋) mod N) − ⌊N/2⌋`, the remainder taken in `[0, N)`.
"""
reduce_offset(o::Int, N::Int) = mod(o + fld(N, 2), N) - fld(N, 2)

"""
    BandReach

What one band entry implies (§5.2.3): `global` when it makes the term
global outright; `covers`, the axes the reach covers; `offsets`, the offsets
per axis (reduced on horizontal axes, raw on the vertical one and on other
axes).
"""
struct BandReach
    global_::Bool
    covers::Set{String}
    offsets::Dict{String, Vector{Int}}
end

"""
    band_reach(b, shapes, axes, sizes) -> BandReach

§5.2.3 for one band entry. `shapes` gives each array variable's index-set
names. Dimensions of the row and column variables are paired by index-set
name: the `j`-th dimension of the column variable on an index set pairs with
the `j`-th dimension of the row variable on it (§5.2.3).
"""
function band_reach(b::BandEntry, shapes, axes::Axes, sizes)
    xs = get(shapes, b.row, String[])
    ys = get(shapes, b.col, String[])
    covers = Set{String}()
    offsets = Dict{String, Vector{Int}}()
    isempty(xs) && isempty(ys) && return BandReach(false, covers, offsets)       # scalar and scalar
    (isempty(xs) || isempty(ys)) && return BandReach(true, covers, offsets)     # one scalar, one array
    contracted = [c[1] for c in b.contracted]
    for (d, X) in enumerate(ys)
        occ = count(==(X), ys[1:d])
        xd = findall(==(X), xs)
        if length(xd) < occ
            push!(covers, X)                 # x has no dimension on X
            continue
        end
        xdim = xd[occ]
        c = b.col_idx[d]
        if any(n -> _mentions(c, n), contracted)
            push!(covers, X)
            continue
        end
        lo, hi = b.rows[xdim]
        raw = raw_offsets(c, b.row_idx[xdim], lo, hi)
        if raw === nothing
            push!(covers, X)
            continue
        end
        os = axis_role(axes, X) === :horizontal ? [reduce_offset(o, sizes[X]) for o in raw] : raw
        append!(get!(offsets, X, Int[]), os)
    end
    return BandReach(false, covers, offsets)
end

"""
    classify_bands(bands, shapes, axes, sizes) -> (class, radius)

§5.2.4: a term's class from its band list, with its radius per horizontal
axis when it is a stencil (`nothing` otherwise).

Offsets on an "other" axis (neither horizontal nor vertical) do not affect
the class, while a reach that covers one makes the term global (§5.2.4,
D82). No radius size check is needed: reduction keeps every horizontal
offset in [−⌊N/2⌋, ⌈N/2⌉ − 1], so lower + upper + 1 never exceeds N.
"""
function classify_bands(bands::AbstractVector{BandEntry}, shapes, axes::Axes, sizes)
    isglobal = false
    covers = Set{String}()
    offs = Dict{String, Vector{Int}}()
    for b in bands
        r = band_reach(b, shapes, axes, sizes)
        isglobal |= r.global_
        union!(covers, r.covers)
        for (X, os) in r.offsets
            append!(get!(offs, X, Int[]), os)
        end
    end
    any(X -> axis_role(axes, X) !== :vertical, covers) && (isglobal = true)
    radius = Dict{String, Tuple{Int, Int}}()
    for X in axes.horizontal
        os = get(offs, X, Int[])
        lower = maximum((-o for o in os if o < 0); init = 0)
        upper = maximum((o for o in os if o > 0); init = 0)
        radius[X] = (lower, upper)
    end
    isglobal && return (:global, nothing)
    any(X -> any(!=(0), get(offs, X, Int[])), axes.horizontal) && return (:stencil, radius)
    v = axes.vertical
    if v !== nothing && (v in covers || any(!=(0), get(offs, v, Int[])))
        return (:column, nothing)
    end
    return (:pointwise, nothing)
end

# ---------------------------------------------------------------------------
# The band source: where a term's bands come from (§5.2.2)

"""
    PlanBackend

What the planner needs from EarthSciASTDiff (M2 stage 2): a term's bands
(§5.2.2), an operator's structure (§5.4), the Gershgorin bound (§5.5.2), and
the derived documents with their validation (§6, §8). Each method may return
`nothing` for "not available in this stage"; the planner raises
[`PlannerStageError`](@ref) where it needs the value.

Methods:
- `term_bands(backend, ctx, term) -> Union{Vector{BandEntry}, Nothing}`
- `operator_structure(backend, ctx, op) -> Union{AbstractDict, Nothing}` (the
  plan's `structure` object)
- `gershgorin_bound(backend, ctx, op) -> Union{Float64, Nothing}` (`max_s ρ(u_s, p, t_s)`)
- `derived_and_validation(backend, ctx) -> Union{Tuple{AbstractDict, AbstractVector}, Nothing}`
"""
abstract type PlanBackend end

"""
    PendingBackend()

M2 stage 1's backend: EarthSciASTDiff is not available, so nothing is
derived. One case needs no bands: a model whose states are all scalars has
only scalar-row, scalar-column bands, which §5.2.3 gives no reach, so every
term is pointwise whatever its band list; `term_bands` returns an empty list
for it, which classifies identically. It returns `nothing` otherwise.
"""
struct PendingBackend <: PlanBackend end

term_bands(::PendingBackend, ctx, t) = all(s -> !haskey(ctx.model.shapes, s), ctx.model.states) ? BandEntry[] : nothing
operator_structure(::PendingBackend, ctx, op) = nothing
gershgorin_bound(::PendingBackend, ctx, op) = nothing
derived_and_validation(::PendingBackend, ctx) = nothing
