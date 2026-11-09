# Evaluating derived documents in canonical order (spec/plan.md §4, §5.5.1,
# §8): the DOF layout, the sampled states, a document's right-hand side and
# its ForwardDiff Jacobian through EarthSciAST's `f!`, and the band Jacobian
# assembled from a Jacobian document's band fields.

"""
    DofLayout

The canonical state order (§4): states sorted by code point, each array's
cells column-major. `offset[x]` is state `x`'s first DOF (0-based) and
`extents[x]` its extents (empty for a scalar); `names[n+1]` is DOF `n`'s
element name as EarthSciAST writes it (`x` or `x[i,j,…]`).
"""
struct DofLayout
    offset::Dict{String, Int}
    extents::Dict{String, Vector{Int}}
    names::Vector{String}
end

"""Element name `x[i,j,…]` (or `x` for a scalar), as EarthSciAST's state layout writes it."""
element_name(x::AbstractString, cell) = isempty(cell) ? String(x) : string(x, "[", join(cell, ","), "]")

"""The cells of a box of extents `ext`, column-major: first index fastest."""
cells(ext::AbstractVector{Int}) = box_cells([(1, n) for n in ext])

"""The cells of the box of inclusive ranges `[(lo, hi), …]`, column-major, as a vector."""
box_cells(rs) = isempty(rs) ? [Int[]] : vec([collect(c) for c in Iterators.product((lo:hi for (lo, hi) in rs)...)])

function DofLayout(m::PlanModel)
    offset = Dict{String, Int}()
    extents = Dict{String, Vector{Int}}()
    names = String[]
    for x in m.canonical
        ext = Int[m.sizes[X] for X in get(m.shapes, x, String[])]
        offset[x] = length(names)
        extents[x] = ext
        append!(names, element_name(x, c) for c in cells(ext))
    end
    return DofLayout(offset, extents, names)
end

ndofs(L::DofLayout) = length(L.names)

"""The 1-based column-major linear index of `cell` in a box of extents `ext`."""
function linear_index(ext::AbstractVector{Int}, cell::AbstractVector{Int})
    i, stride = 1, 1
    for d in eachindex(ext)
        i += (cell[d] - 1) * stride
        stride *= ext[d]
    end
    return i
end

"""The 1-based DOF of state `x`'s `cell`, or `nothing` when the cell is outside the state's extents (a ghost read)."""
function dof(L::DofLayout, x::AbstractString, cell::AbstractVector{Int})
    ext = L.extents[x]
    length(cell) == length(ext) || return nothing
    all(d -> 1 <= cell[d] <= ext[d], eachindex(ext)) || return nothing
    return L.offset[x] + linear_index(ext, cell)
end

"""The 1-based DOFs of every cell of the states `xs`."""
state_dofs(L::DofLayout, xs) = sort!([L.offset[x] + i for x in xs for i in 1:prod(L.extents[x]; init = 1)])

# ---------------------------------------------------------------------------
# Sampled states (§5.5.1)

"""`r_{s,n}` of §5.5.1: SplitMix64 of `z = 2³² s + n`, scaled into [0, 1)."""
function splitmix_unit(s::Integer, n::Integer)
    z = (UInt64(s) << 32) + UInt64(n)
    z += 0x9E3779B97F4A7C15
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    z = z ⊻ (z >> 31)
    return Float64(z >> 11) * 0x1p-53
end

"""
    sampled_states(u0, (t0, tf)) -> Vector{Tuple{Float64, Vector{Float64}}}

§5.5.1: `(t_s, u_s)` for s = 1, 2, 3: `t_s = t0 + (tf − t0) q_s`, `q_s` = 0.25,
0.5, 0.75, and `u_s[n] = b_n (0.5 + r_{s,n})`, `b_n` the initial state's
`u0[n]`, or 1 where that is zero.
"""
function sampled_states(u0::AbstractVector{Float64}, time::Tuple{Float64, Float64})
    t0, tf = time
    return [
        (
                t0 + (tf - t0) * q,
                [(u0[n] != 0 ? u0[n] : 1.0) * (0.5 + splitmix_unit(s, n - 1)) for n in eachindex(u0)],
            ) for (s, q) in enumerate((0.25, 0.5, 0.75))
    ]
end

"""
    sampled_forcing(names, defaults) -> Vector{Dict{String, Vector{Float64}}}

§5.5.1's sampled forcing values for s = 1, 2, 3: over the canonical forcing
order (the used data-fed parameters `names`, sorted by code point, each
column-major; `defaults[n]` its default's values in that order),
`F_s[m] = b^F_m (0.5 + r_{s+3,m})`, by parameter name.
"""
function sampled_forcing(names::AbstractVector{String}, defaults::AbstractDict)
    out = Dict{String, Vector{Float64}}[]
    for s in 1:3
        F = Dict{String, Vector{Float64}}()
        m = 0
        for n in sort(names)
            b = vec(defaults[n])
            F[n] = Float64[b[i] * (0.5 + splitmix_unit(s + 3, m + i - 1)) for i in eachindex(b)]
            m += length(b)
        end
        push!(out, F)
    end
    return out
end

# ---------------------------------------------------------------------------
# A document's right-hand side

"""
    BuiltDocument

A document built with EarthSciAST as spec/operator-interface.md §8.1 has
runtimes build it: `rhs_program(esm_problem(file, tspan))` with the native
compiler, with one workspace, its default forcing set and its parameter
vector at the defaults. The problem's own `f!` (also native) is kept for
ForwardDiff's dual numbers, which the program's Float64 workspace does not
take (§8.2). `aux` is the program's `aux` buffer at zero, of the length
its layout gives (a whole-model program has none). Also: its initial state (its `u0`, in its own slot order), the
number of its state slots, `perm` mapping each canonical DOF to its slot,
and the slots of every band field `J_k` (column-major over the field's
shape) for a Jacobian document.
"""
struct BuiltDocument
    prog::EarthSciAST.RhsProgram
    ws::Any
    fs::EarthSciAST.ForcingSet
    pvec::Vector{Float64}
    aux::Vector{Float64}
    f!::Any
    p::Any
    u0::Vector{Float64}
    nslots::Int
    perm::Vector{Int}
    fields::Vector{Vector{Int}}
    # The data-fed parameters' live arrays of the problem whose `f!` is kept
    # (spec/plan.md §6.3), by flattened name; empty without data-fed
    # parameters. ForwardDiff's evaluations write the forcing into them.
    arrays::Dict{String, Array{Float64}}
end

"""
The externally fed parameters of a flattened document: those whose `update`
has a rule with `from` (EarthSciAST's forcing slots, spec/plan.md §6.3).
"""
fed_parameters(flat::EarthSciAST.FlattenedSystem) = sort!(
    [n for (n, v) in flat.parameters if v.update !== nothing && any(u -> u.from !== nothing, v.update)]
)

"""
The value array of a data-fed parameter's `default` over its declared
`shape` (index-set names; a scalar's is one element): a number broadcast, an
inline array as EarthSciAST holds it.
"""
function default_array(flat::EarthSciAST.FlattenedSystem, n::AbstractString)
    v = flat.parameters[n]
    d = v.default
    dims = v.shape === nothing || isempty(v.shape) ? (1,) :
        Tuple(Int(flat.index_sets[X].size) for X in v.shape)
    d === nothing && error("data-fed parameter $n has no default to bind at construction (spec/plan.md §6.3)")
    d isa Real && return fill(Float64(d), dims)
    a = Array{Float64}(d)
    length(a) == prod(dims) || error("data-fed parameter $n: its default has $(length(a)) elements, its shape $dims")
    return reshape(a, dims)
end

"""Whether an `ic` equation of `flat` reads one of the names `fed`, directly or through observed definitions."""
function ic_reads(flat::EarthSciAST.FlattenedSystem, fed)
    isempty(fed) && return false
    defs = observed_definitions(flat)
    seen = Set{String}()
    stack = String[]
    for (_, e) in flat.field_ics
        append!(stack, EarthSciAST.free_variables(e))
    end
    while !isempty(stack)
        n = pop!(stack)
        n in seen && continue
        push!(seen, n)
        n in fed && return true
        for r in get(defs, n, ASTExpr[])
            append!(stack, EarthSciAST.free_variables(r))
        end
    end
    return false
end

"""
    without_ic(flat, full) -> (FlattenedSystem, u0)

`flat` without its `ic` equations, for a build whose program is used but whose
initial state is not (EarthSciAST's `rhs_program` cannot yet rebuild an `ic`
equation that reads a data-fed parameter), and the caller `u0` that build
needs: a state with no `default`, started only by an `ic` equation, would
otherwise have no starting value (`E_TREEWALK_MISSING_INITIAL_VALUE`). `full`
is the problem built from `flat` as written; its slots name the states. Each
such state is set to one by its name, which sets every element of an array.
The value is a placeholder that nothing reads: the caller takes the initial
state from `full` and evaluates the program only at states it passes. One
rather than zero only so that, were it ever evaluated there, a term dividing
by the state would not be infinite.
"""
function without_ic(flat::EarthSciAST.FlattenedSystem, full)
    slotted = Set{String}(first(split(String(k), '[')) for k in keys(full.var_map))
    u0 = Dict{String, Float64}(n => 1.0 for (n, v) in flat.state_variables if v.default === nothing && n in slotted)
    return EarthSciAST.FlattenedSystem(flat; field_ics = empty(flat.field_ics)), (isempty(u0) ? nothing : u0)
end

"""
Build `file` (a loaded EarthSciAST document or a flattened model) as
[`BuiltDocument`](@ref) describes and map its state layout to canonical
order; `fields` gives each band field's extents for a Jacobian document.
The program's `u` is the problem's flat layout, so the problem's `var_map`
names the slots of both.

A document with data-fed parameters (spec/plan.md §6.3) binds each one at
construction to its `default`, so its initial state is the planner's. Its
program is built from a problem whose data-fed parameters are live arrays
(EarthSciAST's `param_arrays`), which [`eval_slots`](@ref) fills with the
forcing ForwardDiff evaluates at. EarthSciAST's `rhs_program` cannot yet
rebuild an `ic` equation that reads a data-fed array parameter, so when one
does, the program is built without the `ic` equations, and the initial state
is taken from a second build that keeps them.
"""
function build_document(file, L::DofLayout, time; fields = Vector{Int}[])
    flat = file isa EarthSciAST.FlattenedSystem ? file : EarthSciAST.flatten(file)
    fed = fed_parameters(flat)
    arrays = Dict{String, Array{Float64}}(n => default_array(flat, n) for n in fed)
    u0prob = ic_u0 = nothing
    prob = if isempty(fed)
        EarthSciAST.esm_problem(file, time; compiler = :native)
    else
        src = flat
        if ic_reads(flat, Set(fed))
            u0prob = EarthSciAST.esm_problem(flat, time; compiler = :native)
            src, ic_u0 = without_ic(flat, u0prob)
        end
        EarthSciAST.esm_problem(src, time; compiler = :native, u0 = ic_u0, param_arrays = Dict{String, Any}(arrays))
    end
    prog = EarthSciAST.rhs_program(prob)
    EarthSciAST.n_states(prog) == length(prob.u0) ||
        error("EarthSciAST's program of a derived document has $(EarthSciAST.n_states(prog)) states, its problem $(length(prob.u0))")
    vm = Dict{String, Int}(String(k) => Int(v) for (k, v) in prob.var_map)
    slot(name) = haskey(vm, name) ? vm[name] :
        error("EarthSciAST's state layout of a derived document has no element $name")
    perm = Int[slot(n) for n in L.names]
    fslots = [Int[slot(element_name("JacobianEval.J_$k", c)) for c in cells(ext)] for (k, ext) in enumerate(fields)]
    u0 = u0prob === nothing ? prob.u0 : u0prob.u0
    if u0prob !== nothing
        Dict{String, Int}(String(k) => Int(v) for (k, v) in u0prob.var_map) == vm ||
            error("EarthSciAST's two builds of a derived document lay out its states differently")
    end
    return BuiltDocument(
        prog, EarthSciAST.rhs_workspace(prog), EarthSciAST.default_forcing_set(prog), EarthSciAST.parameter_vector(prog),
        zeros(EarthSciAST.tile_layout(prog).n_aux), prob.f!, prob.p, u0, length(prob.u0), perm, fslots, arrays
    )
end

"""The full slot vector of a canonical state `u` (other slots, band fields included, zero)."""
function _slots(B::BuiltDocument, u::AbstractVector{T}) where {T}
    U = zeros(T, B.nslots)
    U[B.perm] = u
    return U
end

"""
    eval_slots(B, u, t, F = nothing)

The document's right-hand side at `(u, t)`, every slot (in EarthSciAST's
order): the program's `rhs!`, or for dual numbers the problem's `f!`. `F`
maps data-fed parameters to their column-major values (§5.5.1's sampled
forcing values); each of the program's forcing slots takes the buffer of
its name (spec/plan.md §8), and ForwardDiff's `f!` reads them from the
problem's live arrays, which are written first.
"""
function eval_slots(B::BuiltDocument, u::AbstractVector{T}, t, F = nothing) where {T}
    U = _slots(B, u)
    du = zeros(T, B.nslots)
    if T === Float64
        fs = F === nothing ? B.fs : EarthSciAST.forcing_set(B.prog, [F[s.name] for s in EarthSciAST.forcing_slots(B.prog)])
        EarthSciAST.rhs!(du, B.prog, B.ws, U, B.aux, B.pvec, t, fs)
    else
        if F !== nothing
            for (n, a) in B.arrays
                vec(a) .= F[n]
                EarthSciAST._bump_buffer_epochs!(a)
            end
        end
        B.f!(du, U, B.p, t)
    end
    return du
end

"""The document's right-hand side at `(u, t)` in canonical order, with forcing `F` (see [`eval_slots`](@ref))."""
eval_rhs(B::BuiltDocument, u, t, F = nothing) = eval_slots(B, u, t, F)[B.perm]

struct BandModelTag end

"""
    BandPattern

An operator's band pattern (§8.2): the distinct `(row, column)` DOF pairs of
its band points, held row by row (compressed sparse rows). Row `r`'s columns
are `cols[rowptr[r]:(rowptr[r + 1] - 1)]`, ascending; `pos[i]` is the entry
of band point `i` (in the order of the points it was built from), so the band
Jacobian is assembled straight into a vector aligned with `cols`.
"""
struct BandPattern
    n::Int
    rowptr::Vector{Int}
    cols::Vector{Int}
    pos::Vector{Int}
end

"""
    BandPattern(points, n)

The pattern of the band points `points` (`(k, f, r, c)`, [`band_points`](@ref))
over `n` DOFs.
"""
function BandPattern(points::AbstractVector{NTuple{4, Int}}, n::Int)
    pairs = sort!(unique!([(r, c) for (_, _, r, c) in points]))
    rowptr = zeros(Int, n + 1)
    rowptr[1] = 1
    for (r, _) in pairs
        rowptr[r + 1] += 1
    end
    cumsum!(rowptr, rowptr)
    cols = Int[c for (_, c) in pairs]
    index = Dict{Tuple{Int, Int}, Int}(p => i for (i, p) in enumerate(pairs))
    pos = Int[index[(r, c)] for (_, _, r, c) in points]
    return BandPattern(n, rowptr, cols, pos)
end

nentries(P::BandPattern) = length(P.cols)
entry_range(P::BandPattern, r::Int) = P.rowptr[r]:(P.rowptr[r + 1] - 1)

"""
    colour_columns(P::BandPattern) -> Vector{Int}

§8.2's colouring of the pattern's columns, 0-based colours: the columns in
canonical order, each taking the lowest colour that no column before it
sharing a pattern row with it has taken (Curtis, Powell and Reid; the Rust
planner's `colour_columns`). A column in no pattern entry takes colour 0.
"""
function colour_columns(P::BandPattern)
    rows_of = [Int[] for _ in 1:P.n]
    for r in 1:P.n, i in entry_range(P, r)
        push!(rows_of[P.cols[i]], r)
    end
    # The colours each row's columns have taken so far.
    taken = [BitSet() for _ in 1:P.n]
    colour = zeros(Int, P.n)
    forbidden = BitSet()
    for c in 1:P.n
        empty!(forbidden)
        for r in rows_of[c]
            union!(forbidden, taken[r])
        end
        k = 0
        while k in forbidden
            k += 1
        end
        colour[c] = k
        for r in rows_of[c]
            push!(taken[r], k)
        end
    end
    return colour
end

"""
    forwarddiff_colours(B, u, t, P, colour, rows; chunk = 8, F = nothing) -> (JI, off)

§8.2's ForwardDiff Jacobian of the document's right-hand side at `(u, t)`, on
the pattern `P` with its column colouring `colour` ([`colour_columns`](@ref)):
each colour's columns are seeded together, in direction `colour + 1`, `chunk`
colours to a dual evaluation. `JI[i]` is the partial of row `r` in the
direction of the colour of entry `i`'s column `c`, which is
``∂f_r/∂u_c`` when no other column of that colour enters row `r`. `JI`
covers every row of `P`. `off` is `nothing`, or the first `(r, k, v)`, `r` in
`rows`, with a nonzero partial `v` in the direction of a colour `k` (0-based)
that no pattern column of row `r` has: a dependence the bands omit.
"""
function forwarddiff_colours(
        B::BuiltDocument, u::Vector{Float64}, t, P::BandPattern, colour::Vector{Int}, rows;
        chunk::Int = 8, F = nothing
    )
    # The chunk width is a type parameter past this function barrier.
    return _forwarddiff_colours(Val(chunk), B, u, t, P, colour, rows, F)
end

function _forwarddiff_colours(
        ::Val{K}, B::BuiltDocument, u::Vector{Float64}, t, P::BandPattern, colour::Vector{Int}, rows, F
    ) where {K}
    n = length(u)
    ncolours = isempty(colour) ? 0 : maximum(colour) + 1
    D = ForwardDiff.Dual{BandModelTag, Float64, K}
    zero_partials = ForwardDiff.Partials(ntuple(_ -> 0.0, Val(K)))
    x = D[D(u[j], zero_partials) for j in 1:n]
    members = [Int[] for _ in 1:ncolours]
    for c in 1:n
        push!(members[colour[c] + 1], c)
    end
    checked = falses(n)
    for r in rows
        checked[r] = true
    end
    JI = zeros(nentries(P))
    has = falses(K)
    off = nothing
    for k0 in 0:K:(ncolours - 1)
        kend = min(k0 + K, ncolours) - 1
        # Seed this chunk's columns only; the previous chunk's are unseeded.
        for k in k0:kend, c in members[k + 1]
            x[c] = D(u[c], ForwardDiff.Partials(ntuple(i -> i == k - k0 + 1 ? 1.0 : 0.0, Val(K))))
        end
        du = eval_rhs(B, x, t, F)
        for k in k0:kend, c in members[k + 1]
            x[c] = D(u[c], zero_partials)
        end
        for r in 1:n
            rng = entry_range(P, r)
            (isempty(rng) && !checked[r]) && continue
            p = ForwardDiff.partials(du[r])
            fill!(has, false)
            for i in rng
                k = colour[P.cols[i]]
                k0 <= k <= kend || continue
                JI[i] = p[k - k0 + 1]
                has[k - k0 + 1] = true
            end
            if checked[r] && off === nothing
                for j in 1:(kend - k0 + 1)
                    v = p[j]
                    (v != 0 && !has[j]) && (off = (r, k0 + j - 1, v); break)
                end
            end
        end
        off === nothing || break
    end
    return JI, off
end

# ---------------------------------------------------------------------------
# Band Jacobians (esm-jacobian-spec §4, spec/operator-interface.md §8.3)

"""
A serialized column index or guard as an expression to evaluate with
[`eval_index`](@ref): a name `String` or a number as itself, anything else
parsed by EarthSciAST once.
"""
index_expr(c) = c isa AbstractString || c isa Real ? c : EarthSciAST.expression_from_json(c)

"""
`e` with every literal `index(const, …)` gather (a connectivity table)
replaced by its value at `bind`, as EarthSciASTDiff's `_fold_const_gathers`
does before evaluating a column index or a guard (esm-jacobian-spec §4).
"""
function fold_const_gathers(e, bind::Dict{String, Float64})
    e isa EarthSciAST.OpExpr || return e
    if e.op == "index" && e.args[1] isa EarthSciAST.OpExpr && e.args[1].op == "const"
        v = e.args[1].value
        for x in e.args[2:end]
            v = v[eval_index(x, bind)]
        end
        return v isa Integer ? EarthSciAST.IntExpr(Int64(v)) : EarthSciAST.NumExpr(Float64(v))
    end
    return EarthSciAST.map_children(x -> fold_const_gathers(x, bind), e)
end

"""The value of a column-index or guard expression ([`index_expr`](@ref)) with index names bound by `bind`."""
function eval_value(e, bind::Dict{String, Float64})
    e isa AbstractString && return haskey(bind, e) ? bind[e] : error("column index names $e, which the band does not bind")
    e isa Real && return Float64(e)
    return EarthSciAST.evaluate_expr(fold_const_gathers(e, bind), bind)
end

"""The integer value of a column-index expression ([`index_expr`](@ref)) with index names bound by `bind`."""
eval_index(e, bind::Dict{String, Float64}) = Int(round(eval_value(e, bind)))

"""
    band_points(L, bands) -> Vector{NTuple{4, Int}}

Every point of every band (esm-jacobian-spec §4), as EarthSciASTDiff's
`_all_points` enumerates them: `(k, f, r, c)` for band `k`, its field cell
`f` (column-major over the row cell and the contracted cell), row DOF `r`
and column DOF `c`. A cell the band's `where` guard excludes is not a
point; a column index outside the column variable's extents is a ghost
read and has no point either.
"""
function band_points(L::DofLayout, bands::AbstractVector{BandEntry})
    pts = NTuple{4, Int}[]
    for (k, b) in enumerate(bands)
        rowcells = box_cells(b.rows)
        ccells = box_cells([(lo, hi) for (_, lo, hi) in b.contracted])
        fext = vcat(L.extents[b.row], Int[hi for (_, _, hi) in b.contracted])
        cidx = Any[index_expr(x) for x in b.col_idx]
        guard = b.guard === nothing ? nothing : index_expr(b.guard)
        for cell in rowcells, cc in ccells
            bind = Dict{String, Float64}(r => cell[d] for (d, r) in enumerate(b.row_idx) if r isa String)
            for (j, (n, _, _)) in enumerate(b.contracted)
                bind[n] = cc[j]
            end
            guard === nothing || eval_value(guard, bind) != 0 || continue
            r = dof(L, b.row, cell)
            c = dof(L, b.col, Int[eval_index(x, bind) for x in cidx])
            (r === nothing || c === nothing) && continue
            push!(pts, (k, linear_index(fext, vcat(cell, cc)), r, c))
        end
    end
    return pts
end

"""The band Jacobian at `(u, t)` as `(r, c) => value`, every band's contribution added before anything else (esm-jacobian-spec §4 additivity)."""
function band_jacobian(B::BuiltDocument, pts, u, t, F = nothing)
    du = eval_slots(B, u, t, F)
    J = Dict{Tuple{Int, Int}, Float64}()
    for (k, f, r, c) in pts
        J[(r, c)] = get(J, (r, c), 0.0) + du[B.fields[k][f]]
    end
    return J
end

"""
    band_values(B, P, pts, u, t, F = nothing) -> Vector{Float64}

[`band_jacobian`](@ref) on the pattern `P` built from the points `pts`: the
band Jacobian's value at each entry of `P`, every band's contribution added.
"""
function band_values(B::BuiltDocument, P::BandPattern, pts, u, t, F = nothing)
    du = eval_slots(B, u, t, F)
    J = zeros(nentries(P))
    for (i, (k, f, _, _)) in enumerate(pts)
        J[P.pos[i]] += du[B.fields[k][f]]
    end
    return J
end

"""The Gershgorin bound (§5.4): `max_r Σ_c |J_rc|` over the rows `rows`."""
function gershgorin(J::AbstractDict, rows)
    sums = Dict{Int, Float64}(r => 0.0 for r in rows)
    for ((r, _), v) in J
        haskey(sums, r) && (sums[r] += abs(v))
    end
    return maximum(values(sums); init = 0.0)
end
