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
state_dofs(L::DofLayout, xs) = sort!([L.offset[x] + i for x in xs for i in 1:max(1, prod(L.extents[x]; init = 1))])

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

# ---------------------------------------------------------------------------
# A document's right-hand side

"""
    BuiltDocument

A document built with EarthSciAST (`esm_problem`, native compiler): its
`f!`, default parameters and initial state (its `u0`, in its own slot
order), the number of its state slots, `perm` mapping
each canonical DOF to its slot, and the slots of every band field `J_k`
(column-major over the field's shape) for a Jacobian document.
"""
struct BuiltDocument
    f!::Any
    p::Any
    u0::Vector{Float64}
    nslots::Int
    perm::Vector{Int}
    fields::Vector{Vector{Int}}
end

"""
Build `file` (a loaded EarthSciAST document or a flattened model) with
EarthSciAST's `esm_problem` and map its state layout to canonical order;
`fields` gives each band field's extents for a Jacobian document.

The planner builds with EarthSciAST's interpreter (`compiler =
:interpreter`), not the native compiler runtimes use
(spec/operator-interface.md §8.1): EarthSciAST.jl 0.6.0's native compiler
declines a contracting `faq` inside a lifted equation (spec/plan.md §10
item 5) and evaluates a lifted `index(makearray(…), i, j, k)` term to zero,
both of which the reseact-shaped fixture has. The interpreter evaluates
the documents as written; the planner's checks are of the documents, not of
a compiler.
"""
function build_document(file, L::DofLayout, time; fields = Vector{Int}[])
    prob = EarthSciAST.esm_problem(file, time; compiler = :interpreter)
    vm = Dict{String, Int}(String(k) => Int(v) for (k, v) in prob.var_map)
    slot(name) = haskey(vm, name) ? vm[name] :
        error("EarthSciAST's state layout of a derived document has no element $name")
    perm = Int[slot(n) for n in L.names]
    fslots = [Int[slot(element_name("_jacobian.J_$k", c)) for c in cells(ext)] for (k, ext) in enumerate(fields)]
    return BuiltDocument(prob.f!, prob.p, prob.u0, length(prob.u0), perm, fslots)
end

"""The full slot vector of a canonical state `u` (other slots, band fields included, zero)."""
function _slots(B::BuiltDocument, u::AbstractVector{T}) where {T}
    U = zeros(T, B.nslots)
    U[B.perm] = u
    return U
end

"""The document's right-hand side at `(u, t)`, every slot (in EarthSciAST's order)."""
function eval_slots(B::BuiltDocument, u::AbstractVector{T}, t) where {T}
    U = _slots(B, u)
    du = zeros(T, B.nslots)
    B.f!(du, U, B.p, t)
    return du
end

"""The document's right-hand side at `(u, t)` in canonical order."""
eval_rhs(B::BuiltDocument, u, t) = eval_slots(B, u, t)[B.perm]

"""ForwardDiff's Jacobian of the document's right-hand side at `(u, t)`, canonical order (§8.2)."""
forwarddiff_jacobian(B::BuiltDocument, u::Vector{Float64}, t) = ForwardDiff.jacobian(x -> eval_rhs(B, x, t), u)

# ---------------------------------------------------------------------------
# Band Jacobians (esm-jacobian-spec §4, spec/operator-interface.md §8.3)

"""The value of a serialized column-index expression with index names bound by `bind`."""
function eval_index(c, bind::Dict{String, Int})
    c isa AbstractString && return haskey(bind, c) ? bind[c] : error("column index names $c, which the band does not bind")
    c isa Integer && return Int(c)
    c isa Real && return Int(round(c))
    op = c["op"]
    args = c["args"]
    if op == "+" && length(keys(c)) == 2
        return sum(a -> eval_index(a, bind), args)
    elseif op == "*" && length(keys(c)) == 2
        return prod(a -> eval_index(a, bind), args)
    elseif op == "-" && length(keys(c)) == 2
        return length(args) == 1 ? -eval_index(args[1], bind) : eval_index(args[1], bind) - eval_index(args[2], bind)
    end
    e = EarthSciAST.expression_from_json(c)
    return Int(round(EarthSciAST.evaluate_expr(e, Dict{String, Float64}(k => Float64(v) for (k, v) in bind))))
end

"""
    band_points(L, bands) -> Vector{NTuple{4, Int}}

Every point of every band (esm-jacobian-spec §4): `(k, f, r, c)` for band
`k`, its field cell `f` (column-major over the row cell and the contracted
cell), row DOF `r` and column DOF `c`. A column index outside the column
variable's extents is a ghost read and has no point.
"""
function band_points(L::DofLayout, bands::AbstractVector{BandEntry})
    pts = NTuple{4, Int}[]
    for (k, b) in enumerate(bands)
        rowcells = box_cells(b.rows)
        ccells = box_cells([(lo, hi) for (_, lo, hi) in b.contracted])
        fext = vcat(L.extents[b.row], Int[hi for (_, _, hi) in b.contracted])
        for cell in rowcells, cc in ccells
            bind = Dict{String, Int}(r => cell[d] for (d, r) in enumerate(b.row_idx) if r isa String)
            for (j, (n, _, _)) in enumerate(b.contracted)
                bind[n] = cc[j]
            end
            r = dof(L, b.row, cell)
            c = dof(L, b.col, Int[eval_index(x, bind) for x in b.col_idx])
            (r === nothing || c === nothing) && continue
            push!(pts, (k, linear_index(fext, vcat(cell, cc)), r, c))
        end
    end
    return pts
end

"""The band Jacobian at `(u, t)` as `(r, c) => value`, every band's contribution added before anything else (esm-jacobian-spec §4 additivity)."""
function band_jacobian(B::BuiltDocument, pts, u, t)
    du = eval_slots(B, u, t)
    J = Dict{Tuple{Int, Int}, Float64}()
    for (k, f, r, c) in pts
        J[(r, c)] = get(J, (r, c), 0.0) + du[B.fields[k][f]]
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
