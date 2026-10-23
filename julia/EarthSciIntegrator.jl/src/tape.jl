# The tape, its JSON Lines form, and checkpoints (spec/tape.md).

"""A tape record: one invocation of one schedule node (spec/tape.md §1–§2)."""
abstract type TapeRecord end

"""
    LeafRecord

A `leaf` invocation: its interval `[t, t+dt]`, the accepted steps `(tₙ, hₙ)`,
the accepted and rejected counts, the return code, and the clip masks
recorded under `domain.on_violation: "clip"` (step index ⇒ mask of DOFs kept,
spec/methods.md §11).
"""
mutable struct LeafRecord <: TapeRecord
    path::String
    t::Float64
    dt::Float64
    steps::Vector{NTuple{2, Float64}}
    accepted::Int
    rejected::Int
    retcode::String
    clips::Vector{Pair{Int, Vector{Bool}}}
end

"""A `map` invocation and the mask of discrete decisions the map recorded."""
struct MapRecord <: TapeRecord
    path::String
    t::Float64
    dt::Float64
    mask::Vector{Bool}
end

"""A `subcycle` invocation and its number of substeps."""
struct SubcycleRecord <: TapeRecord
    path::String
    t::Float64
    dt::Float64
    n::Int
end

"""A macro step: its start, size, and the reason for the size (spec/run-config.md §5)."""
struct MacroRecord <: TapeRecord
    path::String
    t::Float64
    dt::Float64
    reason::String
end

"""
    Tape

The records of a forward run in execution order, indexed by path. A macro
step's record follows the invocations of its step.
"""
struct Tape
    records::Vector{TapeRecord}
    index::Dict{String, Int}
end
Tape() = Tape(TapeRecord[], Dict{String, Int}())

function Base.push!(tape::Tape, r::TapeRecord)
    key = r isa MacroRecord ? "macro:" * r.path : r.path
    haskey(tape.index, key) && error("duplicate tape path $(r.path)")
    push!(tape.records, r)
    tape.index[key] = length(tape.records)
    return tape
end
Base.getindex(tape::Tape, path::AbstractString) = tape.records[tape.index[path]]
Base.haskey(tape::Tape, path::AbstractString) = haskey(tape.index, path)
Base.length(tape::Tape) = length(tape.records)
macro_records(tape::Tape) = [r for r in tape.records if r isa MacroRecord]

"""`path(k, id, occurrence)`: `<k>/<id>[#<occurrence>]` (spec/tape.md §1)."""
invocation_path(k::Integer, id::AbstractString, occ::Integer) = occ == 0 ? "$k/$id" : "$k/$id#$occ"

# ---------------------------------------------------------------------------
# JSON Lines (spec/tape.md §5). Numbers are written in shortest round-trip form
# (Julia's `repr` of a Float64), so a tape read back is bitwise the tape written.

_jnum(x::Float64) = isfinite(x) ? repr(x) : error("non-finite number $x cannot be written to a tape")
_jnum(x::Integer) = string(x)
_jstr(s::AbstractString) = JSON.json(String(s))
_jbool(b::Bool) = b ? "true" : "false"

function _jline(io::IO, r::LeafRecord)
    steps = join(("[" * _jnum(t) * ", " * _jnum(h) * "]" for (t, h) in r.steps), ", ")
    print(io, "{\"path\": ", _jstr(r.path), ", \"node\": \"leaf\", \"t\": ", _jnum(r.t), ", \"dt\": ", _jnum(r.dt),
        ", \"steps\": [", steps, "], \"accepted\": ", r.accepted, ", \"rejected\": ", r.rejected,
        ", \"retcode\": ", _jstr(r.retcode))
    if !isempty(r.clips)
        clips = join(("[" * string(i) * ", [" * join(map(_jbool, m), ", ") * "]]" for (i, m) in r.clips), ", ")
        print(io, ", \"clips\": [", clips, "]")
    end
    print(io, "}")
end
_jline(io::IO, r::MapRecord) = print(io, "{\"path\": ", _jstr(r.path), ", \"node\": \"map\", \"t\": ", _jnum(r.t),
    ", \"dt\": ", _jnum(r.dt), ", \"mask\": [", join(map(_jbool, r.mask), ", "), "]}")
_jline(io::IO, r::SubcycleRecord) = print(io, "{\"path\": ", _jstr(r.path), ", \"node\": \"subcycle\", \"t\": ",
    _jnum(r.t), ", \"dt\": ", _jnum(r.dt), ", \"n\": ", r.n, "}")
_jline(io::IO, r::MacroRecord) = print(io, "{\"path\": ", _jstr(r.path), ", \"node\": \"macro\", \"t\": ", _jnum(r.t),
    ", \"dt\": ", _jnum(r.dt), ", \"reason\": ", _jstr(r.reason), "}")

"""Write a tape as JSON Lines, one object per invocation in execution order."""
function write_tape(io::IO, tape::Tape)
    for r in tape.records
        _jline(io, r)
        print(io, "\n")
    end
end
write_tape(path::AbstractString, tape::Tape) = open(io -> write_tape(io, tape), path, "w")
tape_string(tape::Tape) = sprint(write_tape, tape)

_f(x) = Float64(x)

"""Read a tape written by [`write_tape`](@ref)."""
function read_tape(io::IO)
    tape = Tape()
    for line in eachline(io)
        isempty(strip(line)) && continue
        d = JSON.parse(line)
        node = d["node"]
        if node == "leaf"
            steps = NTuple{2, Float64}[(_f(s[1]), _f(s[2])) for s in d["steps"]]
            clips = Pair{Int, Vector{Bool}}[Int(c[1]) => Bool.(c[2]) for c in get(d, "clips", [])]
            push!(tape, LeafRecord(d["path"], _f(d["t"]), _f(d["dt"]), steps, Int(d["accepted"]), Int(d["rejected"]),
                d["retcode"], clips))
        elseif node == "map"
            push!(tape, MapRecord(d["path"], _f(d["t"]), _f(d["dt"]), Bool.(d["mask"])))
        elseif node == "subcycle"
            push!(tape, SubcycleRecord(d["path"], _f(d["t"]), _f(d["dt"]), Int(d["n"])))
        elseif node == "macro"
            push!(tape, MacroRecord(d["path"], _f(d["t"]), _f(d["dt"]), d["reason"]))
        else
            error("unknown tape node kind '$node'")
        end
    end
    return tape
end
"""`read_tape(s::AbstractString)`: parse a tape from its JSON Lines text."""
read_tape(s::AbstractString) = read_tape(IOBuffer(s))

# ---------------------------------------------------------------------------
# Checkpoints (spec/tape.md §4)

"""
    Checkpoint

Everything needed to resume the forward run bitwise at a macro-step
boundary: the state, every leaf's controller state (spec/methods.md §8.6),
and the macro step index and time. M1 has no `every` or `solve` nodes, so
their parts are empty.
"""
struct Checkpoint
    index::Int
    t::Float64
    u::Vector{Float64}
    controllers::Dict{String, ControllerState}
end
