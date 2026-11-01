# Planner refusals (spec/plan.md §4, spec/planner-errors.json, D79).

"""
    PlannerRefusal(code, at, msg)

The planner refuses its inputs: `code` is one of spec/planner-errors.json's
codes, `at` the JSON pointer into the run config of the offending key
(`/model` for a refusal about the model), and `msg` a message for people.
Conformance fixtures compare `code` and `at` only.
"""
struct PlannerRefusal <: Exception
    code::String
    at::String
    msg::String
end
Base.showerror(io::IO, e::PlannerRefusal) = print(io, "PlannerRefusal(", e.code, " at ", repr(e.at), "): ", e.msg)

refuse(code::AbstractString, at::AbstractString, msg::AbstractString) = throw(PlannerRefusal(code, at, msg))

"""
    PlannerStageError(what)

A part of the plan that needs EarthSciASTDiff (term and operator bands,
structure, the Gershgorin bound, derived documents and validation), asked of
a [`PlanBackend`](@ref) that cannot give it, such as [`PendingBackend`](@ref)
or a test backend. Raised instead of writing a value that would only look
right. The production backend, [`DiffBackend`](@ref), gives every part.
"""
struct PlannerStageError <: Exception
    what::String
end
Base.showerror(io::IO, e::PlannerStageError) =
    print(io, "PlannerStageError: the planner's backend cannot give what EarthSciASTDiff gives: ", e.what)

"""JSON pointer escaping of one reference token (RFC 6901 §3)."""
pointer_token(s) = replace(string(s), "~" => "~0", "/" => "~1")

"""`base` extended by reference tokens."""
jptr(base::AbstractString, toks...) = string(base, (string("/", pointer_token(t)) for t in toks)...)
