# Operator references (esm-spec §9.6.4, spec/plan.md §4 step 5, §6.1.5): the
# one place the planner handles `apply_operator` nodes. The planner never
# expands a reference. The flattened model keeps them, each resolving against
# the merged registry `flatten` carries (`template_registry`, esm-spec §9.6.4
# rule 7), whose bodies `flatten` has closed: a body's free names are trailing
# `params`, bound at every reference to the site's own (flattened) names. So
#
# - the names a reference reads are the names in its `bindings`, which every
#   walk of an expression's children already visits (`free_variables`,
#   `referenced_names`);
# - a reference is one summand when a contribution is split into terms
#   (§5.1.1): it is neither `+` nor `-`;
# - a derived document carries, in each top-level component whose subtree
#   holds a reference, an `operators` block with the definitions its
#   references reach, directly or through bodies (§6.1.5).
#
# What this layer reads of a definition, it reads once per definition: the
# definitions its body references (`callees`) are memoized by name.

"""The `op` of an operator reference (esm-spec §9.6.2)."""
const APPLY_OPERATOR = "apply_operator"

"""
    OperatorRefs(registry)

The flattened model's merged operator registry (`FlattenedSystem.template_registry`:
definition name => its JSON object) and the per-definition facts read from
it, memoized by name.
"""
struct OperatorRefs
    registry::OrderedDict{String, Any}
    callees::Dict{String, Vector{String}}
end

OperatorRefs(registry::AbstractDict) =
    OperatorRefs(OrderedDict{String, Any}(String(k) => v for (k, v) in registry), Dict{String, Vector{String}}())

"""The names of the definitions referenced in the JSON form `x` (an expression, or a definition's body), in first-seen order."""
function referenced_definitions!(out::Vector{String}, x)
    if x isa AbstractDict
        get(x, "op", nothing) == APPLY_OPERATOR && (n = x["name"]; n in out || push!(out, String(n)))
        for v in values(x)
            referenced_definitions!(out, v)
        end
    elseif x isa AbstractVector
        for v in x
            referenced_definitions!(out, v)
        end
    end
    return out
end

"""The definitions the body of definition `name` references directly (computed once per definition)."""
function callees(r::OperatorRefs, name::AbstractString)
    return get!(r.callees, String(name)) do
        haskey(r.registry, name) || error(
            "an operator reference names $name, which the flattened model's registry does not hold (esm-spec §9.6.4 rule 7)"
        )
        referenced_definitions!(String[], get(r.registry[name], "body", nothing))
    end
end

"""Every definition reachable from the definitions `roots` through the bodies' references, sorted by code point."""
function reachable_definitions(r::OperatorRefs, roots)
    seen = Set{String}()
    stack = String[String(n) for n in roots]
    while !isempty(stack)
        n = pop!(stack)
        n in seen && continue
        push!(seen, n)
        append!(stack, callees(r, n))
    end
    return sort!(collect(seen))
end

"""
The entry a derived document writes for definition `name`: the registry's
entry as `flatten` holds it, materialized as esm-spec §9.6.4 rule 5 writes a
referenced `match` rule (without `match`, `where` and `priority`, with
`"reference_only": true`) should the registry hold one.
"""
function materialized_entry(r::OperatorRefs, name::AbstractString)
    e = r.registry[name]
    haskey(e, "match") || return e
    out = Dict{String, Any}(k => v for (k, v) in e if !(k in ("match", "where", "priority")))
    out["reference_only"] = true
    return out
end

"""
    attach_operators!(models, r) -> models

§6.1.5: give each top-level component of the derived document's component
tree `models` (JSON, as written) whose subtree holds an operator reference
an `operators` block with every definition its references reach. A
component with no reference gets none.
"""
function attach_operators!(models::AbstractDict, r::OperatorRefs)
    isempty(r.registry) && return models
    for (_, comp) in models
        delete!(comp, "operators")
        names = referenced_definitions!(String[], comp)
        isempty(names) && continue
        comp["operators"] = Dict{String, Any}(n => materialized_entry(r, n) for n in reachable_definitions(r, names))
    end
    return models
end
