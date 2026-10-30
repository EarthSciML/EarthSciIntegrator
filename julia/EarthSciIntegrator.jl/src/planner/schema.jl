# A JSON Schema (draft 2020-12) validator for the keywords spec/schemas use,
# so the planner can refuse a run config the schema rejects (spec/plan.md §4
# step 1, `run_config_invalid`). JSONSchema.jl (1.5 and 1.6) skips
# `unevaluatedProperties`, on which the run-config schema's leaf and coupled
# nodes rely, so this file implements the vocabulary itself: the applicators
# (`$ref` within the document, `allOf`, `anyOf`, `oneOf`, `not`, `if`/`then`/
# `else`, `properties`, `patternProperties`, `additionalProperties`,
# `propertyNames`, `items`, `prefixItems`, `unevaluatedProperties`) and the
# assertions (`type`, `enum`, `const`, `pattern`, `minLength`, `maxLength`,
# `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `minItems`,
# `maxItems`, `uniqueItems`, `required`, `minProperties`, `maxProperties`,
# `dependentRequired`). Any other keyword is an error, so a schema change that
# needs more is noticed rather than silently ignored.

const _SCHEMA_ANNOTATIONS = Set(["\$schema", "\$id", "title", "description", "\$defs", "\$comment", "examples", "default"])
const _SCHEMA_KEYWORDS = Set(
    [
        "\$ref", "allOf", "anyOf", "oneOf", "not", "if", "then", "else", "properties",
        "patternProperties", "additionalProperties", "propertyNames", "items", "prefixItems", "unevaluatedProperties",
        "type", "enum", "const", "pattern", "minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum",
        "exclusiveMaximum", "minItems", "maxItems", "uniqueItems", "required", "minProperties", "maxProperties",
        "dependentRequired",
    ]
)

struct SchemaValidator
    root::Any
    regexes::Dict{String, Regex}
end
SchemaValidator(root) = SchemaValidator(root, Dict{String, Regex}())

_isnum(x) = x isa Real && !(x isa Bool)

# JSON equality: numbers by value (1 == 1.0), containers structurally.
_json_equal(a, b) = _isnum(a) && _isnum(b) ? a == b :
    (a isa AbstractString && b isa AbstractString) ? a == b :
    (a isa Bool && b isa Bool) ? a == b :
    (a === nothing && b === nothing) ? true :
    (a isa AbstractVector && b isa AbstractVector) ? (length(a) == length(b) && all(_json_equal(x, y) for (x, y) in zip(a, b))) :
    (a isa AbstractDict && b isa AbstractDict) ? (
        length(a) == length(b) &&
        all(k -> haskey(b, k) && _json_equal(a[k], b[k]), keys(a))
    ) : false

function _json_type_ok(x, t::AbstractString)
    t == "object" && return x isa AbstractDict
    t == "array" && return x isa AbstractVector
    t == "string" && return x isa AbstractString
    t == "boolean" && return x isa Bool
    t == "null" && return x === nothing
    t == "number" && return _isnum(x)
    t == "integer" && return _isnum(x) && isfinite(x) && isinteger(x)
    throw(ArgumentError("unknown JSON Schema type '$t'"))
end

function _regex(v::SchemaValidator, p::AbstractString)
    return get!(v.regexes, p) do
        # ECMA-262 semantics for the anchors the schemas use: `$` matches only
        # at the very end (PCRE would also match before a final newline).
        Regex(p, Base.DEFAULT_COMPILER_OPTS | Base.PCRE.DOLLAR_ENDONLY | Base.PCRE.UCP, Base.DEFAULT_MATCH_OPTS)
    end
end

function _resolve_ref(v::SchemaValidator, ref::AbstractString)
    startswith(ref, "#") || throw(ArgumentError("only document-local \$ref is supported, not '$ref'"))
    node = v.root
    for tok in split(ref[2:end], '/'; keepempty = false)
        tok = replace(tok, "~1" => "/", "~0" => "~")
        node = node isa AbstractVector ? node[parse(Int, tok) + 1] : node[tok]
    end
    return node
end

"""
    _sv(v, x, schema, path) -> (ok, evaluated, err)

Validate instance `x` against `schema`. `evaluated` is the set of property
names (for an object) that a successful evaluation annotated, which
`unevaluatedProperties` reads; `err` describes the first failure.
"""
function _sv(v::SchemaValidator, x, schema, path::String)
    E = Set{String}()
    schema === true && return (true, E, "")
    schema === false && return (false, E, "$path: schema false")
    schema isa AbstractDict || throw(ArgumentError("schema is not an object at $path"))
    for k in keys(schema)
        (k in _SCHEMA_KEYWORDS || k in _SCHEMA_ANNOTATIONS) ||
            throw(ArgumentError("JSON Schema keyword '$k' is not supported by this validator"))
    end
    fail(msg) = (false, E, "$path: $msg")
    # Applicators that pass annotations up.
    if haskey(schema, "\$ref")
        ok, ev, err = _sv(v, x, _resolve_ref(v, schema["\$ref"]), path)
        ok || return (false, E, err)
        union!(E, ev)
    end
    if haskey(schema, "allOf")
        for s in schema["allOf"]
            ok, ev, err = _sv(v, x, s, path)
            ok || return (false, E, err)
            union!(E, ev)
        end
    end
    if haskey(schema, "anyOf")
        any_ok = false
        firsterr = ""
        for s in schema["anyOf"]
            ok, ev, err = _sv(v, x, s, path)
            if ok
                any_ok = true
                union!(E, ev)
            elseif isempty(firsterr)
                firsterr = err
            end
        end
        any_ok || return fail("anyOf: no branch matches ($firsterr)")
    end
    if haskey(schema, "oneOf")
        n = 0
        evs = Set{String}()
        errs = String[]
        for s in schema["oneOf"]
            ok, ev, err = _sv(v, x, s, path)
            ok ? (n += 1; evs = ev) : push!(errs, err)
        end
        n == 1 || return fail("oneOf: $n branches match" * (n == 0 ? " ($(join(errs, "; ")))" : ""))
        union!(E, evs)
    end
    if haskey(schema, "not")
        ok, _, _ = _sv(v, x, schema["not"], path)
        ok && return fail("not: the instance matches the negated schema")
    end
    if haskey(schema, "if")
        ok, ev, _ = _sv(v, x, schema["if"], path)
        if ok
            union!(E, ev)
            if haskey(schema, "then")
                ok2, ev2, err2 = _sv(v, x, schema["then"], path)
                ok2 || return (false, E, err2)
                union!(E, ev2)
            end
        elseif haskey(schema, "else")
            ok2, ev2, err2 = _sv(v, x, schema["else"], path)
            ok2 || return (false, E, err2)
            union!(E, ev2)
        end
    end
    # Assertions on any type.
    if haskey(schema, "type")
        t = schema["type"]
        ts = t isa AbstractVector ? t : [t]
        any(tt -> _json_type_ok(x, tt), ts) || return fail("type: not $(join(ts, " or "))")
    end
    if haskey(schema, "enum")
        any(e -> _json_equal(x, e), schema["enum"]) || return fail("enum: $(repr(x)) is not one of $(schema["enum"])")
    end
    if haskey(schema, "const")
        _json_equal(x, schema["const"]) || return fail("const: $(repr(x)) is not $(repr(schema["const"]))")
    end
    if x isa AbstractString
        n = length(x)
        haskey(schema, "minLength") && n < schema["minLength"] && return fail("minLength")
        haskey(schema, "maxLength") && n > schema["maxLength"] && return fail("maxLength")
        haskey(schema, "pattern") && !occursin(_regex(v, schema["pattern"]), x) &&
            return fail("pattern: $(repr(x)) does not match $(schema["pattern"])")
    end
    if _isnum(x)
        haskey(schema, "minimum") && x < schema["minimum"] && return fail("minimum")
        haskey(schema, "maximum") && x > schema["maximum"] && return fail("maximum")
        haskey(schema, "exclusiveMinimum") && x <= schema["exclusiveMinimum"] && return fail("exclusiveMinimum")
        haskey(schema, "exclusiveMaximum") && x >= schema["exclusiveMaximum"] && return fail("exclusiveMaximum")
    end
    if x isa AbstractVector
        n = length(x)
        haskey(schema, "minItems") && n < schema["minItems"] && return fail("minItems")
        haskey(schema, "maxItems") && n > schema["maxItems"] && return fail("maxItems")
        if get(schema, "uniqueItems", false) === true
            for i in 1:n, j in (i + 1):n
                _json_equal(x[i], x[j]) && return fail("uniqueItems")
            end
        end
        np = 0
        if haskey(schema, "prefixItems")
            for (i, s) in enumerate(schema["prefixItems"])
                i > n && break
                ok, _, err = _sv(v, x[i], s, "$path[$(i - 1)]")
                ok || return (false, E, err)
                np = i
            end
        end
        if haskey(schema, "items")
            for i in (np + 1):n
                ok, _, err = _sv(v, x[i], schema["items"], "$path[$(i - 1)]")
                ok || return (false, E, err)
            end
        end
    end
    if x isa AbstractDict
        n = length(x)
        haskey(schema, "minProperties") && n < schema["minProperties"] && return fail("minProperties")
        haskey(schema, "maxProperties") && n > schema["maxProperties"] && return fail("maxProperties")
        if haskey(schema, "required")
            for r in schema["required"]
                haskey(x, r) || return fail("required: '$r' is missing")
            end
        end
        if haskey(schema, "dependentRequired")
            for (k, deps) in schema["dependentRequired"]
                haskey(x, k) || continue
                for d in deps
                    haskey(x, d) || return fail("dependentRequired: '$k' needs '$d'")
                end
            end
        end
        if haskey(schema, "propertyNames")
            for k in keys(x)
                ok, _, err = _sv(v, String(k), schema["propertyNames"], "$path.$k (name)")
                ok || return (false, E, err)
            end
        end
        props = get(schema, "properties", nothing)
        pats = get(schema, "patternProperties", nothing)
        local_ev = Set{String}()
        if props !== nothing
            for (k, s) in props
                haskey(x, k) || continue
                ok, _, err = _sv(v, x[k], s, "$path.$k")
                ok || return (false, E, err)
                push!(local_ev, k)
            end
        end
        if pats !== nothing
            for (p, s) in pats, k in keys(x)
                occursin(_regex(v, p), k) || continue
                ok, _, err = _sv(v, x[k], s, "$path.$k")
                ok || return (false, E, err)
                push!(local_ev, k)
            end
        end
        if haskey(schema, "additionalProperties")
            for k in keys(x)
                k in local_ev && continue
                (props !== nothing && haskey(props, k)) && continue
                ok, _, err = _sv(v, x[k], schema["additionalProperties"], "$path.$k")
                ok || return (
                    false, E, (
                        schema["additionalProperties"] === false ?
                            "$path: additionalProperties: '$k' is not allowed" : err
                    ),
                )
                push!(local_ev, k)
            end
        end
        union!(E, local_ev)
        if haskey(schema, "unevaluatedProperties")
            for k in keys(x)
                k in E && continue
                ok, _, err = _sv(v, x[k], schema["unevaluatedProperties"], "$path.$k")
                ok || return (
                    false, E, (
                        schema["unevaluatedProperties"] === false ?
                            "$path: unevaluatedProperties: '$k' is not allowed" : err
                    ),
                )
                push!(E, k)
            end
        end
    end
    return (true, E, "")
end

"""
    schema_errors(schema, x) -> Union{Nothing, String}

`nothing` when `x` validates against the draft 2020-12 `schema` (a parsed
JSON object), else a description of the first failure found.
"""
function schema_errors(schema, x)
    ok, _, err = _sv(SchemaValidator(schema), x, schema, "\$")
    return ok ? nothing : err
end

const _RUN_CONFIG_SCHEMA = Ref{Any}(nothing)

"""The run-config schema, spec/schemas/run-config.schema.json (read once, from the package's `data/` copy)."""
function run_config_schema()
    if _RUN_CONFIG_SCHEMA[] === nothing
        path = joinpath(@__DIR__, "..", "..", "data", "schemas", "run-config.schema.json")
        _RUN_CONFIG_SCHEMA[] = JSON.parsefile(path)
    end
    return _RUN_CONFIG_SCHEMA[]
end
