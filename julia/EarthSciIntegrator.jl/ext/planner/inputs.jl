# Reading the inputs (spec/plan.md §4 steps 1–6): the run config, the model,
# its load and flatten with EarthSciAST, and the M2 scope.

"""
    PlanInputs

What §4 steps 1–5 read: the run config (parsed, with its bytes for the hash
of §7.7), the model as loaded and as flattened, and the flattened
equations' contributions.
"""
struct PlanInputs
    config_path::String
    config_bytes::Vector{UInt8}
    config::AbstractDict
    model_path::String            # absolute
    model_bytes::Vector{UInt8}
    file::EarthSciAST.EsmFile
    flat::EarthSciAST.FlattenedSystem
    contributions::Vector{Vector{EarthSciAST.Contribution}}
end

# ---------------------------------------------------------------------------
# solver.external (§4 step 4): the run config's own location written as
# esm-spec §2.2.6 writes a file location.

"""Dot segments removed from an absolute path (RFC 3986 §5.2.4), lexically."""
function _remove_dot_segments(path::AbstractString)
    input = String(path)
    out = String[]
    # RFC 3986 §5.2.4 over '/'-separated segments of an absolute path.
    segs = split(input, '/')
    trailing = false
    for (i, s) in enumerate(segs)
        i == 1 && continue                    # the empty segment before the leading '/'
        last = i == length(segs)
        if s == "."
            trailing = last
        elseif s == ".."
            isempty(out) || pop!(out)
            trailing = last
        else
            push!(out, s)
            trailing = false
        end
    end
    return "/" * join(out, "/") * (trailing ? "/" : "")
end

"""`file://` and the absolute path, each UTF-8 byte outside `A-Za-z0-9-._~/:` written `%XX` (esm-spec §2.2.6)."""
function _file_location(abs_path::AbstractString)
    io = IOBuffer()
    print(io, "file://")
    for b in codeunits(String(abs_path))
        if (UInt8('A') <= b <= UInt8('Z')) || (UInt8('a') <= b <= UInt8('z')) ||
           (UInt8('0') <= b <= UInt8('9')) || b in codeunits("-._~/:")
            write(io, b)
        else
            print(io, '%', uppercase(string(b; base = 16, pad = 2)))
        end
    end
    return String(take!(io))
end

"""The run config's location as esm-spec §2.2.6 writes a file: lexical, never `realpath`."""
function run_config_location(path::AbstractString)
    p = replace(String(path), '\\' => '/')
    startswith(p, "/") || (p = replace(pwd(), '\\' => '/') * "/" * p)
    return _file_location(_remove_dot_segments(p))
end

# ---------------------------------------------------------------------------

_load_error_message(e) = sprint(showerror, e)

"""
    read_inputs(config_path) -> PlanInputs

spec/plan.md §4 steps 1–5, refusing with the step's code at its pointer.
"""
function read_inputs(config_path::AbstractString)
    # Step 1: the run config.
    bytes = read(config_path)
    config = try
        JSON.parse(String(copy(bytes)))
    catch e
        refuse("run_config_invalid", "", "the run config is not JSON: " * sprint(showerror, e))
    end
    config isa AbstractDict || refuse("run_config_invalid", "", "the run config is not a JSON object")
    err = schema_errors(run_config_schema(), config)
    # The schema's failure is reported at the document root: which of a
    # oneOf's branches "the" error is in has no single answer.
    err === nothing || refuse("run_config_invalid", "", "the run config does not validate against run-config.schema.json: $err")
    v = config["esm_integrator"]
    VersionNumber(v) == SPEC_VERSION ||
        refuse("version_unsupported", "/esm_integrator", "esm_integrator $v; this planner implements $SPEC_VERSION")
    # Step 2: the model path.
    model = String(config["model"])
    startswith(model, "toy:") &&
        refuse("toy_model_not_plannable", "/model", "'$model' names a hand-written toy problem, which only the conformance harness runs")
    model_path = normpath(joinpath(dirname(abspath(config_path)), model))
    # Step 3: load.
    isfile(model_path) || refuse("model_load_failed", "/model", "no model file at $model_path")
    model_bytes = read(model_path)
    file = try
        EarthSciAST.load_path(model_path)
    catch e
        # A solver.time_span EarthSciAST rejects at load takes its rule-14
        # code (spec/plan.md §7.2).
        doc = try
            JSON.parse(String(copy(model_bytes)))
        catch
            nothing
        end
        r = model_span_refusal(doc)
        r === nothing || throw(r)
        refuse("model_load_failed", "/model", "EarthSciAST refused to load $model: " * _load_error_message(e))
    end
    # Step 4: solver.external.
    ext = try
        EarthSciAST.solver_external_location(file; base_path = dirname(model_path))
    catch e
        refuse("model_load_failed", "/model", "EarthSciAST refused the model's solver.external: " * _load_error_message(e))
    end
    if ext !== nothing
        mine = run_config_location(abspath(config_path))
        ext == mine || refuse("solver_external_mismatch", "/model",
            "the model's solver.external resolves to $ext, not to this run config ($mine)")
    end
    # Step 5: flatten, expand surviving template references, lower table lookups
    # (enums were lowered at load, EarthSciAST's own pass).
    flat = try
        f = EarthSciAST.flatten(file)
        f = EarthSciAST.expand_flattened_refs(f)
        EarthSciAST.lower_table_lookups(f)
    catch e
        e isa PlannerRefusal && rethrow()
        refuse("model_flatten_failed", "/model", "EarthSciAST refused to flatten $model: " * _load_error_message(e))
    end
    return PlanInputs(String(config_path), bytes, config, model_path, model_bytes, file, flat,
        EarthSciAST.equation_contributions(flat))
end

"""
    check_scope(flat)

spec/plan.md §4 step 6: refuse, in this order, algebraic variables, events,
parameter update rules, loader fields, and an index set that is not an
`interval` with an integer size.
"""
function check_scope(flat::EarthSciAST.FlattenedSystem)
    isempty(flat.algebraic_variables) || refuse("unsupported_algebraic_equation", "/model",
        "the flattened model has algebraic unknowns ($(join(keys(flat.algebraic_variables), ", "))); algebraic operators need U6 (M4b)")
    (isempty(flat.continuous_events) && isempty(flat.discrete_events)) || refuse("unsupported_event", "/model",
        "the flattened model has $(length(flat.continuous_events)) continuous and $(length(flat.discrete_events)) discrete events; " *
        "condition and map operators need U6 (M4b)")
    # §4 step 6: at the parameter-update place, unsupported_crossing_update when
    # any parameter has a crossing rule (reported for the first such
    # parameter), else unsupported_parameter_update; algebraic equations and
    # events come first whatever the rules.
    kinds(v) = v.update === nothing ? String[] : [u.kind for u in v.update]
    updated = [(name, kinds(v)) for (name, v) in flat.parameters if !isempty(kinds(v))]
    if !isempty(updated)
        i = something(findfirst(u -> "crossing" in u[2], updated), 1)
        name, ks = updated[i]
        code = "crossing" in ks ? "unsupported_crossing_update" : "unsupported_parameter_update"
        refuse(code, "/model", "parameter $name has update rules of kind $(join(ks, ", ")); none is planned at M2 (D75)")
    end
    isempty(flat.loader_fields) || refuse("unsupported_forcing", "/model",
        "the flattened model reads data-loader fields; forcing epochs arrive at M3")
    for (name, s) in flat.index_sets
        (s.kind == "interval" && s.size !== nothing) || refuse("unsupported_index_set", "/model",
            "index set $name is a $(s.kind)$(s.size === nothing ? " without an integer size" : ""); M2 plans intervals only")
    end
    return nothing
end
