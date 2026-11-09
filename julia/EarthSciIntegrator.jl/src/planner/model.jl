# What the planner reads from the flattened model (spec/plan.md §4, §6.1.3):
# its state, parameter and observed names, the shapes of its arrays by
# index-set name, the index-set sizes, its state equations and observed
# definitions.

"""
    ForcingInfo

A data-fed parameter (spec/plan.md §4.1): its shape (index-set names, empty
for a scalar), the data source and file variable its `loader_fields` entry
names, the source's `kind`, its cadence (`"const"` or `"discrete"`) and its
effective records per sample.
"""
struct ForcingInfo
    shape::Vector{String}
    source::String
    file_variable::String
    source_kind::String
    cadence::String
    records_per_sample::Int
end

"""
    PlanModel

The flattened model as the passes read it.

- `states`: the state variables, the keys of `state_variables` that some
  equation's left-hand side differentiates, in document order;
  `state_set` holds them for lookup, and `canonical` is the same names
  sorted by code point (§4).
- `shapes`: index-set names per array variable: the declared shape, or for
  a lifted state the shape `flatten` records (§6.1.3).
- `sizes`: every index set's size, in the flattened model's `index_sets` order.
- `state_eqs`: `(equation index, dependent variable, lifted)` per state equation.
- `observed_defs`: each observed's defining right-hand side.
- `observed_rhs`: every right-hand side defining each observed, a bare,
  indexed or `faq` left-hand side alike ([`defined_observed`](@ref)).
"""
struct PlanModel
    file::EarthSciAST.EsmFile
    flat::EarthSciAST.FlattenedSystem
    contributions::Vector{Vector{Contribution}}
    states::Vector{String}
    state_set::Set{String}
    canonical::Vector{String}
    params::Vector{String}
    observeds::Vector{String}
    shapes::Dict{String, Vector{String}}
    sizes::OrderedDict{String, Int}
    lifted::Set{String}
    state_eqs::Vector{Tuple{Int, String, Bool}}
    observed_defs::Dict{String, ASTExpr}
    observed_rhs::Dict{String, Vector{ASTExpr}}
    reaction_systems::Set{String}
    # Data-fed parameters (§4.1), every one, by flattened name in code-point
    # order, and the used ones among them (§4.1: read by a term, directly or
    # through observed definitions, or by an `ic` equation), filled once the
    # terms are known ([`mark_used_forcing!`](@ref)).
    forcing::OrderedDict{String, ForcingInfo}
    used_forcing::Set{String}
end

"""The data source `name` of the loaded document, or `nothing`."""
data_source(file, name) = (file === nothing || file.data_sources === nothing) ? nothing : get(file.data_sources, name, nothing)

"""
Whether a data source is `discrete` (spec/plan.md §4.1): its `temporal` has a
`start` and a `frequency` or a `file_period`.
"""
function is_discrete(src)
    t = src.temporal
    t === nothing && return false
    has(x) = x !== nothing && x != ""
    return has(t.start) && (has(t.frequency) || has(t.file_period))
end

"""The data-fed parameters of the flattened model (§4.1), by name in code-point order."""
function forcing_info(file, flat)
    out = OrderedDict{String, ForcingInfo}()
    for lf in sort(flat.loader_fields; by = lf -> lf.name)
        v = flat.parameters[lf.name]
        src = data_source(file, lf.subkey)
        src === nothing && error("data-fed parameter $(lf.name) names data source $(lf.subkey), which the loaded model does not declare")
        shape = v.shape === nothing ? String[] : String.(v.shape)
        out[lf.name] = ForcingInfo(
            shape, lf.subkey, lf.var, src.kind, is_discrete(src) ? "discrete" : "const", lf.records_per_sample
        )
    end
    return out
end

is_forcing(m, n) = haskey(m.forcing, n)

"""The data-fed parameters among `names`, sorted (an operator's `forcing`, §5.3)."""
forcing_among(m, names) = sort!([n for n in names if is_forcing(m, n)])

is_state(m::PlanModel, n) = n in m.state_set
is_param(m::PlanModel, n) = haskey(m.flat.parameters, n)
is_observed(m::PlanModel, n) = haskey(m.flat.observed_variables, n)
is_name(m::PlanModel, n) = is_state(m, n) || is_param(m, n) || is_observed(m, n)

"""The kind of a qualified name in the flattened model: `:state`, `:parameter`, `:observed` or `nothing`."""
name_kind(m::PlanModel, n) = is_state(m, n) ? :state : is_param(m, n) ? :parameter : is_observed(m, n) ? :observed : nothing

"""
The variable a pointwise-lifted equation differentiates, or `nothing` (the
lift's left-hand side is a `faq` over `D(index(x, …))` for a state `x` the lift
recorded in `lifted_shapes`, its right-hand side the `faq` around the
contribution sum; esm-libraries-spec §4.7.5). This is how Julia identifies a
lifted equation (spec/plan.md §5.1.1).
"""
function lifted_variable(eq::Equation, lifted_shapes)
    isempty(lifted_shapes) && return nothing
    (eq.rhs isa OpExpr && eq.rhs.op == "faq" && eq.rhs.expr_body !== nothing) || return nothing
    lhs = eq.lhs
    (lhs isa OpExpr && lhs.op == "faq" && lhs.expr_body isa OpExpr) || return nothing
    d = lhs.expr_body
    (d.op == "D" && length(d.args) == 1 && d.args[1] isa OpExpr) || return nothing
    ix = d.args[1]
    (ix.op == "index" && !isempty(ix.args) && ix.args[1] isa VarExpr) || return nothing
    name = ix.args[1].name
    return haskey(lifted_shapes, name) ? name : nothing
end

"""
The variable a left-hand side differentiates, at any spelling, or `nothing`:
`D(x)`, `D(index(x, …))`, or either inside a `faq` (Rust's `differentiated`).
"""
function differentiated_variable(lhs)
    lhs isa OpExpr || return nothing
    lhs.op == "faq" && return differentiated_variable(lhs.expr_body)
    (lhs.op == "D" && !isempty(lhs.args)) || return nothing
    x = lhs.args[1]
    x isa VarExpr && return x.name
    (x isa OpExpr && x.op == "index" && !isempty(x.args) && x.args[1] isa VarExpr) || return nothing
    return x.args[1].name
end

"""The variable a non-lifted `D(x, t)` equation differentiates, or `nothing`."""
function derivative_variable(eq::Equation)
    lhs = eq.lhs
    (lhs isa OpExpr && lhs.op == "D" && length(lhs.args) == 1 && lhs.args[1] isa VarExpr) || return nothing
    return lhs.args[1].name
end

"""Every reaction-system path of the document: top-level systems and their subsystems, and reaction systems mounted as model subsystems."""
function reaction_system_paths(file::EarthSciAST.EsmFile)
    out = Set{String}()
    function rs!(path, rs)
        push!(out, path)
        for (k, s) in rs.subsystems
            rs!(path * "." * k, s)
        end
        return
    end
    function model!(path, m)
        for (k, s) in m.subsystems
            if s isa EarthSciAST.ReactionSystem
                rs!(path * "." * k, s)
            elseif s isa EarthSciAST.Model
                model!(path * "." * k, s)
            end
        end
        return
    end
    file.reaction_systems === nothing || foreach(((k, s),) -> rs!(k, s), file.reaction_systems)
    file.models === nothing || foreach(((k, m),) -> model!(k, m), file.models)
    return out
end

"""
    plan_model(inputs) -> PlanModel

The flattened model's names, shapes and equations (§4). A lifted state's
shape is the one `flatten` records on it (§6.1.3).
"""
function plan_model(inp::PlanInputs)
    flat = inp.flat
    sizes = OrderedDict{String, Int}(k => Int(s.size) for (k, s) in flat.index_sets)
    state_eqs = Tuple{Int, String, Bool}[]
    observed_defs = Dict{String, ASTExpr}()
    for (i, eq) in enumerate(flat.equations)
        lv = lifted_variable(eq, flat.lifted_shapes)
        if lv !== nothing
            push!(state_eqs, (i, lv, true))
            continue
        end
        dv = derivative_variable(eq)
        if dv !== nothing && haskey(flat.state_variables, dv)
            push!(state_eqs, (i, dv, false))
        elseif eq.lhs isa VarExpr && haskey(flat.observed_variables, eq.lhs.name)
            observed_defs[eq.lhs.name] = eq.rhs
        end
    end
    differentiated = Set(x[2] for x in state_eqs)
    states = [k for k in keys(flat.state_variables) if k in differentiated]
    shapes = Dict{String, Vector{String}}()
    for dict in (flat.state_variables, flat.parameters, flat.observed_variables), (k, v) in dict
        v.shape === nothing || isempty(v.shape) || (shapes[k] = String.(v.shape))
    end
    for k in keys(flat.lifted_shapes)
        haskey(shapes, k) || error("EarthSciAST's flatten recorded no shape for the lifted state $k (spec/plan.md §6.1.3)")
    end
    return PlanModel(
        inp.file, flat, inp.contributions, states, Set(states), sort(states), collect(keys(flat.parameters)),
        collect(keys(flat.observed_variables)), shapes, sizes, Set(keys(flat.lifted_shapes)), state_eqs,
        observed_defs, observed_definitions(flat), reaction_system_paths(inp.file),
        forcing_info(inp.file, flat), Set{String}()
    )
end

"""
    lifted_ranges(m, eq_index, var) -> (output_idx, ranges)

The `faq` wrapper of a lifted state equation as §6.1.2 writes it in a derived
document: `output_idx` copied from the flattened equation's right-hand-side
`faq`, and `ranges` built, not copied, mapping the `d`-th name of
`output_idx` to `{"from": X_d}`, `X_d` the index set of dimension `d` of the
state's shape (§6.1.3). (Before EarthSciAST 0.8, Rust's `flatten` wrote the
interval `[1, 4]` where Julia's wrote `{"from": "cell"}`; building them keeps
the documents the same in both.)
"""
function lifted_ranges(m::PlanModel, ei::Int, var::AbstractString)
    rhs = m.flat.equations[ei].rhs
    O = String[String(x) for x in rhs.output_idx]
    shape = m.shapes[var]
    length(O) == length(shape) || error("lifted equation for $var: output_idx $O does not match its shape $shape")
    G = Dict{String, Any}(O[d] => Dict{String, Any}("from" => shape[d]) for d in eachindex(O))
    return O, G
end

"""The names (states, parameters, observeds) an expression references; `t` and index names are not names (§5.3)."""
referenced_names(m::PlanModel, e::ASTExpr) = sort!([n for n in EarthSciAST.free_variables(e) if is_name(m, n)])

"""A component path: the name without its last dot-separated segment (§5.1.2)."""
function component_of(name::AbstractString)
    i = findlast('.', name)
    return i === nothing ? "" : name[1:prevind(name, i)]
end
