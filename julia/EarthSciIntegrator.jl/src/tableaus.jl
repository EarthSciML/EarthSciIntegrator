# Coefficient tables (spec/methods.md §1, D54).
#
# How the package finds them: the tables are read from the package's own
# `data/tableaus/*.json` when the package is precompiled, and each file is
# registered with `include_dependency`, so editing or regenerating a table
# invalidates the precompile cache. The numbers then live in the package image
# as the constant `TABLEAUS`; nothing reads the files at run time.
#
# `data/tableaus` is a copy of the repository's `spec/tableaus`, which is where
# tables are edited: a registered package ships only this directory, so it
# cannot read spec/. Run `tools/sync-package-data.sh` after changing a table;
# CI fails while the copy differs.

"""Controller constants of a method's table (spec/methods.md §8)."""
struct ControllerConstants
    beta1::Float64
    beta2::Float64
    gamma::Float64
    qmin::Float64
    qmax::Float64
    qmax_first_step::Float64
    qsteady_min::Float64
    qsteady_max::Float64
    failfactor::Float64
    qoldinit::Float64
    pid_beta3::Float64
    pid_accept_safety::Float64
end

"""
    Tableau

One method's coefficient table, as `spec/tableaus/<name>.json` holds it
(spec/schemas/tableau.schema.json). Fields a method's form does not use are
`nothing`. Matrices are row-major in the file and ordinary Julia matrices
here (`A[i, j]` is aᵢⱼ). The Shu–Osher `alpha` and `beta` are lower-triangular
`s × s` matrices with `alpha[i, j+1]` = αᵢⱼ (`j` 0-based as in the spec).
"""
struct Tableau
    name::String
    family::Symbol
    form::Symbol
    stages::Int
    order::Int
    embedded_order::Union{Nothing, Int}
    error_estimate::Symbol
    first_same_as_last::Bool
    stiffly_accurate::Bool
    explicit_first_stage::Union{Nothing, Bool}
    A::Union{Nothing, Matrix{Float64}}
    C::Union{Nothing, Matrix{Float64}}
    b::Union{Nothing, Vector{Float64}}
    btilde::Union{Nothing, Vector{Float64}}
    c::Union{Nothing, Vector{Float64}}
    d::Union{Nothing, Vector{Float64}, Float64}
    gamma::Union{Nothing, Float64}
    c32::Union{Nothing, Float64}
    alpha::Union{Nothing, Matrix{Float64}}
    beta::Union{Nothing, Matrix{Float64}}
    embedded_alpha::Union{Nothing, Vector{Float64}}
    embedded_beta::Union{Nothing, Vector{Float64}}
    embedded_scale::Union{Nothing, Float64}
    residual_sigma::Union{Nothing, Vector{Float64}}
    residual_scale::Union{Nothing, Float64}
    divided_difference::Union{Nothing, Float64}
    controller::Union{Nothing, ControllerConstants}
    ordinarydiffeq_algorithm::Union{Nothing, String}
end

_vec(x) = x === nothing ? nothing : Float64[Float64(v) for v in x]
function _mat(x)
    x === nothing && return nothing
    n, m = length(x), length(first(x))
    M = zeros(n, m)
    for i in 1:n, j in 1:m
        M[i, j] = Float64(x[i][j])
    end
    return M
end
function _triangle(x)
    x === nothing && return nothing
    s = length(x)
    M = zeros(s, s)
    for i in 1:s
        length(x[i]) == i || error("Shu–Osher row $i has $(length(x[i])) entries, expected $i")
        for j in 1:i
            M[i, j] = Float64(x[i][j])
        end
    end
    return M
end
_get(d, k) = haskey(d, k) ? d[k] : nothing
_num(x) = x === nothing ? nothing : Float64(x)

function Tableau(d::AbstractDict)
    ctrl = _get(d, "controller")
    cc = ctrl === nothing ? nothing :
        ControllerConstants((Float64(ctrl[k]) for k in ("beta1", "beta2", "gamma", "qmin", "qmax", "qmax_first_step",
        "qsteady_min", "qsteady_max", "failfactor", "qoldinit", "pid_beta3", "pid_accept_safety"))...)
    emb = _get(d, "embedded")
    rc = _get(d, "residual_control")
    dd = _get(d, "divided_difference")
    dval = _get(d, "d")
    dval = dval === nothing ? nothing : dval isa AbstractVector ? _vec(dval) : Float64(dval)
    eo = _get(d, "embedded_order")
    return Tableau(String(d["name"]), Symbol(d["family"]), Symbol(d["form"]), Int(d["stages"]), Int(d["order"]),
        eo === nothing ? nothing : Int(eo), Symbol(d["error_estimate"]), Bool(d["first_same_as_last"]),
        Bool(d["stiffly_accurate"]), _get(d, "explicit_first_stage"),
        _mat(_get(d, "A")), _mat(_get(d, "C")), _vec(_get(d, "b")), _vec(_get(d, "btilde")), _vec(_get(d, "c")),
        dval, _num(_get(d, "gamma")), _num(_get(d, "c32")),
        _triangle(_get(d, "alpha")), _triangle(_get(d, "beta")),
        emb === nothing ? nothing : _vec(emb["alpha"]), emb === nothing ? nothing : _vec(emb["beta"]),
        emb === nothing ? nothing : Float64(emb["scale"]),
        rc === nothing ? nothing : _vec(rc["sigma"]), rc === nothing ? nothing : Float64(rc["scale"]),
        dd === nothing ? nothing : Float64(dd["constant"]), cc,
        let a = _get(d, "ordinarydiffeq_algorithm"); a === nothing ? nothing : String(a) end)
end

const TABLEAU_DIR = normpath(joinpath(@__DIR__, "..", "data", "tableaus"))

function _load_tableaus(dir)
    isdir(dir) || error("EarthSciIntegrator: coefficient tables not found at $dir (run tools/sync-package-data.sh)")
    tabs = Dict{String, Tableau}()
    for f in sort(readdir(dir))
        endswith(f, ".json") || continue
        path = joinpath(dir, f)
        include_dependency(path)
        t = Tableau(JSON.parsefile(path))
        t.name * ".json" == f || error("table $f names method $(t.name)")
        tabs[t.name] = t
    end
    return tabs
end

"""
    TABLEAUS

Every method's [`Tableau`](@ref), keyed by run-config `alg` name, read from
`data/tableaus/*.json` (a copy of `spec/tableaus`) at precompile time.
"""
const TABLEAUS = _load_tableaus(TABLEAU_DIR)

"""`has_error_estimate(tab)`: whether the method can step adaptively (D52)."""
has_error_estimate(t::Tableau) = t.error_estimate !== :none
