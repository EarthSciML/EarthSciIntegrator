# Extract the coefficient tables of every v1 method into spec/tableaus/<Name>.json.
#
#     julia --project=tools/tableaus tools/tableaus/extract.jl
#
# Coefficients come from the installed OrdinaryDiffEq packages (D31): read from
# their tableau constructors where one exists, transcribed here with a citation
# where OrdinaryDiffEq writes them inline in `perform_step!`. KPP's Rosenbrock
# methods are parsed from WRF-Chem's KPP source; WRF_RK3, IMEXEuler and Theta are
# defined from their references. Controller constants are OrdinaryDiffEqCore's
# defaults for each algorithm.
#
# Before writing, the script checks every table it writes: one step of each
# method, evaluated here from the table in the arithmetic form spec/methods.md
# specifies, must match OrdinaryDiffEq's own step (and its error estimate) to
# roundoff, and every method must show its order on a smooth test problem. The
# output is deterministic: keys are sorted and numbers are printed in shortest
# round-trip form.

using OrdinaryDiffEqCore
using OrdinaryDiffEqLowOrderRK
using OrdinaryDiffEqTsit5
using OrdinaryDiffEqSSPRK
using OrdinaryDiffEqRosenbrock
using OrdinaryDiffEqRosenbrockTableaus
using OrdinaryDiffEqSDIRK
using LinearAlgebra
using Printf

const Core_ = OrdinaryDiffEqCore
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUT = joinpath(ROOT, "spec", "tableaus")
# The KPP source ships with WRF-Chem; WRF is a sibling checkout of this repository.
const KPP_FILE = normpath(joinpath(ROOT, "..", "WRF", "chem", "KPP", "kpp", "kpp-2.1",
    "int", "WRF_conform", "rosenbrock.f90"))

############################################################################
# Deterministic JSON
############################################################################

jsonnum(x::Integer) = string(x)
function jsonnum(x::Real)
    y = Float64(x)
    isfinite(y) || error("non-finite number $y")
    y == 0 && return "0.0"        # never print -0.0
    return repr(y)                # shortest round-trip, e.g. 0.1, 1.0e-5
end

function writejson(io, x, indent = 0)
    pad = "  "^indent
    if x isa AbstractDict
        isempty(x) && return print(io, "{}")
        println(io, "{")
        ks = sort!(collect(keys(x)))
        for (n, k) in enumerate(ks)
            print(io, pad, "  ", repr(String(k)), ": ")
            writejson(io, x[k], indent + 1)
            println(io, n < length(ks) ? "," : "")
        end
        print(io, pad, "}")
    elseif x isa AbstractMatrix
        writejson(io, [collect(r) for r in eachrow(x)], indent)
    elseif x isa AbstractVector && all(v -> v isa Real && !(v isa Bool), x)
        print(io, "[", join(jsonnum.(x), ", "), "]")
    elseif x isa AbstractVector
        isempty(x) && return print(io, "[]")
        println(io, "[")
        for (n, v) in enumerate(x)
            print(io, pad, "  ")
            writejson(io, v, indent + 1)
            println(io, n < length(x) ? "," : "")
        end
        print(io, pad, "]")
    elseif x isa Bool
        print(io, x ? "true" : "false")
    elseif x === nothing
        print(io, "null")
    elseif x isa Real
        print(io, jsonnum(x))
    elseif x isa AbstractString
        print(io, repr(String(x)))
    else
        error("cannot serialize $(typeof(x))")
    end
end

############################################################################
# Citations
############################################################################

function cite(mod::Module, m::Method; note = nothing)
    root = pkgdir(mod)
    d = Dict{String, Any}(
        "package" => string(nameof(mod)),
        "version" => string(pkgversion(mod)),
        "file" => relpath(String(m.file), root),
        "line" => Int(m.line),
    )
    note === nothing || (d["note"] = note)
    return d
end
cite(mod::Module, f, types::Type; kw...) = cite(mod, which(f, types); kw...)
step_method(cache) = which(Core_.perform_step!, Tuple{Any, cache, Any})

controller(alg) = Core_.isadaptive(alg) ? Dict{String, Any}(
    "beta2" => Core_.beta2_default(alg),
    "beta1" => Core_.beta1_default(alg, Core_.beta2_default(alg)),
    "gamma" => Core_.gamma_default(alg),
    "qmin" => Core_.qmin_default(alg),
    "qmax" => Core_.qmax_default(alg),
    "qmax_first_step" => Core_.qmax_first_step_default(alg),
    "qsteady_min" => Core_.qsteady_min_default(alg),
    "qsteady_max" => Core_.qsteady_max_default(alg),
    "failfactor" => Core_.failfactor_default(alg),
    "qoldinit" => 1 // 10^4,
    "pid_beta3" => 0,
    "pid_accept_safety" => 0.81,
) : nothing

const CONTROLLER_SOURCE = Dict{String, Any}(
    "package" => "OrdinaryDiffEqCore",
    "version" => string(pkgversion(OrdinaryDiffEqCore)),
    "file" => "src/alg_utils.jl",
    "line" => Int(which(Core_.beta2_default, Tuple{Core_.OrdinaryDiffEqAlgorithm}).line),
    "note" => "beta1_default, beta2_default, gamma_default, qmin_default, qmax_default, qsteady_*_default, qmax_first_step_default and failfactor_default for this algorithm; qoldinit and the PID constants from src/integrators/controllers.jl (PIController, PIDController).",
)

############################################################################
# Reference evaluators: one step of each form, as spec/methods.md defines it.
# Used only for the self-checks below.
############################################################################

# Butcher form. Returns (u_new, embedded error vector or nothing, stage derivatives).
function step_butcher(f, u, t, h, tab)
    A, b, c = tab["A"], tab["b"], tab["c"]
    s = length(b)
    K = Vector{Vector{Float64}}(undef, s)
    for i in 1:s
        U = copy(u)
        for j in 1:(i - 1)
            A[i][j] == 0 || (U .+= h * A[i][j] .* K[j])
        end
        K[i] = f(U, t + c[i] * h)
    end
    unew = copy(u)
    for i in 1:s
        b[i] == 0 || (unew .+= h * b[i] .* K[i])
    end
    err = nothing
    if haskey(tab, "btilde")
        bt = tab["btilde"]
        # the error weights may reach one stage past b: f(u_new) (first same as last)
        Kx = length(bt) > s ? vcat(K, [f(unew, t + h)]) : K
        err = h .* sum(bt[i] .* Kx[i] for i in eachindex(bt))
    end
    return unew, err
end

function step_shu_osher(f, u, t, h, tab)
    α, β, c = tab["alpha"], tab["beta"], tab["c"]
    s = length(α)
    U = Vector{Vector{Float64}}(undef, s + 1)
    F = Vector{Vector{Float64}}(undef, s)
    U[1] = u
    for i in 1:s
        acc = zero(u)
        for j in 0:(i - 1)
            a, bb = α[i][j + 1], β[i][j + 1]
            a == 0 || (acc .+= a .* U[j + 1])
            if bb != 0
                isassigned(F, j + 1) || (F[j + 1] = f(U[j + 1], t + c[j + 1] * h))
                acc .+= h * bb .* F[j + 1]
            end
        end
        U[i + 1] = acc
    end
    err = nothing
    if haskey(tab, "embedded")
        e = tab["embedded"]
        ut = zero(u)
        for j in 0:s
            a = e["alpha"][j + 1]
            a == 0 || (ut .+= a .* U[j + 1])
            if j < s && e["beta"][j + 1] != 0
                ut .+= h * e["beta"][j + 1] .* F[j + 1]
            end
        end
        err = e["scale"] .* (ut .- U[s + 1])
    end
    return U[s + 1], err
end

# Transformed Rosenbrock form (adjoint.md §4.2).
function step_rosenbrock(f, J, ft, u, t, h, tab)
    A, C, γ, c, d, b = tab["A"], tab["C"], tab["gamma"], tab["c"], tab["d"], tab["b"]
    s = length(b)
    W = I / (h * γ) - J(u, t)
    Wf = lu(W)
    dT = ft(u, t)
    k = Vector{Vector{Float64}}(undef, s)
    for i in 1:s
        U = copy(u)
        for j in 1:(i - 1)
            A[i][j] == 0 || (U .+= A[i][j] .* k[j])
        end
        r = f(U, t + c[i] * h) .+ (h * d[i]) .* dT
        for j in 1:(i - 1)
            C[i][j] == 0 || (r .+= (C[i][j] / h) .* k[j])
        end
        k[i] = Wf \ r
    end
    unew = u .+ sum(b[i] .* k[i] for i in 1:s)
    err = haskey(tab, "btilde") ? sum(tab["btilde"][i] .* k[i] for i in 1:s) : nothing
    return unew, err
end

# Rosenbrock23 in its ode23s form (adjoint.md §4.2).
function step_rosenbrock23(f, J, ft, u, t, h, tab)
    d, c32 = tab["d"], tab["c32"]
    Wf = lu(I - h * d * J(u, t))
    dT = ft(u, t)
    f0 = f(u, t)
    k1 = Wf \ (f0 .+ h * d .* dT)
    f1 = f(u .+ (h / 2) .* k1, t + h / 2)
    k2 = (Wf \ (f1 .- k1)) .+ k1
    unew = u .+ h .* k2
    f2 = f(unew, t + h)
    k3 = Wf \ (f2 .- c32 .* (k2 .- f1) .- 2 .* (k1 .- f0) .+ h .* dT)
    err = (h / 6) .* (k1 .- 2 .* k2 .+ k3)
    return unew, err
end

# SDIRK in the z = hK form, stages solved by Newton with the exact Jacobian to roundoff.
function step_sdirk(f, J, u, t, h, tab; c = tab["c"])
    A, b = tab["A"], tab["b"]
    s = length(b)
    z = Vector{Vector{Float64}}(undef, s)
    for i in 1:s
        tmp = copy(u)
        for j in 1:(i - 1)
            A[i][j] == 0 || (tmp .+= A[i][j] .* z[j])
        end
        γ = A[i][i]
        if γ == 0                       # explicit first stage
            z[i] = h .* f(tmp, t + c[i] * h)
            continue
        end
        zi = h .* f(tmp, t + c[i] * h)
        for _ in 1:50
            Z = tmp .+ γ .* zi
            G = h .* f(Z, t + c[i] * h) .- zi
            dz = (I - h * γ * J(Z, t + c[i] * h)) \ G
            zi = zi .+ dz
            norm(dz) <= 1e-15 * max(1.0, norm(zi)) && break
        end
        z[i] = zi
    end
    unew = u .+ sum(b[i] .* z[i] for i in 1:s)
    err = haskey(tab, "btilde") ? sum(tab["btilde"][i] .* z[i] for i in 1:s) : nothing
    return unew, err
end

############################################################################
# Test problem (smooth, non-autonomous, mildly nonlinear) with exact J and f_t.
############################################################################

ftest(u, t) = [-0.5u[1] + u[2] * u[3] + sin(t), -u[2] + 0.3u[1]^2, -2u[3] + cos(t) * u[1]]
Jtest(u, t) = [-0.5 u[3] u[2]; 0.6u[1] -1.0 0.0; cos(t) 0.0 -2.0]
fttest(u, t) = [cos(t), 0.0, -sin(t) * u[1]]
# The autonomous variant, for comparing with SDIRK stage times that OrdinaryDiffEq sets differently.
fauto(u, t) = [-0.5u[1] + u[2] * u[3], -u[2] + 0.3u[1]^2, -2u[3] + u[1]]
Jauto(u, t) = [-0.5 u[3] u[2]; 0.6u[1] -1.0 0.0; 1.0 0.0 -2.0]
const U0 = [1.0, 0.5, -0.25]

# abstol and reltol set OrdinaryDiffEq's Newton tolerance for implicit stages.
function ode_step(alg, f, u0, t0, h; tol = 1e-12)
    prob = ODEProblem((u, p, t) -> f(u, t), copy(u0), (t0, t0 + 10h))
    integ = init(prob, alg; dt = h, adaptive = false, abstol = tol, reltol = tol, save_everystep = false)
    step!(integ)
    integ.t ≈ t0 + h || error("$(nameof(typeof(alg))): step landed at $(integ.t)")
    return integ.u
end

# OrdinaryDiffEq's scaled error estimate for one step from (u0, t0), abstol = reltol = tol.
function ode_eest(alg, f, u0, t0, h; tol = 1e-6)
    prob = ODEProblem((u, p, t) -> f(u, t), copy(u0), (t0, t0 + 10h))
    integ = init(prob, alg; dt = h, adaptive = true, abstol = tol, reltol = tol, save_everystep = false)
    Core_.perform_step!(integ, integ.cache)
    return Core_.get_EEst(integ), integ.u
end

eest(err, u0, u1; tol = 1e-6) = sqrt(sum(abs2, err ./ (tol .+ max.(abs.(u0), abs.(u1)) .* tol)) / length(err))

# Global error at t = 1 against a fine RK4 reference; returns the observed order.
function observed_order(stepper; n0 = 10)
    ref = let u = copy(U0), N = 20000, h = 1 / N
        rk4 = Dict("A" => [[0.0], [0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0, 0.0]],
            "b" => [1 / 6, 1 / 3, 1 / 3, 1 / 6], "c" => [0.0, 0.5, 0.5, 1.0])
        for n in 0:(N - 1)
            u = step_butcher(ftest, u, n * h, h, rk4)[1]
        end
        u
    end
    errs = map((n0, 2n0)) do N
        u = copy(U0); h = 1 / N
        for n in 0:(N - 1)
            u = stepper(u, n * h, h)
        end
        norm(u - ref)
    end
    return log2(errs[1] / errs[2])
end

############################################################################
# Tables
############################################################################

lower(M) = [collect(M[i, :]) for i in axes(M, 1)]
# OrdinaryDiffEq stores A and C as s×(s-1) for some methods; pad to s×s.
square(M, s) = (Q = zeros(s, s); Q[1:size(M, 1), 1:size(M, 2)] .= M; Q)
padrows(rows) = (s = length(rows); [vcat(Float64.(r), zeros(s - length(r))) for r in rows])

tables = Dict{String, Dict{String, Any}}()
checks = String[]

function add!(name, d)
    d["name"] = name
    tables[name] = d
end

## Explicit, Butcher form -------------------------------------------------

let t = OrdinaryDiffEqTsit5.Tsit5ConstantCacheActual(Float64, Float64)
    g(s) = getfield(t, s)
    A = zeros(7, 7)
    for (i, j) in ((2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3), (5, 1), (5, 2), (5, 3), (5, 4),
            (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6))
        A[i, j] = g(Symbol("a$i$j"))
    end
    add!("Tsit5", Dict{String, Any}(
        "family" => "explicit_rk", "form" => "butcher", "stages" => 7,
        "A" => lower(A), "b" => vcat(A[7, 1:6], 0.0),
        "c" => [0.0, g(:c1), g(:c2), g(:c3), g(:c4), g(:c5), g(:c6)],
        "btilde" => [g(Symbol("btilde$i")) for i in 1:7],
        "error_estimate" => "embedded",
        "source" => cite(OrdinaryDiffEqTsit5, OrdinaryDiffEqTsit5.Tsit5ConstantCacheActual, Tuple{Type{Float64}, Type{Float64}}),
        "notes" => ["Stage 7 is f(u_new): b equals row 7 of A (first same as last), and btilde reads stage 7."],
    ))
end

let t = OrdinaryDiffEqLowOrderRK.BS3ConstantCache(Float64, Float64)
    add!("BS3", Dict{String, Any}(
        "family" => "explicit_rk", "form" => "butcher", "stages" => 4,
        "A" => lower([0 0 0 0; t.a21 0 0 0; 0 t.a32 0 0; t.a41 t.a42 t.a43 0]),
        "b" => [t.a41, t.a42, t.a43, 0.0], "c" => [0.0, t.c1, t.c2, 1.0],
        "btilde" => [t.btilde1, t.btilde2, t.btilde3, t.btilde4],
        "error_estimate" => "embedded",
        "source" => cite(OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqLowOrderRK.BS3ConstantCache, Tuple{Type{Float64}, Type{Float64}}),
        "notes" => ["Stage 4 is f(u_new) (first same as last)."],
    ))
end

add!("Heun", Dict{String, Any}(
    "family" => "explicit_rk", "form" => "butcher", "stages" => 2,
    "A" => [[0.0, 0.0], [1.0, 0.0]], "b" => [0.5, 0.5], "c" => [0.0, 1.0],
    "btilde" => [-0.5, 0.5], "error_estimate" => "embedded",
    "source" => cite(OrdinaryDiffEqLowOrderRK, step_method(OrdinaryDiffEqLowOrderRK.HeunConstantCache);
        note = "Transcribed from the perform_step! body, which writes the coefficients inline (a1 = dt, a2 = dt/2; error a2*(k2 - k1))."),
))

add!("RK4", Dict{String, Any}(
    "family" => "explicit_rk", "form" => "butcher", "stages" => 4,
    "A" => [[0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
    "b" => [1 / 6, 1 / 3, 1 / 3, 1 / 6], "c" => [0.0, 0.5, 0.5, 1.0],
    "error_estimate" => "residual_control",
    "residual_control" => Dict{String, Any}(
        "sigma" => [0.5 - sqrt(3.0) / 6, 0.5 + sqrt(3.0) / 6], "scale" => 2.1342),
    "source" => cite(OrdinaryDiffEqLowOrderRK, step_method(OrdinaryDiffEqLowOrderRK.RK4ConstantCache);
        note = "Transcribed from the perform_step! body, which writes the coefficients inline; the error estimate is Shampine's residual control (two extra f evaluations), spec/methods.md §3.2."),
))

add!("ForwardEuler", Dict{String, Any}(
    "family" => "explicit_rk", "form" => "butcher", "stages" => 1,
    "A" => [[0.0]], "b" => [1.0], "c" => [0.0], "error_estimate" => "none",
    "source" => cite(OrdinaryDiffEqLowOrderRK, step_method(OrdinaryDiffEqLowOrderRK.EulerConstantCache);
        note = "OrdinaryDiffEq's Euler."),
))

add!("WRF_RK3", Dict{String, Any}(
    "family" => "explicit_rk", "form" => "butcher", "stages" => 3,
    "A" => [[0.0, 0.0, 0.0], [1 / 3, 0.0, 0.0], [0.0, 0.5, 0.0]],
    "b" => [0.0, 0.0, 1.0], "c" => [0.0, 1 / 3, 0.5], "error_estimate" => "none",
    "source" => Dict{String, Any}(
        "reference" => "Wicker, L. J. and Skamarock, W. C. (2002), Time-splitting methods for elastic models using forward time schemes, Mon. Wea. Rev. 130, 2088-2097.",
        "file" => "WRF/dyn_em/solve_em.F", "line" => 610,
        "note" => "Stage fractions dt/3, dt/2, dt of the rk_order == 3 branch (lines 610-622, relative to the directory holding the repositories).",
    ),
    "notes" => ["Third order for linear autonomous problems, second order in general (Wicker and Skamarock 2002)."],
))

## Explicit, Shu–Osher form (transcribed: OrdinaryDiffEqSSPRK writes these inline) -

function shu_osher(name, rows, c, m; embedded = nothing, notes = String[])
    s = length(rows)
    α = [zeros(i) for i in 1:s]; β = [zeros(i) for i in 1:s]
    for (i, r) in enumerate(rows), (j, a, b) in r
        α[i][j + 1] = a; β[i][j + 1] = b
    end
    d = Dict{String, Any}("family" => "explicit_rk", "form" => "shu_osher", "stages" => s,
        "alpha" => α, "beta" => β, "c" => c,
        "error_estimate" => embedded === nothing ? "none" : "shu_osher_embedded",
        "source" => cite(OrdinaryDiffEqSSPRK, step_method(m);
            note = "Transcribed from the perform_step! body, which writes the stages inline; alpha and beta are the same arithmetic regrouped (spec/methods.md §3.1)."),
        "notes" => notes)
    embedded === nothing || (d["embedded"] = embedded)
    add!(name, d)
end

# rows[i] lists (j, α_ij, β_ij) for u^(i) = Σ_j α_ij u^(j) + h β_ij f(u^(j), t + c_j h)
shu_osher("SSPRK33",
    [[(0, 1.0, 1.0)], [(0, 3 / 4, 0.0), (1, 1 / 4, 1 / 4)], [(0, 1 / 3, 0.0), (2, 2 / 3, 2 / 3)]],
    [0.0, 1.0, 0.5], OrdinaryDiffEqSSPRK.SSPRK33ConstantCache)
shu_osher("SSPRK43",
    [[(0, 1.0, 0.5)], [(1, 1.0, 0.5)], [(0, 2 / 3, 0.0), (2, 1 / 3, 1 / 6)], [(3, 1.0, 0.5)]],
    [0.0, 0.5, 1.0, 0.5], OrdinaryDiffEqSSPRK.SSPRK43ConstantCache;
    embedded = Dict{String, Any}("alpha" => [1 / 3, 0.0, 2 / 3, 0.0, 0.0], "beta" => [0.0, 0.0, 1 / 3, 0.0],
        "scale" => 0.5),
    notes = ["The embedded solution is u~ = u0/3 + 2/3 (u2 + h/2 f(u2)); the error is scale * (u~ - u_new), which OrdinaryDiffEq comments as bhat = (1/4, 1/4, 1/4, 1/4)."])
shu_osher("SSPRK104",
    [[(0, 1.0, 1 / 6)], [(1, 1.0, 1 / 6)], [(2, 1.0, 1 / 6)], [(3, 1.0, 1 / 6)],
        [(0, 3 / 5, 0.0), (4, 2 / 5, 1 / 15)], [(5, 1.0, 1 / 6)], [(6, 1.0, 1 / 6)], [(7, 1.0, 1 / 6)],
        [(8, 1.0, 1 / 6)], [(0, 1 / 25, 0.0), (4, 9 / 25, 3 / 50), (9, 3 / 5, 1 / 10)]],
    [0.0, 1 / 6, 1 / 3, 1 / 2, 2 / 3, 1 / 3, 1 / 2, 2 / 3, 5 / 6, 1.0],
    OrdinaryDiffEqSSPRK.SSPRK104ConstantCache)

## Rosenbrock ---------------------------------------------------------------

for (name, alg) in (("ROS2", ROS2()), ("ROS3P", ROS3P()), ("Rodas3", Rodas3()), ("Rodas4", Rodas4()), ("Rodas5P", Rodas5P()))
    ctor = OrdinaryDiffEqRosenbrock.tabtype(alg)
    t = ctor(Float64, Float64)
    mod = parentmodule(ctor)
    s = length(t.b)
    d = Dict{String, Any}("family" => "rosenbrock", "form" => "rosenbrock_transformed", "stages" => s,
        "A" => lower(square(t.A, s)), "C" => lower(square(t.C, s)),
        "gamma" => Float64(t.gamma), "c" => Float64.(t.c[1:s]), "d" => Float64.(t.d[1:s]),
        "b" => Float64.(t.b),
        "error_estimate" => t.btilde === nothing ? "none" : "embedded",
        "source" => cite(mod, ctor, Tuple{Type{Float64}, Type{Float64}}))
    t.btilde === nothing || (d["btilde"] = Float64.(t.btilde))
    add!(name, d)
end

let t = OrdinaryDiffEqRosenbrock.Rosenbrock23Tableau(Float64)
    add!("Rosenbrock23", Dict{String, Any}("family" => "rosenbrock", "form" => "rosenbrock23", "stages" => 2,
        "d" => t.d, "c32" => t.c₃₂, "error_estimate" => "rosenbrock23",
        "source" => [cite(OrdinaryDiffEqRosenbrock, OrdinaryDiffEqRosenbrock.Rosenbrock23Tableau, Tuple{Type{Float64}}),
            cite(OrdinaryDiffEqRosenbrock, step_method(OrdinaryDiffEqRosenbrock.Rosenbrock23ConstantCache);
                note = "The ode23s step and its third-stage error estimate.")],
        "notes" => ["OrdinaryDiffEq marks Rosenbrock23 a W-method but defaults it to max_jac_age = 1, so it forms J every step, as this spec requires."]))
end

## KPP Rosenbrock (parsed from WRF-Chem's KPP source) -------------------------

function parse_kpp(sub)
    lines = readlines(KPP_FILE)
    first_ = findfirst(l -> occursin(Regex("SUBROUTINE\\s+KPP_ROOT_$(sub)\\b"), l), lines)
    last_ = findnext(l -> occursin(Regex("END SUBROUTINE\\s+KPP_ROOT_$(sub)\\b"), l), lines, first_)
    env = Dict{Symbol, Any}(:g => 0.0)
    arrays = Dict{String, Dict{Int, Float64}}()
    newf = Dict{Int, Bool}()
    S = nothing
    fortran(expr) = begin
        e = replace(expr, r"_dp" => "", r"([0-9.])[Ee]([+-]?[0-9])" => s"\1e\2", "SQRT" => "sqrt")
        e = replace(e, r"(ros_[A-Za-z]+)\((\d+)\)" => s"\1[\2]")
        Meta.parse(e)
    end
    evalf(ex) = ex isa Number ? Float64(ex) : ex isa Symbol ? Float64(env[ex]) :
        ex.head === :ref ? arrays[String(ex.args[1])][ex.args[2]] :
        ex.head === :call ? Float64(getfield(Base, ex.args[1])(map(evalf, ex.args[2:end])...)) :
        error("cannot evaluate $ex")
    for l in lines[first_:last_]
        l = strip(split(l, '!')[1])
        if (m = match(r"^INTEGER, PARAMETER :: S\s*=\s*(\d+)", l)) !== nothing
            S = parse(Int, m[1])
        elseif (m = match(r"^ros_NewF\((\d+)\)\s*=\s*\.(TRUE|FALSE)\.", l)) !== nothing
            newf[parse(Int, m[1])] = m[2] == "TRUE"
        elseif (m = match(r"^(ros_[A-Za-z]+)\((\d+)\)\s*=\s*(.+)$", l)) !== nothing
            get!(arrays, m[1], Dict{Int, Float64}())[parse(Int, m[2])] = evalf(fortran(m[3]))
        elseif (m = match(r"^(ros_ELO)\s*=\s*(.+)$", l)) !== nothing
            env[:ros_ELO] = evalf(fortran(m[2]))
        elseif (m = match(r"^g\s*=\s*(.+)$", l)) !== nothing
            env[:g] = evalf(fortran(m[1]))
        end
    end
    s = S === nothing ? maximum(keys(arrays["ros_M"])) : S
    vec_(n) = [get(arrays[n], i, 0.0) for i in 1:s]
    packed(n) = (M = zeros(s, s); for i in 2:s, j in 1:(i - 1)
        M[i, j] = get(arrays[n], (i - 1) * (i - 2) ÷ 2 + j, 0.0) end; M)
    return (s = s, A = packed("ros_A"), C = packed("ros_C"), M = vec_("ros_M"), E = vec_("ros_E"),
        Alpha = vec_("ros_Alpha"), Gamma = vec_("ros_Gamma"), ELO = env[:ros_ELO],
        NewF = [newf[i] for i in 1:s], lines = (first_, last_))
end

const KPP_REL = relpath(KPP_FILE, normpath(joinpath(ROOT, "..")))
const KPP_CONTROLLER = Dict{String, Any}("fac_min" => 0.2, "fac_max" => 6.0, "fac_rej" => 0.1, "fac_safe" => 0.9,
    "source" => Dict{String, Any}("file" => KPP_REL, "line" => 317,
        "note" => "KPP's defaults FacMin, FacMax, FacRej, FacSafe (lines 317-348) and its step-size rule (lines 631-660): an I controller with exponent 1/ros_ELO, no growth after a rejection, FacRej after two rejections in a row."))

for (name, sub, order) in (("KPP_Ros2", "Ros2", 2), ("KPP_Ros3", "Ros3", 3), ("KPP_Ros4", "Ros4", 4),
        ("KPP_Rodas3", "Rodas3", 3), ("KPP_Rodas4", "Rodas4", 4))
    k = parse_kpp(sub)
    add!(name, Dict{String, Any}("family" => "rosenbrock", "form" => "rosenbrock_transformed", "stages" => k.s,
        "A" => lower(k.A), "C" => lower(k.C), "gamma" => k.Gamma[1], "c" => k.Alpha, "d" => k.Gamma,
        "b" => k.M, "btilde" => k.E, "error_estimate" => "embedded",
        "order" => order, "embedded_order" => Int(k.ELO) - 1,
        "kpp" => Dict{String, Any}("elo" => k.ELO, "new_f" => k.NewF, "controller" => KPP_CONTROLLER),
        "source" => Dict{String, Any}("file" => KPP_REL, "line" => k.lines[1],
            "note" => "KPP 2.1 (WRF_conform) subroutine KPP_ROOT_$(sub), lines $(k.lines[1])-$(k.lines[2]); KPP's (ros_A, ros_C, ros_Gamma(1), ros_Alpha, ros_Gamma, ros_M, ros_E) are (A, C, gamma, c, d, b, btilde) of the transformed form. Sandu et al. (1997), Atmos. Environ. 31, 3459-3472."),
        "controller" => Dict{String, Any}("from" => "order", "note" => "OrdinaryDiffEqCore's defaults for an adaptive Rosenbrock method of this order; KPP's own controller is under kpp.controller for WRF-Chem parity."),
    ))
end

## SDIRK -------------------------------------------------------------------

for (name, alg, ctorname) in (("BackwardEuler", ImplicitEuler(), :ImplicitEulerESDIRKIMEXTableau),
        ("TRBDF2", TRBDF2(), :TRBDF2ESDIRKIMEXTableau), ("SDIRK2", SDIRK2(), :SDIRK2ESDIRKIMEXTableau),
        ("Kvaerno3", Kvaerno3(), :Kvaerno3ESDIRKIMEXTableau), ("Kvaerno4", Kvaerno4(), :Kvaerno4ESDIRKIMEXTableau))
    t = OrdinaryDiffEqSDIRK.ESDIRKIMEXTableau(alg, Float64, Float64)
    ctor = which(getfield(OrdinaryDiffEqSDIRK, ctorname), Tuple{Type{Float64}, Type{Float64}})
    s = t.s
    A = Float64.(t.Ai)
    rowsum = vec(sum(A; dims = 2))
    # c is the row sum of A, rounded to the abscissa OrdinaryDiffEq writes when the two agree to roundoff.
    c = [abs(rowsum[i] - t.c[i]) <= 1e-12 ? Float64(t.c[i]) : rowsum[i] for i in 1:s]
    d = Dict{String, Any}("family" => "sdirk", "form" => "sdirk", "stages" => s,
        "A" => lower(A), "b" => Float64.(t.bi), "c" => c,
        "explicit_first_stage" => t.explicit_first_stage,
        "error_estimate" => name == "BackwardEuler" ? "divided_difference" : "embedded_smoothed",
        "source" => cite(OrdinaryDiffEqSDIRK, ctor), "notes" => String[])
    isempty(t.btilde) || (d["btilde"] = Float64.(t.btilde))
    if c != Float64.(t.c)
        d["c_ordinarydiffeq"] = Float64.(t.c)
        push!(d["notes"], "OrdinaryDiffEq sets the stage times c to $(Float64.(t.c)); this table uses the row sums of A, which the method's order conditions assume (spec/methods.md §5.4, D55).")
    end
    name == "BackwardEuler" && (d["divided_difference"] = Dict{String, Any}("constant" => 7 / 12))
    isempty(d["notes"]) && delete!(d, "notes")
    add!(name, d)
end

## Defined here --------------------------------------------------------------

add!("IMEXEuler", Dict{String, Any}("family" => "imex", "form" => "imex_euler", "stages" => 1,
    "error_estimate" => "none", "order" => 1, "embedded_order" => nothing, "controller" => nothing,
    "source" => Dict{String, Any}("reference" => "architecture §5 (PISM's explicit horizontal at old time); spec/adjoint.md §4.4."),
))
add!("Theta", Dict{String, Any}("family" => "theta", "form" => "theta", "stages" => 1,
    "error_estimate" => "none", "order" => 1, "embedded_order" => nothing, "controller" => nothing,
    "notes" => ["theta is the run config's leaf option. order is the order for theta ≠ 1/2; theta = 1/2 is second order."],
    "source" => Dict{String, Any}("reference" => "spec/adjoint.md §4.5."),
))

## Orders, flags and controllers from OrdinaryDiffEq's traits ------------------

const ALGS = Dict("Tsit5" => Tsit5(), "BS3" => BS3(), "Heun" => Heun(), "RK4" => RK4(), "ForwardEuler" => Euler(),
    "SSPRK33" => SSPRK33(), "SSPRK43" => SSPRK43(), "SSPRK104" => SSPRK104(),
    "Rosenbrock23" => Rosenbrock23(), "ROS2" => ROS2(), "ROS3P" => ROS3P(), "Rodas3" => Rodas3(),
    "Rodas4" => Rodas4(), "Rodas5P" => Rodas5P(),
    "BackwardEuler" => ImplicitEuler(), "TRBDF2" => TRBDF2(), "SDIRK2" => SDIRK2(),
    "Kvaerno3" => Kvaerno3(), "Kvaerno4" => Kvaerno4())

for (name, d) in tables
    if haskey(ALGS, name)
        alg = ALGS[name]
        d["order"] = Core_.alg_order(alg)
        d["embedded_order"] = Core_.isadaptive(alg) ? Core_.alg_adaptive_order(alg) : nothing
        d["controller"] = controller(alg)
        d["controller"] === nothing || (d["controller"]["source"] = CONTROLLER_SOURCE)
        d["ordinarydiffeq_algorithm"] = string(nameof(typeof(alg)))
    elseif get(d, "controller", nothing) isa Dict && d["controller"]["from"] == "order"
        # KPP: OrdinaryDiffEq's formulas for an adaptive Rosenbrock method of this order.
        alg = d["order"] == 2 ? ROS2() : d["order"] == 3 ? Rodas3() : Rodas4()
        Core_.alg_order(alg) == d["order"] || error("order proxy")
        note = d["controller"]["note"]
        d["controller"] = controller(alg)
        d["controller"]["source"] = merge(CONTROLLER_SOURCE, Dict("note" => note))
    elseif !haskey(d, "order")
        d["order"] = name == "WRF_RK3" ? 2 : error("no order for $name")
        d["embedded_order"] = nothing
        d["controller"] = nothing
    end
    get(d, "error_estimate", "none") == "none" && d["controller"] !== nothing &&
        error("$name: adaptive without an error estimate")
    d["first_same_as_last"] = name in ("Tsit5", "BS3", "Heun", "RK4", "ForwardEuler", "Rosenbrock23")
    if haskey(d, "A") && haskey(d, "b") && d["form"] in ("butcher", "sdirk")
        A = d["A"]; s = length(d["b"])
        d["stiffly_accurate"] = all(isapprox(A[s][j], d["b"][j]; atol = 0) for j in 1:s)
    elseif d["form"] == "rosenbrock_transformed"
        # u_new = U_s + k_s, i.e. b = (A[s, :], 1): stiffly accurate in the transformed form
        A = d["A"]; s = length(d["b"])
        d["stiffly_accurate"] = all(A[s][j] == d["b"][j] for j in 1:(s - 1)) && d["b"][s] == 1
    else
        d["stiffly_accurate"] = false
    end
end

############################################################################
# Self-checks
############################################################################

relerr(a, b) = norm(a - b) / max(norm(b), 1e-300)
function check(cond, msg)
    cond || error("self-check failed: $msg")
    push!(checks, msg)
end

h, t0 = 0.05, 0.3
for (name, d) in sort!(collect(tables); by = first)
    alg = get(ALGS, name, nothing)
    form = d["form"]
    stepper = if form == "butcher"
        (u, t, h) -> step_butcher(ftest, u, t, h, d)
    elseif form == "shu_osher"
        (u, t, h) -> step_shu_osher(ftest, u, t, h, d)
    elseif form == "rosenbrock_transformed"
        (u, t, h) -> step_rosenbrock(ftest, Jtest, fttest, u, t, h, d)
    elseif form == "rosenbrock23"
        (u, t, h) -> step_rosenbrock23(ftest, Jtest, fttest, u, t, h, d)
    elseif form == "sdirk"
        (u, t, h) -> step_sdirk(ftest, Jtest, u, t, h, d)
    else
        nothing
    end
    stepper === nothing && continue
    # 1. Same step as OrdinaryDiffEq, to roundoff (SDIRK on the autonomous problem, see below).
    if alg !== nothing
        if form == "sdirk"
            mine = step_sdirk(fauto, Jauto, U0, t0, h, d)[1]
            ode = ode_step(alg, fauto, U0, t0, h)
            check(relerr(mine, ode) < 1e-9, "$name: one step matches OrdinaryDiffEq (autonomous, Newton tolerance), rel $(@sprintf("%.1e", relerr(mine, ode)))")
        else
            mine = stepper(U0, t0, h)[1]
            ode = ode_step(alg, ftest, U0, t0, h)
            check(relerr(mine, ode) < 1e-13, "$name: one step matches OrdinaryDiffEq, rel $(@sprintf("%.1e", relerr(mine, ode)))")
        end
    end
    # 2. Same scaled error estimate as OrdinaryDiffEq, where the estimate is a plain vector.
    if alg !== nothing && d["error_estimate"] in ("embedded", "shu_osher_embedded", "rosenbrock23")
        E_ode, u1 = ode_eest(alg, ftest, U0, t0, h)
        _, err = stepper(U0, t0, h)
        E_mine = eest(err, U0, u1)
        check(abs(E_mine - E_ode) <= 1e-8 * E_ode, "$name: error estimate matches OrdinaryDiffEq, $(@sprintf("%.6e", E_mine)) vs $(@sprintf("%.6e", E_ode))")
    end
    if alg !== nothing && d["error_estimate"] == "embedded_smoothed"
        a2 = typeof(alg).name.wrapper(smooth_est = false)
        E_ode, u1 = ode_eest(a2, fauto, U0, t0, h)
        _, err = step_sdirk(fauto, Jauto, U0, t0, h, d)
        E_mine = eest(err, U0, u1)
        check(abs(E_mine - E_ode) <= 1e-6 * E_ode, "$name: unsmoothed error estimate matches OrdinaryDiffEq (smooth_est = false), $(@sprintf("%.6e", E_mine)) vs $(@sprintf("%.6e", E_ode))")
    end
    # 3. Observed order on the non-autonomous test problem.
    expected = d["order"]
    p = observed_order((u, t, h) -> stepper(u, t, h)[1]; n0 = d["order"] >= 4 ? 8 : 16)
    check(p > expected - 0.35, "$name: observed order $(@sprintf("%.2f", p)) (expected $expected)")
end

# Which stage times does OrdinaryDiffEq actually use for Kvaerno3/4? Compare on the
# non-autonomous problem against both choices of c.
for name in ("Kvaerno3", "Kvaerno4")
    d = tables[name]
    ode = ode_step(ALGS[name], ftest, U0, t0, h)
    with_rowsum = step_sdirk(ftest, Jtest, U0, t0, h, d)[1]
    with_ode_c = step_sdirk(ftest, Jtest, U0, t0, h, d; c = d["c_ordinarydiffeq"])[1]
    check(relerr(with_ode_c, ode) < 1e-9 && relerr(with_rowsum, ode) > 1e-8,
        "$name: OrdinaryDiffEq evaluates stage 2 at c_ordinarydiffeq (rel $(@sprintf("%.1e", relerr(with_ode_c, ode)))), not at the row sum (rel $(@sprintf("%.1e", relerr(with_rowsum, ode))))")
end

############################################################################
# Write
############################################################################

mkpath(OUT)
for f in readdir(OUT)
    endswith(f, ".json") && rm(joinpath(OUT, f))
end
for (name, d) in tables
    d["schema"] = "https://earthsciml.org/schemas/esm-integrator/0.1.0/tableau.schema.json"
    open(joinpath(OUT, "$name.json"), "w") do io
        writejson(io, d)
        println(io)
    end
end
foreach(println, checks)
println("wrote $(length(tables)) tables to $(relpath(OUT, ROOT)); $(length(checks)) self-checks passed")
