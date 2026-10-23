# The hand-written toy problems of spec/toy-problems.md §1 (D53). Every call
# of operator-interface.md §3 is implemented with hand-derived derivatives; the
# test suite checks each against ForwardDiff.
#
# Indices in the spec are 0-based; Julia's are 1-based, so DOF `i` of cell `c`
# in a species-major layout is at `i*ncells + c + 1`.

"""
    ToyProblem

A toy problem: its state size, initial state, parameters, interval, operators
by name, variable names (each mapped to its DOF indices), invariants, and the
exact solution where one exists.
"""
struct ToyProblem
    name::String
    N::Int
    u0::Vector{Float64}
    p::Vector{Float64}
    pnames::Vector{String}
    tspan::Tuple{Float64, Float64}
    operators::Dict{String, AbstractOperator}
    maps::Dict{String, AbstractMapOperator}
    variables::Vector{Pair{String, Vector{Int}}}
    invariants::Vector{Function}
    exact::Union{Nothing, Function}
    stiff::Bool
end

"""The 1-based DOF indices of variable `name`."""
function variable_dofs(prob::ToyProblem, name::AbstractString)
    i = findfirst(v -> v.first == name, prob.variables)
    i === nothing && throw(ConfigError("toy problem $(prob.name) has no variable '$name'"))
    return prob.variables[i].second
end

# ---------------------------------------------------------------------------
# 1.1 forced_decay: u' = -k u + sin(ω t), p = (k, ω)

struct ForcedDecay <: AbstractOperator end
opname(::ForcedDecay) = "decay"
structure(::ForcedDecay) = DenseStructure()
depends_on_t(::ForcedDecay) = true
linear_in_u(::ForcedDecay) = true

function rhs!(::ForcedDecay, out, u, p, t)
    k, ω = p[1], p[2]
    out[1] += -k * u[1] + sin(ω * t)
    return out
end
jvp_u!(::ForcedDecay, out, u, p, t, v) = (out[1] += -p[1] * v[1]; out)
vjp_u!(::ForcedDecay, out, u, p, t, w) = (out[1] += -p[1] * w[1]; out)
function vjp_p!(::ForcedDecay, out, u, p, t, w)
    ω = p[2]
    out[1] += -u[1] * w[1]
    out[2] += t * cos(ω * t) * w[1]
    return out
end
jac!(::ForcedDecay, out, u, p, t) = addentry!(out, 1, 1, -p[1])
ft!(::ForcedDecay, out, u, p, t) = (out[1] += p[2] * cos(p[2] * t); out)
hess_u!(::ForcedDecay, out, u, p, t, μ) = out
function hess_p!(::ForcedDecay, out, u, p, t, μ)
    # Σᵢ μᵢᵀ J wᵢ = -k Σᵢ μᵢ wᵢ
    for (m, w) in μ
        out[1] += -m[1] * w[1]
    end
    return out
end
ft_vjp_u!(::ForcedDecay, out, u, p, t, ν) = out
function ft_vjp_p!(::ForcedDecay, out, u, p, t, ν)
    ω = p[2]
    # ∂/∂ω (ν ω cos ωt) = ν (cos ωt - ω t sin ωt)
    out[2] += ν[1] * (cos(ω * t) - ω * t * sin(ω * t))
    return out
end

function forced_decay()
    p = [2.0, 3.0]
    u0 = [1.0]
    exact = function (t)
        k, ω = p
        A(s) = (k * sin(ω * s) - ω * cos(ω * s)) / (k^2 + ω^2)
        return [(u0[1] - A(0.0)) * exp(-k * t) + A(t)]
    end
    return ToyProblem("forced_decay", 1, u0, p, ["k", "omega"], (0.0, 2.0),
        Dict{String, AbstractOperator}("decay" => ForcedDecay()), Dict{String, AbstractMapOperator}(),
        ["u" => [1]], Function[], exact, false)
end

# ---------------------------------------------------------------------------
# 1.2 van_der_pol: y' = z, z' = μ(1 - y²) z - y, p = (μ)

struct Oscillator <: AbstractOperator end
opname(::Oscillator) = "oscillator"
structure(::Oscillator) = DenseStructure()
depends_on_t(::Oscillator) = false
linear_in_u(::Oscillator) = false

function rhs!(::Oscillator, out, u, p, t)
    μ = p[1]
    y, z = u[1], u[2]
    out[1] += z
    out[2] += μ * (1 - y^2) * z - y
    return out
end
function jvp_u!(::Oscillator, out, u, p, t, v)
    μ = p[1]
    y, z = u[1], u[2]
    out[1] += v[2]
    out[2] += (-2 * μ * y * z - 1) * v[1] + μ * (1 - y^2) * v[2]
    return out
end
function vjp_u!(::Oscillator, out, u, p, t, w)
    μ = p[1]
    y, z = u[1], u[2]
    out[1] += (-2 * μ * y * z - 1) * w[2]
    out[2] += w[1] + μ * (1 - y^2) * w[2]
    return out
end
function vjp_p!(::Oscillator, out, u, p, t, w)
    y, z = u[1], u[2]
    out[1] += (1 - y^2) * z * w[2]
    return out
end
function jac!(::Oscillator, out, u, p, t)
    μ = p[1]
    y, z = u[1], u[2]
    addentry!(out, 1, 2, one(y))
    addentry!(out, 2, 1, -2 * μ * y * z - 1)
    addentry!(out, 2, 2, μ * (1 - y^2))
    return out
end
ft!(::Oscillator, out, u, p, t) = out
function hess_u!(::Oscillator, out, u, p, t, μs)
    μ = p[1]
    y, z = u[1], u[2]
    # mᵀ J w = m₁w₂ + m₂((-2μyz - 1)w₁ + μ(1 - y²)w₂)
    for (m, w) in μs
        out[1] += m[2] * (-2 * μ * z * w[1] - 2 * μ * y * w[2])
        out[2] += m[2] * (-2 * μ * y * w[1])
    end
    return out
end
function hess_p!(::Oscillator, out, u, p, t, μs)
    y, z = u[1], u[2]
    for (m, w) in μs
        out[1] += m[2] * (-2 * y * z * w[1] + (1 - y^2) * w[2])
    end
    return out
end
ft_vjp_u!(::Oscillator, out, u, p, t, ν) = out
ft_vjp_p!(::Oscillator, out, u, p, t, ν) = out

function van_der_pol(variant::AbstractString)
    if variant == "nonstiff"
        μ, tf, stiff = 1.0, 6.3, false
    elseif variant == "stiff"
        μ, tf, stiff = 1000.0, 1.0, true
    else
        throw(ConfigError("van_der_pol has variants 'nonstiff' and 'stiff', not '$variant'"))
    end
    return ToyProblem("van_der_pol/$variant", 2, [2.0, 0.0], [μ], ["mu"], (0.0, tf),
        Dict{String, AbstractOperator}("oscillator" => Oscillator()), Dict{String, AbstractMapOperator}(),
        ["y" => [1], "z" => [2]], Function[], nothing, stiff)
end

# ---------------------------------------------------------------------------
# 1.3 robertson, p = (k₁, k₂, k₃)

struct Kinetics <: AbstractOperator end
opname(::Kinetics) = "kinetics"
structure(::Kinetics) = DenseStructure()
depends_on_t(::Kinetics) = false
linear_in_u(::Kinetics) = false

function rhs!(::Kinetics, out, u, p, t)
    k1, k2, k3 = p[1], p[2], p[3]
    y1, y2, y3 = u[1], u[2], u[3]
    out[1] += -k1 * y1 + k3 * y2 * y3
    out[2] += k1 * y1 - k3 * y2 * y3 - k2 * y2^2
    out[3] += k2 * y2^2
    return out
end
# J = [-k1  k3y3  k3y2; k1  -k3y3-2k2y2  -k3y2; 0  2k2y2  0]
function jvp_u!(::Kinetics, out, u, p, t, v)
    k1, k2, k3 = p[1], p[2], p[3]
    y2, y3 = u[2], u[3]
    out[1] += -k1 * v[1] + k3 * y3 * v[2] + k3 * y2 * v[3]
    out[2] += k1 * v[1] + (-k3 * y3 - 2 * k2 * y2) * v[2] - k3 * y2 * v[3]
    out[3] += 2 * k2 * y2 * v[2]
    return out
end
function vjp_u!(::Kinetics, out, u, p, t, w)
    k1, k2, k3 = p[1], p[2], p[3]
    y2, y3 = u[2], u[3]
    out[1] += -k1 * w[1] + k1 * w[2]
    out[2] += k3 * y3 * w[1] + (-k3 * y3 - 2 * k2 * y2) * w[2] + 2 * k2 * y2 * w[3]
    out[3] += k3 * y2 * w[1] - k3 * y2 * w[2]
    return out
end
function vjp_p!(::Kinetics, out, u, p, t, w)
    y1, y2, y3 = u[1], u[2], u[3]
    out[1] += -y1 * w[1] + y1 * w[2]
    out[2] += -y2^2 * w[2] + y2^2 * w[3]
    out[3] += y2 * y3 * w[1] - y2 * y3 * w[2]
    return out
end
function jac!(::Kinetics, out, u, p, t)
    k1, k2, k3 = p[1], p[2], p[3]
    y2, y3 = u[2], u[3]
    addentry!(out, 1, 1, -k1)
    addentry!(out, 1, 2, k3 * y3)
    addentry!(out, 1, 3, k3 * y2)
    addentry!(out, 2, 1, k1)
    addentry!(out, 2, 2, -k3 * y3 - 2 * k2 * y2)
    addentry!(out, 2, 3, -k3 * y2)
    addentry!(out, 3, 2, 2 * k2 * y2)
    return out
end
ft!(::Kinetics, out, u, p, t) = out
function hess_u!(::Kinetics, out, u, p, t, μs)
    k2, k3 = p[2], p[3]
    for (m, w) in μs
        out[2] += m[1] * k3 * w[3] + m[2] * (-2 * k2 * w[2] - k3 * w[3]) + m[3] * 2 * k2 * w[2]
        out[3] += m[1] * k3 * w[2] - m[2] * k3 * w[2]
    end
    return out
end
function hess_p!(::Kinetics, out, u, p, t, μs)
    y2, y3 = u[2], u[3]
    for (m, w) in μs
        out[1] += -m[1] * w[1] + m[2] * w[1]
        out[2] += -m[2] * 2 * y2 * w[2] + m[3] * 2 * y2 * w[2]
        out[3] += m[1] * (y3 * w[2] + y2 * w[3]) - m[2] * (y3 * w[2] + y2 * w[3])
    end
    return out
end
ft_vjp_u!(::Kinetics, out, u, p, t, ν) = out
ft_vjp_p!(::Kinetics, out, u, p, t, ν) = out

function robertson()
    return ToyProblem("robertson", 3, [1.0, 0.0, 0.0], [0.04, 3.0e7, 1.0e4], ["k1", "k2", "k3"], (0.0, 40.0),
        Dict{String, AbstractOperator}("kinetics" => Kinetics()), Dict{String, AbstractMapOperator}(),
        ["y1" => [1], "y2" => [2], "y3" => [3]], Function[u -> [u[1] + u[2] + u[3]]], nothing, true)
end

# ---------------------------------------------------------------------------
# 1.4 photostationary_cells, p = (j₀, k, ω); species (NO₂, NO, O₃), species-major

struct Photochemistry <: AbstractOperator
    ncells::Int
end
opname(::Photochemistry) = "photochemistry"
structure(op::Photochemistry) = BlockDiagonalStructure(op.ncells, 3)
depends_on_t(::Photochemistry) = true
linear_in_u(::Photochemistry) = false

@inline _pc_phase(op::Photochemistry, c) = 2π * c / op.ncells
@inline _pc_idx(op::Photochemistry, c) = (c + 1, op.ncells + c + 1, 2 * op.ncells + c + 1)
# j_c(t) = ½ j₀ (1 + sin(ωt + φ_c))
@inline _pc_j(j0, ω, t, φ) = j0 / 2 * (1 + sin(ω * t + φ))

function rhs!(op::Photochemistry, out, u, p, t)
    j0, k, ω = p[1], p[2], p[3]
    for c in 0:(op.ncells - 1)
        iNO2, iNO, iO3 = _pc_idx(op, c)
        r = _pc_j(j0, ω, t, _pc_phase(op, c)) * u[iNO2] - k * u[iNO] * u[iO3]
        out[iNO2] += -r
        out[iNO] += r
        out[iO3] += r
    end
    return out
end
function jvp_u!(op::Photochemistry, out, u, p, t, v)
    j0, k, ω = p[1], p[2], p[3]
    for c in 0:(op.ncells - 1)
        iNO2, iNO, iO3 = _pc_idx(op, c)
        dr = _pc_j(j0, ω, t, _pc_phase(op, c)) * v[iNO2] - k * u[iO3] * v[iNO] - k * u[iNO] * v[iO3]
        out[iNO2] += -dr
        out[iNO] += dr
        out[iO3] += dr
    end
    return out
end
function vjp_u!(op::Photochemistry, out, u, p, t, w)
    j0, k, ω = p[1], p[2], p[3]
    for c in 0:(op.ncells - 1)
        iNO2, iNO, iO3 = _pc_idx(op, c)
        s = -w[iNO2] + w[iNO] + w[iO3]   # cotangent of r_c
        out[iNO2] += _pc_j(j0, ω, t, _pc_phase(op, c)) * s
        out[iNO] += -k * u[iO3] * s
        out[iO3] += -k * u[iNO] * s
    end
    return out
end
function vjp_p!(op::Photochemistry, out, u, p, t, w)
    j0, k, ω = p[1], p[2], p[3]
    for c in 0:(op.ncells - 1)
        iNO2, iNO, iO3 = _pc_idx(op, c)
        φ = _pc_phase(op, c)
        s = -w[iNO2] + w[iNO] + w[iO3]
        out[1] += (1 + sin(ω * t + φ)) / 2 * u[iNO2] * s
        out[2] += -u[iNO] * u[iO3] * s
        out[3] += j0 / 2 * cos(ω * t + φ) * t * u[iNO2] * s
    end
    return out
end
function jac!(op::Photochemistry, out, u, p, t)
    j0, k, ω = p[1], p[2], p[3]
    for c in 0:(op.ncells - 1)
        idx = _pc_idx(op, c)
        iNO2, iNO, iO3 = idx
        dr = (_pc_j(j0, ω, t, _pc_phase(op, c)), -k * u[iO3], -k * u[iNO])
        for (row, sgn) in ((iNO2, -1), (iNO, 1), (iO3, 1)), col in 1:3
            addentry!(out, row, idx[col], sgn * dr[col])
        end
    end
    return out
end
function ft!(op::Photochemistry, out, u, p, t)
    j0, k, ω = p[1], p[2], p[3]
    for c in 0:(op.ncells - 1)
        iNO2, iNO, iO3 = _pc_idx(op, c)
        drdt = j0 / 2 * cos(ω * t + _pc_phase(op, c)) * ω * u[iNO2]
        out[iNO2] += -drdt
        out[iNO] += drdt
        out[iO3] += drdt
    end
    return out
end
function hess_u!(op::Photochemistry, out, u, p, t, μs)
    k = p[2]
    for (m, w) in μs, c in 0:(op.ncells - 1)
        iNO2, iNO, iO3 = _pc_idx(op, c)
        sm = -m[iNO2] + m[iNO] + m[iO3]
        # sm * (j w_NO2 - k O3 w_NO - k NO w_O3)
        out[iNO] += sm * (-k * w[iO3])
        out[iO3] += sm * (-k * w[iNO])
    end
    return out
end
function hess_p!(op::Photochemistry, out, u, p, t, μs)
    j0, k, ω = p[1], p[2], p[3]
    for (m, w) in μs, c in 0:(op.ncells - 1)
        iNO2, iNO, iO3 = _pc_idx(op, c)
        φ = _pc_phase(op, c)
        sm = -m[iNO2] + m[iNO] + m[iO3]
        out[1] += sm * (1 + sin(ω * t + φ)) / 2 * w[iNO2]
        out[2] += sm * (-u[iO3] * w[iNO] - u[iNO] * w[iO3])
        out[3] += sm * j0 / 2 * cos(ω * t + φ) * t * w[iNO2]
    end
    return out
end
function ft_vjp_u!(op::Photochemistry, out, u, p, t, ν)
    j0, k, ω = p[1], p[2], p[3]
    for c in 0:(op.ncells - 1)
        iNO2, iNO, iO3 = _pc_idx(op, c)
        sν = -ν[iNO2] + ν[iNO] + ν[iO3]
        out[iNO2] += sν * j0 / 2 * cos(ω * t + _pc_phase(op, c)) * ω
    end
    return out
end
function ft_vjp_p!(op::Photochemistry, out, u, p, t, ν)
    j0, k, ω = p[1], p[2], p[3]
    for c in 0:(op.ncells - 1)
        iNO2, iNO, iO3 = _pc_idx(op, c)
        φ = _pc_phase(op, c)
        sν = -ν[iNO2] + ν[iNO] + ν[iO3]
        out[1] += sν * cos(ω * t + φ) / 2 * ω * u[iNO2]
        out[3] += sν * j0 / 2 * u[iNO2] * (cos(ω * t + φ) - ω * t * sin(ω * t + φ))
    end
    return out
end

function photostationary_cells()
    ncells = 16
    op = Photochemistry(ncells)
    u0 = vcat(fill(10.0, ncells), fill(5.0, ncells), fill(40.0, ncells))
    inv = u -> vcat([u[c] + u[ncells + c] for c in 1:ncells], [u[c] + u[2ncells + c] for c in 1:ncells])
    return ToyProblem("photostationary_cells", 3ncells, u0, [1.0e-2, 4.4e-4, 2π / 3600], ["j0", "k", "omega"],
        (0.0, 3600.0), Dict{String, AbstractOperator}("photochemistry" => op), Dict{String, AbstractMapOperator}(),
        ["NO2" => collect(1:ncells), "NO" => collect((ncells + 1):(2ncells)), "O3" => collect((2ncells + 1):(3ncells))],
        Function[inv], nothing, true)
end

# ---------------------------------------------------------------------------
# 1.5 advection_reaction, p = (v, k₁, k₂); species (A, B), species-major, periodic

struct Transport <: AbstractOperator
    ncells::Int
end
opname(::Transport) = "transport"
structure(::Transport) = DenseStructure()
depends_on_t(::Transport) = false
linear_in_u(::Transport) = true

# Upwind: A_c' = -v (A_c - A_{c-1}) / Δx, evaluated as ((-v) * (A_c - A_{c-1})) / Δx.
@inline _left(op::Transport, c) = c == 1 ? op.ncells : c - 1
function rhs!(op::Transport, out, u, p, t)
    v = p[1]
    dx = 1 / op.ncells
    for c in 1:op.ncells
        out[c] += -v * (u[c] - u[_left(op, c)]) / dx
    end
    return out
end
function jvp_u!(op::Transport, out, u, p, t, w)
    v = p[1]
    dx = 1 / op.ncells
    for c in 1:op.ncells
        out[c] += -v * (w[c] - w[_left(op, c)]) / dx
    end
    return out
end
function vjp_u!(op::Transport, out, u, p, t, w)
    v = p[1]
    dx = 1 / op.ncells
    for c in 1:op.ncells
        out[c] += -v / dx * w[c]
        out[_left(op, c)] += v / dx * w[c]
    end
    return out
end
function vjp_p!(op::Transport, out, u, p, t, w)
    dx = 1 / op.ncells
    for c in 1:op.ncells
        out[1] += -(u[c] - u[_left(op, c)]) / dx * w[c]
    end
    return out
end
function jac!(op::Transport, out, u, p, t)
    v = p[1]
    dx = 1 / op.ncells
    for c in 1:op.ncells
        addentry!(out, c, c, -v / dx)
        addentry!(out, c, _left(op, c), v / dx)
    end
    return out
end
ft!(::Transport, out, u, p, t) = out
hess_u!(::Transport, out, u, p, t, μ) = out
function hess_p!(op::Transport, out, u, p, t, μs)
    dx = 1 / op.ncells
    for (m, w) in μs, c in 1:op.ncells
        out[1] += m[c] * (-(w[c] - w[_left(op, c)]) / dx)
    end
    return out
end
ft_vjp_u!(::Transport, out, u, p, t, ν) = out
ft_vjp_p!(::Transport, out, u, p, t, ν) = out

struct Reaction <: AbstractOperator
    ncells::Int
end
opname(::Reaction) = "reaction"
structure(op::Reaction) = BlockDiagonalStructure(op.ncells, 2)
depends_on_t(::Reaction) = false
linear_in_u(::Reaction) = true

function rhs!(op::Reaction, out, u, p, t)
    k1, k2 = p[2], p[3]
    n = op.ncells
    for c in 1:n
        A, B = u[c], u[n + c]
        out[c] += -k1 * A + k2 * B
        out[n + c] += k1 * A - k2 * B
    end
    return out
end
function jvp_u!(op::Reaction, out, u, p, t, v)
    k1, k2 = p[2], p[3]
    n = op.ncells
    for c in 1:n
        out[c] += -k1 * v[c] + k2 * v[n + c]
        out[n + c] += k1 * v[c] - k2 * v[n + c]
    end
    return out
end
function vjp_u!(op::Reaction, out, u, p, t, w)
    k1, k2 = p[2], p[3]
    n = op.ncells
    for c in 1:n
        out[c] += -k1 * w[c] + k1 * w[n + c]
        out[n + c] += k2 * w[c] - k2 * w[n + c]
    end
    return out
end
function vjp_p!(op::Reaction, out, u, p, t, w)
    n = op.ncells
    for c in 1:n
        A, B = u[c], u[n + c]
        out[2] += -A * w[c] + A * w[n + c]
        out[3] += B * w[c] - B * w[n + c]
    end
    return out
end
function jac!(op::Reaction, out, u, p, t)
    k1, k2 = p[2], p[3]
    n = op.ncells
    for c in 1:n
        addentry!(out, c, c, -k1)
        addentry!(out, c, n + c, k2)
        addentry!(out, n + c, c, k1)
        addentry!(out, n + c, n + c, -k2)
    end
    return out
end
ft!(::Reaction, out, u, p, t) = out
hess_u!(::Reaction, out, u, p, t, μ) = out
function hess_p!(op::Reaction, out, u, p, t, μs)
    n = op.ncells
    for (m, w) in μs, c in 1:n
        out[2] += -m[c] * w[c] + m[n + c] * w[c]
        out[3] += m[c] * w[n + c] - m[n + c] * w[n + c]
    end
    return out
end
ft_vjp_u!(::Reaction, out, u, p, t, ν) = out
ft_vjp_p!(::Reaction, out, u, p, t, ν) = out

function advection_reaction()
    n = 64
    x = [(c + 0.5) / n for c in 0:(n - 1)]
    u0 = vcat(1 .+ sin.(2π .* x) ./ 2, zeros(n))
    return ToyProblem("advection_reaction", 2n, u0, [1.0, 50.0, 25.0], ["v", "k1", "k2"], (0.0, 0.5),
        Dict{String, AbstractOperator}("transport" => Transport(n), "reaction" => Reaction(n)),
        Dict{String, AbstractMapOperator}(), ["A" => collect(1:n), "B" => collect((n + 1):(2n))],
        Function[u -> [sum(u)]], nothing, false)
end

# ---------------------------------------------------------------------------
# 1.6 source_decay_map, p = (a, r)

struct Source <: AbstractOperator end
opname(::Source) = "source"
structure(::Source) = DenseStructure()
depends_on_t(::Source) = false
linear_in_u(::Source) = true

rhs!(::Source, out, u, p, t) = (out[1] += p[1]; out)
jvp_u!(::Source, out, u, p, t, v) = out
vjp_u!(::Source, out, u, p, t, w) = out
vjp_p!(::Source, out, u, p, t, w) = (out[1] += w[1]; out)
jac!(::Source, out, u, p, t) = out
ft!(::Source, out, u, p, t) = out
hess_u!(::Source, out, u, p, t, μ) = out
hess_p!(::Source, out, u, p, t, μ) = out
ft_vjp_u!(::Source, out, u, p, t, ν) = out
ft_vjp_p!(::Source, out, u, p, t, ν) = out

"""The map `u ← u e^{-rΔ}` (toy-problems.md §1.6). It takes no discrete decisions, so its mask is empty."""
struct DecayMap <: AbstractMapOperator end
opname(::DecayMap) = "decay"
mask_length(::DecayMap, N) = 0

function apply!(::DecayMap, out, mask, u, p, t, Δ)
    out[1] = u[1] * exp(-p[2] * Δ)
    return out
end
function map_vjp!(::DecayMap, out_u, out_p, u, p, t, Δ, mask, w)
    e = exp(-p[2] * Δ)
    out_u[1] += e * w[1]
    out_p[2] += -Δ * u[1] * e * w[1]
    return nothing
end
function map_jvp!(::DecayMap, out, u, p, t, Δ, mask, v_u, v_p)
    e = exp(-p[2] * Δ)
    out[1] = e * v_u[1] + (-Δ * u[1] * e) * v_p[2]
    return out
end

function source_decay_map()
    return ToyProblem("source_decay_map", 1, [0.0], [1.0, 0.5], ["a", "r"], (0.0, 4.0),
        Dict{String, AbstractOperator}("source" => Source()),
        Dict{String, AbstractMapOperator}("decay" => DecayMap()), ["u" => [1]], Function[], nothing, false)
end

# ---------------------------------------------------------------------------

"""
    toy_problem(model) -> ToyProblem

The toy problem a conformance case's `model` names: `toy:<problem>` or
`toy:<problem>/<variant>` (toy-problems.md §2). Accepted only from the test
harness; a real run config names an `.esm` document.
"""
function toy_problem(model::AbstractString)
    startswith(model, "toy:") || throw(ConfigError("model '$model' is not a toy problem (toy:<problem>)"))
    name = model[5:end]
    parts = split(name, '/')
    base = parts[1]
    variant = length(parts) > 1 ? parts[2] : nothing
    base == "van_der_pol" && return van_der_pol(something(variant, ""))
    variant === nothing || throw(ConfigError("toy problem $base has no variants"))
    base == "forced_decay" && return forced_decay()
    base == "robertson" && return robertson()
    base == "photostationary_cells" && return photostationary_cells()
    base == "advection_reaction" && return advection_reaction()
    base == "source_decay_map" && return source_decay_map()
    throw(ConfigError("unknown toy problem '$base'"))
end

"""The names of every toy problem and variant (toy-problems.md §1)."""
const TOY_MODELS = ["toy:forced_decay", "toy:van_der_pol/nonstiff", "toy:van_der_pol/stiff", "toy:robertson",
    "toy:photostationary_cells", "toy:advection_reaction", "toy:source_decay_map"]
