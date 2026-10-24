"""
`Dual` numbers in the error norm (spec/methods.md §7, D41): the controller acts
on primal values only, so partials are stripped before the norm
(`dt_tangents = false`, spec/adjoint.md §2.3).
"""
module EarthSciIntegratorForwardDiffExt

using EarthSciIntegrator
import ForwardDiff

EarthSciIntegrator.primal(x::ForwardDiff.Dual) = EarthSciIntegrator.primal(ForwardDiff.value(x))

# A converged implicit stage keeps its primal value bitwise and takes the
# tangent of the exact stage equation (sdirk.jl, `stage_tangent`).
EarthSciIntegrator.with_value(x::ForwardDiff.Dual{T}, v) where {T} =
    ForwardDiff.Dual{T}(EarthSciIntegrator.with_value(ForwardDiff.value(x), v), ForwardDiff.partials(x))

end
