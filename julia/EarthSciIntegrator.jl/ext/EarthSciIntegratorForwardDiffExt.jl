"""
`Dual` numbers in the error norm (spec/methods.md §7, D41): the controller acts
on primal values only, so partials are stripped before the norm
(`dt_tangents = false`, spec/adjoint.md §2.3).
"""
module EarthSciIntegratorForwardDiffExt

using EarthSciIntegrator
import ForwardDiff

EarthSciIntegrator.primal(x::ForwardDiff.Dual) = EarthSciIntegrator.primal(ForwardDiff.value(x))

end
