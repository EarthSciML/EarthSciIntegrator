"""
The leaf's sum of operators as a `SciMLBase.ODEFunction` (operator-interface.md
§5, D33), so OrdinaryDiffEq can integrate the same problem as an oracle.
"""
module EarthSciIntegratorSciMLBaseExt

using EarthSciIntegrator
using EarthSciIntegrator: OperatorSum, DenseJacobian, eval_rhs!, jac!, ft!, vjp_u!, jvp_u!, depends_on_t
import SciMLBase

function EarthSciIntegrator.odefunction(ops::OperatorSum; inplace::Bool = true)
    jac = (J, u, p, t) -> (fill!(J, zero(eltype(J))); jac!(ops, DenseJacobian(J), u, p, t); J)
    tgrad = (dT, u, p, t) -> (fill!(dT, zero(eltype(dT))); ft!(ops, dT, u, p, t); dT)
    vjp = (out, w, u, p, t) -> (fill!(out, zero(eltype(out))); vjp_u!(ops, out, u, p, t, w); out)
    jvp = (out, v, u, p, t) -> (fill!(out, zero(eltype(out))); jvp_u!(ops, out, u, p, t, v); out)
    if inplace
        f! = (du, u, p, t) -> (eval_rhs!(du, ops, u, p, t); nothing)
        return SciMLBase.ODEFunction{true}(f!; jac, tgrad, vjp, jvp)
    else
        f = function (u, p, t)
            du = zeros(promote_type(eltype(u), eltype(p), typeof(t)), length(u))
            eval_rhs!(du, ops, u, p, t)
            return du
        end
        return SciMLBase.ODEFunction{false}(f)
    end
end

EarthSciIntegrator.odefunction(leaf::EarthSciIntegrator.LeafNode; kw...) = EarthSciIntegrator.odefunction(leaf.ops; kw...)

end
