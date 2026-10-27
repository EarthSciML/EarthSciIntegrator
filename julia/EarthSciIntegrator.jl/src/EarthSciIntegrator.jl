"""
    EarthSciIntegrator

Julia implementation of the EarthSciIntegrator spec (`spec/` at the repository
root).

M1: the operator interface (spec/operator-interface.md), the six
hand-written toy problems (spec/toy-problems.md), the coefficient tables
(spec/tableaus), every v1 method family (explicit Runge–Kutta, Rosenbrock,
SDIRK and ESDIRK, IMEX Euler and the θ-method) with the Newton iteration and
the controllers (spec/methods.md §3–§12), `dense_lu` and batched `block_lu`
linear solves, the schedule interpreter for `leaf`, `sequence`, `strang`,
`subcycle` and `map`, the tape with replay and checkpoints (spec/tape.md),
functionals at the final time, at output times and integrated over time
(single-leaf schedules), and the discrete adjoint of all of these
(spec/adjoint.md §4.1–§4.5, §5, §6).

M2 stage 1: the planner (spec/plan.md) up to what needs EarthSciASTDiff.
Here: the RFC 8785 writer of plan.json, the run-config schema check, the
band calculus of pass 2 and the [`PlanBackend`](@ref) interface. With
`import EarthSciAST`, the extension EarthSciIntegratorEarthSciASTExt adds
[`plan`](@ref): reading and flattening the model, terms, owners and
selection, locality, DOF sets, the resolved schedule, time, step, partition,
precision, checkpoint placeholder and refusals. Bands, structure, Gershgorin
stiffness, derived documents and validation come through `PlanBackend` in
stage 2.
"""
module EarthSciIntegrator

using JSON
using LinearAlgebra: dot

"""Version of the integrator spec (the run config's `esm_integrator` key) this package implements."""
const SPEC_VERSION = v"0.1.0"

"""
    UnsupportedError(msg)

Raised when a run config asks for something this implementation does not
support yet (spec/run-config.md §8 rule 16): a method family not implemented
in this stage, a combinator not yet interpreted, and the like.
"""
struct UnsupportedError <: Exception
    msg::String
end
Base.showerror(io::IO, e::UnsupportedError) = print(io, "UnsupportedError: ", e.msg)

"""
    ConfigError(msg)

Raised when a run config is invalid for a reason the schema cannot check
(spec/run-config.md §8).
"""
struct ConfigError <: Exception
    msg::String
end
Base.showerror(io::IO, e::ConfigError) = print(io, "ConfigError: ", e.msg)

"""
    ReplayMismatchError(msg)

Raised when a tape does not fit the run it is replayed for (spec/tape.md §3):
a record missing or of the wrong kind, an interval other than the one the
schedule gives the invocation, a map decision that flips, or clips outside the
leaf's steps or its clip domain.
"""
struct ReplayMismatchError <: Exception
    msg::String
end
Base.showerror(io::IO, e::ReplayMismatchError) = print(io, "ReplayMismatchError: ", e.msg)

include("operators.jl")
include("toys.jl")
include("tableaus.jl")
include("control.jl")
include("erk.jl")
include("linsolve.jl")
include("rosenbrock.jl")
include("sdirk.jl")
include("schedule.jl")
include("tape.jl")
include("leaf.jl")
include("interpreter.jl")
include("adjoint.jl")
include("results.jl")

# The planner (spec/plan.md), M2 stage 1: the parts that need no EarthSciAST.
# The passes are in ext/EarthSciIntegratorEarthSciASTExt.jl; what needs
# EarthSciASTDiff plugs in through `PlanBackend` (planner/bands.jl).
include("planner/jcs.jl")
include("planner/refusal.jl")
include("planner/schema.jl")
include("planner/bands.jl")

"""
    odefunction(ops::OperatorSum; inplace=true)
    odefunction(leaf::LeafNode; inplace=true)

The sum of a leaf's operators as a `SciMLBase.ODEFunction` with `jac`, `tgrad`,
`vjp` and `jvp` filled from the operator calls (operator-interface.md §5,
D33). Needs `SciMLBase` loaded (package extension).
"""
function odefunction end

export SPEC_VERSION, UnsupportedError, ConfigError, ReplayMismatchError
export rhs!, jvp_u!, vjp_u!, vjp_p!, jac!, ft!, hess_u!, hess_p!, ft_vjp_u!, ft_vjp_p!
export apply!, map_vjp!, map_jvp!
export toy_problem, TABLEAUS, build_run, forward!, replay, gradient, odefunction
export plan, plan_dict, plan_json, PlannerRefusal, PlannerStageError, PlanBackend, PendingBackend, BandEntry, jcs

end
