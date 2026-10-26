"""
The planner's passes (spec/plan.md), M2 stage 1: reading and flattening the
model with EarthSciAST (§4), terms, owners, selection and operators (§5.1),
pass 2 over the model (§5.2), DOF sets (§5.3), the resolved schedule, time,
step, partition, precision and checkpoint placeholder (§7), and the driver
that assembles `plan.json`. What needs EarthSciASTDiff comes through
`EarthSciIntegrator.PlanBackend`.

An extension rather than part of the package proper because EarthSciAST's
compat pins SciMLBase to 2, which would hold the M1 tests' OrdinaryDiffEq
oracle at an older SDIRK (see src/planner/bands.jl).
"""
module EarthSciIntegratorEarthSciASTExt

import EarthSciAST
import EarthSciIntegrator: plan, plan_dict, plan_json
using EarthSciIntegrator: SPEC_VERSION, TABLEAUS, has_error_estimate, macro_bounds, PlannerRefusal, PlannerStageError,
    refuse, jptr, jcs, schema_errors, run_config_schema, BandEntry, Axes, axis_role, classify_bands, coarser, LOCALITY_ORDER, PlanBackend,
    PendingBackend, term_bands, operator_structure, gershgorin_bound, derived_and_validation
using JSON
using SHA: sha256
using OrderedCollections: OrderedDict

include("planner/inputs.jl")
include("planner/model.jl")
include("planner/terms.jl")
include("planner/passes.jl")
include("planner/dofsets.jl")
include("planner/time.jl")
include("planner/schedule_plan.jl")
include("planner/settings.jl")
include("planner/planner.jl")

end
