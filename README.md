# EarthSciIntegrator

A time integrator for large-scale [EarthSciML](https://github.com/EarthSciML) `.esm` models. There is one written specification, implemented natively in **Julia** and in **Rust**. A shared
conformance suite checks each implementation against the other.

**Documentation:** <https://earthsci.dev/integrator/>.

> **Status: under development (spec version 0.1.0).** Both implementations cover milestone M1: the core
> methods, step-size control, operator-splitting combinators, the tape and discrete adjoints, all on
> hand-written toy problems. Milestone M2, the planner, is in progress. Nothing here has run a production model
> yet. See the [milestones](docs/architecture.md#14-milestones).

## What it does

A **run configuration** (a JSON document) describes how to integrate a model:
- which operators make up the model's right-hand side;
- which method integrates each operator, and at what step size or tolerance;
- how the operators are split and subcycled;
- what to differentiate.

A **planner** reads the run configuration together with an `.esm` model. It infers each operator's
locality and Jacobian structure, and writes an inspectable, cacheable **plan**. The **runtime** runs the
plan forward and records a tape, and then optionally runs it backward to compute a discrete adjoint.

Design goals:
- **Methods:** explicit (ERK/SSP), Rosenbrock, SDIRK/ESDIRK, IMEX, and legacy fixed-step and
  split-explicit schemes. Nested nonlinear solves, with index-1 DAEs specified for later.
- **Structure-aware linear algebra by default:** batched block LU, banded column solves, and Krylov
  methods with block-Jacobi or ASM preconditioning. It never forms a dense Jacobian.
- **Operator splitting as a first-class schedule tree** (`leaf`, `sequence`, `strang`, `subcycle`, `map`,
  …) whose callbacks and tolerances actually propagate.
- **Discrete adjoints of whole simulations,** with respect to parameters, the initial state, gridded fields
  and forcing. Every nonlinear solve is differentiated through the implicit function theorem.
- **Partitioning owned by the integrator:** by operator locality (pointwise, column, stencil, global),
  over threads and MPI. Results are bitwise reproducible for a fixed decomposition.
- **Mixed precision per leaf:** f16/bf16 through f64, with time, norms and reductions always in f64.

In Julia, the library uses the SciMLBase interfaces. It uses OrdinaryDiffEq's tableaus and controller
constants, Krylov.jl and NonlinearSolve.jl. The Rust implementation builds on faer.

## Quick start (Julia)

```julia
using EarthSciIntegrator

config = Dict(
    "esm_integrator" => "0.1.0",
    "model" => "toy:forced_decay",            # u' = -k u + sin(ωt)
    "schedule" => Dict("leaf" => "decay", "alg" => "RK4", "dt" => 0.1),
    "step" => Dict("dt" => 2),
    "adjoint" => Dict(
        "functional" => Dict("sum" => "u", "over" => "final"),
        "wrt" => ["u0", "params:numeric"],
    ),
)

run = build_run(config; tspan = (0.0, 2.0))   # validates; refuses with ConfigError
forward!(run)                                 # integrates and records the tape
J, ū0, p̄ = gradient(run)                      # discrete adjoint of the recorded steps
u, outputs, q = replay(run)                   # bit-for-bit replay on the tape
```

The same program in Rust is
[`docs/api/examples/rust/forward_and_gradient.rs`](docs/api/examples/rust/forward_and_gradient.rs). Both
programs run in CI.

## Repository layout

| Path | Contents |
|---|---|
| [`spec/`](spec/) | The normative specification: run config, planner and plan, methods and controllers, operator interface, tape, adjoint, toy problems, JSON Schemas, and method tableaus as data |
| [`julia/EarthSciIntegrator.jl`](julia/EarthSciIntegrator.jl) | The Julia package |
| [`rust/`](rust/) | The Rust crates: `earthsci-integrator` (core), `-planner`, `-solvers`, `-mpi` |
| [`tests/conformance/`](tests/conformance/) | The shared conformance suite: schema fixtures, planner fixtures and goldens, and the M1/M2 cases |
| [`docs/`](docs/) | [Architecture](docs/architecture.md), [decisions and tradeoffs](docs/decisions.md), the [requirements survey](docs/survey.md), and [RFCs](docs/rfcs/) for changes needed in sibling repositories |
| [`tools/`](tools/) | Scripts for goldens, tableau extraction and syncing package data |

## Building and testing

```sh
# Conformance tier 0: schemas and fixtures
pip install "jsonschema>=4.18" "earthsci-ast==0.8.0"
python tests/conformance/validate_schemas.py --require-esm

# Julia (1.10+)
julia --project=julia/EarthSciIntegrator.jl -e 'using Pkg; Pkg.test()'

# Rust
cargo test --manifest-path rust/Cargo.toml --workspace
```

Both test suites write conformance result files. CI then compares the Julia results with the Rust results
(conformance tier 3), using
[`tests/conformance/m1/compare_results.py`](tests/conformance/m1/compare_results.py).

The method tableaus live in `spec/tableaus/`. Each package keeps its own copy, because a released package
cannot reach `spec/`. After changing a tableau, run `tools/sync-package-data.sh` to refresh the copies.

## Dependencies on sibling packages

The specification targets [EarthSciAST](https://github.com/EarthSciML/EarthSciAST) 0.8 (esm format 1.3.0),
which loads and flattens models, and [EarthSciASTDiff](https://github.com/EarthSciML/EarthSciASTDiff) 0.3,
which provides Jacobian band models and block factorization plans. The integrator also needs some upstream
changes, which are drafted as RFCs in [`docs/rfcs/`](docs/rfcs/).

## License

[GNU Affero General Public License v3.0](LICENSE).
