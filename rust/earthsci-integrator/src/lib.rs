//! Rust implementation of the EarthSciIntegrator spec (`spec/` at the
//! repository root).
//!
//! Crate layout (M1: explicit Runge–Kutta and Rosenbrock methods):
//! - this crate: the operator interface ([`operator`]), the toy problems
//!   ([`toy`]), run-config parsing ([`config`]), the leaf ([`leaf`]), the
//!   schedule interpreter ([`schedule`]), the tape and checkpoints
//!   ([`tape`]) and the discrete adjoint ([`adjoint`]);
//! - `earthsci-integrator-solvers`: coefficient tables, method steps, error
//!   norm, controllers, first step and the direct linear solvers,
//!   independent of the operator interface.

pub mod adjoint;
pub mod config;
pub mod conformance;
pub mod jltrig;
pub mod leaf;
pub mod operator;
pub mod schedule;
pub mod tape;
pub mod toy;

pub use earthsci_integrator_solvers::Scalar;

/// Version of the integrator spec (the run config's `esm_integrator` key)
/// this crate implements.
pub const SPEC_VERSION: &str = "0.1.0";

/// Errors: refusals of a run config or schedule, and failed runs.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The run config is invalid or uses something this implementation refuses.
    Config(String),
    /// A leaf's method is in the catalogue but not implemented yet.
    Unimplemented(String),
    /// The run ended with a non-success return code.
    Run(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Config(s) => write!(f, "run config refused: {s}"),
            Error::Unimplemented(s) => write!(f, "not implemented: {s}"),
            Error::Run(s) => write!(f, "run failed: {s}"),
        }
    }
}

impl std::error::Error for Error {}
