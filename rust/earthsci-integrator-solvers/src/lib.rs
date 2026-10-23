//! EarthSciIntegrator methods, step-size control and (from the next M1
//! stages) linear algebra: everything of spec/methods.md that does not need
//! to know what an operator is.
//!
//! - [`tableau`]: the coefficient tables of `spec/tableaus/`, embedded at
//!   compile time (D54).
//! - [`erk`]: explicit Runge–Kutta steps, Butcher and Shu–Osher forms (§3).
//! - [`norm`]: the error norm (§7).
//! - [`controller`]: PI, I and PID step-size control (§8).
//! - [`initdt`]: the first step (§9).
//!
//! Methods see the leaf's right-hand side as a closure `f(out, u, t)` that
//! overwrites `out`; the core crate builds it from the leaf's operators.

pub mod controller;
pub mod erk;
pub mod initdt;
pub mod norm;
pub mod scalar;
pub mod tableau;

pub use scalar::Scalar;
