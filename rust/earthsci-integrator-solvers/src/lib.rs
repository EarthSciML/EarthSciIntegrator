//! EarthSciIntegrator methods, step-size control and linear algebra:
//! everything of spec/methods.md that does not need to know what an
//! operator is.
//!
//! - [`tableau`]: the coefficient tables of `spec/tableaus/`, embedded at
//!   compile time (D54).
//! - [`erk`]: explicit Runge–Kutta steps, Butcher and Shu–Osher forms (§3).
//! - [`rosenbrock`]: Rosenbrock steps, transformed and ode23s forms (§4).
//! - [`linalg`]: matrix storage and the `dense_lu` and `block_lu` solvers.
//! - [`norm`]: the error norm (§7).
//! - [`controller`]: PI, I and PID step-size control (§8).
//! - [`initdt`]: the first step (§9).
//!
//! Methods see the leaf's right-hand side as a closure `f(out, u, t)` that
//! overwrites `out`; the core crate builds it from the leaf's operators.

pub mod controller;
pub mod erk;
pub mod initdt;
pub mod linalg;
pub mod norm;
pub mod rosenbrock;
pub mod scalar;
pub mod tableau;

pub use scalar::Scalar;

/// The leaf's right-hand side as methods see it: `f(out, u, t)` overwrites `out`.
pub type RhsFn<'a, T> = dyn FnMut(&mut [T], &[T], f64) + 'a;

/// A scaled error norm `norm(e, u_n, u_{n+1})` (spec/methods.md §7).
pub type NormFn<'a, T> = dyn FnMut(&[f64], &[T], &[T]) -> f64 + 'a;
