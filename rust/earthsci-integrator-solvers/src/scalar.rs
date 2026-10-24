//! The state element type `T` (spec/operator-interface.md §1, D13).
//!
//! State vectors are `f32` or `f64`. Time, step sizes, error norms,
//! parameters and adjoint accumulators are always `f64`.

use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// A floating-point element type for state vectors: `f32` or `f64`.
pub trait Scalar:
    Copy
    + Send
    + Sync
    + Debug
    + Display
    + PartialOrd
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + 'static
{
    /// The type's name, `"f32"` or `"f64"`.
    const NAME: &'static str;
    /// Machine epsilon of the type.
    const EPSILON: f64;
    /// Zero.
    fn zero() -> Self;
    /// Converts from `f64`, rounding to nearest.
    fn from_f64(x: f64) -> Self;
    /// Converts to `f64` exactly.
    fn to_f64(self) -> f64;
    /// Whether the value is finite.
    fn is_finite(self) -> bool;
    /// Bit pattern, widened to 64 bits, for bitwise comparisons.
    fn to_bits_u64(self) -> u64;
}

impl Scalar for f64 {
    const NAME: &'static str = "f64";
    const EPSILON: f64 = f64::EPSILON;
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    #[inline]
    fn to_bits_u64(self) -> u64 {
        self.to_bits()
    }
}

impl Scalar for f32 {
    const NAME: &'static str = "f32";
    const EPSILON: f64 = f32::EPSILON as f64;
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
    #[inline]
    fn to_bits_u64(self) -> u64 {
        self.to_bits() as u64
    }
}

/// Julia's `eps(x)` for `f64`: the distance from `|x|` to the next larger
/// float. `eps(0.0)` is the smallest subnormal. Used for the tstop tolerance
/// (spec/methods.md §10) and the step-size floor (§8.6).
pub fn eps_f64(x: f64) -> f64 {
    let a = x.abs();
    if !a.is_finite() {
        return f64::NAN;
    }
    next_up(a) - a
}

/// The next representable `f64` above `x` (Julia's `nextfloat`), for finite `x`.
pub fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    let bits = x.to_bits();
    if x > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eps_matches_julia() {
        assert_eq!(eps_f64(1.0), f64::EPSILON);
        assert_eq!(eps_f64(0.0), f64::from_bits(1));
        assert_eq!(eps_f64(2.0), 2.0 * f64::EPSILON);
        assert_eq!(eps_f64(-2.0), 2.0 * f64::EPSILON);
        assert_eq!(next_up(1.0), 1.0 + f64::EPSILON);
        assert_eq!(next_up(-1.0), -1.0 + f64::EPSILON / 2.0);
    }
}
