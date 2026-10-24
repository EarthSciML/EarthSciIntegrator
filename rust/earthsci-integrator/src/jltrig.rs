//! `sin` and `cos` computed exactly as Julia's `Base.sin` and `Base.cos`
//! (Julia 1.12, `base/special/trig.jl` and `base/special/rem_pio2.jl`), for
//! the toy problems' time-dependent coefficients.
//!
//! The toy problems are written by hand in each language (D53), and the
//! cross-language check compares Julia's and Rust's results to 1e-10. An
//! adaptive Rosenbrock23 leaf magnifies a one-ulp difference in a
//! right-hand side into different step sizes (its error estimate cancels to
//! O(h³)), so both languages must evaluate `sin` and `cos` identically.
//! Neither the platform's libm (glibc, correctly rounded) nor the `libm`
//! crate (FreeBSD msun) does what Julia does: Julia's polynomial kernels and
//! Cody–Waite reduction use `muladd`, which Julia contracts into fused
//! multiply-adds on hardware that has them. This port uses `f64::mul_add`
//! there, so it matches Julia on such hardware (every x86-64 and aarch64
//! machine CI runs on).
//!
//! Measured differences (photostationary_cells): at
//! `f64::from_bits(0x4015fdbc370f36da)` glibc's sin is `0xbfe6a09cb1129215`
//! and Julia's `0xbfe6a09cb1129214`; at `0x3fe9222eaaf93691` Julia's sin is
//! `0x3fe6a0c2b3ae3d36` and glibc's and msun's `0x3fe6a0c2b3ae3d37`.
//!
//! Only the reduction Julia uses for `|x| < 2²⁰·π/2` is ported (Cody–Waite);
//! larger arguments, which no toy problem reaches, panic.
//!
//! The kernels are from FDLIBM, as Julia's are:
//!
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//!
//! Developed at SunPro, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this
//! software is freely granted, provided that this notice
//! is preserved.

// The constants are written as FDLIBM and Julia write them.
#![allow(clippy::excessive_precision)]

const DS1: f64 = -1.666_666_666_666_663_243_48e-1;
const DS2: f64 = 8.333_333_333_322_489_461_24e-3;
const DS3: f64 = -1.984_126_982_985_794_931_34e-4;
const DS4: f64 = 2.755_731_370_707_006_767_89e-6;
const DS5: f64 = -2.505_076_025_340_686_341_95e-8;
const DS6: f64 = 1.589_690_995_211_550_102_21e-10;

const DC1: f64 = 4.166_666_666_666_660_190_37e-2;
const DC2: f64 = -1.388_888_888_887_410_957_49e-3;
const DC3: f64 = 2.480_158_728_947_672_941_78e-5;
const DC4: f64 = -2.755_731_435_139_066_330_35e-7;
const DC5: f64 = 2.087_572_321_298_174_827_90e-9;
const DC6: f64 = -1.135_964_755_778_819_482_65e-11;

/// Julia's `@horner(x, a, b, c)`: `muladd(x, muladd(x, c, b), a)`.
#[inline]
fn horner3(x: f64, a: f64, b: f64, c: f64) -> f64 {
    x.mul_add(x.mul_add(c, b), a)
}

#[inline]
fn horner2(x: f64, a: f64, b: f64) -> f64 {
    x.mul_add(b, a)
}

/// `sin` on `[−π/4, π/4]` of a double-double `hi + lo`.
fn sin_kernel_dd(hi: f64, lo: f64) -> f64 {
    let y2 = hi * hi;
    let y4 = y2 * y2;
    let r = horner3(y2, DS2, DS3, DS4) + y2 * y4 * horner2(y2, DS5, DS6);
    let y3 = y2 * hi;
    hi - ((y2 * (0.5 * lo - y3 * r) - lo) - y3 * DS1)
}

/// `sin` on `[−π/4, π/4]`.
fn sin_kernel(y: f64) -> f64 {
    let y2 = y * y;
    let y4 = y2 * y2;
    let r = horner3(y2, DS2, DS3, DS4) + y2 * y4 * horner2(y2, DS5, DS6);
    let y3 = y2 * y;
    y + y3 * (DS1 + y2 * r)
}

/// `cos` on `[−π/4, π/4]` of a double-double `hi + lo`.
fn cos_kernel_dd(hi: f64, lo: f64) -> f64 {
    let y2 = hi * hi;
    let y4 = y2 * y2;
    let r = y2 * horner3(y2, DC1, DC2, DC3) + y4 * y4 * horner3(y2, DC4, DC5, DC6);
    let half = 0.5 * y2;
    let w = 1.0 - half;
    w + (((1.0 - w) - half) + (y2 * r - hi * lo))
}

/// `cos` on `[−π/4, π/4]`.
fn cos_kernel(y: f64) -> f64 {
    let y2 = y * y;
    let y4 = y2 * y2;
    let r = y2 * horner3(y2, DC1, DC2, DC3) + y4 * y4 * horner3(y2, DC4, DC5, DC6);
    let half = 0.5 * y2;
    let w = 1.0 - half;
    w + (((1.0 - w) - half) + (y2 * r))
}

fn highword(x: f64) -> u32 {
    (x.to_bits() >> 32) as u32
}

fn poshighword(x: f64) -> u32 {
    highword(x) & 0x7fff_ffff
}

fn cody_waite_2c(x: f64, fnv: f64, n: i64) -> (i64, f64, f64) {
    const PIO2_1: f64 = 1.570_796_326_734_125_614_17e0;
    const PIO2_1T: f64 = 6.077_100_506_506_192_249_32e-11;
    let z = (-fnv).mul_add(PIO2_1, x);
    let y1 = (-fnv).mul_add(PIO2_1T, z);
    let y2 = (-fnv).mul_add(PIO2_1T, z - y1);
    (n, y1, y2)
}

fn cody_waite_ext(x: f64, xhp: u32) -> (i64, f64, f64) {
    const PIO2_1: f64 = 1.570_796_326_734_125_614_17e0;
    const PIO2_1T: f64 = 6.077_100_506_506_192_249_32e-11;
    const PIO2_2: f64 = 6.077_100_506_303_965_976_60e-11;
    const PIO2_2T: f64 = 2.022_266_248_795_950_631_54e-21;
    const PIO2_3: f64 = 2.022_266_248_711_166_455_80e-21;
    const PIO2_3T: f64 = 8.478_427_660_368_899_569_97e-32;
    let fnv = (x * (2.0 / std::f64::consts::PI)).round_ties_even();
    let mut r = (-fnv).mul_add(PIO2_1, x);
    let mut w = fnv * PIO2_1T;
    let j = xhp >> 20;
    let mut y1 = r - w;
    let mut i = j.wrapping_sub((highword(y1) >> 20) & 0x7ff);
    if i > 16 {
        let t = r;
        w = fnv * PIO2_2;
        r = t - w;
        w = fnv.mul_add(PIO2_2T, -((t - r) - w));
        y1 = r - w;
        i = j.wrapping_sub((highword(y1) >> 20) & 0x7ff);
        if i > 49 {
            let t = r;
            w = fnv * PIO2_3;
            r = t - w;
            w = fnv.mul_add(PIO2_3T, -((t - r) - w));
            y1 = r - w;
        }
    }
    let y2 = (r - y1) - w;
    (fnv as i64, y1, y2)
}

/// Julia's `rem_pio2_kernel` for `|x| < 2²⁰·π/2`: the quadrant and the
/// remainder as a double-double.
fn rem_pio2(x: f64) -> (i64, f64, f64) {
    let xhp = poshighword(x);
    if xhp <= 0x400f_6a7a {
        if (xhp & 0xfffff) == 0x921fb {
            return cody_waite_ext(x, xhp);
        }
        if xhp <= 0x4002_d97c {
            return if x > 0.0 {
                cody_waite_2c(x, 1.0, 1)
            } else {
                cody_waite_2c(x, -1.0, -1)
            };
        }
        return if x > 0.0 {
            cody_waite_2c(x, 2.0, 2)
        } else {
            cody_waite_2c(x, -2.0, -2)
        };
    }
    if xhp <= 0x401c_463b {
        if xhp <= 0x4015_fdbc {
            if xhp == 0x4012_d97c {
                return cody_waite_ext(x, xhp);
            }
            return if x > 0.0 {
                cody_waite_2c(x, 3.0, 3)
            } else {
                cody_waite_2c(x, -3.0, -3)
            };
        }
        if xhp == 0x4019_21fb {
            return cody_waite_ext(x, xhp);
        }
        return if x > 0.0 {
            cody_waite_2c(x, 4.0, 4)
        } else {
            cody_waite_2c(x, -4.0, -4)
        };
    }
    assert!(
        xhp < 0x4139_21fb,
        "jltrig: |x| = {} needs Payne–Hanek reduction, which is not ported",
        x.abs()
    );
    cody_waite_ext(x, xhp)
}

/// Julia's `sin(x::Float64)`.
pub fn sin(x: f64) -> f64 {
    let absx = x.abs();
    if absx < std::f64::consts::PI / 4.0 {
        if absx < f64::EPSILON.sqrt() {
            return x;
        }
        return sin_kernel(x);
    } else if x.is_nan() {
        return x;
    } else if x.is_infinite() {
        return f64::NAN;
    }
    let (n, hi, lo) = rem_pio2(x);
    match n & 3 {
        0 => sin_kernel_dd(hi, lo),
        1 => cos_kernel_dd(hi, lo),
        2 => -sin_kernel_dd(hi, lo),
        _ => -cos_kernel_dd(hi, lo),
    }
}

/// Julia's `cos(x::Float64)`.
pub fn cos(x: f64) -> f64 {
    let absx = x.abs();
    if absx < std::f64::consts::PI / 4.0 {
        if absx < (f64::EPSILON / 2.0).sqrt() {
            return 1.0;
        }
        return cos_kernel(x);
    } else if x.is_nan() {
        return x;
    } else if x.is_infinite() {
        return f64::NAN;
    }
    let (n, hi, lo) = rem_pio2(x);
    match n & 3 {
        0 => cos_kernel_dd(hi, lo),
        1 => -sin_kernel_dd(hi, lo),
        2 => -cos_kernel_dd(hi, lo),
        _ => sin_kernel_dd(hi, lo),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values from Julia 1.12 (`reinterpret(UInt64, sin(x))`), including the
    /// two arguments where glibc or msun differ from Julia.
    #[test]
    fn matches_julia() {
        let cases: [(u64, u64, u64); 3] = [
            (0x4015fdbc370f36da, 0xbfe6a09cb1129214, 0x3fe6a0a01bebc47c),
            (0x3fe9222eaaf93691, 0x3fe6a0c2b3ae3d36, 0x3fe6a07a1915fcd4),
            (0x4002d98954e06430, 0x3fe6a07a1915fcd5, 0xbfe6a0c2b3ae3d36),
        ];
        for (x, s, c) in cases {
            let x = f64::from_bits(x);
            assert_eq!(sin(x).to_bits(), s, "sin({x})");
            assert_eq!(cos(x).to_bits(), c, "cos({x})");
        }
        assert_eq!(sin(1e-9), 1e-9);
        assert_eq!(cos(1e-9), 1.0);
    }

    #[test]
    fn accurate() {
        let mut x = -40.0;
        while x < 40.0 {
            assert!((sin(x) - x.sin()).abs() <= 2.0 * f64::EPSILON, "sin {x}");
            assert!((cos(x) - x.cos()).abs() <= 2.0 * f64::EPSILON, "cos {x}");
            x += 0.0123;
        }
    }
}
