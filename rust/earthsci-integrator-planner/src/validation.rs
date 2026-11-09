//! Pass 8, validation (spec/plan.md §8, D76), and the Gershgorin bound of
//! pass 5 (§5.4).
//!
//! Each check returns the quantity it measured, which goes to
//! `plan.report.json`, or the message of its refusal.

use crate::numeric::{Dofs, Program, Sample, Sparse};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};

/// ε = 2⁻⁵², the spacing of binary64 at 1.
pub const EPS: f64 = f64::EPSILON;

/// The Gershgorin bound (§5.4): the largest absolute row sum of the band
/// Jacobian `jb` over the rows `rows` marks (the operator's `writes` DOFs).
/// `jb` already sums every band's contribution to an entry.
pub fn gershgorin(jb: &Sparse, rows: &[bool]) -> f64 {
    let mut sums: BTreeMap<usize, f64> = BTreeMap::new();
    for ((r, _), v) in jb {
        if rows[*r] {
            *sums.entry(*r).or_insert(0.0) += v.abs();
        }
    }
    sums.values().fold(0.0, |a, b| a.max(*b))
}

/// What `band_model` measured at one sampled state.
#[derive(Debug, Clone, Default)]
pub struct BandModelReport {
    /// The largest `|J^B − J^I| / τ` over the pattern entries, where an
    /// entry the central difference fails but one at a smaller step accepts
    /// (§8.2, "Kinks") contributes that smaller step's ratio.
    pub worst_entry: f64,
    /// The largest off-pattern difference over its floor.
    pub worst_off_pattern: f64,
    /// The number of colours (right-hand-side pairs).
    pub colours: usize,
    /// The number of entries the central difference failed and one at a
    /// smaller step accepted.
    pub accepted_at_smaller_step: usize,
}

impl BandModelReport {
    pub fn to_json(&self) -> Value {
        json!({
            "worst_entry_over_tolerance": self.worst_entry,
            "worst_off_pattern_over_floor": self.worst_off_pattern,
            "colours": self.colours,
            "entries_accepted_at_smaller_step": self.accepted_at_smaller_step,
        })
    }
}

/// The greedy column colouring of §8.2: columns in canonical order, each
/// taking the lowest colour that no earlier column sharing a pattern row
/// with it has taken (Curtis, Powell and Reid); a column in no pattern
/// entry takes colour 0. The Julia planner's `colour_columns` is the same.
pub fn colour_columns(n: usize, pattern: &Sparse) -> Vec<usize> {
    let mut rows_of: Vec<Vec<usize>> = vec![vec![]; n];
    for (r, c) in pattern.keys() {
        rows_of[*c].push(*r);
    }
    // The colours each row's columns have taken so far.
    let mut taken: HashMap<usize, HashSet<usize>> = HashMap::new();
    let mut colour = vec![0usize; n];
    let mut forbidden = HashSet::new();
    for (c, rows) in rows_of.iter().enumerate() {
        forbidden.clear();
        for r in rows {
            if let Some(t) = taken.get(r) {
                forbidden.extend(t.iter().copied());
            }
        }
        let k = (0..)
            .find(|k| !forbidden.contains(k))
            .expect("a free colour");
        for r in rows {
            taken.entry(*r).or_default().insert(k);
        }
        colour[c] = k;
    }
    colour
}

/// Where `band_model` evaluates: a sampled state in the canonical DOF
/// order, with the sampling base `b_n` of §5.5.1 and the DOFs' element
/// names (for messages).
pub struct At<'a> {
    pub dofs: &'a Dofs,
    pub sample: &'a Sample,
    pub base: &'a [f64],
    pub names: &'a [String],
}

/// `band_model` at one sampled state (§8.1, §8.2, Rust): the band
/// Jacobian `jb` against central finite differences of the operator's
/// right-hand side `f`, rows of `rows` (its `writes` DOFs) only, and the
/// band pattern `pattern` checked to cover the true dependence.
pub fn band_model(
    f: &Program,
    at: &At,
    jb: &Sparse,
    pattern: &Sparse,
    rows: &[bool],
) -> Result<BandModelReport, String> {
    let At {
        dofs,
        sample: s,
        base,
        names,
    } = *at;
    let eval = |v: &[f64]| f.eval(dofs, v, s.t, &s.forcing);
    band_model_with(&eval, &s.u, base, names, jb, pattern, rows)
}

/// The divisors of the step at which §8.2's kink fallback takes central
/// differences again, in the order they are tried.
const SMALLER_STEPS: [f64; 2] = [10.0, 100.0];

/// A right-hand side as a function of the state alone, in canonical order.
type Rhs<'a> = dyn Fn(&[f64]) -> Result<Vec<f64>, String> + 'a;

/// A pattern entry `(row, column)` with its tolerance τ.
type Entry = ((usize, usize), f64);

/// [`band_model`] with the right-hand side as a function of the state
/// alone (`f(v)` in canonical order), at the sampled state `u`.
pub(crate) fn band_model_with(
    f: &Rhs,
    u: &[f64],
    base: &[f64],
    names: &[String],
    jb: &Sparse,
    pattern: &Sparse,
    rows: &[bool],
) -> Result<BandModelReport, String> {
    let n = u.len();
    // Steps: h_n = ε^(1/3) max(|u_n|, |b_n|) / d; the step taken is the
    // exact distance (u_n + h_n) − u_n. `d` is 1 for the check and 10, 100
    // for the kink fallback.
    let cbrt_eps = EPS.cbrt();
    let steps = |d: f64| -> Vec<f64> {
        (0..n)
            .map(|k| {
                let h = cbrt_eps * u[k].abs().max(base[k].abs()) / d;
                (u[k] + h) - u[k]
            })
            .collect()
    };
    let step = steps(1.0);
    let colour = colour_columns(n, pattern);
    let colours = colour.iter().copied().max().map_or(0, |m| m + 1);
    // The pattern columns of each row.
    let mut cols_of: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (r, c) in pattern.keys() {
        cols_of.entry(*r).or_default().push(*c);
    }
    // f_r^+ and f_r^- per pattern entry.
    let mut fpm: BTreeMap<(usize, usize), (f64, f64)> = BTreeMap::new();
    let mut report = BandModelReport {
        colours,
        ..Default::default()
    };
    let mut members_of: Vec<Vec<usize>> = vec![Vec::new(); colours];
    for (c, k) in colour.iter().enumerate() {
        members_of[*k].push(c);
    }
    // f at u moved by ±step_c in every column c of colour k.
    let pair = |k: usize, step: &[f64]| -> Result<(Vec<f64>, Vec<f64>), String> {
        let mut up = u.to_vec();
        let mut um = u.to_vec();
        for &c in &members_of[k] {
            up[c] = u[c] + step[c];
            um[c] = u[c] - step[c];
        }
        Ok((f(&up)?, f(&um)?))
    };
    for k in 0..colours {
        let (fp, fm) = pair(k, &step)?;
        for r in (0..n).filter(|r| rows[*r]) {
            let col = cols_of
                .get(&r)
                .and_then(|cs| cs.iter().find(|c| colour[**c] == k));
            match col {
                Some(&c) => {
                    fpm.insert((r, c), (fp[r], fm[r]));
                }
                None => {
                    let big_f = fp[r].abs().max(fm[r].abs());
                    let d = ((fp[r] - fm[r]) / 2.0).abs();
                    let floor = 1e3 * EPS * big_f;
                    if d > floor {
                        return Err(format!(
                            "row {} depends on a column of colour {k} the bands omit: a central difference of {d:e} over a floor of {floor:e}",
                            names[r]
                        ));
                    }
                    if d > 0.0 {
                        report.worst_off_pattern = report.worst_off_pattern.max(d / floor);
                    }
                }
            }
        }
    }
    // J^I, the central difference, of a pattern entry.
    let ji = |r: usize, c: usize| -> f64 {
        let (p, m) = fpm[&(r, c)];
        (p - m) / (2.0 * step[c])
    };
    // M_r: the largest magnitude in row r of either Jacobian.
    let mut m: BTreeMap<usize, f64> = BTreeMap::new();
    let central = fpm.keys().map(|&(r, c)| (r, ji(r, c)));
    for (r, v) in jb.iter().map(|((r, _), v)| (*r, *v)).chain(central) {
        let e = m.entry(r).or_insert(0.0);
        *e = e.max(v.abs());
    }
    // The entries the central difference fails, with their τ, by colour.
    let mut failing: BTreeMap<usize, Vec<Entry>> = BTreeMap::new();
    for (&(r, c), &(p, mm)) in &fpm {
        let vi = ji(r, c);
        let big_f = p.abs().max(mm.abs());
        let vb = jb.get(&(r, c)).copied().unwrap_or(0.0);
        let tau = 1e-6 * m[&r] + 1e3 * EPS * big_f / step[c];
        let d = (vb - vi).abs();
        if d > tau {
            failing.entry(colour[c]).or_default().push(((r, c), tau));
        } else if d > 0.0 {
            report.worst_entry = report.worst_entry.max(d / tau);
        }
    }
    if failing.is_empty() {
        return Ok(report);
    }
    // The kink fallback (§8.2): an entry the central difference fails is
    // accepted when its band value is within τ of the central difference
    // at the step divided by 10, then by 100, τ taken again at that step
    // (its rounding term grows as the step shrinks). Two evaluations per
    // smaller step, for the colours that still have failing entries.
    let smaller: Vec<Vec<f64>> = SMALLER_STEPS.iter().map(|d| steps(*d)).collect();
    for (k, entries) in failing {
        // What each entry's smaller steps gave, for the refusal message.
        let mut tried: Vec<Vec<f64>> = vec![Vec::new(); entries.len()];
        let mut open: Vec<usize> = (0..entries.len()).collect();
        for st in &smaller {
            if open.is_empty() {
                break;
            }
            let (fp, fm) = pair(k, st)?;
            open.retain(|&i| {
                let (r, c) = entries[i].0;
                let central = (fp[r] - fm[r]) / (2.0 * st[c]);
                tried[i].push(central);
                let tau = 1e-6 * m[&r] + 1e3 * EPS * fp[r].abs().max(fm[r].abs()) / st[c];
                let d = (jb.get(&(r, c)).copied().unwrap_or(0.0) - central).abs();
                if d <= tau {
                    report.worst_entry = report.worst_entry.max(d / tau);
                    report.accepted_at_smaller_step += 1;
                    false
                } else {
                    true
                }
            });
        }
        if let Some(&i) = open.first() {
            let ((r, c), tau) = entries[i];
            let t = &tried[i];
            return Err(format!(
                "J[{}, {}]: bands give {:e}, central differences {:e}, beyond the tolerance {tau:e}; central differences at the step over 10 ({:e}) and over 100 ({:e}) are beyond their tolerances too",
                names[r],
                names[c],
                jb.get(&(r, c)).copied().unwrap_or(0.0),
                ji(r, c),
                t[0],
                t[1],
            ));
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sparse(entries: &[((usize, usize), f64)]) -> Sparse {
        entries.iter().copied().collect()
    }

    #[test]
    fn gershgorin_sums_written_rows() {
        let jb = sparse(&[((0, 0), -2.0), ((0, 1), 1.0), ((1, 1), -5.0), ((2, 2), 9.0)]);
        assert_eq!(gershgorin(&jb, &[true, true, false]), 5.0);
        assert_eq!(gershgorin(&jb, &[true, false, false]), 3.0);
        assert_eq!(gershgorin(&Sparse::new(), &[true]), 0.0);
    }

    /// `band_model_with` on `f` at `u` (its own sampling base), every row
    /// written, with the band Jacobian `jb` (its keys the pattern).
    fn check(
        f: impl Fn(&[f64]) -> Vec<f64>,
        u: &[f64],
        jb: &Sparse,
    ) -> Result<BandModelReport, String> {
        let names: Vec<String> = (0..u.len()).map(|i| format!("x[{}]", i + 1)).collect();
        let f = |v: &[f64]| Ok(f(v));
        band_model_with(&f, u, u, &names, jb, jb, &vec![true; u.len()])
    }

    /// The step `band_model` takes at a component of magnitude 1.
    fn h1() -> f64 {
        (1.0 + EPS.cbrt()) - 1.0
    }

    #[test]
    fn kink_within_one_step_is_accepted_at_a_smaller_step() {
        // An upwind-style |·| limiter whose kink lies half a step above the
        // sample: the slope is 1 below the kink, -1 above it. The central
        // difference straddles the kink; the one at a tenth of the step
        // does not.
        let kink = 1.0 + 0.5 * h1();
        let f = |v: &[f64]| vec![-(v[0] - kink).abs(), 3.0 * v[1]];
        let jb = sparse(&[((0, 0), 1.0), ((1, 1), 3.0)]);
        let r = check(f, &[1.0, 2.0], &jb).expect("a tenth of the step matches");
        assert_eq!(r.accepted_at_smaller_step, 1);
        assert!(r.worst_entry <= 1.0);
        // The same kink half a step below.
        let kink = 1.0 - 0.5 * h1();
        let f = |v: &[f64]| vec![(v[0] - kink).abs(), 3.0 * v[1]];
        let r = check(f, &[1.0, 2.0], &jb).expect("a tenth of the step matches");
        assert_eq!(r.accepted_at_smaller_step, 1);
        // A min/max limiter clamping within a fifth of a step on both sides.
        let w = 0.2 * h1();
        let f = |v: &[f64]| vec![v[0].clamp(1.0 - w, 1.0 + w), 3.0 * v[1]];
        let r = check(f, &[1.0, 2.0], &jb).expect("a smaller step matches");
        assert_eq!(r.accepted_at_smaller_step, 1);
        // A clamp within a twentieth of a step needs the step over 100.
        let w = 0.05 * h1();
        let f = |v: &[f64]| vec![v[0].clamp(1.0 - w, 1.0 + w), 3.0 * v[1]];
        let r = check(f, &[1.0, 2.0], &jb).expect("the smallest step matches");
        assert_eq!(r.accepted_at_smaller_step, 1);
        // With no kink near the sample nothing needs a smaller step.
        let f = |v: &[f64]| vec![-(v[0] - 5.0).abs(), 3.0 * v[1]];
        let r = check(f, &[1.0, 2.0], &jb).unwrap();
        assert_eq!(r.accepted_at_smaller_step, 0);
    }

    #[test]
    fn wrong_band_value_still_fails() {
        // A smooth row with a band value off by 1e-3 relative.
        let f = |v: &[f64]| vec![2.0 * v[0] * v[0], 3.0 * v[1]];
        let jb = sparse(&[((0, 0), 4.004), ((1, 1), 3.0)]);
        let e = check(f, &[1.0, 2.0], &jb).unwrap_err();
        assert!(e.starts_with("J[x[1], x[1]]: bands give"), "{e}");
        assert!(e.contains("over 100"), "{e}");
        // A smooth, curved row whose band value is the forward difference:
        // off by half the step times the curvature (about 3e-6 relative),
        // beyond τ's 1e-6. A one-sided difference would accept it; the
        // central differences at smaller steps do not.
        let f = |v: &[f64]| vec![v[0] * v[0], 3.0 * v[1]];
        let forward = ((1.0 + h1()) * (1.0 + h1()) - 1.0) / h1();
        assert!((forward - 2.0) / 2.0 > 2e-6);
        let jb = sparse(&[((0, 0), forward), ((1, 1), 3.0)]);
        let e = check(f, &[1.0, 2.0], &jb).unwrap_err();
        assert!(e.starts_with("J[x[1], x[1]]"), "{e}");
        // A kinked row whose band value is no side's slope: below the kink
        // the slope is 1, above it -1 (0.5, the central difference across
        // the kink, passes the check itself).
        let kink = 1.0 + 0.5 * h1();
        let f = |v: &[f64]| vec![-(v[0] - kink).abs(), 3.0 * v[1]];
        for wrong in [0.0, 0.9, 1.001, -1.0, 2.0] {
            let jb = sparse(&[((0, 0), wrong), ((1, 1), 3.0)]);
            let e = check(f, &[1.0, 2.0], &jb).unwrap_err();
            assert!(e.starts_with("J[x[1], x[1]]"), "{wrong}: {e}");
        }
        // Within a clamp narrower than a hundredth of a step no difference
        // sees the slope, so the check refuses: the fallback's limit.
        let w = 1e-3 * h1();
        let f = |v: &[f64]| vec![v[0].clamp(1.0 - w, 1.0 + w), 3.0 * v[1]];
        let jb = sparse(&[((0, 0), 1.0), ((1, 1), 3.0)]);
        assert!(check(f, &[1.0, 2.0], &jb).is_err());
    }

    /// The tolerance at a smaller step has its own rounding term, which
    /// grows as the step shrinks. A row near 1e8 with a kink half a step
    /// above the sample and a jump of 1e-5 at the sample (standing in for
    /// rounding error) gives, at a tenth of the step, a central difference
    /// 8.3 from the right slope: beyond the full step's τ (about 3.7), within
    /// the tenth's (about 37).
    #[test]
    fn smaller_steps_widen_the_rounding_term() {
        let kink = 1.0 + 0.5 * h1();
        let f = |v: &[f64]| {
            let jump = if v[0] > 1.0 { 1e-5 } else { 0.0 };
            vec![1e8 - 100.0 * (v[0] - kink).abs() + jump, 3.0 * v[1]]
        };
        let jb = sparse(&[((0, 0), 100.0), ((1, 1), 3.0)]);
        let r = check(f, &[1.0, 2.0], &jb).expect("the right slope passes");
        assert_eq!(r.accepted_at_smaller_step, 1);
        assert!(r.worst_entry > 0.1 && r.worst_entry <= 1.0, "{r:?}");
    }

    #[test]
    fn greedy_colouring() {
        // A tridiagonal pattern on 5 columns needs three colours.
        let mut p = Sparse::new();
        for r in 0..5usize {
            for c in r.saturating_sub(1)..=(r + 1).min(4) {
                p.insert((r, c), 0.0);
            }
        }
        assert_eq!(colour_columns(5, &p), [0, 1, 2, 0, 1]);
        // A block-diagonal pattern of two 2 × 2 blocks needs two.
        let p = sparse(&[
            ((0, 0), 0.0),
            ((0, 1), 0.0),
            ((1, 0), 0.0),
            ((1, 1), 0.0),
            ((2, 2), 0.0),
            ((2, 3), 0.0),
            ((3, 2), 0.0),
            ((3, 3), 0.0),
        ]);
        assert_eq!(colour_columns(4, &p), [0, 1, 0, 1]);
        // A column no row reads takes colour 0.
        assert_eq!(colour_columns(2, &sparse(&[((0, 0), 0.0)])), [0, 0]);
    }
}
