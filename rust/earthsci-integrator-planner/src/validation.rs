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
    /// The largest `|J^B − J^I| / τ` over the pattern entries.
    pub worst_entry: f64,
    /// The largest off-pattern difference over its floor.
    pub worst_off_pattern: f64,
    /// The number of colours (right-hand-side pairs).
    pub colours: usize,
}

impl BandModelReport {
    pub fn to_json(&self) -> Value {
        json!({
            "worst_entry_over_tolerance": self.worst_entry,
            "worst_off_pattern_over_floor": self.worst_off_pattern,
            "colours": self.colours,
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
    let n = dofs.n;
    let u = &s.u;
    // Steps: h_n = ε^(1/3) max(|u_n|, |b_n|); the step taken is the exact
    // distance (u_n + h_n) − u_n.
    let cbrt_eps = EPS.cbrt();
    let step: Vec<f64> = (0..n)
        .map(|k| {
            let h = cbrt_eps * u[k].abs().max(base[k].abs());
            (u[k] + h) - u[k]
        })
        .collect();
    let colour = colour_columns(n, pattern);
    let colours = colour.iter().copied().max().map_or(0, |m| m + 1);
    // The pattern columns of each row.
    let mut cols_of: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (r, c) in pattern.keys() {
        cols_of.entry(*r).or_default().push(*c);
    }
    // J^I and F_r per pattern entry.
    let mut ji: BTreeMap<(usize, usize), (f64, f64)> = BTreeMap::new();
    let mut report = BandModelReport {
        colours,
        ..Default::default()
    };
    let mut members_of: Vec<Vec<usize>> = vec![Vec::new(); colours];
    for (c, k) in colour.iter().enumerate() {
        members_of[*k].push(c);
    }
    for (k, members) in members_of.iter().enumerate() {
        let mut up = u.clone();
        let mut um = u.clone();
        for &c in members {
            up[c] = u[c] + step[c];
            um[c] = u[c] - step[c];
        }
        let fp = f.eval(dofs, &up, s.t, &s.forcing)?;
        let fm = f.eval(dofs, &um, s.t, &s.forcing)?;
        for r in (0..n).filter(|r| rows[*r]) {
            let big_f = fp[r].abs().max(fm[r].abs());
            let col = cols_of
                .get(&r)
                .and_then(|cs| cs.iter().find(|c| colour[**c] == k));
            match col {
                Some(&c) => {
                    ji.insert((r, c), ((fp[r] - fm[r]) / (2.0 * step[c]), big_f));
                }
                None => {
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
    // M_r: the largest magnitude in row r of either Jacobian.
    let mut m: BTreeMap<usize, f64> = BTreeMap::new();
    for ((r, _), v) in jb {
        let e = m.entry(*r).or_insert(0.0);
        *e = e.max(v.abs());
    }
    for ((r, _), (v, _)) in &ji {
        let e = m.entry(*r).or_insert(0.0);
        *e = e.max(v.abs());
    }
    for ((r, c), (vi, big_f)) in &ji {
        let vb = jb.get(&(*r, *c)).copied().unwrap_or(0.0);
        let tau = 1e-6 * m[r] + 1e3 * EPS * big_f / step[*c];
        let d = (vb - vi).abs();
        if d > tau {
            return Err(format!(
                "J[{}, {}]: bands give {vb:e}, central differences {vi:e}, beyond the tolerance {tau:e}",
                names[*r], names[*c]
            ));
        }
        if d > 0.0 {
            report.worst_entry = report.worst_entry.max(d / tau);
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
