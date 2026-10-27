//! Pass 2, locality (spec/plan.md §5.2, D66).
//!
//! The band list of a term is EarthSciASTDiff's state Jacobian
//! (`jacobian_bands`, §5.2.2), which arrives with stage 2 of M2. This module
//! holds what does not need EarthSciASTDiff: the band entry as EarthSciASTDiff
//! serializes it ([`BandEntry`]), the [`BandSource`] seam stage 2 plugs
//! `jacobian_bands` into, and the offset, reduction, class and radius rules
//! of §5.2.3–§5.2.4 as pure functions of a band list.

use crate::axes::{Axes, Role};
use crate::model::{node, ModelInfo};
use crate::terms::Term;
use earthsci_ast::Expr;
use indexmap::IndexMap;
use std::collections::{BTreeMap, BTreeSet};

/// A band's row index on one dimension (EarthSciASTDiff's `RIdx`): an index
/// name ranging over the band's row range, or a fixed integer.
#[derive(Debug, Clone, PartialEq)]
pub enum RowIdx {
    Name(String),
    Fixed(i64),
}

/// One band of a Jacobian, with the fields EarthSciASTDiff's
/// `serialize_band` writes, without `coef` (spec/plan.md §5.4).
#[derive(Debug, Clone, PartialEq)]
pub struct BandEntry {
    /// The row variable `x`.
    pub row: String,
    /// The column variable `y`.
    pub col: String,
    /// The row range per dimension of `x`, 1-based inclusive.
    pub rows: Vec<(i64, i64)>,
    /// The row index per dimension of `x`.
    pub row_idx: Vec<RowIdx>,
    /// The column index expression per dimension of `y`.
    pub col_idx: Vec<Expr>,
    /// Contracted index names with their ranges.
    pub contracted: Vec<(String, i64, i64)>,
}

/// A failure of the band source for one term (`band_model_failed`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BandFailure {
    /// The position of the failing term in the slice passed in.
    pub term: usize,
    pub message: String,
}

/// Where term bands come from (spec/plan.md §5.2.2). Stage 2 implements it
/// with EarthSciASTDiff's `jacobian_bands`; until then planning stops at
/// pass 2 unless a caller supplies band lists.
pub trait BandSource {
    /// The band list of each term, in the order given. `Ok(None)` means the
    /// source cannot compute bands yet (stage 1), which is not a refusal.
    fn term_bands(
        &self,
        info: &ModelInfo,
        terms: &[Term],
    ) -> Result<Option<Vec<Vec<BandEntry>>>, BandFailure>;
}

/// The band source of stage 1: it has no bands.
pub struct NoBands;

impl BandSource for NoBands {
    fn term_bands(
        &self,
        _info: &ModelInfo,
        _terms: &[Term],
    ) -> Result<Option<Vec<Vec<BandEntry>>>, BandFailure> {
        Ok(None)
    }
}

/// Locality classes, coarsening in declaration order (§5.2.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Class {
    Pointwise,
    Column,
    Stencil,
    Global,
}

impl Class {
    pub fn as_str(self) -> &'static str {
        match self {
            Class::Pointwise => "pointwise",
            Class::Column => "column",
            Class::Stencil => "stencil",
            Class::Global => "global",
        }
    }

    pub fn parse(s: &str) -> Option<Class> {
        Some(match s {
            "pointwise" => Class::Pointwise,
            "column" => Class::Column,
            "stencil" => Class::Stencil,
            "global" => Class::Global,
            _ => return None,
        })
    }

    /// The default operator of this class (spec/run-config.md §3.3).
    pub fn default_operator(self) -> String {
        format!("default_{}", self.as_str())
    }
}

/// What one band entry reaches (§5.2.3).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Reach {
    /// The band makes the term global (one scalar, one array).
    pub global: bool,
    /// Index sets the reach covers.
    pub covers: BTreeSet<String>,
    /// Offsets per index set: reduced on horizontal axes, raw elsewhere.
    pub offsets: BTreeMap<String, BTreeSet<i64>>,
}

/// An integer literal.
fn int_lit(e: &Expr) -> Option<i64> {
    match e {
        Expr::Integer(k) => Some(*k),
        _ => None,
    }
}

/// An integer constant (§5.2.3), recursively: an integer literal, a `-` of
/// one or two integer constants, or a `+` or `*` of one or more, evaluated
/// exactly in 64-bit signed integers. An expression whose value, or any of
/// whose sub-expressions' values, overflows is not one. A float literal is
/// never one, even when integral.
fn int_const(e: &Expr) -> Option<i64> {
    if let Some(k) = int_lit(e) {
        return Some(k);
    }
    let n = node(e)?;
    let args: Option<Vec<i64>> = n.args.iter().map(int_const).collect();
    let args = args?;
    match (n.op.as_str(), args.as_slice()) {
        ("+", [_, ..]) => args.iter().try_fold(0i64, |a, b| a.checked_add(*b)),
        ("*", [_, ..]) => args.iter().try_fold(1i64, |a, b| a.checked_mul(*b)),
        ("-", [a]) => a.checked_neg(),
        ("-", [a, b]) => a.checked_sub(*b),
        _ => None,
    }
}

/// Whether `e` references any of `names`.
fn references(e: &Expr, names: &BTreeSet<&str>) -> bool {
    let mut found = false;
    let v = serde_json::to_value(e).unwrap_or_default();
    crate::model::collect_strings(&v, &mut |s| found |= names.contains(s));
    found
}

/// The raw offsets of a column index `c` against the row index `r` on one
/// dimension whose row range is `range`; `None` when the reach covers the
/// index set ("anything else", §5.2.3).
fn raw_offsets(c: &Expr, r: &RowIdx, range: (i64, i64)) -> Option<BTreeSet<i64>> {
    // c equal to r.
    match (c, r) {
        (Expr::Variable(v), RowIdx::Name(n)) if v == n => return Some([0].into()),
        _ => {}
    }
    // c of the form r + δ, r − δ or δ + r, δ an integer literal.
    if let (Some(n), RowIdx::Name(rn)) = (node(c), r) {
        let is_r = |e: &Expr| matches!(e, Expr::Variable(v) if v == rn);
        if let [a, b] = n.args.as_slice() {
            match n.op.as_str() {
                "+" if is_r(a) => {
                    if let Some(d) = int_lit(b) {
                        return Some([d].into());
                    }
                }
                "+" if is_r(b) => {
                    if let Some(d) = int_lit(a) {
                        return Some([d].into());
                    }
                }
                "-" if is_r(a) => {
                    if let Some(d) = int_lit(b) {
                        return Some([-d].into());
                    }
                }
                _ => {}
            }
        }
    }
    // c an integer constant k.
    if let Some(k) = int_const(c) {
        return Some(match r {
            RowIdx::Fixed(rho) => [k.checked_sub(*rho)?].into(),
            RowIdx::Name(_) => (k.checked_sub(range.1)?..=k.checked_sub(range.0)?).collect(),
        });
    }
    None
}

/// `o` reduced on a horizontal axis of size `n` (§5.2.3, D66):
/// `((o + ⌊n/2⌋) mod n) − ⌊n/2⌋`, the modulus in `[0, n)`.
pub fn reduce(o: i64, n: i64) -> i64 {
    let h = n.div_euclid(2);
    (o + h).rem_euclid(n) - h
}

/// The reach of one band entry (§5.2.3). `shapes` gives each state's index
/// sets (empty for a scalar).
pub fn band_reach(
    b: &BandEntry,
    shapes: &BTreeMap<String, Vec<String>>,
    sizes: &IndexMap<String, i64>,
    axes: &Axes,
) -> Reach {
    let mut reach = Reach::default();
    let empty = Vec::new();
    let xs = shapes.get(&b.row).unwrap_or(&empty);
    let ys = shapes.get(&b.col).unwrap_or(&empty);
    match (xs.is_empty(), ys.is_empty()) {
        (true, true) => return reach,
        (true, false) | (false, true) => {
            reach.global = true;
            return reach;
        }
        _ => {}
    }
    let contracted: BTreeSet<&str> = b.contracted.iter().map(|(n, _, _)| n.as_str()).collect();
    for (d, set) in ys.iter().enumerate() {
        // The j-th dimension of y on `set` (counting y's dimensions on `set`
        // from the left) pairs with the j-th dimension of x on `set`.
        let j = ys[..d].iter().filter(|s| *s == set).count();
        let Some(xd) = xs
            .iter()
            .enumerate()
            .filter(|(_, s)| *s == set)
            .nth(j)
            .map(|(i, _)| i)
        else {
            reach.covers.insert(set.clone());
            continue;
        };
        let (Some(c), Some(r)) = (b.col_idx.get(d), b.row_idx.get(xd)) else {
            reach.covers.insert(set.clone());
            continue;
        };
        if references(c, &contracted) {
            reach.covers.insert(set.clone());
            continue;
        }
        let range = b.rows.get(xd).copied().unwrap_or((1, sizes[set]));
        match raw_offsets(c, r, range) {
            None => {
                reach.covers.insert(set.clone());
            }
            Some(raw) => {
                let horizontal = axes.role(set) == Role::Horizontal;
                let n = sizes[set];
                let entry = reach.offsets.entry(set.clone()).or_default();
                for o in raw {
                    entry.insert(if horizontal { reduce(o, n) } else { o });
                }
            }
        }
    }
    reach
}

/// A term's locality class and, for a stencil term, its radius on every
/// horizontal axis (§5.2.4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Locality {
    pub class: Class,
    /// `[lower, upper]` per horizontal axis (stencil terms only; empty
    /// otherwise).
    pub radius: BTreeMap<String, (i64, i64)>,
}

/// The class and radius of a term from its band list (§5.2.4).
///
/// Offsets on an other axis do not affect the class; a reach that covers one
/// makes the term global (§5.2.4, D82). No size check on the radius is
/// needed: reduction keeps `lower + upper + 1` within the axis size.
pub fn classify(
    bands: &[BandEntry],
    shapes: &BTreeMap<String, Vec<String>>,
    sizes: &IndexMap<String, i64>,
    axes: &Axes,
) -> Locality {
    let mut global = false;
    let mut covers: BTreeSet<String> = BTreeSet::new();
    let mut offsets: BTreeMap<String, BTreeSet<i64>> = BTreeMap::new();
    for b in bands {
        let r = band_reach(b, shapes, sizes, axes);
        global |= r.global;
        covers.extend(r.covers);
        for (set, os) in r.offsets {
            offsets.entry(set).or_default().extend(os);
        }
    }
    let role = |s: &str| axes.role(s);
    if covers.iter().any(|s| role(s) != Role::Vertical) {
        global = true;
    }
    let mut radius = BTreeMap::new();
    let mut stencil = false;
    for h in &axes.horizontal {
        let os = offsets.get(h).cloned().unwrap_or_default();
        let lower = os.iter().filter(|o| **o < 0).map(|o| -o).max().unwrap_or(0);
        let upper = os.iter().filter(|o| **o > 0).copied().max().unwrap_or(0);
        stencil |= lower > 0 || upper > 0;
        radius.insert(h.clone(), (lower, upper));
    }
    if global {
        return Locality {
            class: Class::Global,
            radius: BTreeMap::new(),
        };
    }
    if stencil {
        return Locality {
            class: Class::Stencil,
            radius,
        };
    }
    let column = axes.vertical.as_ref().is_some_and(|v| {
        covers.contains(v) || offsets.get(v).is_some_and(|os| os.iter().any(|o| *o != 0))
    });
    Locality {
        class: if column {
            Class::Column
        } else {
            Class::Pointwise
        },
        radius: BTreeMap::new(),
    }
}

/// An operator's class (the coarsest of its terms') and, for a stencil
/// operator, its radius on every horizontal axis, each direction the
/// maximum over its stencil terms (§5.2.4).
pub fn operator_locality<'a>(
    terms: impl IntoIterator<Item = &'a Locality>,
    axes: &Axes,
) -> Locality {
    let terms: Vec<&Locality> = terms.into_iter().collect();
    let class = terms
        .iter()
        .map(|l| l.class)
        .max()
        .unwrap_or(Class::Pointwise);
    let mut radius = BTreeMap::new();
    if class == Class::Stencil {
        for h in &axes.horizontal {
            let mut r = (0, 0);
            for l in terms.iter().filter(|l| l.class == Class::Stencil) {
                if let Some(&(lo, up)) = l.radius.get(h) {
                    r = (r.0.max(lo), r.1.max(up));
                }
            }
            radius.insert(h.clone(), r);
        }
    }
    Locality { class, radius }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(s: &str) -> Expr {
        serde_json::from_str(s).unwrap()
    }

    fn v(s: &str) -> Expr {
        Expr::Variable(s.into())
    }

    fn name(s: &str) -> RowIdx {
        RowIdx::Name(s.into())
    }

    struct Grid {
        shapes: BTreeMap<String, Vec<String>>,
        sizes: IndexMap<String, i64>,
        axes: Axes,
    }

    /// `q` over lon × lat × lev (8 × 3 × 4), `s` scalar, `f` over lon × lat
    /// (2-D), `m` over lon × spc (an other axis), with lon, lat horizontal
    /// and lev vertical.
    fn grid() -> Grid {
        let set = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let mut shapes = BTreeMap::new();
        shapes.insert("q".into(), set(&["lon", "lat", "lev"]));
        shapes.insert("p".into(), set(&["lon", "lat", "lev"]));
        shapes.insert("f".into(), set(&["lon", "lat"]));
        shapes.insert("m".into(), set(&["lon", "spc"]));
        shapes.insert("s".into(), vec![]);
        shapes.insert("s2".into(), vec![]);
        let sizes: IndexMap<String, i64> = [("lon", 8), ("lat", 3), ("lev", 4), ("spc", 5)]
            .into_iter()
            .map(|(k, n)| (k.to_string(), n))
            .collect();
        Grid {
            shapes,
            sizes,
            axes: Axes {
                horizontal: vec!["lon".into(), "lat".into()],
                vertical: Some("lev".into()),
            },
        }
    }

    fn band(
        row: &str,
        col: &str,
        rows: &[(i64, i64)],
        ri: Vec<RowIdx>,
        ci: Vec<Expr>,
    ) -> BandEntry {
        BandEntry {
            row: row.into(),
            col: col.into(),
            rows: rows.to_vec(),
            row_idx: ri,
            col_idx: ci,
            contracted: vec![],
        }
    }

    fn q_band(ci: [Expr; 3]) -> BandEntry {
        band(
            "q",
            "q",
            &[(1, 8), (1, 3), (1, 4)],
            vec![name("i"), name("j"), name("k")],
            ci.to_vec(),
        )
    }

    fn reach(b: &BandEntry) -> Reach {
        let g = grid();
        band_reach(b, &g.shapes, &g.sizes, &g.axes)
    }

    fn class(bands: &[BandEntry]) -> Locality {
        let g = grid();
        classify(bands, &g.shapes, &g.sizes, &g.axes)
    }

    fn offs(r: &Reach, set: &str) -> Vec<i64> {
        r.offsets
            .get(set)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default()
    }

    #[test]
    fn scalar_and_mixed_bands() {
        // Scalar row and scalar column: no reach.
        let r = reach(&band("s", "s2", &[], vec![], vec![]));
        assert_eq!(r, Reach::default());
        assert_eq!(
            class(&[band("s", "s2", &[], vec![], vec![])]).class,
            Class::Pointwise
        );
        // One scalar, one array: global, either way round.
        let r = reach(&band("s", "q", &[], vec![], vec![v("i"), v("j"), v("k")]));
        assert!(r.global);
        let r = reach(&band(
            "q",
            "s",
            &[(1, 8), (1, 3), (1, 4)],
            vec![name("i"), name("j"), name("k")],
            vec![],
        ));
        assert!(r.global);
        assert_eq!(
            class(&[band(
                "q",
                "s",
                &[(1, 8), (1, 3), (1, 4)],
                vec![name("i"), name("j"), name("k")],
                vec![]
            )])
            .class,
            Class::Global
        );
        // No bands: pointwise.
        assert_eq!(class(&[]).class, Class::Pointwise);
    }

    #[test]
    fn equal_and_affine_indices() {
        // c equal to r: {0}.
        let r = reach(&q_band([v("i"), v("j"), v("k")]));
        assert_eq!(offs(&r, "lon"), vec![0]);
        assert_eq!(offs(&r, "lev"), vec![0]);
        assert!(r.covers.is_empty());
        // r + δ, δ + r, r − δ.
        let r = reach(&q_band([
            e(r#"{"op":"+","args":["i",2]}"#),
            e(r#"{"op":"+","args":[1,"j"]}"#),
            e(r#"{"op":"-","args":["k",1]}"#),
        ]));
        assert_eq!(offs(&r, "lon"), vec![2]);
        assert_eq!(offs(&r, "lat"), vec![1]);
        assert_eq!(offs(&r, "lev"), vec![-1]);
        // A negative literal after `+` and a δ of 0 are still literals.
        let r = reach(&q_band([
            e(r#"{"op":"+","args":["i",-3]}"#),
            v("j"),
            v("k"),
        ]));
        assert_eq!(offs(&r, "lon"), vec![-3]);
    }

    #[test]
    fn constant_indices() {
        // c an integer constant k and r a fixed integer ρ: {k − ρ}.
        let b = band(
            "q",
            "q",
            &[(1, 1), (1, 3), (1, 4)],
            vec![RowIdx::Fixed(1), name("j"), name("k")],
            vec![Expr::Integer(8), v("j"), v("k")],
        );
        // Raw 7 reduced modulo 8 into [−4, 3]: −1 (the periodic seam, D66).
        assert_eq!(offs(&reach(&b), "lon"), vec![-1]);
        // +, − and * of integer literals are constants.
        let b = band(
            "q",
            "q",
            &[(2, 2), (1, 3), (1, 4)],
            vec![RowIdx::Fixed(2), name("j"), name("k")],
            vec![e(r#"{"op":"*","args":[2,3]}"#), v("j"), v("k")],
        );
        assert_eq!(offs(&reach(&b), "lon"), vec![4 - 8]);
        let b = band(
            "q",
            "q",
            &[(5, 5), (1, 3), (1, 4)],
            vec![RowIdx::Fixed(5), name("j"), name("k")],
            vec![e(r#"{"op":"-","args":[9,2]}"#), v("j"), v("k")],
        );
        assert_eq!(offs(&reach(&b), "lon"), vec![2]);
        // c an integer constant k and r a name over [lo, hi]: {k − hi … k − lo}.
        // Vertical, so not reduced: lev rows 2–4 reading level 1.
        let b = band(
            "q",
            "q",
            &[(1, 8), (1, 3), (2, 4)],
            vec![name("i"), name("j"), name("k")],
            vec![v("i"), v("j"), Expr::Integer(1)],
        );
        assert_eq!(offs(&reach(&b), "lev"), vec![-3, -2, -1]);
        assert_eq!(class(&[b]).class, Class::Column);
    }

    #[test]
    fn covering_reaches() {
        // x has no dimension on the column's index set: covers it.
        // Row f (lon, lat) reading q (lon, lat, lev): covers lev → column.
        let b = band(
            "f",
            "q",
            &[(1, 8), (1, 3)],
            vec![name("i"), name("j")],
            vec![v("i"), v("j"), v("kk")],
        );
        let r = reach(&b);
        assert!(r.covers.contains("lev"));
        assert_eq!(class(&[b]).class, Class::Column);
        // A dimension of x that y lacks is not paired: q reading f is pointwise.
        let b = band(
            "q",
            "f",
            &[(1, 8), (1, 3), (1, 4)],
            vec![name("i"), name("j"), name("k")],
            vec![v("i"), v("j")],
        );
        assert_eq!(class(&[b]).class, Class::Pointwise);
        // A contracted name in a column index covers the index set: column on
        // the vertical axis, global on a horizontal one.
        let mut b = q_band([v("i"), v("j"), v("kk")]);
        b.contracted = vec![("kk".into(), 1, 4)];
        assert!(reach(&b).covers.contains("lev"));
        assert_eq!(class(std::slice::from_ref(&b)).class, Class::Column);
        let mut b = q_band([v("ii"), v("j"), v("k")]);
        b.contracted = vec![("ii".into(), 1, 8)];
        assert_eq!(class(&[b]).class, Class::Global);
        // Anything else covers: another index name, a gather, a nonlinear index.
        for c in [
            v("j"),
            e(r#"{"op":"index","args":["perm","i"]}"#),
            e(r#"{"op":"*","args":[2,"i"]}"#),
            e(r#"{"op":"+","args":["i","j"]}"#),
            e(r#"{"op":"+","args":["i",1.0]}"#),
        ] {
            let b = q_band([c, v("j"), v("k")]);
            assert!(reach(&b).covers.contains("lon"));
            assert_eq!(class(&[b]).class, Class::Global);
        }
        let b = q_band([v("i"), v("j"), e(r#"{"op":"index","args":["lvl","k"]}"#)]);
        assert_eq!(class(&[b]).class, Class::Column);
        // A covered other axis is global.
        let b = band(
            "m",
            "m",
            &[(1, 8), (1, 5)],
            vec![name("i"), name("s")],
            vec![v("i"), v("t")],
        );
        assert_eq!(class(&[b]).class, Class::Global);
    }

    #[test]
    fn integer_constants_are_recursive() {
        // Nested, n-ary + and *, unary and binary -.
        assert_eq!(
            int_const(&e(r#"{"op":"*","args":[2,{"op":"+","args":[3,1]}]}"#)),
            Some(8)
        );
        assert_eq!(int_const(&e(r#"{"op":"+","args":[1,2,3]}"#)), Some(6));
        assert_eq!(int_const(&e(r#"{"op":"+","args":[5]}"#)), Some(5));
        assert_eq!(int_const(&e(r#"{"op":"*","args":[7]}"#)), Some(7));
        assert_eq!(
            int_const(&e(
                r#"{"op":"-","args":[{"op":"-","args":[4]},{"op":"*","args":[2,2,2]}]}"#
            )),
            Some(-12)
        );
        // Not constants: a float literal, a name, three-argument -, other ops.
        assert_eq!(int_const(&e(r#"{"op":"+","args":[1,1.0]}"#)), None);
        assert_eq!(int_const(&e("2.0")), None);
        assert_eq!(int_const(&e(r#"{"op":"+","args":[1,"i"]}"#)), None);
        assert_eq!(int_const(&e(r#"{"op":"-","args":[1,2,3]}"#)), None);
        assert_eq!(int_const(&e(r#"{"op":"/","args":[4,2]}"#)), None);
        // 64-bit overflow anywhere is not a constant, even if the total fits.
        let big = i64::MAX;
        assert_eq!(
            int_const(&e(&format!(r#"{{"op":"+","args":[{big},1]}}"#))),
            None
        );
        assert_eq!(
            int_const(&e(&format!(
                r#"{{"op":"-","args":[{{"op":"+","args":[{big},1]}},1]}}"#
            ))),
            None
        );
        assert_eq!(
            int_const(&e(&format!(r#"{{"op":"-","args":[{big},1]}}"#))),
            Some(big - 1)
        );
        // A nested constant column index against a fixed row: {k − ρ}.
        let b = band(
            "q",
            "q",
            &[(1, 1), (1, 3), (1, 4)],
            vec![RowIdx::Fixed(1), name("j"), name("k")],
            vec![
                e(r#"{"op":"*","args":[2,{"op":"+","args":[1,1]}]}"#),
                v("j"),
                v("k"),
            ],
        );
        assert_eq!(offs(&reach(&b), "lon"), vec![3]);
        // An overflowing index covers the axis.
        let b = band(
            "q",
            "q",
            &[(1, 1), (1, 3), (1, 4)],
            vec![RowIdx::Fixed(1), name("j"), name("k")],
            vec![
                e(&format!(r#"{{"op":"*","args":[{big},2]}}"#)),
                v("j"),
                v("k"),
            ],
        );
        assert!(reach(&b).covers.contains("lon"));
    }

    /// An index set twice in one shape: the j-th dimension of the column on
    /// it pairs with the j-th of the row on it.
    #[test]
    fn repeated_index_set_pairs_in_order() {
        let mut g = grid();
        g.shapes
            .insert("mm".into(), vec!["lon".into(), "lon".into()]);
        let b = band(
            "mm",
            "mm",
            &[(1, 8), (1, 8)],
            vec![name("a"), name("b")],
            vec![v("a"), e(r#"{"op":"+","args":["b",1]}"#)],
        );
        let r = band_reach(&b, &g.shapes, &g.sizes, &g.axes);
        // The first lon pairs with the first (offset 0), the second with the
        // second (offset +1); pairing both with the first would give "b"
        // against "a", a cover.
        assert!(r.covers.is_empty());
        assert_eq!(offs(&r, "lon"), vec![0, 1]);
        // A column with two lon dimensions read from a row with one: the
        // second has no partner and covers lon.
        let b = band(
            "q",
            "mm",
            &[(1, 8), (1, 3), (1, 4)],
            vec![name("i"), name("j"), name("k")],
            vec![v("i"), v("i")],
        );
        let r = band_reach(&b, &g.shapes, &g.sizes, &g.axes);
        assert!(r.covers.contains("lon"));
    }

    #[test]
    fn reduction() {
        // Into [−⌊N/2⌋, ⌈N/2⌉ − 1].
        assert_eq!(reduce(7, 8), -1);
        assert_eq!(reduce(-7, 8), 1);
        assert_eq!(reduce(4, 8), -4);
        assert_eq!(reduce(-4, 8), -4);
        assert_eq!(reduce(3, 8), 3);
        assert_eq!(reduce(63, 64), -1);
        assert_eq!(reduce(2, 5), 2);
        assert_eq!(reduce(3, 5), -2);
        assert_eq!(reduce(0, 1), 0);
        // Vertical offsets are not reduced.
        let b = band(
            "q",
            "q",
            &[(1, 8), (1, 3), (1, 1)],
            vec![name("i"), name("j"), RowIdx::Fixed(1)],
            vec![v("i"), v("j"), Expr::Integer(4)],
        );
        assert_eq!(offs(&reach(&b), "lev"), vec![3]);
    }

    /// The periodic stencil of the `periodic_stencil` fixture: interior rows
    /// read i−3 … i+2, the seam rows literal wrapped indices.
    #[test]
    fn periodic_stencil_radius() {
        let mut bands = Vec::new();
        for d in [-3i64, -2, -1, 0, 1, 2] {
            let c = match d {
                0 => v("i"),
                d if d < 0 => e(&format!(r#"{{"op":"-","args":["i",{}]}}"#, -d)),
                d => e(&format!(r#"{{"op":"+","args":["i",{d}]}}"#)),
            };
            bands.push(band(
                "q",
                "q",
                &[(4, 6), (1, 3), (1, 4)],
                vec![name("i"), name("j"), name("k")],
                vec![c, v("j"), v("k")],
            ));
        }
        for (r, cols) in [
            (1, [6, 7, 8, 1, 2, 3]),
            (2, [7, 8, 1, 2, 3, 4]),
            (3, [8, 1, 2, 3, 4, 5]),
            (7, [4, 5, 6, 7, 8, 1]),
            (8, [5, 6, 7, 8, 1, 2]),
        ] {
            for c in cols {
                bands.push(band(
                    "q",
                    "q",
                    &[(r, r), (1, 3), (1, 4)],
                    vec![RowIdx::Fixed(r), name("j"), name("k")],
                    vec![Expr::Integer(c), v("j"), v("k")],
                ));
            }
        }
        let l = class(&bands);
        assert_eq!(l.class, Class::Stencil);
        assert_eq!(l.radius["lon"], (3, 2));
        assert_eq!(l.radius["lat"], (0, 0));
    }

    #[test]
    fn class_rules() {
        // Stencil on lat, one cell toward lower index.
        let lat = q_band([v("i"), e(r#"{"op":"-","args":["j",1]}"#), v("k")]);
        let l = class(std::slice::from_ref(&lat));
        assert_eq!(l.class, Class::Stencil);
        assert_eq!(l.radius["lat"], (1, 0));
        assert_eq!(l.radius["lon"], (0, 0));
        // Lower + upper + 1 above the axis size is global: lat has 3 cells.
        let wide = q_band([v("i"), e(r#"{"op":"+","args":["j",1]}"#), v("k")]);
        let l = class(&[lat.clone(), wide.clone()]);
        assert_eq!(l.class, Class::Stencil);
        assert_eq!(l.radius["lat"], (1, 1));
        let wider = q_band([v("i"), e(r#"{"op":"-","args":["j",3]}"#), v("k")]);
        // −3 on an axis of 3 reduces to 0: pointwise reach.
        assert_eq!(class(&[wider]).class, Class::Pointwise);
        let lon = |d: i64| {
            q_band([
                if d < 0 {
                    e(&format!(r#"{{"op":"-","args":["i",{}]}}"#, -d))
                } else {
                    e(&format!(r#"{{"op":"+","args":["i",{d}]}}"#))
                },
                v("j"),
                v("k"),
            ])
        };
        // lon has 8 cells, reduced into [−4, 3]: −4..3 is 4 + 3 + 1 = 8, stencil.
        let l = class(&[lon(-4), lon(3)]);
        assert_eq!(l.class, Class::Stencil);
        assert_eq!(l.radius["lon"], (4, 3));
        // Stencil beats column: a vertical offset with a horizontal one.
        let mixed = q_band([
            e(r#"{"op":"+","args":["i",1]}"#),
            v("j"),
            e(r#"{"op":"+","args":["k",1]}"#),
        ]);
        assert_eq!(class(&[mixed]).class, Class::Stencil);
        // A vertical offset alone is column.
        let up = q_band([v("i"), v("j"), e(r#"{"op":"+","args":["k",1]}"#)]);
        assert_eq!(class(&[up]).class, Class::Column);
        // Global beats everything.
        let g = band("s", "q", &[], vec![], vec![v("i"), v("j"), v("k")]);
        assert_eq!(class(&[lat, g]).class, Class::Global);
        // An offset on an other axis alone: pointwise (literal reading).
        let other = band(
            "m",
            "m",
            &[(1, 8), (1, 5)],
            vec![name("i"), name("s")],
            vec![v("i"), e(r#"{"op":"+","args":["s",1]}"#)],
        );
        assert_eq!(class(&[other]).class, Class::Pointwise);
    }

    #[test]
    fn operator_class_and_radius() {
        let g = grid();
        let st = |lon: (i64, i64), lat: (i64, i64)| Locality {
            class: Class::Stencil,
            radius: [("lon".to_string(), lon), ("lat".to_string(), lat)].into(),
        };
        let pw = Locality {
            class: Class::Pointwise,
            radius: BTreeMap::new(),
        };
        let col = Locality {
            class: Class::Column,
            radius: BTreeMap::new(),
        };
        let l = operator_locality([&st((3, 2), (0, 0)), &st((0, 0), (1, 0)), &pw], &g.axes);
        assert_eq!(l.class, Class::Stencil);
        assert_eq!(l.radius["lon"], (3, 2));
        assert_eq!(l.radius["lat"], (1, 0));
        let l = operator_locality([&pw, &col], &g.axes);
        assert_eq!(l.class, Class::Column);
        assert!(l.radius.is_empty());
    }
}
