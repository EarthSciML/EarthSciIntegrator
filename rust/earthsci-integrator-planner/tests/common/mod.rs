//! Shared test helpers: the conformance directory, a hand-written band
//! source, and the conformance harness's band-value comparison (spec/plan.md
//! §8.2).
//!
//! The hand-written band source ([`HandBands`]) was stage 1's stand-in for
//! EarthSciASTDiff, written by hand from each fixture's `expected.md`: for
//! every state a term references, one same-cell band (row and column indices
//! equal, the band EarthSciASTDiff gives a pointwise read), plus, for the
//! terms `expected.md` describes as stencil or column terms, bands with the
//! offsets it lists. Stage 2 keeps it to check that EarthSciASTDiff's band
//! lists give every term the class `expected.md` gives it.
#![allow(dead_code)]

use earthsci_ast::Expr;
use earthsci_integrator_planner::locality::{BandEntry, BandFailure, BandSource, RowIdx};
use earthsci_integrator_planner::model::ModelInfo;
use earthsci_integrator_planner::terms::Term;
use earthsci_integrator_planner::Planned;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;

pub fn conformance() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/conformance")
}

const IDX: [&str; 4] = ["i", "j", "k", "l"];

fn v(s: &str) -> Expr {
    Expr::Variable(s.into())
}

fn e(s: &str) -> Expr {
    serde_json::from_str(s).unwrap()
}

/// A band of row `x` reading `y`, same cell, with the column index on
/// dimension `d` replaced by `col(d)` when given.
fn band(info: &ModelInfo, x: &str, y: &str, col: &dyn Fn(usize) -> Option<Expr>) -> BandEntry {
    let xs = &info.shapes[x];
    let ys = &info.shapes[y];
    BandEntry {
        row: x.into(),
        col: y.into(),
        rows: xs.iter().map(|s| (1, info.index_sets[s])).collect(),
        row_idx: (0..xs.len()).map(|d| RowIdx::Name(IDX[d].into())).collect(),
        col_idx: ys
            .iter()
            .enumerate()
            .map(|(d, s)| {
                col(d).unwrap_or_else(|| {
                    let p = xs.iter().position(|t| t == s).unwrap_or(d);
                    v(IDX[p])
                })
            })
            .collect(),
        contracted: vec![],
    }
}

pub struct HandBands;

impl BandSource for HandBands {
    fn term_bands(
        &self,
        info: &ModelInfo,
        terms: &[Term],
    ) -> Result<Option<Vec<Vec<BandEntry>>>, BandFailure> {
        let model = info.file.metadata.name.clone().unwrap_or_default();
        Ok(Some(
            terms
                .iter()
                .map(|t| {
                    let x = t.variable.as_str();
                    let reads: Vec<&String> = t.names.iter().filter(|n| info.is_state(n)).collect();
                    let same = |col: &dyn Fn(usize) -> Option<Expr>| -> Vec<BandEntry> {
                        reads.iter().map(|y| band(info, x, y, col)).collect()
                    };
                    match (model.as_str(), t.contribution, t.summand) {
                        // PPM-like advection along lon: offsets −3 … 2.
                        ("periodic_stencil", _, _) | ("reseact_shaped", 2, 0) => (-3i64..=2)
                            .flat_map(|o| {
                                let c = match o {
                                    0 => v("i"),
                                    o if o < 0 => {
                                        e(&format!(r#"{{"op":"-","args":["i",{}]}}"#, -o))
                                    }
                                    o => e(&format!(r#"{{"op":"+","args":["i",{o}]}}"#)),
                                };
                                same(&move |d| (d == 0).then(|| c.clone()))
                            })
                            .collect(),
                        // First-order upwind along cell, seam row 1 reading cell 64.
                        ("advection_reaction", 1, 0) => {
                            let mut b = same(&|_| None);
                            b.extend(same(&|_| Some(e(r#"{"op":"-","args":["i",1]}"#))));
                            b
                        }
                        // Upwind along lat.
                        ("reseact_shaped", 2, 1) => {
                            let mut b = same(&|_| None);
                            b.extend(same(&|d| {
                                (d == 1).then(|| e(r#"{"op":"-","args":["j",1]}"#))
                            }));
                            b
                        }
                        // The column mean: a contraction over lev.
                        ("reseact_shaped", 3, 0) | ("column_contraction", _, _) => {
                            let mut b = same(&|_| None);
                            let mut c = same(&|d| (d == 2).then(|| v("kk")));
                            for x in &mut c {
                                x.contracted = vec![("kk".into(), 1, info.index_sets["lev"])];
                            }
                            b.extend(c);
                            b
                        }
                        _ => same(&|_| None),
                    }
                })
                .collect(),
        ))
    }
}

/// The conformance harness's comparison of the Rust planner's band values
/// with a golden's `validation/<op>.bandvalues.json` (spec/plan.md §8.2): at
/// each sample's `u` and `t`, operator `op`'s band Jacobian must have the
/// file's structural key set, and every value must agree within
/// `1e-10 · M_r`, `M_r` the largest magnitude in row `r` of either.
pub fn compare_band_values(p: &Planned, op: &str, file: &Value) -> Result<(), String> {
    let names = p.dofs().element_names();
    let samples = file["samples"]
        .as_array()
        .ok_or("the band-value file has no samples")?;
    for (k, s) in samples.iter().enumerate() {
        let t = s["t"].as_f64().ok_or("a sample has no t")?;
        let u: Vec<f64> = names
            .iter()
            .map(|n| {
                s["u"][n]
                    .as_f64()
                    .ok_or_else(|| format!("sample {k} has no u[{n}]"))
            })
            .collect::<Result<_, _>>()?;
        let got = p.band_jacobian_at(op, &u, t).map_err(|e| e.to_string())?;
        let mut want: BTreeMap<(String, String), f64> = BTreeMap::new();
        for e in s["entries"].as_array().ok_or("a sample has no entries")? {
            let r = e[0].as_str().ok_or("an entry has no row")?;
            let c = e[1].as_str().ok_or("an entry has no column")?;
            let v = e[2].as_f64().ok_or("an entry has no value")?;
            want.insert((r.to_string(), c.to_string()), v);
        }
        if let Some(key) = got.keys().find(|key| !want.contains_key(*key)) {
            return Err(format!(
                "sample {k}: {key:?} is in the Rust bands, not in the file"
            ));
        }
        if let Some(key) = want.keys().find(|key| !got.contains_key(*key)) {
            return Err(format!(
                "sample {k}: {key:?} is in the file, not in the Rust bands"
            ));
        }
        let mut m: BTreeMap<&str, f64> = BTreeMap::new();
        for ((r, _), v) in got.iter().chain(&want) {
            let e = m.entry(r.as_str()).or_insert(0.0);
            *e = e.max(v.abs());
        }
        for (key, g) in &got {
            let w = want[key];
            if (g - w).abs() > 1e-10 * m[key.0.as_str()] {
                return Err(format!(
                    "sample {k}: {key:?} is {g:e} in Rust, {w:e} in the file"
                ));
            }
        }
    }
    Ok(())
}

/// A band-value file in the golden's form, written from the Rust planner's
/// own band Jacobian at `states` (`(t, u)` pairs): for testing the
/// comparison itself.
pub fn band_values_file(p: &Planned, op: &str, states: &[(f64, Vec<f64>)]) -> Value {
    let names = p.dofs().element_names();
    let samples: Vec<Value> = states
        .iter()
        .map(|(t, u)| {
            let jb = p.band_jacobian_at(op, u, *t).unwrap();
            let um: Map<String, Value> = names
                .iter()
                .zip(u)
                .map(|(n, x)| (n.clone(), json!(x)))
                .collect();
            let entries: Vec<Value> = jb.iter().map(|((r, c), v)| json!([r, c, v])).collect();
            json!({"t": t, "u": um, "entries": entries})
        })
        .collect();
    json!({ "samples": samples })
}
