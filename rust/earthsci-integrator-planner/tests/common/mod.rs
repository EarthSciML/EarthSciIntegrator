//! A hand-written band source for the conformance fixtures.
//!
//! Stage 1 has no EarthSciASTDiff, so these band lists are written by hand
//! from each fixture's `expected.md`: for every state a term references,
//! one same-cell band (row and column indices equal, the band EarthSciASTDiff
//! gives a pointwise read), plus, for the terms `expected.md` describes as
//! stencil or column terms, bands with the offsets it lists. They exercise
//! the real offset and class rules of spec/plan.md §5.2.3–§5.2.4; the bands
//! themselves are not EarthSciASTDiff's and are not golden material.
#![allow(dead_code)]

use earthsci_ast::Expr;
use earthsci_integrator_planner::locality::{BandEntry, BandFailure, BandSource, RowIdx};
use earthsci_integrator_planner::model::ModelInfo;
use earthsci_integrator_planner::terms::Term;
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
                                    o if o < 0 => e(&format!(r#"{{"op":"-","args":["i",{}]}}"#, -o)),
                                    o => e(&format!(r#"{{"op":"+","args":["i",{o}]}}"#)),
                                };
                                same(&move |d| (d == 0).then(|| c.clone()))
                            })
                            .collect(),
                        // Upwind along lat.
                        ("reseact_shaped", 2, 1) => {
                            let mut b = same(&|_| None);
                            b.extend(same(&|d| (d == 1).then(|| e(r#"{"op":"-","args":["j",1]}"#))));
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
