//! Pass 4, structure (spec/plan.md §5.4): an operator's band list from
//! EarthSciASTDiff and what the plan records about it.

use crate::astdiff::{band_entry, FlatView};
use crate::axes::Axes;
use crate::error::{PlanError, Result};
use crate::locality::{band_reach, Class};
use crate::model::ModelInfo;
use crate::operators::Operator;
use crate::terms::Term;
use earthsci_astdiff::JacEntry;
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;

/// `structure.block` (§5.4): EarthSciASTDiff's factorization plan
/// (`plan_factorization`), as `serialize_plan` writes it without its `type`
/// key and with `block_vars` written as `vars`.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    /// The block variables, in Markowitz order (`block_vars`).
    pub vars: Vec<String>,
    /// The block as the plan writes it: `vars`, `ordering`, `pattern` and
    /// `lu_pattern` (1-based pairs, sorted).
    pub json: Value,
}

/// `structure.block` from a `serialize_plan` object: `type` dropped,
/// `block_vars` renamed `vars`. `None` when it is not a block LU.
fn block_of(mut plan: Value) -> Option<Block> {
    let m = plan.as_object_mut()?;
    if m.remove("type")? != "block_diagonal_lu" {
        return None;
    }
    let vars = m.remove("block_vars")?;
    let names = vars
        .as_array()?
        .iter()
        .map(|v| v.as_str().map(str::to_string))
        .collect::<Option<Vec<_>>>()?;
    m.insert("vars".into(), vars);
    Some(Block {
        vars: names,
        json: plan,
    })
}

/// An operator's `structure` (§5.4).
#[derive(Debug, Clone)]
pub struct Structure {
    /// EarthSciASTDiff's `detect_structure`.
    pub class: String,
    pub linear_in_u: bool,
    pub depends_on_t: bool,
    /// The band list, in EarthSciASTDiff's order; band `k` (1-based) is the
    /// field `J_k` of the Jacobian document.
    pub bands: Vec<JacEntry>,
    /// Pointwise operators whose class is `block_diagonal`.
    pub block: Option<Block>,
    /// Column operators: the vertical axis, `kl` and `ku`.
    pub column: Option<(String, i64, i64)>,
    /// Stencil operators: the footprint, sorted.
    pub footprint: Option<Vec<Vec<i64>>>,
}

/// One band as `serialize_band` writes it, without `coef` (§5.4).
pub fn band_json(en: &JacEntry) -> Value {
    let mut b = earthsci_astdiff::serialize_band(en);
    if let Some(m) = b.as_object_mut() {
        m.remove("coef");
    }
    b
}

/// Whether any of `names` is `t`.
fn reads_t(names: impl IntoIterator<Item = String>) -> bool {
    names.into_iter().any(|n| n == "t")
}

/// Pass 4 for one operator: `view` is the flattened view of its
/// right-hand-side document and `bands` its band list.
pub fn structure(
    info: &ModelInfo,
    axes: &Axes,
    op: &Operator,
    terms: &[Term],
    view: &FlatView,
    bands: Vec<JacEntry>,
) -> Result<Structure> {
    let ctx = view.ctx().map_err(|e| PlanError::Internal(e.0))?;
    let class = earthsci_astdiff::detect_structure(&bands, &ctx).to_string();

    // linear_in_u: no coefficient reads a state, or an observed whose
    // definition reads one, transitively.
    let state_dependent: BTreeSet<String> = info
        .states
        .iter()
        .cloned()
        .chain(info.observed_defs.keys().filter_map(|o| {
            let reads = info.close_over_observeds(&[o.clone()].into());
            reads.iter().any(|n| info.is_state(n)).then(|| o.clone())
        }))
        .collect();
    let linear_in_u = bands.iter().all(|en| {
        earthsci_ast::free_variables(&en.band.coef)
            .iter()
            .all(|n| !state_dependent.contains(n))
    });

    // depends_on_t: `t` in a state equation of the document (the operator's
    // terms) or in the definition of an observed it keeps.
    let mut names = BTreeSet::new();
    let mut in_terms = false;
    for &i in &op.terms {
        names.extend(terms[i].names.iter().cloned());
        in_terms |= reads_t(earthsci_ast::free_variables(&terms[i].expr));
    }
    let live = crate::derived::live_observeds(info, &names);
    let depends_on_t = in_terms
        || live.iter().any(|o| {
            info.observed_defs
                .get(o)
                .is_some_and(|d| reads_t(earthsci_ast::free_variables(d)))
        });

    let block = if op.locality.class == Class::Pointwise && class == "block_diagonal" {
        let plan =
            earthsci_astdiff::plan_factorization(&bands, &view.sysview(), &ctx, Some(&class))
                .map_err(|e| PlanError::Internal(format!("plan_factorization: {}", e.0)))?
                .ok_or_else(|| {
                    PlanError::Internal("plan_factorization gave no block_diagonal plan".into())
                })?;
        let plan = earthsci_astdiff::serialize_plan(&plan);
        Some(block_of(plan.clone()).ok_or_else(|| {
            PlanError::Internal(format!("serialize_plan gave no block LU: {plan}"))
        })?)
    } else {
        None
    };

    let reaches: Vec<_> = bands
        .iter()
        .map(|en| band_reach(&band_entry(en), &info.shapes, &info.index_sets, axes))
        .collect();

    let column = match (&op.locality.class, &axes.vertical) {
        (Class::Column, Some(v)) => {
            let nv = info.index_sets[v];
            let (mut kl, mut ku) = (0i64, 0i64);
            for r in &reaches {
                if r.covers.contains(v) {
                    kl = kl.max(nv - 1);
                    ku = ku.max(nv - 1);
                }
                for o in r.offsets.get(v).into_iter().flatten() {
                    kl = kl.max(-o);
                    ku = ku.max(*o);
                }
            }
            Some((v.clone(), kl, ku))
        }
        _ => None,
    };

    let footprint = (op.locality.class == Class::Stencil).then(|| {
        let mut out: BTreeSet<Vec<i64>> = BTreeSet::new();
        for r in &reaches {
            // One integer per horizontal axis; a band with several offsets on
            // an axis contributes every combination.
            let mut vecs: Vec<Vec<i64>> = vec![vec![]];
            for h in &axes.horizontal {
                let os: Vec<i64> = match r.offsets.get(h) {
                    Some(s) if !s.is_empty() => s.iter().copied().collect(),
                    _ => vec![0],
                };
                vecs = vecs
                    .into_iter()
                    .flat_map(|v| {
                        os.iter().map(move |o| {
                            let mut v = v.clone();
                            v.push(*o);
                            v
                        })
                    })
                    .collect();
            }
            out.extend(vecs);
        }
        out.into_iter().collect()
    });

    Ok(Structure {
        class,
        linear_in_u,
        depends_on_t,
        bands,
        block,
        column,
        footprint,
    })
}

impl Structure {
    /// `structure` as the plan writes it (§5.4).
    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("class".into(), json!(self.class));
        out.insert("linear_in_u".into(), json!(self.linear_in_u));
        out.insert("depends_on_t".into(), json!(self.depends_on_t));
        out.insert(
            "bands".into(),
            Value::Array(self.bands.iter().map(band_json).collect()),
        );
        if let Some(b) = &self.block {
            out.insert("block".into(), b.json.clone());
        }
        if let Some((axis, kl, ku)) = &self.column {
            out.insert("column".into(), json!({"axis": axis, "kl": kl, "ku": ku}));
        }
        if let Some(f) = &self.footprint {
            out.insert("stencil".into(), json!({"footprint": f}));
        }
        Value::Object(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_from_serialize_plan() {
        let plan = json!({"type": "block_diagonal_lu", "block_vars": ["M.a", "M.b"],
                          "ordering": [2, 1], "pattern": [[1, 1], [2, 2]],
                          "lu_pattern": [[1, 1], [2, 2]]});
        let b = block_of(plan).unwrap();
        assert_eq!(b.vars, ["M.a", "M.b"]);
        assert_eq!(
            b.json,
            json!({"vars": ["M.a", "M.b"], "ordering": [2, 1], "pattern": [[1, 1], [2, 2]],
                   "lu_pattern": [[1, 1], [2, 2]]})
        );
        assert_eq!(block_of(json!({"type": "column_block"})), None);
    }

    #[test]
    fn json_form() {
        let s = Structure {
            class: "banded".into(),
            linear_in_u: true,
            depends_on_t: false,
            bands: vec![],
            block: None,
            column: Some(("lev".into(), 1, 2)),
            footprint: Some(vec![vec![-1, 0], vec![0, 0]]),
        };
        assert_eq!(
            s.to_json(),
            json!({"class": "banded", "linear_in_u": true, "depends_on_t": false, "bands": [],
                   "column": {"axis": "lev", "kl": 1, "ku": 2},
                   "stencil": {"footprint": [[-1, 0], [0, 0]]}})
        );
    }
}
