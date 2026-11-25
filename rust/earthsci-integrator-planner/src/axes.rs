//! Axes (spec/plan.md §5.2.1, spec/run-config.md §6.1, D67).

use crate::error::{codes, refuse, Result};
use crate::model::ModelInfo;
use crate::runconfig::RunConfig;
use serde_json::{json, Value};
use std::collections::BTreeMap;

/// The role of an index set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Horizontal,
    Vertical,
    /// Used by a state but neither horizontal nor vertical.
    Other,
}

/// The resolved axes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Axes {
    /// One or two horizontal axes (none for a model without index sets), in
    /// the order `partition.ranks` and `tiles` divide them.
    pub horizontal: Vec<String>,
    pub vertical: Option<String>,
    /// The declared face sets (`axes.faces`, D105): each face index set
    /// mapped to the cell axis it staggers off, cell `i` owning face `i`,
    /// its low face.
    pub faces: BTreeMap<String, String>,
}

impl Axes {
    pub fn role(&self, set: &str) -> Role {
        if self.vertical.as_deref() == Some(set) {
            Role::Vertical
        } else if self.horizontal.iter().any(|h| h == set) {
            Role::Horizontal
        } else {
            Role::Other
        }
    }

    /// Index set `set` as §5.2.3 pairs it: a declared face set is its cell
    /// axis, any other set itself (D105).
    pub fn cell_axis<'a>(&'a self, set: &'a str) -> &'a str {
        self.faces.get(set).map(String::as_str).unwrap_or(set)
    }
}

/// Resolves `axes`: given entries must exist and not repeat across roles;
/// by default there is no vertical axis and every index set that is a
/// dimension of some state's shape, other than the vertical one, is
/// horizontal, in the flattened model's `index_sets` order. More than two
/// horizontal axes is `axes_unresolved`. Lifted states' shapes must be
/// resolved first (§6.1.3).
pub fn resolve(rc: &RunConfig, info: &ModelInfo) -> Result<Axes> {
    let given = rc.at("/axes");
    let unresolved = |m: String| refuse(codes::AXES_UNRESOLVED, "/axes", m);
    let vertical = given
        .and_then(|a| a.get("vertical"))
        .and_then(Value::as_str)
        .map(str::to_string);
    if let Some(v) = &vertical {
        if !info.index_sets.contains_key(v) {
            return Err(unresolved(format!(
                "vertical axis {v} is not an index set of the model"
            )));
        }
    }
    let horizontal: Vec<String> = match given.and_then(|a| a.get("horizontal")) {
        Some(Value::Array(list)) => {
            let names: Vec<String> = list
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect();
            for h in &names {
                if !info.index_sets.contains_key(h) {
                    return Err(unresolved(format!(
                        "horizontal axis {h} is not an index set of the model"
                    )));
                }
                if vertical.as_deref() == Some(h.as_str()) {
                    return Err(unresolved(format!(
                        "{h} is named both horizontal and vertical"
                    )));
                }
            }
            names
        }
        _ => info
            .index_sets
            .keys()
            .filter(|set| vertical.as_deref() != Some(set.as_str()))
            .filter(|set| info.shapes.values().any(|s| s.contains(set)))
            .cloned()
            .collect(),
    };
    if horizontal.len() > 2 {
        return Err(unresolved(format!(
            "{} horizontal axes ({}); at most two are allowed",
            horizontal.len(),
            horizontal.join(", ")
        )));
    }
    // `axes.faces` (D105), in code-point order of the face set.
    let mut faces = BTreeMap::new();
    if let Some(Value::Object(given)) = given.and_then(|a| a.get("faces")) {
        let mut entries: Vec<(&String, &Value)> = given.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (f, c) in entries {
            let c = c.as_str().unwrap_or_default();
            let bad = |m: String| refuse(codes::AXES_UNRESOLVED, format!("/axes/faces/{f}"), m);
            let Some(&nf) = info.index_sets.get(f) else {
                return Err(bad(format!(
                    "the face set {f} is not an index set of the model"
                )));
            };
            if horizontal.contains(f) || vertical.as_deref() == Some(f.as_str()) {
                return Err(bad(format!(
                    "the face set {f} is itself a horizontal or vertical axis"
                )));
            }
            if !horizontal.iter().any(|h| h == c) && vertical.as_deref() != Some(c) {
                return Err(bad(format!(
                    "the face set {f} staggers off {c}, which is neither a horizontal nor the vertical axis"
                )));
            }
            let nc = info.index_sets[c];
            if nf != nc + 1 {
                return Err(bad(format!(
                    "the face set {f} has {nf} elements; the faces of {c}, of {nc} cells, are {}",
                    nc + 1
                )));
            }
            faces.insert(f.clone(), c.to_string());
        }
    }
    Ok(Axes {
        horizontal,
        vertical,
        faces,
    })
}

/// The plan report's `likely_face_set` warnings (spec/plan.md §5.2.1,
/// D105): one per index set of the model that is neither an axis nor a
/// declared face set and has exactly one more element than a horizontal or
/// the vertical axis, index sets in code-point order, then axes in
/// `horizontal` order, the vertical last. The same list as Julia's.
pub fn face_set_warnings(info: &ModelInfo, axes: &Axes) -> Vec<Value> {
    let mut cells: Vec<&String> = axes.horizontal.iter().collect();
    if let Some(v) = &axes.vertical {
        cells.push(v);
    }
    let mut sets: Vec<&String> = info.index_sets.keys().collect();
    sets.sort();
    let mut out = Vec::new();
    for s in sets {
        if cells.contains(&s) || axes.faces.contains_key(s) {
            continue;
        }
        for c in &cells {
            if info.index_sets[s] == info.index_sets[*c] + 1 {
                out.push(json!({
                    "code": "likely_face_set",
                    "index_set": s,
                    "axis": c,
                    "message": format!(
                        "index set {s} has one element more than axis {c}: if it holds {c}'s faces, declare it in axes.faces"
                    ),
                }));
            }
        }
    }
    out
}
