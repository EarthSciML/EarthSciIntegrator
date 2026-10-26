//! Axes (spec/plan.md §5.2.1, spec/run-config.md §6.1, D67).

use crate::error::{codes, refuse, Result};
use crate::model::ModelInfo;
use crate::runconfig::RunConfig;
use serde_json::Value;

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
            return Err(unresolved(format!("vertical axis {v} is not an index set of the model")));
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
                    return Err(unresolved(format!("{h} is named both horizontal and vertical")));
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
    Ok(Axes {
        horizontal,
        vertical,
    })
}
