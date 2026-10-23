//! The run configuration (spec/run-config.md), the parts M1 runs.
//!
//! Unknown keys are errors at every level (§1). Keys the schema allows but
//! this implementation does not run yet parse, and are refused with a
//! message when the schedule is built.

use crate::Error;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

/// A run configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    pub esm_integrator: String,
    #[serde(default)]
    pub description: Option<String>,
    pub model: String,
    #[serde(default)]
    pub operators: Option<Value>,
    pub schedule: Value,
    #[serde(default)]
    pub step: Option<Step>,
    #[serde(default)]
    pub partition: Option<Value>,
    #[serde(default)]
    pub precision: Option<Value>,
    #[serde(default)]
    pub adjoint: Option<Adjoint>,
    #[serde(default)]
    pub output: Option<Value>,
    #[serde(default)]
    pub restart: Option<Value>,
}

impl RunConfig {
    /// Parses a run config and checks its spec version (spec/README.md).
    pub fn from_value(v: &Value) -> Result<RunConfig, Error> {
        let rc: RunConfig =
            serde_json::from_value(v.clone()).map_err(|e| Error::Config(e.to_string()))?;
        if rc.esm_integrator != crate::SPEC_VERSION {
            return Err(Error::Config(format!(
                "esm_integrator {} is not the version this implementation implements ({})",
                rc.esm_integrator,
                crate::SPEC_VERSION
            )));
        }
        Ok(rc)
    }

    /// The schedule as a node tree.
    pub fn schedule_node(&self) -> Result<Node, Error> {
        if self.schedule.as_str() == Some("auto") {
            return Err(Error::Unimplemented(
                "schedule \"auto\" needs the planner (M2)".into(),
            ));
        }
        Node::from_value(&self.schedule)
    }
}

/// Macro-step control (§5).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Step {
    #[serde(default)]
    pub dt: Option<Value>,
    #[serde(default)]
    pub limits: Option<Value>,
    #[serde(default)]
    pub hit: Option<Vec<String>>,
    #[serde(default)]
    pub hit_multiples: Option<Value>,
    #[serde(default)]
    pub resolution: Option<Value>,
}

/// The `adjoint` key (§7.2).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Adjoint {
    pub functional: Functional,
    pub wrt: Vec<String>,
    #[serde(default)]
    pub checkpoint: Option<Value>,
}

/// The adjoint functional: one reduction of one quantity.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Functional {
    #[serde(default)]
    pub mean: Option<String>,
    #[serde(default)]
    pub sum: Option<String>,
    #[serde(default, rename = "where")]
    pub where_: Option<BTreeMap<String, Value>>,
    pub over: String,
    #[serde(default)]
    pub integrate_in: Option<String>,
}

/// `leaf` and `coupled` options (§4.2).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LeafNode {
    #[serde(default)]
    pub leaf: Option<OperatorRefs>,
    #[serde(default)]
    pub coupled: Option<OperatorRefs>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub alg: Option<String>,
    #[serde(default)]
    pub theta: Option<f64>,
    #[serde(default)]
    pub implicit_solve: Option<String>,
    #[serde(default)]
    pub dt: Option<Value>,
    #[serde(default)]
    pub control: Option<Control>,
    #[serde(default)]
    pub jacobian: Option<String>,
    #[serde(default)]
    pub linsolve: Option<Value>,
    #[serde(default)]
    pub imex: Option<Value>,
    #[serde(default)]
    pub domain: Option<Domain>,
    #[serde(default)]
    pub frozen: Option<Value>,
    #[serde(default)]
    pub partition: Option<Value>,
    #[serde(default)]
    pub bind_dt: Option<BTreeMap<String, String>>,
}

/// One operator name or a list.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum OperatorRefs {
    One(String),
    Many(Vec<String>),
}

impl OperatorRefs {
    pub fn names(&self) -> Vec<String> {
        match self {
            OperatorRefs::One(s) => vec![s.clone()],
            OperatorRefs::Many(v) => v.clone(),
        }
    }
}

/// `control` (§4.2).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Control {
    #[serde(default)]
    pub reltol: Option<f64>,
    #[serde(default)]
    pub abstol: Option<Value>,
    #[serde(default)]
    pub dt0: Option<Value>,
    #[serde(default)]
    pub dtmin: Option<Value>,
    #[serde(default)]
    pub dtmax: Option<Value>,
    #[serde(default)]
    pub controller: Option<String>,
    #[serde(default)]
    pub dt_scope: Option<String>,
    #[serde(default)]
    pub maxiters: Option<u64>,
}

/// `domain` (§4.2, spec/methods.md §11).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Domain {
    #[serde(default)]
    pub nonnegative: Option<Value>,
    #[serde(default)]
    pub on_violation: Option<String>,
    #[serde(default)]
    pub check_stages: Option<bool>,
}

/// A schedule node (§4).
#[derive(Debug, Clone)]
pub enum Node {
    Leaf(Box<LeafNode>),
    Sequence {
        id: Option<String>,
        children: Vec<Node>,
    },
    Strang {
        id: Option<String>,
        children: Vec<Node>,
    },
    Subcycle {
        id: Option<String>,
        child: Box<Node>,
        n: Option<usize>,
        limits: Option<Value>,
    },
    Map {
        id: Option<String>,
        op: String,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SeqNode {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    sequence: Option<Vec<Value>>,
    #[serde(default)]
    strang: Option<Vec<Value>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SubNode {
    #[serde(default)]
    id: Option<String>,
    subcycle: Value,
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    limits: Option<Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MapNode {
    #[serde(default)]
    id: Option<String>,
    map: String,
}

impl Node {
    /// Parses a node; exactly one combinator key identifies its kind.
    pub fn from_value(v: &Value) -> Result<Node, Error> {
        let obj = v
            .as_object()
            .ok_or_else(|| Error::Config(format!("schedule node is not an object: {v}")))?;
        let kinds = [
            "leaf",
            "coupled",
            "sequence",
            "strang",
            "subcycle",
            "every",
            "split_explicit",
            "map",
            "solve",
        ];
        let present: Vec<&str> = kinds
            .iter()
            .copied()
            .filter(|k| obj.contains_key(*k))
            .collect();
        if present.len() != 1 {
            return Err(Error::Config(format!(
                "a schedule node needs exactly one combinator key, found {present:?}"
            )));
        }
        let de = |e: serde_json::Error| Error::Config(format!("{}: {e}", present[0]));
        match present[0] {
            "leaf" => {
                let l: LeafNode = serde_json::from_value(v.clone()).map_err(de)?;
                Ok(Node::Leaf(Box::new(l)))
            }
            "coupled" => Err(Error::Unimplemented(
                "coupled nodes (run-config §4.3) are not implemented at M1".into(),
            )),
            "sequence" | "strang" => {
                let s: SeqNode = serde_json::from_value(v.clone()).map_err(de)?;
                let is_seq = s.sequence.is_some();
                let kids = s.sequence.or(s.strang).unwrap_or_default();
                let children = kids
                    .iter()
                    .map(Node::from_value)
                    .collect::<Result<Vec<_>, _>>()?;
                if is_seq {
                    Ok(Node::Sequence { id: s.id, children })
                } else {
                    if children.len() < 2 {
                        return Err(Error::Config("strang needs at least two children".into()));
                    }
                    Ok(Node::Strang { id: s.id, children })
                }
            }
            "subcycle" => {
                let s: SubNode = serde_json::from_value(v.clone()).map_err(de)?;
                if s.n.is_some() == s.limits.is_some() {
                    return Err(Error::Config(
                        "subcycle needs exactly one of n and limits".into(),
                    ));
                }
                Ok(Node::Subcycle {
                    id: s.id,
                    child: Box::new(Node::from_value(&s.subcycle)?),
                    n: s.n,
                    limits: s.limits,
                })
            }
            "map" => {
                let m: MapNode = serde_json::from_value(v.clone()).map_err(de)?;
                Ok(Node::Map {
                    id: m.id,
                    op: m.map,
                })
            }
            other => Err(Error::Unimplemented(format!(
                "schedule node {other:?} is not implemented at M1"
            ))),
        }
    }
}

/// Parses a duration (§2.2): positive seconds, or `"<number> <unit>"`.
/// Parameter names are refused (the toy problems have none in seconds).
pub fn parse_duration(v: &Value) -> Result<f64, Error> {
    let bad = || Error::Config(format!("invalid duration {v}"));
    let x = if let Some(x) = v.as_f64() {
        x
    } else if let Some(s) = v.as_str() {
        let s = s.trim();
        let split = s
            .find(|c: char| c.is_ascii_alphabetic() && c != 'e' && c != 'E')
            .ok_or_else(bad)?;
        let (num, unit) = s.split_at(split);
        let factor = match unit {
            "s" => 1.0,
            "min" => 60.0,
            "h" => 3600.0,
            "day" => 86400.0,
            _ => {
                return Err(Error::Config(format!(
                    "duration {s:?}: parameter-valued durations are not supported for toy problems"
                )))
            }
        };
        num.trim().parse::<f64>().map_err(|_| bad())? * factor
    } else {
        return Err(bad());
    };
    if x > 0.0 && x.is_finite() {
        Ok(x)
    } else {
        Err(bad())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn durations() {
        assert_eq!(parse_duration(&json!(2.5)).unwrap(), 2.5);
        assert_eq!(parse_duration(&json!("2 h")).unwrap(), 7200.0);
        assert_eq!(parse_duration(&json!("1.5min")).unwrap(), 90.0);
        assert_eq!(parse_duration(&json!("1e2 s")).unwrap(), 100.0);
        assert!(parse_duration(&json!("0 s")).is_err());
        assert!(parse_duration(&json!(-1)).is_err());
    }

    #[test]
    fn unknown_keys_are_refused() {
        let v = json!({"leaf": "decay", "alg": "Tsit5", "bogus": 1});
        assert!(Node::from_value(&v).is_err());
        let v = json!({"sequence": [{"leaf": "a", "alg": "Heun", "dt": 1}, {"map": "m"}]});
        assert!(matches!(
            Node::from_value(&v).unwrap(),
            Node::Sequence { .. }
        ));
        let v = json!({"leaf": "a", "map": "m"});
        assert!(Node::from_value(&v).is_err());
    }
}
