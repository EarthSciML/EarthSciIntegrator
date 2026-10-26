//! Pass 1 selection and operators (spec/plan.md §5.1.3–§5.1.4) and pass 3,
//! the DOF sets (§5.3).

use crate::error::{codes, ptr, refuse, Result};
use crate::locality::{Class, Locality};
use crate::model::ModelInfo;
use crate::runconfig::RunConfig;
use crate::terms::Term;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// An operator: a set of terms in term order, with its `bind_dt`.
#[derive(Debug, Clone, PartialEq)]
pub struct Operator {
    pub name: String,
    /// Positions of its terms in the planner's term list.
    pub terms: Vec<usize>,
    /// The run config's `bind_dt` for it, if any.
    pub bind_dt: Option<Value>,
    /// Whether it is a default operator (spec/run-config.md §3.3).
    pub is_default: bool,
    /// Pass 3: the dependent variables of its terms, sorted.
    pub writes: Vec<String>,
    /// Pass 3: the names its terms read, sorted.
    pub reads: Vec<String>,
    /// Pass 2: its class and radius.
    pub locality: Locality,
}

/// A term operator's selector.
struct Selector<'a> {
    name: &'a str,
    components: Option<Vec<&'a str>>,
    variables: Option<Vec<&'a str>>,
    locality: Option<Vec<Class>>,
    terms: Option<Vec<&'a str>>,
}

fn strings(v: Option<&Value>) -> Option<Vec<&str>> {
    match v? {
        Value::String(s) => Some(vec![s.as_str()]),
        Value::Array(a) => Some(a.iter().filter_map(Value::as_str).collect()),
        _ => None,
    }
}

/// A component `c` in a selector covers a component `p` (§5.1.2).
fn covers(c: &str, p: &str) -> bool {
    p == c || (p.starts_with(c) && p[c.len()..].starts_with('.'))
}

impl Selector<'_> {
    fn matches(&self, t: &Term, loc: Class) -> bool {
        if let Some(cs) = &self.components {
            let Some(owner) = &t.owner else {
                return false;
            };
            if !cs.iter().any(|c| covers(c, owner)) {
                return false;
            }
        }
        if let Some(vs) = &self.variables {
            if !vs.contains(&t.variable.as_str()) {
                return false;
            }
        }
        if let Some(ls) = &self.locality {
            if !ls.contains(&loc) {
                return false;
            }
        }
        if let Some(ts) = &self.terms {
            if !ts.contains(&t.provenance.as_str()) {
                return false;
            }
        }
        true
    }
}

/// Selection (§5.1.3): every term goes to the one term operator whose
/// selector matches it, or to the default operator of its locality.
/// `localities` are the terms' classes (pass 2), in term order.
pub fn select(
    rc: &RunConfig,
    info: &ModelInfo,
    terms: &[Term],
    localities: &[Locality],
) -> Result<BTreeMap<String, Operator>> {
    // Operators in code-point order of their names.
    let declared: BTreeMap<&str, &Value> = rc
        .at("/operators")
        .and_then(Value::as_object)
        .map(|m| m.iter().map(|(k, v)| (k.as_str(), v)).collect())
        .unwrap_or_default();
    for (name, op) in &declared {
        if op.get("map").is_some() {
            return Err(refuse(
                codes::UNSUPPORTED_MAP_OPERATOR,
                ptr("/operators", name),
                format!("{name} is a map operator; map operators need U6 (M4b)"),
            ));
        }
    }
    let mut selectors = Vec::new();
    for (name, op) in &declared {
        let sel = &op["select"];
        let base = ptr(&ptr("/operators", name), "select");
        let variables = strings(sel.get("variables"));
        if let Some(vs) = &variables {
            let list = sel["variables"].is_array();
            for (i, v) in vs.iter().enumerate() {
                let at = if list {
                    ptr(&ptr(&base, "variables"), i)
                } else {
                    ptr(&base, "variables")
                };
                if !info.is_name(v) {
                    return Err(refuse(
                        codes::NAME_UNRESOLVED,
                        at,
                        format!("{v} is not a name of the flattened model"),
                    ));
                }
                if !info.is_state(v) {
                    return Err(refuse(
                        codes::NAME_WRONG_KIND,
                        at,
                        format!("{v} is not a state variable"),
                    ));
                }
            }
        }
        selectors.push(Selector {
            name,
            components: strings(sel.get("components")),
            variables,
            locality: strings(sel.get("locality"))
                .map(|ls| ls.into_iter().filter_map(Class::parse).collect()),
            terms: strings(sel.get("terms")),
        });
    }
    let any_components = selectors.iter().any(|s| s.components.is_some());
    let mut assigned: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut defaults = BTreeSet::new();
    for (i, t) in terms.iter().enumerate() {
        let loc = localities[i].class;
        let matched: Vec<&str> = selectors
            .iter()
            .filter(|s| s.matches(t, loc))
            .map(|s| s.name)
            .collect();
        match matched.as_slice() {
            [one] => assigned.entry(one.to_string()).or_default().push(i),
            [first, second, ..] => {
                return Err(refuse(
                    codes::TERM_SELECTED_TWICE,
                    ptr("/operators", first),
                    format!(
                        "term ({}, {}, {}) is selected by both {first} and {second}",
                        t.variable, t.contribution, t.summand
                    ),
                ))
            }
            [] if t.owner.is_none() && any_components => {
                let reads: BTreeSet<&str> = t
                    .names
                    .iter()
                    .map(|n| crate::terms::component_of(n))
                    .collect();
                return Err(refuse(
                    codes::AMBIGUOUS_TERM_OWNER,
                    "/operators",
                    format!(
                        "term ({}, {}, {}) reads components {} and has no owner; no selector takes it while a selector uses components (select it by terms or variables)",
                        t.variable,
                        t.contribution,
                        t.summand,
                        reads.into_iter().collect::<Vec<_>>().join(", ")
                    ),
                ));
            }
            [] => {
                let name = loc.default_operator();
                defaults.insert(name.clone());
                assigned.entry(name).or_default().push(i);
            }
        }
    }
    for s in &selectors {
        if !assigned.contains_key(s.name) {
            return Err(refuse(
                codes::SELECTOR_MATCHES_NOTHING,
                ptr(&ptr("/operators", s.name), "select"),
                format!("the selector of {} matches no term", s.name),
            ));
        }
    }
    let mut out = BTreeMap::new();
    for (name, idx) in assigned {
        let bind_dt = declared.get(name.as_str()).and_then(|op| op.get("bind_dt")).cloned();
        let (writes, reads) = dof_sets(info, idx.iter().map(|&i| &terms[i]));
        out.insert(
            name.clone(),
            Operator {
                is_default: defaults.contains(&name),
                name,
                terms: idx,
                bind_dt,
                writes,
                reads,
                locality: Locality {
                    class: Class::Pointwise,
                    radius: BTreeMap::new(),
                },
            },
        );
    }
    Ok(out)
}

/// Pass 3 (§5.3): `writes`, the dependent variables of the terms, and
/// `reads`, every name the terms reference with, transitively, the names
/// the definitions of the observeds among them reference. Both sorted by
/// code point and duplicate-free.
pub fn dof_sets<'a>(
    info: &ModelInfo,
    terms: impl IntoIterator<Item = &'a Term>,
) -> (Vec<String>, Vec<String>) {
    let mut writes = BTreeSet::new();
    let mut names = BTreeSet::new();
    for t in terms {
        writes.insert(t.variable.clone());
        names.extend(t.names.iter().cloned());
    }
    let reads = info.close_over_observeds(&names);
    (writes.into_iter().collect(), reads.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::covers;

    #[test]
    fn component_cover() {
        assert!(covers("Chem", "Chem"));
        assert!(covers("Chem", "Chem.Gas"));
        assert!(!covers("Chem", "Chemistry"));
        assert!(!covers("Chem.Gas", "Chem"));
    }
}
