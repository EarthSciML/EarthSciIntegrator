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
    /// Pass 3: the data-fed parameters among `reads` (§5.3, D87), sorted.
    pub forcing: Vec<String>,
    /// Pass 2: its class and radius.
    pub locality: Locality,
    /// An algebraic operator's residuals (§5.9.1), in flattened equation
    /// order; empty for a term operator.
    pub residuals: Vec<crate::algebraic::Residual>,
    /// Once its solving node is paired (§5.9.5), each residual's unknown.
    pub pairing: Vec<String>,
}

impl Operator {
    /// Whether it is an algebraic operator (§5.9.1).
    pub fn is_algebraic(&self) -> bool {
        !self.residuals.is_empty()
    }
}

/// A term operator's selector.
struct Selector<'a> {
    name: &'a str,
    components: Option<Vec<&'a str>>,
    variables: Option<Vec<&'a str>>,
    locality: Option<Vec<Class>>,
    terms: Option<Vec<&'a str>>,
}

/// A string or an array of strings, as a list.
pub(crate) fn strings(v: Option<&Value>) -> Option<Vec<&str>> {
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

    /// spec/run-config.md §3.1 for an algebraic equation: `components`
    /// covers its owner, `variables` lists a name it reads, `terms` lists
    /// its id, and a `locality` key never matches.
    fn matches_residual(&self, r: &crate::algebraic::Residual) -> bool {
        if self.locality.is_some() {
            return false;
        }
        if let Some(cs) = &self.components {
            let Some(owner) = &r.owner else {
                return false;
            };
            if !cs.iter().any(|c| covers(c, owner)) {
                return false;
            }
        }
        if let Some(vs) = &self.variables {
            if !vs.iter().any(|v| r.reads.contains(*v)) {
                return false;
            }
        }
        if let Some(ts) = &self.terms {
            if !ts.contains(&r.id.as_str()) {
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
                        format!("{v} is not a state variable or solved unknown"),
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
    // §5.9.1: algebraic equations; unmatched ones go to `default_algebraic`.
    let mut rassigned: BTreeMap<String, Vec<crate::algebraic::Residual>> = BTreeMap::new();
    for r in &info.residuals {
        let matched: Vec<&str> = selectors
            .iter()
            .filter(|s| s.matches_residual(r))
            .map(|s| s.name)
            .collect();
        if let [first, second, ..] = matched.as_slice() {
            return Err(refuse(
                codes::TERM_SELECTED_TWICE,
                ptr("/operators", first),
                format!("algebraic equation {} is selected by both {first} and {second}", r.id),
            ));
        }
        let name = matched.first().map_or("default_algebraic", |n| *n).to_string();
        rassigned.entry(name).or_default().push(r.clone());
    }
    for s in &selectors {
        if !assigned.contains_key(s.name) && !rassigned.contains_key(s.name) {
            return Err(refuse(
                codes::SELECTOR_MATCHES_NOTHING,
                ptr(&ptr("/operators", s.name), "select"),
                format!("the selector of {} matches no term", s.name),
            ));
        }
    }
    for s in &selectors {
        if assigned.contains_key(s.name) && rassigned.contains_key(s.name) {
            return Err(refuse(
                codes::OPERATOR_MIXES_KINDS,
                ptr("/operators", s.name),
                format!("the selector of {} matches both right-hand-side terms and algebraic equations (spec/plan.md §5.9.1)", s.name),
            ));
        }
    }
    let mut out = BTreeMap::new();
    for (name, rs) in rassigned {
        let bind_dt = declared
            .get(name.as_str())
            .and_then(|op| op.get("bind_dt"))
            .cloned();
        out.insert(
            name.clone(),
            Operator {
                is_default: name == "default_algebraic",
                name,
                terms: vec![],
                bind_dt,
                writes: vec![],
                reads: vec![],
                forcing: vec![],
                locality: Locality {
                    class: Class::Pointwise,
                    radius: BTreeMap::new(),
                },
                residuals: rs,
                pairing: vec![],
            },
        );
    }
    for (name, idx) in assigned {
        let bind_dt = declared
            .get(name.as_str())
            .and_then(|op| op.get("bind_dt"))
            .cloned();
        let (writes, reads) = dof_sets(info, idx.iter().map(|&i| &terms[i]));
        let forcing = info.forcing_among(&reads);
        out.insert(
            name.clone(),
            Operator {
                is_default: defaults.contains(&name),
                name,
                terms: idx,
                bind_dt,
                writes,
                reads,
                forcing,
                locality: Locality {
                    class: Class::Pointwise,
                    radius: BTreeMap::new(),
                },
                residuals: vec![],
                pairing: vec![],
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
