//! Pass 8's `rhs_sum` (spec/plan.md §8.1, D91): the operators'
//! right-hand-side documents, loaded and flattened from their bytes, sum to
//! the flattened model, compared structurally. Nothing is built or
//! evaluated.

use crate::derived;
use crate::model::{differentiated, node, ModelInfo};
use crate::terms::{additive_terms, is_literal_zero};
use earthsci_ast::{Equation, Expr, FlattenedSystem, ModelVariable};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// An expression's canonical form (§8.1): EarthSciAST's serialization as
/// `stable_json` writes it (§2.2).
pub fn canonical(e: &Expr) -> String {
    earthsci_astdiff::stable_json(&serde_json::to_value(e).unwrap_or(Value::Null))
}

/// A JSON value with every number a binary64 value.
fn float_form(v: Value) -> Value {
    match v {
        Value::Number(n) => n.as_f64().map_or(Value::Null, |x| serde_json::json!(x)),
        Value::Array(a) => Value::Array(a.into_iter().map(float_form).collect()),
        Value::Object(o) => Value::Object(o.into_iter().map(|(k, v)| (k, float_form(v))).collect()),
        v => v,
    }
}

/// A declared `default` as §8.1 compares it: its `stable_json` text with
/// every number read as binary64, or `None`.
fn default_form(v: &ModelVariable) -> Option<String> {
    v.default.as_ref().map(|d| {
        earthsci_astdiff::stable_json(&float_form(serde_json::to_value(d).unwrap_or(Value::Null)))
    })
}

/// A declared shape: its index-set names, empty for none.
fn shape_form(v: &ModelVariable) -> Vec<String> {
    v.shape.clone().unwrap_or_default()
}

/// §8.1's summands of a state equation's right-hand side `rhs`, canonical:
/// none when `rhs` is the state's zero right-hand side `zero` (§6.1.2,
/// canonical); else the summands of §5.1.1 (`additive_terms`, literal
/// zeros dropped) of `rhs`, or of the body of its `faq` when `lifted`.
/// `None` when `lifted` and `rhs` is no `faq`.
fn summands(rhs: &Expr, zero: &str, lifted: bool) -> Option<Vec<String>> {
    if canonical(rhs) == zero {
        return Some(Vec::new());
    }
    let body = if lifted {
        node(rhs)
            .filter(|n| n.op == "faq")
            .and_then(|n| n.expr.as_deref())?
    } else {
        rhs
    };
    Some(
        additive_terms(body)
            .iter()
            .filter(|t| !is_literal_zero(t))
            .map(canonical)
            .collect(),
    )
}

/// The `faq` node's `output_idx`, or `None` when `e` is no `faq`.
fn faq_output_idx(e: &Expr) -> Option<Vec<String>> {
    node(e)
        .filter(|n| n.op == "faq")
        .and_then(|n| n.output_idx.clone())
}

/// The variable an equation's left-hand side defines, at any spelling: a
/// bare name, `index(y, …)` or `faq{index(y, …)}`. A derived document
/// declares the observeds it keeps as unknowns (§6.1.1), so the model says
/// which names are observeds.
fn defined_name(eq: &Equation) -> Option<&str> {
    let mut l = &eq.lhs;
    if let Some(n) = node(l).filter(|n| n.op == "faq") {
        l = n.expr.as_deref()?;
    }
    match l {
        Expr::Variable(x) => Some(x),
        _ => {
            let n = node(l).filter(|n| n.op == "index")?;
            match n.args.first()? {
                Expr::Variable(x) => Some(x),
                _ => None,
            }
        }
    }
}

type Definitions = BTreeMap<String, Vec<(String, String)>>;

/// The definitions of the model's observeds `obs` in `flat`, canonical, by
/// observed: `(lhs, rhs)` in equation order.
fn observed_forms(flat: &FlattenedSystem, obs: &BTreeSet<&str>) -> Definitions {
    let mut out = Definitions::new();
    for eq in &flat.equations {
        if differentiated(&eq.lhs).is_some() {
            continue;
        }
        if let Some(y) = defined_name(eq).filter(|y| obs.contains(y)) {
            out.entry(y.to_string())
                .or_default()
                .push((canonical(&eq.lhs), canonical(&eq.rhs)));
        }
    }
    out
}

/// The `ic` equations of `flat`, canonical, sorted.
fn ic_forms(flat: &FlattenedSystem) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = flat
        .field_ics
        .iter()
        .map(|(x, e)| (x.clone(), canonical(e)))
        .collect();
    out.sort();
    out
}

/// §8.1's `rhs_sum`: the flattened right-hand-side documents `docs`
/// (operator, flattened document, in code-point order of the operator)
/// sum to the flattened model. Returns the number of summands compared, or
/// the first difference found.
pub fn rhs_sum(info: &ModelInfo, docs: &[(&str, &FlattenedSystem)]) -> Result<usize, String> {
    let flat = &info.flat;
    let obs: BTreeSet<&str> = flat.observed_variables.keys().map(String::as_str).collect();
    let mut zeros = BTreeMap::new();
    for x in &info.states {
        let e: Expr = serde_json::from_value(derived::zero(info, x))
            .map_err(|e| format!("the zero right-hand side of {x}: {e}"))?;
        zeros.insert(x.as_str(), canonical(&e));
    }
    // The flattened model: each state's summands, lifted or not.
    let mut want: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    let mut lifted: BTreeMap<&str, bool> = BTreeMap::new();
    let mut output_idx: BTreeMap<&str, Option<Vec<String>>> = BTreeMap::new();
    for (i, eq) in flat.equations.iter().enumerate() {
        let Some(x) = differentiated(&eq.lhs).filter(|x| info.is_state(x)) else {
            continue;
        };
        let l = flat.equation_lifted.get(i).copied().unwrap_or(false);
        let s = summands(&eq.rhs, &zeros[x], l).ok_or_else(|| {
            format!("the flattened model's lifted equation for {x} has no faq right-hand side")
        })?;
        if want.insert(x, s).is_some() {
            return Err(format!("the flattened model has two equations for {x}"));
        }
        lifted.insert(x, l);
        output_idx.insert(x, if l { faq_output_idx(&eq.rhs) } else { None });
    }
    let observeds = observed_forms(flat, &obs);
    let ics = ic_forms(flat);
    let declared_params: BTreeSet<&str> = flat
        .parameters
        .keys()
        .map(String::as_str)
        .filter(|p| !info.data_fed.contains_key(*p) || info.used.contains(*p))
        .collect();
    let states: BTreeSet<&str> = info.states.iter().map(String::as_str).collect();
    let mut got: BTreeMap<&str, Vec<String>> = states.iter().map(|x| (*x, Vec::new())).collect();
    for (o, d) in docs {
        let mut seen = BTreeSet::new();
        for eq in &d.equations {
            if let Some(x) = differentiated(&eq.lhs) {
                let Some(&l) = lifted.get(x) else {
                    return Err(format!(
                        "operator {o}'s right-hand-side document has an equation for {x}, which is not a state of the model"
                    ));
                };
                if !seen.insert(x) {
                    return Err(format!(
                        "operator {o}'s right-hand-side document has two equations for {x}"
                    ));
                }
                let s = summands(&eq.rhs, &zeros[x], l).ok_or_else(|| {
                    format!("operator {o}'s right-hand-side document writes the lifted state {x} without the lift's faq")
                })?;
                if l && !s.is_empty() && faq_output_idx(&eq.rhs) != output_idx[x] {
                    return Err(format!(
                        "operator {o}'s right-hand-side document writes {x} over the indices {:?}, the model over {:?}",
                        faq_output_idx(&eq.rhs),
                        output_idx[x]
                    ));
                }
                got.get_mut(x).expect("every state").extend(s);
                continue;
            }
            if !defined_name(eq).is_some_and(|y| obs.contains(y)) {
                return Err(format!(
                    "operator {o}'s right-hand-side document has an equation that defines neither a state nor an observed: {}",
                    canonical(&eq.lhs)
                ));
            }
        }
        if let Some(x) = states.iter().find(|x| !seen.contains(*x)) {
            return Err(format!(
                "operator {o}'s right-hand-side document has no equation for the state {x}"
            ));
        }
        // Observeds: the definitions the model gives them.
        for (y, defs) in observed_forms(d, &obs) {
            if observeds.get(&y) != Some(&defs) {
                return Err(format!(
                    "operator {o}'s right-hand-side document defines the observed {y} otherwise than the model"
                ));
            }
        }
        // Declarations: every state and parameter, with the model's shape
        // and default.
        let dstates: BTreeSet<&str> = d
            .state_variables
            .keys()
            .map(String::as_str)
            .filter(|n| !obs.contains(n))
            .collect();
        if dstates != states {
            return Err(format!(
                "operator {o}'s right-hand-side document declares the states {dstates:?}, the model has {states:?}"
            ));
        }
        let dparams: BTreeSet<&str> = d.parameters.keys().map(String::as_str).collect();
        if dparams != declared_params {
            return Err(format!(
                "operator {o}'s right-hand-side document declares the parameters {dparams:?}, not the model's {declared_params:?}"
            ));
        }
        for (dict, mdict) in [
            (&d.state_variables, &flat.state_variables),
            (&d.parameters, &flat.parameters),
        ] {
            for (n, v) in dict {
                if obs.contains(n.as_str()) {
                    continue;
                }
                let w = &mdict[n];
                if shape_form(v) != shape_form(w) {
                    return Err(format!(
                        "operator {o}'s right-hand-side document declares {n} with the shape {:?}, the model with {:?}",
                        shape_form(v),
                        shape_form(w)
                    ));
                }
                if default_form(v) != default_form(w) {
                    return Err(format!(
                        "operator {o}'s right-hand-side document declares {n} with a default other than the model's"
                    ));
                }
            }
        }
        if ic_forms(d) != ics {
            return Err(format!(
                "operator {o}'s right-hand-side document's ic equations are not the model's"
            ));
        }
    }
    let mut n = 0;
    for x in &info.states {
        let mut a = want[x.as_str()].clone();
        let mut b = got[x.as_str()].clone();
        a.sort();
        b.sort();
        if a != b {
            let missing = a.iter().find(|s| !b.contains(s));
            let extra = b.iter().find(|s| !a.contains(s));
            return Err(format!(
                "the operators' right-hand sides of {x} do not sum to the model's: {}{}({} summands in the model, {} in the operators)",
                missing.map_or(String::new(), |s| format!("the model's summand {s} is in no operator's; ")),
                extra.map_or(String::new(), |s| format!("the operators' summand {s} is not the model's; ")),
                a.len(),
                b.len()
            ));
        }
        n += a.len();
    }
    Ok(n)
}
