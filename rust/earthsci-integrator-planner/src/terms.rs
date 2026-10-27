//! Pass 1, terms (spec/plan.md §5.1.1–§5.1.2, D64, D65).

use crate::error::{codes, refuse, Result};
use crate::model::{differentiated, node, ModelInfo};
use earthsci_ast::{Expr, ExpressionNode, TermOrigin};
use std::collections::BTreeSet;

/// One additive summand of one contribution of a state equation.
#[derive(Debug, Clone)]
pub struct Term {
    /// The dependent variable's flattened name.
    pub variable: String,
    /// The equation's position in the flattened model.
    pub equation: usize,
    /// The contribution's position in the equation's `equation_contributions`.
    pub contribution: usize,
    /// The summand's position in the contribution's decomposition.
    pub summand: usize,
    /// The summand; for a lifted equation, the per-cell body.
    pub expr: Expr,
    /// Whether the equation is a pointwise lift (`equation_lifted`).
    pub lifted: bool,
    /// The contribution's origins.
    pub origins: Vec<TermOrigin>,
    /// The provenance label (§5.1.1).
    pub provenance: String,
    /// The owner (§5.1.2), `None` when the term has none.
    pub owner: Option<String>,
    /// The states, parameters and observeds the summand references.
    pub names: BTreeSet<String>,
}

/// EarthSciASTSplitter's `additive_terms` as spec/plan.md §5.1.1 states it:
/// n-ary `+` concatenates its arguments' decompositions, binary `-` gives
/// the left operand's summands then each of the right operand's negated,
/// unary `-` negates each summand of its operand, anything else is one
/// summand.
pub fn additive_terms(e: &Expr) -> Vec<Expr> {
    let mut out = Vec::new();
    push_terms(e, &mut out);
    out
}

fn negate(t: Expr) -> Expr {
    Expr::operator(ExpressionNode {
        op: "-".into(),
        args: vec![t],
        ..Default::default()
    })
}

fn push_terms(e: &Expr, out: &mut Vec<Expr>) {
    if let Some(n) = node(e) {
        match (n.op.as_str(), n.args.len()) {
            ("+", _) => {
                for a in &n.args {
                    push_terms(a, out);
                }
                return;
            }
            ("-", 2) => {
                push_terms(&n.args[0], out);
                out.extend(additive_terms(&n.args[1]).into_iter().map(negate));
                return;
            }
            ("-", 1) => {
                out.extend(additive_terms(&n.args[0]).into_iter().map(negate));
                return;
            }
            _ => {}
        }
    }
    out.push(e.clone());
}

/// A numeric literal equal to zero.
fn is_literal_zero(e: &Expr) -> bool {
    match e {
        Expr::Integer(0) => true,
        Expr::Number(x) => *x == 0.0,
        _ => false,
    }
}

/// The provenance label of a contribution's origins (§5.1.1).
pub fn provenance(origins: &[TermOrigin]) -> String {
    origins
        .iter()
        .map(|o| match o {
            TermOrigin::Component { component } => component.clone(),
            TermOrigin::Coupling { coupling } => format!("coupling:{coupling}"),
        })
        .collect::<Vec<_>>()
        .join("+")
}

/// A name's component: the name without its last dot-separated segment.
pub fn component_of(name: &str) -> &str {
    name.rsplit_once('.').map(|(c, _)| c).unwrap_or("")
}

/// The owner of a term (§5.1.2, D65).
fn owner(variable: &str, origins: &[TermOrigin], names: &BTreeSet<String>) -> Option<String> {
    if let [TermOrigin::Component { component }] = origins {
        return Some(component.clone());
    }
    let own = component_of(variable);
    let others: BTreeSet<&str> = names
        .iter()
        .map(|n| component_of(n))
        .filter(|c| *c != own)
        .collect();
    match others.len() {
        0 => Some(own.to_string()),
        1 => others.into_iter().next().map(str::to_string),
        _ => None,
    }
}

/// The terms of the flattened model, in flattened-equation order, then
/// contribution order, then summand order (§5.1.1, §2.1). An empty
/// `origins` list is `term_origin_unknown`.
pub fn terms(info: &ModelInfo) -> Result<Vec<Term>> {
    let flat = &info.flat;
    let mut out = Vec::new();
    for (i, eq) in flat.equations.iter().enumerate() {
        let Some(variable) = differentiated(&eq.lhs) else {
            continue;
        };
        if !info.is_state(variable) {
            continue;
        }
        let lifted = flat.equation_lifted.get(i).copied().unwrap_or(false);
        // An equation without a contribution list (a hand-assembled system)
        // is one contribution with no origins.
        let fallback;
        let contributions = match flat.equation_contributions.get(i) {
            Some(c) if !c.is_empty() => c.as_slice(),
            _ => {
                let body = if lifted {
                    node(&eq.rhs)
                        .and_then(|n| n.expr.as_deref())
                        .unwrap_or(&eq.rhs)
                } else {
                    &eq.rhs
                };
                fallback = [earthsci_ast::Contribution {
                    expr: body.clone(),
                    origins: vec![],
                }];
                &fallback
            }
        };
        for (c, contribution) in contributions.iter().enumerate() {
            if contribution.origins.is_empty() {
                return Err(refuse(
                    codes::TERM_ORIGIN_UNKNOWN,
                    "/model",
                    format!("contribution {c} of the equation for {variable} has no origins"),
                ));
            }
            let label = provenance(&contribution.origins);
            let summands = additive_terms(&contribution.expr)
                .into_iter()
                .filter(|s| !is_literal_zero(s));
            for (s, expr) in summands.enumerate() {
                let names = info.names_in(&expr);
                out.push(Term {
                    variable: variable.to_string(),
                    equation: i,
                    contribution: c,
                    summand: s,
                    owner: owner(variable, &contribution.origins, &names),
                    expr,
                    lifted,
                    origins: contribution.origins.clone(),
                    provenance: label.clone(),
                    names,
                });
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(s: &str) -> Expr {
        serde_json::from_str(s).unwrap()
    }

    fn js(v: &[Expr]) -> String {
        serde_json::to_string(v).unwrap()
    }

    #[test]
    fn decomposition() {
        // n-ary +, nested.
        let t = additive_terms(&e(
            r#"{"op":"+","args":["a",{"op":"+","args":["b","c"]},"d"]}"#,
        ));
        assert_eq!(js(&t), r#"["a","b","c","d"]"#);
        // Binary -: the right operand's summands negated one by one.
        let t = additive_terms(&e(
            r#"{"op":"-","args":[{"op":"-","args":["a","b"]},{"op":"+","args":["c","d"]}]}"#,
        ));
        assert_eq!(
            js(&t),
            r#"["a",{"op":"-","args":["b"]},{"op":"-","args":["c"]},{"op":"-","args":["d"]}]"#
        );
        // Unary -.
        let t = additive_terms(&e(r#"{"op":"-","args":[{"op":"+","args":["a","b"]}]}"#));
        assert_eq!(
            js(&t),
            r#"[{"op":"-","args":["a"]},{"op":"-","args":["b"]}]"#
        );
        // Double negation is not simplified.
        let t = additive_terms(&e(r#"{"op":"-","args":["a",{"op":"-","args":["b"]}]}"#));
        assert_eq!(
            js(&t),
            r#"["a",{"op":"-","args":[{"op":"-","args":["b"]}]}]"#
        );
        // Anything else is one summand.
        let t = additive_terms(&e(r#"{"op":"*","args":[{"op":"+","args":["a","b"]},"c"]}"#));
        assert_eq!(t.len(), 1);
        assert!(is_literal_zero(&e("0")));
        assert!(is_literal_zero(&e("0.0")));
        assert!(!is_literal_zero(&e("1")));
        // In `a - 0` the second summand is `-(0)`, an operation, and is kept
        // (spec/plan.md §5.1.1).
        let t = additive_terms(&e(r#"{"op":"-","args":["a",0]}"#));
        assert_eq!(js(&t), r#"["a",{"op":"-","args":[0]}]"#);
        assert!(t.iter().all(|s| !is_literal_zero(s)));
    }

    #[test]
    fn owners() {
        let comp = |c: &str| TermOrigin::Component {
            component: c.into(),
        };
        let names = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<BTreeSet<_>>();
        // Rule 1.
        assert_eq!(
            owner("Chem.O3", &[comp("Chem")], &names(&["Met.h"])),
            Some("Chem".into())
        );
        let cpl = [TermOrigin::Coupling { coupling: 0 }];
        // Rule 2: no other component, one, two.
        assert_eq!(
            owner("Chem.O3", &cpl, &names(&["Chem.O3"])),
            Some("Chem".into())
        );
        assert_eq!(
            owner("Chem.NO", &cpl, &names(&["Chem.NO", "Dep.kNO"])),
            Some("Dep".into())
        );
        assert_eq!(
            owner("Chem.O3", &cpl, &names(&["Chem.O3", "Dep.vd", "Met.h"])),
            None
        );
        // A collapsed contribution goes by rule 2 too.
        assert_eq!(
            owner(
                "A.B.x",
                &[comp("A.B"), TermOrigin::Coupling { coupling: 4 }],
                &names(&["A.B.x", "A.B.C.y"])
            ),
            Some("A.B.C".into())
        );
        assert_eq!(
            provenance(&[comp("SuperFast"), TermOrigin::Coupling { coupling: 4 }]),
            "SuperFast+coupling:4"
        );
    }
}
