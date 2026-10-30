//! Pass 5, stiffness and strategy (spec/plan.md §5.5, D70, D71).
//!
//! Stiffness rules 1–3 of §5.5.2 and the method table of §5.5.3 as a pure
//! function of the leaf's facts. Stiffness rule 4 (the Gershgorin bound,
//! [`crate::validation::gershgorin`]) and the facts that come from structure
//! (the block size and whether every operator has `structure.block`) are
//! computed by the pipeline ([`crate::plan`]); a leaf resolved without them
//! is [`PlanError::NotYetImplemented`].

use crate::error::{codes, refuse, PlanError, Result};
use crate::locality::Class;
use earthsci_integrator_solvers::tableau::{tableau, ErrorEstimate, Family};
use serde_json::{json, Map, Value};

/// A method's family as the planner needs it.
pub fn family(alg: &str) -> Option<Family> {
    tableau(alg).map(|t| t.family)
}

/// Methods that form a Jacobian: Rosenbrock, SDIRK, `IMEXEuler`, `Theta`
/// (spec/plan.md §7.1, §5.5.2 rule 1).
pub fn forms_jacobian(alg: &str) -> bool {
    !matches!(family(alg), Some(Family::ExplicitRk) | None)
}

/// Whether a method has an embedded error estimate (spec/methods.md §2).
pub fn has_error_estimate(alg: &str) -> bool {
    tableau(alg).is_some_and(|t| !matches!(t.error_estimate, ErrorEstimate::None))
}

/// A Rosenbrock method other than `Rosenbrock23` (spec/plan.md §7.5).
pub fn classical_rosenbrock(alg: &str) -> bool {
    matches!(family(alg), Some(Family::Rosenbrock)) && alg != "Rosenbrock23"
}

/// Where an operator's stiffness class came from (§5.5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StiffnessSource {
    RunConfig,
    ReactionSystems,
    SolverHint,
    Gershgorin,
}

impl StiffnessSource {
    pub fn as_str(self) -> &'static str {
        match self {
            StiffnessSource::RunConfig => "run_config",
            StiffnessSource::ReactionSystems => "reaction_systems",
            StiffnessSource::SolverHint => "solver_hint",
            StiffnessSource::Gershgorin => "gershgorin",
        }
    }
}

/// An operator's stiffness (`stiffness: {class, source}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stiffness {
    pub stiff: bool,
    pub source: StiffnessSource,
}

impl Stiffness {
    pub fn to_json(self) -> Value {
        json!({
            "class": if self.stiff { "stiff" } else { "nonstiff" },
            "source": self.source.as_str(),
        })
    }
}

/// Stiffness rules 1–3 of §5.5.2, in order. `None` means rule 4, the
/// Gershgorin bound, decides.
///
/// - `leaf_alg`: the `alg` of the leaf that advances the operator, as given.
/// - `reaction_system_term`: some term has a single component origin that
///   is a reaction system of the model.
pub fn stiffness_rules_1_to_3(
    leaf_alg: Option<&str>,
    reaction_system_term: bool,
    class: Class,
    solver_stiffness: Option<&str>,
) -> Option<Stiffness> {
    if let Some(alg) = leaf_alg.filter(|a| *a != "auto") {
        return Some(Stiffness {
            stiff: forms_jacobian(alg),
            source: StiffnessSource::RunConfig,
        });
    }
    if reaction_system_term {
        return Some(Stiffness {
            stiff: true,
            source: StiffnessSource::ReactionSystems,
        });
    }
    if class == Class::Pointwise && solver_stiffness == Some("high") {
        return Some(Stiffness {
            stiff: true,
            source: StiffnessSource::SolverHint,
        });
    }
    None
}

/// What §5.5.3 needs to know about a leaf.
#[derive(Debug, Clone, PartialEq)]
pub struct LeafFacts {
    /// The coarsest locality of its operators.
    pub class: Class,
    /// Whether any of its operators is stiff; `None` when a stiffness is
    /// not known (rule 4 not computed).
    pub stiff: Option<bool>,
    /// Its operators that are column operators and stiff, in leaf order;
    /// `None` when a column operator's stiffness is not known yet.
    pub stiff_column_ops: Option<Vec<String>>,
    /// Its operators in leaf order with their classes.
    pub ops: Vec<(String, Class)>,
    /// Its resolved `reltol`.
    pub reltol: f64,
    /// Whether it gives a `dt`.
    pub dt_given: bool,
    /// Whether the leaf's state variables are all arrays of one shape or all
    /// scalars (spec/operator-interface.md §8.3).
    pub uniform_shapes: bool,
    /// The size of the union of its operators' `structure.block` variables
    /// when every operator has `structure.block`, `Some(None)` when one has
    /// none; `None` when structure is not known.
    pub block_vars: Option<Option<usize>>,
}

/// The resolved method of a leaf (§5.5.3).
#[derive(Debug, Clone, PartialEq)]
pub struct LeafStrategy {
    pub alg: String,
    /// `jacobian`, for methods that form one.
    pub jacobian: Option<String>,
    /// `linsolve` in object form, for methods that form a Jacobian.
    pub linsolve: Option<Value>,
    pub dt_scope: String,
    /// For an `IMEXEuler` leaf chosen by the table: the implicit operators.
    pub imex_implicit: Option<Vec<String>>,
}

/// `linsolve` in the object form of §7.1, every defaulted option written.
pub fn linsolve_object(given: &Value) -> Value {
    let (name, opts) = match given {
        Value::String(s) => (s.clone(), Map::new()),
        Value::Object(m) => match m.iter().next() {
            Some((k, v)) => (k.clone(), v.as_object().cloned().unwrap_or_default()),
            None => return given.clone(),
        },
        _ => return given.clone(),
    };
    let mut opts = opts;
    match name.as_str() {
        "block_lu" => {
            opts.entry("pivot").or_insert(json!(true));
            opts.entry("refine").or_insert(json!(0));
        }
        "sparse_block_lu" => {
            opts.entry("refine").or_insert(json!(0));
        }
        _ => {}
    }
    let mut out = Map::new();
    out.insert(name, Value::Object(opts));
    Value::Object(out)
}

fn not_known(what: &str) -> PlanError {
    PlanError::NotYetImplemented(format!(
        "{what} needs the operators' structure and Gershgorin bounds"
    ))
}

/// The table row's `linsolve` for a leaf's class, when the leaf forms a
/// Jacobian and gives none (§5.5.3).
fn table_linsolve(facts: &LeafFacts, pointer: &str) -> Result<Value> {
    let unavailable = |m: &str| refuse(codes::AUTO_STRATEGY_UNAVAILABLE, pointer, m.to_string());
    match facts.class {
        Class::Pointwise => {
            // Storage is block_diagonal only when the leaf's states are all
            // arrays of one shape or all scalars and every operator has
            // `structure.block`; the first condition needs no structure.
            let not_block = || {
                unavailable(
                    "the leaf's Jacobian storage is not block_diagonal, which block_lu and sparse_block_lu need",
                )
            };
            if !facts.uniform_shapes {
                return Err(not_block());
            }
            let block = facts
                .block_vars
                .ok_or_else(|| not_known("the block size of a pointwise leaf"))?;
            match block {
                Some(n) => Ok(if n <= 16 {
                    json!({"block_lu": {"pivot": true, "refine": 0}})
                } else {
                    json!({"sparse_block_lu": {"refine": 0}})
                }),
                None => Err(not_block()),
            }
        }
        Class::Column => Ok(json!({"banded": {}})),
        Class::Stencil | Class::Global => Err(unavailable(
            "an implicit method on a stencil or global leaf has no M2 linear solver; give linsolve",
        )),
    }
}

/// §5.5.3: resolve a leaf's `alg`, `jacobian`, `linsolve` and `dt_scope`.
/// `given` is the leaf node as written; `pointer` is where a refusal points.
pub fn resolve_leaf(given: &Value, facts: &LeafFacts, pointer: &str) -> Result<LeafStrategy> {
    let given_str = |k: &str| {
        given
            .get(k)
            .and_then(Value::as_str)
            .filter(|s| *s != "auto")
            .map(str::to_string)
    };
    let given_linsolve = given
        .get("linsolve")
        .filter(|v| v.as_str() != Some("auto"))
        .map(linsolve_object);
    let given_scope = given
        .get("control")
        .and_then(|c| c.get("dt_scope"))
        .and_then(Value::as_str)
        .filter(|s| *s != "auto")
        .map(str::to_string);
    let table_scope = || {
        if facts.class == Class::Pointwise {
            "cell".to_string()
        } else {
            "global".to_string()
        }
    };
    let unavailable = |m: String| refuse(codes::AUTO_STRATEGY_UNAVAILABLE, pointer, m);

    let mut imex_implicit = None;
    // The pointwise stiff row's linsolve. The row needs block_diagonal
    // storage even when the leaf gives linsolve (§5.5.3), so it is resolved,
    // and refused, either way.
    let mut row_linsolve = None;
    let alg = match given_str("alg") {
        Some(alg) => alg,
        None => {
            // The stencil row reads only its column operators' stiffness.
            let stiff = facts.class != Class::Stencil
                && facts.stiff.ok_or_else(|| {
                    not_known("the stiffness of an operator by its Gershgorin bound")
                })?;
            match (facts.class, stiff) {
                (Class::Pointwise, true) => {
                    row_linsolve = Some(table_linsolve(facts, pointer)?);
                    if facts.reltol >= 1e-3 {
                        "Rosenbrock23".into()
                    } else {
                        "Rodas4".into()
                    }
                }
                (Class::Pointwise, false) | (Class::Column, false) => "Tsit5".into(),
                (Class::Column, true) => "TRBDF2".into(),
                (Class::Stencil, _) => {
                    let stiff_cols = facts.stiff_column_ops.clone().ok_or_else(|| {
                        not_known("the stiffness of a column operator by its Gershgorin bound")
                    })?;
                    if stiff_cols.is_empty() {
                        "SSPRK43".into()
                    } else if facts.dt_given {
                        imex_implicit = Some(stiff_cols);
                        "IMEXEuler".into()
                    } else {
                        return Err(unavailable(
                            "a stencil leaf with a stiff column operator and no dt has no M2 method".into(),
                        ));
                    }
                }
                (Class::Global, true) => {
                    return Err(unavailable("a stiff global leaf has no M2 method".into()))
                }
                (Class::Global, false) => "SSPRK43".into(),
            }
        }
    };
    let (jacobian, linsolve) = if forms_jacobian(&alg) {
        let jacobian = given_str("jacobian").unwrap_or_else(|| "symbolic".into());
        let linsolve = match (given_linsolve, row_linsolve) {
            (Some(l), _) | (None, Some(l)) => l,
            (None, None) if imex_implicit.is_some() => json!({"banded": {}}),
            (None, None) => table_linsolve(facts, pointer)?,
        };
        (Some(jacobian), Some(linsolve))
    } else {
        (None, None)
    };
    Ok(LeafStrategy {
        alg,
        jacobian,
        linsolve,
        dt_scope: given_scope.unwrap_or_else(table_scope),
        imex_implicit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(class: Class, stiff: bool) -> LeafFacts {
        LeafFacts {
            class,
            stiff: Some(stiff),
            stiff_column_ops: Some(vec![]),
            ops: vec![("op".into(), class)],
            reltol: 1e-3,
            dt_given: false,
            uniform_shapes: true,
            block_vars: Some(Some(3)),
        }
    }

    fn code(r: Result<LeafStrategy>) -> &'static str {
        match r {
            Err(PlanError::Refused(r)) => r.code,
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn stencil_needs_only_column_stiffness() {
        let f = LeafFacts {
            stiff: None,
            ..facts(Class::Stencil, false)
        };
        let s = resolve_leaf(&json!({"alg": "auto"}), &f, "/schedule").unwrap();
        assert_eq!(s.alg, "SSPRK43");
    }

    #[test]
    fn method_families() {
        assert!(forms_jacobian("Rodas4"));
        assert!(forms_jacobian("Kvaerno4"));
        assert!(forms_jacobian("IMEXEuler"));
        assert!(forms_jacobian("Theta"));
        assert!(!forms_jacobian("SSPRK43"));
        assert!(!forms_jacobian("WRF_RK3"));
        for a in [
            "ForwardEuler",
            "SSPRK33",
            "SSPRK104",
            "WRF_RK3",
            "IMEXEuler",
            "Theta",
        ] {
            assert!(!has_error_estimate(a), "{a}");
        }
        for a in [
            "Tsit5",
            "SSPRK43",
            "Rosenbrock23",
            "Kvaerno4",
            "RK4",
            "BackwardEuler",
        ] {
            assert!(has_error_estimate(a), "{a}");
        }
        assert!(classical_rosenbrock("Rodas4"));
        assert!(classical_rosenbrock("KPP_Ros3"));
        assert!(!classical_rosenbrock("Rosenbrock23"));
        assert!(!classical_rosenbrock("TRBDF2"));
    }

    #[test]
    fn stiffness_rules() {
        let s = |a, r, c, h| stiffness_rules_1_to_3(a, r, c, h);
        assert_eq!(
            s(Some("Rodas4"), false, Class::Stencil, None),
            Some(Stiffness {
                stiff: true,
                source: StiffnessSource::RunConfig
            })
        );
        assert_eq!(
            s(Some("SSPRK33"), true, Class::Pointwise, Some("high")),
            Some(Stiffness {
                stiff: false,
                source: StiffnessSource::RunConfig
            })
        );
        assert_eq!(
            s(Some("auto"), true, Class::Pointwise, Some("high")),
            Some(Stiffness {
                stiff: true,
                source: StiffnessSource::ReactionSystems
            })
        );
        assert_eq!(
            s(None, false, Class::Pointwise, Some("high")),
            Some(Stiffness {
                stiff: true,
                source: StiffnessSource::SolverHint
            })
        );
        assert_eq!(s(None, false, Class::Column, Some("high")), None);
        assert_eq!(s(None, false, Class::Pointwise, Some("moderate")), None);
    }

    #[test]
    fn auto_table() {
        let auto = json!({"leaf": "op", "alg": "auto"});
        let r = resolve_leaf(&auto, &facts(Class::Pointwise, true), "/schedule").unwrap();
        assert_eq!(r.alg, "Rosenbrock23");
        assert_eq!(r.jacobian.as_deref(), Some("symbolic"));
        assert_eq!(
            r.linsolve,
            Some(json!({"block_lu": {"pivot": true, "refine": 0}}))
        );
        assert_eq!(r.dt_scope, "cell");
        let mut f = facts(Class::Pointwise, true);
        f.reltol = 1e-6;
        f.block_vars = Some(Some(17));
        let r = resolve_leaf(&auto, &f, "/schedule").unwrap();
        assert_eq!(r.alg, "Rodas4");
        assert_eq!(r.linsolve, Some(json!({"sparse_block_lu": {"refine": 0}})));
        f.block_vars = Some(None);
        assert_eq!(
            code(resolve_leaf(&auto, &f, "/schedule")),
            codes::AUTO_STRATEGY_UNAVAILABLE
        );
        f.block_vars = Some(Some(2));
        f.uniform_shapes = false;
        assert_eq!(
            code(resolve_leaf(&auto, &f, "/schedule")),
            codes::AUTO_STRATEGY_UNAVAILABLE
        );
        let r = resolve_leaf(&auto, &facts(Class::Pointwise, false), "/s").unwrap();
        assert_eq!(
            (r.alg.as_str(), r.dt_scope.as_str(), r.linsolve),
            ("Tsit5", "cell", None)
        );
        let r = resolve_leaf(&auto, &facts(Class::Column, true), "/s").unwrap();
        assert_eq!(r.alg, "TRBDF2");
        assert_eq!(r.linsolve, Some(json!({"banded": {}})));
        assert_eq!(r.dt_scope, "global");
        let r = resolve_leaf(&auto, &facts(Class::Column, false), "/s").unwrap();
        assert_eq!((r.alg.as_str(), r.dt_scope.as_str()), ("Tsit5", "global"));
        for stiff in [false, true] {
            let r = resolve_leaf(&auto, &facts(Class::Stencil, stiff), "/s").unwrap();
            assert_eq!(r.alg, "SSPRK43");
        }
        let mut f = facts(Class::Stencil, true);
        f.stiff_column_ops = Some(vec!["mix".into()]);
        assert_eq!(
            code(resolve_leaf(&auto, &f, "/s")),
            codes::AUTO_STRATEGY_UNAVAILABLE
        );
        f.dt_given = true;
        let r = resolve_leaf(&auto, &f, "/s").unwrap();
        assert_eq!(r.alg, "IMEXEuler");
        assert_eq!(r.imex_implicit, Some(vec!["mix".to_string()]));
        assert_eq!(r.linsolve, Some(json!({"banded": {}})));
        assert_eq!(
            code(resolve_leaf(&auto, &facts(Class::Global, true), "/s")),
            codes::AUTO_STRATEGY_UNAVAILABLE
        );
        assert_eq!(
            resolve_leaf(&auto, &facts(Class::Global, false), "/s")
                .unwrap()
                .alg,
            "SSPRK43"
        );
        // A given linsolve does not lift the row's block_diagonal condition.
        let given = json!({"leaf": "op", "alg": "auto", "linsolve": "dense_lu"});
        let mut f = facts(Class::Pointwise, true);
        assert_eq!(
            resolve_leaf(&given, &f, "/s").unwrap().linsolve,
            Some(json!({"dense_lu": {}}))
        );
        f.uniform_shapes = false;
        assert_eq!(
            code(resolve_leaf(&given, &f, "/s")),
            codes::AUTO_STRATEGY_UNAVAILABLE
        );
        // Unknown stiffness is not a refusal.
        let mut f = facts(Class::Pointwise, true);
        f.stiff = None;
        assert!(matches!(
            resolve_leaf(&auto, &f, "/s"),
            Err(PlanError::NotYetImplemented(_))
        ));
    }

    #[test]
    fn given_methods() {
        // Explicit: no jacobian, no linsolve, the class's dt_scope.
        let r = resolve_leaf(
            &json!({"alg": "ForwardEuler", "dt": "step"}),
            &facts(Class::Pointwise, false),
            "/s",
        )
        .unwrap();
        assert_eq!(
            (r.jacobian, r.linsolve, r.dt_scope.as_str()),
            (None, None, "cell")
        );
        // Implicit with omitted jacobian and linsolve: the class's row.
        let r = resolve_leaf(
            &json!({"alg": "Rodas4"}),
            &facts(Class::Pointwise, true),
            "/s",
        )
        .unwrap();
        assert_eq!(r.jacobian.as_deref(), Some("symbolic"));
        assert_eq!(
            r.linsolve,
            Some(json!({"block_lu": {"pivot": true, "refine": 0}}))
        );
        // Given options are kept, in object form with defaults.
        let r = resolve_leaf(
            &json!({"alg": "Rodas4", "linsolve": {"block_lu": {"pivot": false}}, "control": {"dt_scope": "global"}}),
            &facts(Class::Pointwise, true),
            "/s",
        )
        .unwrap();
        assert_eq!(
            r.linsolve,
            Some(json!({"block_lu": {"pivot": false, "refine": 0}}))
        );
        assert_eq!(r.dt_scope, "global");
        let r = resolve_leaf(
            &json!({"alg": "Rodas4", "linsolve": "dense_lu"}),
            &facts(Class::Pointwise, true),
            "/s",
        )
        .unwrap();
        assert_eq!(r.linsolve, Some(json!({"dense_lu": {}})));
        // An implicit method on a stencil leaf without linsolve.
        assert_eq!(
            code(resolve_leaf(
                &json!({"alg": "TRBDF2"}),
                &facts(Class::Stencil, true),
                "/s"
            )),
            codes::AUTO_STRATEGY_UNAVAILABLE
        );
        assert_eq!(
            resolve_leaf(&json!({"alg": "TRBDF2"}), &facts(Class::Column, true), "/s")
                .unwrap()
                .linsolve,
            Some(json!({"banded": {}}))
        );
        assert_eq!(
            linsolve_object(&json!("sparse_block_lu")),
            json!({"sparse_block_lu": {"refine": 0}})
        );
        assert_eq!(linsolve_object(&json!("gmres")), json!({"gmres": {}}));
        assert_eq!(
            linsolve_object(&json!({"gmres": {"restart": 30}})),
            json!({"gmres": {"restart": 30}})
        );
    }
}
