//! Method coefficient tables (spec/methods.md §1, D54).
//!
//! Every `spec/tableaus/*.json` file is embedded at compile time with
//! `include_str!` and parsed on first use. No coefficient is restated by hand
//! in this crate.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// The `controller` field of a table: OrdinaryDiffEqCore's defaults for the
/// method (spec/methods.md §8).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControllerConstants {
    pub beta1: f64,
    pub beta2: f64,
    pub gamma: f64,
    pub qmin: f64,
    pub qmax: f64,
    pub qmax_first_step: f64,
    pub qsteady_min: f64,
    pub qsteady_max: f64,
    pub failfactor: f64,
    pub qoldinit: f64,
    pub pid_beta3: f64,
    pub pid_accept_safety: f64,
    pub source: serde_json::Value,
}

/// SSPRK43's embedded solution (spec/methods.md §3.2, `shu_osher_embedded`).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShuOsherEmbedded {
    pub alpha: Vec<f64>,
    pub beta: Vec<f64>,
    pub scale: f64,
}

/// RK4's residual-control constants (spec/methods.md §3.2).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidualControl {
    pub sigma: Vec<f64>,
    pub scale: f64,
}

/// BackwardEuler's divided-difference constant (spec/methods.md §5.5).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DividedDifference {
    pub constant: f64,
}

/// KPP's own step-size rule, recorded for M7 (spec/methods.md §4.4).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KppController {
    pub fac_min: f64,
    pub fac_max: f64,
    pub fac_rej: f64,
    pub fac_safe: f64,
    pub source: serde_json::Value,
}

/// KPP-specific data of a KPP Rosenbrock table.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Kpp {
    pub elo: f64,
    pub new_f: Vec<bool>,
    pub controller: KppController,
}

/// `d` is a vector for the transformed Rosenbrock form and a number for
/// Rosenbrock23.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum DCoefficients {
    Vector(Vec<f64>),
    Scalar(f64),
}

/// The method family (schema `family`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    ExplicitRk,
    Rosenbrock,
    Sdirk,
    Imex,
    Theta,
}

/// The arithmetic form of the step (schema `form`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Form {
    Butcher,
    ShuOsher,
    RosenbrockTransformed,
    Rosenbrock23,
    Sdirk,
    ImexEuler,
    Theta,
}

/// How the error estimate is formed (schema `error_estimate`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorEstimate {
    None,
    Embedded,
    ShuOsherEmbedded,
    ResidualControl,
    Rosenbrock23,
    EmbeddedSmoothed,
    DividedDifference,
}

/// One coefficient table, as `spec/schemas/tableau.schema.json` defines it.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tableau {
    pub schema: String,
    pub name: String,
    pub family: Family,
    pub form: Form,
    #[serde(default)]
    pub ordinarydiffeq_algorithm: Option<String>,
    pub stages: usize,
    pub order: u32,
    pub embedded_order: Option<u32>,
    pub error_estimate: ErrorEstimate,
    pub first_same_as_last: bool,
    pub stiffly_accurate: bool,
    #[serde(default)]
    pub explicit_first_stage: Option<bool>,
    #[serde(rename = "A", default)]
    pub a: Option<Vec<Vec<f64>>>,
    #[serde(rename = "C", default)]
    pub c_mat: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pub b: Option<Vec<f64>>,
    #[serde(default)]
    pub btilde: Option<Vec<f64>>,
    #[serde(default)]
    pub c: Option<Vec<f64>>,
    #[serde(default)]
    pub c_ordinarydiffeq: Option<Vec<f64>>,
    #[serde(default)]
    pub d: Option<DCoefficients>,
    #[serde(default)]
    pub gamma: Option<f64>,
    #[serde(default)]
    pub c32: Option<f64>,
    #[serde(default)]
    pub alpha: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pub beta: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pub embedded: Option<ShuOsherEmbedded>,
    #[serde(default)]
    pub residual_control: Option<ResidualControl>,
    #[serde(default)]
    pub divided_difference: Option<DividedDifference>,
    #[serde(default)]
    pub kpp: Option<Kpp>,
    pub controller: Option<ControllerConstants>,
    pub source: serde_json::Value,
    #[serde(default)]
    pub notes: Vec<String>,
}

macro_rules! tableau_sources {
    ($($name:literal),* $(,)?) => {
        /// `(alg, JSON text)` for every file in `spec/tableaus/`.
        const SOURCES: &[(&str, &str)] = &[
            $(($name, include_str!(concat!("../../../spec/tableaus/", $name, ".json")))),*
        ];
    };
}

tableau_sources!(
    "BS3",
    "BackwardEuler",
    "ForwardEuler",
    "Heun",
    "IMEXEuler",
    "KPP_Rodas3",
    "KPP_Rodas4",
    "KPP_Ros2",
    "KPP_Ros3",
    "KPP_Ros4",
    "Kvaerno3",
    "Kvaerno4",
    "RK4",
    "ROS2",
    "ROS3P",
    "Rodas3",
    "Rodas4",
    "Rodas5P",
    "Rosenbrock23",
    "SDIRK2",
    "SSPRK104",
    "SSPRK33",
    "SSPRK43",
    "TRBDF2",
    "Theta",
    "Tsit5",
    "WRF_RK3",
);

/// Number of embedded tables.
pub const N_TABLEAUS: usize = SOURCES.len();

fn registry() -> &'static BTreeMap<&'static str, Tableau> {
    static REG: OnceLock<BTreeMap<&'static str, Tableau>> = OnceLock::new();
    REG.get_or_init(|| {
        let mut m = BTreeMap::new();
        for (name, text) in SOURCES {
            let t: Tableau = serde_json::from_str(text)
                .unwrap_or_else(|e| panic!("spec/tableaus/{name}.json does not parse: {e}"));
            assert_eq!(
                &t.name, name,
                "spec/tableaus/{name}.json names itself {}",
                t.name
            );
            m.insert(*name, t);
        }
        m
    })
}

/// The table for run-config `alg` value `name`, if there is one.
pub fn tableau(name: &str) -> Option<&'static Tableau> {
    registry().get(name)
}

/// Every table, by name.
pub fn all_tableaus() -> impl Iterator<Item = &'static Tableau> {
    registry().values()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_tables_load() {
        assert_eq!(N_TABLEAUS, 27);
        assert_eq!(all_tableaus().count(), 27);
        for t in all_tableaus() {
            assert!(t.schema.ends_with("tableau.schema.json"));
            if t.error_estimate == ErrorEstimate::None {
                assert!(t.controller.is_none(), "{}", t.name);
            } else {
                assert!(t.controller.is_some(), "{}", t.name);
            }
            match t.form {
                Form::Butcher | Form::Sdirk => {
                    let a = t.a.as_ref().unwrap();
                    assert_eq!(a.len(), t.stages, "{}", t.name);
                    assert_eq!(t.b.as_ref().unwrap().len(), t.stages);
                    assert_eq!(t.c.as_ref().unwrap().len(), t.stages);
                }
                Form::ShuOsher => {
                    let alpha = t.alpha.as_ref().unwrap();
                    assert_eq!(alpha.len(), t.stages, "{}", t.name);
                    for (i, row) in alpha.iter().enumerate() {
                        assert_eq!(row.len(), i + 1);
                    }
                }
                _ => {}
            }
        }
    }

    #[test]
    fn explicit_tables_are_consistent() {
        for t in all_tableaus().filter(|t| t.form == Form::Butcher) {
            let a = t.a.as_ref().unwrap();
            let c = t.c.as_ref().unwrap();
            for (i, row) in a.iter().enumerate() {
                for (j, aij) in row.iter().enumerate() {
                    if j >= i {
                        assert_eq!(*aij, 0.0, "{} is not explicit", t.name);
                    }
                }
                let rowsum: f64 = row.iter().sum();
                assert!((rowsum - c[i]).abs() < 1e-14, "{} row {i}", t.name);
            }
        }
    }
}
