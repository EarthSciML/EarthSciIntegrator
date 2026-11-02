//! The M2 end-to-end cases (spec/toy-problems.md §4): running a case's plan
//! and the `toy_equivalence` check (§4.3).
//!
//! Planning is the planner's (`earthsci-integrator-planner`); this module
//! takes the plan directory it wrote, builds the operators from it
//! ([`crate::planned`]), runs the plan's schedule and writes the result file
//! with `u` in the case's `layout`.

use crate::conformance::{self, Outcome, Paths};
use crate::planned::load_plan;
use crate::schedule::RunSpec;
use crate::tape::{Tape, TapeRecord};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// `tests/conformance/m2/cases.json`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CasesFile {
    pub esm_integrator: String,
    #[serde(default)]
    pub generated_by: Option<String>,
    pub cases: Vec<Case>,
}

/// One M2 case.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub id: String,
    pub run_config: Value,
    pub tspan: [f64; 2],
    #[serde(default)]
    pub saveat: Option<Vec<f64>>,
    /// The variables of `u` in a result file, in order (§4.3).
    pub layout: Vec<String>,
    pub checks: Vec<Check>,
}

/// The check of an M2 case.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "check", rename_all = "snake_case", deny_unknown_fields)]
pub enum Check {
    ToyEquivalence {
        against: String,
        rtol: f64,
        #[serde(default)]
        rtol_steps_differ: Option<f64>,
    },
}

/// A run's result file (spec/toy-problems.md §3) and its tape.
#[derive(Debug, Clone)]
pub struct RunResult {
    pub result: Value,
    pub tape: Tape,
}

/// Builds the operators of the plan in `plan_dir`, runs its schedule over
/// the case's interval and output times, and returns the result with `u` in
/// the case's `layout`.
pub fn run_plan(case: &Case, plan_dir: &Path) -> Result<RunResult, String> {
    let lp = load_plan::<f64>(plan_dir).map_err(|e| e.to_string())?;
    let tspan = (case.tspan[0], case.tspan[1]);
    if lp.tspan != tspan {
        return Err(format!(
            "the plan's interval {:?} is not the case's {tspan:?}",
            lp.tspan
        ));
    }
    let mut order: Vec<usize> = Vec::new();
    for v in &case.layout {
        order.extend(
            lp.state_dofs
                .get(v)
                .ok_or_else(|| format!("layout variable {v} is not a state of the plan"))?,
        );
    }
    let spec =
        RunSpec::<f64>::with_problem(lp.problem, &lp.run_config, tspan, case.saveat.as_deref())
            .map_err(|e| e.to_string())?;
    let u0 = spec.u0();
    let p = spec.problem.p.clone();
    let fwd = spec.forward(&u0, &p).map_err(|e| e.to_string())?;
    let result = json!({
        "case": case.id,
        "producer": "rust",
        "t": fwd.outputs.iter().map(|o| o.0).collect::<Vec<_>>(),
        "u": fwd.outputs.iter().map(|o| order.iter().map(|&j| o.1[j]).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "stats": {"accepted": fwd.accepted, "rejected": fwd.rejected, "rhs_calls": fwd.rhs_calls},
    });
    Ok(RunResult {
        result,
        tape: fwd.tape,
    })
}

/// The forward run of an M1 case on its hand-written problem, as its
/// result file (states and statistics; no gradient) and tape.
pub fn run_m1_forward(case: &conformance::Case) -> Result<RunResult, String> {
    let rc = crate::config::RunConfig::from_value(&case.run_config).map_err(|e| e.to_string())?;
    let spec = RunSpec::<f64>::new(&rc, (case.tspan[0], case.tspan[1]), case.saveat.as_deref())
        .map_err(|e| e.to_string())?;
    let u0 = spec.u0();
    let p = spec.problem.p.clone();
    let fwd = spec.forward(&u0, &p).map_err(|e| e.to_string())?;
    let result = json!({
        "case": case.id,
        "producer": "rust",
        "t": fwd.outputs.iter().map(|o| o.0).collect::<Vec<_>>(),
        "u": fwd.outputs.iter().map(|o| o.1.clone()).collect::<Vec<_>>(),
        "stats": {"accepted": fwd.accepted, "rejected": fwd.rejected, "rhs_calls": fwd.rhs_calls},
    });
    Ok(RunResult {
        result,
        tape: fwd.tape,
    })
}

/// Validates a result against spec/schemas/result.schema.json and writes it,
/// with its tape beside it, to `<result_dir>/<case>.json`.
pub fn write_result(r: &RunResult, paths: &Paths) -> Result<PathBuf, String> {
    let schema: Value = serde_json::from_str(
        &std::fs::read_to_string(&paths.result_schema).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    conformance::validate(&r.result, &schema, &schema, "$")?;
    std::fs::create_dir_all(&paths.result_dir).map_err(|e| e.to_string())?;
    let id = r.result["case"].as_str().unwrap_or("unnamed");
    let path = paths.result_dir.join(format!("{id}.json"));
    std::fs::write(&path, serde_json::to_string(&r.result).unwrap() + "\n")
        .map_err(|e| e.to_string())?;
    std::fs::write(path.with_extension("tape.jsonl"), r.tape.to_jsonl())
        .map_err(|e| e.to_string())?;
    Ok(path)
}

/// Every leaf record's accepted steps `(t, h)`, in tape order.
fn steps_of(tape: &Tape) -> Vec<&[[f64; 2]]> {
    tape.records
        .iter()
        .filter_map(|r| match r {
            TapeRecord::Leaf { steps, .. } => Some(steps.as_slice()),
            _ => None,
        })
        .collect()
}

fn no_states(which: &str) -> String {
    format!(
        "the {which} result's t or u is missing or has a value that is not a finite number \
         (written null)"
    )
}

/// spec/toy-problems.md §4.3: the M2 result against the M1 result by the
/// rule of the `cross_language` check (§2.1). The step sequences differ when
/// the accepted counts do or when any step's `(t, h)` does; then states
/// agree to `rtol_steps_differ`, else to `rtol`. States are compared per
/// component relative to its largest magnitude over the run in either
/// result. Only states and step counts are compared.
pub fn toy_equivalence(
    m2: &RunResult,
    m1: &RunResult,
    rtol: f64,
    rtol_steps_differ: Option<f64>,
) -> Outcome {
    let (a, b) = (&m2.result, &m1.result);
    let acc = |r: &Value| r["stats"]["accepted"].as_u64().unwrap_or(u64::MAX);
    let (na, nb) = (acc(a), acc(b));
    let differ = na != nb || steps_of(&m2.tape) != steps_of(&m1.tape);
    let mut note = if na == nb {
        format!("accepted {na}")
    } else {
        format!("accepted M2 {na}, M1 {nb}")
    };
    let tol = if differ {
        note += ", step sequences differ";
        match rtol_steps_differ {
            Some(r) => r,
            None => {
                return Outcome::Fail(format!(
                    "step sequences differ and the case has no rtol_steps_differ [{note}]"
                ))
            }
        }
    } else {
        rtol
    };
    let parse = |r: &Value| -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
        Some((
            serde_json::from_value(r["t"].clone()).ok()?,
            serde_json::from_value(r["u"].clone()).ok()?,
        ))
    };
    // A non-finite value is written null in a result file, so it fails
    // here, on either side.
    let (ta, ua) = match parse(a) {
        Some(x) => x,
        None => return Outcome::Fail(no_states("M2")),
    };
    let (tb, ub) = match parse(b) {
        Some(x) => x,
        None => return Outcome::Fail(no_states("M1")),
    };
    if ta != tb {
        return Outcome::Fail(format!("output times differ: M2 {ta:?}, M1 {tb:?}"));
    }
    let ncomp = ub.first().map_or(0, Vec::len);
    if ua.iter().chain(&ub).any(|r| r.len() != ncomp) {
        return Outcome::Fail("state lengths differ".into());
    }
    let mut worst = (0.0f64, 0usize, 0usize);
    for j in 0..ncomp {
        let scale = ua.iter().chain(&ub).map(|r| r[j].abs()).fold(0.0, f64::max);
        for (k, (x, y)) in ua.iter().zip(&ub).enumerate() {
            let d = (x[j] - y[j]).abs() / if scale > 0.0 { scale } else { 1.0 };
            if d > worst.0 {
                worst = (d, k, j);
            }
        }
    }
    let msg = format!("{note}; states {:.2e} (rtol {tol:e})", worst.0);
    if worst.0 <= tol {
        Outcome::Pass(msg)
    } else {
        Outcome::Fail(format!(
            "state component {} at t = {}: relative difference {:.3e} > rtol {tol:e} [{msg}]",
            worst.2, ta[worst.1], worst.0
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(u: Vec<Vec<f64>>) -> RunResult {
        let t: Vec<f64> = (0..u.len()).map(|k| k as f64).collect();
        RunResult {
            result: serde_json::json!({"t": t, "u": u, "stats": {"accepted": 1}}),
            tape: Tape::default(),
        }
    }

    #[test]
    fn non_finite_values_fail_on_either_side() {
        let good = result(vec![vec![1.0, 2.0]]);
        let bad = result(vec![vec![1.0, f64::NAN]]);
        assert!(matches!(
            toy_equivalence(&good, &good, 1e-12, None),
            Outcome::Pass(_)
        ));
        for (m2, m1, which) in [(&bad, &good, "M2"), (&good, &bad, "M1")] {
            match toy_equivalence(m2, m1, 1e-12, None) {
                Outcome::Fail(m) => assert!(m.contains(which), "{m}"),
                o => panic!("{which}: {o:?}"),
            }
        }
    }
}
