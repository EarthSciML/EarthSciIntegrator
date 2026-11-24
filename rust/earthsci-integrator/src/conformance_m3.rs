//! The M3 cases (spec/toy-problems.md §5; spec/partitioning.md §9): plan a
//! case's run config for each of its `invariance` variants, run it, and
//! compare the runs bit for bit; compare a case with the case its
//! `scope_agreement` names.

use crate::conformance::Outcome;
use crate::conformance_m2::{Case, RunResult};
use crate::forcing::DataOptions;
use crate::planned::load_plan_with;
use crate::schedule::RunSpec;
use crate::tape::{Tape, TapeRecord};
use serde_json::{json, Value};
use std::path::Path;

/// One `invariance` variant: overrides of the pointwise leaves' partitions
/// and the thread count.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Variant {
    pub shards: Option<u64>,
    pub lanes: Option<u64>,
    pub threads: Option<usize>,
}

impl Variant {
    pub fn from_json(v: &Value) -> Variant {
        Variant {
            shards: v["shards"].as_u64(),
            lanes: v["lanes"].as_u64(),
            threads: v["threads"].as_u64().map(|x| x as usize),
        }
    }

    /// The case's run config with every leaf that has a `partition` given
    /// this variant's `shards` and `lanes`.
    pub fn run_config(&self, rc: &Value) -> Value {
        fn walk(v: &mut Value, var: &Variant) {
            match v {
                Value::Object(o) => {
                    if o.contains_key("leaf") && o.contains_key("partition") {
                        let p = o.get_mut("partition").expect("checked");
                        if let Some(s) = var.shards {
                            p["shards"] = json!(s);
                        }
                        if let Some(l) = var.lanes {
                            p["lanes"] = json!(l);
                        }
                    }
                    for x in o.values_mut() {
                        walk(x, var);
                    }
                }
                Value::Array(a) => a.iter_mut().for_each(|x| walk(x, var)),
                _ => {}
            }
        }
        let mut out = rc.clone();
        walk(&mut out, self);
        out
    }
}

/// [`crate::conformance_m2::run_plan_with`] on `threads` executor threads,
/// returning also the run's per-unit counts: every scoped leaf record's
/// per-cell or per-shard accepted and rejected counts, in tape order.
pub fn run_plan_threads(
    case: &Case,
    plan_dir: &Path,
    data: &DataOptions,
    threads: usize,
) -> Result<RunResult, String> {
    run_plan_with_options(case, plan_dir, data, threads, false)
}

/// [`run_plan_threads`], with `whole_state` the oracle of
/// spec/partitioning.md §9.3: scoped leaves evaluate their operators over
/// the whole model state rather than through lane programs.
pub fn run_plan_with_options(
    case: &Case,
    plan_dir: &Path,
    data: &DataOptions,
    threads: usize,
    whole_state: bool,
) -> Result<RunResult, String> {
    let lp = load_plan_with::<f64>(plan_dir, data).map_err(|e| e.to_string())?;
    let tspan = (case.tspan[0], case.tspan[1]);
    let mut order: Vec<usize> = Vec::new();
    for v in &case.layout {
        order.extend(
            lp.state_dofs
                .get(v)
                .ok_or_else(|| format!("layout variable {v} is not a state of the plan"))?,
        );
    }
    let spec = RunSpec::<f64>::with_problem_and_tstops(
        lp.problem,
        &lp.run_config,
        tspan,
        case.saveat.as_deref(),
        lp.timeline.refresh_times(),
    )
    .map_err(|e| e.to_string())?;
    spec.set_threads(threads);
    spec.set_whole_state_units(whole_state);
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

/// Every scoped leaf record's units, as `(accepted, rejected)` per cell or
/// shard, with the shards' cells, in tape order; and every unscoped leaf
/// record's steps.
pub fn unit_counts(tape: &Tape) -> Vec<Value> {
    tape.records
        .iter()
        .filter_map(|r| match r {
            TapeRecord::Leaf {
                path,
                cells,
                shards,
                steps,
                accepted,
                rejected,
                retcode,
                ..
            } => Some(json!({
                "path": path,
                "cells": cells,
                "shards": shards,
                "steps": steps.iter().map(|s| [s[0].to_bits(), s[1].to_bits()]).collect::<Vec<_>>(),
                "accepted": accepted,
                "rejected": rejected,
                "retcode": format!("{retcode:?}"),
            })),
            _ => None,
        })
        .collect()
}

/// Whether two runs are the same bit for bit: output times and states, and
/// every leaf record's counts and steps.
pub fn bitwise_same(a: &RunResult, b: &RunResult) -> Result<(), String> {
    let bits = |r: &RunResult| -> Vec<Vec<u64>> {
        r.result["u"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|row| {
                row.as_array()
                    .into_iter()
                    .flatten()
                    .map(|x| x.as_f64().unwrap_or(f64::NAN).to_bits())
                    .collect()
            })
            .collect()
    };
    if a.result["t"] != b.result["t"] {
        return Err("output times differ".into());
    }
    let (ua, ub) = (bits(a), bits(b));
    for (k, (ra, rb)) in ua.iter().zip(&ub).enumerate() {
        if let Some(j) = ra.iter().zip(rb).position(|(x, y)| x != y) {
            return Err(format!(
                "output {k}, element {j}: {} against {}",
                f64::from_bits(ra[j]),
                f64::from_bits(rb[j])
            ));
        }
    }
    let (ca, cb) = (unit_counts(&a.tape), unit_counts(&b.tape));
    if ca.len() != cb.len() {
        return Err(format!("{} leaf records against {}", ca.len(), cb.len()));
    }
    for (x, y) in ca.iter().zip(&cb) {
        // Shard records carry their cells; under dt_scope cell the shards
        // are only the grouping of work, so they are not compared.
        let strip = |v: &Value| {
            let mut v = v.clone();
            if v["shards"].is_null() {
                v.as_object_mut().map(|o| o.remove("shards"));
            }
            v
        };
        if strip(x) != strip(y) {
            return Err(format!(
                "leaf record {} differs: {x} against {y}",
                x["path"]
            ));
        }
    }
    Ok(())
}

/// `scope_agreement`: states agree to `rtol`, relative to each component's
/// largest magnitude over the run in either result.
pub fn scope_agreement(a: &RunResult, b: &RunResult, rtol: f64) -> Outcome {
    let rows = |r: &RunResult| -> Vec<Vec<f64>> {
        r.result["u"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|row| {
                row.as_array()
                    .into_iter()
                    .flatten()
                    .map(|x| x.as_f64().unwrap_or(f64::NAN))
                    .collect()
            })
            .collect()
    };
    let (ra, rb) = (rows(a), rows(b));
    if ra.len() != rb.len() {
        return Outcome::Fail(format!("{} outputs against {}", ra.len(), rb.len()));
    }
    let n = ra.first().map_or(0, Vec::len);
    let mut worst = 0.0f64;
    for j in 0..n {
        let scale = ra
            .iter()
            .chain(&rb)
            .map(|r| r[j].abs())
            .fold(0.0, f64::max)
            .max(f64::MIN_POSITIVE);
        for (x, y) in ra.iter().zip(&rb) {
            let d = (x[j] - y[j]).abs() / scale;
            if d.is_nan() {
                return Outcome::Fail(format!("element {j} is not finite"));
            }
            worst = worst.max(d);
        }
    }
    if worst <= rtol {
        Outcome::Pass(format!("worst relative difference {worst:e} <= {rtol:e}"))
    } else {
        Outcome::Fail(format!("worst relative difference {worst:e} > {rtol:e}"))
    }
}
