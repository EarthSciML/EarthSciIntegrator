//! The M1 conformance cases (spec/toy-problems.md §2–§3): parsing, the
//! checks Rust runs, and result files.
//!
//! `model: "toy:..."` is accepted only through this module, the test
//! harness's entry point (spec/toy-problems.md §2).

use crate::config::RunConfig;
use crate::schedule::{ForwardRun, RunSpec};
use crate::tape::Tape;
use crate::toy::{forced_decay_exact, source_decay_map_exact};
use crate::Error;
use serde::Deserialize;
use serde_json::{json, Value};
use std::fmt;
use std::path::{Path, PathBuf};

/// `tests/conformance/m1/cases.json`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CasesFile {
    pub esm_integrator: String,
    #[serde(default)]
    pub generated_by: Option<String>,
    pub cases: Vec<Case>,
}

/// One conformance case.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub id: String,
    pub run_config: Value,
    pub tspan: [f64; 2],
    #[serde(default)]
    pub saveat: Option<Vec<f64>>,
    pub checks: Vec<Check>,
}

/// A check of spec/toy-problems.md §2.1.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "check", rename_all = "snake_case", deny_unknown_fields)]
pub enum Check {
    Order {
        vary: String,
        dts: Vec<f64>,
        expected: f64,
        #[serde(default)]
        against: Option<String>,
    },
    Oracle {
        rtol: f64,
    },
    Invariant {
        rtol: f64,
    },
    ExactDiscrete {
        rtol: f64,
    },
    Reference {
        rtol: f64,
    },
    DotProduct {
        rtol: f64,
    },
    FdGradient {
        rtol: f64,
        epsilons: Vec<f64>,
    },
    ReverseDiff {
        rtol: f64,
    },
    Replay,
    CrossLanguage {
        rtol: f64,
        #[serde(default)]
        rtol_steps_differ: Option<f64>,
    },
}

impl Check {
    /// The check's name in cases.json.
    pub fn name(&self) -> &'static str {
        match self {
            Check::Order { .. } => "order",
            Check::Oracle { .. } => "oracle",
            Check::Invariant { .. } => "invariant",
            Check::ExactDiscrete { .. } => "exact_discrete",
            Check::Reference { .. } => "reference",
            Check::DotProduct { .. } => "dot_product",
            Check::FdGradient { .. } => "fd_gradient",
            Check::ReverseDiff { .. } => "reverse_diff",
            Check::Replay => "replay",
            Check::CrossLanguage { .. } => "cross_language",
        }
    }
}

/// The outcome of one check.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// Passed, with the measured quantity.
    Pass(String),
    /// Failed, with the measured quantity and the threshold.
    Fail(String),
    /// Not run, with the reason.
    Skip(String),
}

/// What happened to one case.
#[derive(Debug, Clone)]
pub struct CaseReport {
    pub id: String,
    /// `None` when the case ran; otherwise why it was skipped.
    pub skipped: Option<String>,
    /// A failure before any check could run.
    pub error: Option<String>,
    pub checks: Vec<(&'static str, Outcome)>,
    pub result_file: Option<PathBuf>,
}

impl fmt::Display for CaseReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(r) = &self.skipped {
            return writeln!(f, "SKIP  {}: {r}", self.id);
        }
        if let Some(e) = &self.error {
            return writeln!(f, "FAIL  {}: {e}", self.id);
        }
        writeln!(f, "RUN   {}", self.id)?;
        for (name, o) in &self.checks {
            match o {
                Outcome::Pass(m) => writeln!(f, "      pass  {name}: {m}")?,
                Outcome::Fail(m) => writeln!(f, "      FAIL  {name}: {m}")?,
                Outcome::Skip(m) => writeln!(f, "      skip  {name}: {m}")?,
            }
        }
        Ok(())
    }
}

/// Paths the harness reads and writes.
#[derive(Debug, Clone)]
pub struct Paths {
    /// `tests/conformance/m1/reference/`.
    pub reference_dir: PathBuf,
    /// Where result files go.
    pub result_dir: PathBuf,
    /// `spec/schemas/result.schema.json`.
    pub result_schema: PathBuf,
}

const JULIA_ONLY: &str = "Julia only (spec/toy-problems.md §2.1, D44)";

/// Runs one case and its checks.
pub fn run_case(case: &Case, paths: &Paths) -> CaseReport {
    let mut rep = CaseReport {
        id: case.id.clone(),
        skipped: None,
        error: None,
        checks: Vec::new(),
        result_file: None,
    };
    let tspan = (case.tspan[0], case.tspan[1]);
    let rc = match RunConfig::from_value(&case.run_config) {
        Ok(rc) => rc,
        Err(e) => {
            rep.error = Some(e.to_string());
            return rep;
        }
    };
    let spec = match RunSpec::<f64>::new(&rc, tspan, case.saveat.as_deref()) {
        Ok(s) => s,
        Err(Error::Unimplemented(r)) => {
            rep.skipped = Some(r);
            return rep;
        }
        Err(e) => {
            rep.error = Some(e.to_string());
            return rep;
        }
    };
    let u0 = spec.u0();
    let p = spec.problem.p.clone();
    let fwd = match spec.forward(&u0, &p) {
        Ok(f) => f,
        Err(e) => {
            rep.error = Some(e.to_string());
            return rep;
        }
    };
    let grad = if spec.functional.is_some() {
        match spec.gradient(&fwd, &p) {
            Ok(g) => Some(g),
            Err(e) => {
                rep.error = Some(format!("adjoint: {e}"));
                return rep;
            }
        }
    } else {
        None
    };

    // The result file (spec/toy-problems.md §3).
    let mut res = json!({
        "case": case.id,
        "producer": "rust",
        "t": fwd.outputs.iter().map(|o| o.0).collect::<Vec<_>>(),
        "u": fwd.outputs.iter().map(|o| o.1.clone()).collect::<Vec<_>>(),
        "stats": {"accepted": fwd.accepted, "rejected": fwd.rejected, "rhs_calls": fwd.rhs_calls},
    });
    if let Some(g) = &grad {
        res["functional"] = json!(g.functional);
        res["gradient"] = json!({"u0": g.u0, "p": g.p});
    }
    match write_result(&res, &case.id, paths) {
        Ok(path) => {
            // The tape beside it, for diagnosis (spec/tape.md §5).
            let _ = std::fs::write(path.with_extension("tape.jsonl"), fwd.tape.to_jsonl());
            rep.result_file = Some(path)
        }
        Err(e) => {
            rep.error = Some(format!("result file: {e}"));
            return rep;
        }
    }

    for check in &case.checks {
        let o = match check {
            Check::Oracle { .. } | Check::DotProduct { .. } | Check::ReverseDiff { .. } => {
                Outcome::Skip(JULIA_ONLY.into())
            }
            Check::CrossLanguage { .. } => Outcome::Skip(
                "compared with Julia's result in CI (compare_results.py); the result file is written".into(),
            ),
            Check::Order {
                vary,
                dts,
                expected,
                against,
            } => check_order(case, &rc, vary, dts, *expected, against.as_deref(), paths),
            Check::Invariant { rtol } => check_invariant(&spec, &fwd, *rtol),
            Check::ExactDiscrete { rtol } => check_exact_discrete(&spec, &fwd, *rtol),
            Check::Reference { rtol } => check_reference(case, &fwd, *rtol, paths),
            Check::FdGradient { rtol, epsilons } => match &grad {
                Some(g) => check_fd(&spec, &fwd, g, *rtol, epsilons),
                None => Outcome::Fail("fd_gradient needs an adjoint functional".into()),
            },
            Check::Replay => check_replay(&spec, &fwd, &u0, &p),
        };
        rep.checks.push((check.name(), o));
    }
    rep
}

fn write_result(res: &Value, id: &str, paths: &Paths) -> Result<PathBuf, String> {
    let schema: Value = serde_json::from_str(
        &std::fs::read_to_string(&paths.result_schema).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    validate(res, &schema, &schema, "$")?;
    std::fs::create_dir_all(&paths.result_dir).map_err(|e| e.to_string())?;
    let path = paths.result_dir.join(format!("{id}.json"));
    // serde_json writes every f64 in shortest round-trip form.
    std::fs::write(&path, serde_json::to_string(res).unwrap() + "\n").map_err(|e| e.to_string())?;
    Ok(path)
}

/// A JSON Schema (2020-12) validator for the keywords the result schema uses:
/// `type`, `required`, `properties`, `additionalProperties`, `items`,
/// `minItems`, `enum`, `const`, `minimum`, `pattern` (as a character-class
/// check of `^[A-Za-z0-9_.-]+$` only) and `dependentRequired`.
pub fn validate(v: &Value, s: &Value, root: &Value, at: &str) -> Result<(), String> {
    let err = |m: String| Err(format!("{at}: {m}"));
    let Some(so) = s.as_object() else {
        return Ok(());
    };
    for (k, kw) in so {
        match k.as_str() {
            "type" => {
                let ok = match kw.as_str().unwrap_or("") {
                    "object" => v.is_object(),
                    "array" => v.is_array(),
                    "number" => v.is_number(),
                    "integer" => v.is_u64() || v.is_i64(),
                    "string" => v.is_string(),
                    "boolean" => v.is_boolean(),
                    t => return err(format!("unsupported type {t}")),
                };
                if !ok {
                    return err(format!("not of type {kw}"));
                }
            }
            "required" => {
                for r in kw.as_array().into_iter().flatten() {
                    let r = r.as_str().unwrap_or("");
                    if v.get(r).is_none() {
                        return err(format!("missing required {r}"));
                    }
                }
            }
            "properties" => {
                if let (Some(vo), Some(props)) = (v.as_object(), kw.as_object()) {
                    for (pk, ps) in props {
                        if let Some(pv) = vo.get(pk) {
                            validate(pv, ps, root, &format!("{at}.{pk}"))?;
                        }
                    }
                }
            }
            "additionalProperties" => {
                if kw == &Value::Bool(false) {
                    if let Some(vo) = v.as_object() {
                        let props = so.get("properties").and_then(|p| p.as_object());
                        for key in vo.keys() {
                            if !props.is_some_and(|p| p.contains_key(key)) {
                                return err(format!("additional property {key}"));
                            }
                        }
                    }
                }
            }
            "items" => {
                for (i, it) in v.as_array().into_iter().flatten().enumerate() {
                    validate(it, kw, root, &format!("{at}[{i}]"))?;
                }
            }
            "minItems" => {
                if let Some(a) = v.as_array() {
                    if (a.len() as u64) < kw.as_u64().unwrap_or(0) {
                        return err(format!("fewer than {kw} items"));
                    }
                }
            }
            "enum" => {
                if !kw.as_array().is_some_and(|a| a.contains(v)) {
                    return err(format!("{v} not in {kw}"));
                }
            }
            "const" => {
                if kw != v {
                    return err(format!("{v} is not {kw}"));
                }
            }
            "minimum" => {
                if let (Some(x), Some(m)) = (v.as_f64(), kw.as_f64()) {
                    if x < m {
                        return err(format!("{x} below {m}"));
                    }
                }
            }
            "pattern" => {
                if kw.as_str() != Some("^[A-Za-z0-9_.-]+$") {
                    return err(format!("unsupported pattern {kw}"));
                }
                let ok = v.as_str().is_some_and(|s| {
                    !s.is_empty()
                        && s.chars()
                            .all(|c| c.is_ascii_alphanumeric() || "_.-".contains(c))
                });
                if !ok {
                    return err(format!("{v} does not match {kw}"));
                }
            }
            "dependentRequired" => {
                if let (Some(vo), Some(dr)) = (v.as_object(), kw.as_object()) {
                    for (key, deps) in dr {
                        if vo.contains_key(key) {
                            for d in deps.as_array().into_iter().flatten() {
                                if !vo.contains_key(d.as_str().unwrap_or("")) {
                                    return err(format!("{key} requires {d}"));
                                }
                            }
                        }
                    }
                }
            }
            "$schema" | "$id" | "title" | "description" => {}
            other => return err(format!("unsupported schema keyword {other}")),
        }
    }
    let _ = root;
    Ok(())
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
}

fn fmt_e(x: f64) -> String {
    format!("{x:.3e}")
}

/// Replaces every leaf's fixed `dt` (vary = "leaf") or the macro step
/// (vary = "macro") in a run config.
fn with_dt(rc: &Value, vary: &str, dt: f64) -> Result<Value, String> {
    let mut v = rc.clone();
    fn leaves(n: &mut Value, dt: f64) {
        if let Some(o) = n.as_object_mut() {
            if o.contains_key("leaf") {
                o.insert("dt".into(), json!(dt));
                return;
            }
            for k in ["sequence", "strang"] {
                if let Some(Value::Array(a)) = o.get_mut(k) {
                    for c in a {
                        leaves(c, dt);
                    }
                }
            }
            for k in ["subcycle", "every"] {
                if let Some(c) = o.get_mut(k) {
                    leaves(c, dt);
                }
            }
        }
    }
    match vary {
        "leaf" => leaves(&mut v["schedule"], dt),
        "macro" => v["step"]["dt"] = json!(dt),
        o => return Err(format!("unknown vary {o:?}")),
    }
    Ok(v)
}

/// A reference result: output times and states.
type Reference = (Vec<f64>, Vec<Vec<f64>>);

fn load_reference(case_id: &str, paths: &Paths) -> Option<Result<Reference, String>> {
    let path = paths.reference_dir.join(format!("{case_id}.json"));
    if !path.exists() {
        return None;
    }
    let parse = || -> Result<Reference, String> {
        let v: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        let t: Vec<f64> = serde_json::from_value(v["t"].clone()).map_err(|e| e.to_string())?;
        let u: Vec<Vec<f64>> = serde_json::from_value(v["u"].clone()).map_err(|e| e.to_string())?;
        Ok((t, u))
    };
    Some(parse())
}

fn check_order(
    case: &Case,
    rc: &RunConfig,
    vary: &str,
    dts: &[f64],
    expected: f64,
    against: Option<&str>,
    paths: &Paths,
) -> Outcome {
    let _ = rc;
    let tspan = (case.tspan[0], case.tspan[1]);
    // The comparison state at the end of the interval.
    let target: Vec<f64> = match against.unwrap_or("exact") {
        "exact" => {
            if case.run_config["model"].as_str() != Some("toy:forced_decay") {
                return Outcome::Fail("no exact solution for this problem".into());
            }
            let pb = crate::toy::toy_problem::<f64>("toy:forced_decay").unwrap();
            vec![forced_decay_exact(tspan.1, pb.u0[0], &pb.p)]
        }
        "reference" => match load_reference(&case.id, paths) {
            None => {
                return Outcome::Skip(format!(
                "reference file {}.json is not present (the Julia M1 implementation produces it)",
                case.id
            ))
            }
            Some(Err(e)) => return Outcome::Fail(format!("reference file: {e}")),
            Some(Ok((t, u))) => match t.iter().position(|&x| x == tspan.1) {
                Some(i) => u[i].clone(),
                None => return Outcome::Fail("reference has no state at the interval end".into()),
            },
        },
        o => return Outcome::Fail(format!("unknown against {o:?}")),
    };
    let mut errs = Vec::new();
    for &dt in dts {
        let v = match with_dt(&case.run_config, vary, dt) {
            Ok(v) => v,
            Err(e) => return Outcome::Fail(e),
        };
        let run = RunConfig::from_value(&v)
            .and_then(|rc| RunSpec::<f64>::new(&rc, tspan, None))
            .and_then(|s| {
                let u0 = s.u0();
                s.forward(&u0, &s.problem.p.clone())
            });
        let run = match run {
            Ok(r) => r,
            Err(e) => return Outcome::Fail(format!("dt {dt}: {e}")),
        };
        let u = run.final_state();
        let e = u
            .iter()
            .zip(&target)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        errs.push(e);
    }
    let xs: Vec<f64> = dts.iter().map(|d| d.ln()).collect();
    let ys: Vec<f64> = errs.iter().map(|e| e.ln()).collect();
    let n = xs.len() as f64;
    let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
    let slope = sxy / sxx;
    let msg = format!(
        "observed order {slope:.3} (threshold {:.2}); errors {}",
        expected - 0.3,
        errs.iter()
            .map(|e| fmt_e(*e))
            .collect::<Vec<_>>()
            .join(", ")
    );
    if slope >= expected - 0.3 {
        Outcome::Pass(msg)
    } else {
        Outcome::Fail(msg)
    }
}

pub fn check_invariant(spec: &RunSpec<f64>, fwd: &ForwardRun<f64>, rtol: f64) -> Outcome {
    if spec.problem.invariants.is_empty() {
        return Outcome::Fail(format!("{} has no invariants", spec.problem.name));
    }
    let u0 = spec.u0();
    let mut worst = 0.0f64;
    for inv in &spec.problem.invariants {
        let val = |u: &[f64]| inv.iter().map(|&(j, w)| w * u[j]).sum::<f64>();
        let i0 = val(&u0);
        for (_, u) in &fwd.outputs {
            worst = worst.max(rel(val(u), i0));
        }
    }
    let msg = format!("max relative drift {} (rtol {rtol:e})", fmt_e(worst));
    if worst <= rtol {
        Outcome::Pass(msg)
    } else {
        Outcome::Fail(msg)
    }
}

fn check_exact_discrete(spec: &RunSpec<f64>, fwd: &ForwardRun<f64>, rtol: f64) -> Outcome {
    if spec.problem.name != "source_decay_map" {
        return Outcome::Fail(format!(
            "no exact discrete solution for {}",
            spec.problem.name
        ));
    }
    let n = fwd.checkpoints.len() - 1;
    let exact = source_decay_map_exact(spec.problem.u0[0], &spec.problem.p, spec.macro_dt, n);
    let got = fwd.final_state()[0];
    let e = rel(got, exact);
    let msg = format!(
        "end state {got} vs exact {exact}: relative error {} (rtol {rtol:e})",
        fmt_e(e)
    );
    if e <= rtol {
        Outcome::Pass(msg)
    } else {
        Outcome::Fail(msg)
    }
}

fn check_reference(case: &Case, fwd: &ForwardRun<f64>, rtol: f64, paths: &Paths) -> Outcome {
    let (t, u) = match load_reference(&case.id, paths) {
        None => {
            return Outcome::Skip(format!(
                "reference file {}.json is not present (the Julia M1 implementation produces it)",
                case.id
            ))
        }
        Some(Err(e)) => return Outcome::Fail(format!("reference file: {e}")),
        Some(Ok(r)) => r,
    };
    let ncomp = u.first().map_or(0, |r| r.len());
    let scale: Vec<f64> = (0..ncomp)
        .map(|j| u.iter().map(|r| r[j].abs()).fold(0.0, f64::max))
        .collect();
    let mut worst = 0.0f64;
    for (tr, ur) in t.iter().zip(&u) {
        let Some((_, ours)) = fwd.outputs.iter().find(|o| o.0 == *tr) else {
            return Outcome::Fail(format!("no output at reference time {tr}"));
        };
        for j in 0..ncomp {
            let d = (ours[j] - ur[j]).abs() / if scale[j] > 0.0 { scale[j] } else { 1.0 };
            worst = worst.max(d);
        }
    }
    let msg = format!("max scaled difference {} (rtol {rtol:e})", fmt_e(worst));
    if worst <= rtol {
        Outcome::Pass(msg)
    } else {
        Outcome::Fail(msg)
    }
}

/// spec/adjoint.md §7.2: directional derivatives `⟨∇J, δ⟩` against central
/// differences of the frozen-tape map `(J(x + εδ) − J(x − εδ))/2ε`, at every
/// epsilon. The directions are: a pseudo-random one over every requested
/// input, one over `u0` only, and each requested parameter alone. A
/// direction's entry for input `x` is `r·|x|` (`r` in `[−1, 1]`), or `r` when
/// `x = 0`, so inputs of very different size are perturbed comparably. A
/// direction passes when its smallest relative error over the epsilons is
/// within `rtol`.
pub fn check_fd(
    spec: &RunSpec<f64>,
    fwd: &ForwardRun<f64>,
    g: &crate::adjoint::Gradient,
    rtol: f64,
    epsilons: &[f64],
) -> Outcome {
    let u0 = spec.u0();
    let p = spec.problem.p.clone();
    let (nu, np) = (u0.len(), p.len());
    let jfun = |u: &[f64], pp: &[f64]| -> Result<f64, Error> {
        let r = spec.replay(u, pp, &fwd.tape)?;
        spec.functional_value(&r)
    };
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let scale = |x: f64| if x != 0.0 { x.abs() } else { 1.0 };
    // Directions as (label, δu0, δp).
    let mut dirs: Vec<(String, Vec<f64>, Vec<f64>)> = Vec::new();
    let du: Vec<f64> = u0
        .iter()
        .map(|&x| if spec.wrt_u0 { rnd() * scale(x) } else { 0.0 })
        .collect();
    let dp: Vec<f64> = p
        .iter()
        .zip(&spec.wrt_p)
        .map(|(&x, &w)| if w { rnd() * scale(x) } else { 0.0 })
        .collect();
    dirs.push(("all inputs".into(), du.clone(), dp));
    if spec.wrt_u0 {
        dirs.push(("u0".into(), du, vec![0.0; np]));
    }
    for (i, &w) in spec.wrt_p.iter().enumerate() {
        if w {
            let mut e = vec![0.0; np];
            e[i] = scale(p[i]);
            dirs.push((
                format!("p[{}]", spec.problem.param_names[i]),
                vec![0.0; nu],
                e,
            ));
        }
    }
    let mut worst = 0.0f64;
    let mut worst_label = String::new();
    let mut trend = String::new();
    for (label, du, dp) in &dirs {
        let exact: f64 = g.u0.iter().zip(du).map(|(a, b)| a * b).sum::<f64>()
            + g.p.iter().zip(dp).map(|(a, b)| a * b).sum::<f64>();
        let mut best = f64::INFINITY;
        let mut errs = Vec::new();
        for &eps in epsilons {
            let shift = |s: f64| -> (Vec<f64>, Vec<f64>) {
                (
                    u0.iter().zip(du).map(|(x, d)| x + s * d).collect(),
                    p.iter().zip(dp).map(|(x, d)| x + s * d).collect(),
                )
            };
            let (up, pp) = shift(eps);
            let (um, pm) = shift(-eps);
            let (jp, jm) = match (jfun(&up, &pp), jfun(&um, &pm)) {
                (Ok(a), Ok(b)) => (a, b),
                (Err(e), _) | (_, Err(e)) => return Outcome::Fail(format!("{label}: replay: {e}")),
            };
            let fd = (jp - jm) / (2.0 * eps);
            let e = (fd - exact).abs() / exact.abs().max(f64::MIN_POSITIVE);
            errs.push(e);
            best = best.min(e);
        }
        if best >= worst {
            worst = best;
            worst_label = label.clone();
            trend = errs
                .iter()
                .map(|e| fmt_e(*e))
                .collect::<Vec<_>>()
                .join(" → ");
        }
    }
    let msg = format!(
        "{} directions; worst {worst_label}: relative error {} at the best epsilon (rtol {rtol:e}; over epsilons {trend})",
        dirs.len(),
        fmt_e(worst)
    );
    if worst <= rtol {
        Outcome::Pass(msg)
    } else {
        Outcome::Fail(msg)
    }
}

/// spec/toy-problems.md §2.1 `replay` and spec/adjoint.md §7.4: the tape
/// round-trips through JSON Lines bitwise; replaying from the start with the
/// tape reproduces every macro-step state, every output and the end state
/// bitwise; and replaying each macro step from its checkpoint with the
/// controller on reproduces its tape records and the next checkpoint bitwise.
pub fn check_replay(spec: &RunSpec<f64>, fwd: &ForwardRun<f64>, u0: &[f64], p: &[f64]) -> Outcome {
    let back = match Tape::from_jsonl(&fwd.tape.to_jsonl()) {
        Ok(t) => t,
        Err(e) => return Outcome::Fail(format!("tape does not read back: {e}")),
    };
    if back.records.len() != fwd.tape.records.len()
        || !back
            .records
            .iter()
            .zip(&fwd.tape.records)
            .all(|(a, b)| a.bitwise_eq(b))
    {
        return Outcome::Fail("the JSON Lines tape does not read back bitwise".into());
    }
    let rep = match spec.replay(u0, p, &back) {
        Ok(r) => r,
        Err(e) => return Outcome::Fail(format!("replay: {e}")),
    };
    let same = |a: &[f64], b: &[f64]| {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    };
    if rep.tape.records.len() != fwd.tape.records.len() {
        return Outcome::Fail("replay produced a different number of invocations".into());
    }
    for (a, b) in rep.tape.records.iter().zip(&fwd.tape.records) {
        if !a.bitwise_eq(b) {
            return Outcome::Fail(format!("replay differs at {}", a.path()));
        }
    }
    for (a, b) in rep.checkpoints.iter().zip(&fwd.checkpoints) {
        if a.t.to_bits() != b.t.to_bits() || !same(&a.u, &b.u) {
            return Outcome::Fail(format!(
                "replayed state differs at macro step {}",
                a.macro_index
            ));
        }
    }
    for (a, b) in rep.outputs.iter().zip(&fwd.outputs) {
        if a.0.to_bits() != b.0.to_bits() || !same(&a.1, &b.1) {
            return Outcome::Fail(format!("replayed output differs at t = {}", a.0));
        }
    }
    let index = fwd.tape.index();
    let nmacro = fwd.checkpoints.len() - 1;
    for k in 0..nmacro {
        let cp = &fwd.checkpoints[k];
        let (t0, delta) = match index.get(k.to_string().as_str()) {
            Some(crate::tape::TapeRecord::Macro { t, dt, .. }) => (*t, *dt),
            _ => return Outcome::Fail(format!("no macro record {k}")),
        };
        let t1 = fwd.checkpoints[k + 1].t;
        let mut u = cp.u.clone();
        let mut persist = cp.leaves.clone();
        let recs = match spec.macro_step(k, &mut u, &mut persist, p, t0, t1, delta, None, None) {
            Ok(((r, _), _)) => r,
            Err(e) => return Outcome::Fail(format!("controller-on replay of macro step {k}: {e}")),
        };
        let seg = fwd.tape.macro_segment(k);
        let seg = &seg[..seg.len() - 1];
        if recs.len() != seg.len() || !recs.iter().zip(seg).all(|(a, b)| a.bitwise_eq(b)) {
            return Outcome::Fail(format!(
                "controller-on replay of macro step {k} took different steps"
            ));
        }
        let next = crate::tape::Checkpoint {
            macro_index: k + 1,
            t: t1,
            u,
            leaves: persist,
        };
        if !next.bitwise_eq(&fwd.checkpoints[k + 1]) {
            return Outcome::Fail(format!(
                "controller-on replay of macro step {k} does not reproduce the next checkpoint bitwise"
            ));
        }
    }
    let (acc, _) = fwd.tape.step_counts();
    Outcome::Pass(format!(
        "{nmacro} macro steps, {acc} accepted steps, end state bitwise equal; checkpoints reproduced with the controller on"
    ))
}

/// Default paths relative to the repository root.
pub fn default_paths(repo_root: &Path, result_dir: PathBuf) -> Paths {
    Paths {
        reference_dir: repo_root.join("tests/conformance/m1/reference"),
        result_dir,
        result_schema: repo_root.join("spec/schemas/result.schema.json"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_schema_validation() {
        let schema: Value =
            serde_json::from_str(include_str!("../../../spec/schemas/result.schema.json")).unwrap();
        let good = json!({"case": "a.b", "producer": "rust", "t": [1.0], "u": [[1.0]],
                          "stats": {"accepted": 1, "rejected": 0}});
        validate(&good, &schema, &schema, "$").unwrap();
        let mut bad = good.clone();
        bad["extra"] = json!(1);
        assert!(validate(&bad, &schema, &schema, "$").is_err());
        let mut bad = good.clone();
        bad["functional"] = json!(1.0);
        assert!(validate(&bad, &schema, &schema, "$").is_err());
        let mut bad = good;
        bad["producer"] = json!("python");
        assert!(validate(&bad, &schema, &schema, "$").is_err());
    }
}
