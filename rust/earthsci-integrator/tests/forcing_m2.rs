//! The forcing epochs of a planned run (spec/operator-interface.md §8.4):
//! the adjoint takes, at every evaluation, the buffers its forward step took
//! (§8.4.1, D88; spec/adjoint.md §2.4, §3), and the providers' clock is
//! converted to model time whatever origin they chose (§8.4.2 item 2).
//!
//! The models are written here, into the target's directory; the data are
//! the monthly inventory files of the `data_monthly` toy
//! (spec/toy-problems.md §4.5), `E_{m,i} = 1e-6 (m + i)` per day for month
//! `m` and file cell `i`.

use earthsci_integrator::config::RunConfig;
use earthsci_integrator::conformance::{check_fd, check_replay, Outcome};
use earthsci_integrator::forcing::DataOptions;
use earthsci_integrator::planned::{load_plan_with, LoadedPlan};
use earthsci_integrator::schedule::RunSpec;
use earthsci_integrator_planner::astdiff::AstDiffBands;
use earthsci_integrator_planner::{plan, Options};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const DAY: f64 = 86400.0;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The data_monthly toy's inventory files, `inv_point_2016MM.nc`.
fn inventory_dir() -> PathBuf {
    repo()
        .join("tests/conformance/toys/data_monthly")
        .canonicalize()
        .unwrap()
}

/// The monthly inventory as a data source starting at `start`, its files
/// named by the month of the anchor (so a 1970 anchor reads the 2016 file
/// of the same month).
fn inventory_source(start: &str) -> Value {
    json!({
        "kind": "grid",
        "source": {"url_template": format!("{}/inv_point_2016{{date:%m}}.nc", inventory_dir().display())},
        "temporal": {
            "start": start, "frequency": "P1M", "file_period": "P1M",
            "records_per_file": 1, "time_variable": "time",
        },
        "metadata": {"esio_format": "netcdf"},
    })
}

/// `E` over the three model cells: the inventory's `E_day` per second, file
/// cells 2–4.
fn e_param() -> Value {
    json!({
        "type": "parameter", "units": "1/s", "shape": ["x"], "default": 1e-11,
        "update": {"kind": "data", "source": "Inventory", "from": {
            "file_variable": "E_day",
            "unit_conversion": {"op": "/", "args": [1.0, 86400.0]},
            "select": {"axes": [{"range": {"start": 1, "stop": 4}}]},
        }},
    })
}

/// `f(i)` over the cells `x`, as the `rhs` of `D(c)`.
fn over_cells(expr: Value) -> Value {
    json!({"op": "makearray", "args": [], "regions": [[[1, 3]]], "values": [
        {"op": "faq", "args": [], "output_idx": ["i"], "ranges": {"i": [1, 3]}, "expr": expr},
    ]})
}

fn ix(v: &str) -> Value {
    json!({"op": "index", "args": [v, "i"]})
}

/// Writes `doc` as `<name>.esm` and a run config of `schedule` over it into
/// a work directory, plans it over `tspan` and loads the plan with its data.
fn planned(
    name: &str,
    doc: &Value,
    schedule: Value,
    step_dt: f64,
    tspan: (f64, f64),
) -> (LoadedPlan<f64>, PathBuf, Value) {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("forcing_m2");
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join(format!("{name}.esm"));
    std::fs::write(&model, serde_json::to_vec_pretty(doc).unwrap()).unwrap();
    let rc = json!({
        "esm_integrator": "0.1.0", "model": format!("{name}.esm"),
        "operators": {"emis": {"select": {"components": "Emis"}}},
        "schedule": schedule,
        "step": {"dt": step_dt, "hit": ["forcing_refresh", "output"]},
    });
    let rc_path = dir.join(format!("{name}.run.json"));
    std::fs::write(&rc_path, serde_json::to_vec_pretty(&rc).unwrap()).unwrap();
    let plan_dir = dir.join(format!("{name}.plan"));
    let _ = std::fs::remove_dir_all(&plan_dir);
    let options = Options {
        time: Some(json!({"start": tspan.0, "end": tspan.1})),
    };
    plan(&rc_path, &options, &AstDiffBands)
        .and_then(|p| p.write(&plan_dir))
        .unwrap();
    let data = DataOptions {
        model: Some(model),
        cache_dir: Some(dir.join("esio-cache")),
        ..Default::default()
    };
    let lp = load_plan_with::<f64>(&plan_dir, &data).unwrap();
    let pj: Value =
        serde_json::from_str(&std::fs::read_to_string(plan_dir.join("plan.json")).unwrap())
            .unwrap();
    (lp, plan_dir, pj)
}

/// `c' = −K E c` with `E` the monthly inventory, a one-record parameter
/// (constant on each month), from 2016-01-20 to 2016-03-10.
fn decay_doc() -> Value {
    json!({
        "esm": "1.3.0", "metadata": {"name": "monthly_decay"},
        "domain": {"temporal": {"reference_time": "2016-01-01T00:00:00Z"}},
        "index_sets": {"x": {"kind": "interval", "size": 3}},
        "data_sources": {"Inventory": inventory_source("2016-01-01T00:00:00Z")},
        "models": {"Emis": {
            "variables": {
                "c": {"type": "unknown", "units": "1", "shape": ["x"], "default": 1.0},
                "K": {"type": "parameter", "units": "1", "default": 10000.0},
                "E": e_param(),
            },
            "equations": [{
                "lhs": {"op": "D", "args": ["c"], "wrt": "t"},
                "rhs": over_cells(json!({"op": "*", "args": [-1, "Emis.K", ix("Emis.E"), ix("c")]})),
            }],
        }},
    })
}

/// §8.4.1 and spec/adjoint.md §2.4, §3: the gradient through a planned leaf
/// whose one-record forcing multiplies the state, over steps that start and
/// end on refresh times (2016-02-01 and 2016-03-01), agrees with central
/// finite differences of the frozen tape, for an explicit method and a
/// Rosenbrock method. The first stage of the step that starts at a refresh
/// time reads the new month's rate; the right-end stage of the step that
/// ends there reads the old month's. An adjoint that took the old month's
/// rate at a step's start (the epoch of a stage at a boundary chosen from
/// its time alone), or that rebuilt the Rosenbrock W at a step's start with
/// it, would differ from the forward step by about `h b_1 K ΔE`.
#[test]
fn adjoint_takes_the_forward_steps_forcing() {
    let tspan = (19.0 * DAY, 69.0 * DAY);
    let ctl = json!({"dt_scope": "global", "reltol": 1e-8, "abstol": 1e-12});
    for alg in ["Tsit5", "Rodas4"] {
        let schedule = json!({"leaf": "emis", "alg": alg, "control": ctl});
        let (lp, _, pj) = planned(
            &format!("monthly_decay_{alg}"),
            &decay_doc(),
            schedule,
            50.0 * DAY,
            tspan,
        );
        assert_eq!(lp.timeline.refresh_times(), [31.0 * DAY, 60.0 * DAY]);
        let rc = RunConfig::from_value(&json!({
            "esm_integrator": "0.1.0", "model": "monthly_decay.esm",
            "schedule": pj["schedule"], "step": pj["step"],
            "adjoint": {"functional": {"sum": "Emis.c", "over": "final"}, "wrt": ["u0"]},
        }))
        .unwrap();
        let tstops = lp.timeline.refresh_times().to_vec();
        let s = RunSpec::with_problem_and_tstops(lp.problem, &rc, tspan, None, &tstops).unwrap();
        let u0 = s.u0();
        let p = s.problem.p.clone();
        let fwd = s.forward(&u0, &p).unwrap();
        // Steps start and end on both refresh times.
        let steps: Vec<[f64; 2]> = fwd
            .tape
            .records
            .iter()
            .filter_map(|r| match r {
                earthsci_integrator::tape::TapeRecord::Leaf { steps, .. } => Some(steps.clone()),
                _ => None,
            })
            .flatten()
            .collect();
        for b in [31.0 * DAY, 60.0 * DAY] {
            assert!(
                steps.iter().any(|s| s[0] == b),
                "{alg}: no step starts at {b}"
            );
            assert!(
                steps
                    .iter()
                    .all(|s| !(s[0] < b && s[0] + s[1] > b * (1.0 + 1e-15))),
                "{alg}: a step crosses {b}"
            );
        }
        let g = s.gradient(&fwd, &p).unwrap();
        // dc_i(t_f)/dc_i(t0) = c_i(t_f), the problem being linear with c(t0) = 1.
        let u1 = fwd.final_state();
        for (i, (&gi, &ci)) in g.u0.iter().zip(u1).enumerate() {
            assert!(
                (gi - ci).abs() < 1e-12 * ci.abs().max(1.0),
                "{alg}: dJ/dc0[{i}] = {gi}, c(t_f) = {ci}"
            );
        }
        for (o, k) in [
            (check_fd(&s, &fwd, &g, 1e-7, &[1e-3, 1e-4, 1e-5]), "fd"),
            (check_replay(&s, &fwd, &u0, &p), "replay"),
        ] {
            match o {
                Outcome::Pass(m) => println!("{alg} {k}: {m}"),
                other => panic!("{alg} {k}: {other:?}"),
            }
        }
    }
}

/// §8.4.2 item 2: a model with no `domain.temporal` counts time from the
/// Unix epoch, while EarthSciAST's providers then count from the run
/// window's start. Run from day 19 to day 69 of 1970, the epochs start at
/// day 19 and at the month starts, 1970-02-01 (day 31) and 1970-03-01 (day
/// 59), and each holds that month's record (`E_{m,i}`, file cells 2–4).
#[test]
fn provider_clock_without_a_time_origin() {
    let doc = json!({
        "esm": "1.3.0", "metadata": {"name": "no_origin"},
        "index_sets": {"x": {"kind": "interval", "size": 3}},
        "data_sources": {"Inventory": inventory_source("1970-01-01T00:00:00Z")},
        "models": {"Emis": {
            "variables": {
                "c": {"type": "unknown", "units": "1", "shape": ["x"], "default": 0.0},
                "E": e_param(),
            },
            "equations": [{
                "lhs": {"op": "D", "args": ["c"], "wrt": "t"},
                "rhs": over_cells(ix("Emis.E")),
            }],
        }},
    });
    let tspan = (19.0 * DAY, 69.0 * DAY);
    let ctl = json!({"dt_scope": "global", "reltol": 1e-8, "abstol": 1e-12});
    let (lp, _, pj) = planned(
        "no_origin",
        &doc,
        json!({"leaf": "emis", "alg": "Tsit5", "control": ctl}),
        50.0 * DAY,
        tspan,
    );
    assert_eq!(
        lp.timeline.boundaries(),
        [19.0 * DAY, 31.0 * DAY, 59.0 * DAY]
    );
    let e = |m: f64, i: f64| 1e-6 * (m + i) / 86400.0;
    for (k, m) in [1.0, 2.0, 3.0].into_iter().enumerate() {
        let got = &lp.timeline.epoch(k).bufs()["Emis.E"];
        let want = [e(m, 2.0), e(m, 3.0), e(m, 4.0)];
        for (g, w) in got.iter().zip(want) {
            assert!(
                (g - w).abs() <= 1e-15 * w,
                "epoch {k}: {got:?}, want {want:?}"
            );
        }
    }
    // The run integrates the right months: c_i(t_f) = Σ_m E_{m,i+1} × the
    // time spent in month m.
    let rc = RunConfig::from_value(&json!({
        "esm_integrator": "0.1.0", "model": "no_origin.esm",
        "schedule": pj["schedule"], "step": pj["step"],
    }))
    .unwrap();
    let tstops = lp.timeline.refresh_times().to_vec();
    let s = RunSpec::with_problem_and_tstops(lp.problem, &rc, tspan, None, &tstops).unwrap();
    let fwd = s.forward(&s.u0(), &s.problem.p).unwrap();
    let u1 = fwd.final_state();
    for (i, &ci) in u1.iter().enumerate() {
        let ii = i as f64 + 2.0;
        let want =
            e(1.0, ii) * (12.0 * DAY) + e(2.0, ii) * (28.0 * DAY) + e(3.0, ii) * (10.0 * DAY);
        assert!(
            (ci - want).abs() < 1e-9 * want,
            "c[{i}] = {ci}, want {want}"
        );
    }
}
