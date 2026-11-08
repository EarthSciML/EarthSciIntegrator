//! The runtimes' own checks on the data-input toy `data_decay`
//! (spec/toy-problems.md §4.4, spec/operator-interface.md §8.4): the forcing
//! buffers at `t0` equal, bit for bit, those of EarthSciAST's `esm_problem`
//! of the model with the same providers (§8.4.2 item 3), and a run of the
//! variant whose initial condition reads data starts from the data, `S_i`,
//! not the default 1.0, and ends at its reference (§8.4.5).

use earthsci_integrator::conformance::Outcome;
use earthsci_integrator::conformance_m2::{check_reference, run_plan_with, Case};
use earthsci_integrator::forcing::{plan_forcing, DataOptions, ModelData};
use earthsci_integrator::planned::load_plan_with;
use earthsci_integrator_planner::astdiff::AstDiffBands;
use earthsci_integrator_planner::{plan, Options};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Plans `m2.data_decay.tsit5`'s run config with the toy `model` in place
/// of its own into a directory of the target's.
fn planned(name: &str, model: &str) -> (Case, PathBuf, DataOptions) {
    planned_at(name, &repo().join("tests/conformance/toys").join(model))
}

/// [`planned`] for the model at `model`, any path.
fn planned_at(name: &str, model: &Path) -> (Case, PathBuf, DataOptions) {
    let cases: Value = serde_json::from_str(
        &std::fs::read_to_string(repo().join("tests/conformance/m2/cases.json")).unwrap(),
    )
    .unwrap();
    let mut case = cases["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["id"] == "m2.data_decay.tsit5")
        .unwrap()
        .clone();
    case["id"] = json!(name);
    case["checks"] = json!([]);
    let work = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("data_m2");
    std::fs::create_dir_all(&work).unwrap();
    // The run config names the toy by its absolute path; the plan goes to
    // the target's directory.
    let mut run_config = case["run_config"].clone();
    run_config["model"] = json!(model.to_string_lossy());
    let rc_path = work.join(format!("{name}.run.json"));
    std::fs::write(&rc_path, run_config.to_string()).unwrap();
    let plan_dir = work.join(name);
    let _ = std::fs::remove_dir_all(&plan_dir);
    let tspan = &case["tspan"];
    let options = Options {
        time: Some(json!({"start": tspan[0], "end": tspan[1]})),
    };
    plan(&rc_path, &options, &AstDiffBands)
        .and_then(|p| p.write(&plan_dir))
        .unwrap();
    let data = DataOptions {
        model: Some(model.to_path_buf()),
        cache_dir: Some(work.join("esio-cache")),
        ..Default::default()
    };
    (serde_json::from_value(case).unwrap(), plan_dir, data)
}

#[test]
fn buffers_at_t0_are_esm_problems() {
    let (case, plan_dir, data) = planned("data_decay_buffers", "data_decay.esm");
    let lp = load_plan_with::<f64>(&plan_dir, &data).unwrap();
    let plan_json: Value =
        serde_json::from_slice(&std::fs::read(plan_dir.join("plan.json")).unwrap()).unwrap();
    let md = ModelData::load(data.model.as_ref().unwrap()).unwrap();
    let tspan = (case.tspan[0], case.tspan[1]);
    let prob = md
        .problem(&plan_forcing(&plan_json).unwrap(), tspan, &data)
        .unwrap();
    let prog = earthsci_ast::rhs_program(&prob).unwrap();
    let theirs = prog.default_forcing();
    let ours = lp.timeline.epoch(0).bufs();
    assert_eq!(theirs.slots().len(), ours.len());
    for (k, slot) in theirs.slots().iter().enumerate() {
        let bits = |b: &[f64]| b.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&ours[&slot.name]),
            bits(theirs.buffer(k)),
            "{}",
            slot.name
        );
    }
    // The epochs: t0, each hour, and t_f (§8.4.3).
    assert_eq!(
        lp.timeline.boundaries(),
        [79200.0, 82800.0, 86400.0, 90000.0, 93600.0]
    );
}

#[test]
fn initial_state_reads_data() {
    let (case, plan_dir, data) = planned("data_decay_ic", "data_decay_ic.esm");
    let lp = load_plan_with::<f64>(&plan_dir, &data).unwrap();
    // S_i = 1 + i/4, not the default 1.0.
    assert_eq!(lp.problem.u0, [1.0, 1.25, 1.5]);
    let r = run_plan_with(&case, &plan_dir, &data).unwrap();
    let reference = repo().join("tests/conformance/m2/reference/data_decay_ic.json");
    match check_reference(&r, &reference, 1e-6) {
        Outcome::Pass(_) => {}
        o => panic!("{o:?}"),
    }
}

/// spec/plan.md §6.2 item 4: `data_decay_ic` with its state's `default`
/// removed, so only the `ic` equation, which reads data, starts it. The
/// planner gives the state the `default` 1.0 in the Jacobian document; the
/// plan builds, loads and starts from the data.
#[test]
fn initial_state_reads_data_without_default() {
    let toys = repo().join("tests/conformance/toys");
    let mut doc: Value =
        serde_json::from_slice(&std::fs::read(toys.join("data_decay_ic.esm")).unwrap()).unwrap();
    assert!(doc["models"]["Decay"]["variables"]["u"]
        .as_object_mut()
        .unwrap()
        .remove("default")
        .is_some());
    // The data by absolute path, so the model can live outside the toys.
    for (_, src) in doc["data_sources"].as_object_mut().unwrap() {
        let t = src["source"]["url_template"].as_str().unwrap();
        src["source"]["url_template"] = json!(toys.join(t).to_string_lossy());
    }
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("data_m2");
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("data_decay_ic_nodefault.esm");
    std::fs::write(&model, doc.to_string()).unwrap();
    let (_, plan_dir, data) = planned_at("data_decay_ic_nodefault", &model);
    let jac: Value = serde_json::from_slice(
        &std::fs::read(plan_dir.join("derived/decay.jacobian.esm")).unwrap(),
    )
    .unwrap();
    assert_eq!(jac["models"]["Decay"]["variables"]["u"]["default"], 1.0);
    let lp = load_plan_with::<f64>(&plan_dir, &data).unwrap();
    assert_eq!(lp.problem.u0, [1.0, 1.25, 1.5]);
}
