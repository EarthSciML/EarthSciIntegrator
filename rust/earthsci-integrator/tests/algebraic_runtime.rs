//! The runtime of algebraic equations on the toy problems with exact answers
//! (spec/toy-problems.md §1.7–§1.10; spec/methods.md §14, §15), as
//! julia/EarthSciIntegrator.jl/test/algebraic_runtime.jl tests them.

use earthsci_integrator::planned::load_plan;
use earthsci_integrator::schedule::{ForwardRun, RunSpec};
use earthsci_integrator::tape::{ActiveSet, Tape, TapeRecord};
use earthsci_integrator_planner::astdiff::AstDiffBands;
use earthsci_integrator_planner::{plan, Options};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/conformance/plans/fixtures")
}

/// Plans the fixture case, loads the plan, and runs it forward.
type Dofs = BTreeMap<String, Vec<usize>>;

fn run_case(fixture: &str, case: &str) -> (ForwardRun<f64>, Dofs) {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("algebraic_runtime")
        .join(format!("{fixture}__{case}"));
    let _ = std::fs::remove_dir_all(&dir);
    plan(
        &fixtures().join(fixture).join(format!("{case}.run.json")),
        &Options::default(),
        &AstDiffBands,
    )
    .and_then(|p| p.write(&dir))
    .unwrap_or_else(|e| panic!("{fixture}/{case}: plan: {e}"));
    let lp = load_plan::<f64>(&dir).unwrap_or_else(|e| panic!("{fixture}/{case}: load: {e}"));
    let dofs = lp.state_dofs.clone();
    let spec = RunSpec::<f64>::with_problem(lp.problem, &lp.run_config, lp.tspan, None)
        .unwrap_or_else(|e| panic!("{fixture}/{case}: run spec: {e}"));
    let run = spec
        .forward(&spec.u0(), &spec.problem.p)
        .unwrap_or_else(|e| panic!("{fixture}/{case}: forward: {e}"));
    (run, dofs)
}

fn at_end(run: &ForwardRun<f64>, lp: &Dofs, v: &str) -> Vec<f64> {
    let u = &run.outputs.last().unwrap().1;
    lp[v].iter().map(|&j| u[j]).collect()
}

fn solve_record(tape: &Tape) -> &TapeRecord {
    let mut it = tape
        .records
        .iter()
        .filter(|r| matches!(r, TapeRecord::Solve { .. }));
    let r = it.next().expect("a solve record");
    assert!(it.next().is_none());
    r
}

fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

#[test]
fn cubic_roots() {
    for case in ["newton", "ptc"] {
        let (run, lp) = run_case("toy_cubic_roots", case);
        assert_eq!(run.outputs.last().unwrap().0, 1.0);
        let z = at_end(&run, &lp, "Cubic.z");
        let want = [
            0.6823278038280193,
            1.0,
            1.2134116627622296,
            1.378796700129551,
        ];
        assert!(close(&z, &want, 1e-11), "{case}: {z:?}");
        let TapeRecord::Solve { active, .. } = solve_record(&run.tape) else {
            unreachable!()
        };
        assert!(active.is_empty());
    }
}

#[test]
fn box_lcp() {
    for case in ["newton", "ptc"] {
        let (run, lp) = run_case("toy_box_lcp", case);
        // Both bounds active: z₁ at its referenced upper bound, z₂ at 0 (§1.8).
        assert_eq!(at_end(&run, &lp, "Box.z"), vec![0.25, 0.0], "{case}");
        let TapeRecord::Solve { active, .. } = solve_record(&run.tape) else {
            unreachable!()
        };
        let want = ActiveSet {
            lower: vec![1],
            upper: vec![0],
            degenerate: vec![],
        };
        assert_eq!(active.get("Box.z"), Some(&want), "{case}");
        // The tape's JSON Lines form reads back the same record (spec/tape.md §5).
        let back = Tape::from_jsonl(&run.tape.to_jsonl()).unwrap();
        assert!(solve_record(&back).bitwise_eq(solve_record(&run.tape)));
    }
}

#[test]
fn phase_joint() {
    let (run, lp) = run_case("toy_phase_joint", "ptc");
    // s column-major over cell × salt; cell 1 saturates salt 1 (§1.9).
    assert!(close(
        &at_end(&run, &lp, "Phase.s"),
        &[1.0, 0.0, 0.0, 0.0],
        1e-11
    ));
    assert!(close(&at_end(&run, &lp, "Phase.w"), &[1.0, 0.75], 1e-11));
}

#[test]
fn robertson_dae() {
    let (ode, olp) = run_case("toy_robertson", "rodas4");
    let y1 = at_end(&ode, &olp, "Kinetics.y1")[0];
    let y2 = at_end(&ode, &olp, "Kinetics.y2")[0];
    for (case, rtol) in [
        ("rodas4", 1e-8),
        ("kvaerno4", 1e-6),
        ("trbdf2", 1e-4),
        ("inconsistent", 1e-8),
    ] {
        let (run, lp) = run_case("toy_robertson_dae", case);
        assert_eq!(run.outputs.last().unwrap().0, 40.0, "{case}");
        let y: Vec<f64> = ["Rober.y1", "Rober.y2", "Rober.y3"]
            .iter()
            .map(|v| at_end(&run, &lp, v)[0])
            .collect();
        assert!(
            (y[0] - y1).abs() <= rtol * y1.abs(),
            "{case}: y1 {} vs {y1}",
            y[0]
        );
        assert!(
            (y[1] - y2).abs() <= rtol * y2.abs(),
            "{case}: y2 {} vs {y2}",
            y[1]
        );
        // The algebraic equation holds at the step ends (stiffly accurate methods).
        assert!(
            (1.0 - y.iter().sum::<f64>()).abs() <= 1e-14,
            "{case}: {y:?}"
        );
    }
}
