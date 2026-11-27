//! The runtime of algebraic equations on the toy problems with exact answers
//! (spec/toy-problems.md §1.7–§1.11; spec/methods.md §14, §15), as
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

/// The blocks a plan's structure gives a solve (spec/plan.md §5.4).
#[test]
fn solve_blocks() {
    for (fixture, case, op, want) in [
        (
            "toy_cubic_roots",
            "newton",
            "cubic",
            vec![vec![0], vec![1], vec![2], vec![3]],
        ),
        // One 3 × 3 block per cell: s[c, 1], s[c, 2], w[c] (positions among
        // the unknowns s then w, column-major).
        (
            "toy_phase_joint",
            "per_cell",
            "phase",
            vec![vec![0, 2, 4], vec![1, 3, 5]],
        ),
    ] {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join("algebraic_runtime")
            .join(format!("{fixture}__{case}__blocks"));
        let _ = std::fs::remove_dir_all(&dir);
        plan(
            &fixtures().join(fixture).join(format!("{case}.run.json")),
            &Options::default(),
            &AstDiffBands,
        )
        .and_then(|p| p.write(&dir))
        .unwrap();
        let lp = load_plan::<f64>(&dir).unwrap();
        assert_eq!(
            lp.problem.algebraic[op].blocks,
            Some(want),
            "{fixture}/{case}"
        );
    }
}

#[test]
fn picard_roots() {
    // The roots of z + z³ = c, c = i/4 (spec/toy-problems.md §1.11).
    let want = [
        0.2367329038645632,
        0.4238537990697833,
        0.5673642266809228,
        0.6823278038280195,
    ];
    for (case, tol) in [
        ("picard", 1e-12),
        ("rel_change", 1e-11),
        ("newton_rel_change", 1e-11),
    ] {
        let (run, lp) = run_case("toy_picard_roots", case);
        let z = at_end(&run, &lp, "Picard.z");
        assert!(close(&z, &want, tol), "{case}: {z:?}");
        let TapeRecord::Solve { iterations, .. } = solve_record(&run.tape) else {
            unreachable!()
        };
        // Picard with ν frozen is the fixed point z ← c/ν(z): linear
        // convergence, at rate 0.64 in the slowest cell; Newton is quadratic.
        if case == "newton_rel_change" {
            assert!(*iterations <= 8, "{case}: {iterations}");
        } else {
            assert!((30..=100).contains(iterations), "{case}: {iterations}");
        }
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
    // `per_cell` names cell the only horizontal axis (salt inside the cell, D123).
    for case in ["ptc", "per_cell"] {
        let (run, lp) = run_case("toy_phase_joint", case);
        // s column-major over cell × salt; cell 1 saturates salt 1 (§1.9).
        assert!(close(
            &at_end(&run, &lp, "Phase.s"),
            &[1.0, 0.0, 0.0, 0.0],
            1e-11
        ));
        assert!(close(&at_end(&run, &lp, "Phase.w"), &[1.0, 0.75], 1e-11));
    }
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
