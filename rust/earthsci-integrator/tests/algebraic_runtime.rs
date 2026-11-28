//! The runtime of algebraic equations on the toy problems with exact answers
//! (spec/toy-problems.md §1.7–§1.11; spec/methods.md §14, §15), as
//! julia/EarthSciIntegrator.jl/test/algebraic_runtime.jl tests them.

use earthsci_integrator::algebraic::{held_set, run_solve, solve_adjoint, solve_tangent, Residual};
use earthsci_integrator::config::RunConfig;
use earthsci_integrator::conformance::{check_fd, check_replay, Outcome};
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
    for case in ["newton", "newton_inexact", "ptc"] {
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
fn box_scaled() {
    for case in ["scaled_ptc", "recover_method"] {
        let (run, lp) = run_case("box_scaled", case);
        // z[1] at the state-free observed upper bound 1.5 zmax[1] = 0.375,
        // z[2] at 0 (D125).
        assert_eq!(at_end(&run, &lp, "Box.z"), vec![0.375, 0.0], "{case}");
        let TapeRecord::Solve {
            active, recover, ..
        } = solve_record(&run.tape)
        else {
            unreachable!()
        };
        assert_eq!(active["Box.z"].upper, vec![0], "{case}");
        assert_eq!(active["Box.z"].lower, vec![1], "{case}");
        let want: Vec<String> = if case == "recover_method" {
            vec!["method".into()]
        } else {
            vec![]
        };
        assert_eq!(recover, &want, "{case}");
    }
}

/// A scale that is not positive fails the invocation (spec/methods.md §15.1).
#[test]
fn a_scale_that_is_not_positive_fails() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("algebraic_runtime")
        .join("box_scaled__negative_scale");
    let _ = std::fs::remove_dir_all(&dir);
    plan(
        &fixtures()
            .join("box_scaled")
            .join("recover_method.run.json"),
        &Options::default(),
        &AstDiffBands,
    )
    .and_then(|p| p.write(&dir))
    .unwrap();
    let f = dir.join("plan.json");
    let mut pj: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&f).unwrap()).unwrap();
    pj["schedule"]["method"]["scale"]["Box.z"] = serde_json::json!(-1.0);
    std::fs::write(&f, serde_json::to_vec(&pj).unwrap()).unwrap();
    let lp = load_plan::<f64>(&dir).unwrap();
    let spec = RunSpec::<f64>::with_problem(lp.problem, &lp.run_config, lp.tspan, None).unwrap();
    let Err(err) = spec.forward(&spec.u0(), &spec.problem.p) else {
        panic!("a scale of -1 was accepted")
    };
    assert!(err.to_string().contains("not positive"), "{err}");
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
    for case in ["newton", "newton_inexact", "ptc"] {
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

/// Plans the fixture case and runs it forward with its run config's
/// `adjoint`: the run spec, the forward run and the state DOFs.
fn adjoint_case(fixture: &str, case: &str) -> (RunSpec<f64>, ForwardRun<f64>, Dofs) {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("algebraic_runtime")
        .join(format!("{fixture}__{case}"));
    let _ = std::fs::remove_dir_all(&dir);
    let rcpath = fixtures().join(fixture).join(format!("{case}.run.json"));
    plan(&rcpath, &Options::default(), &AstDiffBands)
        .and_then(|p| p.write(&dir))
        .unwrap_or_else(|e| panic!("{fixture}/{case}: plan: {e}"));
    let lp = load_plan::<f64>(&dir).unwrap_or_else(|e| panic!("{fixture}/{case}: load: {e}"));
    let dofs = lp.state_dofs.clone();
    let pj: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("plan.json")).unwrap()).unwrap();
    let rj: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&rcpath).unwrap()).unwrap();
    let rc = RunConfig::from_value(&serde_json::json!({
        "esm_integrator": "0.1.0", "model": "model.esm",
        "schedule": pj["schedule"], "step": pj["step"], "adjoint": rj["adjoint"],
    }))
    .unwrap();
    let spec = RunSpec::<f64>::with_problem(lp.problem, &rc, lp.tspan, None)
        .unwrap_or_else(|e| panic!("{fixture}/{case}: run spec: {e}"));
    let run = spec
        .forward(&spec.u0(), &spec.problem.p)
        .unwrap_or_else(|e| panic!("{fixture}/{case}: forward: {e}"));
    (spec, run, dofs)
}

fn pass(o: Outcome, what: &str) {
    match o {
        Outcome::Pass(m) => println!("{what}: {m}"),
        other => panic!("{what}: {other:?}"),
    }
}

/// A pseudo-random number in [−1, 1].
fn rnd(seed: &mut u64) -> f64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    (*seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// The per-invocation dot-product identity of a solve (spec/adjoint.md
/// §7.1, §4.6): ⟨λ, dz⟩ against ⟨λₙ, δu⟩ + ⟨p̄, δp⟩ for the implicit-function
/// tangent and the adjoint at the same converged `z`, relative to the terms'
/// magnitude.
#[allow(clippy::too_many_arguments)]
fn node_dot(
    r: &Residual<'_, f64>,
    info: &earthsci_integrator::algebraic::AlgebraicInfo,
    z: &[f64],
    p: &[f64],
    held: &[bool],
    up: &[bool],
    wrt_p: &[bool],
) -> f64 {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let n = r.u.len();
    let du: Vec<f64> = (0..n).map(|_| rnd(&mut seed)).collect();
    let dp: Vec<f64> = wrt_p
        .iter()
        .map(|&w| if w { rnd(&mut seed) } else { 0.0 })
        .collect();
    let dz = solve_tangent(r, info, z, p, held, up, &du, &dp).unwrap();
    let mut lam = vec![0.0; n];
    let lz: Vec<f64> = info.dofs.iter().map(|_| rnd(&mut seed)).collect();
    for (&d, &v) in info.dofs.iter().zip(&lz) {
        lam[d] = v;
    }
    let mut pbar = vec![0.0; p.len()];
    solve_adjoint(r, info, z, p, held, up, &mut lam, &mut pbar, None).unwrap();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let abs_dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x * y).abs()).sum::<f64>();
    let (lhs, rhs) = (dot(&lz, &dz), dot(&lam, &du) + dot(&pbar, &dp));
    let scale = abs_dot(&lz, &dz).max(abs_dot(&lam, &du) + abs_dot(&pbar, &dp));
    (lhs - rhs).abs() / scale.max(f64::MIN_POSITIVE)
}

/// The adjoint of a solve (spec/adjoint.md §4.6, D127): the parameter
/// products, a parameter and an observed bound, the per-invocation identity,
/// finite differences of the frozen-tape map (the active set held) and the
/// exact replay.
#[test]
fn solve_adjoint_toys() {
    for (fixture, case) in [
        ("adjoint_cubic", "newton"),
        ("adjoint_box", "newton"),
        ("toy_phase_joint", "adjoint"),
    ] {
        let what = format!("{fixture}/{case}");
        let (spec, run, dofs) = adjoint_case(fixture, case);
        let p = spec.problem.p.clone();
        let u0 = spec.u0();
        let TapeRecord::Solve { z, active, t, .. } = solve_record(&run.tape) else {
            unreachable!()
        };
        let s = &spec.schedule.solves[0];
        let (held, up) = held_set(&s.info, active);
        let r = Residual {
            ops: &s.ops,
            dofs: &s.info.dofs,
            u: &u0,
            t: *t,
        };
        let mis = node_dot(&r, &s.info, z, &p, &held, &up, &spec.wrt_p);
        assert!(mis <= 1e-12, "{what}: per-invocation identity {mis:e}");
        let g = spec.gradient(&run, &p).unwrap();
        assert!(g.p.iter().any(|&x| x != 0.0), "{what}: a zero gradient");
        pass(
            check_fd(&spec, &run, &g, 1e-6, &[1e-3, 1e-4, 1e-5]),
            &format!("{what} fd"),
        );
        pass(
            check_replay(&spec, &run, &u0, &p),
            &format!("{what} replay"),
        );
        let at = |n: &str| {
            spec.problem
                .param_names
                .iter()
                .position(|x| x == n)
                .unwrap()
        };
        if fixture == "adjoint_box" {
            // Both bounds active: dz₁/dcap = zmax₁ through the frozen
            // residual's product, dz₂/dzlo = 1 through the parameter bound,
            // and b does not move an unknown held at a bound.
            assert!((g.p[at("Box.cap")] - 0.25).abs() <= 1e-14, "{:?}", g.p);
            assert!((g.p[at("Box.zlo")] - 1.0).abs() <= 1e-14, "{:?}", g.p);
            assert_eq!(g.p[at("Box.beta")], 0.0);
        } else if fixture == "adjoint_cubic" {
            // Σ dz/da = Σ cᵢ / (1 + 3zᵢ²).
            let z = at_end(&run, &dofs, "Cubic.z");
            let want: f64 = [1.0, 2.0, 3.0, 4.0]
                .iter()
                .zip(&z)
                .map(|(c, z)| c / (1.0 + 3.0 * z * z))
                .sum();
            assert!(
                (g.p[at("Cubic.a")] - want).abs() <= 1e-12 * want,
                "{} {want}",
                g.p[0]
            );
        }
    }
}

/// The adjoint of an index-1 DAE (spec/adjoint.md §5.10, D127) with respect
/// to the initial state: ESDIRK steps with σM − J and the consistent
/// initialization as a §4.6 solve; finite differences of the frozen-tape
/// map, the initialization's per-invocation identity, the exact replay.
#[test]
fn dae_adjoint_robertson() {
    for case in ["kvaerno4_adjoint", "trbdf2_adjoint"] {
        let (spec, run, dofs) = adjoint_case("toy_robertson_dae", case);
        let p = spec.problem.p.clone();
        let u0 = spec.u0();
        let g = spec.gradient(&run, &p).unwrap();
        assert_eq!(
            g.u0[dofs["Rober.y3"][0]], 0.0,
            "{case}: y3 only starts the initialization"
        );
        pass(
            check_fd(&spec, &run, &g, 1e-5, &[1e-3, 1e-4, 1e-5]),
            &format!("{case} fd"),
        );
        pass(
            check_replay(&spec, &run, &u0, &p),
            &format!("{case} replay"),
        );
        let d = spec.schedule.leaves[0].dae.as_ref().unwrap();
        let r = Residual {
            ops: &d.ops,
            dofs: &d.info.dofs,
            u: &u0,
            t: 0.0,
        };
        let warm: Vec<f64> = d.info.dofs.iter().map(|&j| u0[j]).collect();
        let o = run_solve(&r, &d.info, &p, &d.pnames, &d.init, &warm, &warm, case).unwrap();
        let nz = o.z.len();
        let mis = node_dot(
            &r,
            &d.info,
            &o.z,
            &p,
            &vec![false; nz],
            &vec![false; nz],
            &vec![false; p.len()],
        );
        assert!(mis <= 1e-12, "{case}: initialization identity {mis:e}");
    }
}
