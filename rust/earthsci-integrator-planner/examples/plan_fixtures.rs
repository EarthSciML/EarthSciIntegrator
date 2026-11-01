//! Plans every case of `tests/conformance/plans/fixtures/` and writes each
//! plan directory to `<out>/<fixture>/<case>/` (`plan.json`, `derived/`,
//! `plan.report.json`), for comparing with the Julia planner's goldens.
//!
//! ```text
//! cargo run -p earthsci-integrator-planner --example plan_fixtures -- <out>
//! ```

use earthsci_integrator_planner::astdiff::AstDiffBands;
use earthsci_integrator_planner::{plan, Options};
use std::path::PathBuf;

fn main() {
    let out = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: plan_fixtures <output directory>"),
    );
    let fixtures =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/conformance/plans/fixtures");
    let mut dirs: Vec<_> = std::fs::read_dir(&fixtures)
        .expect("fixtures")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    let mut failed = false;
    for dir in dirs {
        let fixture = dir.file_name().unwrap().to_string_lossy().into_owned();
        let mut cases: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.to_string_lossy().ends_with(".run.json"))
            .collect();
        cases.sort();
        for rc in cases {
            let name = rc.file_name().unwrap().to_string_lossy().into_owned();
            let case = name.trim_end_matches(".run.json");
            let target = out.join(&fixture).join(case);
            match plan(&rc, &Options::default(), &AstDiffBands).and_then(|p| p.write(&target)) {
                Ok(()) => println!("{fixture}/{case}: {}", target.display()),
                Err(e) => {
                    failed = true;
                    println!("{fixture}/{case}: {e}");
                }
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
}
