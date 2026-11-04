//! Plans one run config and writes its plan directory (`plan.json`,
//! `derived/`, `plan.report.json`), or prints the refusal.
//!
//! ```text
//! cargo run -p earthsci-integrator-planner --example plan_one -- <run config> <out>
//! ```

use earthsci_integrator_planner::astdiff::AstDiffBands;
use earthsci_integrator_planner::{plan, Options};
use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let usage = "usage: plan_one <run config> <output directory>";
    let rc = PathBuf::from(args.next().expect(usage));
    let out = PathBuf::from(args.next().expect(usage));
    match plan(&rc, &Options::default(), &AstDiffBands).and_then(|p| p.write(&out)) {
        Ok(()) => println!("planned {} into {}", rc.display(), out.display()),
        Err(e) => {
            eprintln!("{}: {e}", rc.display());
            std::process::exit(1);
        }
    }
}
