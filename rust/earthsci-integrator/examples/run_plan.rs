//! Loads a plan directory, runs it forward and prints the named variables at
//! the end and the solve records.
//!
//! ```text
//! cargo run --release -p earthsci-integrator --example run_plan -- <plan dir> <variable>...
//! ```

use earthsci_integrator::planned::load_plan;
use earthsci_integrator::schedule::RunSpec;
use earthsci_integrator::tape::TapeRecord;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(
        args.next()
            .expect("usage: run_plan <plan dir> <variable>..."),
    );
    let vars: Vec<String> = args.collect();
    let t0 = Instant::now();
    let lp = load_plan::<f64>(&dir).unwrap_or_else(|e| panic!("load: {e}"));
    let dofs = lp.state_dofs.clone();
    let spec = RunSpec::<f64>::with_problem(lp.problem, &lp.run_config, lp.tspan, None)
        .unwrap_or_else(|e| panic!("run spec: {e}"));
    let tl = t0.elapsed().as_secs_f64();
    let run = match spec.forward(&spec.u0(), &spec.problem.p) {
        Ok(r) => r,
        Err(e) => {
            for op in &spec.problem.operators {
                if let Some(f) = op.fault() {
                    eprintln!("operator {}: {f}", op.name());
                }
            }
            panic!("forward: {e}")
        }
    };
    let u = &run.outputs.last().unwrap().1;
    for v in &vars {
        let x: Vec<f64> = dofs[v].iter().map(|&j| u[j]).collect();
        println!("{v} = {x:?}");
    }
    for r in &run.tape.records {
        if let TapeRecord::Solve { .. } = r {
            println!("{r:?}");
        }
    }
    println!("load {tl:.2} s, total {:.2} s", t0.elapsed().as_secs_f64());
}
