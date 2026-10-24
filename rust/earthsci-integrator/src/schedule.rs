//! The schedule interpreter (spec/run-config.md §4–§5, spec/tape.md).
//!
//! Builds a schedule from a run config against a toy problem, then runs it
//! macro step by macro step, recording the tape and a checkpoint at every
//! macro-step boundary. The same interpreter replays a run from its tape
//! (controller off), which the adjoint and the frozen-tape finite
//! differences use.

use crate::config::{parse_duration, Control, LeafNode, Node, RunConfig};
use crate::leaf::{BindKind, DomainSpec, DtMode, LeafCall, LeafMethod, LeafSpec, StepData};
use crate::operator::{MapOperator, Structure};
use crate::tape::{Checkpoint, LeafPersist, Retcode, Tape, TapeRecord};
use crate::toy::ToyProblem;
use crate::{Error, Scalar};
use earthsci_integrator_solvers::controller::{Controller, ControllerKind};
use earthsci_integrator_solvers::erk::ExplicitRk;
use earthsci_integrator_solvers::norm::Tolerances;
use earthsci_integrator_solvers::scalar::eps_f64;
use earthsci_integrator_solvers::tableau::{tableau, Family};
use std::collections::HashMap;
use std::sync::Arc;

/// A node of the built schedule.
#[derive(Debug, Clone)]
pub enum SNode {
    Leaf {
        id: String,
        idx: usize,
    },
    Map {
        id: String,
        idx: usize,
    },
    Sequence {
        id: String,
        children: Vec<SNode>,
    },
    Strang {
        id: String,
        children: Vec<SNode>,
    },
    Subcycle {
        id: String,
        child: Box<SNode>,
        n: usize,
    },
}

/// A built map node. Toy map operators take their `Δ` from the node's
/// interval (spec/toy-problems.md §1.6: "Δ bound to the node's step").
pub struct MapSpec<T: Scalar> {
    pub id: String,
    pub op: Arc<dyn MapOperator<T>>,
    pub bind: BindKind,
}

/// A built schedule: the node tree plus its leaves and maps.
pub struct Schedule<T: Scalar> {
    pub root: SNode,
    pub leaves: Vec<LeafSpec<T>>,
    pub maps: Vec<MapSpec<T>>,
}

/// What the adjoint will need, so the build can refuse missing calls.
#[derive(Debug, Clone, Copy, Default)]
pub struct Needs {
    pub adjoint: bool,
    pub wrt_p: bool,
}

impl<T: Scalar> Schedule<T> {
    /// Builds the schedule of `node` against problem `pb`.
    pub fn build(node: &Node, pb: &ToyProblem<T>, needs: Needs) -> Result<Self, Error> {
        let mut s = Schedule {
            root: SNode::Map {
                id: String::new(),
                idx: 0,
            },
            leaves: Vec::new(),
            maps: Vec::new(),
        };
        let root = s.build_node(node, "root", pb, needs)?;
        s.root = root;
        let mut ids = std::collections::HashSet::new();
        check_ids(&s.root, &mut ids)?;
        Ok(s)
    }

    fn build_node(
        &mut self,
        node: &Node,
        path: &str,
        pb: &ToyProblem<T>,
        needs: Needs,
    ) -> Result<SNode, Error> {
        let id_or = |id: &Option<String>| id.clone().unwrap_or_else(|| path.to_string());
        Ok(match node {
            Node::Leaf(l) => {
                let id = id_or(&l.id);
                let spec = build_leaf(l, &id, pb, needs)?;
                self.leaves.push(spec);
                SNode::Leaf {
                    id,
                    idx: self.leaves.len() - 1,
                }
            }
            Node::Map { id, op } => {
                let id = id_or(id);
                let m = pb.map(op).ok_or_else(|| {
                    Error::Config(format!("map node {id}: no map operator named {op:?}"))
                })?;
                self.maps.push(MapSpec {
                    id: id.clone(),
                    op: m,
                    bind: BindKind::Node,
                });
                SNode::Map {
                    id,
                    idx: self.maps.len() - 1,
                }
            }
            Node::Sequence { id, children } | Node::Strang { id, children } => {
                // Children are named `<id>.0`, `<id>.1`, … after this node's
                // own id (spec/plan.md §3), as the Julia implementation does.
                let id = id_or(id);
                let kids = children
                    .iter()
                    .enumerate()
                    .map(|(i, c)| self.build_node(c, &format!("{id}.{i}"), pb, needs))
                    .collect::<Result<Vec<_>, _>>()?;
                if matches!(node, Node::Sequence { .. }) {
                    SNode::Sequence { id, children: kids }
                } else {
                    SNode::Strang { id, children: kids }
                }
            }
            Node::Subcycle {
                id,
                child,
                n,
                limits,
            } => {
                if limits.is_some() {
                    return Err(Error::Unimplemented(
                        "subcycle with limits needs limit operators (M2); use n".into(),
                    ));
                }
                let n = n.ok_or_else(|| Error::Config("subcycle needs n".into()))?;
                if n < 1 {
                    return Err(Error::Config(format!(
                        "subcycle {}: n = {n}; it must be at least 1 (run-config §4.8)",
                        id_or(id)
                    )));
                }
                // The child is `<id>.subcycle` (spec/plan.md §3's `root.1.every` pattern).
                let id = id_or(id);
                let child =
                    Box::new(self.build_node(child, &format!("{id}.subcycle"), pb, needs)?);
                SNode::Subcycle { id, child, n }
            }
        })
    }

    /// The leaf indices in the order the schedule lists them.
    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }

    /// Fresh persistent leaf states.
    pub fn initial_persist(&self) -> Vec<LeafPersist> {
        self.leaves.iter().map(|l| l.initial_persist()).collect()
    }

    /// Methods of every leaf, `(id, alg)`.
    pub fn algs(&self) -> Vec<(String, String)> {
        self.leaves
            .iter()
            .map(|l| (l.id.clone(), l.alg.clone()))
            .collect()
    }
}

fn check_ids(n: &SNode, seen: &mut std::collections::HashSet<String>) -> Result<(), Error> {
    let (id, kids): (&String, Vec<&SNode>) = match n {
        SNode::Leaf { id, .. } | SNode::Map { id, .. } => (id, vec![]),
        SNode::Sequence { id, children } | SNode::Strang { id, children } => {
            (id, children.iter().collect())
        }
        SNode::Subcycle { id, child, .. } => (id, vec![child.as_ref()]),
    };
    if !seen.insert(id.clone()) {
        return Err(Error::Config(format!("duplicate node id {id:?}")));
    }
    for k in kids {
        check_ids(k, seen)?;
    }
    Ok(())
}

/// The DOFs whose variable is `name` (a toy problem's variable names are its
/// qualified names).
fn variable_dofs<T: Scalar>(pb: &ToyProblem<T>, name: &str) -> Result<Vec<usize>, Error> {
    let v: Vec<usize> = (0..pb.n).filter(|&j| pb.variables[j] == name).collect();
    if v.is_empty() {
        return Err(Error::Config(format!(
            "{name:?} is not a variable of {}",
            pb.name
        )));
    }
    Ok(v)
}

fn build_leaf<T: Scalar>(
    l: &LeafNode,
    id: &str,
    pb: &ToyProblem<T>,
    needs: Needs,
) -> Result<LeafSpec<T>, Error> {
    let names = l
        .leaf
        .as_ref()
        .ok_or_else(|| Error::Config(format!("leaf {id}: no operators")))?
        .names();
    let mut ops = Vec::new();
    for nm in &names {
        ops.push(pb.operator(nm).ok_or_else(|| {
            Error::Config(format!(
                "leaf {id}: no operator named {nm:?} in {}",
                pb.name
            ))
        })?);
    }
    let alg = l.alg.clone().unwrap_or_else(|| "auto".into());
    if alg == "auto" {
        return Err(Error::Unimplemented(format!(
            "leaf {id}: alg \"auto\" needs the planner (M2)"
        )));
    }
    let tab =
        tableau(&alg).ok_or_else(|| Error::Config(format!("leaf {id}: unknown alg {alg:?}")))?;
    let method = match tab.family {
        Family::ExplicitRk => {
            LeafMethod::Erk(ExplicitRk::from_tableau(tab).map_err(Error::Config)?)
        }
        f => {
            return Err(Error::Unimplemented(format!(
                "leaf {id}: {alg} ({f:?} family) is not implemented in the Rust implementation yet"
            )))
        }
    };
    let mut dofs: Vec<usize> = ops.iter().flat_map(|o| o.writes()).collect();
    dofs.sort_unstable();
    dofs.dedup();
    let structure = Structure::common(ops.iter().map(|o| o.structure()));

    // Calls the method and its adjoint need (operator-interface.md §3).
    let mut calls = method.forward_calls().to_vec();
    if needs.adjoint {
        calls.extend(method.adjoint_calls(needs.wrt_p));
    }
    for op in &ops {
        for c in &calls {
            if !op.provides(*c) {
                return Err(Error::Config(format!(
                    "leaf {id}: operator {} does not provide {}",
                    op.name(),
                    c.name()
                )));
            }
        }
    }

    // Options explicit leaves do not take.
    if l.theta.is_some() {
        return Err(Error::Config(format!(
            "leaf {id}: theta applies to Theta only"
        )));
    }
    if l.imex.is_some() || l.implicit_solve.is_some() {
        return Err(Error::Config(format!(
            "leaf {id}: imex and implicit_solve apply to IMEXEuler only"
        )));
    }
    if l.frozen.is_some() {
        return Err(Error::Unimplemented(format!(
            "leaf {id}: frozen tendencies are not implemented at M1"
        )));
    }
    if l.partition.is_some() {
        return Err(Error::Unimplemented(format!(
            "leaf {id}: partition is not implemented at M1 (single shard)"
        )));
    }
    // jacobian and linsolve are accepted and unused by explicit methods.

    let dt_mode = match &l.dt {
        None => DtMode::Adaptive,
        Some(v) if v.as_str() == Some("step") => DtMode::Step,
        Some(v) if v.as_str() == Some("accumulated") => DtMode::Accumulated,
        Some(v) => DtMode::Fixed(parse_duration(v)?),
    };
    if dt_mode == DtMode::Adaptive && !method.adaptive() {
        return Err(Error::Config(format!(
            "leaf {id}: {alg} has no error estimate, so the leaf must give dt (D52)"
        )));
    }

    let ctl = l.control.clone().unwrap_or_default();
    let (tol, kind, dt0, dtmin, dtmax) = control(&ctl, id, pb)?;
    let controller = if method.adaptive() {
        Controller::new(kind, tab)
    } else {
        None
    };

    let domain = match &l.domain {
        None => None,
        Some(d) => {
            if d.check_stages == Some(true) {
                return Err(Error::Config(format!(
                    "leaf {id}: check_stages applies to Rosenbrock leaves only"
                )));
            }
            let listed = match &d.nonnegative {
                None => vec![],
                Some(v) if v.as_str() == Some("all") => dofs.clone(),
                Some(serde_json::Value::Array(a)) => {
                    let mut v = Vec::new();
                    for name in a {
                        let nm = name.as_str().ok_or_else(|| {
                            Error::Config(format!("leaf {id}: domain.nonnegative entry {name}"))
                        })?;
                        v.extend(
                            variable_dofs(pb, nm)?
                                .into_iter()
                                .filter(|j| dofs.contains(j)),
                        );
                    }
                    v.sort_unstable();
                    v.dedup();
                    v
                }
                Some(v) => return Err(Error::Config(format!("leaf {id}: domain.nonnegative {v}"))),
            };
            let clip = match d.on_violation.as_deref() {
                None | Some("reject") => false,
                Some("clip") => true,
                Some(o) => return Err(Error::Config(format!("leaf {id}: on_violation {o}"))),
            };
            Some(DomainSpec { dofs: listed, clip })
        }
    };

    let mut binds = Vec::new();
    for (name, kind) in l.bind_dt.iter().flatten() {
        let i = pb
            .param_names
            .iter()
            .position(|p| p == name)
            .ok_or_else(|| {
                Error::Config(format!(
                    "leaf {id}: bind_dt names unknown parameter {name:?}"
                ))
            })?;
        let k = match kind.as_str() {
            "leaf" => BindKind::Leaf,
            "node" => BindKind::Node,
            "macro" => BindKind::Macro,
            o => return Err(Error::Config(format!("leaf {id}: bind_dt value {o:?}"))),
        };
        binds.push((i, k));
    }
    // spec/methods.md §9: the first step evaluates f before there is a step
    // whose size a "leaf" binding could give.
    if dt_mode == DtMode::Adaptive && dt0.is_none() && binds.iter().any(|b| b.1 == BindKind::Leaf) {
        return Err(Error::Config(format!(
            "leaf {id}: an adaptive leaf with a bind_dt \"leaf\" parameter needs control.dt0, \
             because the first-step algorithm evaluates f before any step size exists \
             (spec/methods.md §9, run-config §8 rule 14)"
        )));
    }

    Ok(LeafSpec {
        id: id.to_string(),
        ops,
        dofs,
        structure,
        alg,
        method,
        dt_mode,
        tol,
        controller,
        dt0,
        dtmin,
        dtmax,
        maxiters: ctl.maxiters,
        domain,
        binds,
        work: Default::default(),
    })
}

type ControlParts = (Tolerances, ControllerKind, Option<f64>, f64, Option<f64>);

/// Reads `control`. Defaults where the run config omits a key are
/// DiffEqBase's: `reltol = 1e-3`, `abstol = 1e-6`.
fn control<T: Scalar>(c: &Control, id: &str, pb: &ToyProblem<T>) -> Result<ControlParts, Error> {
    let reltol = c.reltol.unwrap_or(1e-3);
    let mut abstol = vec![1e-6; pb.n];
    match &c.abstol {
        None => {}
        Some(v) if v.is_number() => abstol = vec![v.as_f64().unwrap(); pb.n],
        Some(serde_json::Value::Object(m)) => {
            for (name, val) in m {
                let x = val.as_f64().ok_or_else(|| {
                    Error::Config(format!("leaf {id}: abstol for {name} is not a number"))
                })?;
                for j in variable_dofs(pb, name)? {
                    abstol[j] = x;
                }
            }
        }
        Some(v) => return Err(Error::Config(format!("leaf {id}: abstol {v}"))),
    }
    let kind = match c.controller.as_deref() {
        None | Some("auto") | Some("PI") => ControllerKind::PI,
        Some("I") => ControllerKind::I,
        Some("PID") => ControllerKind::PID,
        Some(o) => return Err(Error::Config(format!("leaf {id}: controller {o:?}"))),
    };
    match c.dt_scope.as_deref() {
        None | Some("auto") | Some("global") => {}
        Some(o) => {
            return Err(Error::Unimplemented(format!(
                "leaf {id}: dt_scope {o:?} arrives at M3; M1 runs dt_scope global"
            )))
        }
    }
    let dur = |v: &Option<serde_json::Value>| v.as_ref().map(parse_duration).transpose();
    Ok((
        Tolerances { abstol, reltol },
        kind,
        dur(&c.dt0)?,
        dur(&c.dtmin)?.unwrap_or(0.0),
        dur(&c.dtmax)?,
    ))
}

/// What the reverse pass needs from one invocation, recorded during replay.
#[derive(Debug, Clone)]
pub enum AdjRecord<T> {
    Leaf {
        leaf: usize,
        steps: Vec<StepData<T>>,
    },
    Map {
        map: usize,
        u_before: Vec<T>,
        t: f64,
        dt: f64,
        mask: Vec<u8>,
    },
}

/// Executes the schedule over one macro step.
struct Exec<'a, T: Scalar> {
    sched: &'a Schedule<T>,
    p: &'a [f64],
    macro_index: usize,
    macro_dt: f64,
    persist: &'a mut [LeafPersist],
    replay: Option<&'a HashMap<&'a str, &'a TapeRecord>>,
    tape_out: Vec<TapeRecord>,
    adj: Option<Vec<AdjRecord<T>>>,
    /// Output times strictly inside this macro step, for a root leaf.
    stops: Vec<f64>,
    stop_states: Vec<(f64, Vec<T>)>,
    rhs_calls: u64,
    occ: HashMap<String, usize>,
}

impl<'a, T: Scalar> Exec<'a, T> {
    fn path(&mut self, id: &str) -> String {
        let o = self.occ.entry(id.to_string()).or_insert(0);
        let p = if *o == 0 {
            format!("{}/{}", self.macro_index, id)
        } else {
            format!("{}/{}#{}", self.macro_index, id, o)
        };
        *o += 1;
        p
    }

    fn recorded(&self, path: &str) -> Result<Option<&'a TapeRecord>, Error> {
        let replay: Option<&'a HashMap<&'a str, &'a TapeRecord>> = self.replay;
        match replay {
            None => Ok(None),
            Some(ix) => ix
                .get(path)
                .copied()
                .map(Some)
                .ok_or_else(|| Error::Run(format!("replay: no tape record for {path}"))),
        }
    }

    fn exec(
        &mut self,
        node: &SNode,
        u: &mut [T],
        t0: f64,
        t1: f64,
        is_root: bool,
    ) -> Result<(), Error> {
        // Every invocation's Δ is its interval's length, `t1 − t0`.
        let delta = t1 - t0;
        match node {
            SNode::Leaf { id, idx } => {
                let path = self.path(id);
                let leaf = &self.sched.leaves[*idx];
                let rec = self.recorded(&path)?;
                // Replay takes the recorded steps and reproduces the recorded
                // rejected count, which leaves no other trace (spec/tape.md §2).
                let mut replayed_rejected = None;
                let replay = match rec {
                    Some(TapeRecord::Leaf {
                        steps,
                        clips,
                        rejected,
                        ..
                    }) => {
                        replayed_rejected = Some(*rejected);
                        Some((steps.as_slice(), clips.as_slice()))
                    }
                    Some(_) => {
                        return Err(Error::Run(format!("replay: {path} is not a leaf record")))
                    }
                    None => None,
                };
                let stops: Vec<f64> = if is_root {
                    self.stops
                        .iter()
                        .copied()
                        .filter(|&s| s > t0 && s < t1)
                        .collect()
                } else {
                    vec![]
                };
                let call = LeafCall {
                    p: self.p,
                    t0,
                    t1,
                    delta,
                    macro_dt: self.macro_dt,
                    stops: &stops,
                    replay,
                    record_stages: self.adj.is_some(),
                };
                let out = leaf.invoke(u, &mut self.persist[*idx], &call)?;
                self.rhs_calls += out.rhs_calls;
                self.stop_states.extend(out.stop_states.iter().cloned());
                self.tape_out.push(TapeRecord::Leaf {
                    path: path.clone(),
                    t: t0,
                    dt: delta,
                    steps: out.steps.clone(),
                    accepted: out.accepted,
                    rejected: replayed_rejected.unwrap_or(out.rejected),
                    retcode: out.retcode,
                    clips: out.clips.clone(),
                });
                if out.retcode != Retcode::Success {
                    return Err(Error::Run(format!(
                        "leaf {} ({}) returned {:?} at {path}",
                        leaf.id, leaf.alg, out.retcode
                    )));
                }
                if let Some(adj) = &mut self.adj {
                    adj.push(AdjRecord::Leaf {
                        leaf: *idx,
                        steps: out.stage_data,
                    });
                }
            }
            SNode::Map { id, idx } => {
                let path = self.path(id);
                let m = &self.sched.maps[*idx];
                let dt = match m.bind {
                    BindKind::Leaf | BindKind::Node => delta,
                    BindKind::Macro => self.macro_dt,
                };
                let mut out = u.to_vec();
                let mut mask = Vec::new();
                match self.recorded(&path)? {
                    Some(TapeRecord::Map { mask: mk, .. }) => {
                        m.op.apply_frozen(&mut out, u, self.p, t0, dt, mk);
                        mask = mk.clone();
                    }
                    Some(_) => {
                        return Err(Error::Run(format!("replay: {path} is not a map record")))
                    }
                    None => m.op.apply(&mut out, u, self.p, t0, dt, &mut mask),
                }
                if let Some(adj) = &mut self.adj {
                    adj.push(AdjRecord::Map {
                        map: *idx,
                        u_before: u.to_vec(),
                        t: t0,
                        dt,
                        mask: mask.clone(),
                    });
                }
                u.copy_from_slice(&out);
                self.tape_out.push(TapeRecord::Map {
                    path,
                    t: t0,
                    dt,
                    mask,
                });
            }
            SNode::Sequence { children, .. } => {
                for c in children {
                    self.exec(c, u, t0, t1, false)?;
                }
            }
            SNode::Strang { children, .. } => {
                // c₁(Δ/2) … cₙ₋₁(Δ/2) cₙ(Δ) cₙ₋₁(Δ/2) … c₁(Δ/2) (run-config §4.7).
                // *Reading:* the first halves cover [t, tm], the middle child
                // [t, t+Δ] and the second halves [tm, t+Δ], tm = t + Δ/2.
                let tm = t0 + delta / 2.0;
                let n = children.len();
                for c in &children[..n - 1] {
                    self.exec(c, u, t0, tm, false)?;
                }
                self.exec(&children[n - 1], u, t0, t1, false)?;
                for c in children[..n - 1].iter().rev() {
                    self.exec(c, u, tm, t1, false)?;
                }
            }
            SNode::Subcycle { id, child, n } => {
                let path = self.path(id);
                // Substep i covers [t0 + i·δ, t0 + (i+1)·δ], the last ending at t1.
                let hs = delta / *n as f64;
                for k in 0..*n {
                    let a = t0 + k as f64 * hs;
                    let b = if k + 1 == *n {
                        t1
                    } else {
                        t0 + (k + 1) as f64 * hs
                    };
                    self.exec(child, u, a, b, false)?;
                }
                self.tape_out.push(TapeRecord::Subcycle {
                    path,
                    t: t0,
                    dt: delta,
                    n: *n,
                });
            }
        }
        Ok(())
    }
}

/// The adjoint functional (spec/adjoint.md §2.1), resolved against a problem.
#[derive(Debug, Clone)]
pub struct FunctionalSpec {
    /// DOFs reduced over.
    pub dofs: Vec<usize>,
    /// Weight of each DOF: 1 for `sum`, `1/|dofs|` for `mean`.
    pub weight: f64,
    /// Observation times (macro-step boundaries).
    pub times: Vec<f64>,
}

impl FunctionalSpec {
    /// `φ(u)`.
    pub fn phi<T: Scalar>(&self, u: &[T]) -> f64 {
        let mut s = 0.0;
        for &j in &self.dofs {
            s += self.weight * u[j].to_f64();
        }
        s
    }
}

/// A configured run: problem, schedule, interval, macro step, outputs and
/// the adjoint settings.
pub struct RunSpec<T: Scalar> {
    pub problem: ToyProblem<T>,
    pub schedule: Schedule<T>,
    pub tspan: (f64, f64),
    pub macro_dt: f64,
    /// Output times, ascending.
    pub output_times: Vec<f64>,
    /// Macro-step boundaries `t0 … t_f`.
    pub bounds: Vec<f64>,
    /// The reason for each macro step's size.
    pub reasons: Vec<String>,
    pub functional: Option<FunctionalSpec>,
    pub wrt_u0: bool,
    /// Per parameter: whether its gradient is requested.
    pub wrt_p: Vec<bool>,
}

/// The result of a forward run or a replay.
#[derive(Debug, Clone)]
pub struct ForwardRun<T> {
    /// `(t, u)` at each output time.
    pub outputs: Vec<(f64, Vec<T>)>,
    pub tape: Tape,
    /// Checkpoints at every macro-step boundary, the end included.
    pub checkpoints: Vec<Checkpoint<T>>,
    pub accepted: u64,
    pub rejected: u64,
    pub rhs_calls: u64,
}

impl<T: Scalar> ForwardRun<T> {
    /// The end state.
    pub fn final_state(&self) -> &[T] {
        &self.checkpoints.last().expect("a run has checkpoints").u
    }
}

impl<T: Scalar> RunSpec<T> {
    /// Builds a run from a run config, an interval and output times
    /// (default: the interval end).
    pub fn new(rc: &RunConfig, tspan: (f64, f64), saveat: Option<&[f64]>) -> Result<Self, Error> {
        let problem: ToyProblem<T> = crate::toy::toy_problem(&rc.model)?;
        if rc.operators.is_some() {
            return Err(Error::Config(
                "a toy problem's operators exist under their names; the run config has no operators key (spec/toy-problems.md)".into(),
            ));
        }
        for (k, v) in [
            ("partition", &rc.partition),
            ("precision", &rc.precision),
            ("output", &rc.output),
            ("restart", &rc.restart),
        ] {
            if v.is_some() {
                return Err(Error::Unimplemented(format!("run config key {k:?} at M1")));
            }
        }
        let step = rc.step.clone().unwrap_or_default();
        if step.limits.is_some() || step.hit_multiples.is_some() || step.resolution.is_some() {
            return Err(Error::Unimplemented(
                "step.limits, hit_multiples and resolution need the planner's limit operators (M2)"
                    .into(),
            ));
        }
        let macro_dt = match &step.dt {
            Some(v) => parse_duration(v)?,
            None => {
                return Err(Error::Unimplemented(
                    "a step without dt (the root leaf's steps as macro steps, D49) is not implemented at M1".into(),
                ))
            }
        };

        // The schedule first, so a leaf whose method is not implemented is
        // what a refusal names.
        let needs = Needs {
            adjoint: rc.adjoint.is_some(),
            wrt_p: rc
                .adjoint
                .as_ref()
                .is_some_and(|a| a.wrt.iter().any(|w| w.starts_with("params:"))),
        };
        let node = rc.schedule_node()?;
        let schedule = Schedule::build(&node, &problem, needs)?;

        // Adjoint settings.
        let mut wrt_u0 = false;
        let mut wrt_p = vec![false; problem.p.len()];
        let mut functional = None;
        // Output times: `saveat` (default the end), always with the end; the
        // initial state is output at t0 too (spec/toy-problems.md §3's example).
        let mut out_times: Vec<f64> = saveat.map(|s| s.to_vec()).unwrap_or_default();
        out_times.push(tspan.1);
        out_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        out_times.dedup();
        if let Some(adj) = &rc.adjoint {
            // Parameters bound by bind_dt are tape constants (spec/adjoint.md §5.8).
            let mut bound = vec![false; problem.p.len()];
            for l in &schedule.leaves {
                for &(i, _) in &l.binds {
                    bound[i] = true;
                }
            }
            for w in &adj.wrt {
                if w == "u0" {
                    wrt_u0 = true;
                } else if let Some(pat) = w.strip_prefix("params:") {
                    // A name refers to one parameter, refused when bound
                    // (run-config §8 rule 11); a pattern skips bound ones.
                    let is_pattern = pat == "numeric" || pat.contains('*');
                    let mut any = false;
                    for (i, name) in problem.param_names.iter().enumerate() {
                        if pat == "numeric" || glob(pat, name) {
                            if bound[i] {
                                if !is_pattern {
                                    return Err(Error::Config(format!(
                                        "wrt {w:?} names a parameter bound by bind_dt, a constant of the frozen tape (run-config §8 rule 11, spec/adjoint.md §5.8)"
                                    )));
                                }
                                continue;
                            }
                            wrt_p[i] = true;
                            any = true;
                        }
                    }
                    if !any {
                        return Err(Error::Config(format!("wrt {w:?} matches nothing")));
                    }
                } else {
                    return Err(Error::Unimplemented(format!(
                        "wrt {w:?}: gridded fields and forcing arrive at M3-M4"
                    )));
                }
            }
            let f = &adj.functional;
            let (name, mean) = match (&f.sum, &f.mean) {
                (Some(s), None) => (s, false),
                (None, Some(m)) => (m, true),
                _ => {
                    return Err(Error::Config(
                        "functional needs exactly one of sum and mean".into(),
                    ))
                }
            };
            let mut dofs: Vec<usize> = (0..problem.n)
                .filter(|&j| problem.variables[j] == name)
                .collect();
            if dofs.is_empty() {
                return Err(Error::Config(format!(
                    "functional quantity {name:?} is not a variable of {}",
                    problem.name
                )));
            }
            if let Some(wh) = &f.where_ {
                for (set, sel) in wh {
                    let cells = match (set.as_str(), &problem.cells) {
                        ("cell", Some(c)) => c,
                        _ => {
                            return Err(Error::Config(format!(
                                "{} has no index set {set:?}",
                                problem.name
                            )))
                        }
                    };
                    let (lo, hi) = match sel {
                        serde_json::Value::Number(n) => {
                            let i = n.as_u64().unwrap_or(0) as usize;
                            (i, i)
                        }
                        serde_json::Value::Array(a) if a.len() == 2 => (
                            a[0].as_u64().unwrap_or(0) as usize,
                            a[1].as_u64().unwrap_or(0) as usize,
                        ),
                        _ => return Err(Error::Config(format!("where {set}: {sel}"))),
                    };
                    // 1-based, inclusive.
                    dofs.retain(|&j| cells[j] + 1 >= lo && cells[j] < hi);
                }
            }
            let times = match f.over.as_str() {
                "final" => vec![tspan.1],
                "saveat" => out_times.clone(),
                "time" => {
                    return Err(Error::Unimplemented(
                        "functional over \"time\" (integrated auxiliary state, D26) is not implemented yet".into(),
                    ))
                }
                o => return Err(Error::Config(format!("functional over {o:?}"))),
            };
            let weight = if mean { 1.0 / dofs.len() as f64 } else { 1.0 };
            functional = Some(FunctionalSpec {
                dofs,
                weight,
                times,
            });
        }

        // Macro-step boundaries t0 + k·dt, the last one t_f exactly by the stop
        // rule of spec/methods.md §10 (multiplied, not accumulated).
        let (bounds, reasons) = macro_bounds(tspan.0, tspan.1, macro_dt);
        for &x in &out_times {
            if x <= tspan.0 || x > tspan.1 {
                return Err(Error::Config(format!(
                    "output time {x} is outside ({}, {}]",
                    tspan.0, tspan.1
                )));
            }
            // Output times are tstops: macro-step boundaries under a fixed
            // step.dt (run-config §8 rule 9), or stops inside the interval of a
            // root leaf that substeps (spec/methods.md §10). A "step" or
            // "accumulated" leaf takes exactly one step of the node's Δ
            // (run-config §4.2), so it cannot land on one.
            if !bounds.contains(&x) {
                let substeps = match schedule.root {
                    SNode::Leaf { idx, .. } => matches!(
                        schedule.leaves[idx].dt_mode,
                        DtMode::Adaptive | DtMode::Fixed(_)
                    ),
                    _ => false,
                };
                if !substeps {
                    return Err(Error::Config(format!(
                        "output time {x} is not a multiple of step.dt = {macro_dt}, and the schedule root is not a leaf that substeps to land on it (run-config §4.2, §8 rule 9)"
                    )));
                }
            }
        }
        if let Some(f) = &functional {
            for &x in &f.times {
                if !bounds.contains(&x) {
                    return Err(Error::Config(format!(
                        "observation time {x} is not a multiple of step.dt (run-config §8 rule 9)"
                    )));
                }
            }
        }
        Ok(RunSpec {
            problem,
            schedule,
            tspan,
            macro_dt,
            output_times: out_times,
            bounds,
            reasons,
            functional,
            wrt_u0,
            wrt_p,
        })
    }

    /// The initial state in `T`.
    pub fn u0(&self) -> Vec<T> {
        self.problem.u0.iter().map(|&x| T::from_f64(x)).collect()
    }

    /// Runs forward from `u0` with parameters `p`, recording the tape and
    /// a checkpoint at every macro-step boundary.
    pub fn forward(&self, u0: &[T], p: &[f64]) -> Result<ForwardRun<T>, Error> {
        self.run(u0, p, None)
    }

    /// Replays the run of `tape` from `u0` with parameters `p`: every
    /// recorded step, mask and macro step is held fixed (spec/tape.md §3).
    /// With a perturbed `u0` or `p` this is the frozen-tape map of
    /// spec/adjoint.md §2.2.
    pub fn replay(&self, u0: &[T], p: &[f64], tape: &Tape) -> Result<ForwardRun<T>, Error> {
        self.run(u0, p, Some(tape))
    }

    fn run(&self, u0: &[T], p: &[f64], tape: Option<&Tape>) -> Result<ForwardRun<T>, Error> {
        let t0 = self.tspan.0;
        let mut u = u0.to_vec();
        let mut persist = self.schedule.initial_persist();
        let mut out = ForwardRun {
            outputs: vec![(t0, u.clone())],
            tape: Tape::default(),
            checkpoints: vec![Checkpoint {
                macro_index: 0,
                t: t0,
                u: u.clone(),
                leaves: persist.clone(),
            }],
            accepted: 0,
            rejected: 0,
            rhs_calls: 0,
        };
        let index = tape.map(|tp| tp.index());
        for k in 0..self.bounds.len() - 1 {
            let (ta, tb) = (self.bounds[k], self.bounds[k + 1]);
            let delta = tb - ta;
            if let Some(ix) = &index {
                if !matches!(
                    ix.get(k.to_string().as_str()),
                    Some(TapeRecord::Macro { .. })
                ) {
                    return Err(Error::Run(format!("replay: no macro record {k}")));
                }
            }
            let (seg, rhs) = self.macro_step(
                k,
                &mut u,
                &mut persist,
                p,
                ta,
                tb,
                delta,
                index.as_ref(),
                None,
            )?;
            let (seg_tape, stop_states) = seg;
            out.rhs_calls += rhs;
            out.outputs.extend(stop_states);
            out.tape.records.extend(seg_tape);
            out.tape.records.push(TapeRecord::Macro {
                path: k.to_string(),
                t: ta,
                dt: delta,
                reason: self.reasons[k].clone(),
            });
            out.checkpoints.push(Checkpoint {
                macro_index: k + 1,
                t: tb,
                u: u.clone(),
                leaves: persist.clone(),
            });
            if self.output_times.contains(&tb) {
                out.outputs.push((tb, u.clone()));
            }
        }
        let (a, r) = out.tape.step_counts();
        out.accepted = a;
        out.rejected = r;
        out.outputs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        if out.outputs.len() != self.output_times.len() + 1 {
            return Err(Error::Run(format!(
                "outputs at {:?}, expected t0 and {:?}",
                out.outputs.iter().map(|o| o.0).collect::<Vec<_>>(),
                self.output_times
            )));
        }
        Ok(out)
    }

    /// Runs macro step `k` over `[t0, t1]`, from `u` and `persist`.
    /// Returns the tape records of its invocations, the states at interior
    /// output times and the right-hand-side count; with `adj`, records what
    /// the reverse pass needs.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn macro_step(
        &self,
        k: usize,
        u: &mut [T],
        persist: &mut [LeafPersist],
        p: &[f64],
        t0: f64,
        t1: f64,
        delta: f64,
        replay: Option<&HashMap<&str, &TapeRecord>>,
        adj: Option<&mut Vec<AdjRecord<T>>>,
    ) -> Result<((Vec<TapeRecord>, Vec<(f64, Vec<T>)>), u64), Error> {
        let stops: Vec<f64> = self
            .output_times
            .iter()
            .copied()
            .filter(|&x| x > t0 && x < t1)
            .collect();
        let mut ex = Exec {
            sched: &self.schedule,
            p,
            macro_index: k,
            macro_dt: delta,
            persist,
            replay,
            tape_out: Vec::new(),
            adj: adj.as_ref().map(|_| Vec::new()),
            stops,
            stop_states: Vec::new(),
            rhs_calls: 0,
            occ: HashMap::new(),
        };
        ex.exec(&self.schedule.root, u, t0, t1, true)?;
        if let (Some(a), Some(rec)) = (adj, ex.adj.take()) {
            *a = rec;
        }
        Ok(((ex.tape_out, ex.stop_states), ex.rhs_calls))
    }
}

/// The macro-step boundaries under a fixed `step.dt`: `t0 + k·dt`, the last
/// one `t_f` exactly by the stop rule of spec/methods.md §10, with each step's
/// reason: `"step.dt"`, or `"tstop"` for a last step shortened to land on
/// `t_f` (spec/run-config.md §5).
pub fn macro_bounds(t0: f64, tf: f64, dt: f64) -> (Vec<f64>, Vec<String>) {
    let mut bounds = vec![t0];
    let mut reasons = Vec::new();
    let mut k = 1.0;
    loop {
        let ta = *bounds.last().unwrap();
        let (h, is_stop) = crate::leaf::stop_rule(ta, dt, tf);
        if is_stop {
            bounds.push(tf);
            let tol = 100.0 * eps_f64(ta.abs().max(tf.abs()));
            reasons.push(if h < dt - tol { "tstop" } else { "step.dt" }.to_string());
            return (bounds, reasons);
        }
        bounds.push(t0 + k * dt);
        reasons.push("step.dt".to_string());
        k += 1.0;
    }
}

/// `*` matches any characters, including dots (run-config §7.2).
fn glob(pat: &str, s: &str) -> bool {
    match pat.find('*') {
        None => pat == s,
        Some(i) => {
            let (pre, rest) = (&pat[..i], &pat[i + 1..]);
            if !s.starts_with(pre) {
                return false;
            }
            let tail = &s[pre.len()..];
            (0..=tail.len()).any(|k| tail.is_char_boundary(k) && glob(rest, &tail[k..]))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn globs() {
        assert!(glob("k*", "k1"));
        assert!(glob("*", "a.b"));
        assert!(glob("a*c", "abbc"));
        assert!(!glob("a*c", "abb"));
    }
}
