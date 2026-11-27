//! Operators supplied by a plan (spec/operator-interface.md §8).
//!
//! [`load_plan`] reads a plan directory (spec/plan.md §1): `plan.json` and
//! each operator's two derived documents. Every operator the schedule names
//! is built from its entry `operators.<name>` and its documents, whose right
//! -hand sides are EarthSciAST's public reentrant program
//! (`rhs_program(esm_problem(path, tspan))`, native compiler, §8.1). The
//! result is a [`ToyProblem`] (the runtime's problem type: operators,
//! parameters, initial state, layout) plus the run config the schedule
//! interpreter runs, made from the plan's resolved schedule and step.
//!
//! Layout (§8.2): operators act on the whole model state in the plan's
//! canonical order (spec/plan.md §4): state names sorted by code point, each
//! array's cells column-major. A leaf advances only the union of its
//! operators' writes ([`crate::leaf::Bound`]). Parameters are the flattened model's scalar
//! parameters in code-point order of their names, at their defaults.

use crate::config::RunConfig;
use crate::forcing::{plan_forcing, DataOptions, ForcingTimeline, ModelData, PlanForcing};
use crate::operator::{Call, JacOverScratch, MatrixStorage, Operator, Structure};
use crate::toy::ToyProblem;
use crate::{Error, Scalar};
use earthsci_ast::{esm_problem, rhs_program, Compiler, ProblemInput, ProblemOptions, RhsProgram};
use earthsci_integrator_planner::numeric::{band_field, band_scatter, Dofs, Forcing, Program};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The model component that holds a Jacobian document's band fields
/// (spec/plan.md §6.2).
const JACOBIAN_MODEL: &str = earthsci_integrator_planner::derived::JACOBIAN_MODEL;

/// `esm_problem(path, tspan)` with the native compiler, then `rhs_program`.
/// A native refusal is a build failure: there is no fallback to the
/// interpreter (§8.1).
fn build_program(
    path: &Path,
    tspan: (f64, f64),
) -> Result<(earthsci_ast::EsmProblem, RhsProgram), String> {
    let prob = esm_problem(
        ProblemInput::Path(path),
        tspan,
        ProblemOptions {
            compiler: Some(Compiler::Native),
            ..Default::default()
        },
    )
    .map_err(|e| format!("EarthSciAST esm_problem: {e}"))?;
    let prog = rhs_program(&prob).map_err(|e| format!("EarthSciAST rhs_program: {e}"))?;
    Ok((prob, prog))
}

/// The canonical order of a program's states, leaving out the names `skip`
/// accepts.
fn program_dofs(prog: &RhsProgram, skip: impl Fn(&str) -> bool) -> Dofs {
    Dofs::from_shapes(
        prog.state_layout()
            .iter()
            .filter(|b| !skip(&b.name))
            .map(|b| (b.name.clone(), b.shape.clone())),
    )
}

/// Where a document's parameter takes its value from.
#[derive(Debug, Clone, Copy)]
enum ParamSource {
    /// The problem's `p`, at this index.
    P(usize),
    /// A fixed value: a used data-fed parameter the document declares as an
    /// ordinary parameter, which only `ic` equations read, at its `default`
    /// (spec/plan.md §6.3). It is not in `p` (§8.2).
    Fixed(f64),
}

/// A derived document's right-hand side (§8.1) and, for each of its
/// parameters, where its value comes from.
struct Document {
    prog: Program,
    p_index: Vec<ParamSource>,
    /// The forcing epochs of the run (§8.4).
    timeline: Arc<ForcingTimeline>,
}

impl Document {
    fn new(
        prog: RhsProgram,
        dofs: &Dofs,
        params: &[String],
        forcing: &BTreeMap<String, PlanForcing>,
        timeline: Arc<ForcingTimeline>,
    ) -> Result<Document, String> {
        let p_index = prog
            .parameters()
            .iter()
            .map(|ps| match params.iter().position(|q| *q == ps.name) {
                Some(i) => Ok(ParamSource::P(i)),
                None if forcing.contains_key(&ps.name) => ps
                    .default
                    .map(ParamSource::Fixed)
                    .ok_or_else(|| format!("its data-fed parameter {} has no default", ps.name)),
                None => Err(format!(
                    "its parameter {} is not one of the plan's",
                    ps.name
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        // §8.4.1: every forcing slot is a data-fed parameter of the plan.
        let prog = Program::from_program(prog, dofs)?;
        if let Some(n) = prog
            .forcing_slot_names()
            .into_iter()
            .find(|n| !forcing.contains_key(*n))
        {
            return Err(format!(
                "its forcing slot {n} is not a data-fed parameter of the plan's forcing"
            ));
        }
        Ok(Document {
            prog,
            p_index,
            timeline,
        })
    }

    /// Evaluates the document's right-hand side at the canonical state `u`
    /// with the forcing of the epoch `t` falls in (§8.4.1) and hands its
    /// program-layout result to `g`.
    fn eval<T: Scalar, R>(
        &self,
        u: &[T],
        p: &[f64],
        t: f64,
        g: impl FnOnce(&[f64]) -> R,
    ) -> Result<R, String> {
        self.eval_with(u, p, t, self.timeline.at(t), g)
    }

    /// [`Document::eval`] with the forcing `forcing`.
    fn eval_with<T: Scalar, R>(
        &self,
        u: &[T],
        p: &[f64],
        t: f64,
        forcing: &Forcing,
        g: impl FnOnce(&[f64]) -> R,
    ) -> Result<R, String> {
        self.prog.eval_with(
            |j| u[j].to_f64(),
            |i| match self.p_index[i] {
                ParamSource::P(k) => p[k],
                ParamSource::Fixed(x) => x,
            },
            t,
            forcing,
            g,
        )
    }
}

/// An operator built from a plan (§8).
pub struct PlannedOperator {
    name: String,
    /// Its structure over the model state (§2): `block_diagonal` when it is
    /// pointwise with `structure.block` and every model state shares one
    /// shape. Its `jac` fills storage of this structure; its `jac_over`
    /// fills a leaf's storage directly.
    structure: Structure,
    /// Pointwise, with `structure.block`: a leaf of such operators whose own
    /// state variables share a shape is `block_diagonal` (§8.3).
    blockable: bool,
    /// Every model state's first canonical DOF and extents, in canonical
    /// order (§8.2).
    states: Vec<(usize, Vec<usize>)>,
    depends_on_t: bool,
    linear_in_u: bool,
    /// Canonical DOFs written, ascending.
    writes: Vec<usize>,
    /// Program-layout offset of each written DOF in the right-hand-side
    /// document, beside its canonical DOF.
    rhs_writes: Vec<(usize, usize)>,
    rhs: Document,
    jacobian: Document,
    /// Every band point that lands on a DOF (§8.3):
    /// (program-layout offset of its `J_k` element, row DOF, column DOF).
    /// Ghost reads (a column outside its variable's extent) are left out.
    points: Vec<(usize, usize, usize)>,
    /// `jvp_u`/`vjp_u` accumulators, one per thread that is evaluating.
    acc: Mutex<Vec<Vec<f64>>>,
    /// The first evaluation fault (see [`Operator::fault`]).
    fault: Mutex<Option<String>>,
}

impl PlannedOperator {
    /// Records the first evaluation fault.
    fn record(&self, doc: &str, e: String) {
        let mut f = self.fault.lock().expect("fault");
        if f.is_none() {
            *f = Some(format!("its {doc} document: {e}"));
        }
    }

    /// `out += J v` (`transpose` false) or `out += Jᵀ v` from the assembled
    /// band entries.
    fn jprod<T: Scalar>(
        &self,
        out: &mut [T],
        u: &[T],
        p: &[f64],
        t: f64,
        v: &[T],
        transpose: bool,
    ) {
        let popped = self.acc.lock().expect("accumulator pool").pop();
        let mut acc = popped.unwrap_or_default();
        acc.clear();
        acc.resize(out.len(), 0.0);
        // Each output's contributions are summed in f64, then added into
        // out once.
        let done = self.jacobian.eval(u, p, t, |du| {
            for &(k, r, c) in &self.points {
                if transpose {
                    acc[c] += du[k] * v[r].to_f64();
                } else {
                    acc[r] += du[k] * v[c].to_f64();
                }
            }
        });
        if let Err(e) = done {
            // A fault of EarthSciAST's evaluation makes the product NaN,
            // which the leaf's step rejects.
            self.record("Jacobian", e);
            for &(_, r, c) in &self.points {
                acc[if transpose { c } else { r }] = f64::NAN;
            }
        }
        for (o, a) in out.iter_mut().zip(&acc) {
            if *a != 0.0 {
                *o += T::from_f64(*a);
            }
        }
        self.acc.lock().expect("accumulator pool").push(acc);
    }
}

impl<T: Scalar> Operator<T> for PlannedOperator {
    fn name(&self) -> &str {
        &self.name
    }
    fn structure(&self) -> Structure {
        self.structure
    }
    fn depends_on_t(&self) -> bool {
        self.depends_on_t
    }
    fn linear_in_u(&self) -> bool {
        self.linear_in_u
    }
    fn writes(&self) -> Vec<usize> {
        self.writes.clone()
    }
    fn structure_over(&self, dofs: &[usize], _n_model: usize) -> Result<Structure, String> {
        // §8.3: block_diagonal when every operator of the leaf is pointwise
        // with structure.block and the leaf's own state variables (those
        // with a DOF in `dofs`) are all arrays of one shape (ncells its
        // cells) or all scalars (ncells = 1), n being their number. The
        // model's other states do not matter.
        if !self.blockable {
            return Ok(Structure::Dense);
        }
        // The leaf's own states: those with a DOF in `dofs`. Each must be
        // whole (every DOF of it in the leaf) and they must make up the
        // leaf, so that, states being contiguous in canonical order, the
        // leaf's state vector is species-major with ncells cells: leaf index
        // k·ncells + c is cell c of the k-th own state.
        let mut own: Vec<&Vec<usize>> = Vec::new();
        let mut whole = 0;
        for (start, ext) in &self.states {
            let len: usize = ext.iter().product();
            let lo = dofs.partition_point(|&j| j < *start);
            let hi = dofs.partition_point(|&j| j < start + len);
            if hi > lo {
                if hi - lo != len {
                    return Ok(Structure::Dense);
                }
                own.push(ext);
                whole += len;
            }
        }
        let Some(&shape) = own.first() else {
            return Ok(Structure::Dense);
        };
        let ncells: usize = shape.iter().product();
        if whole != dofs.len() || own.iter().any(|e| *e != shape) {
            return Ok(Structure::Dense);
        }
        // A pointwise operator's entries in the leaf stay inside their cell.
        let loc = |j: usize| dofs.binary_search(&j).ok();
        for &(_, r, c) in &self.points {
            if let (Some(lr), Some(lc)) = (loc(r), loc(c)) {
                if lr % ncells != lc % ncells {
                    return Err(format!(
                        "structure.block, but its band entry ({r}, {c}) joins two cells \
                         (operator-interface.md §8.3)"
                    ));
                }
            }
        }
        Ok(Structure::BlockDiagonal {
            ncells,
            n: own.len(),
        })
    }
    fn reads(&self) -> Option<Vec<usize>> {
        // The columns of the Jacobian's band points: the DOFs the right-hand
        // side depends on.
        let mut r: Vec<usize> = self.points.iter().map(|&(_, _, c)| c).collect();
        r.sort_unstable();
        r.dedup();
        Some(r)
    }
    fn forcing_refresh_times(&self) -> Vec<f64> {
        // The run's epochs are shared by every operator that reads data
        // (§8.4.3); one that reads none adds no stops.
        if self.rhs.prog.forcing_slot_names().is_empty()
            && self.jacobian.prog.forcing_slot_names().is_empty()
        {
            return Vec::new();
        }
        self.rhs.timeline.refresh_times().to_vec()
    }
    fn provides(&self, call: Call) -> bool {
        // ft, vjp_p and the second-order calls wait for U17, U7 and U8
        // (operator-interface.md §8.3).
        matches!(call, Call::Rhs | Call::JvpU | Call::VjpU | Call::Jac)
    }
    fn fault(&self) -> Option<String> {
        self.fault.lock().expect("fault").clone()
    }

    fn rhs(&self, out: &mut [T], u: &[T], p: &[f64], t: f64) {
        let done = self.rhs.eval(u, p, t, |du| {
            for &(dof, off) in &self.rhs_writes {
                out[dof] += T::from_f64(du[off]);
            }
        });
        if let Err(e) = done {
            // NaN, which the leaf's step rejects.
            self.record("right-hand-side", e);
            for &(dof, _) in &self.rhs_writes {
                out[dof] = T::from_f64(f64::NAN);
            }
        }
    }
    fn jvp_u(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, v: &[T]) {
        self.jprod(out, u, p, t, v, false)
    }
    fn vjp_u(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, w: &[T]) {
        self.jprod(out, u, p, t, w, true)
    }
    fn jac_over(
        &self,
        out: &mut MatrixStorage<T>,
        dofs: &[usize],
        u: &[T],
        p: &[f64],
        t: f64,
        scratch: &mut JacOverScratch<T>,
    ) {
        // Each band point at the leaf's rows and columns, at its leaf
        // indices; a column outside the leaf is one of its constants. A
        // block_diagonal leaf's points stay inside their cell (checked by
        // structure_over), so either storage takes every point. They depend
        // only on the leaf's DOFs, so they are found at the first call.
        let at = scratch.points.get_or_insert_with(|| {
            self.points
                .iter()
                .filter_map(|&(k, r, c)| {
                    Some((
                        k,
                        dofs.binary_search(&r).ok()?,
                        dofs.binary_search(&c).ok()?,
                    ))
                })
                .collect()
        });
        let done = self.jacobian.eval(u, p, t, |du| {
            for &(k, a, b) in at.iter() {
                out.add(a, b, T::from_f64(du[k]));
            }
        });
        if let Err(e) = done {
            self.record("Jacobian", e);
            for &(_, a, b) in at.iter() {
                out.add(a, b, T::from_f64(f64::NAN));
            }
        }
    }
    fn jac_entries(
        &self,
        dofs: &[usize],
        u: &[T],
        p: &[f64],
        t: f64,
        f: &mut dyn FnMut(usize, usize, T),
    ) {
        let at = |&(k, r, c): &(usize, usize, usize)| {
            Some((
                k,
                dofs.binary_search(&r).ok()?,
                dofs.binary_search(&c).ok()?,
            ))
        };
        let done = self.jacobian.eval(u, p, t, |du| {
            for (k, a, b) in self.points.iter().filter_map(at) {
                f(a, b, T::from_f64(du[k]));
            }
        });
        if let Err(e) = done {
            self.record("Jacobian", e);
            for (_, a, b) in self.points.iter().filter_map(at) {
                f(a, b, T::from_f64(f64::NAN));
            }
        }
    }

    fn jac(&self, out: &mut MatrixStorage<T>, u: &[T], p: &[f64], t: f64) {
        // A block_diagonal operator's points stay inside their cell (checked
        // when it was built), so either storage takes every point.
        let done = self.jacobian.eval(u, p, t, |du| {
            for &(k, r, c) in &self.points {
                out.add(r, c, T::from_f64(du[k]));
            }
        });
        if let Err(e) = done {
            self.record("Jacobian", e);
            for &(_, r, c) in &self.points {
                out.add(r, c, T::from_f64(f64::NAN));
            }
        }
    }
}

/// A plan loaded for running: the problem its operators make and the run
/// config of its resolved schedule and step.
pub struct LoadedPlan<T: Scalar> {
    pub problem: ToyProblem<T>,
    pub run_config: RunConfig,
    /// The run interval the plan was made for (`time`, spec/plan.md §7.2).
    pub tspan: (f64, f64),
    /// The canonical DOFs of each state, by its flattened name, for writing
    /// results in another layout.
    pub state_dofs: BTreeMap<String, Vec<usize>>,
    /// The forcing epochs (spec/operator-interface.md §8.4.3); one empty
    /// epoch for a plan without `forcing`. Its refresh times are the
    /// `forcing_refresh` tstops.
    pub timeline: Arc<ForcingTimeline>,
}

fn refuse(op: &str, doc: &str, m: impl std::fmt::Display) -> Error {
    Error::Config(format!(
        "operator {op}: {doc}: {m} (operator-interface.md §8.1)"
    ))
}

/// How a schedule node uses an operator (spec/run-config.md §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    /// A term operator of a `leaf`, `coupled` or `split_explicit` node.
    Term,
    /// A map operator (`{"map": op}`, §4.10).
    Map,
    /// An algebraic operator (`{"solve": op}`, §4.10).
    Algebraic,
}

/// Adds `name` to `out` unless it is there already.
fn add_operator(out: &mut Vec<(String, Role)>, name: &str, role: Role) {
    if !out.iter().any(|(n, _)| n == name) {
        out.push((name.to_string(), role));
    }
}

/// The operators a schedule node uses, in tree order (depth first, each
/// node's keys in the order the plan writes them), each once, with how the
/// node uses it.
fn schedule_operators(node: &Value, out: &mut Vec<(String, Role)>) {
    match node {
        Value::Object(o) => {
            for (k, v) in o {
                let role = match k.as_str() {
                    "leaf" | "coupled" | "split_explicit" => Some(Role::Term),
                    "map" => Some(Role::Map),
                    "solve" => Some(Role::Algebraic),
                    _ => None,
                };
                match (role, v) {
                    (Some(r), Value::String(s)) => add_operator(out, s, r),
                    (Some(r), Value::Array(a)) if a.iter().all(Value::is_string) => {
                        for x in a {
                            add_operator(out, x.as_str().expect("a string"), r);
                        }
                    }
                    _ => schedule_operators(v, out),
                }
            }
        }
        Value::Array(a) => a.iter().for_each(|v| schedule_operators(v, out)),
        _ => {}
    }
}

/// Reads the plan directory `dir` and builds every operator its schedule
/// names (operator-interface.md §8). A plan with `forcing` needs the data
/// options' model: see [`load_plan_with`].
pub fn load_plan<T: Scalar>(dir: &Path) -> Result<LoadedPlan<T>, Error> {
    load_plan_with(dir, &DataOptions::default())
}

/// [`load_plan`], fetching the data of a plan with `forcing` as `data`
/// says (operator-interface.md §8.4).
pub fn load_plan_with<T: Scalar>(dir: &Path, data: &DataOptions) -> Result<LoadedPlan<T>, Error> {
    let cfg = |m: String| Error::Config(m);
    let text = std::fs::read_to_string(dir.join("plan.json"))
        .map_err(|e| cfg(format!("{}: {e}", dir.join("plan.json").display())))?;
    let plan: Value = serde_json::from_str(&text).map_err(|e| cfg(format!("plan.json: {e}")))?;
    if plan["esm_integrator"].as_str() != Some(crate::SPEC_VERSION) {
        return Err(cfg(format!(
            "plan.json: esm_integrator {} is not {}",
            plan["esm_integrator"],
            crate::SPEC_VERSION
        )));
    }
    let tspan = match (plan["time"]["start"].as_f64(), plan["time"]["end"].as_f64()) {
        (Some(a), Some(b)) => (a, b),
        _ => return Err(cfg("plan.json: no time interval".into())),
    };
    let mut names = Vec::new();
    schedule_operators(&plan["schedule"], &mut names);
    if names.is_empty() {
        return Err(cfg("plan.json: the schedule names no operator".into()));
    }
    for (name, role) in &names {
        let what = match role {
            Role::Term | Role::Algebraic => continue,
            Role::Map => "a map node",
        };
        return Err(Error::Unimplemented(format!(
            "operator {name}: the schedule runs it from {what}, which a plan's operators do not \
             support at M2 (operator-interface.md §8)"
        )));
    }
    // Each operator's documents, by the plan's `derived` list.
    let mut docs: HashMap<(String, String), String> = HashMap::new();
    for (path, d) in plan["derived"].as_object().into_iter().flatten() {
        if let (Some(op), Some(kind)) = (d["operator"].as_str(), d["kind"].as_str()) {
            docs.insert((op.to_string(), kind.to_string()), path.clone());
        }
    }

    // The canonical order, parameters, initial state and index sets come
    // from the first operator's right-hand-side document; every derived
    // document of a plan declares the same states (§8.2), so the others
    // must agree.
    struct Model {
        dofs: Dofs,
        params: Vec<String>,
        p: Vec<f64>,
        u0: Vec<f64>,
        /// Each state's index sets (empty for a scalar).
        shapes: BTreeMap<String, Vec<String>>,
        /// The number of cells of the states' one shape, when they share
        /// one (§8.3).
        ncells: Option<usize>,
    }
    let forcing = plan_forcing(&plan).map_err(cfg)?;
    let mut model: Option<Model> = None;
    // Every operator's programs first: their forcing slots are what the
    // data layer fills (§8.4.2 item 3).
    struct Built {
        name: String,
        rhs_path: String,
        jac_path: String,
        rhs_prog: RhsProgram,
        jac_prog: RhsProgram,
    }
    let mut built = Vec::new();
    let mut slots: BTreeMap<String, earthsci_ast::ForcingSlot> = BTreeMap::new();
    for (name, _) in &names {
        let entry = &plan["operators"][name];
        if !entry.is_object() {
            return Err(cfg(format!(
                "plan.json: the schedule names {name}, which has no operators entry"
            )));
        }
        // An algebraic operator's first document is its residual document
        // (spec/plan.md §6.4), evaluated as a right-hand side; a `solve` node
        // or a `coupled` node advances it.
        let first = match entry["kind"].as_str() {
            Some("rhs") => "rhs",
            Some("algebraic") => "residual",
            _ => {
                return Err(cfg(format!(
                    "operator {name}: its kind {} is not one a plan's runtime runs",
                    entry["kind"]
                )))
            }
        };
        let path_of = |kind: &str| -> Result<String, Error> {
            docs.get(&(name.clone(), kind.to_string()))
                .cloned()
                .ok_or_else(|| cfg(format!("operator {name}: the plan has no {kind} document")))
        };
        let (rhs_path, jac_path) = (path_of(first)?, path_of("jacobian")?);
        let (rhs_prob, rhs_prog) =
            build_program(&dir.join(&rhs_path), tspan).map_err(|e| refuse(name, &rhs_path, e))?;
        let (_, jac_prog) =
            build_program(&dir.join(&jac_path), tspan).map_err(|e| refuse(name, &jac_path, e))?;

        let this = program_dofs(&rhs_prog, |_| false);
        let jac_states = program_dofs(&jac_prog, |n| n.starts_with(&format!("{JACOBIAN_MODEL}.")));
        if jac_states.states != this.states {
            return Err(refuse(
                name,
                &jac_path,
                "its states are not those of the right-hand-side document",
            ));
        }
        // §8.4.1: a slot names a data-fed parameter of the plan, with its
        // shape.
        let sizes =
            index_set_sizes(&dir.join(&rhs_path)).map_err(|e| refuse(name, &rhs_path, e))?;
        for (prog, path) in [(&rhs_prog, &rhs_path), (&jac_prog, &jac_path)] {
            for slot in prog.forcing_slots() {
                let Some(f) = forcing.get(&slot.name) else {
                    return Err(refuse(
                        name,
                        path,
                        format!(
                            "its forcing slot {} is not in the plan's forcing",
                            slot.name
                        ),
                    ));
                };
                let want: Option<Vec<usize>> =
                    f.shape.iter().map(|x| sizes.get(x).copied()).collect();
                if want.as_ref() != Some(&slot.shape) {
                    return Err(refuse(
                        name,
                        path,
                        format!(
                            "its forcing slot {} has the shape {:?}, the plan's forcing {:?}",
                            slot.name, slot.shape, f.shape
                        ),
                    ));
                }
                slots.insert(slot.name.clone(), slot.clone());
            }
        }
        match &model {
            None => {
                let p = rhs_prog
                    .parameter_vector(&HashMap::new())
                    .map_err(|e| refuse(name, &rhs_path, format!("parameters: {e}")))?;
                // p is the scalar parameters other than data-fed ones (§8.2).
                let (params, p): (Vec<String>, Vec<f64>) = rhs_prog
                    .parameters()
                    .iter()
                    .map(|ps| ps.name.clone())
                    .zip(p)
                    .filter(|(n, _)| !forcing.contains_key(n))
                    .unzip();
                let u0 = initial_state(&rhs_prob, &this).map_err(|e| refuse(name, &rhs_path, e))?;
                let shapes = state_shapes(&dir.join(&rhs_path), &this)
                    .map_err(|e| refuse(name, &rhs_path, e))?;
                let exts: BTreeSet<&Vec<usize>> = this.states.values().map(|(_, e)| e).collect();
                let ncells = match exts.len() {
                    1 => exts.first().map(|e| e.iter().product::<usize>()),
                    _ => None,
                };
                model = Some(Model {
                    dofs: this,
                    params,
                    p,
                    u0,
                    shapes,
                    ncells,
                });
            }
            Some(m) if m.dofs.states != this.states => {
                return Err(refuse(
                    name,
                    &rhs_path,
                    "its states are not those of the other operators' documents",
                ))
            }
            Some(_) => {}
        }
        built.push(Built {
            name: name.clone(),
            rhs_path,
            jac_path,
            rhs_prog,
            jac_prog,
        });
    }
    let mut m = model.expect("at least one operator");

    // The data (§8.4): the plan's model, its providers, the forcing epochs
    // and, when an `ic` equation reads data, the initial state at t0.
    let timeline = if forcing.is_empty() {
        Arc::new(ForcingTimeline::empty(tspan.0))
    } else {
        let model_path = data.model.as_ref().ok_or_else(|| {
            cfg("the plan has forcing, so building it needs its model (DataOptions::model, the plan's inputs.model)".into())
        })?;
        let bytes =
            std::fs::read(model_path).map_err(|e| cfg(format!("{}: {e}", model_path.display())))?;
        let want = plan["inputs"]["model"]["sha256"]
            .as_str()
            .unwrap_or_default();
        let got = earthsci_integrator_planner::settings::sha256_hex(&bytes);
        if got != want {
            return Err(cfg(format!(
                "the model {} has the hash {got}, the plan's inputs.model {want}: the plan is stale",
                model_path.display()
            )));
        }
        let md = ModelData::load(model_path).map_err(cfg)?;
        let read: BTreeMap<String, PlanForcing> = forcing
            .iter()
            .filter(|(n, _)| slots.contains_key(*n))
            .map(|(n, f)| (n.clone(), f.clone()))
            .collect();
        let timeline = md
            .timeline(&read, &slots, tspan, data)
            .map_err(|e| cfg(format!("the data: {e}")))?;
        if md.ic_reads(&forcing).map_err(cfg)? {
            // §8.4.5: the initial state of EarthSciAST's Problem of the model
            // with the providers, permuted into canonical order.
            let prob = md
                .problem(&forcing, tspan, data)
                .map_err(|e| cfg(format!("the data: {e}")))?;
            m.u0 = initial_state(&prob, &m.dofs)
                .map_err(|e| cfg(format!("the initial state with data: {e}")))?;
        }
        Arc::new(timeline)
    };

    let mut operators: Vec<Arc<dyn Operator<T>>> = Vec::new();
    // Each algebraic operator's unknowns and bounds (spec/plan.md §5.9.7).
    let mut algebraic = BTreeMap::new();
    for b in &built {
        let entry = &plan["operators"][&b.name];
        if entry["kind"].as_str() != Some("algebraic") {
            continue;
        }
        let doc: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join(&b.rhs_path))
                .map_err(|e| refuse(&b.name, &b.rhs_path, e))?,
        )
        .map_err(|e| refuse(&b.name, &b.rhs_path, e))?;
        let mut info = crate::algebraic::AlgebraicInfo {
            dofs: vec![],
            names: vec![],
            lower: vec![],
            upper: vec![],
            frozen: None,
            blocks: None,
        };
        for z in entry["writes"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            let r = m
                .dofs
                .dofs_of(z)
                .ok_or_else(|| refuse(&b.name, &b.rhs_path, format!("no state {z}")))?;
            let ext: Vec<usize> = m.dofs.states[z].1.clone();
            let n = r.len();
            info.dofs.extend(r);
            info.names.extend(std::iter::repeat_n(z.to_string(), n));
            let bnds = &entry["bounds"][z];
            for (key, dest) in [("lb", &mut info.lower), ("ub", &mut info.upper)] {
                let v = &bnds[key];
                let src: Vec<Option<crate::algebraic::Bound>> = if v.is_null() {
                    vec![None; n]
                } else if let Some(x) = v.as_f64() {
                    vec![Some(crate::algebraic::Bound::Const(x)); n]
                } else {
                    let pn = v.as_str().unwrap_or_default();
                    match m.params.iter().position(|x| x == pn) {
                        Some(i) => vec![Some(crate::algebraic::Bound::Param(i)); n],
                        None => crate::algebraic::array_default(&doc, pn, &ext)
                            .map_err(|e| refuse(&b.name, &b.rhs_path, e))?
                            .into_iter()
                            .map(|x| Some(crate::algebraic::Bound::Const(x)))
                            .collect(),
                    }
                };
                dest.extend(src);
            }
        }
        // The frozen documents (spec/plan.md §6.5), when the plan has them.
        if let Some(rp) = docs.get(&(b.name.clone(), "frozen_residual".to_string())) {
            let jp = docs.get(&(b.name.clone(), "frozen_jacobian".to_string()));
            let f = PlannedFrozen::load(
                dir, tspan, &b.name, entry, &m.dofs, &m.params, &forcing, &timeline, &info.dofs,
                rp, jp,
            )?;
            info.frozen = Some(crate::algebraic::Frozen(Arc::new(f)));
        }
        info.blocks = solve_blocks(&b.name, entry, &m.dofs, &doc, &info.names)
            .map_err(|e| refuse(&b.name, &b.rhs_path, format!("its block structure: {e}")))?;
        algebraic.insert(b.name.clone(), info);
    }
    for b in built {
        let (name, rhs_path, jac_path) = (&b.name, &b.rhs_path, &b.jac_path);
        let entry = &plan["operators"][name];
        let rhs = Document::new(b.rhs_prog, &m.dofs, &m.params, &forcing, timeline.clone())
            .map_err(|e| refuse(name, rhs_path, e))?;
        let jacobian = Document::new(b.jac_prog, &m.dofs, &m.params, &forcing, timeline.clone())
            .map_err(|e| refuse(name, jac_path, e))?;
        // A document that cannot be evaluated at the initial state is
        // refused here, with EarthSciAST's message, rather than failing
        // every step of the run.
        for (doc, path) in [(&rhs, rhs_path), (&jacobian, jac_path)] {
            doc.eval(&m.u0, &m.p, tspan.0, |_| ())
                .map_err(|e| refuse(name, path, format!("at the initial state: {e}")))?;
        }
        operators.push(Arc::new(build_operator(
            name, entry, &m.dofs, m.ncells, rhs, jacobian,
        )?));
    }

    // The problem: names and layout for run configs' variable references.
    // The toy problems' one index set is `cell` (spec/toy-problems.md), so
    // a cell index is given exactly when every state's shape is `[cell]`.
    let mut variables = vec![String::new(); m.dofs.n];
    let mut cell_of = vec![0usize; m.dofs.n];
    let mut state_dofs = BTreeMap::new();
    for name in m.dofs.states.keys() {
        let r = m.dofs.dofs_of(name).expect("a state");
        for (k, j) in r.clone().enumerate() {
            variables[j] = name.clone();
            cell_of[j] = k;
        }
        state_dofs.insert(name.clone(), r.collect());
    }
    let by_cell = m.shapes.values().all(|s| s.len() == 1 && s[0] == "cell");
    let problem = ToyProblem {
        name: plan["inputs"]["model"]["path"]
            .as_str()
            .unwrap_or("plan")
            .to_string(),
        n: m.dofs.n,
        param_names: m.params,
        p: m.p,
        u0: m.u0,
        tspan,
        variables,
        cells: by_cell.then_some(cell_of),
        operators,
        maps: vec![],
        invariants: vec![],
        algebraic,
    };

    // The run config of the resolved schedule and step (spec/plan.md §7.1,
    // §7.3). The plan's partition is one shard per leaf at M2 and its
    // precision f64 throughout; neither is read.
    let mut rc = json!({
        "esm_integrator": crate::SPEC_VERSION,
        "model": problem.name,
        "schedule": plan["schedule"],
    });
    if !plan["step"].is_null() {
        rc["step"] = plan["step"].clone();
    }
    let run_config = RunConfig::from_value(&rc)?;
    Ok(LoadedPlan {
        problem,
        run_config,
        tspan,
        state_dofs,
        timeline,
    })
}

/// EarthSciAST's initial state of a right-hand-side document (defaults and
/// `ic` equations), in canonical order (spec/plan.md §4, §6.1.2).
fn initial_state(prob: &earthsci_ast::EsmProblem, dofs: &Dofs) -> Result<Vec<f64>, String> {
    let u0 =
        earthsci_ast::initial_state(prob).map_err(|e| format!("EarthSciAST initial_state: {e}"))?;
    let names = prob.state_variable_names();
    if names.len() != u0.len() {
        return Err(format!(
            "EarthSciAST's initial state has {} values for {} state elements",
            u0.len(),
            names.len()
        ));
    }
    let at: HashMap<&str, f64> = names.iter().map(String::as_str).zip(u0).collect();
    dofs.element_names()
        .iter()
        .map(|n| {
            at.get(n.as_str())
                .copied()
                .ok_or_else(|| format!("EarthSciAST's initial state has no {n}"))
        })
        .collect()
}

/// The sizes of a derived document's index sets (spec/plan.md §6.1: every
/// one an interval).
fn index_set_sizes(path: &Path) -> Result<BTreeMap<String, usize>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let doc: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    Ok(doc["index_sets"]
        .as_object()
        .into_iter()
        .flatten()
        .filter_map(|(k, v)| Some((k.clone(), v["size"].as_u64()? as usize)))
        .collect())
}

/// Each state's index sets, as the flattened document declares them
/// (spec/plan.md §6.1: a derived document writes every state's shape).
fn state_shapes(path: &Path, dofs: &Dofs) -> Result<BTreeMap<String, Vec<String>>, String> {
    let file = earthsci_ast::load_path(path).map_err(|e| format!("EarthSciAST load: {e}"))?;
    let flat = earthsci_ast::flatten(&file).map_err(|e| format!("EarthSciAST flatten: {e}"))?;
    dofs.states
        .iter()
        .map(|(name, (_, ext))| {
            let v = flat
                .state_variables
                .get(name)
                .ok_or_else(|| format!("the flattened document has no state {name}"))?;
            let shape = v.shape.clone().unwrap_or_default();
            if shape.len() != ext.len() {
                return Err(format!(
                    "the state {name} has the index sets {shape:?} for {} dimensions",
                    ext.len()
                ));
            }
            Ok((name.clone(), shape))
        })
        .collect()
}

/// An algebraic operator's frozen documents (spec/plan.md §6.5), bound to
/// their own canonical layout: the model's states and the frozen states
/// `FrozenEval.frozen_m`.
struct PlannedFrozen {
    resid: Document,
    jac: Option<Document>,
    /// The length of the frozen layout.
    nf: usize,
    /// Each model DOF's DOF in the frozen layout.
    tolf: Vec<usize>,
    /// Per frozen observed: its frozen state's DOFs in the frozen layout,
    /// and its slots in the frozen residual document's output.
    fdofs: Vec<Vec<usize>>,
    fslots: Vec<Vec<usize>>,
    /// The frozen Jacobian's band points among the unknowns: (output slot,
    /// row position, column position).
    points: Vec<(usize, usize, usize)>,
}

impl PlannedFrozen {
    #[allow(clippy::too_many_arguments)]
    fn load(
        dir: &Path,
        tspan: (f64, f64),
        name: &str,
        entry: &Value,
        dofs: &Dofs,
        params: &[String],
        forcing: &BTreeMap<String, PlanForcing>,
        timeline: &Arc<ForcingTimeline>,
        zdofs: &[usize],
        rpath: &str,
        jpath: Option<&String>,
    ) -> Result<PlannedFrozen, Error> {
        let (_, rprog) =
            build_program(&dir.join(rpath), tspan).map_err(|e| refuse(name, rpath, e))?;
        let lf = program_dofs(&rprog, |_| false);
        let mut tolf = vec![0usize; dofs.n];
        for (st, (start, ext)) in &dofs.states {
            let r = lf.dofs_of(st).ok_or_else(|| {
                refuse(name, rpath, format!("it does not declare the state {st}"))
            })?;
            if lf.states[st].1 != *ext {
                return Err(refuse(
                    name,
                    rpath,
                    format!("its state {st} has other extents than the model's"),
                ));
            }
            for (k, j) in r.enumerate() {
                tolf[start + k] = j;
            }
        }
        let resid = Document::new(rprog, &lf, params, forcing, timeline.clone())
            .map_err(|e| refuse(name, rpath, e))?;
        let (mut fdofs, mut fslots): (Vec<Vec<usize>>, Vec<Vec<usize>>) = (vec![], vec![]);
        loop {
            let x = format!(
                "{}.frozen_{}",
                earthsci_integrator_planner::derived::FROZEN_COMPONENT,
                fdofs.len() + 1
            );
            let Some(r) = lf.dofs_of(&x) else { break };
            let (off, _) = resid
                .prog
                .block_range(&x)
                .ok_or_else(|| refuse(name, rpath, format!("no block {x}")))?;
            fslots.push(r.clone().map(|j| off + j - r.start).collect());
            fdofs.push(r.collect());
        }
        if lf.n != dofs.n + fdofs.iter().map(Vec::len).sum::<usize>() {
            return Err(refuse(
                name,
                rpath,
                "it declares states other than the model's and its frozen states",
            ));
        }
        let mut points = vec![];
        let jac = match jpath {
            None => None,
            Some(jp) => {
                let (_, jprog) =
                    build_program(&dir.join(jp), tspan).map_err(|e| refuse(name, jp, e))?;
                let jac = Document::new(jprog, &lf, params, forcing, timeline.clone())
                    .map_err(|e| refuse(name, jp, e))?;
                let zpos: HashMap<usize, usize> = zdofs
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| (tolf[d], i))
                    .collect();
                for (k, b) in entry["frozen_bands"]
                    .as_array()
                    .ok_or_else(|| refuse(name, jp, "the plan entry has no frozen_bands"))?
                    .iter()
                    .enumerate()
                {
                    let field = band_field(k + 1);
                    let (base, len) = jac
                        .prog
                        .block_range(&field)
                        .ok_or_else(|| refuse(name, jp, format!("it has no {field}")))?;
                    let mut with_coef = b.clone();
                    with_coef["coef"] = json!(0);
                    let en = earthsci_astdiff::parse_band(&with_coef)
                        .map_err(|e| refuse(name, jp, format!("band {}: {e}", k + 1)))?;
                    let scatter = band_scatter(&lf, &en, len, &field)
                        .map_err(|e| refuse(name, jp, format!("band {}: {e}", k + 1)))?;
                    for (off, r, c) in scatter {
                        if let (Some(&ri), Some(&ci)) = (zpos.get(&r), zpos.get(&c)) {
                            points.push((base + off, ri, ci));
                        }
                    }
                }
                Some(jac)
            }
        };
        Ok(PlannedFrozen {
            resid,
            jac,
            nf: lf.n,
            tolf,
            fdofs,
            fslots,
            points,
        })
    }

    /// The frozen layout's state at the model state `x`, its frozen states
    /// zero.
    fn state(&self, x: &[f64]) -> Vec<f64> {
        let mut uf = vec![0.0; self.nf];
        for (i, &j) in self.tolf.iter().enumerate() {
            uf[j] = x[i];
        }
        uf
    }
}

impl crate::algebraic::FrozenEval for PlannedFrozen {
    fn values(&self, x: &[f64], p: &[f64], t: f64) -> Result<Vec<Vec<f64>>, String> {
        let uf = self.state(x);
        self.resid.eval(&uf, p, t, |du| {
            self.fslots
                .iter()
                .map(|s| s.iter().map(|&k| du[k]).collect())
                .collect()
        })
    }

    fn has_jacobian(&self) -> bool {
        self.jac.is_some()
    }

    fn jacobian(&self, x: &[f64], p: &[f64], t: f64) -> Result<crate::algebraic::Entries, String> {
        let jac = self.jac.as_ref().ok_or("no frozen Jacobian document")?;
        let mut uf = self.state(x);
        for (ds, v) in self.fdofs.iter().zip(self.values(x, p, t)?) {
            for (&d, y) in ds.iter().zip(v) {
                uf[d] = y;
            }
        }
        jac.eval(&uf, p, t, |du| {
            let mut a = crate::algebraic::Entries::new();
            for &(s, r, c) in &self.points {
                *a.entry((r, c)).or_insert(0.0) += du[s];
            }
            a
        })
    }
}

/// The index sets of variable `name` as declared in a derived document.
fn declared_shape(doc: &Value, name: &str) -> Result<Vec<String>, String> {
    let segs: Vec<&str> = name.split('.').collect();
    let mut node = &doc["models"][segs[0]];
    for s in &segs[1..segs.len() - 1] {
        node = &node["subsystems"][*s];
    }
    let v = &node["variables"][segs[segs.len() - 1]];
    if v.is_null() {
        return Err(format!("the residual document does not declare {name}"));
    }
    Ok(v["shape"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|x| x.as_str().map(str::to_string))
        .collect())
}

/// The blocks of an algebraic operator's unknowns (`names`, the unknown
/// variable of each position), from its structure's `block` (spec/plan.md
/// §5.4), else `None`: per cell, the positions among the unknowns of the
/// block's elements, in the block's order. A `cell_block` plan lists its
/// elements (a variable and its indices along the axes inside the cell)
/// and its cell axes; a `block_diagonal` plan has one element per
/// variable, in its `ordering` of `vars`, and every axis is a cell axis.
/// Elements of variables the operator only reads are left out. Blocks are
/// in column-major order of their cells.
fn solve_blocks(
    name: &str,
    entry: &Value,
    dofs: &Dofs,
    doc: &Value,
    names: &[String],
) -> Result<Option<Vec<Vec<usize>>>, String> {
    let _ = name;
    let blk = &entry["structure"]["block"];
    if blk.is_null() {
        return Ok(None);
    }
    let strs = |v: &Value| -> Vec<String> {
        v.as_array()
            .into_iter()
            .flatten()
            .filter_map(|x| x.as_str().map(str::to_string))
            .collect()
    };
    let ints = |v: &Value| -> Vec<i64> {
        v.as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_i64)
            .collect()
    };
    type Elements = Vec<(String, Vec<i64>)>;
    let (cell_axes, elems): (Option<Vec<String>>, Elements) =
        if let Some(es) = blk["elements"].as_array() {
            (
                Some(strs(&blk["cell_axes"])),
                es.iter()
                    .map(|e| (e[0].as_str().unwrap_or_default().to_string(), ints(&e[1])))
                    .collect(),
            )
        } else {
            let vars = strs(&blk["vars"]);
            let ord = ints(&blk["ordering"]);
            (
                None,
                ord.iter()
                    .map(|&o| (vars[(o - 1) as usize].clone(), vec![]))
                    .collect(),
            )
        };
    // The block's elements also hold the variables the operator only reads;
    // the solve's blocks are its unknowns', in the block's order.
    let elems: Vec<(String, Vec<i64>)> =
        elems.into_iter().filter(|e| names.contains(&e.0)).collect();
    let at: BTreeMap<&(String, Vec<i64>), usize> =
        elems.iter().enumerate().map(|(i, e)| (e, i)).collect();
    let mut groups: BTreeMap<Vec<i64>, Vec<usize>> = BTreeMap::new();
    let mut k = 0usize;
    let mut seen: Vec<&String> = vec![];
    for z in names {
        if seen.contains(&z) {
            continue;
        }
        seen.push(z);
        let shape = declared_shape(doc, z)?;
        let cd: Vec<usize> = match &cell_axes {
            None => (0..shape.len()).collect(),
            Some(ax) => ax
                .iter()
                .map(|a| {
                    shape
                        .iter()
                        .position(|s| s == a)
                        .ok_or_else(|| format!("{z} does not span the cell axis {a}"))
                })
                .collect::<Result<_, _>>()?,
        };
        let inner: Vec<usize> = (0..shape.len()).filter(|d| !cd.contains(d)).collect();
        let ext = &dofs.states[z].1;
        let ranges: Vec<(i64, i64)> = ext.iter().map(|&n| (1, n as i64)).collect();
        for c in earthsci_integrator_planner::numeric::cells(&ranges) {
            let key: (String, Vec<i64>) = (z.clone(), inner.iter().map(|&d| c[d]).collect());
            let e = *at
                .get(&key)
                .ok_or_else(|| format!("{z}{c:?} is no element of a block"))?;
            // Keyed by the cell's indices reversed: column-major order.
            let cell: Vec<i64> = cd.iter().rev().map(|&d| c[d]).collect();
            let g = groups
                .entry(cell)
                .or_insert_with(|| vec![usize::MAX; elems.len()]);
            if g[e] != usize::MAX {
                return Err(format!("two unknowns are element {} of one block", e + 1));
            }
            g[e] = k;
            k += 1;
        }
    }
    if k != names.len() {
        return Err("the unknowns are not those of the plan entry".into());
    }
    let out: Vec<Vec<usize>> = groups.into_values().collect();
    if out.iter().any(|g| g.contains(&usize::MAX)) {
        return Err("a cell lacks an element of its block".into());
    }
    Ok(Some(out))
}

/// One operator from its plan entry and documents (§8.2, §8.3). `ncells` is
/// the number of cells of the states' one shape, when they share one.
fn build_operator(
    name: &str,
    entry: &Value,
    dofs: &Dofs,
    ncells: Option<usize>,
    rhs: Document,
    jacobian: Document,
) -> Result<PlannedOperator, Error> {
    let bad = |m: String| Error::Config(format!("operator {name}: {m}"));
    let st = &entry["structure"];
    let flag = |k: &str| {
        st[k]
            .as_bool()
            .ok_or_else(|| bad(format!("structure.{k} missing")))
    };
    let (depends_on_t, linear_in_u) = (flag("depends_on_t")?, flag("linear_in_u")?);

    // writes: every cell of every variable in the plan's list, each beside
    // its offset in the right-hand-side document's layout.
    let mut rhs_writes = Vec::new();
    for v in entry["writes"]
        .as_array()
        .ok_or_else(|| bad("no writes".into()))?
    {
        let w = v.as_str().ok_or_else(|| bad(format!("writes entry {v}")))?;
        let r = dofs
            .dofs_of(w)
            .ok_or_else(|| bad(format!("writes {w}, which is not a state")))?;
        let (off, _) = rhs.prog.block_range(w).expect("every state is a block");
        rhs_writes.extend(r.clone().map(|j| (j, off + j - r.start)));
    }
    rhs_writes.sort_unstable();
    rhs_writes.dedup();
    let writes: Vec<usize> = rhs_writes.iter().map(|&(j, _)| j).collect();

    // The band points, scattered once (spec/plan.md §6.2).
    let mut points = Vec::new();
    for (k, b) in st["bands"]
        .as_array()
        .ok_or_else(|| bad("no bands".into()))?
        .iter()
        .enumerate()
    {
        let field = band_field(k + 1);
        let (base, len) = jacobian
            .prog
            .block_range(&field)
            .ok_or_else(|| bad(format!("its Jacobian document has no {field}")))?;
        let mut with_coef = b.clone();
        with_coef["coef"] = json!(0);
        let en = earthsci_astdiff::parse_band(&with_coef)
            .map_err(|e| bad(format!("band {}: {e}", k + 1)))?;
        let scatter = band_scatter(dofs, &en, len, &field)
            .map_err(|e| bad(format!("band {}: {e}", k + 1)))?;
        points.extend(scatter.into_iter().map(|(off, r, c)| (base + off, r, c)));
    }

    // The structure over the model state: block_diagonal when the operator
    // is pointwise with structure.block and the model's states are all
    // arrays of one shape (ncells the cells of that shape) or all scalars
    // (ncells = 1), with n the number of states. A leaf's own structure
    // looks at its own states only (structure_over, §8.3).
    let blockable =
        entry["locality"]["class"].as_str() == Some("pointwise") && st.get("block").is_some();
    let structure = match ncells {
        Some(ncells) if blockable => {
            if let Some((_, r, c)) = points.iter().find(|(_, r, c)| r % ncells != c % ncells) {
                return Err(bad(format!(
                    "structure.block, but its band entry ({r}, {c}) joins two cells"
                )));
            }
            Structure::BlockDiagonal {
                ncells,
                n: dofs.states.len(),
            }
        }
        _ => Structure::Dense,
    };
    Ok(PlannedOperator {
        name: name.to_string(),
        structure,
        blockable,
        states: dofs.states.values().cloned().collect(),
        depends_on_t,
        linear_in_u,
        writes,
        rhs_writes,
        rhs,
        jacobian,
        points,
        acc: Mutex::new(Vec::new()),
        fault: Mutex::new(None),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_operators_in_tree_order() {
        let sched = json!({"sequence": [
            {"leaf": ["kinetics", "advection"], "frozen": [{"every": "advection"}]},
            {"map": "clip"},
            {"solve": "balance", "unknowns": ["x"]},
            {"leaf": "kinetics"}
        ]});
        let mut out = Vec::new();
        schedule_operators(&sched, &mut out);
        assert_eq!(
            out,
            [
                ("kinetics".to_string(), Role::Term),
                ("advection".to_string(), Role::Term),
                ("clip".to_string(), Role::Map),
                ("balance".to_string(), Role::Algebraic),
            ]
        );
    }
}
