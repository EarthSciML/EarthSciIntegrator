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
//! array's cells column-major. Parameters are the flattened model's scalar
//! parameters in code-point order of their names, at their defaults.

use crate::config::RunConfig;
use crate::operator::{Call, MatrixStorage, Operator, Structure};
use crate::toy::ToyProblem;
use crate::{Error, Scalar};
use earthsci_ast::{
    esm_problem, rhs_program, Compiler, ForcingSet, ProblemInput, ProblemOptions, RhsProgram,
    RhsWorkspace,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The model component that holds a Jacobian document's band fields
/// (spec/plan.md §6.2).
const JACOBIAN_MODEL: &str = "JacobianEval";

/// One state of the canonical order: its first DOF and its extents (empty
/// for a scalar).
#[derive(Debug, Clone, PartialEq)]
struct StateSlot {
    start: usize,
    ext: Vec<usize>,
}

/// The canonical state order (spec/plan.md §4).
#[derive(Debug, Clone, PartialEq)]
pub struct Canonical {
    states: BTreeMap<String, StateSlot>,
    /// The number of DOFs.
    pub n: usize,
}

impl Canonical {
    /// The canonical order of `blocks` `(name, extents)`, leaving out the
    /// names `skip` accepts.
    fn new<'a>(
        blocks: impl Iterator<Item = (&'a str, &'a [usize])>,
        skip: impl Fn(&str) -> bool,
    ) -> Self {
        let sorted: BTreeMap<String, Vec<usize>> = blocks
            .filter(|(n, _)| !skip(n))
            .map(|(n, e)| (n.to_string(), e.to_vec()))
            .collect();
        let mut states = BTreeMap::new();
        let mut n = 0;
        for (name, ext) in sorted {
            let len = ext.iter().product::<usize>();
            states.insert(name, StateSlot { start: n, ext });
            n += len;
        }
        Canonical { states, n }
    }

    /// The DOF of `state` at the 1-based `cell`, `None` outside its extents
    /// (a ghost read).
    fn dof(&self, state: &str, cell: &[i64]) -> Option<usize> {
        let s = self.states.get(state)?;
        if cell.len() != s.ext.len() {
            return None;
        }
        let (mut off, mut stride) = (0usize, 1usize);
        for (c, n) in cell.iter().zip(&s.ext) {
            if *c < 1 || *c as usize > *n {
                return None;
            }
            off += (*c as usize - 1) * stride;
            stride *= n;
        }
        Some(s.start + off)
    }

    /// Every DOF's element name as EarthSciAST writes it (`X`, `X[i,j]`).
    fn element_names(&self) -> Vec<String> {
        let mut out = Vec::with_capacity(self.n);
        for (name, s) in &self.states {
            if s.ext.is_empty() {
                out.push(name.clone());
                continue;
            }
            for cell in cells(&s.ext.iter().map(|n| (1, *n as i64)).collect::<Vec<_>>()) {
                let idx: Vec<String> = cell.iter().map(i64::to_string).collect();
                out.push(format!("{name}[{}]", idx.join(",")));
            }
        }
        out
    }

    /// The canonical DOFs of every cell of `state`.
    fn dofs_of(&self, state: &str) -> Option<std::ops::Range<usize>> {
        let s = self.states.get(state)?;
        Some(s.start..s.start + s.ext.iter().product::<usize>())
    }
}

/// Every cell of the box `ranges` (1-based, inclusive), first index fastest.
fn cells(ranges: &[(i64, i64)]) -> Vec<Vec<i64>> {
    if ranges.iter().any(|(lo, hi)| lo > hi) {
        return vec![];
    }
    let mut out = Vec::new();
    let mut cell: Vec<i64> = ranges.iter().map(|(lo, _)| *lo).collect();
    loop {
        out.push(cell.clone());
        let mut k = 0;
        loop {
            if k == cell.len() {
                return out;
            }
            cell[k] += 1;
            if cell[k] <= ranges[k].1 {
                break;
            }
            cell[k] = ranges[k].0;
            k += 1;
        }
    }
}

/// An evaluation workspace with its program-layout `u` and `du`.
type Scratch = (RhsWorkspace, Vec<f64>, Vec<f64>);

/// A derived document's right-hand side as EarthSciAST builds it (§8.1),
/// with the permutation between its own state layout and canonical order.
struct Document {
    prog: RhsProgram,
    forcing: ForcingSet,
    /// Per canonical state: (canonical start, program offset, length).
    map: Vec<(usize, usize, usize)>,
    /// Every block of the program's state layout: offset and length.
    blocks: HashMap<String, (usize, usize)>,
    /// For each of the program's parameters, its index in the problem's `p`.
    p_index: Vec<usize>,
    /// Workspaces with their program-layout `u` and `du`, one per thread
    /// that is evaluating; an `RhsWorkspace` is not `Sync`.
    pool: Mutex<Vec<Scratch>>,
}

/// `esm_problem(path, tspan)` with the native compiler, then `rhs_program`.
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

impl Document {
    fn new(prog: RhsProgram, canon: &Canonical, params: &[String]) -> Result<Document, String> {
        let blocks: HashMap<String, (usize, usize)> = prog
            .state_layout()
            .iter()
            .map(|b| {
                (
                    b.name.clone(),
                    (b.flat_offset, b.shape.iter().product::<usize>().max(1)),
                )
            })
            .collect();
        let mut map = Vec::with_capacity(canon.states.len());
        for (name, s) in &canon.states {
            let n = s.ext.iter().product::<usize>();
            match blocks.get(name) {
                Some((off, len)) if *len == n => map.push((s.start, *off, n)),
                Some((_, len)) => {
                    return Err(format!(
                        "its state {name} has {len} elements; the plan's canonical order has {n}"
                    ))
                }
                None => return Err(format!("it does not declare the state {name}")),
            }
        }
        let p_index =
            prog.parameters()
                .iter()
                .map(|ps| {
                    params.iter().position(|q| *q == ps.name).ok_or_else(|| {
                        format!("its parameter {} is not one of the plan's", ps.name)
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
        let forcing = prog.default_forcing();
        Ok(Document {
            prog,
            forcing,
            map,
            blocks,
            p_index,
            pool: Mutex::new(Vec::new()),
        })
    }

    /// Evaluates the document's right-hand side at the canonical state `u`
    /// and hands its program-layout result to `g`. A fault of EarthSciAST's
    /// evaluation fills the result with NaN, which the leaf's step rejects.
    fn with_du<T: Scalar, R>(&self, u: &[T], p: &[f64], t: f64, g: impl FnOnce(&[f64]) -> R) -> R {
        let popped = self.pool.lock().expect("workspace pool").pop();
        let (mut ws, mut pu, mut du) = popped.unwrap_or_else(|| {
            let n = self.prog.n_states();
            (self.prog.new_workspace(), vec![0.0; n], vec![0.0; n])
        });
        for &(start, off, len) in &self.map {
            for k in 0..len {
                pu[off + k] = u[start + k].to_f64();
            }
        }
        let pp: Vec<f64> = self.p_index.iter().map(|&i| p[i]).collect();
        if self
            .prog
            .eval_into(&pu, &[], &pp, t, &self.forcing, &mut du, &mut ws)
            .is_err()
        {
            du.iter_mut().for_each(|x| *x = f64::NAN);
        }
        let r = g(&du);
        self.pool.lock().expect("workspace pool").push((ws, pu, du));
        r
    }
}

/// An operator built from a plan (§8).
pub struct PlannedOperator {
    name: String,
    structure: Structure,
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
}

impl PlannedOperator {
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
        self.jacobian.with_du(u, p, t, |du| {
            // Sum each entry's contributions before it multiplies v, as the
            // entry (esm-jacobian-spec §4 additivity) would.
            let mut acc = vec![0.0f64; out.len()];
            for &(k, r, c) in &self.points {
                if transpose {
                    acc[c] += du[k] * v[r].to_f64();
                } else {
                    acc[r] += du[k] * v[c].to_f64();
                }
            }
            for (o, a) in out.iter_mut().zip(&acc) {
                if *a != 0.0 {
                    *o += T::from_f64(*a);
                }
            }
        })
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
    fn provides(&self, call: Call) -> bool {
        // ft, vjp_p and the second-order calls wait for U17, U7 and U8
        // (operator-interface.md §8.3).
        matches!(call, Call::Rhs | Call::JvpU | Call::VjpU | Call::Jac)
    }
    fn dense_storage(&self) -> bool {
        true
    }

    fn rhs(&self, out: &mut [T], u: &[T], p: &[f64], t: f64) {
        self.rhs.with_du(u, p, t, |du| {
            for &(dof, off) in &self.rhs_writes {
                out[dof] += T::from_f64(du[off]);
            }
        })
    }
    fn jvp_u(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, v: &[T]) {
        self.jprod(out, u, p, t, v, false)
    }
    fn vjp_u(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, w: &[T]) {
        self.jprod(out, u, p, t, w, true)
    }
    fn jac(&self, out: &mut MatrixStorage<T>, u: &[T], p: &[f64], t: f64) {
        self.jacobian.with_du(u, p, t, |du| match out {
            MatrixStorage::Dense { n, data } => {
                let n = *n;
                for &(k, r, c) in &self.points {
                    data[c * n + r] += T::from_f64(du[k]);
                }
            }
            MatrixStorage::BlockDiagonal { ncells, n, data } => {
                // Species-major (§2): DOF i·ncells + c is species i of cell c.
                let (ncells, n) = (*ncells, *n);
                for &(k, r, c) in &self.points {
                    let (i, cr) = (r / ncells, r % ncells);
                    let (j, cc) = (c / ncells, c % ncells);
                    if cr == cc && i < n && j < n {
                        data[(i * n + j) * ncells + cr] += T::from_f64(du[k]);
                    }
                }
            }
        })
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
}

fn refuse(op: &str, doc: &str, m: impl std::fmt::Display) -> Error {
    Error::Config(format!(
        "operator {op}: {doc}: {m} (operator-interface.md §8.1)"
    ))
}

/// The operator names a schedule node uses, in tree order.
fn schedule_operators(node: &Value, out: &mut BTreeSet<String>) {
    match node {
        Value::Object(o) => {
            for k in ["leaf", "coupled", "map"] {
                match o.get(k) {
                    Some(Value::String(s)) => {
                        out.insert(s.clone());
                    }
                    Some(Value::Array(a)) => {
                        out.extend(a.iter().filter_map(|x| x.as_str().map(String::from)))
                    }
                    _ => {}
                }
            }
            for (k, v) in o {
                if !["leaf", "coupled", "map"].contains(&k.as_str()) {
                    schedule_operators(v, out);
                }
            }
        }
        Value::Array(a) => a.iter().for_each(|v| schedule_operators(v, out)),
        _ => {}
    }
}

/// Reads the plan directory `dir` and builds every operator its schedule
/// names (operator-interface.md §8).
pub fn load_plan<T: Scalar>(dir: &Path) -> Result<LoadedPlan<T>, Error> {
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
    let mut names = BTreeSet::new();
    schedule_operators(&plan["schedule"], &mut names);
    if names.is_empty() {
        return Err(cfg("plan.json: the schedule names no operator".into()));
    }
    // Each operator's documents, by the plan's `derived` list.
    let mut docs: HashMap<(String, String), String> = HashMap::new();
    for (path, d) in plan["derived"].as_object().into_iter().flatten() {
        if let (Some(op), Some(kind)) = (d["operator"].as_str(), d["kind"].as_str()) {
            docs.insert((op.to_string(), kind.to_string()), path.clone());
        }
    }

    let mut canon: Option<Canonical> = None;
    let mut params: Vec<String> = Vec::new();
    let mut p: Vec<f64> = Vec::new();
    let mut u0: Vec<f64> = Vec::new();
    let mut operators: Vec<Arc<dyn Operator<T>>> = Vec::new();
    for name in &names {
        let entry = &plan["operators"][name];
        if !entry.is_object() {
            return Err(cfg(format!(
                "plan.json: the schedule names {name}, which has no operators entry"
            )));
        }
        if entry["kind"].as_str() != Some("rhs") {
            return Err(cfg(format!(
                "operator {name}: kind {} is not run at M2",
                entry["kind"]
            )));
        }
        let path_of = |kind: &str| -> Result<String, Error> {
            docs.get(&(name.clone(), kind.to_string()))
                .cloned()
                .ok_or_else(|| cfg(format!("operator {name}: the plan has no {kind} document")))
        };
        let (rhs_path, jac_path) = (path_of("rhs")?, path_of("jacobian")?);
        let (rhs_prob, rhs_prog) =
            build_program(&dir.join(&rhs_path), tspan).map_err(|e| refuse(name, &rhs_path, e))?;
        let (_, jac_prog) =
            build_program(&dir.join(&jac_path), tspan).map_err(|e| refuse(name, &jac_path, e))?;

        // The canonical order, parameters and initial state come from the
        // first operator's right-hand-side document; every derived document
        // of a plan declares the same states (§8.2), so the others must
        // agree.
        let this = Canonical::new(
            rhs_prog
                .state_layout()
                .iter()
                .map(|b| (b.name.as_str(), b.shape.as_slice())),
            |_| false,
        );
        let jac_states = Canonical::new(
            jac_prog
                .state_layout()
                .iter()
                .map(|b| (b.name.as_str(), b.shape.as_slice())),
            |n| n.starts_with(&format!("{JACOBIAN_MODEL}.")),
        );
        if jac_states != this {
            return Err(refuse(
                name,
                &jac_path,
                "its states are not those of the right-hand-side document",
            ));
        }
        match &canon {
            None => {
                let pv = rhs_prog
                    .parameter_vector(&HashMap::new())
                    .map_err(|e| refuse(name, &rhs_path, format!("parameters: {e}")))?;
                params = rhs_prog
                    .parameters()
                    .iter()
                    .map(|ps| ps.name.clone())
                    .collect();
                p = pv;
                u0 = initial_state(&rhs_prob, &this).map_err(|e| refuse(name, &rhs_path, e))?;
                canon = Some(this);
            }
            Some(c) if *c != this => {
                return Err(refuse(
                    name,
                    &rhs_path,
                    "its states are not those of the other operators' documents",
                ))
            }
            Some(_) => {}
        }
        let c = canon.as_ref().expect("set above");
        let rhs = Document::new(rhs_prog, c, &params).map_err(|e| refuse(name, &rhs_path, e))?;
        let jacobian =
            Document::new(jac_prog, c, &params).map_err(|e| refuse(name, &jac_path, e))?;
        operators.push(Arc::new(build_operator(name, entry, c, rhs, jacobian)?));
    }
    let canon = canon.expect("at least one operator");

    // The problem: names and layout for run configs' variable references.
    let mut variables = vec![String::new(); canon.n];
    let mut cell_of = vec![0usize; canon.n];
    let mut one_d = true;
    let mut state_dofs = BTreeMap::new();
    for (name, s) in &canon.states {
        let r = canon.dofs_of(name).expect("a state");
        for (k, j) in r.clone().enumerate() {
            variables[j] = name.clone();
            cell_of[j] = k;
        }
        one_d &= s.ext.len() <= 1;
        state_dofs.insert(name.clone(), r.collect());
    }
    let problem = ToyProblem {
        name: plan["inputs"]["model"]["path"]
            .as_str()
            .unwrap_or("plan")
            .to_string(),
        n: canon.n,
        param_names: params,
        p,
        u0,
        tspan,
        variables,
        cells: if one_d { Some(cell_of) } else { None },
        operators,
        maps: vec![],
        invariants: vec![],
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
    })
}

/// EarthSciAST's initial state of a right-hand-side document (defaults and
/// `ic` equations), in canonical order (spec/plan.md §4, §6.1.2).
fn initial_state(prob: &earthsci_ast::EsmProblem, canon: &Canonical) -> Result<Vec<f64>, String> {
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
    canon
        .element_names()
        .iter()
        .map(|n| {
            at.get(n.as_str())
                .copied()
                .ok_or_else(|| format!("EarthSciAST's initial state has no {n}"))
        })
        .collect()
}

/// One operator from its plan entry and documents (§8.2, §8.3).
fn build_operator(
    name: &str,
    entry: &Value,
    canon: &Canonical,
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

    // writes: every cell of every variable in the plan's list.
    let wnames: Vec<String> = entry["writes"]
        .as_array()
        .ok_or_else(|| bad("no writes".into()))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(String::from)
                .ok_or_else(|| bad(format!("writes entry {v}")))
        })
        .collect::<Result<_, _>>()?;
    let mut writes = Vec::new();
    for w in &wnames {
        writes.extend(
            canon
                .dofs_of(w)
                .ok_or_else(|| bad(format!("writes {w}, which is not a state")))?,
        );
    }
    writes.sort_unstable();
    writes.dedup();
    let mut owner = vec![usize::MAX; canon.n];
    for (start, off, len) in rhs.map.iter().copied() {
        for k in 0..len {
            owner[start + k] = off + k;
        }
    }
    let rhs_writes = writes.iter().map(|&j| (j, owner[j])).collect();

    // The band points, scattered once: band k's field JacobianEval.J_k over
    // the row's extents, then the contracted extents, column-major.
    let mut points = Vec::new();
    for (k, b) in st["bands"]
        .as_array()
        .ok_or_else(|| bad("no bands".into()))?
        .iter()
        .enumerate()
    {
        let field = format!("{JACOBIAN_MODEL}.J_{}", k + 1);
        let &(base, len) = jacobian
            .blocks
            .get(&field)
            .ok_or_else(|| bad(format!("its Jacobian document has no {field}")))?;
        let mut with_coef = b.clone();
        with_coef["coef"] = json!(0);
        let en = earthsci_astdiff::parse_band(&with_coef)
            .map_err(|e| bad(format!("band {}: {e}", k + 1)))?;
        let mut ext: Vec<usize> = canon
            .states
            .get(&en.u)
            .map(|s| s.ext.clone())
            .unwrap_or_default();
        ext.extend(en.band.contracted.iter().map(|(_, _, hi)| *hi as usize));
        let mut err = None;
        earthsci_astdiff::points::foreach_point(&en.band, &mut |cell, cc, _, col| {
            let (Some(r), Some(c)) = (canon.dof(&en.u, cell), canon.dof(&en.v, col)) else {
                return Ok(()); // a ghost read contributes nothing
            };
            let (mut off, mut stride) = (0usize, 1usize);
            for (x, n) in cell.iter().chain(cc).zip(&ext) {
                off += (*x as usize - 1) * stride;
                stride *= n;
            }
            if off >= len {
                err = Some(format!("{field} has no element {off}"));
            } else {
                points.push((base + off, r, c));
            }
            Ok(())
        })
        .map_err(|e| bad(format!("band {}: {e}", k + 1)))?;
        if let Some(e) = err {
            return Err(bad(e));
        }
    }

    // §8.3: block_diagonal when the operator is pointwise with
    // structure.block and its states are all arrays of one shape (ncells
    // the cells of that shape) or all scalars (ncells = 1).
    let pointwise = entry["locality"]["class"].as_str() == Some("pointwise");
    let shapes: BTreeSet<Vec<usize>> = wnames.iter().map(|w| canon.states[w].ext.clone()).collect();
    let structure = if pointwise && st.get("block").is_some() && shapes.len() == 1 {
        let ncells = shapes
            .iter()
            .next()
            .expect("one shape")
            .iter()
            .product::<usize>();
        Structure::BlockDiagonal {
            ncells,
            n: wnames.len(),
        }
    } else {
        Structure::Dense
    };
    Ok(PlannedOperator {
        name: name.to_string(),
        structure,
        depends_on_t,
        linear_in_u,
        writes,
        rhs_writes,
        rhs,
        jacobian,
        points,
    })
}
