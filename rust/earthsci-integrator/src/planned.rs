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
use earthsci_ast::{esm_problem, rhs_program, Compiler, ProblemInput, ProblemOptions, RhsProgram};
use earthsci_integrator_planner::numeric::{band_field, band_scatter, Dofs, Program};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The model component that holds a Jacobian document's band fields
/// (spec/plan.md §6.2).
const JACOBIAN_MODEL: &str = earthsci_integrator_planner::derived::JACOBIAN_MODEL;

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

/// A derived document's right-hand side (§8.1) and, for each of its
/// parameters, the parameter's index in the problem's `p`.
struct Document {
    prog: Program,
    p_index: Vec<usize>,
}

impl Document {
    fn new(prog: RhsProgram, dofs: &Dofs, params: &[String]) -> Result<Document, String> {
        let p_index =
            prog.parameters()
                .iter()
                .map(|ps| {
                    params.iter().position(|q| *q == ps.name).ok_or_else(|| {
                        format!("its parameter {} is not one of the plan's", ps.name)
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
        Ok(Document {
            prog: Program::from_program(prog, dofs)?,
            p_index,
        })
    }

    /// Evaluates the document's right-hand side at the canonical state `u`
    /// and hands its program-layout result to `g`.
    fn eval<T: Scalar, R>(
        &self,
        u: &[T],
        p: &[f64],
        t: f64,
        g: impl FnOnce(&[f64]) -> R,
    ) -> Result<R, String> {
        self.prog
            .eval_with(|j| u[j].to_f64(), |i| p[self.p_index[i]], t, g)
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
    let mut names = Vec::new();
    schedule_operators(&plan["schedule"], &mut names);
    if names.is_empty() {
        return Err(cfg("plan.json: the schedule names no operator".into()));
    }
    for (name, role) in &names {
        let what = match role {
            Role::Term => continue,
            Role::Map => "a map node",
            Role::Algebraic => "a solve node",
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
    let mut model: Option<Model> = None;
    let mut operators: Vec<Arc<dyn Operator<T>>> = Vec::new();
    for (name, _) in &names {
        let entry = &plan["operators"][name];
        if !entry.is_object() {
            return Err(cfg(format!(
                "plan.json: the schedule names {name}, which has no operators entry"
            )));
        }
        if entry["kind"].as_str() != Some("rhs") {
            return Err(cfg(format!(
                "operator {name}: the schedule uses it as a term operator, but its kind is {}",
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

        let this = program_dofs(&rhs_prog, |_| false);
        let jac_states = program_dofs(&jac_prog, |n| n.starts_with(&format!("{JACOBIAN_MODEL}.")));
        if jac_states.states != this.states {
            return Err(refuse(
                name,
                &jac_path,
                "its states are not those of the right-hand-side document",
            ));
        }
        match &model {
            None => {
                let p = rhs_prog
                    .parameter_vector(&HashMap::new())
                    .map_err(|e| refuse(name, &rhs_path, format!("parameters: {e}")))?;
                let params = rhs_prog
                    .parameters()
                    .iter()
                    .map(|ps| ps.name.clone())
                    .collect();
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
        let m = model.as_ref().expect("set above");
        let rhs =
            Document::new(rhs_prog, &m.dofs, &m.params).map_err(|e| refuse(name, &rhs_path, e))?;
        let jacobian =
            Document::new(jac_prog, &m.dofs, &m.params).map_err(|e| refuse(name, &jac_path, e))?;
        // A document that cannot be evaluated at the initial state is
        // refused here, with EarthSciAST's message, rather than failing
        // every step of the run.
        for (doc, path) in [(&rhs, &rhs_path), (&jacobian, &jac_path)] {
            doc.eval(&m.u0, &m.p, tspan.0, |_| ())
                .map_err(|e| refuse(name, path, format!("at the initial state: {e}")))?;
        }
        operators.push(Arc::new(build_operator(
            name, entry, &m.dofs, m.ncells, rhs, jacobian,
        )?));
    }
    let m = model.expect("at least one operator");

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

    // §8.3: block_diagonal when the operator is pointwise with
    // structure.block and the states are all arrays of one shape (ncells
    // the cells of that shape) or all scalars (ncells = 1), with n the
    // number of states. A leaf of such operators covers every state, so it
    // has the same structure (the leaf refuses a block_diagonal structure
    // that does not cover its state).
    let pointwise = entry["locality"]["class"].as_str() == Some("pointwise");
    let structure = match ncells {
        Some(ncells) if pointwise && st.get("block").is_some() => {
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
