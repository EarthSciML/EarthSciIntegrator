//! The planner's pipeline (spec/plan.md §4 "Order of refusals").
//!
//! [`plan`] runs §4, passes 1–8 and the resolutions of §7 and returns a
//! [`Planned`], whose [`Planned::write`] writes the plan directory:
//! `plan.json`, `derived/*.esm` and `plan.report.json` (§1).

use crate::astdiff;
use crate::axes::{self, Axes};
use crate::derived;
use crate::error::{codes, refuse, PlanError, Result};
use crate::locality::{self, BandSource, Class, Locality};
use crate::model::{self, ModelInfo};
use crate::numeric::{self, Dofs, Program, Sample, Sparse};
use crate::operators::{self, Operator};
use crate::runconfig::{self, RunConfig};
use crate::schedule::{self, CheckContext, Kind, Node};
use crate::settings;
use crate::strategy::{self, LeafFacts, LeafStrategy, Stiffness, StiffnessSource};
use crate::structure::{self, Structure};
use crate::terms::{self, Term};
use crate::time::{self, Interval};
use crate::validation;
use earthsci_ast::TermOrigin;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// What the caller may give besides the run config.
#[derive(Debug, Clone, Default)]
pub struct Options {
    /// The caller's run interval, in the run config's `time` form
    /// (spec/run-config.md §1.1); it overrides every other source.
    pub time: Option<Value>,
}

/// An operator's derived documents (§6), as written.
#[derive(Debug, Clone)]
pub struct Documents {
    /// `<op>.rhs.esm`'s bytes.
    pub rhs: Vec<u8>,
    /// `<op>.jacobian.esm`'s bytes.
    pub jacobian: Vec<u8>,
}

/// Everything the planner computes for a run config.
#[derive(Debug, Clone)]
pub struct Planned {
    pub run_config: RunConfig,
    pub model: ModelInfo,
    pub axes: Axes,
    pub interval: Interval,
    /// Pass 1's terms, in term order.
    pub terms: Vec<Term>,
    /// Pass 2's class and radius of each term, in term order.
    pub term_localities: Vec<Locality>,
    /// The operators by name (passes 1–3).
    pub operators: BTreeMap<String, Operator>,
    /// Pass 4: each operator's structure.
    pub structures: BTreeMap<String, Structure>,
    /// The schedule tree (ids assigned, checks passed).
    pub schedule: Node,
    /// §7.3.
    pub step: Value,
    /// §7.4.
    pub partition: Value,
    /// §7.5.
    pub precision: Value,
    /// §7.6: present exactly when the run config has `adjoint`.
    pub checkpoint_schedule: Option<Value>,
    /// §7.7.
    pub inputs: Value,
    /// §5.5.4: each leaf's reference step, by node id.
    pub reference_steps: BTreeMap<String, f64>,
    /// §5.5.2: every operator's stiffness.
    pub stiffness: BTreeMap<String, Stiffness>,
    /// What §5.5.3 reads from structure and stiffness.
    pub facts: StructureFacts,
    /// The resolved schedule (§7.1, §5.5.3).
    pub resolved_schedule: Value,
    /// Pass 6: each operator's documents.
    pub documents: BTreeMap<String, Documents>,
    /// Pass 8: the checks that ran, in the plan's order (§8.1).
    pub validation: Vec<Value>,
    /// `plan.report.json` (§1): what differs between implementations.
    pub report: Value,
}

/// Facts from structure and stiffness that resolving the schedule needs.
#[derive(Debug, Clone, Default)]
pub struct StructureFacts {
    /// Stiffness by the Gershgorin bound (§5.5.2 rule 4), by operator.
    pub gershgorin_stiff: BTreeMap<String, bool>,
    /// Each operator's `structure.block.vars`, or `None` when it has no
    /// `structure.block`.
    pub block_vars: BTreeMap<String, Option<Vec<String>>>,
}

/// The numbers passes 5 and 8 evaluate, built on first use.
struct Numbers<'a> {
    info: &'a ModelInfo,
    interval: Interval,
    dofs: Dofs,
    names: Vec<String>,
    /// `u_0`, the sampling base and the sampled states (§5.5.1).
    states: Option<(Vec<f64>, Vec<f64>, Vec<Sample>)>,
    /// Per operator: its programs, band pattern, rows and band Jacobians.
    ops: BTreeMap<String, OperatorNumbers>,
}

struct OperatorNumbers {
    rhs: Program,
    pattern: Sparse,
    /// Whether each DOF is a row of the operator's `writes`.
    rows: Vec<bool>,
    /// The band Jacobian at each sampled state.
    jb: Vec<Sparse>,
}

fn internal(m: impl std::fmt::Display) -> PlanError {
    PlanError::Internal(m.to_string())
}

impl<'a> Numbers<'a> {
    fn new(info: &'a ModelInfo, interval: Interval) -> Self {
        let dofs = Dofs::new(info);
        let names = dofs.element_names();
        Numbers {
            info,
            interval,
            dofs,
            names,
            states: None,
            ops: BTreeMap::new(),
        }
    }

    fn tspan(&self) -> (f64, f64) {
        (self.interval.start, self.interval.end)
    }

    fn states(&mut self) -> Result<&(Vec<f64>, Vec<f64>, Vec<Sample>)> {
        if self.states.is_none() {
            let u0 = numeric::initial_state(self.info, &self.dofs, self.interval.start)
                .map_err(internal)?;
            let base = numeric::sampling_base(&u0);
            let samples = numeric::samples(&u0, self.interval.start, self.interval.end);
            self.states = Some((u0, base, samples));
        }
        Ok(self.states.as_ref().expect("set above"))
    }

    /// The operator's numbers: `rhs` and `jacobian` are its documents.
    fn op(
        &mut self,
        name: &str,
        op: &Operator,
        st: &Structure,
        rhs: &Value,
        jacobian: &Value,
    ) -> Result<&OperatorNumbers> {
        if !self.ops.contains_key(name) {
            let samples = self.states()?.2.clone();
            let tspan = self.tspan();
            let rhs_prog = Program::new(rhs, tspan).map_err(internal)?;
            let jac_prog = Program::new(jacobian, tspan).map_err(internal)?;
            let pattern = numeric::pattern(&self.dofs, &st.bands).map_err(internal)?;
            let owners = self.dofs.owners();
            let rows = owners
                .iter()
                .map(|s| op.writes.iter().any(|w| w == s))
                .collect();
            let jb = samples
                .iter()
                .map(|s| numeric::band_jacobian(&jac_prog, &self.dofs, &st.bands, &s.u, s.t))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(internal)?;
            self.ops.insert(
                name.to_string(),
                OperatorNumbers {
                    rhs: rhs_prog,
                    pattern,
                    rows,
                    jb,
                },
            );
        }
        Ok(&self.ops[name])
    }
}

/// Plans the run config at `path`. `bands` gives each term's band list
/// (§5.2.2); [`astdiff::AstDiffBands`] is EarthSciASTDiff's.
pub fn plan(path: &Path, options: &Options, bands: &dyn BandSource) -> Result<Planned> {
    // §4 steps 1–7.
    let rc = runconfig::read(path)?;
    let mut info = model::read(&rc)?;
    // §4 step 8: the shapes of lifted states (§6.1.3), whether or not `axes`
    // is given, then the axes, then the time.
    model::resolve_lifted_shapes(&mut info)?;
    let axes = axes::resolve(&rc, &info)?;
    let interval = time::resolve(&rc, &info, options.time.as_ref())?;

    // Pass 1: terms.
    let terms = terms::terms(&info)?;

    // Each term's locality, before selection, which reads it (§4 "Order of
    // refusals" item 2, D81).
    let bands = match bands.term_bands(&info, &terms) {
        Ok(Some(b)) => b,
        Ok(None) => {
            return Err(PlanError::NotYetImplemented(
                "this band source gives no term bands (spec/plan.md §5.2.2)".into(),
            ))
        }
        Err(f) => {
            let t = terms.get(f.term);
            return Err(refuse(
                codes::BAND_MODEL_FAILED,
                "/model",
                match t {
                    Some(t) => format!(
                        "term ({}, {}, {}): {}",
                        t.variable, t.contribution, t.summand, f.message
                    ),
                    None => f.message,
                },
            ));
        }
    };
    let term_localities: Vec<Locality> = bands
        .iter()
        .map(|b| locality::classify(b, &info.shapes, &info.index_sets, &axes))
        .collect();

    // Pass 1 selection, then pass 2 per operator; pass 3 is in `select`.
    let mut ops = operators::select(&rc, &info, &terms, &term_localities)?;
    for op in ops.values_mut() {
        op.locality =
            locality::operator_locality(op.terms.iter().map(|&i| &term_localities[i]), &axes);
    }

    // Pass 4: each operator's band list, from its right-hand-side document,
    // and its structure.
    let mut rhs_docs = BTreeMap::new();
    let mut views = BTreeMap::new();
    let mut structures = BTreeMap::new();
    for (name, op) in &ops {
        let ts: Vec<&Term> = op.terms.iter().map(|&i| &terms[i]).collect();
        let doc = derived::rhs_document(&info, name, &ts);
        let (view, band_list) = astdiff::document_bands(&doc).map_err(|m| {
            refuse(
                codes::BAND_MODEL_FAILED,
                "/model",
                format!("operator {name}: {m}"),
            )
        })?;
        let st = structure::structure(&info, &axes, op, &terms, &view, band_list)?;
        rhs_docs.insert(name.clone(), doc);
        views.insert(name.clone(), view);
        structures.insert(name.clone(), st);
    }

    // §7.1–§7.6 checks.
    let root = schedule::tree(&rc, &info, &ops);
    schedule::check(
        &root,
        &CheckContext {
            rc: &rc,
            info: &info,
            ops: &ops,
            interval,
        },
    )?;
    let step = settings::step(&rc, &info)?;
    let partition = settings::partition(&rc, &info, &axes, &ops, &root)?;
    // §7.5's classical-Rosenbrock check runs here for the methods the run
    // config gives, and for an `"auto"` method once §5.5.3 resolves it (§4
    // "Order of refusals" item 6).
    let given_algs = given_algs(&root);
    let precision = settings::precision(&rc, &info, &ops, &root, &given_algs)?;
    let checkpoint_schedule = settings::checkpoint(&rc, &info, &root)?;

    // The Jacobian documents (§6.2): built here, since the Gershgorin bound
    // evaluates them; written with the other derived documents.
    let mut jac_docs = BTreeMap::new();
    for (name, st) in &structures {
        jac_docs.insert(
            name.clone(),
            derived::jacobian_document(&info, name, &views[name], &st.bands)?,
        );
    }

    // Pass 5: stiffness, rules 1–3, then rule 4.
    let reference_steps = schedule::reference_steps(&root, &rc, &info, interval)?;
    let reaction_systems: BTreeSet<&str> = info
        .file
        .reaction_systems
        .iter()
        .flat_map(|r| r.keys())
        .map(String::as_str)
        .collect();
    let solver_stiffness = info
        .file
        .solver
        .as_ref()
        .and_then(|s| s.stiffness.as_deref());
    let mut numbers = Numbers::new(&info, interval);
    let mut stiffness = BTreeMap::new();
    let mut gershgorin_report = Map::new();
    let mut facts = StructureFacts::default();
    for (name, op) in &ops {
        let leaf = root
            .leaves()
            .into_iter()
            .find(|n| matches!(&n.kind, Kind::Leaf { ops, .. } if ops.contains(name)));
        let leaf_alg = leaf.and_then(|n| n.value.get("alg").and_then(Value::as_str));
        let rs_term = op.terms.iter().any(|&i| {
            matches!(terms[i].origins.as_slice(),
                [TermOrigin::Component { component }] if reaction_systems.contains(component.as_str()))
        });
        let s = match strategy::stiffness_rules_1_to_3(
            leaf_alg,
            rs_term,
            op.locality.class,
            solver_stiffness,
        ) {
            Some(s) => s,
            None => {
                // Rule 4: max_s ρ(u_s, p, t_s) · Δ ≥ 10.
                let delta = leaf
                    .and_then(|n| reference_steps.get(n.id()).copied())
                    .ok_or_else(|| internal(format!("no leaf advances {name}")))?;
                let on = numbers.op(
                    name,
                    op,
                    &structures[name],
                    &rhs_docs[name],
                    &jac_docs[name],
                )?;
                let rhos: Vec<f64> = on
                    .jb
                    .iter()
                    .map(|jb| validation::gershgorin(jb, &on.rows))
                    .collect();
                let rho = rhos.iter().copied().fold(0.0, f64::max);
                gershgorin_report.insert(
                    name.clone(),
                    json!({"rho": rhos, "delta": delta, "rho_delta": rho * delta}),
                );
                facts
                    .gershgorin_stiff
                    .insert(name.clone(), rho * delta >= 10.0);
                Stiffness {
                    stiff: rho * delta >= 10.0,
                    source: StiffnessSource::Gershgorin,
                }
            }
        };
        stiffness.insert(name.clone(), s);
    }
    for (name, st) in &structures {
        facts
            .block_vars
            .insert(name.clone(), st.block.as_ref().map(|b| b.vars.clone()));
    }
    let inputs = settings::inputs(&rc, &info);

    let mut planned = Planned {
        run_config: rc,
        model: info.clone(),
        axes,
        interval,
        terms,
        term_localities,
        operators: ops,
        structures,
        schedule: root,
        step,
        partition,
        precision,
        checkpoint_schedule,
        inputs,
        reference_steps,
        stiffness,
        facts,
        resolved_schedule: Value::Null,
        documents: BTreeMap::new(),
        validation: Vec::new(),
        report: Value::Null,
    };
    // §5.5.3 for every leaf, in tree order, then §7.5's check for the
    // methods it resolved (§4 "Order of refusals" item 6).
    let mut algs = BTreeMap::new();
    for n in planned.schedule.leaves() {
        let s = planned.leaf_strategy(n, Some(&planned.facts))?;
        algs.insert(n.id().to_string(), s.alg);
    }
    settings::precision(
        &planned.run_config,
        &planned.model,
        &planned.operators,
        &planned.schedule,
        &algs,
    )?;
    planned.resolved_schedule = planned.resolve_schedule(Some(&planned.facts))?;

    // Pass 6: the derived documents.
    for (name, doc) in &rhs_docs {
        planned.documents.insert(
            name.clone(),
            Documents {
                rhs: derived::to_bytes(doc),
                jacobian: derived::to_bytes(&jac_docs[name]),
            },
        );
    }

    // Pass 8: validation.
    let mut report = Map::new();
    let (_, base, samples) = numbers.states()?.clone();
    let model_prog = Program::new(info.path.as_path(), numbers.tspan()).map_err(internal)?;
    for (name, op) in &planned.operators {
        numbers.op(
            name,
            op,
            &planned.structures[name],
            &rhs_docs[name],
            &jac_docs[name],
        )?;
    }
    let mut rhs_sum = Vec::new();
    for s in &samples {
        let f = model_prog
            .eval(&numbers.dofs, &s.u, s.t)
            .map_err(internal)?;
        let parts = numbers
            .ops
            .values()
            .map(|o| o.rhs.eval(&numbers.dofs, &s.u, s.t))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(internal)?;
        let worst = validation::rhs_sum(&f, &parts, &numbers.names).map_err(|m| {
            refuse(
                codes::VALIDATION_RHS_SUM_FAILED,
                "/model",
                format!("at t = {}: {m}", s.t),
            )
        })?;
        rhs_sum.push(worst);
    }
    report.insert("rhs_sum".into(), json!({"worst_relative_error": rhs_sum}));
    planned
        .validation
        .push(json!({"check": "rhs_sum", "passed": true}));
    let mut per_op = Map::new();
    for (name, op) in &planned.operators {
        let on = &numbers.ops[name];
        let mut measured = Vec::new();
        for (s, jb) in samples.iter().zip(&on.jb) {
            let at = validation::At {
                dofs: &numbers.dofs,
                sample: s,
                base: &base,
                names: &numbers.names,
            };
            let r =
                validation::band_model(&on.rhs, &at, jb, &on.pattern, &on.rows).map_err(|m| {
                    refuse(
                        codes::VALIDATION_BAND_MODEL_FAILED,
                        "/model",
                        format!("operator {name} at t = {}: {m}", s.t),
                    )
                })?;
            measured.push(r.to_json());
        }
        planned
            .validation
            .push(json!({"check": "band_model", "operator": name, "passed": true}));
        // non_empty: a term, and a band when some term reads a state.
        let reads_state = op
            .terms
            .iter()
            .any(|&i| planned.terms[i].names.iter().any(|n| info.is_state(n)));
        if op.terms.is_empty() || (reads_state && planned.structures[name].bands.is_empty()) {
            return Err(refuse(
                codes::VALIDATION_EMPTY_OPERATOR,
                "/model",
                format!(
                    "operator {name} has {} terms and {} bands",
                    op.terms.len(),
                    planned.structures[name].bands.len()
                ),
            ));
        }
        planned
            .validation
            .push(json!({"check": "non_empty", "operator": name, "passed": true}));
        per_op.insert(name.clone(), json!({"band_model": measured}));
    }
    report.insert("operators".into(), Value::Object(per_op));
    report.insert("gershgorin".into(), Value::Object(gershgorin_report));
    planned.report = json!({
        "planner": "earthsci-integrator-planner (Rust)",
        "version": env!("CARGO_PKG_VERSION"),
        "upstream": {
            "earthsci-ast": "0.6.0",
            "earthsci-astdiff": "0.1.0 (743d1404f7573e91c6789a62f0cad88fd93fc0ad)",
        },
        "validation": Value::Object(report),
    });
    Ok(planned)
}

/// The `alg` each leaf gives, when it is not `"auto"`, by node id.
fn given_algs(root: &Node) -> BTreeMap<String, String> {
    root.leaves()
        .into_iter()
        .filter_map(|n| {
            let a = n.value.get("alg").and_then(Value::as_str)?;
            (a != "auto").then(|| (n.id().to_string(), a.to_string()))
        })
        .collect()
}

impl Planned {
    /// The plan's `time` (§7.2).
    pub fn time_json(&self) -> Value {
        json!({"start": self.interval.start, "end": self.interval.end})
    }

    /// An operator's `terms` (§5.1.4).
    pub fn terms_json(&self, op: &Operator) -> Value {
        Value::Array(
            op.terms
                .iter()
                .map(|&i| {
                    let t = &self.terms[i];
                    json!({
                        "variable": t.variable,
                        "contribution": t.contribution,
                        "summand": t.summand,
                        "provenance": t.provenance,
                        "locality": self.term_localities[i].class.as_str(),
                    })
                })
                .collect(),
        )
    }

    /// An operator's `locality` (§5.2.4): the class, with `radius` on every
    /// horizontal axis for a stencil operator.
    pub fn locality_json(&self, op: &Operator) -> Value {
        let mut out = Map::new();
        out.insert("class".into(), json!(op.locality.class.as_str()));
        if op.locality.class == Class::Stencil {
            let radius: Map<String, Value> = op
                .locality
                .radius
                .iter()
                .map(|(k, (lo, up))| (k.clone(), json!([lo, up])))
                .collect();
            out.insert("radius".into(), Value::Object(radius));
        }
        Value::Object(out)
    }

    /// An operator's entry in `operators` (§3): `kind`, `locality`,
    /// `writes`, `reads`, `terms`, `structure` and `stiffness`.
    pub fn operator_json(&self, name: &str) -> Value {
        let op = &self.operators[name];
        let mut out = Map::new();
        out.insert("kind".into(), json!("rhs"));
        out.insert("locality".into(), self.locality_json(op));
        out.insert("writes".into(), json!(op.writes));
        out.insert("reads".into(), json!(op.reads));
        out.insert("terms".into(), self.terms_json(op));
        out.insert("structure".into(), self.structures[name].to_json());
        out.insert("stiffness".into(), self.stiffness[name].to_json());
        Value::Object(out)
    }

    /// An operator's stiffness: rules 1–3, else the Gershgorin bound from
    /// `facts`.
    fn stiffness_of(&self, name: &str, facts: Option<&StructureFacts>) -> Option<bool> {
        match self.stiffness.get(name) {
            Some(s) if s.source != StiffnessSource::Gershgorin => Some(s.stiff),
            _ => facts.and_then(|f| f.gershgorin_stiff.get(name).copied()),
        }
    }

    /// What §5.5.3 needs to know about a leaf.
    pub fn leaf_facts(&self, n: &Node, facts: Option<&StructureFacts>) -> LeafFacts {
        let names: Vec<String> = match &n.kind {
            Kind::Leaf { ops, .. } => ops.clone(),
            _ => vec![],
        };
        let ops: Vec<(String, Class)> = names
            .iter()
            .map(|o| (o.clone(), self.operators[o].locality.class))
            .collect();
        let stiffs: Option<Vec<bool>> = names.iter().map(|o| self.stiffness_of(o, facts)).collect();
        let stiff_column_ops: Option<Vec<String>> = ops
            .iter()
            .filter(|(_, c)| *c == Class::Column)
            .map(|(o, _)| self.stiffness_of(o, facts).map(|s| (o.clone(), s)))
            .collect::<Option<Vec<_>>>()
            .map(|v| v.into_iter().filter(|(_, s)| *s).map(|(o, _)| o).collect());
        let states = settings::leaf_states(n, &self.operators);
        let shapes: BTreeSet<&Vec<String>> = states.iter().map(|s| &self.model.shapes[s]).collect();
        let uniform_shapes = shapes.len() <= 1;
        let block_vars = facts.map(|f| {
            let mut union = BTreeSet::new();
            for o in &names {
                match f.block_vars.get(o) {
                    Some(Some(vars)) => union.extend(vars.iter().cloned()),
                    _ => return None,
                }
            }
            Some(union.len())
        });
        LeafFacts {
            class: schedule::leaf_class(n, &self.operators),
            stiff: stiffs.map(|v| v.into_iter().any(|s| s)),
            stiff_column_ops,
            ops,
            reltol: schedule::reltol(n),
            dt_given: n.dt_given(),
            uniform_shapes,
            block_vars,
        }
    }

    /// A leaf's resolved method (§5.5.3).
    pub fn leaf_strategy(&self, n: &Node, facts: Option<&StructureFacts>) -> Result<LeafStrategy> {
        strategy::resolve_leaf(
            &Value::Object(n.value.clone()),
            &self.leaf_facts(n, facts),
            &n.pointer,
        )
    }

    /// The resolved schedule (§7.1) with every leaf's method resolved
    /// (§5.5.3) from `facts`.
    pub fn resolve_schedule(&self, facts: Option<&StructureFacts>) -> Result<Value> {
        let strategy = |n: &Node| self.leaf_strategy(n, facts);
        let resolved = schedule::write(&self.schedule, &self.model, &self.operators, &strategy)?;
        // §7.5's classical-Rosenbrock check for the resolved methods.
        let mut algs = BTreeMap::new();
        for n in self.schedule.leaves() {
            algs.insert(n.id().to_string(), strategy(n)?.alg);
        }
        settings::precision(
            &self.run_config,
            &self.model,
            &self.operators,
            &self.schedule,
            &algs,
        )?;
        Ok(resolved)
    }

    /// The canonical DOF order of the flattened model (§4).
    pub fn dofs(&self) -> Dofs {
        Dofs::new(&self.model)
    }

    /// Operator `op`'s band Jacobian at `(u, t)`, `u` in canonical order:
    /// its Jacobian document evaluated and scattered with
    /// `structure.bands`, keyed by EarthSciAST's element names (`X[i,j]`).
    /// Every band point is a key, whatever its value: the keys are the band
    /// pattern. For the conformance harness's comparison with the Julia
    /// planner's band values (§8.2).
    pub fn band_jacobian_at(
        &self,
        op: &str,
        u: &[f64],
        t: f64,
    ) -> Result<BTreeMap<(String, String), f64>> {
        let docs = self
            .documents
            .get(op)
            .ok_or_else(|| internal(format!("no operator {op}")))?;
        let doc: Value = serde_json::from_slice(&docs.jacobian).map_err(internal)?;
        let prog =
            Program::new(&doc, (self.interval.start, self.interval.end)).map_err(internal)?;
        let dofs = self.dofs();
        let names = dofs.element_names();
        let bands = &self.structures[op].bands;
        let mut out: BTreeMap<(String, String), f64> = numeric::pattern(&dofs, bands)
            .map_err(internal)?
            .keys()
            .map(|(r, c)| ((names[*r].clone(), names[*c].clone()), 0.0))
            .collect();
        let jb = numeric::band_jacobian(&prog, &dofs, bands, u, t).map_err(internal)?;
        for ((r, c), v) in jb {
            out.insert((names[r].clone(), names[c].clone()), v);
        }
        Ok(out)
    }

    /// The path of a derived document relative to the plan directory.
    pub fn derived_path(op: &str, kind: &str) -> String {
        format!("derived/{op}.{kind}.esm")
    }

    /// `derived` (§7.7): every document by path, with its kind, operator
    /// and the hash of the bytes written.
    pub fn derived_json(&self) -> Value {
        let mut out = Map::new();
        for (op, d) in &self.documents {
            for (kind, bytes) in [("rhs", &d.rhs), ("jacobian", &d.jacobian)] {
                out.insert(
                    Self::derived_path(op, kind),
                    json!({"kind": kind, "operator": op, "sha256": settings::sha256_hex(bytes)}),
                );
            }
        }
        Value::Object(out)
    }

    /// The whole `plan.json` (§3).
    pub fn plan_json(&self) -> Value {
        let mut out = Map::new();
        out.insert(
            "esm_integrator".into(),
            self.run_config
                .at("/esm_integrator")
                .cloned()
                .unwrap_or(Value::Null),
        );
        out.insert("inputs".into(), self.inputs.clone());
        out.insert("time".into(), self.time_json());
        let ops: Map<String, Value> = self
            .operators
            .keys()
            .map(|n| (n.clone(), self.operator_json(n)))
            .collect();
        out.insert("operators".into(), Value::Object(ops));
        out.insert("schedule".into(), self.resolved_schedule.clone());
        out.insert("step".into(), self.step.clone());
        out.insert("partition".into(), self.partition.clone());
        out.insert("precision".into(), self.precision.clone());
        if let Some(c) = &self.checkpoint_schedule {
            out.insert("checkpoint_schedule".into(), c.clone());
        }
        out.insert("derived".into(), self.derived_json());
        out.insert("validation".into(), Value::Array(self.validation.clone()));
        Value::Object(out)
    }

    /// `plan.json`'s bytes: RFC 8785 (§2.1).
    pub fn plan_bytes(&self) -> Result<Vec<u8>> {
        crate::jcs::to_string(&self.plan_json())
            .map(String::into_bytes)
            .map_err(PlanError::Internal)
    }

    /// Writes the plan directory `dir` (§1): `plan.json`, `derived/*.esm`
    /// and `plan.report.json`.
    pub fn write(&self, dir: &Path) -> Result<()> {
        let io = |e: std::io::Error| PlanError::Io(format!("{}: {e}", dir.display()));
        std::fs::create_dir_all(dir.join("derived")).map_err(io)?;
        std::fs::write(dir.join("plan.json"), self.plan_bytes()?).map_err(io)?;
        for (op, d) in &self.documents {
            for (kind, bytes) in [("rhs", &d.rhs), ("jacobian", &d.jacobian)] {
                std::fs::write(dir.join(Self::derived_path(op, kind)), bytes).map_err(io)?;
            }
        }
        let report = serde_json::to_vec_pretty(&self.report).map_err(internal)?;
        std::fs::write(dir.join("plan.report.json"), report).map_err(io)?;
        Ok(())
    }
}
