//! The planner's pipeline (spec/plan.md §4 "Order of refusals").
//!
//! Stage 1 of M2 runs §4, passes 1–3, the checks and resolutions of §7, and
//! the parts of pass 5 that need no EarthSciASTDiff. What needs it (term
//! bands come in through a [`BandSource`]; structure, stiffness by the
//! Gershgorin bound, derived documents and validation do not exist yet) is
//! left out, and [`Planned::plan_json`] says so rather than writing a plan
//! with those fields made up.

use crate::axes::{self, Axes};
use crate::error::{codes, refuse, PlanError, Result};
use crate::locality::{self, BandSource, Class, Locality};
use crate::model::{self, ModelInfo};
use crate::operators::{self, Operator};
use crate::runconfig::{self, RunConfig};
use crate::schedule::{self, CheckContext, Kind, Node};
use crate::settings;
use crate::strategy::{self, LeafFacts, LeafStrategy, Stiffness};
use crate::terms::{self, Term};
use crate::time::{self, Interval};
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

/// Everything stage 1 computes for a run config.
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
    /// §5.5.2 rules 1–3 per operator; `None` where rule 4 decides.
    pub stiffness: BTreeMap<String, Option<Stiffness>>,
}

/// Facts that come from EarthSciASTDiff (stage 2) and that resolving the
/// schedule can need.
#[derive(Debug, Clone, Default)]
pub struct StructureFacts {
    /// Stiffness by the Gershgorin bound (§5.5.2 rule 4), by operator.
    pub gershgorin_stiff: BTreeMap<String, bool>,
    /// Each operator's `structure.block.vars`, or `None` when it has no
    /// `structure.block`.
    pub block_vars: BTreeMap<String, Option<Vec<String>>>,
}

/// Plans the run config at `path` as far as stage 1 goes.
pub fn plan(path: &Path, options: &Options, bands: &dyn BandSource) -> Result<Planned> {
    // §4 steps 1–6.
    let rc = runconfig::read(path)?;
    let mut info = model::read(&rc)?;
    // §4 step 7: axes (with the shapes of lifted states, §6.1.3) and time.
    //
    // SPEC FINDING (§4 step 7, §6.1.3): lifted states' shapes are resolved
    // here whether or not `axes` is given, since the default axes and the
    // locality rules both need every state's index sets; the order of
    // `lift_shape_unresolved` against later refusals is otherwise unstated.
    model::resolve_lifted_shapes(&mut info)?;
    let axes = axes::resolve(&rc, &info)?;
    let interval = time::resolve(&rc, &info, options.time.as_ref())?;

    // Pass 1: terms.
    let terms = terms::terms(&info)?;

    // Pass 2, per term.
    //
    // SPEC FINDING (§4, §5.1.3, §5.2): pass 1's selection needs each term's
    // locality, which pass 2 defines, so the per-term half of pass 2 runs
    // between pass 1's terms and its selection; a `band_model_failed` for a
    // term therefore comes before every selection refusal.
    let bands = match bands.term_bands(&info, &terms) {
        Ok(Some(b)) => b,
        Ok(None) => {
            return Err(PlanError::NotYetImplemented(
                "term bands (spec/plan.md §5.2.2) need EarthSciASTDiff's jacobian_bands".into(),
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

    // Pass 4 (structure) arrives with stage 2.

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
    // The classical-Rosenbrock check of §7.5 for the methods the run config
    // gives. SPEC FINDING (§4, §7.5): an `"auto"` method is resolved in pass
    // 5, after §7.5's checks, so a low-precision role on a leaf whose
    // resolved method is a classical Rosenbrock (Rodas4 at a tight `reltol`)
    // is checked again once the method is resolved (`resolve_schedule`).
    let given_algs = given_algs(&root);
    let precision = settings::precision(&rc, &info, &ops, &root, &given_algs)?;
    let checkpoint_schedule = settings::checkpoint(&rc, &info, &root)?;

    // Pass 5, what needs no bands.
    let reference_steps = schedule::reference_steps(&root, &rc, &info, interval)?;
    let reaction_systems: BTreeSet<&str> = info
        .file
        .reaction_systems
        .iter()
        .flat_map(|r| r.keys())
        .map(String::as_str)
        .collect();
    let solver_stiffness = info.file.solver.as_ref().and_then(|s| s.stiffness.as_deref());
    let mut stiffness = BTreeMap::new();
    for (name, op) in &ops {
        let leaf_alg = root
            .leaves()
            .into_iter()
            .find(|n| matches!(&n.kind, Kind::Leaf { ops, .. } if ops.contains(name)))
            .and_then(|n| n.value.get("alg").and_then(Value::as_str));
        let rs_term = op.terms.iter().any(|&i| {
            matches!(terms[i].origins.as_slice(),
                [TermOrigin::Component { component }] if reaction_systems.contains(component.as_str()))
        });
        stiffness.insert(
            name.clone(),
            strategy::stiffness_rules_1_to_3(leaf_alg, rs_term, op.locality.class, solver_stiffness),
        );
    }
    let inputs = settings::inputs(&rc, &info);

    let planned = Planned {
        run_config: rc,
        model: info,
        axes,
        interval,
        terms,
        term_localities,
        operators: ops,
        schedule: root,
        step,
        partition,
        precision,
        checkpoint_schedule,
        inputs,
        reference_steps,
        stiffness,
    };
    // §5.5.3 for every leaf, in tree order, as far as stage 1 can: a refusal
    // ends planning; a leaf that needs stage-2 facts is resolved later.
    let mut algs = BTreeMap::new();
    for n in planned.schedule.leaves() {
        match planned.leaf_strategy(n, None) {
            Ok(s) => {
                algs.insert(n.id().to_string(), s.alg);
            }
            Err(PlanError::NotYetImplemented(_)) => {}
            Err(e) => return Err(e),
        }
    }
    settings::precision(
        &planned.run_config,
        &planned.model,
        &planned.operators,
        &planned.schedule,
        &algs,
    )?;
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

    /// The parts of an operator's entry stage 1 computes: `kind`,
    /// `locality`, `writes`, `reads`, `terms`, and `stiffness` where rules
    /// 1–3 decide it. `structure` needs EarthSciASTDiff.
    pub fn operator_json_stage1(&self, name: &str) -> Value {
        let op = &self.operators[name];
        let mut out = Map::new();
        out.insert("kind".into(), json!("rhs"));
        out.insert("locality".into(), self.locality_json(op));
        out.insert("writes".into(), json!(op.writes));
        out.insert("reads".into(), json!(op.reads));
        out.insert("terms".into(), self.terms_json(op));
        if let Some(Some(s)) = self.stiffness.get(name) {
            out.insert("stiffness".into(), s.to_json());
        }
        Value::Object(out)
    }

    /// An operator's stiffness: rules 1–3, else the Gershgorin bound from
    /// `facts`.
    fn stiffness_of(&self, name: &str, facts: Option<&StructureFacts>) -> Option<bool> {
        match self.stiffness.get(name) {
            Some(Some(s)) => Some(s.stiff),
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
        let stiffs: Option<Vec<bool>> =
            names.iter().map(|o| self.stiffness_of(o, facts)).collect();
        let stiff_column_ops: Option<Vec<String>> = ops
            .iter()
            .filter(|(_, c)| *c == Class::Column)
            .map(|(o, _)| self.stiffness_of(o, facts).map(|s| (o.clone(), s)))
            .collect::<Option<Vec<_>>>()
            .map(|v| v.into_iter().filter(|(_, s)| *s).map(|(o, _)| o).collect());
        let states = settings::leaf_states(n, &self.operators);
        let shapes: BTreeSet<&Vec<String>> =
            states.iter().map(|s| &self.model.shapes[s]).collect();
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

    /// The resolved schedule (§7.1) with every leaf's method resolved
    /// (§5.5.3). Leaves that need stage-2 facts (`facts`) are
    /// [`PlanError::NotYetImplemented`] without them.
    /// A leaf's resolved method (§5.5.3).
    pub fn leaf_strategy(&self, n: &Node, facts: Option<&StructureFacts>) -> Result<LeafStrategy> {
        strategy::resolve_leaf(
            &Value::Object(n.value.clone()),
            &self.leaf_facts(n, facts),
            &n.pointer,
        )
    }

    pub fn resolve_schedule(&self, facts: Option<&StructureFacts>) -> Result<Value> {
        let strategy = |n: &Node| self.leaf_strategy(n, facts);
        let resolved = schedule::write(&self.schedule, &self.model, &self.operators, &strategy)?;
        // §7.5's classical-Rosenbrock check for the resolved methods.
        let mut algs = BTreeMap::new();
        for n in self.schedule.leaves() {
            algs.insert(n.id().to_string(), strategy(n)?.alg);
        }
        settings::precision(&self.run_config, &self.model, &self.operators, &self.schedule, &algs)?;
        Ok(resolved)
    }

    /// The whole `plan.json`. Not available in stage 1: `structure`,
    /// `stiffness` by the Gershgorin bound, `derived` and `validation` need
    /// EarthSciASTDiff (stage 2 of M2).
    pub fn plan_json(&self) -> Result<Value> {
        Err(PlanError::NotYetImplemented(
            "plan.json needs operators.*.structure, stiffness by the Gershgorin bound, derived documents and validation, which need EarthSciASTDiff (stage 2 of M2)".into(),
        ))
    }
}
