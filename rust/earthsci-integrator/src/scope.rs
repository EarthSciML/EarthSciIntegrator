//! `dt_scope: cell` and `shard` (spec/partitioning.md §2.3, §2.4).
//!
//! A scoped leaf integrates **units**, each an independent initial-value
//! problem with its own controller: one cell under `cell`, one shard (a set
//! of cells) under `shard`. A unit is integrated by an ordinary
//! [`LeafSpec`] over the unit's own state vector (species-major over its
//! cells), whose operators ([`UnitOperator`]) evaluate the leaf's operators
//! at the unit's cells only, with the rest of the model state the leaf's
//! constants. So every method, controller, stop rule and domain check is the
//! one a whole leaf uses (spec/methods.md), applied per unit.
//!
//! Executors (§4) are threads: each takes units from a shared counter, in
//! unit order, and integrates each with a unit leaf it owns; units share
//! nothing while they run, so the thread count, the order in which units
//! are taken and the lane capacity change no result (§9.1). Under `cell`
//! the shards only group cells into work; under `shard` they are the units,
//! and re-sharding (§6.2) changes them at macro-step boundaries from each
//! cell's share of its shard's squared error sums.
//!
//! *Lane pools.* A unit's cells are evaluated together through the
//! operators' lane programs ([`crate::lanes`]), `lanes` at a time. Cells of
//! different units are never batched together: lane programs take one time
//! per call (D98), and cells of different units are at different times
//! after their first step. When lane programs take a time per lane (U19),
//! one pool may batch the cells of many units without changing a value.

use crate::lanes::LaneOps;
use crate::leaf::{DomainSpec, DtMode, LeafCall, LeafMethod, LeafOutcome, LeafSpec};
use crate::operator::{Call, JacOverScratch, MatrixStorage, Operator, Structure};
use crate::tape::Retcode;
use crate::tape::{LeafPersist, ShardRecord, UnitCounts};
use crate::{Error, Scalar};
use earthsci_integrator_solvers::controller::Controller;
use earthsci_integrator_solvers::norm::Tolerances;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Which DOFs share a controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    Cell,
    Shard,
}

/// How a unit evaluates one of the leaf's operators.
pub enum UnitSource<T: Scalar> {
    /// A plan's operator, through its lane programs.
    Lanes(Arc<LaneOps>),
    /// Any other operator: evaluated over the whole model state, with the
    /// unit's cells set, and read at the unit's rows. Exact for a pointwise
    /// operator, whose rows at a cell read that cell only; used for
    /// hand-written operators and as the lane programs' oracle in tests.
    Whole,
    #[doc(hidden)]
    _T(std::marker::PhantomData<T>),
}

/// What a unit leaf copies from the scoped leaf.
pub struct Template<T: Scalar> {
    pub alg: String,
    pub method: LeafMethod,
    pub dt_mode: DtMode,
    /// `abstol` per species (equal over a species' cells).
    pub abstol: Vec<f64>,
    pub reltol: f64,
    pub controller: Option<Controller>,
    pub dt0: Option<f64>,
    pub dtmin: f64,
    pub dtmax: Option<f64>,
    pub maxiters: Option<u64>,
    /// `domain`: the species listed (each with every cell), clip, stages.
    pub domain: Option<(Vec<usize>, bool, bool)>,
    pub imex_implicit: Vec<bool>,
    /// The leaf's storage is `dense` (a `dense_lu` leaf).
    pub dense: bool,
    pub ops: Vec<Arc<dyn Operator<T>>>,
    pub sources: Vec<UnitSource<T>>,
}

/// The re-sharding signal (§6.2): `R_c` summed over the current macro step,
/// and over the previous one.
#[derive(Debug, Default)]
struct Signal {
    macro_step: Option<usize>,
    current: Vec<f64>,
    previous: Vec<f64>,
}

/// A scoped leaf's plan: its units and how to build a unit's leaf.
pub struct ScopePlan<T: Scalar> {
    pub kind: ScopeKind,
    /// The leaf's cells and species.
    pub ncells: usize,
    pub n: usize,
    /// The plan's shard count (§6.1).
    pub nshards: usize,
    /// `reshard: {"every": k}` (§6.2), under `shard`.
    pub reshard_every: Option<usize>,
    /// Lanes per lane-program call.
    pub capacity: usize,
    /// Executor threads (a run-time option, §4).
    pub threads: AtomicUsize,
    /// Evaluate every operator over the whole model state instead of its
    /// lane programs: the oracle the lane programs are tested against
    /// (§9.3). Tests only.
    pub whole_state: std::sync::atomic::AtomicBool,
    pub template: Template<T>,
    /// Unit leaves not in use, by unit size.
    pool: Mutex<HashMap<usize, Vec<LeafSpec<T>>>>,
    signal: Mutex<Signal>,
}

/// The cells a unit advances, and the model state at the invocation's
/// start: the leaf's constants.
struct Target<T> {
    cells: Vec<usize>,
    /// The unit's state index `i·m + k` is model DOF `model_dofs[i·m + k]`.
    model_dofs: Vec<usize>,
    bg: Arc<Vec<T>>,
}

/// One of the leaf's operators over a unit's state vector.
struct UnitOperator<T: Scalar> {
    base: Arc<dyn Operator<T>>,
    lanes: Option<Arc<LaneOps>>,
    target: Arc<Mutex<Target<T>>>,
    n: usize,
    m: usize,
    dense: bool,
    fault: Mutex<Option<String>>,
}

impl<T: Scalar> UnitOperator<T> {
    fn record(&self, e: String) {
        let mut f = self.fault.lock().expect("fault");
        if f.is_none() {
            *f = Some(e);
        }
    }

    /// The model state with the unit's state `u` at its DOFs.
    fn full(tg: &Target<T>, u: &[T]) -> Vec<T> {
        let mut full = tg.bg.as_ref().clone();
        for (k, &d) in tg.model_dofs.iter().enumerate() {
            full[d] = u[k];
        }
        full
    }

    /// `out += g(model)` at the unit's rows, for a call `g` over the whole
    /// model state.
    fn whole(&self, out: &mut [T], u: &[T], g: impl FnOnce(&mut [T], &[T])) {
        let tg = self.target.lock().expect("unit target");
        let full = Self::full(&tg, u);
        let mut fo = vec![T::zero(); full.len()];
        g(&mut fo, &full);
        for (k, &d) in tg.model_dofs.iter().enumerate() {
            out[k] += fo[d];
        }
    }
}

impl<T: Scalar> Operator<T> for UnitOperator<T> {
    fn name(&self) -> &str {
        self.base.name()
    }
    fn structure(&self) -> Structure {
        if self.dense {
            Structure::Dense
        } else {
            Structure::BlockDiagonal {
                ncells: self.m,
                n: self.n,
            }
        }
    }
    fn depends_on_t(&self) -> bool {
        self.base.depends_on_t()
    }
    fn linear_in_u(&self) -> bool {
        self.base.linear_in_u()
    }
    fn writes(&self) -> Vec<usize> {
        (0..self.n * self.m).collect()
    }
    fn structure_over(&self, _dofs: &[usize], _n_model: usize) -> Result<Structure, String> {
        Ok(self.structure())
    }
    fn forcing_refresh_times(&self) -> Vec<f64> {
        self.base.forcing_refresh_times()
    }
    fn provides(&self, call: Call) -> bool {
        self.base.provides(call)
    }
    fn fault(&self) -> Option<String> {
        self.fault
            .lock()
            .expect("fault")
            .clone()
            .or_else(|| self.base.fault())
    }
    fn rhs(&self, out: &mut [T], u: &[T], p: &[f64], t: f64) {
        match &self.lanes {
            Some(l) => {
                let tg = self.target.lock().expect("unit target");
                if let Err(e) = l.rhs(out, &tg.cells, u, &tg.bg, p, t) {
                    self.record(e);
                    out.iter_mut().for_each(|o| *o = T::from_f64(f64::NAN));
                }
            }
            None => self.whole(out, u, |o, full| self.base.rhs(o, full, p, t)),
        }
    }
    fn jvp_u(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, v: &[T]) {
        let vt = {
            let tg = self.target.lock().expect("unit target");
            let mut vf = vec![T::zero(); tg.bg.len()];
            for (k, &d) in tg.model_dofs.iter().enumerate() {
                vf[d] = v[k];
            }
            vf
        };
        self.whole(out, u, |o, full| self.base.jvp_u(o, full, p, t, &vt))
    }
    fn vjp_u(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, w: &[T]) {
        let wt = {
            let tg = self.target.lock().expect("unit target");
            let mut wf = vec![T::zero(); tg.bg.len()];
            for (k, &d) in tg.model_dofs.iter().enumerate() {
                wf[d] = w[k];
            }
            wf
        };
        self.whole(out, u, |o, full| self.base.vjp_u(o, full, p, t, &wt))
    }
    fn jac(&self, out: &mut MatrixStorage<T>, u: &[T], p: &[f64], t: f64) {
        let tg = self.target.lock().expect("unit target");
        match &self.lanes {
            Some(l) => {
                if let Err(e) = l.jac(out, &tg.cells, u, &tg.bg, p, t) {
                    self.record(e);
                    for i in 0..self.n * self.m {
                        out.add(i, i, T::from_f64(f64::NAN));
                    }
                }
            }
            None => {
                // The unit's DOFs ascend in the model (cells ascending,
                // species in canonical order), so `jac_over` places them at
                // the unit's indices.
                let full = Self::full(&tg, u);
                let mut scratch = JacOverScratch::default();
                self.base
                    .jac_over(out, &tg.model_dofs, &full, p, t, &mut scratch);
            }
        }
    }
    fn ft(&self, out: &mut [T], u: &[T], p: &[f64], t: f64) {
        match &self.lanes {
            Some(l) => {
                let tg = self.target.lock().expect("unit target");
                if let Err(e) = l.ft(out, &tg.cells, u, &tg.bg, p, t) {
                    self.record(e);
                    out.iter_mut().for_each(|o| *o = T::from_f64(f64::NAN));
                }
            }
            None => self.whole(out, u, |o, full| self.base.ft(o, full, p, t)),
        }
    }
}

impl<T: Scalar> ScopePlan<T> {
    /// A scoped leaf's plan.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kind: ScopeKind,
        ncells: usize,
        n: usize,
        nshards: usize,
        reshard_every: Option<usize>,
        capacity: usize,
        template: Template<T>,
    ) -> ScopePlan<T> {
        ScopePlan {
            kind,
            ncells,
            n,
            nshards: nshards.max(1),
            reshard_every,
            capacity: capacity.max(1),
            threads: AtomicUsize::new(1),
            whole_state: std::sync::atomic::AtomicBool::new(false),
            template,
            pool: Mutex::new(HashMap::new()),
            signal: Mutex::new(Signal::default()),
        }
    }

    /// The initial composition (§6.1): contiguous runs of the canonical
    /// cell order, empty shards dropped.
    pub fn initial_shards(&self) -> Vec<Vec<usize>> {
        let (c, s) = (self.ncells, self.nshards);
        (0..s)
            .map(|k| (k * c / s..(k + 1) * c / s).collect::<Vec<_>>())
            .filter(|v| !v.is_empty())
            .collect()
    }

    /// A unit leaf for `m` cells: one from the pool, or a new one.
    fn take_unit(&self, leaf: &LeafSpec<T>, m: usize) -> LeafSpec<T> {
        if let Some(u) = self
            .pool
            .lock()
            .ok()
            .and_then(|mut p| p.get_mut(&m).and_then(Vec::pop))
        {
            return u;
        }
        let tp = &self.template;
        let n = self.n;
        let target = Arc::new(Mutex::new(Target {
            cells: Vec::new(),
            model_dofs: Vec::new(),
            bg: Arc::new(Vec::new()),
        }));
        let ops: Vec<Arc<dyn Operator<T>>> = tp
            .ops
            .iter()
            .zip(&tp.sources)
            .map(|(base, src)| {
                let o: Arc<dyn Operator<T>> = Arc::new(UnitOperator {
                    base: base.clone(),
                    lanes: match src {
                        UnitSource::Lanes(l) if !self.whole_state.load(Ordering::Relaxed) => {
                            Some(l.clone())
                        }
                        _ => None,
                    },
                    target: target.clone(),
                    n,
                    m,
                    dense: tp.dense,
                    fault: Mutex::new(None),
                });
                o
            })
            .collect();
        let abstol = (0..n * m).map(|k| tp.abstol[k / m]).collect();
        let domain = tp
            .domain
            .as_ref()
            .map(|(species, clip, stages)| DomainSpec {
                dofs: species
                    .iter()
                    .flat_map(|&i| (0..m).map(move |k| i * m + k))
                    .collect(),
                clip: *clip,
                check_stages: *stages,
            });
        let mut u = LeafSpec {
            id: leaf.id.clone(),
            ops,
            dofs: (0..n * m).collect(),
            n_model: n * m,
            structure: if tp.dense {
                Structure::Dense
            } else {
                Structure::BlockDiagonal { ncells: m, n }
            },
            alg: tp.alg.clone(),
            method: tp.method.clone(),
            dt_mode: tp.dt_mode,
            tol: Tolerances {
                abstol,
                reltol: tp.reltol,
            },
            controller: tp.controller.clone(),
            dt0: tp.dt0,
            dtmin: tp.dtmin,
            dtmax: tp.dtmax,
            maxiters: tp.maxiters,
            domain,
            binds: Vec::new(),
            integrand: None,
            integrand_const: Vec::new(),
            imex_implicit: tp.imex_implicit.clone(),
            work: Default::default(),
            bound_work: Default::default(),
            scope: None,
            unit: None,
        };
        u.unit = Some(crate::leaf::UnitState {
            target: Box::new(UnitTargetHandle { target }),
            cells: Mutex::new(vec![0.0; if self.kind == ScopeKind::Shard { m } else { 0 }]),
            last: Mutex::new(vec![0.0; if self.kind == ScopeKind::Shard { m } else { 0 }]),
            ncells: m,
        });
        u
    }

    fn put_unit(&self, m: usize, u: LeafSpec<T>) {
        if let Ok(mut p) = self.pool.lock() {
            p.entry(m).or_default().push(u);
        }
    }
}

/// The handle a unit leaf keeps on its operators' target.
struct UnitTargetHandle<T> {
    target: Arc<Mutex<Target<T>>>,
}

/// Points a unit leaf's operators at a unit.
pub trait UnitTarget<T>: Send + Sync {
    fn set(&self, cells: Vec<usize>, model_dofs: Vec<usize>, bg: Arc<Vec<T>>);
}

impl<T: Send + Sync> UnitTarget<T> for UnitTargetHandle<T> {
    fn set(&self, cells: Vec<usize>, model_dofs: Vec<usize>, bg: Arc<Vec<T>>) {
        let mut tg = self.target.lock().expect("unit target");
        tg.cells = cells;
        tg.model_dofs = model_dofs;
        tg.bg = bg;
    }
}

/// What one unit's integration returned.
struct UnitResult<T> {
    x: Vec<T>,
    persist: LeafPersist,
    out: Result<LeafOutcome<T>, Error>,
    /// `R_c` of each of the unit's cells (§6.2), under `shard`.
    signal: Vec<f64>,
}

/// One invocation of a scoped leaf: every unit integrated over
/// `[call.t0, call.t1]` on the executor threads, then merged.
pub fn invoke<T: Scalar>(
    leaf: &LeafSpec<T>,
    plan: &ScopePlan<T>,
    u: &mut [T],
    persist: &mut LeafPersist,
    call: &LeafCall<'_>,
) -> Result<LeafOutcome<T>, Error> {
    if call.replay.is_some() || call.record_stages {
        return Err(Error::Unimplemented(format!(
            "leaf {}: replay with a recorded step sequence and the adjoint of a dt_scope cell or shard leaf arrive with M4 (spec/partitioning.md §11)",
            leaf.id
        )));
    }
    let (n, nc) = (plan.n, plan.ncells);
    // The units: cells, or the shards of the current composition.
    if plan.kind == ScopeKind::Shard && persist.shards.is_empty() {
        persist.shards = plan.initial_shards();
    }
    let mut signal = plan.signal.lock().expect("signal");
    if signal.current.len() != nc {
        signal.current = vec![0.0; nc];
        signal.previous = vec![0.0; nc];
    }
    if signal.macro_step != Some(call.macro_step) {
        if signal.macro_step.is_some() {
            signal.previous = std::mem::replace(&mut signal.current, vec![0.0; nc]);
        }
        signal.macro_step = Some(call.macro_step);
        if plan.kind == ScopeKind::Shard {
            if let Some(k) = plan.reshard_every {
                if call.macro_step > 0 && call.macro_step % k == 0 && !persist.units.is_empty() {
                    reshard(plan, persist, &signal.previous);
                }
            }
        }
    }
    drop(signal);
    let units: Vec<Vec<usize>> = match plan.kind {
        ScopeKind::Cell => (0..nc).map(|c| vec![c]).collect(),
        ScopeKind::Shard => persist.shards.clone(),
    };
    if persist.units.len() != units.len() {
        persist.units = units.iter().map(|_| leaf.initial_persist()).collect();
    }
    let bg = Arc::new(u.to_vec());
    let results: Vec<Mutex<Option<UnitResult<T>>>> =
        units.iter().map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let threads = plan
        .threads
        .load(Ordering::Relaxed)
        .clamp(1, units.len().max(1));
    let work = |_: usize| {
        let mut held: HashMap<usize, LeafSpec<T>> = HashMap::new();
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= units.len() {
                break;
            }
            let cells = &units[i];
            let m = cells.len();
            let unit = held.entry(m).or_insert_with(|| plan.take_unit(leaf, m));
            let model_dofs: Vec<usize> = (0..n)
                .flat_map(|s| cells.iter().map(move |&c| leaf.dofs[s * nc + c]))
                .collect();
            let mut x: Vec<T> = model_dofs.iter().map(|&d| bg[d]).collect();
            let st = unit.unit.as_ref().expect("a unit leaf");
            st.target.set(cells.clone(), model_dofs, bg.clone());
            if let Ok(mut c) = st.cells.lock() {
                c.iter_mut().for_each(|v| *v = 0.0);
            }
            let mut p = persist.units[i].clone();
            let out = unit.invoke(&mut x, &mut p, call);
            let signal = st.cells.lock().map(|c| c.clone()).unwrap_or_default();
            *results[i].lock().expect("result") = Some(UnitResult {
                x,
                persist: p,
                out,
                signal,
            });
        }
        for (m, u) in held {
            plan.put_unit(m, u);
        }
    };
    if threads == 1 {
        work(0);
    } else {
        std::thread::scope(|s| {
            for k in 0..threads {
                let w = &work;
                s.spawn(move || w(k));
            }
        });
    }

    // Merge, units in order (§2.3, §2.4): counts summed, the first
    // non-success code.
    let mut merged = LeafOutcome {
        steps: Vec::new(),
        accepted: 0,
        rejected: 0,
        retcode: Retcode::Success,
        clips: Vec::new(),
        rhs_calls: 0,
        integral: 0.0,
        stage_data: Vec::new(),
        stop_states: Vec::new(),
        background: None,
        units: None,
    };
    let mut counts = UnitCounts {
        accepted: Vec::new(),
        rejected: Vec::new(),
    };
    let mut shard_records = Vec::new();
    let mut stop_states: Vec<(f64, Vec<T>)> = Vec::new();
    let mut signal = plan.signal.lock().expect("signal");
    for (i, r) in results.into_iter().enumerate() {
        let r = r.into_inner().expect("result").expect("every unit ran");
        let cells = &units[i];
        let m = cells.len();
        let out = r.out?;
        for s in 0..n {
            for (k, &c) in cells.iter().enumerate() {
                u[leaf.dofs[s * nc + c]] = r.x[s * m + k];
            }
        }
        // The model state at each interior stop: the start state with every
        // unit's state at that stop.
        for (j, (ts, xs)) in out.stop_states.iter().enumerate() {
            if stop_states.len() <= j {
                stop_states.push((*ts, bg.as_ref().clone()));
            }
            for s in 0..n {
                for (k, &c) in cells.iter().enumerate() {
                    stop_states[j].1[leaf.dofs[s * nc + c]] = xs[s * m + k];
                }
            }
        }
        persist.units[i] = r.persist;
        merged.accepted += out.accepted;
        merged.rejected += out.rejected;
        merged.rhs_calls += out.rhs_calls;
        if merged.retcode == Retcode::Success && out.retcode != Retcode::Success {
            merged.retcode = out.retcode;
        }
        match plan.kind {
            ScopeKind::Cell => {
                counts.accepted.push(out.accepted);
                counts.rejected.push(out.rejected);
            }
            ScopeKind::Shard => {
                for (k, &c) in cells.iter().enumerate() {
                    signal.current[c] += r.signal.get(k).copied().unwrap_or(0.0);
                }
                shard_records.push(ShardRecord {
                    cells: Some(cells.clone()),
                    steps: out.steps.clone(),
                    accepted: out.accepted,
                    rejected: out.rejected,
                });
            }
        }
    }
    merged.stop_states = stop_states;
    merged.units = Some(match plan.kind {
        ScopeKind::Cell => crate::leaf::UnitOutcome::Cells(counts),
        ScopeKind::Shard => crate::leaf::UnitOutcome::Shards(shard_records),
    });
    Ok(merged)
}

/// Re-composes a `shard` leaf's shards from the signal of the previous
/// macro step (§6.2).
fn reshard<T: Scalar>(plan: &ScopePlan<T>, persist: &mut LeafPersist, r: &[f64]) {
    let mut order: Vec<usize> = (0..plan.ncells).collect();
    // R_c descending, ties (equal bits) by cell index.
    order.sort_by(|&a, &b| {
        if r[a].to_bits() == r[b].to_bits() {
            a.cmp(&b)
        } else {
            r[b].total_cmp(&r[a]).then(a.cmp(&b))
        }
    });
    let (c, s) = (plan.ncells, persist.shards.len().max(1));
    let mut shards: Vec<Vec<usize>> = (0..s)
        .map(|k| {
            let mut v: Vec<usize> = order[k * c / s..(k + 1) * c / s].to_vec();
            v.sort_unstable();
            v
        })
        .filter(|v| !v.is_empty())
        .collect();
    // Each new shard's controller starts from the old shard that held its
    // smallest cell.
    let owner = |cell: usize| {
        persist
            .shards
            .iter()
            .position(|sh| sh.binary_search(&cell).is_ok())
            .unwrap_or(0)
    };
    let units: Vec<LeafPersist> = shards
        .iter()
        .map(|sh| persist.units[owner(sh[0])].clone())
        .collect();
    shards.shrink_to_fit();
    persist.shards = shards;
    persist.units = units;
}
