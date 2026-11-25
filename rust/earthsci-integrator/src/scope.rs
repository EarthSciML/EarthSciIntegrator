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
//! *Lane pools.* Under `cell` each executor runs a pool of `lanes` lanes, one
//! thread per lane, each integrating the cells it takes from the shared
//! queue; their evaluations are served together through lane programs that
//! take a time per lane ([`crate::lanes::Batcher`], U19). A lane's value
//! depends only on its own inputs, so the pool's grouping changes nothing.

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
    /// Tests only: panic in the lane that takes this unit, before it
    /// integrates it, or (with `test_panic_in_serve`) in the lane pool
    /// while serving an evaluation of this cell. `usize::MAX`: never.
    #[doc(hidden)]
    pub test_panic_unit: AtomicUsize,
    #[doc(hidden)]
    pub test_panic_in_serve: std::sync::atomic::AtomicBool,
}

/// The cells a unit advances, and the model state at the invocation's
/// start: the leaf's constants.
struct Target<T> {
    cells: Vec<usize>,
    /// The unit's state index `i·m + k` is model DOF `model_dofs[i·m + k]`.
    model_dofs: Vec<usize>,
    bg: Arc<Vec<T>>,
    /// The lane pool this unit's cell is evaluated in, and its lane
    /// (spec/partitioning.md §3.2); `None` outside a pool.
    batch: Option<(Arc<crate::lanes::Batcher>, usize)>,
    /// Why the unit's operators failed, in the order they did, at most
    /// once per operator: cleared when the target is set to a new unit.
    faults: Vec<(String, String)>,
}

/// One of the leaf's operators over a unit's state vector.
struct UnitOperator<T: Scalar> {
    base: Arc<dyn Operator<T>>,
    lanes: Option<Arc<LaneOps>>,
    target: Arc<Mutex<Target<T>>>,
    n: usize,
    m: usize,
    dense: bool,
}

impl<T: Scalar> UnitOperator<T> {
    /// Records why an evaluation of this operator failed, on the unit.
    fn record(&self, tg: &mut Target<T>, e: String) {
        let name = self.base.name();
        if !tg.faults.iter().any(|(o, _)| o == name) {
            tg.faults.push((name.to_string(), e));
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
        let tg = self.target.lock().expect("unit target");
        tg.faults
            .iter()
            .find(|(o, _)| o == self.base.name())
            .map(|(_, e)| e.clone())
            .or_else(|| self.base.fault())
    }
    fn rhs(&self, out: &mut [T], u: &[T], p: &[f64], t: f64) {
        match &self.lanes {
            Some(l) => {
                let mut tg = self.target.lock().expect("unit target");
                let r = match &tg.batch {
                    Some((b, lane)) => l.rhs_pooled(b, *lane, out, tg.cells[0], u, &tg.bg, p, t),
                    None => l.rhs(out, &tg.cells, u, &tg.bg, p, t),
                };
                if let Err(e) = r {
                    self.record(&mut tg, e);
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
        let mut tg = self.target.lock().expect("unit target");
        match &self.lanes {
            Some(l) => {
                let r = match &tg.batch {
                    Some((b, lane)) => l.jac_pooled(b, *lane, out, tg.cells[0], u, &tg.bg, p, t),
                    None => l.jac(out, &tg.cells, u, &tg.bg, p, t),
                };
                if let Err(e) = r {
                    self.record(&mut tg, e);
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
                let mut tg = self.target.lock().expect("unit target");
                let r = match &tg.batch {
                    Some((b, lane)) => l.ft_pooled(b, *lane, out, tg.cells[0], u, &tg.bg, p, t),
                    None => l.ft(out, &tg.cells, u, &tg.bg, p, t),
                };
                if let Err(e) = r {
                    self.record(&mut tg, e);
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
            test_panic_unit: AtomicUsize::new(usize::MAX),
            test_panic_in_serve: std::sync::atomic::AtomicBool::new(false),
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
            batch: None,
            faults: Vec::new(),
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
            signal: Mutex::new(crate::leaf::UnitSignal::new(
                if self.kind == ScopeKind::Shard { m } else { 0 },
            )),
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
    fn set(
        &self,
        cells: Vec<usize>,
        model_dofs: Vec<usize>,
        bg: Arc<Vec<T>>,
        batch: Option<(Arc<crate::lanes::Batcher>, usize)>,
    );
    /// Why the unit's operators failed since it was set, as
    /// `operator <name>: <reason>`.
    fn faults(&self) -> Vec<String>;
}

impl<T: Send + Sync> UnitTarget<T> for UnitTargetHandle<T> {
    fn set(
        &self,
        cells: Vec<usize>,
        model_dofs: Vec<usize>,
        bg: Arc<Vec<T>>,
        batch: Option<(Arc<crate::lanes::Batcher>, usize)>,
    ) {
        let mut tg = self.target.lock().expect("unit target");
        tg.cells = cells;
        tg.model_dofs = model_dofs;
        tg.bg = bg;
        tg.batch = batch;
        tg.faults.clear();
    }
    fn faults(&self) -> Vec<String> {
        let tg = self.target.lock().expect("unit target");
        tg.faults
            .iter()
            .map(|(o, e)| format!("operator {o}: {e}"))
            .collect()
    }
}

/// What one unit's integration returned.
struct UnitResult<T> {
    x: Vec<T>,
    persist: LeafPersist,
    out: Result<LeafOutcome<T>, Error>,
    /// `R_c` of each of the unit's cells (§6.2) after the unit's attempts,
    /// under `shard`.
    signal: Vec<f64>,
    /// Why the unit's operators failed (§2.3).
    faults: Vec<String>,
}

/// How an error names the unit it happened in: its cell, or its shard.
fn unit_name(kind: ScopeKind, i: usize, cells: &[usize]) -> String {
    match kind {
        ScopeKind::Cell => format!("cell {}", cells[0]),
        ScopeKind::Shard => format!("shard {i} (cells {cells:?})"),
    }
}

/// `e`, naming the unit it happened in.
fn at_unit(e: Error, name: &str) -> Error {
    match e {
        Error::Config(s) => Error::Config(format!("{s} (at {name})")),
        Error::Unimplemented(s) => Error::Unimplemented(format!("{s} (at {name})")),
        Error::Run(s) => Error::Run(format!("{s} (at {name})")),
    }
}

/// Before a `shard` leaf's first invocation of a macro step: re-shards when
/// the macro step is one `reshard: {"every": k}` names, from the signal of
/// the previous macro step, then starts the macro step's signal at zero
/// (§6.2). The signal lives in the leaf's persistent state, so a resumed run
/// sees the one the uninterrupted run had.
fn start_macro_step<T: Scalar>(plan: &ScopePlan<T>, persist: &mut LeafPersist, k: usize) {
    let nc = plan.ncells;
    if persist.signal.len() != nc {
        persist.signal = vec![0.0; nc];
    }
    if persist.signal_step == Some(k) {
        return;
    }
    if let Some(every) = plan.reshard_every {
        if every > 0 && k > 0 && k.is_multiple_of(every) && !persist.units.is_empty() {
            // R_c of the previous macro step; zero when the leaf was not
            // invoked in it.
            let r = if persist.signal_step == Some(k - 1) {
                persist.signal.clone()
            } else {
                vec![0.0; nc]
            };
            reshard(plan, persist, &r);
        }
    }
    persist.signal.iter_mut().for_each(|x| *x = 0.0);
    persist.signal_step = Some(k);
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
    if plan.kind == ScopeKind::Shard {
        if persist.shards.is_empty() {
            persist.shards = plan.initial_shards();
        }
        start_macro_step(plan, persist, call.macro_step);
    }
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
    // Set when a lane panics: the others take no more units.
    let abort = std::sync::atomic::AtomicBool::new(false);
    let threads = plan
        .threads
        .load(Ordering::Relaxed)
        .clamp(1, units.len().max(1));
    // Under `cell` with lane programs, each executor runs a lane pool of
    // `capacity` lanes (§3.2): one thread per lane, each integrating the
    // cells it takes, their evaluations served together (U19).
    let pooled = plan.kind == ScopeKind::Cell
        && plan.capacity > 1
        && !plan.whole_state.load(Ordering::Relaxed)
        && plan
            .template
            .sources
            .iter()
            .all(|s| matches!(s, UnitSource::Lanes(_)));
    let lanes_per = if pooled { plan.capacity } else { 1 };
    let panic_unit = plan.test_panic_unit.load(Ordering::Relaxed);
    let panic_cell = (panic_unit != usize::MAX && plan.test_panic_in_serve.load(Ordering::Relaxed))
        .then_some(panic_unit);
    let batchers: Vec<Option<Arc<crate::lanes::Batcher>>> = (0..threads)
        .map(|_| pooled.then(|| Arc::new(crate::lanes::Batcher::new(lanes_per, panic_cell))))
        .collect();
    let persist_units = &persist.units;
    let signal_in = &persist.signal;
    let work = |k: usize| {
        let batch = batchers[k / lanes_per]
            .as_ref()
            .map(|b| (Arc::clone(b), k % lanes_per));
        // Leaves the pool on every exit, a panic included, so no other lane
        // waits for this one; a panic also stops the other lanes.
        struct Abort<'a>(&'a std::sync::atomic::AtomicBool);
        impl Drop for Abort<'_> {
            fn drop(&mut self) {
                if std::thread::panicking() {
                    self.0.store(true, Ordering::Relaxed);
                }
            }
        }
        let _abort = Abort(&abort);
        let _lane = crate::lanes::LaneGuard(batch.as_ref().map(|(b, _)| Arc::clone(b)));
        let mut held: HashMap<usize, LeafSpec<T>> = HashMap::new();
        loop {
            if abort.load(Ordering::Relaxed) {
                break;
            }
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= units.len() {
                break;
            }
            if panic_cell.is_none() && i == panic_unit {
                panic!("test: a panic in the lane integrating unit {i}");
            }
            let cells = &units[i];
            let m = cells.len();
            let unit = held.entry(m).or_insert_with(|| plan.take_unit(leaf, m));
            let model_dofs: Vec<usize> = (0..n)
                .flat_map(|s| cells.iter().map(move |&c| leaf.dofs[s * nc + c]))
                .collect();
            let mut x: Vec<T> = model_dofs.iter().map(|&d| bg[d]).collect();
            let st = unit.unit.as_ref().expect("a unit leaf");
            st.target
                .set(cells.clone(), model_dofs, bg.clone(), batch.clone());
            // The unit's cells' R_c so far: its attempts add to it in order.
            if let Ok(mut sg) = st.signal.lock() {
                for (r, &c) in sg.r.iter_mut().zip(cells) {
                    *r = signal_in.get(c).copied().unwrap_or(0.0);
                }
            }
            let mut p = persist_units[i].clone();
            let out = unit.invoke(&mut x, &mut p, call);
            let signal = st.signal.lock().map(|s| s.r.clone()).unwrap_or_default();
            let faults = st.target.faults();
            *results[i]
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(UnitResult {
                x,
                persist: p,
                out,
                signal,
                faults,
            });
        }
        for (m, u) in held {
            plan.put_unit(m, u);
        }
    };
    let total = threads * lanes_per;
    if total == 1 {
        work(0);
    } else {
        // A panic in any lane ends every lane (above) and is re-raised here.
        std::thread::scope(|s| {
            for k in 0..total {
                let w = &work;
                s.spawn(move || w(k));
            }
        });
    }

    // Merge, units in order (§2.3, §2.4): counts summed; the first failure
    // in unit order, an error or a non-success code, naming its unit.
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
        failure: None,
    };
    let mut counts = UnitCounts {
        accepted: Vec::new(),
        rejected: Vec::new(),
    };
    let mut shard_records = Vec::new();
    let mut stop_states: Vec<(f64, Vec<T>)> = Vec::new();
    // Clips by step index, as the leaf's state-vector indices (spec/tape.md §2).
    let mut clips: std::collections::BTreeMap<usize, std::collections::BTreeSet<usize>> =
        Default::default();
    for (i, r) in results.into_iter().enumerate() {
        let r = r
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .expect("every unit ran");
        let cells = &units[i];
        let m = cells.len();
        let out = match r.out {
            Ok(o) => o,
            // An earlier unit's non-success code is the invocation's failure.
            Err(_) if merged.failure.is_some() => continue,
            Err(e) => return Err(at_unit(e, &unit_name(plan.kind, i, cells))),
        };
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
        for (step, dofs) in &out.clips {
            let e = clips.entry(*step).or_default();
            e.extend(dofs.iter().map(|&j| (j / m) * nc + cells[j % m]));
        }
        persist.units[i] = r.persist;
        merged.accepted += out.accepted;
        merged.rejected += out.rejected;
        merged.rhs_calls += out.rhs_calls;
        if merged.failure.is_none() && out.retcode != Retcode::Success {
            merged.retcode = out.retcode;
            let mut why = unit_name(plan.kind, i, cells);
            for f in &r.faults {
                why.push_str("; ");
                why.push_str(f);
            }
            merged.failure = Some(why);
        }
        match plan.kind {
            ScopeKind::Cell => {
                counts.accepted.push(out.accepted);
                counts.rejected.push(out.rejected);
            }
            ScopeKind::Shard => {
                for (k, &c) in cells.iter().enumerate() {
                    if let Some(&v) = r.signal.get(k) {
                        persist.signal[c] = v;
                    }
                }
                // A shard's cells on its first record and when they changed
                // since its last (§11).
                if persist.recorded.len() <= i {
                    persist.recorded.resize(i + 1, Vec::new());
                }
                let listed = (persist.recorded[i] != *cells).then(|| cells.clone());
                persist.recorded[i] = cells.clone();
                shard_records.push(ShardRecord {
                    cells: listed,
                    steps: out.steps.clone(),
                    accepted: out.accepted,
                    rejected: out.rejected,
                });
            }
        }
    }
    merged.stop_states = stop_states;
    merged.clips = clips
        .into_iter()
        .map(|(k, d)| (k, d.into_iter().collect()))
        .collect();
    merged.units = Some(match plan.kind {
        ScopeKind::Cell => crate::leaf::UnitOutcome::Cells(counts),
        ScopeKind::Shard => crate::leaf::UnitOutcome::Shards(shard_records),
    });
    Ok(merged)
}

/// Re-composes a `shard` leaf's shards from the signal of the previous
/// macro step (§6.2). The new shards are kept in shard order, by their
/// smallest cell (§2.4).
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
    shards.sort_by_key(|sh| sh[0]);
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
    persist.shards = shards;
    persist.units = units;
}
