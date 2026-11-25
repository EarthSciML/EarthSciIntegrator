//! A plan's operators evaluated over lists of cells (spec/partitioning.md §3).
//!
//! Under `dt_scope: cell` or `shard` (§2.3, §2.4) a pointwise leaf's
//! operators are evaluated at a few cells at a time, never over the whole
//! model state. EarthSciAST's lane programs (`RhsProgram::lanes`, U3) do this
//! with each lane equal, bit for bit, to the whole-domain program at its
//! cell (§3.1 rule 4). A [`LaneOps`] holds an operator's lane programs (its
//! right-hand side and its Jacobian document) and the maps between their lane
//! layouts and a unit's state: a unit is the list of cells one integration
//! advances, its state species-major over those cells.
//!
//! Under `dt_scope: cell` a lane pool (spec/partitioning.md §3.2) integrates
//! each cell on a lane of its own, and [`Batcher`] serves the cells'
//! evaluations together: one lane-program call per document, forcing set and
//! parameter vector, each lane at its own time (U19, EarthSciAST#781). Under
//! `shard` a unit's cells step together and are evaluated `capacity` at a
//! time; a remainder is padded by the program (rule 2).

use crate::forcing::ForcingTimeline;
use crate::operator::MatrixStorage;
use crate::planned::{Document, PlannedOperator};
use crate::Scalar;
use earthsci_ast::{LaneKind, LaneProgram, RhsWorkspace};
use earthsci_integrator_planner::numeric::Program;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Where a lane program's input block takes its value at a cell.
#[derive(Debug, Clone, Copy)]
enum Source {
    /// The unit's state: species `i` of the leaf.
    Leaf(usize),
    /// A constant of the leaf: the model state at `start + cell`.
    Model(usize),
}

/// One derived document's lane program and its inputs.
struct LaneDoc {
    lanes: LaneProgram,
    prog: Arc<Program>,
    /// Each parameter of the program, from the problem's `p` (§8.2).
    p_of: Vec<crate::planned::ParamSource>,
    /// `(u-block lane offset, source)` for every block of `u`, and of `aux`.
    u_fill: Vec<(usize, Source)>,
    aux_fill: Vec<(usize, Source)>,
    u_len: usize,
    aux_len: usize,
    pool: Mutex<Vec<Scratch>>,
}

struct Scratch {
    ws: RhsWorkspace,
    u: Vec<f64>,
    aux: Vec<f64>,
    du: Vec<f64>,
    p: Vec<f64>,
    cells: Vec<i32>,
}

impl LaneDoc {
    fn new(
        doc: &Document,
        kind: &LaneKind,
        capacity: usize,
        state_source: &dyn Fn(&str) -> Option<Source>,
        what: &str,
    ) -> Result<LaneDoc, String> {
        let prog = doc.prog.clone();
        let lanes = prog
            .program()
            .lanes(kind.clone(), capacity)
            .map_err(|e| format!("its {what} document over lanes: {e}"))?;
        let layout = lanes.layout().clone();
        let fill = |blocks: &[earthsci_ast::LaneBlock]| -> Result<Vec<(usize, Source)>, String> {
            let mut out = Vec::new();
            for b in blocks {
                if !b.sets.is_empty() {
                    return Err(format!(
                        "its {what} document's lane block {} has index sets {:?} beside the cell's; \
                         only per-cell blocks are supported (spec/partitioning.md §2.3)",
                        b.name, b.sets
                    ));
                }
                // A block the program writes and does not read (a band field
                // of the Jacobian document) has no input.
                if let Some(src) = state_source(&b.name) {
                    out.push((b.offset, src));
                }
            }
            Ok(out)
        };
        let u_fill = fill(&layout.u)?;
        let aux_fill = fill(&layout.aux)?;
        Ok(LaneDoc {
            u_len: layout.u_len,
            aux_len: layout.aux_len,
            lanes,
            prog,
            p_of: doc.p_index.clone(),
            u_fill,
            aux_fill,
            pool: Mutex::new(Vec::new()),
        })
    }

    fn block_offset(&self, name: &str) -> Option<usize> {
        self.lanes.layout().block(name).map(|b| b.offset)
    }

    /// Evaluates the program at the unit's cells `cells[at..at+nb]`, with
    /// `x` the unit's state (`m` cells, species-major), `bg` the model state
    /// and `coord(c, a)` cell `c`'s 1-based index on lane set `a`; hands the
    /// lane-layout output and the lane count to `g`.
    #[allow(clippy::too_many_arguments)]
    fn eval<T: Scalar>(
        &self,
        cells: &[usize],
        at: usize,
        nb: usize,
        m: usize,
        x: &[T],
        bg: &[T],
        p: &[f64],
        t: f64,
        timeline: &ForcingTimeline,
        coords: &[Vec<i32>],
        g: &mut dyn FnMut(&[f64], usize),
    ) -> Result<(), String> {
        let cap = self.lanes.capacity();
        let d = coords.first().map_or(0, Vec::len);
        let popped = self.pool.lock().expect("lane pool").pop();
        let mut s = popped.unwrap_or_else(|| Scratch {
            ws: self.lanes.new_workspace(),
            u: vec![0.0; self.u_len],
            aux: vec![0.0; self.aux_len],
            du: vec![0.0; self.u_len],
            p: vec![0.0; self.p_of.len()],
            cells: vec![1; d.max(1) * cap],
        });
        let put = |buf: &mut [f64], fills: &[(usize, Source)]| {
            for &(off, src) in fills {
                for l in 0..nb {
                    let k = at + l;
                    buf[off * cap + l] = match src {
                        Source::Leaf(i) => x[i * m + k].to_f64(),
                        Source::Model(start) => bg[start + cells[k]].to_f64(),
                    };
                }
            }
        };
        put(&mut s.u, &self.u_fill);
        put(&mut s.aux, &self.aux_fill);
        for l in 0..nb {
            s.cells[l * d..(l + 1) * d].copy_from_slice(&coords[cells[at + l]]);
        }
        for (i, slot) in s.p.iter_mut().enumerate() {
            *slot = match self.p_of[i] {
                crate::planned::ParamSource::P(k) => p[k],
                crate::planned::ParamSource::Fixed(v) => v,
            };
        }
        let res = self.prog.forcing_set(timeline.at(t)).and_then(|fs| {
            self.lanes
                .eval_into(
                    &s.u,
                    &s.aux,
                    &s.p,
                    t,
                    &fs,
                    &s.cells[..d * nb],
                    nb,
                    &mut s.du,
                    &mut s.ws,
                )
                .map_err(|e| format!("EarthSciAST lane evaluation: {e}"))
        });
        if res.is_ok() {
            g(&s.du, cap);
        }
        self.pool.lock().expect("lane pool").push(s);
        res
    }
}

/// A plan's pointwise operator over lists of cells of one leaf.
pub struct LaneOps {
    name: String,
    /// Each cell's 1-based index on each lane set (the leaf's states'
    /// index sets, in order), by canonical cell.
    coords: Arc<Vec<Vec<i32>>>,
    timeline: Arc<ForcingTimeline>,
    rhs: LaneDoc,
    jac: LaneDoc,
    /// `(du lane offset, leaf species)` of every leaf species the operator
    /// writes.
    rhs_out: Vec<(usize, usize)>,
    /// Per canonical cell, the Jacobian entries in the whole-domain
    /// operator's point order: `(du lane offset of the band field, row leaf
    /// species, column leaf species)`. Points outside the leaf are dropped,
    /// as `jac_over` drops them.
    jac_out: Vec<Vec<(usize, usize, usize)>>,
    /// The time-derivative document (spec/plan.md §6.4) and, per band
    /// field, `(du lane offset, row leaf species)`.
    ft: Option<(LaneDoc, Vec<(usize, usize)>)>,
}

impl LaneOps {
    /// The lane programs of `op` for a leaf whose state variables start at
    /// the canonical DOFs `species` (in leaf order), each of `ncells` cells
    /// over the index sets `sets` with extents `extents`.
    pub fn new(
        op: &PlannedOperator,
        species: &[usize],
        capacity: usize,
    ) -> Result<LaneOps, String> {
        // The cells: the shape of the leaf's first species, whose index
        // sets are the lane sets.
        let first = species
            .first()
            .ok_or_else(|| format!("operator {}: a leaf with no species", op.name))?;
        let extents: Vec<usize> = op
            .states
            .iter()
            .find(|(s, _)| s == first)
            .map(|(_, e)| e.clone())
            .ok_or_else(|| format!("operator {}: no state starts at DOF {first}", op.name))?;
        let ncells: usize = extents.iter().product();
        // Every model state, by name, with its first canonical DOF.
        let names = op.rhs.prog.program().state_layout().to_vec();
        let mut starts: HashMap<String, usize> = HashMap::new();
        {
            let mut by_start: Vec<(usize, Vec<usize>)> = op.states.clone();
            by_start.sort_by_key(|s| s.0);
            // Canonical order is by code point of the state name; the
            // program's layout lists the same states.
            let mut sorted: Vec<&str> = names.iter().map(|b| b.name.as_str()).collect();
            sorted.sort_unstable();
            for (nm, (start, ext)) in sorted.iter().zip(&by_start) {
                if ext.iter().product::<usize>() != ncells {
                    return Err(format!(
                        "operator {}: state {nm} is not over the leaf's cells",
                        op.name
                    ));
                }
                starts.insert((*nm).to_string(), *start);
            }
        }
        let first_name = starts
            .iter()
            .find(|(_, &st)| st == *first)
            .map(|(n, _)| n.clone())
            .expect("the first species is a state");
        let sets: Vec<String> = op
            .rhs
            .prog
            .program()
            .flattened()
            .state_variables
            .get(&first_name)
            .and_then(|v| v.shape.clone())
            .unwrap_or_default();
        if sets.len() != extents.len() {
            return Err(format!(
                "operator {}: the state {first_name} has index sets {sets:?} for extents {extents:?}",
                op.name
            ));
        }
        let kind = LaneKind::Cells { sets: sets.clone() };
        let source = |name: &str| -> Option<Source> {
            let start = *starts.get(name)?;
            Some(match species.iter().position(|&s| s == start) {
                Some(i) => Source::Leaf(i),
                None => Source::Model(start),
            })
        };
        let rhs = LaneDoc::new(&op.rhs, &kind, capacity, &source, "right-hand-side")?;
        let jac = LaneDoc::new(&op.jacobian, &kind, capacity, &source, "Jacobian")?;

        let mut rhs_out = Vec::new();
        for (name, &start) in &starts {
            if let Some(i) = species.iter().position(|&s| s == start) {
                if op.writes.binary_search(&start).is_ok() {
                    let off = rhs
                        .block_offset(name)
                        .ok_or_else(|| format!("operator {}: no lane block for {name}", op.name))?;
                    rhs_out.push((off, i));
                }
            }
        }
        rhs_out.sort_by_key(|&(_, i)| i);

        // The Jacobian document's band fields, by their program offset.
        let mut fields: Vec<(usize, usize, usize)> = Vec::new(); // (prog base, len, lane offset)
        for b in op.jacobian.prog.program().state_layout() {
            if let Some(lo) = jac.block_offset(&b.name) {
                let len = b.shape.iter().product::<usize>().max(1);
                fields.push((b.flat_offset, len, lo));
            }
        }
        let species_of = |dof: usize| -> Option<(usize, usize)> {
            species
                .iter()
                .position(|&s| dof >= s && dof < s + ncells)
                .map(|i| (i, dof - species[i]))
        };
        let mut jac_out = vec![Vec::new(); ncells];
        for &(off, r, c) in &op.points {
            let (Some((i, cr)), Some((j, cc))) = (species_of(r), species_of(c)) else {
                continue;
            };
            if cr != cc {
                return Err(format!(
                    "operator {}: its band entry ({r}, {c}) joins two cells",
                    op.name
                ));
            }
            let Some(&(base, _, lo)) = fields.iter().find(|(b, len, _)| off >= *b && off < b + len)
            else {
                return Err(format!(
                    "operator {}: a band point has no field in the Jacobian document",
                    op.name
                ));
            };
            if off - base != cr {
                return Err(format!(
                    "operator {}: a band field is not over the leaf's cells",
                    op.name
                ));
            }
            jac_out[cr].push((lo, i, j));
        }

        let ft = match &op.time {
            None => None,
            Some((doc, places)) => {
                let lane = LaneDoc::new(doc, &kind, capacity, &source, "time-derivative")?;
                let mut out = Vec::new();
                for (k, &(_, start, _)) in places.iter().enumerate() {
                    let field = earthsci_integrator_planner::numeric::band_field(k + 1);
                    let lo = lane.block_offset(&field).ok_or_else(|| {
                        format!("operator {}: no lane block for {field}", op.name)
                    })?;
                    // A time band of a state outside the leaf adds nothing
                    // to the leaf's f_t.
                    if let Some(i) = species.iter().position(|&s| s == start) {
                        out.push((lo, i));
                    }
                }
                Some((lane, out))
            }
        };

        let mut coords = Vec::with_capacity(ncells);
        for c in 0..ncells {
            let mut rem = c;
            let mut v = Vec::with_capacity(extents.len());
            for &e in &extents {
                v.push((rem % e) as i32 + 1);
                rem /= e;
            }
            coords.push(v);
        }
        Ok(LaneOps {
            name: op.name.clone(),
            coords: Arc::new(coords),
            timeline: op.rhs.timeline.clone(),
            rhs,
            jac,
            rhs_out,
            jac_out,
            ft,
        })
    }

    /// The operator's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// `out += f` at the unit's cells: `x` the unit's state (species-major
    /// over `cells`), `bg` the model state.
    #[allow(clippy::too_many_arguments)]
    pub fn rhs<T: Scalar>(
        &self,
        out: &mut [T],
        cells: &[usize],
        x: &[T],
        bg: &[T],
        p: &[f64],
        t: f64,
    ) -> Result<(), String> {
        let m = cells.len();
        let cap = self.rhs.lanes.capacity();
        let mut at = 0;
        while at < m {
            let nb = cap.min(m - at);
            self.rhs.eval(
                cells,
                at,
                nb,
                m,
                x,
                bg,
                p,
                t,
                &self.timeline,
                &self.coords,
                &mut |du, cap| {
                    for &(off, i) in &self.rhs_out {
                        for l in 0..nb {
                            out[i * m + at + l] += T::from_f64(du[off * cap + l]);
                        }
                    }
                },
            )?;
            at += nb;
        }
        Ok(())
    }

    /// Whether the operator has a time-derivative document.
    pub fn has_ft(&self) -> bool {
        self.ft.is_some()
    }

    /// `out += f_t` at the unit's cells (spec/plan.md §6.4).
    #[allow(clippy::too_many_arguments)]
    pub fn ft<T: Scalar>(
        &self,
        out: &mut [T],
        cells: &[usize],
        x: &[T],
        bg: &[T],
        p: &[f64],
        t: f64,
    ) -> Result<(), String> {
        let Some((doc, outs)) = &self.ft else {
            return Ok(());
        };
        let m = cells.len();
        let cap = doc.lanes.capacity();
        let mut at = 0;
        while at < m {
            let nb = cap.min(m - at);
            doc.eval(
                cells,
                at,
                nb,
                m,
                x,
                bg,
                p,
                t,
                &self.timeline,
                &self.coords,
                &mut |du, cap| {
                    for &(lo, i) in outs {
                        for l in 0..nb {
                            out[i * m + at + l] += T::from_f64(du[lo * cap + l]);
                        }
                    }
                },
            )?;
            at += nb;
        }
        Ok(())
    }

    /// The operator's Jacobian blocks at the unit's cells, added into `out`
    /// (`block_diagonal` over the unit, or dense).
    #[allow(clippy::too_many_arguments)]
    pub fn jac<T: Scalar>(
        &self,
        out: &mut MatrixStorage<T>,
        cells: &[usize],
        x: &[T],
        bg: &[T],
        p: &[f64],
        t: f64,
    ) -> Result<(), String> {
        let m = cells.len();
        let cap = self.jac.lanes.capacity();
        let mut at = 0;
        while at < m {
            let nb = cap.min(m - at);
            self.jac.eval(
                cells,
                at,
                nb,
                m,
                x,
                bg,
                p,
                t,
                &self.timeline,
                &self.coords,
                &mut |du, cap| {
                    for l in 0..nb {
                        let k = at + l;
                        for &(lo, i, j) in &self.jac_out[cells[k]] {
                            out.add(i * m + k, j * m + k, T::from_f64(du[lo * cap + l]));
                        }
                    }
                },
            )?;
            at += nb;
        }
        Ok(())
    }
}

/// Which derived document of an operator a lane request evaluates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Doc {
    Rhs,
    Jac,
    Ft,
}

/// One cell's evaluation, made by the cell's own integration and served
/// together with the other lanes of its pool ([`Batcher`]).
struct Request {
    ops: Arc<LaneOps>,
    doc: Doc,
    cell: usize,
    /// The lane's `u` and `aux` columns, in the program's fill order.
    u: Vec<f64>,
    aux: Vec<f64>,
    p: Vec<f64>,
    t: f64,
    fs: earthsci_ast::ForcingSet,
}

impl LaneOps {
    fn doc(&self, d: Doc) -> &LaneDoc {
        match d {
            Doc::Rhs => &self.rhs,
            Doc::Jac => &self.jac,
            Doc::Ft => &self.ft.as_ref().expect("a time-derivative document").0,
        }
    }

    /// The request for cell `cell` with state `x` (one cell, species in leaf
    /// order), model state `bg`, parameters `p`, time `t`.
    fn request<T: Scalar>(
        self: &Arc<Self>,
        doc: Doc,
        cell: usize,
        x: &[T],
        bg: &[T],
        p: &[f64],
        t: f64,
    ) -> Result<Request, String> {
        let d = self.doc(doc);
        let col = |fills: &[(usize, Source)]| -> Vec<f64> {
            fills
                .iter()
                .map(|&(_, src)| match src {
                    Source::Leaf(i) => x[i].to_f64(),
                    Source::Model(start) => bg[start + cell].to_f64(),
                })
                .collect()
        };
        let pv = d
            .p_of
            .iter()
            .map(|s| match *s {
                crate::planned::ParamSource::P(k) => p[k],
                crate::planned::ParamSource::Fixed(v) => v,
            })
            .collect();
        let fs = d.prog.forcing_set(self.timeline.at(t))?;
        Ok(Request {
            ops: Arc::clone(self),
            doc,
            cell,
            u: col(&d.u_fill),
            aux: col(&d.aux_fill),
            p: pv,
            t,
            fs,
        })
    }

    /// `out += f` at one cell through the pool's batcher (U19: the cells of
    /// one call are at their own times).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rhs_pooled<T: Scalar>(
        self: &Arc<Self>,
        b: &Batcher,
        lane: usize,
        out: &mut [T],
        cell: usize,
        x: &[T],
        bg: &[T],
        p: &[f64],
        t: f64,
    ) -> Result<(), String> {
        let du = b.submit(lane, self.request(Doc::Rhs, cell, x, bg, p, t)?)?;
        for &(off, i) in &self.rhs_out {
            out[i] += T::from_f64(du[off]);
        }
        Ok(())
    }

    /// `out += f_t` at one cell through the pool's batcher.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ft_pooled<T: Scalar>(
        self: &Arc<Self>,
        b: &Batcher,
        lane: usize,
        out: &mut [T],
        cell: usize,
        x: &[T],
        bg: &[T],
        p: &[f64],
        t: f64,
    ) -> Result<(), String> {
        let Some((_, outs)) = &self.ft else {
            return Ok(());
        };
        let du = b.submit(lane, self.request(Doc::Ft, cell, x, bg, p, t)?)?;
        for &(lo, i) in outs {
            out[i] += T::from_f64(du[lo]);
        }
        Ok(())
    }

    /// The Jacobian block at one cell through the pool's batcher, added into
    /// `out` (one cell's storage), in the whole-domain operator's point
    /// order.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn jac_pooled<T: Scalar>(
        self: &Arc<Self>,
        b: &Batcher,
        lane: usize,
        out: &mut MatrixStorage<T>,
        cell: usize,
        x: &[T],
        bg: &[T],
        p: &[f64],
        t: f64,
    ) -> Result<(), String> {
        let du = b.submit(lane, self.request(Doc::Jac, cell, x, bg, p, t)?)?;
        for &(lo, i, j) in &self.jac_out[cell] {
            out.add(i, j, T::from_f64(du[lo]));
        }
        Ok(())
    }
}

impl LaneDoc {
    /// Evaluates `reqs` (one document, one forcing set, equal parameters),
    /// `capacity` at a time, each lane at its own time (U19); returns each
    /// request's output column: the value of every block at its lane.
    fn eval_requests(
        &self,
        reqs: &[&Request],
        coords: &[Vec<i32>],
    ) -> Result<Vec<Vec<f64>>, String> {
        let cap = self.lanes.capacity();
        let d = coords.first().map_or(0, Vec::len);
        let nblocks = self.u_len / cap.max(1);
        let popped = self.pool.lock().expect("lane pool").pop();
        let mut s = popped.unwrap_or_else(|| Scratch {
            ws: self.lanes.new_workspace(),
            u: vec![0.0; self.u_len],
            aux: vec![0.0; self.aux_len],
            du: vec![0.0; self.u_len],
            p: vec![0.0; self.p_of.len()],
            cells: vec![1; d.max(1) * cap],
        });
        let mut out = Vec::with_capacity(reqs.len());
        let mut res = Ok(());
        for chunk in reqs.chunks(cap.max(1)) {
            let nb = chunk.len();
            let mut ts = Vec::with_capacity(nb);
            for (l, r) in chunk.iter().enumerate() {
                for (k, &(off, _)) in self.u_fill.iter().enumerate() {
                    s.u[off * cap + l] = r.u[k];
                }
                for (k, &(off, _)) in self.aux_fill.iter().enumerate() {
                    s.aux[off * cap + l] = r.aux[k];
                }
                s.cells[l * d..(l + 1) * d].copy_from_slice(&coords[r.cell]);
                ts.push(r.t);
            }
            s.p.copy_from_slice(&chunk[0].p);
            res = self
                .lanes
                .eval_times_into(
                    &s.u,
                    &s.aux,
                    &s.p,
                    &ts,
                    &chunk[0].fs,
                    &s.cells[..d * nb],
                    nb,
                    &mut s.du,
                    &mut s.ws,
                )
                .map_err(|e| format!("EarthSciAST lane evaluation: {e}"));
            if res.is_err() {
                break;
            }
            for l in 0..nb {
                out.push((0..nblocks).map(|o| s.du[o * cap + l]).collect());
            }
        }
        self.pool.lock().expect("lane pool").push(s);
        res.map(|()| out)
    }
}

/// The largest number of distinct times any one lane-program call of a pool
/// has served, over the process: how the tests see that cells at different
/// times share a call (U19).
pub static MIXED_TIME_LANES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// The rendezvous of one lane pool (spec/partitioning.md §3.2): each lane
/// integrates its own cell on its own thread, and every evaluation it needs
/// is posted here; when every active lane has posted one, the evaluations
/// are served together, one lane-program call per document, forcing set and
/// parameter vector, each lane at its own time (U19). A lane's result
/// depends only on its own request (the lane programs' contract), so how
/// lanes are grouped changes no value.
pub struct Batcher {
    state: std::sync::Mutex<BatchState>,
    cv: std::sync::Condvar,
    /// Tests only: panic while serving an evaluation of this cell.
    test_panic_cell: Option<usize>,
}

struct BatchState {
    active: usize,
    pending: Vec<(usize, Request)>,
    done: Vec<Option<Result<Vec<f64>, String>>>,
    /// Set when a lane of the pool panicked: every evaluation posted after
    /// it fails, so no lane waits for one that will never post again.
    failed: Option<String>,
}

impl Batcher {
    /// A pool of `lanes` lanes, all active.
    pub(crate) fn new(lanes: usize, test_panic_cell: Option<usize>) -> Batcher {
        Batcher {
            state: std::sync::Mutex::new(BatchState {
                active: lanes,
                pending: Vec::new(),
                done: (0..lanes).map(|_| None).collect(),
                failed: None,
            }),
            cv: std::sync::Condvar::new(),
            test_panic_cell,
        }
    }

    /// The pool's state. A lane that panicked while holding the lock leaves
    /// it poisoned; the state stays consistent (a panic in `serve` is caught
    /// and recorded in `failed` before it unwinds further), so the other
    /// lanes go on reading it.
    fn lock(&self) -> std::sync::MutexGuard<'_, BatchState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn submit(&self, lane: usize, r: Request) -> Result<Vec<f64>, String> {
        let mut st = self.lock();
        if let Some(e) = &st.failed {
            return Err(e.clone());
        }
        st.pending.push((lane, r));
        loop {
            if let Some(res) = st.done[lane].take() {
                return res;
            }
            if let Some(e) = &st.failed {
                let e = e.clone();
                st.pending.retain(|(l, _)| *l != lane);
                return Err(e);
            }
            if !st.pending.is_empty() && st.pending.len() == st.active {
                self.serve_guarded(&mut st);
                continue;
            }
            st = self
                .cv
                .wait(st)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// The lane has no more cells, or it panicked: the pool waits for one
    /// lane fewer. Runs on every exit of a lane ([`LaneGuard`]).
    pub(crate) fn leave(&self, panicked: bool) {
        let mut st = self.lock();
        st.active -= 1;
        if panicked && st.failed.is_none() {
            st.failed = Some("a lane of the pool panicked".to_string());
        }
        if st.failed.is_none() && st.active > 0 && st.pending.len() == st.active {
            self.serve_guarded(&mut st);
        }
        self.cv.notify_all();
    }

    /// Serves the pending evaluations and wakes the lanes. A panic while
    /// serving fails every pending evaluation and the pool, wakes the other
    /// lanes, and then goes on unwinding in the serving lane.
    fn serve_guarded(&self, st: &mut BatchState) {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if let Some(c) = self.test_panic_cell {
                if st.pending.iter().any(|(_, r)| r.cell == c) {
                    panic!("test: a panic serving cell {c}");
                }
            }
            Self::serve(st)
        }));
        if let Err(p) = r {
            let e = "a lane pool's evaluation panicked".to_string();
            st.failed = Some(e.clone());
            for (lane, _) in std::mem::take(&mut st.pending) {
                st.done[lane] = Some(Err(e.clone()));
            }
            self.cv.notify_all();
            std::panic::resume_unwind(p);
        }
        self.cv.notify_all();
    }

    fn serve(st: &mut BatchState) {
        let pending = std::mem::take(&mut st.pending);
        // Groups in order of their first request; requests in lane order.
        let mut order: Vec<usize> = (0..pending.len()).collect();
        order.sort_by_key(|&k| pending[k].0);
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for k in order {
            let r = &pending[k].1;
            let same = |g: &Vec<usize>| {
                let q = &pending[g[0]].1;
                Arc::ptr_eq(&q.ops, &r.ops)
                    && q.doc == r.doc
                    && q.fs.epoch() == r.fs.epoch()
                    && q.p.len() == r.p.len()
                    && q.p
                        .iter()
                        .zip(&r.p)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
            };
            match groups.iter_mut().find(|g| same(g)) {
                Some(g) => g.push(k),
                None => groups.push(vec![k]),
            }
        }
        for g in groups {
            let first = &pending[g[0]].1;
            let mut times: Vec<u64> = g.iter().map(|&k| pending[k].1.t.to_bits()).collect();
            times.sort_unstable();
            times.dedup();
            MIXED_TIME_LANES.fetch_max(times.len(), std::sync::atomic::Ordering::Relaxed);
            let reqs: Vec<&Request> = g.iter().map(|&k| &pending[k].1).collect();
            let res = first
                .ops
                .doc(first.doc)
                .eval_requests(&reqs, &first.ops.coords);
            match res {
                Ok(cols) => {
                    for (&k, col) in g.iter().zip(cols) {
                        st.done[pending[k].0] = Some(Ok(col));
                    }
                }
                Err(e) => {
                    for &k in &g {
                        st.done[pending[k].0] = Some(Err(e.clone()));
                    }
                }
            }
        }
    }
}

/// Leaves a lane's pool when the lane's thread ends, however it ends: on a
/// panic the pool fails, so that the other lanes stop waiting for this one
/// and the panic reaches the executor's scope.
pub(crate) struct LaneGuard(pub Option<Arc<Batcher>>);

impl Drop for LaneGuard {
    fn drop(&mut self) {
        if let Some(b) = &self.0 {
            b.leave(std::thread::panicking());
        }
    }
}
