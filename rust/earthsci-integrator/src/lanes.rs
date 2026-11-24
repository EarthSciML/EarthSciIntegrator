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
//! A lane program takes one time `t` per call (D98, U19), so a call
//! evaluates only cells that share their time, which the callers here always
//! do: a unit's cells step together. Cells are taken `capacity` at a time;
//! a remainder is padded by the program (rule 2).

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
        let prog = match &doc.source {
            Some((path, tspan)) => Arc::new(lane_form(doc, path, *tspan)?),
            None => doc.prog.clone(),
        };
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

/// The right-hand-side document `path` in the form EarthSciAST's lane
/// rewriter takes, compiled: every state equation `D(x) = faq{…}` over the
/// whole array becomes `D(x) = makearray` of one region, the state's whole
/// extent, holding that `faq`. The two are the same array (esm-spec §4.3);
/// EarthSciAST's Rust lane programs refuse the first ("subscripts an array
/// expression a lane cannot restrict", EarthSciAST pinned revision
/// 3a61fe4) and take the second, which is how the planner's Jacobian
/// documents are written. The parameters must come out in the program's
/// order, which is checked.
fn lane_form(doc: &Document, path: &std::path::Path, tspan: (f64, f64)) -> Result<Program, String> {
    use serde_json::{json, Value};
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut v: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let sizes: std::collections::HashMap<String, i64> = v["index_sets"]
        .as_object()
        .into_iter()
        .flatten()
        .filter_map(|(k, d)| Some((k.clone(), d["size"].as_i64()?)))
        .collect();
    let walk = &mut |node: &mut Value| walk_with(node, &sizes);
    fn walk_with(node: &mut Value, sizes: &std::collections::HashMap<String, i64>) {
        if let Some(eqs) = node.get_mut("equations").and_then(Value::as_array_mut) {
            for eq in eqs {
                let is_d = eq["lhs"]["op"] == "D";
                let rhs = &mut eq["rhs"];
                let whole = rhs["op"] == "faq"
                    && rhs["output_idx"].as_array().is_some_and(|o| !o.is_empty());
                if !(is_d && whole) {
                    continue;
                }
                let Some(ranges) = rhs["ranges"].as_object() else {
                    continue;
                };
                let mut region = Vec::new();
                let mut ok = true;
                for k in rhs["output_idx"].as_array().into_iter().flatten() {
                    match k.as_str().and_then(|k| ranges.get(k)) {
                        Some(Value::Array(r)) if r.len() == 2 => region.push(r.clone()),
                        Some(Value::Object(o)) => match o
                            .get("from")
                            .and_then(Value::as_str)
                            .and_then(|set| sizes.get(set))
                        {
                            Some(&n) => region.push(vec![json!(1), json!(n)]),
                            None => ok = false,
                        },
                        _ => ok = false,
                    }
                }
                if ok {
                    let f = rhs.take();
                    *rhs =
                        json!({"op": "makearray", "args": [], "regions": [region], "values": [f]});
                }
            }
        }
        for key in ["models", "subsystems"] {
            if let Some(m) = node.get_mut(key).and_then(Value::as_object_mut) {
                m.values_mut().for_each(|x| walk_with(x, sizes));
            }
        }
    }
    walk(&mut v);
    let (_, prog) = crate::planned::build_program_value(&v, tspan)?;
    if prog.parameters().iter().map(|p| &p.name).ne(doc
        .prog
        .program()
        .parameters()
        .iter()
        .map(|p| &p.name))
    {
        return Err("its lane form has other parameters than the document".into());
    }
    let shapes = prog
        .state_layout()
        .iter()
        .map(|b| (b.name.clone(), b.shape.clone()));
    let dofs = earthsci_integrator_planner::numeric::Dofs::from_shapes(shapes);
    Program::from_program(prog, &dofs)
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
