//! Numeric evaluation for passes 5 and 8 (spec/plan.md §5.5.1, §5.4, §8):
//! the canonical DOF order, the initial state, the sampled states, a
//! document's right-hand side through EarthSciAST's reentrant Rust
//! right-hand side (`RhsProgram`, U4), and the band Jacobian assembled from
//! a Jacobian document's fields.

use crate::model::ModelInfo;
use earthsci_ast::{
    esm_problem, rhs_program, Compiler, Expr, ForcingSet, ProblemInput, ProblemOptions, RhsProgram,
    RhsWorkspace,
};
use earthsci_astdiff::{JacEntry, RIdx};
use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

/// The canonical state order (spec/plan.md §4): states sorted by code
/// point, each array's cells in column-major order. DOF `n` is position `n`.
#[derive(Debug, Clone)]
pub struct Dofs {
    /// Per state: its first DOF and its extents (empty for a scalar).
    pub states: BTreeMap<String, (usize, Vec<usize>)>,
    /// The number of DOFs.
    pub n: usize,
}

impl Dofs {
    pub fn new(info: &ModelInfo) -> Dofs {
        Dofs::from_shapes(info.states.iter().map(|s| {
            let ext: Vec<usize> = info.shapes[s]
                .iter()
                .map(|x| info.index_sets[x] as usize)
                .collect();
            (s.clone(), ext)
        }))
    }

    /// The canonical order of states given as `(name, extents)`, in any
    /// order: sorted by code point, then numbered.
    pub fn from_shapes(states: impl IntoIterator<Item = (String, Vec<usize>)>) -> Dofs {
        let sorted: BTreeMap<String, Vec<usize>> = states.into_iter().collect();
        let mut out = BTreeMap::new();
        let mut n = 0;
        for (name, ext) in sorted {
            let len: usize = ext.iter().product();
            out.insert(name, (n, ext));
            n += len;
        }
        Dofs { states: out, n }
    }

    /// The DOFs of every cell of `state`.
    pub fn dofs_of(&self, state: &str) -> Option<std::ops::Range<usize>> {
        let (start, ext) = self.states.get(state)?;
        Some(*start..start + ext.iter().product::<usize>())
    }

    /// The DOF of `state` at the 1-based `cell`, `None` outside its extents.
    pub fn dof(&self, state: &str, cell: &[i64]) -> Option<usize> {
        let (start, ext) = self.states.get(state)?;
        if cell.len() != ext.len() {
            return None;
        }
        let mut off = 0usize;
        let mut stride = 1usize;
        for (c, n) in cell.iter().zip(ext) {
            if *c < 1 || *c as usize > *n {
                return None;
            }
            off += (*c as usize - 1) * stride;
            stride *= n;
        }
        Some(start + off)
    }

    /// Every DOF's element name as EarthSciAST writes it (`X`, `X[i,j]`), in
    /// canonical order.
    pub fn element_names(&self) -> Vec<String> {
        let mut out = Vec::with_capacity(self.n);
        for (name, (_, ext)) in &self.states {
            if ext.is_empty() {
                out.push(name.clone());
                continue;
            }
            for cell in cells(&ext.iter().map(|n| (1, *n as i64)).collect::<Vec<_>>()) {
                let idx: Vec<String> = cell.iter().map(i64::to_string).collect();
                out.push(format!("{name}[{}]", idx.join(",")));
            }
        }
        out
    }

    /// The state each DOF belongs to, in canonical order.
    pub fn owners(&self) -> Vec<&str> {
        let mut out = vec![""; self.n];
        for (name, (start, ext)) in &self.states {
            let len: usize = ext.iter().product();
            for o in &mut out[*start..start + len] {
                *o = name;
            }
        }
        out
    }
}

/// Every cell of the box `ranges` (1-based, inclusive), first index fastest.
pub fn cells(ranges: &[(i64, i64)]) -> Vec<Vec<i64>> {
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

/// SplitMix64 of `z` as a number in [0, 1) (spec/plan.md §5.5.1).
pub fn splitmix(z: u64) -> f64 {
    let mut z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 * (2f64).powi(-53)
}

/// The quantile of the run interval each sampled state takes (§5.5.1).
pub const SAMPLE_QUANTILES: [f64; 3] = [0.25, 0.5, 0.75];

/// A sampled state (§5.5.1).
#[derive(Debug, Clone)]
pub struct Sample {
    /// The time `t_s`.
    pub t: f64,
    /// The state `u_s`, in canonical order.
    pub u: Vec<f64>,
}

/// The sampling base `b_n`: `u_0[n]` when nonzero, else 1.
pub fn sampling_base(u0: &[f64]) -> Vec<f64> {
    u0.iter()
        .map(|x| if *x != 0.0 { *x } else { 1.0 })
        .collect()
}

/// The three sampled states of §5.5.1 over the interval `[t0, tf]`.
pub fn samples(u0: &[f64], t0: f64, tf: f64) -> Vec<Sample> {
    let base = sampling_base(u0);
    SAMPLE_QUANTILES
        .iter()
        .enumerate()
        .map(|(k, q)| {
            let s = k as u64 + 1;
            Sample {
                t: t0 + (tf - t0) * q,
                u: base
                    .iter()
                    .enumerate()
                    .map(|(n, b)| b * (0.5 + splitmix((s << 32) + n as u64)))
                    .collect(),
            }
        })
        .collect()
}

/// The initial state `u_0` in canonical order: EarthSciAST's
/// `initial_state` of the model's Problem (defaults, initialization and `ic`
/// equations, field initial conditions), matched to the DOFs by
/// `state_variable_names`.
pub fn initial_state(info: &ModelInfo, dofs: &Dofs, t0: f64) -> Result<Vec<f64>, String> {
    let prob = esm_problem(
        ProblemInput::File(&info.file),
        (t0, t0),
        ProblemOptions {
            base_path: info.path.parent().map(|p| p.to_path_buf()),
            ..Default::default()
        },
    )
    .map_err(|e| format!("EarthSciAST esm_problem: {e}"))?;
    let u0 = earthsci_ast::initial_state(&prob)
        .map_err(|e| format!("EarthSciAST initial state: {e}"))?;
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

/// An evaluation workspace and the program-layout `u`, `p` and `du`; `du`
/// is `f(u, p, t)` at the `t` beside it when `valid`.
struct Scratch {
    ws: RhsWorkspace,
    u: Vec<f64>,
    p: Vec<f64>,
    du: Vec<f64>,
    t: f64,
    valid: bool,
}

/// A document's right-hand side, compiled by EarthSciAST (`rhs_program`),
/// with the permutation between its own state layout and the canonical
/// order. It may be evaluated from several threads at once: each takes a
/// workspace from a pool (an `RhsWorkspace` is not `Sync`).
pub struct Program {
    prog: RhsProgram,
    /// The parameters at their defaults, in the program's order.
    p: Vec<f64>,
    forcing: ForcingSet,
    /// Each variable's block in the program's `u`: (offset, length).
    blocks: BTreeMap<String, (usize, usize)>,
    /// Each canonical state's block: (canonical start, program offset,
    /// length). Every state of the DOF order has one.
    map: Vec<(usize, usize, usize)>,
    pool: Mutex<Vec<Scratch>>,
}

impl Program {
    /// Compiles `input` (`native`, falling back to the interpreter when the
    /// native compiler declines a rule) for the DOF order `dofs`. Every
    /// state of `dofs` must be a block of the program's state layout with
    /// the same number of elements.
    pub fn new<'a>(
        input: impl Into<ProblemInput<'a>> + Copy,
        tspan: (f64, f64),
        dofs: &Dofs,
    ) -> Result<Program, String> {
        let build = |c: Compiler| -> Result<RhsProgram, String> {
            let prob = esm_problem(
                input,
                tspan,
                ProblemOptions {
                    compiler: Some(c),
                    ..Default::default()
                },
            )
            .map_err(|e| format!("EarthSciAST esm_problem: {e}"))?;
            rhs_program(&prob).map_err(|e| format!("EarthSciAST rhs_program: {e}"))
        };
        let prog = build(Compiler::Native).or_else(|native| {
            build(Compiler::Interpreter)
                .map_err(|interp| format!("native compiler: {native}; interpreter: {interp}"))
        })?;
        Program::from_program(prog, dofs)
    }

    /// A program already built, for the DOF order `dofs`.
    pub fn from_program(prog: RhsProgram, dofs: &Dofs) -> Result<Program, String> {
        let p = prog
            .parameter_vector(&HashMap::new())
            .map_err(|e| format!("EarthSciAST parameters: {e}"))?;
        let forcing = prog.default_forcing();
        let blocks: BTreeMap<String, (usize, usize)> = prog
            .state_layout()
            .iter()
            .map(|b| {
                (
                    b.name.clone(),
                    (b.flat_offset, b.shape.iter().product::<usize>().max(1)),
                )
            })
            .collect();
        let mut map = Vec::with_capacity(dofs.states.len());
        for (name, (start, ext)) in &dofs.states {
            let n = ext.iter().product::<usize>();
            match blocks.get(name) {
                Some((off, len)) if *len == n => map.push((*start, *off, n)),
                Some((_, len)) => {
                    return Err(format!(
                        "the program's state {name} has {len} elements, the model's has {n}"
                    ))
                }
                None => return Err(format!("the program has no state {name}")),
            }
        }
        Ok(Program {
            prog,
            p,
            forcing,
            blocks,
            map,
            pool: Mutex::new(Vec::new()),
        })
    }

    /// The compiled program.
    pub fn program(&self) -> &RhsProgram {
        &self.prog
    }

    /// The parameters at their defaults, in the program's order.
    pub fn default_parameters(&self) -> &[f64] {
        &self.p
    }

    /// The block of variable `name` in a program-layout vector: (offset,
    /// length).
    pub fn block_range(&self, name: &str) -> Option<(usize, usize)> {
        self.blocks.get(name).copied()
    }

    /// Evaluates `f(u, p, t)`, where `u(j)` is canonical DOF `j` and `p(i)`
    /// the program's parameter `i`, and hands the program-layout result to
    /// `g`. A workspace that last evaluated the same `(u, p, t)`, bit for
    /// bit, is not evaluated again.
    pub fn eval_with<R>(
        &self,
        u: impl Fn(usize) -> f64,
        p: impl Fn(usize) -> f64,
        t: f64,
        g: impl FnOnce(&[f64]) -> R,
    ) -> Result<R, String> {
        let popped = self.pool.lock().expect("workspace pool").pop();
        let mut s = popped.unwrap_or_else(|| {
            let n = self.prog.n_states();
            Scratch {
                ws: self.prog.new_workspace(),
                u: vec![0.0; n],
                p: vec![0.0; self.p.len()],
                du: vec![0.0; n],
                t: f64::NAN,
                valid: false,
            }
        });
        let mut same = s.valid && s.t.to_bits() == t.to_bits();
        for &(start, off, len) in &self.map {
            for k in 0..len {
                let x = u(start + k);
                same &= s.u[off + k].to_bits() == x.to_bits();
                s.u[off + k] = x;
            }
        }
        for (i, slot) in s.p.iter_mut().enumerate() {
            let x = p(i);
            same &= slot.to_bits() == x.to_bits();
            *slot = x;
        }
        s.t = t;
        let res = if same {
            Ok(())
        } else {
            self.prog
                .eval_into(&s.u, &[], &s.p, t, &self.forcing, &mut s.du, &mut s.ws)
                .map_err(|e| format!("EarthSciAST evaluation: {e}"))
        };
        s.valid = res.is_ok();
        let r = res.map(|()| g(&s.du));
        self.pool.lock().expect("workspace pool").push(s);
        r
    }

    /// Evaluates `f(u, p, t)` with the parameters at their defaults and
    /// hands the program-layout result to `g`.
    fn with_du<R>(&self, u: &[f64], t: f64, g: impl FnOnce(&[f64]) -> R) -> Result<R, String> {
        self.eval_with(|j| u[j], |i| self.p[i], t, g)
    }

    /// `f(u, p, t)` in canonical order.
    pub fn eval(&self, dofs: &Dofs, u: &[f64], t: f64) -> Result<Vec<f64>, String> {
        self.with_du(u, t, |du| {
            let mut out = vec![0.0; dofs.n];
            for &(start, off, len) in &self.map {
                out[start..start + len].copy_from_slice(&du[off..off + len]);
            }
            out
        })
    }

    /// The block of variable `name` in a program-layout vector.
    pub fn block<'v>(&self, v: &'v [f64], name: &str) -> Option<&'v [f64]> {
        self.blocks.get(name).map(|(off, len)| &v[*off..off + len])
    }
}

/// A band Jacobian: `(row DOF, column DOF)` to the summed band values
/// (esm-jacobian-spec §4 additivity). Its keys are the band pattern.
pub type Sparse = BTreeMap<(usize, usize), f64>;

/// An integer column index evaluated at the index bindings `bind`.
fn eval_index(c: &Expr, bind: &HashMap<String, f64>) -> Result<i64, String> {
    earthsci_ast::evaluate(c, bind)
        .map(|v| v.round() as i64)
        .map_err(|e| format!("column index: {e:?}"))
}

/// A visitor of band points: `(row cell, contracted cell, column cell)`.
pub type PointVisitor<'a> = dyn FnMut(&[i64], &[i64], &[i64]) + 'a;

/// Every point of band `b`: `f(row cell, contracted cell, column cell)`,
/// row cells first index fastest, then contracted cells.
pub fn band_points(en: &JacEntry, f: &mut PointVisitor) -> Result<(), String> {
    let b = &en.band;
    let cranges: Vec<(i64, i64)> = b.contracted.iter().map(|(_, lo, hi)| (*lo, *hi)).collect();
    let rows = if b.rows.is_empty() {
        vec![vec![]]
    } else {
        cells(&b.rows)
    };
    let ccells = if cranges.is_empty() {
        vec![vec![]]
    } else {
        cells(&cranges)
    };
    for cell in &rows {
        for cc in &ccells {
            let mut bind: HashMap<String, f64> = HashMap::new();
            for (k, r) in b.ridx.iter().enumerate() {
                if let RIdx::Name(n) = r {
                    bind.insert(n.clone(), cell[k] as f64);
                }
            }
            for (k, (n, _, _)) in b.contracted.iter().enumerate() {
                bind.insert(n.clone(), cc[k] as f64);
            }
            let col: Vec<i64> = b
                .cidx
                .iter()
                .map(|c| eval_index(c, &bind))
                .collect::<Result<_, _>>()?;
            f(cell, cc, &col);
        }
    }
    Ok(())
}

/// The band pattern: every `(row DOF, column DOF)` a band reaches.
pub fn pattern(dofs: &Dofs, entries: &[JacEntry]) -> Result<Sparse, String> {
    let mut out = Sparse::new();
    for en in entries {
        band_points(en, &mut |cell, _, col| {
            if let (Some(r), Some(c)) = (dofs.dof(&en.u, cell), dofs.dof(&en.v, col)) {
                out.insert((r, c), 0.0);
            }
        })?;
    }
    Ok(out)
}

/// Where band `en`'s values land: for each point of the band on a DOF,
/// `(offset in the band field, row DOF, column DOF)`. The field holds the
/// row's extents then the contracted extents, column-major, in `len`
/// elements; a ghost read (a column outside its variable's extent) is left
/// out.
pub fn band_scatter(
    dofs: &Dofs,
    en: &JacEntry,
    len: usize,
    field: &str,
) -> Result<Vec<(usize, usize, usize)>, String> {
    let mut ext: Vec<usize> = dofs
        .states
        .get(&en.u)
        .map(|(_, e)| e.clone())
        .unwrap_or_default();
    ext.extend(en.band.contracted.iter().map(|(_, _, hi)| *hi as usize));
    let mut out = Vec::new();
    let mut err = None;
    band_points(en, &mut |cell, cc, col| {
        let (Some(r), Some(c)) = (dofs.dof(&en.u, cell), dofs.dof(&en.v, col)) else {
            return;
        };
        let mut off = 0usize;
        let mut stride = 1usize;
        for (x, n) in cell.iter().chain(cc).zip(&ext) {
            off += (*x as usize - 1) * stride;
            stride *= n;
        }
        if off < len {
            out.push((off, r, c));
        } else {
            err = Some(format!("{field} has no element {off}"));
        }
    })?;
    match err {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

/// The name of band `k`'s field (1-based) in a Jacobian document
/// (spec/plan.md §6.2).
pub fn band_field(k: usize) -> String {
    format!("{}.J_{k}", crate::derived::JACOBIAN_MODEL)
}

/// The band Jacobian at `(u, t)` from the Jacobian document's program: band
/// `k`'s field `JacobianEval.J_k` scattered by [`band_scatter`] and summed.
pub fn band_jacobian(
    jac: &Program,
    dofs: &Dofs,
    entries: &[JacEntry],
    u: &[f64],
    t: f64,
) -> Result<Sparse, String> {
    let mut points = Vec::new();
    for (k, en) in entries.iter().enumerate() {
        let name = band_field(k + 1);
        let (base, len) = jac
            .block_range(&name)
            .ok_or_else(|| format!("the Jacobian document's program has no {name}"))?;
        points.extend(
            band_scatter(dofs, en, len, &name)?
                .into_iter()
                .map(|(off, r, c)| (base + off, r, c)),
        );
    }
    jac.with_du(u, t, |du| {
        let mut out = Sparse::new();
        for (k, r, c) in points {
            *out.entry((r, c)).or_insert(0.0) += du[k];
        }
        out
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The check values of spec/plan.md §5.5.1.
    #[test]
    fn splitmix_check_values() {
        let r = |s: u64, n: u64| splitmix((s << 32) + n);
        assert_eq!(r(1, 0), 0.766301757339086);
        assert_eq!(r(1, 1), 0.12603102042098713);
        assert_eq!(r(2, 0), 0.905065227878185);
        assert_eq!(r(3, 2), 0.03750424614895098);
    }

    #[test]
    fn sampled_states() {
        let s = samples(&[2.0, 0.0], 10.0, 30.0);
        assert_eq!(
            s.iter().map(|x| x.t).collect::<Vec<_>>(),
            [15.0, 20.0, 25.0]
        );
        // b_n = u_0[n], or 1 where u_0[n] is 0.
        assert_eq!(s[0].u[0], 2.0 * (0.5 + 0.766301757339086));
        assert_eq!(s[0].u[1], 0.5 + 0.12603102042098713);
    }

    #[test]
    fn column_major_cells() {
        assert_eq!(
            cells(&[(1, 2), (3, 4)]),
            [vec![1, 3], vec![2, 3], vec![1, 4], vec![2, 4]]
        );
        assert!(cells(&[(2, 1)]).is_empty());
        let dofs = Dofs {
            states: [
                ("A".to_string(), (0, vec![])),
                ("B".to_string(), (1, vec![2, 3])),
            ]
            .into(),
            n: 7,
        };
        assert_eq!(dofs.dof("A", &[]), Some(0));
        assert_eq!(dofs.dof("B", &[2, 1]), Some(2));
        assert_eq!(dofs.dof("B", &[1, 2]), Some(3));
        assert_eq!(dofs.dof("B", &[3, 1]), None);
        assert_eq!(
            dofs.element_names(),
            ["A", "B[1,1]", "B[2,1]", "B[1,2]", "B[2,2]", "B[1,3]", "B[2,3]"]
        );
        assert_eq!(dofs.owners(), ["A", "B", "B", "B", "B", "B", "B"]);
    }
}
