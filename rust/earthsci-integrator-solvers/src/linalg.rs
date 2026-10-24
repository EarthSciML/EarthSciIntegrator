//! Matrix storage and the direct linear solvers of the `linsolve` option
//! (spec/run-config.md §4.2, spec/operator-interface.md §4, architecture §6.1):
//! `dense_lu`, an LU of a column-major `N × N` matrix, and `block_lu`, a
//! batched LU of `ncells` dense `n × n` blocks stored lane-innermost.
//!
//! **Arithmetic.** Both follow Julia's `LinearAlgebra.generic_lufact!` (the
//! LU that LinearSolve's `GenericLUFactorization` and EarthSciMLBase's
//! `BlockDiagonal` use, the oracle architecture §6.1 names), so the two
//! implementations can agree bitwise:
//! - right-looking elimination, column `k = 0, 1, …`;
//! - with pivoting, the pivot row is the first `i ≥ k` of largest `|a_ik|`
//!   (strict `>`, so ties keep the earliest row), and whole rows are swapped;
//! - the multipliers are `a_ik · (1/a_kk)`, the reciprocal formed once;
//! - the trailing update is `a_ij ← a_ij − a_ik·a_kj`, no fused multiply-add;
//! - a zero pivot (after pivoting) is a failed factorization (`info ≠ 0`,
//!   which OrdinaryDiffEq's `issuccess_W` treats as a singular `W`,
//!   spec/methods.md §4.2).
//!
//! The triangular solves are column-oriented, as Julia's generic `naivesub!`
//! (and BLAS `trsv`) write them, so each entry is updated by sequential
//! subtraction: `L y = Pb` over ascending columns, then `U x = y` over
//! descending columns with `x_j = y_j / u_jj`. The transposed solve
//! `Aᵀx = b` is `Uᵀz = b` (ascending columns, divide first), then `Lᵀy = z`
//! (descending columns), then the row interchanges undone in reverse order.
//!
//! `block_lu` performs, for every lane, exactly the arithmetic `dense_lu`
//! performs on that lane's block; its loops run one operation across all
//! lanes at a time (the lane index innermost).

// Index loops mirror the elimination and stage formulas.
#![allow(clippy::needless_range_loop)]

use crate::scalar::Scalar;

/// The layout and Jacobian structure an operator declares
/// (spec/operator-interface.md §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Structure {
    /// No structure; DOF order is the operator set's own.
    Dense,
    /// `ncells` cells of `n` DOFs each, species-major: DOF `i` of cell `c`
    /// is at index `i·ncells + c`. The Jacobian is zero between cells.
    BlockDiagonal { ncells: usize, n: usize },
}

impl Structure {
    /// The coarsest structure all of `structures` share (§2): equal
    /// `block_diagonal` structures stay block diagonal; any other mix is dense.
    pub fn common(structures: impl IntoIterator<Item = Structure>) -> Structure {
        let mut it = structures.into_iter();
        let Some(first) = it.next() else {
            return Structure::Dense;
        };
        if it.all(|s| s == first) {
            first
        } else {
            Structure::Dense
        }
    }
}

/// Storage `jac` accumulates into (spec/operator-interface.md §4).
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixStorage<T> {
    /// An `N × N` column-major matrix: entry `(i, j)` at `j·N + i`.
    Dense { n: usize, data: Vec<T> },
    /// `ncells` dense `n × n` blocks, lane-innermost: entry `(i, j)` of cell
    /// `c` at `(i·n + j)·ncells + c`.
    BlockDiagonal {
        ncells: usize,
        n: usize,
        data: Vec<T>,
    },
}

impl<T: Scalar> MatrixStorage<T> {
    /// Zeroed storage for a structure over a state of length `n_state`.
    pub fn zeros(structure: Structure, n_state: usize) -> Self {
        match structure {
            Structure::Dense => MatrixStorage::Dense {
                n: n_state,
                data: vec![T::zero(); n_state * n_state],
            },
            Structure::BlockDiagonal { ncells, n } => MatrixStorage::BlockDiagonal {
                ncells,
                n,
                data: vec![T::zero(); ncells * n * n],
            },
        }
    }

    /// Sets every entry to zero.
    pub fn clear(&mut self) {
        self.data_mut().iter_mut().for_each(|x| *x = T::zero());
    }

    /// The stored entries.
    pub fn data(&self) -> &[T] {
        match self {
            MatrixStorage::Dense { data, .. } | MatrixStorage::BlockDiagonal { data, .. } => data,
        }
    }

    /// The stored entries, mutably.
    pub fn data_mut(&mut self) -> &mut [T] {
        match self {
            MatrixStorage::Dense { data, .. } | MatrixStorage::BlockDiagonal { data, .. } => data,
        }
    }

    /// Adds `v` to the entry at state indices `(row, col)`. For block-diagonal
    /// storage both indices must be in the same cell.
    pub fn add(&mut self, row: usize, col: usize, v: T) {
        match self {
            MatrixStorage::Dense { n, data } => data[col * *n + row] += v,
            MatrixStorage::BlockDiagonal { ncells, n, data } => {
                let (i, c) = (row / *ncells, row % *ncells);
                let (j, c2) = (col / *ncells, col % *ncells);
                assert_eq!(c, c2, "block-diagonal entry ({row}, {col}) crosses cells");
                data[(i * *n + j) * *ncells + c] += v;
            }
        }
    }

    /// The entry at state indices `(row, col)` (zero off the blocks).
    pub fn get(&self, row: usize, col: usize) -> T {
        match self {
            MatrixStorage::Dense { n, data } => data[col * *n + row],
            MatrixStorage::BlockDiagonal { ncells, n, data } => {
                let (i, c) = (row / *ncells, row % *ncells);
                let (j, c2) = (col / *ncells, col % *ncells);
                if c != c2 {
                    T::zero()
                } else {
                    data[(i * *n + j) * *ncells + c]
                }
            }
        }
    }

    /// Writes `σ I − J` into `self`, with `J` in `jac` (same shape): each
    /// diagonal entry is `σ − J_ii`, rounded once, and each other entry
    /// `−J_ij`. This is the `Ŵ = I/(hγ) − J` of spec/methods.md §4.1 with
    /// `σ = 1/(hγ)`, the exact negation of OrdinaryDiffEq's stored
    /// `W = J − I/(hγ)` (OrdinaryDiffEqDifferentiation `calc_W`).
    pub fn shifted_negation(&mut self, jac: &MatrixStorage<T>, sigma: T) {
        match (self, jac) {
            (MatrixStorage::Dense { n, data }, MatrixStorage::Dense { n: n2, data: j }) => {
                assert_eq!(n, n2);
                for col in 0..*n {
                    for row in 0..*n {
                        let k = col * *n + row;
                        data[k] = if row == col { sigma - j[k] } else { -j[k] };
                    }
                }
            }
            (
                MatrixStorage::BlockDiagonal { ncells, n, data },
                MatrixStorage::BlockDiagonal {
                    ncells: c2,
                    n: n2,
                    data: j,
                },
            ) => {
                assert_eq!((*ncells, *n), (*c2, *n2));
                let nc = *ncells;
                for i in 0..*n {
                    for jj in 0..*n {
                        let base = (i * *n + jj) * nc;
                        for c in 0..nc {
                            data[base + c] = if i == jj {
                                sigma - j[base + c]
                            } else {
                                -j[base + c]
                            };
                        }
                    }
                }
            }
            _ => panic!("shifted_negation: storage shapes differ"),
        }
    }
}

/// A dense LU factorization with partial pivoting (`dense_lu`).
#[derive(Debug, Clone)]
pub struct DenseLu<T> {
    n: usize,
    /// `L` (unit, below the diagonal) and `U`, column-major.
    lu: Vec<T>,
    /// Row interchanges: row `k` was swapped with row `piv[k] ≥ k`.
    piv: Vec<usize>,
    /// Whether the last factorization succeeded.
    ok: bool,
}

impl<T: Scalar> DenseLu<T> {
    /// An empty factorization for `n × n` matrices.
    pub fn new(n: usize) -> Self {
        DenseLu {
            n,
            lu: vec![T::zero(); n * n],
            piv: (0..n).collect(),
            ok: false,
        }
    }

    /// Factors the column-major `n × n` matrix `a`. Returns `false` (and the
    /// factorization is unusable) when a pivot is zero.
    pub fn factor(&mut self, a: &[T]) -> bool {
        let n = self.n;
        assert_eq!(a.len(), n * n);
        self.lu.copy_from_slice(a);
        let m = &mut self.lu;
        self.ok = false;
        for k in 0..n {
            // Pivot: the first row i ≥ k of largest |a_ik|.
            let mut kp = k;
            let mut amax = m[k * n + k].to_f64().abs();
            for i in k + 1..n {
                let ai = m[k * n + i].to_f64().abs();
                if ai > amax {
                    kp = i;
                    amax = ai;
                }
            }
            self.piv[k] = kp;
            if m[k * n + kp] == T::zero() {
                return false;
            }
            if kp != k {
                for j in 0..n {
                    m.swap(j * n + k, j * n + kp);
                }
            }
            let inv = T::from_f64(1.0) / m[k * n + k];
            for i in k + 1..n {
                m[k * n + i] *= inv;
            }
            for j in k + 1..n {
                let akj = m[j * n + k];
                for i in k + 1..n {
                    let lik = m[k * n + i];
                    m[j * n + i] -= lik * akj;
                }
            }
        }
        self.ok = true;
        true
    }

    /// Solves `A x = b` in place.
    pub fn solve(&self, b: &mut [T]) {
        debug_assert!(self.ok);
        let (n, m) = (self.n, &self.lu);
        for k in 0..n {
            b.swap(k, self.piv[k]);
        }
        for j in 0..n {
            let bj = b[j];
            for i in j + 1..n {
                b[i] -= m[j * n + i] * bj;
            }
        }
        for j in (0..n).rev() {
            let xj = b[j] / m[j * n + j];
            b[j] = xj;
            for i in 0..j {
                b[i] -= m[j * n + i] * xj;
            }
        }
    }

    /// Solves `Aᵀ x = b` in place.
    pub fn solve_t(&self, b: &mut [T]) {
        debug_assert!(self.ok);
        let (n, m) = (self.n, &self.lu);
        // Uᵀ z = b: column j of Uᵀ is row j of U.
        for j in 0..n {
            let zj = b[j] / m[j * n + j];
            b[j] = zj;
            for i in j + 1..n {
                b[i] -= m[i * n + j] * zj;
            }
        }
        // Lᵀ y = z.
        for j in (0..n).rev() {
            let yj = b[j];
            for i in 0..j {
                b[i] -= m[i * n + j] * yj;
            }
        }
        for k in (0..n).rev() {
            b.swap(k, self.piv[k]);
        }
    }
}

/// A batched LU of `ncells` dense `n × n` blocks, lane-innermost (`block_lu`).
#[derive(Debug, Clone)]
pub struct BlockLu<T> {
    ncells: usize,
    n: usize,
    pivot: bool,
    /// Factors, lane-innermost like the matrix: `(i·n + j)·ncells + c`.
    lu: Vec<T>,
    /// Row interchanges per lane: `piv[k·ncells + c]`.
    piv: Vec<usize>,
    /// Scratch per lane for one pivot column: its largest magnitude, and the
    /// pivot's reciprocal.
    amax: Vec<f64>,
    inv: Vec<T>,
    ok: bool,
}

impl<T: Scalar> BlockLu<T> {
    /// An empty factorization; `pivot` is `block_lu.pivot`.
    pub fn new(ncells: usize, n: usize, pivot: bool) -> Self {
        BlockLu {
            ncells,
            n,
            pivot,
            lu: vec![T::zero(); ncells * n * n],
            piv: (0..n)
                .flat_map(|k| std::iter::repeat_n(k, ncells))
                .collect(),
            amax: vec![0.0; ncells],
            inv: vec![T::zero(); ncells],
            ok: false,
        }
    }

    /// Factors every block of `a` (lane-innermost). Returns `false` when any
    /// lane has a zero pivot.
    pub fn factor(&mut self, a: &[T]) -> bool {
        let (nc, n) = (self.ncells, self.n);
        assert_eq!(a.len(), nc * n * n);
        self.lu.copy_from_slice(a);
        self.ok = false;
        let at = |i: usize, j: usize| (i * n + j) * nc;
        for k in 0..n {
            let pk = k * nc;
            if self.pivot {
                // Per lane: the first row i ≥ k of largest |a_ik|.
                for c in 0..nc {
                    self.piv[pk + c] = k;
                }
                let amax = &mut self.amax;
                for (c, m) in amax.iter_mut().enumerate() {
                    *m = self.lu[at(k, k) + c].to_f64().abs();
                }
                for i in k + 1..n {
                    let ai = at(i, k);
                    for c in 0..nc {
                        let v = self.lu[ai + c].to_f64().abs();
                        if v > amax[c] {
                            amax[c] = v;
                            self.piv[pk + c] = i;
                        }
                    }
                }
                for c in 0..nc {
                    let kp = self.piv[pk + c];
                    if self.lu[at(kp, k) + c] == T::zero() {
                        return false;
                    }
                }
                for j in 0..n {
                    for c in 0..nc {
                        let kp = self.piv[pk + c];
                        if kp != k {
                            self.lu.swap(at(k, j) + c, at(kp, j) + c);
                        }
                    }
                }
            } else if (0..nc).any(|c| self.lu[at(k, k) + c] == T::zero()) {
                return false;
            }
            let inv = &mut self.inv;
            for (c, v) in inv.iter_mut().enumerate() {
                *v = T::from_f64(1.0) / self.lu[at(k, k) + c];
            }
            for i in k + 1..n {
                let ai = at(i, k);
                for c in 0..nc {
                    self.lu[ai + c] *= inv[c];
                }
            }
            for i in k + 1..n {
                for j in k + 1..n {
                    let (aij, aik, akj) = (at(i, j), at(i, k), at(k, j));
                    for c in 0..nc {
                        let d = self.lu[aik + c] * self.lu[akj + c];
                        self.lu[aij + c] -= d;
                    }
                }
            }
        }
        self.ok = true;
        true
    }

    /// Solves every block's system in place; `b` is species-major (DOF `i` of
    /// cell `c` at `i·ncells + c`), which is lane-innermost.
    pub fn solve(&self, b: &mut [T]) {
        debug_assert!(self.ok);
        let (nc, n, m) = (self.ncells, self.n, &self.lu);
        let at = |i: usize, j: usize| (i * n + j) * nc;
        if self.pivot {
            for k in 0..n {
                for c in 0..nc {
                    b.swap(k * nc + c, self.piv[k * nc + c] * nc + c);
                }
            }
        }
        for j in 0..n {
            for i in j + 1..n {
                let l = at(i, j);
                for c in 0..nc {
                    let d = m[l + c] * b[j * nc + c];
                    b[i * nc + c] -= d;
                }
            }
        }
        for j in (0..n).rev() {
            let d = at(j, j);
            for c in 0..nc {
                b[j * nc + c] = b[j * nc + c] / m[d + c];
            }
            for i in 0..j {
                let u = at(i, j);
                for c in 0..nc {
                    let d = m[u + c] * b[j * nc + c];
                    b[i * nc + c] -= d;
                }
            }
        }
    }

    /// Solves every block's transposed system in place.
    pub fn solve_t(&self, b: &mut [T]) {
        debug_assert!(self.ok);
        let (nc, n, m) = (self.ncells, self.n, &self.lu);
        let at = |i: usize, j: usize| (i * n + j) * nc;
        for j in 0..n {
            let d = at(j, j);
            for c in 0..nc {
                b[j * nc + c] = b[j * nc + c] / m[d + c];
            }
            for i in j + 1..n {
                let u = at(j, i);
                for c in 0..nc {
                    let d = m[u + c] * b[j * nc + c];
                    b[i * nc + c] -= d;
                }
            }
        }
        for j in (0..n).rev() {
            for i in 0..j {
                let l = at(j, i);
                for c in 0..nc {
                    let d = m[l + c] * b[j * nc + c];
                    b[i * nc + c] -= d;
                }
            }
        }
        if self.pivot {
            for k in (0..n).rev() {
                for c in 0..nc {
                    b.swap(k * nc + c, self.piv[k * nc + c] * nc + c);
                }
            }
        }
    }
}

/// The leaf's linear solver (`linsolve`), sized for its matrix storage.
#[derive(Debug, Clone)]
pub enum LinearSolver<T> {
    Dense(DenseLu<T>),
    Block(BlockLu<T>),
}

/// Which direct solver a leaf uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinsolveKind {
    /// `dense_lu`, with partial pivoting.
    DenseLu,
    /// `block_lu`, with `pivot`.
    BlockLu { pivot: bool },
}

impl<T: Scalar> LinearSolver<T> {
    /// A solver of `kind` for matrices of `structure` over a state of length `n`.
    pub fn new(kind: LinsolveKind, structure: Structure, n: usize) -> Result<Self, String> {
        match (kind, structure) {
            (LinsolveKind::DenseLu, Structure::Dense) => Ok(LinearSolver::Dense(DenseLu::new(n))),
            (LinsolveKind::BlockLu { pivot }, Structure::BlockDiagonal { ncells, n: nb }) => {
                Ok(LinearSolver::Block(BlockLu::new(ncells, nb, pivot)))
            }
            (LinsolveKind::DenseLu, Structure::BlockDiagonal { .. }) => Err(
                "dense_lu with block_diagonal matrix storage: the leaf's storage is per block"
                    .into(),
            ),
            (LinsolveKind::BlockLu { .. }, Structure::Dense) => {
                Err("block_lu needs a block_diagonal leaf".into())
            }
        }
    }

    /// Factors `a`; `false` when it is singular.
    pub fn factor(&mut self, a: &MatrixStorage<T>) -> bool {
        match self {
            LinearSolver::Dense(lu) => lu.factor(a.data()),
            LinearSolver::Block(lu) => lu.factor(a.data()),
        }
    }

    /// `x = A⁻¹ b` in place.
    pub fn solve(&self, b: &mut [T]) {
        match self {
            LinearSolver::Dense(lu) => lu.solve(b),
            LinearSolver::Block(lu) => lu.solve(b),
        }
    }

    /// `x = A⁻ᵀ b` in place.
    pub fn solve_t(&self, b: &mut [T]) {
        match self {
            LinearSolver::Dense(lu) => lu.solve_t(b),
            LinearSolver::Block(lu) => lu.solve_t(b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    fn matvec(a: &[f64], n: usize, x: &[f64], transpose: bool) -> Vec<f64> {
        let mut y = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                let aij = a[j * n + i];
                if transpose {
                    y[j] += aij * x[i];
                } else {
                    y[i] += aij * x[j];
                }
            }
        }
        y
    }

    #[test]
    fn dense_lu_solves_and_transposes() {
        let mut r = Lcg(7);
        for n in [1usize, 2, 3, 5, 8] {
            let a: Vec<f64> = (0..n * n).map(|_| r.next()).collect();
            let x: Vec<f64> = (0..n).map(|_| r.next()).collect();
            let mut lu = DenseLu::new(n);
            assert!(lu.factor(&a));
            for tr in [false, true] {
                let mut b = matvec(&a, n, &x, tr);
                if tr {
                    lu.solve_t(&mut b);
                } else {
                    lu.solve(&mut b);
                }
                for (bi, xi) in b.iter().zip(&x) {
                    assert!(
                        (bi - xi).abs() < 1e-10,
                        "n {n} transpose {tr}: {bi} vs {xi}"
                    );
                }
            }
        }
    }

    #[test]
    fn dense_lu_pivots_and_detects_singularity() {
        // A zero leading entry needs a row interchange.
        let a = [0.0, 1.0, 2.0, 3.0]; // [[0, 2], [1, 3]] column-major
        let mut lu = DenseLu::new(2);
        assert!(lu.factor(&a));
        let mut b = [4.0, 7.0]; // x = (1, 2)
        lu.solve(&mut b);
        assert_eq!(b, [1.0, 2.0]);
        let sing = [1.0, 2.0, 2.0, 4.0];
        assert!(!lu.factor(&sing));
        assert!(!DenseLu::<f64>::new(1).factor(&[0.0]));
    }

    /// Every lane of block_lu does dense_lu's arithmetic on its block, bit for bit.
    #[test]
    fn block_lu_matches_dense_lu_per_lane() {
        let mut r = Lcg(11);
        let (nc, n) = (5usize, 4usize);
        for pivot in [true, false] {
            let mut a = vec![0.0; nc * n * n];
            for v in a.iter_mut() {
                *v = r.next();
            }
            // Diagonally dominant without pivoting, so no zero pivots.
            if !pivot {
                for i in 0..n {
                    for c in 0..nc {
                        a[(i * n + i) * nc + c] += 4.0;
                    }
                }
            }
            let mut blu = BlockLu::new(nc, n, pivot);
            assert!(blu.factor(&a));
            let b0: Vec<f64> = (0..nc * n).map(|_| r.next()).collect();
            let (mut b, mut bt) = (b0.clone(), b0.clone());
            blu.solve(&mut b);
            blu.solve_t(&mut bt);
            for c in 0..nc {
                let mut dense = vec![0.0; n * n];
                for i in 0..n {
                    for j in 0..n {
                        dense[j * n + i] = a[(i * n + j) * nc + c];
                    }
                }
                let lane_b: Vec<f64> = (0..n).map(|i| b0[i * nc + c]).collect();
                let x_true = {
                    let mut lu = DenseLu::new(n);
                    assert!(lu.factor(&dense));
                    let (mut x, mut xt) = (lane_b.clone(), lane_b.clone());
                    lu.solve(&mut x);
                    lu.solve_t(&mut xt);
                    (x, xt)
                };
                for i in 0..n {
                    if pivot {
                        assert_eq!(b[i * nc + c].to_bits(), x_true.0[i].to_bits());
                        assert_eq!(bt[i * nc + c].to_bits(), x_true.1[i].to_bits());
                    } else {
                        assert!((b[i * nc + c] - x_true.0[i]).abs() < 1e-12);
                        assert!((bt[i * nc + c] - x_true.1[i]).abs() < 1e-12);
                    }
                }
                // And the residual.
                let ax = matvec(
                    &dense,
                    n,
                    &(0..n).map(|i| b[i * nc + c]).collect::<Vec<_>>(),
                    false,
                );
                for i in 0..n {
                    assert!((ax[i] - lane_b[i]).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn block_lu_singular_lane_fails() {
        let (nc, n) = (3usize, 2usize);
        let mut a = vec![0.0; nc * n * n];
        for c in 0..nc {
            a[c] = 1.0; // (0,0)
            a[3 * nc + c] = 1.0; // (1,1)
        }
        a[3 * nc + 1] = 0.0; // lane 1 singular
        assert!(!BlockLu::<f64>::new(nc, n, true).factor(&a));
        assert!(!BlockLu::<f64>::new(nc, n, false).factor(&a));
        a[3 * nc + 1] = 2.0;
        assert!(BlockLu::<f64>::new(nc, n, false).factor(&a));
    }

    #[test]
    fn shifted_negation_forms_w() {
        let mut j = MatrixStorage::<f64>::zeros(Structure::Dense, 2);
        j.add(0, 0, 1.0);
        j.add(0, 1, 2.0);
        j.add(1, 0, 3.0);
        j.add(1, 1, 4.0);
        let mut w = j.clone();
        w.shifted_negation(&j, 10.0);
        assert_eq!(w.data(), &[9.0, -3.0, -2.0, 6.0]);
        let s = Structure::BlockDiagonal { ncells: 2, n: 2 };
        let mut jb = MatrixStorage::<f64>::zeros(s, 4);
        jb.add(0, 0, 1.0); // cell 0, (0,0)
        jb.add(3, 1, 5.0); // cell 1, (1,0)
        let mut wb = jb.clone();
        wb.shifted_negation(&jb, 2.0);
        assert_eq!(wb.get(0, 0), 1.0);
        assert_eq!(wb.get(1, 1), 2.0);
        assert_eq!(wb.get(3, 1), -5.0);
        assert_eq!(wb.get(3, 3), 2.0);
    }
}
