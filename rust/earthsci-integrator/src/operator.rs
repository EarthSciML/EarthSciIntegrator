//! The operator interface (spec/operator-interface.md).
//!
//! Binding of the spec's names (§5): the spec's call `rhs(out, u, p, t)` is
//! [`Operator::rhs`], and likewise `jvp_u`, `vjp_u`, `vjp_p`, `jac`, `ft`,
//! `hess_u`, `hess_p`, `ft_vjp_u`, `ft_vjp_p`; a map's `apply`, `map_vjp`
//! and `map_jvp` are [`MapOperator::apply`], [`MapOperator::map_vjp`] and
//! [`MapOperator::map_jvp`]. `u`-shaped arguments and outputs have the state
//! element type `T`; `p`-shaped ones are `f64`; `t` and `Δ` are `f64`.
//!
//! Every call except `apply` **accumulates** into its output (§3). An
//! operator holds no state between calls, so it is `Send + Sync` and one
//! object may serve several threads at once (U4).
//!
//! At M1 every operator is defined over the full model state vector (its
//! length is the problem's `N`); [`Operator::writes`] says which DOFs it
//! writes. A leaf's DOFs are the union of its operators' (architecture §3.2
//! pass 3); its error norm runs over those DOFs only.

use crate::Scalar;

/// The layout and Jacobian structure an operator declares (§2).
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

/// Storage `jac` accumulates into (§4).
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
}

/// A call of §3, for refusals that name the missing call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Call {
    Rhs,
    JvpU,
    VjpU,
    VjpP,
    Jac,
    Ft,
    HessU,
    HessP,
    FtVjpU,
    FtVjpP,
}

impl Call {
    /// The spec's name of the call.
    pub fn name(self) -> &'static str {
        match self {
            Call::Rhs => "rhs",
            Call::JvpU => "jvp_u",
            Call::VjpU => "vjp_u",
            Call::VjpP => "vjp_p",
            Call::Jac => "jac",
            Call::Ft => "ft",
            Call::HessU => "hess_u",
            Call::HessP => "hess_p",
            Call::FtVjpU => "ft_vjp_u",
            Call::FtVjpP => "ft_vjp_p",
        }
    }
}

/// A pair `(μ_i, w_i)` of the second-order calls.
pub type MuPair<'a, T> = (&'a [T], &'a [T]);

/// An additive term operator `f_k(u, p, t)` (§1, §3).
///
/// `rhs`, `jvp_u` and `vjp_u` are required. The others have default bodies
/// that panic; an operator that implements one says so in
/// [`provides`](Operator::provides), and a leaf that needs a call its
/// operators do not provide is refused when it is built.
#[allow(unused_variables)]
pub trait Operator<T: Scalar>: Send + Sync {
    /// The operator's name in the schedule.
    fn name(&self) -> &str;
    /// Its structure (§2).
    fn structure(&self) -> Structure;
    /// Whether `f_k` depends on `t` explicitly. When false, `ft` and
    /// `ft_vjp_*` are zero.
    fn depends_on_t(&self) -> bool;
    /// Whether `J_k` is independent of `u`. When true, `hess_u` is zero.
    fn linear_in_u(&self) -> bool;
    /// The DOFs (full-state indices, ascending) the operator writes.
    fn writes(&self) -> Vec<usize>;
    /// Whether the operator implements `call`.
    fn provides(&self, call: Call) -> bool {
        matches!(call, Call::Rhs | Call::JvpU | Call::VjpU)
    }

    /// `out += f_k(u, p, t)`.
    fn rhs(&self, out: &mut [T], u: &[T], p: &[f64], t: f64);
    /// `out += J_k(u) v`.
    fn jvp_u(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, v: &[T]);
    /// `out += J_k(u)ᵀ w`.
    fn vjp_u(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, w: &[T]);
    /// `out += f_{k,p}(u)ᵀ w`.
    fn vjp_p(&self, out: &mut [f64], u: &[T], p: &[f64], t: f64, w: &[T]) {
        unimplemented!("{}: vjp_p", self.name())
    }
    /// `J_k(u)` into the leaf's matrix storage (§4).
    fn jac(&self, out: &mut MatrixStorage<T>, u: &[T], p: &[f64], t: f64) {
        unimplemented!("{}: jac", self.name())
    }
    /// `out += ∂f_k/∂t`.
    fn ft(&self, out: &mut [T], u: &[T], p: &[f64], t: f64) {
        unimplemented!("{}: ft", self.name())
    }
    /// `out += ∇_u Σ_i μ_iᵀ J_k(u) w_i`.
    fn hess_u(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, mu: &[MuPair<'_, T>]) {
        unimplemented!("{}: hess_u", self.name())
    }
    /// `out += ∇_p Σ_i μ_iᵀ J_k(u) w_i`.
    fn hess_p(&self, out: &mut [f64], u: &[T], p: &[f64], t: f64, mu: &[MuPair<'_, T>]) {
        unimplemented!("{}: hess_p", self.name())
    }
    /// `out += ∇_u (νᵀ ∂f_k/∂t)`.
    fn ft_vjp_u(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, nu: &[T]) {
        unimplemented!("{}: ft_vjp_u", self.name())
    }
    /// `out += ∇_p (νᵀ ∂f_k/∂t)`.
    fn ft_vjp_p(&self, out: &mut [f64], u: &[T], p: &[f64], t: f64, nu: &[T]) {
        unimplemented!("{}: ft_vjp_p", self.name())
    }
}

/// A map operator `u ← M(u, p, t, Δ)` (§7).
///
/// The mask records the discrete decisions `apply` takes, as one byte per
/// decision; the adjoint holds them fixed (spec/adjoint.md §5.7).
pub trait MapOperator<T: Scalar>: Send + Sync {
    /// The operator's name in the schedule.
    fn name(&self) -> &str;
    /// The DOFs (full-state indices, ascending) the map writes. The caller
    /// fills `out` with `u` before `apply`, so every other DOF passes
    /// through unchanged, and `map_vjp` covers only the written rows.
    fn writes(&self) -> Vec<usize>;
    /// Writes `M(u, p, t, Δ)` into `out` (overwrites) and the decisions into `mask`.
    fn apply(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, dt: f64, mask: &mut Vec<u8>);
    /// Like `apply`, with the decisions held at `mask` (the frozen-tape map).
    fn apply_frozen(&self, out: &mut [T], u: &[T], p: &[f64], t: f64, dt: f64, mask: &[u8]);
    /// `out_u += (∂M/∂u)ᵀ w`, `out_p += (∂M/∂p)ᵀ w` at the pre-map state `u`.
    #[allow(clippy::too_many_arguments)]
    fn map_vjp(
        &self,
        out_u: &mut [T],
        out_p: &mut [f64],
        u: &[T],
        p: &[f64],
        t: f64,
        dt: f64,
        mask: &[u8],
        w: &[T],
    );
    /// `out += (∂M/∂u) v_u + (∂M/∂p) v_p`.
    #[allow(clippy::too_many_arguments)]
    fn map_jvp(
        &self,
        out: &mut [T],
        u: &[T],
        p: &[f64],
        t: f64,
        dt: f64,
        mask: &[u8],
        v_u: &[T],
        v_p: &[f64],
    );
}
