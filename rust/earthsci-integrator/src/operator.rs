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
//! Every operator is defined over the full model state vector (its length
//! is the problem's `N`); [`Operator::writes`] says which DOFs it writes. A
//! leaf's state vector is the union of its operators' writes (§1, §8.2;
//! architecture §3.2 pass 3): the leaf advances only those DOFs, and every
//! other state is a constant of the leaf ([`crate::leaf::Bound`]).

use crate::Scalar;

/// The layout and Jacobian structure (§2) and the matrix storage `jac`
/// accumulates into (§4) live in the solvers crate, beside the linear
/// solvers that read them.
pub use earthsci_integrator_solvers::linalg::{MatrixStorage, Structure};

/// What an operator's [`jac_over`](Operator::jac_over) keeps between calls
/// for one leaf. The leaf owns it, one per operator, and passes the same
/// one to every call, so an operator, which holds no state of its own,
/// computes what depends only on the leaf's DOFs once (§4, §8.3). It
/// starts empty.
#[derive(Debug, Default)]
pub struct JacOverScratch<T> {
    /// The default `jac_over`'s storage of the operator's structure over
    /// the model state, cleared and refilled at each call.
    pub model: Option<MatrixStorage<T>>,
    /// An operator's entries at the leaf's rows and columns, as
    /// `(entry, leaf row, leaf column)`, for one that knows them (a plan's).
    pub points: Option<Vec<(usize, usize, usize)>>,
}

/// Adds the entries of `m`, storage over the model state, at the rows and
/// columns `dofs` (ascending) into `out`, storage over the state vector the
/// model state at `dofs` makes: leaf index `i` is model DOF `dofs[i]`. Only
/// the entries `out` can hold are read, so a `block_diagonal` `out` takes
/// each of its cells' blocks.
pub fn add_restricted<T: crate::Scalar>(
    out: &mut MatrixStorage<T>,
    m: &MatrixStorage<T>,
    d: &[usize],
) {
    match out {
        MatrixStorage::Dense { n, .. } => {
            let n = *n;
            for b in 0..n {
                for a in 0..n {
                    let v = m.get(d[a], d[b]);
                    if v != T::zero() {
                        out.add(a, b, v);
                    }
                }
            }
        }
        MatrixStorage::BlockDiagonal { ncells, n, .. } => {
            let (nc, n) = (*ncells, *n);
            for i in 0..n {
                for j in 0..n {
                    for c in 0..nc {
                        let v = m.get(d[i * nc + c], d[j * nc + c]);
                        if v != T::zero() {
                            out.add(i * nc + c, j * nc + c, v);
                        }
                    }
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
    /// The DOFs (full-state indices, ascending) `f_k` may depend on, when the
    /// operator knows them; `None` when it does not. A leaf that advances
    /// part of the state needs them for its adjoint: the states it does not
    /// advance are its constants (§1), and their cotangents pass through it
    /// unchanged only when no operator reads them.
    fn reads(&self) -> Option<Vec<usize>> {
        None
    }
    /// The operator's structure over a leaf's state vector, the model state
    /// at `dofs` (ascending) of the `n_model` DOFs (§2, §8.3); the leaf's
    /// structure is the coarsest its operators share. The default restricts
    /// [`structure`](Operator::structure), which is over the model state: a
    /// leaf that writes every DOF has it, one that writes whole species of a
    /// `block_diagonal` structure is `block_diagonal` over its own species,
    /// and any other is `dense`. An operator a plan supplies decides from
    /// the leaf's own state variables instead (§8.3). An error refuses the
    /// leaf.
    fn structure_over(&self, dofs: &[usize], n_model: usize) -> Result<Structure, String> {
        Ok(crate::schedule::leaf_structure(
            self.structure(),
            dofs,
            n_model,
        ))
    }
    /// Whether the operator implements `call`.
    fn provides(&self, call: Call) -> bool {
        matches!(call, Call::Rhs | Call::JvpU | Call::VjpU)
    }
    /// The first fault the operator met evaluating a call, if any: an
    /// operator built from external code (a plan's, §8) returns NaN where
    /// that code fails, and this says why. A leaf that fails reports it.
    fn fault(&self) -> Option<String> {
        None
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
    /// `J_k(u)` into the leaf's matrix storage (§4): the operators' common
    /// structure, or `dense` for a `dense_lu` leaf whatever the structure
    /// (an operator that adds through [`MatrixStorage::add`] fills either).
    fn jac(&self, out: &mut MatrixStorage<T>, u: &[T], p: &[f64], t: f64) {
        unimplemented!("{}: jac", self.name())
    }
    /// `J_k(u)` at a leaf's rows and columns, added into the leaf's storage
    /// over its state vector (§4, §8.3): leaf index `i` is model DOF
    /// `dofs[i]` (ascending), and `u` is the model state. Entries in columns
    /// outside `dofs` (the leaf's constants) are left out. `out` has the
    /// structure the leaf's operators share over `dofs`
    /// ([`structure_over`](Operator::structure_over)), or is dense.
    ///
    /// `scratch` is the leaf's for this operator, kept across calls
    /// ([`JacOverScratch`]): `dofs` is the same at every call with it.
    ///
    /// The default fills storage of [`structure`](Operator::structure) over
    /// the whole model state with [`jac`](Operator::jac) and takes the
    /// leaf's entries from it, so it fills a model-sized matrix, though it
    /// allocates it only once per leaf (in `scratch.model`); an operator
    /// that knows its entries (a plan's, §8) adds them directly.
    fn jac_over(
        &self,
        out: &mut MatrixStorage<T>,
        dofs: &[usize],
        u: &[T],
        p: &[f64],
        t: f64,
        scratch: &mut JacOverScratch<T>,
    ) {
        let m = scratch
            .model
            .get_or_insert_with(|| MatrixStorage::zeros(self.structure(), u.len()));
        m.clear();
        self.jac(m, u, p, t);
        add_restricted(out, m, dofs);
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
