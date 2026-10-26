---
title: "Range-restricted, operator-restricted right-hand sides: tiles with halos, and lanes"
description: "Evaluate one operator's right-hand side — a subset of additive terms selected by provenance, writing only that operator's variables — over a box of cells (a horizontal tile with whole columns) that reads halo cells from an extended local array, or over a runtime list of cells gathered into lanes. Index space stays global, so makearray boundary regions apply exactly where the global domain has a boundary and interior stencils apply at every other tile edge. The contract is bitwise: the tiles' outputs, assembled, equal the operator's global right-hand side bit for bit."
---

Status: draft (proposed by EarthSciIntegrator, 2026-10-13); done upstream: released in EarthSciAST v0.5.0 (EarthSciAST#557)
Target: EarthSciAST API_SPEC §5.8 (new entries beside `esm_problem`; no schema or format-version change); `EarthSciAST.jl` and `earthsci-ast-rs` v0.5.0 (released); Python later.
Needed by: EarthSciIntegrator milestone M3 (tiles in shared memory, and lane pools), then M5 (MPI ranks).
Integrator refs: architecture.md §3.2 (passes 2, 3, 7, 8), §7.1, §7.2, §7.6, §7.7, §13 (U3); decisions D6, D7, D10, D11, D14, D17, D20, decisions.md §2.2

## Summary

The integrator splits `F = Σ_k F_k` into operators by provenance (U2) and runs
each operator on the partition its locality calls for (architecture §7.2):
stencil operators on geographic tiles with per-stage halo exchange, and column
and pointwise operators on lanes. Today both bindings evaluate only the whole
model over the whole domain. On top of U4's program type, this RFC adds:

1. **Operator selection.** Keep only the summands whose U2 provenance is
   selected and write only the operator's variables `W_k`. The other states it
   reads, `R_k \ W_k`, arrive as a separate read-only input.
2. **Tiles.** Restrict that program to an owned box `B` of a horizontal
   decomposition. It reads from a storage box `S ⊇ B` that includes the halo
   and writes only the `(W_k, B)` rows of `du`. Indices stay global, so a
   `makearray` boundary region applies only where its global cells fall
   inside `B`.
3. **Lanes.** Evaluate a pointwise or column operator over a run-time list of
   cells or columns. Each lane carries its global index, from which
   index-derived quantities (coordinates, level gates) are computed.

The normative test: assembling every tile's owned rows reproduces the
operator's global right-hand side **bitwise** under `native` in each binding,
and every lane reproduces its cell's value bitwise.

## Motivation

**Nothing is range-restricted or operator-restricted today.**

- Julia builds one whole-domain `f!` (`EarthSciAST/pkg/EarthSciAST.jl/src/simulate.jl:1189-1207`;
  `src/tree_walk/acc_merge.jl:635-754`).
  - When array observeds exist, every call copies all of `u` into an extended
    vector and fills every observed level over every cell (`build.jl:2757-2772`).
  - Its only cell partition is the Polyester chunking. That is ordinal, not
    spatial, and covers kernels only (`access_kernel.jl:1311-1323`).
- Rust transposes every state variable into a row-major mirror on every call
  (`earthsci-ast-rs/src/simulate_array/tape/exec/mod.rs:530-548`). Each
  instruction then runs over its slot's whole static box (`tape/ir.rs:933-945`).
- Julia never zeroes `du`: writes are assignments (`oop_merge.jl:54-57`). Rust
  zeroes all of it (`driver.rs:621-623`). Tiles sharing one `du` need a rule
  for what a restricted program may touch.
- Flatten sums `operator_compose` terms into one `+` node per target
  (`src/coupling_apply.jl:429-432`). EarthSciASTSplitter's parts are
  full-length (decisions.md §3.2, item 6).

**A tile cannot be a smaller model.**

- Out-of-range reads are decided at build time, against the extent of the array
  read. A state gather outside its variable's box reads a zero ghost:
  - Julia `compile.jl:190-198`, `build.jl:427-433`;
  - Rust `tape/ir.rs:26-28`;
  - CONFORMANCE_SPEC fixes this for all bindings (`CONFORMANCE_SPEC.md:884-886`).
- Boundary conditions live inside `F` as `makearray` regions, later regions
  overwriting earlier ones (`esm-spec.md:4565`, `:1075`).
- EarthSciDiscretizations carves one region per boundary row.
  - PPM's cell form is `[4, N-2]` plus slabs 1, 2, 3, N-1 and N: "three wrap
    columns at the low end and two at the high end"
    (`EarthSciDiscretizations/grids/latlon3d/rules/ppm_D_lon_periodic.esm:5`,
    regions from `:78`).
  - The periodic slabs wrap by explicit index arithmetic:
    `ppm_D_lon_wrap_col1.esm:49-50` reads `f[NLON-1]` and `f[NLON]` from cell 1.
- So a tile re-instantiated as its own model reads zeros at interior edges, or
  applies boundary rules there. Either way it does not match the global run
  (decisions.md §2.2).
- reseact's transport is built from exactly these rules
  (`reseact.esm/reseact.esm:75`, `:91`).



**reseact's lanes need surgery and one build per capacity.**

- Its capacity build re-instantiates chemistry on a `C × 1 × 1` grid
  (`reseact.esm/tools/capacity_chem.jl:59-66`).
- Of the 310 names chemistry needs, 37 read a raw loop index. Eight of those
  need document surgery because "the index itself carries the meaning"
  (`:12-29`).
- Builds cost 4.6 s at C = 288 and 11.4 s plus a 46.5 s Jacobian at C = 4096,
  against 650.2 s plus 567.8 s for the full CONUS reference (`:31-46`).
- A ladder of five capacities keeps padding waste near 3% (`:48-57`).
- Padding lanes must duplicate a real cell, or the error norm goes NaN
  (`:353-364`).

The compiler already knows every one of these index uses, so it can compute
them per lane.

## Decisions of record

Proposed by the integrator, for the maintainers to rule on.

| # | Question | Proposal | Why |
|---|---|---|---|
| 1 | Index space of a tile | **Global.** Boxes, region bounds, coordinates and index-derived values all keep their global 1-based indices; only storage is local. | Boundary regions then apply exactly where the global domain has a boundary, and index-derived values need no rebasing. |
| 2 | Contract | **Bitwise** against the same operator's global evaluation, same binding and compiler | D10. The kernels are per-element pure in both bindings (Julia: "NO `@simd`, NO `@fastmath`, NO `muladd`", `codegen_kernel.jl:19-20`). |
| 3 | Out-of-range reads | **Classified against the global extent.** Out of domain keeps today's semantics. In domain but outside storage is a build error, never a ghost zero. | A silent zero at an interior tile edge is the failure mode this RFC exists to prevent. |
| 4 | Intermediate observeds | **Recomputed on their need box**, not exchanged | One halo exchange per stage (architecture §7.2); the observed chains are short. |
| 5 | Halo filling | **The caller's job.** The compiler reports the halo each tile needs. | The integrator owns exchange and MPI (D6). |
| 6 | Boxes | **Fixed at build** for tiles. Cell lists are **run time** for lanes. | Tile geometry changes only at checkpoints (architecture §7.6); shard composition changes every macro step. |
| 7 | Faces | **Cell `i` owns face `i`**, its low face. The tile that owns cell `N` also owns face `N+1`. | EarthSciDiscretizations: "FACE i IS THE LOWER (WESTERN) EDGE OF CELL i" (`ppmflux_D_lon_interior.esm:5`). This also fixes the meaning of architecture §7.1's "owned by the low-side cell". |
| 8 | What tiles refuse | Writes to 0-d states, and contractions, scans or data-dependent gathers along a decomposed axis | These are `global` locality (architecture §3.2, pass 2). |

## Design

### 1. Terms

- **Decomposition.** The caller names the horizontal index sets it cuts, each
  one's period if periodic, and the face index sets that stagger off it (for
  example `lon_nodes` off `lon`). Every other index set (`lev`, species,
  records) is whole on every tile (D20).
- **Owned box `B`.** One inclusive range per decomposed index set.
- **Storage box `S`.** `B` dilated by the halo. Along a non-periodic axis it may
  extend past `1..N`. Along a periodic axis it is expressed in unwrapped
  coordinates, for example `-2..16`.
- **Face sets.** Their ranges follow from rule 7: owned faces are the owned
  cells, plus `N+1` at the high edge; storage faces are `S`'s cells plus one.
  A face quantity stored on the cell index set is just a cell variable. An
  example is eqice's `Qx`, "flux through the EAST face of each cell"
  (`eqice/stage2/geometry.esm:7317-7324`).
- **Rank storage.** Several tiles of one rank may share one storage box: the
  rank's extended array. With `S` equal to the whole domain, storage is the
  global layout. That is the shared-memory case of architecture §2.2, where the
  halo exchange is a barrier.

### 2. Operator selection

`select(prog; writes, terms)` returns a program for one operator:

- **`terms`** is a set of U2 provenance tags. A summand of a `D(x)` right-hand
  side is kept iff its tag is in the set. The kept summands are added in their
  original relative order.
- **`writes = W_k`.** Rules for variables outside `W_k` are dropped.
- **Inputs.** `u` covers `W_k`. `aux` covers `R_k \ W_k`, the states the kept
  terms read; it is read-only and frozen for the leaf. Both use the flat-layout
  order (scalars, then arrays by sorted name; `src/tree_walk/state_layout.jl:1-16`).
- **Derived documents.** When the planner emits a per-operator document
  (architecture §3.2, pass 6), selection is `terms = all` and `writes = W_k`.
  Both routes give the same program.
- **Bitwise reference.** Every bitwise claim below is against `select(prog; …)`
  over the whole domain. `Σ_k F_k = F` holds only to tolerance, because dropping
  summands changes the association; the planner checks it (§3.2, pass 8).

### 3. Halo report

`halo(op, dec)` returns, for every variable `v ∈ R_k` and every decomposed or
face index set `s` on `v`'s shape, the pair `(lo, hi)`. This is how far the
owned output reaches into `v` along `s`.

- **Composition.** Offsets compose through observeds. If `du[i]` reads `F[i]`
  and `F[i+1]`, and `F[j]` reads `u[j-3..j+2]`, then `u` needs `(3, 3)`.
- **Region reach.** Each `makearray` region contributes its own reach. The
  report is the maximum over regions, so a reach that occurs only in boundary
  slabs is included.
- **Forcing and array parameters.** `forcing_reach(op, dec)` maps the owned box
  affinely to each buffer's index box, including buffers on another grid such as
  GEOS-FP native arrays read at `LON0+gi` (`capacity_chem.jl:20-25`). A
  pre-sliced buffer "is a rebasing problem and nothing more"
  (`reseact.esm/INDEX_RANGE_READS.md:66-72`).
- **Wraps and refusals.** Periodic wraps count as reach across the seam,
  whether the evaluator's periodic gather does them or the rule spells them out
  (`f[NLON-1]` from cell 1). A data-dependent subscript along a decomposed axis
  (Rust `IndexGather`, `tape/ir.rs:303-314`) is refused, naming the rule.
- **Cross-check.** The planner computes the same report from the AST (pass 2).
  The plan golden compares the two.

### 4. Tile semantics (normative)

Let `G_k` be the operator program over the whole domain. Suppose `u_S` and
`aux_S` agree with the global state on every in-domain cell of `S`, and on a
periodic axis every cell outside `1..N` holds the value at its wrapped index.
Then for every `v ∈ W_k` and every owned cell `c` of `v`:

    tile(u_S, aux_S, p, t, forcing)[v, c]  ==  G_k(u, aux, p, t, forcing)[v, c]   (bitwise)

The compiler reaches this by the following rules.

1. **Region clipping.** Each `makearray` region's global box is intersected
   with the need box of its output. Empty intersections are dropped. Order, and
   "later overwrites earlier", are kept. A PPM tile owning `lon ∈ 17..32` keeps
   only the interior `[4, N-2]`; the tile owning `1..16` keeps slabs 1, 2 and 3
   and the interior `4..16`.
2. **Reads.** Each read of `v` at global index `g` along axis `a` resolves in
   this order:
   - `g` in storage → read storage.
   - `a` periodic with period `N`, and `g ± N` in storage → read that cell.
     This covers both explicit wraps and the periodic gather policy.
   - `g` outside `1..N`, `a` not periodic → today's out-of-range rule: zero
     ghost for states and observeds, the declared policy for const arrays
     (`build_helpers.jl:195-216`).
   - Otherwise → build error `range_halo_insufficient`, naming the rule, the
     variable, the axis and the missing cells.
3. **Observeds.** Each array observed is evaluated on its need box. That box is
   the union of its readers' need boxes, each widened by that reader's offsets,
   clipped to the observed's global extent. It is recomputed by every tile that
   needs it.
4. **Writes.** Only the `(W_k, owned)` rows of `du` are written. Every other
   element of `du` is left untouched. Tiles of one rank may therefore write one
   shared `du` concurrently.
5. **Dead halo cells.** Storage cells outside `1..N` on a non-periodic axis are
   never read. The caller may leave them uninitialized; the conformance test
   fills them with NaN.

### 5. Tile layout and signatures

- `u_S`, `aux_S` and `du_S` are species-major. Each variable is one
  column-major block over its storage box (first index fastest), in flat-layout
  order. This is today's layout with `S` in place of the global box.
- `du_S` has `u_S`'s layout; only owned rows are written.
- Index offsets are rebased by storage origin alone:

      slot = base_v + Σ_d (g_d - S_lo[v,d]) · stride[v,d]

  Julia's generated kernels already address `u[oln + Δ]` with bounds, base and
  strides read from a run-time `geo` table (`codegen_kernel.jl:315-325`), so
  only the table changes.

**Julia**

```julia
dec  = Decomposition(axes = ["lon" => (periodic = false,), "lat" => (periodic = false,)],
                     faces = ["lon_nodes" => "lon", "lat_nodes" => "lat"])
op   = select(rhs_program(prob); writes = W, terms = tags)   # U4 program type
h    = halo(op, dec)                                         # Dict{(var, set) => (lo, hi)}
tile = restrict(op, dec; owned = ("lon" => 17:32, "lat" => 1:13),
                         storage = ("lon" => 14:35, "lat" => -2:16))
lay  = tile_layout(tile)          # blocks, bases, strides, owned rows per variable
ws   = rhs_workspace(tile)
rhs!(du_S, tile, ws, u_S, aux_S, p, t, fs)   # fs: U4 ForcingSet, windows allowed
```

**Rust**

```rust
pub struct Decomposition { pub axes: Vec<AxisCut>, pub faces: Vec<(String, String)> }
pub struct AxisCut { pub index_set: String, pub period: Option<i64> }
pub struct TileBox { pub owned: Vec<(String, RangeInclusive<i64>)>,
                     pub storage: Vec<(String, RangeInclusive<i64>)> }

impl RhsProgram {                                         // U4
    pub fn select(&self, sel: &RhsSelection) -> Result<RhsProgram, SimulateError>;
    pub fn halo(&self, dec: &Decomposition) -> Result<HaloReport, SimulateError>;
    pub fn tile(&self, dec: &Decomposition, b: &TileBox) -> Result<TileProgram, SimulateError>;
    pub fn lanes(&self, kind: LaneKind, capacity: usize) -> Result<LaneProgram, SimulateError>;
}
impl TileProgram {                                        // Send + Sync (U4)
    pub fn layout(&self) -> &TileLayout;
    pub fn new_workspace(&self) -> RhsWorkspace;
    pub fn eval_into<T: StateScalar>(&self, u: &[T], aux: &[T], p: &[f64], t: f64,
        forcing: &ForcingSet, du: &mut [T], ws: &mut RhsWorkspace) -> Result<(), RhsError>;
}
```

In Rust a tile is a re-lowering of the tape. Slot boxes become need boxes.
`GatherPlan` segments are computed against storage-box sources, with ghost
classification against global extents (`tape/ir.rs:961-982`). `Region` and
`Assemble` clip. `DyWrite` targets only the owned rows of the storage box. The
state mirror covers `R_k ∩ S` only. `Ramp` keeps its absolute `lo`.

A `ForcingSet` slot may be a window with a per-axis origin. The build checks
that the window covers `forcing_reach`.

### 6. Lanes

`lanes(op, kind, capacity)`, for operators whose locality is `pointwise`
(`kind = Cells`) or `column` (`kind = Columns`).

**Inputs and layout.**

- `cells` is an `i32` array of global indices, `[D, L]`, with `L ≤ capacity`.
  `D` is every spatial index set of the output for `Cells`, and the horizontal
  sets only for `Columns`.
- `u_L`, `aux_L` and `du_L` are species-major with the lane index fastest:
  pointwise slot `(s-1)·L + l` (reseact's `NS` runs of `C` cells), column slot
  `((s-1)·nlev + (k-1))·L + l`. Every per-cell or per-level kernel, and every
  batched block solve (architecture §6.1), then vectorizes across lanes.

**Semantics** (normative): for each lane `l` and each `v ∈ W_k`,
`du_L[v, l] == G_k(…)[v, cells[:, l]]` bitwise. A lane's result depends only on
its own inputs. Consequences:

- Padding a batch with duplicates of a real cell is safe and discardable. This
  is reseact's rule (`capacity_chem.jl:353-364`).
- Lane order and batch composition cannot change a value. That is the property
  lane pools need (architecture §7.6).

**Reads.**

- State reads at offset 0 (pointwise) or along the column (column) come from
  `u_L` and `aux_L`.
- Every read that depends on the cell's index is evaluated from `cells`:
  coordinate expressions (`lon0_deg + (gi-1/2)·dlon_deg`), level gates
  (`gk < 2`), and affine reads of forcing, array parameters, const arrays and
  state-free observeds (`GEOSFP.T[r, gk, LAT0+gj, LON0+gi]`). These are
  reseact's 37 index-reading names: no surgery, and no rebuild per composition.
- State-free observeds keep their whole-domain build-time values and are
  gathered per lane. NEI's conservative regrid is computed once per model, not
  once per capacity.

**Pre-gathered inputs (optional).** `lane_inputs(lanes)` returns the gather
plan for every non-state input. The integrator can then fill per-shard lane
buffers once per epoch with `gather_lane_inputs!` and call the variant that
reads them. This is reseact's `gather_forcing!` (`capacity_chem.jl:414-439`),
moved into the compiler.

**Capacity.**

- Julia loops over lanes with a run-time count, so one build serves every `L`.
- Rust slot boxes are static, so a lane program is built per capacity
  (re-lowering only, no document rebuild). A call passes `n ≤ capacity`; lanes
  `n+1..capacity` are computed and unspecified.
- The XLA lane program is U13's.

**Rust signature.**

```rust
impl LaneProgram {
    pub fn eval_into<T: StateScalar>(&self, u: &[T], aux: &[T], p: &[f64], t: f64,
        forcing: &ForcingSet, cells: &[i32], n: usize,
        du: &mut [T], ws: &mut RhsWorkspace) -> Result<(), RhsError>;
}
```

**Julia signature.** `rhs!(du_L, lanes, ws, u_L, aux_L, p, t, fs, cells, n)`.

**Python.** `restrict` and `lanes` raise `compiler_unavailable` until the Python
native implementation needs them (D1).

## What this is not

- **Not halo exchange, MPI or threading.** The integrator does those (D6). The
  compiler only reports the halo.
- **Not run-time tile boxes.** A tile is rebuilt when geometry changes, which
  happens at checkpoint boundaries only.
- **Not `global` operators.** SSA, connected components and horizontal sums
  stay whole-domain (architecture §7.2).
- **Not a schema change.** The decomposition is an API argument.
  `ModelVariable.location` exists (`src/types.jl:1076-1078`) but is not needed.
- **Not band models or XLA.** Jacobian and VJP band models are derived
  documents compiled by this same path (U7, U8). XLA is U13.

## Conformance and tests

1. **Tile partition (normative, bitwise).** For every stencil fixture, with
   decompositions 1×1, 2×1, 3×2 (uneven tiles), and one tile narrower than the
   boundary band (owned `1..2` against PPM's three low slabs):
   - fill each tile's storage from the global state and dead halo cells with NaN;
   - scatter the owned rows into one vector and assert it equals `G_k` bitwise,
     in Julia and in Rust;
   - assert every element outside the owned rows of a NaN-prefilled `du` is
     untouched.
   
   Across bindings, agreement is the CONFORMANCE_SPEC §5.9 band.
2. **Boundary regions.** The EarthSciDiscretizations 1-D and latlon3d PPM
   fixtures: periodic (explicit wraps), inflow with halo operands
   (`ppm_flux_D_lon_mono_inflow_bc`), zero-gradient, no-flux, and a face field
   `[lon_nodes, lat, lev]` read at `i` and `i+1`.
3. **Halo report.** `halo` equals the planner's AST-derived report. A storage
   box one cell short raises `range_halo_insufficient`.
4. **Lanes.** Random permutation, duplicates and padding: bitwise per cell.
   Negative control: swapping two lanes' inputs must mismatch (reseact's version
   reports 7.6e10, `capacity_chem.jl:34-38`).
5. **Selection.** `Σ_k G_k` against the full `F`, within the planner's
   tolerance. An operator whose `writes` omit a variable that its terms write
   is refused.

## Work breakdown

1. **Selection (both bindings).** Keep U2 tags on summands through
   `coupling_apply.jl:429-432`. Drop unselected summands and rules. Add the
   `aux` layout.
2. **Need-box analysis (both).** Compose offsets through observed levels and
   regions. Produce the halo and forcing-reach reports.
3. **Julia tiles.** Intersect affine box kernels (`stencil_affine.jl:1376-1483`)
   before the class merge (`oop_merge.jl:1105-1135`), or filter merged kernels'
   lane tables per tile (`access_kernel.jl:779-784`). Restrict observed fills,
   prelude tiers and scan folds. Rebase the `geo` tables.
4. **Rust tiles.** Re-lower over need boxes (§5). Restrict the state mirror.
5. **Lanes (both).** Lower index-derived reads to gathers from `cells`. Add the
   optional pre-gather plan.
6. **Fixtures and tests (both).**

## Alternatives considered

- **A tile as its own small model,** like reseact's capacity builds and eqice's
  frozen-ghost blocks. It applies boundary rules or zero ghosts at interior
  edges, needs a build per tile, and needs document surgery.
- **Evaluate the global `F` and discard.** O(domain) work per tile per stage,
  with no MPI.
- **Local index space** (every tile starts at 1). It breaks boundary-region
  identity and every index-derived value. Global indices with a storage offset
  cost nothing.
- **Exchange intermediate observeds.** That means more exchanges per stage, and
  their halos are then halos of halos.
- **Run-time box arguments.** Julia almost has them already (`geo` tables).
  Rust bakes them into `GatherPlan`s. This is deferred; the build-time box is
  enough for checkpoint-boundary geometry changes.

## Open questions

1. Should `terms` selection live here or in U2?
2. Rust re-lowering cost per tile and per capacity is unmeasured *(assumption:
   seconds)*. If it is large, tiles of one class (same shape, same edge flags)
   should share a program with a run-time origin.
3. Must a face set's relation to its cell set (`lon_nodes` to `lon`) be declared
   by the caller, as here, or should esm or EarthSciDiscretizations declare it?
4. Should restricted observeds be exportable for output on a tile, or only
   through U6's observed kernels?
5. Rust and Python infer a `makearray`'s origin differently: Rust uses the
   minimum `lo` (`earthsci-ast-rs/src/simulate_array/eval.rs:5178-5199`),
   Python uses 1. Global index space (decision 1) settles it for tiles. Should
   the spec settle it for whole-domain programs too?
