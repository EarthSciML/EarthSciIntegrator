---
title: "Asynchronous epoch prefetch and per-rank index-range windows"
description: "Two additions to the EarthSciIO Provider, specified once and implemented natively in Python, Julia and Rust. (a) A PrefetchGroup that decodes upcoming forcing epochs on background workers while the caller computes the current one, in forward, reverse or an explicit order, under a memory budget, with cancellation, and with results bitwise identical to synchronous refresh. (b) Named-dimension windows bound to a provider, so each MPI rank fetches and decodes only its horizontal block plus halo, plus a multi-window decode for node-level aggregation. No cache-format change, no MPI dependency."
---

Status: draft (proposed by EarthSciIntegrator, 2026-10-13)
Target: EarthSciIO (spec v1: Provider surface in `spec/README.md`; `select` vocabulary in `spec/registries.md` §2.1–2.2 and `spec/conformance.md` §3; new `spec/prefetch.md`; additive, cache format `v1` unchanged)
Needed by: EarthSciIntegrator milestone M5 (the prefetch half is usable from M3)
Integrator refs: architecture.md §7.1, §7.3, §7.5, §7.8, §10, §13 (U11); decisions D5, D6, D10, D15, D18, D20

## Summary

The integrator refreshes forcing at epoch boundaries and must not stall while it
does. Today a refresh is a synchronous decode on the caller's thread, and the
only `prefetch` downloads files into the disk cache. This RFC adds:

1. **`sample_at(t)`**, a side-effect-free form of `refresh(t)` that touches no
   provider buffer and is safe to call from a worker thread.
2. **`PrefetchGroup`**, which runs `sample_at` for the next epochs on background
   workers while the caller computes. It follows a caller-declared order:
   forward, reverse (the adjoint's backward sweep) or an explicit list (a
   revolve schedule). It is bounded by an epoch depth and a byte budget and can
   be cancelled. Every result is bitwise identical to what synchronous
   `refresh` would have returned.
3. **Named-dimension windows** (`select = {"dims": {...}}`) and
   `with_window(w)`, which binds a window to a provider at construction. Each
   rank then reads and decodes only its horizontal block plus halo.
   `sample_windows_at(t, [w...])` decodes several windows in one pass, so one
   rank per node can serve its neighbours.

EarthSciIO stays MPI-free and solver-free. A window is a plain selection, and an
epoch is a position in the union of the providers' `refresh_times`.

## Motivation

**Refresh dominates at scale, and it is serial.** At 0.25° the reseact forward
pass spent 1231.34 s per refresh, 98.3% of the pass (`reseact.esm/SCALE_025.md:27-33`,
`BUILD_SCALING.md:463`). A profile of that refresh shows it running entirely on
one thread: 34% zlib inflate and 53% the Float32→Float64 convert plus
file-order `permutedims` in the NetCDF reader (`BUILD_SCALING.md:566-578`). The
adjoint pays it twice, because the backward sweep re-reads every epoch:
"320 refreshes (160 forward + 160 backward)" (`SCALE_025.md:178-180`).

**The fix for the decode itself has partly landed; overlap has not.** That
measurement was taken against an EarthSciIO whose Julia NetCDF reader took no
keywords (`SCALE_025.md:54-61`). `main` now pushes `variables`, `select` and
`records` into the decode (`EarthSciIO/julia/src/readers.jl:537-538`; merged
PRs #1 `variables`, #2 `select` and #4 `records`, whose pre-merge branches are
the `EarthSciIO-varproj` and `EarthSciIO-records` worktrees; the
`EarthSciIO-pushdown` worktree now holds the unrelated HTTP-timeout branch). The projected refresh with `variables` alone is about 180 s
(`BUILD_SCALING.md:612-616`). That is still far more than a macro step, so it
has to be hidden behind compute. Nothing can do that today:

- `prefetch` only warms the cache and decodes nothing (`spec/README.md:57`;
  `earthsciio/provider.py:458-466`; `julia/src/provider.jl:595-602`;
  `rust/src/provider.rs:685-688`).
- The provider's decode is entangled with mutable state. Rust `refresh` takes
  `&mut self` and rewrites `buffers`, `coords`, `current_anchor` and a file LRU
  (`provider.rs:349-373`, `560-651`). Python keeps the same state
  (`provider.py:290-299`). A caller cannot run a second decode on another thread
  without racing the one the solver reads.
- No API takes an order. The backward sweep visits epochs in reverse
  (architecture §10), and the providers' file LRUs (`provider.py:547-582`,
  `provider.rs:828-841`) are tuned for forward stepping.

**Per-rank windows can be spelled today, but only badly.**

- *Mixed ranks.* A `select` is positional over file-order dims and applies only
  to fields whose rank equals the axis count (`spec/conformance.md:425-430`). An
  axis count that matches no field is an error (`conformance.md:502-503`). One
  positional window therefore cannot cover a provider that delivers 2-D A1
  fields `[time, lat, lon]` beside 3-D A3dyn fields `[time, lev, lat, lon]`, or a
  staggered `lon_edge` beside `lon`. The Rust `Selection` is positional only
  (`rust/src/format/mod.rs:243-252`).
- *Per-call windows bypass the file buffer.* A per-call `select` is a "peek"
  that re-reads the file fresh and skips the decoded-file LRU
  (`provider.rs:564-572`, `610-622`; `provider.py:370-377`, `552-553`).
  EarthSciAST's Rust adapter passes its declared selection on every refresh
  (`EarthSciAST/pkg/earthsci-ast-rs/src/esio_provider.rs:1689-1692`), so a
  windowed Rust source re-reads and re-decodes its file at every tick instead
  of once per file.
- *Chunk layout.* At 0.25° GEOS-FP A3dyn is chunked `[1, 1, 721, 1152]`, one
  global level per chunk. A window cannot reduce decompression
  (`INDEX_RANGE_READS.md:225-238`). If every rank on a node decodes its own
  window from the same file, the node inflates each global slice once per rank.
  A multi-window decode inflates it once.

**The window is enough.** reseact masked every native cell outside the
halo-inclusive window with NaN and got a bit-identical transport RHS
(`INDEX_RANGE_READS.md:61-72`). The spec already requires a windowed read to be
cell-for-cell identical to the full read sliced afterwards
(`conformance.md:504-507`). That property makes forcing values independent of
the decomposition, which D10 relies on.

## Decisions of record

| # | Question | Decision | Why |
|---|---|---|---|
| 1 | Where does async live | **In EarthSciIO, as `PrefetchGroup`** | The decode, its thread-safety and its equivalence to `refresh` are EarthSciIO's. Three integrators (Julia, Rust, later Python) would otherwise each re-implement them. |
| 2 | Unit of prefetch | **An epoch**: index `k` into `boundaries()`, the sorted union of the members' `refresh_times` | This matches the integrator's tape (§10) and its (t0, tf] refresh rule. EarthSciIO never learns about macro steps. |
| 3 | Order | **Declared up front:** `forward`, `reverse`, or an explicit list of epochs | Covers the forward pass, the backward sweep and revolve replays. A known order also makes look-ahead and eviction exact. |
| 4 | Equivalence | **Normative and bitwise**: `take(k)` equals `sample_at(B[k])`, which equals `refresh(B[k])` | Any scheduling difference would break D10 replay and restart. |
| 5 | Memory | `depth` (default 2 epochs held, decoded or decoding) **and** an optional `max_bytes` | "Two epochs in flight" is the common case. A byte cap protects 0.25° runs, where resident memory itself costs time (+18.8% decode at 73 GB RSS, `BUILD_SCALING.md:542-551`). |
| 6 | Window spelling | **A named-dimension `select`**, 0-based and half-open, in the existing axis vocabulary | Works across mixed ranks and staggered dims. It is one vocabulary with `axes`, and every reader that answers `supports_selection` honours it. |
| 7 | Window binding | **At construction** (`with_window`), never per call | This keeps the decoded-file reuse path. A per-call select remains a peek. |
| 8 | MPI | **None in EarthSciIO** | Ranks bind their own windows. Aggregation uses `sample_windows_at`, and the integrator owns the scatter. |

## Design

### `sample_at` (all bindings)

`sample_at(t)` returns the dataset that `refresh(t)` would return (variables
**and** coordinates), with the same snap-down to the anchor and the same bracket
and end-of-data rules. It reads and writes no provider buffer. It may be called
concurrently from several threads on one provider. Rust: `&self`, and `Provider`
is `Send + Sync` (its `Cache`, `Store`, `Transport` and `Reader` already are:
`rust/src/cache.rs:109-110`, `store/mod.rs:59`, `transport/mod.rs:79`,
`format/mod.rs:379`; a compile-time assertion is added). `refresh(t)` is
re-specified as `sample_at(t)` followed by the buffer update, so the two cannot
drift apart.

### Named-dimension windows

```
select = {"dims": {"<dim>": "all" | {"slice": [start, stop, step?]} | {"indices": [i, ...]}, ...}}
```

- **Exactly one of `axes` and `dims`.** Selectors use the existing vocabulary
  and its rules: 0-based, half-open, bounds checked against the file before any
  read, no dim dropped, order and repetition kept, empty legal
  (`conformance.md:509-533`).
- **Scope.** Each named selector applies **by name** to every decoded field and
  every coordinate that carries the dim, whatever its rank. Unnamed dims are
  `"all"`. A named dim that no field carries is an error.
- **Time.** Naming the provider's `time_dim` is refused: records belong to the
  provider, as for `axes` (`conformance.md:449-453`).
- **Wraparound.** A window that wraps a periodic longitude uses `indices`
  (e.g. `[1150, 1151, 0, 1, …]`), which is returned in the order given.
- **Gate.** The windowed read must be cell-for-cell identical to the full read
  sliced by name. This is the existing gate (`conformance.md:504-507`) extended
  to the new spelling.

`with_window(w)` returns a provider that shares the cache and reader, with `w`
baked as its `select` (Rust `DataSource::select`; Python/Julia
`reader_kwargs["select"]`). It is an error if the loader already bakes a
`select`. `sample_windows_at(t, [w1..wK])` returns K datasets, each identical to
`with_window(wi).sample_at(t)`. It fetches and decodes the bounding union once.

### `PrefetchGroup` (language-neutral)

```
PrefetchGroup(members: {key: Provider}, *, depth=2, max_bytes=None, fetch_ahead=None,
              workers=1, windows=None, transform=None)
  boundaries() -> [time]          # sorted ∪ refresh_times of DISCRETE members; epoch k = [B[k], B[k+1])
  start(order, from=None)         # order: "forward" | "reverse" | [epoch, ...]; cancels any previous order
  take(k) -> EpochSample          # blocks; k must be the next entry of the order
  ready(k) -> bool                # non-blocking
  cancel(wait=false)              # drop pending work; discard in-flight results
  close()                         # cancel(wait=true), join workers
  stats() -> {waited_s, decode_s, peak_bytes, epochs_decoded, overlap: bool}
```

- **Members.** Members must be DISCRETE and share one time type and unit.
  CONST members are rejected; the caller materializes them once.
- **What `take(k)` returns.** It returns, for every member, `Sample{data,
  anchor, changed}`, where `changed` says whether the anchor differs from the
  previous `take` (always true on the first `take` after `start`). This is
  Rust's "None-skip" made explicit (`EarthSciAST/.../provider.rs:139-145`).
- **Ordering rules.** Taking an epoch other than the next one in the order is an
  error that names the expected epoch. To re-seek, call `start`. An order may
  repeat epochs. A sample still held from an earlier slot is reused, not
  decoded again.
- **Budget.** `depth` bounds the epochs held (decoded or decoding) ahead of the
  consumer. `max_bytes` bounds their bytes, using the last decoded size of the
  same members as the estimate. A slot larger than `max_bytes` is still decoded,
  one at a time, and counted in `stats`. `fetch_ahead` (default `depth`) warms
  the cache for later epochs, bounded by disk rather than RAM.
- **Windows and transform.** With `windows`, each member yields K datasets via
  `sample_windows_at`. `transform(key, dataset)` runs on the worker. It must be
  pure, and it may fill caller-owned staging buffers but never live ones.
  EarthSciAST can push unit conversion through it.
- **Errors.** A decode error is stored against its `(epoch, key)` and raised by
  `take(k)`, never earlier. If several members fail, the error raised is the
  first by key order. Scheduling stops past a failed epoch. The error is the one
  `refresh` would have raised, offline-mode `CacheMiss` included.
- **Cancellation.** `cancel` drops queued work at once. Decodes already inside a
  C library run to completion and their results are freed. A group is reusable
  after `start`. `take`, `start` and `cancel` are called from one thread.
- **Determinism.** Workers may finish in any order. Results are keyed and
  delivered only by `take`. Decode is elementwise, with no reductions. Each
  output cell is written once from the same value (the zarr reader already
  states this, `rust/src/format/zarr.rs:28-38`). Depth, worker count, budget and
  timing change when a result is ready, never what it is.
- **Reusing decoded files.** A binding may keep decoded files across epochs
  inside the group. Python and Rust amortize one whole-file decode over many
  ticks (`provider.py:555-570`). Because the order is known, the group can evict
  a file exactly after the last epoch that needs it. This is a binding choice;
  the equivalence gate is what binds.

### Per-binding API

**Python** (`earthsciio.prefetch`; threads by default):

```python
w = Window({"lat": (120, 218), "lon": (300, 497)})  # -> {"dims": {...: {"slice": [a, b]}}}
prov = {s: providers[s].with_window(w) for s in discrete}
with PrefetchGroup(prov, depth=2, max_bytes=8 << 30, workers=2, executor="thread") as g:
    B = g.boundaries()  # list[datetime]
    g.start("forward")
    for k in range(len(B)):
        ep = g.take(k)  # ep["U"].data, ep["U"].changed
    g.start("reverse")
```

`executor="process"` is available when decoding under the GIL limits overlap.
NetCDF decodes hold one module lock, because netCDF4/HDF5 is not assumed to be
thread-safe (*assumption, to verify*).

**Julia** (`EarthSciIO.PrefetchGroup`; `Threads.@spawn` workers):

```julia
w  = Window("lat" => (120, 218), "lon" => (300, 497))       # 0-based half-open, the wire vocabulary
g  = PrefetchGroup(Dict(s => with_window(p, w) for (s, p) in discrete); depth = 2, workers = 2)
B  = boundaries(g)                                           # Vector{Float64}, the providers' time unit
start!(g, :forward)                                          # or :reverse, or a Vector{Int} of 1-based epochs
for k in eachindex(B); ep = take!(g, k); end
close(g)
```

Epoch indices are 1-based in Julia and 0-based elsewhere; the wire form of an
explicit order is 0-based. With `Threads.nthreads() == 1` the group decodes
inside `take!` and reports `overlap = false`. NetCDF decodes take a process-wide
lock: *assumption* that NCDatasets/NetCDF-C calls are not thread-safe; to be
verified before any lock is removed. Zarr decodes run concurrently.

**Rust** (`earthsciio::prefetch`; `std::thread` workers, no async runtime
required):

```rust
let w = Window::new().range("lat", 120, 218).range("lon", 300, 497);   // -> Selection::Named
let mut g = PrefetchGroup::builder()
    .member("U", Arc::new(p_u.with_window(w.clone())?))
    .member("V", Arc::new(p_v.with_window(w)?))
    .depth(2).max_bytes(8 << 30).workers(4)
    .build()?;                                     // PrefetchGroup<NativeDataset>; `.transform(f)` -> PrefetchGroup<T>
let b: Vec<f64> = g.boundaries();                  // Unix seconds, as refresh_times
g.start(Order::Forward)?;                          // Order::Reverse | Order::Explicit(Vec<usize>)
for k in 0..b.len() { let ep = g.take(k)?; /* ep.get("U").changed */ }
// Drop cancels and joins.
```

`Selection` is `#[non_exhaustive]` (`format/mod.rs:244`), so adding
`Selection::Named` is not a breaking change. The pure-Rust NetCDF reader
(`rust/Cargo.toml:153-169`) needs no lock, so members decode in parallel.

### How the integrator drives it

```
per rank:  P[s] = provider(s).with_window(W[rank][s])    # W from the partition plan (§7.1); halo included
           G = PrefetchGroup(P, depth, max_bytes, workers);  B = G.boundaries();  tape.epochs = B
forward:   G.start(forward, from = restart_epoch)
           for k: ep = G.take(k); copy changed samples into rank forcing buffers (host; then device on §7.8)
                  integrate macro steps with B[k] <= t0 < tf <= B[k+1]
backward:  G.start(reverse)            # or the revolve access list
           for k descending: ep = G.take(k); per macro step in reverse: restore, replay, adjoint
```

- **Refresh timing.** Boundaries are tstops, so no step straddles one. The
  refresh at `B[k+1]` fires at the end of the step whose `tf = B[k+1]`, which is
  reseact's (t0, tf] rule (`tools/reactant_handoff/op_split.jl:126-142`).
- **Reverse sweep.** A step's epoch is `searchsortedlast(B, t0)`, the same in
  both directions.
- **Aggregated decode.** For chunk layouts like GEOS-FP 0.25°, one rank per node
  may build its group with `windows = [W[r] for r on node]` and scatter with
  MPI. That choice is the integrator's.
- **Shared cache.** A node's ranks share the cache. Concurrent fetches of one
  URL download once (`spec/cache-format.md:284-298`).
- **Index rebasing.** The integrator, not EarthSciIO, rebases the model's
  index offsets to the window origin (`INDEX_RANGE_READS.md:279-289`).

## What this is not

- Not a scheduler of macro steps or a time blend. Interpolation stays in the
  model.
- Not MPI-aware. EarthSciIO has no communicator, rank, halo exchange or scatter.
- Not a GPU transfer layer. Samples are host arrays, and device double-buffering
  belongs to the integrator (§7.8).
- Not a cache-format or cache-key change. A window never changes the key
  (`registries.md:161-164`).
- Not a decoded-data cache across runs, and not a change to `prefetch`, which
  stays the offline cache warmer.

## Conformance and tests

New corpus cases, three-track, under the existing runner:

1. `named-window-mixed-rank`: NetCDF with 2-D and 3-D variables and a staggered
   dim. The named window must equal the full read sliced by name, coords
   included.
2. `named-window-zarr`: the same on a store-backed v3 store. Unselected chunk
   objects are poisoned, which proves they are never fetched.
3. `window-wrap`: an `indices` window across the last/first longitude.
4. `multi-window`: `sample_windows_at` must equal K separate windowed samples,
   bitwise.
5. `prefetch-equivalence`: a three-file DISCRETE source with
   `records_per_sample` 1 and 2. Run every combination of order (forward,
   reverse, explicit with repeats), depth {1, 2, 3}, workers {1, 4} and a tiny
   `max_bytes`. The `take` sequence must equal the synchronous `refresh`
   sequence: arrays, coords, attrs, dims and dtypes, bitwise. A test hook
   randomizes decode latency so that completion order varies.
6. `prefetch-errors`: a missing file at epoch k raises at `take(k)` and not
   before. `cancel` followed by `start(from=k)` reproduces the uninterrupted
   sequence. With a cold cache, offline mode raises `CacheMiss` at `take`.
7. Per-binding: Rust `Provider: Send + Sync` assertion; a Julia single-thread
   fallback test; and `stats.waited_s` drops to about zero when synthetic
   compute exceeds decode time.

`registries.json` records the `dims` select form per track in `reader_options`.

## Work breakdown

1. **Spec.** Write `spec/prefetch.md`. Extend `registries.md` §2.1–2.2 and
   `conformance.md` §3 with the `dims` form. Add `sample_at`,
   `sample_windows_at`, `with_window` and `PrefetchGroup` to the
   `spec/README.md` Provider surface. Add the cases.
2. **Python.** Named select in the `netcdf` and `zarr` readers, `sample_at`
   (factored out of `refresh`), `with_window`, `sample_windows_at`,
   `PrefetchGroup` with thread and process executors.
3. **Julia.** The same, plus the NetCDF lock, `Threads.@spawn` workers and the
   single-thread fallback.
4. **Rust.** `Selection::Named`, `&self` `sample_at`, the Send/Sync assertion,
   `PrefetchGroup<T>` on `std::thread`.
5. **Bench.** A prefetch-overlap benchmark beside `bench/zarr_peak_memory.py`,
   reporting hidden refresh fraction and peak RSS.
6. **Follow-ups filed against EarthSciAST** (not part of this RFC):
   `provider_sample` over a group, mapping neutral selections to `dims`, and
   the Rust adapter binding its declared select at construction.

## Alternatives considered

- **Background download only.** Rejected. Decode, not transfer, is the measured
  cost (`BUILD_SCALING.md:566-578`). Download look-ahead is kept as
  `fetch_ahead`.
- **The integrator wraps `refresh` in its own thread.** Rejected. Provider
  state is mutable (`provider.rs:560`), so two integrator languages would
  re-solve thread-safety, cancellation and equivalence, and drift apart.
- **One future per call (`refresh_async(t)`).** Simpler, but it gives no shared
  budget, no order-aware reuse and no single place to enforce determinism. It
  can be layered on a one-member group.
- **A positional window per rank-class.** Rejected. It fails on mixed-rank
  providers and staggered dims (`conformance.md:425-430`).
- **Spilling decoded epochs to node-local disk in the forward pass for the
  backward sweep.** This would halve decode in the adjoint, and it is
  attractive at 0.25°. It is left open (below). The integrator can already do
  it with the U12 writer.

## Open questions

1. Default `depth` at 0.25°: one epoch ahead or two, given the residency
   penalty. Settle after re-measuring refresh on the pushdown-enabled `main`.
2. Should decode keep source precision (Float32) instead of widening to
   Float64? The widening is 53% of the measured refresh profile, and D18 allows
   f32 or 16-bit forcing storage. That would be a separate, cross-track RFC.
3. Are NetCDF-C (via NCDatasets) and netCDF4-python safe to call from several
   threads at once? If they are, the decode locks can go.
4. Should a decoded-epoch spill for the backward sweep be an EarthSciIO feature
   or integrator code on top of U12?
5. Should `boundaries()` accept an explicit clip window `(t0, tf)`, or rely on
   each member's construction window as today (`provider.py:429-454`)?
