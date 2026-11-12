---
title: "A Rust streaming Zarr v3 writer, with region writes, a commit protocol and indexed records"
description: "Give the Rust track the open / write_record / flush / close streaming handle the Julia and Python writers already have. Then extend the writer contract in all three bindings so that each MPI rank writes its own horizontal block without a gather (the reserved `region=`). One owner commits shared metadata from tokens the caller passes. Records can be written by index in any order: the adjoint emits forcing gradients in reverse and keeps checkpoint rings. Static variables and per-rank blobs cover gradient fields and bitwise restart. EarthSciIO stays MPI-free."
---

Status: draft (proposed by EarthSciIntegrator, 2026-10-13)
Target: EarthSciIO (write boundary: the streaming-output-sinks contract, EarthSciAST RFC §5 and §16; new `spec/writer.md`; output manifest `earthsciio/output-manifest/v2`, additive beside `v1`)
Needed by: EarthSciIntegrator milestone M4 (single-writer gradient sinks in Rust) and M5 (region writes)
Integrator refs: architecture.md §7.1, §7.4, §10, §11, §13 (U12); decisions D1, D5, D6, D10, D13, D18, D20

## Summary

The Rust writer commits a whole dataset in one call. The Julia and Python
writers stream, but all three refuse `region=`. This RFC specifies one writer
contract and implements it natively in each binding:

1. **A Rust streaming handle.** `ZarrStreamWriter::create / write_record /
   flush / close`, emitting the same sharded Zarr v3 store as the Julia
   reference. `write_zarr_v3` becomes a thin wrapper over it.
2. **Region writes.** A handle is bound at open to a shard-aligned, 0-based,
   half-open block of the non-time dims. Each shard object therefore has exactly
   one writer: no read-modify-write, no locks.
3. **Roles and a commit protocol.** Writers flush data and return a small
   `CommitToken`. The caller (MPI, threads) delivers the tokens to the owner.
   The owner checks that they tile the extent, then bumps shapes and writes the
   manifest, which is the commit point.
4. **Two time modes.** `append`, as today, and `indexed(n)`, where records are
   written by index in any order and may be rewritten.
5. **Static variables, rank-local blobs, more dtypes, and a read-back API**,
   which together cover gradient fields (D5) and bitwise restart (D10).

## Motivation

**Rust has no streaming writer.** `zarr_write.rs` "commits a whole dataset in
one [`write_zarr_v3`] call (the streaming record/flush handle is future work)"
(`EarthSciIO/rust/src/format/zarr_write.rs:19-23`). The schema carries every
record's data up front (`:132-168`), the array is stored with `subset_all`
(`:403-408`), and the manifest is computed from the finished schema
(`:427-547`). The streaming RFC lists Rust's Sink protocol as not ported
(`EarthSciAST/docs/content/rfcs/streaming-output-sinks.md:860-862`). Under D1
the Rust integrator needs the same streaming output as Julia: diagnostics,
gradients and checkpoints.

**Regions are reserved, not implemented.** In Julia, `region` "MUST be
`nothing` in Wave 1" (`julia/src/writer.jl:28-35`), enforced at
`julia/src/zarr_write.jl:486-487`; Python does the same
(`earthsciio/backends/zarr_write.py:386-392`). In EarthSciAST,
`sink_supports_partial` defaults to false and is "INERT in v1"
(`streaming-output-sinks.md:700`; `EarthSciAST.jl/src/data_output.jl:185-194`),
and `ZarrSink` refuses a selection (`ext/EarthSciASTEarthSciIOExt.jl:830-833`).
The constraint the RFC set itself, "each rank … must write its region into a
global logical array **without a global gather**"
(`streaming-output-sinks.md:84-85`), is still open. The integrator decomposes
horizontally by rank (D6, D20), so it hits that constraint at M5.

**Shared metadata is guarded only by file locks.** The Julia writer rewrites
every array's `zarr.json` shape after each flush (`zarr_write.jl:582-599`) and
then the manifest (`:621-637`), both under a `mkpidlock`
(`julia/src/store.jl:138-150`). §16.8 assigns metadata to a rank-0 owner but
calls it "a seam exercised by the object-store phase, not a v1 hot path"
(`streaming-output-sinks.md:768-776`). File locks do not work on S3: the Julia
`S3Store` is a stub (`store.jl:152-166`), and Julia output to `s3://` errors
(`zarr_write.jl:193-196`).

**A missing region would read back as zeros.** The pinned `fill_value` is 0.0,
"never NaN" (`conformance/CROSSLANG.md:280`; `zarr_write.rs:373`). If one rank's
shard is absent, readers see silent zeros, which is as wrong for a checkpoint as
for a gradient. Coverage must be checked before a commit becomes visible.

**What the integrator writes does not fit append-only, f64-only output.**
- *Forcing gradients come out in reverse.* The backward sweep visits epochs in
  reverse (architecture §10). A forcing record's gradient is complete only after
  both brackets that contain it have been processed (D5). Records therefore
  finish in descending time order, and the append-only writer cannot take them.
- *Some outputs have no time axis.* Gradients with respect to gridded fields and
  u0 have none, but every streaming variable must carry the time dim
  (`zarr_write.rs:262-267`; `zarr_write.jl:448-450`).
- *Restart must be bitwise.* For a fixed decomposition (§11, D10) the store
  needs controller state, per-cell dt and integer counts. The streaming RFC's
  restart is "tolerance-consistent" because it stores none of that
  (`streaming-output-sinks.md:746-766`).
- *Rust is f64-only.* Its writer writes only `float64` (`zarr_write.rs:132-145`,
  `:370`), while Julia also writes f32/i32/i64/u32/u64 (`zarr_write.jl:162-168`).
  WRF runs f32 state (D13, D18).
- *Adjoint checkpoints must be read back.* At 0.25° they total about 188 GB
  (`reseact.esm/SCALE_025.md:196`), are spilled to disk, and are read back
  record by record in reverse.

## Decisions of record

| # | Question | Decision | Why |
|---|---|---|---|
| 1 | Region granularity | **Bound at open**; per-call `region=` must be `nothing` or equal the bound region | Buffers are sized once and alignment is checked once. It activates the reserved argument without changing its meaning. |
| 2 | Shared shards | **Refused**: every partitioned dim's region bounds must sit on shard boundaries (or the array end) | One writer per object means no read-modify-write, no locks, and correctness on S3. |
| 3 | Metadata | **Owner-only**: shapes, coordinates, replicated variables, manifest | Follows §16.8. Contributors never write shared objects. |
| 4 | Coordination | **Tokens transported by the caller** | EarthSciIO stays MPI-free. MPI, threads and tests all just pass values. |
| 5 | Commit point | **The manifest write, done last by the owner after a coverage check** | Readers never see a gap filled with zeros. |
| 6 | Time modes | **`append`** (default) and **`indexed(n)`** | Reverse-order gradients and checkpoint rings need random-access records. |
| 7 | Manifest | **`v1` unchanged** for single-writer append; **`v2`** for regions, indexed, statics or blobs | Readers pin the label exactly (`zarr_write.rs:498-503`). |
| 8 | Dtypes | f64, f32, i32, i64, u8 in all three bindings | Restart parity with Julia; u8 for masks. |
| 9 | Precision of reads | **Bitwise**: all profiles are lossless (Blosc-zstd or zstd; `streaming-output-sinks.md:983-991`) and dtypes are preserved | D10 restart. |

## Design

### Schema

`StreamSchema` is today's `OutputSchema` without data, plus:
- `time_mode`: `append` or `indexed{n_records, times}`;
- `statics`: variables without a time dim;
- a `dtype` on every variable.

Dims, chunk and shard shapes, coordinates, group attributes and profile mean
what they mean today (`zarr_write.rs:147-168`; `zarr_write.jl:86-120`). On disk
the record axis comes first (`streaming-output-sinks.md:938-948`).

### Regions

`region = {dim: [start, stop)}` covers the non-time dims and is 0-based. An
omitted dim means the full extent.

- **Partitioned and replicated variables.** A variable that carries any region
  dim with a proper sub-range is *partitioned*: each writer supplies it shaped
  to its region. Any other variable is *replicated* and is written only by the
  owner; a contributor that supplies one gets an error.
- **Alignment rule.** For each partitioned dim `d`, `start % shard[d] == 0`, and
  `stop % shard[d] == 0` or `stop == len(d)`. A violation is
  `RegionNotShardAligned`, raised at open.
- **Chunks.** Inner chunks stay whatever the reader wants. Only shards (the
  write objects) must be owned by one writer.
- **Integrator consequence.** The partition planner must place rank-block
  boundaries on multiples of the output shard length (architecture §7.1, pass
  7). For example, 193 latitudes over 4 ranks with shard length 16 gives blocks
  48/48/48/49.

### Roles and the commit protocol

`Role` is one of:
- `Single`: today's behaviour; `flush` commits;
- `Owner{n_writers}`: also writes its own region;
- `Contributor{id}`.

```
owner:         create(schema, Owner{n}, region_0)          # group, coords, array zarr.json at shape[time]=0
caller:        barrier
contributors:  join(store, Contributor{r}, region_r)       # read + fingerprint-check owner's metadata
all writers:   write_record(t, region-shaped arrays) ...   # a full time shard is encoded and PUT whole
all writers:   tok_r = flush()                             # PUT any partial time shard of the region; return token
caller:        gather tokens to the owner (MPI_Gather in rank order, or in memory)
owner:         commit([tok_0 .. tok_{n-1}])                # verify, then shapes, time coord, manifest LAST
```

**The token.** A `CommitToken` is JSON: writer id, region, schema fingerprint,
the record indices written since the last commit, their `t` values, the
statics and blobs written, and the object keys PUT.

**What `commit` verifies.** The regions are disjoint and tile the extent of
every partitioned dim; every token carries the same record indices with
bitwise-equal `t`; the fingerprints match. On any failure it raises
`CommitError` naming the gap, the overlap or the divergent writer, and commits
nothing.

**Writes.**
- Data objects are atomic per object: stage-and-rename on a filesystem
  (`store.jl:118-136`), a single PUT or multipart-complete on an object store.
- A partial time shard is rewritten whole from the writer's buffer, as Julia
  does today (`zarr_write.jl:557-569`).
- Readers trust only the manifest and the `zarr.json` shapes. Objects beyond the
  committed prefix are invisible and are overwritten later.

### Time modes

- **`append`.** The record index is the running count, as today.
  `write_record(t, arrays)` returns the index.
- **`indexed{n_records, times}`.** The time axis and its coordinate are fixed at
  open, and the owner writes the coordinate.
  - `write_record_at(i, t, arrays)` accepts any order. `t` must equal `times[i]`
    bitwise.
  - A time shard holding some unwritten records is flushed with fill and
    rewritten when they arrive. The writer keeps the open shards' buffers.
  - The manifest records, per record: committed, `t`, generation.
  - Rewriting a committed record that other readers may still need requires the
    owner to `invalidate([i…])` and commit first. Otherwise a crash mid-rewrite
    would leave a corrupt record marked committed.

### Statics, blobs and read-back

- **`write_static(name, array)`** writes a time-less variable, region-shaped.
  Rewrites follow the same invalidate rule.
- **`write_blob(scope, key, bytes)`** stores small rank-local payloads at
  `blobs/<record|static>/<writer>/<key>`, for state that is not gridded.
- **Read-back.** `read_output_manifest(store)`, `read_record(store, var, i,
  region)`, `read_static(store, var, region)` and `read_blob(...)`. They read
  committed data only, directly from the store, not through the
  content-addressed cache: caching a checkpoint store would copy it.
- **`resume(store, role, region, at=t)`.** Reopens a store after preemption.
  - In `append` mode the owner truncates the committed records to `t <= at` and
    rewrites the manifest and shapes. Each writer reloads its region of the open
    time shard so that later flushes rewrite whole shards.
  - Under D10, records re-emitted after a restart are bitwise identical, so
    truncate-and-rewrite is idempotent.

### Manifest `v2`

`v2` contains every `v1` field (`julia/src/manifest.jl:99-141`) plus:
- `time_mode`;
- `writers: [{id, region}]`;
- `records`: committed ranges, plus `t` and `generation` in indexed mode;
- `statics`, `blobs`;
- `commit_seq`.

A JSON Schema lands in `spec/schemas/`. A single-writer append store with no
statics or blobs keeps writing byte-compatible `v1`.

### Per-binding API

**Rust** (`earthsciio::format::zarr_stream`; native and `wasm32`, where only
`Role::Single` is meaningful, via `OpfsStore`):

```rust
pub enum TimeMode { Append, Indexed { n_records: usize, times: Vec<f64> } }
pub enum Role { Single, Owner { n_writers: usize }, Contributor { id: usize } }
pub struct Region(pub Vec<(String, std::ops::Range<usize>)>);   // empty = whole extent

impl<S: ReadableWritableStorageTraits + 'static> ZarrStreamWriter<S> {
    pub fn create(store: Arc<S>, base_url: &str, schema: StreamSchema, role: Role, region: Region) -> Result<Self>;
    pub fn join(store: Arc<S>, base_url: &str, role: Role, region: Region) -> Result<Self>;
    pub fn resume(store: Arc<S>, base_url: &str, role: Role, region: Region, at: Option<f64>) -> Result<Self>;
    pub fn write_record(&mut self, t: f64, arrays: &RecordArrays) -> Result<u64>;
    pub fn write_record_at(&mut self, index: u64, t: f64, arrays: &RecordArrays) -> Result<()>;
    pub fn write_static(&mut self, name: &str, array: TypedArray<'_>) -> Result<()>;
    pub fn write_blob(&mut self, scope: BlobScope, key: &str, bytes: &[u8]) -> Result<()>;
    pub fn flush(&mut self) -> Result<CommitToken>;                       // Single: also commits
    pub fn invalidate(&mut self, indices: &[u64]) -> Result<()>;          // owner
    pub fn commit(&mut self, tokens: &[CommitToken]) -> Result<CommitReceipt>;   // owner
    pub fn close(self, tokens: &[CommitToken]) -> Result<Option<OutputManifestV2>>;  // owner commits; others: tokens = []
}
```

- `RecordArrays` maps a variable name to a typed row-major view of its region
  shape, without the time dim.
- `TypedArray` covers f64, f32, i32, i64 and u8.
- Shards are written whole with `zarrs` `Array::store_chunk`, and the owner
  resizes with `Array::set_shape`. Both exist in `zarrs` 0.23.13
  (`array/array_sync_writable.rs:87`, `array.rs:742`).
- `write_zarr_v3` and `write_all_to_store` become `create(Single)` → records →
  `close`. One code path serves every host (`zarr_write.rs:195-205`).

**Julia.** The existing generics gain `write_open!(w, store, base_url, schema;
role = :single, n_writers, id, region)` and a live-but-bound
`write_record!(…; region = nothing)`. `write_flush!` now returns a token. New:
`write_record_at!`, `write_static!`, `write_blob!`, `write_commit!`,
`write_invalidate!`, `write_resume`, `read_record`, `read_static`, `read_blob`,
and `UInt8`. Regions use the wire form `"lat" => (0, 48)`, 0-based. Single-writer behaviour
and the existing locks are unchanged. Multi-writer stores skip the metadata
locks, because the owner is the only metadata writer.

**Python.** `ZarrWriter.write_open(base_url, schema, role=…, region=…)`,
`write_record(h, t, arrays, region=None)`, `write_record_at`, `write_static`,
`write_blob`, `write_flush(h) -> CommitToken`, `write_commit(h, tokens)`,
`write_invalidate`, `write_resume`, `read_record`. Writes go to zarr-python
whole-shard selections. *Assumption to verify:* zarr-python 3 encodes a fully
covered shard without reading it back.

### Integrator use

- **Diagnostics.** One store per output grid (streaming RFC §9). Rank 0 is
  `Owner`, the others are `Contributor`. Each output tick calls `write_record`;
  each flush cadence runs `flush` → `MPI_Gather` in rank order → owner
  `commit`.
- **Restart store.** `indexed{n=2}`, time shard 1, `checkpoint` profile.
  Variables are prognostic state, per-cell dt, controller history and step
  counts, each region-written in its true dtype. Blobs hold shard compositions
  and other rank-local controller state; the owner writes tape position and
  forcing epoch as group attributes. Restart reads the slot with the highest
  committed generation, so a crash mid-write leaves the other slot valid
  without `invalidate`. Each rank reads its region with `read_record`, which
  gives a bitwise restart (§11).
- **Adjoint disk checkpoints.** Use a `Single` store per rank on node-local
  scratch, `indexed{n = #checkpoints}`, written forward and read with
  `read_record` in reverse. Per-rank stores avoid a coverage check on stores
  that hold only one node's ranks.
- **Gradient sinks (D5), always f64 (D18).**
  - *dJ/d(field) and dJ/d(u0)* are `write_static`, written once after the
    backward sweep. Partial sums live in the restart store.
  - *dJ/d(forcing)* gets one store per forcing cadence, because each schema has
    one `time_dim` (`zarr_write.rs:153-154`). It uses `indexed{n = records in
    the window, times = anchors}`, and the time coordinate is written at open.
  - Record `k+1` is written with `write_record_at` once epoch `k`'s adjoint has
    folded both bracket contributions, so writes descend.
- **Halo contributions.** Ranks' native forcing windows overlap by their halos.
  Halo contributions must be summed into the owning rank, in a fixed order,
  before writing. The writer rejects overlap, so the integrator does this.

## What this is not

- Not MPI. Tokens are values, and the caller moves them.
- Not a gather, and not shared-shard read-modify-write. Misaligned regions are
  refused.
- Not EarthSciAST's Sink protocol. `sink_supports_partial`, the shard-local
  `state_snapshot` and the Rust Sink port are follow-ups filed against
  EarthSciAST.
- Not NetCDF or parallel HDF5 (still out of scope,
  `streaming-output-sinks.md:605-606`). Not a codec change.

## Conformance and tests

The write corpus gains `write_spec_v2.json`. Every case runs in all three
writers, and every reader reads every store:

1. `region-tiling`: one writer against 2×2 region writers, with in-memory
   tokens. Decoded arrays must be bitwise equal. Intra-language shard bytes are
   equal, as §16.6 allows.
2. `region-misaligned`: an error at open. `region-gap` and `region-overlap`:
   `CommitError` at commit, with the committed prefix unchanged.
3. `indexed-reverse`: records written in descending order must equal forward
   append, bitwise.
4. `partial-shard-resume`: flush mid-shard, drop every handle uncommitted,
   resume, continue. The result must equal an uninterrupted run, and
   uncommitted objects must be ignored.
5. `truncate-on-resume`: rewritten records must be identical.
6. `dtypes`: f32, i32, i64 and u8 round-trip bitwise.
7. `statics-and-blobs`.
8. `manifest-v1-compat`: a single-writer append store's manifest equals today's,
   except `created_at`.
9. Object-store variants run offline (`object_store` memory and local FS;
   fsspec memory).

A Rust reader dumper joins `conformance/dumpers/`; today only Python and Julia
read (`CROSSLANG.md:233-237`).

## Work breakdown

1. **Spec.**
   - `spec/writer.md`: the normative writer contract, consolidating the
     EarthSciAST RFC §5 and §16 and this RFC.
   - `spec/schemas/output-manifest.schema.json` covering v1 and v2.
   - `write_spec_v2.json` and its generator.
2. **Rust.** `ZarrStreamWriter`, typed arrays, regions, tokens, indexed mode,
   statics, blobs, resume, read-back. Re-base `write_zarr_v3` on it. Gate the
   `wasm32` `Single` path.
3. **Julia.** Activate `region`; add roles and tokens, indexed mode, statics,
   blobs, resume, read-back and `UInt8`.
4. **Python.** The same, on zarr-python 3.
5. **Harness.** The multi-writer driver, the Rust reader dumper and the v2
   comparator rules.
6. **Follow-up RFC to EarthSciAST.** A region-capable `ZarrSink`, the
   shard-local snapshot, and the Rust Sink protocol.

## Alternatives considered

- **Gather to rank 0** (the v1 host-gather, `streaming-output-sinks.md:717-720`).
  Root memory and bandwidth grow with the globe, and the adjoint checkpoints
  would pass through one node. Kept as `Single`, rejected as the multi-rank
  default.
- **One store per rank everywhere.** Trivial to write, but output stops being
  one logical array. Recommended only for node-local adjoint spill.
- **Shared shards under locks.** Writers serialize and need cross-node locks:
  `mkpidlock` on a parallel filesystem, conditional PUT on S3 (still a stub,
  `store.jl:154-155`). Rejected.
- **Variable-size (rectilinear) chunk grids, one shard per rank.** Not Zarr v3
  core, and every reader would need it, including Julia's hand-rolled one.
  Rejected for now.
- **Append gradients in reverse and flip in a post-pass.** A full extra pass
  over the largest output. Rejected.
- **Per-writer manifest fragments merged by the owner at close.** Needs no
  caller transport, but relies on consistent listing and rename. It is a
  possible fallback for POSIX-only runs.

## Open questions

1. Should `commit` HEAD-check every object named in the tokens? This is cheap
   on a filesystem and costly on S3.
2. `fill_value` stays 0 (pinned). Should checkpoint stores use NaN as a
   defensive fill, even though coverage checks already prevent committed gaps?
3. Should EarthSciAST's tolerance-consistent checkpoint sink (§16.7) adopt this
   bitwise layout, or stay separate from the integrator's restart store?
4. Are tokens a documented wire format, which would allow mixed-language writers
   on one store, or opaque within one binding? Proposed: documented JSON, but
   mixed-language multi-writer is not tested in v1.
5. Should the writer encode shards on a background thread so that `write_record`
   never blocks on compression, matching U11's overlap on the read side?
