---
title: "A time per lane in lane programs"
description: "EarthSciAST's lane programs (U3) evaluate a pointwise or column operator over a run-time list of cells, one call for a batch of lanes, but take one time t per call. Under dt_scope cell each cell has its own step-size controller, so after its first step every cell of a batch is at a different time, and a batch collapses to one call per cell. This RFC asks for lane programs that take one time per lane, with each lane's value equal, bit for bit, to the whole-domain program's at its cell and its time."
---

Status: draft (proposed by EarthSciIntegrator, 2026-11-24); implemented in draft EarthSciAST#781 (stacked on #780): Rust evaluates a batch at mixed times in one pass; Julia offers the call and serves it by groups of equal time, since its 0-d observeds are one value per call.
Target: EarthSciAST (API_SPEC §5.8 "Lanes", CONFORMANCE_SPEC §5.50); Rust (`RhsProgram::lanes`, `LaneProgram`) and Julia (`lanes`, `LaneProgram`); Python if it gains lane programs.
Needed by: EarthSciIntegrator milestone M3: the solve-time budget of the M3 gate (D96) with `dt_scope: cell` (D97, D98).
Integrator refs: spec/partitioning.md §3.2, §3.3; architecture.md §7.6; decisions D17, D96, D97, D98

File citations are relative to the `code/` directory that holds all the repositories, at EarthSciAST `main` as of 2026-11-24 (`e839cf92a`).

## Summary

A lane program evaluates an operator over `L` lanes in one call, each lane one cell (or column), with every lane equal to the whole-domain program at its cell (the U3 contract). The call takes one `t` (Rust `LaneProgram::eval_into(u, aux, p, t, forcing, cells, n, du, ws)`, `pkg/earthsci-ast-rs/src/simulate_array/range/lanes.rs`; Julia `rhs!(du, lanes, ws, u, aux, p, t, fs, cells, n)`, `pkg/EarthSciAST.jl/src/range/lanes.jl`).

EarthSciIntegrator's `dt_scope: cell` (spec/partitioning.md §2.3) gives every cell its own controller. A lane pool steps all its lanes together, but each by its own `h`, so after the first step no two lanes are at the same time, and a right-hand side that reads `t` (reseact's photolysis follows the solar zenith angle) cannot be evaluated for the batch in one call. Today a runtime calls the program once per distinct time, each call at full capacity with the other lanes as padding (D98): correct, but up to `L` times the work.

**Proposal.** A lane program takes one time per lane:

- Rust: `LaneProgram::eval_times_into(u, aux, p, ts, forcing, cells, n, du, ws)` and `eval_gathered_times_into(u, aux, inputs, p, ts, cells, n, du, ws)`, with `ts` holding at least `n` and at most `capacity` times; a padding lane takes lane 1's time.
- Julia: the same, `rhs!(du, lanes, ws, u, aux, p, ts::AbstractVector, fs, cells, n)` dispatching on a vector of times.
- **Contract.** Lane `l`'s value equals, bit for bit, the whole-domain program's at cell `cells[:, l]` and time `ts[l]`, whatever the other lanes hold. With every `ts[l]` equal to `t` the call equals the existing one-time call.

## Design

1. **Time is a lane input.** The program's internal state gains a lane block `__lane_t` of `capacity` entries beside the lane's cell indices; the call writes it from `ts` (or from `t` in the one-time call). A rule's reads of `t` become reads of `__lane_t` at the lane.
2. **Observeds that depend on `t`.** An observed that depends on `t`, directly or through other observeds, is computed per lane, as a state-dependent observed already is: an array observed over the lane's sets in its lane form, a scalar observed as one value per lane. Observeds that depend on neither the state nor `t` stay computed once per call.
3. **Pre-gathered inputs.** A read whose subscripts involve `t` is never hoisted into the pre-gather plan, which is filled once per batch of cells and is therefore time-independent.
4. **Forcing.** One forcing set per call, as today. Lanes that take different sets (one ending a step on a forcing epoch boundary) are evaluated in separate calls (spec/operator-interface.md §8.4.1; spec/partitioning.md §7).

## Conformance and tests

- Each binding: a program reading `t` directly, through a scalar observed and through an array observed; lanes at distinct times and a padding lane; every lane equal bit for bit to the whole-domain program at its time; the one-time call equal to the per-lane call with equal times.
- A CONFORMANCE_SPEC §5.50 addition stating the contract above.

## Alternatives considered

- **One call per distinct time** (the integrator's current workaround, D98). Correct; costs up to `capacity` times the work of a batched call.
- **Grouping lanes by time.** Lanes almost never share a time after their first step, so groups are of size one.
- **Synchronous steps across lanes** (one `h` per batch). That is `dt_scope: shard`, a different integration whose results depend on the batch composition (D97); it does not serve `cell`.

## Work breakdown

1. Rust: lane block `__lane_t`, the rewrite of `t` and of time-dependent observeds, the two entry points, tests.
2. Julia: the same in `src/range/lanes.jl`.
3. API_SPEC §5.8 and CONFORMANCE_SPEC §5.50 text.

## Open questions

1. Should a lane program also take one forcing set per lane, so a batch spanning an epoch boundary is one call? Recommended: not now; such batches are rare (one step per lane per epoch).
