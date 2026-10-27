---
title: "Getting started"
description: "Build a run from a run configuration, integrate it, differentiate it with the discrete adjoint and replay it, in Julia and Rust."
weight: 10
---

This guide runs one small problem through the library's main steps. The
integrator is specified once (see the [specification](/spec/)) and implemented
natively in Julia and Rust, so each step shows both; pick your language on any
tab and the rest of the site follows. Every snippet is cut from a program that
runs, and checks its own result, in CI.

The problem is the toy problem `forced_decay`: the scalar equation
$u' = -k u + \sin(\omega t)$ with $k = 2$, $\omega = 3$ and $u(0) = 1$, which has an
exact solution. The *toy problems* are hand-written models behind the operator
interface that the conformance suite uses; a real run configuration names an
`.esm` document instead (see [the planner](/generated/api/topics/planner)).

## A problem

<!-- example: toy -->

## Forward run

A **run configuration** is a JSON object: the model, a `schedule` that says
which operators each leaf integrates and with which method (`RK4` at a fixed
step of 0.1 here), the macro `step`, and optionally an `adjoint` block naming a
functional to differentiate. Building a run checks the configuration, and
refuses what it cannot run; running it forward integrates over the interval and
records the tape: every step, accepted or rejected, with its size.

<!-- example: forward -->

## Gradient

With an `adjoint` block, the run's functional can be differentiated with respect
to the initial state and the parameters. The gradient is the *discrete* adjoint
of the recorded run: it differentiates exactly the steps the forward run took,
not a continuous approximation of them. See the
[adjoint chapter](/spec/adjoint/) for the definition.

<!-- example: gradient -->

## Replay

A run can be replayed on its tape. Every recorded step is held fixed, so the
replay gives the same result bit for bit, and with a perturbed initial state or
parameters it is the map the adjoint differentiates.

<!-- example: replay -->

## Refusals

A run configuration that asks for something invalid or unsupported is refused
when the run is built, not partway through the integration.

<!-- example: refusal -->

See [`build_run`](/generated/api/ref/build_run) and
[`run_spec`](/generated/api/ref/run_spec) for the full set of options.
