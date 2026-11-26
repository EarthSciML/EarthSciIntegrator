---
title: "EarthSciIntegrator"
description: "One specification for integrating EarthSciML models, implemented natively in Julia and Rust."
---

EarthSciIntegrator is a time integrator for the full 3-D `.esm` models of the
EarthSciML stack. A run configuration says which operators make up a model's
right-hand side, which method integrates each one, how the operators are split,
and what to differentiate; a planner turns that and an `.esm` document into a
plan; and a library runs the plan forward and backward. It supports explicit,
implicit, IMEX and split-explicit schemes, nested nonlinear solves, per-locality
partitioning, and discrete adjoints of whole simulations.

There is one specification, implemented natively in Julia and in Rust, and the
two are checked against each other by a shared conformance suite.

- **[Getting started](/generated/api/guides/getting-started/)**: build a run
  from a run configuration, integrate it, differentiate it and replay it, in
  both languages.
- **[API reference](/generated/api/)**: the Julia and Rust APIs side by side.
- **[Specification](/spec/)**: what an implementation must do: the run
  configuration, the planner, methods and controllers, the operator interface,
  the tape, and the discrete adjoint.
- **[Architecture](/architecture/)** and **[decisions](/decisions/)**: the
  design and the reasons for it. The **[survey](/survey/)** records the
  requirements it rests on, and the **[RFCs](/rfcs/)** the changes it needs in
  sibling repositories.

The status of the implementation is in the [specification index](/spec/): the
Julia and Rust implementations cover the M1 methods and the first stage of the
planner.
