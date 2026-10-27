//! EarthSciIntegrator planner (spec/plan.md).
//!
//! Stage 1 of M2: reading the inputs (§4), pass 1 terms and operators
//! (§5.1), pass 2's locality rules over a band list (§5.2), pass 3 (§5.3),
//! the resolved settings of §7, the parts of pass 5 that need no
//! EarthSciASTDiff (§5.5), the canonical JSON writer (§2.1) and the
//! refusals that can be decided without bands. Band lists come in through
//! [`locality::BandSource`], which stage 2 implements with EarthSciASTDiff's
//! `jacobian_bands`; structure, stiffness by the Gershgorin bound, derived
//! documents and validation arrive with it.
//!
//! - [`runconfig`]: reading and validating the run config (§4 step 1).
//! - [`model`]: loading and flattening the model, M2's scope (§4 steps 2–6).
//! - [`axes`], [`time`]: §5.2.1 and §7.2.
//! - [`terms`]: terms, provenance and owners (§5.1.1–§5.1.2).
//! - [`locality`]: band entries, offsets, classes and radii (§5.2).
//! - [`operators`]: selection, operators and DOF sets (§5.1.3–§5.1.4, §5.3).
//! - [`schedule`]: the `"auto"` schedule, node ids, the schedule's checks,
//!   the resolved tree and reference steps (§5.5.4–§5.5.5, §7.1).
//! - [`strategy`]: stiffness rules 1–3 and the method table (§5.5.2–§5.5.3).
//! - [`settings`]: step, partition, precision, checkpoint, inputs (§7.3–§7.7).
//! - [`plan`]: the pipeline.
//! - [`jcs`]: RFC 8785 canonical JSON (§2.1).

pub mod axes;
pub mod error;
pub mod jcs;
pub mod locality;
pub mod model;
pub mod operators;
pub mod plan;
pub mod runconfig;
pub mod schedule;
pub mod settings;
pub mod strategy;
pub mod terms;
pub mod time;

pub use error::{PlanError, Refusal};
pub use plan::{plan, Options, Planned};
