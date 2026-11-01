//! EarthSciIntegrator planner (spec/plan.md).
//!
//! M2: reading the inputs (§4), pass 1 terms and operators (§5.1), pass 2
//! locality from EarthSciASTDiff's band lists (§5.2), pass 3 (§5.3), pass 4
//! structure (§5.4), pass 5 stiffness and strategy (§5.5), the derived
//! documents (§6), the resolved settings of §7, pass 8 validation (§8) and
//! the canonical JSON writer (§2.1).
//!
//! - [`runconfig`]: reading and validating the run config (§4 step 1).
//! - [`model`]: loading and flattening the model, M2's scope (§4 steps 2–6).
//! - [`axes`], [`time`]: §5.2.1 and §7.2.
//! - [`terms`]: terms, provenance and owners (§5.1.1–§5.1.2).
//! - [`locality`]: band entries, offsets, classes and radii (§5.2).
//! - [`astdiff`]: the EarthSciASTDiff seam: flattened views of derived
//!   documents, band lists, and the term band source (§5.2.2, §5.4).
//! - [`operators`]: selection, operators and DOF sets (§5.1.3–§5.1.4, §5.3).
//! - [`structure`]: pass 4 (§5.4).
//! - [`schedule`]: the `"auto"` schedule, node ids, the schedule's checks,
//!   the resolved tree and reference steps (§5.5.4–§5.5.5, §7.1).
//! - [`strategy`]: stiffness rules 1–3 and the method table (§5.5.2–§5.5.3).
//! - [`settings`]: step, partition, precision, checkpoint, inputs (§7.3–§7.7).
//! - [`derived`]: the right-hand-side and Jacobian documents (§6).
//! - [`numeric`]: DOF order, sampled states, document evaluation and band
//!   Jacobians (§5.5.1, §8).
//! - [`validation`]: the Gershgorin bound (§5.4) and pass 8's checks (§8).
//! - [`plan`]: the pipeline and the plan directory.
//! - [`jcs`]: RFC 8785 canonical JSON (§2.1).

pub mod astdiff;
pub mod axes;
pub mod derived;
pub mod error;
pub mod jcs;
pub mod locality;
pub mod model;
pub mod numeric;
pub mod operators;
pub mod plan;
pub mod runconfig;
pub mod schedule;
pub mod settings;
pub mod strategy;
pub mod structure;
pub mod terms;
pub mod time;
pub mod validation;

pub use error::{PlanError, Refusal};
pub use plan::{plan, Options, Planned};
