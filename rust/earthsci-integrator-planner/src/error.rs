//! Refusals (spec/planner-errors.json, D79) and the planner's other errors.

/// The refusal codes of spec/planner-errors.json that this crate raises.
pub mod codes {
    pub const RUN_CONFIG_INVALID: &str = "run_config_invalid";
    pub const VERSION_UNSUPPORTED: &str = "version_unsupported";
    pub const TOY_MODEL_NOT_PLANNABLE: &str = "toy_model_not_plannable";
    pub const MODEL_LOAD_FAILED: &str = "model_load_failed";
    pub const SOLVER_EXTERNAL_MISMATCH: &str = "solver_external_mismatch";
    pub const MODEL_FLATTEN_FAILED: &str = "model_flatten_failed";
    pub const UNSUPPORTED_ALGEBRAIC_EQUATION: &str = "unsupported_algebraic_equation";
    pub const UNSUPPORTED_EVENT: &str = "unsupported_event";
    pub const UNSUPPORTED_PARAMETER_UPDATE: &str = "unsupported_parameter_update";
    pub const UNSUPPORTED_FORCING: &str = "unsupported_forcing";
    pub const UNSUPPORTED_INDEX_SET: &str = "unsupported_index_set";
    pub const UNSUPPORTED_MAP_OPERATOR: &str = "unsupported_map_operator";
    pub const UNSUPPORTED_STEP_LIMIT: &str = "unsupported_step_limit";
    pub const UNSUPPORTED_SOLVE_NODE: &str = "unsupported_solve_node";
    pub const UNSUPPORTED_SPLIT_EXPLICIT: &str = "unsupported_split_explicit";
    pub const UNSUPPORTED_CROSSING_UPDATE: &str = "unsupported_crossing_update";
    pub const AXES_UNRESOLVED: &str = "axes_unresolved";
    pub const NAME_UNRESOLVED: &str = "name_unresolved";
    pub const NAME_WRONG_KIND: &str = "name_wrong_kind";
    pub const OPERATOR_UNDEFINED: &str = "operator_undefined";
    pub const TERM_SELECTED_TWICE: &str = "term_selected_twice";
    pub const SELECTOR_MATCHES_NOTHING: &str = "selector_matches_nothing";
    pub const AMBIGUOUS_TERM_OWNER: &str = "ambiguous_term_owner";
    pub const TERM_ORIGIN_UNKNOWN: &str = "term_origin_unknown";
    pub const OPERATOR_NOT_ADVANCED: &str = "operator_not_advanced";
    pub const OPERATOR_ADVANCED_TWICE: &str = "operator_advanced_twice";
    pub const IMEX_PARTITION_INVALID: &str = "imex_partition_invalid";
    pub const DUPLICATE_NODE_ID: &str = "duplicate_node_id";
    pub const NODE_ID_RESERVED: &str = "node_id_reserved";
    pub const HOLD_TENDENCY_CHILD_NOT_LEAF: &str = "hold_tendency_child_not_leaf";
    pub const TSTOP_OFF_FIXED_STEP: &str = "tstop_off_fixed_step";
    pub const PRECISION_CONTRADICTS_DECLARATION: &str = "precision_contradicts_declaration";
    pub const PRECISION_RHS_BELOW_STATE: &str = "precision_rhs_below_state";
    pub const PRECISION_CONFLICT: &str = "precision_conflict";
    pub const PRECISION_LOW_WITH_CLASSICAL_ROSENBROCK: &str =
        "precision_low_with_classical_rosenbrock";
    pub const WRT_MATCHES_NOTHING: &str = "wrt_matches_nothing";
    pub const WRT_BOUND_PARAMETER: &str = "wrt_bound_parameter";
    pub const STEP_CONTROL_MISSING: &str = "step_control_missing";
    pub const ADAPTIVE_WITHOUT_ERROR_ESTIMATE: &str = "adaptive_without_error_estimate";
    pub const TIME_INTERVAL_MISSING: &str = "time_interval_missing";
    pub const TIME_INTERVAL_EMPTY: &str = "time_interval_empty";
    pub const TIME_INVALID_DATE: &str = "time_invalid_date";
    pub const TIME_ORIGIN_INVALID: &str = "time_origin_invalid";
    pub const BIND_DT_LEAF_WITHOUT_DT0: &str = "bind_dt_leaf_without_dt0";
    pub const AUTO_STRATEGY_UNAVAILABLE: &str = "auto_strategy_unavailable";
    pub const BAND_MODEL_FAILED: &str = "band_model_failed";
    pub const LIFT_SHAPE_UNRESOLVED: &str = "lift_shape_unresolved";
    pub const RESERVED_COMPONENT_NAME: &str = "reserved_component_name";
    pub const PARTITION_AXIS_MISSING: &str = "partition_axis_missing";
    pub const PARTITION_TOO_FINE: &str = "partition_too_fine";
    pub const PARTITION_CONFLICT: &str = "partition_conflict";

    /// Every code above, for the test that checks them against
    /// spec/planner-errors.json.
    pub const ALL: &[&str] = &[
        RUN_CONFIG_INVALID,
        VERSION_UNSUPPORTED,
        TOY_MODEL_NOT_PLANNABLE,
        MODEL_LOAD_FAILED,
        SOLVER_EXTERNAL_MISMATCH,
        MODEL_FLATTEN_FAILED,
        UNSUPPORTED_ALGEBRAIC_EQUATION,
        UNSUPPORTED_EVENT,
        UNSUPPORTED_PARAMETER_UPDATE,
        UNSUPPORTED_FORCING,
        UNSUPPORTED_INDEX_SET,
        UNSUPPORTED_MAP_OPERATOR,
        UNSUPPORTED_STEP_LIMIT,
        UNSUPPORTED_SOLVE_NODE,
        UNSUPPORTED_SPLIT_EXPLICIT,
        UNSUPPORTED_CROSSING_UPDATE,
        AXES_UNRESOLVED,
        NAME_UNRESOLVED,
        NAME_WRONG_KIND,
        OPERATOR_UNDEFINED,
        TERM_SELECTED_TWICE,
        SELECTOR_MATCHES_NOTHING,
        AMBIGUOUS_TERM_OWNER,
        TERM_ORIGIN_UNKNOWN,
        OPERATOR_NOT_ADVANCED,
        OPERATOR_ADVANCED_TWICE,
        IMEX_PARTITION_INVALID,
        DUPLICATE_NODE_ID,
        NODE_ID_RESERVED,
        HOLD_TENDENCY_CHILD_NOT_LEAF,
        TSTOP_OFF_FIXED_STEP,
        PRECISION_CONTRADICTS_DECLARATION,
        PRECISION_RHS_BELOW_STATE,
        PRECISION_CONFLICT,
        PRECISION_LOW_WITH_CLASSICAL_ROSENBROCK,
        WRT_MATCHES_NOTHING,
        WRT_BOUND_PARAMETER,
        STEP_CONTROL_MISSING,
        ADAPTIVE_WITHOUT_ERROR_ESTIMATE,
        TIME_INTERVAL_MISSING,
        TIME_INTERVAL_EMPTY,
        TIME_INVALID_DATE,
        TIME_ORIGIN_INVALID,
        BIND_DT_LEAF_WITHOUT_DT0,
        AUTO_STRATEGY_UNAVAILABLE,
        BAND_MODEL_FAILED,
        LIFT_SHAPE_UNRESOLVED,
        RESERVED_COMPONENT_NAME,
        PARTITION_AXIS_MISSING,
        PARTITION_TOO_FINE,
        PARTITION_CONFLICT,
    ];
}

/// A refusal: a code of spec/planner-errors.json, the JSON pointer into the
/// run config it concerns (`/model` for the model), and a message. Only the
/// code and the pointer are conformance-tested (spec/plan.md §4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    pub code: &'static str,
    pub pointer: String,
    pub message: String,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at {:?}: {}", self.code, self.pointer, self.message)
    }
}

/// Why planning stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    /// A refusal of the spec (spec/planner-errors.json).
    Refused(Refusal),
    /// A step that needs EarthSciASTDiff (bands, structure, stiffness by the
    /// Gershgorin bound, derived documents, validation), which arrives in
    /// stage 2 of M2. Never a refusal: it names what is missing.
    NotYetImplemented(String),
    /// A file could not be read.
    Io(String),
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::Refused(r) => write!(f, "refused: {r}"),
            PlanError::NotYetImplemented(s) => write!(f, "not yet implemented: {s}"),
            PlanError::Io(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for PlanError {}

/// A refusal with `code` at `pointer`.
pub fn refuse(
    code: &'static str,
    pointer: impl Into<String>,
    message: impl Into<String>,
) -> PlanError {
    PlanError::Refused(Refusal {
        code,
        pointer: pointer.into(),
        message: message.into(),
    })
}

/// A JSON pointer token (RFC 6901): `~` as `~0`, `/` as `~1`.
pub fn token(s: &str) -> String {
    s.replace('~', "~0").replace('/', "~1")
}

/// `base` extended by one reference token.
pub fn ptr(base: &str, tok: impl std::fmt::Display) -> String {
    format!("{base}/{}", token(&tok.to_string()))
}

pub type Result<T> = std::result::Result<T, PlanError>;
