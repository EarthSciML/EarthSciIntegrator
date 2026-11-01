//! The run interval (spec/plan.md §7.2, spec/run-config.md §1.1, D60, D74).

use crate::error::{codes, ptr, refuse, PlanError, Result};
use crate::model::ModelInfo;
use crate::runconfig::{literal_duration, RunConfig};
use earthsci_ast::TimeOriginSource;
use serde_json::Value;

/// The resolved interval in seconds of model time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub start: f64,
    pub end: f64,
}

/// One bound of a run interval, in seconds of model time: a number, or a
/// timestamp converted by EarthSciAST's `timestamp_to_seconds` from
/// `origin`'s time origin (esm-spec §2.2.7), its codes mapped to rule 14's
/// (spec/plan.md §7.2). A bound that is neither is `run_config_invalid`.
fn bound<S: TimeOriginSource + ?Sized>(v: &Value, pointer: &str, origin: &S) -> Result<f64> {
    if let Some(x) = v.as_f64() {
        return Ok(x);
    }
    let Some(text) = v.as_str() else {
        return Err(refuse(
            codes::RUN_CONFIG_INVALID,
            pointer,
            format!("{v} is neither seconds nor a timestamp"),
        ));
    };
    earthsci_ast::timestamp_to_seconds(origin, text).map_err(|e| {
        let code = match e.code {
            "timestamp_malformed" | "timestamp_no_calendar_day" => codes::TIME_INVALID_DATE,
            "time_origin_invalid" => codes::TIME_ORIGIN_INVALID,
            other => {
                return PlanError::Internal(format!(
                    "timestamp_to_seconds({text}) raised {other}: {}",
                    e.message
                ))
            }
        };
        refuse(code, pointer, format!("{text}: {} ({})", e.message, e.code))
    })
}

/// An interval given as `{"start", "end"}` or `{"start", "duration"}`, with
/// the refusals of spec/run-config.md §8 rule 14 at `base`, or at the field
/// under `base` when `fields` (the interval is in the run config).
fn from_value<S: TimeOriginSource + ?Sized>(
    v: &Value,
    base: &str,
    fields: bool,
    origin: &S,
) -> Result<Interval> {
    let at = |key: &str| {
        if fields {
            ptr(base, key)
        } else {
            base.to_string()
        }
    };
    let start = bound(&v["start"], &at("start"), origin)?;
    let end = match (v.get("end"), v.get("duration")) {
        (Some(e), _) => bound(e, &at("end"), origin)?,
        (None, Some(d)) => {
            let d = literal_duration(d).ok_or_else(|| {
                refuse(
                    codes::RUN_CONFIG_INVALID,
                    at("duration"),
                    format!("{d} is not a literal duration"),
                )
            })?;
            start + d
        }
        (None, None) => {
            return Err(refuse(
                codes::RUN_CONFIG_INVALID,
                base,
                "an interval needs end or duration",
            ))
        }
    };
    // "Not after": a NaN bound is no interval either.
    if end.partial_cmp(&start) != Some(std::cmp::Ordering::Greater) {
        return Err(refuse(
            codes::TIME_INTERVAL_EMPTY,
            base,
            format!("the end ({end} s) is not after the start ({start} s)"),
        ));
    }
    Ok(Interval { start, end })
}

/// The rule-14 refusal, at `/model`, for a model document whose
/// `solver.time_span` EarthSciAST rejects. EarthSciAST gives every such
/// rejection the one code `solver_time_span_invalid`; spec/plan.md §7.2 asks
/// for the code of the cause, which is why the planner converts the bounds
/// itself, with `timestamp_to_seconds` over the raw `domain`. `None` when the
/// cause is not one of rule 14's (a bound that is neither a number nor a
/// string).
pub fn model_span_refusal(doc: &Value) -> Option<PlanError> {
    let span = doc.pointer("/solver/time_span")?;
    let domain = doc.get("domain").unwrap_or(&Value::Null);
    match from_value(span, "/model", false, domain) {
        Err(PlanError::Refused(r))
            if [
                codes::TIME_INTERVAL_EMPTY,
                codes::TIME_INVALID_DATE,
                codes::TIME_ORIGIN_INVALID,
            ]
            .contains(&r.code) =>
        {
            Some(PlanError::Refused(r))
        }
        _ => None,
    }
}

/// spec/plan.md §7.2: the caller's interval, else the run config's `time`,
/// else the model's `solver.time_span` (read with EarthSciAST's
/// `resolve_time_span`), the first that gives one giving all of it.
///
/// `caller` has the run config's `time` form. Its refusals carry the empty
/// pointer, since the caller's interval is not in the run config.
pub fn resolve(rc: &RunConfig, info: &ModelInfo, caller: Option<&Value>) -> Result<Interval> {
    if let Some(v) = caller {
        return from_value(v, "", false, &info.file);
    }
    if let Some(v) = rc.at("/time") {
        return from_value(v, "/time", true, &info.file);
    }
    match earthsci_ast::resolve_time_span(&info.file, None) {
        Ok(Some((start, end))) => Ok(Interval { start, end }),
        Ok(None) => Err(refuse(
            codes::TIME_INTERVAL_MISSING,
            "/time",
            "no run interval: the caller, the run config's time and the model's solver.time_span give none",
        )),
        Err(e) => Err(serde_json::to_value(&info.file)
            .ok()
            .and_then(|doc| model_span_refusal(&doc))
            .unwrap_or_else(|| refuse(codes::MODEL_LOAD_FAILED, "/model", e.to_string()))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn refusal(r: Result<Interval>) -> (&'static str, String) {
        match r {
            Err(PlanError::Refused(r)) => (r.code, r.pointer),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn caller_refusals_carry_the_empty_pointer() {
        let none = &Value::Null;
        let bad_day = json!({"start": "2021-02-30T00:00:00Z", "end": "2021-03-02T00:00:00Z"});
        assert_eq!(
            refusal(from_value(&bad_day, "", false, none)),
            (codes::TIME_INVALID_DATE, String::new())
        );
        assert_eq!(
            refusal(from_value(&bad_day, "/time", true, none)),
            (codes::TIME_INVALID_DATE, "/time/start".into())
        );
        // A malformed timestamp the schema has not checked (the caller's).
        let malformed = json!({"start": "2021-03-01", "end": "2021-03-02T00:00:00Z"});
        assert_eq!(
            refusal(from_value(&malformed, "", false, none)),
            (codes::TIME_INVALID_DATE, String::new())
        );
        let bad_origin = json!({"temporal": {"start": "yesterday"}});
        let stamps = json!({"start": "2021-03-01T00:00:00Z", "end": "2021-03-02T00:00:00Z"});
        assert_eq!(
            refusal(from_value(&stamps, "", false, &bad_origin)),
            (codes::TIME_ORIGIN_INVALID, String::new())
        );
        let bad_duration = json!({"start": 0, "duration": "ten s"});
        assert_eq!(
            refusal(from_value(&bad_duration, "", false, none)),
            (codes::RUN_CONFIG_INVALID, String::new())
        );
    }

    #[test]
    fn model_span_refusals_take_their_cause() {
        let code = |doc: Value| match model_span_refusal(&doc) {
            Some(PlanError::Refused(r)) => Some((r.code, r.pointer)),
            _ => None,
        };
        let at_model = |c| Some((c, "/model".to_string()));
        assert_eq!(
            code(json!({"solver": {"time_span": {"start": 100, "end": 50}}})),
            at_model(codes::TIME_INTERVAL_EMPTY)
        );
        assert_eq!(
            code(json!({"solver": {"time_span": {
                "start": "2021-02-30T00:00:00Z", "end": "2021-03-02T00:00:00Z"}}})),
            at_model(codes::TIME_INVALID_DATE)
        );
        assert_eq!(
            code(json!({
                "domain": {"temporal": {"reference_time": "yesterday"}},
                "solver": {"time_span": {
                    "start": "2021-03-01T00:00:00Z", "end": "2021-03-02T00:00:00Z"}}})),
            at_model(codes::TIME_ORIGIN_INVALID)
        );
        assert_eq!(
            code(json!({"solver": {"time_span": {
                "start": "2021-03-01 00:00", "end": "2021-03-02T00:00:00Z"}}})),
            at_model(codes::TIME_INVALID_DATE)
        );
        assert_eq!(
            code(json!({"solver": {"time_span": {"start": 0, "end": 1}}})),
            None
        );
        assert_eq!(
            code(json!({"solver": {"time_span": {"start": true, "end": 1}}})),
            None
        );
        assert_eq!(code(json!({"solver": {}})), None);
    }
}
