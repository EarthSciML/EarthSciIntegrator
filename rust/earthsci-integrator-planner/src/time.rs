//! The run interval (spec/plan.md §7.2, spec/run-config.md §1.1, D60, D74).

use crate::error::{codes, ptr, refuse, PlanError, Result};
use crate::model::ModelInfo;
use crate::runconfig::{literal_duration, RunConfig};
use serde_json::Value;

/// The resolved interval in seconds of model time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub start: f64,
    pub end: f64,
}

/// A date-time as esm-spec §2.2.7 converts it: whole POSIX seconds `W`
/// (proleptic Gregorian, offset subtracted) and the fraction `f`.
#[derive(Debug, Clone, Copy)]
struct Instant {
    whole: i64,
    frac: f64,
}

enum StampError {
    /// Not of the accepted form.
    Form,
    /// The form is right but the day does not exist in its month.
    NoDay,
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        2 if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

fn digits(b: &[u8]) -> Option<i64> {
    if b.is_empty() || !b.iter().all(u8::is_ascii_digit) {
        return None;
    }
    Some(b.iter().fold(0, |acc, &c| acc * 10 + i64::from(c - b'0')))
}

/// Reads `YYYY-MM-DDThh:mm:ss(.f+)?` and a zone field by field, as
/// esm-spec §2.2.7 (and EarthSciAST's `resolve_time_span`) does: `utc_only`
/// is a run-interval bound (`T`, `Z`), otherwise a time origin (`T` or `t`,
/// then `Z`, `z` or `±hh:mm`).
fn parse_date_time(s: &str, utc_only: bool) -> std::result::Result<Instant, StampError> {
    let b = s.as_bytes();
    if b.len() < 20
        || b[4] != b'-'
        || b[7] != b'-'
        || !(b[10] == b'T' || (!utc_only && b[10] == b't'))
        || b[13] != b':'
        || b[16] != b':'
    {
        return Err(StampError::Form);
    }
    let field = |r: std::ops::Range<usize>| digits(&b[r]).ok_or(StampError::Form);
    let (year, month, day) = (field(0..4)?, field(5..7)?, field(8..10)?);
    let (hour, minute, second) = (field(11..13)?, field(14..16)?, field(17..19)?);
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return Err(StampError::Form);
    }
    let mut i = 19;
    let mut frac = 0.0;
    if b[i] == b'.' {
        let start = i + 1;
        let mut j = start;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j == start {
            return Err(StampError::Form);
        }
        frac = format!("0.{}", &s[start..j])
            .parse::<f64>()
            .map_err(|_| StampError::Form)?;
        i = j;
    }
    let offset = match &b[i..] {
        [b'Z'] => 0,
        [b'z'] if !utc_only => 0,
        [sign @ (b'+' | b'-'), h1, h2, b':', m1, m2] if !utc_only => {
            let oh = digits(&[*h1, *h2]).ok_or(StampError::Form)?;
            let om = digits(&[*m1, *m2]).ok_or(StampError::Form)?;
            if oh > 23 || om > 59 {
                return Err(StampError::Form);
            }
            let off = oh * 3600 + om * 60;
            if *sign == b'-' {
                -off
            } else {
                off
            }
        }
        _ => return Err(StampError::Form),
    };
    if day > days_in_month(year, month) {
        return Err(StampError::NoDay);
    }
    let whole =
        days_from_civil(year, month, day) * 86_400 + hour * 3600 + minute * 60 + second - offset;
    Ok(Instant { whole, frac })
}

/// The model's time origin (esm-spec §2.2.7): `domain.temporal.reference_time`,
/// else `domain.temporal.start`, else the POSIX epoch. `Err` carries the
/// unreadable field.
fn time_origin(info: &ModelInfo) -> std::result::Result<Instant, String> {
    let epoch = Instant {
        whole: 0,
        frac: 0.0,
    };
    let temporal = info
        .file
        .domain
        .as_ref()
        .and_then(|d| d.temporal.as_ref())
        .and_then(Value::as_object);
    let Some(temporal) = temporal else {
        return Ok(epoch);
    };
    for key in ["reference_time", "start"] {
        let Some(v) = temporal.get(key) else {
            continue;
        };
        return match v.as_str().map(|s| parse_date_time(s, false)) {
            Some(Ok(origin)) => Ok(origin),
            _ => Err(format!("domain.temporal.{key} = {v}")),
        };
    }
    Ok(epoch)
}

/// One bound of a run interval, in seconds of model time.
fn bound(v: &Value, pointer: &str, info: &ModelInfo) -> Result<f64> {
    if let Some(x) = v.as_f64() {
        return Ok(x);
    }
    let text = v.as_str().unwrap_or_default();
    let stamp = parse_date_time(text, true).map_err(|e| match e {
        StampError::NoDay => refuse(
            codes::TIME_INVALID_DATE,
            pointer,
            format!("{text} names a day its month does not have"),
        ),
        StampError::Form => refuse(
            codes::RUN_CONFIG_INVALID,
            pointer,
            format!("{text} is not an RFC 3339 UTC timestamp"),
        ),
    })?;
    let origin = time_origin(info).map_err(|field| {
        refuse(
            codes::TIME_ORIGIN_INVALID,
            pointer,
            format!("the timestamp {text} counts from the model's time origin, but {field} is not a readable date-time"),
        )
    })?;
    Ok((stamp.whole - origin.whole) as f64 + (stamp.frac - origin.frac))
}

/// An interval given as `{"start", "end"}` or `{"start", "duration"}`, with
/// the refusals of spec/run-config.md §8 rule 14 at `base`.
fn from_value(v: &Value, base: &str, info: &ModelInfo) -> Result<Interval> {
    let start = bound(&v["start"], &ptr(base, "start"), info)?;
    let end = match (v.get("end"), v.get("duration")) {
        (Some(e), _) => bound(e, &ptr(base, "end"), info)?,
        (None, Some(d)) => {
            let d = literal_duration(d).ok_or_else(|| {
                refuse(
                    codes::RUN_CONFIG_INVALID,
                    ptr(base, "duration"),
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

/// spec/plan.md §7.2: the caller's interval, else the run config's `time`,
/// else the model's `solver.time_span` (read with EarthSciAST's
/// `resolve_time_span`), the first that gives one giving all of it.
///
/// `caller` has the run config's `time` form. Its refusals carry the empty
/// pointer, since the caller's interval is not in the run config.
pub fn resolve(rc: &RunConfig, info: &ModelInfo, caller: Option<&Value>) -> Result<Interval> {
    if let Some(v) = caller {
        return from_value(v, "", info);
    }
    if let Some(v) = rc.at("/time") {
        return from_value(v, "/time", info);
    }
    match earthsci_ast::resolve_time_span(&info.file, None) {
        Ok(Some((start, end))) => Ok(Interval { start, end }),
        Ok(None) => Err(refuse(
            codes::TIME_INTERVAL_MISSING,
            "/time",
            "no run interval: the caller, the run config's time and the model's solver.time_span give none",
        )),
        Err(e) => Err(PlanError::Refused(crate::error::Refusal {
            code: codes::MODEL_LOAD_FAILED,
            pointer: "/model".into(),
            message: e.to_string(),
        })),
    }
}
