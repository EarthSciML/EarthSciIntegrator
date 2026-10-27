//! Reading the run config (spec/plan.md §4 step 1, spec/run-config.md).
//!
//! The run config stays a `serde_json::Value` throughout planning: every
//! refusal names a JSON pointer into it, and walking the value keeps the
//! pointer at hand.

use crate::error::{codes, refuse, PlanError, Result};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The version of the integrator spec this planner implements.
pub const SPEC_VERSION: &str = "0.1.0";

const RUN_CONFIG_SCHEMA: &str = include_str!("../../../spec/schemas/run-config.schema.json");

/// A run config as read from its file.
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// The path the planner was given.
    pub path: PathBuf,
    /// The file's bytes, for `inputs.run_config.sha256` (spec/plan.md §7.7).
    pub bytes: Vec<u8>,
    /// The parsed document.
    pub value: Value,
}

impl RunConfig {
    /// The directory `model` is resolved against (spec/plan.md §4 step 2).
    pub fn dir(&self) -> PathBuf {
        match self.path.parent() {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => PathBuf::from("."),
        }
    }

    /// The file name without directories (spec/plan.md §7.7).
    pub fn file_name(&self) -> String {
        self.path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    /// The value at a JSON pointer, if present.
    pub fn at(&self, pointer: &str) -> Option<&Value> {
        self.value.pointer(pointer)
    }

    /// The `model` string as written.
    pub fn model(&self) -> &str {
        self.value
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default()
    }
}

fn validator() -> &'static jsonschema::Validator {
    static V: OnceLock<jsonschema::Validator> = OnceLock::new();
    V.get_or_init(|| {
        let schema: Value =
            serde_json::from_str(RUN_CONFIG_SCHEMA).expect("run-config schema is JSON");
        jsonschema::draft202012::new(&schema).expect("run-config schema compiles")
    })
}

/// Step 1 of spec/plan.md §4: parse the run config, validate it against
/// `schemas/run-config.schema.json` (`run_config_invalid`) and refuse a
/// version this planner does not implement (`version_unsupported`).
pub fn read(path: &Path) -> Result<RunConfig> {
    let bytes = std::fs::read(path)
        .map_err(|e| PlanError::Io(format!("cannot read {}: {e}", path.display())))?;
    let value: Value = serde_json::from_slice(&bytes).map_err(|e| {
        refuse(
            codes::RUN_CONFIG_INVALID,
            "",
            format!("the run config is not JSON: {e}"),
        )
    })?;
    check_schema(&value)?;
    let version = value
        .get("esm_integrator")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if version != SPEC_VERSION {
        return Err(refuse(
            codes::VERSION_UNSUPPORTED,
            "/esm_integrator",
            format!("esm_integrator {version} is not {SPEC_VERSION}, the version this planner implements"),
        ));
    }
    Ok(RunConfig {
        path: path.to_path_buf(),
        bytes,
        value,
    })
}

/// Validates a run config against its schema; the refusal points at the
/// first failing instance location.
pub fn check_schema(value: &Value) -> Result<()> {
    if let Some(e) = validator().iter_errors(value).next() {
        return Err(refuse(
            codes::RUN_CONFIG_INVALID,
            // At the empty pointer: a failing `oneOf` has no single
            // location (spec/plan.md §4 step 1).
            "",
            format!("the run config does not validate against its schema: {e}"),
        ));
    }
    Ok(())
}

/// A literal duration (spec/run-config.md §2.2) in seconds: a number, or a
/// `"<number> <unit>"` string, the number parsed as binary64 and multiplied
/// by the unit's seconds. `None` when `v` is neither (a parameter name).
pub fn literal_duration(v: &Value) -> Option<f64> {
    if let Some(x) = v.as_f64() {
        return Some(x);
    }
    let s = v.as_str()?;
    for (unit, factor) in [("min", 60.0), ("day", 86400.0), ("h", 3600.0), ("s", 1.0)] {
        if let Some(num) = s.strip_suffix(unit) {
            let num = num.strip_suffix(' ').unwrap_or(num);
            if num.is_empty() || !num.starts_with(|c: char| c.is_ascii_digit()) {
                continue;
            }
            if let Ok(x) = num.parse::<f64>() {
                return Some(x * factor);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn durations() {
        assert_eq!(literal_duration(&json!(2.5)), Some(2.5));
        assert_eq!(literal_duration(&json!("2 h")), Some(7200.0));
        assert_eq!(literal_duration(&json!("1.5min")), Some(90.0));
        assert_eq!(literal_duration(&json!("1e2 s")), Some(100.0));
        assert_eq!(literal_duration(&json!("3650 day")), Some(3650.0 * 86400.0));
        assert_eq!(literal_duration(&json!("0.5 s")), Some(0.5));
        // The number as the nearest binary64, times the unit (spec/plan.md §7.1).
        assert_eq!(literal_duration(&json!("1.1 h")), Some(3960.0000000000005));
        assert_eq!(literal_duration(&json!("1.1h")), Some(1.1 * 3600.0));
        assert_eq!(literal_duration(&json!("Met.dt")), None);
    }

    fn check_dir(dir: &str, valid: bool) {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/conformance");
        let mut n = 0;
        for entry in std::fs::read_dir(root.join(dir)).unwrap() {
            let p = entry.unwrap().path();
            if p.extension().and_then(|e| e.to_str()) != Some("json")
                || p.file_name().unwrap() == "expected-errors.json"
            {
                continue;
            }
            let v: Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
            let r = check_schema(&v);
            assert_eq!(r.is_ok(), valid, "{}: {r:?}", p.display());
            n += 1;
        }
        assert!(n > 0);
    }

    /// The schema validator agrees with conformance tier 0 on the run-config
    /// fixtures.
    #[test]
    fn schema_fixtures() {
        check_dir("run-configs/valid", true);
        check_dir("run-configs/invalid", false);
    }
}
