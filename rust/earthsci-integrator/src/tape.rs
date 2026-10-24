//! The tape, checkpoints and the JSON Lines form (spec/tape.md).

use crate::Scalar;
use earthsci_integrator_solvers::controller::ControllerState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A leaf's return code (spec/methods.md §12).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Retcode {
    Success,
    MaxIters,
    DtLessThanMin,
    Unstable,
    DtNaN,
    ConvergenceFailure,
    DomainViolation,
}

/// One record of the tape: one invocation of one schedule node, or a macro
/// step (spec/tape.md §2, §5).
///
/// Records are in execution order; a composite node's record (`subcycle`,
/// the macro step) follows the records of the invocations it contains.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "node", rename_all = "lowercase", deny_unknown_fields)]
pub enum TapeRecord {
    Leaf {
        path: String,
        #[serde(with = "num")]
        t: f64,
        #[serde(with = "num")]
        dt: f64,
        /// The accepted steps `(t_n, h_n)`.
        #[serde(with = "num_pairs")]
        steps: Vec<[f64; 2]>,
        accepted: u64,
        rejected: u64,
        retcode: Retcode,
        /// Clips under `domain.on_violation: "clip"` (spec/tape.md §2):
        /// `[k, [i, ...]]` with `k` the 0-based index of the accepted step in
        /// `steps` and the `i` the ascending 0-based indices, into the leaf's
        /// state vector, of the DOFs that step clipped. Steps that clipped
        /// nothing are omitted, and the key when nothing clipped.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        clips: Vec<(usize, Vec<usize>)>,
    },
    Subcycle {
        path: String,
        #[serde(with = "num")]
        t: f64,
        #[serde(with = "num")]
        dt: f64,
        /// Number of substeps.
        n: usize,
    },
    Map {
        path: String,
        #[serde(with = "num")]
        t: f64,
        #[serde(with = "num")]
        dt: f64,
        /// The map's recorded discrete decisions.
        mask: Vec<u8>,
    },
    Macro {
        path: String,
        #[serde(with = "num")]
        t: f64,
        #[serde(with = "num")]
        dt: f64,
        /// The reason for the macro step's size (spec/run-config.md §5).
        reason: String,
    },
}

impl TapeRecord {
    /// The invocation path (spec/tape.md §1).
    pub fn path(&self) -> &str {
        match self {
            TapeRecord::Leaf { path, .. }
            | TapeRecord::Subcycle { path, .. }
            | TapeRecord::Map { path, .. }
            | TapeRecord::Macro { path, .. } => path,
        }
    }

    /// Bitwise equality: like `==`, but numbers compare by their bits, so
    /// `0.0` and `-0.0` differ and NaNs with equal bits are equal.
    pub fn bitwise_eq(&self, other: &TapeRecord) -> bool {
        use TapeRecord as R;
        let b = |x: &f64, y: &f64| x.to_bits() == y.to_bits();
        match (self, other) {
            (
                R::Leaf {
                    path,
                    t,
                    dt,
                    steps,
                    accepted,
                    rejected,
                    retcode,
                    clips,
                },
                R::Leaf {
                    path: path2,
                    t: t2,
                    dt: dt2,
                    steps: steps2,
                    accepted: accepted2,
                    rejected: rejected2,
                    retcode: retcode2,
                    clips: clips2,
                },
            ) => {
                path == path2
                    && b(t, t2)
                    && b(dt, dt2)
                    && steps.len() == steps2.len()
                    && steps
                        .iter()
                        .zip(steps2)
                        .all(|(x, y)| b(&x[0], &y[0]) && b(&x[1], &y[1]))
                    && accepted == accepted2
                    && rejected == rejected2
                    && retcode == retcode2
                    && clips == clips2
            }
            (
                R::Subcycle { path, t, dt, n },
                R::Subcycle {
                    path: path2,
                    t: t2,
                    dt: dt2,
                    n: n2,
                },
            ) => path == path2 && b(t, t2) && b(dt, dt2) && n == n2,
            (
                R::Map { path, t, dt, mask },
                R::Map {
                    path: path2,
                    t: t2,
                    dt: dt2,
                    mask: mask2,
                },
            ) => path == path2 && b(t, t2) && b(dt, dt2) && mask == mask2,
            (
                R::Macro {
                    path,
                    t,
                    dt,
                    reason,
                },
                R::Macro {
                    path: path2,
                    t: t2,
                    dt: dt2,
                    reason: reason2,
                },
            ) => path == path2 && b(t, t2) && b(dt, dt2) && reason == reason2,
            _ => false,
        }
    }
}

/// Numbers in the JSON Lines form (spec/tape.md §5): finite ones in shortest
/// round-trip form, non-finite ones as the strings `"NaN"`, `"Infinity"` and
/// `"-Infinity"`, which JSON numbers cannot hold.
mod num {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(x: &f64, s: S) -> Result<S::Ok, S::Error> {
        if x.is_finite() {
            s.serialize_f64(*x)
        } else if x.is_nan() {
            s.serialize_str("NaN")
        } else if *x > 0.0 {
            s.serialize_str("Infinity")
        } else {
            s.serialize_str("-Infinity")
        }
    }

    #[derive(Deserialize)]
    #[serde(untagged)]
    pub enum Num {
        F(f64),
        S(String),
    }

    impl Num {
        pub fn value<E: serde::de::Error>(self) -> Result<f64, E> {
            match self {
                Num::F(x) => Ok(x),
                Num::S(s) => match s.as_str() {
                    "NaN" => Ok(f64::NAN),
                    "Infinity" => Ok(f64::INFINITY),
                    "-Infinity" => Ok(f64::NEG_INFINITY),
                    o => Err(E::custom(format!("{o:?} is not a number"))),
                },
            }
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        Num::deserialize(d)?.value()
    }
}

/// `steps` with the number form of [`num`].
mod num_pairs {
    use super::num::Num;
    use serde::ser::SerializeSeq;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    struct N(f64);

    impl Serialize for N {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            super::num::serialize(&self.0, s)
        }
    }

    pub fn serialize<S: Serializer>(v: &[[f64; 2]], s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(v.len()))?;
        for x in v {
            seq.serialize_element(&[N(x[0]), N(x[1])])?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<[f64; 2]>, D::Error> {
        Vec::<[Num; 2]>::deserialize(d)?
            .into_iter()
            .map(|[a, b]| Ok([a.value()?, b.value()?]))
            .collect()
    }
}

/// The tape of a run.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Tape {
    pub records: Vec<TapeRecord>,
}

impl Tape {
    /// The JSON Lines form, one record per line (spec/tape.md §5).
    pub fn to_jsonl(&self) -> String {
        let mut s = String::new();
        for r in &self.records {
            s.push_str(&serde_json::to_string(r).expect("tape record serializes"));
            s.push('\n');
        }
        s
    }

    /// Reads the JSON Lines form back.
    pub fn from_jsonl(text: &str) -> Result<Tape, String> {
        let mut records = Vec::new();
        for (i, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            records
                .push(serde_json::from_str(line).map_err(|e| format!("tape line {}: {e}", i + 1))?);
        }
        Ok(Tape { records })
    }

    /// Records by path.
    pub fn index(&self) -> HashMap<&str, &TapeRecord> {
        self.records.iter().map(|r| (r.path(), r)).collect()
    }

    /// The records of macro step `k`, the macro record last.
    pub fn macro_segment(&self, k: usize) -> &[TapeRecord] {
        let prefix = format!("{k}/");
        let key = k.to_string();
        let start = self
            .records
            .iter()
            .position(|r| r.path().starts_with(&prefix) || r.path() == key)
            .unwrap_or(self.records.len());
        let end = self.records[start..]
            .iter()
            .position(|r| r.path() == key)
            .map(|i| start + i + 1)
            .unwrap_or(self.records.len());
        &self.records[start..end]
    }

    /// Accepted and rejected step counts over all leaf invocations.
    pub fn step_counts(&self) -> (u64, u64) {
        let mut a = 0;
        let mut r = 0;
        for rec in &self.records {
            if let TapeRecord::Leaf {
                accepted, rejected, ..
            } = rec
            {
                a += accepted;
                r += rejected;
            }
        }
        (a, r)
    }
}

/// A leaf's persistent state (spec/methods.md §8.6), part of checkpoints.
#[derive(Debug, Clone, PartialEq)]
pub struct LeafPersist {
    /// The proposed next step; `None` before the leaf's first step.
    pub proposal: Option<f64>,
    /// The controller's history, for adaptive leaves.
    pub ctrl: Option<ControllerState>,
    /// Newton rate `θ_prev` and `η_old` (spec/methods.md §5.2; unused by
    /// explicit leaves, kept so checkpoints have their final shape).
    pub theta_prev: f64,
    pub eta_old: f64,
}

impl LeafPersist {
    /// The state of a fresh run.
    pub fn new(ctrl: Option<ControllerState>) -> Self {
        LeafPersist {
            proposal: None,
            ctrl,
            theta_prev: 1.0,
            eta_old: 1.0,
        }
    }

    /// Bitwise equality of the persistent state.
    pub fn bitwise_eq(&self, o: &LeafPersist) -> bool {
        let b = |x: f64| x.to_bits();
        let ob = |x: Option<f64>| x.map(f64::to_bits);
        let cs = |c: &Option<ControllerState>| {
            c.map(|c| (b(c.eold), b(c.pid_eps[0]), b(c.pid_eps[1]), c.accepted_any))
        };
        ob(self.proposal) == ob(o.proposal)
            && cs(&self.ctrl) == cs(&o.ctrl)
            && b(self.theta_prev) == b(o.theta_prev)
            && b(self.eta_old) == b(o.eta_old)
    }
}

/// A checkpoint at a macro-step boundary (spec/tape.md §4).
#[derive(Debug, Clone)]
pub struct Checkpoint<T> {
    /// Index of the macro step that starts here.
    pub macro_index: usize,
    pub t: f64,
    pub u: Vec<T>,
    /// Per leaf, in schedule order.
    pub leaves: Vec<LeafPersist>,
}

impl<T: Scalar> Checkpoint<T> {
    /// Bitwise equality of state and leaf states.
    pub fn bitwise_eq(&self, o: &Checkpoint<T>) -> bool {
        self.macro_index == o.macro_index
            && self.t.to_bits() == o.t.to_bits()
            && self.u.len() == o.u.len()
            && self
                .u
                .iter()
                .zip(&o.u)
                .all(|(a, b)| a.to_bits_u64() == b.to_bits_u64())
            && self.leaves.len() == o.leaves.len()
            && self
                .leaves
                .iter()
                .zip(&o.leaves)
                .all(|(a, b)| a.bitwise_eq(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_round_trip_is_bitwise() {
        let tape = Tape {
            records: vec![
                TapeRecord::Leaf {
                    path: "0/root.0".into(),
                    t: 0.0,
                    dt: 0.1,
                    steps: vec![[0.0, 0.1 / 3.0], [0.1 / 3.0, 0.1 - 0.1 / 3.0]],
                    accepted: 2,
                    rejected: 1,
                    retcode: Retcode::Success,
                    clips: vec![(1, vec![0, 3])],
                },
                TapeRecord::Map {
                    path: "0/root.1".into(),
                    t: 0.0,
                    dt: 0.1,
                    mask: vec![],
                },
                TapeRecord::Subcycle {
                    path: "0/root.2".into(),
                    t: 0.0,
                    dt: 0.1,
                    n: 4,
                },
                TapeRecord::Macro {
                    path: "0".into(),
                    t: 0.0,
                    dt: 0.1,
                    reason: "step.dt".into(),
                },
            ],
        };
        let s = tape.to_jsonl();
        assert!(s.starts_with("{\"node\":\"leaf\",\"path\":\"0/root.0\""));
        // spec/tape.md §2: [[k, [i, ...]], ...], omitted when nothing clipped.
        assert!(s.lines().next().unwrap().contains("\"clips\":[[1,[0,3]]]"));
        let no_clip = TapeRecord::Leaf {
            path: "0/root".into(),
            t: 0.0,
            dt: 1.0,
            steps: vec![[0.0, 1.0]],
            accepted: 1,
            rejected: 0,
            retcode: Retcode::Success,
            clips: vec![],
        };
        assert!(!serde_json::to_string(&no_clip).unwrap().contains("clips"));
        let back = Tape::from_jsonl(&s).unwrap();
        assert_eq!(back, tape);
        for (a, b) in back.records.iter().zip(&tape.records) {
            assert!(a.bitwise_eq(b));
        }
        assert_eq!(tape.macro_segment(0).len(), 4);
        assert_eq!(tape.step_counts(), (2, 1));
    }

    #[test]
    fn non_finite_numbers_round_trip() {
        let tape = Tape {
            records: vec![
                TapeRecord::Leaf {
                    path: "0/root".into(),
                    t: 0.0,
                    dt: f64::INFINITY,
                    steps: vec![[0.0, f64::NAN], [-0.0, f64::NEG_INFINITY]],
                    accepted: 2,
                    rejected: 0,
                    retcode: Retcode::DtNaN,
                    clips: vec![],
                },
                TapeRecord::Macro {
                    path: "0".into(),
                    t: f64::NAN,
                    dt: f64::NEG_INFINITY,
                    reason: "step.dt".into(),
                },
            ],
        };
        let s = tape.to_jsonl();
        assert!(
            s.contains(r#""steps":[[0.0,"NaN"],[-0.0,"-Infinity"]]"#),
            "{s}"
        );
        assert!(s.contains(r#""dt":"Infinity""#), "{s}");
        let back = Tape::from_jsonl(&s).unwrap();
        for (a, b) in back.records.iter().zip(&tape.records) {
            assert!(a.bitwise_eq(b), "{a:?} {b:?}");
            assert!(a.bitwise_eq(a));
        }
        // Bits, not values: 0.0 and -0.0 differ.
        let mut other = tape.records[0].clone();
        if let TapeRecord::Leaf { steps, .. } = &mut other {
            steps[1][0] = 0.0;
        }
        assert!(!other.bitwise_eq(&tape.records[0]));
        assert!(
            Tape::from_jsonl(r#"{"node":"map","path":"0/m","t":"nan","dt":1,"mask":[]}"#).is_err()
        );
    }
}
