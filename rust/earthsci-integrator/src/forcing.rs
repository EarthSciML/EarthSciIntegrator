//! The runtime's data layer (spec/operator-interface.md §8.4, D87).
//!
//! A plan's operators read data-fed parameters (spec/plan.md §4.1) through
//! forcing sets. The runtime writes no data layer of its own: it loads the
//! plan's model with EarthSciAST, builds one provider per data-fed parameter
//! some operator reads with EarthSciAST's `providers_from_document` (on
//! EarthSciIO), samples them with EarthSciAST's refresh executor and turns
//! each sample into a slot's buffer with `forcing_from_sample` (§8.4.2).
//!
//! The forcing epochs (§8.4.3) start at `t0` and at every refresh time of a
//! discrete provider inside `(t0, t_f]`. Every epoch is sampled when the
//! plan is loaded, in order on the calling thread, which is before the first
//! call whose `t` reaches it; EarthSciIO's prefetch may replace this without
//! changing a result. A call takes the set of the epoch its `t` falls in
//! (§8.4.1); a stage at exactly a boundary reads the new epoch for a
//! two-record parameter and the old one for a one-record parameter, and a
//! step that ends on a boundary does not lend its last stage to the next
//! (the leaf's `no_reuse`).

use earthsci_ast::esio_provider::{providers_from_document_with, DocumentProviderOptions};
use earthsci_ast::provider::{CadenceProvider, ForcingBuffer, NativeField, RefreshExecutor};
use earthsci_ast::{forcing_from_sample, ForcingSlot};
use earthsci_integrator_planner::numeric::Forcing;
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

thread_local! {
    /// The start time of the step the calling thread is taking, NaN outside
    /// a step: a stage at exactly a refresh time that starts the step takes
    /// the new epoch, one that ends it the boundary set (§8.4.1).
    static STEP_START: Cell<f64> = const { Cell::new(f64::NAN) };
}

/// Marks the calling thread as taking a step from `t` until the guard is
/// dropped; the leaf holds one around each step it takes.
pub struct StepGuard {
    prev: f64,
}

impl StepGuard {
    pub fn new(t: f64) -> StepGuard {
        StepGuard {
            prev: STEP_START.with(|c| c.replace(t)),
        }
    }
}

impl Drop for StepGuard {
    fn drop(&mut self) {
        STEP_START.with(|c| c.set(self.prev));
    }
}

/// The run-time options of the data layer (spec/operator-interface.md
/// §8.4.2 item 1): in neither the run config nor the plan, since none
/// changes what is computed. EarthSciIO's offline mode is its own
/// `EARTHSCI_OFFLINE` environment variable.
#[derive(Debug, Clone, Default)]
pub struct DataOptions {
    /// The plan's model (`inputs.model`), which the plan names relative to
    /// its run config's directory. Needed when the plan has `forcing`; its
    /// hash must be the plan's.
    pub model: Option<PathBuf>,
    /// EarthSciIO's cache directory; default its data directory.
    pub cache_dir: Option<PathBuf>,
    /// Source name to replacement URL template (EarthSciAST's
    /// `url_overrides`).
    pub url_overrides: HashMap<String, String>,
}

/// One data-fed parameter of the plan's `forcing` (spec/plan.md §3).
#[derive(Debug, Clone, PartialEq)]
pub struct PlanForcing {
    pub shape: Vec<String>,
    pub source: String,
    pub file_variable: String,
    pub cadence: String,
    pub records_per_sample: u64,
}

/// The plan's `forcing`, by flattened name; empty when it has none.
pub fn plan_forcing(plan: &serde_json::Value) -> Result<BTreeMap<String, PlanForcing>, String> {
    let mut out = BTreeMap::new();
    for (name, f) in plan["forcing"].as_object().into_iter().flatten() {
        let s = |k: &str| {
            f[k].as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("plan.json: forcing.{name} has no {k}"))
        };
        out.insert(
            name.clone(),
            PlanForcing {
                shape: f["shape"]
                    .as_array()
                    .ok_or_else(|| format!("plan.json: forcing.{name} has no shape"))?
                    .iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect(),
                source: s("source")?,
                file_variable: s("file_variable")?,
                cadence: s("cadence")?,
                records_per_sample: f["records_per_sample"].as_u64().unwrap_or(1),
            },
        );
    }
    Ok(out)
}

/// The forcing epochs of a run (§8.4.3) and each epoch's buffers.
#[derive(Debug, Clone)]
pub struct ForcingTimeline {
    /// `B_0 = t0`, then the refresh times inside `(t0, t_f]`, ascending.
    boundaries: Vec<f64>,
    /// Epoch `k`'s buffers, for every data-fed parameter some operator
    /// reads.
    epochs: Vec<Forcing>,
    /// For `k >= 1`, the buffers a stage at exactly `B_k` takes (§8.4.1):
    /// epoch `k`'s for a parameter with two records per sample, whose
    /// time blend is continuous there only with the new bracket, and epoch
    /// `k - 1`'s for one with one record, the limit from the left of its
    /// piecewise-constant values. Entry 0 is epoch 0's.
    at_boundary: Vec<Forcing>,
}

impl ForcingTimeline {
    /// No data: one empty epoch.
    pub fn empty(t0: f64) -> ForcingTimeline {
        ForcingTimeline {
            boundaries: vec![t0],
            epochs: vec![Forcing::empty()],
            at_boundary: vec![Forcing::empty()],
        }
    }

    /// The forcing a call at `t` takes (§8.4.1): epoch `k` with
    /// `B_k <= t < B_{k+1}`, the last epoch for `t` at or after its start
    /// and the first for `t` before `t0`; except that a stage at exactly a
    /// later boundary `B_k` of a step that does not start there (a stage
    /// at the right end of the step that ends on it) takes each parameter's
    /// buffer by its records per sample (`at_boundary`).
    pub fn at(&self, t: f64) -> &Forcing {
        let k = self
            .boundaries
            .partition_point(|&b| b <= t)
            .saturating_sub(1);
        let starts_here = STEP_START.with(|c| c.get()) == t;
        if k >= 1 && self.boundaries[k] == t && !starts_here {
            &self.at_boundary[k]
        } else {
            &self.epochs[k]
        }
    }

    /// Builds the boundary sets from the epochs: a parameter in `two` (two
    /// records per sample) takes the new epoch's buffer, any other the old
    /// one's.
    fn with_boundaries(
        boundaries: Vec<f64>,
        epochs: Vec<Forcing>,
        two: &BTreeSet<String>,
    ) -> ForcingTimeline {
        let mut at_boundary = vec![epochs[0].clone()];
        for k in 1..epochs.len() {
            let bufs = epochs[k]
                .bufs()
                .iter()
                .map(|(name, new)| {
                    let b = if two.contains(name) {
                        new.clone()
                    } else {
                        epochs[k - 1].bufs()[name].clone()
                    };
                    (name.clone(), b)
                })
                .collect();
            at_boundary.push(Forcing::new(bufs));
        }
        ForcingTimeline {
            boundaries,
            epochs,
            at_boundary,
        }
    }

    /// The epoch starts after `t0`: the `forcing_refresh` tstops
    /// (spec/run-config.md §5).
    pub fn refresh_times(&self) -> &[f64] {
        &self.boundaries[1..]
    }

    /// Every epoch's start, `t0` first.
    pub fn boundaries(&self) -> &[f64] {
        &self.boundaries
    }

    /// Epoch `k`'s buffers.
    pub fn epoch(&self, k: usize) -> &Forcing {
        &self.epochs[k]
    }
}

/// The plan's model loaded for its data (§8.4.2 item 1).
pub struct ModelData {
    pub path: PathBuf,
    pub file: earthsci_ast::EsmFile,
    /// The loaded document as JSON, which `providers_from_document` reads.
    doc: serde_json::Value,
    /// The model's time origin, in Unix seconds (D74).
    pub origin: f64,
}

impl ModelData {
    /// Loads the model at `path` with EarthSciAST.
    pub fn load(path: &Path) -> Result<ModelData, String> {
        let file = earthsci_ast::load_path(path)
            .map_err(|e| format!("EarthSciAST load of {}: {e}", path.display()))?;
        let doc = serde_json::to_value(&file)
            .map_err(|e| format!("serializing the loaded model: {e}"))?;
        let origin = earthsci_ast::time_origin(&file)
            .map_err(|e| format!("the model's time origin: {e}"))?;
        Ok(ModelData {
            path: path.to_path_buf(),
            file,
            doc,
            origin,
        })
    }

    /// One provider per parameter of `names`, keyed by its flattened name,
    /// built by EarthSciAST's `providers_from_document` from the model's own
    /// declarations, restricted to the sources `names` read, over the run
    /// window `tspan` (model seconds). A parameter it gives no provider for
    /// is an error.
    pub fn providers(
        &self,
        names: &BTreeMap<String, PlanForcing>,
        tspan: (f64, f64),
        opts: &DataOptions,
    ) -> Result<HashMap<String, Box<dyn CadenceProvider>>, String> {
        if names.is_empty() {
            return Ok(HashMap::new());
        }
        let sources: BTreeSet<String> = names.values().map(|f| f.source.clone()).collect();
        let at = |s: f64| {
            let ns = ((self.origin + s) * 1e9).round() as i128;
            time::OffsetDateTime::from_unix_timestamp_nanos(ns)
                .map_err(|e| format!("run window bound {s} s: {e}"))
        };
        let cache = opts.cache_dir.clone().unwrap_or_else(earthsciio::data_dir);
        let built = providers_from_document_with(
            &self.doc,
            &cache,
            &DocumentProviderOptions {
                loaders: Some(sources.into_iter().collect()),
                url_overrides: opts.url_overrides.clone(),
                // EarthSciIO's window is half-open, so a refresh time at
                // exactly t_f would be left out; §8.4.3 counts it (the last
                // epoch is [t_f, t_f]). The end is moved past t_f by a
                // millisecond, less than any cadence, and refresh times
                // after t_f are dropped by the caller.
                window: Some((at(tspan.0)?, at(tspan.1 + 1e-3)?)),
                ..Default::default()
            },
        )
        .map_err(|e| format!("EarthSciAST providers_from_document: {}", e.0))?;
        let mut out: HashMap<String, Box<dyn CadenceProvider>> = HashMap::new();
        for (key, p) in built {
            if names.contains_key(&key) {
                out.insert(key, Box::new(p));
            }
        }
        if let Some(missing) = names.keys().find(|n| !out.contains_key(*n)) {
            return Err(format!(
                "providers_from_document built no provider for the data-fed parameter {missing} (source {})",
                names[missing].source
            ));
        }
        Ok(out)
    }

    /// The forcing epochs of a run over `tspan` for the parameters `read`
    /// (those some operator reads), each turned into its slot's buffer
    /// (§8.4.2 item 3, §8.4.3).
    pub fn timeline(
        &self,
        read: &BTreeMap<String, PlanForcing>,
        slots: &BTreeMap<String, ForcingSlot>,
        tspan: (f64, f64),
        opts: &DataOptions,
    ) -> Result<ForcingTimeline, String> {
        if read.is_empty() {
            return Ok(ForcingTimeline::empty(tspan.0));
        }
        let mut exec = RefreshExecutor::from_providers(self.providers(read, tspan, opts)?);
        let buf: ForcingBuffer = Rc::new(RefCell::new(HashMap::new()));
        let bufs = |buf: &ForcingBuffer, t: f64| -> Result<Forcing, String> {
            let held = buf.borrow();
            let mut out = BTreeMap::new();
            for name in read.keys() {
                let slot = slots
                    .get(name)
                    .ok_or_else(|| format!("no operator program has a forcing slot {name}"))?;
                let a = held.get(name).ok_or_else(|| {
                    format!(
                        "the data-fed parameter {name} (source {}) has no record in effect at t = {t}",
                        read[name].source
                    )
                })?;
                let sample = HashMap::from([(name.clone(), NativeField::new(a.clone()))]);
                let b: Arc<[f64]> = forcing_from_sample(&sample, slot)
                    .map_err(|e| format!("the data-fed parameter {name} at t = {t}: {e}"))?;
                out.insert(name.clone(), b);
            }
            Ok(Forcing::new(out))
        };
        exec.materialize_const(&buf)
            .map_err(|e| format!("reading const data: {e}"))?;
        exec.sample_at(tspan.0, &buf)
            .map_err(|e| format!("sampling the data at t0 = {}: {e}", tspan.0))?;
        let mut boundaries = vec![tspan.0];
        let mut epochs = vec![bufs(&buf, tspan.0)?];
        for t in exec.refresh_times() {
            if t <= tspan.0 || t > tspan.1 {
                continue;
            }
            exec.refresh_at(t, &buf)
                .map_err(|e| format!("refreshing the data at t = {t}: {e}"))?;
            boundaries.push(t);
            epochs.push(bufs(&buf, t)?);
        }
        let two: BTreeSet<String> = read
            .iter()
            .filter(|(_, f)| f.records_per_sample == 2)
            .map(|(n, _)| n.clone())
            .collect();
        Ok(ForcingTimeline::with_boundaries(boundaries, epochs, &two))
    }

    /// EarthSciAST's Problem of the model over `tspan` with the providers of
    /// `names` (§8.4.2 item 3, §8.4.5), as EarthSciAST's own solve of the
    /// model builds it.
    pub fn problem(
        &self,
        names: &BTreeMap<String, PlanForcing>,
        tspan: (f64, f64),
        opts: &DataOptions,
    ) -> Result<earthsci_ast::EsmProblem, String> {
        earthsci_ast::esm_problem(
            earthsci_ast::ProblemInput::File(&self.file),
            tspan,
            earthsci_ast::ProblemOptions {
                base_path: self.path.parent().map(Path::to_path_buf),
                providers: self.providers(names, tspan, opts)?,
                ..Default::default()
            },
        )
        .map_err(|e| format!("EarthSciAST esm_problem of the model: {e}"))
    }

    /// Whether an `ic` equation of the model reads one of `names`, directly
    /// or through observed definitions (§8.4.5).
    pub fn ic_reads(&self, names: &BTreeMap<String, PlanForcing>) -> Result<bool, String> {
        let flat = earthsci_ast::flatten(&self.file)
            .map_err(|e| format!("EarthSciAST flatten of the model: {e}"))?;
        let reads = earthsci_integrator_planner::model::ic_reads(&flat);
        Ok(names.keys().any(|n| reads.contains(n)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §8.4.1: a call takes the epoch its `t` falls in; a stage at exactly
    /// a later boundary takes the new epoch for a two-record parameter and
    /// the old one for a one-record parameter; `t_f`'s epoch is the last.
    #[test]
    fn epoch_selection() {
        let f = |x: f64| {
            Forcing::new(BTreeMap::from([
                ("P".to_string(), Arc::from(vec![x])),
                ("Q".to_string(), Arc::from(vec![-x])),
            ]))
        };
        let tl = ForcingTimeline::with_boundaries(
            vec![0.0, 10.0, 20.0],
            vec![f(0.0), f(1.0), f(2.0)],
            &BTreeSet::from(["P".to_string()]),
        );
        let at = |t: f64| (tl.at(t).bufs()["P"][0], tl.at(t).bufs()["Q"][0]);
        assert_eq!(at(-1.0), (0.0, 0.0));
        assert_eq!(at(0.0), (0.0, 0.0));
        assert_eq!(at(9.999), (0.0, 0.0));
        assert_eq!(at(10.0), (1.0, 0.0));
        assert_eq!(at(10.001), (1.0, -1.0));
        assert_eq!(at(19.0), (1.0, -1.0));
        assert_eq!(at(20.0), (2.0, -1.0));
        // The first stage of a step that starts on a boundary.
        {
            let _g = StepGuard::new(10.0);
            assert_eq!(at(10.0), (1.0, -1.0));
        }
        assert_eq!(at(10.0), (1.0, 0.0));
        assert_eq!(tl.refresh_times(), [10.0, 20.0]);
        assert_eq!(ForcingTimeline::empty(3.0).refresh_times(), [] as [f64; 0]);
    }
}
