//! The EarthSciASTDiff seam (spec/plan.md §5.2.2, §5.4): a derived
//! document loaded and flattened by EarthSciAST and viewed with
//! EarthSciASTDiff's `sysview_flat`, its band list from
//! `jacobian_bands_view`, and the [`BandSource`] that gives every term its
//! bands from one probe document per batch of terms, searching halves of a
//! batch for the failing term when its probe fails.

use crate::derived;
use crate::locality::{BandEntry, BandFailure, BandSource, BandSourceError, RowIdx};
use crate::model::ModelInfo;
use crate::terms::Term;
use earthsci_ast::{Expr, ExpressionNode, FlattenedSystem};
use earthsci_astdiff::{BandError, JacEntry, RIdx, StructureEntry, SysView, Wrt};
use serde_json::Value;
use std::collections::HashMap;
use std::ops::Range;

/// A derived document, loaded and flattened by EarthSciAST.
#[derive(Debug, Clone)]
pub struct FlatView {
    /// The flattened document.
    pub flat: FlattenedSystem,
}

impl FlatView {
    /// Loads the document whose bytes are `text` and flattens it.
    pub fn new(text: &str) -> Result<FlatView, String> {
        let loaded =
            earthsci_ast::load_string(text).map_err(|e| format!("EarthSciAST load: {e}"))?;
        let flat =
            earthsci_ast::flatten(&loaded).map_err(|e| format!("EarthSciAST flatten: {e}"))?;
        Ok(FlatView { flat })
    }

    /// The document `doc`, written as its bytes, loaded and flattened.
    pub fn of_document(doc: &Value) -> Result<FlatView, String> {
        let text = String::from_utf8(derived::to_bytes(doc)).map_err(|e| e.to_string())?;
        FlatView::new(&text)
    }

    /// EarthSciASTDiff's view of the flattened system (`sysview_flat`),
    /// which checks each lifted state's shape. It is not free: build it
    /// once and pass it to [`bands`] and the passes that read it.
    pub fn sysview(&self) -> Result<SysView<'_>, BandError> {
        earthsci_astdiff::sysview_flat(&self.flat)
    }
}

/// The state Jacobian's band list of the view `sv` (`jacobian_bands_view`,
/// `wrt = states`), in EarthSciASTDiff's order.
pub fn bands(sv: &SysView) -> Result<Vec<JacEntry>, BandError> {
    earthsci_astdiff::jacobian_bands_view(sv, Wrt::States, None)
}

/// The state Jacobian's structure-only entries of the view `sv`
/// (`jacobian_structure_view`, `wrt = states`; esm-jacobian-spec §2.1):
/// the pattern pass 2 reads (§5.2.2), without coefficients.
pub fn structure(sv: &SysView) -> Result<Vec<StructureEntry>, BandError> {
    earthsci_astdiff::jacobian_structure_view(sv, Wrt::States, None)
}

fn row_idx(ridx: &[RIdx]) -> Vec<RowIdx> {
    ridx.iter()
        .map(|r| match r {
            RIdx::Name(n) => RowIdx::Name(n.clone()),
            RIdx::Fixed(i) => RowIdx::Fixed(*i),
        })
        .collect()
}

/// The [`BandEntry`] of the band with row `u` and column `v` whose fields
/// are the rest: the one conversion that [`band_entry`] and
/// [`structure_entry`] share, so the two cannot drift apart.
fn entry(
    u: &str,
    v: &str,
    rows: &[(i64, i64)],
    ridx: &[RIdx],
    cidx: &[Expr],
    contracted: &[(String, i64, i64)],
) -> BandEntry {
    BandEntry {
        row: u.to_string(),
        col: v.to_string(),
        rows: rows.to_vec(),
        row_idx: row_idx(ridx),
        col_idx: cidx.to_vec(),
        contracted: contracted.to_vec(),
    }
}

/// A band entry as the locality rules read it (§5.2.3).
pub fn band_entry(en: &JacEntry) -> BandEntry {
    let b = &en.band;
    entry(&en.u, &en.v, &b.rows, &b.ridx, &b.cidx, &b.contracted)
}

/// A structure-only entry as the locality rules read it (§5.2.3).
pub fn structure_entry(en: &StructureEntry) -> BandEntry {
    let b = &en.band;
    entry(&en.u, &en.v, &b.rows, &b.ridx, &b.cidx, &b.contracted)
}

/// `read` of the view of the document `doc`, loaded and flattened. A
/// failure anywhere is a message for `band_model_failed`.
fn document_view<T>(
    doc: &Value,
    read: impl FnOnce(&SysView) -> Result<T, BandError>,
) -> Result<T, String> {
    let view = FlatView::of_document(doc)?;
    let sv = view.sysview().map_err(astdiff_message)?;
    read(&sv).map_err(astdiff_message)
}

/// The band list of the document `doc` (§5.2.2, §5.4): loaded, flattened,
/// viewed and differentiated. A failure anywhere is a message for
/// `band_model_failed`.
pub fn document_bands(doc: &Value) -> Result<Vec<JacEntry>, String> {
    document_view(doc, bands)
}

/// The structure-only entries of the document `doc` (§5.2.2): loaded,
/// flattened, viewed and read for its pattern. A failure anywhere is a
/// message for `band_model_failed`.
pub fn document_structure(doc: &Value) -> Result<Vec<StructureEntry>, String> {
    document_view(doc, structure)
}

/// Which of EarthSciASTDiff's state Jacobians a search reads: the
/// structure-only entries pass 2 reads (§5.2.2), or the full band list
/// pass 4 reads (§5.4), whose failures pass 2 cannot see.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// `jacobian_structure_view`: no derivative, no coefficient.
    Structure,
    /// `jacobian_bands_view`: every derivative, with its coefficient.
    Full,
}

/// The entries of the document `doc` in `mode`, as the locality rules
/// read them; a full band list's coefficients are dropped here.
fn document_entries(doc: &Value, mode: Mode) -> Result<Vec<BandEntry>, String> {
    match mode {
        Mode::Structure => Ok(document_structure(doc)?
            .iter()
            .map(structure_entry)
            .collect()),
        Mode::Full => Ok(document_bands(doc)?.iter().map(band_entry).collect()),
    }
}

/// An EarthSciASTDiff failure as a `band_model_failed` message.
pub fn astdiff_message(e: BandError) -> String {
    format!("EarthSciASTDiff: {}", e.0)
}

/// The most operator nodes, every occurrence counted, that a probe batch's
/// terms hold in all, unless the batch is a single term (§5.2.2). A term
/// with more (a large term) is therefore a batch by itself.
pub const LARGE_TERM_NODES: u64 = 10_000;

/// The number of operator nodes of `e`, every occurrence counted, in every
/// expression-bearing child, or `limit + 1` when it exceeds `limit`. A node
/// reached through several parents is counted once per path but walked
/// once, and the walk stops as soon as the count passes `limit`.
pub fn operator_nodes(e: &Expr, limit: u64) -> u64 {
    capped_nodes(e, &mut HashMap::new(), limit.saturating_add(1))
}

/// `min(operator nodes of e, cap)`, memoized by node in `memo`. Every value
/// in `memo` is capped at the same `cap`, so one memo can be shared across
/// many calls with that cap. A node's walk stops once its count reaches
/// `cap`; its memoized value is then `cap`, which is still exact.
fn capped_nodes(e: &Expr, memo: &mut HashMap<*const ExpressionNode, u64>, cap: u64) -> u64 {
    let Expr::Operator(node) = e else {
        return 0;
    };
    let key = std::sync::Arc::as_ptr(node);
    if let Some(&n) = memo.get(&key) {
        return n;
    }
    let mut c = 1u64.min(cap);
    if c < cap {
        node.any_child(&mut |x| {
            c = (c + capped_nodes(x, memo, cap)).min(cap);
            c >= cap
        });
    }
    memo.insert(key, c);
    c
}

/// §5.2.2's batches: consecutive runs of `terms`, in term order, whose
/// terms hold at most `large` operator nodes in all, unless the run is a
/// single term. A batch ends just before a term that would take its total
/// over `large`, so a large term (more than `large` operator nodes) is a
/// batch by itself.
pub fn probe_batches(terms: &[Term], large: u64) -> Vec<Range<usize>> {
    // One memo for every term: the interner shares subtrees between terms.
    let (cap, mut memo) = (large.saturating_add(1), HashMap::new());
    let mut out = Vec::new();
    // Each count is capped at `large + 1`, which decides `total + n > large`
    // as the uncapped count would.
    let (mut start, mut total) = (0, 0u64);
    for (k, t) in terms.iter().enumerate() {
        let n = capped_nodes(&t.expr, &mut memo, cap);
        if k > start && total.saturating_add(n) > large {
            out.push(start..k);
            (start, total) = (k, 0);
        }
        total = total.saturating_add(n);
    }
    if !terms.is_empty() {
        out.push(start..terms.len());
    }
    out
}

/// Term bands from EarthSciASTDiff (§5.2.2): each term's band list is that
/// of the right-hand-side document of a hypothetical operator holding only
/// that term. They are computed in batches ([`probe_batches`]), one probe
/// document per batch ([`derived::probe_document`]): one EarthSciAST load
/// and flatten and one call for EarthSciASTDiff's structure-only entries
/// (`jacobian_structure_view`, which take no derivative and carry no
/// coefficient), whose entries with row `probe_state(k)` are the batch's
/// term `k`'s, row renamed to its variable. Each batch's bands are kept as
/// [`BandEntry`] values, and its probe is dropped before the next batch is
/// built, so only one batch's probe is held at a time: of one large term,
/// or of terms with at most [`LARGE_TERM_NODES`] operator nodes in all. A
/// probe that succeeds gives the batch's bands. A failure on it names no
/// term, so the batch is then split in halves and each half probed in
/// turn, first half first, down to single terms, whose own documents'
/// structure is read: the first term whose own document fails, in term
/// order, is `band_model_failed`, with that document's message. The
/// batches are consecutive and taken in order, so that is the first such
/// term of the whole list. Within a failing batch of `n` terms, finding it
/// takes about `2 log₂ n` probe documents rather than one document per
/// term, on top of one probe per batch.
///
/// This search reads structure only, so it cannot find a failure only a
/// derivative shows (`underivable_operator`, such as `erf`). Such a term
/// fails its operator's document in pass 4, where `plan` finds it with
/// [`first_failing_term`]: the same search in [`Mode::Full`].
pub struct AstDiffBands;

/// A probe's band lists, or EarthSciAST's or EarthSciASTDiff's message.
type Probe = std::result::Result<Vec<Vec<BandEntry>>, String>;

impl AstDiffBands {
    /// The probe document's band lists in `mode`, or EarthSciAST's or
    /// EarthSciASTDiff's message when it fails on it. A band whose row is
    /// not a probe state is the planner's own error.
    fn probe(info: &ModelInfo, terms: &[Term], mode: Mode) -> Result<Probe, BandSourceError> {
        Self::probe_with(info, terms, derived::PROBE_TEMPLATE_MIN_NODES, mode)
    }

    /// [`Self::probe`] of the probe document whose templates have at least
    /// `min_nodes` operator nodes ([`derived::probe_document_with`]).
    fn probe_with(
        info: &ModelInfo,
        terms: &[Term],
        min_nodes: u64,
        mode: Mode,
    ) -> Result<Probe, BandSourceError> {
        let entries =
            match document_entries(&derived::probe_document_with(info, terms, min_nodes), mode) {
                Ok(e) => e,
                Err(message) => return Ok(Err(message)),
            };
        let index: HashMap<String, usize> = (0..terms.len())
            .map(|k| (derived::probe_state(k), k))
            .collect();
        let mut out = vec![Vec::new(); terms.len()];
        for mut b in entries {
            // Only the probe states have nonzero right-hand sides.
            let k = *index.get(&b.row).ok_or_else(|| {
                BandSourceError::Internal(format!(
                    "the probe document (spec/plan.md §5.2.2) gives a band with row {}, which is not a probe state",
                    b.row
                ))
            })?;
            b.row = terms[k].variable.clone();
            out[k].push(b);
        }
        Ok(Ok(out))
    }

    /// Term `t`'s band list in `mode` from its own right-hand-side
    /// document; a failure is `band_model_failed` for the term at position
    /// `i`. In [`Mode::Full`] only EarthSciASTDiff's failures are: an
    /// EarthSciAST load or flatten failure is not the term's derivative
    /// failing, and is [`BandSourceError::Internal`], which ends pass 4's
    /// search so that the operator is refused instead (§5.2.2).
    fn own(
        info: &ModelInfo,
        t: &Term,
        i: usize,
        mode: Mode,
    ) -> Result<Vec<BandEntry>, BandSourceError> {
        let doc = derived::rhs_document(info, "term", &[t]);
        let failed = |message| BandSourceError::Failed(BandFailure { term: i, message });
        match mode {
            Mode::Structure => document_entries(&doc, mode).map_err(failed),
            Mode::Full => {
                let view = FlatView::of_document(&doc).map_err(BandSourceError::Internal)?;
                let sv = view.sysview().map_err(|e| failed(astdiff_message(e)))?;
                let list = bands(&sv).map_err(|e| failed(astdiff_message(e)))?;
                Ok(list.iter().map(band_entry).collect())
            }
        }
    }

    /// The band lists of `terms` in `mode`, the terms from position
    /// `first` on, into `out`: one probe, or when it fails each half in
    /// turn, down to single terms' own documents.
    fn search(
        info: &ModelInfo,
        terms: &[Term],
        first: usize,
        mode: Mode,
        out: &mut Vec<Vec<BandEntry>>,
    ) -> Result<(), BandSourceError> {
        match terms {
            [] => return Ok(()),
            [t] => {
                out.push(Self::own(info, t, first, mode)?);
                return Ok(());
            }
            _ => {}
        }
        if let Ok(bands) = Self::probe(info, terms, mode)? {
            out.extend(bands);
            return Ok(());
        }
        let (a, b) = terms.split_at(terms.len() / 2);
        Self::search(info, a, first, mode, out)?;
        Self::search(info, b, first + a.len(), mode, out)
    }

    /// Every term's band list in `mode`, one batch after another
    /// ([`probe_batches`] with bound `large`).
    fn batched(
        info: &ModelInfo,
        terms: &[Term],
        large: u64,
        mode: Mode,
    ) -> Result<Vec<Vec<BandEntry>>, BandSourceError> {
        Self::in_ranges(info, terms, probe_batches(terms, large), mode)
    }

    /// Every term's band list in `mode`, one batch after another, the
    /// batches being `batches`: consecutive ranges covering `terms` in order.
    fn in_ranges(
        info: &ModelInfo,
        terms: &[Term],
        batches: Vec<Range<usize>>,
        mode: Mode,
    ) -> Result<Vec<Vec<BandEntry>>, BandSourceError> {
        let mut out = Vec::with_capacity(terms.len());
        for r in batches {
            let first = r.start;
            Self::search(info, &terms[r], first, mode, &mut out)?;
        }
        Ok(out)
    }
}

/// The first of `terms`, in their order, whose own right-hand-side
/// document fails in EarthSciASTDiff's full mode, with that document's
/// message, or `None` when none does or when the search meets a failure
/// that is not EarthSciASTDiff's, such as an EarthSciAST load or flatten
/// failure on a term's own document (§5.2.2, §4 "Order of refusals" item
/// 4). `plan` calls this on an operator's terms when the operator's own
/// document fails in pass 4, and on `None` refuses against the operator
/// with the operator's own message. It is [`AstDiffBands`]' halving search in
/// [`Mode::Full`], in the same batches, so it takes about `2 log₂ n` probe
/// documents rather than one full differentiation per term; each batch's
/// coefficients are dropped before the next batch is probed.
pub fn first_failing_term(info: &ModelInfo, terms: &[Term]) -> Option<BandFailure> {
    match AstDiffBands::batched(info, terms, LARGE_TERM_NODES, Mode::Full) {
        Err(BandSourceError::Failed(f)) => Some(f),
        Ok(_) | Err(BandSourceError::Internal(_)) => None,
    }
}

/// Test access to the parts of [`AstDiffBands`], not part of the planner's
/// interface.
#[doc(hidden)]
pub mod testing {
    use super::*;

    /// The probe document's structure-only band lists, or its failure's
    /// message.
    pub fn probe(info: &ModelInfo, terms: &[Term]) -> Result<Probe, BandSourceError> {
        AstDiffBands::probe(info, terms, Mode::Structure)
    }

    /// The structure-only band lists of the probe document whose templates
    /// have at least `min_nodes` operator nodes, or its failure's message.
    pub fn probe_with(
        info: &ModelInfo,
        terms: &[Term],
        min_nodes: u64,
    ) -> Result<Probe, BandSourceError> {
        AstDiffBands::probe_with(info, terms, min_nodes, Mode::Structure)
    }

    /// Every term's band list in `mode`, in batches whose terms hold at
    /// most `large` operator nodes in all, unless a batch is a single term;
    /// the first failure is the failing term's.
    pub fn in_batches(
        info: &ModelInfo,
        terms: &[Term],
        large: u64,
        mode: Mode,
    ) -> Result<Vec<Vec<BandEntry>>, BandSourceError> {
        AstDiffBands::batched(info, terms, large, mode)
    }

    /// Every term's band list in `mode`, in the given batches: consecutive
    /// ranges covering `terms` in order. The first failure is the failing
    /// term's.
    pub fn in_given_batches(
        info: &ModelInfo,
        terms: &[Term],
        batches: Vec<Range<usize>>,
        mode: Mode,
    ) -> Result<Vec<Vec<BandEntry>>, BandSourceError> {
        AstDiffBands::in_ranges(info, terms, batches, mode)
    }

    /// Each term's band list in `mode` from its own right-hand-side
    /// document, in term order; the first failure is the failing term's.
    pub fn each_term(
        info: &ModelInfo,
        terms: &[Term],
        mode: Mode,
    ) -> Result<Vec<Vec<BandEntry>>, BandSourceError> {
        terms
            .iter()
            .enumerate()
            .map(|(i, t)| AstDiffBands::own(info, t, i, mode))
            .collect()
    }
}

impl BandSource for AstDiffBands {
    fn term_bands(
        &self,
        info: &ModelInfo,
        terms: &[Term],
    ) -> Result<Option<Vec<Vec<BandEntry>>>, BandSourceError> {
        Self::batched(info, terms, LARGE_TERM_NODES, Mode::Structure).map(Some)
    }
}
