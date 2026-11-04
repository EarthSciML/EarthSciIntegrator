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

/// A term is large when its expression has more than this many operator
/// nodes, every occurrence counted (§5.2.2): a probe holds at most one
/// large term.
pub const LARGE_TERM_NODES: u64 = 10_000;

/// The number of operator nodes of `e`, every occurrence counted, in every
/// expression-bearing child, or `limit + 1` when it exceeds `limit`. A node
/// reached through several parents is counted once per path but walked
/// once.
pub fn operator_nodes(e: &Expr, limit: u64) -> u64 {
    fn walk(e: &Expr, memo: &mut HashMap<*const ExpressionNode, u64>, cap: u64) -> u64 {
        let Expr::Operator(node) = e else {
            return 0;
        };
        let key = std::sync::Arc::as_ptr(node);
        if let Some(&n) = memo.get(&key) {
            return n;
        }
        let mut c = 1u64;
        node.for_each_child(&mut |x| c = (c + walk(x, memo, cap)).min(cap));
        memo.insert(key, c);
        c
    }
    let cap = limit.saturating_add(1);
    walk(e, &mut HashMap::new(), cap)
}

/// §5.2.2's batches: consecutive runs of `terms`, in term order, each
/// holding at most one large term (more than `large` operator nodes). A
/// batch ends just before a large term when it already holds one.
pub fn probe_batches(terms: &[Term], large: u64) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let (mut start, mut seen) = (0, false);
    for (k, t) in terms.iter().enumerate() {
        if operator_nodes(&t.expr, large) <= large {
            continue;
        }
        if seen {
            out.push(start..k);
            start = k;
        }
        seen = true;
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
/// built, so only one large term's probe is held at a time. A probe that
/// succeeds gives the batch's bands. A failure on it names no term, so the
/// batch is then split in halves and each half probed in turn, first half
/// first, down to single terms, whose own documents' structure is read:
/// the first term whose own document fails, in term order, is
/// `band_model_failed`, with that document's message. The batches are
/// consecutive and taken in order, so that is the first such term of the
/// whole list. Finding it takes about `2 log₂ n` probe documents rather
/// than one document per term.
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
    /// `i`.
    fn own(
        info: &ModelInfo,
        t: &Term,
        i: usize,
        mode: Mode,
    ) -> Result<Vec<BandEntry>, BandFailure> {
        let doc = derived::rhs_document(info, "term", &[t]);
        document_entries(&doc, mode).map_err(|message| BandFailure { term: i, message })
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
        let mut out = Vec::with_capacity(terms.len());
        for r in probe_batches(terms, large) {
            let first = r.start;
            Self::search(info, &terms[r], first, mode, &mut out)?;
        }
        Ok(out)
    }
}

/// The first of `terms`, in their order, whose own right-hand-side
/// document fails in EarthSciASTDiff's full mode, with that document's
/// message, or `None` when none does (§5.2.2, §4 "Order of refusals" item
/// 4). `plan` calls this on an operator's terms when the operator's own
/// document fails in pass 4. It is [`AstDiffBands`]' halving search in
/// [`Mode::Full`], in the same batches, so it takes about `2 log₂ n` probe
/// documents rather than one full differentiation per term; each batch's
/// coefficients are dropped before the next batch is probed.
pub fn first_failing_term(
    info: &ModelInfo,
    terms: &[Term],
) -> Result<Option<BandFailure>, BandSourceError> {
    match AstDiffBands::batched(info, terms, LARGE_TERM_NODES, Mode::Full) {
        Ok(_) => Ok(None),
        Err(BandSourceError::Failed(f)) => Ok(Some(f)),
        Err(e) => Err(e),
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

    /// Every term's band list in `mode`, in batches whose large terms have
    /// more than `large` operator nodes; the first failure is the failing
    /// term's.
    pub fn in_batches(
        info: &ModelInfo,
        terms: &[Term],
        large: u64,
        mode: Mode,
    ) -> Result<Vec<Vec<BandEntry>>, BandSourceError> {
        AstDiffBands::batched(info, terms, large, mode)
    }

    /// Each term's band list in `mode` from its own right-hand-side
    /// document, in term order; the first failure is the failing term's.
    pub fn each_term(
        info: &ModelInfo,
        terms: &[Term],
        mode: Mode,
    ) -> Result<Vec<Vec<BandEntry>>, BandFailure> {
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
