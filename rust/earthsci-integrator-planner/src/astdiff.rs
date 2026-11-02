//! The EarthSciASTDiff seam (spec/plan.md §5.2.2, §5.4): a derived
//! document loaded and flattened by EarthSciAST and viewed with
//! EarthSciASTDiff's `sysview_flat`, its band list from
//! `jacobian_bands_view`, and the [`BandSource`] that gives each term the
//! bands of its own right-hand-side document.

use crate::derived;
use crate::locality::{BandEntry, BandFailure, BandSource, RowIdx};
use crate::model::ModelInfo;
use crate::terms::Term;
use earthsci_ast::FlattenedSystem;
use earthsci_astdiff::{BandError, JacEntry, RIdx, SysView, Wrt};
use serde_json::Value;

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

/// A band entry as the locality rules read it (§5.2.3).
pub fn band_entry(en: &JacEntry) -> BandEntry {
    let b = &en.band;
    BandEntry {
        row: en.u.clone(),
        col: en.v.clone(),
        rows: b.rows.clone(),
        row_idx: b
            .ridx
            .iter()
            .map(|r| match r {
                RIdx::Name(n) => RowIdx::Name(n.clone()),
                RIdx::Fixed(i) => RowIdx::Fixed(*i),
            })
            .collect(),
        col_idx: b.cidx.clone(),
        contracted: b.contracted.clone(),
    }
}

/// The band list of the document `doc` (§5.2.2, §5.4): loaded, flattened,
/// viewed and differentiated. A failure anywhere is a message for
/// `band_model_failed`.
pub fn document_bands(doc: &Value) -> Result<Vec<JacEntry>, String> {
    let view = FlatView::of_document(doc)?;
    let sv = view.sysview().map_err(astdiff_message)?;
    bands(&sv).map_err(astdiff_message)
}

/// An EarthSciASTDiff failure as a `band_model_failed` message.
pub fn astdiff_message(e: BandError) -> String {
    format!("EarthSciASTDiff: {}", e.0)
}

/// Term bands from EarthSciASTDiff (§5.2.2): each term's band list is that
/// of the right-hand-side document of a hypothetical operator holding only
/// that term. One document per term: the fixtures are small, and the band
/// lists are exactly the definition's.
pub struct AstDiffBands;

impl BandSource for AstDiffBands {
    fn term_bands(
        &self,
        info: &ModelInfo,
        terms: &[Term],
    ) -> Result<Option<Vec<Vec<BandEntry>>>, BandFailure> {
        let mut out = Vec::with_capacity(terms.len());
        for (i, t) in terms.iter().enumerate() {
            let doc = derived::rhs_document(info, "term", &[t]);
            let bands = document_bands(&doc).map_err(|message| BandFailure { term: i, message })?;
            out.push(bands.iter().map(band_entry).collect());
        }
        Ok(Some(out))
    }
}
