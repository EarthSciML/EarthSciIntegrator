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
use earthsci_astdiff::{BandError, Ctx, JacEntry, RIdx, SysView, Wrt};
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

    /// EarthSciASTDiff's view of the flattened system (`sysview_flat`).
    pub fn sysview(&self) -> SysView<'_> {
        earthsci_astdiff::sysview_flat(&self.flat)
    }

    /// EarthSciASTDiff's context of the view (index-set sizes, shapes).
    pub fn ctx(&self) -> Result<Ctx, BandError> {
        earthsci_astdiff::ctx_of(&self.sysview())
    }

    /// The state Jacobian's band list (`jacobian_bands_view`, `wrt =
    /// states`), in EarthSciASTDiff's order.
    pub fn bands(&self) -> Result<Vec<JacEntry>, BandError> {
        earthsci_astdiff::jacobian_bands_view(&self.sysview(), Wrt::States, None)
    }
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
pub fn document_bands(doc: &Value) -> Result<(FlatView, Vec<JacEntry>), String> {
    let text = String::from_utf8(derived::to_bytes(doc)).map_err(|e| e.to_string())?;
    let view = FlatView::new(&text)?;
    let bands = view
        .bands()
        .map_err(|e| format!("EarthSciASTDiff: {}", e.0))?;
    Ok((view, bands))
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
            let (_, bands) =
                document_bands(&doc).map_err(|message| BandFailure { term: i, message })?;
            out.push(bands.iter().map(band_entry).collect());
        }
        Ok(Some(out))
    }
}
