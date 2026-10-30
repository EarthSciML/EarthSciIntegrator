//! The EarthSciASTDiff seam (spec/plan.md §5.2.2, §5.4): a derived
//! document loaded and flattened by EarthSciAST and viewed as one model, its
//! band list from `jacobian_bands`, and the [`BandSource`] that gives each
//! term the bands of its own right-hand-side document.
//!
//! EarthSciASTDiff's Rust `sysview` views one model of a document; a
//! flattened system is not a document, so the view is a document with one
//! model, [`VIEW_MODEL`], whose variables and equations are the flattened
//! system's (fully scoped names, in the flattened order).

use crate::derived;
use crate::locality::{BandEntry, BandFailure, BandSource, RowIdx};
use crate::model::ModelInfo;
use crate::terms::Term;
use earthsci_ast::{EsmFile, FlattenedSystem};
use earthsci_astdiff::{BandError, Ctx, JacEntry, RIdx, SysView, Wrt};
use serde_json::{json, Map, Value};

/// The name of the one model of a [`FlatView`].
pub const VIEW_MODEL: &str = "flattened";

/// A derived document, loaded and flattened by EarthSciAST, as the one-model
/// document EarthSciASTDiff views.
#[derive(Debug, Clone)]
pub struct FlatView {
    /// The flattened document.
    pub flat: FlattenedSystem,
    /// The one-model document of the flattened system.
    pub file: EsmFile,
}

impl FlatView {
    /// Loads the document whose bytes are `text` and flattens it.
    pub fn new(text: &str) -> Result<FlatView, String> {
        let loaded =
            earthsci_ast::load_string(text).map_err(|e| format!("EarthSciAST load: {e}"))?;
        let flat =
            earthsci_ast::flatten(&loaded).map_err(|e| format!("EarthSciAST flatten: {e}"))?;
        let mut vars = Map::new();
        for (n, v) in flat
            .state_variables
            .iter()
            .chain(&flat.parameters)
            .chain(&flat.observed_variables)
        {
            if !vars.contains_key(n) {
                vars.insert(
                    n.clone(),
                    serde_json::to_value(v).map_err(|e| e.to_string())?,
                );
            }
        }
        let doc = json!({
            "esm": derived::ESM_VERSION,
            "metadata": {"name": VIEW_MODEL},
            "index_sets": serde_json::to_value(&flat.index_sets).map_err(|e| e.to_string())?,
            "models": {VIEW_MODEL: {
                "variables": vars,
                "equations": serde_json::to_value(&flat.equations).map_err(|e| e.to_string())?,
            }},
        });
        let file: EsmFile = serde_json::from_value(doc).map_err(|e| e.to_string())?;
        Ok(FlatView { flat, file })
    }

    /// EarthSciASTDiff's view of the flattened model.
    pub fn sysview(&self) -> Result<SysView<'_>, BandError> {
        earthsci_astdiff::sysview(&self.file, Some(VIEW_MODEL))
    }

    /// EarthSciASTDiff's context of the view (index-set sizes, shapes).
    pub fn ctx(&self) -> Result<Ctx, BandError> {
        earthsci_astdiff::ctx_of(&self.sysview()?)
    }

    /// The state Jacobian's band list (`jacobian_bands`, `wrt = states`), in
    /// EarthSciASTDiff's order.
    pub fn bands(&self) -> Result<Vec<JacEntry>, BandError> {
        earthsci_astdiff::jacobian_bands(&self.file, Some(VIEW_MODEL), Wrt::States)
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
