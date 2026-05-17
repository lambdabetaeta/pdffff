//! The unit of indexing: a `ChunkItem`.
//!
//! Construction goes through [`ChunkItem::new`] so the
//! `char_start ≤ char_end` invariant is checked in exactly one place;
//! every loader (DB stream, overlay publish, bench / test fixture)
//! ends up calling it.

use std::sync::Arc;

/// One indexed chunk loaded from `chunks` into memory.
///
/// `path` and `filename` are `Arc<str>` so all chunks belonging to the
/// same document share a single allocation. `text_utf8` and
/// `text_norm_ascii` are kept distinct because the normalization is
/// lossy (deunicode, lowercase, whitespace collapse) and the original
/// is what we render in snippets.
///
/// Fields are public for read access — the type is an immutable data
/// record — but construction goes through [`ChunkItem::new`] so the
/// `char_start ≤ char_end` invariant is enforced in one place.
#[derive(Debug, Clone)]
pub struct ChunkItem {
    pub chunk_id: i64,
    pub doc_id: i64,
    pub path: Arc<str>,
    pub filename: Arc<str>,
    pub page_no: u32,
    pub chunk_ord: u32,
    pub char_start: u32,
    pub char_end: u32,
    pub text_utf8: Arc<str>,
    pub text_norm_ascii: Arc<[u8]>,
    pub preview: Arc<str>,
    pub doc_mtime_ns: i64,
}

impl ChunkItem {
    /// Build a `ChunkItem` from its components.
    ///
    /// Enforces `char_start ≤ char_end` via `debug_assert!`; in
    /// release builds an out-of-order pair will still construct, but
    /// downstream code (snippet rendering) treats the inversion as a
    /// caller bug rather than a runtime condition.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        chunk_id: i64,
        doc_id: i64,
        path: Arc<str>,
        filename: Arc<str>,
        page_no: u32,
        chunk_ord: u32,
        char_start: u32,
        char_end: u32,
        text_utf8: Arc<str>,
        text_norm_ascii: Arc<[u8]>,
        preview: Arc<str>,
        doc_mtime_ns: i64,
    ) -> Self {
        debug_assert!(
            char_start <= char_end,
            "ChunkItem invariant: char_start ({char_start}) must be ≤ char_end ({char_end})",
        );
        Self {
            chunk_id,
            doc_id,
            path,
            filename,
            page_no,
            chunk_ord,
            char_start,
            char_end,
            text_utf8,
            text_norm_ascii,
            preview,
            doc_mtime_ns,
        }
    }
}
