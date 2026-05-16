//! Day-3/4 placeholder: `ChunkItem`, `BaseIndex`, `Overlay`,
//! `IndexState` (the `ArcSwap<BaseIndex>` + `RwLock<Overlay>` pair).
//!
//! The real types are filled in by the Day-3 agent (`ChunkItem`,
//! `BaseIndex` construction without bigrams), Day-4 agent (bigrams),
//! and Day-5 agent (overlay).

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Arc;

/// One indexed chunk loaded from `chunks` into memory.
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
    pub fn filename_from_path(path: &str) -> Arc<str> {
        let name = std::path::Path::new(path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string());
        Arc::from(name.as_str())
    }
}

/// Placeholder so other modules can name the type. Replaced on Day 3/4.
pub struct BaseIndex {
    pub chunks: Arc<Vec<ChunkItem>>,
}

impl BaseIndex {
    pub fn empty() -> Self {
        BaseIndex {
            chunks: Arc::new(Vec::new()),
        }
    }
}

/// Used by tests / wiring placeholder.
pub type PathRef = PathBuf;
