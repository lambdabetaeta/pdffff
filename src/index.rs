//! In-memory chunk index.
//!
//! Day 3 fills in the real types described in `deep-research-report.md`:
//! a flat `Vec<ChunkItem>` keyed by doc range, wrapped in an
//! [`ArcSwap`] so the query loop can read a consistent snapshot with a
//! single atomic load. The bigram posting list is deferred to Day 4
//! (the `BaseIndex::bigrams` field is left as `None` until then), and
//! the mutable overlay is deferred to Day 5 (the [`Overlay`] field is
//! kept here as a no-op placeholder so wiring downstream is stable).

use anyhow::{Context, Result};
use arc_swap::ArcSwap;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::bigram::BigramIndex;
use crate::db::Db;

/// One indexed chunk loaded from `chunks` into memory.
///
/// `path` and `filename` are `Arc<str>` so all chunks belonging to the
/// same document share a single allocation. `text_utf8` and
/// `text_norm_ascii` are kept distinct because the normalization is
/// lossy (deunicode, lowercase, whitespace collapse) and the original
/// is what we render in snippets.
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

/// The immutable base index: every active chunk in `(doc_id, chunk_ord)`
/// order, with a `doc_id -> range-of-indices` side map so per-document
/// operations (a future incremental update, e.g.) can locate their
/// chunks cheaply.
///
/// `bigrams` is `None` for Day 3; Day 4 fills it in.
pub struct BaseIndex {
    pub chunks: Arc<Vec<ChunkItem>>,
    pub doc_ranges: HashMap<i64, Range<usize>>,
    pub bigrams: Option<Arc<BigramIndex>>,
    pub built_at_ms: i64,
}

impl BaseIndex {
    pub fn empty() -> Self {
        BaseIndex {
            chunks: Arc::new(Vec::new()),
            doc_ranges: HashMap::new(),
            bigrams: None,
            built_at_ms: now_ms(),
        }
    }
}

/// Mutable overlay applied on top of [`BaseIndex`] between rebuilds.
/// Day 5 will fill this in (tombstones over the base + overflow chunks
/// for new/modified docs). Day 3 keeps it present but empty so the
/// query path and call sites already have a stable shape.
#[derive(Default)]
pub struct Overlay {
    pub generation: u64,
}

impl Overlay {
    pub fn is_empty(&self) -> bool {
        self.generation == 0
    }
}

/// The handle the rest of the system reads from: an [`ArcSwap`] over
/// the immutable base index plus a small [`RwLock`]-guarded overlay.
///
/// Day 3 only uses the base half; the overlay is a no-op placeholder.
pub struct IndexState {
    pub base: ArcSwap<BaseIndex>,
    pub overlay: RwLock<Overlay>,
}

impl IndexState {
    pub fn new(base: BaseIndex) -> Self {
        Self {
            base: ArcSwap::new(Arc::new(base)),
            overlay: RwLock::new(Overlay::default()),
        }
    }

    pub fn empty() -> Self {
        Self::new(BaseIndex::empty())
    }

    pub fn load_base(&self) -> Arc<BaseIndex> {
        self.base.load_full()
    }
}

/// Stream `chunks` out of SQLite and assemble a [`BaseIndex`].
///
/// Per-doc `Arc<str>` deduplication is done with a small `HashMap`:
/// the path string from SQLite is copied at most once per document
/// (not once per chunk).
pub fn load_base_index_from_db(db: &Db) -> Result<BaseIndex> {
    let mut chunks: Vec<ChunkItem> = Vec::new();
    let mut doc_ranges: HashMap<i64, Range<usize>> = HashMap::new();
    let mut path_cache: HashMap<i64, (Arc<str>, Arc<str>)> = HashMap::new();
    let mut cur_doc: Option<(i64, usize)> = None;

    db.for_each_active_chunk(|row| {
        // `for_each_active_chunk` is ordered by (doc_id, chunk_ord), so
        // doc transitions are contiguous and one-pass.
        if cur_doc.map(|(id, _)| id) != Some(row.doc_id) {
            if let Some((prev_id, prev_start)) = cur_doc {
                doc_ranges.insert(prev_id, prev_start..chunks.len());
            }
            cur_doc = Some((row.doc_id, chunks.len()));
        }
        let (path, filename) = path_cache
            .entry(row.doc_id)
            .or_insert_with(|| {
                let path_str = row.path.to_string_lossy().into_owned();
                let filename = std::path::Path::new(&path_str)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path_str.clone());
                (Arc::<str>::from(path_str.as_str()), Arc::<str>::from(filename.as_str()))
            })
            .clone();
        chunks.push(ChunkItem {
            chunk_id: row.chunk_id,
            doc_id: row.doc_id,
            path,
            filename,
            page_no: row.page_no,
            chunk_ord: row.chunk_ord,
            char_start: row.char_start,
            char_end: row.char_end,
            text_utf8: Arc::<str>::from(row.text_utf8.as_str()),
            text_norm_ascii: Arc::<[u8]>::from(row.text_norm_ascii.as_bytes()),
            preview: Arc::<str>::from(row.preview.as_str()),
            doc_mtime_ns: row.doc_mtime_ns,
        });
        Ok(())
    })
    .context("loading chunks from SQLite into BaseIndex")?;

    if let Some((prev_id, prev_start)) = cur_doc {
        doc_ranges.insert(prev_id, prev_start..chunks.len());
    }

    Ok(BaseIndex {
        chunks: Arc::new(chunks),
        doc_ranges,
        bigrams: None,
        built_at_ms: now_ms(),
    })
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{ChunkInsert, DocStatus, ExtractedDoc};
    use crate::normalize::{NORM_VERSION, normalize_for_index};
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn fake_doc(path: PathBuf, pages: &[(&str, u32)]) -> ExtractedDoc {
        let chunks: Vec<ChunkInsert> = pages
            .iter()
            .enumerate()
            .map(|(i, (text, page_no))| ChunkInsert {
                page_no: *page_no,
                chunk_ord: i as u32,
                char_start: 0,
                char_end: text.len() as u32,
                text_utf8: (*text).to_string(),
                text_norm_ascii: normalize_for_index(text),
                preview: (*text).to_string(),
            })
            .collect();
        ExtractedDoc {
            path,
            size_bytes: 0,
            mtime_ns: 0,
            dev: None,
            ino: None,
            extractor: "test".into(),
            extractor_version: "test".into(),
            norm_version: NORM_VERSION,
            page_count: pages.len() as u32,
            status: DocStatus::Ok,
            error_text: None,
            chunks,
        }
    }

    #[test]
    fn empty_db_loads_to_empty_index() -> Result<()> {
        let tmp = tempdir()?;
        let db = Db::open(&tmp.path().join("idx.db"))?;
        let base = load_base_index_from_db(&db)?;
        assert!(base.chunks.is_empty());
        assert!(base.doc_ranges.is_empty());
        assert!(base.bigrams.is_none());
        Ok(())
    }

    #[test]
    fn loads_doc_ranges_in_order() -> Result<()> {
        let tmp = tempdir()?;
        let db_path = tmp.path().join("idx.db");
        let mut db = Db::open(&db_path)?;
        let id_a = db.upsert_extracted(&fake_doc(
            PathBuf::from("/tmp/a.pdf"),
            &[("alpha bravo", 1), ("charlie delta", 2)],
        ))?;
        let id_b = db.upsert_extracted(&fake_doc(
            PathBuf::from("/tmp/b.pdf"),
            &[("echo foxtrot", 1)],
        ))?;
        drop(db);

        let db = Db::open(&db_path)?;
        let base = load_base_index_from_db(&db)?;
        assert_eq!(base.chunks.len(), 3);
        let range_a = base.doc_ranges.get(&id_a).expect("doc a present");
        let range_b = base.doc_ranges.get(&id_b).expect("doc b present");
        assert_eq!(range_a.len(), 2);
        assert_eq!(range_b.len(), 1);
        // doc_id is monotone in insertion order, so ranges should be
        // contiguous and ordered.
        assert_eq!(range_a.end, range_b.start);

        // Path is shared per doc — pointer-equal across chunks of the
        // same document, distinct across documents.
        let a0 = &base.chunks[range_a.start];
        let a1 = &base.chunks[range_a.start + 1];
        let b0 = &base.chunks[range_b.start];
        assert!(Arc::ptr_eq(&a0.path, &a1.path));
        assert!(!Arc::ptr_eq(&a0.path, &b0.path));
        assert_eq!(&*a0.filename, "a.pdf");
        assert_eq!(&*b0.filename, "b.pdf");

        // Norm bytes are exactly what the normalizer produced.
        assert_eq!(&*a0.text_norm_ascii, b"alpha bravo");
        assert_eq!(&*b0.text_norm_ascii, b"echo foxtrot");
        Ok(())
    }

    #[test]
    fn deleted_doc_chunks_are_excluded() -> Result<()> {
        let tmp = tempdir()?;
        let db_path = tmp.path().join("idx.db");
        let mut db = Db::open(&db_path)?;
        db.upsert_extracted(&fake_doc(
            PathBuf::from("/tmp/keep.pdf"),
            &[("kept text", 1)],
        ))?;
        db.upsert_extracted(&fake_doc(
            PathBuf::from("/tmp/drop.pdf"),
            &[("dropped text", 1)],
        ))?;
        db.mark_deleted(std::path::Path::new("/tmp/drop.pdf"))?;
        drop(db);

        let db = Db::open(&db_path)?;
        let base = load_base_index_from_db(&db)?;
        assert_eq!(base.chunks.len(), 1);
        assert_eq!(&*base.chunks[0].text_norm_ascii, b"kept text");
        Ok(())
    }
}
