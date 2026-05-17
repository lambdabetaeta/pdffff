//! The immutable base index: every active chunk loaded from SQLite,
//! plus the dense bigram prefilter built over it.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::bigram::{BigramIndex, build_bigram_index_from_chunks};
use crate::db::{Db, LoadedChunkRow};
use crate::index::chunk::ChunkItem;

/// The immutable base index: every active chunk in `(doc_id, chunk_ord)`
/// order, with a `doc_id -> range-of-indices` side map so per-document
/// operations (e.g. tombstoning a doc's base chunks) can locate their
/// chunks cheaply.
///
/// `bigrams` is `Some` whenever the index has at least one chunk; for
/// an empty corpus it is left as `None` because the prefilter has
/// nothing useful to do and the caller should fall straight through
/// to the verification path (an empty `Vec<ChunkItem>` makes that
/// trivial).
pub struct BaseIndex {
    pub chunks: Arc<Vec<ChunkItem>>,
    pub doc_ranges: HashMap<i64, Range<usize>>,
    pub bigrams: Option<Arc<BigramIndex>>,
    /// Per-doc normalised filename, computed once at index build so the
    /// fuzzy query path's filename-match pass can avoid re-running
    /// `normalize_for_index` (which allocates via `deunicode`) on every
    /// filename of every doc on every keystroke. Keyed by `doc_id` to
    /// match `doc_ranges`.
    pub filename_norms: HashMap<i64, String>,
    pub built_at_ms: i64,
}

impl BaseIndex {
    pub fn empty() -> Self {
        BaseIndex {
            chunks: Arc::new(Vec::new()),
            doc_ranges: HashMap::new(),
            bigrams: None,
            filename_norms: HashMap::new(),
            built_at_ms: now_ms(),
        }
    }
}

/// Stream `chunks` out of SQLite and assemble a [`BaseIndex`].
///
/// Per-doc `Arc<str>` deduplication is done with a small `HashMap`:
/// the path string from SQLite is copied at most once per document
/// (not once per chunk). The function is a thin composition over
/// [`stream_chunks_with_doc_ranges`] + [`compute_filename_norms`] +
/// `build_bigram_index_from_chunks` so each phase has a name.
pub fn load_base_index_from_db(db: &Db) -> Result<BaseIndex> {
    let StreamedChunks {
        chunks,
        doc_ranges,
        path_cache,
    } = stream_chunks_with_doc_ranges(db)?;

    let bigrams = if chunks.is_empty() {
        None
    } else {
        Some(Arc::new(build_bigram_index_from_chunks(&chunks)))
    };
    let filename_norms = compute_filename_norms(&path_cache);

    Ok(BaseIndex {
        chunks: Arc::new(chunks),
        doc_ranges,
        bigrams,
        filename_norms,
        built_at_ms: now_ms(),
    })
}

/// Output of the SQL streaming phase.
struct StreamedChunks {
    chunks: Vec<ChunkItem>,
    doc_ranges: HashMap<i64, Range<usize>>,
    /// Per-doc shared (path, filename) `Arc<str>`s, kept so the
    /// downstream filename-norm pass can reuse them without
    /// re-allocating.
    path_cache: HashMap<i64, (Arc<str>, Arc<str>)>,
}

/// One pass over `db.for_each_active_chunk` that:
///
/// * builds the flat `Vec<ChunkItem>` in `(doc_id, chunk_ord)` order;
/// * tracks the per-doc index range (relying on the query's
///   `ORDER BY doc_id, chunk_ord` so doc transitions are contiguous);
/// * dedupes the per-doc `(path, filename)` allocations.
fn stream_chunks_with_doc_ranges(db: &Db) -> Result<StreamedChunks> {
    let mut chunks: Vec<ChunkItem> = Vec::new();
    let mut doc_ranges: HashMap<i64, Range<usize>> = HashMap::new();
    let mut path_cache: HashMap<i64, (Arc<str>, Arc<str>)> = HashMap::new();
    let mut cur_doc: Option<(i64, usize)> = None;

    db.for_each_active_chunk(|row| {
        if doc_changed(cur_doc, row.doc_id) {
            if let Some((prev_id, prev_start)) = cur_doc {
                doc_ranges.insert(prev_id, prev_start..chunks.len());
            }
            cur_doc = Some((row.doc_id, chunks.len()));
        }
        let (path, filename) = path_cache
            .entry(row.doc_id)
            .or_insert_with(|| make_doc_path_arcs(&row))
            .clone();
        chunks.push(chunk_from_row(row, path, filename));
        Ok(())
    })
    .context("loading chunks from SQLite into BaseIndex")?;

    if let Some((prev_id, prev_start)) = cur_doc {
        doc_ranges.insert(prev_id, prev_start..chunks.len());
    }

    Ok(StreamedChunks {
        chunks,
        doc_ranges,
        path_cache,
    })
}

#[inline]
fn doc_changed(cur: Option<(i64, usize)>, new_doc_id: i64) -> bool {
    cur.map(|(id, _)| id) != Some(new_doc_id)
}

/// One owned `Arc<str>` for the doc's full path plus another for its
/// basename. Shared across every chunk of the same doc.
fn make_doc_path_arcs(row: &LoadedChunkRow) -> (Arc<str>, Arc<str>) {
    let path_str = row.path.to_string_lossy().into_owned();
    let filename = std::path::Path::new(&path_str)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path_str.clone());
    (
        Arc::<str>::from(path_str.as_str()),
        Arc::<str>::from(filename.as_str()),
    )
}

/// Materialise one DB row as a [`ChunkItem`], borrowing `path` and
/// `filename` from the per-doc cache.
fn chunk_from_row(row: LoadedChunkRow, path: Arc<str>, filename: Arc<str>) -> ChunkItem {
    ChunkItem::new(
        row.chunk_id,
        row.doc_id,
        path,
        filename,
        row.page_no,
        row.chunk_ord,
        row.char_start,
        row.char_end,
        Arc::<str>::from(row.text_utf8.as_str()),
        Arc::<[u8]>::from(row.text_norm_ascii.as_bytes()),
        Arc::<str>::from(row.preview.as_str()),
        row.doc_mtime_ns,
    )
}

/// Pre-normalised filename cache, keyed by `doc_id`.
///
/// Done eagerly at index build so per-keystroke fuzzy search doesn't
/// re-run `deunicode` over every filename in the corpus.
fn compute_filename_norms(
    path_cache: &HashMap<i64, (Arc<str>, Arc<str>)>,
) -> HashMap<i64, String> {
    path_cache
        .iter()
        .map(|(doc_id, (_, filename))| {
            (*doc_id, crate::normalize::normalize_for_index(filename))
        })
        .collect()
}

pub(crate) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
