//! In-memory chunk index.
//!
//! Day 3 fills in the real types described in `deep-research-report.md`:
//! a flat `Vec<ChunkItem>` keyed by doc range, wrapped in an
//! [`ArcSwap`] so the query loop can read a consistent snapshot with a
//! single atomic load. Day 4 attaches a [`BigramIndex`] to
//! `BaseIndex.bigrams` so the query path can do candidate generation
//! before verifying.
//!
//! Day 5 turns the previously empty [`Overlay`] placeholder into a real
//! mutation overlay: a per-base-chunk tombstone bitset hides stale base
//! rows, and an `overflow_chunks` list publishes new/changed chunks
//! since the last rebuild (each with a pre-extracted deduped bigram
//! set so the query path can prefilter the overflow set the same way
//! it prefilters the base). The base index and overlay together are
//! the live, post-mutation corpus; a future rebuild step (Day 6) folds
//! the overlay back into the base and clears it.
//!
//! Two consistency rules are load-bearing for query correctness:
//!
//! 1. **No double-visibility.** When a doc is modified, the tombstone
//!    must hide the doc's base range *before* the overflow rows
//!    become readable, so a reader never sees both the stale and the
//!    fresh chunks for the same doc. Day 5 ensures this by performing
//!    both mutations under a single `RwLock` write guard in
//!    [`Overlay::modify_doc`].
//! 2. **Snapshot reads.** A query holds `state.load_base()` *and* a
//!    single read guard on `state.overlay` for the entire literal
//!    pass. The overlay's `generation` counter increments on every
//!    write, which makes it cheap for diagnostics to detect that a
//!    snapshot is stale.

use anyhow::{Context, Result};
use arc_swap::ArcSwap;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;

use crate::bigram::{BigramIndex, build_bigram_index_from_chunks, extract_bigrams};
use crate::db::Db;

/// Trigger a base rebuild when the overflow list grows beyond this many
/// chunks. Day 5 only exposes the threshold and the predicate; Day 6
/// owns the actual rebuild routine.
pub const REBUILD_OVERLAY_CHUNKS: usize = 10_000;
/// Trigger a base rebuild when more than this fraction of the base
/// chunks are tombstoned. Same Day 5 / Day 6 split as the chunk
/// threshold.
pub const REBUILD_TOMBSTONE_RATIO: f64 = 0.10;

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
///
/// * `tombstones` hides stale base chunks by *index into `BaseIndex.chunks`*.
///   Each bit covers one base chunk; set ⇒ the chunk is logically deleted
///   from the active corpus (the query path AND-NOTs this against its
///   candidate bitset).
/// * `overflow_chunks` carries new/modified chunks since the last base
///   rebuild. They live in `Vec<ChunkItem>` order — the overlay itself
///   does not impose any (doc_id, chunk_ord) ordering, so callers that
///   need a stable ordering must sort hits at the call site.
/// * `overflow_bigrams[i]` is the deduped bigram set of
///   `overflow_chunks[i].text_norm_ascii`, computed once on insert so
///   the query path doesn't re-extract bigrams per query.
/// * `changed_docs` records every `doc_id` that currently has an
///   overlay entry (a tombstone, an overflow row, or both). This is
///   the set Day-6 rebuild will need to drop from the base.
/// * `generation` increments on every write. Diagnostic only; the
///   query path doesn't rely on it for correctness because it holds
///   the read lock across both verification passes.
pub struct Overlay {
    pub tombstones: Vec<u64>,
    pub overflow_chunks: Vec<ChunkItem>,
    pub overflow_bigrams: Vec<Vec<u16>>,
    pub changed_docs: HashSet<i64>,
    pub generation: u64,
}

/// Snapshot diagnostics returned by [`Overlay::stats`]. Used by future
/// `info` output and by the rebuild trigger.
#[derive(Debug, Clone, Copy)]
pub struct OverlayStats {
    pub tombstones: usize,
    pub overflow_chunks: usize,
    pub generation: u64,
    pub tombstone_ratio: f64,
}

impl Overlay {
    /// Build an empty overlay sized to track `base_chunk_count` tombstones.
    pub fn new(base_chunk_count: usize) -> Self {
        Self {
            tombstones: vec![0u64; words_for(base_chunk_count)],
            overflow_chunks: Vec::new(),
            overflow_bigrams: Vec::new(),
            changed_docs: HashSet::new(),
            generation: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.generation == 0
    }

    /// Tombstone a single base chunk by its index into `BaseIndex.chunks`.
    pub fn tombstone_index(&mut self, base_idx: usize) {
        let word = base_idx / 64;
        let bit = 1u64 << (base_idx % 64);
        if word >= self.tombstones.len() {
            // The tombstone vector was sized to the base at construction
            // time. A base_idx outside that range is a caller bug.
            tracing::warn!(
                base_idx,
                tombstones_len = self.tombstones.len(),
                "tombstone_index called with out-of-range index; ignoring",
            );
            return;
        }
        self.tombstones[word] |= bit;
        self.generation += 1;
    }

    /// Is this base chunk hidden by the overlay?
    #[inline]
    pub fn is_tombstoned(&self, base_idx: usize) -> bool {
        let word = base_idx / 64;
        let bit = 1u64 << (base_idx % 64);
        word < self.tombstones.len() && self.tombstones[word] & bit != 0
    }

    /// Push a new chunk into the overflow set, recording its bigrams
    /// and its parent doc.
    pub fn add_overflow(&mut self, chunk: ChunkItem) {
        let bigrams = extract_bigrams(chunk.text_norm_ascii.as_ref());
        self.changed_docs.insert(chunk.doc_id);
        self.overflow_chunks.push(chunk);
        self.overflow_bigrams.push(bigrams);
        self.generation += 1;
    }

    /// Tombstone every base chunk that belongs to `doc_id`, and drop
    /// any overflow rows previously published for the same doc.
    ///
    /// Dropping the doc's overflow rows is essential for the modify
    /// flow: a doc's second modification must not leave the first
    /// modification's chunks visible. The combined operation is
    /// atomic under the caller's write lock.
    pub fn tombstone_doc(&mut self, doc_id: i64, base: &BaseIndex) {
        if let Some(range) = base.doc_ranges.get(&doc_id) {
            for idx in range.clone() {
                let word = idx / 64;
                let bit = 1u64 << (idx % 64);
                if word < self.tombstones.len() {
                    self.tombstones[word] |= bit;
                }
            }
        }
        // Drop overflow rows for the same doc. We rebuild both vectors
        // in lockstep so `overflow_chunks[i]` continues to align with
        // `overflow_bigrams[i]`.
        if !self.overflow_chunks.is_empty() {
            let mut new_chunks: Vec<ChunkItem> =
                Vec::with_capacity(self.overflow_chunks.len());
            let mut new_bigrams: Vec<Vec<u16>> =
                Vec::with_capacity(self.overflow_bigrams.len());
            let pairs = std::mem::take(&mut self.overflow_chunks)
                .into_iter()
                .zip(std::mem::take(&mut self.overflow_bigrams));
            for (chunk, bigrams) in pairs {
                if chunk.doc_id == doc_id {
                    continue;
                }
                new_chunks.push(chunk);
                new_bigrams.push(bigrams);
            }
            self.overflow_chunks = new_chunks;
            self.overflow_bigrams = new_bigrams;
        }
        self.changed_docs.insert(doc_id);
        self.generation += 1;
    }

    /// Replace a doc's chunks: tombstone its base range and its prior
    /// overflow rows, then publish the new ones. Performed as a single
    /// atomic mutation: the caller holds the write lock for the entire
    /// operation, so a concurrent reader cannot observe the doc with
    /// both the old and new chunks live.
    pub fn modify_doc(&mut self, doc_id: i64, new_chunks: Vec<ChunkItem>, base: &BaseIndex) {
        self.tombstone_doc(doc_id, base);
        for chunk in new_chunks {
            debug_assert_eq!(
                chunk.doc_id, doc_id,
                "modify_doc: incoming chunk's doc_id must match",
            );
            self.add_overflow(chunk);
        }
    }

    /// Reset the overlay to empty. Used after the base index is
    /// rebuilt and the swap is published — the new base already
    /// includes everything the overlay was tracking.
    pub fn clear(&mut self) {
        self.tombstones.clear();
        self.overflow_chunks.clear();
        self.overflow_bigrams.clear();
        self.changed_docs.clear();
        self.generation = 0;
    }

    /// Return the indices into `overflow_chunks` whose deduped bigram
    /// set contains every bigram in `query_bigrams`.
    ///
    /// Mirrors fff's `query_modified` overlay scan: bigram-level
    /// containment is a sound and cheap prefilter; the literal /
    /// regex verifier downstream is still the source of truth.
    /// An empty `query_bigrams` slice returns every overflow index
    /// (the prefilter has nothing to say, so every row is a
    /// candidate).
    pub fn overflow_matches(&self, query_bigrams: &[u16]) -> Vec<usize> {
        if query_bigrams.is_empty() {
            return (0..self.overflow_chunks.len()).collect();
        }
        let mut out = Vec::new();
        for (idx, bigrams) in self.overflow_bigrams.iter().enumerate() {
            // `extract_bigrams` returns first-seen order without
            // dupes, so we can use a tiny set-membership check.
            if query_bigrams.iter().all(|q| bigrams.contains(q)) {
                out.push(idx);
            }
        }
        out
    }

    pub fn stats(&self) -> OverlayStats {
        let tombstones: usize = self
            .tombstones
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum();
        // `tombstones.len() * 64` over-counts when the last word is
        // partial; for ratio purposes that bias is at most one word
        // (≤ 64 chunks) and harmless on any nontrivial corpus.
        let total = (self.tombstones.len() * 64).max(1);
        let tombstone_ratio = tombstones as f64 / total as f64;
        OverlayStats {
            tombstones,
            overflow_chunks: self.overflow_chunks.len(),
            generation: self.generation,
            tombstone_ratio,
        }
    }
}

impl Default for Overlay {
    fn default() -> Self {
        Self::new(0)
    }
}

/// `tombstones.len()` for a base index of `n` chunks.
#[inline]
fn words_for(n: usize) -> usize {
    n.div_ceil(64)
}

/// The handle the rest of the system reads from: an [`ArcSwap`] over
/// the immutable base index plus a small [`RwLock`]-guarded overlay.
pub struct IndexState {
    pub base: ArcSwap<BaseIndex>,
    pub overlay: RwLock<Overlay>,
}

impl IndexState {
    pub fn new(base: BaseIndex) -> Self {
        let chunk_count = base.chunks.len();
        Self {
            base: ArcSwap::new(Arc::new(base)),
            overlay: RwLock::new(Overlay::new(chunk_count)),
        }
    }

    pub fn empty() -> Self {
        Self::new(BaseIndex::empty())
    }

    pub fn load_base(&self) -> Arc<BaseIndex> {
        self.base.load_full()
    }

    /// True when the overlay has drifted far enough from the base
    /// that a rebuild would be cheaper than continuing to scan it.
    /// Day 5 only exposes the predicate; Day 6 owns the rebuild
    /// pipeline.
    ///
    /// Two thresholds:
    /// * Too many overflow chunks: querying them is a linear scan,
    ///   so once we cross [`REBUILD_OVERLAY_CHUNKS`] the overlay
    ///   stops being a prefiltered side-set and starts being a slow
    ///   second corpus.
    /// * Too many tombstones: the dense bitset stays the same size
    ///   regardless of how many bits are set, but verifying
    ///   tombstoned candidates is wasted work.
    ///
    /// Emits a `tracing::info!` the first call that crosses the
    /// threshold so Day 6's rebuild loop has a clear hook.
    pub fn needs_rebuild(&self, ov: &Overlay, base: &BaseIndex) -> bool {
        let stats = ov.stats();
        let by_overflow = stats.overflow_chunks >= REBUILD_OVERLAY_CHUNKS;
        // Compute the ratio against the *base chunk count*, not the
        // bitset capacity, so a corpus of `64 + 1` chunks doesn't
        // get a divide-by-larger-than-base ratio.
        let base_count = base.chunks.len().max(1);
        let tombstone_ratio = stats.tombstones as f64 / base_count as f64;
        let by_tombstones = tombstone_ratio >= REBUILD_TOMBSTONE_RATIO;
        if by_overflow || by_tombstones {
            info!(
                overflow_chunks = stats.overflow_chunks,
                tombstones = stats.tombstones,
                tombstone_ratio,
                "overlay crossed a rebuild threshold",
            );
            return true;
        }
        false
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

    let bigrams = if chunks.is_empty() {
        None
    } else {
        Some(Arc::new(build_bigram_index_from_chunks(&chunks)))
    };

    Ok(BaseIndex {
        chunks: Arc::new(chunks),
        doc_ranges,
        bigrams,
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

    fn mk_chunk(id: i64, doc: i64, text: &str) -> ChunkItem {
        ChunkItem {
            chunk_id: id,
            doc_id: doc,
            path: Arc::from("/x.pdf"),
            filename: Arc::from("x.pdf"),
            page_no: 1,
            chunk_ord: 0,
            char_start: 0,
            char_end: text.len() as u32,
            text_utf8: Arc::from(text),
            text_norm_ascii: Arc::<[u8]>::from(normalize_for_index(text).as_bytes()),
            preview: Arc::from(text),
            doc_mtime_ns: 0,
        }
    }

    fn base_with_three_docs() -> BaseIndex {
        // Three docs, two chunks each, in (doc_id, chunk_ord) order.
        let chunks: Vec<ChunkItem> = vec![
            mk_chunk(1, 1, "alpha bravo charlie"),
            mk_chunk(2, 1, "delta echo foxtrot"),
            mk_chunk(3, 2, "golf hotel india"),
            mk_chunk(4, 2, "juliet kilo lima"),
            mk_chunk(5, 3, "mike november oscar"),
            mk_chunk(6, 3, "papa quebec romeo"),
        ];
        let mut doc_ranges = HashMap::new();
        doc_ranges.insert(1, 0..2);
        doc_ranges.insert(2, 2..4);
        doc_ranges.insert(3, 4..6);
        BaseIndex {
            chunks: Arc::new(chunks),
            doc_ranges,
            bigrams: None,
            built_at_ms: 0,
        }
    }

    #[test]
    fn empty_db_loads_to_empty_index() -> Result<()> {
        let tmp = tempdir()?;
        let db = Db::open(&tmp.path().join("idx.db"))?;
        let base = load_base_index_from_db(&db)?;
        assert!(base.chunks.is_empty());
        assert!(base.doc_ranges.is_empty());
        // No chunks → no prefilter is built (full-scan over the
        // empty Vec is trivial).
        assert!(base.bigrams.is_none());
        Ok(())
    }

    #[test]
    fn populated_db_attaches_bigram_index() -> Result<()> {
        let tmp = tempdir()?;
        let db_path = tmp.path().join("idx.db");
        let mut db = Db::open(&db_path)?;
        db.upsert_extracted(&fake_doc(
            PathBuf::from("/tmp/a.pdf"),
            &[("alpha bravo charlie", 1)],
        ))?;
        drop(db);

        let db = Db::open(&db_path)?;
        let base = load_base_index_from_db(&db)?;
        let bigrams = base.bigrams.as_ref().expect("bigrams attached");
        assert_eq!(bigrams.populated(), base.chunks.len());
        assert_eq!(bigrams.item_count(), base.chunks.len());
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

    #[test]
    fn overlay_tombstone_doc_hides_base_range() {
        let base = base_with_three_docs();
        let mut ov = Overlay::new(base.chunks.len());
        assert!(!ov.is_tombstoned(0));
        ov.tombstone_doc(2, &base);
        // Doc 2 occupies indices 2..4 — both must be tombstoned.
        assert!(ov.is_tombstoned(2));
        assert!(ov.is_tombstoned(3));
        // Other docs untouched.
        assert!(!ov.is_tombstoned(0));
        assert!(!ov.is_tombstoned(1));
        assert!(!ov.is_tombstoned(4));
        assert!(!ov.is_tombstoned(5));
        assert!(ov.changed_docs.contains(&2));
        // Tombstoning bumps generation.
        assert!(ov.generation >= 1);
    }

    #[test]
    fn overlay_tombstone_doc_drops_prior_overflow() {
        let base = base_with_three_docs();
        let mut ov = Overlay::new(base.chunks.len());
        // First publish two overflow rows for doc 2 (simulating a
        // prior modify).
        ov.add_overflow(mk_chunk(100, 2, "new doc 2 chunk one"));
        ov.add_overflow(mk_chunk(101, 2, "new doc 2 chunk two"));
        // Also publish one for doc 3 — must survive.
        ov.add_overflow(mk_chunk(200, 3, "doc 3 overflow"));
        assert_eq!(ov.overflow_chunks.len(), 3);

        // Now tombstone doc 2 — its previous overflow rows must go,
        // doc 3's must stay.
        ov.tombstone_doc(2, &base);
        assert_eq!(ov.overflow_chunks.len(), 1);
        assert_eq!(ov.overflow_chunks[0].doc_id, 3);
        assert_eq!(ov.overflow_bigrams.len(), 1);
        // Doc 2's base range must now be tombstoned.
        assert!(ov.is_tombstoned(2));
        assert!(ov.is_tombstoned(3));
    }

    #[test]
    fn overlay_modify_doc_replaces_atomically() {
        let base = base_with_three_docs();
        let mut ov = Overlay::new(base.chunks.len());
        // Doc 1's base range (indices 0..2) is initially visible.
        assert!(!ov.is_tombstoned(0));
        assert!(!ov.is_tombstoned(1));
        let new_chunks = vec![
            mk_chunk(500, 1, "fresh doc 1 chunk one"),
            mk_chunk(501, 1, "fresh doc 1 chunk two"),
        ];
        ov.modify_doc(1, new_chunks, &base);
        // Base hidden:
        assert!(ov.is_tombstoned(0));
        assert!(ov.is_tombstoned(1));
        // Overflow now carries the two new chunks for doc 1, in
        // insertion order.
        assert_eq!(ov.overflow_chunks.len(), 2);
        assert_eq!(ov.overflow_chunks[0].chunk_id, 500);
        assert_eq!(ov.overflow_chunks[1].chunk_id, 501);
        assert_eq!(ov.overflow_bigrams.len(), 2);
        // And `changed_docs` lists the touched doc.
        assert!(ov.changed_docs.contains(&1));
    }

    #[test]
    fn overlay_modify_doc_replays_replace_themselves() {
        // A doc modified twice must show only the second set of
        // chunks afterwards — the first modify's overflow rows must
        // be dropped by the second call.
        let base = base_with_three_docs();
        let mut ov = Overlay::new(base.chunks.len());

        let first = vec![mk_chunk(700, 3, "first revision")];
        ov.modify_doc(3, first, &base);
        assert_eq!(ov.overflow_chunks.len(), 1);
        assert_eq!(ov.overflow_chunks[0].chunk_id, 700);

        let second = vec![
            mk_chunk(701, 3, "second revision one"),
            mk_chunk(702, 3, "second revision two"),
        ];
        ov.modify_doc(3, second, &base);
        assert_eq!(ov.overflow_chunks.len(), 2);
        assert!(ov.overflow_chunks.iter().all(|c| c.doc_id == 3));
        let ids: Vec<i64> = ov.overflow_chunks.iter().map(|c| c.chunk_id).collect();
        assert_eq!(ids, vec![701, 702]);
    }

    #[test]
    fn overlay_overflow_matches_filters_by_bigrams() {
        let base = base_with_three_docs();
        let mut ov = Overlay::new(base.chunks.len());
        // Three overflow rows with distinctive token tails.
        ov.add_overflow(mk_chunk(10, 1, "the quick brown fox jumps"));
        ov.add_overflow(mk_chunk(11, 2, "lazy dog watches the river"));
        ov.add_overflow(mk_chunk(12, 3, "the quick zebra"));

        // Query bigrams for "quick" → "qu", "ui", "ic", "ck"
        let q = extract_bigrams(b"quick");
        let hits = ov.overflow_matches(&q);
        // Should be indices 0 and 2 (both contain "quick"), in order.
        assert_eq!(hits, vec![0, 2]);

        // A bigram set with a non-existent bigram → no matches.
        let q = extract_bigrams(b"zzzz");
        assert_eq!(ov.overflow_matches(&q), Vec::<usize>::new());

        // Empty query → every overflow row.
        assert_eq!(ov.overflow_matches(&[]), vec![0, 1, 2]);
    }

    #[test]
    fn overlay_clear_resets_everything() {
        let base = base_with_three_docs();
        let mut ov = Overlay::new(base.chunks.len());
        ov.add_overflow(mk_chunk(10, 1, "anything"));
        ov.tombstone_doc(2, &base);
        assert!(!ov.is_empty());
        ov.clear();
        assert!(ov.overflow_chunks.is_empty());
        assert!(ov.overflow_bigrams.is_empty());
        assert!(ov.changed_docs.is_empty());
        assert_eq!(ov.generation, 0);
        // After `clear`, the tombstone vector is empty — the next
        // base index will reset it via `Overlay::new`. Until then no
        // index is in range.
        for i in 0..base.chunks.len() {
            assert!(!ov.is_tombstoned(i));
        }
    }

    #[test]
    fn needs_rebuild_threshold_triggers() {
        // Tombstone enough of a tiny base to cross the ratio.
        let base = base_with_three_docs();
        let state = IndexState::new(base);
        let ov = state.overlay.read();
        let base = state.load_base();
        assert!(!state.needs_rebuild(&ov, &base));
        drop(ov);

        // Tombstone the first doc (2 of 6 chunks = 33%, > 10%).
        let base = state.load_base();
        {
            let mut ov = state.overlay.write();
            ov.tombstone_doc(1, &base);
        }
        let ov = state.overlay.read();
        assert!(state.needs_rebuild(&ov, &base));
    }
}
