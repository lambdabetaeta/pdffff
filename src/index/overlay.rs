//! Mutation overlay over [`super::BaseIndex`].
//!
//! Splits into two cooperating pieces:
//!
//! * [`OverflowSet`] — the chunk-publishing side: new chunks since
//!   the last rebuild, paired with sorted bigram sets so the
//!   per-query prefilter is a binary search per query bigram.
//! * [`Overlay`] — the tombstone bitset + the overflow set + the
//!   `changed_docs` audit trail, with the atomic
//!   tombstone-then-publish dance for doc modifications.

use std::collections::HashSet;

use crate::bigram::extract_bigrams;
use crate::bitset::Bitset;
use crate::index::base::BaseIndex;
use crate::index::chunk::ChunkItem;

/// Trigger a base rebuild when the overflow list grows beyond this many
/// chunks.
pub const REBUILD_OVERLAY_CHUNKS: usize = 10_000;
/// Trigger a base rebuild when more than this fraction of the base
/// chunks are tombstoned.
pub const REBUILD_TOMBSTONE_RATIO: f64 = 0.10;

/// Per-chunk side of the overlay: chunks published since the last base
/// rebuild, paired with their deduped & sorted bigram set so the
/// per-query prefilter is a small binary-search loop rather than a
/// linear scan.
///
/// The two underlying vectors are wrapped together because they share
/// one invariant — `chunks[i]` and `bigrams[i]` must always refer to
/// the same row. Every mutating method on [`OverflowSet`] preserves
/// that invariant, so callers cannot drift them out of sync.
#[derive(Debug, Default)]
pub struct OverflowSet {
    chunks: Vec<ChunkItem>,
    /// Deduped, sorted bigram set for `chunks[i]`. Sorting once at
    /// insert lets [`OverflowSet::matches`] use `binary_search` so the
    /// per-chunk check is O(Q · log C) rather than O(Q · C).
    bigrams: Vec<Vec<u16>>,
}

impl OverflowSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn chunks(&self) -> &[ChunkItem] {
        &self.chunks
    }

    /// Add `chunk`. Extracts and sorts its bigrams in the same step,
    /// so the row enters with both its slots populated.
    pub fn push(&mut self, chunk: ChunkItem) {
        let mut bg = extract_bigrams(chunk.text_norm_ascii.as_ref());
        bg.sort_unstable();
        self.chunks.push(chunk);
        self.bigrams.push(bg);
    }

    /// Drop every row whose chunk belongs to `doc_id`.
    ///
    /// Two-pointer compact: O(n) time, no allocations beyond the
    /// truncation. The chunk and bigram vectors stay aligned because
    /// the same swap pattern drives both.
    pub fn drop_doc(&mut self, doc_id: i64) {
        let mut write = 0;
        for read in 0..self.chunks.len() {
            if self.chunks[read].doc_id != doc_id {
                if write != read {
                    self.chunks.swap(write, read);
                    self.bigrams.swap(write, read);
                }
                write += 1;
            }
        }
        self.chunks.truncate(write);
        self.bigrams.truncate(write);
    }

    /// Reset to empty.
    pub fn clear(&mut self) {
        self.chunks.clear();
        self.bigrams.clear();
    }

    /// Return indices whose deduped bigram set contains every entry of
    /// `query_bigrams`.
    ///
    /// `query_bigrams` may be in any order. The per-chunk side is
    /// sorted at insert time so each `contains` check is a binary
    /// search. An empty query bigram slice returns every row (the
    /// prefilter has nothing to say).
    pub fn matches(&self, query_bigrams: &[u16]) -> Vec<usize> {
        if query_bigrams.is_empty() {
            return (0..self.chunks.len()).collect();
        }
        let mut out = Vec::new();
        for (i, bg) in self.bigrams.iter().enumerate() {
            if query_bigrams.iter().all(|q| bg.binary_search(q).is_ok()) {
                out.push(i);
            }
        }
        out
    }
}

/// Snapshot diagnostics returned by [`Overlay::stats`].
#[derive(Debug, Clone, Copy)]
pub struct OverlayStats {
    pub tombstones: usize,
    pub overflow_chunks: usize,
    pub generation: u64,
    pub tombstone_ratio: f64,
}

/// Mutable overlay applied on top of [`BaseIndex`] between rebuilds.
///
/// * `tombstones` hides stale base chunks by index into
///   `BaseIndex.chunks`. Each bit covers one base chunk; set ⇒ the
///   chunk is logically deleted (the query path AND-NOTs this
///   against its candidate bitset).
/// * `overflow` carries new/modified chunks since the last base
///   rebuild, paired with their sorted bigram sets. Insertion order.
/// * `changed_docs` records every `doc_id` that currently has an
///   overlay entry (a tombstone, an overflow row, or both). This is
///   the set the rebuild routine drops from the base.
/// * `generation` increments on every write. Diagnostic only.
pub struct Overlay {
    pub tombstones: Bitset,
    pub overflow: OverflowSet,
    pub changed_docs: HashSet<i64>,
    pub generation: u64,
}

impl Overlay {
    /// Build an empty overlay sized to track `base_chunk_count` tombstones.
    pub fn new(base_chunk_count: usize) -> Self {
        Self {
            tombstones: Bitset::zeros(base_chunk_count),
            overflow: OverflowSet::new(),
            changed_docs: HashSet::new(),
            generation: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.generation == 0
    }

    /// Tombstone a single base chunk by its index into `BaseIndex.chunks`.
    pub fn tombstone_index(&mut self, base_idx: usize) {
        if base_idx >= self.tombstones.len() {
            // The tombstone bitset was sized to the base at construction
            // time. A base_idx outside that range is a caller bug.
            tracing::warn!(
                base_idx,
                tombstones_len = self.tombstones.len(),
                "tombstone_index called with out-of-range index; ignoring",
            );
            return;
        }
        self.tombstones.set(base_idx);
        self.generation += 1;
    }

    /// Is this base chunk hidden by the overlay?
    #[inline]
    pub fn is_tombstoned(&self, base_idx: usize) -> bool {
        self.tombstones.get(base_idx)
    }

    /// Push a new chunk into the overflow set, recording its bigrams
    /// and its parent doc.
    pub fn add_overflow(&mut self, chunk: ChunkItem) {
        self.changed_docs.insert(chunk.doc_id);
        self.overflow.push(chunk);
        self.generation += 1;
    }

    /// Tombstone every base chunk that belongs to `doc_id`.
    fn tombstone_doc_in_base(&mut self, doc_id: i64, base: &BaseIndex) {
        if let Some(range) = base.doc_ranges.get(&doc_id) {
            for idx in range.clone() {
                self.tombstones.set(idx);
            }
        }
    }

    /// Tombstone every base chunk that belongs to `doc_id`, and drop
    /// any overflow rows previously published for the same doc.
    ///
    /// Dropping the doc's overflow rows is essential for the modify
    /// flow: a doc's second modification must not leave the first
    /// modification's chunks visible. The combined operation is
    /// atomic under the caller's write lock.
    pub fn tombstone_doc(&mut self, doc_id: i64, base: &BaseIndex) {
        self.tombstone_doc_in_base(doc_id, base);
        self.overflow.drop_doc(doc_id);
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
        self.tombstones = Bitset::zeros(0);
        self.overflow.clear();
        self.changed_docs.clear();
        self.generation = 0;
    }

    /// Indices into `overflow.chunks()` whose deduped bigram set
    /// contains every entry of `query_bigrams`.
    ///
    /// Thin convenience over [`OverflowSet::matches`] so callers don't
    /// have to reach through the field.
    pub fn overflow_matches(&self, query_bigrams: &[u16]) -> Vec<usize> {
        self.overflow.matches(query_bigrams)
    }

    pub fn stats(&self) -> OverlayStats {
        let tombstones = self.tombstones.count_ones();
        // Use the logical bit length, not the word-aligned capacity, so
        // a corpus of `64 + 1` chunks gets a ratio against the true
        // chunk count.
        let total = self.tombstones.len().max(1);
        let tombstone_ratio = tombstones as f64 / total as f64;
        OverlayStats {
            tombstones,
            overflow_chunks: self.overflow.len(),
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
