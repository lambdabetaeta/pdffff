// SPDX-License-Identifier: MIT
// Adapted from fff's `bigram_filter.rs`, (c) 2025 Dmitriy Kovalenko, MIT.
// Upstream: <https://github.com/dmitriy-kovalenko/fff>.
//
// Modifications for pdffff:
//   * The unit of indexing is a `ChunkItem` rather than a `FileItem`, so
//     "file_idx" is renamed "item_idx" throughout.
//   * The parallel builder uses `AtomicU64::fetch_or` for the dense bitset
//     writes instead of the partitioned `UnsafeCell` slab from upstream.
//     fff drives writes through `par_chunks` with a word-aligned chunk size
//     so that no two threads ever touch the same `u64` word, which lets it
//     use a plain `*p |= mask` through an `UnsafeCell<Box<[u64]>>`. We
//     prefer the safer `AtomicU64::fetch_or` here: the atomic-RMW cost is
//     invisible at chunk-corpus sizes (we expect ≤ ~200k chunks), and
//     dropping the `unsafe` removes a class of invariants from the
//     caller's contract. See the module-level note on `BigramIndexBuilder`.
//   * Driver-side helpers (mmap / page-cache prefetch, content cache,
//     binary detection, etc.) are dropped: items here are already
//     `Arc<[u8]>` in memory.
//   * The on-disk `BigramOverlay` from upstream is deferred to Day 5.

//! Dense bigram inverted index over `Vec<ChunkItem>`.
//!
//! The index is a candidate-generation prefilter for literal / regex /
//! fuzzy searches: a query's bigrams index into 65536 posting lists, and
//! AND-ing those lists yields a small set of chunks that *might* contain
//! the query. The verification scan (`memchr::memmem`) then decides
//! which candidates are actual hits. This module never decides hits; it
//! only narrows the candidate set.
//!
//! The layout mirrors fff's `BigramFilter`:
//! * `lookup: Vec<u16>` of length 65536 maps each printable-ASCII bigram
//!   key `(hi << 8) | lo` to a dense column index (or `NO_COLUMN` =
//!   `u16::MAX` for "not tracked").
//! * `dense_data: Vec<u64>` is a flat bitset slab at fixed stride
//!   `words = ceil(item_count / 64)`. Column `c` lives at
//!   `c * words .. (c + 1) * words`. Bit `i` of word `w` is set iff
//!   item `64 * w + i` contains the bigram represented by column `c`.
//! * `dense_count` counts how many of the 65536 keys survived
//!   compression; columns ≥ `dense_count` are unused.
//! * An optional skip-1 sub-index (`skip_index`) holds bigrams of bytes
//!   `(content[i], content[i + 2])` for tighter filtering on queries of
//!   length ≥ 3.
//!
//! Compression drops columns whose density is too low (the bigram appears
//! in too few items to be worth its 8 bytes per 64 items) or too high
//! (≥ 90% of items: the column is effectively all-ones and adds AND
//! cycles without removing candidates). Both cutoffs are taken verbatim
//! from fff.
//!
//! ## Parallel build
//!
//! `BigramIndexBuilder` is `Sync`-safe. Multiple threads may call
//! `add_item_content(skip_builder, item_idx, content)` with any
//! distinct `item_idx`s in any order; column allocation goes through
//! atomic RMWs on a 65536-element `Vec<AtomicU16>`, and per-(column,
//! word) writes go through `AtomicU64::fetch_or`. This is strictly
//! safer than fff's partitioned-write trick — it does not require the
//! driver to partition items into word-aligned ranges — at the cost of
//! a `lock or` instead of a plain `or` on x86_64 per record. At chunk
//! scale that cost is invisible.

#![allow(clippy::needless_range_loop)]

use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSlice;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU16, AtomicU64, AtomicUsize, Ordering};

use crate::index::ChunkItem;

/// Maximum number of distinct bigrams the dense slab tracks.
///
/// Printable ASCII (32..=126) lowercased gives ~70 distinct bytes, so
/// at most ~4900 distinct bigrams are reachable; 5000 leaves a margin.
/// Memory cost is `MAX_BIGRAM_COLUMNS * words * 8` bytes during build.
pub const MAX_BIGRAM_COLUMNS: usize = 5000;

/// Sentinel: "this bigram key has no column" in the lookup table.
const NO_COLUMN: u16 = u16::MAX;

/// Skip-1 sub-index column density floor. fff documents this as the
/// value where the skip-1 sub-index keeps ~70-75% of its filtering
/// power while shedding the 25-30% memory the sparse tail consumes.
const SKIP_INDEX_MIN_DENSITY_PCT: u32 = 12;

/// Driver chunk size (in items) for the rayon parallel build. Any
/// positive integer works because the builder uses atomic writes; we
/// pick 256 to balance per-task overhead against load-balancing on
/// small corpora. The value is not load-bearing for correctness.
const BIGRAM_CHUNK_ITEMS: usize = 256;

/// Map a single input byte to its normalised bigram alphabet position:
/// `u16::MAX` if the byte is outside printable ASCII (32..=126),
/// otherwise the lowercased byte value as a `u16`.
///
/// The same routine is used at index time and at query time; both sides
/// must agree on what a "bigram" is or the prefilter will reject
/// candidates that actually do contain the query.
#[inline(always)]
fn normalize_byte_scalar(b: u8) -> u16 {
    let printable = b.wrapping_sub(32) <= 94;
    // Branchless lowercase: OR 0x20 iff byte is in 'A'..='Z'.
    let lower = b | ((b.wrapping_sub(b'A') < 26) as u8 * 0x20);
    if printable { lower as u16 } else { u16::MAX }
}

/// Parallel-safe dense builder for [`BigramIndex`].
///
/// One builder per logical sub-index (consecutive vs skip-1). Items are
/// fed in by `add_item_content`; the builder lazily materialises a
/// `MAX_BIGRAM_COLUMNS * words` `AtomicU64` slab on first use and
/// records bigrams into it. `compress` consumes the builder and emits
/// the final compact [`BigramIndex`].
///
/// Safety: all writes to the slab go through `AtomicU64::fetch_or` with
/// `Ordering::Relaxed`. There is no per-`u64` exclusivity invariant the
/// driver has to enforce; threads may touch overlapping words freely.
pub struct BigramIndexBuilder {
    /// Per-bigram-key column index. `NO_COLUMN` until the first thread
    /// allocates a column for the key.
    lookup: Vec<AtomicU16>,
    /// Dense `MAX_BIGRAM_COLUMNS * words` bitset, lazily allocated on
    /// first record.
    col_data: OnceLock<Box<[AtomicU64]>>,
    /// Monotone allocator for column indices. Capped at
    /// `MAX_BIGRAM_COLUMNS`.
    next_column: AtomicU16,
    /// Words per column. `words = ceil(item_count / 64)`.
    words: usize,
    item_count: usize,
    /// Count of items that contributed content (i.e. items where
    /// `add_item_content` was called with non-empty content).
    populated: AtomicUsize,
}

impl BigramIndexBuilder {
    pub fn new(item_count: usize) -> Self {
        let words = item_count.div_ceil(64);
        let mut lookup: Vec<AtomicU16> = Vec::with_capacity(65536);
        lookup.resize_with(65536, || AtomicU16::new(NO_COLUMN));
        Self {
            lookup,
            col_data: OnceLock::new(),
            next_column: AtomicU16::new(0),
            words,
            item_count,
            populated: AtomicUsize::new(0),
        }
    }

    /// Lazily materialise the `MAX_BIGRAM_COLUMNS * words` atomic
    /// bitset on first use. `OnceLock` makes this safe under
    /// concurrent first-touch from many threads.
    #[inline(always)]
    fn col_data_slab(&self) -> &[AtomicU64] {
        self.col_data.get_or_init(|| {
            let total = MAX_BIGRAM_COLUMNS * self.words;
            let mut v: Vec<AtomicU64> = Vec::with_capacity(total);
            v.resize_with(total, || AtomicU64::new(0));
            v.into_boxed_slice()
        })
    }

    #[inline]
    fn get_or_alloc_column(&self, key: u16) -> u16 {
        let current = self.lookup[key as usize].load(Ordering::Relaxed);
        if current != NO_COLUMN {
            return current;
        }
        let new_col = self.next_column.fetch_add(1, Ordering::Relaxed);
        if new_col >= MAX_BIGRAM_COLUMNS as u16 {
            return NO_COLUMN;
        }
        match self.lookup[key as usize].compare_exchange(
            NO_COLUMN,
            new_col,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => new_col,
            // Another thread allocated first — use theirs. The column
            // we reserved with `fetch_add` is now permanently unused
            // but that's a leak of at most `n_threads * 65536` columns
            // worth of address space in the very worst case, which is
            // bounded by `MAX_BIGRAM_COLUMNS` anyway.
            Err(existing) => existing,
        }
    }

    /// Record every bigram in `content` for `item_idx` into `self`
    /// (consecutive bigrams) and `skip_builder` (skip-1 bigrams).
    ///
    /// Mirrors fff's `add_file_content`:
    /// * Two stack-local 1024-`u64` (8 KiB) bitsets dedup the
    ///   per-item bigram sets so the shared slab is touched at most
    ///   once per (item, bigram) pair, even if `content` repeats a
    ///   bigram many times.
    /// * Each input byte is normalised exactly once and carried via
    ///   `n0`/`n1` so it participates in up to three bigrams (as the
    ///   right side of one consecutive pair, then the left side of
    ///   another, then the left side of one skip-1 pair) without
    ///   redundant normalisation.
    pub fn add_item_content(
        &self,
        skip_builder: &BigramIndexBuilder,
        item_idx: usize,
        content: &[u8],
    ) {
        if content.len() < 2 {
            return;
        }

        debug_assert!(item_idx < self.item_count);
        let word_idx = item_idx / 64;
        let bit_mask = 1u64 << (item_idx % 64);

        // Stack-local dedup bitsets: 1024 × u64 = 8 KiB each, covering
        // all 65536 bigram keys. Has to fit in L1.
        let mut seen_consec = [0u64; 1024];
        let mut seen_skip = [0u64; 1024];

        let bytes = content;
        let len = bytes.len();

        let mut n0 = normalize_byte_scalar(bytes[0]);
        let mut n1 = normalize_byte_scalar(bytes[1]);

        if n0 != u16::MAX && n1 != u16::MAX {
            let key = (n0 << 8) | n1;
            self.record_bigram(&mut seen_consec, key, word_idx, bit_mask);
        }

        for &b in &bytes[2..len] {
            let cur = normalize_byte_scalar(b);
            if cur != u16::MAX {
                if n1 != u16::MAX {
                    let key = (n1 << 8) | cur;
                    self.record_bigram(&mut seen_consec, key, word_idx, bit_mask);
                }
                if n0 != u16::MAX {
                    let key = (n0 << 8) | cur;
                    skip_builder.record_bigram(&mut seen_skip, key, word_idx, bit_mask);
                }
            }
            n0 = n1;
            n1 = cur;
        }

        self.populated.fetch_add(1, Ordering::Relaxed);
        skip_builder.populated.fetch_add(1, Ordering::Relaxed);
    }

    /// Mark `key` as present for the item identified by
    /// `(word_idx, bit_mask)`. Per-item dedup via `seen` keeps the
    /// shared slab cold for repeated bigrams.
    #[inline(always)]
    fn record_bigram(&self, seen: &mut [u64; 1024], key: u16, word_idx: usize, bit_mask: u64) {
        let k = key as usize;
        let w = k >> 6;
        let bit = 1u64 << (k & 63);
        if seen[w] & bit == 0 {
            seen[w] |= bit;
            let col = self.get_or_alloc_column(key);
            if col != NO_COLUMN {
                let slab = self.col_data_slab();
                let idx = col as usize * self.words + word_idx;
                slab[idx].fetch_or(bit_mask, Ordering::Relaxed);
            }
        }
    }

    pub fn columns_used(&self) -> u16 {
        self.next_column
            .load(Ordering::Relaxed)
            .min(MAX_BIGRAM_COLUMNS as u16)
    }

    /// Consume the builder and emit a compact [`BigramIndex`].
    ///
    /// The density filter follows fff exactly:
    /// * Columns appearing in too few items are dropped. When
    ///   `min_density_pct` is `Some(p)`, "too few" means
    ///   `popcount * 100 < populated * p`. When `None`, the default
    ///   heuristic is `popcount * 4 < words * 8` (popcount ≥ words×2,
    ///   i.e. ≥ ~3.1% of items).
    /// * Columns appearing in ≥ 90% of items are dropped as well:
    ///   they're effectively all-ones and add AND cycles without
    ///   shrinking the candidate set.
    pub fn compress(self, min_density_pct: Option<u32>) -> BigramIndex {
        let cols = self.columns_used() as usize;
        let words = self.words;
        let item_count = self.item_count;
        let populated = self.populated.load(Ordering::Relaxed);
        let dense_bytes = words * 8; // bytes per dense column

        let old_lookup = self.lookup;
        let col_data = self.col_data.into_inner();

        let mut lookup: Vec<u16> = vec![NO_COLUMN; 65536];
        let mut dense_data: Vec<u64> = Vec::with_capacity(cols * words);
        let mut dense_count: usize = 0;

        if let Some(col_data) = col_data {
            for key in 0..65536usize {
                let old_col = old_lookup[key].load(Ordering::Relaxed);
                if old_col == NO_COLUMN || old_col as usize >= cols {
                    continue;
                }
                let col_start = old_col as usize * words;
                // Snapshot the atomic words into u64.
                let snapshot: Vec<u64> = (0..words)
                    .map(|i| col_data[col_start + i].load(Ordering::Relaxed))
                    .collect();

                let mut popcount = 0u32;
                for &word in &snapshot {
                    popcount += word.count_ones();
                }

                // Drop too-rare bigrams.
                let not_too_rare = if let Some(min_pct) = min_density_pct {
                    populated > 0 && (popcount as usize) * 100 >= populated * min_pct as usize
                } else {
                    (popcount as usize * 4) >= dense_bytes
                };
                if !not_too_rare {
                    continue;
                }

                // Drop ubiquitous bigrams (≥ 90% of items).
                if populated > 0 && (popcount as usize) * 10 >= populated * 9 {
                    continue;
                }

                let dense_idx = dense_count as u16;
                lookup[key] = dense_idx;
                dense_count += 1;
                dense_data.extend_from_slice(&snapshot);
            }
        }

        BigramIndex {
            lookup,
            dense_data,
            dense_count,
            words,
            item_count,
            populated,
            skip_index: None,
        }
    }
}

/// Compact, immutable bigram inverted index. Layout mirrors fff's
/// `BigramFilter` — see the module-level docs.
#[derive(Debug)]
pub struct BigramIndex {
    lookup: Vec<u16>,
    /// Flat buffer of all dense column data at fixed stride `words`.
    /// Column `i` lives at `i * words .. (i + 1) * words`. Must remain
    /// `u64`-aligned for the AND loop to auto-vectorise.
    dense_data: Vec<u64>,
    dense_count: usize,
    words: usize,
    item_count: usize,
    populated: usize,
    /// Optional skip-1 sub-index (stride-2 bigrams). ANDed into the
    /// candidate bitset on queries of length ≥ 3 to cut false
    /// positives.
    skip_index: Option<Box<BigramIndex>>,
}

/// SIMD-friendly bitwise AND of two equal-length bitsets. Kept
/// `#[inline]` and pointer-free so LLVM autovectorises the loop.
#[inline]
fn bitset_and(result: &mut [u64], bitset: &[u64]) {
    result
        .iter_mut()
        .zip(bitset.iter())
        .for_each(|(r, b)| *r &= *b);
}

impl BigramIndex {
    /// AND the posting lists for all bigrams of `pattern` (consecutive
    /// plus, if `pattern.len() >= 3` and a skip-1 sub-index is set,
    /// stride-2).
    ///
    /// Returns `None` if no bigram of `pattern` is tracked — in that
    /// case the caller must fall back to a full scan because the
    /// prefilter has no information. Otherwise the returned bitset has
    /// bit `i` set iff item `i` *might* contain `pattern`. The trailing
    /// bits above `item_count` are masked off so the caller never
    /// observes spurious out-of-range candidates.
    pub fn query(&self, pattern: &[u8]) -> Option<Vec<u64>> {
        if pattern.len() < 2 {
            return None;
        }

        let mut result = vec![u64::MAX; self.words];
        if self.words > 0 && !self.item_count.is_multiple_of(64) {
            let last = self.words - 1;
            result[last] = (1u64 << (self.item_count % 64)) - 1;
        }

        let words = self.words;
        let mut has_filter = false;

        let mut prev = pattern[0];
        for &b in &pattern[1..] {
            if (32..=126).contains(&prev) && (32..=126).contains(&b) {
                let key =
                    ((prev.to_ascii_lowercase() as u16) << 8) | b.to_ascii_lowercase() as u16;
                let col = self.lookup[key as usize];
                if col != NO_COLUMN {
                    let offset = col as usize * words;
                    let slice = &self.dense_data[offset..offset + words];
                    bitset_and(&mut result, slice);
                    has_filter = true;
                }
            }
            prev = b;
        }

        // Stride-2 (skip-1) bigrams.
        if let Some(skip) = &self.skip_index {
            if pattern.len() >= 3 {
                if let Some(skip_candidates) = skip.query_skip(pattern) {
                    bitset_and(&mut result, &skip_candidates);
                    has_filter = true;
                }
            }
        }

        if has_filter { Some(result) } else { None }
    }

    /// Sibling of `query` for stride-2 bigrams. Identical structure;
    /// kept separate so callers don't pay for the loop on patterns
    /// shorter than 3 bytes.
    fn query_skip(&self, pattern: &[u8]) -> Option<Vec<u64>> {
        let mut result = vec![u64::MAX; self.words];
        if self.words > 0 && !self.item_count.is_multiple_of(64) {
            let last = self.words - 1;
            result[last] = (1u64 << (self.item_count % 64)) - 1;
        }

        let words = self.words;
        let mut has_filter = false;

        for i in 0..pattern.len().saturating_sub(2) {
            let a = pattern[i];
            let b = pattern[i + 2];
            if (32..=126).contains(&a) && (32..=126).contains(&b) {
                let key =
                    ((a.to_ascii_lowercase() as u16) << 8) | b.to_ascii_lowercase() as u16;
                let col = self.lookup[key as usize];
                if col != NO_COLUMN {
                    let offset = col as usize * words;
                    let slice = &self.dense_data[offset..offset + words];
                    bitset_and(&mut result, slice);
                    has_filter = true;
                }
            }
        }

        if has_filter { Some(result) } else { None }
    }

    pub fn set_skip_index(&mut self, skip: BigramIndex) {
        self.skip_index = Some(Box::new(skip));
    }

    #[inline]
    pub fn is_candidate(candidates: &[u64], idx: usize) -> bool {
        let word = idx / 64;
        let bit = idx % 64;
        word < candidates.len() && candidates[word] & (1u64 << bit) != 0
    }

    pub fn count_candidates(candidates: &[u64]) -> usize {
        candidates.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn lookup(&self) -> &[u16] {
        &self.lookup
    }

    pub fn dense_data(&self) -> &[u64] {
        &self.dense_data
    }

    pub fn words(&self) -> usize {
        self.words
    }

    pub fn dense_count(&self) -> usize {
        self.dense_count
    }

    pub fn item_count(&self) -> usize {
        self.item_count
    }

    pub fn populated(&self) -> usize {
        self.populated
    }

    pub fn skip_index(&self) -> Option<&BigramIndex> {
        self.skip_index.as_deref()
    }

    /// Total heap bytes used by `self` (and the skip sub-index if any).
    /// Used by future `info` reporting.
    pub fn heap_bytes(&self) -> usize {
        let lookup_bytes = self.lookup.len() * std::mem::size_of::<u16>();
        let dense_bytes = self.dense_data.len() * std::mem::size_of::<u64>();
        let skip_bytes = self.skip_index.as_ref().map_or(0, |s| s.heap_bytes());
        lookup_bytes + dense_bytes + skip_bytes
    }

    /// Build an empty index — used for unit tests and as the
    /// degenerate-corpus default.
    pub fn empty() -> Self {
        Self {
            lookup: vec![NO_COLUMN; 65536],
            dense_data: Vec::new(),
            dense_count: 0,
            words: 0,
            item_count: 0,
            populated: 0,
            skip_index: None,
        }
    }
}

/// Extract the set of distinct printable-ASCII bigrams from `content`.
///
/// Uses an 8 KiB stack-equivalent bitset (`1024 * u64`) for dedup —
/// the same algorithm as fff's standalone `extract_bigrams`. Returns
/// the bigrams in the order they were first seen.
pub fn extract_bigrams(content: &[u8]) -> Vec<u16> {
    if content.len() < 2 {
        return Vec::new();
    }
    let mut seen = vec![0u64; 1024];
    let mut bigrams = Vec::new();

    let mut prev = content[0];
    for &b in &content[1..] {
        if (32..=126).contains(&prev) && (32..=126).contains(&b) {
            let key =
                ((prev.to_ascii_lowercase() as u16) << 8) | b.to_ascii_lowercase() as u16;
            let word = key as usize / 64;
            let bit = 1u64 << (key as usize % 64);
            if seen[word] & bit == 0 {
                seen[word] |= bit;
                bigrams.push(key);
            }
        }
        prev = b;
    }
    bigrams
}

/// Build a [`BigramIndex`] from `chunks` in parallel.
///
/// The consecutive-bigram index uses fff's default density heuristic
/// (`min_density_pct = None`); the skip-1 sub-index uses
/// `SKIP_INDEX_MIN_DENSITY_PCT = 12` — the value cited in the
/// deep-research report for shedding sparse skip columns without
/// losing meaningful filter power.
pub fn build_bigram_index_from_chunks(chunks: &[ChunkItem]) -> BigramIndex {
    if chunks.is_empty() {
        return BigramIndex::empty();
    }

    let builder = BigramIndexBuilder::new(chunks.len());
    let skip_builder = BigramIndexBuilder::new(chunks.len());

    // Parallel feed. The `AtomicU64::fetch_or` writes mean we don't
    // need to enforce word-aligned chunk sizes (cf. fff, which uses
    // an `UnsafeCell` slab and relies on `par_chunks` partitioning to
    // make plain `|=` race-free). Any positive chunk size is correct.
    chunks
        .par_chunks(BIGRAM_CHUNK_ITEMS)
        .enumerate()
        .for_each(|(chunk_idx, batch)| {
            let base_idx = chunk_idx * BIGRAM_CHUNK_ITEMS;
            for (offset, item) in batch.iter().enumerate() {
                let item_idx = base_idx + offset;
                let content: &[u8] = &item.text_norm_ascii;
                if content.len() < 2 {
                    continue;
                }
                builder.add_item_content(&skip_builder, item_idx, content);
            }
        });

    let mut index = builder.compress(None);
    let skip_index = skip_builder.compress(Some(SKIP_INDEX_MIN_DENSITY_PCT));
    index.set_skip_index(skip_index);
    index
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::ChunkItem;
    use std::sync::Arc;

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
            text_norm_ascii: Arc::<[u8]>::from(text.as_bytes()),
            preview: Arc::from(text),
            doc_mtime_ns: 0,
        }
    }

    fn bigram_key(a: u8, b: u8) -> u16 {
        ((a.to_ascii_lowercase() as u16) << 8) | b.to_ascii_lowercase() as u16
    }

    #[test]
    fn extract_bigrams_basic() {
        let bg = extract_bigrams(b"abc");
        // Both bigrams must be present; order is "first-seen".
        assert_eq!(bg.len(), 2);
        assert!(bg.contains(&bigram_key(b'a', b'b')));
        assert!(bg.contains(&bigram_key(b'b', b'c')));
        // First-seen order means "ab" precedes "bc".
        assert_eq!(bg[0], bigram_key(b'a', b'b'));
        assert_eq!(bg[1], bigram_key(b'b', b'c'));
    }

    #[test]
    fn extract_bigrams_short_input() {
        assert!(extract_bigrams(b"").is_empty());
        assert!(extract_bigrams(b"a").is_empty());
    }

    #[test]
    fn extract_bigrams_dedups() {
        // "ababab" has many repeats — each unique bigram appears once.
        let bg = extract_bigrams(b"ababab");
        assert_eq!(bg.len(), 2);
        assert!(bg.contains(&bigram_key(b'a', b'b')));
        assert!(bg.contains(&bigram_key(b'b', b'a')));
    }

    #[test]
    fn query_matches_only_chunk_with_token() {
        // Query for a token shared by exactly 3 of 200 chunks. With
        // populated = 200, words = ceil(200/64) = 4, popcount = 3:
        //   * density floor: 3 × 4 = 12 >= 4 × 8 = 32? No.
        //   * But the *skip-1* index uses a 12 % density threshold:
        //     3 × 100 = 300 >= 200 × 12 = 2400? No. Drops too.
        // So neither consec nor skip will track our 3-of-200 bigram.
        // Use a denser sharing: put the shared bigram in 60 of 200.
        let mut chunks: Vec<ChunkItem> = Vec::new();
        let target_text = "the quick brown fox kwjkwjkwj"; // distinctive
        chunks.push(mk_chunk(1, 1, target_text));
        // 100 chunks that share a "qkqlqmq" common bigram chain (≈ 50 %).
        for i in 0..100 {
            chunks.push(mk_chunk(
                (10 + i) as i64,
                (10 + i) as i64,
                "filler text qkqlqmq with shared bigrams",
            ));
        }
        // 100 more chunks that don't share any of those.
        for i in 0..100 {
            chunks.push(mk_chunk(
                (200 + i) as i64,
                (200 + i) as i64,
                "different padding here for ballast",
            ));
        }
        let idx = build_bigram_index_from_chunks(&chunks);
        // "qkql" has bigrams "qk", "kq", "ql" — each shared by 100 of
        // 201 chunks (chunks 1..=100), well within the density and
        // ubiquity bands. Only those 100 chunks should be candidates.
        let cand = idx.query(b"qkql").expect("qkql bigrams tracked");
        // Chunk 0 (the distinctive "kwjkwjkwj" target) should NOT be
        // a candidate.
        assert!(
            !BigramIndex::is_candidate(&cand, 0),
            "chunk 0 must not be a candidate for 'qkql'",
        );
        for i in 1..=100 {
            assert!(
                BigramIndex::is_candidate(&cand, i),
                "shared chunk {i} should be a candidate for 'qkql'",
            );
        }
        for i in 101..201 {
            assert!(
                !BigramIndex::is_candidate(&cand, i),
                "unrelated chunk {i} must not be a candidate for 'qkql'",
            );
        }
    }

    #[test]
    fn rare_token_returns_no_prefilter_info() {
        // A token unique to one chunk in a 200-chunk corpus has
        // popcount = 1 for each of its bigrams; the density floor
        // drops them. The prefilter then has no information and
        // `query` returns `None`, signalling "fall back to full scan".
        let mut chunks: Vec<ChunkItem> = Vec::new();
        chunks.push(mk_chunk(1, 1, "the unique token xylotomous"));
        for i in 0..200 {
            chunks.push(mk_chunk(
                (10 + i) as i64,
                (10 + i) as i64,
                "filler chunk without the unique token",
            ));
        }
        let idx = build_bigram_index_from_chunks(&chunks);
        // Even when query() returns Some (some bigrams of the pattern
        // happen to be common), the rare-bigram chunk's filter would
        // still be missing — but for "xylotomous" none of the bigrams
        // are common at all, so the prefilter has no information.
        match idx.query(b"xylotomous") {
            None => {
                // Correct: caller falls back to full scan.
            }
            Some(_) => panic!(
                "expected the prefilter to return None for a 1-of-201 rare token",
            ),
        }
    }

    #[test]
    fn query_unknown_returns_no_candidates() {
        // A bigram nobody contains: if the prefilter does track it
        // (which it usually won't, the density filter drops empty
        // columns) every candidate bit must be zero.
        let chunks: Vec<ChunkItem> = (0..200)
            .map(|i| mk_chunk(i as i64, i as i64, "filler text padding here for density"))
            .collect();
        let idx = build_bigram_index_from_chunks(&chunks);
        if let Some(cand) = idx.query(b"zq") {
            assert_eq!(BigramIndex::count_candidates(&cand), 0);
        }
        // If `None`, the caller would fall back to full scan — that's
        // also a valid outcome.
    }

    #[test]
    fn skip_one_catches_cross_byte() {
        // Verify the skip-1 sub-index records `content[i], content[i+2]`
        // as a bigram. We pad the corpus so the (q, q) skip bigram
        // exercised below survives the density and ubiquity filters:
        // it appears in 100 of 200 chunks via the "qkq" / "qlq" /
        // "qmq" runs.
        let mut chunks: Vec<ChunkItem> = Vec::new();
        for i in 0..100 {
            chunks.push(mk_chunk(
                i as i64,
                i as i64,
                "filler text qkqlqmq with shared bigrams",
            ));
        }
        for i in 100..200 {
            chunks.push(mk_chunk(
                i as i64,
                i as i64,
                "different padding here for ballast",
            ));
        }
        let idx = build_bigram_index_from_chunks(&chunks);
        // The pattern "qkq" has consec bigrams "qk", "kq" and skip-1
        // bigram (q, q). All three are populated in chunks 0..100.
        let cand = idx.query(b"qkq").expect("qkq bigrams tracked");
        for i in 0..100 {
            assert!(
                BigramIndex::is_candidate(&cand, i),
                "chunk {i} should be a candidate for 'qkq'",
            );
        }
        for i in 100..200 {
            assert!(
                !BigramIndex::is_candidate(&cand, i),
                "chunk {i} must not be a candidate for 'qkq'",
            );
        }
    }

    #[test]
    fn query_short_returns_none() {
        let chunks = vec![mk_chunk(1, 1, "ab")];
        let idx = build_bigram_index_from_chunks(&chunks);
        // Pattern length < 2: no bigram, no information.
        assert!(idx.query(b"").is_none());
        assert!(idx.query(b"a").is_none());
    }

    #[test]
    fn empty_corpus_builds_empty_index() {
        let idx = build_bigram_index_from_chunks(&[]);
        assert_eq!(idx.item_count(), 0);
        assert_eq!(idx.populated(), 0);
        assert_eq!(idx.dense_count(), 0);
        assert!(idx.query(b"abc").is_none());
    }

    #[test]
    fn populated_counts_each_item_once() {
        let chunks = vec![mk_chunk(1, 1, "hello world"), mk_chunk(2, 2, "another item")];
        let idx = build_bigram_index_from_chunks(&chunks);
        assert_eq!(idx.populated(), 2);
    }

    #[test]
    fn item_boundary_word_split() {
        // Items at indices 63 and 64 straddle the u64 word boundary in
        // the bitset. The build must place them into adjacent words
        // for the candidate lookup to address them correctly.
        //
        // To survive the density filter we need every "target" bigram
        // we'll query to:
        //   * appear in ≥ 2 of the 200 items (density floor: popcount
        //     × 4 ≥ words × 8 with words = ceil(200/64) = 4 means
        //     popcount ≥ 8);
        //   * appear in < 90 % of the 200 items (ubiquity cutoff).
        //
        // We embed "qkqlqmq" in every other chunk (≈ 50 items, well
        // between 8 and 180) and query for "qk" / "ql".
        let n = 200usize;
        let mut chunks: Vec<ChunkItem> = Vec::with_capacity(n);
        for i in 0..n {
            let text = if i % 2 == 0 {
                "filler text qkqlqmq with target bigrams"
            } else {
                "filler text padding here for density"
            };
            chunks.push(mk_chunk(i as i64, i as i64, text));
        }
        let idx = build_bigram_index_from_chunks(&chunks);
        let cand = idx.query(b"qkql").expect("qk/ql bigrams tracked");
        // Every even item (0, 2, ..., 198) — including the
        // word-boundary indices 62, 64 — should be a candidate.
        for i in (0..n).step_by(2) {
            assert!(
                BigramIndex::is_candidate(&cand, i),
                "item {i} should be a candidate for 'qkql'",
            );
        }
        // Odd items must not be candidates.
        for i in (1..n).step_by(2) {
            assert!(
                !BigramIndex::is_candidate(&cand, i),
                "odd item {i} must not be a candidate for 'qkql'",
            );
        }
    }
}
