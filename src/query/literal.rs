//! Literal substring search.

use anyhow::Result;
use memchr::memmem;
use tracing::warn;

use crate::bigram::extract_bigrams;
use crate::index::{ChunkItem, IndexState};
use crate::normalize::normalize_query_ascii;

use super::Hit;
use super::candidate::{CandidateSet, MatchLocation};
use super::rank::cheap_rank;
use super::walk::{walk_base_chunks, walk_overflow};

/// Below this length the bigram prefilter has too little information to
/// be useful (only one or zero bigrams), so we fall back to a full
/// scan. Warn at that point so the user understands why a 1-byte
/// query is slow on a large corpus.
const NO_BIGRAM_FULLSCAN_WARN_LEN: usize = 2;

pub(super) fn literal_search(
    state: &IndexState,
    query: &str,
    limit: usize,
) -> Result<Vec<Hit>> {
    let q = normalize_query_ascii(query);
    if q.is_empty() || limit == 0 {
        return Ok(Vec::new());
    }
    let needle = q.as_bytes();
    let finder = memmem::Finder::new(needle);

    let base = state.load_base();
    let ov = state.overlay.read();

    if needle.len() < NO_BIGRAM_FULLSCAN_WARN_LEN {
        warn!(
            len = needle.len(),
            "literal query is too short for the bigram prefilter; falling back to full scan",
        );
    }

    let lookup = base.bigrams.as_ref().and_then(|idx| idx.query(needle));
    let candidates = CandidateSet::from_bigram_lookup(lookup, base.chunks.len(), &ov);

    let verify = |chunk: &ChunkItem| {
        finder
            .find(&chunk.text_norm_ascii)
            .map(|offset| MatchLocation::Norm { offset, query_len: needle.len() })
    };

    let mut hits: Vec<Hit> = Vec::new();
    walk_base_chunks(&base, &ov, &candidates, limit, &mut hits, &verify);

    if hits.len() < limit && !ov.overflow.is_empty() {
        let query_bigrams = extract_bigrams(needle);
        walk_overflow(
            &ov,
            ov.overflow_matches(&query_bigrams),
            limit,
            &mut hits,
            &verify,
        );
    }

    // Stable (doc_id, page, chunk_id) ordering — needed when overflow
    // and base both contributed. Then run the cheap deterministic
    // ranker so the most-relevant hits land at the top.
    hits.sort_by_key(|h| (h.doc_id, h.page_no, h.chunk_id));
    cheap_rank(&mut hits, &q);
    hits.truncate(limit);

    Ok(hits)
}
