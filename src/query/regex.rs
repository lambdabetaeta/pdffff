//! Regex search.

use anyhow::{Context, Result};

use crate::bigram_query::regex_to_bigram_query;
use crate::index::{ChunkItem, IndexState};

use super::Hit;
use super::candidate::{CandidateSet, MatchLocation};
use super::rank::cheap_rank;
use super::walk::{walk_base_chunks, walk_overflow};

pub(super) fn regex_search(
    state: &IndexState,
    pattern: &str,
    limit: usize,
) -> Result<Vec<Hit>> {
    if pattern.is_empty() || limit == 0 {
        return Ok(Vec::new());
    }

    // The bigram decomposition lowercases ASCII as it extracts;
    // `regex::RegexBuilder` is told to ignore case so the engine's
    // matching semantics agree with what the prefilter assumed. We
    // pass the pattern verbatim: lowercasing the raw pattern would
    // break character classes (e.g. `[A-Z]` becomes `[a-z]`) and is
    // wrong in general for regex syntax.
    let bq = regex_to_bigram_query(pattern);
    let regex = ::regex::RegexBuilder::new(pattern)
        .case_insensitive(true)
        .build()
        .with_context(|| format!("compiling regex {pattern:?}"))?;

    let base = state.load_base();
    let ov = state.overlay.read();

    let lookup = if bq.is_any() {
        None
    } else {
        base.bigrams.as_ref().and_then(|idx| bq.evaluate(idx))
    };
    let candidates = CandidateSet::from_bigram_lookup(lookup, base.chunks.len(), &ov);

    let verify = |chunk: &ChunkItem| {
        regex
            .find(&chunk.text_utf8)
            .map(|m| MatchLocation::Utf8 { offset: m.start(), match_len: m.len() })
    };

    let mut hits: Vec<Hit> = Vec::new();
    walk_base_chunks(&base, &ov, &candidates, limit, &mut hits, verify);

    // Overlay overflow: conservatively check every row. Regex bigrams
    // don't always survive overlay-side bigram dedup, so we let the
    // regex engine itself act as the verifier here. The overflow set
    // is bounded by the rebuild threshold, so the linear scan is
    // bounded too.
    if hits.len() < limit {
        let all_overflow: Vec<usize> = (0..ov.overflow.len()).collect();
        walk_overflow(&ov, all_overflow, limit, &mut hits, verify);
    }

    hits.sort_by_key(|h| (h.doc_id, h.page_no, h.chunk_id));
    // No phrase-rank for regex queries; cheap_rank's "more matched
    // terms" lever still helps if the user typed multi-word literal
    // fragments inside their regex. We pass the raw pattern so the
    // phrase / term split is at least defensible.
    cheap_rank(&mut hits, pattern);
    hits.truncate(limit);

    Ok(hits)
}
