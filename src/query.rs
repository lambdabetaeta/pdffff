//! Search engine over an [`IndexState`].
//!
//! Literal search runs in two stages:
//!
//! 1. **Candidate generation.** Normalize the query through
//!    `crate::normalize::normalize_query_ascii`, then ask the
//!    [`BigramIndex`] for a bitset of items that *might* contain the
//!    query. When the index has no information about the query (either
//!    the index is missing, the query is too short, or none of the
//!    query's bigrams survived compression), the candidate set falls
//!    back to "every active chunk".
//! 2. **Verification.** For each candidate, run a compiled
//!    `memchr::memmem::Finder` over `text_norm_ascii`. The finder is
//!    the source of truth for whether a chunk is a real hit; the
//!    bigram prefilter never decides hits, only narrows.
//!
//! Day 6 plumbs the same prefilter into regex / fuzzy modes.
//!
//! Snippet rendering is best-effort: the normalized bytes
//! (`text_norm_ascii`) do not byte-align with the original
//! `text_utf8` after deunicode + lowercase + whitespace collapse, so a
//! position in the norm cannot be mapped exactly back into the UTF-8
//! text. The strategy is:
//!
//! 1. Try a *proportional* mapping (norm_offset / norm_len ≈
//!    utf8_offset / utf8_len) to find a byte index in `text_utf8`.
//! 2. Snap to the nearest UTF-8 char boundary.
//! 3. Take a window of [`SNIPPET_CONTEXT_BYTES`] on each side, snapped
//!    to UTF-8 boundaries.
//! 4. Collapse whitespace runs to single spaces.
//!
//! The result *visually approximates* the match position but is not a
//! cryptographic alignment. The matched substring may or may not be
//! literally present in the snippet (the original text often has
//! ligatures or accents that the norm has flattened); we mark this as
//! a known limitation rather than working around it with hacks.

use anyhow::{Result, bail};
use memchr::memmem;
use tracing::warn;

use crate::bigram::BigramIndex;
use crate::index::{ChunkItem, IndexState};
use crate::normalize::normalize_query_ascii;

/// CLI / TUI cap on the number of hits surfaced to the user.
pub const DISPLAY_LIMIT: usize = 200;

/// Below this length the bigram prefilter has too little information to
/// be useful (only one or zero bigrams), so we fall back to a full
/// scan. Warn at that point so the user understands why a 1-byte
/// query is slow on a large corpus.
const NO_BIGRAM_FULLSCAN_WARN_LEN: usize = 2;

/// How many bytes of `text_utf8` to include on each side of the
/// approximate match offset when rendering a snippet.
const SNIPPET_CONTEXT_BYTES: usize = 60;

/// Which query engine to run. Only [`QueryMode::Literal`] is implemented
/// in Day 3; the others are stubs that error cleanly until Day 6 wires
/// them up via the bigram prefilter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryMode {
    Literal,
    Regex,
    Fuzzy,
}

/// One search result, ready to format for the CLI.
#[derive(Debug, Clone)]
pub struct Hit {
    pub chunk_id: i64,
    pub doc_id: i64,
    pub path: String,
    pub page_no: u32,
    pub score: f32,
    pub snippet: String,
}

/// Run `query` against the current `BaseIndex` snapshot.
///
/// Contract:
/// * Empty / whitespace-only queries return no hits.
/// * `QueryMode::Literal` narrows the candidate set with the bigram
///   prefilter (when available), then verifies each candidate with
///   `memchr::memmem`; results are returned in `(doc_id, chunk_ord)`
///   order, capped at `limit`.
/// * `QueryMode::Regex` and `QueryMode::Fuzzy` return an error — they
///   land in the Day-6 milestone.
pub fn search(
    state: &IndexState,
    query: &str,
    mode: QueryMode,
    limit: usize,
) -> Result<Vec<Hit>> {
    match mode {
        QueryMode::Literal => literal_search(state, query, limit),
        QueryMode::Regex | QueryMode::Fuzzy => {
            bail!("query mode {mode:?} is not implemented yet (planned for day 6)")
        }
    }
}

fn literal_search(state: &IndexState, query: &str, limit: usize) -> Result<Vec<Hit>> {
    let q = normalize_query_ascii(query);
    if q.is_empty() || limit == 0 {
        return Ok(Vec::new());
    }
    let needle = q.as_bytes();
    let finder = memmem::Finder::new(needle);

    let base = state.load_base();

    // Stage 1: candidate generation via the bigram prefilter.
    //
    // `bigrams.query()` returns `Some(bitset)` when at least one of
    // the query's bigrams was tracked. If `None`, the prefilter has
    // zero information and we must consider every chunk a candidate.
    let candidates: Option<Vec<u64>> = base
        .bigrams
        .as_ref()
        .and_then(|idx| idx.query(needle));

    if needle.len() < NO_BIGRAM_FULLSCAN_WARN_LEN {
        warn!(
            len = needle.len(),
            "literal query is too short for the bigram prefilter; falling back to full scan",
        );
    }

    // Stage 2: verification scan.
    let mut hits: Vec<Hit> = Vec::new();
    match candidates {
        Some(bitset) => {
            for (i, chunk) in base.chunks.iter().enumerate() {
                if !BigramIndex::is_candidate(&bitset, i) {
                    continue;
                }
                let Some(pos) = finder.find(&chunk.text_norm_ascii) else {
                    continue;
                };
                hits.push(make_hit(chunk, pos, needle.len()));
                if hits.len() >= limit {
                    break;
                }
            }
        }
        None => {
            for chunk in base.chunks.iter() {
                let Some(pos) = finder.find(&chunk.text_norm_ascii) else {
                    continue;
                };
                hits.push(make_hit(chunk, pos, needle.len()));
                if hits.len() >= limit {
                    break;
                }
            }
        }
    }
    Ok(hits)
}

fn make_hit(chunk: &ChunkItem, match_offset_in_norm: usize, query_len: usize) -> Hit {
    Hit {
        chunk_id: chunk.chunk_id,
        doc_id: chunk.doc_id,
        path: chunk.path.to_string(),
        page_no: chunk.page_no,
        // Day 3 has no real scoring — that's the fuzzy path on Day 6.
        // A constant 1.0 keeps the field meaningful without lying.
        score: 1.0,
        snippet: render_snippet(chunk, match_offset_in_norm, query_len),
    }
}

/// Build a short snippet around the approximate UTF-8 location of the
/// match. See the module-level docs for why this is best-effort.
pub fn render_snippet(
    chunk: &ChunkItem,
    match_offset_in_norm: usize,
    query_len: usize,
) -> String {
    let text = &*chunk.text_utf8;
    if text.is_empty() {
        return String::new();
    }

    let norm_len = chunk.text_norm_ascii.len();
    let approx_byte = if norm_len == 0 {
        0
    } else {
        // Proportional mapping: norm and utf8 share roughly the same
        // shape (whitespace runs and accents aside), so this gets us
        // within a few bytes of the right spot.
        let ratio = match_offset_in_norm as f64 / norm_len as f64;
        ((text.len() as f64) * ratio).round() as usize
    };
    let approx_byte = approx_byte.min(text.len());
    let center = snap_char_boundary(text, approx_byte);

    let want_match_end = (center + query_len).min(text.len());
    let left = center.saturating_sub(SNIPPET_CONTEXT_BYTES);
    let right = (want_match_end + SNIPPET_CONTEXT_BYTES).min(text.len());
    let left = snap_char_boundary(text, left);
    let right = snap_char_boundary(text, right);

    collapse_whitespace(&text[left..right])
}

/// Move `idx` left until it lies on a UTF-8 char boundary (or 0).
fn snap_char_boundary(text: &str, mut idx: usize) -> usize {
    if idx >= text.len() {
        return text.len();
    }
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = true;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{BaseIndex, ChunkItem, IndexState};
    use std::collections::HashMap;
    use std::sync::Arc;

    fn synthetic_state(chunks: Vec<ChunkItem>) -> IndexState {
        let mut doc_ranges: HashMap<i64, std::ops::Range<usize>> = HashMap::new();
        let mut cur: Option<(i64, usize)> = None;
        for (i, c) in chunks.iter().enumerate() {
            if cur.map(|(d, _)| d) != Some(c.doc_id) {
                if let Some((d, s)) = cur {
                    doc_ranges.insert(d, s..i);
                }
                cur = Some((c.doc_id, i));
            }
        }
        if let Some((d, s)) = cur {
            doc_ranges.insert(d, s..chunks.len());
        }
        // Build the bigram prefilter from the same chunks so unit
        // tests exercise the candidate-generation path (not just the
        // fallback). Empty corpora skip it because no chunks means no
        // candidates anyway.
        let bigrams = if chunks.is_empty() {
            None
        } else {
            Some(Arc::new(crate::bigram::build_bigram_index_from_chunks(
                &chunks,
            )))
        };
        let base = BaseIndex {
            chunks: Arc::new(chunks),
            doc_ranges,
            bigrams,
            built_at_ms: 0,
        };
        IndexState::new(base)
    }

    fn mk_chunk(id: i64, doc: i64, path: &str, page: u32, text: &str) -> ChunkItem {
        ChunkItem {
            chunk_id: id,
            doc_id: doc,
            path: Arc::from(path),
            filename: Arc::from(
                std::path::Path::new(path)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
                    .as_str(),
            ),
            page_no: page,
            chunk_ord: 0,
            char_start: 0,
            char_end: text.len() as u32,
            text_utf8: Arc::from(text),
            text_norm_ascii: Arc::<[u8]>::from(
                crate::normalize::normalize_for_index(text).as_bytes(),
            ),
            preview: Arc::from(text),
            doc_mtime_ns: 0,
        }
    }

    #[test]
    fn empty_query_returns_no_hits() {
        let state =
            synthetic_state(vec![mk_chunk(1, 1, "/a.pdf", 1, "the quick brown fox")]);
        let hits = search(&state, "", QueryMode::Literal, 10).unwrap();
        assert!(hits.is_empty());
        let hits = search(&state, "   ", QueryMode::Literal, 10).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn literal_finds_known_token() {
        let state = synthetic_state(vec![
            mk_chunk(1, 1, "/a.pdf", 1, "the quick brown fox"),
            mk_chunk(2, 2, "/b.pdf", 4, "no match here"),
            mk_chunk(3, 3, "/c.pdf", 7, "another quick result"),
        ]);
        let hits = search(&state, "quick", QueryMode::Literal, 10).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].path, "/a.pdf");
        assert_eq!(hits[0].page_no, 1);
        assert_eq!(hits[1].path, "/c.pdf");
        assert_eq!(hits[1].page_no, 7);
        assert!(hits[0].snippet.contains("quick"));
    }

    #[test]
    fn unknown_token_returns_no_hits() {
        let state = synthetic_state(vec![
            mk_chunk(1, 1, "/a.pdf", 1, "the quick brown fox"),
            mk_chunk(2, 2, "/b.pdf", 1, "second"),
        ]);
        let hits = search(&state, "zebra", QueryMode::Literal, 10).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn limit_caps_hits() {
        let chunks: Vec<ChunkItem> = (0..10)
            .map(|i| mk_chunk(i, i, "/c.pdf", 1, "matches matches matches"))
            .collect();
        let state = synthetic_state(chunks);
        let hits = search(&state, "matches", QueryMode::Literal, 3).unwrap();
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn case_and_unicode_are_normalized() {
        let state = synthetic_state(vec![mk_chunk(
            1,
            1,
            "/a.pdf",
            1,
            "Café résumé in mixed Case",
        )]);
        let hits = search(&state, "RESUME", QueryMode::Literal, 10).unwrap();
        assert_eq!(hits.len(), 1);
        let hits = search(&state, "café", QueryMode::Literal, 10).unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn regex_and_fuzzy_modes_are_unimplemented() {
        let state = synthetic_state(vec![mk_chunk(1, 1, "/a.pdf", 1, "anything")]);
        assert!(search(&state, "x", QueryMode::Regex, 10).is_err());
        assert!(search(&state, "x", QueryMode::Fuzzy, 10).is_err());
    }

    #[test]
    fn snippet_snaps_to_char_boundaries() {
        // Multi-byte UTF-8 around the approximate match offset must not
        // panic and must produce valid UTF-8.
        let chunk = mk_chunk(1, 1, "/a.pdf", 1, "αβγδε needle αβγδε");
        let state = synthetic_state(vec![chunk]);
        let hits = search(&state, "needle", QueryMode::Literal, 10).unwrap();
        assert_eq!(hits.len(), 1);
        // Snippet must be valid UTF-8 by construction.
        let _ = hits[0].snippet.as_str();
    }
}
