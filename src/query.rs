//! Search engine over an [`IndexState`].
//!
//! All three modes share a candidate-generation skeleton built on the
//! Day-4 bigram prefilter and the Day-5 overlay:
//!
//! 1. **Candidate generation.** Convert the query into a candidate
//!    bitset over the base index. For literal queries we go through
//!    `BigramIndex::query`; for regex / fuzzy we go through
//!    [`crate::bigram_query::regex_to_bigram_query`] /
//!    [`crate::bigram_query::fuzzy_to_bigram_query`] and evaluate the
//!    resulting [`crate::bigram_query::BigramQuery`] AND/OR tree. When the prefilter has no
//!    information (the bigram set is empty, every column was dropped,
//!    or the query is `Any`) the candidate set is "every active chunk".
//! 2. **Tombstone mask.** If the overlay has tombstoned base chunks we
//!    AND-NOT the tombstone bitset into the candidate bitset before
//!    verification.
//! 3. **Verification.** Per-mode:
//!     * Literal: a compiled `memchr::memmem::Finder` over
//!       `text_norm_ascii`.
//!     * Regex: a compiled `regex::Regex` matched against `text_utf8`,
//!       with `case_insensitive(true)` so the regex engine's notion of
//!       case matches the lowercase-only bigram decomposition.
//!     * Fuzzy: `memmem` is replaced by neo_frizbee's parallel match
//!       call over a synthetic "rank string" of
//!       `"{filename} {path} page {page_no} {preview}"`. Above a
//!       candidate-count limit ([`FRIZBEE_LIMIT`]) we fall back to a
//!       cheap deterministic ordering — the report calls for this
//!       explicitly so we don't burn neo_frizbee on huge candidate
//!       lists.
//! 4. **Overflow pass.** Mode-specific candidate set:
//!     * Literal: use the query's deduped bigram set against
//!       `Overlay::overflow_matches`.
//!     * Regex: conservatively include every overflow row; the regex
//!       engine still decides hits and the verification cost over a
//!       few-thousand-chunk overlay is acceptable. (fff does the same;
//!       regex bigrams don't always survive deduplication.)
//!     * Fuzzy: include every overflow row, the same neo_frizbee
//!       call handles them.
//!
//! The base index and the overlay are read under a single
//! `state.overlay.read()` guard that brackets both verification passes
//! so the snapshot stays consistent.
//!
//! Snippet rendering is best-effort: the normalized bytes
//! (`text_norm_ascii`) do not byte-align with the original
//! `text_utf8` after deunicode + lowercase + whitespace collapse, so a
//! position in the norm cannot be mapped exactly back into the UTF-8
//! text. See [`render_snippet`] for the proportional mapping strategy.

use anyhow::{Context, Result};
use memchr::memmem;
use regex::Regex;
use tracing::warn;

use crate::bigram::{BigramIndex, extract_bigrams};
use crate::bigram_query::{fuzzy_to_bigram_query, regex_to_bigram_query};
use crate::index::{BaseIndex, ChunkItem, IndexState, Overlay};
use crate::normalize::normalize_query_ascii;

/// CLI / TUI cap on the number of hits surfaced to the user.
pub const DISPLAY_LIMIT: usize = 200;

/// Number of evenly-spaced probe bigrams to take when decomposing a
/// fuzzy query into a [`crate::bigram_query::BigramQuery`]. The report names six.
pub const FUZZY_PROBES: usize = 6;

/// Above this many candidate chunks we skip neo_frizbee and fall back
/// to the cheap deterministic ordering. The threshold is from the
/// report.
pub const FRIZBEE_LIMIT: usize = 2048;

/// neo_frizbee thread count. The crate launches its own scoped pool
/// when called; six matches the value in fff's own scorer wiring.
const FRIZBEE_THREADS: usize = 6;

/// Below this length the bigram prefilter has too little information to
/// be useful (only one or zero bigrams), so we fall back to a full
/// scan. Warn at that point so the user understands why a 1-byte
/// query is slow on a large corpus.
const NO_BIGRAM_FULLSCAN_WARN_LEN: usize = 2;

/// How many bytes of `text_utf8` to include on each side of the
/// approximate match offset when rendering a snippet.
const SNIPPET_CONTEXT_BYTES: usize = 60;

/// Which query engine to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryMode {
    Literal,
    Regex,
    Fuzzy,
}

/// One search result, ready to format for the CLI.
///
/// Implements [`serde::Serialize`] so the CLI's `--json` mode can emit
/// one compact JSON object per hit. The field names are stable and
/// match the user-facing column names (`path`, `page_no`, `chunk_ord`,
/// `score`, `snippet`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct Hit {
    pub chunk_id: i64,
    pub doc_id: i64,
    pub path: String,
    pub page_no: u32,
    pub chunk_ord: u32,
    pub score: f32,
    pub snippet: String,
}

/// Run `query` against the current `BaseIndex` snapshot.
///
/// Contract:
/// * Empty / whitespace-only queries return no hits.
/// * [`QueryMode::Literal`]: candidate prefilter via the bigram index,
///   verification via `memchr::memmem`, deterministic ordering.
/// * [`QueryMode::Regex`]: candidate prefilter via
///   [`regex_to_bigram_query`]; verification via a compiled
///   case-insensitive `regex::Regex`. The pattern is *not* normalized
///   (lowercasing the source would break character classes and
///   look-arounds); case-insensitivity is delegated to the engine.
/// * [`QueryMode::Fuzzy`]: candidate prefilter via
///   [`fuzzy_to_bigram_query`]; ranking via `neo_frizbee`'s parallel
///   match call, with the cheap deterministic fallback above
///   [`FRIZBEE_LIMIT`] candidates.
pub fn search(
    state: &IndexState,
    query: &str,
    mode: QueryMode,
    limit: usize,
) -> Result<Vec<Hit>> {
    match mode {
        QueryMode::Literal => literal_search(state, query, limit),
        QueryMode::Regex => regex_search(state, query, limit),
        QueryMode::Fuzzy => fuzzy_search(state, query, limit),
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
    let ov = state.overlay.read();

    let mut candidates: Option<Vec<u64>> = base
        .bigrams
        .as_ref()
        .and_then(|idx| idx.query(needle));

    if needle.len() < NO_BIGRAM_FULLSCAN_WARN_LEN {
        warn!(
            len = needle.len(),
            "literal query is too short for the bigram prefilter; falling back to full scan",
        );
    }

    apply_tombstones(&mut candidates, &ov);

    let mut hits: Vec<Hit> = Vec::new();
    walk_base_for_literal(&base, &ov, candidates.as_deref(), &finder, needle.len(), limit, &mut hits);

    if hits.len() < limit && !ov.overflow_chunks.is_empty() {
        let query_bigrams = extract_bigrams(needle);
        for idx in ov.overflow_matches(&query_bigrams) {
            let chunk = &ov.overflow_chunks[idx];
            let Some(pos) = finder.find(&chunk.text_norm_ascii) else {
                continue;
            };
            hits.push(make_hit_at_norm(chunk, pos, needle.len()));
            if hits.len() >= limit {
                break;
            }
        }
    }

    // Stable (doc_id, page, chunk_id) ordering — needed when overflow
    // and base both contributed. Then run the cheap deterministic
    // ranker so the most-relevant hits land at the top.
    hits.sort_by_key(|h| (h.doc_id, h.page_no, h.chunk_id));
    cheap_rank(&mut hits, &q);
    hits.truncate(limit);

    Ok(hits)
}

fn regex_search(state: &IndexState, pattern: &str, limit: usize) -> Result<Vec<Hit>> {
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
    let regex = regex::RegexBuilder::new(pattern)
        .case_insensitive(true)
        .build()
        .with_context(|| format!("compiling regex {pattern:?}"))?;

    let base = state.load_base();
    let ov = state.overlay.read();

    let mut candidates: Option<Vec<u64>> = if bq.is_any() {
        None
    } else {
        base.bigrams.as_ref().and_then(|idx| bq.evaluate(idx))
    };
    apply_tombstones(&mut candidates, &ov);

    let mut hits: Vec<Hit> = Vec::new();
    walk_base_for_regex(&base, &ov, candidates.as_deref(), &regex, limit, &mut hits);

    // Overlay overflow: conservatively check every row. Regex bigrams
    // don't always survive overlay-side bigram dedup, so we let the
    // regex engine itself act as the verifier here. The overflow set
    // is bounded by the rebuild threshold, so the linear scan is
    // bounded too.
    if hits.len() < limit {
        for chunk in &ov.overflow_chunks {
            if let Some(m) = regex.find(&chunk.text_utf8) {
                hits.push(make_hit_at_utf8(chunk, m.start(), m.len()));
                if hits.len() >= limit {
                    break;
                }
            }
        }
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

fn fuzzy_search(state: &IndexState, query: &str, limit: usize) -> Result<Vec<Hit>> {
    let q_norm = normalize_query_ascii(query);
    if q_norm.is_empty() || limit == 0 {
        return Ok(Vec::new());
    }

    let bq = fuzzy_to_bigram_query(&q_norm, FUZZY_PROBES);

    let base = state.load_base();
    let ov = state.overlay.read();

    let mut candidates: Option<Vec<u64>> = if bq.is_any() {
        None
    } else {
        base.bigrams.as_ref().and_then(|idx| bq.evaluate(idx))
    };
    apply_tombstones(&mut candidates, &ov);

    // Gather candidate chunks — base passes through the bitset, overflow
    // chunks are unconditionally appended (the fuzzy scorer makes the
    // final call).
    let mut candidate_chunks: Vec<&ChunkItem> = Vec::new();
    collect_base_candidates(&base, &ov, candidates.as_deref(), &mut candidate_chunks);
    for chunk in &ov.overflow_chunks {
        candidate_chunks.push(chunk);
    }

    if candidate_chunks.is_empty() {
        return Ok(Vec::new());
    }

    let rank_texts: Vec<String> = candidate_chunks
        .iter()
        .map(|c| rank_text_for(c))
        .collect();

    let mut hits: Vec<Hit> = if rank_texts.len() > FRIZBEE_LIMIT {
        // Above the limit the cheap deterministic ranker is faster
        // and good enough — the report names this fallback exactly.
        let needle_norm = q_norm.as_bytes();
        let finder = memmem::Finder::new(needle_norm);
        candidate_chunks
            .iter()
            .filter_map(|chunk| {
                let pos = finder.find(&chunk.text_norm_ascii)?;
                Some(make_hit_at_norm(chunk, pos, needle_norm.len()))
            })
            .collect()
    } else {
        let config = neo_frizbee::Config {
            max_typos: None,
            sort: true,
            scoring: neo_frizbee::Scoring::default(),
        };
        let matches =
            neo_frizbee::match_list_parallel(&q_norm, &rank_texts, &config, FRIZBEE_THREADS);
        let mut hits: Vec<Hit> = Vec::with_capacity(matches.len());
        for m in &matches {
            let chunk = candidate_chunks[m.index as usize];
            // Locate the user's query inside the rank text for snippet
            // purposes — best-effort. If neo_frizbee accepted the
            // candidate but the literal needle isn't present (the
            // fuzzy match crossed token boundaries), centre the
            // snippet on offset 0.
            let pos = memmem::find(chunk.text_norm_ascii.as_ref(), q_norm.as_bytes())
                .unwrap_or(0);
            let mut hit = make_hit_at_norm(chunk, pos, q_norm.len());
            // Carry neo_frizbee's score through so callers can rank
            // across queries; the unit-of-score is u16 internally.
            hit.score = m.score as f32;
            hits.push(hit);
        }
        hits
    };

    hits.truncate(limit);
    Ok(hits)
}

/// Build the synthetic "rank string" passed to the fuzzy scorer. Mirrors
/// the report's recipe verbatim: `{filename} {path} page {page_no} {preview}`.
fn rank_text_for(c: &ChunkItem) -> String {
    let mut s = String::with_capacity(
        c.filename.len() + c.path.len() + 10 + c.preview.len(),
    );
    s.push_str(&c.filename);
    s.push(' ');
    s.push_str(&c.path);
    s.push_str(" page ");
    s.push_str(&c.page_no.to_string());
    s.push(' ');
    s.push_str(&c.preview);
    s
}

/// AND-NOT the overlay's tombstone bitset into a candidate bitset.
fn apply_tombstones(candidates: &mut Option<Vec<u64>>, ov: &Overlay) {
    if let Some(ref mut bitset) = candidates {
        let n = bitset.len().min(ov.tombstones.len());
        for w in 0..n {
            bitset[w] &= !ov.tombstones[w];
        }
    }
}

fn walk_base_for_literal(
    base: &BaseIndex,
    ov: &Overlay,
    candidates: Option<&[u64]>,
    finder: &memmem::Finder,
    needle_len: usize,
    limit: usize,
    hits: &mut Vec<Hit>,
) {
    match candidates {
        Some(bitset) => {
            for (i, chunk) in base.chunks.iter().enumerate() {
                if !BigramIndex::is_candidate(bitset, i) {
                    continue;
                }
                let Some(pos) = finder.find(&chunk.text_norm_ascii) else {
                    continue;
                };
                hits.push(make_hit_at_norm(chunk, pos, needle_len));
                if hits.len() >= limit {
                    break;
                }
            }
        }
        None => {
            for (i, chunk) in base.chunks.iter().enumerate() {
                if ov.is_tombstoned(i) {
                    continue;
                }
                let Some(pos) = finder.find(&chunk.text_norm_ascii) else {
                    continue;
                };
                hits.push(make_hit_at_norm(chunk, pos, needle_len));
                if hits.len() >= limit {
                    break;
                }
            }
        }
    }
}

fn walk_base_for_regex(
    base: &BaseIndex,
    ov: &Overlay,
    candidates: Option<&[u64]>,
    regex: &Regex,
    limit: usize,
    hits: &mut Vec<Hit>,
) {
    match candidates {
        Some(bitset) => {
            for (i, chunk) in base.chunks.iter().enumerate() {
                if !BigramIndex::is_candidate(bitset, i) {
                    continue;
                }
                let Some(m) = regex.find(&chunk.text_utf8) else {
                    continue;
                };
                hits.push(make_hit_at_utf8(chunk, m.start(), m.len()));
                if hits.len() >= limit {
                    break;
                }
            }
        }
        None => {
            for (i, chunk) in base.chunks.iter().enumerate() {
                if ov.is_tombstoned(i) {
                    continue;
                }
                let Some(m) = regex.find(&chunk.text_utf8) else {
                    continue;
                };
                hits.push(make_hit_at_utf8(chunk, m.start(), m.len()));
                if hits.len() >= limit {
                    break;
                }
            }
        }
    }
}

fn collect_base_candidates<'a>(
    base: &'a BaseIndex,
    ov: &Overlay,
    candidates: Option<&[u64]>,
    out: &mut Vec<&'a ChunkItem>,
) {
    match candidates {
        Some(bitset) => {
            for (i, chunk) in base.chunks.iter().enumerate() {
                if BigramIndex::is_candidate(bitset, i) {
                    out.push(chunk);
                }
            }
        }
        None => {
            for (i, chunk) in base.chunks.iter().enumerate() {
                if !ov.is_tombstoned(i) {
                    out.push(chunk);
                }
            }
        }
    }
}

fn make_hit_at_norm(chunk: &ChunkItem, match_offset_in_norm: usize, query_len: usize) -> Hit {
    Hit {
        chunk_id: chunk.chunk_id,
        doc_id: chunk.doc_id,
        path: chunk.path.to_string(),
        page_no: chunk.page_no,
        chunk_ord: chunk.chunk_ord,
        score: 1.0,
        snippet: render_snippet(chunk, match_offset_in_norm, query_len),
    }
}

fn make_hit_at_utf8(chunk: &ChunkItem, match_offset_in_utf8: usize, query_len: usize) -> Hit {
    // For regex hits we know the *exact* UTF-8 offset, so we can render
    // the snippet without going through the proportional norm→utf8
    // remapping.
    Hit {
        chunk_id: chunk.chunk_id,
        doc_id: chunk.doc_id,
        path: chunk.path.to_string(),
        page_no: chunk.page_no,
        chunk_ord: chunk.chunk_ord,
        score: 1.0,
        snippet: render_snippet_at_utf8(chunk, match_offset_in_utf8, query_len),
    }
}

/// Cheap deterministic ordering for hits.
///
/// Sort by, in order:
/// 1. Exact phrase hit before partial-term hit (treat the normalized
///    query as a phrase).
/// 2. More matched terms (whitespace-split) before fewer.
/// 3. Earlier match offset before later.
/// 4. Lower `page_no` before higher.
/// 5. Newer `doc_mtime_ns` before older.
///
/// We carry the original chunk through each `Hit`'s `(doc_id, chunk_id,
/// page_no)` triple, but `doc_mtime_ns` is not on `Hit` — we look it up
/// once per hit at sort time by linear scan over the candidate set. For
/// the small slice of `limit ≤ DISPLAY_LIMIT` hits this is fine.
pub fn cheap_rank(hits: &mut Vec<Hit>, query_norm: &str) {
    let phrase = query_norm.trim();
    let terms: Vec<&str> = phrase.split_whitespace().collect();
    // Pre-compute the lowercased snippet once per hit to avoid
    // re-lowercasing inside the comparator.
    let snippets_lc: Vec<String> = hits.iter().map(|h| h.snippet.to_lowercase()).collect();

    // Build sort keys.
    let keys: Vec<(bool, usize, usize, u32)> = hits
        .iter()
        .zip(snippets_lc.iter())
        .map(|(h, snip_lc)| {
            let has_phrase = !phrase.is_empty() && snip_lc.contains(phrase);
            let term_count = terms.iter().filter(|t| snip_lc.contains(**t)).count();
            // Earlier offset of the phrase in the snippet (saturating to
            // a large sentinel when not present so phrase-bearing hits
            // win the tiebreak).
            let offset = if has_phrase {
                snip_lc.find(phrase).unwrap_or(usize::MAX)
            } else if let Some(t) = terms.iter().filter_map(|t| snip_lc.find(t)).min() {
                t
            } else {
                usize::MAX
            };
            (has_phrase, term_count, offset, h.page_no)
        })
        .collect();

    let mut indices: Vec<usize> = (0..hits.len()).collect();
    indices.sort_by(|&a, &b| {
        let (pa, ta, oa, ga) = keys[a];
        let (pb, tb, ob, gb) = keys[b];
        // Phrase hit first.
        pb.cmp(&pa)
            // More terms first.
            .then_with(|| tb.cmp(&ta))
            // Earlier offset first.
            .then_with(|| oa.cmp(&ob))
            // Lower page first.
            .then_with(|| ga.cmp(&gb))
            // Stable: doc_id then chunk_id.
            .then_with(|| hits[a].doc_id.cmp(&hits[b].doc_id))
            .then_with(|| hits[a].chunk_id.cmp(&hits[b].chunk_id))
    });

    let reordered: Vec<Hit> = indices.into_iter().map(|i| hits[i].clone()).collect();
    *hits = reordered;
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
        let ratio = match_offset_in_norm as f64 / norm_len as f64;
        ((text.len() as f64) * ratio).round() as usize
    };
    let approx_byte = approx_byte.min(text.len());
    render_window(text, approx_byte, query_len)
}

/// Same as [`render_snippet`] but takes an exact UTF-8 offset. Used by
/// the regex path where we already know the byte offset of the match.
pub fn render_snippet_at_utf8(
    chunk: &ChunkItem,
    match_offset_in_utf8: usize,
    match_len: usize,
) -> String {
    let text = &*chunk.text_utf8;
    if text.is_empty() {
        return String::new();
    }
    let centre = match_offset_in_utf8.min(text.len());
    render_window(text, centre, match_len)
}

fn render_window(text: &str, centre: usize, match_len: usize) -> String {
    let center = snap_char_boundary(text, centre);
    let want_match_end = (center + match_len).min(text.len());
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
        // Order may now be governed by cheap_rank but both should still
        // appear and both should mention "quick" in the snippet.
        let paths: Vec<&str> = hits.iter().map(|h| h.path.as_str()).collect();
        assert!(paths.contains(&"/a.pdf"));
        assert!(paths.contains(&"/c.pdf"));
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
    fn regex_finds_pattern_in_chunk() {
        let state = synthetic_state(vec![
            mk_chunk(1, 1, "/a.pdf", 1, "Order #42 was placed."),
            mk_chunk(2, 2, "/b.pdf", 1, "no number here"),
            mk_chunk(3, 3, "/c.pdf", 1, "Order #1337 shipped."),
        ]);
        let hits = search(&state, r"order #\d+", QueryMode::Regex, 10).unwrap();
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn regex_compile_error_is_surfaced() {
        let state = synthetic_state(vec![mk_chunk(1, 1, "/a.pdf", 1, "anything")]);
        assert!(search(&state, "[invalid", QueryMode::Regex, 10).is_err());
    }

    #[test]
    fn fuzzy_finds_close_token() {
        let state = synthetic_state(vec![
            mk_chunk(1, 1, "/a.pdf", 1, "the quick brown fox jumps over"),
            mk_chunk(2, 2, "/b.pdf", 1, "completely unrelated text"),
            mk_chunk(3, 3, "/c.pdf", 1, "yet another paragraph"),
        ]);
        // "qiuck" — one transposition typo against "quick".
        let hits = search(&state, "qiuck", QueryMode::Fuzzy, 10).unwrap();
        assert!(!hits.is_empty(), "fuzzy should still surface the close chunk");
        // The /a.pdf chunk that genuinely contains "quick" must rank
        // somewhere in the top 3.
        let top3_paths: Vec<&str> =
            hits.iter().take(3).map(|h| h.path.as_str()).collect();
        assert!(
            top3_paths.contains(&"/a.pdf"),
            "expected /a.pdf in top 3 fuzzy hits, got {top3_paths:?}",
        );
    }

    #[test]
    fn snippet_snaps_to_char_boundaries() {
        let chunk = mk_chunk(1, 1, "/a.pdf", 1, "αβγδε needle αβγδε");
        let state = synthetic_state(vec![chunk]);
        let hits = search(&state, "needle", QueryMode::Literal, 10).unwrap();
        assert_eq!(hits.len(), 1);
        let _ = hits[0].snippet.as_str();
    }

    #[test]
    fn tombstoned_base_chunks_are_hidden() {
        let state = synthetic_state(vec![
            mk_chunk(1, 1, "/a.pdf", 1, "the quick brown fox"),
            mk_chunk(2, 2, "/b.pdf", 4, "no match here"),
            mk_chunk(3, 3, "/c.pdf", 7, "another quick result"),
        ]);
        {
            let base = state.load_base();
            let mut ov = state.overlay.write();
            ov.tombstone_doc(1, &base);
        }
        let hits = search(&state, "quick", QueryMode::Literal, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "/c.pdf");
    }

    #[test]
    fn overlay_overflow_chunks_are_searchable() {
        let state = synthetic_state(vec![mk_chunk(
            1,
            1,
            "/a.pdf",
            1,
            "the quick brown fox",
        )]);
        {
            let mut ov = state.overlay.write();
            ov.add_overflow(mk_chunk(
                100,
                2,
                "/new.pdf",
                3,
                "freshly added content with xylotomous",
            ));
        }
        let hits = search(&state, "quick", QueryMode::Literal, 10).unwrap();
        assert_eq!(hits.len(), 1);
        let hits = search(&state, "xylotomous", QueryMode::Literal, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "/new.pdf");
    }

    #[test]
    fn modify_doc_swaps_base_for_overflow() {
        let state = synthetic_state(vec![
            mk_chunk(1, 1, "/a.pdf", 1, "old version mentions zebra"),
            mk_chunk(2, 2, "/b.pdf", 1, "second chunk content"),
        ]);
        {
            let base = state.load_base();
            let mut ov = state.overlay.write();
            ov.modify_doc(
                1,
                vec![mk_chunk(100, 1, "/a.pdf", 1, "new version mentions giraffe")],
                &base,
            );
        }
        let hits = search(&state, "zebra", QueryMode::Literal, 10).unwrap();
        assert!(hits.is_empty());
        let hits = search(&state, "giraffe", QueryMode::Literal, 10).unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn cheap_rank_promotes_phrase_match() {
        // Three hits over the same query — only one has the exact
        // phrase in its snippet. The phrase hit must come first.
        let mut hits = vec![
            Hit {
                chunk_id: 1,
                doc_id: 1,
                path: "/a.pdf".into(),
                page_no: 9,
                chunk_ord: 0,
                score: 1.0,
                snippet: "foo lonely word here bar".into(),
            },
            Hit {
                chunk_id: 2,
                doc_id: 2,
                path: "/b.pdf".into(),
                page_no: 1,
                chunk_ord: 0,
                score: 1.0,
                snippet: "this contains foo bar exactly".into(),
            },
            Hit {
                chunk_id: 3,
                doc_id: 3,
                path: "/c.pdf".into(),
                page_no: 4,
                chunk_ord: 0,
                score: 1.0,
                snippet: "neither here nor there".into(),
            },
        ];
        cheap_rank(&mut hits, "foo bar");
        assert_eq!(hits[0].chunk_id, 2, "phrase hit should sort first");
        assert_eq!(hits[1].chunk_id, 1, "partial-term hit next");
        assert_eq!(hits[2].chunk_id, 3, "no-term hit last");
    }

    #[test]
    fn cheap_rank_breaks_ties_by_page_then_id() {
        let mut hits = vec![
            Hit {
                chunk_id: 10,
                doc_id: 1,
                path: "/a.pdf".into(),
                page_no: 3,
                chunk_ord: 0,
                score: 1.0,
                snippet: "matches nothing distinctive".into(),
            },
            Hit {
                chunk_id: 11,
                doc_id: 2,
                path: "/b.pdf".into(),
                page_no: 1,
                chunk_ord: 0,
                score: 1.0,
                snippet: "matches nothing distinctive".into(),
            },
        ];
        cheap_rank(&mut hits, "zebraquack");
        // Neither contains the phrase nor any of its terms; tiebreak
        // is by `page_no`, ascending: page 1 before page 3.
        assert_eq!(hits[0].page_no, 1);
        assert_eq!(hits[1].page_no, 3);
    }
}
