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

/// Cap on the number of hits surfaced to the user in the TUI.
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

/// One search result, rendered by the TUI.
///
/// `filename` is the basename of `path` (carried separately so the TUI
/// can render it without re-splitting the full path on every keystroke).
#[derive(Debug, Clone)]
pub struct Hit {
    pub chunk_id: i64,
    pub doc_id: i64,
    pub path: String,
    pub filename: String,
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

    // Find docs whose filename *itself* matches the query, and union
    // their chunks into the candidate set. The chunk-text bigram
    // prefilter is computed only over `text_norm_ascii`, so a doc whose
    // only match is in its filename (e.g. the user types part of a
    // paper title) would otherwise never reach the fuzzy scorer.
    //
    // We match the filename by `memmem` against the normalized query
    // bytes — a substring test on the ASCII-normalized form. This is
    // looser than the bigram prefilter (so it includes any reasonable
    // typo-free hit) and is the same normalisation the scorer's
    // `rank_text` uses, so the downstream ranking stays consistent.
    let filename_match_docs = docs_with_filename_match(&base, &q_norm);

    // Gather candidate chunks — base passes through the bitset OR the
    // filename-match doc set, overflow chunks are unconditionally
    // appended (the fuzzy scorer makes the final call).
    let mut candidate_chunks: Vec<&ChunkItem> = Vec::new();
    collect_base_candidates_with_filename_docs(
        &base,
        &ov,
        candidates.as_deref(),
        &filename_match_docs,
        &mut candidate_chunks,
    );
    for chunk in &ov.overflow_chunks {
        candidate_chunks.push(chunk);
    }

    if candidate_chunks.is_empty() {
        return Ok(Vec::new());
    }

    let mut hits: Vec<Hit> = if candidate_chunks.len() > FRIZBEE_LIMIT {
        // Above the limit the cheap deterministic ranker is faster
        // and good enough — the report names this fallback exactly.
        // We still let filename-match docs through even when the
        // chunk text doesn't contain the query, so the "type part of
        // a filename" UX survives on large corpora; the snippet
        // anchors at offset 0 in that case.
        //
        // The cheap branch doesn't need the `rank_text_for(c)`
        // strings the neo_frizbee branch builds, so we skip that
        // pass entirely. We also break at `limit`: on a 1-char fuzzy
        // query against a large corpus the bigram prefilter has no
        // information and every chunk lands in `candidate_chunks`,
        // so the early break is what keeps the first keystroke from
        // doing N_chunks worth of unbounded work.
        let needle_norm = q_norm.as_bytes();
        let finder = memmem::Finder::new(needle_norm);
        let mut hits: Vec<Hit> = Vec::with_capacity(limit.min(candidate_chunks.len()));
        for chunk in &candidate_chunks {
            let hit = if let Some(pos) = finder.find(&chunk.text_norm_ascii) {
                make_hit_at_norm(chunk, pos, needle_norm.len())
            } else if filename_match_docs.contains(&chunk.doc_id) {
                make_hit_at_norm(chunk, 0, needle_norm.len())
            } else {
                continue;
            };
            hits.push(hit);
            if hits.len() >= limit {
                break;
            }
        }
        hits
    } else {
        // neo_frizbee needs one "rank string" per candidate; build it
        // only here, since the cheap branch above doesn't use it.
        let rank_texts: Vec<String> = candidate_chunks
            .iter()
            .map(|c| rank_text_for(c))
            .collect();
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
            // Locate the user's query inside the chunk text for snippet
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

/// Walk the base chunks and push every chunk that either survives the
/// bigram prefilter *or* belongs to a doc in `filename_match_docs`.
/// Tombstoned base chunks are always hidden.
///
/// Used by the fuzzy path so that a query matching only a paper's
/// filename (and not the body text) still surfaces the paper. Pass an
/// empty `filename_match_docs` to get the strict bigram-prefilter
/// behaviour.
fn collect_base_candidates_with_filename_docs<'a>(
    base: &'a BaseIndex,
    ov: &Overlay,
    candidates: Option<&[u64]>,
    filename_match_docs: &std::collections::HashSet<i64>,
    out: &mut Vec<&'a ChunkItem>,
) {
    for (i, chunk) in base.chunks.iter().enumerate() {
        if ov.is_tombstoned(i) {
            continue;
        }
        let bigram_ok = match candidates {
            Some(bitset) => BigramIndex::is_candidate(bitset, i),
            None => true,
        };
        if bigram_ok || filename_match_docs.contains(&chunk.doc_id) {
            out.push(chunk);
        }
    }
}

/// Doc IDs whose normalised filename contains every whitespace-delimited
/// term of `q_norm` as a substring.
///
/// `q_norm` is produced by [`normalize_query_ascii`] (deunicode + ASCII
/// lowercase + whitespace collapse); we run the same normalisation on
/// each filename so that, e.g., "café_2023.pdf" matches a query of
/// "cafe". One allocation per doc, not per chunk.
///
/// The per-term AND (rather than a single contiguous-substring check) is
/// what lets a multi-word query like `streicher 1994` match a filename
/// like `Streicher - 1994 - A universality.pdf` — the separators between
/// the terms in the filename would defeat a single `memmem` of the joined
/// query. Filenames in academic corpora routinely separate the author,
/// year, and title with hyphens or underscores, so the substring-only
/// rule was too strict in practice.
fn docs_with_filename_match(
    base: &BaseIndex,
    q_norm: &str,
) -> std::collections::HashSet<i64> {
    use std::collections::HashSet;
    let mut out: HashSet<i64> = HashSet::new();
    let terms: Vec<&str> = q_norm.split_whitespace().collect();
    if terms.is_empty() {
        return out;
    }
    let finders: Vec<memmem::Finder> = terms
        .iter()
        .map(|t| memmem::Finder::new(t.as_bytes()))
        .collect();
    // Read pre-normalised filenames from the BaseIndex cache. They were
    // computed once at index-build time so per-keystroke fuzzy search
    // doesn't re-run `deunicode` over every filename in the corpus.
    for (doc_id, fn_norm) in &base.filename_norms {
        let fn_bytes = fn_norm.as_bytes();
        if finders.iter().all(|f| f.find(fn_bytes).is_some()) {
            out.insert(*doc_id);
        }
    }
    out
}

fn make_hit_at_norm(chunk: &ChunkItem, match_offset_in_norm: usize, query_len: usize) -> Hit {
    Hit {
        chunk_id: chunk.chunk_id,
        doc_id: chunk.doc_id,
        path: chunk.path.to_string(),
        filename: chunk.filename.to_string(),
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
        filename: chunk.filename.to_string(),
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
        // Treat both whitespace and non-whitespace control characters
        // as space. Raw PDF text occasionally contains ESC (\x1b),
        // backspace (\x08), bell (\x07), or other sub-0x20 bytes that
        // are *not* whitespace per `char::is_whitespace`; rendered into
        // a TUI cell they get interpreted by the host terminal as
        // escape sequences / cursor moves and corrupt the screen
        // (random letters re-flowing, beeps, half-erased cells). The
        // safe thing is to normalise every control char to a space at
        // the snippet boundary — the same rule the bigram normaliser
        // uses for the indexed copy.
        let is_problem_ctrl = ch.is_control() && !ch.is_whitespace();
        if ch.is_whitespace() || is_problem_ctrl {
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
        let mut filename_norms: HashMap<i64, String> = HashMap::new();
        for (doc_id, range) in &doc_ranges {
            if let Some(chunk) = chunks.get(range.start) {
                filename_norms.insert(
                    *doc_id,
                    crate::normalize::normalize_for_index(&chunk.filename),
                );
            }
        }
        let base = BaseIndex {
            chunks: Arc::new(chunks),
            doc_ranges,
            bigrams,
            filename_norms,
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
                filename: "a.pdf".into(),
                page_no: 9,
                chunk_ord: 0,
                score: 1.0,
                snippet: "foo lonely word here bar".into(),
            },
            Hit {
                chunk_id: 2,
                doc_id: 2,
                path: "/b.pdf".into(),
                filename: "b.pdf".into(),
                page_no: 1,
                chunk_ord: 0,
                score: 1.0,
                snippet: "this contains foo bar exactly".into(),
            },
            Hit {
                chunk_id: 3,
                doc_id: 3,
                path: "/c.pdf".into(),
                filename: "c.pdf".into(),
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
    fn snippet_strips_terminal_control_chars() {
        // ESC, BEL and BS inside the chunk text would corrupt the
        // host terminal if rendered raw; the snippet must collapse
        // them to whitespace so the TUI stays stable.
        let chunk =
            mk_chunk(1, 1, "/a.pdf", 1, "before\x1b[31mneedle\x1b[m\x07 after \x08x");
        let state = synthetic_state(vec![chunk]);
        let hits = search(&state, "needle", QueryMode::Literal, 10).unwrap();
        assert_eq!(hits.len(), 1);
        let snip = &hits[0].snippet;
        assert!(!snip.chars().any(|c| c.is_control() && !c.is_whitespace()),
            "snippet must not contain raw control chars: {snip:?}");
        // The needle and the surrounding visible text should survive,
        // separated by spaces where control runs were.
        assert!(snip.contains("needle"), "snippet should still contain the needle: {snip:?}");
    }

    #[test]
    fn fuzzy_matches_against_filename() {
        // Doc whose body text contains nothing related to "thesis",
        // but whose filename does. Fuzzy should still surface it.
        let state = synthetic_state(vec![
            mk_chunk(1, 1, "/work/thesis-final.pdf", 1, "totally unrelated body text here"),
            mk_chunk(2, 2, "/work/random.pdf", 1, "another unrelated chunk"),
        ]);
        let hits = search(&state, "thesis", QueryMode::Fuzzy, 10).unwrap();
        assert!(!hits.is_empty(), "fuzzy should match against the filename");
        let paths: Vec<&str> = hits.iter().map(|h| h.path.as_str()).collect();
        assert!(
            paths.contains(&"/work/thesis-final.pdf"),
            "expected /work/thesis-final.pdf in fuzzy results, got {paths:?}",
        );
    }

    #[test]
    fn fuzzy_matches_author_and_title_tokens_in_filename() {
        // Same shape as the Streicher-1994 case, but with the two
        // query terms being the author and a word from the title —
        // covers the variant the user reported alongside the year one.
        let state = synthetic_state(vec![
            mk_chunk(
                1,
                1,
                "/papers/Streicher - 1994 - A universality.pdf",
                1,
                "totally unrelated body text here",
            ),
            mk_chunk(2, 2, "/papers/random.pdf", 1, "another unrelated chunk"),
        ]);
        let hits = search(&state, "streicher universality", QueryMode::Fuzzy, 10).unwrap();
        let paths: Vec<&str> = hits.iter().map(|h| h.path.as_str()).collect();
        assert!(
            paths.contains(&"/papers/Streicher - 1994 - A universality.pdf"),
            "expected the Streicher universality paper in fuzzy results, got {paths:?}",
        );
    }

    #[test]
    fn fuzzy_matches_multiword_query_across_filename_separators() {
        // Real-world academic-paper case: the filename embeds the
        // author and year separated by " - ", and the user types just
        // those two tokens with a space. A single contiguous-substring
        // check against the normalised filename would miss this; the
        // per-term match must succeed.
        let state = synthetic_state(vec![
            mk_chunk(
                1,
                1,
                "/papers/Streicher - 1994 - A universality.pdf",
                1,
                "totally unrelated body text here",
            ),
            mk_chunk(2, 2, "/papers/random.pdf", 1, "another unrelated chunk"),
        ]);
        let hits = search(&state, "streicher 1994", QueryMode::Fuzzy, 10).unwrap();
        let paths: Vec<&str> = hits.iter().map(|h| h.path.as_str()).collect();
        assert!(
            paths.contains(&"/papers/Streicher - 1994 - A universality.pdf"),
            "expected the Streicher 1994 paper in fuzzy results, got {paths:?}",
        );
    }

    #[test]
    fn cheap_rank_breaks_ties_by_page_then_id() {
        let mut hits = vec![
            Hit {
                chunk_id: 10,
                doc_id: 1,
                path: "/a.pdf".into(),
                filename: "a.pdf".into(),
                page_no: 3,
                chunk_ord: 0,
                score: 1.0,
                snippet: "matches nothing distinctive".into(),
            },
            Hit {
                chunk_id: 11,
                doc_id: 2,
                path: "/b.pdf".into(),
                filename: "b.pdf".into(),
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
