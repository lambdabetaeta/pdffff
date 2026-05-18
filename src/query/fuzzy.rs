//! Fuzzy search via `neo_frizbee`, with a cheap deterministic
//! fallback for large candidate sets.
//!
//! Filename matches are surfaced as a separate, prioritised band:
//! when a doc's normalised filename contains every whitespace-delimited
//! query term as a substring, the doc gets one representative hit
//! (its first non-tombstoned chunk) at the top of the result list,
//! ahead of any body-text matches. The body-text path then runs over
//! every chunk that does *not* belong to a filename-matched doc, so
//! filename-matched docs never get diluted by their own body chunks
//! competing for top spots.

use anyhow::Result;
use memchr::memmem;
use std::collections::HashSet;

use crate::bigram_query::fuzzy_to_bigram_query;
use crate::index::{BaseIndex, ChunkItem, IndexState, Overlay};
use crate::normalize::normalize_query_ascii;

use super::Hit;
use super::candidate::{CandidateSet, MatchLocation};
use super::snippet::make_hit;
use super::{FRIZBEE_LIMIT, FRIZBEE_THREADS, FUZZY_PROBES};

pub(super) fn fuzzy_search(
    state: &IndexState,
    query: &str,
    limit: usize,
) -> Result<Vec<Hit>> {
    let q_norm = normalize_query_ascii(query);
    if q_norm.is_empty() || limit == 0 {
        return Ok(Vec::new());
    }

    let base = state.load_base();
    let ov = state.overlay.read();

    // Filename band: any doc whose normalised filename contains every
    // whitespace-delimited query term as a substring. These outrank
    // body-text matches absolutely, and each doc contributes exactly
    // one hit (the doc's first non-tombstoned chunk).
    let filename_match_docs = docs_with_filename_match(&base, &q_norm);
    let filename_hits =
        build_filename_hits(&base, &ov, &filename_match_docs, &q_norm, limit);

    // Body band: remaining `limit` slots after the filename hits, over
    // candidates that do not belong to any filename-matched doc.
    let body_slots = limit.saturating_sub(filename_hits.len());
    let body_hits = if body_slots == 0 {
        Vec::new()
    } else {
        body_fuzzy_hits(&base, &ov, &q_norm, &filename_match_docs, body_slots)
    };

    let mut hits = filename_hits;
    hits.extend(body_hits);
    hits.truncate(limit);
    Ok(hits)
}

/// Body-text fuzzy ranking over every candidate chunk whose doc is
/// *not* in `filename_match_docs`.
fn body_fuzzy_hits(
    base: &BaseIndex,
    ov: &Overlay,
    q_norm: &str,
    filename_match_docs: &HashSet<i64>,
    limit: usize,
) -> Vec<Hit> {
    let bq = fuzzy_to_bigram_query(q_norm, FUZZY_PROBES);
    let lookup = if bq.is_any() {
        None
    } else {
        base.bigrams.as_ref().and_then(|idx| bq.evaluate(idx))
    };
    let candidates = CandidateSet::from_bigram_lookup(lookup, base.chunks.len(), ov);

    let candidate_chunks =
        gather_body_candidates(base, ov, &candidates, filename_match_docs);

    if candidate_chunks.is_empty() {
        return Vec::new();
    }

    if candidate_chunks.len() > FRIZBEE_LIMIT {
        rank_fuzzy_cheap(&candidate_chunks, q_norm, limit)
    } else {
        rank_fuzzy_frizbee(&candidate_chunks, q_norm)
    }
}

/// Base chunks surviving the bigram prefilter, plus every overflow
/// chunk — minus any chunk whose doc is already represented in the
/// filename band.
fn gather_body_candidates<'a>(
    base: &'a BaseIndex,
    ov: &'a Overlay,
    candidates: &CandidateSet,
    filename_match_docs: &HashSet<i64>,
) -> Vec<&'a ChunkItem> {
    let mut out: Vec<&'a ChunkItem> = Vec::new();
    for (i, chunk) in base.chunks.iter().enumerate() {
        if ov.is_tombstoned(i) {
            continue;
        }
        if filename_match_docs.contains(&chunk.doc_id) {
            continue;
        }
        if candidates.includes(i, ov) {
            out.push(chunk);
        }
    }
    for chunk in ov.overflow.chunks() {
        if filename_match_docs.contains(&chunk.doc_id) {
            continue;
        }
        out.push(chunk);
    }
    out
}

/// Cheap deterministic ordering used when the body candidate set
/// exceeds [`FRIZBEE_LIMIT`]. On a 1-char fuzzy query against a large
/// corpus the prefilter has no information and every chunk is a
/// candidate, so the early break at `limit` is what keeps the first
/// keystroke bounded.
fn rank_fuzzy_cheap(
    candidate_chunks: &[&ChunkItem],
    q_norm: &str,
    limit: usize,
) -> Vec<Hit> {
    let needle_norm = q_norm.as_bytes();
    let finder = memmem::Finder::new(needle_norm);
    let mut hits: Vec<Hit> = Vec::with_capacity(limit.min(candidate_chunks.len()));
    for chunk in candidate_chunks {
        let Some(offset) = finder.find(&chunk.text_norm_ascii) else {
            continue;
        };
        let loc = MatchLocation::Norm { offset, query_len: needle_norm.len() };
        hits.push(make_hit(chunk, loc));
        if hits.len() >= limit {
            break;
        }
    }
    hits
}

/// Score with `neo_frizbee` over the full body candidate set.
fn rank_fuzzy_frizbee(candidate_chunks: &[&ChunkItem], q_norm: &str) -> Vec<Hit> {
    // neo_frizbee needs one "rank string" per candidate; we only build
    // these on the frizbee path because the cheap fallback doesn't use
    // them.
    let rank_texts: Vec<String> = candidate_chunks.iter().map(|c| rank_text_for(c)).collect();
    let config = neo_frizbee::Config {
        max_typos: None,
        sort: true,
        scoring: neo_frizbee::Scoring::default(),
    };
    let matches =
        neo_frizbee::match_list_parallel(q_norm, &rank_texts, &config, FRIZBEE_THREADS);
    let mut hits: Vec<Hit> = Vec::with_capacity(matches.len());
    for m in &matches {
        let chunk = candidate_chunks[m.index as usize];
        // Locate the user's query inside the chunk text for snippet
        // purposes — best-effort. If neo_frizbee accepted the
        // candidate but the literal needle isn't present (the fuzzy
        // match crossed token boundaries), centre the snippet on
        // offset 0.
        let offset = memmem::find(chunk.text_norm_ascii.as_ref(), q_norm.as_bytes()).unwrap_or(0);
        let mut hit = make_hit(chunk, MatchLocation::Norm { offset, query_len: q_norm.len() });
        // Carry neo_frizbee's score through so callers can rank across
        // queries; the unit-of-score is u16 internally.
        hit.score = m.score as f32;
        hits.push(hit);
    }
    hits
}

/// Build the synthetic "rank string" passed to the fuzzy scorer for
/// body chunks. Filename and path stay in front so a typo against the
/// filename still scores, but every doc whose filename is a clean
/// substring match has already been pulled into the filename band and
/// excluded from this set.
fn rank_text_for(c: &ChunkItem) -> String {
    let mut s =
        String::with_capacity(c.filename.len() + c.path.len() + 10 + c.preview.len());
    s.push_str(&c.filename);
    s.push(' ');
    s.push_str(&c.path);
    s.push_str(" page ");
    s.push_str(&c.page_no.to_string());
    s.push(' ');
    s.push_str(&c.preview);
    s
}

/// One [`Hit`] per filename-matched doc, ranked so the most relevant
/// filenames sort first.
///
/// Ranking is by `(sum of first-occurrence offsets across the query
/// terms, filename length, doc_id)` — earlier matches and shorter
/// filenames win, with `doc_id` as a stable tiebreak. `neo_frizbee`
/// would over-filter here (it requires the query characters to appear
/// in order, but our substring-AND admits filenames where the terms
/// are reordered), so we don't route through it.
fn build_filename_hits(
    base: &BaseIndex,
    ov: &Overlay,
    filename_match_docs: &HashSet<i64>,
    q_norm: &str,
    limit: usize,
) -> Vec<Hit> {
    if filename_match_docs.is_empty() || limit == 0 {
        return Vec::new();
    }
    let terms: Vec<&str> = q_norm.split_whitespace().collect();
    if terms.is_empty() {
        return Vec::new();
    }
    let finders: Vec<memmem::Finder> = terms
        .iter()
        .map(|t| memmem::Finder::new(t.as_bytes()))
        .collect();

    struct Entry<'a> {
        chunk: &'a ChunkItem,
        score_key: (usize, usize, i64),
    }

    let mut entries: Vec<Entry> = Vec::with_capacity(filename_match_docs.len());
    for &doc_id in filename_match_docs {
        let Some(chunk) = pick_representative_chunk(base, ov, doc_id) else {
            continue;
        };
        let fn_norm = base
            .filename_norms
            .get(&doc_id)
            .map(String::as_str)
            .unwrap_or("");
        let fn_bytes = fn_norm.as_bytes();
        // Sum the first-occurrence offsets of each term in the
        // normalised filename. Every term must be present (that's what
        // membership in `filename_match_docs` guarantees), so the
        // `unwrap_or(0)` is defensive and never triggered in practice.
        let offset_sum: usize = finders
            .iter()
            .map(|f| f.find(fn_bytes).unwrap_or(0))
            .sum();
        entries.push(Entry {
            chunk,
            score_key: (offset_sum, fn_norm.len(), doc_id),
        });
    }

    entries.sort_by_key(|e| e.score_key);

    entries
        .into_iter()
        .take(limit)
        .map(|e| {
            make_hit(
                e.chunk,
                MatchLocation::Norm { offset: 0, query_len: q_norm.len() },
            )
        })
        .collect()
}

/// First non-tombstoned chunk for `doc_id`, scanning base then
/// overflow. Returns `None` only if the doc has no live chunks at all.
fn pick_representative_chunk<'a>(
    base: &'a BaseIndex,
    ov: &'a Overlay,
    doc_id: i64,
) -> Option<&'a ChunkItem> {
    if let Some(range) = base.doc_ranges.get(&doc_id) {
        for i in range.clone() {
            if !ov.is_tombstoned(i) {
                return Some(&base.chunks[i]);
            }
        }
    }
    ov.overflow
        .chunks()
        .iter()
        .filter(|c| c.doc_id == doc_id)
        .min_by_key(|c| (c.page_no, c.chunk_ord))
}

/// Doc IDs whose normalised filename contains every whitespace-delimited
/// term of `q_norm` as a substring.
///
/// `q_norm` is produced by `normalize_query_ascii` (deunicode + ASCII
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
fn docs_with_filename_match(base: &BaseIndex, q_norm: &str) -> HashSet<i64> {
    let mut out: HashSet<i64> = HashSet::new();
    let terms: Vec<&str> = q_norm.split_whitespace().collect();
    if terms.is_empty() {
        return out;
    }
    let finders: Vec<memmem::Finder> = terms
        .iter()
        .map(|t| memmem::Finder::new(t.as_bytes()))
        .collect();
    for (doc_id, fn_norm) in &base.filename_norms {
        let fn_bytes = fn_norm.as_bytes();
        if finders.iter().all(|f| f.find(fn_bytes).is_some()) {
            out.insert(*doc_id);
        }
    }
    out
}
