//! Fuzzy search via `neo_frizbee`, with a cheap deterministic
//! fallback for large candidate sets.

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

    let bq = fuzzy_to_bigram_query(&q_norm, FUZZY_PROBES);

    let base = state.load_base();
    let ov = state.overlay.read();

    let lookup = if bq.is_any() {
        None
    } else {
        base.bigrams.as_ref().and_then(|idx| bq.evaluate(idx))
    };
    let candidates = CandidateSet::from_bigram_lookup(lookup, base.chunks.len(), &ov);

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

    let candidate_chunks =
        gather_fuzzy_candidates(&base, &ov, &candidates, &filename_match_docs);

    if candidate_chunks.is_empty() {
        return Ok(Vec::new());
    }

    let mut hits = if candidate_chunks.len() > FRIZBEE_LIMIT {
        rank_fuzzy_cheap(&candidate_chunks, &q_norm, &filename_match_docs, limit)
    } else {
        rank_fuzzy_frizbee(&candidate_chunks, &q_norm)
    };

    hits.truncate(limit);
    Ok(hits)
}

/// Base chunks surviving the bigram prefilter (or belonging to a
/// filename-matched doc), followed by every overflow chunk.
fn gather_fuzzy_candidates<'a>(
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
        if candidates.includes(i, ov) || filename_match_docs.contains(&chunk.doc_id) {
            out.push(chunk);
        }
    }
    for chunk in ov.overflow.chunks() {
        out.push(chunk);
    }
    out
}

/// Cheap deterministic ordering used when the candidate set exceeds
/// [`FRIZBEE_LIMIT`]. The report names this fallback exactly: on a
/// 1-char fuzzy query against a large corpus the prefilter has no
/// information and every chunk is a candidate, so the early break at
/// `limit` is what keeps the first keystroke bounded.
fn rank_fuzzy_cheap(
    candidate_chunks: &[&ChunkItem],
    q_norm: &str,
    filename_match_docs: &HashSet<i64>,
    limit: usize,
) -> Vec<Hit> {
    let needle_norm = q_norm.as_bytes();
    let finder = memmem::Finder::new(needle_norm);
    let mut hits: Vec<Hit> = Vec::with_capacity(limit.min(candidate_chunks.len()));
    for chunk in candidate_chunks {
        let loc = if let Some(offset) = finder.find(&chunk.text_norm_ascii) {
            MatchLocation::Norm { offset, query_len: needle_norm.len() }
        } else if filename_match_docs.contains(&chunk.doc_id) {
            // Filename-only match: anchor the snippet at offset 0.
            MatchLocation::Norm { offset: 0, query_len: needle_norm.len() }
        } else {
            continue;
        };
        hits.push(make_hit(chunk, loc));
        if hits.len() >= limit {
            break;
        }
    }
    hits
}

/// Score with `neo_frizbee` over the full candidate set.
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

/// Build the synthetic "rank string" passed to the fuzzy scorer. Mirrors
/// the report's recipe verbatim: `{filename} {path} page {page_no} {preview}`.
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
