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
use tracing::warn;

use crate::bigram::extract_bigrams;
use crate::bigram_query::{fuzzy_to_bigram_query, regex_to_bigram_query};
use crate::bitset::Bitset;
use crate::index::{BaseIndex, ChunkItem, IndexState, Overlay};
use crate::normalize::{collapse_whitespace_for_display, normalize_query_ascii};

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

/// Where a match was found inside a chunk.
///
/// The two variants exist because literal / fuzzy search locate matches
/// in `text_norm_ascii` (the normalised byte string used for the bigram
/// prefilter and `memchr`), while regex search locates them directly in
/// `text_utf8` (the original UTF-8 the regex engine ran against). The
/// snippet renderer needs to know which: an in-norm offset must be
/// proportionally remapped to a byte offset in `text_utf8` because the
/// two strings diverge after deunicode + lowercase + whitespace
/// collapse; an in-utf8 offset is already exact.
#[derive(Debug, Clone, Copy)]
enum MatchLocation {
    /// Byte offset within `text_norm_ascii`.
    Norm { offset: usize, query_len: usize },
    /// Byte offset within `text_utf8`.
    Utf8 { offset: usize, match_len: usize },
}

/// Candidate set fed into the verification pass.
///
/// Either:
/// * `Unconstrained` — the bigram prefilter had nothing to say (no
///   bigrams in the query, or `BigramIndex::query` returned `None`).
///   The verifier must visit every base chunk that isn't tombstoned.
/// * `Restricted(Bitset)` — a bigram bitset already AND-NOTed with
///   the overlay's tombstones at construction.
///
/// Hiding the two cases behind one type lets the base-walk be a single
/// loop over `[0, base.chunks.len())` instead of two near-identical
/// match arms.
enum CandidateSet {
    Unconstrained,
    Restricted(Bitset),
}

impl CandidateSet {
    /// Lift the bigram prefilter's `Option<Vec<u64>>` into a
    /// [`CandidateSet`] sized to `base_chunk_count`.
    ///
    /// `None` ⇒ `Unconstrained`; `Some(words)` ⇒ `Restricted` over
    /// exactly `base_chunk_count` bits, with the overlay's tombstones
    /// masked out in the same step.
    fn from_bigram_lookup(
        lookup: Option<Vec<u64>>,
        base_chunk_count: usize,
        ov: &Overlay,
    ) -> Self {
        match lookup {
            None => Self::Unconstrained,
            Some(words) => {
                let mut bits = Bitset::from_words(words, base_chunk_count);
                bits.and_not_assign(&ov.tombstones);
                Self::Restricted(bits)
            }
        }
    }

    /// Should chunk index `i` be visited by the verifier?
    ///
    /// `Restricted`: check the bitset. Tombstones are already masked
    /// out at construction, so a single lookup suffices.
    /// `Unconstrained`: skip tombstoned indices; everything else passes.
    #[inline]
    fn includes(&self, i: usize, ov: &Overlay) -> bool {
        match self {
            Self::Restricted(bits) => bits.get(i),
            Self::Unconstrained => !ov.is_tombstoned(i),
        }
    }
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

    if needle.len() < NO_BIGRAM_FULLSCAN_WARN_LEN {
        warn!(
            len = needle.len(),
            "literal query is too short for the bigram prefilter; falling back to full scan",
        );
    }

    let lookup = base.bigrams.as_ref().and_then(|idx| idx.query(needle));
    let candidates = CandidateSet::from_bigram_lookup(lookup, base.chunks.len(), &ov);

    let literal_verifier = |chunk: &ChunkItem| {
        finder
            .find(&chunk.text_norm_ascii)
            .map(|offset| MatchLocation::Norm { offset, query_len: needle.len() })
    };

    let mut hits: Vec<Hit> = Vec::new();
    walk_base_chunks(&base, &ov, &candidates, limit, &mut hits, &literal_verifier);

    if hits.len() < limit && !ov.overflow.is_empty() {
        let query_bigrams = extract_bigrams(needle);
        walk_overflow(&ov, ov.overflow_matches(&query_bigrams), limit, &mut hits, &literal_verifier);
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

    let lookup = if bq.is_any() {
        None
    } else {
        base.bigrams.as_ref().and_then(|idx| bq.evaluate(idx))
    };
    let candidates = CandidateSet::from_bigram_lookup(lookup, base.chunks.len(), &ov);

    let regex_verifier = |chunk: &ChunkItem| {
        regex
            .find(&chunk.text_utf8)
            .map(|m| MatchLocation::Utf8 { offset: m.start(), match_len: m.len() })
    };

    let mut hits: Vec<Hit> = Vec::new();
    walk_base_chunks(&base, &ov, &candidates, limit, &mut hits, &regex_verifier);

    // Overlay overflow: conservatively check every row. Regex bigrams
    // don't always survive overlay-side bigram dedup, so we let the
    // regex engine itself act as the verifier here. The overflow set
    // is bounded by the rebuild threshold, so the linear scan is
    // bounded too.
    if hits.len() < limit {
        let all_overflow: Vec<usize> = (0..ov.overflow.len()).collect();
        walk_overflow(&ov, all_overflow, limit, &mut hits, &regex_verifier);
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

    // Gather candidate chunks — base passes through the bitset OR the
    // filename-match doc set, overflow chunks are unconditionally
    // appended (the fuzzy scorer makes the final call).
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
    filename_match_docs: &std::collections::HashSet<i64>,
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
    filename_match_docs: &std::collections::HashSet<i64>,
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

/// Walk the base chunks, applying `verify` to each survivor of the
/// candidate set. Stops once `hits` reaches `limit`.
///
/// The two-arm `Some(bitset)` / `None` pyramid the search functions
/// used to carry collapses to one loop here; the asymmetry between
/// "have prefilter" and "no prefilter" is encapsulated by
/// [`CandidateSet::includes`].
fn walk_base_chunks<F>(
    base: &BaseIndex,
    ov: &Overlay,
    candidates: &CandidateSet,
    limit: usize,
    hits: &mut Vec<Hit>,
    verify: F,
) where
    F: Fn(&ChunkItem) -> Option<MatchLocation>,
{
    for (i, chunk) in base.chunks.iter().enumerate() {
        if !candidates.includes(i, ov) {
            continue;
        }
        if let Some(loc) = verify(chunk) {
            hits.push(make_hit(chunk, loc));
            if hits.len() >= limit {
                break;
            }
        }
    }
}

/// Same as [`walk_base_chunks`] over a list of overflow indices.
fn walk_overflow<F>(
    ov: &Overlay,
    indices: Vec<usize>,
    limit: usize,
    hits: &mut Vec<Hit>,
    verify: F,
) where
    F: Fn(&ChunkItem) -> Option<MatchLocation>,
{
    let chunks = ov.overflow.chunks();
    for idx in indices {
        let chunk = &chunks[idx];
        if let Some(loc) = verify(chunk) {
            hits.push(make_hit(chunk, loc));
            if hits.len() >= limit {
                break;
            }
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

/// Construct a [`Hit`] for `chunk` with its snippet anchored at `loc`.
fn make_hit(chunk: &ChunkItem, loc: MatchLocation) -> Hit {
    Hit {
        chunk_id: chunk.chunk_id,
        doc_id: chunk.doc_id,
        path: chunk.path.to_string(),
        filename: chunk.filename.to_string(),
        page_no: chunk.page_no,
        chunk_ord: chunk.chunk_ord,
        score: 1.0,
        snippet: render_snippet(chunk, loc),
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

/// Build a short snippet around `loc` inside `chunk.text_utf8`.
///
/// For [`MatchLocation::Norm`] offsets the function proportionally
/// remaps the `text_norm_ascii` offset to a `text_utf8` byte offset:
/// the two strings diverge after deunicode + lowercase + whitespace
/// collapse, so the remap is best-effort. For [`MatchLocation::Utf8`]
/// the offset is used directly. See the module-level docs.
fn render_snippet(chunk: &ChunkItem, loc: MatchLocation) -> String {
    let text = &*chunk.text_utf8;
    if text.is_empty() {
        return String::new();
    }
    let (centre, match_len) = match loc {
        MatchLocation::Norm { offset, query_len } => {
            let norm_len = chunk.text_norm_ascii.len();
            let approx_byte = if norm_len == 0 {
                0
            } else {
                ((text.len() as f64) * (offset as f64 / norm_len as f64)).round() as usize
            };
            (approx_byte.min(text.len()), query_len)
        }
        MatchLocation::Utf8 { offset, match_len } => (offset.min(text.len()), match_len),
    };
    render_window(text, centre, match_len)
}

fn render_window(text: &str, centre: usize, match_len: usize) -> String {
    let center = snap_char_boundary(text, centre);
    let want_match_end = (center + match_len).min(text.len());
    let left = center.saturating_sub(SNIPPET_CONTEXT_BYTES);
    let right = (want_match_end + SNIPPET_CONTEXT_BYTES).min(text.len());
    let left = snap_char_boundary(text, left);
    let right = snap_char_boundary(text, right);
    collapse_whitespace_for_display(&text[left..right])
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
