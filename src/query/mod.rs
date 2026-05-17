//! Search engine over an [`IndexState`].
//!
//! All three modes share a candidate-generation skeleton built on the
//! bigram prefilter and the mutation overlay:
//!
//! 1. **Candidate generation.** Convert the query into a candidate
//!    bitset over the base index. The decision "do we have prefilter
//!    information at all?" plus the AND-NOT against the overlay's
//!    tombstones is encapsulated in [`candidate::CandidateSet`].
//! 2. **Verification.** Per-mode:
//!     * Literal: a compiled `memchr::memmem::Finder` over
//!       `text_norm_ascii`.
//!     * Regex: a compiled `regex::Regex` matched against `text_utf8`,
//!       with `case_insensitive(true)` so the engine's notion of case
//!       matches the lowercase-only bigram decomposition.
//!     * Fuzzy: two bands. A *filename band* of one hit per doc whose
//!       normalised filename contains every query term as a substring
//!       (ranked by earliest-match offset, then filename length); and
//!       a *body band* of `neo_frizbee`'s parallel match call over a
//!       synthetic "rank string" of `"{filename} {path} page {page_no}
//!       {preview}"` over the remaining chunks. Above [`FRIZBEE_LIMIT`]
//!       body candidates we fall back to a cheap deterministic
//!       ordering. The filename band is concatenated *before* the body
//!       band, so filename matches always outrank body-only matches.
//! 3. **Overflow pass.** Mode-specific candidate set:
//!     * Literal: use the query's deduped bigram set against
//!       `Overlay::overflow_matches`.
//!     * Regex: conservatively check every overflow row.
//!     * Fuzzy: include every overflow row except those whose doc
//!       is already represented in the filename band.
//!
//! The base index and the overlay are read under a single
//! `state.overlay.read()` guard that brackets both verification passes
//! so the snapshot stays consistent.
//!
//! Snippet rendering is best-effort: the normalized bytes
//! (`text_norm_ascii`) do not byte-align with the original `text_utf8`
//! after deunicode + lowercase + whitespace collapse, so a position in
//! the norm cannot be mapped exactly back into the UTF-8 text. See
//! [`snippet`] for the proportional mapping strategy.
//!
//! Module layout:
//!
//! * [`candidate`] — [`candidate::CandidateSet`] and
//!   [`candidate::MatchLocation`] (shared types).
//! * [`walk`]      — `walk_base_chunks` / `walk_overflow` generic over a
//!   verifier closure.
//! * [`snippet`]   — `make_hit` + `render_snippet`.
//! * [`rank`]      — [`rank::cheap_rank`] post-rank.
//! * [`literal`] / [`regex`] / [`fuzzy`] — the three query modes.

mod candidate;
mod fuzzy;
mod literal;
mod rank;
mod regex;
mod snippet;
mod walk;

use anyhow::Result;

use crate::index::IndexState;

pub use rank::cheap_rank;

/// Cap on the number of hits surfaced to the user in the TUI.
pub const DISPLAY_LIMIT: usize = 200;

/// Number of evenly-spaced probe bigrams to take when decomposing a
/// fuzzy query into a `BigramQuery`. The report names six.
pub const FUZZY_PROBES: usize = 6;

/// Above this many candidate chunks we skip neo_frizbee and fall back
/// to the cheap deterministic ordering. The threshold is from the
/// report.
pub const FRIZBEE_LIMIT: usize = 2048;

/// neo_frizbee thread count. The crate launches its own scoped pool
/// when called; six matches the value in fff's own scorer wiring.
pub(crate) const FRIZBEE_THREADS: usize = 6;

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
/// * [`QueryMode::Regex`]: candidate prefilter via the bigram-query
///   decomposition; verification via a compiled case-insensitive
///   `regex::Regex`. The pattern is *not* normalized (lowercasing the
///   source would break character classes and look-arounds);
///   case-insensitivity is delegated to the engine.
/// * [`QueryMode::Fuzzy`]: candidate prefilter via the fuzzy bigram
///   query; ranking via `neo_frizbee`'s parallel match call, with the
///   cheap deterministic fallback above [`FRIZBEE_LIMIT`] candidates.
pub fn search(
    state: &IndexState,
    query: &str,
    mode: QueryMode,
    limit: usize,
) -> Result<Vec<Hit>> {
    match mode {
        QueryMode::Literal => literal::literal_search(state, query, limit),
        QueryMode::Regex => regex::regex_search(state, query, limit),
        QueryMode::Fuzzy => fuzzy::fuzzy_search(state, query, limit),
    }
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
        let filename = std::path::Path::new(path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        ChunkItem::new(
            id,
            doc,
            Arc::from(path),
            Arc::from(filename.as_str()),
            page,
            0,
            0,
            text.len() as u32,
            Arc::from(text),
            Arc::<[u8]>::from(crate::normalize::normalize_for_index(text).as_bytes()),
            Arc::from(text),
            0,
        )
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
    fn fuzzy_filename_match_outranks_body_match() {
        // Doc A: filename matches the query, body does not.
        // Doc B: filename does NOT match, body contains the query
        // verbatim — so neo_frizbee scores it very highly on body.
        // Expectation: the filename-band hit comes first regardless.
        let state = synthetic_state(vec![
            mk_chunk(1, 1, "/papers/Streicher-1994.pdf", 1, "totally unrelated body text"),
            mk_chunk(2, 2, "/papers/other.pdf", 1, "this chunk repeatedly says streicher streicher streicher"),
        ]);
        let hits = search(&state, "streicher", QueryMode::Fuzzy, 10).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(
            hits[0].path, "/papers/Streicher-1994.pdf",
            "filename-matched doc must rank above the body-only match, got order {:?}",
            hits.iter().map(|h| h.path.as_str()).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn fuzzy_filename_match_contributes_one_hit_per_doc() {
        // The filename-matched doc has many chunks. Only one should
        // surface in the filename band; the doc's other chunks must
        // not flood the result list.
        let state = synthetic_state(vec![
            mk_chunk(1, 1, "/papers/thesis-final.pdf", 1, "chunk one of thesis"),
            mk_chunk(2, 1, "/papers/thesis-final.pdf", 2, "chunk two of thesis"),
            mk_chunk(3, 1, "/papers/thesis-final.pdf", 3, "chunk three of thesis"),
            mk_chunk(4, 2, "/papers/other.pdf", 1, "an unrelated chunk"),
        ]);
        let hits = search(&state, "thesis", QueryMode::Fuzzy, 10).unwrap();
        let from_doc1 = hits.iter().filter(|h| h.doc_id == 1).count();
        assert_eq!(
            from_doc1, 1,
            "filename-matched doc must contribute exactly one hit, got {} (paths: {:?})",
            from_doc1,
            hits.iter().map(|h| (h.doc_id, h.page_no)).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn fuzzy_filename_band_ranks_shorter_or_earlier_matches_first() {
        // Both filenames match "streicher", but one is short and the
        // term appears at offset 0; the other buries the term deep
        // inside a long name. The compact, early-match filename wins.
        let state = synthetic_state(vec![
            mk_chunk(
                1,
                1,
                "/papers/long-prefix-that-buries-the-author-streicher.pdf",
                1,
                "body text",
            ),
            mk_chunk(2, 2, "/papers/streicher.pdf", 1, "body text"),
        ]);
        let hits = search(&state, "streicher", QueryMode::Fuzzy, 10).unwrap();
        assert!(hits.len() >= 2);
        assert_eq!(
            hits[0].path, "/papers/streicher.pdf",
            "shorter / earlier-match filename should rank first, got {:?}",
            hits.iter().map(|h| h.path.as_str()).collect::<Vec<_>>(),
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
