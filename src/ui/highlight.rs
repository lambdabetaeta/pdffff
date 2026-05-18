//! Frontend-agnostic snippet highlighting.
//!
//! Both the TUI and the GUI need to draw a snippet (or a filename) with
//! the query matches visually distinct from the surrounding text. The
//! *where* of the highlight is identical between frontends — it is a
//! pure function of `(text, query)`. The *how* of the highlight is
//! frontend-specific: ratatui paints styled `Span`s, egui paints
//! `LayoutJob` runs with a background colour.
//!
//! This module owns the *where*. It exposes [`highlight_segments`],
//! which returns a flat sequence of [`SnippetSegment`]s — each segment
//! carries the text it covers and a tag indicating whether the segment
//! is a query match. Each frontend then maps the segments to its own
//! widget vocabulary in a one-line `match`.
//!
//! Algorithm
//! ---------
//! Three composed passes, identical to the previous in-TUI
//! implementation:
//!
//! 1. [`build_needles`] — full phrase + whitespace-split terms, deduped
//!    and sorted longest-first so the greedy matcher prefers the
//!    full phrase over its constituents.
//! 2. [`build_lc_offset_map`] — a lowercased copy of `text` together
//!    with a table mapping lowercase byte offsets back to original
//!    byte offsets. Lowercasing changes byte length per codepoint
//!    ('İ' → "i\u{307}" grows, 'ẞ' → "ß" shrinks), so the table is
//!    the only correct way to recover original-side spans.
//! 3. [`scan_for_match_ranges`] — greedy left-to-right scan over the
//!    lowercase bytes, emitting original-side, non-overlapping
//!    `Range<usize>`s.
//!
//! Segment construction then weaves unmatched runs between match
//! ranges. The function is style-agnostic; both the snippet body and
//! the card-title filename use the same pipeline.

use std::ops::Range;

/// One run of the highlighted output.
///
/// `text` is an owned substring of the input; `kind` records whether
/// that substring corresponds to a query match. Owned strings (rather
/// than borrows) keep the segments cheaply movable across the worker
/// → render boundary in both frontends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnippetSegment {
    pub text: String,
    pub kind: SegmentKind,
}

/// What the segment represents. `Plain` is the default "body" text;
/// `Match` is a query hit and should be rendered with the frontend's
/// hit-highlight style.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentKind {
    Plain,
    Match,
}

/// Split `text` into highlighted / unhighlighted runs against `query`.
///
/// Empty or whitespace-only `query` yields a single `Plain` segment
/// covering the whole input. Match detection is case-insensitive and
/// UTF-8 safe.
pub fn highlight_segments(text: &str, query: &str) -> Vec<SnippetSegment> {
    let needles = build_needles(query);
    if needles.is_empty() {
        return vec![SnippetSegment {
            text: text.to_string(),
            kind: SegmentKind::Plain,
        }];
    }
    let map = build_lc_offset_map(text);
    let ranges = scan_for_match_ranges(text, &map, &needles);
    weave_segments(text, &ranges)
}

/// Ordered, deduped match needles: full phrase + whitespace-split
/// terms, sorted longest-first so longest-match-wins is a simple loop.
fn build_needles(query: &str) -> Vec<String> {
    let phrase = query.trim().to_lowercase();
    if phrase.is_empty() {
        return Vec::new();
    }
    let mut v = vec![phrase.clone()];
    v.extend(
        phrase
            .split_whitespace()
            .filter(|t| *t != phrase)
            .map(|t| t.to_lowercase()),
    );
    v.sort_by_key(|s| std::cmp::Reverse(s.len()));
    v.dedup();
    v
}

/// Lowercased copy of `snippet` together with a `lc_byte → orig_byte`
/// table.
///
/// Lowercasing changes byte length per codepoint ('İ' → "i\u{307}"
/// grows, 'ẞ' → "ß" shrinks), so positions in the two strings don't
/// coincide. `lc_to_orig[i] = Some(orig)` marks lc byte offset `i` as
/// a char boundary corresponding to original byte offset `orig`; mid-
/// codepoint and lowercase-expansion offsets are `None`. The final
/// entry pins the end-of-string boundary so a needle that lands at the
/// very end has a well-defined `orig_end`.
struct LcOffsetMap {
    lc: String,
    /// Indexed by lowercase byte offset, length = `lc.len() + 1`.
    lc_to_orig: Vec<Option<usize>>,
}

fn build_lc_offset_map(snippet: &str) -> LcOffsetMap {
    let mut lc = String::with_capacity(snippet.len());
    let mut lc_to_orig: Vec<Option<usize>> = Vec::new();
    for (orig_byte, ch) in snippet.char_indices() {
        while lc_to_orig.len() < lc.len() {
            lc_to_orig.push(None);
        }
        lc_to_orig.push(Some(orig_byte));
        for lc_ch in ch.to_lowercase() {
            lc.push(lc_ch);
        }
    }
    while lc_to_orig.len() < lc.len() {
        lc_to_orig.push(None);
    }
    lc_to_orig.push(Some(snippet.len()));
    LcOffsetMap { lc, lc_to_orig }
}

/// Greedy left-to-right scan: at each char boundary in `map.lc`, try
/// each needle in order (longest first) and emit the original-side
/// match range. Returns ranges in left-to-right order, non-overlapping.
fn scan_for_match_ranges(
    snippet: &str,
    map: &LcOffsetMap,
    needles: &[String],
) -> Vec<Range<usize>> {
    let lc_bytes = map.lc.as_bytes();
    let mut ranges: Vec<Range<usize>> = Vec::new();
    let mut lc_cursor = 0usize;
    while lc_cursor < lc_bytes.len() {
        let Some(orig_cursor) = map.lc_to_orig.get(lc_cursor).copied().flatten() else {
            lc_cursor += 1;
            continue;
        };
        let matched = needles.iter().find_map(|n| {
            let lc_end = lc_cursor + n.len();
            if lc_end > lc_bytes.len() || &lc_bytes[lc_cursor..lc_end] != n.as_bytes() {
                return None;
            }
            map.lc_to_orig
                .get(lc_end)
                .copied()
                .flatten()
                .map(|orig_end| (n.len(), orig_end))
        });
        if let Some((lc_len, orig_end)) = matched {
            ranges.push(orig_cursor..orig_end);
            lc_cursor += lc_len;
        } else {
            // No match here — advance by the lowercase byte length of
            // the next codepoint so the cursor stays on a boundary.
            let lc_step: usize = match snippet[orig_cursor..].chars().next() {
                Some(c) => c.to_lowercase().map(|x| x.len_utf8()).sum(),
                None => break,
            };
            lc_cursor += lc_step.max(1);
        }
    }
    ranges
}

/// Weave the unmatched prefixes / suffixes between the highlighted
/// match `ranges` into a segment list. Adjacent `Plain` segments cannot
/// be produced — every emitted segment is non-empty and alternates as
/// the input dictates.
fn weave_segments(text: &str, ranges: &[Range<usize>]) -> Vec<SnippetSegment> {
    let mut segments: Vec<SnippetSegment> = Vec::new();
    let mut cursor = 0usize;
    for r in ranges {
        if cursor < r.start {
            segments.push(SnippetSegment {
                text: text[cursor..r.start].to_string(),
                kind: SegmentKind::Plain,
            });
        }
        segments.push(SnippetSegment {
            text: text[r.start..r.end].to_string(),
            kind: SegmentKind::Match,
        });
        cursor = r.end;
    }
    if cursor < text.len() {
        segments.push(SnippetSegment {
            text: text[cursor..].to_string(),
            kind: SegmentKind::Plain,
        });
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flatten(segments: &[SnippetSegment]) -> String {
        segments.iter().map(|s| s.text.as_str()).collect()
    }

    fn matches(segments: &[SnippetSegment]) -> Vec<&str> {
        segments
            .iter()
            .filter(|s| s.kind == SegmentKind::Match)
            .map(|s| s.text.as_str())
            .collect()
    }

    #[test]
    fn highlight_plain_ascii() {
        let segs = highlight_segments("Hello world", "world");
        assert_eq!(flatten(&segs), "Hello world");
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].text, "Hello ");
        assert_eq!(segs[0].kind, SegmentKind::Plain);
        assert_eq!(segs[1].text, "world");
        assert_eq!(segs[1].kind, SegmentKind::Match);
    }

    #[test]
    fn highlight_preserves_original_case() {
        let segs = highlight_segments("HeLLo WoRLd", "hello");
        assert_eq!(flatten(&segs), "HeLLo WoRLd");
        assert_eq!(matches(&segs), vec!["HeLLo"]);
    }

    // Regression: lowercasing 'ẞ' (U+1E9E, 3 bytes UTF-8) yields "ß"
    // (2 bytes), so byte positions in `snippet` and its lowercase form
    // diverge. Previously this overflowed the lowercase slice.
    #[test]
    fn highlight_handles_shrinking_lowercase() {
        let segs = highlight_segments("STRAẞE", "stra");
        assert_eq!(flatten(&segs), "STRAẞE");
    }

    // Regression: lowercasing 'İ' (U+0130, 2 bytes) yields "i\u{307}"
    // (3 bytes), the other direction of length divergence.
    #[test]
    fn highlight_handles_growing_lowercase() {
        let segs = highlight_segments("İstanbul", "istanbul");
        assert_eq!(flatten(&segs), "İstanbul");
    }

    #[test]
    fn highlight_empty_query_passes_through() {
        let segs = highlight_segments("anything", "");
        assert_eq!(flatten(&segs), "anything");
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].kind, SegmentKind::Plain);
    }

    #[test]
    fn highlight_longest_needle_wins() {
        // "foo bar" (full phrase) should win over "foo" / "bar" splits.
        let segs = highlight_segments("a foo bar b", "foo bar");
        assert_eq!(flatten(&segs), "a foo bar b");
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[1].text, "foo bar");
        assert_eq!(segs[1].kind, SegmentKind::Match);
    }
}
