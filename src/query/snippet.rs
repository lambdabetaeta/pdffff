//! Hit construction and snippet rendering.
//!
//! The two pairs that used to live here separately — `make_hit_at_norm`
//! / `make_hit_at_utf8` and `render_snippet` / `render_snippet_at_utf8`
//! — collapse to one [`make_hit`] and one [`render_snippet`] thanks to
//! the [`MatchLocation`] enum.

use crate::index::ChunkItem;
use crate::normalize::collapse_whitespace_for_display;

use super::Hit;
use super::candidate::MatchLocation;

/// How many bytes of `text_utf8` to include on each side of the
/// approximate match offset when rendering a snippet.
const SNIPPET_CONTEXT_BYTES: usize = 60;

/// Construct a [`Hit`] for `chunk` with its snippet anchored at `loc`.
pub(crate) fn make_hit(chunk: &ChunkItem, loc: MatchLocation) -> Hit {
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

/// Build a short snippet around `loc` inside `chunk.text_utf8`.
///
/// For [`MatchLocation::Norm`] offsets the function proportionally
/// remaps the `text_norm_ascii` offset to a `text_utf8` byte offset:
/// the two strings diverge after deunicode + lowercase + whitespace
/// collapse, so the remap is best-effort. For [`MatchLocation::Utf8`]
/// the offset is used directly.
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
