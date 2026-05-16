//! Text normalization for indexing and querying.
//!
//! The bigram index is printable-ASCII only — that mirrors the choice in
//! `fff`'s `bigram_filter` (`normalize_byte_scalar`) and keeps the lookup
//! table at a fixed 65536 entries. To make the index useful on real PDF
//! text (which contains accented letters, smart quotes, ligatures, etc.)
//! we first transliterate Unicode to ASCII with `deunicode`, then
//! lowercase, then collapse all whitespace runs to a single space.
//!
//! The bigram extractor itself also performs an in-line lowercasing pass,
//! so this routine's lowercasing is redundant on bytes that survive the
//! ASCII fold, but it is essential to keep `text_norm_ascii` directly
//! usable for cheap literal verification with `memchr` / `aho-corasick`.
//!
//! Two related routines are exported:
//! * [`normalize_for_index`] – used for indexed text and for query
//!   strings before bigram extraction.
//! * [`normalize_query_ascii`] – an explicit alias used at query time so
//!   the call site reads correctly.

use deunicode::deunicode;

/// `BUMP_VERSION` is recorded into `documents.norm_version` so a future
/// change to the algorithm can force re-extraction of stale rows.
pub const NORM_VERSION: i64 = 1;

/// Normalize free-form Unicode text into ASCII bytes suitable for the
/// bigram index and for fast literal verification.
///
/// Steps:
/// 1. `deunicode` — transliterate non-ASCII characters to ASCII.
/// 2. Lowercase ASCII letters.
/// 3. Replace any control char other than '\n' with a single space.
/// 4. Replace '\n' with a single space (we keep page structure separately
///    via the `chunks.page_no` column, not inline).
/// 5. Collapse runs of ASCII whitespace into a single space, trim ends.
pub fn normalize_for_index(text: &str) -> String {
    // deunicode handles Unicode → ASCII transliteration; the result is
    // already pure ASCII bytes that can include letters, digits, punctuation.
    let ascii = deunicode(text);
    let bytes = ascii.as_bytes();

    let mut out = String::with_capacity(bytes.len());
    let mut prev_space = true; // collapse leading whitespace
    for &b in bytes {
        let c = if b.is_ascii_whitespace() || (b < 0x20 && b != b'\n') {
            b' '
        } else if b.is_ascii_uppercase() {
            b | 0x20
        } else {
            b
        };
        if c == b' ' {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(c as char);
            prev_space = false;
        }
    }
    // trim trailing space
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

/// Same as [`normalize_for_index`] – name used at query call sites for
/// readability.
#[inline]
pub fn normalize_query_ascii(query: &str) -> String {
    normalize_for_index(query)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowercases_and_collapses() {
        assert_eq!(normalize_for_index("Hello   WORLD"), "hello world");
    }

    #[test]
    fn deunicodes() {
        let n = normalize_for_index("café — résumé");
        assert!(n.contains("cafe"));
        assert!(n.contains("resume"));
    }

    #[test]
    fn handles_newlines_as_spaces() {
        assert_eq!(normalize_for_index("a\nb\n\nc"), "a b c");
    }

    #[test]
    fn trims_ends() {
        assert_eq!(normalize_for_index("  foo  "), "foo");
    }

    #[test]
    fn empty_input() {
        assert_eq!(normalize_for_index(""), "");
    }
}
