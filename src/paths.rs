//! Path-shape predicates.
//!
//! Single source of truth for "does this path look like something the
//! pipeline should care about." Today that's just PDF detection; future
//! adapters (epub, html, …) would hang off here.

use std::path::Path;

/// True iff `path`'s extension equals `pdf` case-insensitively.
///
/// Pure on the path string — does not touch the filesystem. The
/// scanner / watcher / coordinator all need the same predicate so they
/// agree on what enters the pipeline; routing through here keeps the
/// decision in one place.
pub fn is_pdf(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e.to_string_lossy().eq_ignore_ascii_case("pdf"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case_insensitive_extension() {
        assert!(is_pdf(Path::new("foo.pdf")));
        assert!(is_pdf(Path::new("FOO.PDF")));
        assert!(is_pdf(Path::new("dir/sub/A.Pdf")));
        assert!(!is_pdf(Path::new("foo.txt")));
        assert!(!is_pdf(Path::new("foo")));
    }
}
