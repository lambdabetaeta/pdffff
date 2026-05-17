//! PDF extraction via Poppler's `pdftotext`.
//!
//! Architecture matches the report's prescription and rga's adapter:
//!
//! 1. Stream the PDF bytes into `pdftotext - -` (read PDF on stdin, write
//!    text on stdout). This is exactly what `rga`'s built-in poppler
//!    adapter does and avoids leaving temp files on disk.
//! 2. Capture UTF-8 stdout and split on ASCII form-feed `\x0c` into pages.
//!    `pdftotext` emits one `\x0c` after every page, so the natural split
//!    yields one entry per page (with a trailing empty fragment after the
//!    last form-feed which we drop).
//! 3. Normalize each page through [`crate::normalize::normalize_for_index`]
//!    into the column stored in `chunks.text_norm_ascii`.
//! 4. Chunk each page:
//!    * page text ≤ 1200 *chars* (Unicode scalar values) → one chunk;
//!    * page text >  1200 chars  → sliding 1200-char windows with
//!      200-char overlap (step = 1000 chars).
//!    `char_start` / `char_end` are recorded as **byte** offsets into the
//!    page's original UTF-8 text (which is what `text_utf8` is a slice of).
//! 5. `preview` is a short head-of-chunk (≤ 200 chars of `text_utf8`) with
//!    whitespace collapsed for cheap UI use.
//!
//! Error handling:
//! * `pdftotext` missing on PATH ⇒ surfaced at startup via
//!   [`ensure_pdftotext_available`]. Per-document extraction also returns
//!   the underlying I/O error if exec fails mid-run.
//! * `pdftotext` exits nonzero ⇒ [`ExtractedDoc`] with
//!   `status = DocStatus::Error`, stderr captured in `error_text`, zero
//!   chunks.
//! * stdout empty / whitespace-only ⇒ `status = DocStatus::Empty`, zero
//!   chunks.
//! * Only `status = DocStatus::Ok` carries chunks.

use anyhow::{Context, Result, anyhow, bail};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::db::{ChunkInsert, DocStatus, EXTRACTOR_NAME, ExtractedDoc, extractor_version};
use crate::normalize::{NORM_VERSION, normalize_for_index};
use crate::scanner::ScanJob;

/// Window size (in Unicode chars) for chunking a long page.
pub const CHUNK_WINDOW_CHARS: usize = 1200;
/// Overlap (in Unicode chars) between consecutive windows.
pub const CHUNK_OVERLAP_CHARS: usize = 200;
/// Maximum length of the `chunks.preview` field, in bytes (ASCII-only).
pub const PREVIEW_MAX_BYTES: usize = 200;

/// Verify that `pdftotext` is available on `PATH` *and* usable before
/// launching the extractor pool. Two failure modes are surfaced
/// distinctly so the user gets an actionable error instead of a stream
/// of per-doc extractor failures later:
///
/// * `pdftotext` is not on `PATH` ⇒ point at the install command.
/// * `pdftotext` is on `PATH` but `pdftotext -v` exits nonzero with no
///   stderr ⇒ refuse to run; a silent failure typically means the
///   binary is broken (mismatched libraries, sandbox blocking exec,
///   etc.) and continuing would just churn through `Error` rows.
///
/// `pdftotext -v` writes its banner to stderr and exits 0 on every
/// poppler version I've seen; we accept either exit-0 or a nonempty
/// stderr banner.
pub fn ensure_pdftotext_available() -> Result<()> {
    let out = match Command::new("pdftotext").arg("-v").output() {
        Ok(o) => o,
        Err(err) => {
            return Err(anyhow!(
                "pdftotext (poppler-utils) is required but not on PATH: {err}.\n\
                 Install it (e.g. `apt-get install poppler-utils` or `brew install poppler`) \
                 and retry."
            ));
        }
    };
    if !out.status.success() && out.stderr.is_empty() {
        bail!(
            "pdftotext is present but `pdftotext -v` failed silently \
             (status {:?}); refusing to run.",
            out.status.code()
        );
    }
    Ok(())
}

/// Split the full extracted document text on ASCII form-feed (`\x0c`).
///
/// `pdftotext` emits a form-feed after every page, so a document with `N`
/// pages produces `N` `\x0c` bytes. The trailing empty fragment after the
/// final form-feed is dropped; any other empty pages are preserved so
/// downstream metadata keeps the same numbering.
pub fn split_pages(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut pages: Vec<&str> = text.split('\x0c').collect();
    // If the last element is empty (typical: `pdftotext` always ends with
    // a form feed), drop it — otherwise we'd over-count by one page.
    if pages.last().is_some_and(|p| p.is_empty()) {
        pages.pop();
    }
    pages
}

/// Run `pdftotext - -` on `path` and capture stdout + stderr + status.
///
/// Streams the PDF bytes in via stdin (the first `-`) and reads the
/// extracted text from stdout (the second `-`) so no temporary files are
/// written. This matches rga's built-in poppler adapter.
fn run_pdftotext(path: &Path) -> Result<PdftotextOutput> {
    let pdf_bytes = std::fs::read(path)
        .with_context(|| format!("reading PDF bytes from {}", path.display()))?;

    let mut child = Command::new("pdftotext")
        .arg("-")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| "spawning pdftotext")?;

    // Write the PDF on stdin in a scope so the pipe is closed before we
    // wait — otherwise `pdftotext` would block forever on EOF.
    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("pdftotext stdin not piped"))?;
        stdin
            .write_all(&pdf_bytes)
            .with_context(|| "writing PDF bytes to pdftotext stdin")?;
    }

    let output = child
        .wait_with_output()
        .with_context(|| "waiting for pdftotext")?;
    Ok(PdftotextOutput {
        status_success: output.status.success(),
        exit_code: output.status.code(),
        stdout: output.stdout,
        stderr: output.stderr,
    })
}

struct PdftotextOutput {
    status_success: bool,
    exit_code: Option<i32>,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

/// Extract `job.path` end-to-end and produce the [`ExtractedDoc`] that
/// will go to the DB writer thread.
///
/// Never panics on PDF errors — they are converted into
/// `status = Error` rows with stderr in `error_text` so the document is
/// recorded as having been attempted (and won't loop forever in the
/// scanner's `RetryAfterError` path until something on disk changes).
pub fn extract_pdf(job: &ScanJob) -> Result<ExtractedDoc> {
    let extractor_ver = extractor_version();

    let base = |status: DocStatus,
                error_text: Option<String>,
                page_count: u32,
                chunks: Vec<ChunkInsert>| {
        ExtractedDoc {
            path: job.path.clone(),
            size_bytes: job.size_bytes,
            mtime_ns: job.mtime_ns,
            dev: job.dev,
            ino: job.ino,
            extractor: EXTRACTOR_NAME.to_string(),
            extractor_version: extractor_ver.clone(),
            norm_version: NORM_VERSION,
            page_count,
            status,
            error_text,
            chunks,
        }
    };

    tracing::debug!(path = %job.path.display(), reason = ?job.reason, "extracting");

    let out = match run_pdftotext(&job.path) {
        Ok(o) => o,
        Err(err) => {
            // I/O failure spawning / writing to pdftotext. Record an
            // `error` row so the scanner's `RetryAfterError` path can pick
            // it up again on the next run.
            tracing::warn!(path = %job.path.display(), ?err, "pdftotext exec failed");
            return Ok(base(
                DocStatus::Error,
                Some(format!("pdftotext exec failed: {err:#}")),
                0,
                Vec::new(),
            ));
        }
    };

    if !out.status_success {
        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
        tracing::warn!(
            path = %job.path.display(),
            code = ?out.exit_code,
            stderr = %stderr,
            "pdftotext exited nonzero",
        );
        return Ok(base(
            DocStatus::Error,
            Some(format_error_text(out.exit_code, &stderr)),
            0,
            Vec::new(),
        ));
    }

    // pdftotext is documented to emit UTF-8. If a PDF's text stream
    // contains bytes that aren't valid UTF-8 we replace them rather than
    // discarding the page — this matches rga's lenient decoding path.
    let stdout_text: String = match String::from_utf8(out.stdout) {
        Ok(s) => s,
        Err(err) => {
            tracing::debug!(
                path = %job.path.display(),
                "pdftotext stdout was not valid UTF-8; lossy-decoding",
            );
            String::from_utf8_lossy(err.as_bytes()).into_owned()
        }
    };

    if stdout_text.trim().is_empty() {
        tracing::info!(path = %job.path.display(), "pdftotext stdout empty");
        return Ok(base(DocStatus::Empty, None, 0, Vec::new()));
    }

    let pages = split_pages(&stdout_text);
    let page_count = pages.len() as u32;
    let mut all_chunks: Vec<ChunkInsert> = Vec::new();
    for (i, page_text) in pages.iter().enumerate() {
        let page_no = (i + 1) as u32; // 1-based, matches user-facing page numbers
        let page_chunks = chunk_page(page_text, page_no);
        all_chunks.extend(page_chunks);
    }

    if all_chunks.is_empty() {
        // Every page was whitespace-only.
        return Ok(base(DocStatus::Empty, None, page_count, Vec::new()));
    }

    tracing::debug!(
        path = %job.path.display(),
        pages = page_count,
        chunks = all_chunks.len(),
        "extracted ok",
    );
    Ok(base(DocStatus::Ok, None, page_count, all_chunks))
}

fn format_error_text(code: Option<i32>, stderr: &str) -> String {
    match code {
        Some(c) => {
            if stderr.is_empty() {
                format!("pdftotext exited with status {c}")
            } else {
                format!("pdftotext exited with status {c}: {stderr}")
            }
        }
        None => {
            if stderr.is_empty() {
                "pdftotext terminated by signal".to_string()
            } else {
                format!("pdftotext terminated by signal: {stderr}")
            }
        }
    }
}

/// Chunk a single page's UTF-8 text.
///
/// Returns one chunk if the page is at most [`CHUNK_WINDOW_CHARS`] chars,
/// or sliding windows of [`CHUNK_WINDOW_CHARS`] with
/// [`CHUNK_OVERLAP_CHARS`] overlap otherwise. Whitespace-only pages yield
/// no chunks. `char_start` / `char_end` are byte offsets into `page_text`.
pub fn chunk_page(page_text: &str, page_no: u32) -> Vec<ChunkInsert> {
    if page_text.trim().is_empty() {
        return Vec::new();
    }

    // Precompute byte offset for every char boundary, then a trailing
    // offset = page_text.len(). Indexing by char index is then a single
    // array lookup, and windowing is a simple stride over the array.
    let mut char_byte_offsets: Vec<usize> = page_text.char_indices().map(|(b, _)| b).collect();
    char_byte_offsets.push(page_text.len());
    let total_chars = char_byte_offsets.len() - 1;

    let mut chunks = Vec::new();
    if total_chars <= CHUNK_WINDOW_CHARS {
        if let Some(chunk) = build_chunk(page_text, page_no, 0, 0, total_chars, &char_byte_offsets)
        {
            chunks.push(chunk);
        }
        return chunks;
    }

    let step = CHUNK_WINDOW_CHARS - CHUNK_OVERLAP_CHARS;
    let mut chunk_ord: u32 = 0;
    let mut start_char: usize = 0;
    loop {
        let end_char = (start_char + CHUNK_WINDOW_CHARS).min(total_chars);
        if let Some(chunk) =
            build_chunk(page_text, page_no, chunk_ord, start_char, end_char, &char_byte_offsets)
        {
            chunks.push(chunk);
        }
        chunk_ord += 1;
        if end_char == total_chars {
            break;
        }
        start_char += step;
    }
    chunks
}

fn build_chunk(
    page_text: &str,
    page_no: u32,
    chunk_ord: u32,
    start_char: usize,
    end_char: usize,
    char_byte_offsets: &[usize],
) -> Option<ChunkInsert> {
    if start_char >= end_char {
        return None;
    }
    let byte_start = char_byte_offsets[start_char];
    let byte_end = char_byte_offsets[end_char];
    let text_utf8 = &page_text[byte_start..byte_end];
    // Skip windows that are pure whitespace — they contribute nothing to
    // search and would waste a row.
    if text_utf8.trim().is_empty() {
        return None;
    }
    let text_norm_ascii = normalize_for_index(text_utf8);
    let preview = make_preview(text_utf8);
    Some(ChunkInsert {
        page_no,
        chunk_ord,
        char_start: byte_start as u32,
        char_end: byte_end as u32,
        text_utf8: text_utf8.to_string(),
        text_norm_ascii,
        preview,
    })
}

/// Build a short head-of-chunk preview: take the first
/// [`PREVIEW_MAX_BYTES`] ASCII-ish characters of `text_utf8`, collapse
/// whitespace runs, trim ends.
fn make_preview(text: &str) -> String {
    let mut out = String::with_capacity(PREVIEW_MAX_BYTES.min(text.len()));
    let mut prev_space = true;
    for ch in text.chars() {
        if out.len() >= PREVIEW_MAX_BYTES {
            break;
        }
        let c = if ch.is_whitespace() { ' ' } else { ch };
        if c == ' ' {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            // If adding this char would overflow the byte budget, stop —
            // we must not split a multi-byte char.
            let mut buf = [0u8; 4];
            let s = c.encode_utf8(&mut buf);
            if out.len() + s.len() > PREVIEW_MAX_BYTES {
                break;
            }
            out.push_str(s);
            prev_space = false;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

/// Return the first line of `pdftotext -v` output, or the string
/// `"missing"` if the binary is not on PATH / refused to run. Used by
/// `pdffff diagnose` to surface the installed poppler version.
pub fn extractor_version_or_missing() -> String {
    crate::db::extractor_version()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_pages_drops_trailing_formfeed() {
        // pdftotext emits a trailing \x0c after the last page.
        let s = "page one\x0cpage two\x0c";
        let pages = split_pages(s);
        assert_eq!(pages, vec!["page one", "page two"]);
    }

    #[test]
    fn split_pages_keeps_empty_interior_pages() {
        let s = "p1\x0c\x0cp3\x0c";
        let pages = split_pages(s);
        assert_eq!(pages, vec!["p1", "", "p3"]);
    }

    #[test]
    fn split_pages_empty_input() {
        assert!(split_pages("").is_empty());
    }

    #[test]
    fn chunk_short_page_makes_one_chunk() {
        let page = "Hello world. This is a short page.";
        let chunks = chunk_page(page, 1);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].page_no, 1);
        assert_eq!(chunks[0].chunk_ord, 0);
        assert_eq!(chunks[0].char_start, 0);
        assert_eq!(chunks[0].char_end as usize, page.len());
        assert_eq!(chunks[0].text_utf8, page);
        assert!(chunks[0].text_norm_ascii.contains("hello world"));
    }

    #[test]
    fn chunk_empty_page_yields_no_chunks() {
        assert!(chunk_page("", 1).is_empty());
        assert!(chunk_page("    \n\t  ", 1).is_empty());
    }

    #[test]
    fn chunk_long_page_slides_with_overlap() {
        // 3500 'a' chars; expected windows at char ranges
        // [0,1200), [1000,2200), [2000,3200), [3000,3500)
        let page: String = "a".repeat(3500);
        let chunks = chunk_page(&page, 7);
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].char_start, 0);
        assert_eq!(chunks[0].char_end, 1200);
        assert_eq!(chunks[1].char_start, 1000);
        assert_eq!(chunks[1].char_end, 2200);
        assert_eq!(chunks[2].char_start, 2000);
        assert_eq!(chunks[2].char_end, 3200);
        assert_eq!(chunks[3].char_start, 3000);
        assert_eq!(chunks[3].char_end, 3500);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.page_no, 7);
            assert_eq!(c.chunk_ord, i as u32);
        }
    }

    #[test]
    fn chunk_long_page_covers_every_char() {
        let page: String = (0..2500u32).map(|i| (b'a' + (i % 26) as u8) as char).collect();
        let chunks = chunk_page(&page, 1);
        assert!(chunks.len() >= 2);
        let mut covered = vec![false; page.len()];
        for c in &chunks {
            assert!(c.char_end as usize - c.char_start as usize <= CHUNK_WINDOW_CHARS);
            for i in c.char_start as usize..c.char_end as usize {
                covered[i] = true;
            }
        }
        assert!(covered.iter().all(|b| *b), "every byte must be in at least one chunk");
    }

    #[test]
    fn preview_collapses_and_caps() {
        let p = make_preview("  hello\nworld\t\tfoo    bar  ");
        assert_eq!(p, "hello world foo bar");
        // Capacity bound: a long input is truncated to <= PREVIEW_MAX_BYTES.
        let long = "x".repeat(1000);
        let p = make_preview(&long);
        assert!(p.len() <= PREVIEW_MAX_BYTES);
    }

    #[test]
    fn no_page_n_text_embedded_in_indexed_strings() {
        // The page number must be metadata only, never embedded into
        // text_utf8 / text_norm_ascii / preview. (Report rule 6.)
        let page = "the content of page one";
        let chunks = chunk_page(page, 42);
        let c = &chunks[0];
        assert!(!c.text_utf8.starts_with("Page 42:"));
        assert!(!c.text_norm_ascii.contains("page 42"));
        assert!(!c.preview.starts_with("Page 42:"));
    }
}
