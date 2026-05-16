//! End-to-end Day-2 pipeline tests.
//!
//! Each test:
//! 1. synthesizes one or more tiny real PDFs with `printpdf`,
//! 2. runs the full `pdffff::app::run_index` pipeline against a fresh
//!    SQLite file (scanner → rayon extractor pool → flume channel →
//!    single DB-writer thread),
//! 3. reopens the DB on the main thread and asserts on the rows that
//!    came out the other side.
//!
//! Tests are tagged with `#[ignore]` *only* if the local environment
//! lacks `pdftotext` — see [`require_pdftotext_or_skip`] below.

mod common;

use std::path::Path;

use anyhow::Result;
use rusqlite::params;
use tempfile::tempdir;

use pdffff::app::{IndexOptions, run_index};
use pdffff::db::Db;
use pdffff::extract::{
    CHUNK_OVERLAP_CHARS, CHUNK_WINDOW_CHARS, chunk_page, ensure_pdftotext_available,
};

/// Skip a test gracefully if `pdftotext` isn't on PATH. We still prefer
/// CI to install poppler-utils, but a developer running `cargo test` on
/// a fresh machine should get a useful skip message rather than a panic.
fn require_pdftotext_or_skip(test_name: &str) -> bool {
    if let Err(err) = ensure_pdftotext_available() {
        eprintln!("[{test_name}] skipping: {err}");
        return false;
    }
    true
}

fn opts() -> IndexOptions {
    IndexOptions {
        respect_gitignore: false,
        follow_symlinks: false,
        // A pool size of 2 is enough to exercise the channel without
        // pinning all cores during `cargo test`.
        jobs: Some(2),
        require_pdftotext: true,
    }
}

fn count_chunks(db: &Db, path: &Path) -> Result<i64> {
    let n: i64 = db.conn.query_row(
        "SELECT COUNT(*) FROM chunks c \
         JOIN documents d ON d.doc_id = c.doc_id \
         WHERE d.path = ?1",
        params![path.to_string_lossy()],
        |r| r.get(0),
    )?;
    Ok(n)
}

fn page_nos(db: &Db, path: &Path) -> Result<Vec<i64>> {
    let mut stmt = db.conn.prepare(
        "SELECT DISTINCT c.page_no FROM chunks c \
         JOIN documents d ON d.doc_id = c.doc_id \
         WHERE d.path = ?1 \
         ORDER BY c.page_no",
    )?;
    let rows = stmt.query_map(params![path.to_string_lossy()], |r| r.get::<_, i64>(0))?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r?);
    }
    Ok(out)
}

#[test]
fn pipeline_indexes_two_page_pdf() -> Result<()> {
    if !require_pdftotext_or_skip("pipeline_indexes_two_page_pdf") {
        return Ok(());
    }

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    std::fs::create_dir_all(&root)?;
    let pdf_path = root.join("two-page.pdf");

    // Two pages of distinctive ASCII text — easy for pdftotext to
    // reproduce exactly and easy to assert on.
    common::make_pdf_two_pages(
        &pdf_path,
        &["First page contains alpha bravo charlie."],
        &["Second page mentions delta echo foxtrot."],
    );

    let db_path = tmp.path().join("idx.db");
    let stats = run_index(&db_path, &root, &opts())?;
    assert_eq!(stats.seen, 1, "scanner should see one PDF");
    assert_eq!(stats.dirty, 1, "the new PDF should be dirty");
    assert_eq!(stats.ok, 1, "extraction should report ok");
    assert_eq!(stats.error, 0);
    assert_eq!(stats.empty, 0);
    assert_eq!(stats.deleted, 0);

    let db = Db::open(&db_path)?;
    let docs = db.load_all_documents()?;
    assert_eq!(docs.len(), 1);
    let doc = &docs[0];
    assert_eq!(doc.path, pdf_path, "stored path is the absolute path the scanner emitted");
    assert_eq!(doc.status, pdffff::db::DocStatus::Ok);

    // Page numbering: both pages must produce at least one chunk.
    let pages = page_nos(&db, &pdf_path)?;
    assert_eq!(pages, vec![1, 2], "both pages produce chunks numbered 1, 2");

    // Total chunks should be small (the fixture text is well under
    // CHUNK_WINDOW_CHARS per page).
    let n_chunks = count_chunks(&db, &pdf_path)?;
    assert_eq!(n_chunks, 2, "two short pages should yield exactly two chunks");

    // Content sanity: page 1 mentions "alpha", page 2 mentions "delta".
    let mut stmt = db.conn.prepare(
        "SELECT c.page_no, c.text_utf8, c.text_norm_ascii FROM chunks c \
         JOIN documents d ON d.doc_id = c.doc_id \
         WHERE d.path = ?1 ORDER BY c.page_no",
    )?;
    let rows: Vec<(i64, String, String)> = stmt
        .query_map(params![pdf_path.to_string_lossy()], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    let (p1_no, p1_utf8, p1_norm) = &rows[0];
    let (p2_no, p2_utf8, p2_norm) = &rows[1];
    assert_eq!(*p1_no, 1);
    assert_eq!(*p2_no, 2);
    assert!(p1_utf8.contains("alpha"), "page-1 utf8 has 'alpha': {p1_utf8:?}");
    assert!(p2_utf8.contains("delta"), "page-2 utf8 has 'delta': {p2_utf8:?}");
    assert!(p1_norm.contains("alpha bravo charlie"));
    assert!(p2_norm.contains("delta echo foxtrot"));

    // Report rule 6: "Page N:" must not be embedded in indexed strings.
    assert!(!p1_utf8.to_lowercase().starts_with("page 1"));
    assert!(!p1_norm.starts_with("page 1"));
    Ok(())
}

#[test]
fn pipeline_marks_text_free_pdf_empty() -> Result<()> {
    if !require_pdftotext_or_skip("pipeline_marks_text_free_pdf_empty") {
        return Ok(());
    }

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    std::fs::create_dir_all(&root)?;
    let pdf_path = root.join("empty.pdf");
    common::make_pdf_no_text(&pdf_path);

    let db_path = tmp.path().join("idx.db");
    let stats = run_index(&db_path, &root, &opts())?;
    assert_eq!(stats.seen, 1);
    assert_eq!(stats.dirty, 1);
    assert_eq!(
        stats.empty, 1,
        "a pdf with no text content should land in DocStatus::Empty",
    );
    assert_eq!(stats.ok, 0);
    assert_eq!(stats.error, 0);

    let db = Db::open(&db_path)?;
    let docs = db.load_all_documents()?;
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].status, pdffff::db::DocStatus::Empty);
    assert_eq!(count_chunks(&db, &pdf_path)?, 0, "no chunks for empty doc");
    Ok(())
}

#[test]
fn pipeline_tombstones_deleted_paths() -> Result<()> {
    if !require_pdftotext_or_skip("pipeline_tombstones_deleted_paths") {
        return Ok(());
    }

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    std::fs::create_dir_all(&root)?;
    let a = root.join("a.pdf");
    let b = root.join("b.pdf");
    common::make_pdf_two_pages(&a, &["alpha"], &["alpha2"]);
    common::make_pdf_two_pages(&b, &["bravo"], &["bravo2"]);

    let db_path = tmp.path().join("idx.db");
    let _ = run_index(&db_path, &root, &opts())?;
    {
        let db = Db::open(&db_path)?;
        assert_eq!(db.load_all_documents()?.len(), 2);
    }

    // Delete one file from disk and re-index.
    std::fs::remove_file(&a)?;
    let stats = run_index(&db_path, &root, &opts())?;
    assert_eq!(stats.deleted, 1, "the missing PDF must be tombstoned");

    let db = Db::open(&db_path)?;
    let docs = db.load_all_documents()?;
    let by_path: std::collections::HashMap<_, _> =
        docs.iter().map(|d| (d.path.clone(), d.status)).collect();
    assert_eq!(by_path.get(&a), Some(&pdffff::db::DocStatus::Deleted));
    assert_eq!(by_path.get(&b), Some(&pdffff::db::DocStatus::Ok));
    Ok(())
}

#[test]
fn chunker_sliding_window_invariants() {
    // Pure-Rust check of the chunker's invariants, independent of any
    // PDF or pdftotext. Important: every Unicode char must appear in at
    // least one chunk; consecutive chunks must overlap by exactly
    // CHUNK_OVERLAP_CHARS chars; no chunk exceeds CHUNK_WINDOW_CHARS.

    let total_chars = 5 * CHUNK_WINDOW_CHARS - 137; // 5963 — not a multiple of the step
    let page: String = (0..total_chars)
        .map(|i| (b'a' + (i % 26) as u8) as char)
        .collect();
    let chunks = chunk_page(&page, 1);
    assert!(chunks.len() >= 2, "long page should produce multiple chunks");

    // Char-length bound.
    for c in &chunks {
        let span = (c.char_end - c.char_start) as usize;
        assert!(
            span <= CHUNK_WINDOW_CHARS,
            "chunk span {span} exceeds {CHUNK_WINDOW_CHARS}"
        );
    }

    // Step / overlap: chunk_i starts CHUNK_WINDOW_CHARS - CHUNK_OVERLAP_CHARS
    // chars after chunk_{i-1}, except possibly the last (which is clamped
    // to the page length and may carry < CHUNK_WINDOW_CHARS chars).
    let step = CHUNK_WINDOW_CHARS - CHUNK_OVERLAP_CHARS;
    for w in chunks.windows(2) {
        let a = &w[0];
        let b = &w[1];
        let advance = (b.char_start - a.char_start) as usize;
        assert_eq!(
            advance, step,
            "consecutive chunks must advance by exactly {step} chars (got {advance})"
        );
        let overlap = (a.char_end - b.char_start) as usize;
        // Every non-final pair must overlap by CHUNK_OVERLAP_CHARS chars.
        // (The final chunk may itself be truncated, but the *previous*
        // chunk is always full-window-sized — so the overlap with the
        // last chunk's start is still CHUNK_OVERLAP_CHARS.)
        assert!(
            overlap >= CHUNK_OVERLAP_CHARS,
            "consecutive chunks should overlap by >= {CHUNK_OVERLAP_CHARS} chars (got {overlap})",
        );
    }

    // Coverage: every byte of the page must be in at least one chunk
    // (pure-ASCII text means byte == char, so a byte bitmap suffices).
    let mut covered = vec![false; page.len()];
    for c in &chunks {
        for i in c.char_start as usize..c.char_end as usize {
            covered[i] = true;
        }
    }
    assert!(
        covered.iter().all(|b| *b),
        "every char of the page must appear in at least one chunk",
    );
}
