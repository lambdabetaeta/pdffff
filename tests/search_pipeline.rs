//! End-to-end search tests.
//!
//! Synthesizes a small PDF with `printpdf`, runs the full
//! `pdffff::app::run_index` pipeline to populate SQLite, then runs
//! `pdffff::app::run_search` against the resulting database and
//! asserts on the returned hits. This is the closest analog to a user
//! running `pdffff index` followed by `pdffff search`.
//!
//! Day-6 coverage: the regex/fuzzy modes and `run_rebuild` are also
//! exercised through their public APIs.

mod common;

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Result;
use tempfile::tempdir;

use pdffff::app::{IndexOptions, WatchOptions, run_index, run_rebuild, run_search, run_watch};
use pdffff::db::Db;
use pdffff::extract::ensure_pdftotext_available;
use pdffff::index::load_base_index_from_db;
use pdffff::query::{DISPLAY_LIMIT, Hit, QueryMode, search};

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
        jobs: Some(2),
        require_pdftotext: true,
    }
}

#[test]
fn search_finds_known_token_in_indexed_pdf() -> Result<()> {
    if !require_pdftotext_or_skip("search_finds_known_token_in_indexed_pdf") {
        return Ok(());
    }

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    std::fs::create_dir_all(&root)?;
    let pdf_path = root.join("paper.pdf");
    common::make_pdf_two_pages(
        &pdf_path,
        &["First page contains alpha bravo charlie."],
        &["Second page mentions delta echo foxtrot."],
    );

    let db_path = tmp.path().join("idx.db");
    let stats = run_index(&db_path, &root, &opts())?;
    assert_eq!(stats.ok, 1);

    let hits = run_search(&db_path, "delta echo", QueryMode::Literal, DISPLAY_LIMIT)?;
    assert!(!hits.is_empty(), "expected at least one hit for 'delta echo'");
    let hit = hits.iter().find(|h| h.page_no == 2).expect("hit on page 2");
    assert_eq!(hit.path, pdf_path.to_string_lossy(), "path matches indexed path");
    assert!(
        hit.snippet.to_lowercase().contains("delta"),
        "snippet should mention the match: {:?}",
        hit.snippet
    );
    Ok(())
}

#[test]
fn search_for_missing_token_returns_zero_hits() -> Result<()> {
    if !require_pdftotext_or_skip("search_for_missing_token_returns_zero_hits") {
        return Ok(());
    }

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    std::fs::create_dir_all(&root)?;
    let pdf_path = root.join("paper.pdf");
    common::make_pdf_two_pages(&pdf_path, &["alpha bravo charlie"], &["delta echo foxtrot"]);

    let db_path = tmp.path().join("idx.db");
    run_index(&db_path, &root, &opts())?;

    let hits = run_search(
        &db_path,
        "no_such_token_xyzzy",
        QueryMode::Literal,
        DISPLAY_LIMIT,
    )?;
    assert!(hits.is_empty(), "unknown token must return no hits");
    Ok(())
}

#[test]
fn search_respects_display_limit() -> Result<()> {
    if !require_pdftotext_or_skip("search_respects_display_limit") {
        return Ok(());
    }

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    std::fs::create_dir_all(&root)?;

    // A page of mostly repeated tokens — every chunk on every page
    // will contain "needle", so a corpus of N PDFs with M pages each
    // produces at least N*M hits.
    let page = "needle ".repeat(80);
    for i in 0..5 {
        let pdf_path = root.join(format!("paper-{i}.pdf"));
        common::make_pdf_two_pages(&pdf_path, &[page.as_str()], &[page.as_str()]);
    }

    let db_path = tmp.path().join("idx.db");
    let stats = run_index(&db_path, &root, &opts())?;
    assert_eq!(stats.ok, 5);

    let hits = run_search(&db_path, "needle", QueryMode::Literal, 3)?;
    assert_eq!(hits.len(), 3, "limit must cap the returned hits");
    Ok(())
}

#[test]
fn loaded_base_index_has_bigram_prefilter_attached() -> Result<()> {
    if !require_pdftotext_or_skip("loaded_base_index_has_bigram_prefilter_attached") {
        return Ok(());
    }

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    std::fs::create_dir_all(&root)?;
    let pdf_path = root.join("paper.pdf");
    common::make_pdf_two_pages(
        &pdf_path,
        &["alpha bravo charlie"],
        &["delta echo foxtrot"],
    );

    let db_path = tmp.path().join("idx.db");
    let stats = run_index(&db_path, &root, &opts())?;
    assert_eq!(stats.ok, 1);

    // Reload the way `run_search` would and inspect the attached
    // bigram index directly.
    let db = Db::open(&db_path)?;
    let base = load_base_index_from_db(&db)?;
    let bigrams = base.bigrams.as_ref().expect("bigrams attached after Day 4");
    assert_eq!(
        bigrams.populated(),
        base.chunks.len(),
        "every chunk should contribute to the bigram index",
    );
    Ok(())
}

#[test]
fn search_finds_token_unique_to_one_document() -> Result<()> {
    if !require_pdftotext_or_skip("search_finds_token_unique_to_one_document") {
        return Ok(());
    }

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    std::fs::create_dir_all(&root)?;
    let a = root.join("a.pdf");
    let b = root.join("b.pdf");
    let c = root.join("c.pdf");
    common::make_pdf_two_pages(&a, &["common page words"], &["nothing special here"]);
    common::make_pdf_two_pages(&b, &["unrelated content"], &["another generic page"]);
    // Only c contains the distinctive token.
    common::make_pdf_two_pages(
        &c,
        &["totally normal text"],
        &["the magic word is xylotomous"],
    );

    let db_path = tmp.path().join("idx.db");
    let stats = run_index(&db_path, &root, &opts())?;
    assert_eq!(stats.ok, 3);

    let hits = run_search(&db_path, "xylotomous", QueryMode::Literal, DISPLAY_LIMIT)?;
    assert_eq!(hits.len(), 1, "exactly one hit for the unique token");
    assert_eq!(
        hits[0].path,
        c.to_string_lossy(),
        "the hit must be in the document containing the token",
    );
    Ok(())
}

#[test]
fn regex_mode_finds_anchored_literal_match() -> Result<()> {
    if !require_pdftotext_or_skip("regex_mode_finds_anchored_literal_match") {
        return Ok(());
    }

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    std::fs::create_dir_all(&root)?;
    let pdf_path = root.join("paper.pdf");
    common::make_pdf_two_pages(
        &pdf_path,
        &["Page one mentions invoice number 42 here."],
        &["Page two has nothing useful."],
    );

    let db_path = tmp.path().join("idx.db");
    let stats = run_index(&db_path, &root, &opts())?;
    assert_eq!(stats.ok, 1);

    // A regex with a literal anchor ("invoice") and a one-digit
    // wildcard pattern — must hit the document on page 1.
    let hits = run_search(
        &db_path,
        r"invoice number \d+",
        QueryMode::Regex,
        DISPLAY_LIMIT,
    )?;
    assert!(!hits.is_empty(), "expected at least one regex hit");
    let hit = hits.iter().find(|h| h.page_no == 1);
    assert!(
        hit.is_some(),
        "expected regex hit on page 1 of paper.pdf, got hits: {:?}",
        hits.iter().map(|h| (h.path.as_str(), h.page_no)).collect::<Vec<_>>(),
    );
    Ok(())
}

#[test]
fn regex_invalid_pattern_returns_error() -> Result<()> {
    if !require_pdftotext_or_skip("regex_invalid_pattern_returns_error") {
        return Ok(());
    }

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    std::fs::create_dir_all(&root)?;
    let pdf_path = root.join("paper.pdf");
    common::make_pdf_two_pages(&pdf_path, &["alpha"], &["bravo"]);

    let db_path = tmp.path().join("idx.db");
    run_index(&db_path, &root, &opts())?;

    let err = run_search(&db_path, "[invalid", QueryMode::Regex, DISPLAY_LIMIT);
    assert!(err.is_err(), "malformed regex must surface a compile error");
    Ok(())
}

#[test]
fn fuzzy_mode_tolerates_a_single_typo() -> Result<()> {
    if !require_pdftotext_or_skip("fuzzy_mode_tolerates_a_single_typo") {
        return Ok(());
    }

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    std::fs::create_dir_all(&root)?;
    let target_pdf = root.join("target.pdf");
    let other_pdf = root.join("other.pdf");
    common::make_pdf_two_pages(
        &target_pdf,
        &["The kavinsky monorail is awaiting passengers."],
        &["unrelated content on page two."],
    );
    common::make_pdf_two_pages(
        &other_pdf,
        &["Completely different sentence one."],
        &["Completely different sentence two."],
    );

    let db_path = tmp.path().join("idx.db");
    let stats = run_index(&db_path, &root, &opts())?;
    assert_eq!(stats.ok, 2);

    // Query has a one-char transposition vs. the indexed token
    // "kavinsky".
    let hits = run_search(&db_path, "kavnisky", QueryMode::Fuzzy, DISPLAY_LIMIT)?;
    assert!(!hits.is_empty(), "fuzzy must yield at least one hit");
    let top3 = &hits[..hits.len().min(3)];
    assert!(
        top3.iter().any(|h| Path::new(&h.path) == target_pdf.as_path()),
        "target.pdf must appear in fuzzy top-3; got {:?}",
        top3.iter().map(|h| h.path.as_str()).collect::<Vec<_>>(),
    );
    Ok(())
}

#[test]
fn rebuild_returns_stats_matching_the_database() -> Result<()> {
    if !require_pdftotext_or_skip("rebuild_returns_stats_matching_the_database") {
        return Ok(());
    }

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    std::fs::create_dir_all(&root)?;
    let a = root.join("a.pdf");
    let b = root.join("b.pdf");
    common::make_pdf_two_pages(&a, &["alpha bravo charlie"], &["delta echo foxtrot"]);
    common::make_pdf_two_pages(&b, &["golf hotel india"], &["juliet kilo lima"]);

    let db_path = tmp.path().join("idx.db");
    run_index(&db_path, &root, &opts())?;

    let stats = run_rebuild(&db_path)?;
    // Two docs and at least one chunk per page.
    assert_eq!(stats.docs, 2);
    assert!(stats.chunks >= 4, "two two-page pdfs should produce ≥ 4 chunks");
    assert!(
        stats.bigram_heap_bytes > 0,
        "a populated corpus must build a non-trivial bigram index",
    );
    Ok(())
}

#[test]
fn rebuild_after_watch_modifies_collapses_overlay() -> Result<()> {
    if !require_pdftotext_or_skip("rebuild_after_watch_modifies_collapses_overlay") {
        return Ok(());
    }

    const PATIENCE: Duration = Duration::from_secs(10);
    const POLL: Duration = Duration::from_millis(50);

    fn wait_for<F>(
        state: &pdffff::index::IndexState,
        query: &str,
        predicate: F,
        label: &str,
    ) where
        F: Fn(&[Hit]) -> bool,
    {
        let start = Instant::now();
        loop {
            let hits = search(state, query, QueryMode::Literal, DISPLAY_LIMIT)
                .expect("literal search never errors");
            if predicate(&hits) {
                return;
            }
            if start.elapsed() >= PATIENCE {
                panic!("timed out waiting for {label} (query={query:?})");
            }
            std::thread::sleep(POLL);
        }
    }

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    std::fs::create_dir_all(&root)?;
    let a = root.join("a.pdf");
    common::make_pdf_two_pages(&a, &["originalrebuildtoken"], &["page two stays"]);

    let db_path = tmp.path().join("idx.db");

    let watch_opts = WatchOptions {
        respect_gitignore: false,
        follow_symlinks: false,
        jobs: Some(2),
        require_pdftotext: true,
        debounce: Some(Duration::from_millis(60)),
    };
    let handle = run_watch(&db_path, &root, &watch_opts)?;
    let state = handle.state.clone();

    // Initial token visible via the (now-loaded) base index.
    wait_for(
        &state,
        "originalrebuildtoken",
        |hits| hits.iter().any(|h| Path::new(&h.path) == a.as_path()),
        "initial token",
    );

    // Modify a.pdf — the writer thread publishes the new chunks to
    // the overlay (and may or may not rebuild, depending on whether
    // the overlay crosses the threshold).
    common::make_pdf_two_pages(&a, &["modifiedrebuildtoken now lives here"], &["page two stays"]);

    // The fresh token must appear via the overlay (or via the new
    // base, if a rebuild fired).
    wait_for(
        &state,
        "modifiedrebuildtoken",
        |hits| hits.iter().any(|h| Path::new(&h.path) == a.as_path()),
        "modified token after watcher pickup",
    );

    handle.stop()?;

    // Now force a rebuild explicitly and assert the new base index
    // both contains the modified content and has zero overlay rows.
    let stats = run_rebuild(&db_path)?;
    assert_eq!(stats.docs, 1, "exactly one doc after the rebuild");
    let db = Db::open_reader(&db_path)?;
    let base = load_base_index_from_db(&db)?;
    let any_modified = base
        .chunks
        .iter()
        .any(|c| c.text_norm_ascii.windows(b"modifiedrebuildtoken".len())
            .any(|w| w == b"modifiedrebuildtoken"));
    assert!(any_modified, "base index must carry the modified content");
    let any_original = base
        .chunks
        .iter()
        .any(|c| c.text_norm_ascii.windows(b"originalrebuildtoken".len())
            .any(|w| w == b"originalrebuildtoken"));
    assert!(!any_original, "base index must NOT carry the stale content");

    Ok(())
}
