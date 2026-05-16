//! End-to-end Day-3 search tests.
//!
//! Synthesizes a small PDF with `printpdf`, runs the full
//! `pdffff::app::run_index` pipeline to populate SQLite, then runs
//! `pdffff::app::run_search` against the resulting database and
//! asserts on the returned hits. This is the closest analog to a user
//! running `pdffff index` followed by `pdffff search`.

mod common;

use anyhow::Result;
use tempfile::tempdir;

use pdffff::app::{IndexOptions, run_index, run_search};
use pdffff::extract::ensure_pdftotext_available;
use pdffff::query::{DISPLAY_LIMIT, QueryMode};

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
fn regex_and_fuzzy_modes_are_not_yet_implemented() -> Result<()> {
    if !require_pdftotext_or_skip("regex_and_fuzzy_modes_are_not_yet_implemented") {
        return Ok(());
    }

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    std::fs::create_dir_all(&root)?;
    let pdf_path = root.join("paper.pdf");
    common::make_pdf_two_pages(&pdf_path, &["alpha"], &["bravo"]);

    let db_path = tmp.path().join("idx.db");
    run_index(&db_path, &root, &opts())?;

    assert!(run_search(&db_path, "alpha", QueryMode::Regex, DISPLAY_LIMIT).is_err());
    assert!(run_search(&db_path, "alpha", QueryMode::Fuzzy, DISPLAY_LIMIT).is_err());
    Ok(())
}
