//! End-to-end Day-5 watch pipeline tests.
//!
//! These tests spin up the real `run_watch` pipeline against a fresh
//! tempdir + sqlite, observe its index live via the `WatchHandle`'s
//! `IndexState`, and poll until the live index reflects the on-disk
//! state. No `sleep`-then-assert anywhere: every check is a bounded
//! polling loop that times out at 10s.

mod common;

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use tempfile::tempdir;

use pdffff::app::{WatchOptions, run_watch};
use pdffff::extract::ensure_pdftotext_available;
use pdffff::index::IndexState;
use pdffff::query::{DISPLAY_LIMIT, Hit, QueryMode, search};

const PATIENCE: Duration = Duration::from_secs(10);
const POLL: Duration = Duration::from_millis(50);

fn require_pdftotext_or_skip(test_name: &str) -> bool {
    if let Err(err) = ensure_pdftotext_available() {
        eprintln!("[{test_name}] skipping: {err}");
        return false;
    }
    true
}

fn opts() -> WatchOptions {
    WatchOptions {
        respect_gitignore: false,
        follow_symlinks: false,
        jobs: Some(2),
        require_pdftotext: true,
        // Use the bottom of the 50-250 ms band so tests don't wait
        // longer than they have to for events to flush.
        debounce: Some(Duration::from_millis(60)),
    }
}

/// Poll `search(query)` until `predicate` is true or the patience
/// timeout elapses.
fn wait_for(
    state: &Arc<IndexState>,
    query: &str,
    predicate: impl Fn(&[Hit]) -> bool,
    label: &str,
) {
    let start = Instant::now();
    loop {
        let hits = search(state, query, QueryMode::Literal, DISPLAY_LIMIT)
            .expect("query never errors in literal mode");
        if predicate(&hits) {
            return;
        }
        if start.elapsed() >= PATIENCE {
            panic!(
                "timed out waiting for `{label}` (query={query:?}): \
                 got {} hits after {:?}",
                hits.len(),
                start.elapsed(),
            );
        }
        std::thread::sleep(POLL);
    }
}

fn paths_in_hits(hits: &[Hit]) -> Vec<&str> {
    hits.iter().map(|h| h.path.as_str()).collect()
}

#[test]
fn watch_picks_up_initial_pdfs_and_new_arrivals() -> Result<()> {
    if !require_pdftotext_or_skip("watch_picks_up_initial_pdfs_and_new_arrivals") {
        return Ok(());
    }

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    std::fs::create_dir_all(&root)?;
    let a = root.join("a.pdf");
    common::make_pdf_two_pages(
        &a,
        &["alpha bravo charlie quickbrownfoxtoken"],
        &["page two of a"],
    );

    let db_path = tmp.path().join("idx.db");
    let handle = run_watch(&db_path, &root, &opts())?;
    let state = handle.state.clone();

    // The initial synchronous scan must already have indexed a.pdf.
    wait_for(
        &state,
        "quickbrownfoxtoken",
        |hits| hits.iter().any(|h| Path::new(&h.path) == a.as_path()),
        "initial token in a.pdf",
    );

    // Now drop a second PDF; the watcher must pick it up.
    let b = root.join("b.pdf");
    common::make_pdf_two_pages(&b, &["bravoecho secondonlytoken"], &["page two of b"]);
    wait_for(
        &state,
        "secondonlytoken",
        |hits| hits.iter().any(|h| Path::new(&h.path) == b.as_path()),
        "secondonlytoken in b.pdf",
    );

    handle.stop()?;
    Ok(())
}

#[test]
fn watch_removes_deleted_pdfs_from_index() -> Result<()> {
    if !require_pdftotext_or_skip("watch_removes_deleted_pdfs_from_index") {
        return Ok(());
    }

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    std::fs::create_dir_all(&root)?;
    let a = root.join("a.pdf");
    let b = root.join("b.pdf");
    common::make_pdf_two_pages(&a, &["uniquealphatoken in document a"], &["page two of a"]);
    common::make_pdf_two_pages(&b, &["uniquebravotoken in document b"], &["page two of b"]);

    let db_path = tmp.path().join("idx.db");
    let handle = run_watch(&db_path, &root, &opts())?;
    let state = handle.state.clone();

    // Both must be present initially.
    wait_for(
        &state,
        "uniquealphatoken",
        |hits| hits.iter().any(|h| Path::new(&h.path) == a.as_path()),
        "alpha token in a",
    );
    wait_for(
        &state,
        "uniquebravotoken",
        |hits| hits.iter().any(|h| Path::new(&h.path) == b.as_path()),
        "bravo token in b",
    );

    // Delete a.pdf.
    std::fs::remove_file(&a)?;

    // Searching for a's token must eventually return zero hits.
    wait_for(
        &state,
        "uniquealphatoken",
        |hits| hits.is_empty(),
        "alpha token disappears",
    );

    // b's token must still be present (sanity).
    let hits = search(&state, "uniquebravotoken", QueryMode::Literal, DISPLAY_LIMIT)?;
    assert!(
        hits.iter().any(|h| Path::new(&h.path) == b.as_path()),
        "b.pdf must still be indexed; got paths = {:?}",
        paths_in_hits(&hits),
    );

    handle.stop()?;
    Ok(())
}

#[test]
fn watch_reflects_modified_pdf_contents() -> Result<()> {
    if !require_pdftotext_or_skip("watch_reflects_modified_pdf_contents") {
        return Ok(());
    }

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    std::fs::create_dir_all(&root)?;
    let a = root.join("a.pdf");
    common::make_pdf_two_pages(&a, &["originalmagictoken"], &["page two stays"]);

    let db_path = tmp.path().join("idx.db");
    let handle = run_watch(&db_path, &root, &opts())?;
    let state = handle.state.clone();

    // Original content visible.
    wait_for(
        &state,
        "originalmagictoken",
        |hits| hits.iter().any(|h| Path::new(&h.path) == a.as_path()),
        "original token",
    );
    let no_new = search(&state, "freshmagictoken", QueryMode::Literal, DISPLAY_LIMIT)?;
    assert!(no_new.is_empty(), "freshmagictoken must not be present yet");

    // Rewrite the same path with new contents.
    common::make_pdf_two_pages(&a, &["freshmagictoken now lives here"], &["page two stays"]);

    // The new token appears, and the old one disappears.
    wait_for(
        &state,
        "freshmagictoken",
        |hits| hits.iter().any(|h| Path::new(&h.path) == a.as_path()),
        "fresh token after modify",
    );
    wait_for(
        &state,
        "originalmagictoken",
        |hits| hits.is_empty(),
        "original token disappears after modify",
    );

    handle.stop()?;
    Ok(())
}
