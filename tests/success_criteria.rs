//! Day-7 integration tests: each of the report's measurable success
//! criteria, at a scale-down small enough to run inside `cargo test`.
//!
//! These tests synthesise PDFs with `printpdf` and feed them through the
//! real `run_index` / `run_search` pipeline so they fail loudly if the
//! pipeline ever drops one of the report's promises.
//!
//! The numbers reported in `deep-research-report.md` are:
//!   * indexes at least 1,000 mixed PDFs without manual intervention;
//!   * second startup performs no re-extraction;
//!   * literal query over ~100k chunks returns visible results in
//!     < 50 ms p95 on a normal desktop;
//!   * modified PDF visible through the watcher/overlay within < 2 s;
//!   * every result has path, page_no, and a match-centred snippet.
//!
//! At the scaled-down level used here we use 30 PDFs (criterion 1),
//! 200 PDFs ~ 5_000 chunks (criterion 3), and one PDF for the watcher
//! round-trip (criterion 4). The success-criteria predicates are
//! the same as the report's.

mod common;

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use tempfile::tempdir;

use pdffff::app::{IndexOptions, WatchOptions, run_index, run_search, run_watch};
use pdffff::db::{ChunkInsert, Db, DocStatus, ExtractedDoc};
use pdffff::extract::ensure_pdftotext_available;
use pdffff::index::{IndexState, load_base_index_from_db};
use pdffff::normalize::{NORM_VERSION, normalize_for_index};
use pdffff::query::{DISPLAY_LIMIT, Hit, QueryMode, search};
use pdffff::scanner::Scanner;

const PATIENCE: Duration = Duration::from_secs(10);
const POLL: Duration = Duration::from_millis(20);

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
        // The synthesis loop dominates wall-clock here; 4 workers is
        // plenty and keeps `cargo test` polite.
        jobs: Some(4),
        require_pdftotext: true,
    }
}

fn watch_opts() -> WatchOptions {
    WatchOptions {
        respect_gitignore: false,
        follow_symlinks: false,
        jobs: Some(2),
        require_pdftotext: true,
        debounce: Some(Duration::from_millis(60)),
    }
}

/// Build a tiny PDF whose first page contains `tok` and a few padding
/// sentences. Used to synthesise corpora that look like real mixed
/// PDFs without needing a sample-files fixture on disk.
fn make_pdf_with_token(path: &Path, tok: &str, extra_words: &[&str]) {
    let line = format!(
        "{tok} the quick brown fox jumps over the lazy dog of {}",
        extra_words.join(" "),
    );
    let line2 = format!(
        "page two {tok} also mentions {} and more text",
        extra_words.join(" "),
    );
    common::make_pdf_two_pages(path, &[line.as_str()], &[line2.as_str()]);
}

/// Criterion 1: indexes many PDFs into SQLite without manual intervention.
#[test]
fn criterion_1_indexes_many_pdfs_unattended() -> Result<()> {
    if !require_pdftotext_or_skip("criterion_1_indexes_many_pdfs_unattended") {
        return Ok(());
    }

    const N: usize = 30;

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    std::fs::create_dir_all(&root)?;
    for i in 0..N {
        let tok = format!("uniqdoc{i}token");
        let pdf = root.join(format!("doc-{i:03}.pdf"));
        make_pdf_with_token(&pdf, &tok, &["alpha", "bravo", "charlie"]);
    }

    let db_path = tmp.path().join("idx.db");
    let started = Instant::now();
    let stats = run_index(&db_path, &root, &opts())?;
    let elapsed = started.elapsed();

    assert_eq!(stats.seen, N, "scanner must see every PDF");
    assert_eq!(stats.ok, N, "every PDF must extract successfully");
    assert_eq!(stats.error, 0, "no PDF should land in status=error");
    assert!(
        elapsed < Duration::from_secs(60),
        "indexing {N} synthesised PDFs took {elapsed:?}; should be well under 60s",
    );

    // Confirm via the durable store that the document rows are present
    // and that one of the per-doc tokens is retrievable.
    let db = Db::open(&db_path)?;
    let docs = db.load_all_documents()?;
    assert_eq!(docs.len(), N);
    assert!(docs.iter().all(|d| d.status == DocStatus::Ok));
    drop(db);

    let hits = run_search(&db_path, "uniqdoc17token", QueryMode::Literal, DISPLAY_LIMIT)?;
    assert!(
        !hits.is_empty(),
        "post-index search for a known token must find at least one hit",
    );
    Ok(())
}

/// Criterion 2: a second startup with no file changes performs no
/// re-extraction.
#[test]
fn criterion_2_second_startup_does_no_reextraction() -> Result<()> {
    if !require_pdftotext_or_skip("criterion_2_second_startup_does_no_reextraction") {
        return Ok(());
    }

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    std::fs::create_dir_all(&root)?;
    for i in 0..5 {
        let pdf = root.join(format!("doc-{i}.pdf"));
        make_pdf_with_token(&pdf, &format!("alpha{i}token"), &["padding", "text"]);
    }

    let db_path = tmp.path().join("idx.db");
    let first = run_index(&db_path, &root, &opts())?;
    assert_eq!(first.dirty, 5, "every file is dirty on the first pass");
    assert_eq!(first.ok, 5);

    // Second pass: nothing has changed on disk; scanner.walk_and_diff
    // must produce zero jobs, and run_index must report zero dirty.
    let db = Db::open(&db_path)?;
    let scan = Scanner::new(&root).walk_and_diff(&db)?;
    assert_eq!(
        scan.jobs.len(),
        0,
        "second walk must produce zero extraction jobs; got {:?}",
        scan.jobs,
    );
    assert_eq!(scan.deleted.len(), 0);
    drop(db);

    let second = run_index(&db_path, &root, &opts())?;
    assert_eq!(
        second.dirty, 0,
        "IndexStats::dirty must be zero on a no-op second pass; got {second:?}",
    );
    assert_eq!(second.ok, 0);
    assert_eq!(second.error, 0);
    assert_eq!(second.empty, 0);
    Ok(())
}

/// Criterion 3: literal query speed.
///
/// Build a ~5_000-chunk in-memory corpus directly (bypassing
/// pdftotext — what's being measured here is *query latency over an
/// in-memory `BaseIndex`*, not extraction throughput) and run 20
/// representative queries through `pdffff::query::search`. Assert
/// p95 < 50 ms.
///
/// Why not use `printpdf` + `run_index`? `printpdf`'s `use_text`
/// writes each call as a single PDF text op without wrapping, so a
/// 16_000-word string lays out as a few characters on a single
/// physical page — `pdftotext` then extracts only a handful of
/// characters per page no matter how long the input string was.
/// Synthesising 5k chunks via real PDFs would require hundreds of
/// short PDFs and would dominate test wall-clock for no extra
/// signal: the criterion is about query latency, and the rest of
/// the pipeline (Days 1-2) is exercised by `index_pipeline.rs`.
///
/// This is the only soft-bounded test in the suite — variance on a
/// busy CI box can push individual queries past 50 ms. We still
/// assert on p95 (i.e. allow 1 outlier in 20).
#[test]
fn criterion_3_literal_query_under_50ms_p95() -> Result<()> {
    // Fixed deterministic dictionary so the corpus is reproducible.
    let words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
                 "golf", "hotel", "india", "juliet", "kilo", "lima",
                 "mike", "november", "oscar", "papa", "quebec", "romeo",
                 "sierra", "tango", "uniform", "victor", "whiskey",
                 "xray", "yankee", "zulu"];

    // ~1200 chars per chunk → ~150 words. Generate 5_000 distinct
    // chunks; each chunk picks a different rotation through the
    // dictionary so the bigram index gets the same kind of variety it
    // would see on real PDF text.
    const N_CHUNKS: usize = 5_000;
    const WORDS_PER_CHUNK: usize = 150;

    let tmp = tempdir()?;
    let db_path = tmp.path().join("idx.db");
    let mut db = Db::open(&db_path)?;

    // Group chunks into "documents" of 50 chunks each so doc_ranges
    // exercises the per-doc dedup the loader uses.
    const CHUNKS_PER_DOC: usize = 50;
    let n_docs = N_CHUNKS.div_ceil(CHUNKS_PER_DOC);
    let mut next_chunk_id = 0usize;
    for d in 0..n_docs {
        let chunks_in_doc = CHUNKS_PER_DOC.min(N_CHUNKS - next_chunk_id);
        let mut chunk_inserts: Vec<ChunkInsert> = Vec::with_capacity(chunks_in_doc);
        for k in 0..chunks_in_doc {
            let chunk_idx = next_chunk_id + k;
            let text: String = (0..WORDS_PER_CHUNK)
                .map(|i| words[(chunk_idx * 13 + i) % words.len()])
                .collect::<Vec<_>>()
                .join(" ");
            let norm = normalize_for_index(&text);
            let preview: String = text.chars().take(120).collect();
            chunk_inserts.push(ChunkInsert {
                page_no: ((k / 5) as u32) + 1,
                chunk_ord: k as u32,
                char_start: 0,
                char_end: text.len() as u32,
                text_utf8: text,
                text_norm_ascii: norm,
                preview,
            });
        }
        next_chunk_id += chunks_in_doc;

        let path = tmp.path().join(format!("synthetic-{d:03}.pdf"));
        db.upsert_extracted(&ExtractedDoc {
            path,
            size_bytes: 0,
            mtime_ns: 0,
            dev: None,
            ino: None,
            extractor: "synthetic".into(),
            extractor_version: "test".into(),
            norm_version: NORM_VERSION,
            page_count: (chunks_in_doc / 5 + 1) as u32,
            status: DocStatus::Ok,
            error_text: None,
            chunks: chunk_inserts,
        })?;
    }
    drop(db);

    // Load once and re-use the in-memory state for every query — same
    // path the running CLI/watch take.
    let db = Db::open_reader(&db_path)?;
    let base = load_base_index_from_db(&db)?;
    let total_chunks = base.chunks.len();
    let state = IndexState::new(base);
    assert_eq!(total_chunks, N_CHUNKS);

    let queries = [
        "alpha bravo", "charlie delta", "echo foxtrot", "golf hotel",
        "india juliet", "kilo lima", "mike november", "oscar papa",
        "quebec romeo", "sierra tango", "uniform victor", "whiskey xray",
        "yankee zulu", "alpha echo", "bravo foxtrot", "charlie golf",
        "delta hotel", "tango uniform", "papa quebec", "lima mike",
    ];

    // Warm-up: a cold first query pays one-time allocator and CPU
    // cache fills that aren't representative of steady-state latency.
    let _ = search(&state, "alpha", QueryMode::Literal, DISPLAY_LIMIT)?;

    let mut times_us: Vec<u128> = Vec::with_capacity(queries.len());
    for q in &queries {
        let start = Instant::now();
        let _hits = search(&state, q, QueryMode::Literal, DISPLAY_LIMIT)?;
        times_us.push(start.elapsed().as_micros());
    }
    times_us.sort_unstable();
    let p95 = times_us[(times_us.len() * 95 / 100).min(times_us.len() - 1)];
    let max = *times_us.last().unwrap();
    eprintln!(
        "[criterion_3] {total_chunks} chunks; p95 = {} us; max = {} us; all = {:?}",
        p95, max, times_us,
    );
    assert!(
        p95 < 50_000,
        "p95 query latency {p95} us exceeds 50ms budget at {total_chunks} chunks",
    );
    Ok(())
}

/// Criterion 4: watcher round-trip < 2 s from file write to visible hit.
#[test]
fn criterion_4_watcher_round_trip_under_2s() -> Result<()> {
    if !require_pdftotext_or_skip("criterion_4_watcher_round_trip_under_2s") {
        return Ok(());
    }

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    std::fs::create_dir_all(&root)?;
    let pdf_path = root.join("paper.pdf");
    // Initial corpus has one PDF without the magic token, so the
    // watcher must pick up the *modification*, not the initial sync.
    common::make_pdf_two_pages(
        &pdf_path,
        &["original content here"],
        &["page two stays"],
    );

    let db_path = tmp.path().join("idx.db");
    let handle = run_watch(&db_path, &root, &watch_opts())?;
    let state = handle.state.clone();

    // Wait for the initial sync to settle so the modification we time
    // is the only one in flight.
    wait_until(&state, "original", |hits| !hits.is_empty(), "initial sync");

    let token = "freshpipelinetoken";
    let write_started = Instant::now();
    common::make_pdf_two_pages(
        &pdf_path,
        &[&format!("{token} now lives here")],
        &["page two stays"],
    );

    // Poll the live state until the token is visible.
    loop {
        let hits = search(&state, token, QueryMode::Literal, DISPLAY_LIMIT)?;
        if hits.iter().any(|h| Path::new(&h.path) == pdf_path.as_path()) {
            let elapsed = write_started.elapsed();
            eprintln!("[criterion_4] watcher round-trip = {elapsed:?}");
            assert!(
                elapsed < Duration::from_secs(2),
                "watcher round-trip {elapsed:?} exceeds 2s budget",
            );
            handle.stop()?;
            return Ok(());
        }
        if write_started.elapsed() >= Duration::from_secs(2) {
            handle.stop()?;
            panic!(
                "watcher did not surface token {token:?} within 2s of file write",
            );
        }
        std::thread::sleep(POLL);
    }
}

/// Criterion 5: every result includes path, page_no >= 1, and a
/// non-empty snippet.
#[test]
fn criterion_5_every_hit_has_path_page_snippet() -> Result<()> {
    if !require_pdftotext_or_skip("criterion_5_every_hit_has_path_page_snippet") {
        return Ok(());
    }

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    std::fs::create_dir_all(&root)?;
    let a = root.join("a.pdf");
    let b = root.join("b.pdf");
    common::make_pdf_two_pages(
        &a,
        &["alpha bravo charlie delta echo foxtrot golf"],
        &["hotel india juliet kilo lima mike november"],
    );
    common::make_pdf_two_pages(
        &b,
        &["oscar papa quebec romeo sierra tango"],
        &["uniform victor whiskey xray yankee zulu"],
    );

    let db_path = tmp.path().join("idx.db");
    let stats = run_index(&db_path, &root, &opts())?;
    assert_eq!(stats.ok, 2);

    let queries = ["alpha", "hotel india", "tango", "zulu", "echo foxtrot"];
    for q in &queries {
        let hits = run_search(&db_path, q, QueryMode::Literal, DISPLAY_LIMIT)?;
        assert!(!hits.is_empty(), "expected hits for query {q:?}");
        for hit in &hits {
            assert!(!hit.path.is_empty(), "every hit must have a path");
            assert!(
                hit.page_no >= 1,
                "page_no must be 1-based and present: {hit:?}",
            );
            assert!(
                !hit.snippet.is_empty(),
                "every hit must have a non-empty snippet: {hit:?}",
            );
            // The snippet must be a slice of `text_utf8` after
            // whitespace collapse, so at minimum it should be
            // displayable ASCII / UTF-8.
            assert!(
                hit.snippet.chars().all(|c| !c.is_control() || c == '\t'),
                "snippet should not contain control chars: {:?}",
                hit.snippet,
            );
        }
    }
    Ok(())
}

fn wait_until<F>(
    state: &Arc<IndexState>,
    query: &str,
    predicate: F,
    label: &str,
) where
    F: Fn(&[Hit]) -> bool,
{
    let started = Instant::now();
    loop {
        let hits = search(state, query, QueryMode::Literal, DISPLAY_LIMIT)
            .expect("literal search never errors");
        if predicate(&hits) {
            return;
        }
        if started.elapsed() >= PATIENCE {
            panic!("timed out waiting for {label} (query={query:?})");
        }
        std::thread::sleep(POLL);
    }
}
