//! Top-level orchestrator: workers, channels, lifecycle.
//!
//! The pure-TUI build exposes exactly one entry point — [`run_watch`] —
//! which starts the long-lived pipeline:
//!
//! ```text
//!   coordinator thread ──► rayon extractor pool ──► flume bounded channel ──►
//!     single DB-writer thread (owns the only writer Connection,
//!                              publishes overlay mutations)
//! ```
//!
//! On startup the coordinator does **one** synchronous
//! [`crate::scanner::Scanner::walk_and_diff`] pass to discover what's
//! on disk, then dispatches the dirty PDFs to the same rayon pool the
//! `notify-debouncer-full` watcher feeds. Both initial-sync extractions
//! and live filesystem events therefore travel the same path: there is
//! no separate "initial sync" phase that blocks the caller. The TUI
//! receives the [`WatchHandle`] immediately and starts serving queries
//! against whatever's already in SQLite.
//!
//! The orchestrator is split across:
//!
//! * [`options`]     — [`WatchOptions`] knobs.
//! * [`handle`]      — [`WatchHandle`] + [`IndexProgress`] counters.
//! * [`db_path`]     — [`resolve_db_path`] convention.
//! * [`coordinator`] — the scanner / watcher event handler.
//! * [`writer`]      — the single DB writer thread + overlay
//!   publishing.

mod coordinator;
mod db_path;
mod handle;
mod options;
mod writer;

use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;

use crate::db::Db;
use crate::extract::ensure_pdftotext_available;
use crate::index::{IndexState, load_base_index_from_db};
use crate::watcher::spawn_watcher;

use coordinator::{ScanParams, coordinator_thread};
use writer::{WriterMsg, writer_thread};

pub use db_path::resolve_db_path;
pub use handle::{IndexProgress, WatchHandle};
pub use options::WatchOptions;

/// Wall-clock milliseconds since the Unix epoch. Used as `indexed_at_ms`
/// / `deleted_at_ms` timestamps in `documents`.
pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Long-lived watch pipeline.
///
/// 1. Verify `pdftotext` (unless the caller disables that for tests).
/// 2. Open the DB and load whatever is currently in it into
///    [`IndexState`]. This may be empty on first launch, or a previous
///    session's snapshot — either way, queries work immediately.
/// 3. Spawn the long-lived DB-writer + rayon extractor pool + watcher.
/// 4. Spawn the coordinator. Its first action is a synchronous
///    [`crate::scanner::Scanner::walk_and_diff`] pass that discovers
///    everything on disk and dispatches the dirty PDFs to the same
///    pool the watcher uses; then it enters the live event loop.
/// 5. Return a [`WatchHandle`] the caller can use to run queries and
///    stop the pipeline cleanly.
///
/// The function itself does no extraction work — every PDF, whether
/// found by the initial walk or by a later filesystem event, is
/// extracted on a background thread.
pub fn run_watch(db_path: &Path, root: &Path, opts: &WatchOptions) -> Result<WatchHandle> {
    if opts.require_pdftotext {
        ensure_pdftotext_available()?;
    }

    // Open one connection up front: apply the schema (so the writer /
    // coordinator threads find a populated DB when they open their own
    // connections) and stream the existing chunks into the in-memory
    // BaseIndex from the same connection. On a brand-new corpus the
    // load yields an empty BaseIndex — queries against it return zero
    // hits, which is fine, because the coordinator's initial scan
    // will start streaming results into the overlay almost immediately.
    let state = {
        let db = Db::open(db_path)
            .with_context(|| format!("opening DB at {} for initial load", db_path.display()))?;
        let base = load_base_index_from_db(&db)?;
        Arc::new(IndexState::new(base))
    };

    // ---- Long-lived DB writer ------------------------------------------
    let (writer_tx, writer_rx) = flume::bounded::<WriterMsg>(64);
    let counters = Arc::new(IndexProgress::default());
    let writer_db_path = db_path.to_path_buf();
    let writer_state = state.clone();
    let writer_counters = counters.clone();
    let writer_handle = thread::Builder::new()
        .name("pdffff-db-writer".into())
        .spawn(move || writer_thread(writer_db_path, writer_rx, writer_counters, writer_state))
        .context("spawning DB writer thread")?;

    // ---- Long-lived extractor pool -------------------------------------
    let pool_size = opts.jobs.unwrap_or_else(default_pool_size);
    info!(pool_size, "watch extractor pool spawned");
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(pool_size)
            .thread_name(|i| format!("pdffff-extractor-{i}"))
            .build()
            .context("building rayon extractor pool")?,
    );

    // ---- Filesystem watcher --------------------------------------------
    //
    // Spawn the watcher *before* the initial scan so that any
    // filesystem event firing during the scan is buffered on the
    // channel rather than lost. The coordinator may end up
    // double-extracting a path that was both seen by the scan and
    // touched by the user mid-scan, but the writer just re-UPSERTs —
    // double work, not incorrect work, and a vanishingly rare race.
    let (watch_tx, watch_rx) = flume::unbounded();
    let watcher = spawn_watcher(root, watch_tx, opts.debounce)?;

    // ---- Coordinator ---------------------------------------------------
    let stop = Arc::new(AtomicBool::new(false));
    let (stop_tx, stop_rx) = flume::bounded::<()>(1);

    let coord_pool = pool.clone();
    let coord_writer_tx = writer_tx.clone();
    let coord_stop = stop.clone();
    let coord_progress = counters.clone();
    let coord_root = root.to_path_buf();
    let coord_db_path = db_path.to_path_buf();
    let coord_scan = ScanParams {
        respect_gitignore: opts.respect_gitignore,
        follow_symlinks: opts.follow_symlinks,
    };
    let coordinator = thread::Builder::new()
        .name("pdffff-watch-coordinator".into())
        .spawn(move || {
            coordinator_thread(
                coord_root,
                coord_db_path,
                coord_scan,
                coord_pool,
                watch_rx,
                coord_writer_tx,
                stop_rx,
                coord_stop,
                coord_progress,
            )
        })
        .context("spawning watch coordinator")?;

    // Drop our own writer_tx clone so the writer exits when the
    // coordinator drops its clone on shutdown.
    drop(writer_tx);

    Ok(WatchHandle {
        stop,
        stop_tx,
        coordinator: Some(coordinator),
        writer: Some(writer_handle),
        _watcher: Some(watcher),
        state,
        progress: counters,
    })
}

fn default_pool_size() -> usize {
    let n = thread::available_parallelism().map(|n| n.get()).unwrap_or(2);
    n.min(6).max(1)
}
