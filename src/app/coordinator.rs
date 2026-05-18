//! The coordinator thread: turns scanner / watcher events into
//! extractor-pool jobs and writer-thread deletes.

use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tracing::{info, warn};

use crate::db::Db;
use crate::extract::extract_pdf;
use crate::scanner::{DirtyReason, ScanJob, Scanner, scan_one};
use crate::watcher::WatchEvent;

use super::handle::IndexProgress;
use super::writer::WriterMsg;

/// Subset of [`super::WatchOptions`] the coordinator needs for the
/// initial scan. Kept as a separate struct so the coordinator's
/// signature does not grow every time we add a watcher-only option.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScanParams {
    pub respect_gitignore: bool,
    pub follow_symlinks: bool,
}

/// The coordinator thread:
///
/// 1. Does one synchronous [`Scanner::walk_and_diff`] pass against
///    `root`, dispatches every dirty `ScanJob` to the rayon extractor
///    pool, and sends one [`WriterMsg::Delete`] per disappeared path.
///    Both flows update the `progress` counters so the TUI's spinner
///    keeps ticking through the initial sync.
/// 2. Enters the long-running event loop: receives `WatchEvent`s from
///    the watcher, converts them to `ScanJob`s (Dirty) or
///    `WriterMsg::Delete` messages (Removed), and dispatches them via
///    the same pool / writer channel.
///
/// The selector loop watches both the watcher channel and a small
/// stop channel; a stop signal or a closed watcher channel causes a
/// clean exit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn coordinator_thread(
    root: PathBuf,
    db_path: PathBuf,
    scan: ScanParams,
    pool: Arc<rayon::ThreadPool>,
    watch_rx: flume::Receiver<WatchEvent>,
    writer_tx: flume::Sender<WriterMsg>,
    stop_rx: flume::Receiver<()>,
    stop: Arc<AtomicBool>,
    progress: Arc<IndexProgress>,
) -> Result<()> {
    // ---- Initial scan ---------------------------------------------------
    //
    // Open a *reader* DB just for the diff. The writer thread owns the
    // only writer connection from here on. SQLite's WAL allows any
    // number of readers concurrent with the single writer, so this is
    // safe.
    if !stop.load(Ordering::Relaxed) {
        run_initial_scan(&db_path, &root, scan, &pool, &writer_tx, &progress, &stop);
    }

    // ---- Live event loop ------------------------------------------------
    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let sel = flume::Selector::new()
            .recv(&watch_rx, |r| r.ok())
            .recv(&stop_rx, |_| None);
        let Some(ev) = sel.wait() else {
            // Either stop was signalled or the watcher channel hung
            // up. In both cases we exit the loop cleanly.
            break;
        };
        if !handle_watch_event(ev, &pool, &writer_tx, &progress) {
            break;
        }
    }
    // Drop the writer sender so the writer thread sees disconnection
    // and exits.
    drop(writer_tx);
    Ok(())
}

/// One synchronous walk + diff. Logs and swallows errors so the
/// coordinator can still enter its live loop; the watcher will pick
/// up subsequent changes regardless.
fn run_initial_scan(
    db_path: &std::path::Path,
    root: &std::path::Path,
    scan: ScanParams,
    pool: &rayon::ThreadPool,
    writer_tx: &flume::Sender<WriterMsg>,
    progress: &Arc<IndexProgress>,
    stop: &Arc<AtomicBool>,
) {
    let db = match Db::open_reader(db_path) {
        Ok(db) => db,
        Err(err) => {
            warn!(?err, "opening reader DB for initial scan failed");
            return;
        }
    };
    let mut scanner = Scanner::new(root);
    scanner.respect_gitignore = scan.respect_gitignore;
    scanner.follow_symlinks = scan.follow_symlinks;
    let result = match scanner.walk_and_diff(&db) {
        Ok(r) => r,
        Err(err) => {
            warn!(?err, "initial scanner walk failed");
            return;
        }
    };
    info!(
        seen = result.seen_count,
        dirty = result.jobs.len(),
        deleted = result.deleted.len(),
        "initial scan complete",
    );
    for job in result.jobs {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        dispatch_extract(pool, writer_tx, progress, job);
    }
    for path in result.deleted {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        if writer_tx.send(WriterMsg::Delete(path)).is_err() {
            warn!("writer thread closed before initial deletes were enqueued");
            return;
        }
    }
}

/// Apply one watcher event. Returns `false` when the loop should
/// exit (writer channel hung up).
fn handle_watch_event(
    ev: WatchEvent,
    pool: &rayon::ThreadPool,
    writer_tx: &flume::Sender<WriterMsg>,
    progress: &Arc<IndexProgress>,
) -> bool {
    match ev {
        WatchEvent::Dirty(path) => {
            // Stat and submit an extraction job. `RetryAfterError`
            // is a fine reason here: a Dirty event after a prior
            // extraction error should re-extract on the next
            // mutation anyway.
            match scan_one(&path, DirtyReason::Modified) {
                Ok(Some(job)) => dispatch_extract(pool, writer_tx, progress, job),
                Ok(None) => {}
                Err(err) => warn!(path = %path.display(), ?err, "stat failed"),
            }
            true
        }
        WatchEvent::Removed(path) => {
            if writer_tx.send(WriterMsg::Delete(path.clone())).is_err() {
                warn!(path = %path.display(), "writer thread closed before delete enqueued");
                return false;
            }
            true
        }
    }
}

/// Push one `ScanJob` onto the extractor pool. Used both by the
/// initial scan and by the live `Dirty` branch — keeping a single
/// dispatch path means the `progress.pending` counter stays consistent
/// regardless of how the job entered the pipeline.
fn dispatch_extract(
    pool: &rayon::ThreadPool,
    writer_tx: &flume::Sender<WriterMsg>,
    progress: &Arc<IndexProgress>,
    job: ScanJob,
) {
    progress.pending.fetch_add(1, Ordering::Relaxed);
    let tx = writer_tx.clone();
    let job_progress = progress.clone();
    pool.spawn(move || {
        let _guard = PendingGuard::new(&job_progress.pending);
        let extracted = match extract_pdf(&job) {
            Ok(d) => d,
            Err(err) => {
                warn!(
                    path = %job.path.display(), ?err,
                    "extractor returned hard error",
                );
                return;
            }
        };
        if tx.send(WriterMsg::Doc(Box::new(extracted))).is_err() {
            warn!(path = %job.path.display(), "writer thread closed; discarding result");
        }
    });
}

/// RAII guard that decrements a `pending` atomic when dropped.
///
/// Used to ensure the coordinator's `progress.pending` counter is
/// always decremented when an extractor closure finishes, regardless of
/// which branch the closure takes (success, extract error, send
/// failure, or panic).
struct PendingGuard<'a> {
    counter: &'a AtomicUsize,
}

impl<'a> PendingGuard<'a> {
    fn new(counter: &'a AtomicUsize) -> Self {
        Self { counter }
    }
}

impl Drop for PendingGuard<'_> {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::Relaxed);
    }
}
