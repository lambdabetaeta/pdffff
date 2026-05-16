//! Top-level orchestrator: workers, channels, lifecycle.
//!
//! Day 2 wired the one-shot *indexing* pipeline. Day 5 adds the
//! long-running *watch* pipeline:
//!
//! ```text
//!   notify-debouncer-full ──► coordinator thread ──►
//!     rayon extractor pool ──► flume bounded channel ──►
//!       single DB-writer thread (owns the only writer Connection,
//!                                publishes overlay mutations)
//! ```
//!
//! Both pipelines share the same building blocks (`Scanner`,
//! `extract_pdf`, `Db::upsert_extracted`, `Db::mark_deleted`). The
//! writer thread is the only mutator of both `documents` /
//! `chunks` rows and the live [`IndexState::overlay`], so every
//! mutation has exactly one serializer and the overlay never sees
//! interleaved updates from two threads.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

use crate::db::{Db, DocStatus, ExtractedDoc, LoadedChunkRow};
use crate::extract::{ensure_pdftotext_available, extract_pdf, probe_pdftotext_or_explain};
use crate::index::{ChunkItem, IndexState, load_base_index_from_db};
use crate::query::{Hit, QueryMode, search};
use crate::scanner::{DirtyReason, ScanJob, Scanner, scan_one};
use crate::watcher::{WatchEvent, WatcherHandle, spawn_watcher};

/// Wall-clock milliseconds since the Unix epoch. Used as `indexed_at_ms`
/// / `deleted_at_ms` timestamps in `documents`.
pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Outcome of a full `index` run — useful to tests and to the CLI's
/// terminal output.
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub seen: usize,
    pub dirty: usize,
    pub ok: usize,
    pub empty: usize,
    pub error: usize,
    pub deleted: usize,
    pub elapsed_secs: f64,
}

/// Knobs for [`run_index`].
#[derive(Debug, Clone)]
pub struct IndexOptions {
    pub respect_gitignore: bool,
    pub follow_symlinks: bool,
    /// Override extractor pool size. Default: `min(num_cpus, 6)`.
    pub jobs: Option<usize>,
    /// If true, fail fast at startup when `pdftotext` is missing. Tests
    /// can disable this only when they have pre-checked.
    pub require_pdftotext: bool,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            respect_gitignore: false,
            follow_symlinks: false,
            jobs: None,
            require_pdftotext: true,
        }
    }
}

/// Top-level pipeline driver for the one-shot `index` mode.
///
/// 1. Verify `pdftotext` if `opts.require_pdftotext`.
/// 2. Open one `Db` to run the scanner diff, then drop it.
/// 3. Spawn the single DB-writer thread (it opens the *only* writer
///    connection for the duration of indexing).
/// 4. Spawn a bounded rayon pool of extractors. Each extractor sends its
///    [`ExtractedDoc`] on a `flume::bounded` channel.
/// 5. Send `Delete(path)` messages for every disappeared path.
/// 6. Drop the sender; join the writer.
pub fn run_index(db_path: &Path, root: &Path, opts: &IndexOptions) -> Result<IndexStats> {
    if opts.require_pdftotext {
        ensure_pdftotext_available()?;
        probe_pdftotext_or_explain()?;
    }

    let started = Instant::now();

    // Scan first with a short-lived reader Db. Then drop it: the writer
    // thread opens the only writer connection from here on, matching
    // SQLite WAL's single-writer model.
    let scan_result = {
        let db = Db::open(db_path)?;
        let mut scanner = Scanner::new(root);
        scanner.respect_gitignore = opts.respect_gitignore;
        scanner.follow_symlinks = opts.follow_symlinks;
        scanner.walk_and_diff(&db)?
    };

    info!(
        seen = scan_result.seen_count,
        dirty = scan_result.jobs.len(),
        deleted = scan_result.deleted.len(),
        "scan complete",
    );

    // ---- DB writer thread ----------------------------------------------
    let (tx, rx) = flume::bounded::<WriterMsg>(64);
    let writer_db_path = db_path.to_path_buf();
    let counters = Arc::new(WriterCounters::default());
    let writer_counters = counters.clone();
    let writer_handle = thread::Builder::new()
        .name("pdffff-db-writer".into())
        .spawn(move || writer_thread(writer_db_path, rx, writer_counters, None))
        .context("spawning DB writer thread")?;

    // ---- Extractor pool ------------------------------------------------
    let pool_size = opts.jobs.unwrap_or_else(default_pool_size);
    info!(pool_size, "extractor pool spawned");

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(pool_size)
        .thread_name(|i| format!("pdffff-extractor-{i}"))
        .build()
        .context("building rayon extractor pool")?;

    let job_count = scan_result.jobs.len();
    let progress = AtomicUsize::new(0);
    pool.scope(|s| {
        for job in &scan_result.jobs {
            let tx = tx.clone();
            let progress = &progress;
            s.spawn(move |_| {
                extract_and_send(job, &tx, progress, job_count);
            });
        }
    });

    // Tombstone deletions through the *same* writer thread so all DB
    // mutations are serialized.
    for path in &scan_result.deleted {
        if tx.send(WriterMsg::Delete(path.clone())).is_err() {
            warn!(path = %path.display(), "writer thread closed before delete enqueued");
            break;
        }
    }

    // Drop sender, then join — writer exits when the channel disconnects.
    drop(tx);
    let writer_result = writer_handle
        .join()
        .map_err(|_| anyhow::anyhow!("DB writer thread panicked"))?;
    writer_result?;

    Ok(IndexStats {
        seen: scan_result.seen_count,
        dirty: scan_result.jobs.len(),
        ok: counters.ok.load(Ordering::Relaxed),
        empty: counters.empty.load(Ordering::Relaxed),
        error: counters.error.load(Ordering::Relaxed),
        deleted: counters.deleted.load(Ordering::Relaxed),
        elapsed_secs: started.elapsed().as_secs_f64(),
    })
}

/// One-shot search: open the DB, load the base index, run a literal
/// query, return the hits. Day 3 keeps this synchronous — there is no
/// watcher, rebuild loop, or persistent process yet.
pub fn run_search(
    db_path: &Path,
    query: &str,
    mode: QueryMode,
    limit: usize,
) -> Result<Vec<Hit>> {
    let db = Db::open_reader(db_path)
        .with_context(|| format!("opening DB at {} for search", db_path.display()))?;
    let base = load_base_index_from_db(&db)?;
    let state = IndexState::new(base);
    search(&state, query, mode, limit)
}

/// Knobs for [`run_watch`]. Mirrors [`IndexOptions`] + the watcher's
/// debounce window.
#[derive(Debug, Clone)]
pub struct WatchOptions {
    pub respect_gitignore: bool,
    pub follow_symlinks: bool,
    pub jobs: Option<usize>,
    pub require_pdftotext: bool,
    /// Debounce window for the filesystem watcher. `None` ⇒ the
    /// watcher module's default ([`crate::watcher::DEFAULT_DEBOUNCE`]).
    pub debounce: Option<Duration>,
}

impl Default for WatchOptions {
    fn default() -> Self {
        Self {
            respect_gitignore: false,
            follow_symlinks: false,
            jobs: None,
            require_pdftotext: true,
            debounce: None,
        }
    }
}

/// Live handle exposed by [`run_watch`] when the caller asks for
/// programmatic control (tests). Owns every thread the watch loop
/// spawned and signals them to shut down on [`WatchHandle::stop`].
pub struct WatchHandle {
    /// Shared with the coordinator: when set, the writer / coordinator
    /// drain their queues and exit at the next opportunity.
    stop: Arc<AtomicBool>,
    /// Sender into the coordinator's stop channel. Sending one `()`
    /// wakes the selector immediately.
    stop_tx: flume::Sender<()>,
    /// Joined by [`stop`] to surface coordinator / writer panics.
    coordinator: Option<thread::JoinHandle<Result<()>>>,
    writer: Option<thread::JoinHandle<Result<()>>>,
    /// Kept alive for the lifetime of the watcher; dropped in `stop`
    /// to halt notify's internal thread.
    _watcher: Option<WatcherHandle>,
    /// Live in-memory index. Exposed so tests (and a future TUI) can
    /// run queries while the watcher is active.
    pub state: Arc<IndexState>,
}

impl WatchHandle {
    pub fn stop(mut self) -> Result<()> {
        self.stop.store(true, Ordering::Relaxed);
        // Best-effort wake: if the channel is already full or torn
        // down, the coordinator will still notice via the AtomicBool.
        let _ = self.stop_tx.send(());
        // Dropping the watcher handle stops the notify thread and
        // disconnects the WatchEvent sender, which lets the
        // coordinator's loop see a clean shutdown.
        self._watcher.take();
        if let Some(h) = self.coordinator.take() {
            let res = h.join().map_err(|_| anyhow::anyhow!("watch coordinator panicked"))?;
            res?;
        }
        if let Some(h) = self.writer.take() {
            let res = h.join().map_err(|_| anyhow::anyhow!("watch writer panicked"))?;
            res?;
        }
        Ok(())
    }
}

/// Long-lived watch pipeline.
///
/// Steps, in the order they happen:
///
/// 1. Verify `pdftotext`.
/// 2. Open the DB, run one synchronous `Scanner::walk_and_diff` pass
///    to bring it up to date with the filesystem (the same machinery
///    `run_index` uses, minus the early-exit at the end).
/// 3. Load `IndexState` from the now-up-to-date DB.
/// 4. Start the long-lived DB-writer + extractor-pool + watcher chain.
/// 5. Return a [`WatchHandle`] the caller can use to issue queries
///    and stop the pipeline cleanly.
///
/// The watch loop runs on background threads; the caller's thread is
/// free to run queries against `state` immediately.
pub fn run_watch(db_path: &Path, root: &Path, opts: &WatchOptions) -> Result<WatchHandle> {
    if opts.require_pdftotext {
        ensure_pdftotext_available()?;
        probe_pdftotext_or_explain()?;
    }

    // ---- Initial sync ---------------------------------------------------
    //
    // We run the same scan+extract pipeline `run_index` uses, but
    // synchronously and only once — its only job is to converge the
    // DB with whatever's currently on disk before the watcher starts
    // emitting deltas.
    let index_opts = IndexOptions {
        respect_gitignore: opts.respect_gitignore,
        follow_symlinks: opts.follow_symlinks,
        jobs: opts.jobs,
        // We've already verified pdftotext above; skip the re-check.
        require_pdftotext: false,
    };
    let _ = run_index(db_path, root, &index_opts)?;

    // Load the up-to-date base index now.
    let state = {
        let db = Db::open_reader(db_path)?;
        let base = load_base_index_from_db(&db)?;
        Arc::new(IndexState::new(base))
    };

    // ---- Long-lived DB writer ------------------------------------------
    let (writer_tx, writer_rx) = flume::bounded::<WriterMsg>(64);
    let counters = Arc::new(WriterCounters::default());
    let writer_db_path = db_path.to_path_buf();
    let writer_state = state.clone();
    let writer_counters = counters.clone();
    let writer_handle = thread::Builder::new()
        .name("pdffff-db-writer".into())
        .spawn(move || writer_thread(writer_db_path, writer_rx, writer_counters, Some(writer_state)))
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
    let (watch_tx, watch_rx) = flume::unbounded::<WatchEvent>();
    let watcher = spawn_watcher(root, watch_tx, opts.debounce)?;

    // ---- Coordinator ---------------------------------------------------
    let stop = Arc::new(AtomicBool::new(false));
    let (stop_tx, stop_rx) = flume::bounded::<()>(1);

    let coord_pool = pool.clone();
    let coord_writer_tx = writer_tx.clone();
    let coord_stop = stop.clone();
    let coordinator = thread::Builder::new()
        .name("pdffff-watch-coordinator".into())
        .spawn(move || {
            coordinator_thread(
                coord_pool,
                watch_rx,
                coord_writer_tx,
                stop_rx,
                coord_stop,
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
    })
}

/// The coordinator thread:
///
/// * receives `WatchEvent`s from the watcher,
/// * converts them to `ScanJob`s (Dirty) or `WriterMsg::Delete`
///   messages (Removed),
/// * fans Dirty jobs out to the rayon extractor pool, which sends
///   `WriterMsg::Doc` on the writer channel.
///
/// The selector loop watches both the watcher channel and a small
/// stop channel; either a stop signal or a closed watcher channel
/// causes a clean exit.
fn coordinator_thread(
    pool: Arc<rayon::ThreadPool>,
    watch_rx: flume::Receiver<WatchEvent>,
    writer_tx: flume::Sender<WriterMsg>,
    stop_rx: flume::Receiver<()>,
    stop: Arc<AtomicBool>,
) -> Result<()> {
    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let sel = flume::Selector::new()
            .recv(&watch_rx, |r| match r {
                Ok(ev) => Some(ev),
                Err(_) => None,
            })
            .recv(&stop_rx, |_| None);
        let ev = sel.wait();
        let Some(ev) = ev else {
            // Either stop was signalled or the watcher channel hung
            // up. In both cases we exit the loop cleanly.
            break;
        };
        match ev {
            WatchEvent::Dirty(path) => {
                // Stat and submit an extraction job. `RetryAfterError`
                // is a fine reason here: a Dirty event after a prior
                // extraction error should re-extract on the next
                // mutation anyway.
                let job = match scan_one(&path, DirtyReason::Modified) {
                    Ok(Some(j)) => j,
                    Ok(None) => continue,
                    Err(err) => {
                        warn!(path = %path.display(), ?err, "stat failed");
                        continue;
                    }
                };
                let tx = writer_tx.clone();
                pool.spawn(move || {
                    let extracted = match extract_pdf(&job) {
                        Ok(d) => d,
                        Err(err) => {
                            warn!(path = %job.path.display(), ?err, "extractor returned hard error");
                            return;
                        }
                    };
                    if tx.send(WriterMsg::Doc(Box::new(extracted))).is_err() {
                        warn!(path = %job.path.display(), "writer thread closed; discarding result");
                    }
                });
            }
            WatchEvent::Removed(path) => {
                if writer_tx.send(WriterMsg::Delete(path.clone())).is_err() {
                    warn!(path = %path.display(), "writer thread closed before delete enqueued");
                    break;
                }
            }
        }
    }
    // Drop the writer sender so the writer thread sees disconnection
    // and exits.
    drop(writer_tx);
    Ok(())
}

fn default_pool_size() -> usize {
    let n = thread::available_parallelism().map(|n| n.get()).unwrap_or(2);
    n.min(6).max(1)
}

fn extract_and_send(
    job: &ScanJob,
    tx: &flume::Sender<WriterMsg>,
    progress: &AtomicUsize,
    total: usize,
) {
    let n = progress.fetch_add(1, Ordering::Relaxed) + 1;
    let extracted = match extract_pdf(job) {
        Ok(doc) => doc,
        Err(err) => {
            // `extract_pdf` converts almost every per-PDF failure into an
            // Error-status row; reaching this branch means we couldn't
            // build *any* `ExtractedDoc` at all. Log and skip — the
            // scanner will retry on the next run via `RetryAfterError`.
            warn!(path = %job.path.display(), ?err, "extractor returned hard error");
            return;
        }
    };
    info!(
        n,
        total,
        path = %extracted.path.display(),
        status = extracted.status.as_str(),
        pages = extracted.page_count,
        chunks = extracted.chunks.len(),
        "extracted",
    );
    if tx.send(WriterMsg::Doc(Box::new(extracted))).is_err() {
        warn!(path = %job.path.display(), "writer thread closed; discarding result");
    }
}

/// Bounded-channel message into the DB writer thread.
///
/// `ExtractedDoc` is boxed because individual results can carry megabytes
/// of chunk text and the channel keeps several slots in flight.
enum WriterMsg {
    Doc(Box<ExtractedDoc>),
    Delete(PathBuf),
}

#[derive(Default)]
struct WriterCounters {
    ok: AtomicUsize,
    empty: AtomicUsize,
    error: AtomicUsize,
    deleted: AtomicUsize,
}

/// Run the DB writer until the channel disconnects.
///
/// `live_state` is `Some` when the writer is part of the long-lived
/// `run_watch` pipeline: each successful UPSERT and tombstone is
/// reflected into the supplied [`IndexState`]'s overlay so a query
/// run between two mutations sees a consistent snapshot.
///
/// When `live_state` is `None` (the `run_index` one-shot path), the
/// overlay isn't involved — `run_index` rebuilds the base index from
/// scratch on the next process start.
fn writer_thread(
    db_path: PathBuf,
    rx: flume::Receiver<WriterMsg>,
    counters: Arc<WriterCounters>,
    live_state: Option<Arc<IndexState>>,
) -> Result<()> {
    let mut db = Db::open(&db_path).context("writer thread: opening SQLite")?;
    while let Ok(msg) = rx.recv() {
        match msg {
            WriterMsg::Doc(doc) => {
                let status = doc.status;
                let path = doc.path.clone();
                match db.upsert_extracted(&doc) {
                    Ok(doc_id) => {
                        match status {
                            DocStatus::Ok => &counters.ok,
                            DocStatus::Empty => &counters.empty,
                            DocStatus::Error => &counters.error,
                            DocStatus::Deleted => &counters.deleted,
                        }
                        .fetch_add(1, Ordering::Relaxed);

                        // Reflect into the live overlay if we have one.
                        if let Some(state) = &live_state {
                            if let Err(err) = apply_overlay_for_upsert(&db, state, doc_id) {
                                warn!(path = %path.display(), ?err, "applying overlay update");
                            }
                        }
                    }
                    Err(err) => {
                        warn!(path = %path.display(), ?err, "upsert_extracted failed");
                        counters.error.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            WriterMsg::Delete(path) => match db.mark_deleted(&path) {
                Ok(Some(doc_id)) => {
                    counters.deleted.fetch_add(1, Ordering::Relaxed);
                    if let Some(state) = &live_state {
                        let base = state.load_base();
                        let mut ov = state.overlay.write();
                        ov.tombstone_doc(doc_id, &base);
                    }
                }
                Ok(None) => {
                    // Path wasn't known to the DB — nothing to tombstone.
                }
                Err(err) => {
                    warn!(path = %path.display(), ?err, "mark_deleted failed");
                }
            },
        }
    }
    Ok(())
}

/// After the writer has UPSERTed `doc_id`, fetch the freshly active
/// chunks from the DB and publish them into the overlay so they are
/// immediately searchable. The base index keeps the *old* chunks
/// (or none, for a brand-new doc); the overlay's tombstone hides the
/// stale ones, and the overlay's overflow carries the fresh ones.
fn apply_overlay_for_upsert(db: &Db, state: &IndexState, doc_id: i64) -> Result<()> {
    let rows = db.load_chunks_for_doc(doc_id)?;
    let chunks = build_chunk_items(rows);
    let base = state.load_base();
    let mut ov = state.overlay.write();
    if chunks.is_empty() {
        // The doc upserted with `status != Ok` (empty / error). We
        // still want to hide the stale base chunks: tombstone the
        // doc and drop any prior overflow rows.
        ov.tombstone_doc(doc_id, &base);
    } else {
        // The doc upserted with `Ok` and at least one chunk: swap
        // base for overflow atomically.
        ov.modify_doc(doc_id, chunks, &base);
    }
    // Day 6 owns the actual rebuild; here we only log the threshold.
    let _ = state.needs_rebuild(&ov, &base);
    Ok(())
}

/// Materialize `LoadedChunkRow`s into `ChunkItem`s with a single
/// shared `Arc<str>` for `path`/`filename` (matches what
/// `load_base_index_from_db` does for the base index).
fn build_chunk_items(rows: Vec<LoadedChunkRow>) -> Vec<ChunkItem> {
    if rows.is_empty() {
        return Vec::new();
    }
    let path_str = rows[0].path.to_string_lossy().into_owned();
    let filename = std::path::Path::new(&path_str)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path_str.clone());
    let path: Arc<str> = Arc::from(path_str.as_str());
    let filename: Arc<str> = Arc::from(filename.as_str());
    rows.into_iter()
        .map(|row| ChunkItem {
            chunk_id: row.chunk_id,
            doc_id: row.doc_id,
            path: path.clone(),
            filename: filename.clone(),
            page_no: row.page_no,
            chunk_ord: row.chunk_ord,
            char_start: row.char_start,
            char_end: row.char_end,
            text_utf8: Arc::<str>::from(row.text_utf8.as_str()),
            text_norm_ascii: Arc::<[u8]>::from(row.text_norm_ascii.as_bytes()),
            preview: Arc::<str>::from(row.preview.as_str()),
            doc_mtime_ns: row.doc_mtime_ns,
        })
        .collect()
}
