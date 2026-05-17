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
//! [`Scanner::walk_and_diff`] pass to discover what's on disk, then
//! dispatches the dirty PDFs to the same rayon pool the
//! `notify-debouncer-full` watcher feeds. Both initial-sync extractions
//! and live filesystem events therefore travel the same path: there is
//! no separate "initial sync" phase that blocks the caller. The TUI
//! receives the [`WatchHandle`] immediately and starts serving queries
//! against whatever's already in SQLite (which may be empty on a fresh
//! corpus, or a previous session's snapshot); each successful
//! extraction streams into the overlay so results pop in progressively.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

use crate::db::{Db, DocStatus, ExtractedDoc, LoadedChunkRow};
use crate::extract::{ensure_pdftotext_available, extract_pdf};
use crate::index::{ChunkItem, IndexState, load_base_index_from_db};
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

/// Knobs for [`run_watch`].
#[derive(Debug, Clone)]
pub struct WatchOptions {
    pub respect_gitignore: bool,
    pub follow_symlinks: bool,
    /// Override extractor pool size. Default: `min(num_cpus, 6)`.
    pub jobs: Option<usize>,
    /// If true, fail fast at startup when `pdftotext` is missing. Tests
    /// can disable this only when they have pre-checked.
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

/// Live handle exposed by [`run_watch`]. Owns every thread the watch
/// loop spawned and signals them to shut down on [`WatchHandle::stop`].
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
    /// Live in-memory index. Exposed so the TUI can run queries while
    /// the watcher is active.
    pub state: Arc<IndexState>,
    /// Running counters of writer-thread activity. The TUI samples
    /// these every tick to render the indexer status bar.
    pub progress: Arc<IndexProgress>,
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
/// 1. Verify `pdftotext` (unless the caller disables that for tests).
/// 2. Open the DB and load whatever is currently in it into
///    [`IndexState`]. This may be empty on first launch, or a previous
///    session's snapshot — either way, queries work immediately.
/// 3. Spawn the long-lived DB-writer + rayon extractor pool + watcher.
/// 4. Spawn the coordinator. Its first action is a synchronous
///    [`Scanner::walk_and_diff`] pass that discovers everything on disk
///    and dispatches the dirty PDFs to the same pool the watcher uses;
///    then it enters the live event loop.
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
    let (watch_tx, watch_rx) = flume::unbounded::<WatchEvent>();
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

/// Resolve where pdffff stores the SQLite DB for a given corpus root.
///
/// Convention: `<data_dir>/pdffff/<basename>-<8-hex of blake3(canonical)>.db`,
/// where `data_dir` is `dirs::data_dir()` (`$XDG_DATA_HOME` on Linux,
/// `~/Library/Application Support` on macOS, `%APPDATA%` on Windows).
/// The basename gives the file a human-readable hint of which corpus
/// it backs; the hash disambiguates two folders that happen to share a
/// basename.
///
/// Side-effect: creates the parent directory if it doesn't exist, so
/// the caller can hand the returned path directly to `rusqlite`.
pub fn resolve_db_path(root: &Path) -> Result<PathBuf> {
    let canonical = root.canonicalize().with_context(|| {
        format!("canonicalising corpus root {}", root.display())
    })?;
    let basename = canonical
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        // Falls back when root is `/` — vanishingly rare but possible.
        .unwrap_or_else(|| "root".to_string());
    let hash = blake3::hash(canonical.as_os_str().as_encoded_bytes());
    let short = hex8(hash.as_bytes());
    let mut dir = dirs::data_dir()
        .context("could not determine the user's data directory (XDG_DATA_HOME / equivalent)")?;
    dir.push("pdffff");
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating data dir {}", dir.display()))?;
    Ok(dir.join(format!("{}-{}.db", sanitize(&basename), short)))
}

/// Tiny basename sanitiser: keep alphanumerics, dash, underscore, dot;
/// replace everything else with `_`. Stops the DB filename from
/// inheriting awkward characters (spaces, `:`, etc.) from the corpus
/// folder name.
fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn hex8(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(16);
    for &b in bytes.iter().take(8) {
        use std::fmt::Write;
        let _ = write!(out, "{b:02x}");
    }
    out
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
fn coordinator_thread(
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
        match Db::open_reader(&db_path) {
            Ok(db) => {
                let mut scanner = Scanner::new(&root);
                scanner.respect_gitignore = scan.respect_gitignore;
                scanner.follow_symlinks = scan.follow_symlinks;
                match scanner.walk_and_diff(&db) {
                    Ok(result) => {
                        info!(
                            seen = result.seen_count,
                            dirty = result.jobs.len(),
                            deleted = result.deleted.len(),
                            "initial scan complete",
                        );
                        for job in result.jobs {
                            if stop.load(Ordering::Relaxed) {
                                break;
                            }
                            dispatch_extract(&pool, &writer_tx, &progress, job);
                        }
                        for path in result.deleted {
                            if stop.load(Ordering::Relaxed) {
                                break;
                            }
                            if writer_tx.send(WriterMsg::Delete(path)).is_err() {
                                warn!("writer thread closed before initial deletes were enqueued");
                                break;
                            }
                        }
                    }
                    Err(err) => warn!(?err, "initial scanner walk failed"),
                }
            }
            Err(err) => warn!(?err, "opening reader DB for initial scan failed"),
        }
    }

    // ---- Live event loop ------------------------------------------------
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
                dispatch_extract(&pool, &writer_tx, &progress, job);
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

/// Subset of [`WatchOptions`] the coordinator needs for the initial
/// scan. Kept as a separate struct so the coordinator's signature does
/// not grow every time we add a watcher-only option.
#[derive(Debug, Clone, Copy)]
struct ScanParams {
    respect_gitignore: bool,
    follow_symlinks: bool,
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

fn default_pool_size() -> usize {
    let n = thread::available_parallelism().map(|n| n.get()).unwrap_or(2);
    n.min(6).max(1)
}

/// Bounded-channel message into the DB writer thread.
///
/// `ExtractedDoc` is boxed because individual results can carry megabytes
/// of chunk text and the channel keeps several slots in flight.
enum WriterMsg {
    Doc(Box<ExtractedDoc>),
    Delete(PathBuf),
}

/// Running counters of indexer activity.
///
/// Updated from the watcher's coordinator (for `pending`) and from the
/// DB writer thread (for the terminal-status counters). All fields are
/// observable from any thread, including the TUI's render loop, so
/// they are atomics rather than counters guarded by a mutex.
///
/// `pending` is "extraction jobs the coordinator has dispatched to the
/// rayon pool but for which the writer has not yet observed a `Doc`
/// message" — it's the right cardinality for a "currently indexing"
/// status bar.
#[derive(Default)]
pub struct IndexProgress {
    pub ok: AtomicUsize,
    pub empty: AtomicUsize,
    pub error: AtomicUsize,
    pub deleted: AtomicUsize,
    pub pending: AtomicUsize,
}

/// Run the DB writer until the channel disconnects. Every successful
/// UPSERT and tombstone is reflected into the supplied [`IndexState`]'s
/// overlay so a query run between two mutations sees a consistent
/// snapshot.
fn writer_thread(
    db_path: PathBuf,
    rx: flume::Receiver<WriterMsg>,
    counters: Arc<IndexProgress>,
    live_state: Arc<IndexState>,
) -> Result<()> {
    let mut db = Db::open(&db_path).context("writer thread: opening SQLite")?;
    while let Ok(msg) = rx.recv() {
        let mutated = process_writer_msg(&mut db, &live_state, &counters, msg);

        // After each mutation that touched the overlay, check the
        // rebuild thresholds. The check itself is cheap (one stats
        // sweep); the rebuild only fires when the predicate trips.
        //
        // Doing this on the writer thread (rather than a separate
        // rebuilder thread) is intentional: the writer is the *only*
        // mutator of the overlay, so rebuilding here means we cannot
        // race a concurrent overlay update against an in-flight
        // rebuild. The brief stall on the writer is acceptable —
        // rebuild_from_db on the threshold-tripping corpora (10k
        // overflow chunks, ~10% tombstones) is bounded by the time
        // to stream chunks from SQLite plus the dense-bigram build,
        // which is the same work startup pays on every process boot.
        if mutated {
            match live_state.rebuild_if_needed(&db) {
                Ok(true) => info!("writer thread completed a base rebuild"),
                Ok(false) => {}
                Err(err) => warn!(?err, "rebuild_if_needed failed"),
            }
        }
    }
    Ok(())
}

/// Apply one [`WriterMsg`] to the DB and the live overlay.
///
/// Returns `true` if the overlay was mutated (so the writer loop knows
/// to re-check the rebuild thresholds), `false` otherwise. Failures are
/// logged and folded into the `false` branch — the writer keeps running
/// so one bad row can't take down the indexer.
fn process_writer_msg(
    db: &mut Db,
    state: &IndexState,
    counters: &IndexProgress,
    msg: WriterMsg,
) -> bool {
    match msg {
        WriterMsg::Doc(doc) => apply_doc(db, state, counters, *doc),
        WriterMsg::Delete(path) => apply_delete(db, state, counters, &path),
    }
}

fn apply_doc(db: &mut Db, state: &IndexState, counters: &IndexProgress, doc: ExtractedDoc) -> bool {
    let status = doc.status;
    let path = doc.path.clone();
    match db.upsert_extracted(&doc) {
        Ok(doc_id) => {
            counter_for(counters, status).fetch_add(1, Ordering::Relaxed);
            if let Err(err) = apply_overlay_for_upsert(db, state, doc_id) {
                warn!(path = %path.display(), ?err, "applying overlay update");
            }
            true
        }
        Err(err) => {
            warn!(path = %path.display(), ?err, "upsert_extracted failed");
            counters.error.fetch_add(1, Ordering::Relaxed);
            false
        }
    }
}

fn apply_delete(db: &mut Db, state: &IndexState, counters: &IndexProgress, path: &Path) -> bool {
    match db.mark_deleted(path) {
        Ok(Some(doc_id)) => {
            counters.deleted.fetch_add(1, Ordering::Relaxed);
            let base = state.load_base();
            let mut ov = state.overlay.write();
            ov.tombstone_doc(doc_id, &base);
            true
        }
        // Path wasn't known to the DB — nothing to tombstone.
        Ok(None) => false,
        Err(err) => {
            warn!(path = %path.display(), ?err, "mark_deleted failed");
            false
        }
    }
}

/// Counter inside [`IndexProgress`] corresponding to `status`.
///
/// Centralising the dispatch means a future `DocStatus` variant has
/// exactly one place to consider; the writer thread itself never sees
/// the mapping.
fn counter_for(counters: &IndexProgress, status: DocStatus) -> &AtomicUsize {
    match status {
        DocStatus::Ok => &counters.ok,
        DocStatus::Empty => &counters.empty,
        DocStatus::Error => &counters.error,
        DocStatus::Deleted => &counters.deleted,
    }
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
    // The writer thread runs `state.rebuild_if_needed(&db)` after this
    // returns — once the write lock has been dropped — so the threshold
    // check is part of the message-loop tick, not this function.
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
