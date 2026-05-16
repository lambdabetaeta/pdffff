//! Top-level orchestrator: workers, channels, lifecycle.
//!
//! Day 2 puts the *indexing* pipeline here:
//!
//! ```text
//!   Scanner ──► rayon extractor pool ──► flume bounded channel ──►
//!     single DB-writer thread (owns the only writer Connection)
//! ```
//!
//! Keeping the pipeline in a library module rather than `main.rs` lets
//! integration tests drive the same code paths without going through
//! clap. `main.rs` is a thin wrapper around [`run_index`].
//!
//! Later days will add the in-memory `BaseIndex` rebuild step after
//! extraction, the watcher for incremental updates, and the query loop.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

use crate::db::{Db, DocStatus, ExtractedDoc};
use crate::extract::{ensure_pdftotext_available, extract_pdf, probe_pdftotext_or_explain};
use crate::index::{IndexState, load_base_index_from_db};
use crate::query::{Hit, QueryMode, search};
use crate::scanner::{ScanJob, Scanner};

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

/// Top-level pipeline driver.
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
        .spawn(move || writer_thread(writer_db_path, rx, writer_counters))
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

fn writer_thread(
    db_path: PathBuf,
    rx: flume::Receiver<WriterMsg>,
    counters: Arc<WriterCounters>,
) -> Result<()> {
    let mut db = Db::open(&db_path).context("writer thread: opening SQLite")?;
    while let Ok(msg) = rx.recv() {
        match msg {
            WriterMsg::Doc(doc) => match db.upsert_extracted(&doc) {
                Ok(_) => {
                    match doc.status {
                        DocStatus::Ok => &counters.ok,
                        DocStatus::Empty => &counters.empty,
                        DocStatus::Error => &counters.error,
                        DocStatus::Deleted => &counters.deleted,
                    }
                    .fetch_add(1, Ordering::Relaxed);
                }
                Err(err) => {
                    warn!(path = %doc.path.display(), ?err, "upsert_extracted failed");
                    counters.error.fetch_add(1, Ordering::Relaxed);
                }
            },
            WriterMsg::Delete(path) => match db.mark_deleted(&path) {
                Ok(Some(_)) => {
                    counters.deleted.fetch_add(1, Ordering::Relaxed);
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
