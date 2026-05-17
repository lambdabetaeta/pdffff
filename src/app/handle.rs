//! The live handle that [`super::run_watch`] returns and the running
//! counters of indexer activity it exposes.

use anyhow::Result;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;

use crate::index::IndexState;
use crate::watcher::WatcherHandle;

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

/// Live handle exposed by [`super::run_watch`]. Owns every thread the
/// watch loop spawned and signals them to shut down on
/// [`WatchHandle::stop`].
pub struct WatchHandle {
    /// Shared with the coordinator: when set, the writer / coordinator
    /// drain their queues and exit at the next opportunity.
    pub(crate) stop: Arc<AtomicBool>,
    /// Sender into the coordinator's stop channel. Sending one `()`
    /// wakes the selector immediately.
    pub(crate) stop_tx: flume::Sender<()>,
    /// Joined by [`stop`] to surface coordinator / writer panics.
    pub(crate) coordinator: Option<thread::JoinHandle<Result<()>>>,
    pub(crate) writer: Option<thread::JoinHandle<Result<()>>>,
    /// Kept alive for the lifetime of the watcher; dropped in `stop`
    /// to halt notify's internal thread.
    pub(crate) _watcher: Option<WatcherHandle>,
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
            let res = h
                .join()
                .map_err(|_| anyhow::anyhow!("watch coordinator panicked"))?;
            res?;
        }
        if let Some(h) = self.writer.take() {
            let res = h
                .join()
                .map_err(|_| anyhow::anyhow!("watch writer panicked"))?;
            res?;
        }
        Ok(())
    }
}
