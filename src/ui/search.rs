//! Off-thread search worker shared by every interactive frontend.
//!
//! Both the TUI and the GUI need a background thread that:
//!
//! * owns an `Arc<IndexState>` and runs `query::search` against it,
//! * accepts queries from the input thread without blocking it,
//! * coalesces bursts of input (typing, key-repeat on Backspace) so
//!   only the *latest* request actually runs after the in-flight one
//!   finishes, and
//! * publishes results back to the input thread so the renderer can
//!   pick them up without waiting on a search.
//!
//! That is exactly the same concurrency contract regardless of whether
//! the frontend is ratatui or egui — so this module owns the
//! implementation and both frontends consume it as a library.
//!
//! Wire-up
//! -------
//! The frontend creates the worker once at startup from a
//! [`WatchHandle::state`](crate::app::WatchHandle::state) clone,
//! [`submit`](SearchWorker::submit)s a [`SearchRequest`] on every
//! query-affecting input event, and polls
//! [`take_result`](SearchWorker::take_result) once per render tick to
//! drain finished work. Every request carries a monotonic stamp; the
//! frontend echoes it back on the result and drops results that
//! predate the latest submitted stamp (the user has typed past them).
//!
//! Shutdown is via [`SearchWorker::stop`], which signals the worker
//! and joins it. Idempotent and safe to call from a panic path.

use anyhow::{Context, Result};
use parking_lot::{Condvar, Mutex};
use std::sync::Arc;
use std::thread;

use crate::index::IndexState;
use crate::query::{Hit, QueryMode, search};

/// A single pending search submitted to the worker.
///
/// `stamp` is a frontend-supplied monotonic counter the worker echoes
/// back in the result so the frontend can drop stale results.
#[derive(Debug, Clone)]
pub struct SearchRequest {
    pub stamp: u64,
    pub query: String,
    pub mode: QueryMode,
    pub limit: usize,
}

/// A search the worker has finished, ready for the frontend to apply.
pub struct SearchResult {
    pub stamp: u64,
    pub hits: Result<Vec<Hit>>,
}

/// One-slot mailbox guarded by a condvar.
///
/// `pending` is the next request to run; the worker takes it and runs
/// it, then loops back to wait for the next. `closed` lets the
/// frontend signal shutdown without sending a sentinel request.
struct SearchSlot {
    pending: Option<SearchRequest>,
    closed: bool,
}

/// Handle held by the frontend. Drop / [`SearchWorker::stop`] tear down
/// the background thread.
pub struct SearchWorker {
    slot: Arc<(Mutex<SearchSlot>, Condvar)>,
    result: Arc<Mutex<Option<SearchResult>>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl SearchWorker {
    /// Spawn the background worker. The worker holds its own clone of
    /// the index `Arc`, so the caller is free to drop its handle once
    /// the worker is alive.
    pub fn spawn(state: Arc<IndexState>) -> Result<Self> {
        let slot = Arc::new((
            Mutex::new(SearchSlot {
                pending: None,
                closed: false,
            }),
            Condvar::new(),
        ));
        let result: Arc<Mutex<Option<SearchResult>>> = Arc::new(Mutex::new(None));
        let slot_for_thread = slot.clone();
        let result_for_thread = result.clone();
        let handle = thread::Builder::new()
            .name("pdffff-search".into())
            .spawn(move || worker_loop(state, slot_for_thread, result_for_thread))
            .context("spawning pdffff search worker thread")?;
        Ok(Self {
            slot,
            result,
            handle: Some(handle),
        })
    }

    /// Overwrite the pending slot with `req` and wake the worker. If
    /// the worker is mid-search it finishes that search first; the new
    /// request runs next. Older pending requests are dropped silently
    /// — that's the coalescing behaviour the input thread depends on.
    pub fn submit(&self, req: SearchRequest) {
        let (m, cv) = &*self.slot;
        let mut s = m.lock();
        s.pending = Some(req);
        cv.notify_one();
    }

    /// Pop the latest published result, if any.
    pub fn take_result(&self) -> Option<SearchResult> {
        self.result.lock().take()
    }

    /// Signal the worker to exit and join it. Idempotent; safe to call
    /// from a drop guard or an explicit shutdown.
    pub fn stop(mut self) {
        {
            let (m, cv) = &*self.slot;
            let mut s = m.lock();
            s.closed = true;
            cv.notify_one();
        }
        if let Some(h) = self.handle.take() {
            // Best-effort: a worker panic would already have been
            // surfaced through `result` as an `Err`, so a Join error
            // here is purely shutdown noise we don't want to leak into
            // the frontend's exit path.
            let _ = h.join();
        }
    }
}

fn worker_loop(
    state: Arc<IndexState>,
    slot: Arc<(Mutex<SearchSlot>, Condvar)>,
    result: Arc<Mutex<Option<SearchResult>>>,
) {
    let (m, cv) = &*slot;
    loop {
        let request = {
            let mut s = m.lock();
            loop {
                if s.closed {
                    return;
                }
                if let Some(r) = s.pending.take() {
                    break r;
                }
                cv.wait(&mut s);
            }
        };
        let hits = search(&state, &request.query, request.mode, request.limit);
        *result.lock() = Some(SearchResult {
            stamp: request.stamp,
            hits,
        });
    }
}
