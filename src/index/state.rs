//! The handle the rest of the system reads from: an
//! [`ArcSwap<BaseIndex>`] paired with a [`RwLock<Overlay>`].
//!
//! Two consistency rules are load-bearing:
//!
//! 1. **No double-visibility.** When a doc is modified, the tombstone
//!    hides the doc's base range *before* the overflow rows become
//!    readable, so a reader never sees both the stale and the fresh
//!    chunks for the same doc. Atomicity is enforced by
//!    [`Overlay::modify_doc`] under one write guard.
//! 2. **Snapshot reads.** A query holds `state.load_base()` *and* a
//!    single read guard on `state.overlay` for the entire literal
//!    pass.

use anyhow::Result;
use arc_swap::ArcSwap;
use parking_lot::RwLock;
use std::sync::Arc;
use tracing::info;

use crate::db::Db;
use crate::index::base::{BaseIndex, load_base_index_from_db};
use crate::index::overlay::{
    Overlay, REBUILD_OVERLAY_CHUNKS, REBUILD_TOMBSTONE_RATIO,
};

pub struct IndexState {
    pub base: ArcSwap<BaseIndex>,
    pub overlay: RwLock<Overlay>,
}

impl IndexState {
    pub fn new(base: BaseIndex) -> Self {
        let chunk_count = base.chunks.len();
        Self {
            base: ArcSwap::new(Arc::new(base)),
            overlay: RwLock::new(Overlay::new(chunk_count)),
        }
    }

    pub fn empty() -> Self {
        Self::new(BaseIndex::empty())
    }

    pub fn load_base(&self) -> Arc<BaseIndex> {
        self.base.load_full()
    }

    /// True when the overlay has drifted far enough from the base
    /// that a rebuild would be cheaper than continuing to scan it.
    ///
    /// Two thresholds:
    /// * Too many overflow chunks: querying them is a linear scan,
    ///   so once we cross [`REBUILD_OVERLAY_CHUNKS`] the overlay
    ///   stops being a prefiltered side-set and starts being a slow
    ///   second corpus.
    /// * Too many tombstones: the dense bitset stays the same size
    ///   regardless of how many bits are set, but verifying
    ///   tombstoned candidates is wasted work.
    pub fn needs_rebuild(&self, ov: &Overlay, base: &BaseIndex) -> bool {
        let stats = ov.stats();
        let by_overflow = stats.overflow_chunks >= REBUILD_OVERLAY_CHUNKS;
        // Compute the ratio against the *base chunk count*, not the
        // bitset capacity, so a corpus of `64 + 1` chunks doesn't
        // get a divide-by-larger-than-base ratio.
        let base_count = base.chunks.len().max(1);
        let tombstone_ratio = stats.tombstones as f64 / base_count as f64;
        let by_tombstones = tombstone_ratio >= REBUILD_TOMBSTONE_RATIO;
        if by_overflow || by_tombstones {
            info!(
                overflow_chunks = stats.overflow_chunks,
                tombstones = stats.tombstones,
                tombstone_ratio,
                "overlay crossed a rebuild threshold",
            );
            return true;
        }
        false
    }

    /// Check `needs_rebuild` against the current `(base, overlay)`
    /// snapshot. If true, drop the read guards, run
    /// [`rebuild_from_db`], and emit a `tracing::info!` summarising
    /// the new state.
    pub fn rebuild_if_needed(&self, db: &Db) -> Result<bool> {
        // Hold the read locks for *exactly* the check; drop them
        // before acquiring the write lock inside `rebuild_from_db` so
        // we don't deadlock against ourselves.
        let needs = {
            let ov = self.overlay.read();
            let base = self.base.load();
            self.needs_rebuild(&ov, &base)
        };
        if !needs {
            return Ok(false);
        }
        rebuild_from_db(self, db)?;
        Ok(true)
    }
}

/// Rebuild the [`BaseIndex`] from SQLite and atomically swap it into
/// `state`, then reset the overlay.
///
/// The swap is intentionally minimal: a single `ArcSwap::store` for the
/// base plus an overlay reset under a single write lock. A reader either
/// sees the *old* `(base, overlay)` pair or the *new* `(base, fresh
/// overlay)` pair, never a torn snapshot — because every reader holds
/// its `overlay.read()` guard for the duration of its candidate +
/// verification passes, and the swap of `base` is itself atomic via
/// `arc-swap`.
pub fn rebuild_from_db(state: &IndexState, db: &Db) -> Result<()> {
    let new_base = load_base_index_from_db(db)?;
    let new_chunk_count = new_base.chunks.len();
    let prev_overflow;
    let prev_tombstones;
    let prev_generation;
    {
        // Acquire the write lock *before* publishing the new base so
        // concurrent readers don't observe a (new_base, stale_overlay)
        // pair where stale_overlay's tombstones index into the
        // previous base layout.
        let mut ov = state.overlay.write();
        prev_overflow = ov.overflow.len();
        prev_tombstones = ov.tombstones.count_ones();
        prev_generation = ov.generation;
        state.base.store(Arc::new(new_base));
        *ov = Overlay::new(new_chunk_count);
    }
    info!(
        chunks = new_chunk_count,
        prev_overflow,
        prev_tombstones,
        prev_generation,
        "base index rebuilt and overlay reset",
    );
    Ok(())
}
