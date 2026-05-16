//! Top-level helpers and orchestration glue.
//!
//! Day 1 only needs a `now_ms` helper used by [`crate::db`]. The full
//! orchestrator (channel wiring, worker pool, watcher) lands as later
//! days replace the placeholder modules.

use std::time::{SystemTime, UNIX_EPOCH};

/// Wall-clock milliseconds since the Unix epoch. Used as `indexed_at_ms`
/// / `deleted_at_ms` timestamps in `documents`.
pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
