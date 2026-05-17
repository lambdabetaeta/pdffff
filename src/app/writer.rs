//! The single DB-writer thread.
//!
//! Owns the only writer SQLite connection. Each [`WriterMsg`] is
//! applied to the DB and the live overlay under a single tick of the
//! receive loop; after every overlay mutation the rebuild thresholds
//! are checked so the overflow stays bounded.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::{info, warn};

use crate::db::{Db, DocStatus, ExtractedDoc, LoadedChunkRow};
use crate::index::{ChunkItem, IndexState};

use super::handle::IndexProgress;

/// Bounded-channel message into the DB writer thread.
///
/// `ExtractedDoc` is boxed because individual results can carry
/// megabytes of chunk text and the channel keeps several slots in
/// flight.
pub enum WriterMsg {
    Doc(Box<ExtractedDoc>),
    Delete(PathBuf),
}

/// Run the DB writer until the channel disconnects. Every successful
/// UPSERT and tombstone is reflected into the supplied [`IndexState`]'s
/// overlay so a query run between two mutations sees a consistent
/// snapshot.
pub(crate) fn writer_thread(
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
/// to re-check the rebuild thresholds), `false` otherwise. Failures
/// are logged and folded into the `false` branch — the writer keeps
/// running so one bad row can't take down the indexer.
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
/// immediately searchable. The base index keeps the *old* chunks (or
/// none, for a brand-new doc); the overlay's tombstone hides the
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
    Ok(())
}

/// Materialise `LoadedChunkRow`s into `ChunkItem`s with a single
/// shared `Arc<str>` for `path`/`filename` (matches what
/// `load_base_index_from_db` does for the base index).
fn build_chunk_items(rows: Vec<LoadedChunkRow>) -> Vec<ChunkItem> {
    if rows.is_empty() {
        return Vec::new();
    }
    let (path, filename) = shared_path_arcs(&rows[0].path);
    rows.into_iter()
        .map(|row| {
            ChunkItem::new(
                row.chunk_id,
                row.doc_id,
                path.clone(),
                filename.clone(),
                row.page_no,
                row.chunk_ord,
                row.char_start,
                row.char_end,
                Arc::<str>::from(row.text_utf8.as_str()),
                Arc::<[u8]>::from(row.text_norm_ascii.as_bytes()),
                Arc::<str>::from(row.preview.as_str()),
                row.doc_mtime_ns,
            )
        })
        .collect()
}

/// `(path, filename)` shared `Arc<str>`s for a single doc — one
/// allocation each, cloned into every chunk of the doc.
fn shared_path_arcs(path: &Path) -> (Arc<str>, Arc<str>) {
    let path_str = path.to_string_lossy().into_owned();
    let filename = std::path::Path::new(&path_str)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path_str.clone());
    (
        Arc::<str>::from(path_str.as_str()),
        Arc::<str>::from(filename.as_str()),
    )
}
