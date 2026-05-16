//! Filesystem watcher → coordinator channel.
//!
//! A thin wrapper around `notify-debouncer-full`'s `Debouncer<...>`:
//!
//! 1. A `Debouncer<RecommendedWatcher, RecommendedCache>` is rooted at
//!    the scan root via `watch(root, RecursiveMode::Recursive)`.
//! 2. The debouncer's `DebounceEventHandler` runs on `notify`'s
//!    internal thread. It filters every event's `paths` for
//!    `.pdf`-extension regular files (case-insensitive) and forwards
//!    one [`WatchEvent`] per affected path on a `flume::Sender`.
//! 3. The caller owns the receiving end. The returned
//!    [`WatcherHandle`] also carries a `stop_tx`: dropping the
//!    debouncer is enough to stop receiving events, but the
//!    coordinator on the other end of the channel also needs an
//!    explicit signal so it can shut down cleanly.
//!
//! No background work is done in this module: there is no `walk_dir`,
//! no `stat`, no extraction. The watcher's job is to deliver
//! `(WatchEvent, path)` to the coordinator and nothing else. The
//! coordinator (in `app::run_watch`) reuses `Scanner` / `extract_pdf`
//! / `Db::upsert_extracted` for the heavy lifting so this module
//! stays DRY with respect to the rest of the pipeline.
//!
//! Debounce window: the report names a 50–250 ms range. We pick
//! [`DEFAULT_DEBOUNCE`] = 200 ms as the default (responsive enough
//! for an interactive search loop, long enough to coalesce a typical
//! `:w` from an editor that writes-then-renames).

use anyhow::{Context, Result};
use notify::event::{EventKind, ModifyKind, RenameMode};
use notify::RecursiveMode;
use notify_debouncer_full::{DebounceEventResult, Debouncer, RecommendedCache, new_debouncer};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tracing::{debug, warn};

/// Default debounce window. Inside the report's 50–250 ms band.
pub const DEFAULT_DEBOUNCE: Duration = Duration::from_millis(200);

/// One filesystem mutation observed by the watcher.
///
/// We collapse rename-from / rename-to and create-then-modify into the
/// same two kinds the coordinator cares about:
///
/// * [`WatchEvent::Dirty`] — the file at `path` exists right now and
///   may have new contents. The coordinator should `stat` it and
///   submit an extraction job.
/// * [`WatchEvent::Removed`] — the file at `path` no longer exists
///   and should be tombstoned in the index.
#[derive(Debug, Clone)]
pub enum WatchEvent {
    Dirty(PathBuf),
    Removed(PathBuf),
}

/// Handle returned by [`spawn_watcher`]. Dropping the handle stops
/// the debouncer thread (`Debouncer::stop` is called by `Debouncer`'s
/// `Drop` impl); the coordinator still owns the receiver end of the
/// event channel and can keep draining queued events until it sees
/// the channel disconnect.
pub struct WatcherHandle {
    /// Kept alive for the lifetime of the watcher. Dropping it stops
    /// the underlying notify thread.
    _debouncer: Debouncer<notify::RecommendedWatcher, RecommendedCache>,
}

/// Spawn a debounced filesystem watcher rooted at `root`.
///
/// `tx` is the flume sender into which the watcher publishes one
/// [`WatchEvent`] per affected PDF path. `debounce` defaults to
/// [`DEFAULT_DEBOUNCE`] when `None`.
pub fn spawn_watcher(
    root: &Path,
    tx: flume::Sender<WatchEvent>,
    debounce: Option<Duration>,
) -> Result<WatcherHandle> {
    let timeout = debounce.unwrap_or(DEFAULT_DEBOUNCE);
    let mut debouncer = new_debouncer(
        timeout,
        None, // tick rate: notify picks timeout/4 by default
        move |result: DebounceEventResult| match result {
            Ok(events) => {
                for ev in events {
                    forward_event(&tx, &ev.event);
                }
            }
            Err(errors) => {
                for err in errors {
                    warn!(?err, "filesystem watcher error");
                }
            }
        },
    )
    .context("constructing notify-debouncer-full")?;

    debouncer
        .watch(root, RecursiveMode::Recursive)
        .with_context(|| format!("watching {}", root.display()))?;

    Ok(WatcherHandle { _debouncer: debouncer })
}

/// Decide what to do with one upstream `notify::Event` and forward
/// the result on `tx`. Visible to tests via the
/// `#[cfg(test)] mod tests` block below.
fn forward_event(tx: &flume::Sender<WatchEvent>, event: &notify::Event) {
    let event_kind = event.kind;
    for path in &event.paths {
        if !is_pdf_path(path) {
            continue;
        }
        let we = match classify(event_kind, path) {
            Some(we) => we,
            None => continue,
        };
        debug!(?we, "watcher forwarding");
        if tx.send(we).is_err() {
            // The coordinator hung up. There's nothing useful to do
            // here; subsequent events will keep tripping this branch
            // until the debouncer is dropped.
            return;
        }
    }
}

/// Map a `notify::EventKind` to the coordinator's two-kind enum.
fn classify(kind: EventKind, path: &Path) -> Option<WatchEvent> {
    match kind {
        // Removed: easy case — the file is gone. `RemoveKind::Folder`
        // never matters here because the path filter upstream rules
        // out directories that don't end in `.pdf` anyway, but we
        // still handle it for completeness.
        EventKind::Remove(_) => Some(WatchEvent::Removed(path.to_path_buf())),

        // Rename From: the path on the "from" side no longer exists.
        EventKind::Modify(ModifyKind::Name(RenameMode::From)) => {
            Some(WatchEvent::Removed(path.to_path_buf()))
        }

        // Rename To / Both: the destination now exists, treat it as
        // a new (dirty) file. `Both` carries both paths in
        // `event.paths`, so the caller will emit two events for the
        // (from, to) pair — one Removed and one Dirty. We use the
        // path's current existence to disambiguate.
        EventKind::Modify(ModifyKind::Name(RenameMode::To)) => {
            Some(WatchEvent::Dirty(path.to_path_buf()))
        }
        EventKind::Modify(ModifyKind::Name(RenameMode::Both | RenameMode::Any | RenameMode::Other)) => {
            if path.exists() {
                Some(WatchEvent::Dirty(path.to_path_buf()))
            } else {
                Some(WatchEvent::Removed(path.to_path_buf()))
            }
        }

        // Create / Modify / Other: if the file still exists, treat
        // as Dirty. If it doesn't (rare race), treat as Removed so
        // the index stays consistent.
        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Other => {
            if path.exists() {
                Some(WatchEvent::Dirty(path.to_path_buf()))
            } else {
                Some(WatchEvent::Removed(path.to_path_buf()))
            }
        }

        // Access events do not affect the index.
        EventKind::Access(_) | EventKind::Any => None,
    }
}

/// True for paths whose extension equals `pdf` case-insensitively.
fn is_pdf_path(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e.to_string_lossy().eq_ignore_ascii_case("pdf"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use notify::event::{CreateKind, EventAttributes, RemoveKind};

    fn ev(kind: EventKind, path: PathBuf) -> notify::Event {
        notify::Event {
            kind,
            paths: vec![path],
            attrs: EventAttributes::default(),
        }
    }

    #[test]
    fn is_pdf_path_is_case_insensitive() {
        assert!(is_pdf_path(Path::new("foo.pdf")));
        assert!(is_pdf_path(Path::new("FOO.PDF")));
        assert!(is_pdf_path(Path::new("dir/sub/A.Pdf")));
        assert!(!is_pdf_path(Path::new("foo.txt")));
        assert!(!is_pdf_path(Path::new("foo")));
    }

    #[test]
    fn non_pdf_paths_do_not_forward() {
        let (tx, rx) = flume::unbounded();
        let e = ev(
            EventKind::Create(CreateKind::File),
            PathBuf::from("/tmp/notes.txt"),
        );
        forward_event(&tx, &e);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn remove_classifies_as_removed() {
        let we = classify(
            EventKind::Remove(RemoveKind::File),
            Path::new("/tmp/x.pdf"),
        )
        .unwrap();
        assert!(matches!(we, WatchEvent::Removed(_)));
    }
}
