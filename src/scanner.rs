//! Filesystem walker + diff against `documents`.
//!
//! The scanner is the single producer of [`ScanJob`] values consumed by
//! the extractor pool. Day 1 supplies:
//!
//! * [`Scanner::walk_and_diff`] — synchronously walk a root directory
//!   with `ignore::WalkBuilder`, compute the diff against the `documents`
//!   table, and return a [`ScanResult`] describing the work to do.
//!
//! Later days plug this into a long-running worker thread; the
//! synchronous API stays useful for tests and one-shot `index` runs.

use anyhow::Result;
use ignore::WalkBuilder;
use std::collections::HashMap;
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;

use crate::db::{Db, DocStatus, DocumentRow, EXTRACTOR_NAME, extractor_version};
use crate::normalize::NORM_VERSION;

/// Why a scan decided a file needed extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirtyReason {
    /// Path not seen before.
    New,
    /// `mtime_ns` or `size_bytes` changed.
    Modified,
    /// `extractor_version` or `norm_version` no longer matches.
    StaleExtractor,
    /// The row exists but its previous status was `error` — retry.
    RetryAfterError,
}

/// One unit of work consumed by the extractor pool.
#[derive(Debug, Clone)]
pub struct ScanJob {
    pub path: PathBuf,
    pub size_bytes: i64,
    pub mtime_ns: i64,
    pub dev: Option<i64>,
    pub ino: Option<i64>,
    pub reason: DirtyReason,
}

/// Aggregated outcome of one full walk: paths that need extraction plus
/// paths that disappeared from disk and should be tombstoned.
#[derive(Debug, Default)]
pub struct ScanResult {
    pub jobs: Vec<ScanJob>,
    pub deleted: Vec<PathBuf>,
    pub seen_count: usize,
}

pub struct Scanner {
    pub root: PathBuf,
    pub follow_symlinks: bool,
    pub respect_gitignore: bool,
}

impl Scanner {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            follow_symlinks: false,
            respect_gitignore: false,
        }
    }

    /// Synchronously walk `self.root`, compare every PDF against the
    /// `documents` table, and return the diff.
    pub fn walk_and_diff(&self, db: &Db) -> Result<ScanResult> {
        let existing = db.load_all_documents()?;
        let mut by_path: HashMap<PathBuf, DocumentRow> = existing
            .into_iter()
            .map(|d| (d.path.clone(), d))
            .collect();

        let cur_extractor_version = extractor_version();
        let mut result = ScanResult::default();

        let walker = WalkBuilder::new(&self.root)
            .standard_filters(self.respect_gitignore)
            .git_ignore(self.respect_gitignore)
            .git_exclude(self.respect_gitignore)
            .git_global(self.respect_gitignore)
            .hidden(self.respect_gitignore)
            .follow_links(self.follow_symlinks)
            .build();

        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                Err(err) => {
                    tracing::debug!(?err, "walk error");
                    continue;
                }
            };

            // We only consider regular files with a `.pdf` extension.
            let path = entry.path();
            if !path.extension().is_some_and(|e| {
                e.to_string_lossy().eq_ignore_ascii_case("pdf")
            }) {
                continue;
            }
            let md = match entry.metadata() {
                Ok(m) => m,
                Err(err) => {
                    tracing::debug!(path = %path.display(), ?err, "stat failed");
                    continue;
                }
            };
            if !md.is_file() {
                continue;
            }

            result.seen_count += 1;
            let size_bytes = md.len() as i64;
            let mtime_ns = md
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos() as i64)
                .unwrap_or(0);
            let dev = Some(md.dev() as i64);
            let ino = Some(md.ino() as i64);

            let abs_path = path.to_path_buf();
            let reason = match by_path.remove(&abs_path) {
                None => Some(DirtyReason::New),
                Some(row) => decide_dirty(
                    &row,
                    size_bytes,
                    mtime_ns,
                    &cur_extractor_version,
                ),
            };
            if let Some(reason) = reason {
                result.jobs.push(ScanJob {
                    path: abs_path,
                    size_bytes,
                    mtime_ns,
                    dev,
                    ino,
                    reason,
                });
            }
        }

        // Anything left in `by_path` was on disk previously but is gone
        // now (or is `deleted` already, which we filter out).
        for (path, row) in by_path {
            if row.status != DocStatus::Deleted {
                result.deleted.push(path);
            }
        }

        Ok(result)
    }
}

fn decide_dirty(
    row: &DocumentRow,
    size_bytes: i64,
    mtime_ns: i64,
    cur_extractor_version: &str,
) -> Option<DirtyReason> {
    if row.status == DocStatus::Error {
        return Some(DirtyReason::RetryAfterError);
    }
    if row.size_bytes != size_bytes || row.mtime_ns != mtime_ns {
        return Some(DirtyReason::Modified);
    }
    if row.extractor != EXTRACTOR_NAME
        || row.extractor_version != cur_extractor_version
        || row.norm_version != NORM_VERSION
    {
        return Some(DirtyReason::StaleExtractor);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Db;
    use std::fs;
    use std::path::Path;
    use tempfile::tempdir;

    fn touch_pdf(p: &Path) {
        // Anything with a .pdf extension is enough for the scanner — we
        // don't extract here.
        fs::write(p, b"%PDF-1.4\nstub\n").unwrap();
    }

    #[test]
    fn first_walk_marks_everything_new() -> Result<()> {
        let tmp = tempdir()?;
        touch_pdf(&tmp.path().join("a.pdf"));
        touch_pdf(&tmp.path().join("b.pdf"));
        fs::write(tmp.path().join("not-a-pdf.txt"), b"ignore me").unwrap();

        let db_path = tmp.path().join("idx.db");
        let db = Db::open(&db_path)?;

        let scanner = Scanner::new(tmp.path());
        let result = scanner.walk_and_diff(&db)?;
        assert_eq!(result.jobs.len(), 2);
        assert!(result.jobs.iter().all(|j| j.reason == DirtyReason::New));
        assert_eq!(result.deleted.len(), 0);
        Ok(())
    }
}
