//! SQLite-backed durability layer.
//!
//! `documents` carries the re-extraction contract (extractor + norm
//! version + file identity); `chunks` carries display text plus
//! search-normalized text. WAL mode + `synchronous=NORMAL` matches
//! rga's choice.

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};
use std::path::{Path, PathBuf};

pub const EXTRACTOR_NAME: &str = "pdftotext";

/// Records pdftotext's reported version so a poppler upgrade can force
/// re-extraction by mismatching the stored `extractor_version`.
pub fn extractor_version() -> String {
    use std::process::Command;
    let output = Command::new("pdftotext").arg("-v").output();
    match output {
        Ok(o) => {
            // `pdftotext -v` writes the banner to stderr.
            let banner = String::from_utf8_lossy(&o.stderr);
            banner
                .lines()
                .next()
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| "unknown".to_string())
        }
        Err(_) => "missing".to_string(),
    }
}

const SCHEMA: &str = r#"
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA temp_store = MEMORY;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
  doc_id              INTEGER PRIMARY KEY,
  path                TEXT NOT NULL UNIQUE,
  size_bytes          INTEGER NOT NULL,
  mtime_ns            INTEGER NOT NULL,
  dev                 INTEGER,
  ino                 INTEGER,
  extractor           TEXT NOT NULL,
  extractor_version   TEXT NOT NULL,
  norm_version        INTEGER NOT NULL,
  page_count          INTEGER NOT NULL DEFAULT 0,
  status              TEXT NOT NULL,
  error_text          TEXT,
  indexed_at_ms       INTEGER NOT NULL,
  deleted_at_ms       INTEGER
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id            INTEGER PRIMARY KEY,
  doc_id              INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  page_no             INTEGER NOT NULL,
  chunk_ord           INTEGER NOT NULL,
  char_start          INTEGER NOT NULL,
  char_end            INTEGER NOT NULL,
  text_utf8           TEXT NOT NULL,
  text_norm_ascii     TEXT NOT NULL,
  preview             TEXT NOT NULL,
  active              INTEGER NOT NULL DEFAULT 1,
  UNIQUE(doc_id, page_no, chunk_ord)
);

CREATE INDEX IF NOT EXISTS idx_documents_status_path
  ON documents(status, path);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_ord
  ON chunks(doc_id, chunk_ord);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_page
  ON chunks(doc_id, page_no, chunk_ord);
"#;

/// PRAGMAs to apply to every fresh connection (PRAGMAs are connection-local
/// except WAL mode itself, which persists once set).
const CONNECTION_PRAGMAS: &str = r#"
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA temp_store = MEMORY;
PRAGMA busy_timeout = 5000;
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocStatus {
    Ok,
    Empty,
    Error,
    Deleted,
}

impl DocStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            DocStatus::Ok => "ok",
            DocStatus::Empty => "empty",
            DocStatus::Error => "error",
            DocStatus::Deleted => "deleted",
        }
    }
    /// Parse the textual form persisted in `documents.status`. Returns
    /// `None` for any value not produced by `as_str`. We do not
    /// implement `std::str::FromStr` because the parse here is total in
    /// our domain (any unknown text is treated as a corrupt row by the
    /// loader) and the `Option` shape matches that contract better.
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "ok" => DocStatus::Ok,
            "empty" => DocStatus::Empty,
            "error" => DocStatus::Error,
            "deleted" => DocStatus::Deleted,
            _ => return None,
        })
    }
}

/// Snapshot of a row in `documents` used by the scanner's diff.
#[derive(Debug, Clone)]
pub struct DocumentRow {
    pub doc_id: i64,
    pub path: PathBuf,
    pub size_bytes: i64,
    pub mtime_ns: i64,
    pub extractor: String,
    pub extractor_version: String,
    pub norm_version: i64,
    pub status: DocStatus,
}

/// One chunk's worth of data, as fed into an UPSERT.
#[derive(Debug, Clone)]
pub struct ChunkInsert {
    pub page_no: u32,
    pub chunk_ord: u32,
    pub char_start: u32,
    pub char_end: u32,
    pub text_utf8: String,
    pub text_norm_ascii: String,
    pub preview: String,
}

/// The fully-extracted result for a single document, ready to be persisted
/// in one transaction by the DB writer.
#[derive(Debug, Clone)]
pub struct ExtractedDoc {
    pub path: PathBuf,
    pub size_bytes: i64,
    pub mtime_ns: i64,
    pub dev: Option<i64>,
    pub ino: Option<i64>,
    pub extractor: String,
    pub extractor_version: String,
    pub norm_version: i64,
    pub page_count: u32,
    pub status: DocStatus,
    pub error_text: Option<String>,
    pub chunks: Vec<ChunkInsert>,
}

/// Connection helper used for both reader and writer threads.
pub struct Db {
    pub conn: Connection,
}

impl Db {
    /// Open / create the SQLite database at `path` and run migrations.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("opening SQLite at {}", path.display()))?;
        conn.execute_batch(CONNECTION_PRAGMAS)
            .context("applying connection pragmas")?;
        conn.execute_batch(SCHEMA).context("applying schema")?;
        Ok(Db { conn })
    }

    /// Open a fresh reader connection — same DB file, applies only the
    /// per-connection pragmas (no migration writes).
    pub fn open_reader(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("opening SQLite at {}", path.display()))?;
        conn.execute_batch(CONNECTION_PRAGMAS)
            .context("applying connection pragmas")?;
        Ok(Db { conn })
    }

    /// Load every (path, mtime, size, version) row needed by the scanner
    /// to decide whether to re-extract.
    pub fn load_all_documents(&self) -> Result<Vec<DocumentRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT doc_id, path, size_bytes, mtime_ns, extractor, \
                    extractor_version, norm_version, status \
             FROM documents",
        )?;
        let rows = stmt.query_map([], |row| {
            let path: String = row.get(1)?;
            let status: String = row.get(7)?;
            Ok(DocumentRow {
                doc_id: row.get(0)?,
                path: PathBuf::from(path),
                size_bytes: row.get(2)?,
                mtime_ns: row.get(3)?,
                extractor: row.get(4)?,
                extractor_version: row.get(5)?,
                norm_version: row.get(6)?,
                status: DocStatus::parse(&status).unwrap_or(DocStatus::Error),
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Stream every active chunk into the supplied callback. Used at
    /// startup to populate the in-memory `BaseIndex`.
    pub fn for_each_active_chunk(
        &self,
        mut f: impl FnMut(LoadedChunkRow) -> Result<()>,
    ) -> Result<()> {
        let mut stmt = self.conn.prepare(
            "SELECT c.chunk_id, c.doc_id, d.path, d.mtime_ns, c.page_no, c.chunk_ord, \
                    c.char_start, c.char_end, c.text_utf8, c.text_norm_ascii, c.preview \
             FROM chunks c \
             JOIN documents d ON d.doc_id = c.doc_id \
             WHERE c.active = 1 AND d.status = 'ok' \
             ORDER BY c.doc_id, c.chunk_ord",
        )?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let path_str: String = row.get(2)?;
            f(LoadedChunkRow {
                chunk_id: row.get(0)?,
                doc_id: row.get(1)?,
                path: PathBuf::from(path_str),
                doc_mtime_ns: row.get(3)?,
                page_no: row.get::<_, i64>(4)? as u32,
                chunk_ord: row.get::<_, i64>(5)? as u32,
                char_start: row.get::<_, i64>(6)? as u32,
                char_end: row.get::<_, i64>(7)? as u32,
                text_utf8: row.get(8)?,
                text_norm_ascii: row.get(9)?,
                preview: row.get(10)?,
            })?;
        }
        Ok(())
    }

    /// Persist a fully-extracted document and all its chunks in one
    /// transaction. Replaces any prior chunks for the same `path`.
    pub fn upsert_extracted(&mut self, ex: &ExtractedDoc) -> Result<i64> {
        let now_ms = crate::app::now_ms();
        let tx = self.conn.transaction()?;
        // Look up existing doc_id (so chunks can be replaced cleanly).
        let existing: Option<i64> = tx
            .query_row(
                "SELECT doc_id FROM documents WHERE path = ?1",
                params![ex.path.to_string_lossy()],
                |r| r.get(0),
            )
            .optional()?;
        let doc_id = if let Some(id) = existing {
            tx.execute(
                "UPDATE documents SET size_bytes=?2, mtime_ns=?3, dev=?4, ino=?5, \
                  extractor=?6, extractor_version=?7, norm_version=?8, \
                  page_count=?9, status=?10, error_text=?11, \
                  indexed_at_ms=?12, deleted_at_ms=NULL WHERE doc_id=?1",
                params![
                    id,
                    ex.size_bytes,
                    ex.mtime_ns,
                    ex.dev,
                    ex.ino,
                    ex.extractor,
                    ex.extractor_version,
                    ex.norm_version,
                    ex.page_count as i64,
                    ex.status.as_str(),
                    ex.error_text,
                    now_ms,
                ],
            )?;
            tx.execute("DELETE FROM chunks WHERE doc_id = ?1", params![id])?;
            id
        } else {
            tx.execute(
                "INSERT INTO documents \
                  (path, size_bytes, mtime_ns, dev, ino, \
                   extractor, extractor_version, norm_version, \
                   page_count, status, error_text, indexed_at_ms) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                params![
                    ex.path.to_string_lossy(),
                    ex.size_bytes,
                    ex.mtime_ns,
                    ex.dev,
                    ex.ino,
                    ex.extractor,
                    ex.extractor_version,
                    ex.norm_version,
                    ex.page_count as i64,
                    ex.status.as_str(),
                    ex.error_text,
                    now_ms,
                ],
            )?;
            tx.last_insert_rowid()
        };

        if matches!(ex.status, DocStatus::Ok) && !ex.chunks.is_empty() {
            let mut stmt = tx.prepare(
                "INSERT INTO chunks \
                   (doc_id, page_no, chunk_ord, char_start, char_end, \
                    text_utf8, text_norm_ascii, preview, active) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1)",
            )?;
            for c in &ex.chunks {
                stmt.execute(params![
                    doc_id,
                    c.page_no as i64,
                    c.chunk_ord as i64,
                    c.char_start as i64,
                    c.char_end as i64,
                    c.text_utf8,
                    c.text_norm_ascii,
                    c.preview,
                ])?;
            }
        }
        tx.commit()?;
        Ok(doc_id)
    }

    /// Load every active chunk for a single doc, in `chunk_ord` order.
    ///
    /// Used by the live watch pipeline to convert a freshly-UPSERTed
    /// `ExtractedDoc` into the `ChunkItem` rows the overlay needs.
    /// Returns the rows together with the `doc_id` and current
    /// `mtime_ns` so the overlay can construct `ChunkItem`s with the
    /// same identity the base loader would have produced.
    pub fn load_chunks_for_doc(&self, doc_id: i64) -> Result<Vec<LoadedChunkRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.chunk_id, c.doc_id, d.path, d.mtime_ns, c.page_no, c.chunk_ord, \
                    c.char_start, c.char_end, c.text_utf8, c.text_norm_ascii, c.preview \
             FROM chunks c \
             JOIN documents d ON d.doc_id = c.doc_id \
             WHERE c.active = 1 AND c.doc_id = ?1 \
             ORDER BY c.chunk_ord",
        )?;
        let rows = stmt.query_map(params![doc_id], |row| {
            let path_str: String = row.get(2)?;
            Ok(LoadedChunkRow {
                chunk_id: row.get(0)?,
                doc_id: row.get(1)?,
                path: PathBuf::from(path_str),
                doc_mtime_ns: row.get(3)?,
                page_no: row.get::<_, i64>(4)? as u32,
                chunk_ord: row.get::<_, i64>(5)? as u32,
                char_start: row.get::<_, i64>(6)? as u32,
                char_end: row.get::<_, i64>(7)? as u32,
                text_utf8: row.get(8)?,
                text_norm_ascii: row.get(9)?,
                preview: row.get(10)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Mark a path as deleted. Also tombstones its chunks (deletes them
    /// via FK cascade is fine, but we keep them and just deactivate so an
    /// undo is easy).
    pub fn mark_deleted(&mut self, path: &Path) -> Result<Option<i64>> {
        let now = crate::app::now_ms();
        let tx = self.conn.transaction()?;
        let doc_id: Option<i64> = tx
            .query_row(
                "SELECT doc_id FROM documents WHERE path = ?1",
                params![path.to_string_lossy()],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(id) = doc_id {
            tx.execute(
                "UPDATE documents SET status='deleted', deleted_at_ms=?2 WHERE doc_id=?1",
                params![id, now],
            )?;
            tx.execute(
                "UPDATE chunks SET active=0 WHERE doc_id=?1",
                params![id],
            )?;
            tx.commit()?;
            Ok(Some(id))
        } else {
            Ok(None)
        }
    }

    /// Set a `meta` key.
    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?1,?2) \
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![key, value],
        )?;
        Ok(())
    }
}

/// Wide row used by [`Db::for_each_active_chunk`].
#[derive(Debug)]
pub struct LoadedChunkRow {
    pub chunk_id: i64,
    pub doc_id: i64,
    pub path: PathBuf,
    pub doc_mtime_ns: i64,
    pub page_no: u32,
    pub chunk_ord: u32,
    pub char_start: u32,
    pub char_end: u32,
    pub text_utf8: String,
    pub text_norm_ascii: String,
    pub preview: String,
}
