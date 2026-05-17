//! Per-corpus SQLite-DB path resolution.
//!
//! Convention: `<data_dir>/pdffff/<basename>-<8-hex of blake3(canonical)>.db`,
//! where `data_dir` is `dirs::data_dir()` (`$XDG_DATA_HOME` on Linux,
//! `~/Library/Application Support` on macOS, `%APPDATA%` on Windows).
//! The basename gives the file a human-readable hint of which corpus
//! it backs; the hash disambiguates two folders that happen to share a
//! basename.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Resolve where pdffff stores the SQLite DB for a given corpus root.
///
/// Side-effect: creates the parent directory if it doesn't exist, so
/// the caller can hand the returned path directly to `rusqlite`.
pub fn resolve_db_path(root: &Path) -> Result<PathBuf> {
    let canonical = root
        .canonicalize()
        .with_context(|| format!("canonicalising corpus root {}", root.display()))?;
    let basename = canonical
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        // Falls back when root is `/` — vanishingly rare but possible.
        .unwrap_or_else(|| "root".to_string());
    let hash = blake3::hash(canonical.as_os_str().as_encoded_bytes());
    let short = hex8(hash.as_bytes());
    let mut dir = dirs::data_dir().context(
        "could not determine the user's data directory (XDG_DATA_HOME / equivalent)",
    )?;
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

/// Hex-encode the first 8 bytes of `bytes` (16 hex chars).
fn hex8(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(16);
    for &b in bytes.iter().take(8) {
        use std::fmt::Write;
        let _ = write!(out, "{b:02x}");
    }
    out
}
