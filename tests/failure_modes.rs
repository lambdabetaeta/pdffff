//! Day-7 failure-mode tests.
//!
//! Each test pushes the indexer into a known-bad state (corrupted PDF,
//! image-only PDF, missing pdftotext, read-only DB) and asserts that
//! the pipeline degrades cleanly rather than panicking. The indexer's
//! contract is: a single broken file must not stop the rest of the
//! corpus from being indexed, and unrecoverable failures must surface
//! a real `anyhow::Error` instead of an out-of-band panic.

mod common;

use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

use anyhow::Result;
use tempfile::tempdir;

use pdffff::app::{IndexOptions, run_index};
use pdffff::db::{Db, DocStatus};
use pdffff::extract::ensure_pdftotext_available;

fn require_pdftotext_or_skip(test_name: &str) -> bool {
    if let Err(err) = ensure_pdftotext_available() {
        eprintln!("[{test_name}] skipping: {err}");
        return false;
    }
    true
}

fn opts() -> IndexOptions {
    IndexOptions {
        respect_gitignore: false,
        follow_symlinks: false,
        jobs: Some(2),
        require_pdftotext: true,
    }
}

/// Random bytes with a `%PDF-1.4` header so the file is *recognised* as
/// a PDF by the scanner but rejected by pdftotext. pdftotext exits
/// nonzero ⇒ we expect `status='error'` with non-empty `error_text`
/// and zero chunks.
#[test]
fn corrupted_pdf_lands_as_error_and_indexer_continues() -> Result<()> {
    if !require_pdftotext_or_skip("corrupted_pdf_lands_as_error_and_indexer_continues") {
        return Ok(());
    }

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    fs::create_dir_all(&root)?;

    // One valid PDF + one corrupted PDF: the valid one MUST still be
    // extracted to its full chunk set even though the corrupted one
    // failed.
    let good = root.join("good.pdf");
    let bad = root.join("bad.pdf");
    common::make_pdf_two_pages(&good, &["alpha bravo charlie"], &["delta echo foxtrot"]);

    // Random-looking bytes with a PDF banner.
    let mut bytes: Vec<u8> = b"%PDF-1.4\n".to_vec();
    for i in 0..1024u32 {
        bytes.push((i * 31 + 7) as u8);
    }
    fs::write(&bad, &bytes)?;

    let db_path = tmp.path().join("idx.db");
    let stats = run_index(&db_path, &root, &opts())?;

    // The good PDF makes it through; the bad PDF surfaces as an
    // `Error` row. The indexer reports both outcomes via counters.
    assert_eq!(stats.seen, 2);
    assert_eq!(stats.dirty, 2);
    assert_eq!(stats.ok, 1, "good PDF must extract");
    assert_eq!(stats.error, 1, "bad PDF must land as error");
    assert_eq!(stats.empty, 0);

    let db = Db::open(&db_path)?;
    let docs = db.load_all_documents()?;
    let by_path: std::collections::HashMap<_, _> =
        docs.iter().map(|d| (d.path.clone(), d)).collect();
    let bad_row = by_path.get(&bad).expect("bad.pdf was scanned");
    assert_eq!(bad_row.status, DocStatus::Error);

    // The `error_text` column is non-NULL and non-empty for the bad
    // row.
    let error_text: Option<String> = db.conn.query_row(
        "SELECT error_text FROM documents WHERE path = ?1",
        rusqlite::params![bad.to_string_lossy()],
        |r| r.get(0),
    )?;
    let error_text = error_text.expect("error_text must be set for status=error");
    assert!(
        !error_text.trim().is_empty(),
        "error_text must be informative; got {error_text:?}",
    );

    // The good PDF has its chunks intact.
    let n_chunks: i64 = db.conn.query_row(
        "SELECT COUNT(*) FROM chunks c \
         JOIN documents d ON d.doc_id = c.doc_id \
         WHERE d.path = ?1",
        rusqlite::params![good.to_string_lossy()],
        |r| r.get(0),
    )?;
    assert!(n_chunks >= 2, "good PDF should still have its chunks; got {n_chunks}");

    // The bad PDF has zero chunks (the writer never UPSERTs chunks for
    // a status=error row).
    let bad_chunks: i64 = db.conn.query_row(
        "SELECT COUNT(*) FROM chunks c \
         JOIN documents d ON d.doc_id = c.doc_id \
         WHERE d.path = ?1",
        rusqlite::params![bad.to_string_lossy()],
        |r| r.get(0),
    )?;
    assert_eq!(bad_chunks, 0);
    Ok(())
}

/// Image-only PDF ⇒ pdftotext emits empty stdout ⇒ `status='empty'`,
/// zero chunks, indexer reports the empty count.
#[test]
fn image_only_pdf_lands_as_empty_and_indexer_continues() -> Result<()> {
    if !require_pdftotext_or_skip("image_only_pdf_lands_as_empty_and_indexer_continues") {
        return Ok(());
    }

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    fs::create_dir_all(&root)?;
    let good = root.join("good.pdf");
    let empty = root.join("empty.pdf");
    common::make_pdf_two_pages(&good, &["alpha bravo"], &["charlie delta"]);
    common::make_pdf_no_text(&empty);

    let db_path = tmp.path().join("idx.db");
    let stats = run_index(&db_path, &root, &opts())?;
    assert_eq!(stats.seen, 2);
    assert_eq!(stats.dirty, 2);
    assert_eq!(stats.ok, 1);
    assert_eq!(stats.empty, 1);
    assert_eq!(stats.error, 0);

    let db = Db::open(&db_path)?;
    let docs = db.load_all_documents()?;
    let by_path: std::collections::HashMap<_, _> =
        docs.iter().map(|d| (d.path.clone(), d)).collect();
    assert_eq!(by_path[&empty].status, DocStatus::Empty);
    assert_eq!(by_path[&good].status, DocStatus::Ok);

    let empty_chunks: i64 = db.conn.query_row(
        "SELECT COUNT(*) FROM chunks c \
         JOIN documents d ON d.doc_id = c.doc_id \
         WHERE d.path = ?1",
        rusqlite::params![empty.to_string_lossy()],
        |r| r.get(0),
    )?;
    assert_eq!(empty_chunks, 0);
    Ok(())
}

/// pdftotext missing on PATH ⇒ `pdffff index` exits with a non-zero
/// status. Done by spawning the actual binary with an empty PATH so we
/// don't have to mutate the test process's environment.
#[test]
fn missing_pdftotext_surfaces_a_clean_error() -> Result<()> {
    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    fs::create_dir_all(&root)?;
    fs::write(root.join("dummy.pdf"), b"%PDF-1.4\n")?;
    let db_path = tmp.path().join("idx.db");

    let exe = env!("CARGO_BIN_EXE_pdffff");
    let output = Command::new(exe)
        .arg("--db")
        .arg(&db_path)
        .arg("index")
        .arg(&root)
        .env_clear()
        .env("PATH", "/nonexistent")
        .output()?;

    assert!(
        !output.status.success(),
        "pdffff must fail when pdftotext is missing on PATH; got success",
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.to_lowercase().contains("pdftotext"),
        "stderr should explain that pdftotext is missing; got {stderr:?}",
    );
    Ok(())
}

/// Unwritable SQLite ⇒ the indexer surfaces a clean error rather than
/// panicking. We create + populate a fresh DB, then chmod it 0o444 to
/// remove write access, then try to run a second index pass which has
/// to UPSERT new rows.
#[cfg(unix)]
#[test]
fn unwritable_sqlite_returns_clean_error() -> Result<()> {
    if !require_pdftotext_or_skip("unwritable_sqlite_returns_clean_error") {
        return Ok(());
    }
    use std::os::unix::fs::PermissionsExt;

    let tmp = tempdir()?;
    let root = tmp.path().join("docs");
    fs::create_dir_all(&root)?;
    let a = root.join("a.pdf");
    common::make_pdf_two_pages(&a, &["alpha bravo"], &["charlie delta"]);

    let db_path = tmp.path().join("idx.db");
    let _ = run_index(&db_path, &root, &opts())?;

    // Remove the SQLite WAL / shm files first so SQLite has to recreate
    // them (which requires write to the parent directory but not to
    // the main file — to actually fail we have to make the *directory*
    // read-only since SQLite always recreates wal/shm sidecars).
    fs::set_permissions(&db_path, fs::Permissions::from_mode(0o444))?;
    // Also chmod the parent dir read-only so SQLite can't create
    // wal/shm sidecars.
    fs::set_permissions(tmp.path(), fs::Permissions::from_mode(0o555))?;

    // Modify the corpus so a second pass has work to do.
    let b = root.join("b.pdf");
    // The parent dir is now read-only; this write fails. Skip the
    // assertion path if so — we only care about exercising the DB-write
    // failure mode.
    if fs::write(&b, b"%PDF-1.4\n").is_err() {
        // Restore perms before bailing so tempdir cleanup works.
        fs::set_permissions(tmp.path(), fs::Permissions::from_mode(0o755))?;
        fs::set_permissions(&db_path, fs::Permissions::from_mode(0o644))?;
        eprintln!("[unwritable_sqlite_returns_clean_error] could not stage second corpus, skipping");
        return Ok(());
    }

    let result = run_index(&db_path, &root, &opts());
    // Restore permissions so the tempdir Drop can clean up.
    fs::set_permissions(tmp.path(), fs::Permissions::from_mode(0o755))?;
    fs::set_permissions(&db_path, fs::Permissions::from_mode(0o644))?;

    assert!(
        result.is_err(),
        "writing to a read-only SQLite file must return an error, not a panic",
    );
    let err_msg = format!("{:#}", result.unwrap_err());
    assert!(
        err_msg.to_lowercase().contains("read") || err_msg.to_lowercase().contains("permission")
            || err_msg.to_lowercase().contains("readonly")
            || err_msg.to_lowercase().contains("sqlite"),
        "error must mention the underlying SQLite / permission cause: {err_msg}",
    );
    Ok(())
}

/// Sanity: `ensure_pdftotext_available` rejects an empty PATH up-front.
/// Done via a subprocess so the test doesn't leak `env::set_var` into
/// other parallel tests.
#[test]
fn ensure_pdftotext_available_rejects_empty_path() -> Result<()> {
    // Use the binary as a shim: `pdffff diagnose` calls
    // `ensure_pdftotext_available`. With an empty PATH it should print
    // a "MISSING" line.
    let exe = env!("CARGO_BIN_EXE_pdffff");
    let tmp = tempdir()?;
    let db_path = tmp.path().join("idx.db");
    let output = Command::new(exe)
        .arg("--db")
        .arg(&db_path)
        .arg("diagnose")
        .env_clear()
        .env("PATH", "/nonexistent")
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.to_lowercase().contains("missing"),
        "diagnose must report pdftotext missing on empty PATH; got stdout {stdout:?}",
    );
    // Wait for output buffer flush; we don't care about exit code (the
    // `diagnose` subcommand prints "MISSING" but continues to inspect
    // SQLite, which is intentional behaviour for a diagnostic tool).
    drop(Duration::from_secs(0));
    let _ = Path::new("");
    Ok(())
}
