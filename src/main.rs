//! pdffff command-line entry point.
//!
//! Every subcommand is a thin wrapper around a function in
//! [`pdffff::app`] so the same building blocks can be exercised from
//! tests, benchmarks, and the binary. Result printing lives here (and
//! only here): the library never writes to stdout itself.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use owo_colors::OwoColorize;
use std::fs::OpenOptions;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing_subscriber::{EnvFilter, fmt};

use pdffff::app::{IndexOptions, WatchOptions, run_index, run_rebuild, run_search, run_watch};
use pdffff::db::Db;
use pdffff::extract::{ensure_pdftotext_available, extractor_version_or_missing};
use pdffff::query::{DISPLAY_LIMIT, Hit, QueryMode, search};
use pdffff::scanner::Scanner;
use pdffff::tui::{TuiOptions, run_tui};

#[derive(Parser, Debug)]
#[command(name = "pdffff", version, about = "Durable, fast PDF folder search")]
struct Cli {
    /// SQLite database file. Default: ./pdffff.db
    #[arg(long, global = true, default_value = "pdffff.db")]
    db: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Dry-run scanner: walks a directory and reports what would be indexed.
    ///
    /// Useful to check `.gitignore` interactions or symlink handling
    /// before committing to a long extraction run.
    Scan {
        /// Directory to walk recursively.
        root: PathBuf,
        /// Respect .gitignore / .ignore files.
        #[arg(long)]
        respect_ignore: bool,
        /// Follow symlinks.
        #[arg(long)]
        follow_symlinks: bool,
    },
    /// Index a directory of PDFs into SQLite (one-shot).
    ///
    /// Scans, extracts every dirty PDF with a worker pool, and persists
    /// chunks to SQLite. Tombstones any paths that disappeared from disk.
    Index {
        /// Directory to walk recursively.
        root: PathBuf,
        /// Respect .gitignore / .ignore files.
        #[arg(long)]
        respect_ignore: bool,
        /// Follow symlinks.
        #[arg(long)]
        follow_symlinks: bool,
        /// Override extractor pool size. Default: min(num_cpus, 6).
        #[arg(long)]
        jobs: Option<usize>,
    },
    /// Watch a folder for changes and answer interactive queries.
    ///
    /// Long-lived watch mode: synchronous scan + extract pass to
    /// converge with disk, then a notify-based watcher that keeps the
    /// in-memory index live. Type queries on stdin and the process
    /// answers them against the current snapshot. Press Ctrl-C (or
    /// close stdin) to exit cleanly.
    Watch {
        /// Directory to watch recursively.
        root: PathBuf,
        /// Respect .gitignore / .ignore files.
        #[arg(long)]
        respect_ignore: bool,
        /// Follow symlinks.
        #[arg(long)]
        follow_symlinks: bool,
        /// Override extractor pool size. Default: min(num_cpus, 6).
        #[arg(long)]
        jobs: Option<usize>,
        /// Watcher debounce window in milliseconds. Default: 200 ms
        /// (inside the 50-250 ms band from the report).
        #[arg(long)]
        debounce_ms: Option<u64>,
    },
    /// Interactive fzf-style TUI: watch a folder and search live.
    ///
    /// Runs the same scan + watch + extractor pool that `watch` does,
    /// but renders a full-screen terminal UI for typing queries and
    /// browsing hits. The indexer keeps running in the background so
    /// the displayed results reflect the current on-disk state. Press
    /// Ctrl+C, Esc, Ctrl+D, or Ctrl+Q to quit cleanly — the watcher
    /// stops, the writer drains its queue, and the SQLite database
    /// is left in a consistent state on every commit boundary.
    Tui {
        /// Directory to watch recursively.
        root: PathBuf,
        /// Respect .gitignore / .ignore files.
        #[arg(long)]
        respect_ignore: bool,
        /// Follow symlinks.
        #[arg(long)]
        follow_symlinks: bool,
        /// Override extractor pool size. Default: min(num_cpus, 6).
        #[arg(long)]
        jobs: Option<usize>,
        /// Watcher debounce window in milliseconds. Default: 200 ms.
        #[arg(long)]
        debounce_ms: Option<u64>,
        /// Initial query mode. Tab cycles literal → regex → fuzzy
        /// while the UI is running.
        #[arg(long, value_enum, default_value_t = CliQueryMode::Literal)]
        mode: CliQueryMode,
        /// Cap on number of hits surfaced per query.
        #[arg(long, default_value_t = DISPLAY_LIMIT)]
        limit: usize,
        /// Tracing log file. When the TUI is active stderr is taken
        /// over by the alternate screen, so logs go here instead.
        /// Default: $TMPDIR/pdffff-tui.log.
        #[arg(long)]
        log_file: Option<PathBuf>,
    },
    /// Search the indexed corpus.
    ///
    /// Three modes are supported: literal (the default), regex, and
    /// fuzzy.  Results are printed one per blank-separated paragraph
    /// with path, page, chunk number, score, and a match-centred
    /// snippet — or, with `--json`, one compact JSON object per line.
    Search {
        /// Query string. For literal/fuzzy the query is normalized
        /// through the same ASCII / lowercase pipeline used at index
        /// time. For regex the pattern is passed through verbatim —
        /// case-insensitivity is handled by the regex engine so that
        /// the bigram prefilter's lowercase-only assumption stays
        /// consistent with what the verifier sees.
        query: String,
        /// Query engine.
        #[arg(long, value_enum, default_value_t = CliQueryMode::Literal)]
        mode: CliQueryMode,
        /// Cap on number of hits to print.
        #[arg(long, default_value_t = DISPLAY_LIMIT)]
        limit: usize,
        /// Emit one compact JSON object per hit on stdout, one per
        /// line. Suitable for piping to `jq` / xargs / other scripts.
        #[arg(long)]
        json: bool,
    },
    /// Force a base-index rebuild from SQLite and report stats.
    ///
    /// Useful for diagnostics and for validating that an extracted
    /// corpus survives a round-trip. The long-lived `watch` mode
    /// triggers rebuilds automatically when the overlay exceeds its
    /// thresholds.
    Rebuild,
    /// Print database statistics (one-shot).
    ///
    /// Reports total documents broken down by status, total active
    /// chunks, and the approximate memory cost of the in-memory bigram
    /// prefilter that would be built at startup.
    Info,
    /// Diagnose the install / database / corpus end-to-end.
    ///
    /// Verifies `pdftotext` is available and reports its version,
    /// asks SQLite for an integrity check, summarises document
    /// statuses, and lists up to 20 documents currently in
    /// `status='error'` along with their stored `error_text`.
    Diagnose,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum CliQueryMode {
    Literal,
    Regex,
    Fuzzy,
}

impl CliQueryMode {
    fn to_query_mode(self) -> QueryMode {
        match self {
            CliQueryMode::Literal => QueryMode::Literal,
            CliQueryMode::Regex => QueryMode::Regex,
            CliQueryMode::Fuzzy => QueryMode::Fuzzy,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    // The TUI takes over stderr (alternate screen + raw mode), so we
    // route tracing to a file for that path and to stderr for every
    // other subcommand.
    match &cli.command {
        Command::Tui { log_file, .. } => init_tracing_to_file(log_file.as_deref())?,
        _ => init_tracing_to_stderr(),
    }
    match cli.command {
        Command::Scan {
            root,
            respect_ignore,
            follow_symlinks,
        } => cmd_scan(&cli.db, &root, respect_ignore, follow_symlinks),
        Command::Index {
            root,
            respect_ignore,
            follow_symlinks,
            jobs,
        } => cmd_index(&cli.db, &root, respect_ignore, follow_symlinks, jobs),
        Command::Watch {
            root,
            respect_ignore,
            follow_symlinks,
            jobs,
            debounce_ms,
        } => cmd_watch(&cli.db, &root, respect_ignore, follow_symlinks, jobs, debounce_ms),
        Command::Tui {
            root,
            respect_ignore,
            follow_symlinks,
            jobs,
            debounce_ms,
            mode,
            limit,
            log_file: _,
        } => cmd_tui(
            &cli.db,
            &root,
            respect_ignore,
            follow_symlinks,
            jobs,
            debounce_ms,
            mode,
            limit,
        ),
        Command::Search {
            query,
            mode,
            limit,
            json,
        } => cmd_search(&cli.db, &query, mode, limit, json),
        Command::Rebuild => cmd_rebuild(&cli.db),
        Command::Info => cmd_info(&cli.db),
        Command::Diagnose => cmd_diagnose(&cli.db),
    }
}

fn init_tracing_to_stderr() {
    fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();
}

/// Tracing subscriber that writes to `path` (or `$TMPDIR/pdffff-tui.log`
/// by default). Keeps the TUI's alternate screen clean. The file is
/// opened in append mode and shared across all spans via a `Mutex`.
fn init_tracing_to_file(path: Option<&Path>) -> Result<()> {
    let default_path = std::env::temp_dir().join("pdffff-tui.log");
    let path = path.unwrap_or(&default_path);
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening tracing log file {}", path.display()))?;
    let writer = Mutex::new(file);
    fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with_writer(writer)
        .with_ansi(false)
        .init();
    Ok(())
}

/// True iff stdout is a real terminal and `NO_COLOR` is not set.
fn use_color() -> bool {
    if std::env::var_os("NO_COLOR").is_some() {
        return false;
    }
    std::io::stdout().is_terminal()
}

fn cmd_search(
    db_path: &PathBuf,
    query: &str,
    mode: CliQueryMode,
    limit: usize,
    json: bool,
) -> Result<()> {
    let hits = run_search(db_path, query, mode.to_query_mode(), limit)?;
    if json {
        let mut stdout = std::io::stdout().lock();
        for hit in &hits {
            serde_json::to_writer(&mut stdout, hit)
                .context("writing JSON hit to stdout")?;
            std::io::Write::write_all(&mut stdout, b"\n")?;
        }
    } else {
        print_hits_human(&hits, query, use_color());
    }
    Ok(())
}

/// Render hits in the human-readable format described in Day 7's brief:
///
/// ```text
/// {n}. {path} (page {page_no}, chunk #{chunk_ord}, score {score:.2})
///      {snippet}
/// ```
///
/// When `colored` is true, the filename is bold and the page-number
/// metadata is dim; inside the snippet the normalised query phrase
/// (and each of its whitespace-split terms) is rendered with inverse
/// video so the matched substring stands out.
fn print_hits_human(hits: &[Hit], query: &str, colored: bool) {
    for (i, hit) in hits.iter().enumerate() {
        let header_path: String = if colored {
            hit.path.bold().to_string()
        } else {
            hit.path.clone()
        };
        let metadata = format!(
            "(page {}, chunk #{}, score {:.2})",
            hit.page_no, hit.chunk_ord, hit.score,
        );
        let metadata: String = if colored {
            metadata.dimmed().to_string()
        } else {
            metadata
        };
        println!("{}. {} {}", i + 1, header_path, metadata);
        let snippet_line = if colored {
            highlight_snippet(&hit.snippet, query)
        } else {
            hit.snippet.clone()
        };
        println!("     {snippet_line}");
        println!();
    }
}

/// Wrap occurrences of `query` (or its whitespace-split terms) in
/// `owo_colors` reverse-video escapes. Case-insensitive; non-overlapping
/// from left to right; the longest token wins at each starting position
/// (so the full phrase, if present, takes precedence over a single term).
fn highlight_snippet(snippet: &str, query: &str) -> String {
    let phrase = query.trim().to_lowercase();
    let mut needles: Vec<String> = if phrase.is_empty() {
        Vec::new()
    } else {
        let mut v = vec![phrase.clone()];
        v.extend(
            phrase
                .split_whitespace()
                .filter(|t| *t != phrase)
                .map(|t| t.to_lowercase()),
        );
        v
    };
    // Longest first so the phrase wins over its constituent terms.
    needles.sort_by_key(|s| std::cmp::Reverse(s.len()));
    needles.dedup();
    if needles.is_empty() {
        return snippet.to_string();
    }
    let snippet_lc = snippet.to_lowercase();
    let bytes = snippet.as_bytes();
    let lc_bytes = snippet_lc.as_bytes();
    let mut out = String::with_capacity(snippet.len() + 16);
    let mut cursor = 0;
    while cursor < bytes.len() {
        // Skip past any byte boundary inside a multi-byte char.
        if !snippet.is_char_boundary(cursor) {
            out.push(snippet[cursor..].chars().next().unwrap());
            cursor += snippet[cursor..].chars().next().unwrap().len_utf8();
            continue;
        }
        let mut matched: Option<&str> = None;
        for n in &needles {
            if cursor + n.len() <= bytes.len() && &lc_bytes[cursor..cursor + n.len()] == n.as_bytes()
            {
                // Must still be a char boundary on both sides.
                if snippet.is_char_boundary(cursor + n.len()) {
                    matched = Some(n);
                    break;
                }
            }
        }
        if let Some(n) = matched {
            let end = cursor + n.len();
            let original = &snippet[cursor..end];
            out.push_str(&original.reversed().to_string());
            cursor = end;
        } else {
            let ch = snippet[cursor..].chars().next().unwrap();
            out.push(ch);
            cursor += ch.len_utf8();
        }
    }
    out
}

fn cmd_rebuild(db_path: &PathBuf) -> Result<()> {
    let stats = run_rebuild(db_path)?;
    println!(
        "rebuild: docs={} chunks={} bigram_bytes={} elapsed={:.2}s",
        stats.docs, stats.chunks, stats.bigram_heap_bytes, stats.elapsed_secs,
    );
    Ok(())
}

fn cmd_scan(db_path: &PathBuf, root: &PathBuf, respect_ignore: bool, follow_symlinks: bool) -> Result<()> {
    let db = Db::open(db_path)?;
    let mut scanner = Scanner::new(root);
    scanner.respect_gitignore = respect_ignore;
    scanner.follow_symlinks = follow_symlinks;
    let result = scanner.walk_and_diff(&db)?;
    println!(
        "scanned {} files; {} need extraction, {} disappeared",
        result.seen_count,
        result.jobs.len(),
        result.deleted.len(),
    );
    for job in &result.jobs {
        println!("  [{:?}] {}", job.reason, job.path.display());
    }
    for path in &result.deleted {
        println!("  [DELETED] {}", path.display());
    }
    Ok(())
}

fn cmd_index(
    db_path: &PathBuf,
    root: &PathBuf,
    respect_ignore: bool,
    follow_symlinks: bool,
    jobs: Option<usize>,
) -> Result<()> {
    let opts = IndexOptions {
        respect_gitignore: respect_ignore,
        follow_symlinks,
        jobs,
        require_pdftotext: true,
    };
    let stats = run_index(db_path, root, &opts)?;
    println!(
        "indexed: seen={} dirty={} ok={} empty={} error={} deleted={} elapsed={:.2}s",
        stats.seen,
        stats.dirty,
        stats.ok,
        stats.empty,
        stats.error,
        stats.deleted,
        stats.elapsed_secs,
    );
    Ok(())
}

fn cmd_watch(
    db_path: &PathBuf,
    root: &PathBuf,
    respect_ignore: bool,
    follow_symlinks: bool,
    jobs: Option<usize>,
    debounce_ms: Option<u64>,
) -> Result<()> {
    let opts = WatchOptions {
        respect_gitignore: respect_ignore,
        follow_symlinks,
        jobs,
        require_pdftotext: true,
        debounce: debounce_ms.map(Duration::from_millis),
    };
    let handle = run_watch(db_path, root, &opts)?;
    println!(
        "watching {} (debounce {} ms). type a literal query and press enter; empty line or EOF to quit.",
        root.display(),
        debounce_ms.unwrap_or(200),
    );
    let state: Arc<pdffff::index::IndexState> = handle.state.clone();
    let colored = use_color();

    use std::io::{BufRead, Write};
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let mut line = String::new();
    loop {
        write!(stdout, "> ")?;
        stdout.flush()?;
        line.clear();
        let n = match stdin.lock().read_line(&mut line) {
            Ok(n) => n,
            Err(err) => {
                eprintln!("stdin error: {err}");
                break;
            }
        };
        if n == 0 {
            break;
        }
        let q = line.trim();
        if q.is_empty() {
            break;
        }
        match search(&state, q, QueryMode::Literal, DISPLAY_LIMIT) {
            Ok(hits) => {
                if hits.is_empty() {
                    println!("(no hits)");
                } else {
                    print_hits_human(&hits, q, colored);
                }
            }
            Err(err) => eprintln!("query error: {err}"),
        }
    }
    handle.stop()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_tui(
    db_path: &PathBuf,
    root: &PathBuf,
    respect_ignore: bool,
    follow_symlinks: bool,
    jobs: Option<usize>,
    debounce_ms: Option<u64>,
    mode: CliQueryMode,
    limit: usize,
) -> Result<()> {
    // The watch pipeline performs an initial sync before returning the
    // WatchHandle; on a fresh corpus that can take a while. We can't
    // render the TUI before the handle exists (we'd have nothing to
    // search against), so print a one-line note to the user before
    // taking over the terminal.
    eprintln!("syncing {} (initial scan + extract) …", root.display());
    let opts = WatchOptions {
        respect_gitignore: respect_ignore,
        follow_symlinks,
        jobs,
        require_pdftotext: true,
        debounce: debounce_ms.map(Duration::from_millis),
    };
    let handle = run_watch(db_path, root, &opts)?;

    let tui_opts = TuiOptions {
        limit,
        initial_mode: mode.to_query_mode(),
        root: root.clone(),
    };
    let chosen = run_tui(handle, tui_opts)?;
    if let Some(hit) = chosen {
        // After teardown we are back on the original screen; printing
        // the path to stdout here makes `pdffff tui` usable in shell
        // pipelines (e.g. piped into `xargs open`).
        println!("{}", hit.path);
    }
    Ok(())
}

fn cmd_info(db_path: &PathBuf) -> Result<()> {
    let db = Db::open(db_path)?;
    let docs = db.load_all_documents()?;
    let active = docs.iter().filter(|d| d.status == pdffff::db::DocStatus::Ok).count();
    let errors = docs.iter().filter(|d| d.status == pdffff::db::DocStatus::Error).count();
    let empty = docs.iter().filter(|d| d.status == pdffff::db::DocStatus::Empty).count();
    let deleted = docs.iter().filter(|d| d.status == pdffff::db::DocStatus::Deleted).count();
    println!("documents: {} total", docs.len());
    println!("  ok:      {active}");
    println!("  empty:   {empty}");
    println!("  error:   {errors}");
    println!("  deleted: {deleted}");

    // Load the base index purely to report chunk count and bigram heap
    // size. This is the same load `run_search` would do, so the
    // reported numbers are the ones a real query would face.
    let base = pdffff::index::load_base_index_from_db(&db)?;
    let bigram_bytes = base.bigrams.as_ref().map_or(0, |b| b.heap_bytes());
    let bigram_mib = bigram_bytes as f64 / (1024.0 * 1024.0);
    println!("chunks:    {} active", base.chunks.len());
    println!("bigram heap: {bigram_bytes} bytes ({bigram_mib:.2} MiB)");
    Ok(())
}

fn cmd_diagnose(db_path: &PathBuf) -> Result<()> {
    println!("== pdftotext ==");
    match ensure_pdftotext_available() {
        Ok(()) => {
            let v = extractor_version_or_missing();
            println!("  ok: {v}");
        }
        Err(err) => println!("  MISSING: {err}"),
    }

    println!("\n== sqlite ==");
    diagnose_db(db_path)?;
    Ok(())
}

fn diagnose_db(db_path: &Path) -> Result<()> {
    if !db_path.exists() {
        println!("  database file does not exist: {}", db_path.display());
        return Ok(());
    }
    let db = Db::open_reader(db_path)?;
    let integrity: String = db
        .conn
        .query_row("PRAGMA integrity_check", [], |r| r.get(0))
        .context("running PRAGMA integrity_check")?;
    println!("  integrity_check: {integrity}");

    let docs = db.load_all_documents()?;
    let mut counts = [0usize; 4]; // ok, empty, error, deleted
    for d in &docs {
        let i = match d.status {
            pdffff::db::DocStatus::Ok => 0,
            pdffff::db::DocStatus::Empty => 1,
            pdffff::db::DocStatus::Error => 2,
            pdffff::db::DocStatus::Deleted => 3,
        };
        counts[i] += 1;
    }
    println!("  documents: {} total", docs.len());
    println!("    ok:      {}", counts[0]);
    println!("    empty:   {}", counts[1]);
    println!("    error:   {}", counts[2]);
    println!("    deleted: {}", counts[3]);

    if counts[2] > 0 {
        println!("\n== errored documents (up to 20) ==");
        let mut stmt = db.conn.prepare(
            "SELECT path, error_text FROM documents \
             WHERE status = 'error' \
             ORDER BY indexed_at_ms DESC \
             LIMIT 20",
        )?;
        let rows = stmt.query_map([], |r| {
            let path: String = r.get(0)?;
            let err: Option<String> = r.get(1)?;
            Ok((path, err))
        })?;
        for row in rows {
            let (path, err) = row?;
            println!(
                "  {} :: {}",
                path,
                err.unwrap_or_else(|| "(no error_text)".to_string()),
            );
        }
    }
    Ok(())
}
