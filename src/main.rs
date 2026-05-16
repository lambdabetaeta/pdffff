use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tracing_subscriber::{EnvFilter, fmt};

use pdffff::app::{IndexOptions, WatchOptions, run_index, run_rebuild, run_search, run_watch};
use pdffff::db::Db;
use pdffff::query::{DISPLAY_LIMIT, QueryMode, search};
use pdffff::scanner::Scanner;

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
    /// Walk a directory and report the diff against the database (dry run).
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
    /// Scan, extract every dirty PDF with a worker pool, and persist
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
        /// (inside the 50–250 ms band from the report).
        #[arg(long)]
        debounce_ms: Option<u64>,
    },
    /// Search the indexed corpus. Three modes are supported.
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
    },
    /// Force a rebuild of the in-memory base index from SQLite and
    /// print the resulting stats. Useful for diagnostics and for
    /// validating that an extracted corpus survives a round-trip. The
    /// long-lived `watch` mode triggers rebuilds automatically when
    /// the overlay exceeds its thresholds.
    Rebuild,
    /// Show database statistics.
    Info,
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
    fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();
    let cli = Cli::parse();
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
        Command::Search { query, mode, limit } => cmd_search(&cli.db, &query, mode, limit),
        Command::Rebuild => cmd_rebuild(&cli.db),
        Command::Info => cmd_info(&cli.db),
    }
}

fn cmd_search(db_path: &PathBuf, query: &str, mode: CliQueryMode, limit: usize) -> Result<()> {
    let hits = run_search(db_path, query, mode.to_query_mode(), limit)?;
    for hit in &hits {
        println!("{}:{}  {}", hit.path, hit.page_no, hit.snippet);
    }
    Ok(())
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

    // Read queries from stdin until EOF / blank line. Each query
    // runs against the *current* snapshot so the user can observe
    // the watcher updating the index live.
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
                }
                for hit in &hits {
                    println!("{}:{}  {}", hit.path, hit.page_no, hit.snippet);
                }
            }
            Err(err) => eprintln!("query error: {err}"),
        }
    }
    handle.stop()?;
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
    Ok(())
}
