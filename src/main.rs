use anyhow::{Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use tracing_subscriber::{EnvFilter, fmt};

use pdffff::app::{IndexOptions, run_index, run_search};
use pdffff::db::Db;
use pdffff::query::{DISPLAY_LIMIT, QueryMode};
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
    /// Search the indexed corpus for a literal query. Requires that
    /// `pdffff index` has been run first.
    Search {
        /// Query string. Normalized through the same ASCII / lowercase
        /// pipeline used at index time.
        query: String,
        /// Query engine: only `literal` works today. `regex` and
        /// `fuzzy` ship on Day 6.
        #[arg(long, value_enum, default_value_t = CliQueryMode::Literal)]
        mode: CliQueryMode,
        /// Cap on number of hits to print.
        #[arg(long, default_value_t = DISPLAY_LIMIT)]
        limit: usize,
    },
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
        Command::Search { query, mode, limit } => cmd_search(&cli.db, &query, mode, limit),
        Command::Info => cmd_info(&cli.db),
    }
}

fn cmd_search(db_path: &PathBuf, query: &str, mode: CliQueryMode, limit: usize) -> Result<()> {
    match mode {
        CliQueryMode::Literal => {}
        CliQueryMode::Regex | CliQueryMode::Fuzzy => {
            // We could let `run_search` produce this error, but giving
            // the message at the CLI boundary makes the day-by-day
            // status explicit to users.
            bail!("--mode {mode:?} is not implemented yet (planned for day 6)");
        }
    }
    let hits = run_search(db_path, query, mode.to_query_mode(), limit)?;
    for hit in &hits {
        println!("{}:{}  {}", hit.path, hit.page_no, hit.snippet);
    }
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
