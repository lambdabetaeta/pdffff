use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing_subscriber::{EnvFilter, fmt};

use pdffff::db::Db;
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
    /// Walk a directory and report the diff against the database.
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
    /// Show database statistics.
    Info,
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
        Command::Info => cmd_info(&cli.db),
    }
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
