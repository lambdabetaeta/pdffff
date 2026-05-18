//! pdffff binary: a pure-TUI launcher.
//!
//! The entry point is a single command:
//!
//! ```text
//! pdffff <ROOT> [flags]
//! ```
//!
//! On launch the binary:
//!
//! 1. Resolves where to put the SQLite DB. By default this is
//!    `<data_dir>/pdffff/<corpus-basename>-<hash>.db` so each corpus
//!    gets an isolated, durable store. Pass `--db <path>` to override.
//! 2. Redirects `tracing` to a log file (the TUI takes over stderr).
//! 3. Calls [`pdffff::app::run_watch`], which spawns the long-lived
//!    indexer threads but **does not block** on the initial scan —
//!    the coordinator streams dirty PDFs into the live index in the
//!    background as `pdftotext` extractions finish.
//! 4. Hands the resulting `WatchHandle` to [`pdffff::tui::run_tui`].
//!    Enter on a selected result hands the path to the host's PDF
//!    viewer via [`pdffff::ui::launch::open_in_system_viewer`] while
//!    the TUI keeps running; the launcher returns only when the user
//!    explicitly quits.

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;
use tracing_subscriber::{EnvFilter, fmt};

use pdffff::app::{WatchOptions, resolve_db_path, run_watch};
use pdffff::query::{DISPLAY_LIMIT, QueryMode};
use pdffff::tui::{TuiOptions, run_tui};

#[derive(Parser, Debug)]
#[command(
    name = "pdffff",
    version,
    about = "Durable, fast PDF folder search — pure-TUI"
)]
struct Cli {
    /// Directory to watch and index. Searched recursively.
    root: PathBuf,

    /// Respect .gitignore / .ignore files when walking `root`.
    #[arg(long)]
    respect_ignore: bool,

    /// Follow symlinks during the filesystem walk.
    #[arg(long)]
    follow_symlinks: bool,

    /// Override extractor pool size. Default: min(num_cpus, 6).
    #[arg(long)]
    jobs: Option<usize>,

    /// Watcher debounce window in milliseconds. Default: 200 ms.
    #[arg(long)]
    debounce_ms: Option<u64>,

    /// Initial query mode. Tab cycles literal → regex → fuzzy in the TUI.
    #[arg(long, value_enum, default_value_t = CliQueryMode::Fuzzy)]
    mode: CliQueryMode,

    /// Cap on hits surfaced per query.
    #[arg(long, default_value_t = DISPLAY_LIMIT)]
    limit: usize,

    /// Override the SQLite DB path. Default:
    /// $XDG_DATA_HOME/pdffff/<basename>-<hash>.db (per-corpus,
    /// platform-aware).
    #[arg(long)]
    db: Option<PathBuf>,

    /// Tracing log file. The TUI takes over stderr, so logs go here.
    /// Default: $TMPDIR/pdffff.log.
    #[arg(long)]
    log_file: Option<PathBuf>,
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

    // pdffff always runs the TUI now, which takes over the terminal —
    // route tracing to a file unconditionally.
    init_tracing_to_file(cli.log_file.as_deref())?;

    let db_path = match cli.db {
        Some(p) => p,
        None => resolve_db_path(&cli.root)?,
    };

    let opts = WatchOptions {
        respect_gitignore: cli.respect_ignore,
        follow_symlinks: cli.follow_symlinks,
        jobs: cli.jobs,
        require_pdftotext: true,
        debounce: cli.debounce_ms.map(Duration::from_millis),
    };
    let handle = run_watch(&db_path, &cli.root, &opts)?;

    let tui_opts = TuiOptions {
        limit: cli.limit,
        initial_mode: cli.mode.to_query_mode(),
        root: cli.root.clone(),
    };
    // Enter on a hit now opens the file in the host's PDF viewer
    // without exiting the TUI; the launcher just runs the session
    // until the user quits via Esc / Ctrl+C/D/Q.
    run_tui(handle, tui_opts)
}

/// Tracing subscriber that writes to `path` (or `$TMPDIR/pdffff.log`
/// by default). Keeps the TUI's alternate screen clean. The file is
/// opened in append mode and shared across all spans via a `Mutex`.
fn init_tracing_to_file(path: Option<&Path>) -> Result<()> {
    let default_path = std::env::temp_dir().join("pdffff.log");
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
