//! pdffff-gui: Win98/NT-flavoured desktop launcher for `pdffff`.
//!
//! Mirrors `src/main.rs` parameter-for-parameter; the only difference
//! is the rendering frontend at the end of the wire-up. The shared
//! kernel (`SearchWorker`, `highlight_segments`, `WatchHandle`,
//! `IndexState`) is identical to the TUI's.

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;
use tracing_subscriber::{EnvFilter, fmt};

use pdffff::app::{WatchOptions, resolve_db_path, run_watch};
use pdffff::query::{DISPLAY_LIMIT, QueryMode};
use pdffff::ui::gui::{GuiOptions, run_gui};

#[derive(Parser, Debug)]
#[command(
    name = "pdffff-gui",
    version,
    about = "Durable, fast PDF folder search — Win98/NT-flavoured desktop GUI"
)]
struct Cli {
    root: PathBuf,

    #[arg(long)]
    respect_ignore: bool,

    #[arg(long)]
    follow_symlinks: bool,

    #[arg(long)]
    jobs: Option<usize>,

    #[arg(long)]
    debounce_ms: Option<u64>,

    #[arg(long, value_enum, default_value_t = CliQueryMode::Fuzzy)]
    mode: CliQueryMode,

    #[arg(long, default_value_t = DISPLAY_LIMIT)]
    limit: usize,

    #[arg(long)]
    db: Option<PathBuf>,

    /// Tracing log file. Default: $TMPDIR/pdffff-gui.log.
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

    let gui_opts = GuiOptions {
        limit: cli.limit,
        initial_mode: cli.mode.to_query_mode(),
        root: cli.root.clone(),
    };
    // Activating a hit now opens the file in the host's PDF viewer
    // without exiting the GUI; the launcher just runs the session
    // until the user closes the window.
    run_gui(handle, gui_opts)
}

fn init_tracing_to_file(path: Option<&Path>) -> Result<()> {
    let default_path = std::env::temp_dir().join("pdffff-gui.log");
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
