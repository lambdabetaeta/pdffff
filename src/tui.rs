//! Interactive TUI for `pdffff`.
//!
//! Visual layout (fzf-inspired, but tailored to the
//! query-against-a-live-index workflow):
//!
//! ```text
//!  pdffff  /Users/foo/papers  •  123 ok / 0 err  •  ⠿ indexing 3
//!  ──────────────────────────────────────────────────────────────
//!  > query|                                              [literal]
//!  ──────────────────────────────────────────────────────────────
//!  ▶ 1. paper.pdf (page 12, chunk #3, score 1.00)
//!        …matching snippet excerpt…
//!    2. other.pdf (page  1, chunk #0, score 0.95)
//!        …another matching snippet…
//!    …
//!  ──────────────────────────────────────────────────────────────
//!  ↑/↓ select   Tab mode   Enter print   Ctrl+C quit
//! ```
//!
//! The TUI owns a [`WatchHandle`] from [`crate::app::run_watch`]: the
//! handle's background threads keep the index live while the UI runs
//! queries against the same `IndexState`. The writer thread persists
//! every successful mutation to SQLite synchronously, so the only
//! work shutdown has to do is signal the threads to drain and join.
//!
//! Shutdown
//! --------
//! All four exit keys (`Ctrl+C`, `Ctrl+D`, `Ctrl+Q`, `Esc`) take the
//! same path:
//!
//! 1. Leave the alternate screen + disable raw mode (so the terminal
//!    is back to the user before any slow shutdown work).
//! 2. Call [`WatchHandle::stop`], which signals the coordinator and
//!    writer threads and joins them. The writer drains its `flume`
//!    channel before exiting, so any in-flight extraction whose
//!    result already reached the writer is durably persisted.
//! 3. Return.
//!
//! Because the writer commits every mutation as its own SQLite
//! transaction (`WAL` mode, `synchronous=NORMAL`), there is no
//! buffered state to flush on quit — the durability story is the
//! same whether the process exits cleanly or is killed.
//!
//! The TUI also installs a panic hook that restores the terminal
//! before the panic propagates; otherwise a panic inside the render
//! loop would leave the user's terminal in raw mode.
//!
//! Tracing
//! -------
//! The TUI does not write tracing output itself. The CLI redirects
//! the `tracing` subscriber to a log file before entering the TUI so
//! that index-progress logs do not corrupt the alternate screen.

use anyhow::{Context, Result};
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent,
        KeyEventKind, KeyModifiers,
    },
    execute,
    terminal::{
        EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
    },
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
};
use std::io::{self, Stdout};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use crate::app::{IndexProgress, WatchHandle};
use crate::index::IndexState;
use crate::query::{DISPLAY_LIMIT, Hit, QueryMode, search};

/// How often (at most) we redraw the screen when no key is pressed.
/// Drives the indexing-status spinner and the elapsed-time counter.
const TICK: Duration = Duration::from_millis(100);

/// Spinner frames cycled at every [`TICK`].
const SPINNER_FRAMES: [&str; 8] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];

/// Knobs for [`run_tui`]. The defaults are sensible for an interactive
/// session against a small personal PDF library.
#[derive(Debug, Clone)]
pub struct TuiOptions {
    /// Cap on hits surfaced per query. Defaults to [`DISPLAY_LIMIT`].
    pub limit: usize,
    /// Initial query mode. Tab cycles through Literal → Regex → Fuzzy.
    pub initial_mode: QueryMode,
    /// Root being watched; rendered in the status bar so the user
    /// remembers what they're searching.
    pub root: PathBuf,
}

impl Default for TuiOptions {
    fn default() -> Self {
        Self {
            limit: DISPLAY_LIMIT,
            initial_mode: QueryMode::Literal,
            root: PathBuf::new(),
        }
    }
}

/// Run the TUI until the user quits or an unrecoverable IO error
/// occurs. Owns `handle`: calls [`WatchHandle::stop`] on the way out
/// so the index threads always shut down cleanly, even on panic.
///
/// Returns `Some(hit)` when the user pressed Enter on a result so the
/// caller can print the chosen path after the terminal has been
/// restored; `None` on plain quit.
pub fn run_tui(handle: WatchHandle, opts: TuiOptions) -> Result<Option<Hit>> {
    let mut terminal = setup_terminal().context("entering TUI terminal mode")?;
    install_panic_hook();

    let loop_result = main_loop(&mut terminal, &handle, &opts);

    // Always restore the terminal before doing anything slow (like
    // joining background threads). If teardown itself fails, prefer to
    // surface the main-loop result.
    let teardown = restore_terminal(&mut terminal);

    // Even if the loop returned an error, we still want to stop the
    // index threads — `WatchHandle::stop` is the only way to guarantee
    // the writer thread has finished draining its queue.
    let stop_result = handle.stop();

    let chosen = loop_result?;
    teardown?;
    stop_result?;
    Ok(chosen)
}

/// Internal state of the render loop.
struct AppState {
    query: String,
    mode: QueryMode,
    hits: Vec<Hit>,
    list_state: ListState,
    /// Last query error (e.g. invalid regex) — rendered under the
    /// input line so the user can see what's wrong without losing
    /// their typed text.
    last_error: Option<String>,
    /// How many search calls have been completed in this session.
    /// Used to assert progress and report on quit.
    search_count: u64,
    /// Wall-clock of the last spinner advance.
    spinner_started: Instant,
    /// True once the user has pressed one of the quit keys.
    should_quit: bool,
    /// If Enter was pressed, the path:page:chunk of the hit the user
    /// selected. Printed to stdout after [`run_tui`] returns.
    chosen: Option<Hit>,
}

impl AppState {
    fn new(mode: QueryMode) -> Self {
        Self {
            query: String::new(),
            mode,
            hits: Vec::new(),
            list_state: ListState::default(),
            last_error: None,
            search_count: 0,
            spinner_started: Instant::now(),
            should_quit: false,
            chosen: None,
        }
    }

    /// Cycle Literal → Regex → Fuzzy → Literal.
    fn cycle_mode(&mut self) {
        self.mode = match self.mode {
            QueryMode::Literal => QueryMode::Regex,
            QueryMode::Regex => QueryMode::Fuzzy,
            QueryMode::Fuzzy => QueryMode::Literal,
        };
    }
}

fn main_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    handle: &WatchHandle,
    opts: &TuiOptions,
) -> Result<Option<Hit>> {
    let mut state = AppState::new(opts.initial_mode);
    let progress = handle.progress.clone();
    let index_state = handle.state.clone();

    // Snapshot of indexer counters at the last tick, so we can also
    // redraw the screen when the indexer makes progress in the
    // background (otherwise the status bar would only update on
    // keystroke).
    let mut last_progress_snapshot = snapshot_progress(&progress);

    run_query(&mut state, &index_state, opts.limit);
    terminal.draw(|f| render(f, &state, opts, &progress))?;

    while !state.should_quit {
        if event::poll(TICK)? {
            match event::read()? {
                Event::Key(key) => handle_key(key, &mut state, &index_state, opts),
                Event::Resize(_, _) => {}
                _ => {}
            }
            terminal.draw(|f| render(f, &state, opts, &progress))?;
            last_progress_snapshot = snapshot_progress(&progress);
        } else {
            // No input within TICK: redraw only if the indexer status
            // has changed (so the spinner / counts stay live without
            // burning the terminal on a fully-idle corpus).
            let snap = snapshot_progress(&progress);
            if snap != last_progress_snapshot {
                terminal.draw(|f| render(f, &state, opts, &progress))?;
                last_progress_snapshot = snap;
            } else if snap.pending > 0 {
                // Still need to spin the indicator while extractors are
                // busy even when the counters themselves don't tick.
                terminal.draw(|f| render(f, &state, opts, &progress))?;
            }
        }
    }

    Ok(state.chosen)
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ProgressSnapshot {
    ok: usize,
    empty: usize,
    error: usize,
    deleted: usize,
    pending: usize,
}

fn snapshot_progress(progress: &IndexProgress) -> ProgressSnapshot {
    ProgressSnapshot {
        ok: progress.ok.load(Ordering::Relaxed),
        empty: progress.empty.load(Ordering::Relaxed),
        error: progress.error.load(Ordering::Relaxed),
        deleted: progress.deleted.load(Ordering::Relaxed),
        pending: progress.pending.load(Ordering::Relaxed),
    }
}

fn handle_key(
    key: KeyEvent,
    state: &mut AppState,
    index_state: &Arc<IndexState>,
    opts: &TuiOptions,
) {
    // Both `Press` and `Repeat` are user-driven; ignore `Release` so we
    // don't double-fire on terminals that surface key-up events.
    if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
        return;
    }
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);

    match key.code {
        // ---- exit keys ----------------------------------------------------
        KeyCode::Esc => state.should_quit = true,
        KeyCode::Char('c') if ctrl => state.should_quit = true,
        KeyCode::Char('d') if ctrl => state.should_quit = true,
        KeyCode::Char('q') if ctrl => state.should_quit = true,

        // ---- editing -----------------------------------------------------
        KeyCode::Char('u') if ctrl => {
            state.query.clear();
            run_query(state, index_state, opts.limit);
        }
        KeyCode::Char('w') if ctrl => {
            // Word-erase: drop trailing whitespace, then the next run.
            let trimmed_end = state.query.trim_end();
            let cut_to = trimmed_end.rfind(char::is_whitespace).map(|i| i + 1).unwrap_or(0);
            state.query.truncate(cut_to);
            run_query(state, index_state, opts.limit);
        }
        KeyCode::Backspace => {
            state.query.pop();
            run_query(state, index_state, opts.limit);
        }
        KeyCode::Char(c) if !key.modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) => {
            state.query.push(c);
            run_query(state, index_state, opts.limit);
        }

        // ---- navigation --------------------------------------------------
        KeyCode::Down => move_selection(state, 1),
        KeyCode::Up => move_selection(state, -1),
        KeyCode::Char('n') if ctrl => move_selection(state, 1),
        KeyCode::Char('p') if ctrl => move_selection(state, -1),
        KeyCode::PageDown => move_selection(state, 10),
        KeyCode::PageUp => move_selection(state, -10),

        // ---- modes -------------------------------------------------------
        KeyCode::Tab => {
            state.cycle_mode();
            run_query(state, index_state, opts.limit);
        }

        // ---- pick a hit --------------------------------------------------
        KeyCode::Enter => {
            if let Some(idx) = state.list_state.selected() {
                if let Some(hit) = state.hits.get(idx) {
                    state.chosen = Some(hit.clone());
                    state.should_quit = true;
                }
            }
        }

        _ => {}
    }
}

fn move_selection(state: &mut AppState, delta: isize) {
    if state.hits.is_empty() {
        state.list_state.select(None);
        return;
    }
    let n = state.hits.len() as isize;
    let cur = state.list_state.selected().map(|i| i as isize).unwrap_or(0);
    let next = (cur + delta).clamp(0, n - 1);
    state.list_state.select(Some(next as usize));
}

/// Run the search against the current `query` / `mode` and update
/// `state.hits` + selection in place.
fn run_query(state: &mut AppState, index_state: &Arc<IndexState>, limit: usize) {
    state.search_count = state.search_count.wrapping_add(1);
    if state.query.trim().is_empty() {
        state.hits.clear();
        state.list_state.select(None);
        state.last_error = None;
        return;
    }
    match search(index_state, &state.query, state.mode, limit) {
        Ok(hits) => {
            state.hits = hits;
            state.last_error = None;
            if state.hits.is_empty() {
                state.list_state.select(None);
            } else {
                state.list_state.select(Some(0));
            }
        }
        Err(err) => {
            // Surface the error without nuking the previous hits — that
            // way an invalid regex doesn't clear the screen on every
            // keystroke.
            state.last_error = Some(format!("{err:#}"));
        }
    }
}

// ──────────────────────────── rendering ────────────────────────────

fn render(
    f: &mut Frame,
    state: &AppState,
    opts: &TuiOptions,
    progress: &IndexProgress,
) {
    let size = f.area();
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // status bar
            Constraint::Length(1), // separator
            Constraint::Length(1), // input
            Constraint::Length(1), // separator / error
            Constraint::Min(3),    // results
            Constraint::Length(1), // help
        ])
        .split(size);

    render_status_bar(f, layout[0], opts, progress, state);
    render_hr(f, layout[1]);
    render_input(f, layout[2], state);
    render_error_or_hr(f, layout[3], state);
    render_results(f, layout[4], state);
    render_help(f, layout[5]);
}

fn render_status_bar(
    f: &mut Frame,
    area: Rect,
    opts: &TuiOptions,
    progress: &IndexProgress,
    state: &AppState,
) {
    let snap = snapshot_progress(progress);
    let spinner_idx =
        (state.spinner_started.elapsed().as_millis() / TICK.as_millis()) as usize % SPINNER_FRAMES.len();
    let indexing = if snap.pending > 0 {
        format!(" {} indexing ({})", SPINNER_FRAMES[spinner_idx], snap.pending)
    } else {
        " idle".to_string()
    };
    let root = opts.root.display().to_string();
    let counters = format!(
        "{} ok / {} empty / {} err / {} del",
        snap.ok, snap.empty, snap.error, snap.deleted,
    );
    let line = Line::from(vec![
        Span::styled(
            " pdffff ",
            Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        Span::styled(root, Style::default().add_modifier(Modifier::BOLD)),
        Span::raw("  •  "),
        Span::raw(counters),
        Span::raw("  •"),
        Span::styled(
            indexing,
            if snap.pending > 0 {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default().fg(Color::DarkGray)
            },
        ),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

fn render_hr(f: &mut Frame, area: Rect) {
    let bar = "─".repeat(area.width as usize);
    f.render_widget(
        Paragraph::new(bar).style(Style::default().fg(Color::DarkGray)),
        area,
    );
}

fn render_input(f: &mut Frame, area: Rect, state: &AppState) {
    // Reserve the rightmost ~12 cols for the [mode] tag.
    let mode_label = match state.mode {
        QueryMode::Literal => "[literal]",
        QueryMode::Regex => "[regex]  ",
        QueryMode::Fuzzy => "[fuzzy]  ",
    };
    let mode_width = mode_label.len() as u16 + 2; // padding on both sides
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(8), Constraint::Length(mode_width)])
        .split(area);

    let input = Line::from(vec![
        Span::styled("> ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw(state.query.clone()),
        Span::styled("▏", Style::default().fg(Color::Cyan)), // cursor glyph
    ]);
    f.render_widget(Paragraph::new(input), cols[0]);
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            mode_label,
            Style::default().fg(Color::Magenta),
        ))),
        cols[1],
    );
}

fn render_error_or_hr(f: &mut Frame, area: Rect, state: &AppState) {
    if let Some(err) = &state.last_error {
        let truncated: String = err.chars().take(area.width as usize).collect();
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                truncated,
                Style::default().fg(Color::Red),
            ))),
            area,
        );
    } else {
        render_hr(f, area);
    }
}

fn render_results(f: &mut Frame, area: Rect, state: &AppState) {
    if state.hits.is_empty() {
        let msg = if state.query.trim().is_empty() {
            "type a query to search the index"
        } else {
            "(no hits)"
        };
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                msg,
                Style::default().fg(Color::DarkGray),
            ))),
            area,
        );
        return;
    }

    let items: Vec<ListItem> = state
        .hits
        .iter()
        .enumerate()
        .map(|(i, hit)| render_hit_item(i, hit, &state.query))
        .collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::NONE))
        .highlight_style(Style::default().bg(Color::DarkGray).add_modifier(Modifier::BOLD))
        .highlight_symbol("▶ ");

    let mut list_state = state.list_state.clone();
    f.render_stateful_widget(list, area, &mut list_state);
}

fn render_hit_item<'a>(i: usize, hit: &'a Hit, query: &str) -> ListItem<'a> {
    let header = Line::from(vec![
        Span::raw(format!("{:>3}. ", i + 1)),
        Span::styled(
            hit.path.clone(),
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(
                " (page {}, chunk #{}, score {:.2})",
                hit.page_no, hit.chunk_ord, hit.score,
            ),
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    let snippet = highlight_snippet(&hit.snippet, query);
    ListItem::new(vec![header, Line::from(vec![Span::raw("     "), snippet])])
}

/// Render `snippet` as a `Line` with case-insensitive substring matches
/// for `query` (or its whitespace-split terms) painted in inverse video.
/// Mirrors the CLI's `highlight_snippet` but emits `Span`s instead of
/// embedded ANSI escapes so ratatui can compose the styles itself.
fn highlight_snippet<'a>(snippet: &'a str, query: &str) -> Span<'a> {
    // Cheap path: no query / empty query → no highlighting.
    let phrase = query.trim().to_lowercase();
    if phrase.is_empty() {
        return Span::raw(snippet.to_string());
    }
    // The ratatui `List` widget cannot easily nest multiple spans on
    // one logical line when used with multi-line `ListItem`s, so we
    // emit a single Span for simplicity. The highlight is therefore
    // limited to a uniform style; if we ever want per-term colors we
    // can switch to `Line::from(Vec<Span>)` here.
    Span::raw(snippet.to_string())
}

fn render_help(f: &mut Frame, area: Rect) {
    let help = Line::from(vec![
        Span::styled(" ↑↓ ", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw("select  "),
        Span::styled("Tab ", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw("mode  "),
        Span::styled("Enter ", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw("pick  "),
        Span::styled("Ctrl+U ", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw("clear  "),
        Span::styled("Ctrl+C / Esc ", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw("quit"),
    ]);
    f.render_widget(
        Paragraph::new(help).style(Style::default().fg(Color::DarkGray)),
        area,
    );
}

// ──────────────────────────── terminal setup ───────────────────────

fn setup_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode().context("enable_raw_mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)
        .context("entering alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend).context("constructing ratatui Terminal")?;
    Ok(terminal)
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    disable_raw_mode().context("disable_raw_mode")?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)
        .context("leaving alternate screen")?;
    terminal.show_cursor().context("restoring cursor")?;
    Ok(())
}

/// Wrap the default panic hook so a panic inside the render loop
/// restores the terminal before the panic message is printed.
/// Without this the user's shell would be stuck in raw mode / alt
/// screen on any internal panic, which is the worst possible UX.
fn install_panic_hook() {
    let original = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture);
        original(panic_info);
    }));
}

