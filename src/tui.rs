//! Interactive TUI for `pdffff`.
//!
//! Visual layout — Norton-Commander-revival palette with each visual
//! variable carrying exactly one semantic role (Bertin's "semiology of
//! graphics" applied to a 16-colour TTY): blue chrome, white primary
//! text, dim secondary, magenta-bold for the selected card border, and
//! yellow-on-black reserved exclusively for query matches (in titles
//! *and* snippet bodies):
//!
//! ```text
//!  ╭─ pdffff /Users/foo/papers ─────────── 123 ok · 0 err · ⠿ indexing 3 ─╮
//!  │                                                                       │
//!  │  ❯ alpha synthesis▏                                            LIT    │
//!  │  ───────────────────────────────────────────────────────────────────  │
//!  │                                                                       │
//!  │  ╭ 1. paper.pdf ─────────────────────────────────── p.12 · #3 ─────╮  │
//!  │  │  …matching snippet excerpt with highlighted terms…              │  │
//!  │  ╰─────────────────────────────────────────────────────────────────╯  │
//!  │                                                                       │
//!  │  ╭ 2. other.pdf ──────────────────── p. 1 · #0 · score 1247 ──────╮   │
//!  │  │  …another snippet…                                              │  │
//!  │  ╰─────────────────────────────────────────────────────────────────╯  │
//!  ╰─ ↑↓ select · Tab mode · Enter open · Ctrl+U clear · Esc quit ─────────╯
//! ```
//!
//! The TUI owns a [`WatchHandle`] from [`crate::app::run_watch`]: the
//! handle's background threads keep the index live while the UI runs
//! queries against the same `IndexState`. The writer thread persists
//! every successful mutation to SQLite synchronously, so the only
//! work shutdown has to do is signal the threads to drain and join.
//!
//! Concurrency
//! -----------
//! Searches do not run on the input thread. A dedicated
//! [`SearchWorker`] thread owns an `Arc<IndexState>` and a one-slot
//! mailbox: every keystroke / mode change overwrites the slot, so a
//! burst of input (typing, key-repeat on Backspace) coalesces into the
//! *latest* query rather than running a full search per key. The worker
//! publishes its result into a `Mutex<Option<…>>`; the render loop
//! polls that mutex each iteration. A monotonic stamp on every request
//! is echoed back so we drop results for queries the user has already
//! moved past.
//!
//! Net effect: keystrokes never block on a search, and a long-running
//! search against a large corpus does not freeze the UI — the
//! coordinator / writer / extractor threads from `run_watch` continue
//! to index in the background while the TUI stays responsive.
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
//! The TUI does not write tracing output itself. The launcher
//! (`main.rs`) redirects the `tracing` subscriber to a log file before
//! entering the TUI so that index-progress logs do not corrupt the
//! alternate screen.

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
    layout::{Alignment, Constraint, Direction, Layout, Position, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, ListState, Padding, Paragraph},
};
use std::io::{self, Stdout};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use crate::app::{IndexProgress, WatchHandle};
use crate::query::{DISPLAY_LIMIT, Hit, QueryMode};
use crate::ui::highlight::{SegmentKind, SnippetSegment, highlight_segments};
use crate::ui::launch::open_in_system_viewer;
use crate::ui::search::{SearchRequest, SearchWorker};

/// How often (at most) we redraw the screen when no key is pressed.
/// Drives the indexing-status spinner and the elapsed-time counter.
const TICK: Duration = Duration::from_millis(100);

/// Spinner frames cycled at every [`TICK`].
const SPINNER_FRAMES: [&str; 10] =
    ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

// ─────────────────────────── palette ───────────────────────────
//
// One hue, one job. Background colour is reserved exclusively for
// query matches — the most important pop-out signal in the UI — so
// selection is signalled on the border (hue + bold) instead of as a
// row background. Named / 256-indexed terminal colours throughout, so
// the UI inherits the user's terminal theme rather than baking in RGB.

/// Chrome — outer & card borders, prompt arrow, brand pill, key
/// chips, mode pill background, separator rule. Saturated blue, the
/// classic Norton-Commander-era status colour.
const CHROME: Color = Color::Blue;
/// Secondary text — meta lines, counters, idle/indexing status,
/// numbering, separators. Regular ANSI gray (7) rather than
/// "bright black" (8 / DarkGray) which is near-invisible on
/// true-black terminals.
const DIM: Color = Color::Gray;
/// Primary text — query, filenames, brand wordmark, mode pill label.
const PRIMARY: Color = Color::White;
/// Focus accent — the only place magenta appears, applied to the
/// border of the currently-selected card.
const SEL: Color = Color::Magenta;
/// Match highlight — reserved exclusively for query matches inside
/// snippet bodies and card-title filenames. Reverse-video against
/// yellow gives the highest selective power Bertin's "value"
/// variable can offer on a 16-colour terminal.
const HL_BG: Color = Color::Yellow;
const HL_FG: Color = Color::Black;
/// Error pill background. The single use of red in the UI.
const ERROR: Color = Color::Red;

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
            initial_mode: QueryMode::Fuzzy,
            root: PathBuf::new(),
        }
    }
}

/// Run the TUI until the user quits or an unrecoverable IO error
/// occurs. Owns `handle`: calls [`WatchHandle::stop`] on the way out
/// so the index threads always shut down cleanly, even on panic.
///
/// Pressing Enter on a selected result hands the path to the host's
/// default PDF viewer via [`crate::ui::launch::open_in_system_viewer`]
/// without ending the session — the user can keep searching and open
/// further results. Errors from the launcher surface in the in-screen
/// error pill rather than corrupting the alternate-screen output.
pub fn run_tui(handle: WatchHandle, opts: TuiOptions) -> Result<()> {
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

    loop_result?;
    teardown?;
    stop_result?;
    Ok(())
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
    /// Monotonic stamp bumped on every query / mode edit. Submitted to
    /// the worker and echoed back in the result so we can drop hits for
    /// queries the user has already moved past.
    submitted_stamp: u64,
    /// Stamp of the result currently displayed in `hits`. When this
    /// trails `submitted_stamp` a search is in flight on the worker —
    /// we leave the previous hits on screen rather than blanking the
    /// list so the UI never goes empty between keystrokes.
    applied_stamp: u64,
    /// Wall-clock of the last spinner advance.
    spinner_started: Instant,
    /// Index of the first card visible in the results area. Tracks
    /// scroll position across renders so the view doesn't jump when the
    /// terminal repaints. Mutated by the renderer once it knows the
    /// available area; reset to 0 on every fresh result set.
    scroll_top: usize,
    /// True once the user has pressed one of the quit keys.
    should_quit: bool,
}

impl AppState {
    fn new(mode: QueryMode) -> Self {
        Self {
            query: String::new(),
            mode,
            hits: Vec::new(),
            list_state: ListState::default(),
            last_error: None,
            submitted_stamp: 0,
            applied_stamp: 0,
            spinner_started: Instant::now(),
            scroll_top: 0,
            should_quit: false,
        }
    }

    /// True when the latest submitted query has not yet been answered
    /// by the worker. Drives the prompt-position spinner.
    fn is_searching(&self) -> bool {
        self.applied_stamp != self.submitted_stamp
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
) -> Result<()> {
    let worker = SearchWorker::spawn(handle.state.clone())
        .context("spawning TUI search worker")?;
    // Run the loop body in a helper so worker shutdown happens on every
    // exit path (including `?` propagation), without a Drop guard.
    let outcome = run_event_loop(terminal, &worker, &handle.progress, opts);
    worker.stop();
    outcome
}

fn run_event_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    worker: &SearchWorker,
    progress: &IndexProgress,
    opts: &TuiOptions,
) -> Result<()> {
    let mut state = AppState::new(opts.initial_mode);

    // Snapshot of indexer counters at the last tick, so we can also
    // redraw the screen when the indexer makes progress in the
    // background (otherwise the status bar would only update on
    // keystroke).
    let mut last_progress_snapshot = snapshot_progress(progress);

    terminal.draw(|f| render(f, &mut state, opts, progress))?;

    while !state.should_quit {
        let had_event = event::poll(TICK)?;
        if had_event {
            match event::read()? {
                Event::Key(key) => handle_key(key, &mut state, worker, opts),
                Event::Resize(_, _) => {}
                _ => {}
            }
        }
        let got_result = drain_results(&mut state, worker);
        let snap = snapshot_progress(progress);
        // Redraw if anything visible could have changed: user input,
        // a fresh search result, the indexer counters ticked, or the
        // extractor pool is still busy (spinner needs to keep moving).
        if had_event
            || got_result
            || snap != last_progress_snapshot
            || snap.pending > 0
        {
            terminal.draw(|f| render(f, &mut state, opts, progress))?;
            last_progress_snapshot = snap;
        }
    }

    Ok(())
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
    worker: &SearchWorker,
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
        KeyCode::Char('c' | 'd' | 'q') if ctrl => state.should_quit = true,

        // ---- editing -----------------------------------------------------
        KeyCode::Char('u') if ctrl => edit_and_search(state, worker, opts, String::clear),
        KeyCode::Char('w') if ctrl => edit_and_search(state, worker, opts, word_erase),
        KeyCode::Backspace => edit_and_search(state, worker, opts, |q| {
            q.pop();
        }),
        KeyCode::Char(c) if !key.modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) => {
            edit_and_search(state, worker, opts, |q| q.push(c));
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
            submit_query(state, worker, opts.limit);
        }

        // ---- pick a hit --------------------------------------------------
        // Pressing Enter on a selected result hands the path to the
        // host's PDF viewer and keeps the search session running, so
        // multiple files can be opened in sequence. Errors surface in
        // the in-screen error pill rather than corrupting stderr.
        KeyCode::Enter => {
            if let Some(idx) = state.list_state.selected() {
                if let Some(hit) = state.hits.get(idx) {
                    if let Err(err) = open_in_system_viewer(&hit.path) {
                        state.last_error = Some(format!(
                            "could not open {}: {err:#}",
                            hit.path
                        ));
                    } else {
                        state.last_error = None;
                    }
                }
            }
        }

        _ => {}
    }
}

/// Mutate `state.query` through `edit` and submit a fresh search.
///
/// Every editing key (clear / word-erase / backspace / character
/// insert) follows the same shape; centralising it keeps the four
/// branches in `handle_key` to one line each and ensures no future
/// editing key can forget to dispatch a search.
fn edit_and_search(
    state: &mut AppState,
    worker: &SearchWorker,
    opts: &TuiOptions,
    edit: impl FnOnce(&mut String),
) {
    edit(&mut state.query);
    submit_query(state, worker, opts.limit);
}

/// Drop the trailing whitespace-bounded word from `q`.
///
/// Matches the readline / shell convention: trailing whitespace first,
/// then everything back to the next whitespace boundary.
fn word_erase(q: &mut String) {
    let trimmed_end = q.trim_end();
    let cut_to = trimmed_end
        .rfind(char::is_whitespace)
        .map(|i| i + 1)
        .unwrap_or(0);
    q.truncate(cut_to);
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

/// Bump the query stamp and either clear the UI (empty query) or hand
/// the work to the [`SearchWorker`]. Never blocks: the worker may still
/// be busy on an older query — its mailbox is one-slot, so this just
/// overwrites the pending request.
fn submit_query(state: &mut AppState, worker: &SearchWorker, limit: usize) {
    state.submitted_stamp = state.submitted_stamp.wrapping_add(1);
    if state.query.trim().is_empty() {
        state.hits.clear();
        state.list_state.select(None);
        state.scroll_top = 0;
        state.last_error = None;
        state.applied_stamp = state.submitted_stamp;
        return;
    }
    worker.submit(SearchRequest {
        stamp: state.submitted_stamp,
        query: state.query.clone(),
        mode: state.mode,
        limit,
    });
}

/// Pull any pending result from the worker and apply it. Returns true
/// when a fresh result was applied so the caller knows to redraw.
///
/// Results whose stamp predates `state.submitted_stamp` (i.e. the user
/// has typed more or cleared the query in the meantime) are dropped on
/// the floor — we never paint hits that no longer match the visible
/// query.
fn drain_results(state: &mut AppState, worker: &SearchWorker) -> bool {
    let Some(result) = worker.take_result() else {
        return false;
    };
    if result.stamp != state.submitted_stamp {
        return false;
    }
    state.applied_stamp = result.stamp;
    match result.hits {
        Ok(hits) => {
            state.hits = hits;
            state.last_error = None;
            state.list_state.select(if state.hits.is_empty() {
                None
            } else {
                Some(0)
            });
            state.scroll_top = 0;
        }
        Err(err) => {
            // Surface the error without nuking the previous hits — that
            // way an invalid regex doesn't clear the screen on every
            // keystroke.
            state.last_error = Some(format!("{err:#}"));
        }
    }
    true
}

// ──────────────────────────── rendering ────────────────────────────

fn render(
    f: &mut Frame,
    state: &mut AppState,
    opts: &TuiOptions,
    progress: &IndexProgress,
) {
    let size = f.area();
    let snap = snapshot_progress(progress);

    let outer = build_outer_frame(&snap, opts, state);
    let inner = outer.inner(size);
    f.render_widget(outer, size);

    // Vertical layout *inside* the outer frame.
    //   row 0  : top breathing room
    //   row 1  : input + mode pill
    //   row 2  : separator (or inline error)
    //   row 3  : breathing room before the result cards
    //   rest   : result cards
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(1),
        ])
        .split(inner);

    render_input(f, layout[1], state);
    render_error_or_hr(f, layout[2], state);
    render_results(f, layout[4], state);
}

/// The outer rounded frame. Title-top-left carries the brand pill and
/// the watched root; title-top-right carries the live counters and
/// indexing spinner; title-bottom carries the keybinding hints.
fn build_outer_frame<'a>(
    snap: &ProgressSnapshot,
    opts: &'a TuiOptions,
    state: &AppState,
) -> Block<'a> {
    let brand = Span::styled(
        " pdffff ",
        Style::default().bg(CHROME).fg(PRIMARY).add_modifier(Modifier::BOLD),
    );
    let root = Span::styled(
        format!(" {} ", opts.root.display()),
        Style::default().fg(PRIMARY).add_modifier(Modifier::BOLD),
    );
    let top_left = Line::from(vec![Span::raw(" "), brand, root]).left_aligned();

    let counters = Span::styled(
        format!(
            "{} ok · {} empty · {} err · {} del",
            snap.ok, snap.empty, snap.error, snap.deleted,
        ),
        Style::default().fg(DIM),
    );
    let spinner_idx = (state.spinner_started.elapsed().as_millis()
        / TICK.as_millis()) as usize
        % SPINNER_FRAMES.len();
    // Both branches use the same dim style — the spinner motion is the
    // activity signal, no extra colour required (Norton-Commander-style
    // status row, not a modern progress bar).
    let activity = if snap.pending > 0 {
        Span::styled(
            format!(" {} indexing {} ", SPINNER_FRAMES[spinner_idx], snap.pending),
            Style::default().fg(DIM),
        )
    } else {
        Span::styled(" idle ", Style::default().fg(DIM))
    };
    let top_right = Line::from(vec![
        counters,
        Span::styled(" · ", Style::default().fg(DIM)),
        activity,
        Span::raw(" "),
    ])
    .right_aligned();

    let help = Line::from(vec![
        Span::raw(" "),
        key_chip("↑↓"),
        Span::raw(" select  "),
        key_chip("Tab"),
        Span::raw(" mode  "),
        key_chip("Enter"),
        Span::raw(" open  "),
        key_chip("Ctrl+U"),
        Span::raw(" clear  "),
        key_chip("Esc"),
        Span::raw(" quit "),
    ])
    .left_aligned()
    .style(Style::default().fg(DIM));

    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(CHROME))
        .padding(Padding::new(1, 1, 0, 0))
        .title_top(top_left)
        .title_top(top_right)
        .title_bottom(help)
}

/// A keybinding label, styled in the chrome hue against the dim help
/// footer.
fn key_chip(text: &str) -> Span<'_> {
    Span::styled(
        text,
        Style::default()
            .fg(CHROME)
            .add_modifier(Modifier::BOLD),
    )
}

fn render_input(f: &mut Frame, area: Rect, state: &AppState) {
    // Short labels: mode is a nominal distinction, and the label text
    // itself encodes it. We do not lean on hue here — every mode pill
    // shares the same chrome background, so hue stays free to do its
    // one job (matches).
    let mode_label = match state.mode {
        QueryMode::Literal => " LIT ",
        QueryMode::Regex => " RE ",
        QueryMode::Fuzzy => " FZ ",
    };
    let mode_w = mode_label.chars().count() as u16;

    // Left: prompt + query.  Right: mode pill.
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Min(8),
            Constraint::Length(mode_w),
        ])
        .split(area);

    // Prompt glyph: chevron when idle, animated spinner frame while a
    // search is in flight. Same `applied_stamp != submitted_stamp`
    // contract as the GUI's prompt spinner — drains the same one-slot
    // worker mailbox.
    let prompt_glyph = if state.is_searching() {
        let idx = (state.spinner_started.elapsed().as_millis() / TICK.as_millis())
            as usize
            % SPINNER_FRAMES.len();
        SPINNER_FRAMES[idx]
    } else {
        "❯"
    };
    let prompt = Span::styled(
        format!("{prompt_glyph} "),
        Style::default().fg(CHROME).add_modifier(Modifier::BOLD),
    );
    let query = Span::styled(
        state.query.clone(),
        Style::default().fg(PRIMARY).add_modifier(Modifier::BOLD),
    );
    f.render_widget(Paragraph::new(Line::from(vec![prompt, query])), cols[0]);

    let pill = Paragraph::new(Line::from(Span::styled(
        mode_label,
        Style::default()
            .bg(CHROME)
            .fg(PRIMARY)
            .add_modifier(Modifier::BOLD),
    )))
    .alignment(Alignment::Right);
    f.render_widget(pill, cols[1]);

    // Real terminal cursor at the end of the query, clamped to the
    // visible width of the input column. Visual width of `❯ ` is 2.
    const PROMPT_W: u16 = 2;
    let query_chars = state.query.chars().count() as u16;
    let cursor_x = cols[0]
        .x
        .saturating_add(PROMPT_W)
        .saturating_add(query_chars);
    let max_x = cols[0]
        .x
        .saturating_add(cols[0].width.saturating_sub(1));
    f.set_cursor_position(Position::new(cursor_x.min(max_x), cols[0].y));
}

/// When there is a query error, render it under the input as a red
/// pill plus message; otherwise draw a dim separator across the
/// content area (a horizontal rule that visually splits input from
/// results, but inside the rounded frame so it never touches the
/// border).
fn render_error_or_hr(f: &mut Frame, area: Rect, state: &AppState) {
    if let Some(err) = &state.last_error {
        let label = Span::styled(
            " error ",
            Style::default()
                .bg(ERROR)
                .fg(PRIMARY)
                .add_modifier(Modifier::BOLD),
        );
        // Reserve room for the label, two spaces of padding, and a
        // little safety margin so we never wrap to a second line.
        let budget = area.width.saturating_sub(10) as usize;
        let truncated: String = err.chars().take(budget).collect();
        f.render_widget(
            Paragraph::new(Line::from(vec![
                label,
                Span::raw(" "),
                Span::styled(truncated, Style::default().fg(ERROR)),
            ])),
            area,
        );
    } else if area.width > 0 {
        let bar = "─".repeat(area.width as usize);
        f.render_widget(
            Paragraph::new(bar).style(Style::default().fg(CHROME)),
            area,
        );
    }
}

/// Total vertical rows one result card occupies (top border + one
/// snippet row + bottom border). Snippets are pre-bounded by the
/// snippet builder; we never wrap, so a single content row suffices.
const CARD_HEIGHT: u16 = 3;
/// Blank rows between adjacent cards. Bertin: separation is itself a
/// visual variable — explicit empty space says "different unit" more
/// clearly than any divider could.
const CARD_GAP: u16 = 1;

fn render_results(f: &mut Frame, area: Rect, state: &mut AppState) {
    if state.hits.is_empty() {
        let msg = if state.query.trim().is_empty() {
            "type a query to search the index"
        } else {
            "no hits"
        };
        let placeholder = Paragraph::new(Line::from(Span::styled(
            msg,
            Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
        )))
        .alignment(Alignment::Center);
        f.render_widget(placeholder, area);
        return;
    }

    let visible = visible_card_count(area.height);
    if visible == 0 {
        return;
    }
    let selected = state.list_state.selected().unwrap_or(0);
    state.scroll_top =
        compute_scroll_top(selected, state.hits.len(), visible, state.scroll_top);
    let scroll_top = state.scroll_top;

    let n = visible.min(state.hits.len() - scroll_top);
    for slot in 0..n {
        let i = scroll_top + slot;
        let y = area.y + slot as u16 * (CARD_HEIGHT + CARD_GAP);
        if y + CARD_HEIGHT > area.y + area.height {
            break;
        }
        let card_area = Rect::new(area.x, y, area.width, CARD_HEIGHT);
        render_hit_card(
            f,
            card_area,
            i,
            &state.hits[i],
            &state.query,
            state.mode,
            i == selected,
        );
    }
}

/// How many cards fit in `height` rows, given `CARD_HEIGHT` rows per
/// card and `CARD_GAP` blank rows between cards. Returns 0 when even
/// one card would not fit.
fn visible_card_count(height: u16) -> usize {
    if height < CARD_HEIGHT {
        return 0;
    }
    // n cards take CARD_HEIGHT + (n-1) * (CARD_HEIGHT + CARD_GAP) rows.
    let extra = height - CARD_HEIGHT;
    1 + (extra / (CARD_HEIGHT + CARD_GAP)) as usize
}

/// Pick a scroll position that keeps `selected` visible while disturbing
/// `prev` as little as possible. The selection moves by user input; the
/// scroll position only shifts when selection would otherwise fall out
/// of frame.
fn compute_scroll_top(
    selected: usize,
    total: usize,
    visible: usize,
    prev: usize,
) -> usize {
    if total == 0 || visible == 0 {
        return 0;
    }
    let max_top = total.saturating_sub(visible);
    let mut top = prev.min(max_top);
    if selected < top {
        top = selected;
    } else if selected >= top + visible {
        top = selected + 1 - visible;
    }
    top.min(max_top)
}

fn render_hit_card(
    f: &mut Frame,
    area: Rect,
    i: usize,
    hit: &Hit,
    query: &str,
    mode: QueryMode,
    selected: bool,
) {
    // Score is only meaningful in fuzzy mode (it's the neo_frizbee u16
    // turned into f32). For literal and regex matches it is a hardcoded
    // `1.0` — showing it there is noise, so we hide the field.
    let meta = match mode {
        QueryMode::Fuzzy => format!(
            " p.{} · #{} · score {:.2} ",
            hit.page_no, hit.chunk_ord, hit.score,
        ),
        QueryMode::Literal | QueryMode::Regex => {
            format!(" p.{} · #{} ", hit.page_no, hit.chunk_ord)
        }
    };

    // Display the filename rather than the full path: the full path is
    // typically the watched root + a long suffix, which crowds the
    // header off the right edge and adds no information the user can't
    // get from the corpus context. The filename inherits the highlight
    // style so a filename match (the whole reason there's a
    // filename-priority band in the fuzzy ranker) is visible at a
    // glance.
    let name_base = Style::default().fg(PRIMARY).add_modifier(Modifier::BOLD);
    let filename_spans = highlight_spans(&hit.filename, query, name_base, match_hl_style());

    let mut title_left_spans: Vec<Span<'static>> = vec![
        Span::raw(" "),
        Span::styled(format!("{}. ", i + 1), Style::default().fg(DIM)),
    ];
    title_left_spans.extend(filename_spans);
    title_left_spans.push(Span::raw(" "));
    let title_left = Line::from(title_left_spans).left_aligned();

    let title_right =
        Line::from(Span::styled(meta, Style::default().fg(DIM))).right_aligned();

    let border_style = if selected {
        Style::default().fg(SEL).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(CHROME)
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(border_style)
        .title_top(title_left)
        .title_top(title_right);

    let inner = block.inner(area);
    f.render_widget(block, area);

    let snippet_spans =
        highlight_spans(&hit.snippet, query, Style::default(), match_hl_style());
    f.render_widget(Paragraph::new(Line::from(snippet_spans)), inner);
}

/// The single match-highlight style, used identically in card titles
/// and snippet bodies. Centralising it makes the "yellow background is
/// reserved for matches" invariant a one-liner to audit.
fn match_hl_style() -> Style {
    Style::default()
        .bg(HL_BG)
        .fg(HL_FG)
        .add_modifier(Modifier::BOLD)
}

/// Thin frontend adapter: ask the shared highlighter where the
/// matches are, then paint each segment with `base` / `hl`.
///
/// The actual where-do-the-matches-fall logic lives in
/// [`crate::ui::highlight`]; this module is style-agnostic so the GUI
/// frontend can reuse exactly the same segmentation.
fn highlight_spans(
    text: &str,
    query: &str,
    base: Style,
    hl: Style,
) -> Vec<Span<'static>> {
    highlight_segments(text, query)
        .into_iter()
        .map(|SnippetSegment { text, kind }| {
            let style = match kind {
                SegmentKind::Plain => base,
                SegmentKind::Match => hl,
            };
            Span::styled(text, style)
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    // Highlighter correctness is exercised in `crate::ui::highlight`
    // — the snippet/title pipeline lives there and is frontend
    // agnostic. The TUI-side adapter is a one-liner; here we cover
    // only the TUI-specific concerns (style mapping + card layout
    // maths).

    #[test]
    fn highlight_spans_apply_correct_styles() {
        let base = Style::default().fg(PRIMARY);
        let hl = match_hl_style();
        let spans = highlight_spans("Hello world", "world", base, hl);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].content.as_ref(), "Hello ");
        assert_eq!(spans[0].style, base);
        assert_eq!(spans[1].content.as_ref(), "world");
        assert_eq!(spans[1].style, hl);
    }

    // ── card-layout maths ─────────────────────────────────────────────

    #[test]
    fn visible_card_count_respects_card_geometry() {
        // CARD_HEIGHT = 3, CARD_GAP = 1 → 3, 7, 11, 15 rows fit 1, 2, 3, 4.
        assert_eq!(visible_card_count(0), 0);
        assert_eq!(visible_card_count(2), 0);
        assert_eq!(visible_card_count(3), 1);
        assert_eq!(visible_card_count(6), 1);
        assert_eq!(visible_card_count(7), 2);
        assert_eq!(visible_card_count(10), 2);
        assert_eq!(visible_card_count(11), 3);
    }

    #[test]
    fn scroll_top_keeps_selection_in_frame() {
        // Selection inside the visible window: scroll_top doesn't move.
        assert_eq!(compute_scroll_top(2, 20, 5, 0), 0);
        // Selection past the bottom of the window: scroll_top advances
        // just enough to keep it visible.
        assert_eq!(compute_scroll_top(5, 20, 5, 0), 1);
        assert_eq!(compute_scroll_top(9, 20, 5, 0), 5);
        // Selection above the window: scroll_top jumps back to it.
        assert_eq!(compute_scroll_top(2, 20, 5, 7), 2);
        // Stale prev gets clamped to max_top so we never paint past the
        // end of the list.
        assert_eq!(compute_scroll_top(19, 20, 5, 999), 15);
        // List shorter than the window: scroll_top is pinned at 0.
        assert_eq!(compute_scroll_top(2, 3, 5, 4), 0);
        // Degenerate inputs don't panic.
        assert_eq!(compute_scroll_top(0, 0, 5, 0), 0);
        assert_eq!(compute_scroll_top(0, 5, 0, 0), 0);
    }
}
