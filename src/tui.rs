//! Interactive TUI for `pdffff`.
//!
//! Visual layout (fzf-inspired, but tailored to the
//! query-against-a-live-index workflow):
//!
//! ```text
//!  ╭─ pdffff /Users/foo/papers ──────────  123 ok · 0 err · ⠿ indexing 3 ─╮
//!  │                                                                       │
//!  │  ❯ query▏                                                  LITERAL    │
//!  │  ───────────────────────────────────────────────────────────────────  │
//!  │   ▌ 1. paper.pdf                                            p.12 · #3 │
//!  │       …matching snippet excerpt with highlighted terms…               │
//!  │     2. other.pdf                              p. 1 · #0 · score 1247  │
//!  │       …another snippet…                                               │
//!  │                                                                       │
//!  ╰─ ↑↓ select · Tab mode · Enter pick · Ctrl+U clear · Esc quit ─────────╯
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
    widgets::{
        Block, BorderType, Borders, HighlightSpacing, List, ListItem, ListState, Padding,
        Paragraph,
    },
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
const SPINNER_FRAMES: [&str; 10] =
    ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

// ─────────────────────────── palette ───────────────────────────
//
// We stick to *named* / 256-indexed terminal colours so the UI
// respects whatever colour theme the user has configured for their
// terminal (light, dark, Solarized, …) rather than baking in RGB
// values that fight a user's theme.

/// Brand accent. Used for the pdffff pill, prompt, key chips, and
/// the LITERAL mode pill.
const ACCENT: Color = Color::Cyan;
/// Subtle dim — borders, meta text, middot separators. We use the
/// regular ANSI gray (7) rather than "bright black" (8 / DarkGray)
/// because the latter is near-invisible on true-black terminals.
const DIM: Color = Color::Gray;
/// Border colour for the outer frame and the inline separator.
const BORDER: Color = Color::Gray;
/// Selection background for the currently-highlighted hit. Indexed
/// 237 is a near-black grey on a 256-colour terminal which contrasts
/// gently with most themes; it falls back to terminal default on
/// 16-colour TTYs.
const SEL_BG: Color = Color::Indexed(237);
/// Foreground used for matched query substrings inside snippets.
const HL_FG: Color = Color::Black;
/// Background used for matched query substrings inside snippets.
const HL_BG: Color = Color::Yellow;

/// Pick a foreground that reads against a coloured pill background on
/// both light and dark terminal themes.
///
/// `Color::Black` looked sharp on a paper-white terminal but vanished
/// on dark themes where the user's "black" is mapped close to the
/// terminal background; `Color::White` is the inverse trap on yellow.
/// We pick per-background and never rely on `Modifier::BOLD` to do
/// double duty as a brightness hint (some terminals interpret bold as
/// "lighten the foreground", which moves black toward grey on cyan).
fn pill_fg_for(bg: Color) -> Color {
    match bg {
        // Bright / pale backgrounds need dark text.
        Color::Yellow | Color::LightYellow | Color::White | Color::Gray => Color::Black,
        // Saturated / dark backgrounds need bright text.
        _ => Color::White,
    }
}

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
        KeyCode::Char('c' | 'd' | 'q') if ctrl => state.should_quit = true,

        // ---- editing -----------------------------------------------------
        KeyCode::Char('u') if ctrl => edit_and_search(state, index_state, opts, String::clear),
        KeyCode::Char('w') if ctrl => edit_and_search(state, index_state, opts, word_erase),
        KeyCode::Backspace => edit_and_search(state, index_state, opts, |q| {
            q.pop();
        }),
        KeyCode::Char(c) if !key.modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) => {
            edit_and_search(state, index_state, opts, |q| q.push(c));
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

/// Mutate `state.query` through `edit` then re-run the search.
///
/// Every editing key (clear / word-erase / backspace / character
/// insert) follows the same shape; centralising it keeps the four
/// branches in `handle_key` to one line each and ensures no future
/// editing key can forget to re-run the search.
fn edit_and_search(
    state: &mut AppState,
    index_state: &Arc<IndexState>,
    opts: &TuiOptions,
    edit: impl FnOnce(&mut String),
) {
    edit(&mut state.query);
    run_query(state, index_state, opts.limit);
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
    let snap = snapshot_progress(progress);

    let outer = build_outer_frame(&snap, opts, state);
    let inner = outer.inner(size);
    f.render_widget(outer, size);

    // Vertical layout *inside* the outer frame.
    //   row 0  : breathing room
    //   row 1  : input + mode pill
    //   row 2  : separator (or inline error)
    //   rest   : results list
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(1),
        ])
        .split(inner);

    render_input(f, layout[1], state);
    render_error_or_hr(f, layout[2], state);
    render_results(f, layout[3], state);
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
        Style::default().bg(ACCENT).fg(pill_fg_for(ACCENT)).add_modifier(Modifier::BOLD),
    );
    let root = Span::styled(
        format!(" {} ", opts.root.display()),
        Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
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
    let activity = if snap.pending > 0 {
        Span::styled(
            format!(" {} indexing {} ", SPINNER_FRAMES[spinner_idx], snap.pending),
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
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
        Span::raw(" pick  "),
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
        .border_style(Style::default().fg(BORDER))
        .padding(Padding::new(1, 1, 0, 0))
        .title_top(top_left)
        .title_top(top_right)
        .title_bottom(help)
}

/// A keybinding label, styled bright against the dim help footer.
fn key_chip(text: &str) -> Span<'_> {
    Span::styled(
        text,
        Style::default()
            .fg(ACCENT)
            .add_modifier(Modifier::BOLD),
    )
}

fn render_input(f: &mut Frame, area: Rect, state: &AppState) {
    let (mode_label, mode_bg) = match state.mode {
        QueryMode::Literal => (" LITERAL ", Color::Cyan),
        QueryMode::Regex => (" REGEX ", Color::Yellow),
        QueryMode::Fuzzy => (" FUZZY ", Color::Magenta),
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

    let prompt = Span::styled(
        "❯ ",
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
    );
    let query = Span::styled(
        state.query.clone(),
        Style::default().add_modifier(Modifier::BOLD),
    );
    f.render_widget(Paragraph::new(Line::from(vec![prompt, query])), cols[0]);

    let pill = Paragraph::new(Line::from(Span::styled(
        mode_label,
        Style::default()
            .bg(mode_bg)
            .fg(pill_fg_for(mode_bg))
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
                .bg(Color::Red)
                .fg(pill_fg_for(Color::Red))
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
                Span::styled(truncated, Style::default().fg(Color::Red)),
            ])),
            area,
        );
    } else if area.width > 0 {
        let bar = "─".repeat(area.width as usize);
        f.render_widget(
            Paragraph::new(bar).style(Style::default().fg(BORDER)),
            area,
        );
    }
}

fn render_results(f: &mut Frame, area: Rect, state: &AppState) {
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

    let items: Vec<ListItem> = state
        .hits
        .iter()
        .enumerate()
        .map(|(i, hit)| render_hit_item(i, hit, &state.query, state.mode))
        .collect();

    let list = List::new(items)
        .highlight_style(
            Style::default().bg(SEL_BG).add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▌ ")
        .highlight_spacing(HighlightSpacing::Always);

    let mut list_state = state.list_state.clone();
    f.render_stateful_widget(list, area, &mut list_state);
}

fn render_hit_item(i: usize, hit: &Hit, query: &str, mode: QueryMode) -> ListItem<'static> {
    // Score is only meaningful in fuzzy mode (it's the neo_frizbee u16
    // turned into f32). For literal and regex matches it is a hardcoded
    // `1.0` — showing it there is noise, so we hide the field.
    let meta = match mode {
        QueryMode::Fuzzy => format!(
            "p.{} · #{} · score {:.2}",
            hit.page_no, hit.chunk_ord, hit.score,
        ),
        QueryMode::Literal | QueryMode::Regex => {
            format!("p.{} · #{}", hit.page_no, hit.chunk_ord)
        }
    };
    // Display the filename rather than the full path: the full path is
    // typically the watched root + a long suffix, which crowds the
    // header off the right edge and adds no information the user can't
    // get from the corpus context.
    let header = Line::from(vec![
        Span::styled(
            format!("{:>2}. ", i + 1),
            Style::default().fg(DIM),
        ),
        Span::styled(
            hit.filename.clone(),
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ·  ", Style::default().fg(DIM)),
        Span::styled(meta, Style::default().fg(DIM)),
    ]);

    let mut snippet_spans: Vec<Span<'static>> = vec![Span::raw("   ")];
    snippet_spans.extend(highlight_snippet_spans(&hit.snippet, query));
    ListItem::new(vec![header, Line::from(snippet_spans)])
}

/// Render `snippet` as a list of styled spans, painting case-insensitive
/// substring matches for `query` (or its whitespace-split terms) on a
/// yellow background. Greedy left-to-right, longest-needle-wins — so a
/// match on the full phrase takes precedence over its constituent
/// terms. UTF-8 safe.
///
/// Three composed passes:
///
/// 1. [`build_needles`] — phrase + whitespace-split terms, deduped and
///    sorted longest-first so the greedy matcher prefers the phrase
///    over its constituents.
/// 2. [`build_lc_offset_map`] — a lowercased copy of `snippet` and a
///    table mapping lowercase byte offsets back to original byte
///    offsets. Lowercasing changes byte length per codepoint, so the
///    table is the only correct way to recover original-side spans.
/// 3. [`scan_for_match_ranges`] — greedy left-to-right scan over the
///    lowercase bytes for needle matches, emitting original-side
///    `Range<usize>`s.
///
/// Span construction then weaves the unhighlighted runs between the
/// match ranges.
fn highlight_snippet_spans(snippet: &str, query: &str) -> Vec<Span<'static>> {
    let needles = build_needles(query);
    if needles.is_empty() {
        return vec![Span::raw(snippet.to_string())];
    }
    let map = build_lc_offset_map(snippet);
    let ranges = scan_for_match_ranges(snippet, &map, &needles);
    weave_spans(snippet, &ranges)
}

/// Ordered, deduped match needles: full phrase + whitespace-split terms,
/// sorted longest-first so longest-match-wins is a simple loop.
fn build_needles(query: &str) -> Vec<String> {
    let phrase = query.trim().to_lowercase();
    if phrase.is_empty() {
        return Vec::new();
    }
    let mut v = vec![phrase.clone()];
    v.extend(
        phrase
            .split_whitespace()
            .filter(|t| *t != phrase)
            .map(|t| t.to_lowercase()),
    );
    v.sort_by_key(|s| std::cmp::Reverse(s.len()));
    v.dedup();
    v
}

/// Lowercased copy of `snippet` together with a `lc_byte → orig_byte`
/// table.
///
/// Lowercasing changes byte length per codepoint ('İ' → "i\u{307}"
/// grows, 'ẞ' → "ß" shrinks), so positions in the two strings don't
/// coincide. `lc_to_orig[i] = Some(orig)` marks lc byte offset `i` as
/// a char boundary corresponding to original byte offset `orig`; mid-
/// codepoint and lowercase-expansion offsets are `None`. The final
/// entry pins the end-of-string boundary so a needle that lands at the
/// very end has a well-defined `orig_end`.
struct LcOffsetMap {
    lc: String,
    /// Indexed by lowercase byte offset, length = `lc.len() + 1`.
    lc_to_orig: Vec<Option<usize>>,
}

fn build_lc_offset_map(snippet: &str) -> LcOffsetMap {
    let mut lc = String::with_capacity(snippet.len());
    let mut lc_to_orig: Vec<Option<usize>> = Vec::new();
    for (orig_byte, ch) in snippet.char_indices() {
        while lc_to_orig.len() < lc.len() {
            lc_to_orig.push(None);
        }
        lc_to_orig.push(Some(orig_byte));
        for lc_ch in ch.to_lowercase() {
            lc.push(lc_ch);
        }
    }
    while lc_to_orig.len() < lc.len() {
        lc_to_orig.push(None);
    }
    lc_to_orig.push(Some(snippet.len()));
    LcOffsetMap { lc, lc_to_orig }
}

/// Greedy left-to-right scan: at each char boundary in `map.lc`, try
/// each needle in order (longest first) and emit the original-side
/// match range. Returns ranges in left-to-right order, non-overlapping.
fn scan_for_match_ranges(
    snippet: &str,
    map: &LcOffsetMap,
    needles: &[String],
) -> Vec<std::ops::Range<usize>> {
    let lc_bytes = map.lc.as_bytes();
    let mut ranges: Vec<std::ops::Range<usize>> = Vec::new();
    let mut lc_cursor = 0usize;
    while lc_cursor < lc_bytes.len() {
        let Some(orig_cursor) = map.lc_to_orig.get(lc_cursor).copied().flatten() else {
            lc_cursor += 1;
            continue;
        };
        let matched = needles.iter().find_map(|n| {
            let lc_end = lc_cursor + n.len();
            if lc_end > lc_bytes.len() || &lc_bytes[lc_cursor..lc_end] != n.as_bytes() {
                return None;
            }
            map.lc_to_orig
                .get(lc_end)
                .copied()
                .flatten()
                .map(|orig_end| (n.len(), orig_end))
        });
        if let Some((lc_len, orig_end)) = matched {
            ranges.push(orig_cursor..orig_end);
            lc_cursor += lc_len;
        } else {
            // No match here — advance by the lowercase byte length of
            // the next codepoint so the cursor stays on a boundary.
            let lc_step: usize = match snippet[orig_cursor..].chars().next() {
                Some(c) => c.to_lowercase().map(|x| x.len_utf8()).sum(),
                None => break,
            };
            lc_cursor += lc_step.max(1);
        }
    }
    ranges
}

/// Weave the unhighlighted prefixes / suffixes between the highlighted
/// match `ranges` into a span list.
fn weave_spans(
    snippet: &str,
    ranges: &[std::ops::Range<usize>],
) -> Vec<Span<'static>> {
    let hl = Style::default()
        .bg(HL_BG)
        .fg(HL_FG)
        .add_modifier(Modifier::BOLD);
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut cursor = 0usize;
    for r in ranges {
        if cursor < r.start {
            spans.push(Span::raw(snippet[cursor..r.start].to_string()));
        }
        spans.push(Span::styled(snippet[r.start..r.end].to_string(), hl));
        cursor = r.end;
    }
    if cursor < snippet.len() {
        spans.push(Span::raw(snippet[cursor..].to_string()));
    }
    spans
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

    fn render(spans: &[Span<'static>]) -> String {
        spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn highlight_plain_ascii() {
        let spans = highlight_snippet_spans("Hello world", "world");
        assert_eq!(render(&spans), "Hello world");
        // "Hello " unhighlighted, "world" highlighted.
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].content.as_ref(), "Hello ");
        assert_eq!(spans[1].content.as_ref(), "world");
    }

    #[test]
    fn highlight_preserves_original_case() {
        let spans = highlight_snippet_spans("HeLLo WoRLd", "hello");
        assert_eq!(render(&spans), "HeLLo WoRLd");
        assert_eq!(spans[0].content.as_ref(), "HeLLo");
    }

    // Regression: lowercasing 'ẞ' (U+1E9E, 3 bytes UTF-8) yields "ß"
    // (2 bytes), so byte positions in `snippet` and its lowercase form
    // diverge. Previously this overflowed the lowercase slice.
    #[test]
    fn highlight_handles_shrinking_lowercase() {
        let spans = highlight_snippet_spans("STRAẞE", "stra");
        assert_eq!(render(&spans), "STRAẞE");
    }

    // Regression: lowercasing 'İ' (U+0130, 2 bytes) yields "i\u{307}"
    // (3 bytes), the other direction of length divergence.
    #[test]
    fn highlight_handles_growing_lowercase() {
        let spans = highlight_snippet_spans("İstanbul", "istanbul");
        assert_eq!(render(&spans), "İstanbul");
    }

    #[test]
    fn highlight_empty_query_passes_through() {
        let spans = highlight_snippet_spans("anything", "");
        assert_eq!(render(&spans), "anything");
        assert_eq!(spans.len(), 1);
    }

    #[test]
    fn highlight_longest_needle_wins() {
        // "foo bar" (full phrase) should win over "foo" / "bar" splits.
        let spans = highlight_snippet_spans("a foo bar b", "foo bar");
        assert_eq!(render(&spans), "a foo bar b");
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[1].content.as_ref(), "foo bar");
    }
}
