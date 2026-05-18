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
//!  ╰─ ↑↓ select · Tab mode · Enter open · Ctrl+U clear · Esc quit ─────────╯
//! ```
//!
//! The TUI owns a [`WatchHandle`] from [`crate::app::run_watch`]: the
//! handle's background threads keep the index live while the UI runs
//! queries against the same `IndexState`. The writer thread persists
//! every successful mutation to SQLite synchronously, so the only
//! work shutdown has to do is signal the threads to drain and join.
//!
//! Layout
//! ------
//! * [`palette`]  — Bertin-disciplined colour roles.
//! * [`layout`]   — pure card-geometry / scroll maths.
//! * [`render`]   — `render_*` functions; takes `&AppState`.
//! * [`keys`]     — keystroke dispatch + worker submission.
//! * [`term`]     — raw mode / alt screen / panic hook plumbing.
//!
//! Concurrency
//! -----------
//! Searches do not run on the input thread. A dedicated
//! [`SearchWorker`] thread owns an `Arc<IndexState>` and a one-slot
//! mailbox: every keystroke / mode change overwrites the slot, so a
//! burst of input coalesces into the *latest* query rather than
//! running a full search per key. The worker publishes its result
//! into a `Mutex<Option<…>>`; the render loop polls that mutex each
//! iteration. A monotonic stamp on every request is echoed back so we
//! drop results for queries the user has already moved past.
//!
//! Shutdown
//! --------
//! All four exit keys (`Ctrl+C`, `Ctrl+D`, `Ctrl+Q`, `Esc`) take the
//! same path: restore the terminal, then call [`WatchHandle::stop`]
//! to signal-and-join the coordinator + writer threads. The writer
//! commits every mutation as its own SQLite transaction (`WAL` mode,
//! `synchronous=NORMAL`), so there is no buffered state to flush on
//! quit.

mod keys;
mod layout;
mod palette;
mod render;
mod term;

use anyhow::{Context, Result};
use crossterm::event::{self, Event};
use ratatui::{Terminal, backend::CrosstermBackend, widgets::ListState};
use std::io::Stdout;
use std::path::PathBuf;
use std::time::Instant;

use crate::app::{IndexProgress, ProgressSnapshot, WatchHandle};
use crate::query::{DISPLAY_LIMIT, Hit, QueryMode};
use crate::ui::launch::OnPick;
use crate::ui::search::SearchWorker;
use crate::ui::spinner::TICK;

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
    /// What Enter on a result does. Defaults to opening the file in
    /// the host's PDF viewer and keeping the session alive.
    pub on_pick: OnPick,
}

impl Default for TuiOptions {
    fn default() -> Self {
        Self {
            limit: DISPLAY_LIMIT,
            initial_mode: QueryMode::Fuzzy,
            root: PathBuf::new(),
            on_pick: OnPick::default(),
        }
    }
}

/// Run the TUI until the user quits or an unrecoverable IO error
/// occurs. Owns `handle`: calls [`WatchHandle::stop`] on the way out
/// so the index threads always shut down cleanly, even on panic.
///
/// Behaviour on Enter depends on `opts.on_pick`:
/// * [`OnPick::OpenInViewer`] (default) — hand the path to the host's
///   PDF viewer via [`crate::ui::launch::open_in_system_viewer`] and
///   keep the session alive. Errors surface in the in-screen error
///   pill rather than corrupting the alternate-screen output.
///   `run_tui` returns `Ok(None)` on plain quit.
/// * [`OnPick::SelectAndExit`] — capture the chosen `Hit`, exit the
///   session, and return `Ok(Some(hit))` so the launcher can print
///   the path to stdout for shell pipelines.
pub fn run_tui(handle: WatchHandle, opts: TuiOptions) -> Result<Option<Hit>> {
    let mut terminal = term::setup().context("entering TUI terminal mode")?;
    term::install_panic_hook();

    let loop_result = main_loop(&mut terminal, &handle, &opts);

    // Always restore the terminal before doing anything slow (like
    // joining background threads). If teardown itself fails, prefer
    // to surface the main-loop result.
    let teardown = term::restore(&mut terminal);

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
pub(crate) struct AppState {
    pub(crate) query: String,
    pub(crate) mode: QueryMode,
    pub(crate) hits: Vec<Hit>,
    pub(crate) list_state: ListState,
    /// Last query error (e.g. invalid regex) — rendered under the
    /// input line so the user can see what's wrong without losing
    /// their typed text.
    pub(crate) last_error: Option<String>,
    /// Monotonic stamp bumped on every query / mode edit. Submitted
    /// to the worker and echoed back in the result so we can drop
    /// hits for queries the user has already moved past.
    pub(crate) submitted_stamp: u64,
    /// Stamp of the result currently displayed in `hits`. When this
    /// trails `submitted_stamp` a search is in flight on the worker
    /// — we leave the previous hits on screen rather than blanking
    /// the list so the UI never goes empty between keystrokes.
    pub(crate) applied_stamp: u64,
    /// Wall-clock of the last spinner advance.
    pub(crate) spinner_started: Instant,
    /// Index of the first card visible in the results area. Tracks
    /// scroll position across renders so the view doesn't jump when
    /// the terminal repaints.
    pub(crate) scroll_top: usize,
    /// True once the user has pressed one of the quit keys.
    pub(crate) should_quit: bool,
    /// In [`OnPick::SelectAndExit`] mode, the hit the user picked on
    /// Enter.
    pub(crate) chosen: Option<Hit>,
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
            chosen: None,
        }
    }

    /// True when the latest submitted query has not yet been answered
    /// by the worker. Drives the prompt-position spinner.
    pub(crate) fn is_searching(&self) -> bool {
        self.applied_stamp != self.submitted_stamp
    }
}

fn main_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    handle: &WatchHandle,
    opts: &TuiOptions,
) -> Result<Option<Hit>> {
    let worker = SearchWorker::spawn(handle.state.clone())
        .context("spawning TUI search worker")?;
    // Helper so worker shutdown happens on every exit path (including
    // `?` propagation), without a Drop guard.
    let outcome = run_event_loop(terminal, &worker, &handle.progress, opts);
    worker.stop();
    outcome
}

fn run_event_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    worker: &SearchWorker,
    progress: &IndexProgress,
    opts: &TuiOptions,
) -> Result<Option<Hit>> {
    let mut state = AppState::new(opts.initial_mode);

    // Snapshot of indexer counters at the last tick, so we can redraw
    // the screen when the indexer makes progress in the background
    // (otherwise the status bar would only update on keystroke).
    let mut last_snapshot: ProgressSnapshot = progress.snapshot();
    terminal.draw(|f| render::render(f, &mut state, opts, last_snapshot))?;

    while !state.should_quit {
        let had_event = event::poll(TICK)?;
        if had_event {
            if let Event::Key(key) = event::read()? {
                keys::handle_key(key, &mut state, worker, opts);
            }
        }
        let got_result = keys::drain_results(&mut state, worker);
        let snap = progress.snapshot();
        // Redraw if anything visible could have changed: user input,
        // a fresh search result, the indexer counters ticked, or the
        // extractor pool is still busy (spinner needs to keep moving).
        if had_event || got_result || snap != last_snapshot || snap.pending > 0 {
            terminal.draw(|f| render::render(f, &mut state, opts, snap))?;
            last_snapshot = snap;
        }
    }

    Ok(state.chosen)
}

#[cfg(test)]
mod tests {
    use super::layout::{compute_scroll_top, visible_card_count};
    use super::palette::{PRIMARY, match_hl_style};
    use super::render::highlight_spans;
    use ratatui::style::Style;

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
        // Stale prev gets clamped to max_top so we never paint past
        // the end of the list.
        assert_eq!(compute_scroll_top(19, 20, 5, 999), 15);
        // List shorter than the window: scroll_top is pinned at 0.
        assert_eq!(compute_scroll_top(2, 3, 5, 4), 0);
        // Degenerate inputs don't panic.
        assert_eq!(compute_scroll_top(0, 0, 5, 0), 0);
        assert_eq!(compute_scroll_top(0, 5, 0, 0), 0);
    }
}
