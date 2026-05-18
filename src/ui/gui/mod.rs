//! Win98/NT-flavoured cross-platform desktop GUI for `pdffff`.
//!
//! The GUI mirrors the TUI feature-for-feature:
//!
//! ```text
//!  ╔════════════════════════════════════════════════════════════════════╗
//!  ║ pdffff  /Users/foo/papers                          12 ok · ⠿ idx 3 ║
//!  ╟────────────────────────────────────────────────────────────────────╢
//!  ║  ❯ [ alpha synthesis▏          ]                            [ FZ ] ║
//!  ║  ─────────────────────────────────────────────────────────────────  ║
//!  ║  ┌ 1. paper.pdf ─────────────────────────────────── p.12 · #3 ───┐ ║
//!  ║  │ …matching snippet with [highlighted] terms…                   │ ║
//!  ║  └───────────────────────────────────────────────────────────────┘ ║
//!  ║                                                                    ║
//!  ║  ┌ 2. other.pdf ────────────────────── p. 1 · #0 · score 1247 ──┐ ║
//!  ║  │ …another snippet…                                             │ ║
//!  ║  └───────────────────────────────────────────────────────────────┘ ║
//!  ╟────────────────────────────────────────────────────────────────────╢
//!  ║ ↑↓ select   Tab mode   Enter open   Ctrl+U clear   Esc quit        ║
//!  ╚════════════════════════════════════════════════════════════════════╝
//! ```
//!
//! Layout
//! ------
//! * [`palette`] — Win95 colours + spacing constants.
//! * [`paint`]   — bevel / drop-shadow / gradient / etched-hr helpers.
//! * [`style`]   — global egui visuals stamping.
//! * [`keys`]    — shortcut consumption + dispatch to `GuiApp`.
//! * [`render`]  — every `render_*` / `paint_*` function.
//!
//! Shared kernel
//! -------------
//! Everything above the rendering layer is reused from the TUI:
//!
//! * [`crate::ui::search::SearchWorker`] runs queries on a background
//!   thread with a one-slot mailbox + stamp-based stale-result
//!   rejection.
//! * [`crate::ui::highlight::highlight_segments`] gives match-aware
//!   snippet/title segments.
//! * [`crate::ui::input::cycle_mode`], `word_erase`, `move_selection`
//!   own the input-edit primitives.
//! * [`crate::ui::spinner::frame_at`] picks the animation frame.
//!
//! Concurrency
//! -----------
//! The search worker publishes results into a polled mutex and pings
//! `Context::request_repaint` so the egui event loop wakes immediately
//! without forcing a global 60fps spin. The renderer also schedules a
//! cheap 100ms fallback repaint to keep the indexer-spinner animating
//! and the live status counters honest even when no input is arriving
//! and no search result is ready.
//!
//! Shutdown
//! --------
//! The window owns the [`WatchHandle`] and the [`SearchWorker`]. On
//! window close both are torn down inside `Drop` — best-effort, since
//! Drop swallows errors; for a definitive shutdown path use
//! [`run_gui`], which calls `stop` synchronously after `run_native`
//! returns and propagates errors to the caller.

mod keys;
mod paint;
mod palette;
mod render;
mod style;

use anyhow::{Context, Result, anyhow};
use eframe::egui;
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::app::{IndexProgress, WatchHandle};
use crate::query::{DISPLAY_LIMIT, Hit, QueryMode};
use crate::ui::launch::{OnPick, open_in_system_viewer};
use crate::ui::search::{SearchRequest, SearchWorker};

use palette::{FACE, PANEL_MARGIN};

/// Knobs for [`run_gui`]. Mirrors [`crate::tui::TuiOptions`] field-for-
/// field so a launcher can pick either frontend without remapping.
#[derive(Debug, Clone)]
pub struct GuiOptions {
    pub limit: usize,
    pub initial_mode: QueryMode,
    pub root: PathBuf,
    /// What Enter / double-click on a result does. Defaults to
    /// opening the file in the host's PDF viewer and keeping the
    /// window alive.
    pub on_pick: OnPick,
}

impl Default for GuiOptions {
    fn default() -> Self {
        Self {
            limit: DISPLAY_LIMIT,
            initial_mode: QueryMode::Fuzzy,
            root: PathBuf::new(),
            on_pick: OnPick::default(),
        }
    }
}

/// Run the GUI until the user closes the window or picks a hit.
///
/// Mirrors `tui::run_tui`'s contract: owns `handle`, tears down the
/// indexer threads on the way out, returns `Some(hit)` if the user
/// activated a result so the caller can hand it to the system PDF
/// viewer.
pub fn run_gui(handle: WatchHandle, opts: GuiOptions) -> Result<Option<Hit>> {
    // Cross-thread handoff for the "selector" mode: the GuiApp writes
    // the chosen hit here before requesting window close, and we read
    // it out after `run_native` returns. Stays `None` in OpenInViewer
    // mode.
    let chosen: Arc<Mutex<Option<Hit>>> = Arc::new(Mutex::new(None));
    let chosen_for_app = chosen.clone();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1020.0, 720.0])
            .with_min_inner_size([620.0, 380.0])
            .with_title("pdffff"),
        ..Default::default()
    };

    let opts_for_app = opts.clone();
    eframe::run_native(
        "pdffff",
        native_options,
        Box::new(move |cc| {
            style::apply_font_fallback(&cc.egui_ctx);
            style::apply_win98_visuals(&cc.egui_ctx);
            let app = GuiApp::new(handle, opts_for_app, chosen_for_app)?;
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow!("eframe::run_native: {e}"))?;

    let result = chosen.lock().take();
    Ok(result)
}

/// Live GUI state, owned by `eframe::run_native` for the lifetime of
/// the window.
pub(crate) struct GuiApp {
    // Shared kernel.
    pub(crate) handle: Option<WatchHandle>,
    pub(crate) worker: Option<SearchWorker>,
    pub(crate) progress: Arc<IndexProgress>,
    pub(crate) opts: GuiOptions,

    // UI state.
    pub(crate) query: String,
    pub(crate) mode: QueryMode,
    pub(crate) hits: Vec<Hit>,
    pub(crate) selected: Option<usize>,
    /// Selection at the end of the previous frame. When this differs
    /// from `selected` the renderer scrolls the new card into view.
    /// Tracking the *change* (rather than scrolling every frame) is
    /// what keeps user-initiated scroll-wheel input from being
    /// stomped on by the auto-scroll.
    pub(crate) prev_selected: Option<usize>,
    pub(crate) last_error: Option<String>,
    pub(crate) submitted_stamp: u64,
    /// Stamp of the most recent search result we *applied*. When this
    /// trails `submitted_stamp` a search is in flight (or the latest
    /// keystroke hasn't been picked up by the worker yet) — that's
    /// the signal we hand to [`is_searching`](Self::is_searching) to
    /// drive the prompt-position spinner.
    pub(crate) applied_stamp: u64,
    pub(crate) spinner_started: Instant,
    /// Whether we've already grabbed initial keyboard focus for the
    /// query input. After the first frame the user owns focus; we do
    /// not steal it back on every repaint.
    pub(crate) did_initial_focus: bool,

    /// In `OnPick::SelectAndExit` mode, written when the user
    /// activates a result so the outer `run_gui` can read it after
    /// `run_native` returns. Stays empty in `OpenInViewer` mode.
    pub(crate) chosen: Arc<Mutex<Option<Hit>>>,

    /// True once we've requested the system window to close.
    pub(crate) closing: bool,
}

impl GuiApp {
    fn new(
        handle: WatchHandle,
        opts: GuiOptions,
        chosen: Arc<Mutex<Option<Hit>>>,
    ) -> Result<Self> {
        let progress = handle.progress.clone();
        let worker =
            SearchWorker::spawn(handle.state.clone()).context("spawning GUI search worker")?;
        Ok(Self {
            handle: Some(handle),
            worker: Some(worker),
            progress,
            opts: opts.clone(),
            query: String::new(),
            mode: opts.initial_mode,
            hits: Vec::new(),
            selected: None,
            prev_selected: None,
            last_error: None,
            submitted_stamp: 0,
            applied_stamp: 0,
            spinner_started: Instant::now(),
            did_initial_focus: false,
            chosen,
            closing: false,
        })
    }

    fn shutdown(&mut self) -> Result<()> {
        if let Some(w) = self.worker.take() {
            w.stop();
        }
        if let Some(h) = self.handle.take() {
            h.stop()?;
        }
        Ok(())
    }

    pub(crate) fn submit_query(&mut self) {
        self.submitted_stamp = self.submitted_stamp.wrapping_add(1);
        if self.query.trim().is_empty() {
            self.hits.clear();
            self.selected = None;
            self.last_error = None;
            // An empty query "completes" immediately — there's no
            // worker round-trip — so mark this stamp as applied so
            // the prompt spinner doesn't keep spinning forever.
            self.applied_stamp = self.submitted_stamp;
            return;
        }
        if let Some(w) = &self.worker {
            w.submit(SearchRequest {
                stamp: self.submitted_stamp,
                query: self.query.clone(),
                mode: self.mode,
                limit: self.opts.limit,
            });
        }
    }

    /// True when the latest submitted query has not yet been answered
    /// by the worker. Drives the prompt-position spinner.
    pub(crate) fn is_searching(&self) -> bool {
        self.applied_stamp != self.submitted_stamp
    }

    fn drain_results(&mut self) {
        let Some(worker) = &self.worker else { return };
        let Some(result) = worker.take_result() else {
            return;
        };
        if result.stamp != self.submitted_stamp {
            return;
        }
        self.applied_stamp = result.stamp;
        match result.hits {
            Ok(hits) => {
                self.hits = hits;
                self.last_error = None;
                self.selected = if self.hits.is_empty() { None } else { Some(0) };
            }
            Err(err) => {
                // Keep the old hits visible so an invalid regex while
                // typing doesn't blank the screen on every keystroke.
                self.last_error = Some(format!("{err:#}"));
            }
        }
    }

    /// Activate the currently-selected result. Branches on
    /// `opts.on_pick`:
    ///
    /// * `OpenInViewer` (default) — hand the path to the host's PDF
    ///   viewer and keep the window open. Errors surface in the
    ///   in-screen error pill (the GUI owns the window and cannot
    ///   print to stderr).
    /// * `SelectAndExit` — stash the hit in `self.chosen` and request
    ///   window close so `run_gui` can read it back out after
    ///   `run_native` returns.
    pub(crate) fn pick_selected(&mut self) {
        let Some(i) = self.selected else { return };
        let Some(hit) = self.hits.get(i) else { return };
        match self.opts.on_pick {
            OnPick::OpenInViewer => match open_in_system_viewer(&hit.path) {
                Ok(()) => self.last_error = None,
                Err(err) => {
                    self.last_error =
                        Some(format!("could not open {}: {err:#}", hit.path));
                }
            },
            OnPick::SelectAndExit => {
                *self.chosen.lock() = Some(hit.clone());
                self.closing = true;
            }
        }
    }
}

impl Drop for GuiApp {
    fn drop(&mut self) {
        // Best-effort: if shutdown errors here we have nowhere to
        // propagate them. The supervised path is via `run_gui` after
        // `run_native` returns.
        let _ = self.shutdown();
    }
}

impl eframe::App for GuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Keystrokes that move state without touching widgets.
        keys::handle_global_keys(self, ctx);

        // Pull any finished search before we lay out the result list.
        self.drain_results();

        // The whole window is one panel; we paint our own bevels and
        // chrome inside it.
        egui::CentralPanel::default()
            .frame(
                egui::Frame::none()
                    .fill(FACE)
                    .inner_margin(egui::Margin::same(PANEL_MARGIN)),
            )
            .show(ctx, |ui| render::render(self, ui));

        // Keep the indexer-spinner moving and the live counters honest
        // even when no input is arriving. ~10fps idle is plenty and
        // egui's diff-based renderer makes it effectively free.
        if self.closing {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        } else {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ui::input::word_erase;

    // The GUI's editing primitives now live in `ui::input` and are
    // tested there. We keep parity tests for the three trailing-word-
    // erase scenarios as a regression net — they previously lived in
    // this module and the GUI's Ctrl+W binding still flows through
    // them.

    #[test]
    fn word_erase_drops_trailing_word() {
        let mut s = String::from("alpha beta gamma");
        word_erase(&mut s);
        assert_eq!(s, "alpha beta ");
        word_erase(&mut s);
        assert_eq!(s, "alpha ");
        word_erase(&mut s);
        assert_eq!(s, "");
    }

    #[test]
    fn word_erase_handles_trailing_whitespace() {
        let mut s = String::from("alpha beta   ");
        word_erase(&mut s);
        assert_eq!(s, "alpha ");
    }

    #[test]
    fn word_erase_on_empty_is_noop() {
        let mut s = String::new();
        word_erase(&mut s);
        assert_eq!(s, "");
    }
}
