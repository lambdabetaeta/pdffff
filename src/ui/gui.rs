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
//! The faux 3D bevels (light-top/left, dark-bottom/right for raised
//! surfaces, inverted for sunken ones) are painted via the egui
//! `Painter` API; egui's default rounded `Frame` chrome is suppressed
//! globally in [`apply_win98_visuals`].
//!
//! Shared kernel
//! -------------
//! Everything above the rendering layer is reused from the TUI:
//!
//! * [`crate::ui::search::SearchWorker`] runs queries on a background
//!   thread with the same one-slot mailbox + stamp-based stale-result
//!   rejection.
//! * [`crate::ui::highlight::highlight_segments`] gives match-aware
//!   snippet/title segments; the GUI maps each `SegmentKind` to a
//!   `RichText` run with the appropriate background colour.
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

use anyhow::{Context, Result, anyhow};
use eframe::egui::{
    self, Color32, FontFamily, FontId, Painter, Pos2, Rect, RichText, Sense, Stroke,
    TextEdit, TextStyle, Ui, Vec2,
};
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use crate::app::{IndexProgress, WatchHandle};
use crate::query::{DISPLAY_LIMIT, Hit, QueryMode};
use crate::ui::highlight::{SegmentKind, highlight_segments};
use crate::ui::search::{SearchRequest, SearchWorker};

// ─────────────────────── Win98/NT palette ───────────────────────────
//
// One hue, one job (same Bertin discipline as the TUI palette). The
// 3D-button-face grey is the body of the world; navy blue is reserved
// for chrome and focus; yellow-on-black is reserved exclusively for
// query matches; red is reserved for errors.

/// Classic Windows 9x/NT "3D button face" — the body colour of every
/// non-control surface.
const FACE: Color32 = Color32::from_rgb(0xc0, 0xc0, 0xc0);
/// Highlight edge of a raised bevel (top + left).
const BEVEL_LIGHT: Color32 = Color32::from_rgb(0xff, 0xff, 0xff);
/// Shadow edge of a raised bevel (bottom + right). Inverted for sunken
/// bevels (text inputs, the inset card body).
const BEVEL_DARK: Color32 = Color32::from_rgb(0x80, 0x80, 0x80);
/// Title bar / focus highlight. The Windows "Active Window" navy.
const TITLE_BLUE: Color32 = Color32::from_rgb(0x00, 0x00, 0x80);
/// Hit highlight, identical role to the TUI's yellow-on-black.
const MATCH_BG: Color32 = Color32::from_rgb(0xff, 0xff, 0x00);
const MATCH_FG: Color32 = Color32::BLACK;
/// Error pill background.
const ERROR_BG: Color32 = Color32::from_rgb(0x80, 0x00, 0x00);

const SPINNER_FRAMES: [&str; 10] =
    ["|", "/", "-", "\\", "|", "/", "-", "\\", "*", "+"];

/// Knobs for [`run_gui`]. Mirrors [`crate::tui::TuiOptions`] field-for-
/// field so a launcher can pick either frontend without remapping.
#[derive(Debug, Clone)]
pub struct GuiOptions {
    pub limit: usize,
    pub initial_mode: QueryMode,
    pub root: PathBuf,
}

impl Default for GuiOptions {
    fn default() -> Self {
        Self {
            limit: DISPLAY_LIMIT,
            initial_mode: QueryMode::Fuzzy,
            root: PathBuf::new(),
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
    let chosen: Arc<Mutex<Option<Hit>>> = Arc::new(Mutex::new(None));
    let chosen_for_app = chosen.clone();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 640.0])
            .with_min_inner_size([520.0, 320.0])
            .with_title("pdffff"),
        ..Default::default()
    };

    let opts_for_app = opts.clone();
    eframe::run_native(
        "pdffff",
        native_options,
        Box::new(move |cc| {
            apply_win98_visuals(&cc.egui_ctx);
            let app = GuiApp::new(handle, opts_for_app, chosen_for_app, cc.egui_ctx.clone())?;
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow!("eframe::run_native: {e}"))?;

    let result = chosen.lock().take();
    Ok(result)
}

// ─────────────────────────── styling ────────────────────────────────

/// Stamp every egui surface with the Win98/NT look. Called once at
/// `CreationContext`; `set_visuals` / `set_style` apply globally for
/// the life of the window.
///
/// Win9x had sharp 1px corners on every chrome element, so every
/// rounding parameter goes to zero. egui's stock palette assumes a
/// modern flat-grey-on-grey aesthetic, so we replace each role colour
/// rather than tweaking; that way no remaining widget falls back to a
/// stock blue/grey that breaks the period look.
fn apply_win98_visuals(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::light();

    // Panel + window body — the canonical 3D button face.
    visuals.window_fill = FACE;
    visuals.panel_fill = FACE;
    visuals.faint_bg_color = FACE;
    // Inset (sunken) areas use white for the input wells.
    visuals.extreme_bg_color = Color32::WHITE;

    // Kill all rounding — Win9x is square.
    let zero = egui::Rounding::ZERO;
    visuals.window_rounding = zero;
    visuals.menu_rounding = zero;
    visuals.window_shadow = egui::epaint::Shadow::NONE;
    visuals.popup_shadow = egui::epaint::Shadow::NONE;

    // Text colour everywhere — pure black on grey, pure black on white.
    let black_fg = Stroke::new(1.0, Color32::BLACK);
    let no_stroke = Stroke::NONE;

    for w in [
        &mut visuals.widgets.noninteractive,
        &mut visuals.widgets.inactive,
        &mut visuals.widgets.hovered,
        &mut visuals.widgets.active,
        &mut visuals.widgets.open,
    ] {
        w.rounding = zero;
        w.bg_fill = FACE;
        w.weak_bg_fill = FACE;
        w.bg_stroke = no_stroke;
        w.fg_stroke = black_fg;
        w.expansion = 0.0;
    }
    // Hovered/active keep the same grey body — the bevel does the work,
    // not a colour shift.

    // Selection colour — the canonical Windows navy.
    visuals.selection.bg_fill = TITLE_BLUE;
    visuals.selection.stroke = Stroke::new(1.0, Color32::WHITE);

    ctx.set_visuals(visuals);

    // Slightly tighter spacing than egui's defaults; Win9x dialogs are
    // dense.
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = Vec2::new(6.0, 4.0);
    style.spacing.button_padding = Vec2::new(8.0, 3.0);
    style.spacing.interact_size = Vec2::new(24.0, 22.0);
    // A single readable proportional font at a modest size — egui's
    // default font (Ubuntu Light) at 13pt is the cleanest "MS Sans
    // Serif-adjacent" we can ship without bundling a non-free font.
    style.text_styles.insert(
        TextStyle::Body,
        FontId::new(13.0, FontFamily::Proportional),
    );
    style.text_styles.insert(
        TextStyle::Button,
        FontId::new(13.0, FontFamily::Proportional),
    );
    style.text_styles.insert(
        TextStyle::Heading,
        FontId::new(15.0, FontFamily::Proportional),
    );
    style.text_styles.insert(
        TextStyle::Monospace,
        FontId::new(12.0, FontFamily::Monospace),
    );
    ctx.set_style(style);
}

/// Paint a 1px Win9x bevel on the outer perimeter of `rect`.
///
/// `raised = true` paints buttons / toolbars (light top-left, dark
/// bottom-right); `raised = false` paints text inputs and the inset
/// card body (dark top-left, light bottom-right).
fn paint_bevel(painter: &Painter, rect: Rect, raised: bool) {
    let (light, dark) = if raised {
        (BEVEL_LIGHT, BEVEL_DARK)
    } else {
        (BEVEL_DARK, BEVEL_LIGHT)
    };
    let s = Stroke::new(1.0, light);
    let d = Stroke::new(1.0, dark);
    // Lines are painted *inside* the rectangle's footprint so the
    // bevel never overlaps neighbouring widgets.
    let top_l = rect.left_top();
    let top_r = Pos2::new(rect.right() - 1.0, rect.top());
    let bot_l = Pos2::new(rect.left(), rect.bottom() - 1.0);
    let bot_r = Pos2::new(rect.right() - 1.0, rect.bottom() - 1.0);
    painter.line_segment([top_l, top_r], s);
    painter.line_segment([top_l, bot_l], s);
    painter.line_segment([bot_l, bot_r], d);
    painter.line_segment([top_r, bot_r], d);
}

// ─────────────────────────── app state ──────────────────────────────

struct GuiApp {
    // Shared kernel.
    handle: Option<WatchHandle>,
    worker: Option<SearchWorker>,
    progress: Arc<IndexProgress>,
    opts: GuiOptions,

    // UI state.
    query: String,
    mode: QueryMode,
    hits: Vec<Hit>,
    selected: Option<usize>,
    /// Selection at the end of the previous frame. When this differs
    /// from `selected` the renderer scrolls the new card into view.
    /// Tracking the *change* (rather than scrolling every frame) is
    /// what keeps user-initiated scroll-wheel input from being
    /// stomped on by the auto-scroll.
    prev_selected: Option<usize>,
    last_error: Option<String>,
    submitted_stamp: u64,
    spinner_started: Instant,
    /// Whether we've already grabbed initial keyboard focus for the
    /// query input. After the first frame the user owns focus; we do
    /// not steal it back on every repaint.
    did_initial_focus: bool,

    // Communication out: written when the user activates a result.
    chosen: Arc<Mutex<Option<Hit>>>,

    // True once we've requested the system window to close.
    closing: bool,
}

impl GuiApp {
    fn new(
        handle: WatchHandle,
        opts: GuiOptions,
        chosen: Arc<Mutex<Option<Hit>>>,
        _ctx: egui::Context,
    ) -> Result<Self> {
        let progress = handle.progress.clone();
        let state = handle.state.clone();
        let worker = SearchWorker::spawn(state).context("spawning GUI search worker")?;
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

    fn submit_query(&mut self) {
        self.submitted_stamp = self.submitted_stamp.wrapping_add(1);
        if self.query.trim().is_empty() {
            self.hits.clear();
            self.selected = None;
            self.last_error = None;
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

    fn drain_results(&mut self) {
        let Some(worker) = &self.worker else { return };
        let Some(result) = worker.take_result() else {
            return;
        };
        if result.stamp != self.submitted_stamp {
            return;
        }
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

    fn cycle_mode(&mut self) {
        self.mode = match self.mode {
            QueryMode::Literal => QueryMode::Regex,
            QueryMode::Regex => QueryMode::Fuzzy,
            QueryMode::Fuzzy => QueryMode::Literal,
        };
        self.submit_query();
    }

    fn move_selection(&mut self, delta: isize) {
        if self.hits.is_empty() {
            self.selected = None;
            return;
        }
        let n = self.hits.len() as isize;
        let cur = self.selected.map(|i| i as isize).unwrap_or(0);
        let next = (cur + delta).clamp(0, n - 1);
        self.selected = Some(next as usize);
    }

    fn pick_selected(&mut self) {
        if let Some(i) = self.selected {
            if let Some(hit) = self.hits.get(i) {
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
        self.handle_global_keys(ctx);

        // Pull any finished search before we lay out the result list.
        self.drain_results();

        // The whole window is one panel; we paint our own bevels and
        // chrome inside it.
        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(FACE).inner_margin(egui::Margin::same(8.0)))
            .show(ctx, |ui| self.render(ui));

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

// ─────────────────────────── input ──────────────────────────────────

impl GuiApp {
    fn handle_global_keys(&mut self, ctx: &egui::Context) {
        // Consume each shortcut from egui's event queue *before* any
        // widget processes it. Critical for Tab (which egui otherwise
        // routes to focus traversal) and the Ctrl+U / Ctrl+W edit
        // shortcuts (which the TextEdit would otherwise treat as
        // characters).
        let (esc, enter, up, down, tab, page_up, page_down, ctrl_u, ctrl_w) = ctx
            .input_mut(|i| {
                let none = egui::Modifiers::NONE;
                let ctrl = egui::Modifiers::COMMAND;
                (
                    i.consume_key(none, egui::Key::Escape),
                    i.consume_key(none, egui::Key::Enter),
                    i.consume_key(none, egui::Key::ArrowUp),
                    i.consume_key(none, egui::Key::ArrowDown),
                    i.consume_key(none, egui::Key::Tab),
                    i.consume_key(none, egui::Key::PageUp),
                    i.consume_key(none, egui::Key::PageDown),
                    i.consume_key(ctrl, egui::Key::U),
                    i.consume_key(ctrl, egui::Key::W),
                )
            });

        if esc {
            self.closing = true;
        }
        if enter {
            self.pick_selected();
        }
        if up {
            self.move_selection(-1);
        }
        if down {
            self.move_selection(1);
        }
        if page_up {
            self.move_selection(-10);
        }
        if page_down {
            self.move_selection(10);
        }
        if tab {
            self.cycle_mode();
        }
        if ctrl_u {
            self.query.clear();
            self.submit_query();
        }
        if ctrl_w {
            word_erase(&mut self.query);
            self.submit_query();
        }
    }
}

/// Drop the trailing whitespace-bounded word from `q`. Mirrors the
/// TUI's `word_erase` (readline/shell convention).
fn word_erase(q: &mut String) {
    let trimmed_end = q.trim_end();
    let cut_to = trimmed_end
        .rfind(char::is_whitespace)
        .map(|i| i + 1)
        .unwrap_or(0);
    q.truncate(cut_to);
}

// ─────────────────────────── rendering ──────────────────────────────

impl GuiApp {
    fn render(&mut self, ui: &mut Ui) {
        self.render_titlebar(ui);
        ui.add_space(6.0);
        self.render_input(ui);
        ui.add_space(4.0);
        self.render_separator_or_error(ui);
        ui.add_space(6.0);
        self.render_results(ui);
        self.render_help_bar(ui);
    }

    /// Faux Win9x title strip — rendered inside the client area so the
    /// system window decorations stay clickable but the *content*
    /// announces the look. Carries the brand pill, the watched root,
    /// and the live indexer counters.
    fn render_titlebar(&self, ui: &mut Ui) {
        let snap = snapshot_progress(&self.progress);
        let height = 22.0;
        let (rect, _) = ui.allocate_exact_size(
            Vec2::new(ui.available_width(), height),
            Sense::hover(),
        );
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, TITLE_BLUE);

        // Left: "pdffff <root>".
        let brand = format!("  pdffff  ");
        let body_font = FontId::new(13.0, FontFamily::Proportional);
        let brand_galley = painter.layout_no_wrap(brand, body_font.clone(), Color32::WHITE);
        let brand_pos = Pos2::new(rect.left() + 4.0, rect.center().y - brand_galley.size().y / 2.0);
        painter.galley(brand_pos, brand_galley.clone(), Color32::WHITE);

        let root_text = format!("{}", self.opts.root.display());
        let root_galley =
            painter.layout_no_wrap(root_text, body_font.clone(), Color32::WHITE);
        let root_pos = Pos2::new(
            brand_pos.x + brand_galley.size().x + 4.0,
            rect.center().y - root_galley.size().y / 2.0,
        );
        painter.galley(root_pos, root_galley, Color32::WHITE);

        // Right: status counters + spinner.
        let spinner_idx = (self.spinner_started.elapsed().as_millis() / 100) as usize
            % SPINNER_FRAMES.len();
        let activity = if snap.pending > 0 {
            format!("{} indexing {}", SPINNER_FRAMES[spinner_idx], snap.pending)
        } else {
            "idle".to_string()
        };
        let counters = format!(
            "{} ok · {} empty · {} err · {} del · {}  ",
            snap.ok, snap.empty, snap.error, snap.deleted, activity,
        );
        let counters_galley =
            painter.layout_no_wrap(counters, body_font, Color32::WHITE);
        let counters_pos = Pos2::new(
            rect.right() - counters_galley.size().x - 4.0,
            rect.center().y - counters_galley.size().y / 2.0,
        );
        painter.galley(counters_pos, counters_galley, Color32::WHITE);
    }

    /// Query input + mode pill. The input is a sunken text well with a
    /// "❯" prompt; the mode pill is a raised button that cycles the
    /// mode on click.
    fn render_input(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("❯").color(Color32::BLACK).strong());

            let edit = TextEdit::singleline(&mut self.query)
                .desired_width(ui.available_width() - 80.0)
                .frame(false)
                .text_color(Color32::BLACK)
                .margin(Vec2::new(4.0, 3.0));
            let edit_rect_before = ui.available_rect_before_wrap();
            let resp = ui.add(edit);
            // Paint the sunken well behind the text.
            let well = resp.rect.expand2(Vec2::new(0.0, 0.0));
            // We need to draw the bevel *behind* the text — easiest is
            // to use the painter on a layer below. Since egui paints
            // bottom-up in order of widget addition, the bevel painted
            // here lands on top. To avoid that, paint a filled white
            // rect *and* the bevel *before* the TextEdit by allocating
            // the area first; but that requires a layout dance. The
            // simpler, cheap-to-reason about path: paint the well via
            // an early-layer painter that targets the same rect.
            let painter = ui.ctx().layer_painter(egui::LayerId::background());
            painter.rect_filled(well, 0.0, Color32::WHITE);
            paint_bevel(&painter, well, false);

            if resp.changed() {
                self.submit_query();
            }
            // Grab focus exactly once at startup so the user can begin
            // typing without first clicking the input. After that, the
            // user owns focus — we don't steal it back on every
            // repaint.
            if !self.did_initial_focus {
                resp.request_focus();
                self.did_initial_focus = true;
            }
            // Keep the rect-before-wrap usable downstream.
            let _ = edit_rect_before;

            ui.add_space(8.0);
            // Mode pill — raised button look.
            let label = match self.mode {
                QueryMode::Literal => " LIT ",
                QueryMode::Regex => " RE ",
                QueryMode::Fuzzy => " FZ ",
            };
            let (pill_rect, pill_resp) = ui.allocate_exact_size(
                Vec2::new(46.0, 22.0),
                Sense::click(),
            );
            let pp = ui.painter_at(pill_rect);
            pp.rect_filled(pill_rect, 0.0, FACE);
            paint_bevel(&pp, pill_rect, !pill_resp.is_pointer_button_down_on());
            let body_font = FontId::new(13.0, FontFamily::Proportional);
            let galley = pp.layout_no_wrap(label.to_string(), body_font, Color32::BLACK);
            let p = Pos2::new(
                pill_rect.center().x - galley.size().x / 2.0,
                pill_rect.center().y - galley.size().y / 2.0,
            );
            pp.galley(p, galley, Color32::BLACK);
            if pill_resp.clicked() {
                self.cycle_mode();
            }
        });
    }

    fn render_separator_or_error(&self, ui: &mut Ui) {
        if let Some(err) = &self.last_error {
            ui.horizontal(|ui| {
                // Red "error" pill.
                let label_text = " error ";
                let body_font = FontId::new(13.0, FontFamily::Proportional);
                let (label_rect, _) = ui.allocate_exact_size(
                    Vec2::new(56.0, 20.0),
                    Sense::hover(),
                );
                let painter = ui.painter_at(label_rect);
                painter.rect_filled(label_rect, 0.0, ERROR_BG);
                let galley = painter.layout_no_wrap(
                    label_text.to_string(),
                    body_font,
                    Color32::WHITE,
                );
                let p = Pos2::new(
                    label_rect.center().x - galley.size().x / 2.0,
                    label_rect.center().y - galley.size().y / 2.0,
                );
                painter.galley(p, galley, Color32::WHITE);
                ui.add_space(4.0);
                ui.label(RichText::new(err).color(ERROR_BG));
            });
        } else {
            // Thin grey horizontal rule, drawn as two adjacent 1px
            // lines to mirror the Win9x "etched groupbox" separator.
            let (rect, _) = ui.allocate_exact_size(
                Vec2::new(ui.available_width(), 2.0),
                Sense::hover(),
            );
            let painter = ui.painter_at(rect);
            painter.line_segment(
                [rect.left_top(), rect.right_top()],
                Stroke::new(1.0, BEVEL_DARK),
            );
            painter.line_segment(
                [
                    Pos2::new(rect.left(), rect.top() + 1.0),
                    Pos2::new(rect.right(), rect.top() + 1.0),
                ],
                Stroke::new(1.0, BEVEL_LIGHT),
            );
        }
    }

    fn render_results(&mut self, ui: &mut Ui) {
        let total = self.hits.len();
        if total == 0 {
            let msg = if self.query.trim().is_empty() {
                "type a query to search the index"
            } else {
                "no hits"
            };
            ui.vertical_centered(|ui| {
                ui.add_space(20.0);
                ui.label(RichText::new(msg).italics().color(BEVEL_DARK));
            });
            self.prev_selected = self.selected;
            return;
        }
        let selected = self.selected;
        // Only auto-scroll on the frame where selection changed; on
        // every other frame the user owns the scroll position.
        let scroll_target = if selected != self.prev_selected {
            selected
        } else {
            None
        };
        let query = self.query.clone();
        let mode = self.mode;
        // The result list is scrollable. We reserve room at the bottom
        // for the help bar (which is rendered *after* the results, so
        // its height is taken from the layout cursor).
        let available_h = ui.available_height() - 28.0;
        let mut activated: Option<usize> = None;
        egui::ScrollArea::vertical()
            .max_height(available_h.max(60.0))
            .auto_shrink([false; 2])
            .show(ui, |ui| {
                for i in 0..total {
                    let hit = &self.hits[i];
                    let is_sel = selected == Some(i);
                    let resp = render_hit_card(ui, i, hit, &query, mode, is_sel);
                    if resp.clicked() {
                        activated = Some(i);
                    }
                    if scroll_target == Some(i) {
                        resp.scroll_to_me(Some(egui::Align::Center));
                    }
                    ui.add_space(4.0);
                }
            });
        if let Some(i) = activated {
            self.selected = Some(i);
            self.pick_selected();
        }
        self.prev_selected = self.selected;
    }

    fn render_help_bar(&self, ui: &mut Ui) {
        ui.add_space(4.0);
        let (rect, _) = ui.allocate_exact_size(
            Vec2::new(ui.available_width(), 20.0),
            Sense::hover(),
        );
        let painter = ui.painter_at(rect);
        // Two-tone etched rule across the top of the strip.
        painter.line_segment(
            [rect.left_top(), rect.right_top()],
            Stroke::new(1.0, BEVEL_DARK),
        );
        painter.line_segment(
            [
                Pos2::new(rect.left(), rect.top() + 1.0),
                Pos2::new(rect.right(), rect.top() + 1.0),
            ],
            Stroke::new(1.0, BEVEL_LIGHT),
        );
        let help =
            "↑↓ select   Tab mode   Enter open   Ctrl+U clear   Ctrl+W word-erase   Esc quit";
        let font = FontId::new(12.0, FontFamily::Proportional);
        let galley =
            painter.layout_no_wrap(help.to_string(), font, BEVEL_DARK);
        let p = Pos2::new(
            rect.left() + 4.0,
            rect.center().y - galley.size().y / 2.0 + 1.0,
        );
        painter.galley(p, galley, BEVEL_DARK);
    }
}

/// One result card. Returns the click response so the caller can pick
/// it. The card is a raised bevelled tile when unselected; the
/// selected card replaces the top edge with a navy bar to mirror the
/// TUI's magenta-bordered selection while keeping the surrounding
/// chrome consistent.
fn render_hit_card(
    ui: &mut Ui,
    i: usize,
    hit: &Hit,
    query: &str,
    mode: QueryMode,
    selected: bool,
) -> egui::Response {
    let meta = match mode {
        QueryMode::Fuzzy => format!(
            "p.{} · #{} · score {:.2}",
            hit.page_no, hit.chunk_ord, hit.score,
        ),
        QueryMode::Literal | QueryMode::Regex => {
            format!("p.{} · #{}", hit.page_no, hit.chunk_ord)
        }
    };

    let body_font = FontId::new(13.0, FontFamily::Proportional);

    // Reserve space: title row (20px) + snippet row (40px wrapping) +
    // 8px padding top/bottom + 1px bevel.
    let desired = Vec2::new(ui.available_width(), 64.0);
    let (rect, resp) = ui.allocate_exact_size(desired, Sense::click());
    let painter = ui.painter_at(rect);

    // Card body — raised tile.
    painter.rect_filled(rect, 0.0, FACE);
    paint_bevel(&painter, rect, true);

    // Selection accent: a 3px navy strip along the top, inside the
    // bevel. Matches the TUI's "border-only" selection cue while
    // playing nice with the Win9x palette.
    if selected {
        let accent = Rect::from_min_max(
            Pos2::new(rect.left() + 1.0, rect.top() + 1.0),
            Pos2::new(rect.right() - 1.0, rect.top() + 4.0),
        );
        painter.rect_filled(accent, 0.0, TITLE_BLUE);
    }

    let inner = rect.shrink2(Vec2::new(8.0, 6.0));

    // Title row: "1. filename" left-aligned, meta right-aligned.
    let title_y = inner.top();
    let prefix = format!("{}. ", i + 1);
    let prefix_galley =
        painter.layout_no_wrap(prefix.clone(), body_font.clone(), Color32::BLACK);
    let prefix_pos = Pos2::new(inner.left(), title_y);
    painter.galley(prefix_pos, prefix_galley.clone(), Color32::BLACK);

    paint_highlighted(
        &painter,
        Pos2::new(prefix_pos.x + prefix_galley.size().x, title_y),
        &hit.filename,
        query,
        &body_font,
        Color32::BLACK,
        true,
    );

    // Meta string right-aligned.
    let meta_galley = painter.layout_no_wrap(meta.clone(), body_font.clone(), BEVEL_DARK);
    let meta_pos = Pos2::new(inner.right() - meta_galley.size().x, title_y);
    painter.galley(meta_pos, meta_galley, BEVEL_DARK);

    // Snippet row, wrapped to inner width.
    let snippet_origin = Pos2::new(inner.left(), title_y + 20.0);
    paint_highlighted(
        &painter,
        snippet_origin,
        &hit.snippet,
        query,
        &body_font,
        Color32::BLACK,
        false,
    );

    resp
}

/// Paint `text` left-to-right starting at `origin`, highlighting query
/// matches with a yellow background + black foreground via the shared
/// [`highlight_segments`] kernel.
///
/// `bold_plain` paints non-match runs in bold (used for card titles);
/// otherwise non-match runs use the regular weight. The function does
/// not wrap — call sites that need wrapping (snippet bodies) supply a
/// pre-bounded snippet, mirroring the TUI which also assumes
/// one-line-fit snippets.
fn paint_highlighted(
    painter: &Painter,
    origin: Pos2,
    text: &str,
    query: &str,
    font: &FontId,
    plain_color: Color32,
    bold_plain: bool,
) {
    let mut x = origin.x;
    for seg in highlight_segments(text, query) {
        let (fg, bg) = match seg.kind {
            SegmentKind::Plain => (plain_color, None),
            SegmentKind::Match => (MATCH_FG, Some(MATCH_BG)),
        };
        let galley = painter.layout_no_wrap(seg.text.clone(), font.clone(), fg);
        let size = galley.size();
        if let Some(bg) = bg {
            let bg_rect = Rect::from_min_size(
                Pos2::new(x, origin.y),
                Vec2::new(size.x, size.y),
            );
            painter.rect_filled(bg_rect, 0.0, bg);
        }
        painter.galley(Pos2::new(x, origin.y), galley, fg);
        // egui's stock proportional font has no separate bold face; for
        // card titles we widen the weight by overpainting one pixel
        // right. Cheap, doesn't hurt the snippet bodies (`bold_plain =
        // false`) and matches the TUI's bold-filename convention.
        if seg.kind == SegmentKind::Plain && bold_plain {
            let g2 = painter.layout_no_wrap(seg.text, font.clone(), fg);
            painter.galley(Pos2::new(x + 1.0, origin.y), g2, fg);
        }
        x += size.x;
    }
}

// ─────────────────────── progress snapshot ──────────────────────────

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

#[cfg(test)]
mod tests {
    use super::*;

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
