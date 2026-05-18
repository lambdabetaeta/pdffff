//! Vapourwave-flavoured cross-platform desktop GUI for `pdffff`.
//!
//! Same skeleton as a Win9x dialog (1px bevels, square corners, a
//! status strip up top, a help strip at the bottom, raised cards in
//! the middle) but recoloured in the canonical late-Tumblr palette:
//! deep violet body, hot-pink + neon-cyan edges, peachy cream text,
//! drop shadows under every card. The TUI's Bertin-style discipline
//! survives the recolour — one hue, one job:
//!
//! * **pink** is chrome (bevel highlight, mode pill, prompt)
//! * **cyan** is focus (selection accent, gradient end-stop, hovered
//!   stroke)
//! * **violet** is body (panels, card bodies, input wells)
//! * **yellow-on-magenta** is reserved exclusively for query matches
//!   (CRT-style hot-pop)
//! * **bright pink** is error
//!
//! The faux 3D bevels are still painted via the egui `Painter` API;
//! egui's default rounded `Frame` chrome is still suppressed globally
//! in [`apply_vapourwave_visuals`]. Drop shadows under each card are
//! painted as a stack of translucent pink/cyan rects (egui has no
//! gaussian blur, so the soft-glow look is approximated by a few
//! offset layers with decaying alpha).
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

// ─────────────────────── vapourwave palette ─────────────────────────
//
// One hue, one job (Bertin discipline preserved from the TUI). Deep
// violet body, hot-pink chrome, neon-cyan focus, yellow-on-magenta
// reserved exclusively for query matches, peachy cream for body text.

/// Window background — deep midnight violet, just enough purple to
/// register as not-black.
const BG_DEEP: Color32 = Color32::from_rgb(0x0d, 0x05, 0x1f);
/// Panel / card body — one step up from `BG_DEEP`, the canonical
/// "face" colour every raised surface fills with.
const FACE: Color32 = Color32::from_rgb(0x2a, 0x10, 0x4a);
/// One more step up — used for the input well (sunken text input) so
/// the well reads as a recess rather than another raised panel.
const WELL_FILL: Color32 = Color32::from_rgb(0x1a, 0x08, 0x33);
/// Highlight edge of a raised bevel (top + left). Hot pink — the
/// signature vapourwave neon.
const BEVEL_LIGHT: Color32 = Color32::from_rgb(0xff, 0x71, 0xce);
/// Shadow edge of a raised bevel (bottom + right). Cool, deep violet
/// so the bevel reads as light-from-the-northwest the same way Win9x
/// does, but the colour speaks the vapourwave dialect.
const BEVEL_DARK: Color32 = Color32::from_rgb(0x5a, 0x18, 0x8a);
/// Focus accent / selection — the cooler half of the palette so it
/// pops against the pink chrome.
const ACCENT_CYAN: Color32 = Color32::from_rgb(0x01, 0xcd, 0xfe);
/// Gradient endpoint paired with `ACCENT_CYAN` across the title strip.
const ACCENT_MAGENTA: Color32 = Color32::from_rgb(0xff, 0x36, 0x9f);
/// Body text — cream so it has warmth on the violet body without
/// glaring like pure white.
const TEXT_LIGHT: Color32 = Color32::from_rgb(0xff, 0xf3, 0xe0);
/// Secondary / dim text — peachy pink, half the saturation of the
/// neon edges so meta strings recede from the title.
const TEXT_DIM: Color32 = Color32::from_rgb(0xd6, 0xa3, 0xd9);
/// Hit highlight — bright yellow on hot magenta, the CRT-pop the TUI
/// achieves with yellow-on-black. Reserved exclusively for matches.
const MATCH_BG: Color32 = Color32::from_rgb(0xff, 0x2d, 0x95);
const MATCH_FG: Color32 = Color32::from_rgb(0xff, 0xfb, 0x96);
/// Error pill — a brighter pink than the chrome so it doesn't fight
/// the magenta match highlight.
const ERROR_BG: Color32 = Color32::from_rgb(0xff, 0x4f, 0x5e);
/// Translucent pink used to fake a soft drop-shadow under each card.
/// Stacked at decaying alpha for a glow-like falloff (egui has no
/// gaussian blur primitive, so this is the cheapest fake).
const SHADOW_PINK: Color32 = Color32::from_rgba_premultiplied(0x40, 0x10, 0x30, 0x40);

const SPINNER_FRAMES: [&str; 10] =
    ["◜", "◝", "◞", "◟", "◜", "◝", "◞", "◟", "✦", "✧"];

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
            apply_vapourwave_visuals(&cc.egui_ctx);
            let app = GuiApp::new(handle, opts_for_app, chosen_for_app, cc.egui_ctx.clone())?;
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow!("eframe::run_native: {e}"))?;

    let result = chosen.lock().take();
    Ok(result)
}

// ─────────────────────────── styling ────────────────────────────────

/// Stamp every egui surface with the vapourwave look. Called once at
/// `CreationContext`; `set_visuals` / `set_style` apply globally for
/// the life of the window.
///
/// The Win9x bones are preserved (square 1px corners everywhere; no
/// egui-default rounded chrome) — only the palette is replaced. We
/// rewrite each widget-state role explicitly so no fallback grey-on-
/// grey survives anywhere.
fn apply_vapourwave_visuals(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();

    // Outer window + inner panels use the deep-midnight backdrop; the
    // raised "face" colour is reserved for cards / the mode pill so
    // raised vs body chrome is itself a visual variable.
    visuals.window_fill = BG_DEEP;
    visuals.panel_fill = BG_DEEP;
    visuals.faint_bg_color = FACE;
    // Sunken text wells — the input box reads as a recess in the
    // panel rather than a separate raised tile.
    visuals.extreme_bg_color = WELL_FILL;

    // Kill all rounding — square corners everywhere.
    let zero = egui::Rounding::ZERO;
    visuals.window_rounding = zero;
    visuals.menu_rounding = zero;
    visuals.window_shadow = egui::epaint::Shadow::NONE;
    visuals.popup_shadow = egui::epaint::Shadow::NONE;

    let body_fg = Stroke::new(1.0, TEXT_LIGHT);
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
        w.fg_stroke = body_fg;
        w.expansion = 0.0;
    }
    // A subtle hover-glow: the hovered/active states keep the face
    // colour but pick up a thin cyan outline. The card painter
    // currently doesn't use this (it draws its own bevel), but stock
    // egui widgets like the scrollbar handle benefit.
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT_CYAN);
    visuals.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT_CYAN);

    // Selection — the cool half of the palette.
    visuals.selection.bg_fill = ACCENT_CYAN;
    visuals.selection.stroke = Stroke::new(1.0, BG_DEEP);

    // Hyperlinks (egui paints one in `code` blocks etc) — keep on-brand.
    visuals.hyperlink_color = ACCENT_CYAN;

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

/// Paint a horizontal gradient from `left` to `right` across `rect`.
///
/// egui's primitive set has no native gradient brush; we approximate
/// one by painting `rect.width()` 1px vertical strips with the
/// channel-wise linear interpolation between `left` and `right`. At
/// typical title-bar / horizon-rule sizes (a few hundred pixels) the
/// cost is invisible — it's still O(width) draw calls but each one
/// is a tiny filled rectangle.
fn paint_horizontal_gradient(painter: &Painter, rect: Rect, left: Color32, right: Color32) {
    let w = rect.width().max(1.0);
    let steps = w.ceil() as i32;
    for x in 0..steps {
        let t = x as f32 / w;
        let c = lerp_color(left, right, t);
        let strip = Rect::from_min_max(
            Pos2::new(rect.left() + x as f32, rect.top()),
            Pos2::new(rect.left() + x as f32 + 1.0, rect.bottom()),
        );
        painter.rect_filled(strip, 0.0, c);
    }
}

/// Channel-wise linear interpolation between two `Color32`s. Operates
/// on premultiplied RGBA, which is what egui stores internally; the
/// alpha channel travels with the colour so a translucent endpoint
/// fades into a translucent midpoint as expected.
fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let lerp = |x: u8, y: u8| -> u8 {
        (x as f32 * (1.0 - t) + y as f32 * t).round().clamp(0.0, 255.0) as u8
    };
    Color32::from_rgba_premultiplied(
        lerp(a.r(), b.r()),
        lerp(a.g(), b.g()),
        lerp(a.b(), b.b()),
        lerp(a.a(), b.a()),
    )
}

/// Paint a 1px bevel on the outer perimeter of `rect`.
///
/// `raised = true` paints buttons / cards (light top-left, dark
/// bottom-right); `raised = false` paints text inputs and inset
/// surfaces (dark top-left, light bottom-right). "Light" and "dark"
/// here are the vapourwave hot-pink / deep-violet bevel pair, not
/// literal grey-on-grey.
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
        // chrome inside it. The panel fill is the deep-violet body
        // colour so cards (the raised `FACE` colour) read as floating
        // above it.
        egui::CentralPanel::default()
            .frame(
                egui::Frame::none()
                    .fill(BG_DEEP)
                    .inner_margin(egui::Margin::same(10.0)),
            )
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

    /// Title strip across the top of the client area. Filled with a
    /// horizontal magenta→cyan gradient (the iconic 80s sunset
    /// reduced to two waypoints), brand + watched root on the left,
    /// live indexer counters on the right. Outlined with a 1px hot-
    /// pink stroke so the gradient still reads as a bounded chrome
    /// element.
    fn render_titlebar(&self, ui: &mut Ui) {
        let snap = snapshot_progress(&self.progress);
        let height = 26.0;
        let (rect, _) = ui.allocate_exact_size(
            Vec2::new(ui.available_width(), height),
            Sense::hover(),
        );
        let painter = ui.painter_at(rect);

        paint_horizontal_gradient(&painter, rect, ACCENT_MAGENTA, ACCENT_CYAN);
        // Outline the strip in hot pink. Keeps the gradient bounded
        // on the eye even when the window is very wide.
        painter.rect_stroke(rect, 0.0, Stroke::new(1.0, BEVEL_LIGHT));

        let body_font = FontId::new(13.0, FontFamily::Proportional);

        // Left: "pdffff <root>" in cream, faintly shadow-stamped for
        // the chunky text feel.
        let brand = "  pdffff  ".to_string();
        let brand_galley =
            painter.layout_no_wrap(brand, body_font.clone(), TEXT_LIGHT);
        let brand_pos = Pos2::new(
            rect.left() + 6.0,
            rect.center().y - brand_galley.size().y / 2.0,
        );
        // 1px text-shadow for that CRT-bleed feel.
        let brand_shadow =
            painter.layout_no_wrap("  pdffff  ".to_string(), body_font.clone(), BG_DEEP);
        painter.galley(brand_pos + Vec2::new(1.0, 1.0), brand_shadow, BG_DEEP);
        painter.galley(brand_pos, brand_galley.clone(), TEXT_LIGHT);

        let root_text = format!("{}", self.opts.root.display());
        let root_galley =
            painter.layout_no_wrap(root_text, body_font.clone(), TEXT_LIGHT);
        let root_pos = Pos2::new(
            brand_pos.x + brand_galley.size().x + 6.0,
            rect.center().y - root_galley.size().y / 2.0,
        );
        painter.galley(root_pos, root_galley, TEXT_LIGHT);

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
        let counters_galley = painter.layout_no_wrap(counters, body_font, TEXT_LIGHT);
        let counters_pos = Pos2::new(
            rect.right() - counters_galley.size().x - 6.0,
            rect.center().y - counters_galley.size().y / 2.0,
        );
        painter.galley(counters_pos, counters_galley, TEXT_LIGHT);
    }

    /// Query input + mode pill. The input is a sunken text well with a
    /// hot-pink prompt arrow; the mode pill is a raised neon-bordered
    /// button that cycles the mode on click.
    fn render_input(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("❯")
                    .color(BEVEL_LIGHT)
                    .strong(),
            );

            let edit = TextEdit::singleline(&mut self.query)
                .desired_width(ui.available_width() - 80.0)
                .frame(false)
                .text_color(TEXT_LIGHT)
                .margin(Vec2::new(6.0, 4.0));
            let resp = ui.add(edit);
            // Paint the sunken well behind the text on the background
            // layer (so widget paint lands on top).
            let well = resp.rect;
            let painter = ui.ctx().layer_painter(egui::LayerId::background());
            painter.rect_filled(well, 0.0, WELL_FILL);
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

            ui.add_space(8.0);
            // Mode pill — raised neon-bordered button.
            let label = match self.mode {
                QueryMode::Literal => " LIT ",
                QueryMode::Regex => " RE ",
                QueryMode::Fuzzy => " FZ ",
            };
            let (pill_rect, pill_resp) = ui.allocate_exact_size(
                Vec2::new(52.0, 24.0),
                Sense::click(),
            );
            let pp = ui.painter_at(pill_rect);
            pp.rect_filled(pill_rect, 0.0, FACE);
            // Pressed → invert the bevel so the click reads as a real
            // depress. Hovered → keep the raised look but pre-glow the
            // background via a hint of pink to telegraph interactivity.
            let pressed = pill_resp.is_pointer_button_down_on();
            if pill_resp.hovered() && !pressed {
                pp.rect_filled(
                    pill_rect.shrink(1.0),
                    0.0,
                    Color32::from_rgba_premultiplied(0x40, 0x18, 0x60, 0xff),
                );
            }
            paint_bevel(&pp, pill_rect, !pressed);
            let body_font = FontId::new(13.0, FontFamily::Proportional);
            let galley = pp.layout_no_wrap(label.to_string(), body_font, TEXT_LIGHT);
            let p = Pos2::new(
                pill_rect.center().x - galley.size().x / 2.0,
                pill_rect.center().y - galley.size().y / 2.0,
            );
            pp.galley(p, galley, TEXT_LIGHT);
            if pill_resp.clicked() {
                self.cycle_mode();
            }
        });
    }

    fn render_separator_or_error(&self, ui: &mut Ui) {
        if let Some(err) = &self.last_error {
            ui.horizontal(|ui| {
                // Hot-pink "error" pill.
                let label_text = " error ";
                let body_font = FontId::new(13.0, FontFamily::Proportional);
                let (label_rect, _) = ui.allocate_exact_size(
                    Vec2::new(60.0, 22.0),
                    Sense::hover(),
                );
                let painter = ui.painter_at(label_rect);
                painter.rect_filled(label_rect, 0.0, ERROR_BG);
                paint_bevel(&painter, label_rect, true);
                let galley = painter.layout_no_wrap(
                    label_text.to_string(),
                    body_font,
                    BG_DEEP,
                );
                let p = Pos2::new(
                    label_rect.center().x - galley.size().x / 2.0,
                    label_rect.center().y - galley.size().y / 2.0,
                );
                painter.galley(p, galley, BG_DEEP);
                ui.add_space(6.0);
                ui.label(RichText::new(err).color(ERROR_BG));
            });
        } else {
            // 1px gradient rule magenta → cyan, with a faint glow row
            // below it. Replaces the etched-groove Win9x separator
            // with the vapourwave horizon line.
            let (rect, _) = ui.allocate_exact_size(
                Vec2::new(ui.available_width(), 3.0),
                Sense::hover(),
            );
            let painter = ui.painter_at(rect);
            paint_horizontal_gradient(
                &painter,
                Rect::from_min_max(
                    rect.left_top(),
                    Pos2::new(rect.right(), rect.top() + 1.0),
                ),
                ACCENT_MAGENTA,
                ACCENT_CYAN,
            );
            // A 1px translucent twin slightly offset for the "neon
            // bleed" feel.
            paint_horizontal_gradient(
                &painter,
                Rect::from_min_max(
                    Pos2::new(rect.left(), rect.top() + 2.0),
                    Pos2::new(rect.right(), rect.top() + 3.0),
                ),
                Color32::from_rgba_premultiplied(0x6a, 0x18, 0x52, 0x80),
                Color32::from_rgba_premultiplied(0x00, 0x52, 0x67, 0x80),
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
                ui.label(RichText::new(msg).italics().color(TEXT_DIM));
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
            Vec2::new(ui.available_width(), 22.0),
            Sense::hover(),
        );
        let painter = ui.painter_at(rect);
        // Gradient horizon line on top of the strip — mirrors the
        // separator above the results.
        paint_horizontal_gradient(
            &painter,
            Rect::from_min_max(
                rect.left_top(),
                Pos2::new(rect.right(), rect.top() + 1.0),
            ),
            ACCENT_CYAN,
            ACCENT_MAGENTA,
        );
        let help =
            "↑↓ select   Tab mode   Enter open   Ctrl+U clear   Ctrl+W word-erase   Esc quit";
        let font = FontId::new(12.0, FontFamily::Proportional);
        let galley =
            painter.layout_no_wrap(help.to_string(), font, TEXT_DIM);
        let p = Pos2::new(
            rect.left() + 6.0,
            rect.center().y - galley.size().y / 2.0 + 1.0,
        );
        painter.galley(p, galley, TEXT_DIM);
    }
}

/// One result card. Returns the click response so the caller can pick
/// it. The card is a raised tile with a hot-pink bevel; the selected
/// card replaces the top edge with a cyan glow strip and outlines the
/// whole tile in cyan, mirroring the TUI's "border-only" selection
/// cue in vapourwave grammar. Each card sits on top of a faux
/// drop-shadow stack painted on the panel-level painter so the shadow
/// extends beyond the card's own footprint.
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

    // Reserve space: title row (20px) + snippet row + padding +
    // bevel + room for the drop-shadow stack below.
    let desired = Vec2::new(ui.available_width(), 66.0);
    let (rect, resp) = ui.allocate_exact_size(desired, Sense::click());

    // Drop shadow: stack three translucent rects offset down-right,
    // each smaller in alpha. egui has no gaussian blur; this fake is
    // cheap and reads as a soft glow. Painted on the panel-level
    // painter (not `ui.painter_at(rect)`) so it can spill outside the
    // card's own allocation.
    let panel_painter = ui.painter();
    for step in 1..=4 {
        let off = step as f32 * 1.5;
        let alpha = (0x50_u8).saturating_sub(step as u8 * 0x12);
        let c = Color32::from_rgba_unmultiplied(
            SHADOW_PINK.r(),
            SHADOW_PINK.g(),
            SHADOW_PINK.b(),
            alpha,
        );
        let shadow_rect = rect.translate(Vec2::new(off, off));
        panel_painter.rect_filled(shadow_rect, 0.0, c);
    }

    let painter = ui.painter_at(rect);

    // Card body — raised tile with a hot-pink bevel.
    painter.rect_filled(rect, 0.0, FACE);
    paint_bevel(&painter, rect, true);

    // Selection cue: outline the whole tile in cyan, plus a 3px cyan
    // strip along the top of the body for that "active row" glow.
    if selected {
        painter.rect_stroke(rect, 0.0, Stroke::new(1.0, ACCENT_CYAN));
        let accent = Rect::from_min_max(
            Pos2::new(rect.left() + 1.0, rect.top() + 1.0),
            Pos2::new(rect.right() - 1.0, rect.top() + 4.0),
        );
        painter.rect_filled(accent, 0.0, ACCENT_CYAN);
    }

    let inner = rect.shrink2(Vec2::new(10.0, 8.0));

    // Title row: "1. filename" left-aligned, meta right-aligned.
    let title_y = inner.top();
    let prefix = format!("{}. ", i + 1);
    let prefix_galley =
        painter.layout_no_wrap(prefix.clone(), body_font.clone(), TEXT_DIM);
    let prefix_pos = Pos2::new(inner.left(), title_y);
    painter.galley(prefix_pos, prefix_galley.clone(), TEXT_DIM);

    paint_highlighted(
        &painter,
        Pos2::new(prefix_pos.x + prefix_galley.size().x, title_y),
        &hit.filename,
        query,
        &body_font,
        TEXT_LIGHT,
        true,
    );

    // Meta string right-aligned, in the secondary lavender.
    let meta_galley = painter.layout_no_wrap(meta.clone(), body_font.clone(), TEXT_DIM);
    let meta_pos = Pos2::new(inner.right() - meta_galley.size().x, title_y);
    painter.galley(meta_pos, meta_galley, TEXT_DIM);

    // Snippet row.
    let snippet_origin = Pos2::new(inner.left(), title_y + 20.0);
    paint_highlighted(
        &painter,
        snippet_origin,
        &hit.snippet,
        query,
        &body_font,
        TEXT_LIGHT,
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
