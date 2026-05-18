//! All GUI render functions.
//!
//! Each function owns one visual region (titlebar, input row,
//! separator, results, help bar, a single hit card). State mutation
//! lives in [`super::keys`] / [`super::GuiApp`]; this module owns
//! presentation.

use eframe::egui::{
    self, Color32, FontFamily, FontId, Painter, Pos2, Rect, RichText, Sense, TextEdit,
    Ui, Vec2,
};

use crate::query::{Hit, QueryMode};
use crate::ui::highlight::{SegmentKind, highlight_segments};
use crate::ui::spinner::frame_at;

use super::paint::{bevel, drop_shadow, etched_hr, horizontal_gradient};
use super::palette::{
    BEVEL_DARK, CARD_GAP, CARD_HEIGHT, CARD_PAD_X, CARD_PAD_Y, ERROR_BG, FACE,
    FONT_BODY, FONT_HEADING, FONT_HELP, HELPBAR_HEIGHT, MATCH_BG, MATCH_FG,
    MODE_PILL_W, PROMPT_W, SELECT_NAVY, TITLE_BLUE_LEFT, TITLE_BLUE_RIGHT,
    TITLEBAR_HEIGHT,
};
use super::GuiApp;

pub(super) fn render(app: &mut GuiApp, ui: &mut Ui) {
    render_titlebar(app, ui);
    ui.add_space(10.0);
    render_input(app, ui);
    ui.add_space(8.0);
    render_separator_or_error(app, ui);
    ui.add_space(10.0);
    render_results(app, ui);
    render_help_bar(ui);
}

/// Faux Win95 title strip — rendered inside the client area so the
/// system window decorations stay clickable but the *content*
/// announces the look. Plus!-pack gradient (navy → steel blue)
/// carries the brand pill on the left and the live indexer counters
/// on the right, with a soft drop shadow underneath so the strip
/// floats over the panel body.
fn render_titlebar(app: &GuiApp, ui: &mut Ui) {
    let snap = app.progress.snapshot();
    let (rect, _) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), TITLEBAR_HEIGHT),
        Sense::hover(),
    );
    // Shadow first, on the panel-level painter so it can spill
    // outside `rect`.
    drop_shadow(ui.painter(), rect);

    let painter = ui.painter_at(rect);
    horizontal_gradient(&painter, rect, TITLE_BLUE_LEFT, TITLE_BLUE_RIGHT);
    // 1px bevel around the strip so it reads as a chrome element
    // rather than an arbitrary gradient band.
    bevel(&painter, rect, true);

    let heading_font = FontId::new(FONT_HEADING, FontFamily::Proportional);

    // Left: "pdffff <root>". Brand bolded by overpainting at +1px.
    let brand = "  pdffff  ".to_string();
    let brand_galley =
        painter.layout_no_wrap(brand.clone(), heading_font.clone(), Color32::WHITE);
    let brand_pos = Pos2::new(
        rect.left() + 6.0,
        rect.center().y - brand_galley.size().y / 2.0,
    );
    painter.galley(brand_pos, brand_galley.clone(), Color32::WHITE);
    let brand_bold = painter.layout_no_wrap(brand, heading_font.clone(), Color32::WHITE);
    painter.galley(brand_pos + Vec2::new(1.0, 0.0), brand_bold, Color32::WHITE);

    let root_text = format!("{}", app.opts.root.display());
    let root_galley =
        painter.layout_no_wrap(root_text, heading_font.clone(), Color32::WHITE);
    let root_pos = Pos2::new(
        brand_pos.x + brand_galley.size().x + 6.0,
        rect.center().y - root_galley.size().y / 2.0,
    );
    painter.galley(root_pos, root_galley, Color32::WHITE);

    // Right: status counters + spinner.
    let activity = if snap.pending > 0 {
        format!("{} indexing {}", frame_at(app.spinner_started.elapsed()), snap.pending)
    } else {
        "idle".to_string()
    };
    let counters = format!(
        "{} ok · {} empty · {} err · {} del · {}  ",
        snap.ok, snap.empty, snap.error, snap.deleted, activity,
    );
    let counters_galley = painter.layout_no_wrap(counters, heading_font, Color32::WHITE);
    let counters_pos = Pos2::new(
        rect.right() - counters_galley.size().x - 6.0,
        rect.center().y - counters_galley.size().y / 2.0,
    );
    painter.galley(counters_pos, counters_galley, Color32::WHITE);
}

/// Query input + mode pill.
///
/// The white well is painted via `egui::Frame::fill`, *not* via a
/// separate `layer_painter` call: `CentralPanel` and our paints both
/// live on `LayerId::background()`, and within a single egui layer
/// draw commands run in insertion order — a rect queued after the
/// TextEdit would cover its text. Routing the fill through a Frame
/// means the white lands at the right point in the widget pass
/// (under the text). The 1px bevel is painted *after* `Frame::show`
/// returns, on top of everything; that's fine because the bevel only
/// touches the outer-edge pixels and never overlaps the text.
fn render_input(app: &mut GuiApp, ui: &mut Ui) {
    ui.horizontal(|ui| {
        render_prompt_glyph(app, ui);
        render_input_well(app, ui);
        ui.add_space(12.0);
        render_mode_pill(app, ui);
    });
}

fn render_prompt_glyph(app: &GuiApp, ui: &mut Ui) {
    let prompt_char = if app.is_searching() {
        frame_at(app.spinner_started.elapsed())
    } else {
        "❯"
    };
    let prompt_font = FontId::new(FONT_BODY + 1.0, FontFamily::Proportional);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(PROMPT_W, 30.0), Sense::hover());
    let painter = ui.painter_at(rect);
    let galley = painter.layout_no_wrap(prompt_char.to_string(), prompt_font, Color32::BLACK);
    let pos = Pos2::new(
        rect.center().x - galley.size().x / 2.0,
        rect.center().y - galley.size().y / 2.0,
    );
    painter.galley(pos, galley, Color32::BLACK);
}

fn render_input_well(app: &mut GuiApp, ui: &mut Ui) {
    // egui's horizontal layout inserts an `item_spacing.x` between
    // every two items, *including* before the mode pill and around
    // `add_space`. We have to subtract both item_spacings explicitly
    // or the pill overflows the right edge.
    let item_spacing = ui.spacing().item_spacing.x;
    let between = item_spacing + 12.0 + item_spacing;
    let well_target_w = (ui.available_width() - MODE_PILL_W - between).max(40.0);

    let inner = egui::Frame::none()
        .fill(Color32::WHITE)
        .inner_margin(egui::Margin {
            left: 10.0,
            right: 10.0,
            top: 6.0,
            bottom: 6.0,
        })
        .show(ui, |ui| {
            ui.add(
                TextEdit::singleline(&mut app.query)
                    .desired_width(well_target_w - 20.0)
                    .frame(false)
                    .text_color(Color32::BLACK)
                    .font(FontId::new(FONT_BODY, FontFamily::Proportional)),
            )
        });
    let well_rect = inner.response.rect;
    bevel(&ui.painter_at(well_rect), well_rect, false);
    let resp = inner.inner;

    if resp.changed() {
        app.submit_query();
    }
    if !app.did_initial_focus {
        resp.request_focus();
        app.did_initial_focus = true;
    }
}

fn render_mode_pill(app: &mut GuiApp, ui: &mut Ui) {
    let label = match app.mode {
        QueryMode::Literal => " LITERAL ",
        QueryMode::Regex => " REGEX ",
        QueryMode::Fuzzy => " FUZZY ",
    };
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(MODE_PILL_W, 30.0), Sense::click());
    drop_shadow(ui.painter(), rect);
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, FACE);
    bevel(&painter, rect, !resp.is_pointer_button_down_on());
    let font = FontId::new(FONT_BODY, FontFamily::Proportional);
    let galley = painter.layout_no_wrap(label.to_string(), font, Color32::BLACK);
    let pos = Pos2::new(
        rect.center().x - galley.size().x / 2.0,
        rect.center().y - galley.size().y / 2.0,
    );
    painter.galley(pos, galley, Color32::BLACK);
    if resp.clicked() {
        app.mode = crate::ui::input::cycle_mode(app.mode);
        app.submit_query();
    }
}

fn render_separator_or_error(app: &GuiApp, ui: &mut Ui) {
    if let Some(err) = &app.last_error {
        render_error_pill(err, ui);
    } else {
        let (rect, _) = ui.allocate_exact_size(
            Vec2::new(ui.available_width(), 2.0),
            Sense::hover(),
        );
        etched_hr(&ui.painter_at(rect), rect);
    }
}

fn render_error_pill(err: &str, ui: &mut Ui) {
    ui.horizontal(|ui| {
        let label_text = " error ";
        let body_font = FontId::new(FONT_BODY, FontFamily::Proportional);
        let (rect, _) = ui.allocate_exact_size(Vec2::new(68.0, 26.0), Sense::hover());
        drop_shadow(ui.painter(), rect);
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, ERROR_BG);
        bevel(&painter, rect, true);
        let galley = painter.layout_no_wrap(label_text.to_string(), body_font, Color32::WHITE);
        let pos = Pos2::new(
            rect.center().x - galley.size().x / 2.0,
            rect.center().y - galley.size().y / 2.0,
        );
        painter.galley(pos, galley, Color32::WHITE);
        ui.add_space(8.0);
        ui.label(RichText::new(err).color(ERROR_BG).size(FONT_BODY));
    });
}

fn render_results(app: &mut GuiApp, ui: &mut Ui) {
    let total = app.hits.len();
    if total == 0 {
        let msg = if app.query.trim().is_empty() {
            "type a query to search the index"
        } else {
            "no hits"
        };
        ui.vertical_centered(|ui| {
            ui.add_space(28.0);
            ui.label(
                RichText::new(msg)
                    .italics()
                    .color(BEVEL_DARK)
                    .size(FONT_BODY),
            );
        });
        app.prev_selected = app.selected;
        return;
    }
    let selected = app.selected;
    // Only auto-scroll on the frame where selection changed; on every
    // other frame the user owns the scroll position.
    let scroll_target = if selected != app.prev_selected {
        selected
    } else {
        None
    };
    let query = app.query.clone();
    let mode = app.mode;
    // The result list is scrollable. We reserve room at the bottom
    // for the help bar (which is rendered *after* the results, so
    // its height is taken from the layout cursor).
    let available_h = ui.available_height() - (HELPBAR_HEIGHT + 8.0);
    let mut activated: Option<usize> = None;
    egui::ScrollArea::vertical()
        .max_height(available_h.max(CARD_HEIGHT))
        .auto_shrink([false; 2])
        .show(ui, |ui| {
            for i in 0..total {
                let hit = &app.hits[i];
                let is_sel = selected == Some(i);
                let resp = render_hit_card(ui, i, hit, &query, mode, is_sel);
                if resp.clicked() {
                    activated = Some(i);
                }
                if scroll_target == Some(i) {
                    resp.scroll_to_me(Some(egui::Align::Center));
                }
                ui.add_space(CARD_GAP);
            }
        });
    if let Some(i) = activated {
        app.selected = Some(i);
        app.pick_selected();
    }
    app.prev_selected = app.selected;
}

fn render_help_bar(ui: &mut Ui) {
    ui.add_space(8.0);
    let (rect, _) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), HELPBAR_HEIGHT),
        Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    etched_hr(&painter, rect);
    let help =
        "↑↓ select   Tab mode   Enter open   Ctrl+U clear   Ctrl+W word-erase   Esc quit";
    let font = FontId::new(FONT_HELP, FontFamily::Proportional);
    let galley = painter.layout_no_wrap(help.to_string(), font, BEVEL_DARK);
    let pos = Pos2::new(
        rect.left() + 6.0,
        rect.center().y - galley.size().y / 2.0 + 2.0,
    );
    painter.galley(pos, galley, BEVEL_DARK);
}

/// One result card. Returns the click response so the caller can pick
/// it. The card is a raised bevelled tile with a soft drop shadow
/// underneath; selected cards add a navy strip along the top edge.
fn render_hit_card(
    ui: &mut Ui,
    i: usize,
    hit: &Hit,
    query: &str,
    mode: QueryMode,
    selected: bool,
) -> egui::Response {
    let body_font = FontId::new(FONT_BODY, FontFamily::Proportional);
    let desired = Vec2::new(ui.available_width(), CARD_HEIGHT);
    let (rect, resp) = ui.allocate_exact_size(desired, Sense::click());

    // Drop shadow first, on the panel-level painter so it can spill
    // past `rect`'s allocated footprint.
    drop_shadow(ui.painter(), rect);
    let painter = ui.painter_at(rect);

    // Card body — raised tile.
    painter.rect_filled(rect, 0.0, FACE);
    bevel(&painter, rect, true);

    // Selection accent: a 3px navy strip along the top, inside the
    // bevel. Mirrors the TUI's border-only selection cue using the
    // Win95 active-title navy.
    if selected {
        let accent = Rect::from_min_max(
            Pos2::new(rect.left() + 1.0, rect.top() + 1.0),
            Pos2::new(rect.right() - 1.0, rect.top() + 4.0),
        );
        painter.rect_filled(accent, 0.0, SELECT_NAVY);
    }

    let inner = rect.shrink2(Vec2::new(CARD_PAD_X, CARD_PAD_Y));
    let title_h = body_font.size + 4.0;
    let rule_y = inner.top() + title_h;

    paint_card_title_row(&painter, inner, i, hit, query, mode, &body_font);

    // Hairline separating the title row from the snippet — etched
    // groupbox-style, dark over light.
    etched_hr(
        &painter,
        Rect::from_min_max(
            Pos2::new(inner.left(), rule_y),
            Pos2::new(inner.right(), rule_y + 2.0),
        ),
    );

    paint_highlighted(
        &painter,
        Pos2::new(inner.left(), rule_y + 6.0),
        &hit.snippet,
        query,
        &body_font,
        Color32::BLACK,
        false,
    );

    resp
}

fn paint_card_title_row(
    painter: &Painter,
    inner: Rect,
    i: usize,
    hit: &Hit,
    query: &str,
    mode: QueryMode,
    body_font: &FontId,
) {
    let title_y = inner.top();
    let prefix = format!("{}. ", i + 1);
    let prefix_galley =
        painter.layout_no_wrap(prefix.clone(), body_font.clone(), BEVEL_DARK);
    let prefix_pos = Pos2::new(inner.left(), title_y);
    painter.galley(prefix_pos, prefix_galley.clone(), BEVEL_DARK);

    paint_highlighted(
        painter,
        Pos2::new(prefix_pos.x + prefix_galley.size().x, title_y),
        &hit.filename,
        query,
        body_font,
        Color32::BLACK,
        true,
    );

    let meta = match mode {
        QueryMode::Fuzzy => format!(
            "p.{} · #{} · score {:.2}",
            hit.page_no, hit.chunk_ord, hit.score,
        ),
        QueryMode::Literal | QueryMode::Regex => {
            format!("p.{} · #{}", hit.page_no, hit.chunk_ord)
        }
    };
    let meta_galley =
        painter.layout_no_wrap(meta, body_font.clone(), BEVEL_DARK);
    let meta_pos = Pos2::new(inner.right() - meta_galley.size().x, title_y);
    painter.galley(meta_pos, meta_galley, BEVEL_DARK);
}

/// Paint `text` left-to-right starting at `origin`, highlighting query
/// matches with a yellow background + black foreground via the shared
/// [`highlight_segments`] kernel.
///
/// `bold_plain` paints non-match runs in bold (used for card titles);
/// otherwise non-match runs use the regular weight. The function does
/// not wrap — call sites that need wrapping supply a pre-bounded
/// snippet.
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
            let bg_rect =
                Rect::from_min_size(Pos2::new(x, origin.y), Vec2::new(size.x, size.y));
            painter.rect_filled(bg_rect, 0.0, bg);
        }
        painter.galley(Pos2::new(x, origin.y), galley, fg);
        // egui's stock proportional font has no separate bold face;
        // for card titles we widen the weight by overpainting one
        // pixel right. Cheap, doesn't hurt the snippet bodies
        // (`bold_plain = false`) and matches the TUI's bold-filename
        // convention.
        if seg.kind == SegmentKind::Plain && bold_plain {
            let g2 = painter.layout_no_wrap(seg.text, font.clone(), fg);
            painter.galley(Pos2::new(x + 1.0, origin.y), g2, fg);
        }
        x += size.x;
    }
}

