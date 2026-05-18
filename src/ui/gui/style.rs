//! Global egui style stamping.
//!
//! Called once at the egui `CreationContext` so every widget
//! downstream inherits the Win9x palette + spacing.

use eframe::egui::{self, Color32, FontFamily, FontId, Stroke, TextStyle, Vec2};

use super::palette::{FACE, FONT_BODY, FONT_HEADING, SELECT_NAVY};

/// Add Braille-capable fallback fonts to the `Proportional` family.
///
/// egui's default proportional font is `Ubuntu-Light`, which covers
/// the western text we draw but lacks Dingbats (e.g. `❯` U+276F) and
/// Braille Patterns (the spinner frames). egui only falls back along
/// the *family* chain, never across families, so without this `❯`
/// and `⠋..⠏` render as tofu wherever the system font cascade isn't
/// reachable from inside the bundled font stack.
///
/// On Windows we load `Segoe UI Symbol` (system font with guaranteed
/// Braille coverage). `Hack` covers both blocks on macOS/Linux and
/// serves as a secondary fallback on Windows.
pub fn apply_font_fallback(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    #[cfg(target_os = "windows")]
    {
        let sym_path = r"C:\Windows\Fonts\seguisym.ttf";
        if let Ok(data) = std::fs::read(sym_path) {
            fonts
                .font_data
                .insert("Segoe UI Symbol".to_string(), egui::FontData::from_owned(data));
        }
    }

    if let Some(chain) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
        let mut ins = 1;
        #[cfg(target_os = "windows")]
        if !chain.iter().any(|n| n == "Segoe UI Symbol") {
            chain.insert(ins, "Segoe UI Symbol".to_string());
            ins += 1;
        }
        if !chain.iter().any(|n| n == "Hack") {
            chain.insert(ins, "Hack".to_string());
        }
    }
    ctx.set_fonts(fonts);
}

/// Stamp every egui surface with the Win9x look.
///
/// Win9x had sharp 1px corners on every chrome element, so every
/// rounding parameter goes to zero. egui's stock palette assumes a
/// modern flat-grey-on-grey aesthetic, so we replace each role colour
/// rather than tweaking; that way no remaining widget falls back to a
/// stock blue/grey that breaks the period look.
pub fn apply_win98_visuals(ctx: &egui::Context) {
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
    // Hovered/active keep the same grey body — the bevel does the
    // work, not a colour shift.

    // Selection colour — the canonical Windows navy.
    visuals.selection.bg_fill = SELECT_NAVY;
    visuals.selection.stroke = Stroke::new(1.0, Color32::WHITE);

    ctx.set_visuals(visuals);

    // Airy spacing — Win95 home-dialog generosity, not Win98
    // enterprise density.
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = Vec2::new(10.0, 8.0);
    style.spacing.button_padding = Vec2::new(12.0, 6.0);
    style.spacing.interact_size = Vec2::new(28.0, 28.0);
    // Chunky proportional body font. egui's stock Ubuntu Light at
    // 15pt is the cleanest "MS Sans Serif-adjacent" we can ship
    // without bundling a non-free bitmap font.
    style.text_styles.insert(
        TextStyle::Body,
        FontId::new(FONT_BODY, FontFamily::Proportional),
    );
    style.text_styles.insert(
        TextStyle::Button,
        FontId::new(FONT_BODY, FontFamily::Proportional),
    );
    style.text_styles.insert(
        TextStyle::Heading,
        FontId::new(FONT_HEADING, FontFamily::Proportional),
    );
    style.text_styles.insert(
        TextStyle::Monospace,
        FontId::new(FONT_BODY - 1.0, FontFamily::Monospace),
    );
    ctx.set_style(style);
}
