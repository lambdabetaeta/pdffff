//! Pure-graphics helpers reused across every Win9x chrome element.
//!
//! Bevels, drop shadows, and the Plus!-pack title gradient live here
//! so card / titlebar / mode-pill code only needs to know *which*
//! treatment to apply, not how to draw it.

use eframe::egui::{Color32, Painter, Pos2, Rect, Stroke, Vec2};

use super::palette::{BEVEL_DARK, BEVEL_LIGHT, SHADOW};

/// Paint a horizontal gradient from `left` to `right` across `rect`.
///
/// Used for the Win95 Plus!-pack title strip (navy → steel blue).
/// egui has no gradient brush primitive, so we approximate it with
/// `rect.width()` 1px vertical strips and a channel-wise lerp; at
/// title-bar sizes the cost is invisible.
pub fn horizontal_gradient(painter: &Painter, rect: Rect, left: Color32, right: Color32) {
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

/// Channel-wise linear interpolation between two `Color32`s on the
/// premultiplied-RGBA space egui stores natively.
pub fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let lerp = |x: u8, y: u8| -> u8 {
        (x as f32 * (1.0 - t) + y as f32 * t)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    Color32::from_rgba_premultiplied(
        lerp(a.r(), b.r()),
        lerp(a.g(), b.g()),
        lerp(a.b(), b.b()),
        lerp(a.a(), b.a()),
    )
}

/// Paint a soft Win95-style drop shadow underneath `rect`.
///
/// Win95 menus / tooltips carried a hard black-grey shadow offset by
/// a few pixels to the bottom-right; we use the same offset direction
/// but stack three translucent layers so the falloff reads as soft
/// rather than the harder dithered shadow of the period.
///
/// Must be called via the panel-level painter so the shadow can
/// extend past `rect`'s own allocated footprint.
pub fn drop_shadow(painter: &Painter, rect: Rect) {
    for step in 1..=3 {
        let off = step as f32 * 1.5;
        // Alpha decays roughly geometrically: 0x50 → 0x28 → 0x14.
        let alpha = (0x50_u8) >> (step - 1) as u8;
        let c = Color32::from_rgba_unmultiplied(SHADOW.r(), SHADOW.g(), SHADOW.b(), alpha);
        let shadow_rect = rect.translate(Vec2::new(off, off));
        painter.rect_filled(shadow_rect, 0.0, c);
    }
}

/// Paint a 1px Win9x bevel on the outer perimeter of `rect`.
///
/// `raised = true` paints buttons / toolbars (light top-left, dark
/// bottom-right); `raised = false` paints text inputs and the inset
/// card body (dark top-left, light bottom-right).
pub fn bevel(painter: &Painter, rect: Rect, raised: bool) {
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

/// Two-tone etched horizontal rule (dark over light), the Win9x
/// dialog vocabulary for separating regions inside a groupbox.
pub fn etched_hr(painter: &Painter, rect: Rect) {
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
