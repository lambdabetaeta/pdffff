//! Windows 95 palette + geometry constants.
//!
//! One hue, one job (same Bertin discipline as the TUI palette). The
//! 3D-button-face grey is the body of the world; the Plus!-pack title
//! gradient (navy → steel blue) is reserved for chrome and focus;
//! yellow-on-black is reserved exclusively for query matches; red is
//! reserved for errors. Soft drop shadows under raised tiles match
//! the period menu-shadow effect from Win95 OSR2 / Win98.

use eframe::egui::Color32;

/// Classic Windows 9x "3D button face" — the body colour of every
/// non-control surface.
pub const FACE: Color32 = Color32::from_rgb(0xc0, 0xc0, 0xc0);
/// Highlight edge of a raised bevel (top + left).
pub const BEVEL_LIGHT: Color32 = Color32::from_rgb(0xff, 0xff, 0xff);
/// Shadow edge of a raised bevel (bottom + right). Inverted for sunken
/// bevels (text inputs, the inset card body).
pub const BEVEL_DARK: Color32 = Color32::from_rgb(0x80, 0x80, 0x80);
/// Title bar gradient endpoints — the iconic Plus!-pack "active
/// window" gradient (Win95 OSR2 onwards), running from saturated navy
/// on the left to a lighter steel blue on the right.
pub const TITLE_BLUE_LEFT: Color32 = Color32::from_rgb(0x00, 0x00, 0x80);
pub const TITLE_BLUE_RIGHT: Color32 = Color32::from_rgb(0x10, 0x84, 0xd0);
/// Selection accent — keep the saturated navy; the gradient is
/// reserved for the title strip so the selection cue stays distinct.
pub const SELECT_NAVY: Color32 = TITLE_BLUE_LEFT;
/// Hit highlight, identical role to the TUI's yellow-on-black.
pub const MATCH_BG: Color32 = Color32::from_rgb(0xff, 0xff, 0x00);
pub const MATCH_FG: Color32 = Color32::BLACK;
/// Error pill background.
pub const ERROR_BG: Color32 = Color32::from_rgb(0x80, 0x00, 0x00);

/// Drop-shadow colour. Win95 menus used a hard offset shadow in
/// `Color32::from_rgb(0x40, 0x40, 0x40)`; we lighten it slightly and
/// stack a couple of translucent layers for a softer falloff.
pub const SHADOW: Color32 = Color32::from_rgb(0x00, 0x00, 0x00);

// ────────────────────── spacing / sizing ──────────────────────
//
// Bigger and airier than the bare Win98 dialog defaults — Win95
// dialogs through the eyes of nostalgia, not as a corporate 1999
// data-entry form.

/// Outer margin inside the central panel.
pub const PANEL_MARGIN: f32 = 12.0;
/// Inset of card content from the card's bevel.
pub const CARD_PAD_X: f32 = 12.0;
pub const CARD_PAD_Y: f32 = 10.0;
/// One result card's total height. Sized for the bumped 15pt body
/// font + a 20px gap between the title row and the snippet row +
/// `CARD_PAD_Y` top/bottom.
pub const CARD_HEIGHT: f32 = 78.0;
/// Vertical breathing space between adjacent cards.
pub const CARD_GAP: f32 = 8.0;
/// Title strip height (the navy gradient bar at the top).
pub const TITLEBAR_HEIGHT: f32 = 30.0;
/// Help strip height (the keybinding hints at the bottom).
pub const HELPBAR_HEIGHT: f32 = 26.0;
/// Mode-pill width — sized for the longest label ("LITERAL") plus
/// padding so all three states (LITERAL / REGEX / FUZZY) fit the
/// same pill without resizing as the mode cycles.
pub const MODE_PILL_W: f32 = 112.0;
/// Prompt position width — wide enough to host the widest spinner
/// frame without the cursor of the adjacent text-input jittering as
/// the spinner animates.
pub const PROMPT_W: f32 = 22.0;
/// Body / button font size — the chunky end of "MS Sans Serif at
/// 8pt", scaled up for legible nostalgia.
pub const FONT_BODY: f32 = 15.0;
/// Title-strip + heading font size.
pub const FONT_HEADING: f32 = 16.0;
/// Help-strip font size — slightly smaller than body so the strip
/// reads as secondary chrome.
pub const FONT_HELP: f32 = 13.0;
