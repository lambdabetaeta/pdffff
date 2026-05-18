//! Norton-Commander-revival palette for the TUI.
//!
//! One hue, one job (Bertin's "semiology of graphics" applied to a
//! 16-colour TTY). Background colour is reserved exclusively for query
//! matches — the most important pop-out signal in the UI — so
//! selection is signalled on the border (hue + bold) instead of as a
//! row background. Named / 256-indexed terminal colours throughout, so
//! the UI inherits the user's terminal theme rather than baking in
//! RGB.

use ratatui::style::{Color, Modifier, Style};

/// Chrome — outer & card borders, prompt arrow, brand pill, key
/// chips, mode pill background, separator rule. Saturated blue, the
/// classic Norton-Commander-era status colour.
pub const CHROME: Color = Color::Blue;
/// Secondary text — meta lines, counters, idle/indexing status,
/// numbering, separators. Regular ANSI gray (7) rather than
/// "bright black" (8 / DarkGray) which is near-invisible on
/// true-black terminals.
pub const DIM: Color = Color::Gray;
/// Primary text — query, filenames, brand wordmark, mode pill label.
pub const PRIMARY: Color = Color::White;
/// Focus accent — the only place magenta appears, applied to the
/// border of the currently-selected card.
pub const SEL: Color = Color::Magenta;
/// Match highlight — reserved exclusively for query matches inside
/// snippet bodies and card-title filenames. Reverse-video against
/// yellow gives the highest selective power Bertin's "value"
/// variable can offer on a 16-colour terminal.
pub const HL_BG: Color = Color::Yellow;
pub const HL_FG: Color = Color::Black;
/// Error pill background. The single use of red in the UI.
pub const ERROR: Color = Color::Red;

/// The single match-highlight style, used identically in card titles
/// and snippet bodies. Centralising it makes the "yellow background is
/// reserved for matches" invariant a one-liner to audit.
pub fn match_hl_style() -> Style {
    Style::default()
        .bg(HL_BG)
        .fg(HL_FG)
        .add_modifier(Modifier::BOLD)
}
