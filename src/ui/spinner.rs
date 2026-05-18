//! Shared spinner animation for the TUI and GUI.
//!
//! Both frontends drive a single Braille spinner against the same
//! search-worker mailbox, so the activity signal looks identical
//! across the two frontends. Centralising the frame table and the
//! `Duration → frame` lookup means a future palette tweak (denser
//! frames, slower cadence) propagates to every frontend.

use std::time::Duration;

/// Braille spinner frames. Same set ratatui's default examples ship,
/// chosen because the glyphs land on a single column at every common
/// terminal font width.
pub const FRAMES: [&str; 10] =
    ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// One animation tick. Both frontends repaint at this cadence while
/// the indexer (or a search) is busy. Public so the GUI's idle
/// `request_repaint_after` and the TUI's `event::poll` stay in sync.
pub const TICK: Duration = Duration::from_millis(100);

/// Pick the spinner frame for the given elapsed time. Wraps cleanly at
/// the end of the table so the caller does not need to track an index.
pub fn frame_at(elapsed: Duration) -> &'static str {
    let idx = (elapsed.as_millis() / TICK.as_millis().max(1)) as usize % FRAMES.len();
    FRAMES[idx]
}
