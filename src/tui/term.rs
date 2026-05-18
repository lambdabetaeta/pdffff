//! Terminal lifecycle: raw mode, alternate screen, panic hook.
//!
//! Kept off the render hot path so the rest of the TUI module can
//! ignore crossterm setup/teardown noise.

use anyhow::{Context, Result};
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{
        EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
    },
};
use ratatui::{Terminal, backend::CrosstermBackend};
use std::io::{self, Stdout};

pub(super) fn setup() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode().context("enable_raw_mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)
        .context("entering alternate screen")?;
    Terminal::new(CrosstermBackend::new(stdout))
        .context("constructing ratatui Terminal")
}

pub(super) fn restore(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    disable_raw_mode().context("disable_raw_mode")?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)
        .context("leaving alternate screen")?;
    terminal.show_cursor().context("restoring cursor")?;
    Ok(())
}

/// Wrap the default panic hook so a panic inside the render loop
/// restores the terminal before the panic message is printed.
/// Without this the user's shell would be stuck in raw mode / alt
/// screen on any internal panic, which is the worst possible UX.
pub(super) fn install_panic_hook() {
    let original = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture);
        original(panic_info);
    }));
}
