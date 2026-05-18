//! Hand a chosen file to the host platform's default viewer.
//!
//! Used by both the TUI and the GUI on Enter / double-click. The
//! helper spawns the platform-specific opener detached
//! (`stdout`/`stderr`/`stdin` nulled, no `wait()`) so a viewer that
//! prints to its launching terminal cannot corrupt the alternate
//! screen the TUI owns or the egui event loop the GUI owns, and so
//! the search session stays responsive while the viewer initialises.

use anyhow::{Context, Result};
use std::process::{Command, Stdio};

/// Open `path` in the host's default handler.
///
/// On macOS this is `open`; on Windows `cmd /C start ""` so file
/// associations behave like a double-click; on Linux/BSD/etc.
/// `xdg-open`. Returns as soon as the helper is spawned — the viewer
/// process detaches and runs independently of the caller.
pub fn open_in_system_viewer(path: &str) -> Result<()> {
    #[cfg(target_os = "macos")]
    let mut cmd = {
        let mut c = Command::new("open");
        c.arg(path);
        c
    };
    #[cfg(target_os = "windows")]
    let mut cmd = {
        let mut c = Command::new("cmd");
        c.args(["/C", "start", "", path]);
        c
    };
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let mut cmd = {
        let mut c = Command::new("xdg-open");
        c.arg(path);
        c
    };

    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("launching system viewer for {path}"))?;
    Ok(())
}
