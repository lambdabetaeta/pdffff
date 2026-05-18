//! What to do when the user activates a result, and the helper that
//! does the default thing.
//!
//! The default — `OpenInViewer` — hands the chosen path to the
//! host's PDF viewer via `open_in_system_viewer` and keeps the
//! search session running so the user can pick more files. The
//! alternative — `SelectAndExit` — is a "selector" mode that exits
//! immediately and yields the chosen `Hit` back to the launcher, so
//! pdffff composes with shell pipelines:
//!
//! ```sh
//! cd $(pdffff /papers --print-path | xargs dirname)
//! ```
//!
//! The viewer launcher (when used) spawns the platform-specific
//! opener detached (`stdout`/`stderr`/`stdin` nulled, no `wait()`) so
//! a viewer that prints to its launching terminal cannot corrupt the
//! alternate screen the TUI owns or the egui event loop the GUI
//! owns, and so the search session stays responsive while the viewer
//! initialises.

use anyhow::{Context, Result};
use std::process::{Command, Stdio};

/// What pressing Enter on a selected result does.
///
/// `OnPick` is plumbed through both `TuiOptions` and `GuiOptions` so
/// the binary launchers can pick the behaviour from a CLI flag
/// without either frontend re-implementing the dispatch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OnPick {
    /// Open the file in the host's PDF viewer. The search session
    /// stays alive — Enter can be pressed again for more results.
    #[default]
    OpenInViewer,
    /// Exit the frontend immediately and return the chosen `Hit` to
    /// the launcher (which prints the path to stdout, so pdffff
    /// composes with shell pipelines).
    SelectAndExit,
}

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
