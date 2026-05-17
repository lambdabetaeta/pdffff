//! Knobs for [`super::run_watch`].

use std::time::Duration;

#[derive(Debug, Clone)]
pub struct WatchOptions {
    pub respect_gitignore: bool,
    pub follow_symlinks: bool,
    /// Override extractor pool size. Default: `min(num_cpus, 6)`.
    pub jobs: Option<usize>,
    /// If true, fail fast at startup when `pdftotext` is missing. Tests
    /// can disable this only when they have pre-checked.
    pub require_pdftotext: bool,
    /// Debounce window for the filesystem watcher. `None` ⇒ the
    /// watcher module's default ([`crate::watcher::DEFAULT_DEBOUNCE`]).
    pub debounce: Option<Duration>,
}

impl Default for WatchOptions {
    fn default() -> Self {
        Self {
            respect_gitignore: false,
            follow_symlinks: false,
            jobs: None,
            require_pdftotext: true,
            debounce: None,
        }
    }
}
