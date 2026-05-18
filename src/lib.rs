//! pdffff — durable, fast PDF folder search.
//!
//! pdffff extracts PDF text once via Poppler's `pdftotext`, persists the
//! result in SQLite, and answers literal / regex / fuzzy queries against
//! an in-process dense bigram inverted index. A small mutation overlay
//! sits over the base index so a `notify-debouncer-full`-based watcher
//! can reflect filesystem changes within ~200 ms without rebuilding.
//!
//! See [`deep-research-report.md`](../../../deep-research-report.md) in
//! the repository root for the architectural rationale, and
//! [`docs/architecture.md`](../../../docs/architecture.md) for the
//! developer-facing layout summary.
//!
//! The library is split along the boundaries described in those docs:
//!
//! * [`db`]           – SQLite schema, migrations, and statements (durability).
//! * [`extract`]      – `pdftotext` invocation, normalization, chunking.
//! * [`normalize`]    – `deunicode` + lowercase + whitespace-collapse pipeline.
//! * [`paths`]        – path-shape predicates (`is_pdf`).
//! * [`scanner`]      – filesystem walk + diff against `documents`.
//! * [`watcher`]      – debounced filesystem events feeding the scanner.
//! * [`bigram`]       – dense bigram posting-list index over chunks.
//! * [`bigram_query`] – regex/fuzzy → bigram-query decomposition and evaluation.
//! * [`index`]        – `BaseIndex` + `Overlay` glued together with `arc-swap`.
//! * [`query`]        – literal / regex / fuzzy search with snippet rendering.
//! * [`app`]          – top-level orchestrator (`run_watch`, `resolve_db_path`).
//!
//! The binary in `src/main.rs` is a pure-TUI launcher: it resolves the
//! per-corpus DB path, redirects tracing to a log file, calls
//! [`app::run_watch`] (which returns immediately and indexes
//! progressively in the background), and hands the resulting
//! `WatchHandle` to [`tui::run_tui`]. The library never writes to
//! stdout itself.

pub mod app;
pub mod bigram;
pub mod bigram_query;
pub mod bitset;
pub mod db;
pub mod extract;
pub mod index;
pub mod normalize;
pub mod paths;
pub mod query;
pub mod scanner;
pub mod tui;
pub mod ui;
pub mod watcher;
