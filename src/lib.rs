//! pdffff — durable, fast PDF folder search.
//!
//! The library is split along the boundaries described in `deep-research-report.md`:
//!
//! * [`db`]         – SQLite schema, migrations, and statements (durability).
//! * [`extract`]    – `pdftotext` invocation, normalization, chunking.
//! * [`scanner`]    – filesystem walk + diff against `documents`.
//! * [`watcher`]    – debounced filesystem events feeding the scanner.
//! * [`bigram`]     – dense bigram posting-list index over chunks.
//! * [`bigram_query`] – regex/fuzzy → bigram-query decomposition and evaluation.
//! * [`index`]      – `BaseIndex` + `Overlay` glued together with `arc-swap`.
//! * [`query`]      – literal / regex / fuzzy search with snippet rendering.
//! * [`app`]        – top-level orchestrator (workers, channels, lifecycle).

pub mod app;
pub mod bigram;
pub mod bigram_query;
pub mod db;
pub mod extract;
pub mod index;
pub mod normalize;
pub mod query;
pub mod scanner;
pub mod watcher;
