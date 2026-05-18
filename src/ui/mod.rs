//! UI-agnostic plumbing shared by the TUI and the GUI.
//!
//! Both frontends draw the same data (a query, a list of `Hit`s with
//! highlighted snippets, an indexer progress snapshot) on top of the
//! same `IndexState`. The only differences live at the rendering layer
//! — colour palette, layout primitives, drawing API. This module owns
//! everything strictly above the rendering layer:
//!
//! * [`highlight`] — neutral snippet/title highlighting that returns
//!   frontend-agnostic [`highlight::SnippetSegment`]s. Each frontend
//!   maps segments to its own widget vocabulary.
//! * [`input`] — pure input helpers (`cycle_mode`, `word_erase`,
//!   `move_selection`) that both frontends call on the matching key
//!   events.
//! * [`launch`] — `OnPick` policy + `open_in_system_viewer` helper.
//! * [`search`] — the off-thread search worker every frontend uses to
//!   keep input responsive against a large corpus. One-slot mailbox,
//!   stamp-based stale-result rejection.
//! * [`spinner`] — shared Braille spinner frames + cadence.
//!
//! Each frontend lives in its own module / binary on top of this
//! kernel.

pub mod highlight;
pub mod input;
pub mod launch;
pub mod search;
pub mod spinner;

#[cfg(feature = "gui")]
pub mod gui;
