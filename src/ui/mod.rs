//! UI-agnostic plumbing shared by the TUI and the GUI.
//!
//! Both frontends draw the same data (a query, a list of `Hit`s with
//! highlighted snippets, an indexer progress snapshot) on top of the
//! same `IndexState`. The only differences live at the rendering layer
//! — colour palette, layout primitives, input handling. This module
//! owns everything strictly above the rendering layer:
//!
//! * [`highlight`] — neutral snippet/title highlighting that returns
//!   frontend-agnostic [`highlight::SnippetSegment`]s. Each frontend
//!   maps segments to its own widget vocabulary.
//! * [`search`] — the off-thread search worker every frontend uses to
//!   keep input responsive against a large corpus. One-slot mailbox,
//!   stamp-based stale-result rejection.
//!
//! Each frontend lives in its own module / binary on top of this
//! kernel.

pub mod highlight;
pub mod search;

#[cfg(feature = "gui")]
pub mod gui;
