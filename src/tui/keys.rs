//! Keystroke → state mutation. Pure dispatch: the four exit keys, the
//! editing keys (each routed through the shared editing helper), the
//! navigation keys, the mode-cycle key, and Enter.
//!
//! Every editing key goes through [`edit_and_search`] so a future
//! keybinding cannot forget to dispatch a fresh search.

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use crate::query::Hit;
use crate::ui::input::{cycle_mode, move_selection, word_erase};
use crate::ui::launch::{OnPick, open_in_system_viewer};
use crate::ui::search::{SearchRequest, SearchWorker};

use super::{AppState, TuiOptions};

pub(super) fn handle_key(
    key: KeyEvent,
    state: &mut AppState,
    worker: &SearchWorker,
    opts: &TuiOptions,
) {
    // Both `Press` and `Repeat` are user-driven; ignore `Release` so we
    // don't double-fire on terminals that surface key-up events.
    if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
        return;
    }
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);

    match key.code {
        // ---- exit keys ---------------------------------------------------
        KeyCode::Esc => state.should_quit = true,
        KeyCode::Char('c' | 'd' | 'q') if ctrl => state.should_quit = true,

        // ---- editing -----------------------------------------------------
        KeyCode::Char('u') if ctrl => edit_and_search(state, worker, opts, String::clear),
        KeyCode::Char('w') if ctrl => edit_and_search(state, worker, opts, word_erase),
        KeyCode::Backspace => edit_and_search(state, worker, opts, |q| {
            q.pop();
        }),
        KeyCode::Char(c)
            if !key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
        {
            edit_and_search(state, worker, opts, |q| q.push(c));
        }

        // ---- navigation --------------------------------------------------
        KeyCode::Down => nav(state, 1),
        KeyCode::Up => nav(state, -1),
        KeyCode::Char('n') if ctrl => nav(state, 1),
        KeyCode::Char('p') if ctrl => nav(state, -1),
        KeyCode::PageDown => nav(state, 10),
        KeyCode::PageUp => nav(state, -10),

        // ---- modes -------------------------------------------------------
        KeyCode::Tab => {
            state.mode = cycle_mode(state.mode);
            submit_query(state, worker, opts.limit);
        }

        // ---- pick a hit --------------------------------------------------
        KeyCode::Enter => pick_selected(state, opts),

        _ => {}
    }
}

/// Mutate `state.query` through `edit` and submit a fresh search.
///
/// Every editing key follows the same shape; centralising it keeps the
/// four branches above to one line each and ensures no future editing
/// key can forget to dispatch a search.
fn edit_and_search(
    state: &mut AppState,
    worker: &SearchWorker,
    opts: &TuiOptions,
    edit: impl FnOnce(&mut String),
) {
    edit(&mut state.query);
    submit_query(state, worker, opts.limit);
}

fn nav(state: &mut AppState, delta: isize) {
    let next = move_selection(state.list_state.selected(), state.hits.len(), delta);
    state.list_state.select(next);
}

/// Behaviour on Enter depends on `opts.on_pick`:
/// * `OpenInViewer` (default) — open the file in the host's PDF
///   viewer; errors land in the in-screen error pill.
/// * `SelectAndExit` — capture the hit and quit so the launcher can
///   print the path to stdout.
fn pick_selected(state: &mut AppState, opts: &TuiOptions) {
    let Some(idx) = state.list_state.selected() else {
        return;
    };
    let Some(hit) = state.hits.get(idx) else {
        return;
    };
    match opts.on_pick {
        OnPick::OpenInViewer => match open_in_system_viewer(&hit.path) {
            Ok(()) => state.last_error = None,
            Err(err) => {
                state.last_error =
                    Some(format!("could not open {}: {err:#}", hit.path));
            }
        },
        OnPick::SelectAndExit => {
            state.chosen = Some(hit.clone());
            state.should_quit = true;
        }
    }
}

/// Bump the query stamp and either clear the UI (empty query) or hand
/// the work to the [`SearchWorker`]. Never blocks: the worker may
/// still be busy on an older query — its mailbox is one-slot, so this
/// just overwrites the pending request.
pub(super) fn submit_query(state: &mut AppState, worker: &SearchWorker, limit: usize) {
    state.submitted_stamp = state.submitted_stamp.wrapping_add(1);
    if state.query.trim().is_empty() {
        state.hits.clear();
        state.list_state.select(None);
        state.scroll_top = 0;
        state.last_error = None;
        state.applied_stamp = state.submitted_stamp;
        return;
    }
    worker.submit(SearchRequest {
        stamp: state.submitted_stamp,
        query: state.query.clone(),
        mode: state.mode,
        limit,
    });
}

/// Pull any pending result from the worker and apply it. Returns true
/// when a fresh result was applied so the caller knows to redraw.
///
/// Results whose stamp predates `state.submitted_stamp` are dropped on
/// the floor — we never paint hits that no longer match the visible
/// query.
pub(super) fn drain_results(state: &mut AppState, worker: &SearchWorker) -> bool {
    let Some(result) = worker.take_result() else {
        return false;
    };
    if result.stamp != state.submitted_stamp {
        return false;
    }
    state.applied_stamp = result.stamp;
    match result.hits {
        Ok(hits) => apply_hits(state, hits),
        Err(err) => {
            // Surface the error without nuking the previous hits — an
            // invalid regex while typing shouldn't blank the screen on
            // every keystroke.
            state.last_error = Some(format!("{err:#}"));
        }
    }
    true
}

fn apply_hits(state: &mut AppState, hits: Vec<Hit>) {
    state.hits = hits;
    state.last_error = None;
    state.list_state.select(if state.hits.is_empty() {
        None
    } else {
        Some(0)
    });
    state.scroll_top = 0;
}
