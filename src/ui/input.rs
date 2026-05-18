//! Frontend-agnostic input helpers shared by the TUI and the GUI.
//!
//! The query bar lives in two completely different widget vocabularies
//! (ratatui's `ListState`, egui's `TextEdit`), but the *editing logic*
//! is identical: cycling the query mode, word-erasing the trailing
//! token, clamping a delta-driven selection inside the hit list. This
//! module owns those operations so both frontends call the same
//! function rather than re-implementing it side-by-side.

use crate::query::QueryMode;

/// Cycle Literal → Regex → Fuzzy → Literal. Both frontends bind Tab
/// to this; centralising the order keeps the two cycle modes in lock-
/// step.
pub fn cycle_mode(mode: QueryMode) -> QueryMode {
    match mode {
        QueryMode::Literal => QueryMode::Regex,
        QueryMode::Regex => QueryMode::Fuzzy,
        QueryMode::Fuzzy => QueryMode::Literal,
    }
}

/// Drop the trailing whitespace-bounded word from `q`.
///
/// Matches the readline / shell convention used by Ctrl-W in both
/// frontends: trim trailing whitespace first, then everything back to
/// the next whitespace boundary.
pub fn word_erase(q: &mut String) {
    let trimmed_end = q.trim_end();
    let cut_to = trimmed_end
        .rfind(char::is_whitespace)
        .map(|i| i + 1)
        .unwrap_or(0);
    q.truncate(cut_to);
}

/// Apply `delta` to the current selection and clamp into `0..total`.
///
/// Returns `None` when the list is empty (i.e. nothing to select).
/// Both frontends call this on every navigation key and only need to
/// store the result in their respective widget state.
pub fn move_selection(current: Option<usize>, total: usize, delta: isize) -> Option<usize> {
    if total == 0 {
        return None;
    }
    let n = total as isize;
    let cur = current.map(|i| i as isize).unwrap_or(0);
    Some((cur + delta).clamp(0, n - 1) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycle_mode_walks_all_three_states() {
        assert_eq!(cycle_mode(QueryMode::Literal), QueryMode::Regex);
        assert_eq!(cycle_mode(QueryMode::Regex), QueryMode::Fuzzy);
        assert_eq!(cycle_mode(QueryMode::Fuzzy), QueryMode::Literal);
    }

    #[test]
    fn word_erase_drops_trailing_word() {
        let mut s = String::from("alpha beta gamma");
        word_erase(&mut s);
        assert_eq!(s, "alpha beta ");
        word_erase(&mut s);
        assert_eq!(s, "alpha ");
        word_erase(&mut s);
        assert_eq!(s, "");
    }

    #[test]
    fn word_erase_handles_trailing_whitespace() {
        let mut s = String::from("alpha beta   ");
        word_erase(&mut s);
        assert_eq!(s, "alpha ");
    }

    #[test]
    fn word_erase_on_empty_is_noop() {
        let mut s = String::new();
        word_erase(&mut s);
        assert_eq!(s, "");
    }

    #[test]
    fn move_selection_empty_list_returns_none() {
        assert_eq!(move_selection(Some(0), 0, 1), None);
        assert_eq!(move_selection(None, 0, 0), None);
    }

    #[test]
    fn move_selection_clamps_to_bounds() {
        assert_eq!(move_selection(Some(0), 5, -10), Some(0));
        assert_eq!(move_selection(Some(4), 5, 10), Some(4));
        assert_eq!(move_selection(Some(2), 5, 1), Some(3));
        assert_eq!(move_selection(Some(2), 5, -1), Some(1));
        assert_eq!(move_selection(None, 5, 0), Some(0));
    }
}
