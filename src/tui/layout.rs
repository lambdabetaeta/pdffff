//! Pure-data card geometry + the scroll/visibility maths.
//!
//! Both helpers are pure functions, exercised by their own tests in
//! the parent module. Keeping them away from the render code makes
//! the rules a one-grep proposition.

/// Total vertical rows one result card occupies (top border + one
/// snippet row + bottom border). Snippets are pre-bounded by the
/// snippet builder; we never wrap, so a single content row suffices.
pub const CARD_HEIGHT: u16 = 3;
/// Blank rows between adjacent cards. Bertin: separation is itself a
/// visual variable — explicit empty space says "different unit" more
/// clearly than any divider could.
pub const CARD_GAP: u16 = 1;

/// How many cards fit in `height` rows, given `CARD_HEIGHT` rows per
/// card and `CARD_GAP` blank rows between cards. Returns 0 when even
/// one card would not fit.
pub fn visible_card_count(height: u16) -> usize {
    if height < CARD_HEIGHT {
        return 0;
    }
    // n cards take CARD_HEIGHT + (n-1) * (CARD_HEIGHT + CARD_GAP) rows.
    let extra = height - CARD_HEIGHT;
    1 + (extra / (CARD_HEIGHT + CARD_GAP)) as usize
}

/// Pick a scroll position that keeps `selected` visible while
/// disturbing `prev` as little as possible. The selection moves by
/// user input; the scroll position only shifts when selection would
/// otherwise fall out of frame.
pub fn compute_scroll_top(selected: usize, total: usize, visible: usize, prev: usize) -> usize {
    if total == 0 || visible == 0 {
        return 0;
    }
    let max_top = total.saturating_sub(visible);
    let mut top = prev.min(max_top);
    if selected < top {
        top = selected;
    } else if selected >= top + visible {
        top = selected + 1 - visible;
    }
    top.min(max_top)
}
