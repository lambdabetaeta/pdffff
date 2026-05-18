//! All `render_*` functions for the TUI.
//!
//! The render layer is pure: it takes `&AppState` + the latest
//! [`ProgressSnapshot`] and produces ratatui widgets. State mutation
//! lives in [`super::keys`] and [`super::AppState`]; this module owns
//! presentation only.

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Position, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Padding, Paragraph},
};

use crate::app::ProgressSnapshot;
use crate::query::{Hit, QueryMode};
use crate::ui::highlight::{SegmentKind, SnippetSegment, highlight_segments};
use crate::ui::spinner::{TICK, frame_at};

use super::layout::{CARD_GAP, CARD_HEIGHT, compute_scroll_top, visible_card_count};
use super::palette::{CHROME, DIM, ERROR, PRIMARY, SEL, match_hl_style};
use super::{AppState, TuiOptions};

pub(super) fn render(
    f: &mut Frame,
    state: &mut AppState,
    opts: &TuiOptions,
    snap: ProgressSnapshot,
) {
    let size = f.area();
    let outer = build_outer_frame(&snap, opts, state);
    let inner = outer.inner(size);
    f.render_widget(outer, size);

    // Vertical layout *inside* the outer frame.
    //   row 0  : top breathing room
    //   row 1  : input + mode pill
    //   row 2  : separator (or inline error)
    //   row 3  : breathing room before the result cards
    //   rest   : result cards
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(1),
        ])
        .split(inner);

    render_input(f, rows[1], state);
    render_error_or_hr(f, rows[2], state);
    render_results(f, rows[4], state);
}

/// The outer rounded frame: brand pill + watched root on the top-left,
/// live counters + indexing spinner on the top-right, keybinding help
/// on the bottom.
fn build_outer_frame<'a>(
    snap: &ProgressSnapshot,
    opts: &'a TuiOptions,
    state: &AppState,
) -> Block<'a> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(CHROME))
        .padding(Padding::new(1, 1, 0, 0))
        .title_top(top_left_title(opts))
        .title_top(top_right_title(snap, state))
        .title_bottom(help_line())
}

fn top_left_title(opts: &TuiOptions) -> Line<'_> {
    let brand = Span::styled(
        " pdffff ",
        Style::default().bg(CHROME).fg(PRIMARY).add_modifier(Modifier::BOLD),
    );
    let root = Span::styled(
        format!(" {} ", opts.root.display()),
        Style::default().fg(PRIMARY).add_modifier(Modifier::BOLD),
    );
    Line::from(vec![Span::raw(" "), brand, root]).left_aligned()
}

fn top_right_title<'a>(snap: &ProgressSnapshot, state: &AppState) -> Line<'a> {
    let counters = Span::styled(
        format!(
            "{} ok · {} empty · {} err · {} del",
            snap.ok, snap.empty, snap.error, snap.deleted,
        ),
        Style::default().fg(DIM),
    );
    // Both branches use the same dim style — the spinner motion is the
    // activity signal, no extra colour required (Norton-Commander-style
    // status row, not a modern progress bar).
    let activity = if snap.pending > 0 {
        Span::styled(
            format!(" {} indexing {} ", frame_at(state.spinner_started.elapsed()), snap.pending),
            Style::default().fg(DIM),
        )
    } else {
        Span::styled(" idle ", Style::default().fg(DIM))
    };
    Line::from(vec![
        counters,
        Span::styled(" · ", Style::default().fg(DIM)),
        activity,
        Span::raw(" "),
    ])
    .right_aligned()
}

fn help_line<'a>() -> Line<'a> {
    Line::from(vec![
        Span::raw(" "),
        key_chip("↑↓"),
        Span::raw(" select  "),
        key_chip("Tab"),
        Span::raw(" mode  "),
        key_chip("Enter"),
        Span::raw(" open  "),
        key_chip("Ctrl+U"),
        Span::raw(" clear  "),
        key_chip("Esc"),
        Span::raw(" quit "),
    ])
    .left_aligned()
    .style(Style::default().fg(DIM))
}

/// A keybinding label, styled in the chrome hue against the dim help
/// footer.
fn key_chip(text: &str) -> Span<'_> {
    Span::styled(
        text,
        Style::default().fg(CHROME).add_modifier(Modifier::BOLD),
    )
}

fn render_input(f: &mut Frame, area: Rect, state: &AppState) {
    // Short labels: mode is a nominal distinction, and the label text
    // itself encodes it. We do not lean on hue here — every mode pill
    // shares the same chrome background, so hue stays free to do its
    // one job (matches).
    let mode_label = match state.mode {
        QueryMode::Literal => " LIT ",
        QueryMode::Regex => " RE ",
        QueryMode::Fuzzy => " FZ ",
    };
    let mode_w = mode_label.chars().count() as u16;

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(8), Constraint::Length(mode_w)])
        .split(area);

    render_prompt(f, cols[0], state);
    render_mode_pill(f, cols[1], mode_label);
}

/// Prompt glyph + query text, plus the real terminal cursor.
///
/// Glyph is a chevron when idle and an animated spinner frame while a
/// search is in flight — same `applied_stamp != submitted_stamp`
/// contract as the GUI's prompt spinner.
fn render_prompt(f: &mut Frame, area: Rect, state: &AppState) {
    let prompt_glyph = if state.is_searching() {
        frame_at(state.spinner_started.elapsed())
    } else {
        "❯"
    };
    let prompt = Span::styled(
        format!("{prompt_glyph} "),
        Style::default().fg(CHROME).add_modifier(Modifier::BOLD),
    );
    let query = Span::styled(
        state.query.clone(),
        Style::default().fg(PRIMARY).add_modifier(Modifier::BOLD),
    );
    f.render_widget(Paragraph::new(Line::from(vec![prompt, query])), area);

    // Real terminal cursor at the end of the query, clamped to the
    // visible width of the input column. Visual width of `❯ ` is 2.
    const PROMPT_W: u16 = 2;
    let query_chars = state.query.chars().count() as u16;
    let cursor_x = area
        .x
        .saturating_add(PROMPT_W)
        .saturating_add(query_chars);
    let max_x = area.x.saturating_add(area.width.saturating_sub(1));
    f.set_cursor_position(Position::new(cursor_x.min(max_x), area.y));

    // Touch the TICK constant — keeps the dependency direction
    // explicit (renderer takes its cadence from the spinner module).
    let _: std::time::Duration = TICK;
}

fn render_mode_pill(f: &mut Frame, area: Rect, label: &'static str) {
    let pill = Paragraph::new(Line::from(Span::styled(
        label,
        Style::default()
            .bg(CHROME)
            .fg(PRIMARY)
            .add_modifier(Modifier::BOLD),
    )))
    .alignment(Alignment::Right);
    f.render_widget(pill, area);
}

/// When there is a query error, render it under the input as a red
/// pill plus message; otherwise draw a dim separator across the
/// content area.
fn render_error_or_hr(f: &mut Frame, area: Rect, state: &AppState) {
    if let Some(err) = &state.last_error {
        let label = Span::styled(
            " error ",
            Style::default()
                .bg(ERROR)
                .fg(PRIMARY)
                .add_modifier(Modifier::BOLD),
        );
        // Reserve room for the label, two spaces of padding, and a
        // little safety margin so we never wrap to a second line.
        let budget = area.width.saturating_sub(10) as usize;
        let truncated: String = err.chars().take(budget).collect();
        f.render_widget(
            Paragraph::new(Line::from(vec![
                label,
                Span::raw(" "),
                Span::styled(truncated, Style::default().fg(ERROR)),
            ])),
            area,
        );
    } else if area.width > 0 {
        let bar = "─".repeat(area.width as usize);
        f.render_widget(
            Paragraph::new(bar).style(Style::default().fg(CHROME)),
            area,
        );
    }
}

fn render_results(f: &mut Frame, area: Rect, state: &mut AppState) {
    if state.hits.is_empty() {
        let msg = if state.query.trim().is_empty() {
            "type a query to search the index"
        } else {
            "no hits"
        };
        let placeholder = Paragraph::new(Line::from(Span::styled(
            msg,
            Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
        )))
        .alignment(Alignment::Center);
        f.render_widget(placeholder, area);
        return;
    }

    let visible = visible_card_count(area.height);
    if visible == 0 {
        return;
    }
    let selected = state.list_state.selected().unwrap_or(0);
    state.scroll_top =
        compute_scroll_top(selected, state.hits.len(), visible, state.scroll_top);
    let scroll_top = state.scroll_top;

    let n = visible.min(state.hits.len() - scroll_top);
    for slot in 0..n {
        let i = scroll_top + slot;
        let y = area.y + slot as u16 * (CARD_HEIGHT + CARD_GAP);
        if y + CARD_HEIGHT > area.y + area.height {
            break;
        }
        let card_area = Rect::new(area.x, y, area.width, CARD_HEIGHT);
        render_hit_card(
            f,
            card_area,
            i,
            &state.hits[i],
            &state.query,
            state.mode,
            i == selected,
        );
    }
}

fn render_hit_card(
    f: &mut Frame,
    area: Rect,
    i: usize,
    hit: &Hit,
    query: &str,
    mode: QueryMode,
    selected: bool,
) {
    let block = card_block(i, hit, query, mode, selected);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let snippet_spans =
        highlight_spans(&hit.snippet, query, Style::default(), match_hl_style());
    f.render_widget(Paragraph::new(Line::from(snippet_spans)), inner);
}

fn card_block<'a>(
    i: usize,
    hit: &'a Hit,
    query: &str,
    mode: QueryMode,
    selected: bool,
) -> Block<'a> {
    // Score is only meaningful in fuzzy mode (it's the neo_frizbee u16
    // turned into f32). For literal and regex matches it is a hardcoded
    // `1.0` — showing it there is noise, so we hide the field.
    let meta = match mode {
        QueryMode::Fuzzy => format!(
            " p.{} · #{} · score {:.2} ",
            hit.page_no, hit.chunk_ord, hit.score,
        ),
        QueryMode::Literal | QueryMode::Regex => {
            format!(" p.{} · #{} ", hit.page_no, hit.chunk_ord)
        }
    };

    // Display the filename rather than the full path — the full path
    // is typically the watched root + a long suffix, which crowds the
    // header off the right edge and adds no information the corpus
    // context doesn't already supply.
    let name_base = Style::default().fg(PRIMARY).add_modifier(Modifier::BOLD);
    let filename_spans = highlight_spans(&hit.filename, query, name_base, match_hl_style());

    let mut title_left_spans: Vec<Span<'static>> = vec![
        Span::raw(" "),
        Span::styled(format!("{}. ", i + 1), Style::default().fg(DIM)),
    ];
    title_left_spans.extend(filename_spans);
    title_left_spans.push(Span::raw(" "));
    let title_left = Line::from(title_left_spans).left_aligned();

    let title_right =
        Line::from(Span::styled(meta, Style::default().fg(DIM))).right_aligned();

    let border_style = if selected {
        Style::default().fg(SEL).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(CHROME)
    };
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(border_style)
        .title_top(title_left)
        .title_top(title_right)
}

/// Thin adapter: ask the shared highlighter where the matches are,
/// then paint each segment with `base` / `hl`.
pub(super) fn highlight_spans(
    text: &str,
    query: &str,
    base: Style,
    hl: Style,
) -> Vec<Span<'static>> {
    highlight_segments(text, query)
        .into_iter()
        .map(|SnippetSegment { text, kind }| {
            let style = match kind {
                SegmentKind::Plain => base,
                SegmentKind::Match => hl,
            };
            Span::styled(text, style)
        })
        .collect()
}
