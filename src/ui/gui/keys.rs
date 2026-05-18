//! Global keystroke dispatch for the GUI.
//!
//! Every shortcut the GUI handles is consumed from egui's input queue
//! *before* any widget sees it, so Tab routes to mode-cycling rather
//! than focus traversal and Ctrl+U / Ctrl+W never reach the TextEdit
//! as characters.

use eframe::egui;

use crate::ui::input::{cycle_mode, move_selection, word_erase};

use super::GuiApp;

/// Snapshot of which shortcuts fired this frame.
struct Shortcuts {
    esc: bool,
    enter: bool,
    up: bool,
    down: bool,
    tab: bool,
    page_up: bool,
    page_down: bool,
    ctrl_u: bool,
    ctrl_w: bool,
}

fn consume_shortcuts(ctx: &egui::Context) -> Shortcuts {
    ctx.input_mut(|i| {
        let none = egui::Modifiers::NONE;
        let ctrl = egui::Modifiers::COMMAND;
        Shortcuts {
            esc: i.consume_key(none, egui::Key::Escape),
            enter: i.consume_key(none, egui::Key::Enter),
            up: i.consume_key(none, egui::Key::ArrowUp),
            down: i.consume_key(none, egui::Key::ArrowDown),
            tab: i.consume_key(none, egui::Key::Tab),
            page_up: i.consume_key(none, egui::Key::PageUp),
            page_down: i.consume_key(none, egui::Key::PageDown),
            ctrl_u: i.consume_key(ctrl, egui::Key::U),
            ctrl_w: i.consume_key(ctrl, egui::Key::W),
        }
    })
}

pub(super) fn handle_global_keys(app: &mut GuiApp, ctx: &egui::Context) {
    let s = consume_shortcuts(ctx);
    if s.esc {
        app.closing = true;
    }
    if s.enter {
        app.pick_selected();
    }
    if s.up {
        app.selected = move_selection(app.selected, app.hits.len(), -1);
    }
    if s.down {
        app.selected = move_selection(app.selected, app.hits.len(), 1);
    }
    if s.page_up {
        app.selected = move_selection(app.selected, app.hits.len(), -10);
    }
    if s.page_down {
        app.selected = move_selection(app.selected, app.hits.len(), 10);
    }
    if s.tab {
        app.mode = cycle_mode(app.mode);
        app.submit_query();
    }
    if s.ctrl_u {
        app.query.clear();
        app.submit_query();
    }
    if s.ctrl_w {
        word_erase(&mut app.query);
        app.submit_query();
    }
}
