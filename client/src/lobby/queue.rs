use std::sync::mpsc::{Receiver, SyncSender};
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph},
    Terminal,
};
use std::io;

use novacana_engine::protocol::{ClientMsg, ServerMsg};

pub enum QueueResult {
    /// A game was found; players list is the match.
    GameFound { players: Vec<String> },
    /// The user cancelled and returned to the main menu.
    Cancelled,
}

/// Shows queue size picker, then waiting screen. Returns when a game is found or cancelled.
pub fn run(
    msg_tx: &SyncSender<ClientMsg>,
    msg_rx: &Receiver<ServerMsg>,
) -> io::Result<QueueResult> {
    let size = match pick_size()? {
        Some(s) => s,
        None => return Ok(QueueResult::Cancelled),
    };

    // Send join request.
    if msg_tx.send(ClientMsg::JoinQueue { size }).is_err() {
        return Ok(QueueResult::Cancelled);
    }

    // Wait in queue.
    wait_in_queue(size, msg_tx, msg_rx)
}

fn pick_size() -> io::Result<Option<u8>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let sizes: &[u8] = &[2, 3, 4];
    let mut selected = 0usize;

    let result = loop {
        terminal.draw(|f| {
            let area = f.area();
            f.render_widget(Clear, area);

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(40), Constraint::Min(5), Constraint::Length(1)])
                .split(area);

            let title = Paragraph::new(Line::from(Span::styled(
                "NOVACANA - Play",
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            )))
            .alignment(Alignment::Center);
            f.render_widget(title, chunks[0]);

            let items: Vec<ListItem> = sizes.iter().enumerate().map(|(i, &s)| {
                let label = format!("  {}  {}-Player Game  ", if i == selected { "▶" } else { " " }, s);
                if i == selected {
                    ListItem::new(label).style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
                } else {
                    ListItem::new(label).style(Style::default().fg(Color::White))
                }
            }).collect();

            let list = List::new(items).block(Block::default().borders(Borders::NONE));
            f.render_widget(list, chunks[1]);

            let hint = Paragraph::new(Line::from(Span::styled(
                "↑/↓ select · Enter join · Esc back",
                Style::default().fg(Color::DarkGray),
            )))
            .alignment(Alignment::Center);
            f.render_widget(hint, chunks[2]);
        })?;

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('Q') => { break None; }
                    KeyCode::Up | KeyCode::Char('k') => { selected = selected.saturating_sub(1); }
                    KeyCode::Down | KeyCode::Char('j') => { selected = (selected + 1).min(sizes.len() - 1); }
                    KeyCode::Enter | KeyCode::Char(' ') => { break Some(sizes[selected]); }
                    _ => {}
                }
            }
        }
    };

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(result)
}

fn wait_in_queue(size: u8, msg_tx: &SyncSender<ClientMsg>, msg_rx: &Receiver<ServerMsg>) -> io::Result<QueueResult> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut dots = 0u32;
    let result = loop {
        dots = (dots + 1) % 4;
        let spinner = ".".repeat(dots as usize);

        terminal.draw(|f| {
            let area = f.area();
            f.render_widget(Clear, area);

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(45), Constraint::Length(2), Constraint::Length(1)])
                .split(area);

            let msg = format!("In queue for {}-player game{}", size, spinner);
            let waiting = Paragraph::new(Line::from(Span::styled(msg, Style::default().fg(Color::Cyan))))
                .alignment(Alignment::Center);
            f.render_widget(waiting, chunks[1]);

            let hint = Paragraph::new(Line::from(Span::styled(
                "C to cancel",
                Style::default().fg(Color::DarkGray),
            )))
            .alignment(Alignment::Center);
            f.render_widget(hint, chunks[2]);
        })?;

        // Poll for server messages.
        match msg_rx.try_recv() {
            Ok(ServerMsg::GameFound { players }) => {
                break QueueResult::GameFound { players };
            }
            Ok(ServerMsg::QueueLeft) => {
                break QueueResult::Cancelled;
            }
            _ => {}
        }

        if event::poll(Duration::from_millis(250))? {
            if let Event::Key(key) = event::read()? {
                if matches!(key.code, KeyCode::Char('c') | KeyCode::Char('C') | KeyCode::Esc) {
                    let _ = msg_tx.send(ClientMsg::LeaveQueue);
                    break QueueResult::Cancelled;
                }
            }
        }
    };

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(result)
}
