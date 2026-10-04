use crossterm::event::{self, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
    Terminal,
};
use std::io;

pub struct ConnectResult {
    pub host: String,
    pub port: u16,
}

/// Shows the connection screen and returns the host:port entered by the user.
/// Returns `None` if the user pressed Esc/Q to quit.
pub fn run() -> io::Result<Option<ConnectResult>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut input = String::from("localhost:7878");
    let mut cursor = input.len();
    let mut error: Option<String> = None;

    let result = loop {
        terminal.draw(|f| {
            let area = f.area();
            f.render_widget(Clear, area);

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Percentage(35),
                    Constraint::Length(3),
                    Constraint::Length(3),
                    Constraint::Percentage(50),
                ])
                .split(area);

            let title = Paragraph::new(vec![
                Line::from(Span::styled(
                    "NOVACANA",
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                )),
                Line::from(Span::styled(
                    "Connect to Server",
                    Style::default().fg(Color::DarkGray),
                )),
            ])
            .alignment(Alignment::Center);
            f.render_widget(title, chunks[0]);

            let display = format!("{}█", &input[..cursor]);
            let rest = &input[cursor..];
            let field = Paragraph::new(Line::from(vec![
                Span::raw(display),
                Span::styled(rest, Style::default().fg(Color::DarkGray)),
            ]))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Host:Port ")
                    .border_style(Style::default().fg(Color::Cyan)),
            );
            f.render_widget(field, chunks[1]);

            let hint_text = if let Some(ref err) = error {
                Line::from(Span::styled(err.clone(), Style::default().fg(Color::Red)))
            } else {
                Line::from(Span::styled(
                    "Enter to connect · Esc/Q to quit",
                    Style::default().fg(Color::DarkGray),
                ))
            };
            let hint = Paragraph::new(hint_text).alignment(Alignment::Center);
            f.render_widget(hint, chunks[2]);
        })?;

        if event::poll(std::time::Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('Q') => {
                        break None;
                    }
                    KeyCode::Enter => {
                        match parse_host_port(&input) {
                            Some(r) => break Some(r),
                            None => error = Some("Invalid format - use host:port (e.g. localhost:7878)".into()),
                        }
                    }
                    KeyCode::Char(c) => {
                        input.insert(cursor, c);
                        cursor += 1;
                        error = None;
                    }
                    KeyCode::Backspace if cursor > 0 => {
                        cursor -= 1;
                        input.remove(cursor);
                        error = None;
                    }
                    KeyCode::Delete if cursor < input.len() => {
                        input.remove(cursor);
                        error = None;
                    }
                    KeyCode::Left if cursor > 0 => { cursor -= 1; }
                    KeyCode::Right if cursor < input.len() => { cursor += 1; }
                    KeyCode::Home => { cursor = 0; }
                    KeyCode::End => { cursor = input.len(); }
                    _ => {}
                }
            }
        }
    };

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(result)
}

fn parse_host_port(s: &str) -> Option<ConnectResult> {
    let (host, port_str) = s.rsplit_once(':')?;
    let port: u16 = port_str.parse().ok()?;
    if host.is_empty() { return None; }
    Some(ConnectResult { host: host.to_string(), port })
}
