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
    widgets::{Block, Borders, Clear, Paragraph},
    Terminal,
};
use std::io;

use novacana_engine::protocol::{ClientMsg, ServerMsg};

#[derive(Clone, Copy, PartialEq)]
enum AuthMode { Login, Register }

#[derive(Clone, Copy, PartialEq)]
enum Field { Username, Password }

pub struct AuthResult {
    pub username: String,
}

/// Runs the login/register screen. Returns the logged-in username, or None to quit.
pub fn run(
    msg_tx: &SyncSender<ClientMsg>,
    msg_rx: &Receiver<ServerMsg>,
) -> io::Result<Option<AuthResult>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut mode = AuthMode::Login;
    let mut username = String::new();
    let mut password = String::new();
    let mut active_field = Field::Username;
    let mut status: Option<(String, bool)> = None; // (message, is_error)
    let mut waiting = false;

    let result = loop {
        terminal.draw(|f| {
            let area = f.area();
            f.render_widget(Clear, area);

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Percentage(25),
                    Constraint::Length(1),
                    Constraint::Length(3),
                    Constraint::Length(3),
                    Constraint::Length(3),
                    Constraint::Length(1),
                    Constraint::Percentage(40),
                ])
                .split(area);

            let title_style = Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD);
            let title = Paragraph::new(vec![
                Line::from(Span::styled("NOVACANA", title_style)),
            ])
            .alignment(Alignment::Center);
            f.render_widget(title, chunks[0]);

            // Mode toggle
            let (login_style, reg_style) = if mode == AuthMode::Login {
                (
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
                    Style::default().fg(Color::DarkGray),
                )
            } else {
                (
                    Style::default().fg(Color::DarkGray),
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
                )
            };
            let mode_line = Paragraph::new(Line::from(vec![
                Span::styled("  [Tab] ", Style::default().fg(Color::DarkGray)),
                Span::styled("Login", login_style),
                Span::styled("  /  ", Style::default().fg(Color::DarkGray)),
                Span::styled("Register", reg_style),
                Span::styled("  ", Style::default()),
            ]))
            .alignment(Alignment::Center);
            f.render_widget(mode_line, chunks[1]);

            let username_border_color = if active_field == Field::Username { Color::Cyan } else { Color::DarkGray };
            let username_field = Paragraph::new(username.as_str())
                .block(Block::default().borders(Borders::ALL).title(" Username ").border_style(Style::default().fg(username_border_color)));
            f.render_widget(username_field, chunks[2]);

            let password_border_color = if active_field == Field::Password { Color::Cyan } else { Color::DarkGray };
            let masked: String = "*".repeat(password.len());
            let password_field = Paragraph::new(masked.as_str())
                .block(Block::default().borders(Borders::ALL).title(" Password ").border_style(Style::default().fg(password_border_color)));
            f.render_widget(password_field, chunks[3]);

            let action_label = if waiting {
                "..."
            } else if mode == AuthMode::Login {
                "Enter to Log In"
            } else {
                "Enter to Register"
            };
            let action = Paragraph::new(Line::from(Span::styled(action_label, Style::default().fg(Color::DarkGray))))
                .alignment(Alignment::Center);
            f.render_widget(action, chunks[4]);

            if let Some((ref msg, is_error)) = status {
                let color = if is_error { Color::Red } else { Color::Green };
                let status_widget = Paragraph::new(Line::from(Span::styled(msg.clone(), Style::default().fg(color))))
                    .alignment(Alignment::Center);
                f.render_widget(status_widget, chunks[5]);
            }
        })?;

        // Check for server response when waiting.
        if waiting {
            match msg_rx.try_recv() {
                Ok(ServerMsg::AuthOk { username: uname }) => {
                    let uname_clone = uname.clone();
                    disable_raw_mode()?;
                    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
                    return Ok(Some(AuthResult { username: uname_clone }));
                }
                Ok(ServerMsg::AuthErr { reason }) => {
                    waiting = false;
                    status = Some((reason, true));
                }
                _ => {}
            }
        }

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if waiting { continue; }
                match key.code {
                    KeyCode::Esc => { break None; }
                    KeyCode::Tab => {
                        mode = if mode == AuthMode::Login { AuthMode::Register } else { AuthMode::Login };
                        status = None;
                    }
                    KeyCode::Enter => {
                        if username.is_empty() {
                            status = Some(("Username cannot be empty".into(), true));
                        } else if password.is_empty() {
                            status = Some(("Password cannot be empty".into(), true));
                        } else {
                            status = None;
                            waiting = true;
                            let msg = if mode == AuthMode::Login {
                                ClientMsg::Login { username: username.clone(), password: password.clone() }
                            } else {
                                ClientMsg::Register { username: username.clone(), password: password.clone() }
                            };
                            if msg_tx.send(msg).is_err() {
                                break None;
                            }
                        }
                    }
                    KeyCode::BackTab => {
                        active_field = match active_field {
                            Field::Username => Field::Password,
                            Field::Password => Field::Username,
                        };
                    }
                    KeyCode::Down | KeyCode::Char('\t') => {
                        active_field = match active_field {
                            Field::Username => Field::Password,
                            Field::Password => Field::Username,
                        };
                    }
                    KeyCode::Up => {
                        active_field = match active_field {
                            Field::Username => Field::Password,
                            Field::Password => Field::Username,
                        };
                    }
                    KeyCode::Char(c) => {
                        match active_field {
                            Field::Username => username.push(c),
                            Field::Password => password.push(c),
                        }
                        status = None;
                    }
                    KeyCode::Backspace => {
                        match active_field {
                            Field::Username => { username.pop(); }
                            Field::Password => { password.pop(); }
                        }
                        status = None;
                    }
                    _ => {}
                }
            }
        }
    };

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(result)
}
