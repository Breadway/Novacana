use std::time::Duration;

use crossterm::event::{self, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
    Terminal,
};
use std::io;

const RULEBOOK: &str = include_str!("../../../../Overview.md");

const TITLE_ART: &str = r"
  _   _  _____  _   _  ___  ___  ___   _   _  ___
 | \ | ||  _  || | | |/ _ \/ __|/ _ \ | \ | |/ _ \
 |  \| || | | || | | | |/ / (__| |/ / |  \| | |/ /
 | |\  || |_| ||\ \_/ /\__ \\__ \\__ \ | |\  /\__ \
 |_| \_|\___/ | \___/ \___/\___/\___/ |_| \_\___/
";

pub enum MenuAction {
    Play,
    Quit,
}

pub fn run(username: &str, online: u32) -> io::Result<MenuAction> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let items = ["Play", "Rulebook", "Card Reference", "Quit"];
    let mut selected = 0usize;

    let result = loop {
        terminal.draw(|f| {
            let area = f.area();
            f.render_widget(Clear, area);

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(7),
                    Constraint::Length(1),
                    Constraint::Min(4),
                    Constraint::Length(1),
                ])
                .split(area);

            let art = Paragraph::new(TITLE_ART)
                .alignment(Alignment::Center)
                .style(Style::default().fg(Color::Cyan));
            f.render_widget(art, chunks[0]);

            let info = Paragraph::new(Line::from(vec![
                Span::styled(format!("  {} ", username), Style::default().fg(Color::Yellow)),
                Span::styled(format!("· {} online", online), Style::default().fg(Color::DarkGray)),
            ]));
            f.render_widget(info, chunks[1]);

            let menu_items: Vec<ListItem> = items.iter().enumerate().map(|(i, name)| {
                if i == selected {
                    ListItem::new(format!("  ▶  {}  ", name))
                        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
                } else {
                    ListItem::new(format!("     {}  ", name))
                        .style(Style::default().fg(Color::White))
                }
            }).collect();

            let menu = List::new(menu_items)
                .block(Block::default().borders(Borders::NONE));
            let mut list_state = ListState::default();
            list_state.select(Some(selected));
            f.render_stateful_widget(menu, chunks[2], &mut list_state);

            let hint = Paragraph::new(Line::from(Span::styled(
                "  ↑/↓ navigate · Enter select · Q quit",
                Style::default().fg(Color::DarkGray),
            )));
            f.render_widget(hint, chunks[3]);
        })?;

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => {
                        break MenuAction::Quit;
                    }
                    KeyCode::Up | KeyCode::Char('k') => {
                        selected = selected.saturating_sub(1);
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        selected = (selected + 1).min(items.len() - 1);
                    }
                    KeyCode::Enter | KeyCode::Char(' ') => {
                        match selected {
                            0 => { break MenuAction::Play; }
                            1 => {
                                disable_raw_mode()?;
                                execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
                                run_rulebook()?;
                                enable_raw_mode()?;
                                execute!(io::stdout(), EnterAlternateScreen)?;
                                // Rebuild terminal after returning from sub-screen.
                                let backend = CrosstermBackend::new(io::stdout());
                                terminal = Terminal::new(backend)?;
                            }
                            2 => {
                                disable_raw_mode()?;
                                execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
                                run_card_ref()?;
                                enable_raw_mode()?;
                                execute!(io::stdout(), EnterAlternateScreen)?;
                                let backend = CrosstermBackend::new(io::stdout());
                                terminal = Terminal::new(backend)?;
                            }
                            3 => { break MenuAction::Quit; }
                            _ => {}
                        }
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

fn run_rulebook() -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let lines: Vec<String> = RULEBOOK.lines().map(|l| l.to_string()).collect();
    let mut scroll: usize = 0;

    loop {
        terminal.draw(|f| {
            let area = f.area();
            f.render_widget(Clear, area);
            let visible_height = area.height.saturating_sub(2) as usize;
            let visible: Vec<Line> = lines.iter().skip(scroll).take(visible_height)
                .map(|l| Line::from(Span::raw(l.clone())))
                .collect();
            let para = Paragraph::new(visible)
                .block(Block::default().borders(Borders::ALL).title(" Rulebook - ↑/↓ PgUp/PgDn · Q to return "))
                .wrap(Wrap { trim: false });
            f.render_widget(para, area);
        })?;

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                let page = terminal.size()?.height.saturating_sub(4) as usize;
                match key.code {
                    KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => break,
                    KeyCode::Up | KeyCode::Char('k') => { scroll = scroll.saturating_sub(1); }
                    KeyCode::Down | KeyCode::Char('j') => { scroll = (scroll + 1).min(lines.len().saturating_sub(1)); }
                    KeyCode::PageUp => { scroll = scroll.saturating_sub(page); }
                    KeyCode::PageDown => { scroll = (scroll + page).min(lines.len().saturating_sub(1)); }
                    KeyCode::Home => { scroll = 0; }
                    KeyCode::End => { scroll = lines.len().saturating_sub(1); }
                    _ => {}
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

fn run_card_ref() -> io::Result<()> {
    use novacana_engine::card_db;

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let cards = card_db::all_cards();
    let mut scroll: usize = 0;

    loop {
        terminal.draw(|f| {
            let area = f.area();
            f.render_widget(Clear, area);

            let visible_height = area.height.saturating_sub(2) as usize;

            // Build lines: each card = name + type + effect (3 lines)
            let all_lines: Vec<Line> = cards.iter().flat_map(|c| {
                vec![
                    Line::from(vec![
                        Span::styled(c.name.clone(), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                        Span::styled(format!("  [{}]", c.card_type), Style::default().fg(Color::Yellow)),
                    ]),
                    Line::from(Span::styled(format!("  {}", c.effect), Style::default().fg(Color::White))),
                    Line::from(Span::raw("")),
                ]
            }).collect();

            let visible: Vec<Line> = all_lines.iter().skip(scroll).take(visible_height).cloned().collect();
            let para = Paragraph::new(visible)
                .block(Block::default().borders(Borders::ALL).title(" Card Reference - ↑/↓ PgUp/PgDn · Q to return "))
                .wrap(Wrap { trim: false });
            f.render_widget(para, area);
        })?;

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                let page = terminal.size()?.height.saturating_sub(4) as usize;
                match key.code {
                    KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => break,
                    KeyCode::Up | KeyCode::Char('k') => { scroll = scroll.saturating_sub(1); }
                    KeyCode::Down | KeyCode::Char('j') => { scroll += 1; }
                    KeyCode::PageUp => { scroll = scroll.saturating_sub(page); }
                    KeyCode::PageDown => { scroll += page; }
                    KeyCode::Home => { scroll = 0; }
                    _ => {}
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}
