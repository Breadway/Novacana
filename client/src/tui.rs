use std::collections::HashSet;
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError};
use std::time::Duration;

use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap},
    Frame, Terminal,
};

use novacana_engine::game::{GameEvent, GameResult, GameSnapshot, PlayerSnapshot, WinCondition};
use novacana_engine::human::{HandCardDisplay, HumanPrompt, HumanResponse, PromptKind};

// ── Palette ───────────────────────────────────────────────────────────────────

const PALETTE: [Color; 4] = [Color::Cyan, Color::Green, Color::Yellow, Color::Magenta];

fn player_color(idx: usize) -> Color {
    PALETTE[idx % PALETTE.len()]
}

fn player_color_by_name(name: &str, players: &[PlayerSnapshot]) -> Color {
    let idx = players.iter().position(|p| p.name == name).unwrap_or(0);
    player_color(idx)
}

// ── State ─────────────────────────────────────────────────────────────────────

struct ActiveInput {
    prompt: HumanPrompt,
    selected: usize,
}

struct TuiState {
    snapshot: Option<GameSnapshot>,
    log: Vec<String>,
    thinking: Option<String>,
    thought_player: String,
    stream_buf: String,
    thought_lines: Vec<String>,
    in_thought: bool,
    game_over: Option<GameResult>,
    input: Option<ActiveInput>,
    /// Last known hand for the human player; persists between prompts.
    last_hand: Vec<HandCardDisplay>,
    /// Offset from the bottom of the log. 0 = live (newest). Increases = older.
    log_scroll: usize,
    /// True while the user hasn't manually scrolled (auto-follow new lines).
    log_pinned: bool,
    show_help: bool,
}

impl TuiState {
    fn new() -> Self {
        Self {
            snapshot: None,
            log: Vec::new(),
            thinking: None,
            thought_player: String::new(),
            stream_buf: String::new(),
            thought_lines: Vec::new(),
            in_thought: false,
            game_over: None,
            input: None,
            last_hand: Vec::new(),
            log_scroll: 0,
            log_pinned: true,
            show_help: false,
        }
    }

    fn apply(&mut self, event: GameEvent) {
        match event {
            GameEvent::Log(msg) => {
                for line in msg.split('\n') {
                    let t = line.trim();
                    if t.is_empty() { continue; }

                    if !t.starts_with("[/") && t.ends_with(" reasoning]") && t.starts_with('[') {
                        self.in_thought = true;
                        self.thought_lines.clear();
                        let inner = &t[1..t.len() - 1];
                        self.thought_player = inner
                            .strip_suffix(" reasoning")
                            .unwrap_or(inner)
                            .to_string();
                    } else if t.starts_with("[/") && t.ends_with(" reasoning]") {
                        self.in_thought = false;
                    } else if self.in_thought {
                        self.thought_lines.push(t.to_string());
                    } else {
                        self.log.push(t.to_string());
                        if self.log.len() > 800 {
                            self.log.drain(0..100);
                        }
                    }
                }
                self.thinking = None;
            }
            GameEvent::ThinkingChunk { player_name, text } => {
                if self.thought_player != player_name {
                    self.thought_player = player_name;
                    self.stream_buf.clear();
                    self.thought_lines.clear();
                }
                self.stream_buf.push_str(&text);
                self.thought_lines = self.stream_buf
                    .lines()
                    .filter(|l| !l.trim().is_empty())
                    .map(|l| l.to_string())
                    .collect();
                if self.thought_lines.len() > 200 {
                    let drop = self.thought_lines.len() - 200;
                    self.thought_lines.drain(0..drop);
                }
            }
            GameEvent::Snapshot(snap) => {
                self.snapshot = Some(snap);
                self.thinking = None;
            }
            GameEvent::Thinking { player_name } => {
                if self.thought_player != player_name {
                    self.thought_player = player_name.clone();
                    self.stream_buf.clear();
                    self.thought_lines.clear();
                }
                self.thinking = Some(player_name);
            }
            GameEvent::GameOver(result) => {
                self.game_over = Some(result);
            }
        }
    }
}

// ── Entry points ──────────────────────────────────────────────────────────────

/// Pure observer — no human input; Q/Esc to quit.
pub fn run_observer(rx: Receiver<GameEvent>) -> std::io::Result<()> {
    run_tui(rx, None)
}

/// Interactive — includes a HumanPlayer communicating through channels.
///
/// SERVER PATH: `prompt_rx` becomes a WebSocket receiver and `response_tx`
/// becomes a WebSocket sender. No changes to the game engine are needed.
pub fn run_interactive(
    game_rx: Receiver<GameEvent>,
    prompt_rx: Receiver<HumanPrompt>,
    response_tx: SyncSender<HumanResponse>,
) -> std::io::Result<()> {
    run_tui(game_rx, Some((prompt_rx, response_tx)))
}

fn run_tui(
    game_rx: Receiver<GameEvent>,
    human: Option<(Receiver<HumanPrompt>, SyncSender<HumanResponse>)>,
) -> std::io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut state = TuiState::new();
    let is_interactive = human.is_some();

    loop {
        // Drain game events
        loop {
            match game_rx.try_recv() {
                Ok(ev) => state.apply(ev),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }

        // Pick up a new human prompt (only when not already handling one)
        if let Some((ref prompt_rx, _)) = human {
            if state.input.is_none() {
                if let Ok(prompt) = prompt_rx.try_recv() {
                    state.last_hand = prompt.hand_cards.clone();
                    state.thought_lines.clear();
                    state.thinking = None;
                    state.input = Some(ActiveInput { prompt, selected: 0 });
                }
            }
        }

        terminal.draw(|f| render(f, &state, is_interactive))?;

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                // Help toggle works in all modes.
                if key.code == KeyCode::Char('?') {
                    state.show_help = !state.show_help;
                } else if let Some(input) = state.input.as_mut() {
                    // Input mode: route keys to the selector
                    let response_tx = human.as_ref().map(|(_, tx)| tx);
                    if let Some(tx) = response_tx {
                        match handle_input_key(key.code, input, tx) {
                            InputResult::Done => { state.input = None; }
                            InputResult::Quit => break,
                            InputResult::Pending => {}
                        }
                    }
                } else {
                    // Observer / idle mode
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => {
                            // Send a default so the game thread can exit
                            if let Some((_, ref tx)) = human {
                                let _ = tx.send(HumanResponse::Pass);
                            }
                            break;
                        }
                        // ↑/k scrolls toward older entries (increases offset from bottom)
                        KeyCode::Char('k') | KeyCode::Up => {
                            state.log_scroll = state.log_scroll.saturating_add(1);
                            state.log_pinned = false;
                        }
                        // ↓/j scrolls toward newer entries (decreases offset)
                        KeyCode::Char('j') | KeyCode::Down => {
                            if state.log_scroll > 0 {
                                state.log_scroll -= 1;
                            }
                            if state.log_scroll == 0 {
                                state.log_pinned = true;
                            }
                        }
                        // G/End snaps back to live view
                        KeyCode::Char('G') | KeyCode::Char('g') | KeyCode::End => {
                            state.log_scroll = 0;
                            state.log_pinned = true;
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

// ── Input key handler ─────────────────────────────────────────────────────────

#[derive(PartialEq)]
enum InputResult {
    /// Keep waiting.
    Pending,
    /// Selection confirmed — close the input panel.
    Done,
    /// User wants to quit entirely.
    Quit,
}

fn handle_input_key(
    code: KeyCode,
    input: &mut ActiveInput,
    tx: &SyncSender<HumanResponse>,
) -> InputResult {
    let options_len = input.prompt.kind.options().len();
    let can_pass = input.prompt.kind.can_pass();
    let total = options_len + if can_pass { 1 } else { 0 };

    match code {
        KeyCode::Up => {
            if input.selected == 0 {
                input.selected = total.saturating_sub(1);
            } else {
                input.selected -= 1;
            }
        }
        KeyCode::Down => {
            if total > 0 {
                input.selected = (input.selected + 1) % total;
            }
        }
        KeyCode::Char(c @ '1'..='9') => {
            let n = (c as u8 - b'0') as usize;
            if n >= 1 && n <= options_len {
                let _ = tx.send(HumanResponse::Pick(n - 1));
                return InputResult::Done;
            }
            if can_pass && n == options_len + 1 {
                let _ = tx.send(HumanResponse::Pass);
                return InputResult::Done;
            }
        }
        KeyCode::Char('p') | KeyCode::Char('P') if can_pass => {
            let _ = tx.send(HumanResponse::Pass);
            return InputResult::Done;
        }
        KeyCode::Enter => {
            if options_len == 0 {
                let _ = tx.send(HumanResponse::Pass);
            } else if can_pass && input.selected == options_len {
                let _ = tx.send(HumanResponse::Pass);
            } else {
                let _ = tx.send(HumanResponse::Pick(input.selected.min(options_len - 1)));
            }
            return InputResult::Done;
        }
        KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => {
            // Send a default so the game thread can unblock, then quit
            let _ = tx.send(HumanResponse::Pass);
            return InputResult::Quit;
        }
        _ => {}
    }
    InputResult::Pending
}

// ── Render ────────────────────────────────────────────────────────────────────

fn render(f: &mut Frame, state: &TuiState, is_interactive: bool) {
    let area = f.area();

    let chain_links = state.snapshot.as_ref()
        .map(|s| s.chain_description.len())
        .unwrap_or(0);
    let chain_height: u16 = if chain_links > 0 {
        2 + chain_links.min(10) as u16
    } else {
        0
    };

    // Hand panel: always shown in interactive mode, sized to fit hand
    let hand_height: u16 = if is_interactive {
        (2 + state.last_hand.len().max(1)).min(9) as u16
    } else {
        0
    };

    // Input panel takes priority over thought panel
    let (input_height, thought_height) = if let Some(input) = &state.input {
        let opts = input.prompt.kind.options().len();
        let chain_supplement = match &input.prompt.kind {
            PromptKind::RespondToChain { chain, .. } => chain.len().min(4) + 1,
            _ => 0,
        };
        let h = (2 + 1 + chain_supplement + opts + 2).min(24) as u16;
        (h, 0u16)
    } else if !state.thought_lines.is_empty() || state.thinking.is_some() {
        let h = (2 + state.thought_lines.len().min(12)).max(3) as u16;
        (0u16, h)
    } else {
        (0u16, 0u16)
    };

    let log_height: u16 = if is_interactive { 10 } else { 14 };

    let vsplit = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Fill(1),
            Constraint::Length(hand_height),
            Constraint::Length(chain_height),
            Constraint::Length(input_height),
            Constraint::Length(thought_height),
            Constraint::Length(log_height),
        ])
        .split(area);

    // Playable card names: cards that appear in at least one ChooseAction option.
    let playable_names: HashSet<String> = if let Some(input) = &state.input {
        if let PromptKind::ChooseAction { options, .. } = &input.prompt.kind {
            state.last_hand.iter()
                .filter(|card| options.iter().any(|opt| opt.contains(card.name.as_str())))
                .map(|card| card.name.clone())
                .collect()
        } else {
            HashSet::new()
        }
    } else {
        HashSet::new()
    };

    render_header(f, vsplit[0], state, is_interactive);
    render_players(f, vsplit[1], state);
    if hand_height > 0 {
        render_hand_panel(f, vsplit[2], &state.last_hand, &playable_names);
    }
    if chain_height > 0 {
        render_chain(f, vsplit[3], state);
    }
    if input_height > 0 {
        render_input_panel(f, vsplit[4], state);
    }
    if thought_height > 0 {
        render_thought(f, vsplit[5], state);
    }
    render_log(f, vsplit[6], state);

    // Help overlay drawn last so it appears on top of everything
    if state.show_help {
        render_help_overlay(f, area);
    }
}

fn render_header(f: &mut Frame, area: Rect, state: &TuiState, is_interactive: bool) {
    // Game-over replaces the header with a full-width winner banner.
    if let Some(result) = &state.game_over {
        let cond = match result.condition {
            WinCondition::AstronomicalDomination => "Astronomical Domination",
            WinCondition::CosmicCataclysm => "Cosmic Cataclysm",
        };
        let players = state.snapshot.as_ref().map(|s| s.players.as_slice()).unwrap_or(&[]);
        let winner_color = player_color_by_name(&result.winner_name, players);
        let content = vec![
            Span::styled("  ★ ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(
                result.winner_name.clone(),
                Style::default().fg(winner_color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    " wins by {} in {} turns  ·  press Q to exit",
                    cond, result.turns_elapsed
                ),
                Style::default().fg(Color::Cyan),
            ),
        ];
        let para = Paragraph::new(Line::from(content))
            .block(Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
                .title(Span::styled(
                    " NOVACANA — GAME OVER ",
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                )));
        f.render_widget(para, area);
        return;
    }

    let content = if let Some(snap) = &state.snapshot {
        let active_player = snap.players.iter().find(|p| p.is_active);
        let active_name = active_player.map(|p| p.name.as_str()).unwrap_or("?");
        let active_idx = snap.players.iter().position(|p| p.is_active).unwrap_or(0);
        let active_color = player_color(active_idx);

        let dmv = snap.dark_matter_void
            .map(|c| format!("  ▲ class {} void", c))
            .unwrap_or_default();

        // Derive the current turn phase from the pending prompt kind.
        let status = if let Some(input) = &state.input {
            let phase = match &input.prompt.kind {
                PromptKind::ChooseAction { .. }   => "Action Phase",
                PromptKind::RespondToChain { .. } => "Chain — respond?",
                PromptKind::ChooseDiscard { .. }  => "End Phase — discard",
                PromptKind::Decide { .. }
                | PromptKind::PickTarget { .. }   => "Effect resolving",
            };
            Span::styled(
                format!("  ← YOUR TURN  [{}]", phase),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            )
        } else if let Some(t) = &state.thinking {
            Span::styled(format!("  ⟳ {} thinking…", t), Style::default().fg(Color::DarkGray))
        } else {
            Span::raw("")
        };

        let left = format!("Turn {}  Deck: {}{}  ", snap.turn, snap.deck_size, dmv);

        vec![
            Span::styled(left, Style::default().fg(Color::White)),
            Span::styled("▶ ", Style::default().fg(active_color).add_modifier(Modifier::BOLD)),
            Span::styled(active_name.to_string(), Style::default().fg(active_color).add_modifier(Modifier::BOLD)),
            status,
        ]
    } else {
        vec![Span::styled("Waiting for game to start…", Style::default().fg(Color::DarkGray))]
    };

    let title_label = if is_interactive {
        " NOVACANA — INTERACTIVE  [? help] "
    } else {
        " NOVACANA  [? help] "
    };
    let para = Paragraph::new(Line::from(content))
        .block(Block::default().borders(Borders::ALL)
            .title(Span::styled(
                title_label,
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            )));
    f.render_widget(para, area);
}

fn render_players(f: &mut Frame, area: Rect, state: &TuiState) {
    let snap = match &state.snapshot {
        Some(s) => s,
        None => {
            let p = Paragraph::new("Waiting for first snapshot…")
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(p, area);
            return;
        }
    };

    let players = &snap.players;
    let count = players.len();

    match count {
        0 => {}
        1 | 2 => {
            let cols: Vec<Constraint> = (0..count).map(|_| Constraint::Fill(1)).collect();
            let col_areas = Layout::default()
                .direction(Direction::Horizontal)
                .constraints(cols)
                .split(area);
            for (i, p) in players.iter().enumerate() {
                render_player_panel(f, col_areas[i], p, i);
            }
        }
        3 => {
            let rows = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Fill(1), Constraint::Fill(1)])
                .split(area);
            let top = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Fill(1), Constraint::Fill(1)])
                .split(rows[0]);
            render_player_panel(f, top[0], &players[0], 0);
            render_player_panel(f, top[1], &players[1], 1);
            render_player_panel(f, rows[1], &players[2], 2);
        }
        _ => {
            let rows = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Fill(1), Constraint::Fill(1)])
                .split(area);
            for row in 0..2 {
                let cols = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Fill(1), Constraint::Fill(1)])
                    .split(rows[row]);
                for col in 0..2 {
                    let idx = row * 2 + col;
                    if idx < players.len() {
                        render_player_panel(f, cols[col], &players[idx], idx);
                    }
                }
            }
        }
    }
}

fn render_player_panel(f: &mut Frame, area: Rect, p: &PlayerSnapshot, player_idx: usize) {
    let color = player_color(player_idx);
    let complete_count = p.star_chart.complete_systems().len();

    let mut status_parts: Vec<&str> = Vec::new();
    if p.is_active { status_parts.push("ACTIVE"); }
    if p.skip_next_action { status_parts.push("SKIP"); }
    if p.astral_projected_turns > 0 { status_parts.push("REVEALED"); }
    let status_str = if status_parts.is_empty() {
        String::new()
    } else {
        format!(" [{}]", status_parts.join(" · "))
    };

    let mut lines: Vec<Line> = vec![
        Line::from(vec![
            Span::styled(
                format!("Hand: {}   ", p.hand_size),
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("Systems: {}/3", complete_count),
                Style::default().fg(if complete_count >= 3 { Color::Cyan } else { Color::White }),
            ),
            Span::styled(status_str, Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        ]),
    ];

    if p.star_chart.systems.is_empty() {
        lines.push(Line::from(Span::styled(
            "  (no star chart yet)",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        for sys in &p.star_chart.systems {
            let filled = sys.discoveries.len();
            let complete = filled >= sys.class as usize;
            let eclipse = sys.eclipse_id.is_some();

            let check = if complete { "✓" } else { "·" };
            let sys_color = if complete { Color::Green } else { color };

            let bar = progress_bar(filled, sys.class as usize);
            let eclipse_tag = if eclipse { " ⊙" } else { "" };

            lines.push(Line::from(vec![
                Span::styled(format!("  {} ", check), Style::default().fg(sys_color)),
                Span::styled(
                    sys.star_name.to_string(),
                    Style::default().fg(sys_color).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" cl.{}  {}{}", sys.class, bar, eclipse_tag),
                    Style::default().fg(Color::DarkGray),
                ),
            ]));

            if !sys.discoveries.is_empty() {
                let mut disc_spans: Vec<Span> = vec![Span::raw("      ")];
                for (i, d) in sys.discoveries.iter().enumerate() {
                    if i > 0 {
                        disc_spans.push(Span::styled("  ·  ", Style::default().fg(Color::DarkGray)));
                    }
                    disc_spans.push(Span::styled(
                        d.name.to_string(),
                        Style::default().fg(Color::DarkGray),
                    ));
                    // Planet of Life marker: ⊕ in cyan after the discovery name
                    if d.planet_of_life_id.is_some() {
                        disc_spans.push(Span::styled(" ⊕", Style::default().fg(Color::Cyan)));
                    }
                }
                lines.push(Line::from(disc_spans));
            }
        }
    }

    let border_style = if p.is_active {
        Style::default().fg(color).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(color)
    };

    let title = format!(" {} ", p.name);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(Span::styled(title, border_style));

    let para = Paragraph::new(lines).block(block);
    f.render_widget(para, area);
}

fn progress_bar(filled: usize, total: usize) -> String {
    let width = 6usize;
    let fill = if total == 0 { width } else { (filled * width / total).min(width) };
    let empty = width - fill;
    format!("[{}{}]", "█".repeat(fill), "░".repeat(empty))
}

fn render_chain(f: &mut Frame, area: Rect, state: &TuiState) {
    let players = state.snapshot.as_ref().map(|s| s.players.as_slice()).unwrap_or(&[]);
    let links = state.snapshot.as_ref()
        .map(|s| &s.chain_description[..])
        .unwrap_or(&[]);

    let visible = area.height.saturating_sub(2) as usize;
    let start = links.len().saturating_sub(visible);

    let items: Vec<ListItem> = links[start..].iter().enumerate().map(|(rel_i, l)| {
        let abs_i = start + rel_i;
        let line_color = extract_player_name(l)
            .map(|name| player_color_by_name(name, players))
            .unwrap_or(Color::Magenta);

        let is_last = abs_i == links.len() - 1;
        let prefix = if is_last { "▶ " } else { "  " };

        ListItem::new(Line::from(vec![
            Span::styled(prefix, Style::default().fg(Color::White)),
            Span::styled(l.as_str(), Style::default().fg(line_color)),
        ]))
    }).collect();

    let title = format!(" CHAIN ({}) — resolves last-first ", links.len());
    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Magenta))
            .title(Span::styled(title, Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)))
    );
    f.render_widget(list, area);
}

fn render_hand_panel(f: &mut Frame, area: Rect, hand: &[HandCardDisplay], playable: &HashSet<String>) {
    // Name column is 24 chars, type tag column is 13 chars, rest goes to effect.
    const NAME_W: usize = 24;
    const TAG_W: usize  = 13;
    const SEP: &str     = " — ";
    const LEAD: usize   = 2; // leading marker width

    let inner_w = area.width.saturating_sub(2) as usize; // subtract borders

    let items: Vec<ListItem> = if hand.is_empty() {
        vec![ListItem::new(Line::from(Span::styled(
            "  Waiting for your first turn…",
            Style::default().fg(Color::DarkGray),
        )))]
    } else {
        hand.iter().map(|card| {
            let is_playable = playable.contains(&card.name);
            let type_color = card_type_color(&card.card_type);
            let name  = fit_str(&card.name, NAME_W);
            let tag   = fit_str(&format!("[{}]", card.card_type), TAG_W);
            let eff_w = inner_w.saturating_sub(LEAD + NAME_W + TAG_W + SEP.len());
            let eff   = fit_str(&card.effect, eff_w);

            let (marker, name_style) = if is_playable {
                (
                    Span::styled("► ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                    Style::default().fg(type_color).add_modifier(Modifier::BOLD),
                )
            } else {
                (
                    Span::raw("  "),
                    Style::default().fg(Color::White),
                )
            };

            ListItem::new(Line::from(vec![
                marker,
                Span::styled(format!("{:<NAME_W$}", name), name_style),
                Span::styled(format!("{:<TAG_W$}", tag), Style::default().fg(type_color)),
                Span::styled(SEP, Style::default().fg(Color::DarkGray)),
                Span::styled(eff, Style::default().fg(Color::DarkGray)),
            ]))
        }).collect()
    };

    let title = if hand.is_empty() {
        " YOUR HAND ".to_string()
    } else if !playable.is_empty() {
        format!(" YOUR HAND ({} cards — ► = has play) ", hand.len())
    } else {
        format!(" YOUR HAND ({} cards) ", hand.len())
    };

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Green))
            .title(Span::styled(
                title,
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            ))
    );
    f.render_widget(list, area);
}

fn card_type_color(card_type: &str) -> Color {
    match card_type {
        "Star"      => Color::Yellow,
        "Discovery" => Color::Green,
        "Flare"     => Color::Magenta,
        "Omen"      => Color::Cyan,
        "Fracture"  => Color::Red,
        _           => Color::White,
    }
}

/// Truncate `s` to `max` characters, appending `…` if truncated.
fn fit_str(s: &str, max: usize) -> String {
    if max == 0 { return String::new(); }
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        chars.into_iter().collect()
    } else {
        let truncated: String = chars.into_iter().take(max.saturating_sub(1)).collect();
        format!("{}…", truncated)
    }
}

fn render_input_panel(f: &mut Frame, area: Rect, state: &TuiState) {
    let Some(input) = &state.input else { return };
    let kind = &input.prompt.kind;
    let options = kind.options();
    let options_len = options.len();
    let can_pass = kind.can_pass();

    let mut lines: Vec<Line> = Vec::new();

    // Show chain context for chain responses
    if let PromptKind::RespondToChain { chain, .. } = kind {
        lines.push(Line::from(Span::styled(
            "  Open chain (resolves last-first):",
            Style::default().fg(Color::DarkGray),
        )));
        for link in chain.iter().take(4) {
            lines.push(Line::from(Span::styled(
                format!("   {}", link),
                Style::default().fg(Color::Magenta),
            )));
        }
        if chain.len() > 4 {
            lines.push(Line::from(Span::styled(
                format!("   … and {} more link(s)", chain.len() - 4),
                Style::default().fg(Color::DarkGray),
            )));
        }
        lines.push(Line::default());
    }

    // Numbered options — colored by card type for ChooseAction, white for others
    for (i, opt) in options.iter().enumerate() {
        let is_sel = i == input.selected;
        let type_color = if let PromptKind::ChooseAction { action_types, .. } = kind {
            action_types.get(i)
                .and_then(|t| t.as_deref())
                .map(card_type_color)
                .unwrap_or(Color::DarkGray)
        } else {
            Color::White
        };

        let label = format!("  [{}] {}", i + 1, opt);
        let style = if is_sel {
            let bg = if matches!(type_color, Color::White | Color::DarkGray) {
                Color::Cyan
            } else {
                type_color
            };
            Style::default().fg(Color::Black).bg(bg).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(type_color)
        };
        lines.push(Line::from(Span::styled(label, style)));
    }

    // Pass option
    if can_pass {
        let is_pass = input.selected == options_len;
        let style = if is_pass {
            Style::default().fg(Color::Black).bg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Yellow)
        };
        lines.push(Line::from(Span::styled(
            "  [P] Pass — don't respond to the chain",
            style,
        )));
    }

    // Hint bar
    lines.push(Line::default());
    let hints = if can_pass {
        "  1-9 · ↑↓ navigate · Enter confirm · P pass · Q quit · ? help"
    } else {
        "  1-9 · ↑↓ navigate · Enter confirm · Q quit · ? help"
    };
    lines.push(Line::from(Span::styled(hints, Style::default().fg(Color::DarkGray))));

    let title = format!(" {} ", kind.title());
    let para = Paragraph::new(lines)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .title(Span::styled(
                title,
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            )));
    f.render_widget(para, area);
}

fn render_thought(f: &mut Frame, area: Rect, state: &TuiState) {
    let players = state.snapshot.as_ref().map(|s| s.players.as_slice()).unwrap_or(&[]);
    let thinker_color = player_color_by_name(&state.thought_player, players);

    let visible = area.height.saturating_sub(2) as usize;
    let lines = &state.thought_lines;
    let start = lines.len().saturating_sub(visible);

    let items: Vec<Line> = lines[start..].iter()
        .map(|l| {
            let display = l
                .trim_start_matches("<think>")
                .trim_end_matches("</think>")
                .trim();
            Line::from(Span::styled(display.to_string(), Style::default().fg(Color::White)))
        })
        .collect();

    let indicator = if state.thinking.is_some() { " ⟳" } else { "" };
    let title = format!(" {} thinking{} ", state.thought_player, indicator);

    let para = Paragraph::new(items)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(thinker_color))
            .title(Span::styled(
                title,
                Style::default().fg(thinker_color).add_modifier(Modifier::BOLD),
            )))
        .wrap(Wrap { trim: true });
    f.render_widget(para, area);
}

fn render_log(f: &mut Frame, area: Rect, state: &TuiState) {
    let players = state.snapshot.as_ref().map(|s| s.players.as_slice()).unwrap_or(&[]);
    let visible = area.height.saturating_sub(2) as usize;
    let log = &state.log;

    // Clamp scroll so we can't scroll past the earliest entries.
    let max_scroll = log.len().saturating_sub(visible);
    let scroll = state.log_scroll.min(max_scroll);
    let end = log.len().saturating_sub(scroll);
    let start = end.saturating_sub(visible);

    let mut lines: Vec<Line> = log[start..end].iter()
        .map(|l| colorize_log_line(l, players))
        .collect();

    let title = if scroll > 0 || !state.log_pinned {
        " EVENT LOG  [SCROLLED — G/End to resume] "
    } else if let Some(result) = &state.game_over {
        let cond = match result.condition {
            WinCondition::AstronomicalDomination => "Astronomical Domination",
            WinCondition::CosmicCataclysm => "Cosmic Cataclysm",
        };
        lines.push(Line::from(Span::styled(
            format!(
                "  ★ {} wins by {} in {} turns!  Press Q to exit.",
                result.winner_name, cond, result.turns_elapsed
            ),
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )));
        " EVENT LOG — GAME OVER "
    } else {
        " EVENT LOG  [Q quit · ↑/k scroll · ? help] "
    };

    let para = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(title));
    f.render_widget(para, area);
}

// ── Help overlay ──────────────────────────────────────────────────────────────

fn centered_rect(percent_w: u16, percent_h: u16, area: Rect) -> Rect {
    let w = (area.width * percent_w / 100).max(40).min(area.width);
    let h = (area.height * percent_h / 100).max(20).min(area.height);
    let x = area.x + area.width.saturating_sub(w) / 2;
    let y = area.y + area.height.saturating_sub(h) / 2;
    Rect::new(x, y, w, h)
}

fn render_help_overlay(f: &mut Frame, area: Rect) {
    let popup = centered_rect(54, 72, area);

    let lines = vec![
        Line::default(),
        Line::from(Span::styled("  INPUT MODE", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(vec![
            Span::styled("  1-9        ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("select option by number", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  ↑ / ↓      ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("navigate options", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  Enter      ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("confirm selection", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  P          ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled("pass (chain responses only)", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  Q / Esc    ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("quit game", Style::default().fg(Color::White)),
        ]),
        Line::default(),
        Line::from(Span::styled("  LOG SCROLL  (idle / observer mode)", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(vec![
            Span::styled("  ↑ / k      ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("older entries", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  ↓ / j      ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("newer entries", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  G / End    ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("resume live log", Style::default().fg(Color::White)),
        ]),
        Line::default(),
        Line::from(Span::styled("  LEGEND", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(vec![
            Span::styled("  ►  ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::styled("card has at least one legal play this turn", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  ⊕  ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled("Planet of Life omen attached", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  ⊙  ", Style::default().fg(Color::DarkGray)),
            Span::styled("Solar Eclipse protection active", Style::default().fg(Color::White)),
        ]),
        Line::default(),
        Line::from(Span::styled("  ? to close", Style::default().fg(Color::DarkGray))),
    ];

    let para = Paragraph::new(lines)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::White).add_modifier(Modifier::BOLD))
            .title(Span::styled(
                " KEYBINDINGS ",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            )));

    f.render_widget(Clear, popup);
    f.render_widget(para, popup);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn extract_player_name(s: &str) -> Option<&str> {
    let open = s.find('[')?;
    let close = s[open..].find(']')?;
    Some(&s[open + 1..open + close])
}

fn colorize_log_line<'a>(line: &'a str, players: &[PlayerSnapshot]) -> Line<'a> {
    if line.starts_with("══") || line.starts_with("==") {
        return Line::from(Span::styled(
            line,
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ));
    }

    if line.contains("wins") || line.contains("Cataclysm") || line.contains("Domination") {
        return Line::from(Span::styled(
            line,
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ));
    }

    if line.contains("Chain") || line.contains("chain") || line.contains("resolves") || line.contains("NEGATED") {
        return Line::from(Span::styled(line, Style::default().fg(Color::Magenta)));
    }

    let trimmed = line.trim_start_matches(' ');
    for (i, p) in players.iter().enumerate() {
        if trimmed.starts_with(p.name.as_str()) {
            let rest = &trimmed[p.name.len()..];
            let is_routine = rest.contains("draws") || rest.contains("passes")
                || rest.contains("skips") || rest.contains("discards");
            let fg = if is_routine { Color::DarkGray } else { player_color(i) };
            let leading = &line[..line.len() - trimmed.len()];
            return Line::from(vec![
                Span::raw(leading.to_string()),
                Span::styled(p.name.clone(), Style::default().fg(player_color(i)).add_modifier(Modifier::BOLD)),
                Span::styled(rest.to_string(), Style::default().fg(fg)),
            ]);
        }
    }

    let color = if line.contains("Fracture") || line.contains("Flare") || line.contains("responds") {
        Color::Magenta
    } else if line.contains("complete") || line.contains("★") || line.contains("✓") {
        Color::Green
    } else {
        Color::White
    };
    Line::from(Span::styled(line, Style::default().fg(color)))
}
