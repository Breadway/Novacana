use std::io::{self, BufRead, Write};
use std::sync::mpsc;

use novacana_engine::{
    card::{Card, CardId},
    chain::ChainView,
    cpu::{AdvancedCpuPlayer, CpuPlayer, EasyCpuPlayer},
    deck::Deck,
    game::{GameEngine, GameResult, WinCondition},
    human::HumanPlayer,
    ollama::OllamaPlayer,
    player::{Action, ChainAction, ChoiceRequest, GameView, OmenAttachment, Player, ProtostarChoice, TargetOption},
};

mod lobby;
mod sim;
mod tui;

// ── Random bot ────────────────────────────────────────────────────────────────

pub struct RandomBot {
    name: String,
}

impl RandomBot {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl Player for RandomBot {
    fn name(&self) -> &str {
        &self.name
    }

    fn choose_action(&self, _view: &GameView, legal_actions: &[Action]) -> usize {
        rand::random::<usize>() % legal_actions.len()
    }

    fn respond_to_chain(
        &self,
        _view: &GameView,
        _chain: &ChainView,
        legal_responses: &[ChainAction],
    ) -> Option<usize> {
        if legal_responses.is_empty() {
            return None;
        }
        if rand::random::<u8>() % 10 < 3 {
            Some(rand::random::<usize>() % legal_responses.len())
        } else {
            None
        }
    }

    fn decide(&self, _view: &GameView, request: &ChoiceRequest) -> usize {
        rand::random::<usize>() % request.options.len()
    }

    fn pick_targets(
        &self,
        _view: &GameView,
        _prompt: &str,
        options: &[TargetOption],
        count: usize,
    ) -> Vec<usize> {
        let n = count.min(options.len());
        (0..n).collect()
    }

    fn choose_discards(
        &self,
        _view: &GameView,
        hand: &[Card],
        must_discard: usize,
    ) -> Vec<CardId> {
        hand.iter().rev().take(must_discard).map(|c| c.id).collect()
    }
}

// ── CLI player ────────────────────────────────────────────────────────────────

pub struct CliPlayer {
    name: String,
}

impl CliPlayer {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    fn prompt_pick(&self, prompt: &str, options: &[String]) -> usize {
        loop {
            println!("\n{}", prompt);
            for (i, opt) in options.iter().enumerate() {
                println!("  [{}] {}", i + 1, opt);
            }
            print!("Enter choice (1-{}): ", options.len());
            io::stdout().flush().unwrap();

            let stdin = io::stdin();
            if let Some(line) = stdin.lock().lines().next() {
                if let Ok(input) = line {
                    if let Ok(n) = input.trim().parse::<usize>() {
                        if n >= 1 && n <= options.len() {
                            return n - 1;
                        }
                    }
                }
            }
            println!("  (invalid - try again)");
        }
    }
}

impl Player for CliPlayer {
    fn name(&self) -> &str {
        &self.name
    }

    fn choose_action(&self, view: &GameView, legal_actions: &[Action]) -> usize {
        println!("{}", view);
        let opts: Vec<String> = legal_actions.iter().enumerate()
            .map(|(i, a)| describe_action(a, view, i))
            .collect();
        self.prompt_pick("── YOUR TURN: choose an action ──", &opts)
    }

    fn respond_to_chain(
        &self,
        view: &GameView,
        chain: &ChainView,
        legal_responses: &[ChainAction],
    ) -> Option<usize> {
        println!("\n── CHAIN IS OPEN ──");
        println!("{}", chain);
        if legal_responses.is_empty() {
            println!("  (no legal responses - you pass)");
            return None;
        }
        println!("  Your hand has {} card(s).", view.my_hand.len());
        let mut opts: Vec<String> = legal_responses
            .iter()
            .filter_map(|r| view.my_hand.iter().find(|c| c.id == r.card))
            .map(|c| c.to_string())
            .collect();
        opts.push("Pass (don't respond)".into());

        let choice = self.prompt_pick("Respond to chain?", &opts);
        if choice == opts.len() - 1 { None } else { Some(choice) }
    }

    fn decide(&self, _view: &GameView, request: &ChoiceRequest) -> usize {
        self.prompt_pick(&request.prompt, &request.options)
    }

    fn pick_targets(
        &self,
        _view: &GameView,
        prompt: &str,
        options: &[TargetOption],
        count: usize,
    ) -> Vec<usize> {
        if options.is_empty() { return vec![]; }
        let labels: Vec<String> = options.iter().map(|o| o.label.clone()).collect();
        let mut picks = Vec::new();
        for i in 0..count.min(options.len()) {
            let choice = self.prompt_pick(
                &format!("{} (pick {} of {})", prompt, i + 1, count),
                &labels,
            );
            picks.push(choice);
        }
        picks
    }

    fn choose_discards(
        &self,
        _view: &GameView,
        hand: &[Card],
        must_discard: usize,
    ) -> Vec<CardId> {
        let opts: Vec<String> = hand.iter().map(|c| c.to_string()).collect();
        let mut discards = Vec::new();
        for i in 0..must_discard {
            let available: Vec<String> = opts.iter().enumerate()
                .filter(|(_, s)| !discards.iter().any(|&id| {
                    hand.iter().find(|c| c.id == id)
                        .map(|c| c.to_string() == **s)
                        .unwrap_or(false)
                }))
                .map(|(_, s)| s.clone())
                .collect();
            let choice = self.prompt_pick(
                &format!("Choose card to discard ({} of {}):", i + 1, must_discard),
                &available,
            );
            let non_discarded: Vec<&Card> = hand.iter().filter(|c| !discards.contains(&c.id)).collect();
            if let Some(card) = non_discarded.get(choice) {
                discards.push(card.id);
            }
        }
        discards
    }
}

// ── Action description (CLI) ──────────────────────────────────────────────────

fn describe_action(action: &Action, view: &GameView, _idx: usize) -> String {
    let card_name = |id: CardId| {
        view.my_hand.iter().find(|c| c.id == id).map(|c| c.to_string()).unwrap_or_default()
    };
    match action {
        Action::Pass => "Pass (end action phase)".into(),
        Action::PlayStar(id) => format!("Place {} in Star Chart", card_name(*id)),
        Action::PlayDiscovery { card, onto_star } => {
            let sys_name = view.my_star_chart.system_for_star(*onto_star)
                .map(|s| s.star_name.to_string()).unwrap_or_default();
            format!("Attach {} to {} system", card_name(*card), sys_name)
        }
        Action::PlayFlare(id) => format!("Play {} (Flare → Chain)", card_name(*id)),
        Action::PlayFracture(id) => format!("Play {} (Fracture → Chain)", card_name(*id)),
        Action::PlayOmen { card, attachment } => {
            let desc = match attachment {
                OmenAttachment::SolarEclipse { system_star } => {
                    view.my_star_chart.system_for_star(*system_star)
                        .map(|s| format!("on {} system", s.star_name))
                        .unwrap_or_default()
                }
                OmenAttachment::PlanetOfLife { .. } => "on Rocky Planet".into(),
            };
            format!("Play {} {}", card_name(*card), desc)
        }
        Action::PlaySolarSystem { star, discoveries } => {
            let disc_names: Vec<String> = discoveries.iter().map(|&id| card_name(id)).collect();
            format!("Complete Solar System: {} + [{}]", card_name(*star), disc_names.join(", "))
        }
        Action::PlaySupernovas { .. } => "⚠ PLAY ALL 3 SUPERNOVAS - Cosmic Cataclysm!".into(),
        Action::PlayProtostar { card, choice } => match choice {
            ProtostarChoice::DrawTwo => format!("Play {} → draw 2 cards", card_name(*card)),
            ProtostarChoice::SearchWhiteDwarf => {
                format!("Play {} → search White Dwarf → place in Star Chart", card_name(*card))
            }
        },
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn model_short_name(tag: &str, fallback_idx: usize) -> String {
    let base = tag.split(':').next().unwrap_or(tag);
    let size = tag.split(':').nth(1).unwrap_or("");
    let short = match base {
        s if s.starts_with("deepseek") => "DeepSeek",
        s if s.starts_with("llama3.1") => "Llama3.1",
        s if s.starts_with("llama3.2") => "Llama3.2",
        s if s.starts_with("llama") => "Llama",
        s if s.starts_with("qwen3.5") => "Qwen3.5",
        s if s.starts_with("qwen3") => "Qwen3",
        s if s.starts_with("qwen2") => "Qwen2.5",
        s if s.starts_with("phi") => "Phi",
        s if s.starts_with("gemma4") => "Gemma4",
        s if s.starts_with("gemma") => "Gemma",
        s if s.starts_with("gpt-oss") => "GPT-OSS",
        s if s.starts_with("lfm") => "LFM",
        "cpu" => return format!("CPU-{}", fallback_idx + 1),
        "easy" | "easycpu" => return format!("Easy-{}", fallback_idx + 1),
        "advcpu" | "advanced" | "pro" => return format!("AdvCPU-{}", fallback_idx + 1),
        _ => &base[..base.len().min(8)],
    };
    if size.is_empty() {
        format!("{}-{}", short, fallback_idx + 1)
    } else {
        format!("{}-{}", short, size)
    }
}

fn print_result(result: &GameResult) {
    println!("\n╔══════════════════════════════════╗");
    println!("║            GAME OVER             ║");
    println!("╠══════════════════════════════════╣");
    println!("║ Winner: {:24} ║", result.winner_name);
    let condition = match result.condition {
        WinCondition::AstronomicalDomination => "Astronomical Domination",
        WinCondition::CosmicCataclysm => "Cosmic Cataclysm",
    };
    println!("║ Condition: {:21} ║", condition);
    println!("║ Turns: {:25} ║", result.turns_elapsed);
    println!("╚══════════════════════════════════╝");
}

// ── Server connect mode ───────────────────────────────────────────────────────

fn run_connect(host_arg: Option<String>) {
    use std::sync::mpsc;
    use novacana_engine::game::GameEvent;
    use novacana_engine::human::{HumanPrompt, HumanResponse};
    use novacana_engine::protocol::{ClientMsg, ServerMsg};
    use lobby::{menu, queue};

    // 1 - Connect screen (or use CLI arg).
    let (host, port) = if let Some(arg) = host_arg {
        match arg.rsplit_once(':') {
            Some((h, p)) => match p.parse::<u16>() {
                Ok(p) => (h.to_string(), p),
                Err(_) => { eprintln!("Invalid port in '{}'", arg); return; }
            },
            None => (arg, 7878),
        }
    } else {
        match lobby::connect::run() {
            Ok(Some(r)) => (r.host, r.port),
            Ok(None) => return,
            Err(e) => { eprintln!("Connect screen error: {}", e); return; }
        }
    };

    // 2 - Open WebSocket.
    let url = format!("ws://{}:{}/ws", host, port);
    let ws = match tungstenite::connect(&url) {
        Ok((ws, _)) => ws,
        Err(e) => { eprintln!("Failed to connect to {}: {}", url, e); return; }
    };

    // 3 - Bridge channels.
    let (game_event_tx, game_event_rx) = mpsc::sync_channel::<GameEvent>(256);
    let (prompt_tx, prompt_rx) = mpsc::sync_channel::<HumanPrompt>(4);
    let (lobby_tx, lobby_rx) = mpsc::sync_channel::<ServerMsg>(32);
    let (client_msg_tx, client_msg_rx) = mpsc::sync_channel::<ClientMsg>(16);
    let (response_fwd_tx, response_fwd_rx) = mpsc::sync_channel::<HumanResponse>(4);

    // Forward human responses as ClientMsg::Action.
    {
        let msg_tx = client_msg_tx.clone();
        std::thread::spawn(move || {
            while let Ok(r) = response_fwd_rx.recv() {
                let _ = msg_tx.send(ClientMsg::Action(r));
            }
        });
    }

    // Split WebSocket into read/write halves via a mutex.
    let ws = std::sync::Arc::new(std::sync::Mutex::new(ws));

    // Write thread: drain client_msg_rx → socket.
    {
        let ws = ws.clone();
        std::thread::spawn(move || {
            while let Ok(msg) = client_msg_rx.recv() {
                match serde_json::to_string(&msg) {
                    Ok(json) => {
                        let mut ws = ws.lock().unwrap();
                        let _ = ws.send(tungstenite::Message::Text(json.into()));
                    }
                    Err(_) => {}
                }
            }
        });
    }

    // Read thread: route ServerMsg to the right channel.
    {
        let ws = ws.clone();
        let game_event_tx2 = game_event_tx.clone();
        let prompt_tx2 = prompt_tx.clone();
        let lobby_tx2 = lobby_tx.clone();
        std::thread::spawn(move || {
            loop {
                let msg = {
                    let mut ws = ws.lock().unwrap();
                    ws.read()
                };
                match msg {
                    Ok(tungstenite::Message::Text(text)) => {
                        let server_msg: ServerMsg = match serde_json::from_str(&text) {
                            Ok(m) => m,
                            Err(_) => continue,
                        };
                        match server_msg {
                            ServerMsg::Event(e) => { let _ = game_event_tx2.send(e); }
                            ServerMsg::Prompt(p) => { let _ = prompt_tx2.send(p); }
                            other => { let _ = lobby_tx2.send(other); }
                        }
                    }
                    Ok(tungstenite::Message::Close(_)) | Err(_) => break,
                    _ => {}
                }
            }
        });
    }

    // 4 - Auth screen.
    let username = match lobby::auth::run(&client_msg_tx, &lobby_rx) {
        Ok(Some(r)) => r.username,
        Ok(None) => return,
        Err(e) => { eprintln!("Auth error: {}", e); return; }
    };

    // 5 - Drain LobbyInfo to get online count.
    let mut online: u32 = 0;
    while let Ok(msg) = lobby_rx.try_recv() {
        if let ServerMsg::LobbyInfo { online: n } = msg { online = n; }
    }

    // 6 - Main menu loop.
    loop {
        match menu::run(&username, online) {
            Ok(menu::MenuAction::Quit) => return,
            Ok(menu::MenuAction::Play) => {}
            Err(e) => { eprintln!("Menu error: {}", e); return; }
        }

        // Drain any stale LobbyInfo updates.
        while let Ok(msg) = lobby_rx.try_recv() {
            if let ServerMsg::LobbyInfo { online: n } = msg { online = n; }
        }

        // 7 - Queue.
        match queue::run(&client_msg_tx, &lobby_rx) {
            Ok(queue::QueueResult::Cancelled) => continue,
            Ok(queue::QueueResult::GameFound { .. }) => {}
            Err(e) => { eprintln!("Queue error: {}", e); return; }
        }

        // 8 - Game TUI.
        if let Err(e) = tui::run_interactive(game_event_rx, prompt_rx, response_fwd_tx) {
            eprintln!("Game TUI error: {}", e);
            return;
        }
        // After game ends, re-borrow the channels and go back to menu.
        // NOTE: game_event_rx and prompt_rx are consumed here - after the first
        // game this would fail. Ideally we rebuild them, but for v1 the client
        // exits after one game and the user reconnects.
        return;
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────
//
// Usage:
//   cargo run -p novacana -- play                    you vs 3 full CPUs (TUI)
//   cargo run -p novacana -- onboarding              you vs 3 easy CPUs (TUI)
//   cargo run -p novacana -- observe [models…]       TUI: N LLMs / CPUs play
//   cargo run -p novacana -- bots [N=2]              random bots only
//   cargo run -p novacana -- ollama [N=2] [model]    N Ollama LLM players
//   cargo run -p novacana -- human-vs-ollama [model] you vs 1 Ollama player (CLI)
//   cargo run -p novacana -- ollama-vs-bots [model]  1 Ollama + 1 random bot
//   cargo run -p novacana -- sim [N=100] [cpu=advanced] [players=4]  balance sim
//   cargo run -p novacana -- connect [host:port]     connect to a server

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("bots");

    let mut player_count: usize = 2;
    let mut models: Vec<String> = Vec::new();
    for arg in args.iter().skip(2) {
        if let Ok(n) = arg.parse::<usize>() {
            player_count = n.clamp(2, 4);
        } else {
            models.push(arg.clone());
        }
    }
    let model = models.first().cloned().unwrap_or_else(|| "llama3.2:3b".to_string());

    println!("╔══════════════════════════════════╗");
    println!("║         NOVACANA ENGINE          ║");
    println!("╠══════════════════════════════════╣");
    println!("║ mode:  {:25} ║", mode);
    if models.len() > 1 {
        for (i, m) in models.iter().enumerate() {
            println!("║ p{}:    {:25} ║", i + 1, m);
        }
    } else {
        println!("║ model: {:25} ║", model);
    }
    println!("║ players: {:23} ║", if models.len() > 1 { models.len() } else { player_count });
    println!("╚══════════════════════════════════╝\n");

    let players: Vec<Box<dyn Player>> = match mode {
        "play" | "onboarding" => {
            let (game_tx, game_rx) = std::sync::mpsc::sync_channel(256);
            let (prompt_tx, prompt_rx) = std::sync::mpsc::sync_channel(1);
            let (response_tx, response_rx) = std::sync::mpsc::sync_channel(1);

            let human: Box<dyn Player> = Box::new(HumanPlayer::new("You", prompt_tx, response_rx));
            let opponents: Vec<Box<dyn Player>> = if mode == "onboarding" {
                vec![
                    Box::new(EasyCpuPlayer::new("Learner-A")),
                    Box::new(EasyCpuPlayer::new("Learner-B")),
                    Box::new(EasyCpuPlayer::new("Learner-C")),
                ]
            } else {
                vec![
                    Box::new(AdvancedCpuPlayer::new("Alpha")),
                    Box::new(AdvancedCpuPlayer::new("Beta")),
                    Box::new(AdvancedCpuPlayer::new("Gamma")),
                ]
            };

            let mut all: Vec<Box<dyn Player>> = vec![human];
            all.extend(opponents);

            let deck = Deck::build_classic();
            std::thread::spawn(move || {
                let mut engine = GameEngine::new(deck, all);
                engine.set_observer(game_tx);
                let result = engine.run();
                if let Err(e) = engine.save_log(&result) {
                    eprintln!("Failed to save game log: {}", e);
                }
            });

            if let Err(e) = tui::run_interactive(game_rx, prompt_rx, response_tx) {
                eprintln!("TUI error: {}", e);
            }
            return;
        }

        "human" => {
            let mut ps: Vec<Box<dyn Player>> = vec![Box::new(CliPlayer::new("Human"))];
            for i in 1..player_count {
                ps.push(Box::new(RandomBot::new(format!("Bot {}", i))));
            }
            ps
        }

        "ollama" => (0..player_count)
            .map(|i| -> Box<dyn Player> { Box::new(OllamaPlayer::new(format!("LLM {}", i + 1), &model)) })
            .collect(),

        "human-vs-ollama" => vec![
            Box::new(CliPlayer::new("Human")) as Box<dyn Player>,
            Box::new(OllamaPlayer::new("LLM", &model)),
        ],

        "ollama-vs-bots" => {
            let mut ps: Vec<Box<dyn Player>> =
                vec![Box::new(OllamaPlayer::new(format!("LLM ({})", model), &model))];
            for i in 1..player_count {
                ps.push(Box::new(RandomBot::new(format!("Bot {}", i))));
            }
            ps
        }

        "observe" => {
            let player_models: Vec<(String, String)> = if models.len() >= 2 {
                models.iter().take(4).enumerate()
                    .map(|(i, m)| (model_short_name(m, i), m.clone()))
                    .collect()
            } else {
                let count = player_count.clamp(2, 4);
                (0..count).map(|i| (format!("LLM {}", i + 1), model.clone())).collect()
            };

            let players: Vec<Box<dyn Player>> = player_models.iter()
                .map(|(name, m)| -> Box<dyn Player> {
                    match m.as_str() {
                        "cpu" => Box::new(CpuPlayer::new(name)),
                        "easy" | "easycpu" => Box::new(EasyCpuPlayer::new(name)),
                        "advcpu" | "advanced" | "pro" => Box::new(AdvancedCpuPlayer::new(name)),
                        _ => Box::new(OllamaPlayer::new(name, m).with_silent()),
                    }
                })
                .collect();

            let deck = Deck::build_classic();
            let (tx, rx) = mpsc::sync_channel(256);
            std::thread::spawn(move || {
                let mut engine = GameEngine::new(deck, players);
                engine.set_observer(tx);
                let result = engine.run();
                if let Err(e) = engine.save_log(&result) {
                    eprintln!("Failed to save game log: {}", e);
                }
            });

            if let Err(e) = tui::run_observer(rx) {
                eprintln!("TUI error: {}", e);
            }
            return;
        }

        "sim" => {
            let mut num_games = 100usize;
            let mut num_players = 4usize;
            let mut cpu = sim::CpuKind::Advanced;
            for arg in args.iter().skip(2) {
                if let Ok(n) = arg.parse::<usize>() {
                    if n >= 2 && n <= 4 {
                        num_players = n;
                    } else {
                        num_games = n;
                    }
                } else if let Some(k) = sim::CpuKind::from_str(arg) {
                    cpu = k;
                }
            }
            sim::run(num_games, num_players, cpu);
            return;
        }

        "connect" => {
            let host_arg = args.get(2).cloned();
            run_connect(host_arg);
            return;
        }

        _ => (0..player_count)
            .map(|i| -> Box<dyn Player> { Box::new(RandomBot::new(format!("Bot {}", i + 1))) })
            .collect(),
    };

    let deck = Deck::build_classic();
    let mut engine = GameEngine::new(deck, players);
    let result = engine.run();
    if let Err(e) = engine.save_log(&result) {
        eprintln!("Failed to save game log: {}", e);
    }
    print_result(&result);
}
