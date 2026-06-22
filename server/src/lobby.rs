use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{self, Receiver, SyncSender};

use novacana_engine::card::CardId;
use novacana_engine::chain::ChainView;
use novacana_engine::deck::Deck;
use novacana_engine::game::{GameEngine, GameEvent};
use novacana_engine::human::{HumanPrompt, HumanResponse, PromptKind};
use novacana_engine::player::{Action, ChainAction, ChoiceRequest, GameView, Player, TargetOption};
use novacana_engine::protocol::ServerMsg;

// ── WebSocket-backed human player ─────────────────────────────────────────────

/// Implements `Player` by routing prompts over a channel to the WebSocket handler.
pub struct WsHumanPlayer {
    pub name: String,
    prompt_tx: SyncSender<ServerMsg>,
    response_rx: Mutex<Receiver<HumanResponse>>,
}

impl WsHumanPlayer {
    pub fn new(
        name: String,
        prompt_tx: SyncSender<ServerMsg>,
        response_rx: Receiver<HumanResponse>,
    ) -> Self {
        Self { name, prompt_tx, response_rx: Mutex::new(response_rx) }
    }

    fn send_prompt(&self, prompt: HumanPrompt) {
        let _ = self.prompt_tx.send(ServerMsg::Prompt(prompt));
    }

    fn recv_pick(&self) -> usize {
        match self.response_rx.lock().unwrap().recv() {
            Ok(HumanResponse::Pick(i)) => i,
            _ => 0,
        }
    }

    fn recv_chain(&self) -> Option<usize> {
        match self.response_rx.lock().unwrap().recv() {
            Ok(HumanResponse::Pick(i)) => Some(i),
            Ok(HumanResponse::Pass) | Err(_) => None,
        }
    }

    fn hand_cards(view: &GameView) -> Vec<novacana_engine::human::HandCardDisplay> {
        use novacana_engine::card_db;
        use novacana_engine::human::HandCardDisplay;
        view.my_hand.iter().map(|c| {
            let name = c.kind.short_name();
            let card_type = c.kind.type_name().to_string();
            let effect = card_db::lookup(&name).map(|e| e.effect.clone()).unwrap_or_default();
            HandCardDisplay { name, card_type, effect }
        }).collect()
    }
}

impl Player for WsHumanPlayer {
    fn name(&self) -> &str { &self.name }

    fn choose_action(&self, view: &GameView, legal_actions: &[Action]) -> usize {
        use novacana_engine::human::describe_action;
        let options = legal_actions.iter().map(|a| describe_action(a, view)).collect();
        let action_types = legal_actions.iter().map(|a| {
            match a {
                Action::Pass => None,
                Action::PlayStar(_) | Action::PlayProtostar { .. } | Action::PlaySolarSystem { .. } => Some("Star".to_string()),
                Action::PlayDiscovery { .. } => Some("Discovery".to_string()),
                Action::PlayFlare(_) | Action::PlaySupernovas { .. } => Some("Flare".to_string()),
                Action::PlayFracture(_) => Some("Fracture".to_string()),
                Action::PlayOmen { .. } => Some("Omen".to_string()),
            }
        }).collect();
        self.send_prompt(HumanPrompt {
            kind: PromptKind::ChooseAction { options, action_types },
            hand_cards: Self::hand_cards(view),
        });
        self.recv_pick().min(legal_actions.len().saturating_sub(1))
    }

    fn respond_to_chain(&self, view: &GameView, chain: &ChainView, legal_responses: &[ChainAction]) -> Option<usize> {
        if legal_responses.is_empty() { return None; }
        let chain_lines = chain.links.iter().map(|l| {
            let neg = if l.negated { " (NEGATED)" } else { "" };
            format!("CL{} [{}] {} — {}{}", l.index + 1, l.player_name, l.card.kind.type_name(), l.card, neg)
        }).collect();
        let options = legal_responses.iter()
            .filter_map(|r| view.my_hand.iter().find(|c| c.id == r.card))
            .map(|c| c.to_string())
            .collect();
        self.send_prompt(HumanPrompt {
            kind: PromptKind::RespondToChain { chain: chain_lines, options },
            hand_cards: Self::hand_cards(view),
        });
        let pick = self.recv_chain()?;
        Some(pick.min(legal_responses.len().saturating_sub(1)))
    }

    fn decide(&self, view: &GameView, request: &ChoiceRequest) -> usize {
        self.send_prompt(HumanPrompt {
            kind: PromptKind::Decide { prompt: request.prompt.clone(), options: request.options.clone() },
            hand_cards: Self::hand_cards(view),
        });
        self.recv_pick().min(request.options.len().saturating_sub(1))
    }

    fn pick_targets(&self, view: &GameView, prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
        let labels: Vec<String> = options.iter().map(|o| o.label.clone()).collect();
        let mut picked = Vec::new();
        let mut available: Vec<usize> = (0..labels.len()).collect();
        for i in 0..count.min(options.len()) {
            let opts: Vec<String> = available.iter().map(|&j| labels[j].clone()).collect();
            let p = if count > 1 { format!("{} ({} of {})", prompt, i + 1, count) } else { prompt.to_string() };
            self.send_prompt(HumanPrompt {
                kind: PromptKind::PickTarget { prompt: p, options: opts },
                hand_cards: Self::hand_cards(view),
            });
            let pick = self.recv_pick().min(available.len().saturating_sub(1));
            picked.push(available.remove(pick));
        }
        picked
    }

    fn choose_discards(&self, view: &GameView, hand: &[novacana_engine::card::Card], must_discard: usize) -> Vec<CardId> {
        let mut remaining: Vec<&novacana_engine::card::Card> = hand.iter().collect();
        let mut result = Vec::new();
        for step in 0..must_discard {
            let options: Vec<String> = remaining.iter().map(|c| c.to_string()).collect();
            self.send_prompt(HumanPrompt {
                kind: PromptKind::ChooseDiscard { options, step: step + 1, total: must_discard },
                hand_cards: Self::hand_cards(view),
            });
            let pick = self.recv_pick().min(remaining.len().saturating_sub(1));
            result.push(remaining[pick].id);
            remaining.remove(pick);
        }
        result
    }
}

// ── PlayerHandle ──────────────────────────────────────────────────────────────

/// Channels connecting the per-connection WebSocket task to the game thread.
pub struct PlayerHandle {
    pub username: String,
    /// Send game messages (prompts, events, game-over) to the client.
    pub tx: SyncSender<ServerMsg>,
    /// Send player responses from the client into the game engine.
    pub response_tx: SyncSender<HumanResponse>,
    /// The game engine reads from this end.
    pub response_rx: Option<Receiver<HumanResponse>>,
}

impl PlayerHandle {
    pub fn new(username: String, tx: SyncSender<ServerMsg>) -> Self {
        let (response_tx, response_rx) = mpsc::sync_channel(4);
        Self { username, tx, response_tx, response_rx: Some(response_rx) }
    }
}

// ── LobbyState ────────────────────────────────────────────────────────────────

pub struct LobbyState {
    /// Queues by game size (2, 3, or 4).
    queues: HashMap<u8, Vec<PlayerHandle>>,
    pub online: u32,
}

impl LobbyState {
    pub fn new() -> Self {
        Self {
            queues: HashMap::from([(2, vec![]), (3, vec![]), (4, vec![])]),
            online: 0,
        }
    }

    pub fn join_queue(&mut self, size: u8, handle: PlayerHandle) -> Option<Vec<PlayerHandle>> {
        let queue = self.queues.entry(size).or_default();
        queue.push(handle);
        if queue.len() >= size as usize {
            let players: Vec<PlayerHandle> = queue.drain(..size as usize).collect();
            return Some(players);
        }
        None
    }

    /// Remove a player from any queue they are in (returns true if found).
    pub fn leave_queue(&mut self, username: &str) -> bool {
        for queue in self.queues.values_mut() {
            if let Some(pos) = queue.iter().position(|h| h.username == username) {
                queue.remove(pos);
                return true;
            }
        }
        false
    }
}

pub type SharedLobby = Arc<Mutex<LobbyState>>;

// ── Game spawning ─────────────────────────────────────────────────────────────

pub fn spawn_game(mut handles: Vec<PlayerHandle>) {
    let player_names: Vec<String> = handles.iter().map(|h| h.username.clone()).collect();
    let txs: Vec<SyncSender<ServerMsg>> = handles.iter().map(|h| h.tx.clone()).collect();

    // Notify all players that the game was found.
    for tx in &txs {
        let _ = tx.send(ServerMsg::GameFound { players: player_names.clone() });
    }

    // Build WsHumanPlayer for each handle, consuming the response_rx.
    let players: Vec<Box<dyn Player>> = handles.iter_mut().map(|h| {
        let rx = h.response_rx.take().expect("response_rx already consumed");
        Box::new(WsHumanPlayer::new(h.username.clone(), h.tx.clone(), rx)) as Box<dyn Player>
    }).collect();

    // Build an observer channel that broadcasts GameEvents to all clients.
    let (obs_tx, obs_rx) = mpsc::sync_channel::<GameEvent>(64);
    let broadcast_txs = txs.clone();
    std::thread::spawn(move || {
        while let Ok(event) = obs_rx.recv() {
            let msg = ServerMsg::Event(event);
            for tx in &broadcast_txs {
                let _ = tx.send(msg.clone());
            }
        }
    });

    std::thread::spawn(move || {
        let deck = Deck::build_classic();
        let mut engine = GameEngine::new(deck, players);
        engine.set_observer(obs_tx);
        let result = engine.run();
        for tx in &txs {
            let _ = tx.send(ServerMsg::GameEnded { result: result.clone() });
        }
    });
}
