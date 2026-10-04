use std::collections::HashMap;
use std::sync::mpsc::SyncSender;
use std::sync::{Arc, Mutex};

use crate::card::{Card, CardId, CardKind, DiscoveryName, FlareName, FractureName, OmenName, StarName};
use crate::chain::{Chain, ChainLink, ChainResumeCursor, ChainView};
use crate::deck::Deck;
use crate::player::{
    Action, ChainAction, ChoiceRequest, GameView, OmenAttachment, OpponentView, Player,
    ProtostarChoice, Target, TargetOption,
};
use crate::star_chart::StarChart;

// Keep formatting and string allocation out of silent search simulations.
macro_rules! emit {
    ($engine:ident, $message:expr) => {
        if !$engine.silent || $engine.observer.is_some() {
            $engine.emit($message);
        }
    };
}

// ── Observer events ───────────────────────────────────────────────────────────

/// Events sent to the TUI observer thread after each significant state change.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum GameEvent {
    /// A plain-text log line.
    Log(String),
    /// A full state snapshot - sent after each meaningful action.
    Snapshot(GameSnapshot),
    /// Fired before an LLM player starts thinking (for "… is thinking" indicator).
    Thinking { player_name: String },
    /// A raw token chunk streamed live from a generating model.
    ThinkingChunk { player_name: String, text: String },
    /// Fired when the game ends.
    GameOver(GameResult),
}

/// Observable slice of the game state, cloned and sent across the channel.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GameSnapshot {
    pub players: Vec<PlayerSnapshot>,
    pub deck_size: usize,
    pub turn: u32,
    pub dark_matter_void: Option<u8>,
    /// Human-readable chain link descriptions (empty when no chain is open).
    pub chain_description: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlayerSnapshot {
    pub name: String,
    pub hand_size: usize,
    pub star_chart: StarChart,
    pub is_active: bool,
    pub skip_next_action: bool,
    pub astral_projected_turns: u8,
}

// ── Player state ──────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct PlayerState {
    pub name: String,
    pub hand: Vec<Card>,
    pub star_chart: StarChart,
    /// Set by Black Hole - skip the next Action Phase (draw and discard still happen).
    pub skip_next_action: bool,
    /// Turns remaining where this player's hand is revealed (Astral Projection).
    pub astral_projected_turns: u8,
    /// Omen card IDs whose Planet of Life ability was used last turn (one-turn cooldown).
    pub planet_of_life_on_cooldown: Vec<CardId>,
}

impl PlayerState {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            hand: Vec::new(),
            star_chart: StarChart::new(),
            skip_next_action: false,
            astral_projected_turns: 0,
            planet_of_life_on_cooldown: Vec::new(),
        }
    }

    pub fn take_from_hand(&mut self, card_id: CardId) -> Option<Card> {
        let pos = self.hand.iter().position(|c| c.id == card_id)?;
        Some(self.hand.remove(pos))
    }

    pub fn hand_contains(&self, card_id: CardId) -> bool {
        self.hand.iter().any(|c| c.id == card_id)
    }
}

// ── Global effect state ───────────────────────────────────────────────────────

#[derive(Clone)]
struct DarkMatterVoid {
    restricted_class: u8,
    caster_idx: usize,
}

// ── Win condition ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum WinCondition {
    /// Controlled 3 complete Solar Systems at end of turn.
    AstronomicalDomination,
    /// Played all three unique Supernova Flares simultaneously.
    CosmicCataclysm,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GameResult {
    pub winner_idx: usize,
    pub winner_name: String,
    pub condition: WinCondition,
    pub turns_elapsed: u32,
}

// ── Game state ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct GameState {
    pub deck: Deck,
    /// Arc, mirroring `Deck::registry` (see its doc comment) - always the
    /// same underlying map as `deck.registry`, just cloned out to its own
    /// field for convenient access without going through `deck.`.
    pub registry: Arc<HashMap<CardId, CardKind>>,
    pub players: Vec<PlayerState>,
    pub active_player_idx: usize,
    pub turn_number: u32,
    dark_matter_void: Option<DarkMatterVoid>,
    stellar_alignment_this_turn: bool,
    /// Turn number when the deck first ran out. Drives post-exhaustion hand-limit pressure.
    deck_exhausted_at_turn: Option<u32>,
    /// Rolling log of the last 20 card plays: (player_idx, card_kind). v6 feature input.
    pub play_history: Vec<(usize, CardKind)>,
    /// Cumulative count of how many times each card type has been played this game.
    /// Indexed by CardKind::type_idx(). v6 feature input.
    pub played_counts: [u8; 29],
    /// Whether initial 5-card hands have been dealt yet. Explicit flag rather
    /// than inferring "fresh game" from `turn_number`/hand-emptiness: this is
    /// exactly the class of bug that corrupted simulations for weeks (see
    /// `GameEngine::run`'s deal guard) - a boolean set once at deal time and
    /// carried forward by `Clone` can't drift out of sync with turn-number
    /// bookkeeping the way a heuristic comparison can.
    hands_dealt: bool,
}

impl GameState {
    fn new(deck: Deck, player_names: Vec<String>) -> Self {
        let registry = deck.registry.clone();
        let players = player_names.into_iter().map(PlayerState::new).collect();
        Self {
            deck,
            registry,
            players,
            active_player_idx: 0,
            turn_number: 1,
            dark_matter_void: None,
            stellar_alignment_this_turn: false,
            deck_exhausted_at_turn: None,
            play_history: Vec::new(),
            played_counts: [0u8; 29],
            hands_dealt: false,
        }
    }

    pub fn record_play(&mut self, player_idx: usize, kind: CardKind) {
        self.played_counts[kind.type_idx()] = self.played_counts[kind.type_idx()].saturating_add(1);
        // 64, not 20: v7's history tokens (see ASTRA7.md §3.1) need enough
        // window to see a full round or two, not just the last few plays.
        if self.play_history.len() == 64 {
            self.play_history.remove(0);
        }
        self.play_history.push((player_idx, kind));
    }

    pub fn reconstruct(&self, id: CardId) -> Option<Card> {
        self.registry.get(&id).map(|k| Card::new(id, k.clone()))
    }
}

// ── Game engine ───────────────────────────────────────────────────────────────

pub struct GameEngine {
    pub state: GameState,
    pub players: Vec<Box<dyn Player>>,
    pub log: Arc<Mutex<Vec<String>>>,
    observer: Option<SyncSender<GameEvent>>,
    silent: bool,
    max_turns: Option<u32>,
    /// Total-decision-count watchdog: an ADDITIONAL safety net alongside
    /// `max_turns` and `run_action_phase`'s 50-action-per-turn cap. Those two
    /// only bound cost inside `run()`'s own turn loop; `simulate_finish_chain`
    /// never enters that loop at all (see `cpu.rs`), so before this field it
    /// had literally no cost cap - a chain-response search that kept finding
    /// legal reactions could run indefinitely. `max_decisions` bounds the
    /// TOTAL number of real decision points (choose_action / respond_to_chain
    /// / decide / pick_targets / choose_discards) taken by this engine
    /// instance, regardless of which loop is responsible. None (unset) by
    /// default, matching `max_turns`'s pattern - production/live games never
    /// opt in, only batch simulation code does.
    max_decisions: Option<usize>,
    decisions_taken: usize,
}

impl GameEngine {
    pub fn new(mut deck: Deck, players: Vec<Box<dyn Player>>) -> Self {
        let names: Vec<String> = players.iter().map(|p| p.name().to_string()).collect();
        deck.shuffle();
        let state = GameState::new(deck, names);
        Self {
            state, players, log: Arc::new(Mutex::new(Vec::new())), observer: None,
            silent: false, max_turns: None, max_decisions: None, decisions_taken: 0,
        }
    }

    /// Construct from a pre-existing state snapshot (used by MCTS simulations).
    /// The deck is NOT re-shuffled; players must already be in the correct seats.
    pub fn from_state(state: GameState, players: Vec<Box<dyn Player>>) -> Self {
        Self {
            state, players, log: Arc::new(Mutex::new(Vec::new())), observer: None,
            silent: true, max_turns: None, max_decisions: None, decisions_taken: 0,
        }
    }

    /// Suppress stdout logging when no observer is attached (useful for batch simulations).
    pub fn set_silent(&mut self) { self.silent = true; }

    /// End the game after this many turns, awarding victory to whoever has the
    /// most complete systems (tiebreak: lowest player index). Use in batch sims
    /// to prevent runaway games from stalling the pool.
    pub fn set_max_turns(&mut self, n: u32) { self.max_turns = Some(n); }

    /// End the game (same truncation rule as `set_max_turns`: award whoever
    /// has the most complete systems) once this many real decision points
    /// have been taken. See the `max_decisions` field doc for why this exists
    /// as a separate cap from `max_turns`.
    pub fn set_max_decisions(&mut self, n: usize) { self.max_decisions = Some(n); }

    /// Award the current leader (most complete systems, tiebreak: lowest
    /// player index) - the truncation rule shared by `max_turns` and
    /// `max_decisions` timeouts.
    fn award_leader(&self) -> GameResult {
        let (winner_idx, _) = self.state.players.iter().enumerate()
            .max_by_key(|(_, p)| p.star_chart.complete_count())
            .unwrap();
        GameResult {
            winner_idx,
            winner_name: self.state.players[winner_idx].name.clone(),
            condition: WinCondition::AstronomicalDomination,
            turns_elapsed: self.state.turn_number,
        }
    }

    /// Count one real decision point toward the `max_decisions` budget.
    fn tick_decision(&mut self) {
        self.decisions_taken += 1;
    }

    /// `Some(result)` iff `max_decisions` is set and has been exceeded -
    /// callers treat this exactly like a `max_turns` timeout: stop
    /// immediately and award the current leader.
    fn decision_budget_exceeded(&self) -> Option<GameResult> {
        let max = self.max_decisions?;
        if self.decisions_taken > max {
            Some(self.award_leader())
        } else {
            None
        }
    }

    /// Ask `idx` a mid-effect decision. Centralizes view-building and
    /// decision-budget bookkeeping so every `decide()` call site across the
    /// ~20 card effects can't forget to tick the counter. Routes through
    /// `decide_with_state` (real state access for the ~20 call sites this
    /// covers) rather than `decide` directly.
    fn ask_decide(&mut self, idx: usize, req: &ChoiceRequest) -> usize {
        self.tick_decision();
        let view = self.build_view(idx);
        self.players[idx].decide_with_state(&view, &self.state, req)
    }

    /// Ask `idx` to pick targets. See `ask_decide`.
    fn ask_pick_targets(&mut self, idx: usize, prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
        self.tick_decision();
        let view = self.build_view(idx);
        self.players[idx].pick_targets_with_state(&view, &self.state, prompt, options, count)
    }

    /// Ask `idx` to choose discards. See `ask_decide`.
    fn ask_choose_discards(&mut self, idx: usize, hand: &[Card], must_discard: usize) -> Vec<CardId> {
        self.tick_decision();
        let view = self.build_view(idx);
        self.players[idx].choose_discards_with_state(&view, &self.state, hand, must_discard)
    }

    /// Current hand limit. Starts at 5; drops by 1 for each full round after deck exhaustion.
    fn hand_limit(&self) -> usize {
        match self.state.deck_exhausted_at_turn {
            None => 5,
            Some(t) => {
                let n = self.state.players.len() as u32;
                let rounds = self.state.turn_number.saturating_sub(t) / n;
                5usize.saturating_sub(rounds as usize).max(1)
            }
        }
    }

    /// Attach a TUI observer. When set, `emit()` suppresses stdout and sends
    /// `GameEvent::Log` instead; snapshots are pushed after each action.
    /// Also registers a streaming callback on each player so live token chunks
    /// are forwarded to the TUI as `GameEvent::ThinkingChunk`.
    pub fn set_observer(&mut self, tx: SyncSender<GameEvent>) {
        for (i, player) in self.players.iter().enumerate() {
            let name = self.state.players[i].name.clone();
            let tx2 = tx.clone();
            player.set_thinking_callback(std::sync::Arc::new(move |chunk: String| {
                let _ = tx2.send(GameEvent::ThinkingChunk {
                    player_name: name.clone(),
                    text: chunk,
                });
            }));
        }
        self.observer = Some(tx);
    }

    /// Drain any reasoning text the player generated for their last decision
    /// and emit it to the game log (indented, prefixed with the player's name).
    fn drain_thinking(&mut self, player_idx: usize) {
        if let Some(thought) = self.players[player_idx].take_thinking() {
            let name = self.state.players[player_idx].name.clone();
            emit!(self, format!("[{} reasoning]", name));
            for line in thought.lines() {
                let t = line.trim();
                if !t.is_empty() {
                    emit!(self, format!("  {}", t));
                }
            }
            emit!(self, format!("[/{} reasoning]", name));
        }
    }

    fn send_snapshot(&self, chain: Option<&Chain>) {
        let Some(tx) = &self.observer else { return };
        let active = self.state.active_player_idx;
        let players = self.state.players.iter().enumerate().map(|(i, p)| PlayerSnapshot {
            name: p.name.clone(),
            hand_size: p.hand.len(),
            star_chart: p.star_chart.clone(),
            is_active: i == active,
            skip_next_action: p.skip_next_action,
            astral_projected_turns: p.astral_projected_turns,
        }).collect();

        let chain_description = chain.map(|c| {
            c.resolution_order().iter().enumerate().map(|(i, &raw_idx)| {
                let link = &c.links[raw_idx];
                let player_name = self.state.players.get(link.player_idx)
                    .map(|p| p.name.as_str())
                    .unwrap_or("?");
                format!(
                    "CL{} [{}] {} [{}]{}",
                    i + 1,
                    player_name,
                    link.card.kind.type_name(),
                    link.card,
                    if link.negated { " (NEGATED)" } else { "" }
                )
            }).collect()
        }).unwrap_or_default();

        let snap = GameSnapshot {
            players,
            deck_size: self.state.deck.len(),
            turn: self.state.turn_number,
            dark_matter_void: self.state.dark_matter_void.as_ref().map(|d| d.restricted_class),
            chain_description,
        };
        let _ = tx.send(GameEvent::Snapshot(snap));
    }

    fn notify_thinking(&self, player_idx: usize) {
        let Some(tx) = &self.observer else { return };
        let name = self.state.players.get(player_idx)
            .map(|p| p.name.clone())
            .unwrap_or_default();
        let _ = tx.send(GameEvent::Thinking { player_name: name });
    }

    fn send_game_over(&self, result: &GameResult) {
        let Some(tx) = &self.observer else { return };
        let _ = tx.send(GameEvent::GameOver(GameResult {
            winner_idx: result.winner_idx,
            winner_name: result.winner_name.clone(),
            condition: result.condition.clone(),
            turns_elapsed: result.turns_elapsed,
        }));
    }

    // ── Main loop ─────────────────────────────────────────────────────────────

    pub fn run(&mut self) -> GameResult {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::{Duration, SystemTime, UNIX_EPOCH};

        // Only deal for a fresh game - engines resumed via from_state (MCTS
        // simulations, lookahead search, CPU 1-ply search) already have hands
        // (and `hands_dealt` carries forward through Clone), so dealing again
        // would silently hand every player 5 extra cards. Previously this
        // guard compared `turn_number == 0`, but `GameState::new` initializes
        // `turn_number` to 1 (so "Turn 1" displays correctly) - that
        // mismatch meant fresh games never dealt hands at all. Caught by
        // `tests::from_state_preserves_dealt_state` / `tests::fresh_game_deals_hands`.
        if !self.state.hands_dealt {
            self.deal_initial_hands();
        }

        let stop_saver = (!self.silent).then(|| Arc::new(AtomicBool::new(false)));
        let start_ts = if self.silent { 0 } else {
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
        };

        // Auto-save the log every 10 seconds when not in silent/batch mode.
        if !self.silent {
            let log_arc = Arc::clone(&self.log);
            let player_names: Vec<String> = self.state.players.iter()
                .map(|p| p.name.clone()).collect();
            let stop_clone = Arc::clone(stop_saver.as_ref().unwrap());

            std::thread::spawn(move || {
                let _ = std::fs::create_dir_all("games");
                loop {
                    std::thread::sleep(Duration::from_secs(10));
                    if stop_clone.load(Ordering::Relaxed) { break; }

                    let lines = log_arc.lock().unwrap().clone();
                    let path = format!("games/{}_autosave.log", start_ts);
                    if let Ok(mut f) = std::fs::File::create(&path) {
                        use std::io::Write;
                        let _ = writeln!(f, "NOVACANA - autosave (in progress)");
                        let _ = writeln!(f, "Players: {}", player_names.join(", "));
                        let _ = writeln!(f, "{}", "─".repeat(60));
                        for line in &lines {
                            for sub in line.split('\n') {
                                let t = sub.trim();
                                if !t.is_empty() { let _ = writeln!(f, "{}", t); }
                            }
                        }
                    }
                }
            });
        }

        loop {
            let idx = self.state.active_player_idx;
            emit!(self, format!(
                "\n══ Turn {} - {} ══",
                self.state.turn_number,
                self.state.players[idx].name
            ));

            self.run_start_of_turn(idx);
            self.run_draw_phase(idx);
            self.send_snapshot(None);

            let hand_after_draw = self.state.players[idx].hand.len();
            let limit = self.hand_limit();
            if hand_after_draw <= limit {
                self.state.players[idx].skip_next_action = false;
                emit!(self, format!(
                    "{} has only {} card(s) - action phase skipped.",
                    self.state.players[idx].name, hand_after_draw
                ));
            } else if let Some(result) = self.run_action_phase(idx) {
                self.send_game_over(&result);
                if let Some(stop_saver) = &stop_saver {
                    stop_saver.store(true, std::sync::atomic::Ordering::Relaxed);
                    let _ = std::fs::remove_file(format!("games/{}_autosave.log", start_ts));
                }
                return result;
            }
            if let Some(result) = self.run_end_phase(idx) {
                self.send_game_over(&result);
                if let Some(stop_saver) = &stop_saver {
                    stop_saver.store(true, std::sync::atomic::Ordering::Relaxed);
                    let _ = std::fs::remove_file(format!("games/{}_autosave.log", start_ts));
                }
                return result;
            }

            self.state.stellar_alignment_this_turn = false;
            let n = self.state.players.len();
            self.state.active_player_idx = (self.state.active_player_idx + 1) % n;
            self.state.turn_number += 1;

            // Turn limit: award win to whoever has the most complete systems.
            if let Some(max) = self.max_turns {
                if self.state.turn_number > max {
                    let result = self.award_leader();
                    self.send_game_over(&result);
                    if let Some(stop_saver) = &stop_saver {
                        stop_saver.store(true, std::sync::atomic::Ordering::Relaxed);
                        let _ = std::fs::remove_file(format!("games/{}_autosave.log", start_ts));
                    }
                    return result;
                }
            }

            // Decision-count watchdog: an additional safety net alongside
            // max_turns and the 50-action-per-turn cap (see the
            // `max_decisions` field doc) - catches runaway cost that
            // accumulates WITHIN turns (nested decide()/pick_targets() calls
            // in card effects) rather than across them.
            if let Some(result) = self.decision_budget_exceeded() {
                self.send_game_over(&result);
                if let Some(stop_saver) = &stop_saver {
                    stop_saver.store(true, std::sync::atomic::Ordering::Relaxed);
                    let _ = std::fs::remove_file(format!("games/{}_autosave.log", start_ts));
                }
                return result;
            }
        }
    }

    fn deal_initial_hands(&mut self) {
        for i in 0..self.state.players.len() {
            for _ in 0..5 {
                if let Some(card) = self.state.deck.draw() {
                    self.state.players[i].hand.push(card);
                }
            }
        }
        self.state.hands_dealt = true;
    }

    // ── Phases ────────────────────────────────────────────────────────────────

    fn run_start_of_turn(&mut self, idx: usize) {
        // Expire Dark Matter Void at caster's turn start
        if let Some(dmv) = &self.state.dark_matter_void {
            if dmv.caster_idx == idx {
                emit!(self, "Dark Matter Void expires.".into());
                self.state.dark_matter_void = None;
            }
        }

        // Tick Astral Projection
        if self.state.players[idx].astral_projected_turns > 0 {
            self.state.players[idx].astral_projected_turns -= 1;
            if self.state.players[idx].astral_projected_turns == 0 {
                emit!(self, format!(
                    "{}'s hand is no longer revealed (Astral Projection expired).",
                    self.state.players[idx].name
                ));
            }
        }

        // Reset Planet of Life cooldowns for this player
        self.state.players[idx].planet_of_life_on_cooldown.clear();

        // Solar Eclipse maintenance: each protected system requires discarding 1 card
        let protected: Vec<CardId> = self.state.players[idx].star_chart.eclipse_protected_stars();
        for star_id in protected {
            self.handle_eclipse_maintenance(idx, star_id);
        }
    }

    fn handle_eclipse_maintenance(&mut self, player_idx: usize, star_id: CardId) {
        if self.state.players[player_idx].hand.is_empty() {
            // Cannot pay - detach
            if let Some(omen_id) =
                self.state.players[player_idx].star_chart.detach_solar_eclipse(star_id)
            {
                if let Some(c) = self.state.reconstruct(omen_id) {
                    self.state.deck.push_bottom(c);
                }
                let sys_name = self.state.players[player_idx]
                    .star_chart
                    .system_for_star(star_id)
                    .map(|s| s.star_name.to_string())
                    .unwrap_or_default();
                emit!(self, format!("{}'s Solar Eclipse on {} detaches (no cards to pay).", self.state.players[player_idx].name, sys_name));
            }
            return;
        }

        let sys_name = self.state.players[player_idx]
            .star_chart
            .system_for_star(star_id)
            .map(|s| s.star_name.to_string())
            .unwrap_or_default();

        let req = ChoiceRequest {
            prompt: format!(
                "Solar Eclipse on {} requires discarding 1 card to maintain. Pay?",
                sys_name
            ),
            options: vec!["Yes, pay 1 discard".into(), "No, let Eclipse lapse".into()],
            pick: 1,
        };
        let choice = self.ask_decide(player_idx, &req);
                self.drain_thinking(player_idx);

        if choice == 0 {
            let hand_clone = self.state.players[player_idx].hand.clone();
            let discards = self.ask_choose_discards(player_idx, &hand_clone, 1);
                self.drain_thinking(player_idx);
            if let Some(&id) = discards.first() {
                if let Some(card) = self.state.players[player_idx].take_from_hand(id) {
                    emit!(self, format!(
                        "{} discards {} to maintain Solar Eclipse on {}.",
                        self.state.players[player_idx].name, card, sys_name
                    ));
                    self.state.deck.push_bottom(card);
                }
            }
        } else {
            if let Some(omen_id) =
                self.state.players[player_idx].star_chart.detach_solar_eclipse(star_id)
            {
                if let Some(c) = self.state.reconstruct(omen_id) {
                    self.state.deck.push_bottom(c);
                }
                emit!(self, format!("{} lets Solar Eclipse on {} lapse.", self.state.players[player_idx].name, sys_name));
            }
        }
    }

    fn run_draw_phase(&mut self, idx: usize) {
        match self.state.deck.draw() {
            Some(card) => {
                emit!(self, format!("{} draws a card.", self.state.players[idx].name));
                self.state.players[idx].hand.push(card);
            }
            None => {
                emit!(self, "Deck is empty - no card drawn.".into());
                if self.state.deck_exhausted_at_turn.is_none() {
                    self.state.deck_exhausted_at_turn = Some(self.state.turn_number);
                    emit!(self, "Deck exhausted. Hand limits will shrink by 1 each round.".into());
                }
            }
        }
    }

    fn run_action_phase(&mut self, idx: usize) -> Option<GameResult> {
        if self.state.players[idx].skip_next_action {
            self.state.players[idx].skip_next_action = false;
            emit!(self, format!(
                "{} skips their Action Phase (Black Hole effect).",
                self.state.players[idx].name
            ));
            return None;
        }

        let mut actions_taken = 0usize;
        loop {
            if self.state.players[idx].hand.len() <= self.hand_limit() {
                break;
            }
            // Safety cap: a player whose choices are net-card-positive (e.g.
            // repeated Protostar draws) can stay above the hand limit forever
            // once the discard reshuffles. No legitimate turn comes anywhere
            // near 50 actions; force the phase to end instead of looping.
            if actions_taken >= 50 {
                emit!(self, format!(
                    "{} action phase force-ended (50-action cap).",
                    self.state.players[idx].name
                ));
                break;
            }
            // Decision-count watchdog checkpoint - see the `max_decisions`
            // field doc. Catches cost racked up by nested decide()/
            // pick_targets() calls inside prior actions' effects, not just
            // the action count itself.
            if let Some(result) = self.decision_budget_exceeded() {
                emit!(self, format!(
                    "{} action phase truncated (decision budget exceeded).",
                    self.state.players[idx].name
                ));
                return Some(result);
            }

            let legal = self.compute_legal_actions(idx);
            let view = self.build_view(idx);
            self.notify_thinking(idx);
            self.tick_decision();
            let choice = self.players[idx].choose_action_with_state(&view, &self.state, &legal);
                self.drain_thinking(idx);
            let action = legal[choice.min(legal.len() - 1)].clone();

            if matches!(action, Action::Pass) {
                emit!(self, format!("{} passes.", self.state.players[idx].name));
                self.send_snapshot(None);
                break;
            }

            if let Some(result) = self.execute_action(action, idx) {
                return Some(result);
            }
            self.send_snapshot(None);
            actions_taken += 1;
        }

        None
    }

    fn run_end_phase(&mut self, idx: usize) -> Option<GameResult> {
        let limit = self.hand_limit();
        let excess = self.state.players[idx].hand.len().saturating_sub(limit);
        if excess > 0 {
            if limit < 5 {
                emit!(self, format!(
                    "Hand limit is now {} (deck exhaustion pressure).",
                    limit
                ));
            }
            let hand_clone = self.state.players[idx].hand.clone();
            let to_discard = self.ask_choose_discards(idx, &hand_clone, excess);
            self.drain_thinking(idx);
            for id in to_discard {
                if let Some(card) = self.state.players[idx].take_from_hand(id) {
                    emit!(self, format!("{} discards {}.", self.state.players[idx].name, card));
                    self.state.deck.push_bottom(card);
                }
            }
            // Discard any remaining excess (fallback if player returned wrong count)
            while self.state.players[idx].hand.len() > limit {
                let card = self.state.players[idx].hand.pop().unwrap();
                self.state.deck.push_bottom(card);
            }
        }

        // Astronomical Domination victory check
        let complete = self.state.players[idx].star_chart.complete_count();
        if complete >= 3 {
            return Some(GameResult {
                winner_idx: idx,
                winner_name: self.state.players[idx].name.clone(),
                condition: WinCondition::AstronomicalDomination,
                turns_elapsed: self.state.turn_number,
            });
        }

        None
    }

    // ── Action execution ──────────────────────────────────────────────────────

    fn execute_action(&mut self, action: Action, player_idx: usize) -> Option<GameResult> {
        match action {
            Action::Pass => None,

            Action::PlayStar(card_id) => {
                let card = self.state.players[player_idx].take_from_hand(card_id)?;
                self.state.record_play(player_idx, card.kind.clone());
                let (name, class) = match &card.kind {
                    CardKind::Star(n) => (*n, n.class()?),
                    _ => return None,
                };
                emit!(self, format!(
                    "{} places {} (Class {}) in Star Chart.",
                    self.state.players[player_idx].name, name, class
                ));
                self.state.players[player_idx].star_chart.place_star(card_id, name, class);
                None
            }

            Action::PlayDiscovery { card: card_id, onto_star } => {
                let card = self.state.players[player_idx].take_from_hand(card_id)?;
                self.state.record_play(player_idx, card.kind.clone());
                let disc_name = match card.kind {
                    CardKind::Discovery(n) => n,
                    _ => return None,
                };
                let star_name = self.state.players[player_idx]
                    .star_chart
                    .system_for_star(onto_star)
                    .map(|s| s.star_name)?;
                self.state.players[player_idx]
                    .star_chart
                    .attach_discovery(onto_star, card_id, disc_name);
                emit!(self, format!(
                    "{} attaches {} to {}.",
                    self.state.players[player_idx].name, disc_name, star_name
                ));
                None
            }

            Action::PlaySolarSystem { star, discoveries } => {
                let star_card = self.state.players[player_idx].take_from_hand(star)?;
                self.state.record_play(player_idx, star_card.kind.clone());
                let (star_name, class) = match &star_card.kind {
                    CardKind::Star(n) => (*n, n.class()?),
                    _ => return None,
                };
                self.state.players[player_idx].star_chart.place_star(star, star_name, class);
                let mut disc_names = Vec::new();
                for disc_id in &discoveries {
                    if let Some(card) = self.state.players[player_idx].take_from_hand(*disc_id) {
                        self.state.record_play(player_idx, card.kind.clone());
                        if let CardKind::Discovery(dn) = card.kind {
                            self.state.players[player_idx]
                                .star_chart
                                .attach_discovery(star, *disc_id, dn);
                            disc_names.push(dn.to_string());
                        }
                    }
                }
                emit!(self, format!(
                    "{} completes a Solar System: {} with [{}].",
                    self.state.players[player_idx].name,
                    star_name,
                    disc_names.join(", ")
                ));
                None
            }

            Action::PlayOmen { card: card_id, attachment } => {
                let card = self.state.players[player_idx].take_from_hand(card_id)?;
                self.state.record_play(player_idx, card.kind.clone());
                emit!(self, format!("{} plays {} into Chain.", self.state.players[player_idx].name, card));
                let link = ChainLink::new(card, player_idx, vec![]);
                let mut chain = Chain::new(link);
                if let Some(result) = self.run_chain_responses(&mut chain, player_idx) {
                    self.flush_chain(chain);
                    return Some(result);
                }
                // Resolve LIFO; omens attach on resolution
                self.resolve_chain_omens(chain, player_idx, &attachment)
            }

            Action::PlayFlare(card_id) => {
                let card = self.state.players[player_idx].take_from_hand(card_id)?;
                self.state.record_play(player_idx, card.kind.clone());
                emit!(self, format!("{} plays {} into Chain.", self.state.players[player_idx].name, card));
                let link = ChainLink::new(card, player_idx, vec![]);
                let mut chain = Chain::new(link);
                if let Some(result) = self.run_chain_responses(&mut chain, player_idx) {
                    self.flush_chain(chain);
                    return Some(result);
                }
                self.resolve_chain(chain)
            }

            Action::PlayFracture(card_id) => {
                let card = self.state.players[player_idx].take_from_hand(card_id)?;
                self.state.record_play(player_idx, card.kind.clone());
                emit!(self, format!("{} plays {} into Chain.", self.state.players[player_idx].name, card));
                let link = ChainLink::new(card, player_idx, vec![]);
                let mut chain = Chain::new(link);
                if let Some(result) = self.run_chain_responses(&mut chain, player_idx) {
                    self.flush_chain(chain);
                    return Some(result);
                }
                self.resolve_chain(chain)
            }

            Action::PlaySupernovas { fusion, gravity, gas } => {
                let fc = self.state.players[player_idx].take_from_hand(fusion)?;
                self.state.record_play(player_idx, fc.kind.clone());
                let gc = self.state.players[player_idx].take_from_hand(gravity)?;
                self.state.record_play(player_idx, gc.kind.clone());
                let gac = self.state.players[player_idx].take_from_hand(gas)?;
                self.state.record_play(player_idx, gac.kind.clone());
                emit!(self, format!(
                    "{} plays all three Supernovas - Cosmic Cataclysm attempt!",
                    self.state.players[player_idx].name
                ));
                let mut chain = Chain::new(ChainLink::new(fc, player_idx, vec![]));
                chain.push(ChainLink::new(gc, player_idx, vec![]));
                chain.push(ChainLink::new(gac, player_idx, vec![]));
                // Opponents may respond with Cosmic Denial
                if let Some(result) = self.run_chain_responses(&mut chain, player_idx) {
                    self.flush_chain(chain);
                    return Some(result);
                }
                self.resolve_chain(chain)
            }

            Action::PlayProtostar { card: card_id, choice } => {
                let card = self.state.players[player_idx].take_from_hand(card_id)?;
                self.state.record_play(player_idx, card.kind.clone());
                emit!(self, format!("{} plays Protostar.", self.state.players[player_idx].name));
                // Protostar shuffles itself back into deck first
                self.state.deck.shuffle_in(card);
                emit!(self, "  Protostar shuffled into deck.".into());
                match choice {
                    ProtostarChoice::DrawTwo => {
                        self.draw_n(player_idx, 2);
                    }
                    ProtostarChoice::SearchWhiteDwarf => {
                        let found = self.state.deck.search_and_remove(|c| {
                            matches!(&c.kind, CardKind::Star(StarName::WhiteDwarf))
                        });
                        if let Some(wd) = found {
                            let wd_id = wd.id;
                            emit!(self, format!(
                                "  {} finds White Dwarf - placed directly in Star Chart.",
                                self.state.players[player_idx].name
                            ));
                            self.state.players[player_idx]
                                .star_chart
                                .place_star(wd_id, StarName::WhiteDwarf, 3);
                            self.state.deck.shuffle();
                        } else {
                            emit!(self, "  No White Dwarf in deck - drawing 2 instead.".into());
                            self.state.deck.shuffle();
                            self.draw_n(player_idx, 2);
                        }
                    }
                }
                None
            }
        }
    }

    // ── Chain ─────────────────────────────────────────────────────────────────

    /// Collect reactive responses to an open chain, starting a NEW response
    /// window after `last_adder`'s addition. Used by every live call site in
    /// `execute_action`. Returns `Some(result)` only if the decision-count
    /// watchdog truncated the game mid-collection (see `max_decisions`); the
    /// caller must flush the chain and propagate the result exactly like a
    /// `max_turns` timeout.
    fn run_chain_responses(&mut self, chain: &mut Chain, last_adder: usize) -> Option<GameResult> {
        let n = self.state.players.len();
        let start = (last_adder + 1) % n;
        self.run_chain_responses_from(chain, start, 0)
    }

    /// Core chain-response collection loop, resumable from an explicit
    /// `(start, consecutive_passes)` cursor - see `ChainResumeCursor`.
    /// After each addition, each of the other (n-1) players gets one chance to respond in
    /// clockwise order starting from the player left of the adder. If all pass (or have no
    /// legal responses), the chain closes. The adder is not offered their own addition.
    fn run_chain_responses_from(
        &mut self,
        chain: &mut Chain,
        start: usize,
        consecutive_passes: usize,
    ) -> Option<GameResult> {
        let n = self.state.players.len();
        let mut start = start;
        let mut consecutive_passes = consecutive_passes;

        while consecutive_passes < n - 1 {
            // Decision-count watchdog checkpoint - see the `max_decisions`
            // field doc. This loop has no other bound: in principle every
            // player could keep having a legal reactive card to play,
            // extending the window indefinitely (bounded in practice only by
            // finite hand sizes, which is exactly the kind of implicit bound
            // this watchdog exists to not have to rely on).
            if let Some(result) = self.decision_budget_exceeded() {
                emit!(self, "Chain response collection truncated (decision budget exceeded).".into());
                return Some(result);
            }

            let responder = (start + consecutive_passes) % n;
            let legal = self.compute_chain_responses(responder, chain);

            if legal.is_empty() {
                consecutive_passes += 1;
                continue;
            }

            let view = self.build_view(responder);
            let chain_view = ChainView::from(&*chain);
            self.notify_thinking(responder);
            let cursor = ChainResumeCursor { start, consecutive_passes };
            self.tick_decision();
            let resp = self.players[responder]
                .respond_to_chain_with_state(&view, &self.state, &chain_view, &*chain, cursor, &legal);
            self.drain_thinking(responder);

            match resp {
                Some(choice) => {
                    let chosen = &legal[choice.min(legal.len() - 1)];
                    if let Some(card) = self.state.players[responder].take_from_hand(chosen.card) {
                        emit!(self, format!(
                            "  {} responds: {}",
                            self.state.players[responder].name, card
                        ));
                        // Chain responses were never in play_history before -
                        // a real gap given how much reactive-card timing
                        // turned out to matter (see ASTRA7.md §1.1).
                        self.state.record_play(responder, card.kind.clone());
                        chain.push(ChainLink::new(card, responder, chosen.targets.clone()));
                        self.send_snapshot(Some(chain));
                        start = (responder + 1) % n;
                        consecutive_passes = 0;
                    }
                }
                None => {
                    consecutive_passes += 1;
                }
            }
        }
        None
    }

    /// Simulation-only entry point: continue chain-response resolution on a
    /// CLONED engine from the TRUE resume cursor the live game had reached
    /// (see `ChainResumeCursor`), then resolve the chain. Never called from
    /// `run()` - used exclusively by search-based CPU players (see
    /// `engine/src/cpu.rs`) to score candidate chain responses by their
    /// actual resolved effect instead of a static heuristic.
    ///
    /// Resumes with the EXACT continuation semantics the real game would
    /// use - no approximation: `cursor` carries the same `(start,
    /// consecutive_passes)` position `run_chain_responses_from` had reached
    /// live, so players who already passed in this window before the
    /// caller's turn are correctly not re-asked. (Previously this
    /// reconstructed the window from scratch assuming nobody had responded
    /// yet, which could ask 0-2 extra players a "will you respond" question
    /// the true continuation wouldn't have reached - see git history for the
    /// old approximation if useful context.) Only ever runs on an
    /// already-cloned GameEngine (see `simulate_chain_response` in cpu.rs) -
    /// never mutates a live game. Bounded by `max_decisions` like any other
    /// engine path - callers should set it, since this path never enters
    /// `run()`'s main loop and so gets none of `max_turns`'/the 50-action
    /// cap's protection otherwise.
    pub fn simulate_finish_chain(&mut self, mut chain: Chain, cursor: ChainResumeCursor) -> Option<GameResult> {
        if let Some(result) = self.run_chain_responses_from(&mut chain, cursor.start, cursor.consecutive_passes) {
            self.flush_chain(chain);
            return Some(result);
        }
        self.resolve_chain(chain)
    }

    /// Resolve a closed chain in LIFO order.
    fn resolve_chain(&mut self, mut chain: Chain) -> Option<GameResult> {
        emit!(self, "Chain resolves (LIFO):".into());

        for raw_idx in chain.resolution_order() {
            if chain.links[raw_idx].negated {
                emit!(self, format!("  [negated] {}", chain.links[raw_idx].card));
                continue;
            }

            let card = chain.links[raw_idx].card.clone();
            let actor = chain.links[raw_idx].player_idx;
            emit!(self, format!("  Resolving {}", card));

            // Cosmic Denial handled here to allow mutating chain
            if matches!(&card.kind, CardKind::Flare(FlareName::CosmicDenial)) {
                // Only deniable links that come before this one in the chain (lower indices)
                let deniable: Vec<usize> = chain
                    .deniable_links()
                    .into_iter()
                    .filter(|&i| i < raw_idx)
                    .collect();
                if deniable.is_empty() {
                    emit!(self, "  Cosmic Denial: no valid targets.".into());
                } else {
                    let opts: Vec<String> = deniable
                        .iter()
                        .map(|&i| format!("CL{}: {}", i + 1, chain.links[i].card))
                        .collect();
                    let req = ChoiceRequest {
                        prompt: "Cosmic Denial: choose a Flare or Fracture to negate.".into(),
                        options: opts,
                        pick: 1,
                    };
                    let choice = self.ask_decide(actor, &req);
                self.drain_thinking(actor);
                    let target = deniable[choice.min(deniable.len() - 1)];
                    chain.negate(target);
                    let negated_card_name = chain.links[target].card.to_string();
                    emit!(self, format!(
                        "  Cosmic Denial negates CL{}: {}",
                        target + 1,
                        negated_card_name
                    ));
                }
                continue;
            }

            // Omen links that are reactive: attach them
            if let CardKind::Omen(omen_name) = &card.kind {
                let omen_name = *omen_name;
                let omen_id = card.id;
                // The chain link should carry attachment info in targets, but for now
                // we ask the acting player where they want to attach
                self.attach_reactive_omen(omen_id, omen_name, actor);
                continue;
            }

            // Regular Flare / Fracture effects
            if let Some(result) = self.apply_card_effect(card, actor) {
                self.flush_chain(chain);
                return Some(result);
            }
        }

        // Cosmic Cataclysm check (Supernovas win after full resolution)
        if chain.has_winning_supernovas() {
            let winner_idx = chain
                .links
                .iter()
                .find(|l| l.card.is_supernova() && !l.negated)
                .map(|l| l.player_idx)
                .unwrap_or(self.state.active_player_idx);
            self.flush_chain(chain);
            emit!(self, format!(
                "{} wins via Cosmic Cataclysm!",
                self.state.players[winner_idx].name
            ));
            return Some(GameResult {
                winner_idx,
                winner_name: self.state.players[winner_idx].name.clone(),
                condition: WinCondition::CosmicCataclysm,
                turns_elapsed: self.state.turn_number,
            });
        }

        self.flush_chain(chain);
        None
    }

    /// Resolve a chain that was opened by an Omen play (the caster's omen attaches on resolution).
    fn resolve_chain_omens(
        &mut self,
        mut chain: Chain,
        caster_idx: usize,
        attachment: &OmenAttachment,
    ) -> Option<GameResult> {
        emit!(self, "Chain resolves (LIFO):".into());

        for raw_idx in chain.resolution_order() {
            if chain.links[raw_idx].negated {
                emit!(self, format!("  [negated] {}", chain.links[raw_idx].card));
                // Negated omen - put on deck
                let oid = chain.links[raw_idx].card.id;
                if let Some(c) = self.state.reconstruct(oid) {
                    self.state.deck.push_bottom(c);
                }
                continue;
            }

            let card = chain.links[raw_idx].card.clone();
            let actor = chain.links[raw_idx].player_idx;

            if actor == caster_idx && matches!(card.kind, CardKind::Omen(_)) {
                emit!(self, format!("  {} attaches.", card));
                self.apply_omen_attachment(card.id, actor, attachment);
                continue;
            }

            if matches!(&card.kind, CardKind::Omen(_)) {
                let omen_name = match card.kind { CardKind::Omen(n) => n, _ => unreachable!() };
                self.attach_reactive_omen(card.id, omen_name, actor);
                continue;
            }

            if matches!(&card.kind, CardKind::Flare(FlareName::CosmicDenial)) {
                let deniable: Vec<usize> = chain
                    .deniable_links()
                    .into_iter()
                    .filter(|&i| i < raw_idx)
                    .collect();
                if !deniable.is_empty() {
                    let opts: Vec<String> = deniable
                        .iter()
                        .map(|&i| format!("CL{}: {}", i + 1, chain.links[i].card))
                        .collect();
                    let req = ChoiceRequest {
                        prompt: "Cosmic Denial: choose a target to negate.".into(),
                        options: opts,
                        pick: 1,
                    };
                    let choice = self.ask_decide(actor, &req);
                self.drain_thinking(actor);
                    chain.negate(deniable[choice.min(deniable.len() - 1)]);
                }
                continue;
            }

            if let Some(result) = self.apply_card_effect(card, actor) {
                self.flush_chain(chain);
                return Some(result);
            }
        }

        self.flush_chain(chain);
        None
    }

    fn apply_omen_attachment(&mut self, omen_id: CardId, player_idx: usize, attachment: &OmenAttachment) {
        match attachment {
            OmenAttachment::SolarEclipse { system_star } => {
                if self.state.players[player_idx]
                    .star_chart
                    .attach_solar_eclipse(*system_star, omen_id)
                {
                    let name = self.state.players[player_idx]
                        .star_chart
                        .system_for_star(*system_star)
                        .map(|s| s.star_name.to_string())
                        .unwrap_or_default();
                    emit!(self, format!(
                        "{}'s Solar Eclipse protects {}.",
                        self.state.players[player_idx].name, name
                    ));
                } else {
                    // No valid target - return to deck
                    if let Some(c) = self.state.reconstruct(omen_id) {
                        self.state.deck.push_bottom(c);
                    }
                    emit!(self, "Solar Eclipse: no valid target - returned to deck.".into());
                }
            }
            OmenAttachment::PlanetOfLife { rocky_planet_disc } => {
                if self.state.players[player_idx]
                    .star_chart
                    .attach_planet_of_life(*rocky_planet_disc, omen_id)
                {
                    emit!(self, format!(
                        "{} attaches Planet of Life to a Rocky Planet.",
                        self.state.players[player_idx].name
                    ));
                } else {
                    if let Some(c) = self.state.reconstruct(omen_id) {
                        self.state.deck.push_bottom(c);
                    }
                    emit!(self, "Planet of Life: no valid target - returned to deck.".into());
                }
            }
        }
    }

    fn attach_reactive_omen(&mut self, omen_id: CardId, omen_name: OmenName, player_idx: usize) {
        let attachment = match omen_name {
            OmenName::SolarEclipse => {
                // Ask player which completed system to attach to
                let stars: Vec<CardId> = self.state.players[player_idx]
                    .star_chart
                    .complete_systems()
                    .iter()
                    .filter(|s| s.eclipse_id.is_none())
                    .map(|s| s.star_id)
                    .collect();
                if stars.is_empty() {
                    if let Some(c) = self.state.reconstruct(omen_id) {
                        self.state.deck.push_bottom(c);
                    }
                    emit!(self, "Solar Eclipse: no valid target - returned to deck.".into());
                    return;
                }
                let opts: Vec<String> = stars
                    .iter()
                    .filter_map(|&sid| {
                        self.state.players[player_idx]
                            .star_chart
                            .system_for_star(sid)
                            .map(|s| s.star_name.to_string())
                    })
                    .collect();
                let req = ChoiceRequest {
                    prompt: "Solar Eclipse: attach to which completed system?".into(),
                    options: opts,
                    pick: 1,
                };
                let choice = self.ask_decide(player_idx, &req);
                self.drain_thinking(player_idx);
                OmenAttachment::SolarEclipse { system_star: stars[choice.min(stars.len() - 1)] }
            }
            OmenName::PlanetOfLife => {
                let rockies: Vec<CardId> = self.state.players[player_idx]
                    .star_chart
                    .systems
                    .iter()
                    .flat_map(|s| s.rocky_planets())
                    .collect();
                if rockies.is_empty() {
                    if let Some(c) = self.state.reconstruct(omen_id) {
                        self.state.deck.push_bottom(c);
                    }
                    emit!(self, "Planet of Life: no Rocky Planet to attach to.".into());
                    return;
                }
                let req = ChoiceRequest {
                    prompt: "Planet of Life: attach to which Rocky Planet?".into(),
                    options: (0..rockies.len()).map(|i| format!("Rocky Planet #{}", i + 1)).collect(),
                    pick: 1,
                };
                let choice = self.ask_decide(player_idx, &req);
                self.drain_thinking(player_idx);
                OmenAttachment::PlanetOfLife {
                    rocky_planet_disc: rockies[choice.min(rockies.len() - 1)],
                }
            }
        };
        self.apply_omen_attachment(omen_id, player_idx, &attachment);
    }

    fn flush_chain(&mut self, chain: Chain) {
        for link in chain.links {
            // Attached omens are already tracked in star charts - only flush non-omen cards
            if !matches!(link.card.kind, CardKind::Omen(_)) {
                self.state.deck.push_bottom(link.card);
            }
            // Negated omens were already pushed to deck in resolve_chain_omens
        }
    }

    // ── Card effect resolution ────────────────────────────────────────────────

    fn apply_card_effect(&mut self, card: Card, actor: usize) -> Option<GameResult> {
        match card.kind {
            CardKind::Flare(f) => self.apply_flare(f, actor),
            CardKind::Fracture(f) => self.apply_fracture(f, actor),
            _ => None,
        }
    }

    fn apply_flare(&mut self, flare: FlareName, actor: usize) -> Option<GameResult> {
        match flare {
            FlareName::StellarAlignment => {
                emit!(self, format!("{} Stellar Alignment - draw 2.", self.state.players[actor].name));
                self.draw_n(actor, 2);
                self.state.stellar_alignment_this_turn = true;
                None
            }

            FlareName::CosmicDenial => {
                // Handled specially in resolve_chain; shouldn't reach here
                None
            }

            FlareName::Wormhole => {
                let opponents = self.opponent_indices(actor);
                if opponents.is_empty() {
                    return None;
                }
                let mode = self.ask_decide(
                    actor,
                    &ChoiceRequest {
                        prompt: "Wormhole - choose mode:".into(),
                        options: vec![
                            "Take 1 card blindly from an opponent's hand".into(),
                            "Take up to 2 Star/Discovery cards from an opponent's Star Chart".into(),
                        ],
                        pick: 1,
                    },
                );
                self.drain_thinking(actor);

                let opp_opts: Vec<String> =
                    opponents.iter().map(|&i| self.state.players[i].name.clone()).collect();
                let opp_choice = self.ask_decide(
                    actor,
                    &ChoiceRequest {
                        prompt: "Choose target opponent:".into(),
                        options: opp_opts,
                        pick: 1,
                    },
                );
                self.drain_thinking(actor);
                let target = opponents[opp_choice.min(opponents.len() - 1)];

                if mode == 0 {
                    self.wormhole_blind_steal(actor, target);
                } else {
                    self.wormhole_chart_steal(actor, target);
                }
                None
            }

            FlareName::Ignition => {
                let mode = self.ask_decide(
                    actor,
                    &ChoiceRequest {
                        prompt: "Ignition - choose mode:".into(),
                        options: vec![
                            "Destroy a Gas Giant in an opponent's Star Chart, then search for Yellow Dwarf".into(),
                            "Target opponent reveals hand; if they hold a Gas Giant, put it on bottom of deck, then search for Yellow Dwarf".into(),
                        ],
                        pick: 1,
                    },
                );
                self.drain_thinking(actor);

                let opponents = self.opponent_indices(actor);
                let opp_opts: Vec<String> =
                    opponents.iter().map(|&i| self.state.players[i].name.clone()).collect();
                let opp_choice = self.ask_decide(
                    actor,
                    &ChoiceRequest {
                        prompt: "Choose target opponent:".into(),
                        options: opp_opts,
                        pick: 1,
                    },
                );
                self.drain_thinking(actor);
                let target = opponents[opp_choice.min(opponents.len() - 1)];

                if mode == 0 {
                    if let Some((gid, star_name)) =
                        self.state.players[target].star_chart.take_gas_giant(false)
                    {
                        if let Some(c) = self.state.reconstruct(gid) {
                            emit!(self, format!(
                                "Ignition destroys Gas Giant from {}'s {} system.",
                                self.state.players[target].name, star_name
                            ));
                            self.state.deck.push_bottom(c);
                        }
                    } else {
                        emit!(self, "Ignition: no Gas Giant in target's Star Chart.".into());
                    }
                } else {
                    emit!(self, format!(
                        "Ignition: {} reveals hand.",
                        self.state.players[target].name
                    ));
                    let gas_id = self.state.players[target]
                        .hand
                        .iter()
                        .find(|c| matches!(&c.kind, CardKind::Discovery(DiscoveryName::GasGiant)))
                        .map(|c| c.id);
                    if let Some(gid) = gas_id {
                        if let Some(card) = self.state.players[target].take_from_hand(gid) {
                            emit!(self, format!(
                                "Ignition puts Gas Giant from {}'s hand on bottom of deck.",
                                self.state.players[target].name
                            ));
                            self.state.deck.push_bottom(card);
                        }
                    } else {
                        emit!(self, "Ignition: no Gas Giant in target's hand.".into());
                    }
                }

                // Both modes: search for Yellow Dwarf
                let yd = self.state.deck.search_and_remove(|c| {
                    matches!(&c.kind, CardKind::Star(StarName::YellowDwarf))
                });
                match yd {
                    Some(card) => {
                        emit!(self, format!(
                            "{} finds Yellow Dwarf via Ignition.",
                            self.state.players[actor].name
                        ));
                        self.state.players[actor].hand.push(card);
                    }
                    None => {
                        emit!(self, "No Yellow Dwarf in deck.".into());
                    }
                }
                self.state.deck.shuffle();
                None
            }

            FlareName::AstralProjection => {
                let opponents = self.opponent_indices(actor);
                let opp_opts: Vec<String> =
                    opponents.iter().map(|&i| self.state.players[i].name.clone()).collect();
                let choice = self.ask_decide(
                    actor,
                    &ChoiceRequest {
                        prompt: "Astral Projection: choose target opponent.".into(),
                        options: opp_opts,
                        pick: 1,
                    },
                );
                self.drain_thinking(actor);
                let target = opponents[choice.min(opponents.len() - 1)];
                self.state.players[target].astral_projected_turns = 2;
                emit!(self, format!(
                    "{} Astral Projection on {} - hand revealed for 2 of their turns.",
                    self.state.players[actor].name, self.state.players[target].name
                ));
                None
            }

            FlareName::SupernovaFusion
            | FlareName::SupernovaGravity
            | FlareName::SupernovaGas => {
                // Win check happens after full chain resolution in resolve_chain
                None
            }
        }
    }

    fn apply_fracture(&mut self, frac: FractureName, actor: usize) -> Option<GameResult> {
        match frac {
            FractureName::BlackHole => {
                emit!(self, format!(
                    "{} Black Hole - everyone shuffles hand and field into deck!",
                    self.state.players[actor].name
                ));
                let n = self.state.players.len();
                for i in 0..n {
                    // Collect hand
                    let hand: Vec<Card> = self.state.players[i].hand.drain(..).collect();
                    // Collect star/discovery cards (Omens survive per rules)
                    let wipe = self.state.players[i].star_chart.drain_for_board_wipe();
                    let field_cards: Vec<Card> = wipe
                        .to_deck
                        .iter()
                        .filter_map(|&id| self.state.reconstruct(id))
                        .collect();

                    self.state.deck.shuffle_in_many(hand);
                    self.state.deck.shuffle_in_many(field_cards);
                    // floating_eclipse_ids are already tracked in star_chart.floating_omens
                }
                // Caster skips next Action Phase
                self.state.players[actor].skip_next_action = true;
                self.state.deck.shuffle();
                None
            }

            FractureName::Athena => {
                let opponents = self.opponent_indices(actor);
                let opp_opts: Vec<String> =
                    opponents.iter().map(|&i| self.state.players[i].name.clone()).collect();
                let mode = self.ask_decide(
                    actor,
                    &ChoiceRequest {
                        prompt: "Athena - choose mode:".into(),
                        options: vec![
                            "Destroy a Discovery in an opponent's Star Chart".into(),
                            "Return a Gas Giant from an opponent's Star Chart to their hand".into(),
                        ],
                        pick: 1,
                    },
                );
                self.drain_thinking(actor);
                let opp_choice = self.ask_decide(
                    actor,
                    &ChoiceRequest {
                        prompt: "Choose target opponent:".into(),
                        options: opp_opts,
                        pick: 1,
                    },
                );
                self.drain_thinking(actor);
                let target = opponents[opp_choice.min(opponents.len() - 1)];

                if mode == 0 {
                    let opts = self.disc_target_options(target);
                    if opts.is_empty() {
                        emit!(self, format!(
                            "Athena: {} has no valid Discovery targets.",
                            self.state.players[target].name
                        ));
                        return None;
                    }
                    let picks = self.ask_pick_targets(actor, "Choose a Discovery to destroy:", &opts, 1);
                self.drain_thinking(actor);
                    if let Some(&pick) = picks.first() {
                        let pick = pick.min(opts.len() - 1);
                        if let Target::StarChartCard { card_id, .. } = opts[pick].target.clone() {
                            if let Some((_sn, dn, freed_pol_id)) = self.state.players[target]
                                .star_chart
                                .remove_discovery(card_id, false)
                            {
                                if let Some(c) = self.state.reconstruct(card_id) {
                                    emit!(self, format!(
                                        "Athena destroys {} from {}'s Star Chart.",
                                        dn, self.state.players[target].name
                                    ));
                                    self.state.deck.push_bottom(c);
                                }
                                // The destroyed Discovery may have had a Planet
                                // of Life omen attached - it must return to the
                                // deck too, or it's lost from the game entirely.
                                if let Some(pol_id) = freed_pol_id {
                                    if let Some(c) = self.state.reconstruct(pol_id) {
                                        self.state.deck.push_bottom(c);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if let Some((gid, sn)) =
                        self.state.players[target].star_chart.take_gas_giant(false)
                    {
                        if let Some(c) = self.state.reconstruct(gid) {
                            emit!(self, format!(
                                "Athena returns Gas Giant from {}'s {} system to their hand.",
                                self.state.players[target].name, sn
                            ));
                            self.state.players[target].hand.push(c);
                        }
                    } else {
                        emit!(self, "Athena: no Gas Giant in target's Star Chart.".into());
                    }
                }
                None
            }

            FractureName::Pulsar => {
                emit!(self, format!(
                    "{} Pulsar - shuffles own hand and field into deck, draws up to 7.",
                    self.state.players[actor].name
                ));
                let hand: Vec<Card> = self.state.players[actor].hand.drain(..).collect();
                let wipe = self.state.players[actor].star_chart.drain_for_board_wipe();
                let field_cards: Vec<Card> = wipe
                    .to_deck
                    .iter()
                    .filter_map(|&id| self.state.reconstruct(id))
                    .collect();
                self.state.deck.shuffle_in_many(hand);
                self.state.deck.shuffle_in_many(field_cards);
                self.state.deck.shuffle();
                self.draw_n(actor, 7);
                None
            }

            FractureName::DarkMatterVoid => {
                let choice = self.ask_decide(
                    actor,
                    &ChoiceRequest {
                        prompt: "Dark Matter Void: choose a Star Class to restrict for opponents.".into(),
                        options: vec![
                            "Class 2 (Red Dwarf / Brown Dwarf)".into(),
                            "Class 3 (White Dwarf)".into(),
                            "Class 4 (Yellow Dwarf)".into(),
                            "Class 5 (Red Giant)".into(),
                            "Class 6 (Red Supergiant)".into(),
                        ],
                        pick: 1,
                    },
                );
                self.drain_thinking(actor);
                let restricted_class = (choice.min(4) as u8) + 2;
                emit!(self, format!(
                    "{} Dark Matter Void - Class {} Stars locked for opponents until {}'s next turn.",
                    self.state.players[actor].name,
                    restricted_class,
                    self.state.players[actor].name
                ));
                self.state.dark_matter_void = Some(DarkMatterVoid { restricted_class, caster_idx: actor });
                None
            }

            FractureName::QuantumEntanglementFailure => {
                let n = self.state.players.len();
                let all_opts: Vec<String> =
                    (0..n).map(|i| self.state.players[i].name.clone()).collect();

                let p1 = self.ask_decide(
                    actor,
                    &ChoiceRequest {
                        prompt: "QEF: choose first player.".into(),
                        options: all_opts.clone(),
                        pick: 1,
                    },
                );
                self.drain_thinking(actor);
                let p1 = p1.min(n - 1);

                let remaining_opts: Vec<(usize, String)> = (0..n)
                    .filter(|&i| i != p1)
                    .map(|i| (i, self.state.players[i].name.clone()))
                    .collect();
                let p2_local = self.ask_decide(
                    actor,
                    &ChoiceRequest {
                        prompt: "QEF: choose second player.".into(),
                        options: remaining_opts.iter().map(|(_, n)| n.clone()).collect(),
                        pick: 1,
                    },
                );
                self.drain_thinking(actor);
                let p2 = remaining_opts[p2_local.min(remaining_opts.len() - 1)].0;

                let pick1 = if !self.state.players[p1].hand.is_empty() {
                    let opts: Vec<String> =
                        self.state.players[p1].hand.iter().map(|c| c.to_string()).collect();
                    let c = self.ask_decide(
                        p1,
                        &ChoiceRequest {
                            prompt: format!(
                                "{}: secretly choose a card to swap.",
                                self.state.players[p1].name
                            ),
                            options: opts,
                            pick: 1,
                        },
                    );
                self.drain_thinking(p1);
                    Some(self.state.players[p1].hand[c.min(self.state.players[p1].hand.len() - 1)].id)
                } else {
                    None
                };

                let pick2 = if !self.state.players[p2].hand.is_empty() {
                    let opts: Vec<String> =
                        self.state.players[p2].hand.iter().map(|c| c.to_string()).collect();
                    let c = self.ask_decide(
                        p2,
                        &ChoiceRequest {
                            prompt: format!(
                                "{}: secretly choose a card to swap.",
                                self.state.players[p2].name
                            ),
                            options: opts,
                            pick: 1,
                        },
                    );
                self.drain_thinking(p2);
                    Some(self.state.players[p2].hand[c.min(self.state.players[p2].hand.len() - 1)].id)
                } else {
                    None
                };

                if let (Some(id1), Some(id2)) = (pick1, pick2) {
                    let c1 = self.state.players[p1].take_from_hand(id1);
                    let c2 = self.state.players[p2].take_from_hand(id2);
                    if let (Some(c1), Some(c2)) = (c1, c2) {
                        emit!(self, format!(
                            "QEF: {} and {} swap cards.",
                            self.state.players[p1].name, self.state.players[p2].name
                        ));
                        self.state.players[p1].hand.push(c2);
                        self.state.players[p2].hand.push(c1);
                    }
                } else {
                    emit!(self, "QEF: one or both players have no cards - no swap.".into());
                }
                None
            }

            FractureName::StellarNurseryCollapse => {
                emit!(self, format!(
                    "{} Stellar Nursery Collapse - shuffles deck, privately views top 5.",
                    self.state.players[actor].name
                ));
                self.state.deck.shuffle();
                let top5: Vec<Card> = self.state.deck.peek_top(5).to_vec();
                let n = top5.len();
                if n == 0 {
                    return None;
                }
                // Player picks the order to place them on the bottom
                // Simplified: player makes n sequential picks to specify order
                let mut ordered_ids: Vec<CardId> = Vec::with_capacity(n);
                let mut remaining: Vec<(usize, &Card)> = top5.iter().enumerate().collect();
                for pick_num in 0..n {
                    let opts: Vec<String> = remaining.iter().map(|(_, c)| c.to_string()).collect();
                    let choice = self.ask_decide(
                        actor,
                        &ChoiceRequest {
                            prompt: format!(
                                "Stellar Nursery Collapse: pick card #{} to place at the bottom (first pick = deepest).",
                                pick_num + 1
                            ),
                            options: opts,
                            pick: 1,
                        },
                    );
                self.drain_thinking(actor);
                    let picked = remaining.remove(choice.min(remaining.len() - 1));
                    ordered_ids.push(picked.1.id);
                }
                self.state.deck.nursery_collapse(ordered_ids);
                None
            }

            FractureName::Shockwave => {
                let choice = self.ask_decide(
                    actor,
                    &ChoiceRequest {
                        prompt: "Shockwave: which Supernova to search?".into(),
                        options: vec![
                            "Supernova - Fusion".into(),
                            "Supernova - Gravity".into(),
                            "Supernova - Gas".into(),
                        ],
                        pick: 1,
                    },
                );
                self.drain_thinking(actor);
                let target_flare = match choice.min(2) {
                    0 => FlareName::SupernovaFusion,
                    1 => FlareName::SupernovaGravity,
                    _ => FlareName::SupernovaGas,
                };
                let found = self.state.deck.search_and_remove(|c| {
                    matches!(&c.kind, CardKind::Flare(f) if *f == target_flare)
                });
                match found {
                    Some(card) => {
                        emit!(self, format!(
                            "{} Shockwave finds {}.",
                            self.state.players[actor].name, card
                        ));
                        self.state.players[actor].hand.push(card);
                    }
                    None => {
                        emit!(self, "Shockwave: Supernova not found in deck.".into());
                    }
                }
                self.state.deck.shuffle();
                None
            }
        }
    }

    // ── Wormhole helpers ──────────────────────────────────────────────────────

    fn wormhole_blind_steal(&mut self, actor: usize, target: usize) {
        if self.state.players[target].hand.is_empty() {
            emit!(self, format!("Wormhole: {} has no cards.", self.state.players[target].name));
            return;
        }
        let pos = (crate::random::random::<usize>()) % self.state.players[target].hand.len();
        let stolen = self.state.players[target].hand.remove(pos);
        emit!(self, format!(
            "{} blindly steals a card from {}.",
            self.state.players[actor].name, self.state.players[target].name
        ));
        self.state.players[actor].hand.push(stolen);
    }

    fn wormhole_chart_steal(&mut self, actor: usize, target: usize) {
        let opts = self.chart_target_options(target, false);
        if opts.is_empty() {
            emit!(self, format!(
                "Wormhole: {}'s Star Chart has no valid targets.",
                self.state.players[target].name
            ));
            return;
        }
        let picks = self.ask_pick_targets(
            actor,
            "Wormhole: choose up to 2 Star or Discovery cards to take (Wormhole host rules apply).",
            &opts,
            2,
        );
                self.drain_thinking(actor);
        let picked_targets: Vec<Target> = picks
            .iter()
            .filter_map(|&i| opts.get(i.min(opts.len() - 1)).map(|o| o.target.clone()))
            .collect();

        for t in picked_targets {
            if let Target::StarChartCard { card_id, player_idx: tidx } = t {
                // Attempt to take as a discovery
                if let Some((_sn, dn, freed_pol_id)) =
                    self.state.players[tidx].star_chart.remove_discovery(card_id, false)
                {
                    // Try to attach to an incomplete system; otherwise add to hand
                    let inc_star = self.state.players[actor]
                        .star_chart
                        .incomplete_systems()
                        .first()
                        .map(|s| s.star_id);
                    match inc_star {
                        Some(sid) => {
                            self.state.players[actor]
                                .star_chart
                                .attach_discovery(sid, card_id, dn);
                            emit!(self, format!(
                                "Wormhole: {} takes {} - attached to existing system.",
                                self.state.players[actor].name, dn
                            ));
                        }
                        None => {
                            if let Some(c) = self.state.reconstruct(card_id) {
                                self.state.players[actor].hand.push(c);
                                emit!(self, format!(
                                    "Wormhole: {} takes {} - added to hand.",
                                    self.state.players[actor].name, dn
                                ));
                            }
                        }
                    }
                    // A Planet of Life omen attached to the stolen Discovery does
                    // not travel with it (rules: destroyed when its host Rocky
                    // Planet leaves) - but the physical card must still return
                    // to the deck rather than vanish from the game entirely.
                    if let Some(pol_id) = freed_pol_id {
                        if let Some(c) = self.state.reconstruct(pol_id) {
                            self.state.deck.push_bottom(c);
                        }
                    }
                }
                // Note: taking Stars requires all attached discoveries come with it;
                // full validation is deferred to the player's pick logic.
            }
        }
    }

    // ── Legal action computation ──────────────────────────────────────────────

    pub fn compute_legal_actions(&self, player_idx: usize) -> Vec<Action> {
        let mut actions = vec![Action::Pass];
        let p = &self.state.players[player_idx];

        if p.hand.len() < 6 {
            return actions;
        }

        // Dark Matter Void restricted class (does not apply to the caster)
        let restricted: Option<u8> = self.state.dark_matter_void.as_ref().filter(|d| d.caster_idx != player_idx).map(|d| d.restricted_class);

        // Supernova trio - requires at least 1 complete Solar System as prerequisite
        let has_f = p.hand.iter().any(|c| matches!(&c.kind, CardKind::Flare(FlareName::SupernovaFusion)));
        let has_g = p.hand.iter().any(|c| matches!(&c.kind, CardKind::Flare(FlareName::SupernovaGravity)));
        let has_s = p.hand.iter().any(|c| matches!(&c.kind, CardKind::Flare(FlareName::SupernovaGas)));
        let has_system = !p.star_chart.complete_systems().is_empty();
        if has_f && has_g && has_s && has_system {
            let fid = p.hand.iter().find(|c| matches!(&c.kind, CardKind::Flare(FlareName::SupernovaFusion))).unwrap().id;
            let gid = p.hand.iter().find(|c| matches!(&c.kind, CardKind::Flare(FlareName::SupernovaGravity))).unwrap().id;
            let sid = p.hand.iter().find(|c| matches!(&c.kind, CardKind::Flare(FlareName::SupernovaGas))).unwrap().id;
            actions.push(Action::PlaySupernovas { fusion: fid, gravity: gid, gas: sid });
        }

        for card in &p.hand {
            match &card.kind {
                CardKind::Star(StarName::Protostar) => {
                    actions.push(Action::PlayProtostar { card: card.id, choice: ProtostarChoice::DrawTwo });
                    actions.push(Action::PlayProtostar { card: card.id, choice: ProtostarChoice::SearchWhiteDwarf });
                }
                CardKind::Star(name) => {
                    if let Some(class) = name.class() {
                        if restricted.map_or(true, |r| r != class) {
                            actions.push(Action::PlayStar(card.id));
                        }
                    }
                }
                CardKind::Discovery(_) => {
                    for sys in p.star_chart.incomplete_iter() {
                        actions.push(Action::PlayDiscovery { card: card.id, onto_star: sys.star_id });
                    }
                }
                CardKind::Flare(f) if f.is_supernova() => { /* only as trio */ }
                CardKind::Flare(FlareName::StellarAlignment) if self.state.stellar_alignment_this_turn => { /* one per turn */ }
                CardKind::Flare(_) => {
                    actions.push(Action::PlayFlare(card.id));
                }
                CardKind::Omen(OmenName::SolarEclipse) => {
                    for sys in p.star_chart.complete_systems() {
                        if sys.eclipse_id.is_none() {
                            actions.push(Action::PlayOmen {
                                card: card.id,
                                attachment: OmenAttachment::SolarEclipse { system_star: sys.star_id },
                            });
                        }
                    }
                }
                CardKind::Omen(OmenName::PlanetOfLife) => {
                    for sys in &p.star_chart.systems {
                        for rocky_id in sys.rocky_planets() {
                            // Find if this rocky planet already has a Planet of Life
                            let already_attached = sys.discoveries.iter()
                                .any(|d| d.card_id == rocky_id && d.planet_of_life_id.is_some());
                            if !already_attached {
                                actions.push(Action::PlayOmen {
                                    card: card.id,
                                    attachment: OmenAttachment::PlanetOfLife { rocky_planet_disc: rocky_id },
                                });
                            }
                        }
                    }
                }
                CardKind::Fracture(_) => {
                    actions.push(Action::PlayFracture(card.id));
                }
            }
        }

        // Complete Solar System plays
        self.enumerate_solar_system_actions(player_idx, restricted, &mut actions);

        actions
    }

    fn enumerate_solar_system_actions(
        &self,
        player_idx: usize,
        restricted: Option<u8>,
        actions: &mut Vec<Action>,
    ) {
        let p = &self.state.players[player_idx];
        let stars: Vec<&Card> = p.hand.iter().filter(|c| {
            matches!(&c.kind, CardKind::Star(n) if n.class().is_some() && *n != StarName::Protostar)
                && !matches!(&c.kind, CardKind::Star(StarName::Protostar))
        }).collect();

        let disc_ids: Vec<CardId> = p
            .hand
            .iter()
            .filter(|c| matches!(&c.kind, CardKind::Discovery(_)))
            .map(|c| c.id)
            .collect();

        for star in stars {
            let class = match star.star_class() { Some(c) => c, None => continue };
            if restricted.map_or(false, |r| r == class) { continue; }
            if disc_ids.len() < class as usize { continue; }

            for_each_combination(&disc_ids, class as usize, |combo| {
                actions.push(Action::PlaySolarSystem { star: star.id, discoveries: combo.to_vec() });
            });
        }
    }

    fn compute_chain_responses(&self, player_idx: usize, chain: &Chain) -> Vec<ChainAction> {
        let p = &self.state.players[player_idx];
        let mut responses = Vec::new();

        for card in &p.hand {
            if !card.kind.can_enter_chain() { continue; }
            if card.is_supernova() { continue; }
            if matches!(&card.kind, CardKind::Flare(FlareName::CosmicDenial)) && chain.deniable_links().is_empty() {
                continue;
            }
            responses.push(ChainAction { card: card.id, targets: vec![], omen_attachment: None });
        }

        responses
    }

    // ── View building ─────────────────────────────────────────────────────────

    pub fn build_view(&self, for_player: usize) -> GameView {
        let me = &self.state.players[for_player];
        let opponents = (0..self.state.players.len())
            .filter(|&i| i != for_player)
            .map(|i| {
                let opp = &self.state.players[i];
                OpponentView {
                    idx: i,
                    name: opp.name.clone(),
                    hand_size: opp.hand.len(),
                    hand_revealed: if opp.astral_projected_turns > 0 {
                        opp.hand.clone()
                    } else {
                        vec![]
                    },
                    star_chart: opp.star_chart.clone(),
                }
            })
            .collect();

        // Batch simulations run silent with no observer, so `emit()` has
        // discarded every narrative log line. Avoid locking and scanning the
        // empty log on each simulated decision; this path is exercised many
        // times per training/search game. Preserve the observer case because
        // observer-backed games still collect log messages while silent.
        let history = if self.silent && self.observer.is_none() {
            Vec::new()
        } else {
            let log = self.log.lock().unwrap();
            let start = log.len().saturating_sub(30);
            log[start..].iter()
                .flat_map(|s| s.split('\n'))
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<String>>()
        };

        GameView {
            my_idx: for_player,
            my_name: me.name.clone(),
            my_hand: me.hand.clone(),
            my_star_chart: me.star_chart.clone(),
            opponents,
            deck_size: self.state.deck.len(),
            active_player_idx: self.state.active_player_idx,
            turn_number: self.state.turn_number,
            dark_matter_void: self.state.dark_matter_void.as_ref().map(|d| d.restricted_class),
            history,
            recent_plays: self.state.play_history.clone(),
            discard_pile_counts: self.state.played_counts,
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn draw_n(&mut self, player_idx: usize, n: usize) {
        let mut drawn = 0;
        for _ in 0..n {
            if let Some(card) = self.state.deck.draw() {
                self.state.players[player_idx].hand.push(card);
                drawn += 1;
            }
        }
        if drawn > 0 {
            emit!(self, format!(
                "  {} draws {}.",
                self.state.players[player_idx].name, drawn
            ));
        }
    }

    fn opponent_indices(&self, player_idx: usize) -> Vec<usize> {
        (0..self.state.players.len()).filter(|&i| i != player_idx).collect()
    }

    fn disc_target_options(&self, player_idx: usize) -> Vec<TargetOption> {
        let mut opts = Vec::new();
        for sys in &self.state.players[player_idx].star_chart.systems {
            if sys.is_eclipse_protected() { continue; }
            for disc in &sys.discoveries {
                opts.push(TargetOption::new(
                    format!("{} (in {} system of {})", disc.name, sys.star_name, self.state.players[player_idx].name),
                    Target::StarChartCard { player_idx, card_id: disc.card_id },
                ));
            }
        }
        opts
    }

    fn chart_target_options(&self, player_idx: usize, bypass_eclipse: bool) -> Vec<TargetOption> {
        let mut opts = Vec::new();
        for sys in &self.state.players[player_idx].star_chart.systems {
            if sys.is_eclipse_protected() && !bypass_eclipse { continue; }
            opts.push(TargetOption::new(
                format!("{} [Star]", sys.star_name),
                Target::StarChartCard { player_idx, card_id: sys.star_id },
            ));
            for disc in &sys.discoveries {
                opts.push(TargetOption::new(
                    format!("{} [Discovery]", disc.name),
                    Target::StarChartCard { player_idx, card_id: disc.card_id },
                ));
            }
        }
        opts
    }

    fn emit(&mut self, msg: String) {
        if let Some(tx) = &self.observer {
            let _ = tx.send(GameEvent::Log(msg.clone()));
            self.log.lock().unwrap().push(msg);
        } else if self.silent {
            // Batch simulation: discard log entirely to keep memory per-game minimal.
        } else {
            println!("{}", msg);
            self.log.lock().unwrap().push(msg);
        }
    }

    /// Write the full game log to `./games/<timestamp>_<winner>.log`.
    /// Creates the directory if it doesn't exist.
    pub fn save_log(&self, result: &GameResult) -> std::io::Result<()> {
        use std::fs;
        use std::io::Write;
        use std::time::{SystemTime, UNIX_EPOCH};

        fs::create_dir_all("games")?;

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let winner_slug: String = result.winner_name
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect();

        let cond = match result.condition {
            WinCondition::AstronomicalDomination => "astro",
            WinCondition::CosmicCataclysm => "cataclysm",
        };

        let path = format!("games/{}_{}_{}turns_{}.log", ts, winner_slug, result.turns_elapsed, cond);
        let mut file = fs::File::create(&path)?;

        // Header
        writeln!(file, "NOVACANA - Game Log")?;
        writeln!(file, "Winner : {} ({})", result.winner_name, cond)?;
        writeln!(file, "Turns  : {}", result.turns_elapsed)?;
        writeln!(file, "Players: {}", self.state.players.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(", "))?;
        writeln!(file, "{}", "─".repeat(60))?;

        for line in self.log.lock().unwrap().iter() {
            for subline in line.split('\n') {
                let t = subline.trim();
                if !t.is_empty() {
                    writeln!(file, "{}", t)?;
                }
            }
        }

        // Print path so the user knows where it went (suppressed in TUI mode via caller)
        eprintln!("Game log saved: {}", path);
        Ok(())
    }
}

// ── Combination enumeration ───────────────────────────────────────────────────

fn for_each_combination<T: Clone>(items: &[T], k: usize, mut visit: impl FnMut(&[T])) {
    fn visit_from<T: Clone>(
        items: &[T],
        start: usize,
        remaining: usize,
        chosen: &mut Vec<T>,
        visit: &mut impl FnMut(&[T]),
    ) {
        if remaining == 0 {
            visit(chosen);
            return;
        }
        if items.len().saturating_sub(start) < remaining {
            return;
        }
        let last = items.len() - remaining;
        for i in start..=last {
            chosen.push(items[i].clone());
            visit_from(items, i + 1, remaining - 1, chosen, visit);
            chosen.pop();
        }
    }

    if k > items.len() {
        return;
    }
    let mut chosen = Vec::with_capacity(k);
    visit_from(items, 0, k, &mut chosen, &mut visit);
}

// ── Simulation correctness tests ────────────────────────────────────────────
//
// These exist because a real bug (`run()` re-dealing 5 fresh cards to every
// player on every `from_state` resume) went undetected for weeks: it never
// panicked, it just silently corrupted every simulated/searched game. The
// tests below are the regression coverage for that bug plus its general
// class - state-fidelity drift across `from_state`, card accounting drift
// over the course of play, and (where it actually holds) determinism.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::AdvancedCpuPlayer;

    /// Minimal deterministic `Player` stand-in: always passes / picks the
    /// first option. Used where a test needs a turn's hand-size delta to be
    /// fully predictable (exactly the draw-phase card, offset by any forced
    /// end-of-turn discard) rather than whatever AdvCPU's heuristics decide.
    struct PassPlayer;

    impl Player for PassPlayer {
        fn name(&self) -> &str { "pass" }
        fn choose_action(&self, _view: &GameView, _legal: &[Action]) -> usize { 0 }
        fn respond_to_chain(
            &self,
            _view: &GameView,
            _chain: &ChainView,
            _legal: &[ChainAction],
        ) -> Option<usize> {
            None
        }
        fn decide(&self, _view: &GameView, _req: &ChoiceRequest) -> usize { 0 }
        fn pick_targets(
            &self,
            _view: &GameView,
            _prompt: &str,
            _options: &[TargetOption],
            count: usize,
        ) -> Vec<usize> {
            (0..count).collect()
        }
        fn choose_discards(&self, _view: &GameView, hand: &[Card], must_discard: usize) -> Vec<CardId> {
            hand.iter().take(must_discard).map(|c| c.id).collect()
        }
    }

    fn pass_players(n: usize) -> Vec<Box<dyn Player>> {
        (0..n).map(|_| Box::new(PassPlayer) as Box<dyn Player>).collect()
    }

    fn adv_players(n: usize) -> Vec<Box<dyn Player>> {
        (0..n)
            .map(|i| Box::new(AdvancedCpuPlayer::new(format!("Adv{}", i))) as Box<dyn Player>)
            .collect()
    }

    /// Every `CardId` the deck ever created (`registry.keys()`) must appear
    /// EXACTLY ONCE across: the deck, every player's hand, every player's
    /// star chart (stars + discoveries), and every player's attached omens
    /// (Solar Eclipse + Planet of Life + floating omens from board wipes).
    /// A card appearing 0 times means it vanished from the game; appearing
    /// 2+ times means it was duplicated - both are silent-corruption bugs
    /// this is meant to catch the moment they're introduced.
    fn assert_card_conservation(state: &GameState, when: &str) {
        let mut seen: HashMap<CardId, u32> = HashMap::new();
        for id in state.deck.all_ids() {
            *seen.entry(id).or_insert(0) += 1;
        }
        for p in &state.players {
            for c in &p.hand {
                *seen.entry(c.id).or_insert(0) += 1;
            }
            for id in p.star_chart.all_field_ids() {
                *seen.entry(id).or_insert(0) += 1;
            }
            for id in p.star_chart.attached_omen_ids() {
                *seen.entry(id).or_insert(0) += 1;
            }
        }
        for &id in state.registry.keys() {
            let count = seen.get(&id).copied().unwrap_or(0);
            assert_eq!(
                count, 1,
                "[{}] card {} appears {} time(s) across deck/hands/star charts (want exactly 1)",
                when, id, count
            );
        }
        assert_eq!(
            seen.len(), state.registry.len(),
            "[{}] found card ids that aren't in the registry at all", when
        );
    }

    // ── 1a. Direct regression test for the re-deal bug ──────────────────────

    #[test]
    fn fresh_game_deals_initial_hands() {
        let mut engine = GameEngine::new(Deck::build_classic(), pass_players(4));
        assert!(!engine.state.hands_dealt);
        assert!(engine.state.players.iter().all(|p| p.hand.is_empty()));

        engine.set_max_turns(1);
        let _ = engine.run();

        assert!(engine.state.hands_dealt);
        for p in &engine.state.players {
            // Dealt 5, then only the active player draws (+1) and - since
            // PassPlayer never plays a card - discards straight back down to
            // the 5-card limit (-1) at end of turn. Net: everyone at 5.
            assert_eq!(
                p.hand.len(), 5,
                "{} should hold exactly 5 cards after turn 1 of a fresh game", p.name
            );
        }
    }

    #[test]
    fn from_state_does_not_redeal_hands() {
        // The actual bug: `run()`, when resumed via `from_state` (the path
        // used by EVERY simulation/search caller - MCTS, CPU 1-ply search,
        // chain-response search), used to re-deal 5 fresh cards to every
        // player. PassPlayer makes the one extra turn this test plays fully
        // deterministic, so any deviation from "everyone still has 5 cards"
        // can only be explained by an unwanted deal.
        let mut engine = GameEngine::new(Deck::build_classic(), pass_players(4));
        engine.deal_initial_hands();
        let hand_sizes_before: Vec<usize> =
            engine.state.players.iter().map(|p| p.hand.len()).collect();
        assert_eq!(hand_sizes_before, vec![5, 5, 5, 5]);

        let mut resumed = GameEngine::from_state(engine.state.clone(), pass_players(4));
        // from_state itself must not have touched anything yet.
        assert_eq!(
            resumed.state.players.iter().map(|p| p.hand.len()).collect::<Vec<_>>(),
            hand_sizes_before
        );
        assert!(resumed.state.hands_dealt, "hands_dealt must survive Clone/from_state");

        let turn_before = resumed.state.turn_number;
        resumed.set_max_turns(turn_before); // process exactly one more turn, then stop
        let _ = resumed.run();

        let hand_sizes_after: Vec<usize> =
            resumed.state.players.iter().map(|p| p.hand.len()).collect();
        assert_eq!(
            hand_sizes_after, vec![5, 5, 5, 5],
            "hand sizes changed after resuming via from_state and running one turn - looks like a re-deal"
        );
    }

    // ── 1b. from_state state-fidelity (the general class of that bug) ───────

    #[test]
    fn from_state_preserves_state_fidelity() {
        let mut engine = GameEngine::new(Deck::build_classic(), adv_players(4));
        engine.set_silent();
        engine.set_max_turns(6);
        let _ = engine.run();
        let original = engine.state.clone();

        let resumed = GameEngine::from_state(original.clone(), adv_players(4));

        assert_eq!(
            resumed.state.players.iter().map(|p| p.hand.len()).collect::<Vec<_>>(),
            original.players.iter().map(|p| p.hand.len()).collect::<Vec<_>>(),
            "hand sizes must match exactly after from_state"
        );
        assert_eq!(
            resumed.state.deck.len(), original.deck.len(),
            "deck size must match exactly after from_state"
        );
        assert_eq!(resumed.state.turn_number, original.turn_number);
        assert_eq!(resumed.state.played_counts, original.played_counts);
        assert_eq!(resumed.state.play_history, original.play_history);
    }

    // ── 1c. Card-conservation invariant ──────────────────────────────────────

    #[test]
    fn card_conservation_after_from_state() {
        let mut engine = GameEngine::new(Deck::build_classic(), adv_players(4));
        engine.set_silent();
        engine.set_max_turns(8);
        let _ = engine.run();
        assert_card_conservation(&engine.state, "after real turns");

        let resumed = GameEngine::from_state(engine.state.clone(), adv_players(4));
        assert_card_conservation(&resumed.state, "immediately after from_state");
    }

    #[test]
    fn card_conservation_after_simulated_turns() {
        let mut engine = GameEngine::new(Deck::build_classic(), adv_players(4));
        engine.set_silent();
        engine.set_max_turns(5);
        let _ = engine.run();

        let mut resumed = GameEngine::from_state(engine.state.clone(), adv_players(4));
        resumed.set_silent();
        let cap = resumed.state.turn_number + 15;
        resumed.set_max_turns(cap);
        let _ = resumed.run();
        assert_card_conservation(&resumed.state, "after from_state + 15 more simulated turns");
    }

    // ── 1d. Determinism ───────────────────────────────────────────────────
    //
    // FINDING: `from_state(state.clone(), adv_players).run()` is NOT
    // deterministic across repeated calls, even with identical AdvCPU
    // players and identical starting state. Empirically (20 repeats from the
    // same 4-turn-old snapshot, played out another ~400 turns): 19/20 runs
    // diverged in winner, turn count, final hand sizes, and deck order.
    // Root cause is NOT AdvCPU's decision logic (that part IS a pure
    // function of view/state - see the test below) but unseeded
    // `rand::thread_rng()`/`rand::random()` calls reachable through entirely
    // ordinary play: `Deck::shuffle`/`shuffle_in` (used by Protostar,
    // Ignition, Shockwave, Black Hole, Pulsar, Stellar Nursery Collapse -
    // all common cards) and `wormhole_blind_steal`'s random hand-position
    // pick. This is pre-existing engine behavior, not something introduced
    // by today's changes, and out of scope to fix here (would mean
    // threading a seeded RNG through the whole engine) - reported instead of
    // forced. A hard `assert_eq!` on repeated `run()` outputs would be
    // permanently flaky, so it's deliberately not asserted here.
    #[test]
    fn adv_cpu_decision_is_pure_given_identical_state() {
        let mut engine = GameEngine::new(Deck::build_classic(), adv_players(4));
        engine.set_silent();
        engine.set_max_turns(5);
        let _ = engine.run();
        let snapshot = engine.state.clone();
        let active = snapshot.active_player_idx;

        let e1 = GameEngine::from_state(snapshot.clone(), adv_players(4));
        let e2 = GameEngine::from_state(snapshot.clone(), adv_players(4));

        let legal1 = e1.compute_legal_actions(active);
        let legal2 = e2.compute_legal_actions(active);
        assert_eq!(
            legal1.len(), legal2.len(),
            "compute_legal_actions must be pure given identical state"
        );

        let view1 = e1.build_view(active);
        let view2 = e2.build_view(active);
        let choice1 = e1.players[active].choose_action_with_state(&view1, &e1.state, &legal1);
        let choice2 = e2.players[active].choose_action_with_state(&view2, &e2.state, &legal2);
        assert_eq!(
            choice1, choice2,
            "AdvancedCpuPlayer::choose_action_with_state must be deterministic given identical inputs"
        );
    }
}
