// "Spectator game" recording for the mission-control dashboard's replay
// theater + card-usage spectrum (2026-07-15, user request + Fable 5
// consult - see STATE.md's "make it fun" writeup). Not part of the
// training pipeline itself: piggybacks on the existing win-rate eval
// cadence in `run_bc` (same seat setup as `az::eval_win_rate` - seat 0 =
// greedy net, seats 1..N = AdvancedCpuPlayer) to produce one extra game's
// move-by-move JSON trace the dashboard can animate, at the cost of one
// extra game per eval (cheap relative to the eval's own `winrate_games`
// games).
//
// Scoped to `choose_action`-level decisions only (the ones that place or
// attach a real card) - chain responses/decide/targets are sub-resolution
// detail with no direct visual payoff on the Star Chart, and evals.rs's
// `ValueRankRecorder` already established the precedent of scoping a
// recorder this way rather than instrumenting every decision kind.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::Serialize;

use burn::tensor::backend::Backend;

use novacana_engine::card::{Card, CardId, CardKind};
use novacana_engine::chain::ChainView;
use novacana_engine::cpu::AdvancedCpuPlayer;
use novacana_engine::deck::Deck;
use novacana_engine::game::{GameEngine, GameState};
use novacana_engine::player::{
    Action, ChainAction, ChoiceRequest, GameView, OmenAttachment, Player, ProtostarChoice, TargetOption,
};
use novacana_engine::star_chart::StarChart;

use crate::az::GreedyNetPlayer;
use crate::batch_infer::BatchedNet;
use crate::net::Net;

#[derive(Serialize)]
pub struct BoardSnapshot {
    pub turn: u32,
    pub active_player: usize,
    pub dark_matter_void: Option<u8>,
    pub hand_sizes: Vec<usize>,
    pub star_charts: Vec<StarChart>,
    /// GameState's own cumulative per-kind play tally (CardKind::type_idx
    /// 0..29) as of this decision - the authoritative source for the
    /// "stellar spectrum" chart, not reconstructed from board state.
    pub played_counts: [u8; 29],
}

#[derive(Serialize)]
pub struct Frame {
    pub player: usize,
    pub player_name: String,
    pub action: String,
    pub board: BoardSnapshot,
}

#[derive(Serialize)]
pub struct SpectatorGame {
    pub players: Vec<String>,
    pub frames: Vec<Frame>,
    pub winner_idx: usize,
    /// Cumulative play count per card type (CardKind::type_idx 0..29),
    /// read from the last recorded decision's GameState - may slightly
    /// undercount the very last chain-response/discard-only actions of
    /// the game (those decision kinds aren't recorded), which is fine for
    /// a "what did this policy tend to play" visualization.
    pub card_spectrum: [u8; 29],
}

fn describe_action(action: &Action, registry: &HashMap<CardId, CardKind>) -> String {
    let name = |id: &CardId| registry.get(id).map(|k| k.short_name()).unwrap_or_else(|| "an unknown card".into());
    match action {
        Action::Pass => "passed".to_string(),
        Action::PlayStar(id) => format!("placed {} into the Star Chart", name(id)),
        Action::PlayDiscovery { card, onto_star } => format!("attached {} to {}", name(card), name(onto_star)),
        Action::PlayFlare(id) => format!("played {} (Flare)", name(id)),
        Action::PlayFracture(id) => format!("played {} (Fracture)", name(id)),
        Action::PlayOmen { card, attachment } => match attachment {
            OmenAttachment::SolarEclipse { system_star } => {
                format!("attached {} to protect the {} system", name(card), name(system_star))
            }
            OmenAttachment::PlanetOfLife { rocky_planet_disc } => {
                format!("attached {} to {}", name(card), name(rocky_planet_disc))
            }
        },
        Action::PlaySolarSystem { star, discoveries } => format!(
            "played a complete system: {} + {}",
            name(star),
            discoveries.iter().map(name).collect::<Vec<_>>().join(", ")
        ),
        Action::PlaySupernovas { fusion, gravity, gas } => format!(
            "triggered the Supernova win with {}, {}, {}",
            name(fusion), name(gravity), name(gas)
        ),
        Action::PlayProtostar { card, choice } => format!(
            "played {} and chose {}",
            name(card),
            match choice {
                ProtostarChoice::DrawTwo => "Draw Two",
                ProtostarChoice::SearchWhiteDwarf => "Search White Dwarf",
            }
        ),
    }
}

struct SpectatorRecorder {
    name_str: String,
    fallback: Box<dyn Player>,
    frames: Arc<Mutex<Vec<Frame>>>,
}

impl Player for SpectatorRecorder {
    fn name(&self) -> &str {
        &self.name_str
    }
    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        self.fallback.choose_action(view, legal)
    }
    fn choose_action_with_state(&self, view: &GameView, state: &GameState, legal: &[Action]) -> usize {
        let chosen = self.fallback.choose_action_with_state(view, state, legal);
        let board = BoardSnapshot {
            turn: state.turn_number,
            active_player: state.active_player_idx,
            dark_matter_void: view.dark_matter_void,
            hand_sizes: state.players.iter().map(|p| p.hand.len()).collect(),
            star_charts: state.players.iter().map(|p| p.star_chart.clone()).collect(),
            played_counts: state.played_counts,
        };
        self.frames.lock().unwrap().push(Frame {
            player: view.my_idx,
            player_name: view.my_name.clone(),
            action: describe_action(&legal[chosen], &state.registry),
            board,
        });
        chosen
    }
    fn respond_to_chain(&self, view: &GameView, chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        self.fallback.respond_to_chain(view, chain, legal)
    }
    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        self.fallback.decide(view, req)
    }
    fn pick_targets(&self, view: &GameView, prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
        self.fallback.pick_targets(view, prompt, options, count)
    }
    fn choose_discards(&self, view: &GameView, hand: &[Card], must_discard: usize) -> Vec<CardId> {
        self.fallback.choose_discards(view, hand, must_discard)
    }
}

/// Plays one spectator game (seat 0 = greedy net, seats 1..N = AdvCPU -
/// mirrors `az::eval_win_rate`'s setup) and returns a full move-by-move
/// trace plus the final card-play spectrum.
pub fn record_spectator_game<B: Backend>(net: &Net<B>, device: &B::Device, num_players: usize) -> SpectatorGame
where
    B::Device: Clone + Send + Sync + 'static,
{
    let bnet = BatchedNet::spawn(Arc::new(net.clone()), device.clone(), 64);
    let frames = Arc::new(Mutex::new(Vec::new()));

    let mut players: Vec<Box<dyn Player>> = Vec::new();
    let mut names = Vec::new();

    let net_name = "AzEvalNet".to_string();
    names.push(net_name.clone());
    players.push(Box::new(SpectatorRecorder {
        name_str: net_name.clone(),
        fallback: Box::new(GreedyNetPlayer { name_str: net_name, bnet }),
        frames: Arc::clone(&frames),
    }));
    for i in 1..num_players {
        let adv_name = format!("AdvCPU{}", i + 1);
        names.push(adv_name.clone());
        players.push(Box::new(SpectatorRecorder {
            name_str: adv_name.clone(),
            fallback: Box::new(AdvancedCpuPlayer::new(adv_name)),
            frames: Arc::clone(&frames),
        }));
    }

    let mut engine = GameEngine::new(Deck::build_classic(), players);
    engine.set_silent();
    engine.set_max_turns(400);
    let result = engine.run();

    let frames = Arc::try_unwrap(frames).unwrap_or_else(|f| Mutex::new(std::mem::take(&mut *f.lock().unwrap()))).into_inner().unwrap();
    let card_spectrum = frames.last().map(|f| f.board.played_counts).unwrap_or([0u8; 29]);

    SpectatorGame { players: names, frames, winner_idx: result.winner_idx, card_spectrum }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use burn_ndarray::NdArrayDevice;

    #[test]
    fn records_a_full_game_with_at_least_one_frame() {
        let device = NdArrayDevice::Cpu;
        let net = Net::<NdArray>::fresh(&device);
        let game = record_spectator_game(&net, &device, 4);
        assert_eq!(game.players.len(), 4);
        assert!(!game.frames.is_empty(), "expected at least one recorded choose_action decision");
        for f in &game.frames {
            assert_eq!(f.board.star_charts.len(), 4);
            assert_eq!(f.board.hand_sizes.len(), 4);
            assert!(!f.action.is_empty());
        }
    }
}
