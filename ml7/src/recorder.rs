// BC data generation: an AdvancedCpuPlayer-imitating recorder that emits
// (token sequence, chosen action index) samples across all five decision
// kinds through the tokens.rs pointer-policy tokenizer. Ports the
// `AdvCpuRecorder` shape from ml/src/main.rs, but there is no abstract
// action layer or candidate head to split samples across anymore — every
// decision kind produces exactly one pointer-policy sample.

use std::sync::{Arc, Mutex};

use novacana_engine::card::CardId;
use novacana_engine::chain::ChainView;
use novacana_engine::cpu::AdvancedCpuPlayer;
use novacana_engine::deck::Deck;
use novacana_engine::game::GameEngine;
use novacana_engine::player::{
    Action, ChainAction, ChoiceRequest, GameView, Player, TargetOption,
};

use crate::tokens::{self, TokenSeq};

struct Pending {
    tokens: TokenSeq,
    target: u16,
}

/// One BC training example: a fully assembled token sequence, the pointer
/// target (index into the action-token block AdvCPU actually chose), and
/// the terminal game outcome (1.0 = this seat won) used as the BC-stage
/// value warm-start target.
pub struct Sample {
    pub tokens: TokenSeq,
    pub target: u16,
    pub outcome: f32,
}

struct BcRecorder {
    name_str: String,
    fallback: AdvancedCpuPlayer,
    buffer: Arc<Mutex<Vec<Pending>>>,
}

impl BcRecorder {
    fn new(name: impl Into<String>, buffer: Arc<Mutex<Vec<Pending>>>) -> Self {
        let n = name.into();
        Self { fallback: AdvancedCpuPlayer::new(n.clone()), name_str: n, buffer }
    }

    fn push_if_encodable(&self, view: &GameView, action_tokens: &[tokens::Token], target: u16) {
        if let Some(enc) = tokens::assemble(view, action_tokens) {
            if (target as usize) < enc.n_actions {
                self.buffer.lock().unwrap().push(Pending { tokens: enc.tokens, target });
            }
        }
    }
}

impl Player for BcRecorder {
    fn name(&self) -> &str {
        &self.name_str
    }

    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        let chosen = self.fallback.choose_action(view, legal);
        let action_tokens = tokens::encode_action_tokens_choose_action(view, legal);
        self.push_if_encodable(view, &action_tokens, chosen as u16);
        chosen
    }

    fn respond_to_chain(&self, view: &GameView, chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        let result = self.fallback.respond_to_chain(view, chain, legal);
        let action_tokens = tokens::encode_action_tokens_chain(view, legal);
        // index 0 = pass (see tokens::encode_action_tokens_chain doc comment).
        let target = result.map(|i| i + 1).unwrap_or(0) as u16;
        self.push_if_encodable(view, &action_tokens, target);
        result
    }

    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        if req.options.is_empty() {
            return 0;
        }
        let chosen = self.fallback.decide(view, req);
        let action_tokens = tokens::encode_action_tokens_decide(view, req);
        self.push_if_encodable(view, &action_tokens, chosen as u16);
        chosen
    }

    fn pick_targets(&self, view: &GameView, prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
        if options.is_empty() {
            return Vec::new();
        }
        let chosen = self.fallback.pick_targets(view, prompt, options, count);
        if !chosen.is_empty() {
            let action_tokens = tokens::encode_action_tokens_targets(view, options);
            self.push_if_encodable(view, &action_tokens, chosen[0] as u16);
        }
        chosen
    }

    fn choose_discards(&self, view: &GameView, hand: &[novacana_engine::card::Card], must_discard: usize) -> Vec<CardId> {
        if hand.is_empty() {
            return Vec::new();
        }
        let chosen_ids = self.fallback.choose_discards(view, hand, must_discard);
        if let Some(&first) = chosen_ids.first() {
            if let Some(slot) = hand.iter().position(|c| c.id == first) {
                let action_tokens = tokens::encode_action_tokens_discards(hand);
                self.push_if_encodable(view, &action_tokens, slot as u16);
            }
        }
        chosen_ids
    }
}

fn run_one_game(num_players: usize) -> Vec<Sample> {
    let mut bufs: Vec<Arc<Mutex<Vec<Pending>>>> = Vec::new();
    let mut players: Vec<Box<dyn Player>> = Vec::new();
    for i in 0..num_players {
        let buf = Arc::new(Mutex::new(Vec::new()));
        bufs.push(Arc::clone(&buf));
        players.push(Box::new(BcRecorder::new(format!("AdvCPU{}", i + 1), buf)));
    }
    let mut engine = GameEngine::new(Deck::build_classic(), players);
    engine.set_silent();
    engine.set_max_turns(1000); // ASTRA7.md §8.1: guard against pathological single-game loops
    let result = engine.run();

    let mut out = Vec::new();
    for (player_idx, buf) in bufs.into_iter().enumerate() {
        let outcome = if result.winner_idx == player_idx { 1.0f32 } else { 0.0 };
        let pending = Arc::try_unwrap(buf)
            .unwrap_or_else(|b| Mutex::new(std::mem::take(&mut *b.lock().unwrap())))
            .into_inner()
            .unwrap();
        for p in pending {
            out.push(Sample { tokens: p.tokens, target: p.target, outcome });
        }
    }
    out
}

/// Generate BC samples from `num_games` fresh self-play games (AdvCPU vs
/// itself, `num_players` seats). Streams: caller controls how many games are
/// live in memory at once by calling this in chunks — never pregenerate more
/// than ~8-10K games at once (ASTRA7.md §8.3 — hermes runs earlyoom).
pub fn generate_samples(num_games: usize, num_players: usize) -> Vec<Sample> {
    use rayon::prelude::*;
    (0..num_games).into_par_iter().flat_map(|_| run_one_game(num_players)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::{TokenType, ACTION_MAX, HISTORY_MAX, STATE_MAX};

    // Integration-style round-trip test: run real games through the actual
    // engine and assert the tokenizer never panics and every recorded
    // sample's target lands on a genuinely non-Pad action-token slot.
    #[test]
    fn real_games_produce_well_formed_samples() {
        let samples = generate_samples(20, 4);
        assert!(!samples.is_empty(), "expected at least some recorded decisions from 20 games");

        let action_base = STATE_MAX + HISTORY_MAX;
        let mut kind_counts = [0usize; 16];
        for s in &samples {
            assert!((s.target as usize) < ACTION_MAX);
            let tok = s.tokens[action_base + s.target as usize];
            assert_eq!(tok.ttype, TokenType::Action as u8, "target must point at a real Action token, not Pad");
            let kind = tok.fields[0] as usize;
            if kind < kind_counts.len() {
                kind_counts[kind] += 1;
            }
            assert!(s.outcome == 0.0 || s.outcome == 1.0);
        }

        // Sanity: choose_action (kind 0..10) and chain/decide/target/discard
        // (kind 11..15) should BOTH appear across 20 real games — if one
        // decision kind is silently never recorded, that's a tokenizer bug,
        // not a game-rules quirk (every game has discards and most have
        // decide/target prompts).
        let choose_action_like: usize = kind_counts[0..11].iter().sum();
        let other_kinds: usize = kind_counts[11..16].iter().sum();
        assert!(choose_action_like > 0, "expected at least one choose_action sample");
        assert!(other_kinds > 0, "expected at least one non-choose_action decision sample (chain/decide/target/discard)");
    }
}
