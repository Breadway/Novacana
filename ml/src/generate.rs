use std::io::{self, BufWriter, Write};
use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use novacana_engine::{
    cpu::AdvancedCpuPlayer,
    deck::Deck,
    game::GameEngine,
    player::{Action, ChainAction, ChoiceRequest, GameView, Player, TargetOption},
    card::{Card, CardId},
    chain::ChainView,
};

use crate::action::{action_to_abstract, chain_response_to_abstract, chain_legal_mask, legal_mask};
use crate::candidate::{CAND_DIM, encode_card_candidate, encode_target_candidate};
use crate::feature::encode;

// ── Sample ────────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone)]
pub struct Sample {
    pub features:   Vec<f32>,
    pub legal_mask: u64,
    pub action:     u8,
    pub outcome:    f32, // 1.0 = win, 0.0 = loss
    pub is_chain:   bool,
}

// ── CandidateSample ───────────────────────────────────────────────────────────
//
// A supervised sample for the candidate-scoring head. Features are the full
// game state; candidates are the encoded options; chosen_idx is which one
// AdvancedCpuPlayer picked. No outcome label needed — the imitation target
// is AdvCPU's choice, not the game result.

#[derive(Serialize, Deserialize, Clone)]
pub struct CandidateSample {
    pub features:    Vec<f32>,
    pub candidates:  Vec<[f32; CAND_DIM]>,
    pub chosen_idx:  u8,
}

// ── RecordingPlayer ───────────────────────────────────────────────────────────

struct PendingSample {
    features:   Vec<f32>,
    legal_mask: u64,
    action:     u8,
    is_chain:   bool,
}

struct RecordingPlayer {
    inner:      AdvancedCpuPlayer,
    buffer:     Arc<Mutex<Vec<PendingSample>>>,
    cand_buf:   Arc<Mutex<Vec<CandidateSample>>>,
}

impl RecordingPlayer {
    fn new(name: impl Into<String>) -> (Self, Arc<Mutex<Vec<PendingSample>>>, Arc<Mutex<Vec<CandidateSample>>>) {
        let buf      = Arc::new(Mutex::new(Vec::new()));
        let cand_buf = Arc::new(Mutex::new(Vec::new()));
        let p = Self {
            inner:    AdvancedCpuPlayer::new(name),
            buffer:   Arc::clone(&buf),
            cand_buf: Arc::clone(&cand_buf),
        };
        (p, buf, cand_buf)
    }

    fn record(&self, features: Vec<f32>, mask: u64, action: u8, is_chain: bool) {
        self.buffer.lock().unwrap().push(PendingSample { features, legal_mask: mask, action, is_chain });
    }

    fn record_candidate(&self, features: Vec<f32>, candidates: Vec<[f32; CAND_DIM]>, chosen_idx: u8) {
        if candidates.len() > 1 {
            self.cand_buf.lock().unwrap().push(CandidateSample { features, candidates, chosen_idx });
        }
    }
}

impl Player for RecordingPlayer {
    fn name(&self) -> &str { self.inner.name() }

    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        let idx = self.inner.choose_action(view, legal);
        let features = encode(view).to_vec();
        let mask = legal_mask(legal, &view.my_hand);
        if let Some(a) = action_to_abstract(&legal[idx], &view.my_hand) {
            self.record(features, mask, a as u8, false);
        }
        idx
    }

    fn respond_to_chain(&self, view: &GameView, chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        let result = self.inner.respond_to_chain(view, chain, legal);
        let features = encode(view).to_vec();
        let mask = chain_legal_mask(legal, &view.my_hand);
        let abstract_action = match result {
            None => 0u8, // pass
            Some(idx) => chain_response_to_abstract(&legal[idx], &view.my_hand)
                .unwrap_or(0) as u8,
        };
        self.record(features, mask, abstract_action, true);
        result
    }

    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        let chosen = self.inner.decide(view, req);
        let features = encode(view).to_vec();
        let total = req.options.len();
        // Encode each option positionally — no text semantics, but position + count
        // let the candidate head learn context-relative preferences.
        let cands: Vec<[f32; CAND_DIM]> = (0..total).map(|i| {
            let mut c = [0.0f32; CAND_DIM];
            c[22] = i as f32 / total.max(1) as f32;
            c[23] = total as f32 / 10.0;
            c
        }).collect();
        self.record_candidate(features, cands, chosen as u8);
        chosen
    }

    fn pick_targets(&self, view: &GameView, prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
        let chosen = self.inner.pick_targets(view, prompt, options, count);
        let features = encode(view).to_vec();
        let total = options.len();
        let cands: Vec<[f32; CAND_DIM]> = options.iter().enumerate()
            .map(|(i, opt)| encode_target_candidate(opt, view, i, total))
            .collect();
        // Record one sample per target picked (independent approximation for multi-pick).
        for &c in &chosen {
            if c < 255 {
                self.record_candidate(features.clone(), cands.clone(), c as u8);
            }
        }
        chosen
    }

    fn choose_discards(&self, view: &GameView, hand: &[Card], n: usize) -> Vec<CardId> {
        let chosen_ids = self.inner.choose_discards(view, hand, n);
        let features = encode(view).to_vec();
        let total = hand.len();
        let cands: Vec<[f32; CAND_DIM]> = hand.iter().enumerate()
            .map(|(i, card)| encode_card_candidate(card, i, total))
            .collect();
        // For each discarded card, emit one candidate sample where chosen_idx is its hand position.
        for id in &chosen_ids {
            if let Some(pos) = hand.iter().position(|c| c.id == *id) {
                if pos < 255 {
                    self.record_candidate(features.clone(), cands.clone(), pos as u8);
                }
            }
        }
        chosen_ids
    }
}

// ── Run one game, return labeled samples ──────────────────────────────────────

const MAX_TURNS: u32 = 1000;

fn run_one_game(num_players: usize) -> (Vec<Sample>, Vec<CandidateSample>) {
    let mut bufs:      Vec<Arc<Mutex<Vec<PendingSample>>>> = Vec::new();
    let mut cand_bufs: Vec<Arc<Mutex<Vec<CandidateSample>>>> = Vec::new();
    let mut players:   Vec<Box<dyn Player>> = Vec::new();

    for i in 0..num_players {
        let (p, buf, cand_buf) = RecordingPlayer::new(format!("P{}", i + 1));
        bufs.push(buf);
        cand_bufs.push(cand_buf);
        players.push(Box::new(p));
    }

    let mut engine = GameEngine::new(Deck::build_classic(), players);
    engine.set_silent();
    engine.set_max_turns(MAX_TURNS);
    let result = engine.run();

    let mut samples    = Vec::new();
    let mut cand_out   = Vec::new();

    for (player_idx, buf) in bufs.into_iter().enumerate() {
        let outcome = if result.winner_idx == player_idx { 1.0f32 } else { 0.0 };
        let pending = Arc::try_unwrap(buf)
            .unwrap_or_else(|b| std::mem::take(&mut *b.lock().unwrap()).into())
            .into_inner()
            .unwrap();
        for p in pending {
            samples.push(Sample {
                features: p.features,
                legal_mask: p.legal_mask,
                action: p.action,
                outcome,
                is_chain: p.is_chain,
            });
        }
    }

    for cand_buf in cand_bufs {
        let cs = Arc::try_unwrap(cand_buf)
            .unwrap_or_else(|b| std::mem::take(&mut *b.lock().unwrap()).into())
            .into_inner()
            .unwrap();
        cand_out.extend(cs);
    }

    (samples, cand_out)
}

// ── Public API ────────────────────────────────────────────────────────────────

pub fn run_games_all(num_games: usize, num_players: usize) -> (Vec<Sample>, Vec<CandidateSample>) {
    assert!((2..=4).contains(&num_players));
    let results: Vec<(Vec<Sample>, Vec<CandidateSample>)> = (0..num_games)
        .into_par_iter()
        .map(|_| run_one_game(num_players))
        .collect();
    let mut samples = Vec::new();
    let mut cands   = Vec::new();
    for (s, c) in results {
        samples.extend(s);
        cands.extend(c);
    }
    (samples, cands)
}

pub fn run_games(num_games: usize, num_players: usize) -> Vec<Sample> {
    run_games_all(num_games, num_players).0
}

pub fn save_samples(samples: &[Sample], path: &str) -> io::Result<()> {
    save_bincode_vec(samples, path)
}

pub fn load_samples(path: &str) -> io::Result<Vec<Sample>> {
    load_bincode_vec(path)
}

pub fn save_candidate_samples(samples: &[CandidateSample], path: &str) -> io::Result<()> {
    save_bincode_vec(samples, path)
}

pub fn load_candidate_samples(path: &str) -> io::Result<Vec<CandidateSample>> {
    load_bincode_vec(path)
}

// ── Shared bincode framing ─────────────────────────────────────────────────────

fn save_bincode_vec<T: Serialize>(items: &[T], path: &str) -> io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut w = BufWriter::new(file);
    w.write_all(&(items.len() as u64).to_le_bytes())?;
    for item in items {
        let enc = bincode::serialize(item).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        w.write_all(&(enc.len() as u32).to_le_bytes())?;
        w.write_all(&enc)?;
    }
    w.flush()
}

fn load_bincode_vec<T: for<'de> Deserialize<'de>>(path: &str) -> io::Result<Vec<T>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut count_buf = [0u8; 8];
    file.read_exact(&mut count_buf)?;
    let count = u64::from_le_bytes(count_buf) as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut len_buf = [0u8; 4];
        file.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut data = vec![0u8; len];
        file.read_exact(&mut data)?;
        out.push(bincode::deserialize(&data).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?);
    }
    Ok(out)
}
