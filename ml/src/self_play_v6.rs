use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use novacana_engine::{
    cpu::AdvancedCpuPlayer,
    deck::Deck,
    game::GameEngine,
    player::{Action, ChainAction, ChoiceRequest, GameView, Player, TargetOption},
    card::{Card, CardId},
    chain::ChainView,
};

use crate::action::{
    action_to_abstract, chain_response_to_abstract,
    chain_legal_mask, legal_mask,
    abstract_to_chain,
};
use crate::candidate::{CAND_DIM, encode_card_candidate, encode_system_candidate, encode_target_candidate};
use crate::feature_v6::{encode_v6, FEATURE_DIM_V6};
use crate::net_v6::{NetV6, WeightsV6, INPUT_DIM};
use crate::net_v6_burn::MAX_CANDS;

// ── Sample produced by a self-play game ───────────────────────────────────────

pub struct RlSampleV6 {
    pub features:    Vec<f32>,  // length INPUT_DIM (3 stacked snapshots)
    pub legal_mask:  u64,
    pub action:      u8,
    pub outcome:     f32,
    pub log_prob:    f32,
    pub td_error:    f32,       // |advantage|, used for PER priority
    pub value:       f32,       // value estimate at decision time (for advantage)
    pub cand_feats:  Vec<f32>,  // flat: n_cands * CAND_DIM; empty if no candidate choice
    pub cand_chosen: u8,        // index into cand_feats / CAND_DIM
}

// ── Per-decision buffer entry ─────────────────────────────────────────────────

struct Pending {
    features:      Vec<f32>,
    legal_mask:    u64,
    action:        u8,
    log_prob:      f32,
    shaped_reward: f32,
    value:         f32,
    cand_feats:    Vec<f32>,  // flat: n_cands * CAND_DIM; empty if no candidate choice
    cand_chosen:   u8,
}

// ── Stacked state buffer ──────────────────────────────────────────────────────

struct StateStack {
    buf: [[f32; FEATURE_DIM_V6]; 3],
    count: usize,
}

impl StateStack {
    fn new() -> Self {
        Self { buf: [[0.0; FEATURE_DIM_V6]; 3], count: 0 }
    }

    fn push(&mut self, snapshot: [f32; FEATURE_DIM_V6]) {
        self.buf[0] = self.buf[1];
        self.buf[1] = self.buf[2];
        self.buf[2] = snapshot;
        if self.count < 3 { self.count += 1; }
    }

    fn stacked(&self) -> [f32; INPUT_DIM] {
        let mut out = [0.0f32; INPUT_DIM];
        out[..FEATURE_DIM_V6].copy_from_slice(&self.buf[0]);
        out[FEATURE_DIM_V6..2 * FEATURE_DIM_V6].copy_from_slice(&self.buf[1]);
        out[2 * FEATURE_DIM_V6..].copy_from_slice(&self.buf[2]);
        out
    }
}

// ── Player that acts stochastically and records decisions ─────────────────────

struct SelfPlayPlayerV6 {
    name_str:         String,
    net:              Arc<NetV6>,
    buffer:           Arc<Mutex<Vec<Pending>>>,
    stack:            Mutex<StateStack>,
    temperature:      f32,
    fallback:         AdvancedCpuPlayer,
    prev_my_systems:  AtomicUsize,
    prev_opp_systems: Mutex<Vec<usize>>,
}

impl SelfPlayPlayerV6 {
    fn new(
        name: impl Into<String>,
        net: Arc<NetV6>,
        temperature: f32,
    ) -> (Self, Arc<Mutex<Vec<Pending>>>) {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let n   = name.into();
        let p   = Self {
            fallback:         AdvancedCpuPlayer::new(n.clone()),
            name_str:         n,
            net,
            buffer:           Arc::clone(&buf),
            stack:            Mutex::new(StateStack::new()),
            temperature,
            prev_my_systems:  AtomicUsize::new(0),
            prev_opp_systems: Mutex::new(Vec::new()),
        };
        (p, buf)
    }

    fn update_shaped_reward(&self, view: &GameView) {
        let cur_my = view.my_star_chart.complete_systems().len();
        let prv_my = self.prev_my_systems.load(Ordering::Relaxed);

        let new_mine = cur_my.saturating_sub(prv_my);
        let mut reward = new_mine as f32 * 0.1;

        if cur_my >= 3 && prv_my < 3 {
            reward += 0.15;
        }

        {
            let prv_opps = self.prev_opp_systems.lock().unwrap();
            for (i, opp) in view.opponents.iter().enumerate() {
                let cur_opp = opp.star_chart.complete_systems().len();
                let prv_opp = prv_opps.get(i).copied().unwrap_or(0);
                reward -= cur_opp.saturating_sub(prv_opp) as f32 * 0.05;
            }
        }

        if reward.abs() > 1e-9 {
            let mut buf = self.buffer.lock().unwrap();
            if let Some(last) = buf.last_mut() {
                last.shaped_reward += reward;
            }
        }

        self.prev_my_systems.store(cur_my, Ordering::Relaxed);
        *self.prev_opp_systems.lock().unwrap() = view.opponents.iter()
            .map(|o| o.star_chart.complete_systems().len())
            .collect();
    }

    fn encode_and_stack(&self, view: &GameView) -> [f32; INPUT_DIM] {
        let snap = encode_v6(view);
        let mut stack = self.stack.lock().unwrap();
        stack.push(snap);
        stack.stacked()
    }

    fn record(
        &self, features: Vec<f32>, mask: u64, action: u8, log_prob: f32, value: f32,
        cand_feats: Vec<f32>, cand_chosen: u8,
    ) {
        self.buffer.lock().unwrap().push(Pending {
            features, legal_mask: mask, action, log_prob, shaped_reward: 0.0, value,
            cand_feats, cand_chosen,
        });
    }
}

impl Player for SelfPlayPlayerV6 {
    fn name(&self) -> &str { &self.name_str }

    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        self.update_shaped_reward(view);

        let features = self.encode_and_stack(view);
        let mask = legal_mask(legal, &view.my_hand);
        let u: f32 = rand::random();
        let (abs_idx, log_prob, val, body) =
            self.net.infer_stochastic_with_body(&features, mask, self.temperature, u);

        let hand = &view.my_hand;
        let matching: Vec<usize> = legal.iter().enumerate()
            .filter(|(_, a)| action_to_abstract(a, hand) == Some(abs_idx))
            .map(|(i, _)| i)
            .collect();

        let result_idx;
        let cand_feats: Vec<f32>;
        let cand_chosen: u8;

        if !matching.is_empty() {
            if matching.len() == 1 {
                result_idx = matching[0];
                cand_feats = Vec::new();
                cand_chosen = 0;
            } else {
                let total = matching.len();
                let cands: Vec<[f32; CAND_DIM]> = matching.iter().enumerate().map(|(slot, &ci)| {
                    match &legal[ci] {
                        novacana_engine::player::Action::PlayDiscovery { onto_star, .. } => {
                            view.my_star_chart.systems.iter()
                                .find(|s| s.star_id == *onto_star)
                                .map(|s| encode_system_candidate(s, false, slot, total))
                                .unwrap_or([0.0; CAND_DIM])
                        }
                        novacana_engine::player::Action::PlayStar(id) => {
                            hand.iter().find(|c| c.id == *id)
                                .map(|c| encode_card_candidate(c, slot, total))
                                .unwrap_or([0.0; CAND_DIM])
                        }
                        _ => {
                            let mut c = [0.0f32; CAND_DIM];
                            c[22] = slot as f32 / total as f32;
                            c[23] = total as f32 / 10.0;
                            c
                        }
                    }
                }).collect();
                // Only score / record up to MAX_CANDS candidates
                let nc = cands.len().min(MAX_CANDS);
                let scores = self.net.score_candidates(&body, &cands[..nc]);
                let best = scores.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i).unwrap_or(0);
                result_idx = matching[best];
                cand_feats = cands[..nc].iter().flat_map(|c| c.iter().copied()).collect();
                cand_chosen = best as u8;
            }
        } else {
            let idx = self.fallback.choose_action(view, legal);
            let abs = action_to_abstract(&legal[idx], hand).unwrap_or(abs_idx);
            self.record(features.to_vec(), mask, abs as u8, log_prob, val, Vec::new(), 0);
            return idx;
        }

        self.record(features.to_vec(), mask, abs_idx as u8, log_prob, val, cand_feats, cand_chosen);
        result_idx
    }

    fn respond_to_chain(&self, view: &GameView, _chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        let features = self.encode_and_stack(view);
        let mask = chain_legal_mask(legal, &view.my_hand);
        let u: f32 = rand::random();
        let (abs_idx, log_prob, val, _body) =
            self.net.infer_stochastic_with_body(&features, mask, self.temperature, u);

        self.record(features.to_vec(), mask, abs_idx as u8, log_prob, val, Vec::new(), 0);

        if abs_idx == 0 { return None; }
        if let Some(r) = abstract_to_chain(abs_idx, legal, &view.my_hand) {
            return legal.iter().position(|x| std::ptr::eq(x, r));
        }

        let result = self.fallback.respond_to_chain(view, _chain, legal);
        if let Some(idx) = result {
            if let Some(a) = chain_response_to_abstract(&legal[idx], &view.my_hand) {
                self.buffer.lock().unwrap().last_mut().map(|p| p.action = a as u8);
            }
        }
        result
    }

    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        let total = req.options.len();
        if total == 0 { return 0; }
        let features = self.encode_and_stack(view);
        let body = self.net.body_forward(&features);
        let cands: Vec<[f32; CAND_DIM]> = (0..total).map(|i| {
            let mut c = [0.0f32; CAND_DIM];
            c[22] = i as f32 / total as f32;
            c[23] = total as f32 / 10.0;
            c
        }).collect();
        let scores = self.net.score_candidates(&body, &cands);
        scores.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn pick_targets(&self, view: &GameView, _prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
        if options.is_empty() { return Vec::new(); }
        let features = self.encode_and_stack(view);
        let body = self.net.body_forward(&features);
        let total = options.len();
        let cands: Vec<[f32; CAND_DIM]> = options.iter().enumerate()
            .map(|(i, opt)| encode_target_candidate(opt, view, i, total))
            .collect();
        let scores = self.net.score_candidates(&body, &cands);
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        order.truncate(count);
        order
    }

    fn choose_discards(&self, view: &GameView, hand: &[Card], n: usize) -> Vec<CardId> {
        if hand.is_empty() { return Vec::new(); }
        let features = self.encode_and_stack(view);
        let body = self.net.body_forward(&features);
        let total = hand.len();
        let cands: Vec<[f32; CAND_DIM]> = hand.iter().enumerate()
            .map(|(i, card)| encode_card_candidate(card, i, total))
            .collect();
        let scores = self.net.score_candidates(&body, &cands);
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        order.truncate(n);
        order.iter().map(|&i| hand[i].id).collect()
    }
}

// ── Run one self-play game ────────────────────────────────────────────────────
//
// Pool drain fix: ALL seats run SelfPlayPlayerV6 for sample recording.
// Pool snapshots only influence which net drives OPPONENT moves; every seat
// still contributes samples to training.

fn run_one_game_v6(
    current_net:  Arc<NetV6>,
    pool_nets:    &[Arc<NetV6>],
    num_players:  usize,
    temperature:  f32,
    gamma:        f32,
    lambda:       f32,
    advcpu_share: f32,
) -> Vec<RlSampleV6> {
    let use_pool = !pool_nets.is_empty() && rand::random::<f32>() < 0.5;

    let mut bufs:    Vec<Option<Arc<Mutex<Vec<Pending>>>>> = Vec::new();
    let mut players: Vec<Box<dyn Player>>                  = Vec::new();

    for i in 0..num_players {
        // In pool games, opponent seats have advcpu_share chance of being AdvCPU
        // (no sample collection) to maintain opponent diversity.
        if use_pool && i != 0 && rand::random::<f32>() < advcpu_share {
            bufs.push(None);
            players.push(Box::new(AdvancedCpuPlayer::new(format!("P{}", i + 1))));
        } else {
            let net = if use_pool && i != 0 {
                let idx = rand::random::<usize>() % pool_nets.len();
                Arc::clone(&pool_nets[idx])
            } else {
                Arc::clone(&current_net)
            };
            let (p, buf) = SelfPlayPlayerV6::new(format!("P{}", i + 1), net, temperature);
            bufs.push(Some(buf));
            players.push(Box::new(p));
        }
    }

    let mut engine = GameEngine::new(Deck::build_classic(), players);
    engine.set_silent();
    engine.set_max_turns(1000);
    let result = engine.run();

    let mut out = Vec::new();
    for (player_idx, buf_opt) in bufs.into_iter().enumerate() {
        let buf = match buf_opt {
            Some(b) => b,
            None => continue,
        };
        let raw_outcome = if result.winner_idx == player_idx { 1.0f32 } else { 0.0 };
        let pending = Arc::try_unwrap(buf)
            .unwrap_or_else(|b| std::mem::take(&mut *b.lock().unwrap()).into())
            .into_inner()
            .unwrap();
        let t = pending.len();
        if t == 0 { continue; }

        let gl = gamma * lambda;
        let mut returns = vec![0.0f32; t];
        let mut gae = 0.0f32;
        for i in (0..t).rev() {
            let r      = pending[i].shaped_reward + if i + 1 == t { raw_outcome } else { 0.0 };
            let v_next = if i + 1 < t { pending[i + 1].value } else { 0.0 };
            let delta  = r + gamma * v_next - pending[i].value;
            gae        = delta + gl * gae;
            returns[i] = gae + pending[i].value;
        }

        for (p, ret) in pending.into_iter().zip(returns) {
            let td_error = (ret - p.value).abs();
            out.push(RlSampleV6 {
                features:    p.features,
                legal_mask:  p.legal_mask,
                action:      p.action,
                outcome:     ret,
                log_prob:    p.log_prob,
                td_error,
                value:       p.value,
                cand_feats:  p.cand_feats,
                cand_chosen: p.cand_chosen,
            });
        }
    }
    out
}

// ── Public API ────────────────────────────────────────────────────────────────

pub fn run_self_play_v6(
    num_games:    usize,
    num_players:  usize,
    current:      Arc<WeightsV6>,
    pool:         &[Arc<WeightsV6>],
    temperature:  f32,
    gamma:        f32,
    lambda:       f32,
    advcpu_share: f32,
) -> Vec<RlSampleV6> {
    assert!((2..=4).contains(&num_players));

    let current_net = Arc::new(NetV6::from_weights(&current));
    let pool_nets: Vec<Arc<NetV6>> = pool.iter()
        .map(|w| Arc::new(NetV6::from_weights(w)))
        .collect();

    (0..num_games)
        .into_par_iter()
        .flat_map(|_| run_one_game_v6(
            Arc::clone(&current_net),
            &pool_nets,
            num_players,
            temperature,
            gamma,
            lambda,
            advcpu_share,
        ))
        .collect()
}

// ── Prioritized Experience Replay buffer ──────────────────────────────────────

pub struct PerBuffer {
    pub samples:    Vec<RlSampleV6>,
    priorities: Vec<f32>,
    alpha: f32,   // priority exponent (0.6 recommended)
    beta:  f32,   // IS weight exponent, annealed 0.4 → 1.0
    max_size: usize,
    write_pos: usize,
}

impl PerBuffer {
    pub fn new(max_size: usize, alpha: f32, beta_start: f32) -> Self {
        Self {
            samples: Vec::with_capacity(max_size),
            priorities: Vec::with_capacity(max_size),
            alpha,
            beta: beta_start,
            max_size,
            write_pos: 0,
        }
    }

    pub fn push_batch(&mut self, batch: Vec<RlSampleV6>) {
        let max_p = self.priorities.iter().cloned().fold(1.0f32, f32::max);
        for s in batch {
            let p = (s.td_error.abs() + 1e-6).powf(self.alpha);
            let p = p.min(max_p);
            if self.samples.len() < self.max_size {
                self.samples.push(s);
                self.priorities.push(p);
            } else {
                self.samples[self.write_pos] = s;
                self.priorities[self.write_pos] = p;
                self.write_pos = (self.write_pos + 1) % self.max_size;
            }
        }
    }

    pub fn len(&self) -> usize { self.samples.len() }

    /// Sample `n` indices proportional to priority. Returns (indices, IS weights).
    pub fn sample_indices(&self, n: usize) -> (Vec<usize>, Vec<f32>) {
        let total: f32 = self.priorities.iter().sum();
        let mut indices = Vec::with_capacity(n);
        let mut is_weights = Vec::with_capacity(n);
        let n_f = self.samples.len() as f32;
        let max_w = (n_f * self.priorities.iter().cloned().fold(f32::INFINITY, f32::min) / total)
            .powf(-self.beta)
            .max(1.0);

        for _ in 0..n {
            let mut r = rand::random::<f32>() * total;
            let mut idx = self.samples.len() - 1;
            for (i, &p) in self.priorities.iter().enumerate() {
                r -= p;
                if r <= 0.0 { idx = i; break; }
            }
            let p_i = self.priorities[idx] / total;
            let w = (n_f * p_i).powf(-self.beta) / max_w;
            indices.push(idx);
            is_weights.push(w);
        }
        (indices, is_weights)
    }

    pub fn update_priorities(&mut self, indices: &[usize], td_errors: &[f32]) {
        for (&i, &e) in indices.iter().zip(td_errors.iter()) {
            if i < self.priorities.len() {
                self.priorities[i] = (e.abs() + 1e-6).powf(self.alpha);
            }
        }
    }

    pub fn anneal_beta(&mut self, target: f32, frac: f32) {
        self.beta = self.beta + (target - self.beta) * frac;
    }
}
