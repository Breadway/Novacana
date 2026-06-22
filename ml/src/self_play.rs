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
use crate::feature::encode;
use crate::net::{Net, Weights};

// ── Sample produced by a self-play game ───────────────────────────────────────

pub struct RlSample {
    pub features:   Vec<f32>,
    pub legal_mask: u64,
    pub action:     u8,
    pub outcome:    f32,
    pub log_prob:   f32,
}

// ── Per-decision buffer entry ─────────────────────────────────────────────────

struct Pending {
    features:      Vec<f32>,
    legal_mask:    u64,
    action:        u8,
    #[allow(dead_code)]
    step:          usize,
    log_prob:      f32,
    shaped_reward: f32,
    value:         f32,
}

// ── Player that acts stochastically and records decisions ─────────────────────

struct SelfPlayPlayer {
    name_str:         String,
    net:              Arc<Net>,
    buffer:           Arc<Mutex<Vec<Pending>>>,
    temperature:      f32,
    fallback:         AdvancedCpuPlayer,
    prev_my_systems:  AtomicUsize,
    prev_opp_systems: Mutex<Vec<usize>>,
}

impl SelfPlayPlayer {
    fn new(
        name: impl Into<String>,
        net: Arc<Net>,
        temperature: f32,
    ) -> (Self, Arc<Mutex<Vec<Pending>>>) {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let n   = name.into();
        let p   = Self {
            fallback:         AdvancedCpuPlayer::new(n.clone()),
            name_str:         n,
            net,
            buffer:           Arc::clone(&buf),
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

    fn record(&self, features: Vec<f32>, mask: u64, action: u8, log_prob: f32, value: f32) {
        let mut buf = self.buffer.lock().unwrap();
        let step = buf.len();
        buf.push(Pending { features, legal_mask: mask, action, step, log_prob, shaped_reward: 0.0, value });
    }
}

impl Player for SelfPlayPlayer {
    fn name(&self) -> &str { &self.name_str }

    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        self.update_shaped_reward(view);

        let features = encode(view);
        let mask = legal_mask(legal, &view.my_hand);
        let u: f32 = rand::random();
        let (abs_idx, log_prob, val, body) = self.net.infer_stochastic_with_body(&features, mask, self.temperature, u);

        self.record(features.to_vec(), mask, abs_idx as u8, log_prob, val);

        let hand = &view.my_hand;
        let matching: Vec<usize> = legal.iter().enumerate()
            .filter(|(_, a)| action_to_abstract(a, hand) == Some(abs_idx))
            .map(|(i, _)| i)
            .collect();

        if !matching.is_empty() {
            if matching.len() == 1 { return matching[0]; }
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
                    _ => { let mut c = [0.0f32; CAND_DIM]; c[22] = slot as f32 / total as f32; c[23] = total as f32 / 10.0; c }
                }
            }).collect();
            let scores = self.net.score_candidates(&body, &cands);
            let best = scores.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            return matching[best];
        }

        let idx = self.fallback.choose_action(view, legal);
        if let Some(a) = action_to_abstract(&legal[idx], &view.my_hand) {
            if let Some(last) = self.buffer.lock().unwrap().last_mut() { last.action = a as u8; }
        }
        idx
    }

    fn respond_to_chain(&self, view: &GameView, _chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        let features = encode(view);
        let mask = chain_legal_mask(legal, &view.my_hand);
        let u: f32 = rand::random();
        let (abs_idx, log_prob, val, _body) = self.net.infer_stochastic_with_body(&features, mask, self.temperature, u);

        self.record(features.to_vec(), mask, abs_idx as u8, log_prob, val);

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
        let features = encode(view);
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
        let features = encode(view);
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
        let features = encode(view);
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

fn run_one_game(
    current_net:  Arc<Net>,
    pool_nets:    &[Arc<Net>],
    num_players:  usize,
    temperature:  f32,
    gamma:        f32,
    lambda:       f32,
    advcpu_share: f32,
) -> Vec<RlSample> {
    let use_pool   = !pool_nets.is_empty() && rand::random::<f32>() < 0.5;
    let learn_seat = if use_pool { rand::random::<usize>() % num_players } else { usize::MAX };

    let mut bufs:    Vec<(Arc<Mutex<Vec<Pending>>>, bool)> = Vec::new();
    let mut players: Vec<Box<dyn Player>>                  = Vec::new();

    for i in 0..num_players {
        let is_learner = !use_pool || i == learn_seat;

        if is_learner {
            let (p, buf) = SelfPlayPlayer::new(format!("P{}", i + 1), Arc::clone(&current_net), temperature);
            bufs.push((buf, true));
            players.push(Box::new(p));
        } else if !pool_nets.is_empty() && rand::random::<f32>() >= advcpu_share {
            let idx = rand::random::<usize>() % pool_nets.len();
            let (p, buf) = SelfPlayPlayer::new(format!("P{}", i + 1), Arc::clone(&pool_nets[idx]), temperature);
            bufs.push((buf, false));
            players.push(Box::new(p));
        } else {
            let buf: Arc<Mutex<Vec<Pending>>> = Arc::new(Mutex::new(Vec::new()));
            bufs.push((Arc::clone(&buf), false));
            players.push(Box::new(AdvancedCpuPlayer::new(format!("P{}", i + 1))));
        }
    }

    let mut engine = GameEngine::new(Deck::build_classic(), players);
    engine.set_silent();
    engine.set_max_turns(1000);
    let result = engine.run();

    let mut out = Vec::new();
    for (player_idx, (buf, is_learner)) in bufs.into_iter().enumerate() {
        if !is_learner { continue; }

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
            out.push(RlSample {
                features:   p.features,
                legal_mask: p.legal_mask,
                action:     p.action,
                outcome:    ret,
                log_prob:   p.log_prob,
            });
        }
    }
    out
}

// ── Public API ────────────────────────────────────────────────────────────────

pub fn run_self_play(
    num_games:    usize,
    num_players:  usize,
    current:      Arc<Weights>,
    pool:         &[Arc<Weights>],
    temperature:  f32,
    gamma:        f32,
    lambda:       f32,
    advcpu_share: f32,
) -> Vec<RlSample> {
    assert!((2..=4).contains(&num_players));

    let current_net = Arc::new(Net::from_weights(&current));
    let pool_nets: Vec<Arc<Net>> = pool.iter()
        .map(|w| Arc::new(Net::from_weights(w)))
        .collect();

    (0..num_games)
        .into_par_iter()
        .flat_map(|_| run_one_game(
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
