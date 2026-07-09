use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use novacana_engine::{
    card::{Card, CardId},
    chain::ChainView,
    cpu::AdvancedCpuPlayer,
    deck::Deck,
    game::{GameEngine, GameState},
    player::{Action, ChainAction, ChoiceRequest, GameView, Player, TargetOption},
};

use crate::action::{
    NUM_ACTIONS, abstract_to_chain, action_to_abstract,
    chain_legal_mask, chain_response_to_abstract, legal_mask,
};
use crate::candidate::{CAND_DIM, encode_action_candidate, encode_card_candidate, encode_decide_candidate, encode_target_candidate};
use crate::feature::encode;
use crate::mcts::{mcts_search_v6, StateStack};
use crate::net::{masked_softmax, Net, Weights, INPUT_DIM};
use crate::net_burn::MAX_CANDS;

// ── PPO/RL sample ─────────────────────────────────────────────────────────────

pub struct RlSample {
    pub features:    Vec<f32>,
    pub legal_mask:  u64,
    pub action:      u8,
    pub outcome:     f32,   // GAE return
    pub log_prob:    f32,
    pub td_error:    f32,   // |advantage|, used for PER priority
    pub value:       f32,   // value estimate at decision time
    pub cand_feats:  Vec<f32>,
    pub cand_chosen: u8,
    pub has_policy:  bool,  // false: aux cand sample (decide/targets/discards) — cand head only
}

struct Pending {
    features:      Vec<f32>,
    legal_mask:    u64,
    action:        u8,
    log_prob:      f32,
    shaped_reward: f32,
    value:         f32,
    cand_feats:    Vec<f32>,
    cand_chosen:   u8,
    has_policy:    bool,
}

// Sample an index from softmax(scores / temp). Aux choices must be SAMPLED,
// not argmaxed, for REINFORCE to be unbiased — with argmax, a negative
// advantage on the current pick just thrashes between candidates.
fn sample_scores(scores: &[f32], temp: f32, u: f32) -> usize {
    let t = temp.max(1e-6);
    let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|s| ((s - mx) / t).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut cum = 0.0f32;
    for (i, e) in exps.iter().enumerate() {
        cum += e / sum;
        if u < cum { return i; }
    }
    scores.len() - 1
}

// ── PPO self-play player ──────────────────────────────────────────────────────

struct SelfPlayPlayer {
    name_str:         String,
    net:              Arc<Net>,
    buffer:           Arc<Mutex<Vec<Pending>>>,
    stack:            Mutex<StateStack>,
    temperature:      f32,
    fallback:         AdvancedCpuPlayer,
    prev_my_systems:  AtomicUsize,
    prev_opp_systems: Mutex<Vec<usize>>,
}

impl SelfPlayPlayer {
    fn new(name: impl Into<String>, net: Arc<Net>, temperature: f32) -> (Self, Arc<Mutex<Vec<Pending>>>) {
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

    fn encode_and_stack(&self, view: &GameView) -> [f32; INPUT_DIM] {
        let snap = encode(view);
        let mut st = self.stack.lock().unwrap();
        st.push(snap);
        st.stacked()
    }

    fn update_shaped_reward(&self, view: &GameView) {
        let cur_my = view.my_star_chart.complete_systems().len();
        let prv_my = self.prev_my_systems.load(Ordering::Relaxed);
        let new_mine = cur_my.saturating_sub(prv_my);
        let mut reward = new_mine as f32 * 0.10;
        if cur_my >= 3 && prv_my < 3 { reward += 0.15; }
        {
            let prv = self.prev_opp_systems.lock().unwrap();
            for (i, opp) in view.opponents.iter().enumerate() {
                let prv_opp = prv.get(i).copied().unwrap_or(0);
                reward -= opp.star_chart.complete_systems().len().saturating_sub(prv_opp) as f32 * 0.05;
            }
        }
        if reward.abs() > 1e-9 {
            if let Some(last) = self.buffer.lock().unwrap().last_mut() {
                last.shaped_reward += reward;
            }
        }
        self.prev_my_systems.store(cur_my, Ordering::Relaxed);
        *self.prev_opp_systems.lock().unwrap() =
            view.opponents.iter().map(|o| o.star_chart.complete_systems().len()).collect();
    }

    fn record(&self, p: Pending) {
        self.buffer.lock().unwrap().push(p);
    }
}

impl Player for SelfPlayPlayer {
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

        let result_idx: usize;
        let cand_feats: Vec<f32>;
        let cand_chosen: u8;

        if !matching.is_empty() {
            if matching.len() == 1 {
                result_idx = matching[0];
                cand_feats = Vec::new();
                cand_chosen = 0;
            } else {
                let nc = matching.len().min(MAX_CANDS);
                let cands: Vec<[f32; CAND_DIM]> = matching[..nc].iter().enumerate()
                    .map(|(slot, &ci)| encode_action_candidate(&legal[ci], view, slot, nc))
                    .collect();
                let scores = self.net.score_candidates(&body, &cands);
                let best = sample_scores(&scores, self.temperature, rand::random());
                result_idx = matching[best];
                cand_feats = cands.iter().flat_map(|c| c.iter().copied()).collect();
                cand_chosen = best as u8;
            }
        } else {
            let idx = self.fallback.choose_action(view, legal);
            let abs = action_to_abstract(&legal[idx], hand).unwrap_or(abs_idx);
            self.record(Pending {
                features: features.to_vec(), legal_mask: mask, action: abs as u8,
                log_prob, shaped_reward: 0.0, value: val, cand_feats: Vec::new(), cand_chosen: 0,
                has_policy: true,
            });
            return idx;
        }

        self.record(Pending {
            features: features.to_vec(), legal_mask: mask, action: abs_idx as u8,
            log_prob, shaped_reward: 0.0, value: val, cand_feats, cand_chosen,
            has_policy: true,
        });
        result_idx
    }

    fn respond_to_chain(&self, view: &GameView, chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        let features = self.encode_and_stack(view);
        let mask = chain_legal_mask(legal, &view.my_hand);
        let u: f32 = rand::random();
        let (abs_idx, log_prob, val, _body) =
            self.net.infer_stochastic_with_body(&features, mask, self.temperature, u);
        self.record(Pending {
            features: features.to_vec(), legal_mask: mask, action: abs_idx as u8,
            log_prob, shaped_reward: 0.0, value: val, cand_feats: Vec::new(), cand_chosen: 0,
            has_policy: true,
        });
        if abs_idx == 0 { return None; }
        let result = abstract_to_chain(abs_idx, legal, &view.my_hand)
            .and_then(|r| legal.iter().position(|x| std::ptr::eq(x, r)));
        if result.is_none() {
            let fb = self.fallback.respond_to_chain(view, chain, legal);
            if let Some(idx) = fb {
                if let Some(a) = chain_response_to_abstract(&legal[idx], &view.my_hand) {
                    self.buffer.lock().unwrap().last_mut().map(|p| p.action = a as u8);
                }
            }
            return fb;
        }
        result
    }

    // The three aux paths record REINFORCE candidate samples (has_policy: false)
    // so outcome signal reaches decide/target/discard behaviour — the component
    // the 2x2 eval isolated as the ~7-point win-rate bottleneck. infer_with_body
    // (not body_forward) so the pending sample carries a real value estimate and
    // the GAE chain stays consistent.
    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        let total = req.options.len();
        if total == 0 { return 0; }
        let features = self.encode_and_stack(view);
        let (_, value, body) = self.net.infer_with_body(&features);
        let nc = total.min(MAX_CANDS);
        let cands: Vec<[f32; CAND_DIM]> = req.options.iter().enumerate().take(nc)
            .map(|(i, opt)| encode_decide_candidate(&req.prompt, opt, view, i, total))
            .collect();
        let scores = self.net.score_candidates(&body, &cands);
        let best = sample_scores(&scores, self.temperature, rand::random());
        if nc >= 2 {
            self.record(Pending {
                features: features.to_vec(), legal_mask: 0, action: 0, log_prob: 0.0,
                shaped_reward: 0.0, value,
                cand_feats: cands.iter().flat_map(|c| c.iter().copied()).collect(),
                cand_chosen: best as u8, has_policy: false,
            });
        }
        best
    }

    fn pick_targets(&self, view: &GameView, _prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
        if options.is_empty() { return Vec::new(); }
        let features = self.encode_and_stack(view);
        let (_, value, body) = self.net.infer_with_body(&features);
        let total = options.len();
        let cands: Vec<[f32; CAND_DIM]> = options.iter().enumerate()
            .map(|(i, opt)| encode_target_candidate(opt, view, i, total))
            .collect();
        let scores = self.net.score_candidates(&body, &cands);
        let first = sample_scores(&scores, self.temperature, rand::random());
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        order.retain(|&i| i != first);
        order.insert(0, first);
        let nc = total.min(MAX_CANDS);
        if nc >= 2 && order[0] < nc {
            self.record(Pending {
                features: features.to_vec(), legal_mask: 0, action: 0, log_prob: 0.0,
                shaped_reward: 0.0, value,
                cand_feats: cands[..nc].iter().flat_map(|c| c.iter().copied()).collect(),
                cand_chosen: order[0] as u8, has_policy: false,
            });
        }
        order.truncate(count);
        order
    }

    fn choose_discards(&self, view: &GameView, hand: &[Card], n: usize) -> Vec<CardId> {
        if hand.is_empty() { return Vec::new(); }
        let features = self.encode_and_stack(view);
        let (_, value, body) = self.net.infer_with_body(&features);
        let total = hand.len();
        let cands: Vec<[f32; CAND_DIM]> = hand.iter().enumerate()
            .map(|(i, card)| encode_card_candidate(card, i, total))
            .collect();
        let scores = self.net.score_candidates(&body, &cands);
        let first = sample_scores(&scores, self.temperature, rand::random());
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        order.retain(|&i| i != first);
        order.insert(0, first);
        let nc = total.min(MAX_CANDS);
        if nc >= 2 && order[0] < nc {
            self.record(Pending {
                features: features.to_vec(), legal_mask: 0, action: 0, log_prob: 0.0,
                shaped_reward: 0.0, value,
                cand_feats: cands[..nc].iter().flat_map(|c| c.iter().copied()).collect(),
                cand_chosen: order[0] as u8, has_policy: false,
            });
        }
        order.truncate(n);
        order.iter().map(|&i| hand[i].id).collect()
    }
}

// ── Run one PPO self-play game ────────────────────────────────────────────────

fn run_one_game(
    current_net:  Arc<Net>,
    pool_nets:    &[Arc<Net>],
    num_players:  usize,
    temperature:  f32,
    gamma:        f32,
    lambda:       f32,
    advcpu_share: f32,
) -> Vec<RlSample> {
    let use_pool = !pool_nets.is_empty() && rand::random::<f32>() < 0.5;
    let mut bufs:    Vec<Option<Arc<Mutex<Vec<Pending>>>>> = Vec::new();
    let mut players: Vec<Box<dyn Player>>                  = Vec::new();

    for i in 0..num_players {
        if use_pool && i != 0 && rand::random::<f32>() < advcpu_share {
            bufs.push(None);
            players.push(Box::new(AdvancedCpuPlayer::new(format!("P{}", i + 1))));
        } else {
            let net = if use_pool && i != 0 {
                Arc::clone(&pool_nets[rand::random::<usize>() % pool_nets.len()])
            } else {
                Arc::clone(&current_net)
            };
            let (p, buf) = SelfPlayPlayer::new(format!("P{}", i + 1), net, temperature);
            bufs.push(Some(buf));
            players.push(Box::new(p));
        }
    }

    let mut engine = GameEngine::new(Deck::build_classic(), players);
    engine.set_silent();
    engine.set_max_turns(1000);
    let result = engine.run();

    let gl = gamma * lambda;
    let mut out = Vec::new();

    for (player_idx, buf_opt) in bufs.into_iter().enumerate() {
        let buf = match buf_opt { Some(b) => b, None => continue };
        let raw_outcome = if result.winner_idx == player_idx { 1.0f32 } else { 0.0 };
        let inner = Arc::try_unwrap(buf)
            .unwrap_or_else(|b| Mutex::new(std::mem::take(&mut *b.lock().unwrap())));
        let pending = inner.into_inner().unwrap();
        let t = pending.len();
        if t == 0 { continue; }

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
                features:    p.features,
                legal_mask:  p.legal_mask,
                action:      p.action,
                outcome:     ret,
                log_prob:    p.log_prob,
                td_error:    (ret - p.value).abs(),
                value:       p.value,
                cand_feats:  p.cand_feats,
                cand_chosen: p.cand_chosen,
                has_policy:  p.has_policy,
            });
        }
    }
    out
}

// ── Public PPO API ────────────────────────────────────────────────────────────

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
    let pool_nets: Vec<Arc<Net>> = pool.iter().map(|w| Arc::new(Net::from_weights(w))).collect();
    (0..num_games).into_par_iter()
        .flat_map(|_| run_one_game(
            Arc::clone(&current_net), &pool_nets, num_players, temperature, gamma, lambda, advcpu_share,
        ))
        .collect()
}

// ── Prioritized Experience Replay ─────────────────────────────────────────────

pub struct PerBuffer {
    pub samples:  Vec<RlSample>,
    priorities:   Vec<f32>,
    alpha:        f32,
    beta:         f32,
    max_size:     usize,
    write_pos:    usize,
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

    pub fn push_batch(&mut self, batch: Vec<RlSample>) {
        let max_p = self.priorities.iter().cloned().fold(1.0f32, f32::max);
        for s in batch {
            let p = (s.td_error.abs() + 1e-6).powf(self.alpha).min(max_p);
            if self.samples.len() < self.max_size {
                self.samples.push(s);
                self.priorities.push(p);
            } else {
                self.samples[self.write_pos]    = s;
                self.priorities[self.write_pos] = p;
                self.write_pos = (self.write_pos + 1) % self.max_size;
            }
        }
    }

    pub fn len(&self) -> usize { self.samples.len() }

    pub fn sample_indices(&self, n: usize) -> (Vec<usize>, Vec<f32>) {
        let total: f32 = self.priorities.iter().sum();
        let n_f = self.samples.len() as f32;
        let min_p = self.priorities.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_w = (n_f * min_p / total).powf(-self.beta).max(1.0);

        // Cumulative prefix sum built once per call, then a binary search per
        // draw (O(log n)) instead of a linear scan per draw (O(n)).
        let mut prefix = Vec::with_capacity(self.priorities.len());
        let mut running = 0.0f32;
        for &p in &self.priorities {
            running += p;
            prefix.push(running);
        }

        let mut indices    = Vec::with_capacity(n);
        let mut is_weights = Vec::with_capacity(n);
        for _ in 0..n {
            let r = rand::random::<f32>() * total;
            let idx = prefix.partition_point(|&c| c < r).min(self.samples.len() - 1);
            let p_i = self.priorities[idx] / total;
            is_weights.push((n_f * p_i).powf(-self.beta) / max_w);
            indices.push(idx);
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
        self.beta += (target - self.beta) * frac;
    }
}

// ── AlphaZero sample ──────────────────────────────────────────────────────────

pub struct AzSample {
    pub features:      Vec<f32>,
    pub policy_target: [f32; NUM_ACTIONS],  // MCTS visit distribution (sums to 1)
    pub legal_mask:    u64,
    pub outcome:       f32,                 // raw game outcome
    pub value_est:     f32,                 // network value at decision time
    pub has_policy:    bool,                // use for policy CE + value MSE training
    pub cand_feats:    Vec<f32>,
    pub cand_chosen:   u8,
    pub has_cand:      bool,
}

struct AzPending {
    features:      Vec<f32>,
    policy_target: [f32; NUM_ACTIONS],
    legal_mask:    u64,
    value_est:     f32,
    has_policy:    bool,
    cand_feats:    Vec<f32>,
    cand_chosen:   u8,
    has_cand:      bool,
}

// Sampling temperature for AZ candidate picks — fixed because the AZ loop has
// no temperature schedule. Sampled (not argmax) picks keep the REINFORCE
// estimator unbiased, same lesson as the PPO legs.
const AZ_CAND_TEMP: f32 = 0.35;

// ── AlphaZero self-play player ────────────────────────────────────────────────

struct AzSelfPlayPlayer {
    name_str:  String,
    net:       Arc<Net>,
    buffer:    Arc<Mutex<Vec<AzPending>>>,
    stack:     Mutex<StateStack>,
    mcts_sims: usize,
    c_puct:    f32,
    fallback:  AdvancedCpuPlayer,
    moves:     AtomicUsize,   // decisions taken; first SAMPLE_MOVES are sampled
    explore:   bool,          // root noise + visit-temp openings (see mcts.rs warning)
}

impl AzSelfPlayPlayer {
    fn new(
        name: impl Into<String>,
        net: Arc<Net>,
        mcts_sims: usize,
        c_puct: f32,
        explore: bool,
    ) -> (Self, Arc<Mutex<Vec<AzPending>>>) {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let n   = name.into();
        let p   = Self {
            fallback:  AdvancedCpuPlayer::new(n.clone()),
            name_str:  n,
            net,
            buffer:    Arc::clone(&buf),
            stack:     Mutex::new(StateStack::new()),
            mcts_sims,
            c_puct,
            moves:     AtomicUsize::new(0),
            explore,
        };
        (p, buf)
    }

    fn encode_and_stack(&self, view: &GameView) -> [f32; INPUT_DIM] {
        let snap = encode(view);
        let mut st = self.stack.lock().unwrap();
        st.push(snap);
        st.stacked()
    }

    fn record(&self, p: AzPending) {
        self.buffer.lock().unwrap().push(p);
    }
}

impl Player for AzSelfPlayPlayer {
    fn name(&self) -> &str { &self.name_str }

    // Greedy fallback when no state is available — should be rare in practice.
    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        let features = self.encode_and_stack(view);
        let mask = legal_mask(legal, &view.my_hand);
        let (logits, _, body) = self.net.infer_with_body(&features);
        let probs = masked_softmax(&logits, mask);
        let mut best_p = -1.0f32;
        let mut best_abs = 0usize;
        for i in 0..NUM_ACTIONS {
            if mask & (1u64 << i) != 0 && probs[i] > best_p { best_p = probs[i]; best_abs = i; }
        }
        let hand = &view.my_hand;
        let matching: Vec<usize> = legal.iter().enumerate()
            .filter(|(_, a)| action_to_abstract(a, hand) == Some(best_abs))
            .map(|(i, _)| i).collect();
        if matching.len() <= 1 { return matching.into_iter().next().unwrap_or(0); }
        let nc = matching.len().min(MAX_CANDS);
        let cands: Vec<[f32; CAND_DIM]> = matching[..nc].iter().enumerate()
            .map(|(slot, &ci)| encode_action_candidate(&legal[ci], view, slot, nc))
            .collect();
        let scores = self.net.score_candidates(&body, &cands);
        let best = scores.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        matching[best]
    }

    fn choose_action_with_state(&self, view: &GameView, state: &GameState, legal: &[Action]) -> usize {
        let features = self.encode_and_stack(view);
        let mask = legal_mask(legal, &view.my_hand);

        let (visit_dist, best_abs) = mcts_search_v6(
            state, &self.net, view, legal, &features, mask, self.mcts_sims, self.c_puct,
            self.explore,
        );
        let (_, value, body) = self.net.infer_with_body(&features);

        // Visit-temperature move selection (OPT-IN with `explore`, same caveat
        // as root noise): sample from the visit distribution for the first
        // SAMPLE_MOVES decisions. The policy TARGET stays the full visit dist.
        const SAMPLE_MOVES: usize = 15;
        let move_n   = self.moves.fetch_add(1, Ordering::Relaxed);
        let best_abs = if self.explore && move_n < SAMPLE_MOVES {
            let u: f32 = rand::random();
            let mut cum = 0.0f32;
            let mut pick = best_abs;
            for i in 0..NUM_ACTIONS {
                if visit_dist[i] > 0.0 {
                    cum += visit_dist[i];
                    if u < cum { pick = i; break; }
                }
            }
            pick
        } else { best_abs };

        let hand = &view.my_hand;
        let matching: Vec<usize> = legal.iter().enumerate()
            .filter(|(_, a)| action_to_abstract(a, hand) == Some(best_abs))
            .map(|(i, _)| i).collect();

        let result_idx: usize;
        let cand_feats: Vec<f32>;
        let cand_chosen: u8;

        if matching.is_empty() {
            result_idx = self.fallback.choose_action(view, legal);
            self.record(AzPending {
                features: features.to_vec(), policy_target: visit_dist, legal_mask: mask,
                value_est: value, has_policy: true, cand_feats: Vec::new(), cand_chosen: 0, has_cand: false,
            });
            return result_idx;
        } else if matching.len() == 1 {
            result_idx = matching[0];
            cand_feats = Vec::new();
            cand_chosen = 0;
        } else {
            let nc = matching.len().min(MAX_CANDS);
            let cands: Vec<[f32; CAND_DIM]> = matching[..nc].iter().enumerate()
                .map(|(slot, &ci)| encode_action_candidate(&legal[ci], view, slot, nc))
                .collect();
            let scores = self.net.score_candidates(&body, &cands);
            let best = sample_scores(&scores, AZ_CAND_TEMP, rand::random());
            result_idx = matching[best];
            cand_feats = cands.iter().flat_map(|c| c.iter().copied()).collect();
            cand_chosen = best as u8;
        }

        let has_cand = !cand_feats.is_empty();
        self.record(AzPending {
            features: features.to_vec(), policy_target: visit_dist, legal_mask: mask,
            value_est: value, has_policy: true, cand_feats, cand_chosen, has_cand,
        });
        result_idx
    }

    fn respond_to_chain(&self, view: &GameView, _chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        let features = self.encode_and_stack(view);
        let mask = chain_legal_mask(legal, &view.my_hand);
        let (logits, _, _) = self.net.infer_with_body(&features);
        let probs = masked_softmax(&logits, mask);
        let mut best_p = -1.0f32;
        let mut best_abs = 0usize;
        for i in 0..NUM_ACTIONS { if mask & (1u64 << i) != 0 && probs[i] > best_p { best_p = probs[i]; best_abs = i; } }
        if best_abs == 0 { return None; }
        abstract_to_chain(best_abs, legal, &view.my_hand)
            .and_then(|r| legal.iter().position(|x| std::ptr::eq(x, r)))
    }

    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        let total = req.options.len();
        if total == 0 { return 0; }
        let features = self.encode_and_stack(view);
        let body = self.net.body_forward(&features);
        let nc = total.min(MAX_CANDS);
        let cands: Vec<[f32; CAND_DIM]> = req.options.iter().take(nc).enumerate()
            .map(|(i, opt)| encode_decide_candidate(&req.prompt, opt, view, i, total))
            .collect();
        let scores = self.net.score_candidates(&body, &cands);
        let best = sample_scores(&scores, AZ_CAND_TEMP, rand::random());
        if nc >= 2 && best < nc {
            let flat: Vec<f32> = cands.iter().flat_map(|c| c.iter().copied()).collect();
            self.record(AzPending {
                features: features.to_vec(), policy_target: [0.0; NUM_ACTIONS],
                legal_mask: 0, value_est: 0.0, has_policy: false,
                cand_feats: flat, cand_chosen: best as u8, has_cand: true,
            });
        }
        best
    }

    fn pick_targets(&self, view: &GameView, _prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
        if options.is_empty() { return Vec::new(); }
        let features = self.encode_and_stack(view);
        let body = self.net.body_forward(&features);
        let nc = options.len().min(MAX_CANDS);
        let cands: Vec<[f32; CAND_DIM]> = options.iter().take(nc).enumerate()
            .map(|(i, opt)| encode_target_candidate(opt, view, i, nc))
            .collect();
        let scores = self.net.score_candidates(&body, &cands);
        let first = sample_scores(&scores, AZ_CAND_TEMP, rand::random());
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        order.retain(|&i| i != first);
        order.insert(0, first);
        if nc >= 2 && !order.is_empty() && order[0] < nc {
            let flat: Vec<f32> = cands.iter().flat_map(|c| c.iter().copied()).collect();
            self.record(AzPending {
                features: features.to_vec(), policy_target: [0.0; NUM_ACTIONS],
                legal_mask: 0, value_est: 0.0, has_policy: false,
                cand_feats: flat, cand_chosen: order[0] as u8, has_cand: true,
            });
        }
        order.truncate(count);
        order
    }

    fn choose_discards(&self, view: &GameView, hand: &[Card], n: usize) -> Vec<CardId> {
        if hand.is_empty() { return Vec::new(); }
        let features = self.encode_and_stack(view);
        let body = self.net.body_forward(&features);
        let nc = hand.len().min(MAX_CANDS);
        let cands: Vec<[f32; CAND_DIM]> = hand.iter().take(nc).enumerate()
            .map(|(i, card)| encode_card_candidate(card, i, nc))
            .collect();
        let scores = self.net.score_candidates(&body, &cands);
        let first = sample_scores(&scores, AZ_CAND_TEMP, rand::random());
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        order.retain(|&i| i != first);
        order.insert(0, first);
        if nc >= 2 && !order.is_empty() && order[0] < nc {
            let flat: Vec<f32> = cands.iter().flat_map(|c| c.iter().copied()).collect();
            self.record(AzPending {
                features: features.to_vec(), policy_target: [0.0; NUM_ACTIONS],
                legal_mask: 0, value_est: 0.0, has_policy: false,
                cand_feats: flat, cand_chosen: order[0] as u8, has_cand: true,
            });
        }
        order.truncate(n);
        order.iter().map(|&i| hand[i].id).collect()
    }
}

// ── Run one AlphaZero game ────────────────────────────────────────────────────

fn run_one_az_game(
    current_net:  Arc<Net>,
    pool_nets:    &[Arc<Net>],
    num_players:  usize,
    mcts_sims:    usize,
    c_puct:       f32,
    advcpu_share: f32,
    explore:      bool,
) -> Vec<AzSample> {
    let use_pool = !pool_nets.is_empty() && rand::random::<f32>() < 0.5;
    let mut bufs:    Vec<Option<Arc<Mutex<Vec<AzPending>>>>> = Vec::new();
    let mut players: Vec<Box<dyn Player>>                    = Vec::new();

    for i in 0..num_players {
        if use_pool && i != 0 && rand::random::<f32>() < advcpu_share {
            bufs.push(None);
            players.push(Box::new(AdvancedCpuPlayer::new(format!("P{}", i + 1))));
        } else if use_pool && i != 0 {
            // Pool opponent: greedy SelfPlayPlayer, not recorded for AZ training
            let net = Arc::clone(&pool_nets[rand::random::<usize>() % pool_nets.len()]);
            let (p, _) = SelfPlayPlayer::new(format!("P{}", i + 1), net, 0.3);
            bufs.push(None);
            players.push(Box::new(p));
        } else {
            let (p, buf) = AzSelfPlayPlayer::new(
                format!("P{}", i + 1), Arc::clone(&current_net), mcts_sims, c_puct, explore,
            );
            bufs.push(Some(buf));
            players.push(Box::new(p));
        }
    }

    let mut engine = GameEngine::new(Deck::build_classic(), players);
    engine.set_silent();
    // 400, not 1000: with MCTS at every decision a degenerate stalemate game
    // at the full cap costs HOURS on one core. Typical games end well under
    // 200 turns; truncation awards the leader, which is fine as a training
    // outcome.
    engine.set_max_turns(400);
    let result = engine.run();

    let mut out = Vec::new();
    for (player_idx, buf_opt) in bufs.into_iter().enumerate() {
        let buf = match buf_opt { Some(b) => b, None => continue };
        let outcome = if result.winner_idx == player_idx { 1.0f32 } else { 0.0 };
        let inner = Arc::try_unwrap(buf)
            .unwrap_or_else(|b| Mutex::new(std::mem::take(&mut *b.lock().unwrap())));
        let pending = inner.into_inner().unwrap();
        for p in pending {
            out.push(AzSample {
                features:      p.features,
                policy_target: p.policy_target,
                legal_mask:    p.legal_mask,
                outcome,
                value_est:     p.value_est,
                has_policy:    p.has_policy,
                cand_feats:    p.cand_feats,
                cand_chosen:   p.cand_chosen,
                has_cand:      p.has_cand,
            });
        }
    }
    out
}

// ── Public AlphaZero API ──────────────────────────────────────────────────────

pub fn run_self_play_az(
    num_games:    usize,
    num_players:  usize,
    current:      Arc<Weights>,
    pool:         &[Arc<Weights>],
    mcts_sims:    usize,
    c_puct:       f32,
    advcpu_share: f32,
    explore:      bool,
) -> Vec<AzSample> {
    assert!((2..=4).contains(&num_players));
    let current_net = Arc::new(Net::from_weights(&current));
    let pool_nets: Vec<Arc<Net>> = pool.iter().map(|w| Arc::new(Net::from_weights(w))).collect();
    (0..num_games).into_par_iter()
        .flat_map(|_| run_one_az_game(
            Arc::clone(&current_net), &pool_nets, num_players, mcts_sims, c_puct, advcpu_share, explore,
        ))
        .collect()
}
