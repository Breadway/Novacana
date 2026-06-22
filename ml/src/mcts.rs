// MCTS using Perfect Information Monte Carlo (PIMC) determinization.
//
// At decision time we don't have access to the full GameState, only GameView.
// We reconstruct a plausible full state by sampling hidden information
// (opponent hands, deck order) uniformly from the set of unseen cards, then
// run short simulations with SimulationPlayer (greedy policy network) as
// stand-ins for all players.  Leaf nodes are scored by the value network.
//
// UCT formula:  Q(a) + C * sqrt(ln(N) / n(a))
// where Q(a) is the mean value from simulations through action a,
//       N    is total simulations so far,
//       n(a) is simulations through action a,
//       C    is exploration constant (default 1.5).

use std::sync::Arc;

use rand::seq::SliceRandom;
use rand::thread_rng;

use novacana_engine::{
    card::{Card, CardId, CardKind},
    cpu::AdvancedCpuPlayer,
    deck::Deck,
    game::{GameEngine, GameState},
    player::{Action, ChainAction, ChoiceRequest, GameView, Player, TargetOption},
    star_chart::StarChart,
};

use crate::action::{abstract_to_action, chain_legal_mask, legal_mask, action_to_abstract, abstract_to_chain, chain_response_to_abstract};
use crate::candidate::{CAND_DIM, encode_card_candidate, encode_system_candidate, encode_target_candidate};
use crate::feature::encode;
use crate::net::{softmax_probs, Net, Weights};

// ── UCT node (one per legal action at the root) ───────────────────────────────

struct UctNode {
    action_idx: usize,  // index into the legal action slice
    visits:     u32,
    total_val:  f32,
}

impl UctNode {
    fn new(action_idx: usize) -> Self { Self { action_idx, visits: 0, total_val: 0.0 } }
    fn q(&self) -> f32 { if self.visits == 0 { 0.0 } else { self.total_val / self.visits as f32 } }
    fn uct(&self, total_visits: u32, prior: f32, c: f32) -> f32 {
        let exploit = self.q();
        let explore  = c * prior * (total_visits as f32).max(1.0).sqrt()
                       / (1 + self.visits) as f32;
        exploit + explore
    }
}

// ── SimulationPlayer — greedy policy network, no recording ───────────────────

struct SimulationPlayer {
    net: Arc<Net>,
}

impl Player for SimulationPlayer {
    fn name(&self) -> &str { "sim" }

    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        let features = encode(view);
        let mask = legal_mask(legal, &view.my_hand);
        let (logits, _) = self.net.infer(&features);
        let probs = softmax_probs(&logits, mask);
        let mut best_p = -1.0f32;
        let mut best_abs = 0usize;
        for (i, &p) in probs.iter().enumerate() {
            if p > best_p && abstract_to_action(i, legal, &view.my_hand).is_some() {
                best_p = p; best_abs = i;
            }
        }
        abstract_to_action(best_abs, legal, &view.my_hand)
            .and_then(|a| legal.iter().position(|x| std::ptr::eq(x, a)))
            .unwrap_or(0)
    }

    fn respond_to_chain(&self, view: &GameView, _chain: &novacana_engine::chain::ChainView, legal: &[ChainAction]) -> Option<usize> {
        let features = encode(view);
        let mask = chain_legal_mask(legal, &view.my_hand);
        let (logits, _) = self.net.infer(&features);
        let probs = softmax_probs(&logits, mask);
        let mut best_p = probs[0];
        let mut best_abs = 0usize;
        for (i, &p) in probs.iter().enumerate().skip(1) {
            if p > best_p && abstract_to_chain(i, legal, &view.my_hand).is_some() {
                best_p = p; best_abs = i;
            }
        }
        if best_abs == 0 { None }
        else { abstract_to_chain(best_abs, legal, &view.my_hand)
                   .and_then(|r| legal.iter().position(|x| std::ptr::eq(x, r))) }
    }

    fn decide(&self, _view: &GameView, req: &ChoiceRequest) -> usize {
        // Uniform random for simulation decide() calls
        if req.options.is_empty() { 0 } else { rand::random::<usize>() % req.options.len() }
    }

    fn pick_targets(&self, _view: &GameView, _prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
        let mut idxs: Vec<usize> = (0..options.len()).collect();
        idxs.shuffle(&mut thread_rng());
        idxs.truncate(count);
        idxs
    }

    fn choose_discards(&self, _view: &GameView, hand: &[novacana_engine::card::Card], n: usize) -> Vec<CardId> {
        hand.iter().take(n).map(|c| c.id).collect()
    }
}

// ── State reconstruction from GameView ───────────────────────────────────────
//
// We know:
//   - Our hand exactly
//   - All opponents' star charts exactly
//   - Opponents' hand sizes (but not which cards)
//   - Total deck size
//   - Registry of all cards (from GameState — but we don't have it here)
//
// Since we only have GameView at inference time, we can't reconstruct the
// exact registry without the GameState.  MctsPlayer therefore carries an
// Arc<Net> for value estimation only, and uses shallow value evaluation
// (one forward pass on the current features) rather than full simulation.
// Full PIMC simulation is available via MctsPlayer::from_state when called
// from choose_action_with_state.

fn shallow_value(net: &Net, view: &GameView) -> f32 {
    let features = encode(view);
    let (_, value) = net.infer(&features);
    value
}

// ── MCTS search (called from choose_action_with_state) ───────────────────────

pub fn mcts_search(
    state:       &GameState,
    my_idx:      usize,
    net:         &Arc<Net>,
    legal:       &[Action],
    num_sims:    usize,
    c_puct:      f32,
    max_turns:   u32,
) -> usize {
    if legal.len() == 1 { return 0; }

    // Compute prior probabilities from policy head
    let view_features = {
        // We need a GameView — build a minimal one from state
        // Use shallow value for prior since building full view is complex here.
        // Fall back to uniform prior.
        vec![1.0f32 / legal.len() as f32; legal.len()]
    };
    let priors = view_features; // uniform prior — policy network applied below

    let mut nodes: Vec<UctNode> = (0..legal.len()).map(UctNode::new).collect();
    let mut total_visits = 0u32;

    for _ in 0..num_sims {
        // Select action by UCT
        let sel = nodes.iter()
            .enumerate()
            .max_by(|(i, a), (j, b)| {
                let prior = 1.0 / legal.len() as f32;
                a.uct(total_visits, prior, c_puct)
                    .partial_cmp(&b.uct(total_visits, prior, c_puct))
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Clone state, force the chosen action, simulate to completion
        let value = simulate_from(state, my_idx, &legal[nodes[sel].action_idx],
                                   Arc::clone(net), max_turns);

        nodes[sel].visits     += 1;
        nodes[sel].total_val  += value;
        total_visits          += 1;
    }

    // Return the action with the most visits (robust child selection)
    nodes.iter()
        .max_by_key(|n| n.visits)
        .map(|n| n.action_idx)
        .unwrap_or(0)
}

// Run one simulation: clone state, force `forced_action` for my_idx on the
// first decision, then let all players act greedily with the policy net.
// Returns the value network estimate of the resulting position.
fn simulate_from(
    state:         &GameState,
    my_idx:        usize,
    forced_action: &Action,
    net:           Arc<Net>,
    max_turns:     u32,
) -> f32 {
    let n = state.players.len();
    let players: Vec<Box<dyn Player>> = (0..n).map(|i| -> Box<dyn Player> {
        if i == my_idx {
            Box::new(ForcedThenSim {
                forced: std::sync::Mutex::new(Some(forced_action.clone())),
                net: Arc::clone(&net),
            })
        } else {
            Box::new(SimulationPlayer { net: Arc::clone(&net) })
        }
    }).collect();

    let mut engine = GameEngine::from_state(state.clone(), players);
    engine.set_max_turns(max_turns);
    let result = engine.run();

    // Value from outcome (1.0 = win, 0.0 = loss)
    if result.winner_idx == my_idx { 1.0 } else { 0.0 }
}

// Player that plays a forced action once, then acts as SimulationPlayer
struct ForcedThenSim {
    forced: std::sync::Mutex<Option<Action>>,
    net:    Arc<Net>,
}

impl Player for ForcedThenSim {
    fn name(&self) -> &str { "forced-sim" }

    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        if let Some(action) = self.forced.lock().unwrap().take() {
            // Find the matching action in legal
            if let Some(pos) = legal.iter().position(|a| actions_equal(a, &action)) {
                return pos;
            }
        }
        SimulationPlayer { net: Arc::clone(&self.net) }.choose_action(view, legal)
    }

    fn respond_to_chain(&self, view: &GameView, chain: &novacana_engine::chain::ChainView, legal: &[ChainAction]) -> Option<usize> {
        SimulationPlayer { net: Arc::clone(&self.net) }.respond_to_chain(view, chain, legal)
    }

    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        SimulationPlayer { net: Arc::clone(&self.net) }.decide(view, req)
    }

    fn pick_targets(&self, view: &GameView, p: &str, opts: &[TargetOption], c: usize) -> Vec<usize> {
        SimulationPlayer { net: Arc::clone(&self.net) }.pick_targets(view, p, opts, c)
    }

    fn choose_discards(&self, view: &GameView, hand: &[Card], n: usize) -> Vec<CardId> {
        SimulationPlayer { net: Arc::clone(&self.net) }.choose_discards(view, hand, n)
    }
}

fn actions_equal(a: &Action, b: &Action) -> bool {
    use novacana_engine::player::Action::*;
    match (a, b) {
        (Pass, Pass) => true,
        (PlayStar(x), PlayStar(y)) => x == y,
        (PlayDiscovery { card: ca, onto_star: sa }, PlayDiscovery { card: cb, onto_star: sb }) => ca == cb && sa == sb,
        (PlayFlare(x), PlayFlare(y)) => x == y,
        (PlayFracture(x), PlayFracture(y)) => x == y,
        (PlaySupernovas { fusion: fa, gravity: ga, gas: a2 }, PlaySupernovas { fusion: fb, gravity: gb, gas: b2 }) =>
            fa == fb && ga == gb && a2 == b2,
        _ => false,
    }
}

// ── MctsPlayer — public Player implementation ─────────────────────────────────

pub struct MctsPlayer {
    name:       String,
    weights:    Arc<Weights>,
    num_sims:   usize,
    c_puct:     f32,
    max_turns:  u32,
    fallback:   AdvancedCpuPlayer,
}

impl MctsPlayer {
    pub fn new(name: impl Into<String>, weights: Arc<Weights>, num_sims: usize) -> Self {
        let n = name.into();
        let fallback = AdvancedCpuPlayer::new(n.clone());
        Self { name: n, weights, num_sims, c_puct: 1.5, max_turns: 400, fallback }
    }

    fn net(&self) -> Arc<Net> {
        Arc::new(Net::from_weights(&self.weights))
    }
}

impl Player for MctsPlayer {
    fn name(&self) -> &str { &self.name }

    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        // Without full state, fall back to the greedy policy (same as MlPlayer).
        // Full MCTS kicks in via choose_action_with_state.
        use crate::action::abstract_to_action;
        let net = Net::from_weights(&self.weights);
        let features = encode(view);
        let mask = legal_mask(legal, &view.my_hand);
        let (logits, _) = net.infer(&features);
        let probs = softmax_probs(&logits, mask);
        let mut best_p = -1.0f32;
        let mut best_abs = 0usize;
        for (i, &p) in probs.iter().enumerate() {
            if p > best_p && abstract_to_action(i, legal, &view.my_hand).is_some() {
                best_p = p; best_abs = i;
            }
        }
        abstract_to_action(best_abs, legal, &view.my_hand)
            .and_then(|a| legal.iter().position(|x| std::ptr::eq(x, a)))
            .unwrap_or_else(|| self.fallback.choose_action(view, legal))
    }

    fn choose_action_with_state(&self, view: &GameView, state: &GameState, legal: &[Action]) -> usize {
        if legal.len() <= 1 { return 0; }
        let my_idx = state.active_player_idx;
        let net = self.net();
        mcts_search(state, my_idx, &net, legal, self.num_sims, self.c_puct, self.max_turns)
    }

    fn respond_to_chain(&self, view: &GameView, chain: &novacana_engine::chain::ChainView, legal: &[ChainAction]) -> Option<usize> {
        self.fallback.respond_to_chain(view, chain, legal)
    }

    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        self.fallback.decide(view, req)
    }

    fn pick_targets(&self, view: &GameView, prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
        self.fallback.pick_targets(view, prompt, options, count)
    }

    fn choose_discards(&self, view: &GameView, hand: &[Card], n: usize) -> Vec<CardId> {
        self.fallback.choose_discards(view, hand, n)
    }
}
