use std::sync::Arc;

use novacana_engine::{
    card::{Card, CardId},
    chain::ChainView,
    cpu::AdvancedCpuPlayer,
    player::{Action, ChainAction, ChoiceRequest, GameView, Player, TargetOption},
};

use crate::action::{abstract_to_action, abstract_to_chain, chain_legal_mask, legal_mask};
use crate::candidate::{encode_card_candidate, encode_system_candidate, encode_target_candidate};
use crate::feature::encode;
use crate::net::{softmax_probs, Net, Weights};
use ndarray::Array1;

pub struct MlPlayer {
    name:    String,
    weights: Arc<Weights>,
    // Fallback for decisions the policy network doesn't cover (mid-effect choices).
    fallback: AdvancedCpuPlayer,
}

impl MlPlayer {
    pub fn new(name: impl Into<String>, weights: Arc<Weights>) -> Self {
        let n = name.into();
        let fallback = AdvancedCpuPlayer::new(n.clone());
        Self { name: n, weights, fallback }
    }

    fn net(&self) -> Net {
        Net::from_weights(&self.weights)
    }

    fn pick_action(&self, view: &GameView, legal: &[Action]) -> usize {
        let net = self.net();
        let features = encode(view);
        let mask = legal_mask(legal, &view.my_hand);
        let (logits, _value, body) = net.infer_with_body(&features);
        let probs = softmax_probs(&logits, mask);

        // Greedy: pick the abstract action with highest probability.
        let mut best_prob = -1.0f32;
        let mut best_abs  = 0usize;
        for (abs_idx, &prob) in probs.iter().enumerate() {
            if prob <= best_prob { continue; }
            if abstract_to_action(abs_idx, legal, &view.my_hand).is_some() {
                best_prob = prob;
                best_abs  = abs_idx;
            }
        }

        // Use candidate head to break ties among concrete actions mapping to the same abstract.
        self.best_concrete_for_abstract(best_abs, legal, view, &net, &body)
            .unwrap_or_else(|| self.fallback.choose_action(view, legal))
    }

    // Find the best concrete action for a given abstract index, using the candidate head
    // when multiple concrete actions map to the same abstract.
    fn best_concrete_for_abstract(
        &self,
        abs_idx: usize,
        legal:   &[Action],
        view:    &GameView,
        net:     &Net,
        body:    &Array1<f32>,
    ) -> Option<usize> {
        use novacana_engine::player::Action;
        use crate::candidate::CAND_DIM;

        let hand = &view.my_hand;
        let matching: Vec<usize> = legal.iter().enumerate()
            .filter(|(_, a)| crate::action::action_to_abstract(a, hand) == Some(abs_idx))
            .map(|(i, _)| i)
            .collect();

        if matching.is_empty() { return None; }
        if matching.len() == 1 { return Some(matching[0]); }

        // Multiple concrete actions — score each via candidate head.
        // Encode the differentiating feature of each concrete action.
        let total = matching.len();
        let cands: Vec<[f32; CAND_DIM]> = matching.iter().enumerate().map(|(slot, &ci)| {
            match &legal[ci] {
                Action::PlayDiscovery { onto_star, .. } => {
                    // Differentiating factor: which system we'd attach the discovery to.
                    let sys = view.my_star_chart.systems.iter()
                        .find(|s| s.star_id == *onto_star);
                    sys.map(|s| encode_system_candidate(s, false, slot, total))
                       .unwrap_or([0.0; CAND_DIM])
                }
                Action::PlayStar(id) | Action::PlayFlare(id) | Action::PlayFracture(id) => {
                    let card = hand.iter().find(|c| c.id == *id);
                    card.map(|c| encode_card_candidate(c, slot, total))
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

        let scores = net.score_candidates(body, &cands);
        let best = scores.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        Some(matching[best])
    }
}

impl Player for MlPlayer {
    fn name(&self) -> &str { &self.name }

    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        self.pick_action(view, legal)
    }

    fn respond_to_chain(&self, view: &GameView, _chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        let net = self.net();
        let features = encode(view);
        let mask = chain_legal_mask(legal, &view.my_hand);
        let (logits, _value) = net.infer(&features);
        let probs = softmax_probs(&logits, mask);

        // Highest-prob legal chain action.
        let mut best_prob = probs[0]; // pass probability
        let mut best_abs  = 0usize;
        for (abs_idx, &prob) in probs.iter().enumerate().skip(1) {
            if prob > best_prob {
                if abstract_to_chain(abs_idx, legal, &view.my_hand).is_some() {
                    best_prob = prob;
                    best_abs  = abs_idx;
                }
            }
        }

        if best_abs == 0 {
            None // pass
        } else {
            abstract_to_chain(best_abs, legal, &view.my_hand)
                .and_then(|r| legal.iter().position(|x| std::ptr::eq(x, r)))
        }
    }

    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        let total = req.options.len();
        if total == 0 { return 0; }
        let net = self.net();
        let features = encode(view);
        let body = net.body_forward(&features);
        let cands: Vec<_> = (0..total).map(|i| {
            let mut c = [0.0f32; crate::candidate::CAND_DIM];
            c[22] = i as f32 / total as f32;
            c[23] = total as f32 / 10.0;
            c
        }).collect();
        let scores = net.score_candidates(&body, &cands);
        scores.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn pick_targets(&self, view: &GameView, _prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
        if options.is_empty() { return Vec::new(); }
        let net = self.net();
        let features = encode(view);
        let body = net.body_forward(&features);
        let total = options.len();
        let cands: Vec<_> = options.iter().enumerate()
            .map(|(i, opt)| encode_target_candidate(opt, view, i, total))
            .collect();
        let scores = net.score_candidates(&body, &cands);
        // Take the top `count` indices by score.
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        order.truncate(count);
        order
    }

    fn choose_discards(&self, view: &GameView, hand: &[Card], n: usize) -> Vec<CardId> {
        if hand.is_empty() { return Vec::new(); }
        let net = self.net();
        let features = encode(view);
        let body = net.body_forward(&features);
        let total = hand.len();
        let cands: Vec<_> = hand.iter().enumerate()
            .map(|(i, card)| encode_card_candidate(card, i, total))
            .collect();
        let scores = net.score_candidates(&body, &cands);
        // Discard the n highest-scored cards (candidate head is trained to score desirability of
        // KEEPING; here we interpret it as discard priority — trained via imitation so sign is
        // established by the training data).
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        order.truncate(n);
        order.iter().map(|&i| hand[i].id).collect()
    }
}
