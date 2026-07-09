mod action;
mod candidate;
mod feature;
mod mcts;
mod net;
mod net_burn;
mod self_play;

use std::sync::{Arc, Mutex};
use std::time::Instant;

use rand::seq::SliceRandom;
use rand::thread_rng;

use novacana_engine::{
    card::{Card, CardId},
    chain::ChainView,
    cpu::AdvancedCpuPlayer,
    deck::Deck,
    game::{GameEngine, GameState},
    player::{Action, ChainAction, ChoiceRequest, GameView, Player, TargetOption},
};

use burn::backend::Autodiff;
use burn::module::AutodiffModule;
use burn::tensor::backend::AutodiffBackend;
use burn_ndarray::{NdArray, NdArrayDevice};
#[cfg(feature = "gpu")]
use burn_wgpu::{Wgpu, WgpuDevice};

use action::{action_to_abstract, chain_legal_mask, chain_response_to_abstract, legal_mask};
use candidate::{CAND_DIM, encode_action_candidate, encode_card_candidate, encode_decide_candidate, encode_target_candidate};
use feature::encode;
use mcts::StateStack;
use net::{Net, Weights, INPUT_DIM, masked_softmax};
use net_burn::{
    NetBurn, MAX_CANDS, make_optimizer,
    rl_train_step_burn, az_train_step_burn, cand_step_burn,
    pretrain_value_step_burn, pretrain_joint_step_burn, policy_eval_burn,
};
use self_play::{run_self_play, run_self_play_az, AzSample, PerBuffer};


// ── Eval player (greedy + candidate head) ────────────────────────────────────

struct MlPlayer {
    name_str: String,
    net:      Arc<Net>,
    stack:    Mutex<StateStack>,
}

impl MlPlayer {
    fn new(name: impl Into<String>, net: Arc<Net>) -> Self {
        Self { name_str: name.into(), net, stack: Mutex::new(StateStack::new()) }
    }

    fn features(&self, view: &GameView) -> [f32; INPUT_DIM] {
        let snap = encode(view);
        let mut st = self.stack.lock().unwrap();
        st.push(snap);
        st.stacked()
    }
}

impl Player for MlPlayer {
    fn name(&self) -> &str { &self.name_str }

    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        let features = self.features(view);
        let mask = legal_mask(legal, &view.my_hand);
        let (abs_idx, _lp, _val, body) =
            self.net.infer_stochastic_with_body(&features, mask, 0.1, rand::random());
        let hand = &view.my_hand;
        let matching: Vec<usize> = legal.iter().enumerate()
            .filter(|(_, a)| action_to_abstract(a, hand) == Some(abs_idx))
            .map(|(i, _)| i).collect();
        if matching.is_empty() { return 0; }
        if matching.len() == 1 { return matching[0]; }
        let nc = matching.len().min(MAX_CANDS);
        let cands: Vec<[f32; CAND_DIM]> = matching[..nc].iter().enumerate()
            .map(|(slot, &ci)| encode_action_candidate(&legal[ci], view, slot, nc))
            .collect();
        let scores = self.net.score_candidates(&body, &cands);
        let best = scores.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        matching[best]
    }

    fn respond_to_chain(&self, view: &GameView, _chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        let features = self.features(view);
        let mask = chain_legal_mask(legal, &view.my_hand);
        let (abs_idx, _, _, _) = self.net.infer_stochastic_with_body(&features, mask, 0.1, rand::random());
        if abs_idx == 0 { return None; }
        action::abstract_to_chain(abs_idx, legal, &view.my_hand)
            .and_then(|r| legal.iter().position(|x| std::ptr::eq(x, r)))
    }

    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        let total = req.options.len();
        if total == 0 { return 0; }
        let features = self.features(view);
        let body = self.net.body_forward(&features);
        let cands: Vec<[f32; CAND_DIM]> = req.options.iter().enumerate().take(MAX_CANDS)
            .map(|(i, opt)| encode_decide_candidate(&req.prompt, opt, view, i, total))
            .collect();
        self.net.score_candidates(&body, &cands).iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0)
    }

    fn pick_targets(&self, view: &GameView, _: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
        if options.is_empty() { return Vec::new(); }
        let features = self.features(view);
        let body = self.net.body_forward(&features);
        let total = options.len();
        let cands: Vec<[f32; CAND_DIM]> = options.iter().enumerate()
            .map(|(i, opt)| encode_target_candidate(opt, view, i, total)).collect();
        let scores = self.net.score_candidates(&body, &cands);
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        order.truncate(count);
        order
    }

    fn choose_discards(&self, view: &GameView, hand: &[Card], n: usize) -> Vec<CardId> {
        if hand.is_empty() { return Vec::new(); }
        let features = self.features(view);
        let body = self.net.body_forward(&features);
        let total = hand.len();
        let cands: Vec<[f32; CAND_DIM]> = hand.iter().enumerate()
            .map(|(i, card)| encode_card_candidate(card, i, total)).collect();
        let scores = self.net.score_candidates(&body, &cands);
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        order.truncate(n);
        order.iter().map(|&i| hand[i].id).collect()
    }
}

// ── Inference-fidelity player ─────────────────────────────────────────────────

struct FidelityRec {
    turn:                 usize,
    dtype:                u8,    // 0 = choose_action, 1 = respond_to_chain
    agreed:               bool,
    top2:                 bool,  // advcpu_abs is #1 or #2 in masked probs
    prob_mass_on_advcpu:  f32,
    unmasked_max_illegal: bool,
    legal_set_size:       usize,
    advcpu_in_mask:       bool,  // false = action_to_abstract returned None → not in mask
    ml_abs:               u8,
    advcpu_abs:           u8,
}

struct FidelityPlayer {
    name_str: String,
    net:      Arc<Net>,
    stack:    Mutex<StateStack>,
    fallback: AdvancedCpuPlayer,
    records:  Arc<Mutex<Vec<FidelityRec>>>,
    turn:     Mutex<usize>,
}

impl Player for FidelityPlayer {
    fn name(&self) -> &str { &self.name_str }

    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        // Exact SelfPlayPlayer path: encode → push frame → stack
        let features = {
            let snap = encode(view);
            let mut st = self.stack.lock().unwrap();
            st.push(snap);
            st.stacked()
        };
        let mask = legal_mask(legal, &view.my_hand);
        let advcpu_idx = self.fallback.choose_action(view, legal);
        // Detect None-mapping: if action_to_abstract returns None, AdvCPU's action has no
        // abstract representation → it won't be in the mask and ML can't agree.
        let advcpu_abs_opt = action_to_abstract(&legal[advcpu_idx], &view.my_hand);
        let advcpu_abs     = advcpu_abs_opt.unwrap_or(0);
        let advcpu_in_mask = advcpu_abs_opt.map(|a| mask & (1u64 << a) != 0).unwrap_or(false);

        let (logits, _val, _body) = self.net.infer_with_body(&features);
        let probs = masked_softmax(&logits, mask);

        // Rank legal actions by prob to get #1 and #2
        let mut ranked: Vec<(usize, f32)> = probs.iter().copied().enumerate()
            .filter(|&(i, _)| mask & (1u64 << i) != 0).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let ml_abs  = ranked.first().map(|&(i, _)| i).unwrap_or(0);
        let top2    = ranked.iter().take(2).any(|&(i, _)| i == advcpu_abs);

        let unmasked_max = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        let mass = if advcpu_abs < probs.len() { probs[advcpu_abs] } else { 0.0 };

        let mut t = self.turn.lock().unwrap();
        self.records.lock().unwrap().push(FidelityRec {
            turn: *t, dtype: 0,
            agreed: ml_abs == advcpu_abs, top2, advcpu_in_mask,
            prob_mass_on_advcpu:  mass,
            unmasked_max_illegal: mask & (1u64 << unmasked_max) == 0,
            legal_set_size:       ranked.len(),
            ml_abs:    ml_abs    as u8,
            advcpu_abs: advcpu_abs as u8,
        });
        *t += 1;
        advcpu_idx
    }

    fn respond_to_chain(&self, view: &GameView, chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        let features = {
            let snap = encode(view);
            let mut st = self.stack.lock().unwrap();
            st.push(snap);
            st.stacked()
        };
        let mask = chain_legal_mask(legal, &view.my_hand);
        let advcpu_result = self.fallback.respond_to_chain(view, chain, legal);
        let advcpu_abs = advcpu_result
            .and_then(|i| chain_response_to_abstract(&legal[i], &view.my_hand))
            .unwrap_or(0);

        let (logits, _val, _body) = self.net.infer_with_body(&features);
        let probs = masked_softmax(&logits, mask);
        let mut ranked: Vec<(usize, f32)> = probs.iter().copied().enumerate()
            .filter(|&(i, _)| mask & (1u64 << i) != 0).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let ml_abs = ranked.first().map(|&(i, _)| i).unwrap_or(0);
        let top2   = ranked.iter().take(2).any(|&(i, _)| i == advcpu_abs);
        let unmasked_max = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        let mass = if advcpu_abs < probs.len() { probs[advcpu_abs] } else { 0.0 };
        let advcpu_in_mask = mask & (1u64 << advcpu_abs) != 0;

        let mut t = self.turn.lock().unwrap();
        self.records.lock().unwrap().push(FidelityRec {
            turn: *t, dtype: 1,
            agreed: ml_abs == advcpu_abs, top2, advcpu_in_mask,
            prob_mass_on_advcpu:  mass,
            unmasked_max_illegal: mask & (1u64 << unmasked_max) == 0,
            legal_set_size:       ranked.len(),
            ml_abs:     ml_abs    as u8,
            advcpu_abs: advcpu_abs as u8,
        });
        *t += 1;
        advcpu_result
    }

    // These three push a frame just like SelfPlayPlayer does, so the frame buffer
    // stays in sync with what the training path (AdvCpuRecorder) sees.
    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        { let snap = encode(view); let mut st = self.stack.lock().unwrap(); st.push(snap); }
        self.fallback.decide(view, req)
    }
    fn pick_targets(&self, view: &GameView, p: &str, opts: &[TargetOption], n: usize) -> Vec<usize> {
        { let snap = encode(view); let mut st = self.stack.lock().unwrap(); st.push(snap); }
        self.fallback.pick_targets(view, p, opts, n)
    }
    fn choose_discards(&self, view: &GameView, hand: &[Card], n: usize) -> Vec<CardId> {
        { let snap = encode(view); let mut st = self.stack.lock().unwrap(); st.push(snap); }
        self.fallback.choose_discards(view, hand, n)
    }
}

fn cmd_fidelity(args: &[String]) {
    let model_path  = flag_str(args, "--model", "models/pretrain.bin");
    let num_games   = flag_usize(args, "--games", 500);
    let num_players = flag_usize(args, "--players", 4);
    let weights = net::load_weights(&model_path)
        .unwrap_or_else(|e| panic!("Failed to load {}: {}", model_path, e));
    let net = Arc::new(Net::from_weights(&weights));
    println!("Fidelity test: {} games × {} players, model={}", num_games, num_players, model_path);

    let mut all: Vec<FidelityRec> = Vec::new();
    for _ in 0..num_games {
        let records: Arc<Mutex<Vec<FidelityRec>>> = Arc::new(Mutex::new(Vec::new()));
        let player = FidelityPlayer {
            name_str: "Fidelity".into(),
            net:      Arc::clone(&net),
            stack:    Mutex::new(StateStack::new()),
            fallback: AdvancedCpuPlayer::new("Fidelity".to_string()),
            records:  Arc::clone(&records),
            turn:     Mutex::new(0),
        };
        let mut players: Vec<Box<dyn Player>> = vec![Box::new(player)];
        for i in 1..num_players { players.push(Box::new(AdvancedCpuPlayer::new(format!("CPU{}", i)))); }
        let mut engine = GameEngine::new(Deck::build_classic(), players);
        engine.set_silent();
        engine.set_max_turns(1000);
        engine.run();
        drop(engine);
        let recs = Arc::try_unwrap(records)
            .unwrap_or_else(|_| panic!("records Arc still held after engine drop"))
            .into_inner().unwrap();
        all.extend(recs);
    }

    let report = |label: &str, recs: &[&FidelityRec]| {
        if recs.is_empty() { return; }
        let n         = recs.len();
        let agreed    = recs.iter().filter(|r| r.agreed).count();
        let top2_n    = recs.iter().filter(|r| r.top2).count();
        let in_mask_n = recs.iter().filter(|r| r.advcpu_in_mask).count();
        let mean_legal= recs.iter().map(|r| r.legal_set_size as f32).sum::<f32>() / n as f32;
        let agrees_v: Vec<&&FidelityRec> = recs.iter().filter(|r| r.agreed).collect();
        let misses: Vec<&&FidelityRec>   = recs.iter().filter(|r| !r.agreed).collect();
        let agree_mass = if agrees_v.is_empty() { 0.0 } else {
            agrees_v.iter().map(|r| r.prob_mass_on_advcpu).sum::<f32>() / agrees_v.len() as f32
        };
        let miss_mass = if misses.is_empty() { 0.0 } else {
            misses.iter().map(|r| r.prob_mass_on_advcpu).sum::<f32>() / misses.len() as f32
        };
        let agree_ce = -agree_mass.max(1e-9).ln();
        let miss_ce  = -miss_mass.max(1e-9).ln();
        let mean_ce  = agrees_v.len() as f32 / n as f32 * agree_ce
                     + misses.len()  as f32 / n as f32 * miss_ce;
        println!("  {:<22}  agree={:5.1}%  top2={:5.1}%  agree_mass={:.3}(CE={:.2})  miss_mass={:.3}(CE={:.2})  mean_CE={:.2}  in_mask={:5.1}%  legal_avg={:.1}  n={}",
                 label,
                 100.0 * agreed as f32 / n as f32,
                 100.0 * top2_n as f32 / n as f32,
                 agree_mass, agree_ce,
                 miss_mass, miss_ce,
                 mean_ce,
                 100.0 * in_mask_n as f32 / n as f32,
                 mean_legal, n);
    };

    println!("\n--- choose_action ---");
    let ca: Vec<&FidelityRec> = all.iter().filter(|r| r.dtype == 0).collect();
    report("overall", &ca);
    for (lo, hi, lbl) in [(0,4,"turns  1–5"), (5,14,"turns  6–15"), (15,29,"turns 16–30"), (30,usize::MAX,"turns 31+")] {
        let sub: Vec<&FidelityRec> = ca.iter().copied().filter(|r| r.turn >= lo && r.turn <= hi).collect();
        report(lbl, &sub);
    }
    // Bucket misses by legal-set size (difficulty curve)
    println!("  -- misses by legal-set size (difficulty curve) --");
    for sz in 1usize..=8 {
        let sub: Vec<&FidelityRec> = ca.iter().copied().filter(|r| r.legal_set_size == sz).collect();
        if sub.is_empty() { continue; }
        let ag = sub.iter().filter(|r| r.agreed).count();
        let t2 = sub.iter().filter(|r| r.top2).count();
        let im = sub.iter().filter(|r| r.advcpu_in_mask).count();
        let miss: Vec<&&FidelityRec> = sub.iter().filter(|r| !r.agreed).collect();
        let mm = if miss.is_empty() { 0.0 } else {
            miss.iter().map(|r| r.prob_mass_on_advcpu).sum::<f32>() / miss.len() as f32
        };
        println!("    legal={sz}  agree={:5.1}%  top2={:5.1}%  miss_mass={:.3}  in_mask={:5.1}%  n={}",
                 100.0*ag as f32/sub.len() as f32, 100.0*t2 as f32/sub.len() as f32,
                 mm, 100.0*im as f32/sub.len() as f32, sub.len());
    }
    let sz_many: Vec<&FidelityRec> = ca.iter().copied().filter(|r| r.legal_set_size >= 9).collect();
    if !sz_many.is_empty() {
        let ag = sz_many.iter().filter(|r| r.agreed).count();
        let t2 = sz_many.iter().filter(|r| r.top2).count();
        let im = sz_many.iter().filter(|r| r.advcpu_in_mask).count();
        let miss: Vec<&&FidelityRec> = sz_many.iter().filter(|r| !r.agreed).collect();
        let mm = if miss.is_empty() { 0.0 } else {
            miss.iter().map(|r| r.prob_mass_on_advcpu).sum::<f32>() / miss.len() as f32
        };
        println!("    legal=9+  agree={:5.1}%  top2={:5.1}%  miss_mass={:.3}  in_mask={:5.1}%  n={}",
                 100.0*ag as f32/sz_many.len() as f32, 100.0*t2 as f32/sz_many.len() as f32,
                 mm, 100.0*im as f32/sz_many.len() as f32, sz_many.len());
    }

    // Sample 20 disagreement cases
    println!("\n  -- 20 sample choose_action disagreements (ml_abs / advcpu_abs / mass / legal_sz / in_mask) --");
    let mut misses: Vec<&FidelityRec> = ca.iter().copied().filter(|r| !r.agreed).collect();
    misses.sort_by(|a, b| a.prob_mass_on_advcpu.partial_cmp(&b.prob_mass_on_advcpu).unwrap());
    for r in misses.iter().take(10) {
        println!("    turn={:3}  ml={:2} adv={:2}  mass={:.3}  legal={}  in_mask={}  [low-mass miss]",
                 r.turn, r.ml_abs, r.advcpu_abs, r.prob_mass_on_advcpu, r.legal_set_size, r.advcpu_in_mask);
    }
    for r in misses.iter().rev().take(10) {
        println!("    turn={:3}  ml={:2} adv={:2}  mass={:.3}  legal={}  in_mask={}  [high-mass miss]",
                 r.turn, r.ml_abs, r.advcpu_abs, r.prob_mass_on_advcpu, r.legal_set_size, r.advcpu_in_mask);
    }

    println!("\n--- respond_to_chain ---");
    let rc: Vec<&FidelityRec> = all.iter().filter(|r| r.dtype == 1).collect();
    report("overall", &rc);

    println!("\nDiagnostic thresholds: top2 ~85%+ = difficulty (#1); top2 barely above agree = bug (#2/#3)");
    println!("  miss_mass near zero = structured wrong picks; miss_mass 0.2-0.35 = appropriate uncertainty");
}

// ── Hybrid eval player (ML policy, AdvCPU candidates) ────────────────────────
// Diagnostic: isolates policy head quality from candidate head quality.

struct MlHybridPlayer {
    name_str: String,
    net:      Arc<Net>,
    stack:    Mutex<StateStack>,
    fallback: AdvancedCpuPlayer,
}

impl MlHybridPlayer {
    fn new(name: impl Into<String>, net: Arc<Net>) -> Self {
        let n = name.into();
        Self {
            fallback: AdvancedCpuPlayer::new(n.clone()),
            name_str: n,
            net,
            stack: Mutex::new(StateStack::new()),
        }
    }

    fn features(&self, view: &GameView) -> [f32; INPUT_DIM] {
        let snap = encode(view);
        let mut st = self.stack.lock().unwrap();
        st.push(snap);
        st.stacked()
    }
}

impl Player for MlHybridPlayer {
    fn name(&self) -> &str { &self.name_str }

    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        let features = self.features(view);
        let mask = legal_mask(legal, &view.my_hand);
        let (abs_idx, _lp, _val, body) =
            self.net.infer_stochastic_with_body(&features, mask, 0.1, rand::random());
        let hand = &view.my_hand;
        let matching: Vec<usize> = legal.iter().enumerate()
            .filter(|(_, a)| action_to_abstract(a, hand) == Some(abs_idx))
            .map(|(i, _)| i).collect();
        if matching.is_empty() { return 0; }
        if matching.len() == 1 { return matching[0]; }
        let nc = matching.len().min(MAX_CANDS);
        let cands: Vec<[f32; CAND_DIM]> = matching[..nc].iter().enumerate()
            .map(|(slot, &ci)| encode_action_candidate(&legal[ci], view, slot, nc))
            .collect();
        let scores = self.net.score_candidates(&body, &cands);
        let best = scores.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        matching[best]
    }

    fn respond_to_chain(&self, view: &GameView, chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        let features = self.features(view);
        let mask = chain_legal_mask(legal, &view.my_hand);
        let (abs_idx, _, _, _) = self.net.infer_stochastic_with_body(&features, mask, 0.1, rand::random());
        if abs_idx == 0 { return None; }
        action::abstract_to_chain(abs_idx, legal, &view.my_hand)
            .and_then(|r| legal.iter().position(|x| std::ptr::eq(x, r)))
    }

    // These three push a frame just like SelfPlayPlayer does, so the frame buffer
    // stays in sync with what the training path (AdvCpuRecorder) sees.
    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        { let snap = encode(view); let mut st = self.stack.lock().unwrap(); st.push(snap); }
        self.fallback.decide(view, req)
    }

    fn pick_targets(&self, view: &GameView, prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
        { let snap = encode(view); let mut st = self.stack.lock().unwrap(); st.push(snap); }
        self.fallback.pick_targets(view, prompt, options, count)
    }

    fn choose_discards(&self, view: &GameView, hand: &[Card], n: usize) -> Vec<CardId> {
        { let snap = encode(view); let mut st = self.stack.lock().unwrap(); st.push(snap); }
        self.fallback.choose_discards(view, hand, n)
    }
}

// ── Lookahead eval player (ML policy, 1-step value lookahead candidates) ─────
// Like MlHybridPlayer (AdvCPU aux paths) but disambiguates concrete actions by
// applying each candidate to a cloned state and reading the value head on the
// successor (mcts::value_lookahead) — no candidate head involved.

struct MlLookaheadPlayer {
    name_str: String,
    net:      Arc<Net>,
    stack:    Mutex<StateStack>,
    fallback: AdvancedCpuPlayer,
    aux_ml:   bool,   // true: candidate head handles decide/targets/discards; false: AdvCPU
}

impl MlLookaheadPlayer {
    fn new(name: impl Into<String>, net: Arc<Net>, aux_ml: bool) -> Self {
        let n = name.into();
        Self {
            fallback: AdvancedCpuPlayer::new(n.clone()),
            name_str: n,
            net,
            stack: Mutex::new(StateStack::new()),
            aux_ml,
        }
    }

    fn features(&self, view: &GameView) -> [f32; INPUT_DIM] {
        let snap = encode(view);
        let mut st = self.stack.lock().unwrap();
        st.push(snap);
        st.stacked()
    }
}

impl Player for MlLookaheadPlayer {
    fn name(&self) -> &str { &self.name_str }

    // Without a state we can't simulate; fall back to the candidate head.
    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        let features = self.features(view);
        let mask = legal_mask(legal, &view.my_hand);
        let (abs_idx, _lp, _val, body) =
            self.net.infer_stochastic_with_body(&features, mask, 0.1, rand::random());
        let hand = &view.my_hand;
        let matching: Vec<usize> = legal.iter().enumerate()
            .filter(|(_, a)| action_to_abstract(a, hand) == Some(abs_idx))
            .map(|(i, _)| i).collect();
        if matching.is_empty() { return 0; }
        if matching.len() == 1 { return matching[0]; }
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
        let features = self.features(view);
        let mask = legal_mask(legal, &view.my_hand);
        let (abs_idx, _lp, _val, _body) =
            self.net.infer_stochastic_with_body(&features, mask, 0.1, rand::random());
        let hand = &view.my_hand;
        let matching: Vec<usize> = legal.iter().enumerate()
            .filter(|(_, a)| action_to_abstract(a, hand) == Some(abs_idx))
            .map(|(i, _)| i).collect();
        if matching.is_empty() { return 0; }
        if matching.len() == 1 { return matching[0]; }
        // Cap the simulated candidates to bound per-decision cost.
        let my_idx = view.my_idx;
        matching.iter().take(16)
            .map(|&ci| (ci, mcts::value_lookahead(state, my_idx, &legal[ci], &self.net)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(ci, _)| ci)
            .unwrap_or(matching[0])
    }

    fn respond_to_chain(&self, view: &GameView, _chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        let features = self.features(view);
        let mask = chain_legal_mask(legal, &view.my_hand);
        let (abs_idx, _, _, _) = self.net.infer_stochastic_with_body(&features, mask, 0.1, rand::random());
        if abs_idx == 0 { return None; }
        action::abstract_to_chain(abs_idx, legal, &view.my_hand)
            .and_then(|r| legal.iter().position(|x| std::ptr::eq(x, r)))
    }

    // Aux paths: candidate head when aux_ml, else AdvCPU. Frames pushed either way.
    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        let total = req.options.len();
        if total == 0 { return 0; }
        let features = self.features(view);
        if !self.aux_ml { return self.fallback.decide(view, req); }
        let body = self.net.body_forward(&features);
        let cands: Vec<[f32; CAND_DIM]> = req.options.iter().enumerate().take(MAX_CANDS)
            .map(|(i, opt)| encode_decide_candidate(&req.prompt, opt, view, i, total))
            .collect();
        self.net.score_candidates(&body, &cands).iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0)
    }

    fn pick_targets(&self, view: &GameView, prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
        if options.is_empty() { return Vec::new(); }
        let features = self.features(view);
        if !self.aux_ml { return self.fallback.pick_targets(view, prompt, options, count); }
        let body = self.net.body_forward(&features);
        let total = options.len();
        let cands: Vec<[f32; CAND_DIM]> = options.iter().enumerate()
            .map(|(i, opt)| encode_target_candidate(opt, view, i, total)).collect();
        let scores = self.net.score_candidates(&body, &cands);
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        order.truncate(count);
        order
    }

    fn choose_discards(&self, view: &GameView, hand: &[Card], n: usize) -> Vec<CardId> {
        if hand.is_empty() { return Vec::new(); }
        let features = self.features(view);
        if !self.aux_ml { return self.fallback.choose_discards(view, hand, n); }
        let body = self.net.body_forward(&features);
        let total = hand.len();
        let cands: Vec<[f32; CAND_DIM]> = hand.iter().enumerate()
            .map(|(i, card)| encode_card_candidate(card, i, total)).collect();
        let scores = self.net.score_candidates(&body, &cands);
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        order.truncate(n);
        order.iter().map(|&i| hand[i].id).collect()
    }
}

// ── Imitation (AdvCPU recording) ──────────────────────────────────────────────

struct ImPending {
    features:    Vec<f32>,
    legal_mask:  u64,
    action:      u8,
    has_policy:  bool,
    cand_feats:  Vec<f32>,
    cand_chosen: u8,
    has_cand:    bool,
}

pub struct ImitateSample {
    pub features:    Vec<f32>,
    pub legal_mask:  u64,
    pub action:      u8,
    pub outcome:     f32,
    pub has_policy:  bool,
    pub cand_feats:  Vec<f32>,
    pub cand_chosen: u8,
    pub has_cand:    bool,
}

struct AdvCpuRecorder {
    name_str: String,
    fallback: AdvancedCpuPlayer,
    stack:    Mutex<StateStack>,
    buffer:   Arc<Mutex<Vec<ImPending>>>,
}

impl AdvCpuRecorder {
    fn new(name: impl Into<String>, buffer: Arc<Mutex<Vec<ImPending>>>) -> Self {
        let n = name.into();
        Self {
            fallback: AdvancedCpuPlayer::new(n.clone()),
            name_str: n,
            stack:    Mutex::new(StateStack::new()),
            buffer,
        }
    }

    fn encode_and_stack(&self, view: &GameView) -> [f32; INPUT_DIM] {
        let snap = encode(view);
        let mut st = self.stack.lock().unwrap();
        st.push(snap);
        st.stacked()
    }

    fn push(&self, p: ImPending) {
        self.buffer.lock().unwrap().push(p);
    }
}

impl Player for AdvCpuRecorder {
    fn name(&self) -> &str { &self.name_str }

    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        let features = self.encode_and_stack(view);
        let mask = legal_mask(legal, &view.my_hand);
        let chosen_idx = self.fallback.choose_action(view, legal);
        let hand = &view.my_hand;
        let abs = action_to_abstract(&legal[chosen_idx], hand).unwrap_or(0);
        let matching: Vec<usize> = legal.iter().enumerate()
            .filter(|(_, a)| action_to_abstract(a, hand) == Some(abs))
            .map(|(i, _)| i).collect();
        let (cand_feats, cand_chosen, has_cand) = if matching.len() >= 2 {
            let slot = matching.iter().position(|&ci| ci == chosen_idx).unwrap_or(0);
            if slot < MAX_CANDS {
                let nc = matching.len().min(MAX_CANDS);
                let cands: Vec<[f32; CAND_DIM]> = matching[..nc].iter().enumerate()
                    .map(|(s, &ci)| encode_action_candidate(&legal[ci], view, s, nc))
                    .collect();
                let flat = cands.iter().flat_map(|c| c.iter().copied()).collect();
                (flat, slot as u8, true)
            } else { (Vec::new(), 0, false) }
        } else { (Vec::new(), 0, false) };
        self.push(ImPending {
            features: features.to_vec(), legal_mask: mask, action: abs as u8,
            has_policy: true, cand_feats, cand_chosen, has_cand,
        });
        chosen_idx
    }

    fn respond_to_chain(&self, view: &GameView, chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        let features = self.encode_and_stack(view);
        let mask = chain_legal_mask(legal, &view.my_hand);
        let result = self.fallback.respond_to_chain(view, chain, legal);
        let abs = result.and_then(|i| chain_response_to_abstract(&legal[i], &view.my_hand)).unwrap_or(0);
        self.push(ImPending {
            features: features.to_vec(), legal_mask: mask, action: abs as u8,
            has_policy: true, cand_feats: Vec::new(), cand_chosen: 0, has_cand: false,
        });
        result
    }

    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        let total = req.options.len();
        if total == 0 { return 0; }
        let features = self.encode_and_stack(view);
        let chosen = self.fallback.decide(view, req);
        if total >= 2 && chosen < MAX_CANDS {
            let nc = total.min(MAX_CANDS);
            let cands: Vec<[f32; CAND_DIM]> = req.options.iter().take(nc).enumerate()
                .map(|(i, opt)| encode_decide_candidate(&req.prompt, opt, view, i, nc))
                .collect();
            self.push(ImPending {
                features: features.to_vec(), legal_mask: 0, action: 0, has_policy: false,
                cand_feats: cands.iter().flat_map(|c| c.iter().copied()).collect(),
                cand_chosen: chosen as u8, has_cand: true,
            });
        }
        chosen
    }

    fn pick_targets(&self, view: &GameView, prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
        if options.is_empty() { return Vec::new(); }
        let features = self.encode_and_stack(view);
        let chosen = self.fallback.pick_targets(view, prompt, options, count);
        if options.len() >= 2 && !chosen.is_empty() && chosen[0] < MAX_CANDS {
            let nc = options.len().min(MAX_CANDS);
            let cands: Vec<[f32; CAND_DIM]> = options.iter().take(nc).enumerate()
                .map(|(i, opt)| encode_target_candidate(opt, view, i, nc)).collect();
            self.push(ImPending {
                features: features.to_vec(), legal_mask: 0, action: 0, has_policy: false,
                cand_feats: cands.iter().flat_map(|c| c.iter().copied()).collect(),
                cand_chosen: chosen[0] as u8, has_cand: true,
            });
        }
        chosen
    }

    fn choose_discards(&self, view: &GameView, hand: &[Card], n: usize) -> Vec<CardId> {
        if hand.is_empty() { return Vec::new(); }
        let features = self.encode_and_stack(view);
        let chosen_ids = self.fallback.choose_discards(view, hand, n);
        if hand.len() >= 2 && !chosen_ids.is_empty() {
            let slot = hand.iter().position(|c| c.id == chosen_ids[0]).unwrap_or(0);
            if slot < MAX_CANDS {
                let nc = hand.len().min(MAX_CANDS);
                let cands: Vec<[f32; CAND_DIM]> = hand.iter().take(nc).enumerate()
                    .map(|(i, card)| encode_card_candidate(card, i, nc)).collect();
                self.push(ImPending {
                    features: features.to_vec(), legal_mask: 0, action: 0, has_policy: false,
                    cand_feats: cands.iter().flat_map(|c| c.iter().copied()).collect(),
                    cand_chosen: slot as u8, has_cand: true,
                });
            }
        }
        chosen_ids
    }
}

fn run_one_imitate_game(num_players: usize) -> Vec<ImitateSample> {
    let mut bufs:    Vec<Arc<Mutex<Vec<ImPending>>>> = Vec::new();
    let mut players: Vec<Box<dyn Player>>             = Vec::new();
    for i in 0..num_players {
        let buf = Arc::new(Mutex::new(Vec::new()));
        bufs.push(Arc::clone(&buf));
        players.push(Box::new(AdvCpuRecorder::new(format!("AdvCPU{}", i + 1), buf)));
    }
    let mut engine = GameEngine::new(Deck::build_classic(), players);
    engine.set_silent();
    engine.set_max_turns(1000);
    let result = engine.run();
    let mut out = Vec::new();
    for (player_idx, buf) in bufs.into_iter().enumerate() {
        let outcome = if result.winner_idx == player_idx { 1.0f32 } else { 0.0 };
        let pending = Arc::try_unwrap(buf)
            .unwrap_or_else(|b| Mutex::new(std::mem::take(&mut *b.lock().unwrap())))
            .into_inner().unwrap();
        for p in pending {
            out.push(ImitateSample {
                features: p.features, legal_mask: p.legal_mask, action: p.action,
                outcome, has_policy: p.has_policy, cand_feats: p.cand_feats,
                cand_chosen: p.cand_chosen, has_cand: p.has_cand,
            });
        }
    }
    out
}

fn generate_imitate_data(num_games: usize, num_players: usize) -> Vec<ImitateSample> {
    use rayon::prelude::*;
    (0..num_games).into_par_iter()
        .flat_map(|_| run_one_imitate_game(num_players))
        .collect()
}

// ── Eval ──────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum EvalMode {
    Full,      // MlPlayer: ML policy + candidate head everywhere
    Hybrid,    // MlHybridPlayer: ML policy + cand head; AdvCPU aux paths
    Lookahead, // MlLookaheadPlayer: ML policy + 1-step value lookahead; AdvCPU aux
    FullLookahead, // as Lookahead but candidate head handles the aux paths
}

fn eval(net: &Arc<Net>, num_games: usize, num_players: usize, mode: EvalMode) -> f32 {
    let mut wins = 0usize;
    for _ in 0..num_games {
        let p0: Box<dyn Player> = match mode {
            EvalMode::Hybrid    => Box::new(MlHybridPlayer::new("Astra-6", Arc::clone(net))),
            EvalMode::Lookahead => Box::new(MlLookaheadPlayer::new("Astra-6", Arc::clone(net), false)),
            EvalMode::FullLookahead => Box::new(MlLookaheadPlayer::new("Astra-6", Arc::clone(net), true)),
            EvalMode::Full      => Box::new(MlPlayer::new("Astra-6", Arc::clone(net))),
        };
        let mut players: Vec<Box<dyn Player>> = vec![p0];
        for i in 1..num_players {
            players.push(Box::new(AdvancedCpuPlayer::new(format!("CPU{}", i))));
        }
        let mut engine = GameEngine::new(Deck::build_classic(), players);
        engine.set_silent();
        engine.set_max_turns(1000);
        if engine.run().winner_idx == 0 { wins += 1; }
    }
    wins as f32 / num_games as f32
}

// ── CPU-vs-CPU bench (no ML) ─────────────────────────────────────────────────

fn cmd_cpubench(args: &[String]) {
    use novacana_engine::cpu::ExpertCpuPlayer;
    use rayon::prelude::*;
    let num_games   = flag_usize(args, "--games", 1000);
    let num_players = flag_usize(args, "--players", 4);
    println!("CPU bench: ExpertCpu vs {}× AdvCPU, {} games (seat rotated)", num_players - 1, num_games);
    let t0 = Instant::now();
    let wins: usize = (0..num_games).into_par_iter().map(|g| {
        let expert_seat = g % num_players;
        let players: Vec<Box<dyn Player>> = (0..num_players).map(|i| -> Box<dyn Player> {
            if i == expert_seat {
                Box::new(ExpertCpuPlayer::new(format!("Expert-P{}", i + 1)))
            } else {
                Box::new(AdvancedCpuPlayer::new(format!("CPU-P{}", i + 1)))
            }
        }).collect();
        let mut engine = GameEngine::new(Deck::build_classic(), players);
        engine.set_silent();
        engine.set_max_turns(1000);
        (engine.run().winner_idx == expert_seat) as usize
    }).sum();
    let wr = wins as f32 / num_games as f32;
    println!("ExpertCpu: {}/{} = {:.1}%  (parity {:.1}%, {:.0}s)",
        wins, num_games, wr * 100.0, 100.0 / num_players as f32, t0.elapsed().as_secs_f32());
}

// ── Expert eval-weight fitting data (states → outcome) ──────────────────────

struct ExpertFitRecorder {
    name_str: String,
    adv:      AdvancedCpuPlayer,
    rows:     Arc<Mutex<Vec<[f32; novacana_engine::cpu::EXPERT_NF]>>>,
}

impl Player for ExpertFitRecorder {
    fn name(&self) -> &str { &self.name_str }
    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        // Sample ~every 3rd decision to decorrelate within-game states.
        if rand::random::<f32>() < 0.33 {
            self.rows.lock().unwrap().push(novacana_engine::cpu::expert_features(view));
        }
        self.adv.choose_action(view, legal)
    }
    fn respond_to_chain(&self, v: &GameView, c: &ChainView, l: &[ChainAction]) -> Option<usize> {
        self.adv.respond_to_chain(v, c, l)
    }
    fn decide(&self, v: &GameView, r: &ChoiceRequest) -> usize { self.adv.decide(v, r) }
    fn pick_targets(&self, v: &GameView, p: &str, o: &[TargetOption], n: usize) -> Vec<usize> {
        self.adv.pick_targets(v, p, o, n)
    }
    fn choose_discards(&self, v: &GameView, h: &[Card], n: usize) -> Vec<CardId> {
        self.adv.choose_discards(v, h, n)
    }
}

// Recorder variant that PLAYS as ExpertCpu (sim search) while recording —
// used for the policy-iteration refit so weights are fit on states the
// improved policy actually reaches.
struct ExpertPlayRecorder {
    name_str: String,
    expert:   novacana_engine::cpu::ExpertCpuPlayer,
    rows:     Arc<Mutex<Vec<[f32; novacana_engine::cpu::EXPERT_NF]>>>,
}

impl Player for ExpertPlayRecorder {
    fn name(&self) -> &str { &self.name_str }
    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        self.expert.choose_action(view, legal)
    }
    fn choose_action_with_state(&self, view: &GameView, state: &GameState, legal: &[Action]) -> usize {
        if rand::random::<f32>() < 0.33 {
            self.rows.lock().unwrap().push(novacana_engine::cpu::expert_features(view));
        }
        self.expert.choose_action_with_state(view, state, legal)
    }
    fn respond_to_chain(&self, v: &GameView, c: &ChainView, l: &[ChainAction]) -> Option<usize> {
        self.expert.respond_to_chain(v, c, l)
    }
    fn decide(&self, v: &GameView, r: &ChoiceRequest) -> usize { self.expert.decide(v, r) }
    fn pick_targets(&self, v: &GameView, p: &str, o: &[TargetOption], n: usize) -> Vec<usize> {
        self.expert.pick_targets(v, p, o, n)
    }
    fn choose_discards(&self, v: &GameView, h: &[Card], n: usize) -> Vec<CardId> {
        self.expert.choose_discards(v, h, n)
    }
}

fn cmd_expertfit(args: &[String]) {
    use rayon::prelude::*;
    use std::io::Write;
    let num_games  = flag_usize(args, "--games", 10_000);
    let out_path   = flag_str(args, "--out", "expertfit.tsv");
    let expert_gen = has_flag(args, "--expert");
    println!("expertfit: {} games ({}) → {}", num_games,
        if expert_gen { "Expert seat rotated vs 3 AdvCPU" } else { "4× AdvCPU" }, out_path);
    let t0 = Instant::now();
    let rows: Vec<(Vec<[f32; novacana_engine::cpu::EXPERT_NF]>, f32)> = (0..num_games)
        .into_par_iter()
        .flat_map(|g| {
            if expert_gen {
                let seat = g % 4;
                let buf: Arc<Mutex<Vec<_>>> = Arc::new(Mutex::new(Vec::new()));
                let players: Vec<Box<dyn Player>> = (0..4).map(|i| -> Box<dyn Player> {
                    if i == seat {
                        Box::new(ExpertPlayRecorder {
                            name_str: format!("E{}", i + 1),
                            expert: novacana_engine::cpu::ExpertCpuPlayer::new(format!("E{}", i + 1)),
                            rows: Arc::clone(&buf),
                        })
                    } else {
                        Box::new(AdvancedCpuPlayer::new(format!("R{}", i + 1)))
                    }
                }).collect();
                let mut engine = GameEngine::new(Deck::build_classic(), players);
                engine.set_silent();
                engine.set_max_turns(1000);
                let winner = engine.run().winner_idx;
                let v = std::mem::take(&mut *buf.lock().unwrap());
                return vec![(v, (winner == seat) as i32 as f32)];
            }
            let bufs: Vec<Arc<Mutex<Vec<_>>>> = (0..4).map(|_| Arc::new(Mutex::new(Vec::new()))).collect();
            let players: Vec<Box<dyn Player>> = (0..4).map(|i| -> Box<dyn Player> {
                Box::new(ExpertFitRecorder {
                    name_str: format!("R{}", i + 1),
                    adv: AdvancedCpuPlayer::new(format!("R{}", i + 1)),
                    rows: Arc::clone(&bufs[i]),
                })
            }).collect();
            let mut engine = GameEngine::new(Deck::build_classic(), players);
            engine.set_silent();
            engine.set_max_turns(1000);
            let winner = engine.run().winner_idx;
            bufs.into_iter().enumerate().map(|(i, b)| {
                let v = std::mem::take(&mut *b.lock().unwrap());
                (v, (i == winner) as i32 as f32)
            }).collect::<Vec<_>>()
        })
        .collect();
    let mut f = std::io::BufWriter::new(std::fs::File::create(&out_path).unwrap());
    let mut n = 0usize;
    for (feats, y) in &rows {
        for row in feats {
            let cols: Vec<String> = row.iter().map(|v| format!("{:.4}", v)).collect();
            writeln!(f, "{}\t{}", y, cols.join("\t")).unwrap();
            n += 1;
        }
    }
    println!("{} rows in {:.1}s", n, t0.elapsed().as_secs_f32());
}

// ── CLI helpers ───────────────────────────────────────────────────────────────

fn flag_usize(args: &[String], flag: &str, default: usize) -> usize {
    args.windows(2).find(|w| w[0] == flag).and_then(|w| w[1].parse().ok()).unwrap_or(default)
}
fn flag_f32(args: &[String], flag: &str, default: f32) -> f32 {
    args.windows(2).find(|w| w[0] == flag).and_then(|w| w[1].parse().ok()).unwrap_or(default)
}
fn flag_str(args: &[String], flag: &str, default: &str) -> String {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone()).unwrap_or_else(|| default.to_string())
}
fn has_flag(args: &[String], flag: &str) -> bool { args.iter().any(|a| a == flag) }

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if has_flag(&args, "pretrain")  { cmd_pretrain(&args);  return; }
    if has_flag(&args, "rl")        { cmd_rl(&args);        return; }
    if has_flag(&args, "az")        { cmd_az(&args);        return; }
    if has_flag(&args, "eval")      { cmd_eval(&args);      return; }
    if has_flag(&args, "fidelity")  { cmd_fidelity(&args);  return; }
    if has_flag(&args, "cpubench")  { cmd_cpubench(&args);  return; }
    if has_flag(&args, "expertfit") { cmd_expertfit(&args); return; }

    println!("Usage:");
    println!("  novacana-ml pretrain [options]");
    println!("  novacana-ml rl [options]");
    println!("  novacana-ml az [options]");
    println!("  novacana-ml eval --model <path> [--games 400] [--players 4] [--advcpu-cands | --lookahead-cands]");
    println!();
    println!("Pretrain options:");
    println!("  --out <path>          [models/astra6_pretrain.bin]");
    println!("  --model <path>        warm-start weights; cand head reinit on dim change");
    println!("  --games <n>           imitation games [20000]");
    println!("  --players <n>         [4]");
    println!("  --lr <f>              [0.0003]");
    println!("  --joint-epochs <n>    [30]");
    println!("  --value-epochs <n>    [10]");
    println!("  --cand-epochs <n>     [30]");
    println!("  --batch-size <n>      [512]");
    println!("  --chunk-games <n>     games per epoch instead of all upfront [0=all]");
    println!("  --label-smoothing <f> spread mass over legal set [0.05]");
    println!("  --val-games <n>       held-out games for best-policy checkpointing [500]");
    println!("  --no-freeze-trunk     train trunk during value/cand fine-tune (old behaviour)");
    println!();
    println!("RL options (PPO+PER):");
    println!("  --model <path>        starting weights (omit for random)");
    println!("  --out <path>          [models/net_v6.bin]");
    println!("  --iters <n>           [300]");
    println!("  --games <n>           games per iter [2000]");
    println!("  --players <n>         [4]");
    println!("  --lr <f>              [0.0001]");
    println!("  --temp-start/final    [1.0/0.3]");
    println!("  --entropy-start/final [0.01/0.001]");
    println!("  --ppo-epochs <n>      [4]");
    println!("  --batch-size <n>      [512]");
    println!("  --clip-eps <f>        [0.2]");
    println!("  --snapshot-every <n>  [20]");
    println!("  --pool-cap <n>        [10]");
    println!("  --per-alpha/beta/size [0.6/0.4/500000]");
    println!("  --gamma/lambda        [0.99/0.95]");
    println!("  --cand-lr-mult <f>    [3.0]");
    println!("  --advcpu-share <f>    [0.3]");
    println!("  --eval-every <n>      [50]");
    println!("  --eval-games <n>      [400]");
    println!();
    println!("AZ options (AlphaZero):");
    println!("  --model <path>        starting weights (omit for random)");
    println!("  --out <path>          [models/net_az.bin]");
    println!("  --iters <n>           [500]");
    println!("  --games <n>           games per iter [200]");
    println!("  --players <n>         [4]");
    println!("  --lr <f>              [0.0001]");
    println!("  --mcts-sims <n>       MCTS simulations per decision [50]");
    println!("  --c-puct <f>          PUCT exploration constant [1.5]");
    println!("  --az-epochs <n>       gradient passes per iter [4]");
    println!("  --az-explore          root Dirichlet noise + sampled openings (needs high sims)");
    println!("  --batch-size <n>      [512]");
    println!("  --window <n>          rolling sample window size [200000]");
    println!("  --snapshot-every <n>  [20]");
    println!("  --pool-cap <n>        [10]");
    println!("  --advcpu-share <f>    [0.3]");
    println!("  --cand-lr-mult <f>    [3.0]");
    println!("  --eval-every <n>      [50]");
    println!("  --eval-games <n>      [400]");
}

// ── Pretraining ───────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_pretrain_burn<B: AutodiffBackend>(
    initial_weights: Weights,
    out_path:        &str,
    num_games:       usize,
    num_players:     usize,
    lr:              f32,
    joint_epochs:    usize,
    value_epochs:    usize,
    cand_epochs:     usize,
    batch_size:      usize,
    chunk_games:     usize,
    label_smoothing: f32,
    freeze_trunk:    bool,
    val_games:       usize,
    device:          &B::Device,
) {
    let mut net = NetBurn::<B>::from_weights(&initial_weights, device);
    let mut opt = make_optimizer::<B>();
    let mut rng = thread_rng();

    // chunk_games == 0: generate all upfront and reuse across epochs (original behaviour)
    // chunk_games >  0: generate fresh data each epoch (memory-efficient streaming)
    let pregenerated: Option<Vec<ImitateSample>> = if chunk_games == 0 {
        println!("Generating imitation data: {} games × {} players...", num_games, num_players);
        let t0 = Instant::now();
        let s = generate_imitate_data(num_games, num_players);
        let pi = s.iter().filter(|x| x.has_policy).count();
        let ci = s.iter().filter(|x| x.has_cand).count();
        println!("  {} samples in {:.1}s  (policy={} cand={})", s.len(), t0.elapsed().as_secs_f32(), pi, ci);
        Some(s)
    } else {
        println!("Chunked pretrain: {} games/epoch × {} players", chunk_games, num_players);
        None
    };

    // Dedicated held-out validation set for policy-generalization checkpointing.
    // Generated independently of the training data (and from the same AdvCPU
    // process — no covariate shift), so val argmax-accuracy ≈ fidelity accuracy.
    let val_samples: Vec<ImitateSample> = if val_games > 0 {
        let v = generate_imitate_data(val_games, num_players);
        let vp = v.iter().filter(|s| s.has_policy).count();
        println!("Validation set: {} games → {} policy samples", val_games, vp);
        v
    } else { Vec::new() };
    let val_idx: Vec<usize> = val_samples.iter().enumerate()
        .filter(|(_, s)| s.has_policy).map(|(i, _)| i).collect();

    // Best-policy snapshot: checkpoint on val accuracy, not epoch count, to avoid
    // saving a regressed-policy epoch (the epoch-8→30 problem).
    let mut best_acc:     f32 = -1.0;
    let mut best_weights: Option<Weights> = None;

    println!("Joint policy+value pretraining ({} epochs, lr={}, label_smoothing={})...",
        joint_epochs, lr, label_smoothing);
    for epoch in 0..joint_epochs {
        let fresh: Vec<ImitateSample>;
        let samples: &[ImitateSample] = if let Some(ref s) = pregenerated { s } else {
            fresh = generate_imitate_data(chunk_games, num_players);
            &fresh
        };
        let mut idx: Vec<usize> = samples.iter().enumerate()
            .filter(|(_, s)| s.has_policy).map(|(i, _)| i).collect();
        idx.shuffle(&mut rng);
        let (mut tpl, mut tvl, mut nb) = (0.0f64, 0.0f64, 0usize);
        for chunk in idx.chunks(batch_size) {
            if chunk.len() < 2 { continue; }
            let feats:    Vec<Vec<f32>> = chunk.iter().map(|&i| samples[i].features.clone()).collect();
            let masks:    Vec<u64>      = chunk.iter().map(|&i| samples[i].legal_mask).collect();
            let actions:  Vec<u8>       = chunk.iter().map(|&i| samples[i].action).collect();
            let outcomes: Vec<f32>      = chunk.iter().map(|&i| samples[i].outcome).collect();
            let (nn, pl, vl) = pretrain_joint_step_burn(net, &mut opt, lr as f64, &feats, &masks, &actions, &outcomes, label_smoothing, device);
            net = nn; tpl += pl as f64; tvl += vl as f64; nb += 1;
        }

        // Held-out policy evaluation + best-checkpoint selection.
        // Runs on the inner (non-autodiff) net: forward passes on the autodiff
        // backend accumulate graph state that only backward() frees.
        if !val_idx.is_empty() {
            let vnet = net.valid();
            let (mut correct, mut ce, mut n) = (0usize, 0.0f32, 0usize);
            for chunk in val_idx.chunks(batch_size) {
                let feats:   Vec<Vec<f32>> = chunk.iter().map(|&i| val_samples[i].features.clone()).collect();
                let masks:   Vec<u64>      = chunk.iter().map(|&i| val_samples[i].legal_mask).collect();
                let actions: Vec<u8>       = chunk.iter().map(|&i| val_samples[i].action).collect();
                let (c, cs) = policy_eval_burn(&vnet, &feats, &masks, &actions, device);
                correct += c; ce += cs; n += chunk.len();
            }
            let acc = correct as f32 / n.max(1) as f32;
            let vce = ce / n.max(1) as f32;
            let star = if acc > best_acc { "  *best" } else { "" };
            println!("  epoch {}/{}: policy={:.4} value={:.4}  val_acc={:.4} val_ce={:.4}{}",
                epoch + 1, joint_epochs, tpl / nb.max(1) as f64, tvl / nb.max(1) as f64, acc, vce, star);
            if acc > best_acc {
                best_acc = acc;
                best_weights = Some(net.to_weights());
            }
        } else {
            println!("  epoch {}/{}: policy={:.4} value={:.4}",
                epoch + 1, joint_epochs, tpl / nb.max(1) as f64, tvl / nb.max(1) as f64);
        }
    }

    // Restore the best-policy trunk before value fine-tune so value MSE can't
    // regress the policy we just selected. Reset optimizer state to match.
    if let Some(ref w) = best_weights {
        println!("Restoring best-policy checkpoint (val_acc={:.4})", best_acc);
        net = NetBurn::<B>::from_weights(w, device);
        opt = make_optimizer::<B>();
    }

    let value_lr = (lr * 0.5) as f64;
    println!("Value fine-tune ({} epochs, lr={})...", value_epochs, value_lr);
    for epoch in 0..value_epochs {
        let fresh: Vec<ImitateSample>;
        let samples: &[ImitateSample] = if let Some(ref s) = pregenerated { s } else {
            fresh = generate_imitate_data(chunk_games, num_players);
            &fresh
        };
        let mut indices: Vec<usize> = (0..samples.len()).collect();
        indices.shuffle(&mut rng);
        let (mut tl, mut nb) = (0.0f64, 0usize);
        for chunk in indices.chunks(batch_size) {
            if chunk.len() < 2 { continue; }
            let feats:    Vec<Vec<f32>> = chunk.iter().map(|&i| samples[i].features.clone()).collect();
            let outcomes: Vec<f32>      = chunk.iter().map(|&i| samples[i].outcome).collect();
            let (nn, loss) = pretrain_value_step_burn(net, &mut opt, value_lr, &feats, &outcomes, freeze_trunk, device);
            net = nn; tl += loss as f64; nb += 1;
        }
        println!("  epoch {}/{}: loss={:.4}", epoch + 1, value_epochs, tl / nb.max(1) as f64);
    }

    println!("Candidate pretraining ({} epochs)...", cand_epochs);
    for epoch in 0..cand_epochs {
        let fresh: Vec<ImitateSample>;
        let samples: &[ImitateSample] = if let Some(ref s) = pregenerated { s } else {
            fresh = generate_imitate_data(chunk_games, num_players);
            &fresh
        };
        let mut idx: Vec<usize> = samples.iter().enumerate()
            .filter(|(_, s)| s.has_cand).map(|(i, _)| i).collect();
        idx.shuffle(&mut rng);
        let (mut tl, mut nb) = (0.0f64, 0usize);
        for chunk in idx.chunks(batch_size) {
            if chunk.len() < 2 { continue; }
            let feats:  Vec<Vec<f32>> = chunk.iter().map(|&i| samples[i].features.clone()).collect();
            let cands:  Vec<Vec<f32>> = chunk.iter().map(|&i| samples[i].cand_feats.clone()).collect();
            let chosen: Vec<u8>       = chunk.iter().map(|&i| samples[i].cand_chosen).collect();
            let (nn, loss) = cand_step_burn(net, &mut opt, lr as f64, &feats, &cands, &chosen, None, freeze_trunk, device);
            net = nn; tl += loss as f64; nb += 1;
        }
        println!("  epoch {}/{}: loss={:.4}", epoch + 1, cand_epochs, tl / nb.max(1) as f64);
    }

    let w = net.to_weights();
    net::save_weights(&w, out_path).unwrap_or_else(|e| eprintln!("Warning: save failed: {}", e));
    println!("Pretrain complete → {}", out_path);
}

fn cmd_pretrain(args: &[String]) {
    let out_path     = flag_str(args, "--out",          "models/astra6_pretrain.bin");
    let num_games    = flag_usize(args, "--games",      20_000);
    let num_players  = flag_usize(args, "--players",    4);
    let lr           = flag_f32(args, "--lr",           0.0003);
    let joint_epochs = flag_usize(args, "--joint-epochs", 30);
    let value_epochs = flag_usize(args, "--value-epochs", 10);
    let cand_epochs  = flag_usize(args, "--cand-epochs", 30);
    let batch_size   = flag_usize(args, "--batch-size", 512);
    let chunk_games  = flag_usize(args, "--chunk-games", 0);
    let label_smoothing = flag_f32(args, "--label-smoothing", 0.05);
    let val_games    = flag_usize(args, "--val-games", 500);
    // Trunk is frozen during value/candidate fine-tune by default (policy-first);
    // pass --no-freeze-trunk to restore the old joint-everything behaviour.
    let freeze_trunk = !has_flag(args, "--no-freeze-trunk");

    std::fs::create_dir_all(
        std::path::Path::new(&out_path).parent().unwrap_or(std::path::Path::new("."))
    ).unwrap();
    // --model: warm-start from an existing checkpoint. If the candidate-head
    // dims differ from the current build (e.g. HC widened), keep the trunk /
    // policy / value weights and reinitialize just the candidate head.
    let model_path = flag_str(args, "--model", "");
    let initial_weights = if !model_path.is_empty() {
        let mut w = net::load_weights(&model_path)
            .unwrap_or_else(|e| panic!("Failed to load {}: {}", model_path, e));
        let fresh = Net::new().to_weights();
        assert_eq!(w.w1.len(), fresh.w1.len(), "trunk dims of {} don't match this build", model_path);
        if w.wc1.len() != fresh.wc1.len() || w.wc2.len() != fresh.wc2.len() {
            println!("Warm start from {}: cand head dims changed — reinitializing cand head only", model_path);
            w.wc1 = fresh.wc1; w.bc1 = fresh.bc1;
            w.wc2 = fresh.wc2; w.bc2 = fresh.bc2;
        } else {
            println!("Warm start from {}", model_path);
        }
        w
    } else { Net::new().to_weights() };

    #[cfg(feature = "gpu")]
    { let device = WgpuDevice::default(); run_pretrain_burn::<Autodiff<Wgpu>>(initial_weights, &out_path, num_games, num_players, lr, joint_epochs, value_epochs, cand_epochs, batch_size, chunk_games, label_smoothing, freeze_trunk, val_games, &device); return; }
    #[cfg(not(feature = "gpu"))]
    { let device = NdArrayDevice::Cpu; run_pretrain_burn::<Autodiff<NdArray>>(initial_weights, &out_path, num_games, num_players, lr, joint_epochs, value_epochs, cand_epochs, batch_size, chunk_games, label_smoothing, freeze_trunk, val_games, &device); }
}

fn cmd_eval(args: &[String]) {
    let model_path  = flag_str(args, "--model",   "models/net_v6.bin");
    let num_games   = flag_usize(args, "--games",  400);
    let num_players = flag_usize(args, "--players", 4);
    let weights = net::load_weights(&model_path)
        .unwrap_or_else(|e| panic!("Failed to load {}: {}", model_path, e));
    let net = Arc::new(Net::from_weights(&weights));
    let t0 = Instant::now();
    let mode = if has_flag(args, "--lookahead-full") { EvalMode::FullLookahead }
               else if has_flag(args, "--lookahead-cands") { EvalMode::Lookahead }
               else if has_flag(args, "--advcpu-cands") { EvalMode::Hybrid }
               else { EvalMode::Full };
    let win_rate = eval(&net, num_games, num_players, mode);
    println!("Eval ({} games, {} players): {:.1}%  ({:.1}s)", num_games, num_players, win_rate * 100.0, t0.elapsed().as_secs_f32());
}

// ── PPO/RL training loop ──────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_rl_burn<B: AutodiffBackend>(
    initial_weights: Weights,
    out_path:        &str,
    num_iters:       usize,
    games_per:       usize,
    num_players:     usize,
    lr:              f32,
    temp_start:      f32,
    temp_final:      f32,
    ent_start:       f32,
    ent_final:       f32,
    ppo_epochs:      usize,
    batch_size:      usize,
    clip_eps:        f32,
    snapshot_every:  usize,
    pool_cap:        usize,
    per_alpha:       f32,
    per_beta:        f32,
    per_size:        usize,
    eval_every:      usize,
    eval_games:      usize,
    gamma:           f32,
    lambda:          f32,
    cand_lr_mult:    f32,
    advcpu_share:    f32,
    device:          &B::Device,
) {
    let mut net_burn        = NetBurn::<B>::from_weights(&initial_weights, device);
    let mut opt             = make_optimizer::<B>();
    let mut ndarray_weights = initial_weights;
    let mut snapshot_pool:  Vec<Arc<Weights>> = Vec::new();
    let mut per             = PerBuffer::new(per_size, per_alpha, per_beta);
    let mut rng             = thread_rng();
    let mut best_eval       = -1.0f32;

    println!("RL (PPO+PER): {} iters × {} games × {} ppo-epochs  lr={}  T={:.2}→{:.2}",
        num_iters, games_per, ppo_epochs, lr, temp_start, temp_final);

    for iter in 0..num_iters {
        if iter > 0 && iter % snapshot_every == 0 {
            snapshot_pool.push(Arc::new(ndarray_weights.clone()));
            if snapshot_pool.len() > pool_cap { snapshot_pool.remove(0); }
            println!("  [snapshot — pool {}]", snapshot_pool.len());
        }

        let frac          = iter as f32 / num_iters.saturating_sub(1).max(1) as f32;
        let temperature   = temp_start + (temp_final - temp_start) * frac;
        let entropy_coeff = ent_start  + (ent_final  - ent_start)  * frac;

        let t0 = Instant::now();
        let new_samples = run_self_play(
            games_per, num_players, Arc::new(ndarray_weights.clone()), &snapshot_pool,
            temperature, gamma, lambda, advcpu_share,
        );
        let new_count = new_samples.len();
        per.push_batch(new_samples);

        let (mut tpl, mut tvl, mut nb) = (0.0f64, 0.0f64, 0usize);
        let per_len = per.len();

        if per_len >= batch_size {
            for _epoch in 0..ppo_epochs {
                let sample_n = per_len.min(batch_size * 32);
                let (indices, is_weights) = per.sample_indices(sample_n);
                let mut order: Vec<usize> = (0..indices.len()).collect();
                order.shuffle(&mut rng);

                for chunk in order.chunks(batch_size) {
                    if chunk.len() < 2 { continue; }
                    let mut feats:     Vec<Vec<f32>> = Vec::new();
                    let mut masks:     Vec<u64>      = Vec::new();
                    let mut actions:   Vec<u8>       = Vec::new();
                    let mut outcomes:  Vec<f32>      = Vec::new();
                    let mut old_lps:   Vec<f32>      = Vec::new();
                    let mut is_w:      Vec<f32>      = Vec::new();
                    let mut cand_f:    Vec<Vec<f32>> = Vec::new();
                    let mut cand_d:    Vec<Vec<f32>> = Vec::new();
                    let mut cand_ch:   Vec<u8>       = Vec::new();
                    // Buffer indices actually used, parallel to feats — the loop
                    // below skips samples failing the legal_mask check, so this
                    // can't just be `indices`.
                    let mut used_idx:  Vec<usize>    = Vec::new();
                    let mut cand_adv:  Vec<f32>      = Vec::new();

                    for &j in chunk {
                        let s = &per.samples[indices[j]];
                        // PPO batch: policy decisions only. Aux cand samples
                        // (has_policy=false) carry no abstract action.
                        if s.has_policy && s.legal_mask & (1u64 << s.action) != 0 {
                            feats.push(s.features.clone()); masks.push(s.legal_mask);
                            actions.push(s.action); outcomes.push(s.outcome);
                            old_lps.push(s.log_prob); is_w.push(is_weights[j]);
                            used_idx.push(indices[j]);
                        }
                        if !s.cand_feats.is_empty() {
                            cand_f.push(s.features.clone());
                            cand_d.push(s.cand_feats.clone());
                            cand_ch.push(s.cand_chosen);
                            // REINFORCE weight for the cand head: signed
                            // advantage from collection time, clipped for
                            // stability. Without this the cand step is pure
                            // self-imitation (chosen = the head's own pick).
                            cand_adv.push((s.outcome - s.value).clamp(-2.0, 2.0));
                        }
                    }

                    if !feats.is_empty() {
                        let (nb2, pl, vl, td_errors) = rl_train_step_burn(
                            net_burn, &mut opt, lr as f64, &feats, &masks, &actions,
                            &outcomes, &old_lps, &is_w, entropy_coeff, clip_eps, device,
                        );
                        net_burn = nb2; tpl += pl as f64; tvl += vl as f64; nb += 1;

                        // Real priority update: write the freshly-computed
                        // |advantage| back into the buffer and re-derive
                        // priorities from it.
                        for (&idx, &e) in used_idx.iter().zip(td_errors.iter()) {
                            per.samples[idx].td_error = e;
                        }
                        per.update_priorities(&used_idx, &td_errors);
                    }

                    if cand_f.len() >= 2 {
                        // Mean-center the advantages per batch (REINFORCE
                        // baseline): keeps the update pushing probability
                        // between candidates rather than uniformly up/down.
                        let mean = cand_adv.iter().sum::<f32>() / cand_adv.len() as f32;
                        let adv_c: Vec<f32> = cand_adv.iter().map(|a| (a - mean).clamp(-2.0, 2.0)).collect();
                        // freeze_trunk: noisy REINFORCE gradients must not
                        // move the shared body — PPO owns trunk updates. The
                        // v4 run's post-peak decline (25.0→21.3 over 50 iters)
                        // tracked exactly this leakage.
                        let (nb3, _) = cand_step_burn(
                            net_burn, &mut opt, (lr * cand_lr_mult) as f64, &cand_f, &cand_d, &cand_ch, Some(&adv_c), true, device,
                        );
                        net_burn = nb3;
                    }
                }
            }
        }

        per.anneal_beta(1.0, 1.0 / num_iters as f32);
        ndarray_weights = net_burn.to_weights();
        let elapsed = t0.elapsed().as_secs_f32();

        println!("Iter {:4}  new={:6}  buf={:7}  pl={:+.4}  vl={:.4}  T={:.2}  ent={:.4}  pool={}  ({:.1}s)",
            iter + 1, new_count, per.len(), tpl / nb.max(1) as f64, tvl / nb.max(1) as f64,
            temperature, entropy_coeff, snapshot_pool.len(), elapsed);

        net::save_weights(&ndarray_weights, out_path)
            .unwrap_or_else(|e| eprintln!("Warning: save failed: {}", e));

        if eval_every > 0 && (iter + 1) % eval_every == 0 {
            let net_arc = Arc::new(Net::from_weights(&ndarray_weights));
            let wr = eval(&net_arc, eval_games, num_players, EvalMode::Full);
            let star = if wr > best_eval { "  *best" } else { "" };
            println!("  Eval @ iter {}: {:.1}%  ({} games){}", iter + 1, wr * 100.0, eval_games, star);
            // Peaks are real deliverables — never lose one to a later decline.
            if wr > best_eval {
                best_eval = wr;
                let best_path = out_path.replace(".bin", "_best.bin");
                net::save_weights(&ndarray_weights, &best_path)
                    .unwrap_or_else(|e| eprintln!("Warning: best save failed: {}", e));
            }
        }
    }
    println!("RL training complete → {}", out_path);
}

fn cmd_rl(args: &[String]) {
    let model_path     = flag_str(args, "--model", "");
    let out_path       = flag_str(args, "--out", "models/net_v6.bin");
    let num_iters      = flag_usize(args, "--iters", 300);
    let games_per      = flag_usize(args, "--games", 2000);
    let num_players    = flag_usize(args, "--players", 4);
    let lr             = flag_f32(args, "--lr", 0.0001);
    let temp_start     = flag_f32(args, "--temp-start", 1.0);
    let temp_final     = flag_f32(args, "--temp-final", 0.3);
    let ent_start      = flag_f32(args, "--entropy-start", 0.01);
    let ent_final      = flag_f32(args, "--entropy-final", 0.001);
    let ppo_epochs     = flag_usize(args, "--ppo-epochs", 4);
    let batch_size     = flag_usize(args, "--batch-size", 512);
    let clip_eps       = flag_f32(args, "--clip-eps", 0.2);
    let snapshot_every = flag_usize(args, "--snapshot-every", 20);
    let pool_cap       = flag_usize(args, "--pool-cap", 10);
    let per_alpha      = flag_f32(args, "--per-alpha", 0.6);
    let per_beta       = flag_f32(args, "--per-beta", 0.4);
    let per_size       = flag_usize(args, "--per-size", 500_000);
    let cand_lr_mult   = flag_f32(args, "--cand-lr-mult", 3.0);
    let advcpu_share   = flag_f32(args, "--advcpu-share", 0.3);
    let eval_every     = flag_usize(args, "--eval-every", 50);
    let eval_games     = flag_usize(args, "--eval-games", 400);
    let gamma          = flag_f32(args, "--gamma", 0.99);
    let lambda         = flag_f32(args, "--lambda", 0.95);

    std::fs::create_dir_all(
        std::path::Path::new(&out_path).parent().unwrap_or(std::path::Path::new("."))
    ).unwrap();
    let initial_weights = if !model_path.is_empty() {
        net::load_weights(&model_path).unwrap_or_else(|e| {
            eprintln!("Warning: could not load {}: {} — starting fresh", model_path, e);
            Net::new().to_weights()
        })
    } else { Net::new().to_weights() };

    #[cfg(feature = "gpu")]
    { let device = WgpuDevice::default(); run_rl_burn::<Autodiff<Wgpu>>(initial_weights, &out_path, num_iters, games_per, num_players, lr, temp_start, temp_final, ent_start, ent_final, ppo_epochs, batch_size, clip_eps, snapshot_every, pool_cap, per_alpha, per_beta, per_size, eval_every, eval_games, gamma, lambda, cand_lr_mult, advcpu_share, &device); return; }
    #[cfg(not(feature = "gpu"))]
    { let device = NdArrayDevice::Cpu; run_rl_burn::<Autodiff<NdArray>>(initial_weights, &out_path, num_iters, games_per, num_players, lr, temp_start, temp_final, ent_start, ent_final, ppo_epochs, batch_size, clip_eps, snapshot_every, pool_cap, per_alpha, per_beta, per_size, eval_every, eval_games, gamma, lambda, cand_lr_mult, advcpu_share, &device); }
}

// ── AlphaZero training loop ───────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_az_burn<B: AutodiffBackend>(
    initial_weights: Weights,
    out_path:        &str,
    num_iters:       usize,
    games_per:       usize,
    num_players:     usize,
    lr:              f32,
    mcts_sims:       usize,
    c_puct:          f32,
    az_epochs:       usize,
    az_explore:      bool,
    batch_size:      usize,
    window:          usize,
    snapshot_every:  usize,
    pool_cap:        usize,
    advcpu_share:    f32,
    cand_lr_mult:    f32,
    eval_every:      usize,
    eval_games:      usize,
    device:          &B::Device,
) {
    let mut net_burn        = NetBurn::<B>::from_weights(&initial_weights, device);
    let mut opt             = make_optimizer::<B>();
    let mut ndarray_weights = initial_weights;
    let mut snapshot_pool:  Vec<Arc<Weights>> = Vec::new();
    let mut replay:         Vec<AzSample>     = Vec::new();
    let mut rng             = thread_rng();
    let mut best_eval       = -1.0f32;

    println!("AlphaZero: {} iters × {} games × {}sims × {} epochs  lr={}  c_puct={}  window={}",
        num_iters, games_per, mcts_sims, az_epochs, lr, c_puct, window);

    for iter in 0..num_iters {
        if iter > 0 && iter % snapshot_every == 0 {
            snapshot_pool.push(Arc::new(ndarray_weights.clone()));
            if snapshot_pool.len() > pool_cap { snapshot_pool.remove(0); }
            println!("  [snapshot — pool {}]", snapshot_pool.len());
        }

        let t0 = Instant::now();

        let new_samples = run_self_play_az(
            games_per, num_players, Arc::new(ndarray_weights.clone()), &snapshot_pool,
            mcts_sims, c_puct, advcpu_share, az_explore,
        );
        let new_count = new_samples.len();

        // Rolling window: drop oldest if over capacity
        replay.extend(new_samples);
        if replay.len() > window {
            let excess = replay.len() - window;
            replay.drain(..excess);
        }

        let (mut tpl, mut tvl, mut nb) = (0.0f64, 0.0f64, 0usize);

        let policy_idx: Vec<usize> = replay.iter().enumerate()
            .filter(|(_, s)| s.has_policy).map(|(i, _)| i).collect();
        let cand_idx: Vec<usize>   = replay.iter().enumerate()
            .filter(|(_, s)| s.has_cand).map(|(i, _)| i).collect();

        if policy_idx.len() >= batch_size {
            for _epoch in 0..az_epochs {
                let mut idx = policy_idx.clone();
                idx.shuffle(&mut rng);

                for chunk in idx.chunks(batch_size) {
                    if chunk.len() < 2 { continue; }
                    let feats:    Vec<Vec<f32>>            = chunk.iter().map(|&i| replay[i].features.clone()).collect();
                    let visits:   Vec<Vec<f32>>            = chunk.iter().map(|&i| replay[i].policy_target.to_vec()).collect();
                    let outcomes: Vec<f32>                 = chunk.iter().map(|&i| replay[i].outcome).collect();
                    let masks:    Vec<u64>                 = chunk.iter().map(|&i| replay[i].legal_mask).collect();

                    let (nn, pl, vl) = az_train_step_burn(
                        net_burn, &mut opt, lr as f64, &feats, &visits, &outcomes, &masks, device,
                    );
                    net_burn = nn; tpl += pl as f64; tvl += vl as f64; nb += 1;
                }

                // Candidate head update from cand_idx
                let cand_sample_n = cand_idx.len().min(batch_size * 4);
                if cand_sample_n >= 2 {
                    let mut cidx: Vec<usize> = cand_idx.clone();
                    cidx.shuffle(&mut rng);
                    for chunk in cidx[..cand_sample_n].chunks(batch_size) {
                        if chunk.len() < 2 { continue; }
                        let cand_feats:  Vec<Vec<f32>> = chunk.iter().map(|&i| replay[i].features.clone()).collect();
                        let cand_data:   Vec<Vec<f32>> = chunk.iter().map(|&i| replay[i].cand_feats.clone()).collect();
                        let cand_chosen: Vec<u8>       = chunk.iter().map(|&i| replay[i].cand_chosen).collect();
                        // REINFORCE weight (chosen = the head's own SAMPLED
                        // pick). Mean-centered per batch (baseline) and trunk
                        // frozen — both lessons from the PPO legs.
                        let raw_adv: Vec<f32> = chunk.iter()
                            .map(|&i| replay[i].outcome - replay[i].value_est).collect();
                        let mean = raw_adv.iter().sum::<f32>() / raw_adv.len() as f32;
                        let cand_adv: Vec<f32> = raw_adv.iter().map(|a| (a - mean).clamp(-2.0, 2.0)).collect();
                        let (nn, _) = cand_step_burn(
                            net_burn, &mut opt, (lr * cand_lr_mult) as f64,
                            &cand_feats, &cand_data, &cand_chosen, Some(&cand_adv), true, device,
                        );
                        net_burn = nn;
                    }
                }
            }
        }

        ndarray_weights = net_burn.to_weights();
        let elapsed = t0.elapsed().as_secs_f32();

        println!("Iter {:4}  new={:6}  buf={:7}  pl={:+.4}  vl={:.4}  pool={}  ({:.1}s)",
            iter + 1, new_count, replay.len(), tpl / nb.max(1) as f64, tvl / nb.max(1) as f64,
            snapshot_pool.len(), elapsed);

        net::save_weights(&ndarray_weights, out_path)
            .unwrap_or_else(|e| eprintln!("Warning: save failed: {}", e));

        if eval_every > 0 && (iter + 1) % eval_every == 0 {
            let net_arc = Arc::new(Net::from_weights(&ndarray_weights));
            let wr = eval(&net_arc, eval_games, num_players, EvalMode::Full);
            let star = if wr > best_eval { "  *best" } else { "" };
            println!("  Eval @ iter {}: {:.1}%  ({} games){}", iter + 1, wr * 100.0, eval_games, star);
            if wr > best_eval {
                best_eval = wr;
                let best_path = out_path.replace(".bin", "_best.bin");
                net::save_weights(&ndarray_weights, &best_path)
                    .unwrap_or_else(|e| eprintln!("Warning: best save failed: {}", e));
            }
        }
    }
    println!("AZ training complete → {}", out_path);
}

fn cmd_az(args: &[String]) {
    let model_path     = flag_str(args, "--model",          "");
    let out_path       = flag_str(args, "--out",            "models/net_az.bin");
    let num_iters      = flag_usize(args, "--iters",        500);
    let games_per      = flag_usize(args, "--games",        200);
    let num_players    = flag_usize(args, "--players",      4);
    let lr             = flag_f32(args, "--lr",             0.0001);
    let mcts_sims      = flag_usize(args, "--mcts-sims",    50);
    let c_puct         = flag_f32(args, "--c-puct",         1.5);
    let az_epochs      = flag_usize(args, "--az-epochs",    4);
    let az_explore     = has_flag(args, "--az-explore");
    let batch_size     = flag_usize(args, "--batch-size",   512);
    let window         = flag_usize(args, "--window",       200_000);
    let snapshot_every = flag_usize(args, "--snapshot-every", 20);
    let pool_cap       = flag_usize(args, "--pool-cap",     10);
    let advcpu_share   = flag_f32(args, "--advcpu-share",   0.3);
    let cand_lr_mult   = flag_f32(args, "--cand-lr-mult",   3.0);
    let eval_every     = flag_usize(args, "--eval-every",   50);
    let eval_games     = flag_usize(args, "--eval-games",   400);

    std::fs::create_dir_all(
        std::path::Path::new(&out_path).parent().unwrap_or(std::path::Path::new("."))
    ).unwrap();
    let initial_weights = if !model_path.is_empty() {
        net::load_weights(&model_path).unwrap_or_else(|e| {
            eprintln!("Warning: could not load {}: {} — starting fresh", model_path, e);
            Net::new().to_weights()
        })
    } else { Net::new().to_weights() };

    #[cfg(feature = "gpu")]
    { let device = WgpuDevice::default(); run_az_burn::<Autodiff<Wgpu>>(initial_weights, &out_path, num_iters, games_per, num_players, lr, mcts_sims, c_puct, az_epochs, az_explore, batch_size, window, snapshot_every, pool_cap, advcpu_share, cand_lr_mult, eval_every, eval_games, &device); return; }
    #[cfg(not(feature = "gpu"))]
    { let device = NdArrayDevice::Cpu; run_az_burn::<Autodiff<NdArray>>(initial_weights, &out_path, num_iters, games_per, num_players, lr, mcts_sims, c_puct, az_epochs, az_explore, batch_size, window, snapshot_every, pool_cap, advcpu_share, cand_lr_mult, eval_every, eval_games, &device); }
}
