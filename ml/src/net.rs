use ndarray::{Array1, Array2, Axis};
use ndarray_rand::{rand_distr::Normal, RandomExt};
use serde::{Deserialize, Serialize};

use crate::feature::FEATURE_DIM;
use crate::action::NUM_ACTIONS;
use crate::candidate::CAND_DIM;

// 104→256→256(+res)→256(+res)→128→{35,1}  plus candidate head [128+24→32→1]
const H1: usize = 256;
const H2: usize = 256; // residual from L1 (same dim)
const H3: usize = 256; // residual from L2 (same dim)
const H4: usize = 128; // bottleneck before heads
const HC: usize = 32;  // candidate head hidden size

// ── Flat-weight serialisation ─────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone)]
pub struct Weights {
    pub w1: Vec<f32>, pub b1: Vec<f32>,
    pub w2: Vec<f32>, pub b2: Vec<f32>,
    pub w3: Vec<f32>, pub b3: Vec<f32>,
    pub w4: Vec<f32>, pub b4: Vec<f32>,
    pub wp: Vec<f32>, pub bp: Vec<f32>,
    pub wv: Vec<f32>, pub bv: Vec<f32>,
    // candidate head: [H4+CAND_DIM → HC → 1]
    pub wc1: Vec<f32>, pub bc1: Vec<f32>,
    pub wc2: Vec<f32>, pub bc2: Vec<f32>,
}

// ── Layer ─────────────────────────────────────────────────────────────────────

struct Layer {
    w: Array2<f32>, // [out, in]
    b: Array1<f32>, // [out]
}

impl Layer {
    fn rand(in_dim: usize, out_dim: usize) -> Self {
        let std = (2.0f32 / in_dim as f32).sqrt();
        let w = Array2::random((out_dim, in_dim), Normal::new(0.0f32, std).unwrap());
        let b = Array1::zeros(out_dim);
        Self { w, b }
    }

    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        x.dot(&self.w.t()) + &self.b
    }

    fn forward_single(&self, x: ndarray::ArrayView1<f32>) -> Array1<f32> {
        self.w.dot(&x) + &self.b
    }
}

// ── Network ───────────────────────────────────────────────────────────────────

pub struct Net {
    l1: Layer,
    l2: Layer, // residual: output += input (same dim as L1)
    l3: Layer, // residual: output += input (same dim as L2)
    l4: Layer, // no residual (dim change 256→128)
    lp: Layer,
    lv: Layer,
    lc1: Layer, // candidate head: (H4+CAND_DIM) → HC
    lc2: Layer, // candidate head: HC → 1
}

impl Net {
    pub fn new() -> Self {
        Self {
            l1:  Layer::rand(FEATURE_DIM, H1),
            l2:  Layer::rand(H1, H2),
            l3:  Layer::rand(H2, H3),
            l4:  Layer::rand(H3, H4),
            lp:  Layer::rand(H4, NUM_ACTIONS),
            lv:  Layer::rand(H4, 1),
            lc1: Layer::rand(H4 + CAND_DIM, HC),
            lc2: Layer::rand(HC, 1),
        }
    }

    pub fn from_weights(w: &Weights) -> Self {
        let load2 = |data: &[f32], rows: usize, cols: usize| {
            Array2::from_shape_vec((rows, cols), data.to_vec()).unwrap()
        };
        let load1 = |data: &[f32]| Array1::from_vec(data.to_vec());
        Self {
            l1:  Layer { w: load2(&w.w1,  H1, FEATURE_DIM),    b: load1(&w.b1)  },
            l2:  Layer { w: load2(&w.w2,  H2, H1),             b: load1(&w.b2)  },
            l3:  Layer { w: load2(&w.w3,  H3, H2),             b: load1(&w.b3)  },
            l4:  Layer { w: load2(&w.w4,  H4, H3),             b: load1(&w.b4)  },
            lp:  Layer { w: load2(&w.wp,  NUM_ACTIONS, H4),    b: load1(&w.bp)  },
            lv:  Layer { w: load2(&w.wv,  1, H4),              b: load1(&w.bv)  },
            lc1: Layer { w: load2(&w.wc1, HC, H4 + CAND_DIM), b: load1(&w.bc1) },
            lc2: Layer { w: load2(&w.wc2, 1,  HC),             b: load1(&w.bc2) },
        }
    }

    pub fn to_weights(&self) -> Weights {
        Weights {
            w1:  self.l1.w.iter().copied().collect(),  b1:  self.l1.b.iter().copied().collect(),
            w2:  self.l2.w.iter().copied().collect(),  b2:  self.l2.b.iter().copied().collect(),
            w3:  self.l3.w.iter().copied().collect(),  b3:  self.l3.b.iter().copied().collect(),
            w4:  self.l4.w.iter().copied().collect(),  b4:  self.l4.b.iter().copied().collect(),
            wp:  self.lp.w.iter().copied().collect(),  bp:  self.lp.b.iter().copied().collect(),
            wv:  self.lv.w.iter().copied().collect(),  bv:  self.lv.b.iter().copied().collect(),
            wc1: self.lc1.w.iter().copied().collect(), bc1: self.lc1.b.iter().copied().collect(),
            wc2: self.lc2.w.iter().copied().collect(), bc2: self.lc2.b.iter().copied().collect(),
        }
    }

    // Batch variant: runs one GEMM per layer across all samples instead of N GEMVs.
    // Returns [B × H4] body outputs.
    pub fn body_forward_batch(&self, features: &[[f32; FEATURE_DIM]]) -> Array2<f32> {
        let b = features.len();
        let flat: Vec<f32> = features.iter().flat_map(|f| f.iter().copied()).collect();
        let x  = Array2::from_shape_vec((b, FEATURE_DIM), flat).unwrap();
        let a1 = relu2(self.l1.forward(&x).view());
        let z2 = self.l2.forward(&a1) + &a1;
        let a2 = relu2(z2.view());
        let z3 = self.l3.forward(&a2) + &a2;
        let a3 = relu2(z3.view());
        relu2(self.l4.forward(&a3).view())
    }

    // Batched stochastic inference. Returns one (abs_idx, log_prob, value, body) per sample.
    pub fn infer_stochastic_batch(
        &self,
        features: &[[f32; FEATURE_DIM]],
        masks:    &[u64],
        temps:    &[f32],
        us:       &[f32],
    ) -> Vec<(usize, f32, f32, Array1<f32>)> {
        let bodies = self.body_forward_batch(features);
        (0..features.len()).map(|i| {
            let body  = bodies.row(i).to_owned();
            let lp    = self.lp.forward_single(body.view());
            let value = sigmoid(self.lv.forward_single(body.view())[0]);
            let mut logits = [0.0f32; NUM_ACTIONS];
            for (j, v) in lp.iter().enumerate() { logits[j] = *v; }
            let t      = temps[i].max(1e-6);
            let scaled: [f32; NUM_ACTIONS] = std::array::from_fn(|j| logits[j] / t);
            let probs  = masked_softmax(&scaled, masks[i]);
            let mut cum = 0.0f32;
            let mut chosen = NUM_ACTIONS - 1;
            for (j, &p) in probs.iter().enumerate() {
                cum += p;
                if us[i] < cum { chosen = j; break; }
            }
            if masks[i] & (1u64 << chosen) == 0 {
                chosen = (0..NUM_ACTIONS).find(|&j| masks[i] & (1u64 << j) != 0).unwrap_or(0);
            }
            (chosen, probs[chosen].max(1e-9).ln(), value, body)
        }).collect()
    }

    pub fn infer(&self, features: &[f32; FEATURE_DIM]) -> ([f32; NUM_ACTIONS], f32) {
        let (logits, value, _) = self.infer_with_body(features);
        (logits, value)
    }

    // Single body pass returning logits, value, AND the body output.
    // Use this instead of calling infer() + body_forward() separately.
    pub fn infer_with_body(&self, features: &[f32; FEATURE_DIM]) -> ([f32; NUM_ACTIONS], f32, Array1<f32>) {
        let body = self.body_forward(features);
        let lp    = self.lp.forward_single(body.view());
        let value = sigmoid(self.lv.forward_single(body.view())[0]);
        let mut out = [0.0f32; NUM_ACTIONS];
        for (i, v) in lp.iter().enumerate() { out[i] = *v; }
        (out, value, body)
    }

    // Run the shared body, return the 128-dim output (used to score candidates)
    pub fn body_forward(&self, features: &[f32; FEATURE_DIM]) -> Array1<f32> {
        let x  = ndarray::aview1(features);
        let a1 = relu1(self.l1.forward_single(x));
        let a2 = relu1(self.l2.forward_single(a1.view()) + &a1);
        let a3 = relu1(self.l3.forward_single(a2.view()) + &a2);
        relu1(self.l4.forward_single(a3.view()))
    }

    // Score a single candidate given pre-computed body output
    pub fn score_candidate(&self, body: &Array1<f32>, cand: &[f32; CAND_DIM]) -> f32 {
        let mut inp = Vec::with_capacity(H4 + CAND_DIM);
        inp.extend_from_slice(body.as_slice().unwrap());
        inp.extend_from_slice(cand);
        let inp_arr = ndarray::aview1(&inp);
        let h = relu1(self.lc1.forward_single(inp_arr));
        self.lc2.forward_single(h.view())[0]
    }

    // Score a list of candidates, return raw scores (caller applies softmax or argmax)
    pub fn score_candidates(&self, body: &Array1<f32>, cands: &[[f32; CAND_DIM]]) -> Vec<f32> {
        cands.iter().map(|c| self.score_candidate(body, c)).collect()
    }

    pub fn infer_stochastic(&self, features: &[f32; FEATURE_DIM], mask: u64, temperature: f32, u: f32) -> (usize, f32, f32) {
        let (chosen, log_prob, value, _) = self.infer_stochastic_with_body(features, mask, temperature, u);
        (chosen, log_prob, value)
    }

    // Like infer_stochastic but also returns the body output so callers can
    // reuse it for candidate scoring without a second forward pass.
    pub fn infer_stochastic_with_body(&self, features: &[f32; FEATURE_DIM], mask: u64, temperature: f32, u: f32) -> (usize, f32, f32, Array1<f32>) {
        let (logits, value, body) = self.infer_with_body(features);
        let t = temperature.max(1e-6);
        let scaled: [f32; NUM_ACTIONS] = std::array::from_fn(|i| logits[i] / t);
        let probs = masked_softmax(&scaled, mask);

        let mut cum = 0.0f32;
        let mut chosen = NUM_ACTIONS - 1;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if u < cum { chosen = i; break; }
        }
        if mask & (1u64 << chosen) == 0 {
            chosen = (0..NUM_ACTIONS).find(|&i| mask & (1u64 << i) != 0).unwrap_or(0);
        }
        (chosen, probs[chosen].max(1e-9).ln(), value, body)
    }
}

// ── Adam optimizer ────────────────────────────────────────────────────────────

struct Adam2 { m: Array2<f32>, v: Array2<f32> }
struct Adam1 { m: Array1<f32>, v: Array1<f32> }

impl Adam2 {
    fn new(shape: (usize, usize)) -> Self {
        Self { m: Array2::zeros(shape), v: Array2::zeros(shape) }
    }
    fn step(&mut self, w: &mut Array2<f32>, g: &Array2<f32>, t: u32, lr: f32, b1: f32, b2: f32, eps: f32) {
        self.m = &self.m * b1 + g * (1.0 - b1);
        self.v = &self.v * b2 + &g.mapv(|x| x * x) * (1.0 - b2);
        let mh = self.m.mapv(|x| x / (1.0 - b1.powi(t as i32)));
        let vh = self.v.mapv(|x| x / (1.0 - b2.powi(t as i32)));
        *w -= &(mh / (vh.mapv(f32::sqrt) + eps) * lr);
    }
}

impl Adam1 {
    fn new(len: usize) -> Self {
        Self { m: Array1::zeros(len), v: Array1::zeros(len) }
    }
    fn step(&mut self, b: &mut Array1<f32>, g: &Array1<f32>, t: u32, lr: f32, b1: f32, b2: f32, eps: f32) {
        self.m = &self.m * b1 + g * (1.0 - b1);
        self.v = &self.v * b2 + &g.mapv(|x| x * x) * (1.0 - b2);
        let mh = self.m.mapv(|x| x / (1.0 - b1.powi(t as i32)));
        let vh = self.v.mapv(|x| x / (1.0 - b2.powi(t as i32)));
        *b -= &(mh / (vh.mapv(f32::sqrt) + eps) * lr);
    }
}

pub struct Optimizer {
    l1_w: Adam2, l1_b: Adam1,
    l2_w: Adam2, l2_b: Adam1,
    l3_w: Adam2, l3_b: Adam1,
    l4_w: Adam2, l4_b: Adam1,
    lp_w: Adam2, lp_b: Adam1,
    lv_w: Adam2, lv_b: Adam1,
    lc1_w: Adam2, lc1_b: Adam1,
    lc2_w: Adam2, lc2_b: Adam1,
    pub t: u32,
    pub lr: f32,
    b1: f32, b2: f32, eps: f32,
}

impl Optimizer {
    pub fn new(lr: f32) -> Self {
        Self {
            l1_w:  Adam2::new((H1, FEATURE_DIM)),    l1_b:  Adam1::new(H1),
            l2_w:  Adam2::new((H2, H1)),             l2_b:  Adam1::new(H2),
            l3_w:  Adam2::new((H3, H2)),             l3_b:  Adam1::new(H3),
            l4_w:  Adam2::new((H4, H3)),             l4_b:  Adam1::new(H4),
            lp_w:  Adam2::new((NUM_ACTIONS, H4)),    lp_b:  Adam1::new(NUM_ACTIONS),
            lv_w:  Adam2::new((1, H4)),              lv_b:  Adam1::new(1),
            lc1_w: Adam2::new((HC, H4 + CAND_DIM)), lc1_b: Adam1::new(HC),
            lc2_w: Adam2::new((1, HC)),              lc2_b: Adam1::new(1),
            t: 0, lr, b1: 0.9, b2: 0.999, eps: 1e-8,
        }
    }
}

// ── Supervised imitation training step ────────────────────────────────────────

pub struct Batch<'a> {
    pub features: &'a [Vec<f32>],
    pub masks:    &'a [u64],
    pub actions:  &'a [u8],
    pub outcomes: &'a [f32],
}

pub fn train_step(net: &mut Net, opt: &mut Optimizer, batch: &Batch) -> (f32, f32) {
    let b = batch.features.len();
    assert!(b > 0);

    let (x, z1, a1, z2, a2, z3, a3, z4, a4, lp, _lv, v) = forward(net, batch.features);

    let mut dlp = Array2::<f32>::zeros((b, NUM_ACTIONS));
    let mut policy_loss = 0.0f32;

    for i in 0..b {
        let row    = lp.row(i);
        let mask   = batch.masks[i];
        let target = batch.actions[i] as usize;
        let probs  = masked_softmax(row.as_slice().unwrap(), mask);

        policy_loss -= probs[target].max(1e-9).ln() / b as f32;
        for j in 0..NUM_ACTIONS {
            if mask & (1u64 << j) != 0 {
                dlp[[i, j]] = (probs[j] - if j == target { 1.0 } else { 0.0 }) / b as f32;
            }
        }
    }

    let (value_loss, dlv) = value_loss_and_grad(&v, batch.outcomes, b);

    opt.t += 1;
    backward(net, opt, &dlp, &dlv, &x, &z1, &a1, &z2, &a2, &z3, &a3, &z4, &a4, b);
    (policy_loss, value_loss)
}

// ── Value pretraining step (only updates feature extractor + value head) ──────

pub struct ValueBatch<'a> {
    pub features: &'a [Vec<f32>],
    pub outcomes: &'a [f32],
}

pub fn value_pretrain_step(net: &mut Net, opt: &mut Optimizer, batch: &ValueBatch) -> f32 {
    let b = batch.features.len();
    assert!(b > 0);

    let (x, z1, a1, z2, a2, z3, a3, z4, a4, _lp, _lv, v) = forward(net, batch.features);

    let (value_loss, dlv) = value_loss_and_grad(&v, batch.outcomes, b);

    opt.t += 1;
    let (lr, b1, b2, eps, t) = (opt.lr, opt.b1, opt.b2, opt.eps, opt.t);

    // Value head + shared layers only (skip policy head).
    let (gw_lv, gb_lv, da4) = linear_backward(&dlv, &a4, &net.lv.w, b);
    opt.lv_w.step(&mut net.lv.w, &gw_lv, t, lr, b1, b2, eps);
    opt.lv_b.step(&mut net.lv.b, &gb_lv, t, lr, b1, b2, eps);

    let dz4 = da4 * relu2_grad(z4.view());
    let (gw4, gb4, da3) = linear_backward(&dz4, &a3, &net.l4.w, b);
    opt.l4_w.step(&mut net.l4.w, &gw4, t, lr, b1, b2, eps);
    opt.l4_b.step(&mut net.l4.b, &gb4, t, lr, b1, b2, eps);

    let dz3 = da3 * relu2_grad(z3.view());
    let (gw3, gb3, da2_lin) = linear_backward(&dz3, &a2, &net.l3.w, b);
    let da2 = da2_lin + &dz3;
    opt.l3_w.step(&mut net.l3.w, &gw3, t, lr, b1, b2, eps);
    opt.l3_b.step(&mut net.l3.b, &gb3, t, lr, b1, b2, eps);

    let dz2 = da2 * relu2_grad(z2.view());
    let (gw2, gb2, da1_lin) = linear_backward(&dz2, &a1, &net.l2.w, b);
    let da1 = da1_lin + &dz2;
    opt.l2_w.step(&mut net.l2.w, &gw2, t, lr, b1, b2, eps);
    opt.l2_b.step(&mut net.l2.b, &gb2, t, lr, b1, b2, eps);

    let dz1 = da1 * relu2_grad(z1.view());
    let (gw1, gb1, _) = linear_backward(&dz1, &x, &net.l1.w, b);
    opt.l1_w.step(&mut net.l1.w, &gw1, t, lr, b1, b2, eps);
    opt.l1_b.step(&mut net.l1.b, &gb1, t, lr, b1, b2, eps);

    value_loss
}

// ── RL training step (PPO + entropy bonus) ───────────────────────────────────

pub struct RlBatch<'a> {
    pub features:      &'a [Vec<f32>],
    pub masks:         &'a [u64],
    pub actions:       &'a [u8],
    pub outcomes:      &'a [f32],
    pub old_log_probs: &'a [f32],  // log π_old(a|s) from sample collection
}

pub fn rl_train_step(
    net: &mut Net,
    opt: &mut Optimizer,
    batch: &RlBatch,
    entropy_coeff: f32,
    clip_eps: f32,
) -> (f32, f32) {
    let b = batch.features.len();
    assert!(b > 0);

    let (x, z1, a1, z2, a2, z3, a3, z4, a4, lp, _lv, v) = forward(net, batch.features);

    let mut dlp = Array2::<f32>::zeros((b, NUM_ACTIONS));
    let mut policy_loss = 0.0f32;

    for i in 0..b {
        let row          = lp.row(i);
        let mask         = batch.masks[i];
        let action       = batch.actions[i] as usize;
        let probs        = masked_softmax(row.as_slice().unwrap(), mask);
        let new_log_prob = probs[action].max(1e-9).ln();
        let old_log_prob = batch.old_log_probs[i];

        let ratio     = (new_log_prob - old_log_prob).exp();
        let advantage = batch.outcomes[i] - v[[i, 0]];
        let entropy: f32 = probs.iter().filter(|&&p| p > 1e-9).map(|&p| -p * p.ln()).sum();

        // PPO clip: disable gradient when ratio is already outside bounds
        // in the direction that would increase the loss
        let is_clipped = (ratio > 1.0 + clip_eps && advantage > 0.0)
                      || (ratio < 1.0 - clip_eps && advantage < 0.0);

        let effective_ratio = if is_clipped {
            ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
        } else {
            ratio
        };
        policy_loss += (-effective_ratio * advantage - entropy_coeff * entropy) / b as f32;

        for j in 0..NUM_ACTIONS {
            if mask & (1u64 << j) != 0 {
                let ent_grad = if probs[j] > 1e-9 {
                    probs[j] * (probs[j].ln() + entropy)
                } else {
                    0.0
                };
                let policy_grad = if is_clipped {
                    0.0  // gradient through clipped term is zero
                } else {
                    // ∂(-ratio * advantage) / ∂logit_j
                    // = advantage * ratio * (probs[j] - 1[j==action])
                    advantage * ratio * (probs[j] - if j == action { 1.0 } else { 0.0 })
                };
                dlp[[i, j]] = (policy_grad - entropy_coeff * ent_grad) / b as f32;
            }
        }
    }

    let (value_loss, dlv) = value_loss_and_grad(&v, batch.outcomes, b);

    opt.t += 1;
    backward(net, opt, &dlp, &dlv, &x, &z1, &a1, &z2, &a2, &z3, &a3, &z4, &a4, b);
    (policy_loss, value_loss)
}

// ── Candidate head training step ─────────────────────────────────────────────
//
// Each sample has K candidates; exactly one was chosen (softmax cross-entropy).
// Gradients flow through lc2 and lc1 only — body is treated as fixed context.

pub struct CandidateSampleBatch<'a> {
    pub features:    &'a [Vec<f32>],          // one feature vec per sample
    pub candidates:  &'a [Vec<[f32; CAND_DIM]>], // K candidates per sample
    pub chosen_idxs: &'a [u8],               // which candidate was chosen
}

pub fn candidate_train_step(net: &mut Net, opt: &mut Optimizer, batch: &CandidateSampleBatch) -> f32 {
    let n = batch.features.len();
    assert!(n > 0);

    // Accumulate gradients for the candidate head layers across all samples
    let mut gw_c1 = Array2::<f32>::zeros((HC, H4 + CAND_DIM));
    let mut gb_c1 = Array1::<f32>::zeros(HC);
    let mut gw_c2 = Array2::<f32>::zeros((1, HC));
    let mut gb_c2 = Array1::<f32>::zeros(1);
    let mut total_loss = 0.0f32;

    for i in 0..n {
        // Body forward (no grad propagation back through body)
        let body = net.body_forward(
            &batch.features[i].as_slice().try_into()
                .unwrap_or_else(|_| panic!("feature dim mismatch"))
        );

        let cands = &batch.candidates[i];
        let k = cands.len();
        if k == 0 { continue; }

        // Forward through candidate head for each candidate
        let mut scores = Vec::with_capacity(k);
        let mut hs:     Vec<Array1<f32>> = Vec::with_capacity(k);
        let mut inps:   Vec<Array1<f32>> = Vec::with_capacity(k);
        for cand in cands.iter() {
            let mut inp_vec = Vec::with_capacity(H4 + CAND_DIM);
            inp_vec.extend_from_slice(body.as_slice().unwrap());
            inp_vec.extend_from_slice(cand);
            let inp = Array1::from_vec(inp_vec);
            let h   = relu1(net.lc1.forward_single(inp.view()));
            let s   = net.lc2.forward_single(h.view())[0];
            scores.push(s);
            hs.push(h);
            inps.push(inp);
        }

        // Softmax over scores
        let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
        let sum_e: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e / sum_e.max(1e-10)).collect();

        let chosen = batch.chosen_idxs[i] as usize;
        total_loss += -probs[chosen].max(1e-9).ln() / n as f32;

        // Backward: dL/d_score_j = (probs[j] - 1[j==chosen]) / n
        for j in 0..k {
            let ds = (probs[j] - if j == chosen { 1.0 } else { 0.0 }) / n as f32;
            let ds_arr = Array2::from_shape_vec((1, 1), vec![ds]).unwrap();

            // lc2 gradient
            let (gw2_j, gb2_j, dh_j) = linear_backward(&ds_arr, &hs[j].clone().insert_axis(Axis(0)), &net.lc2.w, 1);
            gw_c2 += &gw2_j;
            gb_c2 += &gb2_j;

            // lc1 gradient (through ReLU)
            // Need z (pre-ReLU) for relu grad — recompute from inp
            let z_h = net.lc1.forward_single(inps[j].view());
            let relu_mask = z_h.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
            let dh_flat = dh_j.into_shape(HC).unwrap();
            let dz_h: Array1<f32> = dh_flat * &relu_mask;
            let dz_arr = dz_h.clone().insert_axis(Axis(0));
            let (gw1_j, gb1_j, _) = linear_backward(&dz_arr, &inps[j].clone().insert_axis(Axis(0)), &net.lc1.w, 1);
            gw_c1 += &gw1_j;
            gb_c1 += &gb1_j;
        }
    }

    opt.t += 1;
    let (lr, b1, b2, eps, t) = (opt.lr, opt.b1, opt.b2, opt.eps, opt.t);
    opt.lc1_w.step(&mut net.lc1.w, &gw_c1, t, lr, b1, b2, eps);
    opt.lc1_b.step(&mut net.lc1.b, &gb_c1, t, lr, b1, b2, eps);
    opt.lc2_w.step(&mut net.lc2.w, &gw_c2, t, lr, b1, b2, eps);
    opt.lc2_b.step(&mut net.lc2.b, &gb_c2, t, lr, b1, b2, eps);

    total_loss
}

// ── Shared forward / backward helpers ────────────────────────────────────────

#[allow(clippy::type_complexity)]
fn forward(net: &Net, features: &[Vec<f32>])
    -> (Array2<f32>, Array2<f32>, Array2<f32>,
        Array2<f32>, Array2<f32>, Array2<f32>,
        Array2<f32>, Array2<f32>, Array2<f32>,
        Array2<f32>, Array2<f32>, Array2<f32>)
{
    let x  = feats_to_array(features);
    let z1 = net.l1.forward(&x);
    let a1 = relu2(z1.view());
    let z2 = net.l2.forward(&a1) + &a1; // residual
    let a2 = relu2(z2.view());
    let z3 = net.l3.forward(&a2) + &a2; // residual
    let a3 = relu2(z3.view());
    let z4 = net.l4.forward(&a3);
    let a4 = relu2(z4.view());
    let lp = net.lp.forward(&a4);
    let lv = net.lv.forward(&a4);
    let v  = lv.mapv(sigmoid);
    (x, z1, a1, z2, a2, z3, a3, z4, a4, lp, lv, v)
}

fn value_loss_and_grad(v: &Array2<f32>, outcomes: &[f32], b: usize) -> (f32, Array2<f32>) {
    let outcomes_arr = Array2::from_shape_vec(
        (b, 1), outcomes.iter().copied().collect(),
    ).unwrap();
    let diff       = v - &outcomes_arr;
    let value_loss = diff.mapv(|x| x * x).mean().unwrap_or(0.0);
    let dv         = diff.mapv(|x| 2.0 * x / b as f32);
    let dlv        = &dv * &v.mapv(|vi| vi * (1.0 - vi));
    (value_loss, dlv)
}

#[allow(clippy::too_many_arguments)]
fn backward(
    net: &mut Net, opt: &mut Optimizer,
    dlp: &Array2<f32>, dlv: &Array2<f32>,
    x: &Array2<f32>, z1: &Array2<f32>, a1: &Array2<f32>,
    z2: &Array2<f32>, a2: &Array2<f32>, z3: &Array2<f32>,
    a3: &Array2<f32>, z4: &Array2<f32>, a4: &Array2<f32>,
    b: usize,
) {
    let (lr, b1, b2, eps, t) = (opt.lr, opt.b1, opt.b2, opt.eps, opt.t);

    let (gw_lp, gb_lp, da4_p) = linear_backward(dlp, a4, &net.lp.w, b);
    opt.lp_w.step(&mut net.lp.w, &gw_lp, t, lr, b1, b2, eps);
    opt.lp_b.step(&mut net.lp.b, &gb_lp, t, lr, b1, b2, eps);

    let (gw_lv, gb_lv, da4_v) = linear_backward(dlv, a4, &net.lv.w, b);
    opt.lv_w.step(&mut net.lv.w, &gw_lv, t, lr, b1, b2, eps);
    opt.lv_b.step(&mut net.lv.b, &gb_lv, t, lr, b1, b2, eps);

    let dz4 = (da4_p + da4_v) * relu2_grad(z4.view());
    let (gw4, gb4, da3) = linear_backward(&dz4, a3, &net.l4.w, b);
    opt.l4_w.step(&mut net.l4.w, &gw4, t, lr, b1, b2, eps);
    opt.l4_b.step(&mut net.l4.b, &gb4, t, lr, b1, b2, eps);

    // L3 residual: z3 = L3(a2) + a2  →  da2 = dz3 @ w3 + dz3
    let dz3 = da3 * relu2_grad(z3.view());
    let (gw3, gb3, da2_lin) = linear_backward(&dz3, a2, &net.l3.w, b);
    let da2 = da2_lin + &dz3;
    opt.l3_w.step(&mut net.l3.w, &gw3, t, lr, b1, b2, eps);
    opt.l3_b.step(&mut net.l3.b, &gb3, t, lr, b1, b2, eps);

    // L2 residual: z2 = L2(a1) + a1  →  da1 = dz2 @ w2 + dz2
    let dz2 = da2 * relu2_grad(z2.view());
    let (gw2, gb2, da1_lin) = linear_backward(&dz2, a1, &net.l2.w, b);
    let da1 = da1_lin + &dz2;
    opt.l2_w.step(&mut net.l2.w, &gw2, t, lr, b1, b2, eps);
    opt.l2_b.step(&mut net.l2.b, &gb2, t, lr, b1, b2, eps);

    let dz1 = da1 * relu2_grad(z1.view());
    let (gw1, gb1, _) = linear_backward(&dz1, x, &net.l1.w, b);
    opt.l1_w.step(&mut net.l1.w, &gw1, t, lr, b1, b2, eps);
    opt.l1_b.step(&mut net.l1.b, &gb1, t, lr, b1, b2, eps);
}

// ── Inference helpers ─────────────────────────────────────────────────────────

pub fn softmax_probs(logits: &[f32; NUM_ACTIONS], mask: u64) -> [f32; NUM_ACTIONS] {
    let v = masked_softmax(logits, mask);
    let mut out = [0.0f32; NUM_ACTIONS];
    out.copy_from_slice(&v);
    out
}

// ── Low-level helpers ─────────────────────────────────────────────────────────

fn relu1(x: Array1<f32>) -> Array1<f32> { x.mapv(|v| v.max(0.0)) }
fn relu2(x: ndarray::ArrayView2<f32>) -> Array2<f32> { x.mapv(|v| v.max(0.0)) }
fn relu2_grad(z: ndarray::ArrayView2<f32>) -> Array2<f32> { z.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }) }
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

fn masked_softmax(logits: &[f32], mask: u64) -> Vec<f32> {
    let n = logits.len();
    let max = (0..n)
        .filter(|&i| mask & (1u64 << i) != 0)
        .map(|i| logits[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs = vec![0.0f32; n];
    if !max.is_finite() { return probs; }
    let mut sum = 0.0f32;
    for i in 0..n {
        if mask & (1u64 << i) != 0 {
            probs[i] = (logits[i] - max).exp();
            sum += probs[i];
        }
    }
    if sum > 1e-10 { for p in probs.iter_mut() { *p /= sum; } }
    probs
}

fn feats_to_array(features: &[Vec<f32>]) -> Array2<f32> {
    let b = features.len();
    let flat: Vec<f32> = features.iter().flat_map(|f| f.iter().copied()).collect();
    Array2::from_shape_vec((b, FEATURE_DIM), flat).unwrap()
}

fn linear_backward(
    upstream: &Array2<f32>,
    input: &Array2<f32>,
    w: &Array2<f32>,
    b: usize,
) -> (Array2<f32>, Array1<f32>, Array2<f32>) {
    let gw = upstream.t().dot(input) / b as f32;
    let gb = upstream.mean_axis(Axis(0)).unwrap();
    let gi = upstream.dot(w);
    (gw, gb, gi)
}
