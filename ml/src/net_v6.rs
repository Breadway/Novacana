// 486→512→512(+res)→512(+res)→512→{35,1}  +  candidate head [512+24→32→1]
// Input = 3 stacked GameView snapshots × 162 features each.
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::{rand_distr::Normal, RandomExt};
use serde::{Deserialize, Serialize};

use crate::action::NUM_ACTIONS;
use crate::candidate::CAND_DIM;
use crate::feature_v6::FEATURE_DIM_V6;

pub const INPUT_DIM: usize = FEATURE_DIM_V6 * 3; // 486

const H1: usize = 512;
const H2: usize = 512;
const H3: usize = 512;
const H4: usize = 512;
const HC: usize = 32;

// ── Flat-weight serialisation ─────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone)]
pub struct WeightsV6 {
    pub w1: Vec<f32>, pub b1: Vec<f32>,
    pub w2: Vec<f32>, pub b2: Vec<f32>,
    pub w3: Vec<f32>, pub b3: Vec<f32>,
    pub w4: Vec<f32>, pub b4: Vec<f32>,
    pub wp: Vec<f32>, pub bp: Vec<f32>,
    pub wv: Vec<f32>, pub bv: Vec<f32>,
    pub wc1: Vec<f32>, pub bc1: Vec<f32>,
    pub wc2: Vec<f32>, pub bc2: Vec<f32>,
}

// ── Layer ─────────────────────────────────────────────────────────────────────

struct Layer {
    w: Array2<f32>,
    b: Array1<f32>,
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

pub struct NetV6 {
    l1: Layer,
    l2: Layer,
    l3: Layer,
    l4: Layer,
    lp: Layer,
    lv: Layer,
    lc1: Layer,
    lc2: Layer,
}

impl NetV6 {
    pub fn new() -> Self {
        Self {
            l1:  Layer::rand(INPUT_DIM, H1),
            l2:  Layer::rand(H1, H2),
            l3:  Layer::rand(H2, H3),
            l4:  Layer::rand(H3, H4),
            lp:  Layer::rand(H4, NUM_ACTIONS),
            lv:  Layer::rand(H4, 1),
            lc1: Layer::rand(H4 + CAND_DIM, HC),
            lc2: Layer::rand(HC, 1),
        }
    }

    pub fn from_weights(w: &WeightsV6) -> Self {
        let load2 = |data: &[f32], rows: usize, cols: usize| {
            Array2::from_shape_vec((rows, cols), data.to_vec()).unwrap()
        };
        let load1 = |data: &[f32]| Array1::from_vec(data.to_vec());
        Self {
            l1:  Layer { w: load2(&w.w1,  H1, INPUT_DIM),         b: load1(&w.b1)  },
            l2:  Layer { w: load2(&w.w2,  H2, H1),                b: load1(&w.b2)  },
            l3:  Layer { w: load2(&w.w3,  H3, H2),                b: load1(&w.b3)  },
            l4:  Layer { w: load2(&w.w4,  H4, H3),                b: load1(&w.b4)  },
            lp:  Layer { w: load2(&w.wp,  NUM_ACTIONS, H4),       b: load1(&w.bp)  },
            lv:  Layer { w: load2(&w.wv,  1, H4),                 b: load1(&w.bv)  },
            lc1: Layer { w: load2(&w.wc1, HC, H4 + CAND_DIM),    b: load1(&w.bc1) },
            lc2: Layer { w: load2(&w.wc2, 1,  HC),               b: load1(&w.bc2) },
        }
    }

    pub fn to_weights(&self) -> WeightsV6 {
        WeightsV6 {
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

    pub fn body_forward(&self, features: &[f32; INPUT_DIM]) -> Array1<f32> {
        let x  = ndarray::aview1(features);
        let a1 = relu1(self.l1.forward_single(x));
        let a2 = relu1(self.l2.forward_single(a1.view()) + &a1);
        let a3 = relu1(self.l3.forward_single(a2.view()) + &a2);
        relu1(self.l4.forward_single(a3.view()))
    }

    pub fn body_forward_batch(&self, features: &[[f32; INPUT_DIM]]) -> Array2<f32> {
        let b = features.len();
        let flat: Vec<f32> = features.iter().flat_map(|f| f.iter().copied()).collect();
        let x  = Array2::from_shape_vec((b, INPUT_DIM), flat).unwrap();
        let a1 = relu2(self.l1.forward(&x).view());
        let z2 = self.l2.forward(&a1) + &a1;
        let a2 = relu2(z2.view());
        let z3 = self.l3.forward(&a2) + &a2;
        let a3 = relu2(z3.view());
        relu2(self.l4.forward(&a3).view())
    }

    pub fn infer_with_body(&self, features: &[f32; INPUT_DIM]) -> ([f32; NUM_ACTIONS], f32, Array1<f32>) {
        let body  = self.body_forward(features);
        let lp    = self.lp.forward_single(body.view());
        let value = sigmoid(self.lv.forward_single(body.view())[0]);
        let mut out = [0.0f32; NUM_ACTIONS];
        for (i, v) in lp.iter().enumerate() { out[i] = *v; }
        (out, value, body)
    }

    pub fn infer_stochastic_with_body(
        &self,
        features: &[f32; INPUT_DIM],
        mask: u64,
        temperature: f32,
        u: f32,
    ) -> (usize, f32, f32, Array1<f32>) {
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

    pub fn score_candidates(&self, body: &Array1<f32>, cands: &[[f32; CAND_DIM]]) -> Vec<f32> {
        cands.iter().map(|c| {
            let mut inp = Vec::with_capacity(H4 + CAND_DIM);
            inp.extend_from_slice(body.as_slice().unwrap());
            inp.extend_from_slice(c);
            let h = relu1(self.lc1.forward_single(ndarray::aview1(&inp)));
            self.lc2.forward_single(h.view())[0]
        }).collect()
    }
}

// ── Optimizer ─────────────────────────────────────────────────────────────────

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

pub struct OptimizerV6 {
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
    /// LR multiplier for the candidate head (3–5× recommended).
    pub cand_lr_mult: f32,
    b1: f32, b2: f32, eps: f32,
}

impl OptimizerV6 {
    pub fn new(lr: f32, cand_lr_mult: f32) -> Self {
        Self {
            l1_w:  Adam2::new((H1, INPUT_DIM)),    l1_b:  Adam1::new(H1),
            l2_w:  Adam2::new((H2, H1)),           l2_b:  Adam1::new(H2),
            l3_w:  Adam2::new((H3, H2)),           l3_b:  Adam1::new(H3),
            l4_w:  Adam2::new((H4, H3)),           l4_b:  Adam1::new(H4),
            lp_w:  Adam2::new((NUM_ACTIONS, H4)),  lp_b:  Adam1::new(NUM_ACTIONS),
            lv_w:  Adam2::new((1, H4)),            lv_b:  Adam1::new(1),
            lc1_w: Adam2::new((HC, H4 + CAND_DIM)), lc1_b: Adam1::new(HC),
            lc2_w: Adam2::new((1, HC)),            lc2_b: Adam1::new(1),
            t: 0, lr, cand_lr_mult, b1: 0.9, b2: 0.999, eps: 1e-8,
        }
    }
}

// ── RL training step (PPO + entropy bonus) ────────────────────────────────────

pub struct RlBatchV6<'a> {
    pub features:      &'a [Vec<f32>],
    pub masks:         &'a [u64],
    pub actions:       &'a [u8],
    pub outcomes:      &'a [f32],
    pub old_log_probs: &'a [f32],
    /// Per-sample importance-sampling weights (from PER). Use 1.0 for uniform.
    pub is_weights:    &'a [f32],
}

pub fn rl_train_step_v6(
    net: &mut NetV6,
    opt: &mut OptimizerV6,
    batch: &RlBatchV6,
    entropy_coeff: f32,
    clip_eps: f32,
) -> (f32, f32) {
    let b = batch.features.len();
    assert!(b > 0);

    let (x, z1, a1, z2, a2, z3, a3, z4, a4, lp, _lv, v) = forward_v6(net, batch.features);

    let mut dlp = Array2::<f32>::zeros((b, NUM_ACTIONS));
    let mut policy_loss = 0.0f32;

    for i in 0..b {
        let row          = lp.row(i);
        let mask         = batch.masks[i];
        let action       = batch.actions[i] as usize;
        let probs        = masked_softmax(row.as_slice().unwrap(), mask);
        let new_log_prob = probs[action].max(1e-9).ln();
        let old_log_prob = batch.old_log_probs[i];
        let is_w         = batch.is_weights[i];

        let ratio     = (new_log_prob - old_log_prob).exp();
        let advantage = batch.outcomes[i] - v[[i, 0]];
        let entropy: f32 = probs.iter().filter(|&&p| p > 1e-9).map(|&p| -p * p.ln()).sum();

        let is_clipped = (ratio > 1.0 + clip_eps && advantage > 0.0)
                      || (ratio < 1.0 - clip_eps && advantage < 0.0);

        let effective_ratio = if is_clipped {
            ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
        } else {
            ratio
        };
        policy_loss += is_w * (-effective_ratio * advantage - entropy_coeff * entropy) / b as f32;

        for j in 0..NUM_ACTIONS {
            if mask & (1u64 << j) != 0 {
                let ent_grad = if probs[j] > 1e-9 {
                    probs[j] * (probs[j].ln() + entropy)
                } else {
                    0.0
                };
                let policy_grad = if is_clipped {
                    0.0
                } else {
                    advantage * ratio * (probs[j] - if j == action { 1.0 } else { 0.0 })
                };
                dlp[[i, j]] = is_w * (policy_grad - entropy_coeff * ent_grad) / b as f32;
            }
        }
    }

    let (value_loss, dlv) = value_loss_and_grad_weighted(&v, batch.outcomes, batch.is_weights, b);

    opt.t += 1;
    backward_v6(net, opt, &dlp, &dlv, &x, &z1, &a1, &z2, &a2, &z3, &a3, &z4, &a4, b);
    (policy_loss, value_loss)
}

// ── Candidate head training step ─────────────────────────────────────────────

pub struct CandidateBatchV6<'a> {
    pub features:    &'a [Vec<f32>],
    pub candidates:  &'a [Vec<[f32; CAND_DIM]>],
    pub chosen_idxs: &'a [u8],
}

pub fn candidate_train_step_v6(net: &mut NetV6, opt: &mut OptimizerV6, batch: &CandidateBatchV6) -> f32 {
    let n = batch.features.len();
    assert!(n > 0);

    let mut gw_c1 = Array2::<f32>::zeros((HC, H4 + CAND_DIM));
    let mut gb_c1 = Array1::<f32>::zeros(HC);
    let mut gw_c2 = Array2::<f32>::zeros((1, HC));
    let mut gb_c2 = Array1::<f32>::zeros(1);
    let mut total_loss = 0.0f32;

    for i in 0..n {
        let feats_slice: &[f32; INPUT_DIM] = batch.features[i].as_slice()
            .try_into()
            .unwrap_or_else(|_| panic!("v6 feature dim mismatch: got {}", batch.features[i].len()));
        let body = net.body_forward(feats_slice);

        let cands = &batch.candidates[i];
        let k = cands.len();
        if k == 0 { continue; }

        let mut inputs: Vec<Array1<f32>> = Vec::with_capacity(k);
        let mut hiddens: Vec<Array1<f32>> = Vec::with_capacity(k);
        let mut scores = Vec::with_capacity(k);

        for c in cands {
            let mut inp = Vec::with_capacity(H4 + CAND_DIM);
            inp.extend_from_slice(body.as_slice().unwrap());
            inp.extend_from_slice(c);
            let inp_arr = Array1::from_vec(inp);
            let h = relu1(net.lc1.forward_single(inp_arr.view()));
            let s = net.lc2.forward_single(h.view())[0];
            inputs.push(inp_arr);
            hiddens.push(h);
            scores.push(s);
        }

        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
        let sum_e: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e / sum_e.max(1e-10)).collect();

        let chosen = batch.chosen_idxs[i] as usize;
        total_loss -= probs[chosen].max(1e-9).ln() / n as f32;

        for j in 0..k {
            let dp = (probs[j] - if j == chosen { 1.0 } else { 0.0 }) / n as f32;
            let d_lc2_w = Array2::from_shape_vec((1, HC),
                hiddens[j].iter().map(|&h| dp * h).collect()).unwrap();
            let d_lc2_b = Array1::from_vec(vec![dp]);
            gw_c2 = gw_c2 + d_lc2_w;
            gb_c2 = gb_c2 + d_lc2_b;

            let d_h_pre = dp * net.lc2.w.row(0).to_owned();
            let d_h = &d_h_pre * &hiddens[j].mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
            let d_lc1_w = Array2::from_shape_vec(
                (HC, H4 + CAND_DIM),
                d_h.iter().flat_map(|&dh| inputs[j].iter().map(move |&x| dh * x)).collect(),
            ).unwrap();
            let d_lc1_b = d_h.clone();
            gw_c1 = gw_c1 + d_lc1_w;
            gb_c1 = gb_c1 + d_lc1_b;
        }
    }

    let (lr, b1, b2, eps, t) = (opt.lr, opt.b1, opt.b2, opt.eps, opt.t);
    let clr = lr * opt.cand_lr_mult;
    opt.t += 1;
    opt.lc1_w.step(&mut net.lc1.w, &gw_c1, t, clr, b1, b2, eps);
    opt.lc1_b.step(&mut net.lc1.b, &gb_c1, t, clr, b1, b2, eps);
    opt.lc2_w.step(&mut net.lc2.w, &gw_c2, t, clr, b1, b2, eps);
    opt.lc2_b.step(&mut net.lc2.b, &gb_c2, t, clr, b1, b2, eps);

    total_loss
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn forward_v6(net: &NetV6, features: &[Vec<f32>])
    -> (Array2<f32>, Array2<f32>, Array2<f32>,
        Array2<f32>, Array2<f32>, Array2<f32>,
        Array2<f32>, Array2<f32>, Array2<f32>,
        Array2<f32>, Array2<f32>, Array2<f32>)
{
    let b = features.len();
    let flat: Vec<f32> = features.iter().flat_map(|f| f.iter().copied()).collect();
    let x  = Array2::from_shape_vec((b, INPUT_DIM), flat).unwrap();
    let z1 = net.l1.forward(&x);
    let a1 = relu2(z1.view());
    let z2 = net.l2.forward(&a1) + &a1;
    let a2 = relu2(z2.view());
    let z3 = net.l3.forward(&a2) + &a2;
    let a3 = relu2(z3.view());
    let z4 = net.l4.forward(&a3);
    let a4 = relu2(z4.view());
    let lp = net.lp.forward(&a4);
    let lv = net.lv.forward(&a4);
    let v  = lv.mapv(sigmoid);
    (x, z1, a1, z2, a2, z3, a3, z4, a4, lp, lv, v)
}

fn value_loss_and_grad_weighted(
    v: &Array2<f32>,
    outcomes: &[f32],
    is_weights: &[f32],
    b: usize,
) -> (f32, Array2<f32>) {
    let outcomes_arr = Array2::from_shape_vec(
        (b, 1), outcomes.iter().copied().collect(),
    ).unwrap();
    let diff = v - &outcomes_arr;
    let mut value_loss = 0.0f32;
    let mut dlv = Array2::<f32>::zeros((b, 1));
    for i in 0..b {
        let d = diff[[i, 0]];
        let vi = v[[i, 0]];
        value_loss += is_weights[i] * d * d / b as f32;
        dlv[[i, 0]] = is_weights[i] * 2.0 * d * vi * (1.0 - vi) / b as f32;
    }
    (value_loss, dlv)
}

#[allow(clippy::too_many_arguments)]
fn backward_v6(
    net: &mut NetV6, opt: &mut OptimizerV6,
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

    let dz3 = da3 * relu2_grad(z3.view());
    let (gw3, gb3, da2_lin) = linear_backward(&dz3, a2, &net.l3.w, b);
    let da2 = da2_lin + &dz3;
    opt.l3_w.step(&mut net.l3.w, &gw3, t, lr, b1, b2, eps);
    opt.l3_b.step(&mut net.l3.b, &gb3, t, lr, b1, b2, eps);

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
