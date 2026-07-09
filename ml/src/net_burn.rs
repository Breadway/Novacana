// Burn-based training backend.
// Game generation and inference run on NetV6 (ndarray/CPU).
// All gradient steps run here (WGPU GPU or NdArray CPU fallback).

use burn::{
    module::{AutodiffModule, Module},
    nn::{Linear, LinearConfig},
    grad_clipping::GradientClippingConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::{activation, backend::AutodiffBackend, Int, TensorData},
};

use crate::action::NUM_ACTIONS;
use crate::candidate::CAND_DIM;
use crate::net::{Weights, INPUT_DIM};

const H:  usize = 512;
const HC: usize = 128;

// 16, not 8: the collapsed PlayDiscovery abstract action's matching set is a
// cross-product (discovery types in hand × incomplete systems) and can exceed 8.
pub const MAX_CANDS: usize = 16;

// ── Network definition ────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct NetBurn<B: Backend> {
    l1:  Linear<B>,
    l2:  Linear<B>,
    l3:  Linear<B>,
    l4:  Linear<B>,
    lp:  Linear<B>,   // policy:   H  → NUM_ACTIONS
    lv:  Linear<B>,   // value:    H  → 1
    lc1: Linear<B>,   // cand h:   H+CAND_DIM → HC
    lc2: Linear<B>,   // cand out: HC → 1
}

// ndarray weights are [out, in]; Burn Linear stores [in, out].
fn t_flat(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

// Replace a 2-D param's tensor data while preserving require_grad.
// data is in [out_dim, in_dim] (ndarray) format; stored as [in_dim, out_dim] (burn row layout).
fn load_w2<B: Backend>(
    param: burn::module::Param<Tensor<B, 2>>,
    data: &[f32], out_dim: usize, in_dim: usize,
    device: &B::Device,
) -> burn::module::Param<Tensor<B, 2>> {
    let flat = t_flat(data, out_dim, in_dim);
    param.map(|t| {
        let rg = t.is_require_grad();
        Tensor::from_data(TensorData::new(flat, [in_dim, out_dim]), device)
            .set_require_grad(rg)
    })
}

// Replace a 1-D param's tensor data while preserving require_grad.
fn load_w1<B: Backend>(
    param: burn::module::Param<Tensor<B, 1>>,
    data: &[f32],
    device: &B::Device,
) -> burn::module::Param<Tensor<B, 1>> {
    let len = data.len();
    let flat = data.to_vec();
    param.map(|t| {
        let rg = t.is_require_grad();
        Tensor::from_data(TensorData::new(flat, [len]), device)
            .set_require_grad(rg)
    })
}

impl<B: Backend> NetBurn<B> {
    // Initialize from pre-trained Weights.
    // Starts with LinearConfig::init (proper grad tracking), then replaces tensor
    // data via Param::map which preserves require_grad status.
    pub fn from_weights(w: &Weights, device: &B::Device) -> Self {
        let mut net = Self::fresh(device);
        net.l1.weight = load_w2(net.l1.weight, &w.w1, H,           INPUT_DIM,    device);
        net.l1.bias   = net.l1.bias.map(|p| load_w1(p, &w.b1,  device));
        net.l2.weight = load_w2(net.l2.weight, &w.w2, H,           H,            device);
        net.l2.bias   = net.l2.bias.map(|p| load_w1(p, &w.b2,  device));
        net.l3.weight = load_w2(net.l3.weight, &w.w3, H,           H,            device);
        net.l3.bias   = net.l3.bias.map(|p| load_w1(p, &w.b3,  device));
        net.l4.weight = load_w2(net.l4.weight, &w.w4, H,           H,            device);
        net.l4.bias   = net.l4.bias.map(|p| load_w1(p, &w.b4,  device));
        net.lp.weight = load_w2(net.lp.weight, &w.wp, NUM_ACTIONS, H,            device);
        net.lp.bias   = net.lp.bias.map(|p| load_w1(p, &w.bp,  device));
        net.lv.weight = load_w2(net.lv.weight, &w.wv, 1,           H,            device);
        net.lv.bias   = net.lv.bias.map(|p| load_w1(p, &w.bv,  device));
        net.lc1.weight = load_w2(net.lc1.weight, &w.wc1, HC, H + CAND_DIM, device);
        net.lc1.bias   = net.lc1.bias.map(|p| load_w1(p, &w.bc1, device));
        net.lc2.weight = load_w2(net.lc2.weight, &w.wc2, 1,  HC,           device);
        net.lc2.bias   = net.lc2.bias.map(|p| load_w1(p, &w.bc2, device));
        net
    }

    // Fresh random init using LinearConfig so grad tracking is correct from the start.
    pub fn fresh(device: &B::Device) -> Self {
        Self {
            l1:  LinearConfig::new(INPUT_DIM,    H).init(device),
            l2:  LinearConfig::new(H,            H).init(device),
            l3:  LinearConfig::new(H,            H).init(device),
            l4:  LinearConfig::new(H,            H).init(device),
            lp:  LinearConfig::new(H,            NUM_ACTIONS).init(device),
            lv:  LinearConfig::new(H,            1).init(device),
            lc1: LinearConfig::new(H + CAND_DIM, HC).init(device),
            lc2: LinearConfig::new(HC,           1).init(device),
        }
    }

    pub fn to_weights(&self) -> Weights {
        let get2_t = |l: &Linear<B>, in_dim: usize, out_dim: usize| -> Vec<f32> {
            let data = l.weight.val().into_data().to_vec::<f32>().expect("weight f32");
            t_flat(&data, in_dim, out_dim)
        };
        let get1 = |l: &Linear<B>| -> Vec<f32> {
            l.bias.as_ref().unwrap().val().into_data().to_vec::<f32>().expect("bias f32")
        };
        Weights {
            w1: get2_t(&self.l1, INPUT_DIM,     H),            b1: get1(&self.l1),
            w2: get2_t(&self.l2, H,             H),            b2: get1(&self.l2),
            w3: get2_t(&self.l3, H,             H),            b3: get1(&self.l3),
            w4: get2_t(&self.l4, H,             H),            b4: get1(&self.l4),
            wp: get2_t(&self.lp, H,             NUM_ACTIONS),  bp: get1(&self.lp),
            wv: get2_t(&self.lv, H,             1),            bv: get1(&self.lv),
            wc1: get2_t(&self.lc1, H + CAND_DIM, HC),         bc1: get1(&self.lc1),
            wc2: get2_t(&self.lc2, HC,           1),           bc2: get1(&self.lc2),
        }
    }
}

// ── Shared body forward ───────────────────────────────────────────────────────

pub fn body_forward<B: Backend>(net: &NetBurn<B>, x: Tensor<B, 2>) -> Tensor<B, 2> {
    let a1 = activation::relu(net.l1.forward(x));
    let a2 = activation::relu(net.l2.forward(a1.clone()) + a1);
    let a3 = activation::relu(net.l3.forward(a2.clone()) + a2);
    activation::relu(net.l4.forward(a3))
}

// ── Optimizer factory ─────────────────────────────────────────────────────────

pub fn make_optimizer<B: AutodiffBackend>() -> impl Optimizer<NetBurn<B>, B> {
    AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init()
}

// ── PPO training step ─────────────────────────────────────────────────────────

pub fn rl_train_step_burn<B: AutodiffBackend, O: Optimizer<NetBurn<B>, B>>(
    net:           NetBurn<B>,
    optim:         &mut O,
    lr:            f64,
    feats:         &[Vec<f32>],
    masks:         &[u64],
    actions:       &[u8],
    outcomes:      &[f32],
    old_lps:       &[f32],
    is_weights_in: &[f32],
    entropy_coeff: f32,
    clip_eps:      f32,
    device:        &B::Device,
) -> (NetBurn<B>, f32, f32, Vec<f32>) {
    let b = feats.len();
    assert!(b > 0);

    let feat_flat: Vec<f32> = feats.iter().flat_map(|f| f.iter().copied()).collect();
    let x = Tensor::<B, 2>::from_data(TensorData::new(feat_flat, [b, INPUT_DIM]), device);

    let body   = body_forward(&net, x);
    let logits = net.lp.forward(body.clone());
    let values = activation::sigmoid(net.lv.forward(body));

    let mask_flat: Vec<f32> = masks.iter().flat_map(|&m| {
        (0..NUM_ACTIONS).map(move |i| if m & (1u64 << i) != 0 { 0.0_f32 } else { -1e9_f32 })
    }).collect();
    let mask_t  = Tensor::<B, 2>::from_data(TensorData::new(mask_flat, [b, NUM_ACTIONS]), device);
    let masked  = logits + mask_t;
    let log_p   = activation::log_softmax(masked.clone(), 1);

    let act_data: Vec<i32> = actions.iter().map(|&a| a as i32).collect();
    let act_t   = Tensor::<B, 2, Int>::from_data(TensorData::new(act_data, [b, 1]), device);
    let new_lp  = log_p.clone().gather(1, act_t);

    let old_lp_t   = Tensor::<B, 2>::from_data(TensorData::new(old_lps.to_vec(), [b, 1]), device);
    let ratio      = (new_lp - old_lp_t).exp();
    let outcomes_t = Tensor::<B, 2>::from_data(TensorData::new(outcomes.to_vec(), [b, 1]), device);
    // Detach: the policy surrogate must not backprop through the value head.
    let advantage  = (outcomes_t.clone() - values.clone()).detach();
    let surr1 = ratio.clone() * advantage.clone();
    let surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantage.clone();
    let ppo_obj = (surr1.clone() + surr2.clone() - (surr1 - surr2).abs()) * 0.5;

    let probs   = activation::softmax(masked, 1);
    let entropy = -(probs * log_p).sum_dim(1);

    let is_t = Tensor::<B, 2>::from_data(TensorData::new(is_weights_in.to_vec(), [b, 1]), device);

    let policy_loss = -(is_t.clone() * (ppo_obj + entropy * entropy_coeff)).mean();
    // GAE returns can fall outside [0,1] due to shaped rewards while the value
    // head is sigmoid — clamp the value-loss target only (advantage above stays
    // on the unclamped returns).
    let value_tgt   = outcomes_t.clamp(0.0, 1.0);
    let diff        = values - value_tgt;
    let value_loss  = (is_t * diff.clone() * diff).mean();

    let pl_val: f32 = policy_loss.clone().into_data().to_vec::<f32>().unwrap_or_default()[0];
    let vl_val: f32 = value_loss.clone().into_data().to_vec::<f32>().unwrap_or_default()[0];
    let td_errors: Vec<f32> = advantage.abs().into_data().to_vec::<f32>().unwrap_or_default();

    let total = policy_loss + value_loss;
    let grads = GradientsParams::from_grads(total.backward(), &net);
    (optim.step(lr, net, grads), pl_val, vl_val, td_errors)
}

// ── AlphaZero training step ───────────────────────────────────────────────────
//
// Policy loss: cross-entropy against MCTS visit distribution (soft targets).
// Value loss:  MSE against game outcome.
// No PPO clipping or IS weights — this is supervised learning against MCTS targets.

pub fn az_train_step_burn<B: AutodiffBackend, O: Optimizer<NetBurn<B>, B>>(
    net:         NetBurn<B>,
    optim:       &mut O,
    lr:          f64,
    feats:       &[Vec<f32>],
    visit_dists: &[Vec<f32>],  // [B][NUM_ACTIONS] soft policy targets (sum to 1)
    outcomes:    &[f32],
    masks:       &[u64],
    device:      &B::Device,
) -> (NetBurn<B>, f32, f32) {
    let b = feats.len();
    assert!(b > 0);

    let feat_flat: Vec<f32> = feats.iter().flat_map(|f| f.iter().copied()).collect();
    let x = Tensor::<B, 2>::from_data(TensorData::new(feat_flat, [b, INPUT_DIM]), device);

    let body   = body_forward(&net, x);
    let logits = net.lp.forward(body.clone());
    let values = activation::sigmoid(net.lv.forward(body));

    // Masked log-softmax: illegal actions get -1e9 before softmax
    let mask_flat: Vec<f32> = masks.iter().flat_map(|&m| {
        (0..NUM_ACTIONS).map(move |i| if m & (1u64 << i) != 0 { 0.0f32 } else { -1e9f32 })
    }).collect();
    let mask_t = Tensor::<B, 2>::from_data(TensorData::new(mask_flat, [b, NUM_ACTIONS]), device);
    let log_p  = activation::log_softmax(logits + mask_t, 1); // [B, NUM_ACTIONS]

    // CE against visit distribution: -sum(visit * log_p, dim=1).mean()
    // illegal entries have visit=0 so 0.0 * finite = 0.0 (no NaN risk)
    let visit_flat: Vec<f32> = visit_dists.iter().flat_map(|v| v.iter().copied()).collect();
    let visit_t    = Tensor::<B, 2>::from_data(TensorData::new(visit_flat, [b, NUM_ACTIONS]), device);
    let policy_loss = -(visit_t * log_p).sum_dim(1).mean();

    // Value MSE
    let tgt        = Tensor::<B, 2>::from_data(TensorData::new(outcomes.to_vec(), [b, 1]), device);
    let diff       = values - tgt;
    let value_loss = (diff.clone() * diff).mean();

    let pl: f32 = policy_loss.clone().into_data().to_vec::<f32>().unwrap_or_default()[0];
    let vl: f32 = value_loss.clone().into_data().to_vec::<f32>().unwrap_or_default()[0];

    let total = policy_loss + value_loss;
    let grads = GradientsParams::from_grads(total.backward(), &net);
    (optim.step(lr, net, grads), pl, vl)
}

// ── Supervised pretraining steps ─────────────────────────────────────────────

// Body forward for head-only fine-tune steps. freeze=true runs the trunk on
// the INNER (non-autodiff) backend and re-imports the activations as a
// constant leaf. This both stops gradients at the trunk and avoids building
// the trunk's autodiff graph at all — a detach()ed subgraph is never reached
// by backward(), and burn only frees graph state on backward, so detach leaks
// the whole trunk graph every batch.
fn frozen_or_live_body<B: AutodiffBackend>(
    net:       &NetBurn<B>,
    feat_flat: Vec<f32>,
    b:         usize,
    freeze:    bool,
    device:    &B::Device,
) -> Tensor<B, 2> {
    if freeze {
        let x = Tensor::<B::InnerBackend, 2>::from_data(TensorData::new(feat_flat, [b, INPUT_DIM]), device);
        Tensor::from_inner(body_forward(&net.valid(), x))
    } else {
        let x = Tensor::<B, 2>::from_data(TensorData::new(feat_flat, [b, INPUT_DIM]), device);
        body_forward(net, x)
    }
}

pub fn pretrain_value_step_burn<B: AutodiffBackend, O: Optimizer<NetBurn<B>, B>>(
    net:          NetBurn<B>,
    optim:        &mut O,
    lr:           f64,
    feats:        &[Vec<f32>],
    outcomes:     &[f32],
    freeze_trunk: bool,
    device:       &B::Device,
) -> (NetBurn<B>, f32) {
    let b = feats.len();
    assert!(b > 0);
    let feat_flat: Vec<f32> = feats.iter().flat_map(|f| f.iter().copied()).collect();
    // When freeze_trunk, run the body on the INNER backend so gradients train
    // only the value head: the value head becomes a readout on the
    // policy-optimized trunk, preventing value MSE from degrading the policy
    // (the epoch-8→30 policy regression). Inner backend, not detach(): a
    // detached trunk subgraph is never reached by backward(), and burn only
    // frees autodiff graph state on backward — detach leaked ~5GB/epoch.
    let body = frozen_or_live_body(&net, feat_flat, b, freeze_trunk, device);
    let values = activation::sigmoid(net.lv.forward(body));
    let tgt    = Tensor::<B, 2>::from_data(TensorData::new(outcomes.to_vec(), [b, 1]), device);
    let diff   = values - tgt;
    let loss   = (diff.clone() * diff).mean();
    let lv = loss.clone().into_data().to_vec::<f32>().unwrap_or_default()[0];
    let grads = GradientsParams::from_grads(loss.backward(), &net);
    (optim.step(lr, net, grads), lv)
}

pub fn pretrain_policy_step_burn<B: AutodiffBackend, O: Optimizer<NetBurn<B>, B>>(
    net:             NetBurn<B>,
    optim:           &mut O,
    lr:              f64,
    feats:           &[Vec<f32>],
    masks:           &[u64],
    actions:         &[u8],
    label_smoothing: f32,
    device:          &B::Device,
) -> (NetBurn<B>, f32) {
    let b = feats.len();
    assert!(b > 0);
    let feat_flat: Vec<f32> = feats.iter().flat_map(|f| f.iter().copied()).collect();
    let x      = Tensor::<B, 2>::from_data(TensorData::new(feat_flat, [b, INPUT_DIM]), device);
    let logits = net.lp.forward(body_forward(&net, x));
    let mask_flat: Vec<f32> = masks.iter().flat_map(|&m| {
        (0..NUM_ACTIONS).map(move |i| if m & (1u64 << i) != 0 { 0.0f32 } else { -1e9f32 })
    }).collect();
    let mask_t = Tensor::<B, 2>::from_data(TensorData::new(mask_flat, [b, NUM_ACTIONS]), device);
    let log_p  = activation::log_softmax(logits + mask_t, 1);
    let act_t  = Tensor::<B, 2, Int>::from_data(
        TensorData::new(actions.iter().map(|&a| a as i32).collect::<Vec<_>>(), [b, 1]), device,
    );
    let hard = -log_p.clone().gather(1, act_t);
    let loss = if label_smoothing > 0.0 {
        let legal_flat: Vec<f32> = masks.iter().flat_map(|&m| {
            (0..NUM_ACTIONS).map(move |i| if m & (1u64 << i) != 0 { 1.0f32 } else { 0.0f32 })
        }).collect();
        let legal_ind = Tensor::<B, 2>::from_data(TensorData::new(legal_flat, [b, NUM_ACTIONS]), device);
        let num_legal = legal_ind.clone().sum_dim(1);
        let uniform   = -(log_p * legal_ind).sum_dim(1) / num_legal;
        let eps = label_smoothing;
        (hard.clone() * (1.0 - eps) + uniform * eps).mean()
    } else {
        hard.clone().mean()
    };
    let lv = hard.mean().into_data().to_vec::<f32>().unwrap_or_default()[0];
    let grads = GradientsParams::from_grads(loss.backward(), &net);
    (optim.step(lr, net, grads), lv)
}

/// Forward-only policy evaluation on a held-out batch.
/// Returns (argmax_correct_count, summed_hard_cross_entropy) so callers can
/// accumulate across batches and checkpoint on *generalization* rather than
/// epoch count. Takes a plain-Backend net: autodiff callers must pass
/// `net.valid()` — forward passes on the autodiff backend accumulate graph
/// state that only backward() releases (~600MB/epoch leak, earlyoom kill).
pub fn policy_eval_burn<B: Backend>(
    net:     &NetBurn<B>,
    feats:   &[Vec<f32>],
    masks:   &[u64],
    actions: &[u8],
    device:  &B::Device,
) -> (usize, f32) {
    let b = feats.len();
    if b == 0 { return (0, 0.0); }
    let feat_flat: Vec<f32> = feats.iter().flat_map(|f| f.iter().copied()).collect();
    let x      = Tensor::<B, 2>::from_data(TensorData::new(feat_flat, [b, INPUT_DIM]), device);
    let logits = net.lp.forward(body_forward(net, x));
    let mask_flat: Vec<f32> = masks.iter().flat_map(|&m| {
        (0..NUM_ACTIONS).map(move |i| if m & (1u64 << i) != 0 { 0.0f32 } else { -1e9f32 })
    }).collect();
    let mask_t = Tensor::<B, 2>::from_data(TensorData::new(mask_flat, [b, NUM_ACTIONS]), device);
    let masked = (logits + mask_t).into_data().to_vec::<f32>().unwrap_or_default();

    let mut correct = 0usize;
    let mut ce_sum  = 0.0f32;
    for i in 0..b {
        let row = &masked[i * NUM_ACTIONS..(i + 1) * NUM_ACTIONS];
        let argmax = row.iter().enumerate()
            .max_by(|a, c| a.1.partial_cmp(c.1).unwrap())
            .map(|(j, _)| j).unwrap_or(0);
        if argmax == actions[i] as usize { correct += 1; }
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse = m + row.iter().map(|v| (v - m).exp()).sum::<f32>().ln();
        ce_sum += lse - row[actions[i] as usize];
    }
    (correct, ce_sum)
}

/// Combined policy CE + value MSE in one backward pass.
pub fn pretrain_joint_step_burn<B: AutodiffBackend, O: Optimizer<NetBurn<B>, B>>(
    net:             NetBurn<B>,
    optim:           &mut O,
    lr:              f64,
    feats:           &[Vec<f32>],
    masks:           &[u64],
    actions:         &[u8],
    outcomes:        &[f32],
    label_smoothing: f32,
    device:          &B::Device,
) -> (NetBurn<B>, f32, f32) {
    let b = feats.len();
    assert!(b > 0);
    let feat_flat: Vec<f32> = feats.iter().flat_map(|f| f.iter().copied()).collect();
    let x    = Tensor::<B, 2>::from_data(TensorData::new(feat_flat, [b, INPUT_DIM]), device);
    let body = body_forward(&net, x);

    let logits = net.lp.forward(body.clone());
    let mask_flat: Vec<f32> = masks.iter().flat_map(|&m| {
        (0..NUM_ACTIONS).map(move |i| if m & (1u64 << i) != 0 { 0.0f32 } else { -1e9f32 })
    }).collect();
    let mask_t = Tensor::<B, 2>::from_data(TensorData::new(mask_flat, [b, NUM_ACTIONS]), device);
    let log_p  = activation::log_softmax(logits + mask_t, 1);
    let act_t  = Tensor::<B, 2, Int>::from_data(
        TensorData::new(actions.iter().map(|&a| a as i32).collect::<Vec<_>>(), [b, 1]), device,
    );
    // Hard cross-entropy against AdvCPU's chosen action.
    let hard = -log_p.clone().gather(1, act_t);          // [b, 1]
    // Masked label smoothing: spread `eps` mass uniformly over the *legal* set,
    // so the policy keeps nonzero probability on alternatives. This directly
    // attacks the near-degenerate delta-function policy (agree_mass≈1.0) that
    // leaves PPO with no gradient signal on alternative actions.
    let policy_loss = if label_smoothing > 0.0 {
        let legal_flat: Vec<f32> = masks.iter().flat_map(|&m| {
            (0..NUM_ACTIONS).map(move |i| if m & (1u64 << i) != 0 { 1.0f32 } else { 0.0f32 })
        }).collect();
        let legal_ind = Tensor::<B, 2>::from_data(TensorData::new(legal_flat, [b, NUM_ACTIONS]), device);
        let num_legal = legal_ind.clone().sum_dim(1);                  // [b, 1]
        let uniform   = -(log_p * legal_ind).sum_dim(1) / num_legal;   // [b, 1]
        let eps = label_smoothing;
        (hard.clone() * (1.0 - eps) + uniform * eps).mean()
    } else {
        hard.clone().mean()
    };

    let values     = activation::sigmoid(net.lv.forward(body));
    let tgt        = Tensor::<B, 2>::from_data(TensorData::new(outcomes.to_vec(), [b, 1]), device);
    let diff       = values - tgt;
    let value_loss = (diff.clone() * diff).mean();

    let pl = hard.mean().into_data().to_vec::<f32>().unwrap_or_default()[0];
    let vl = value_loss.clone().into_data().to_vec::<f32>().unwrap_or_default()[0];

    let total = policy_loss + value_loss;
    let grads = GradientsParams::from_grads(total.backward(), &net);
    (optim.step(lr, net, grads), pl, vl)
}

/// CE loss on candidate head. Used for both pretraining and RL/AZ fine-tuning.
pub fn cand_step_burn<B: AutodiffBackend, O: Optimizer<NetBurn<B>, B>>(
    net:          NetBurn<B>,
    optim:        &mut O,
    lr:           f64,
    feats:        &[Vec<f32>],
    cands_flat:   &[Vec<f32>],
    chosen:       &[u8],
    advantages:   Option<&[f32]>,
    freeze_trunk: bool,
    device:       &B::Device,
) -> (NetBurn<B>, f32) {
    let b = feats.len();
    assert!(b > 0);

    let feat_flat: Vec<f32> = feats.iter().flat_map(|f| f.iter().copied()).collect();
    // freeze_trunk: inner-backend body so candidate training only updates the
    // candidate head, preserving the policy-optimized trunk during pretraining
    // (see frozen_or_live_body for why this is not a detach()).
    let body = frozen_or_live_body(&net, feat_flat, b, freeze_trunk, device);

    let mut cand_data  = vec![0.0f32; b * MAX_CANDS * CAND_DIM];
    let mut valid_mask = vec![-1e9f32; b * MAX_CANDS];
    for (i, cf) in cands_flat.iter().enumerate() {
        let nc = (cf.len() / CAND_DIM).min(MAX_CANDS);
        for j in 0..nc {
            let src = j * CAND_DIM;
            let dst = (i * MAX_CANDS + j) * CAND_DIM;
            cand_data[dst..dst + CAND_DIM].copy_from_slice(&cf[src..src + CAND_DIM]);
            valid_mask[i * MAX_CANDS + j] = 0.0;
        }
    }

    let body_3d: Tensor<B, 3>  = body.reshape([b, 1, H]);
    let body_rep: Tensor<B, 3> = body_3d.repeat_dim(1, MAX_CANDS);
    let body_flat: Tensor<B, 2> = body_rep.reshape([b * MAX_CANDS, H]);

    let cand_t = Tensor::<B, 2>::from_data(
        TensorData::new(cand_data, [b * MAX_CANDS, CAND_DIM]), device,
    );
    let inp = Tensor::cat(vec![body_flat, cand_t], 1);

    let scores: Tensor<B, 2> = net.lc2
        .forward(activation::relu(net.lc1.forward(inp)))
        .reshape([b, MAX_CANDS]);

    let mask_t  = Tensor::<B, 2>::from_data(TensorData::new(valid_mask, [b, MAX_CANDS]), device);
    let log_p   = activation::log_softmax(scores + mask_t, 1);
    let chosen_t = Tensor::<B, 2, Int>::from_data(
        TensorData::new(chosen.iter().map(|&c| c as i32).collect::<Vec<_>>(), [b, 1]), device,
    );
    let picked = log_p.gather(1, chosen_t);   // [b, 1]
    // advantages=None → plain CE toward `chosen` (BC imitation of AdvCPU).
    // advantages=Some → REINFORCE: -A·log p(chosen). During RL `chosen` is the
    // head's own pick, so unweighted CE is pure self-imitation with zero
    // learning signal; advantage weighting reinforces picks that won and
    // pushes away from picks that lost.
    let loss = match advantages {
        Some(a) => {
            let w = Tensor::<B, 2>::from_data(TensorData::new(a.to_vec(), [b, 1]), device);
            -(picked * w).mean()
        }
        None => -picked.mean(),
    };
    let lv = loss.clone().into_data().to_vec::<f32>().unwrap_or_default()[0];
    let grads = GradientsParams::from_grads(loss.backward(), &net);
    (optim.step(lr, net, grads), lv)
}
