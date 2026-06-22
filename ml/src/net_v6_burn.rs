// Burn-based training backend for Astra-6.
// Game generation and inference remain in net_v6.rs / self_play_v6.rs (ndarray, CPU).
// Only the RL training step runs on burn (GPU via WGPU, or NdArray fallback).

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    optim::{Adam, AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::{activation, backend::AutodiffBackend, Int, TensorData},
};

use crate::action::NUM_ACTIONS;
use crate::candidate::CAND_DIM;
use crate::feature_v6::FEATURE_DIM_V6;
use crate::net_v6::{WeightsV6, INPUT_DIM};
use crate::self_play_v6::RlSampleV6;

const H:  usize = 512;
const HC: usize = 32;

// ── Network definition ────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct NetV6Burn<B: Backend> {
    l1:  Linear<B>,
    l2:  Linear<B>,
    l3:  Linear<B>,
    l4:  Linear<B>,
    lp:  Linear<B>,   // policy:    H  → NUM_ACTIONS
    lv:  Linear<B>,   // value:     H  → 1
    lc1: Linear<B>,   // cand h:    H+CAND_DIM → HC
    lc2: Linear<B>,   // cand out:  HC → 1
}

// Burn's Linear stores weight as [d_input, d_output] and computes y = x @ W (no transpose).
// WeightsV6 (ndarray convention) stores weights as [d_output, d_input].
// These helpers convert between the two layouts.
fn t_flat(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    // [rows, cols] row-major → [cols, rows] row-major
    let mut out = vec![0.0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

impl<B: Backend> NetV6Burn<B> {
    /// Load from serialisable WeightsV6 (ndarray inference format).
    pub fn from_weights(w: &WeightsV6, device: &B::Device) -> Self {
        use burn::module::{Param, ParamId};

        // mk2_t: ndarray weight [out, in] → burn weight [in, out]
        let mk2_t = |data: &[f32], out_dim: usize, in_dim: usize| -> Tensor<B, 2> {
            Tensor::from_data(TensorData::new(t_flat(data, out_dim, in_dim), [in_dim, out_dim]), device)
        };
        let mk1 = |data: &[f32], len: usize| -> Tensor<B, 1> {
            Tensor::from_data(TensorData::new(data.to_vec(), [len]), device)
        };
        let lin = |wt: Tensor<B, 2>, bt: Tensor<B, 1>| Linear {
            weight: Param::initialized(ParamId::new(), wt),
            bias:   Some(Param::initialized(ParamId::new(), bt)),
        };

        Self {
            l1:  lin(mk2_t(&w.w1,  H,           INPUT_DIM),    mk1(&w.b1,  H)),
            l2:  lin(mk2_t(&w.w2,  H,           H),            mk1(&w.b2,  H)),
            l3:  lin(mk2_t(&w.w3,  H,           H),            mk1(&w.b3,  H)),
            l4:  lin(mk2_t(&w.w4,  H,           H),            mk1(&w.b4,  H)),
            lp:  lin(mk2_t(&w.wp,  NUM_ACTIONS, H),            mk1(&w.bp,  NUM_ACTIONS)),
            lv:  lin(mk2_t(&w.wv,  1,           H),            mk1(&w.bv,  1)),
            lc1: lin(mk2_t(&w.wc1, HC,          H + CAND_DIM), mk1(&w.bc1, HC)),
            lc2: lin(mk2_t(&w.wc2, 1,           HC),           mk1(&w.bc2, 1)),
        }
    }

    /// Extract back to WeightsV6 for ndarray inference / checkpointing.
    pub fn to_weights(&self) -> WeightsV6 {
        // get2_t: burn weight [in, out] → ndarray weight [out, in]
        let get2_t = |l: &Linear<B>, in_dim: usize, out_dim: usize| -> Vec<f32> {
            let data = l.weight.val().into_data().to_vec::<f32>().expect("weight f32");
            t_flat(&data, in_dim, out_dim)
        };
        let get1 = |l: &Linear<B>| -> Vec<f32> {
            l.bias.as_ref().unwrap().val().into_data().to_vec::<f32>().expect("bias f32")
        };
        WeightsV6 {
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

// ── Forward (body only; autodiff-compatible) ──────────────────────────────────

fn body_forward<B: Backend>(net: &NetV6Burn<B>, x: Tensor<B, 2>) -> Tensor<B, 2> {
    let a1 = activation::relu(net.l1.forward(x));
    let a2 = activation::relu(net.l2.forward(a1.clone()) + a1);
    let a3 = activation::relu(net.l3.forward(a2.clone()) + a2);
    activation::relu(net.l4.forward(a3))
}

// ── Optimizer factory ─────────────────────────────────────────────────────────

/// Create a fresh Adam optimiser for NetV6Burn.
pub fn make_optimizer<B: AutodiffBackend>() -> impl Optimizer<NetV6Burn<B>, B> {
    AdamConfig::new().init()
}

// ── PPO training step ─────────────────────────────────────────────────────────

/// One PPO gradient step on the burn backend.
/// Consumes `net` and returns updated `net` (burn functional API).
pub fn rl_train_step_burn<B: AutodiffBackend, O: Optimizer<NetV6Burn<B>, B>>(
    net:           NetV6Burn<B>,
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
) -> (NetV6Burn<B>, f32, f32) {
    let b = feats.len();
    assert!(b > 0);

    // ── Input tensor [B, INPUT_DIM] ──────────────────────────────────────────
    let feat_flat: Vec<f32> = feats.iter().flat_map(|f| f.iter().copied()).collect();
    let x = Tensor::<B, 2>::from_data(TensorData::new(feat_flat, [b, INPUT_DIM]), device);

    // ── Forward ──────────────────────────────────────────────────────────────
    let body    = body_forward(&net, x);
    let logits  = net.lp.forward(body.clone());                     // [B, NUM_ACTIONS]
    let values  = activation::sigmoid(net.lv.forward(body));        // [B, 1]

    // ── Masked log-softmax ───────────────────────────────────────────────────
    let mask_flat: Vec<f32> = masks.iter().flat_map(|&m| {
        (0..NUM_ACTIONS).map(move |i| if m & (1u64 << i) != 0 { 0.0_f32 } else { -1e9_f32 })
    }).collect();
    let mask_t = Tensor::<B, 2>::from_data(TensorData::new(mask_flat, [b, NUM_ACTIONS]), device);
    let masked  = logits + mask_t;
    let log_p   = activation::log_softmax(masked.clone(), 1);       // [B, NUM_ACTIONS]

    // ── Gather new log-probs at taken actions [B, 1] ─────────────────────────
    let act_data: Vec<i32> = actions.iter().map(|&a| a as i32).collect();
    let act_t = Tensor::<B, 2, Int>::from_data(TensorData::new(act_data, [b, 1]), device);
    let new_lp = log_p.clone().gather(1, act_t);                    // [B, 1]

    // ── PPO ratio & clipped objective ────────────────────────────────────────
    let old_lp_t  = Tensor::<B, 2>::from_data(TensorData::new(old_lps.to_vec(), [b, 1]), device);
    let ratio     = (new_lp - old_lp_t).exp();                      // [B, 1]
    let outcomes_t = Tensor::<B, 2>::from_data(TensorData::new(outcomes.to_vec(), [b, 1]), device);
    let advantage = outcomes_t.clone() - values.clone();             // [B, 1]
    let surr1 = ratio.clone() * advantage.clone();
    let surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantage;
    // element-wise min: (a+b-|a-b|)/2
    let ppo_obj = (surr1.clone() + surr2.clone() - (surr1 - surr2).abs()) * 0.5;

    // ── Entropy ──────────────────────────────────────────────────────────────
    let probs   = activation::softmax(masked, 1);                    // [B, NUM_ACTIONS]
    let entropy = -(probs * log_p).sum_dim(1);                       // [B, 1]

    // ── IS weights [B, 1] ────────────────────────────────────────────────────
    let is_t = Tensor::<B, 2>::from_data(TensorData::new(is_weights_in.to_vec(), [b, 1]), device);

    // ── Losses ───────────────────────────────────────────────────────────────
    let policy_loss = -(is_t.clone() * (ppo_obj + entropy * entropy_coeff)).mean();
    let diff        = values - outcomes_t;
    let value_loss  = (is_t * diff.clone() * diff).mean();

    // Extract scalars for logging (clone before backward consumes tensors)
    let pl_val: f32 = policy_loss.clone().into_data().to_vec::<f32>().unwrap_or_default()[0];
    let vl_val: f32 = value_loss.clone().into_data().to_vec::<f32>().unwrap_or_default()[0];

    // ── Backward + Adam step ─────────────────────────────────────────────────
    let total_loss = policy_loss + value_loss;
    let grads = GradientsParams::from_grads(total_loss.backward(), &net);
    let net = optim.step(lr, net, grads);

    (net, pl_val, vl_val)
}
