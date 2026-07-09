// ASTRA-7 entity transformer (ASTRA7.md §3.2).
//
// Architecture: 6-layer pre-norm transformer encoder, d_model 256, 8 heads,
// FFN 1024 (~6-8M params) over the fixed-layout token sequence from
// tokens.rs. Two heads read out of the shared trunk:
//
//   - Policy (pointer): a small 2-layer MLP scores each encoder output in
//     the action-token block directly (Linear(d,d)+relu -> Linear(d,1)).
//     This IS a pointer-attention readout: the final Linear's weight row is
//     a single learned query vector dotted against the (relu-transformed)
//     per-token key — the same computation a 1-head attention-as-scoring
//     mechanism performs, just without a separate softmax-weighted value
//     aggregation step, because here we want raw per-token logits (the
//     softmax happens once, in the training loss / inference sampler, over
//     however many legal actions exist for this decision — not a pooled
//     vector). Structural masking: positions beyond the true legal-action
//     count are Pad tokens (see tokens.rs), so the mask is derived, not
//     stored.
//   - Value: mean-pool the state-token block (excluding Pad), MLP(256->256->1),
//     sigmoid.
//
// Auxiliary opponent-hand-prediction head (ASTRA7.md §3.2, "recommended"):
// DEFERRED for this BC-pipeline pass — see the TODO below. The BC gate
// (§4.1) only requires policy + value; time was spent validating the
// pointer-policy plumbing end-to-end instead. Flagged, not forgotten.
//
// Weight persistence: unlike ml/src/net_burn.rs (which hand-rolls a flat
// Vec<f32> <-> Param round trip via Param::map, because v6 keeps a SEPARATE
// non-burn ndarray inference net that the flat format has to interoperate
// with), v7 has no such second net — this Net<B> IS the only representation.
// So it uses burn's own Module::save_file/load_file (BinFileRecorder), which
// goes through Module::into_record/load_record and never touches
// Tensor::from_data on a live Param — the §8.2 require_grad footgun doesn't
// apply because that code path is never exercised.

use burn::{
    grad_clipping::GradientClippingConfig,
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{activation, backend::AutodiffBackend, backend::Backend, Bool, Int, Tensor, TensorData},
};

use crate::tokens::{
    TokenSeq, TokenType, ACTION_MAX, FIELD_VOCAB, HISTORY_MAX, NUM_ACTION_KINDS, NUM_FIELDS, SEQ_LEN, STATE_MAX,
    TYPE_VOCAB,
};

pub const D_MODEL: usize = 256;
pub const N_HEADS: usize = 8;
pub const N_LAYERS: usize = 6;
pub const D_FF: usize = 1024;
pub const VALUE_HIDDEN: usize = 256;
pub const POINTER_HIDDEN: usize = 256;

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    type_embed: Embedding<B>,
    field_embed: Vec<Embedding<B>>,
    encoder: TransformerEncoder<B>,
    pointer_hidden: Linear<B>,
    pointer_out: Linear<B>,
    value_hidden: Linear<B>,
    value_out: Linear<B>,
}

impl<B: Backend> Net<B> {
    pub fn fresh(device: &B::Device) -> Self {
        let type_embed = EmbeddingConfig::new(TYPE_VOCAB, D_MODEL).init(device);
        let field_embed = (0..NUM_FIELDS)
            .map(|_| EmbeddingConfig::new(FIELD_VOCAB, D_MODEL).init(device))
            .collect();
        let encoder = TransformerEncoderConfig::new(D_MODEL, D_FF, N_HEADS, N_LAYERS)
            .with_norm_first(true) // ASTRA7.md §3.2: pre-norm
            .with_dropout(0.1)
            .init(device);
        let pointer_hidden = LinearConfig::new(D_MODEL, POINTER_HIDDEN).init(device);
        let pointer_out = LinearConfig::new(POINTER_HIDDEN, 1).init(device);
        let value_hidden = LinearConfig::new(D_MODEL, VALUE_HIDDEN).init(device);
        let value_out = LinearConfig::new(VALUE_HIDDEN, 1).init(device);
        Self { type_embed, field_embed, encoder, pointer_hidden, pointer_out, value_hidden, value_out }
    }
}

// ── Batch tensor construction ──────────────────────────────────────────────

struct BatchTensors<B: Backend> {
    type_ids: Tensor<B, 2, Int>,
    fields: Vec<Tensor<B, 2, Int>>,
    pad_mask: Tensor<B, 2, Bool>, // true = pad
}

fn build_batch<B: Backend>(seqs: &[TokenSeq], device: &B::Device) -> BatchTensors<B> {
    let b = seqs.len();
    let mut type_flat = vec![0i32; b * SEQ_LEN];
    let mut field_flat = vec![vec![0i32; b * SEQ_LEN]; NUM_FIELDS];
    let mut pad_flat = vec![0i32; b * SEQ_LEN];

    for (bi, seq) in seqs.iter().enumerate() {
        for (si, tok) in seq.iter().enumerate() {
            let idx = bi * SEQ_LEN + si;
            type_flat[idx] = tok.ttype as i32;
            for f in 0..NUM_FIELDS {
                field_flat[f][idx] = tok.fields[f] as i32;
            }
            pad_flat[idx] = (tok.ttype == TokenType::Pad as u8) as i32;
        }
    }

    let type_ids = Tensor::<B, 2, Int>::from_data(TensorData::new(type_flat, [b, SEQ_LEN]), device);
    let fields = field_flat
        .into_iter()
        .map(|f| Tensor::<B, 2, Int>::from_data(TensorData::new(f, [b, SEQ_LEN]), device))
        .collect();
    let pad_mask = Tensor::<B, 2, Int>::from_data(TensorData::new(pad_flat, [b, SEQ_LEN]), device).equal_elem(1);

    BatchTensors { type_ids, fields, pad_mask }
}

// ── Forward pass ────────────────────────────────────────────────────────

/// Runs the shared trunk + both heads.
/// Returns (`policy_logits` [batch, ACTION_MAX], `value` [batch, 1]).
/// Illegal action-token slots (Pad, beyond a decision's true legal-action
/// count) are NOT masked here — callers must mask before softmax/loss using
/// the same derivation net.rs uses internally (type_id == Pad in the action
/// block), exactly like v6's `masks: &[u64]` convention in net_burn.rs.
pub fn forward<B: Backend>(net: &Net<B>, seqs: &[TokenSeq], device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let b = seqs.len();
    let batch = build_batch::<B>(seqs, device);

    let mut emb = net.type_embed.forward(batch.type_ids);
    for (f, embed) in net.field_embed.iter().enumerate() {
        emb = emb + embed.forward(batch.fields[f].clone());
    }

    let input = TransformerEncoderInput::new(emb).mask_pad(batch.pad_mask.clone());
    let enc = net.encoder.forward(input); // [b, SEQ_LEN, D_MODEL]

    // ── Value: mean-pool state block, excluding Pad ──────────────────────
    let state_block = enc.clone().narrow(1, 0, STATE_MAX); // [b, STATE_MAX, D]
    let state_pad = batch.pad_mask.clone().narrow(1, 0, STATE_MAX); // [b, STATE_MAX]
    let keep = state_pad.bool_not().float().reshape([b, STATE_MAX, 1]); // 1.0 = real token
    let summed = (state_block * keep.clone()).sum_dim(1); // [b, 1, D]
    let counts = keep.sum_dim(1).clamp_min(1.0f32); // [b, 1, 1]
    let pooled = (summed / counts).reshape([b, D_MODEL]);

    let vh = activation::relu(net.value_hidden.forward(pooled));
    let value = activation::sigmoid(net.value_out.forward(vh)); // [b, 1]

    // ── Policy: pointer scores over the action-token block ───────────────
    let action_block = enc.narrow(1, STATE_MAX + HISTORY_MAX, ACTION_MAX); // [b, ACTION_MAX, D]
    let ph = activation::relu(net.pointer_hidden.forward(action_block));
    let logits = net.pointer_out.forward(ph).reshape([b, ACTION_MAX]); // [b, ACTION_MAX]

    (logits, value)
}

/// Boolean action-validity mask derived from the same Pad convention used in
/// tokens.rs — `true` = legal (real) action slot. Shape [batch, ACTION_MAX].
pub fn action_valid_mask<B: Backend>(seqs: &[TokenSeq], device: &B::Device) -> Tensor<B, 2, Bool> {
    let b = seqs.len();
    let base = STATE_MAX + HISTORY_MAX;
    let mut flat = vec![0i32; b * ACTION_MAX];
    for (bi, seq) in seqs.iter().enumerate() {
        for j in 0..ACTION_MAX {
            let is_pad = seq[base + j].ttype == TokenType::Pad as u8;
            flat[bi * ACTION_MAX + j] = (!is_pad) as i32;
        }
    }
    Tensor::<B, 2, Int>::from_data(TensorData::new(flat, [b, ACTION_MAX]), device).equal_elem(1)
}

// ── Optimizer ──────────────────────────────────────────────────────────────

pub fn make_optimizer<B: AutodiffBackend>() -> impl Optimizer<Net<B>, B> {
    AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init()
}

// ── BC training step ────────────────────────────────────────────────────
//
// Pointer policy CE (masked label smoothing over the legal set, ported from
// ml/src/net_burn.rs's pretrain_joint_step_burn, generalized from a fixed
// NUM_ACTIONS=31 softmax to the variable-length ACTION_MAX pointer logits)
// + value MSE against the terminal game outcome in one backward pass. This
// value target is a BC-stage warm start, not the TD(λ) target ASTRA7.md
// §4.2 specifies for the dedicated value stage (out of scope here).

#[allow(clippy::too_many_arguments)]
pub fn bc_train_step<B: AutodiffBackend, O: Optimizer<Net<B>, B>>(
    net: Net<B>,
    optim: &mut O,
    lr: f64,
    seqs: &[TokenSeq],
    targets: &[u16],
    outcomes: &[f32],
    label_smoothing: f32,
    device: &B::Device,
) -> (Net<B>, f32, f32) {
    let b = seqs.len();
    assert!(b > 0);

    let (logits, values) = forward(&net, seqs, device);
    let valid = action_valid_mask::<B>(seqs, device); // true = legal
    let neg_inf = valid.clone().bool_not().float() * (-1e9f32);
    let masked_logits = logits + neg_inf;
    let log_p = activation::log_softmax(masked_logits, 1);

    let act_t = Tensor::<B, 2, Int>::from_data(
        TensorData::new(targets.iter().map(|&a| a as i32).collect::<Vec<_>>(), [b, 1]),
        device,
    );
    let hard = -log_p.clone().gather(1, act_t); // [b, 1]

    let policy_loss = if label_smoothing > 0.0 {
        let legal_f = valid.float(); // [b, ACTION_MAX]
        let num_legal = legal_f.clone().sum_dim(1); // [b, 1]
        let uniform = -(log_p * legal_f).sum_dim(1) / num_legal; // [b, 1]
        let eps = label_smoothing;
        (hard.clone() * (1.0 - eps) + uniform * eps).mean()
    } else {
        hard.clone().mean()
    };

    let tgt = Tensor::<B, 2>::from_data(TensorData::new(outcomes.to_vec(), [b, 1]), device);
    let diff = values - tgt;
    let value_loss = (diff.clone() * diff).mean();

    let pl = hard.mean().into_data().to_vec::<f32>().unwrap_or_default()[0];
    let vl = value_loss.clone().into_data().to_vec::<f32>().unwrap_or_default()[0];

    // ASTRA7.md §10: value target is only a terminal-outcome warm start (the
    // real value signal comes from TD-value/AZ later) — down-weight so it
    // doesn't distract the policy gradient, which is what BC is actually for.
    const VALUE_LOSS_WEIGHT: f32 = 0.3;
    let total = policy_loss + value_loss * VALUE_LOSS_WEIGHT;
    let grads = GradientsParams::from_grads(total.backward(), &net);
    (optim.step(lr, net, grads), pl, vl)
}

/// Forward-only policy evaluation (argmax-correct count + summed hard CE).
/// Callers must pass `net.valid()` — forward passes on the autodiff backend
/// accumulate graph state that only `backward()` releases (ASTRA7.md §8.2).
pub fn policy_eval<B: Backend>(net: &Net<B>, seqs: &[TokenSeq], targets: &[u16], device: &B::Device) -> (usize, f32) {
    let b = seqs.len();
    if b == 0 {
        return (0, 0.0);
    }
    let (logits, _values) = forward(net, seqs, device);
    let valid = action_valid_mask::<B>(seqs, device);
    let neg_inf = valid.bool_not().float() * (-1e9f32);
    let masked = (logits + neg_inf).into_data().to_vec::<f32>().unwrap_or_default();

    let mut correct = 0usize;
    let mut ce_sum = 0.0f32;
    for i in 0..b {
        let row = &masked[i * ACTION_MAX..(i + 1) * ACTION_MAX];
        let argmax = row
            .iter()
            .enumerate()
            .max_by(|a, c| a.1.partial_cmp(c.1).unwrap())
            .map(|(j, _)| j)
            .unwrap_or(0);
        if argmax == targets[i] as usize {
            correct += 1;
        }
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse = m + row.iter().map(|v| (v - m).exp()).sum::<f32>().ln();
        ce_sum += lse - row[targets[i] as usize];
    }
    (correct, ce_sum)
}

/// Same forward-only evaluation as `policy_eval`, bucketed by the TARGET
/// action's `ActionKind` (tokens.rs field[0] of the token at the correct
/// pointer index) instead of one aggregate number — answers "is any
/// decision kind (esp. chain responses, this project's single largest
/// measured lever — ASTRA7.md §1.1) lagging the others" directly, rather
/// than guessing from an average. Returns `[(correct, total, ce_sum); 16]`
/// indexed by `ActionKind as usize`. Same `net.valid()` calling convention
/// as `policy_eval` (§8.2) — this is a forward-only pass.
pub fn policy_eval_by_kind<B: Backend>(
    net: &Net<B>,
    seqs: &[TokenSeq],
    targets: &[u16],
    device: &B::Device,
) -> [(usize, usize, f32); NUM_ACTION_KINDS] {
    let mut buckets = [(0usize, 0usize, 0.0f32); NUM_ACTION_KINDS];
    let b = seqs.len();
    if b == 0 {
        return buckets;
    }
    let (logits, _values) = forward(net, seqs, device);
    let valid = action_valid_mask::<B>(seqs, device);
    let neg_inf = valid.bool_not().float() * (-1e9f32);
    let masked = (logits + neg_inf).into_data().to_vec::<f32>().unwrap_or_default();

    let action_base = STATE_MAX + HISTORY_MAX;
    for i in 0..b {
        let row = &masked[i * ACTION_MAX..(i + 1) * ACTION_MAX];
        let target_idx = targets[i] as usize;
        let kind = seqs[i][action_base + target_idx].fields[0] as usize;
        if kind >= NUM_ACTION_KINDS {
            continue; // defensive — should never happen given the encoders
        }
        let argmax = row
            .iter()
            .enumerate()
            .max_by(|a, c| a.1.partial_cmp(c.1).unwrap())
            .map(|(j, _)| j)
            .unwrap_or(0);
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse = m + row.iter().map(|v| (v - m).exp()).sum::<f32>().ln();
        buckets[kind].1 += 1;
        buckets[kind].2 += lse - row[target_idx];
        if argmax == target_idx {
            buckets[kind].0 += 1;
        }
    }
    buckets
}
