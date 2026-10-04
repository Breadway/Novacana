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
//     per-token key - the same computation a 1-head attention-as-scoring
//     mechanism performs, just without a separate softmax-weighted value
//     aggregation step, because here we want raw per-token logits (the
//     softmax happens once, in the training loss / inference sampler, over
//     however many legal actions exist for this decision - not a pooled
//     vector). Structural masking: positions beyond the true legal-action
//     count are Pad tokens (see tokens.rs), so the mask is derived, not
//     stored.
//   - Value: mean-pool the state-token block (excluding Pad), MLP(256->256->1),
//     sigmoid.
//
// Auxiliary opponent-hand-prediction head (ASTRA7.md §3.2, "recommended"):
// DEFERRED for this BC-pipeline pass - see the TODO below. The BC gate
// (§4.1) only requires policy + value; time was spent validating the
// pointer-policy plumbing end-to-end instead. Flagged, not forgotten.
//
// Weight persistence: unlike ml/src/net_burn.rs (which hand-rolls a flat
// Vec<f32> <-> Param round trip via Param::map, because v6 keeps a SEPARATE
// non-burn ndarray inference net that the flat format has to interoperate
// with), v7 has no such second net - this Net<B> IS the only representation.
// So it uses burn's own Module::save_file/load_file (BinFileRecorder), which
// goes through Module::into_record/load_record and never touches
// Tensor::from_data on a live Param - the §8.2 require_grad footgun doesn't
// apply because that code path is never exercised.

use burn::{
    grad_clipping::GradientClippingConfig,
    module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param, ParamId},
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{
        activation, backend::AutodiffBackend, backend::Backend, Bool, FloatDType, Int, Tensor,
        TensorData,
    },
};

/// Read floats back as f32. Callers that score or log a scalar want host f32.
/// Illegal-logit fill. A fill of -1e9 overflows fp16 to infinity, and
/// `0 * inf` then turns every legal logit into NaN, so the mask stays -1e4.
pub(crate) const LOGIT_MASK: f32 = -1.0e4;

pub(crate) fn download_f32<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor.cast(FloatDType::F32).into_data().to_vec::<f32>().unwrap_or_default()
}

use crate::tokens::{
    TokenSeq, TokenType, ACTION_MAX, FIELD_VOCAB, HISTORY_MAX, NUM_ACTION_KINDS, NUM_FIELDS, STATE_MAX,
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
        Self::fresh_with_dropout(device, 0.1)
    }

    pub fn fresh_with_dropout(device: &B::Device, dropout: f64) -> Self {
        let type_embed = EmbeddingConfig::new(TYPE_VOCAB, D_MODEL).init(device);
        let field_embed = (0..NUM_FIELDS)
            .map(|_| EmbeddingConfig::new(FIELD_VOCAB, D_MODEL).init(device))
            .collect();
        let encoder_cfg = TransformerEncoderConfig::new(D_MODEL, D_FF, N_HEADS, N_LAYERS)
            .with_norm_first(true) // ASTRA7.md §3.2: pre-norm
            .with_dropout(dropout);
        let encoder = encoder_cfg.init(device);
        let pointer_hidden = LinearConfig::new(D_MODEL, POINTER_HIDDEN).init(device);
        let pointer_out = LinearConfig::new(POINTER_HIDDEN, 1).init(device);
        let value_hidden = LinearConfig::new(D_MODEL, VALUE_HIDDEN).init(device);
        let value_out = LinearConfig::new(VALUE_HIDDEN, 1).init(device);
        Self { type_embed, field_embed, encoder, pointer_hidden, pointer_out, value_hidden, value_out }
    }
}

// ── Batch tensor construction ──────────────────────────────────────────────

/// Padding is trimmed independently from each block, rounded to eight slots
/// to limit GPU kernel shapes. The transformer has no positional embeddings;
/// positions with real tokens and their fields retain exactly the same meaning.
#[derive(Clone, Copy, Debug)]
struct BatchLayout {
    state: usize,
    history: usize,
    actions: usize,
}

impl BatchLayout {
    fn dense() -> Self { Self { state: STATE_MAX, history: HISTORY_MAX, actions: ACTION_MAX } }
    fn for_batch(seqs: &[TokenSeq]) -> Self {
        assert!(!seqs.is_empty(), "a batch must contain at least one sequence");
        let extent = |start: usize, cap: usize| {
            let used = seqs.iter().map(|s| {
                s[start..start + cap].iter().rposition(|t| t.ttype != TokenType::Pad as u8)
                    .map_or(0, |i| i + 1)
            }).max().unwrap_or(0);
            // Keep a nonempty dimension for each block, including empty history.
            used.max(1).div_ceil(8).saturating_mul(8).min(cap)
        };
        Self {
            state: extent(0, STATE_MAX),
            history: extent(STATE_MAX, HISTORY_MAX),
            actions: extent(STATE_MAX + HISTORY_MAX, ACTION_MAX),
        }
    }

    fn len(self) -> usize { self.state + self.history + self.actions }
}

/// Unpadded token count for one sequence. Training sorts on this so a long
/// decision does not stretch every row in its batch out to the same length.
pub(crate) fn sequence_span(seq: &TokenSeq) -> usize {
    let used = |start: usize, cap: usize| {
        seq[start..start + cap]
            .iter()
            .rposition(|t| t.ttype != TokenType::Pad as u8)
            .map_or(0, |i| i + 1)
    };
    used(0, STATE_MAX) + used(STATE_MAX, HISTORY_MAX) + used(STATE_MAX + HISTORY_MAX, ACTION_MAX)
}

struct BatchTensors<B: Backend> {
    type_ids: Tensor<B, 2, Int>,
    fields: Vec<Tensor<B, 2, Int>>,
    pad_mask: Tensor<B, 2, Bool>, // true = pad
}

fn build_batch<B: Backend>(seqs: &[TokenSeq], layout: BatchLayout, device: &B::Device) -> BatchTensors<B> {
    let b = seqs.len();
    let len = layout.len();
    let mut type_flat = Vec::with_capacity(b * len);
    let mut field_flat = vec![Vec::with_capacity(b * len); NUM_FIELDS];
    for seq in seqs {
        for tok in seq[..layout.state].iter()
            .chain(&seq[STATE_MAX..STATE_MAX + layout.history])
            .chain(&seq[STATE_MAX + HISTORY_MAX..STATE_MAX + HISTORY_MAX + layout.actions])
        {
            type_flat.push(tok.ttype as i32);
            for (values, field) in field_flat.iter_mut().zip(tok.fields) {
                values.push(field as i32);
            }
        }
    }
    let type_ids = Tensor::<B, 2, Int>::from_data(TensorData::new(type_flat, [b, len]), device);
    // Derive padding on device instead of uploading a ninth integer buffer.
    let pad_mask = type_ids.clone().equal_elem(TokenType::Pad as i32);
    let fields = field_flat.into_iter()
        .map(|f| Tensor::<B, 2, Int>::from_data(TensorData::new(f, [b, len]), device))
        .collect();
    BatchTensors { type_ids, fields, pad_mask }
}

// ── Forward pass ────────────────────────────────────────────────────────

/// Trunk-only forward pass: embeddings + transformer encoder. Factored out of
/// `forward()` so TD-value training (net.rs below, ASTRA7.md §4.2/§8.2) can
/// run the trunk on the INNER backend (via `net.valid()`) while the value
/// head stays on the live autodiff backend - the "freeze trunk" mechanism.
/// Returns the compact encoder output, padding mask, and block layout.
fn encode_trunk<B: Backend>(
    net: &Net<B>,
    seqs: &[TokenSeq],
    device: &B::Device,
) -> (Tensor<B, 3>, Tensor<B, 2, Bool>, BatchLayout) {
    encode_trunk_layout(net, seqs, BatchLayout::for_batch(seqs), device)
}

fn encode_trunk_layout<B: Backend>(
    net: &Net<B>, seqs: &[TokenSeq], layout: BatchLayout, device: &B::Device,
) -> (Tensor<B, 3>, Tensor<B, 2, Bool>, BatchLayout) {
    let batch = build_batch::<B>(seqs, layout, device);

    let mut emb = net.type_embed.forward(batch.type_ids);
    for (f, embed) in net.field_embed.iter().enumerate() {
        emb = emb + embed.forward(batch.fields[f].clone());
    }

    let input = TransformerEncoderInput::new(emb).mask_pad(batch.pad_mask.clone());
    let enc = net.encoder.forward(input);
    (enc, batch.pad_mask, layout)
}

/// Mean-pool the state-token block of an encoder output, excluding Pad.
/// Shared by `forward()`'s value head and `td_train_step`'s frozen-trunk
/// value readout (both need identically-computed pooled features).
fn pool_state<B: Backend>(enc: Tensor<B, 3>, pad_mask: Tensor<B, 2, Bool>, state_len: usize) -> Tensor<B, 2> {
    let b = enc.dims()[0];
    let state_block = enc.narrow(1, 0, state_len); // [b, STATE_MAX, D]
    let state_pad = pad_mask.narrow(1, 0, state_len); // [b, STATE_MAX]
    let keep = state_pad.bool_not().float().reshape([b, state_len, 1]); // 1.0 = real token
    let summed = (state_block * keep.clone()).sum_dim(1); // [b, 1, D]
    let counts = keep.sum_dim(1).clamp_min(1.0f32); // [b, 1, 1]
    (summed / counts).reshape([b, D_MODEL])
}

/// Value-head readout from already-pooled state features. Split out of
/// `forward()` so `td_train_step` can feed it a trunk output computed on a
/// different backend (see `pool_state`/`encode_trunk` doc).
fn value_from_pooled<B: Backend>(net: &Net<B>, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
    let vh = activation::relu(net.value_hidden.forward(pooled));
    activation::sigmoid(net.value_out.forward(vh))
}

/// Runs the shared trunk + both heads.
/// Returns (`policy_logits` [batch, ACTION_MAX], `value` [batch, 1]).
/// Illegal action-token slots (Pad, beyond a decision's true legal-action
/// count) are NOT masked here - callers must mask before softmax/loss using
/// the same derivation net.rs uses internally (type_id == Pad in the action
/// block), exactly like v6's `masks: &[u64]` convention in net_burn.rs.
pub fn forward<B: Backend>(net: &Net<B>, seqs: &[TokenSeq], device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 2>) {
    forward_layout(net, seqs, BatchLayout::for_batch(seqs), device)
}

fn forward_layout<B: Backend>(net: &Net<B>, seqs: &[TokenSeq], layout: BatchLayout, device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let b = seqs.len();
    let (enc, pad_mask, _) = encode_trunk_layout(net, seqs, layout, device);

    // ── Value: mean-pool state block, excluding Pad ──────────────────────
    let pooled = pool_state(enc.clone(), pad_mask, layout.state);
    let value = value_from_pooled(net, pooled); // [b, 1]

    // ── Policy: pointer scores over the action-token block ───────────────
    let action_block = enc.narrow(1, layout.state + layout.history, layout.actions); // [b, ACTION_MAX, D]
    let ph = activation::relu(net.pointer_hidden.forward(action_block));
    let logits = net.pointer_out.forward(ph).reshape([b, layout.actions]);
    // Restore the public fixed-width action interface. These slots are always
    // masked by every caller, so computing a pointer head for them is wasted.
    let logits = if layout.actions < ACTION_MAX {
        Tensor::cat(vec![logits, Tensor::zeros([b, ACTION_MAX - layout.actions], device)], 1)
    } else { logits }; // [b, ACTION_MAX]

    (logits, value)
}

/// Boolean action-validity mask derived from the same Pad convention used in
/// tokens.rs - `true` = legal (real) action slot. Shape [batch, ACTION_MAX].
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
    // Default epsilon is 1e-5. The fp16 Vulkan run raised it to 1e-2 because
    // 1e-5 flushed to zero and the first step produced NaNs. Steady-state
    // Adam then added that 1e-2 on every update, which stalled the policy
    // once per-weight gradients fell below it. Vulkan training is fp32.
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
    report_metrics: bool,
    device: &B::Device,
) -> (Net<B>, Option<f32>, Option<f32>) {
    bc_train_step_layout(net, optim, lr, seqs, targets, outcomes, label_smoothing,
        report_metrics, BatchLayout::for_batch(seqs), device)
}

#[allow(clippy::too_many_arguments)]
fn bc_train_step_layout<B: AutodiffBackend, O: Optimizer<Net<B>, B>>(
    net: Net<B>, optim: &mut O, lr: f64, seqs: &[TokenSeq], targets: &[u16],
    outcomes: &[f32], label_smoothing: f32, report_metrics: bool,
    layout: BatchLayout, device: &B::Device,
) -> (Net<B>, Option<f32>, Option<f32>) {
    let fwd = bc_forward_layout(&net, seqs, targets, outcomes, label_smoothing,
        report_metrics, layout, device);
    // Keep the Fusion queue drain before backward on the one-device path.
    let pl = fwd.policy_ce.map(|t| download_f32(t).first().copied().unwrap_or(f32::NAN));
    let vl = fwd.value_mse.map(|t| download_f32(t).first().copied().unwrap_or(f32::NAN));
    let grads = GradientsParams::from_grads(fwd.total.backward(), &net);
    (bc_apply_step(net, optim, lr, grads), pl, vl)
}

struct BcForward<B: AutodiffBackend> {
    total: Tensor<B, 1>,
    policy_ce: Option<Tensor<B, 1>>,
    value_mse: Option<Tensor<B, 1>>,
    sequences: usize,
}

#[allow(clippy::too_many_arguments)]
fn bc_forward_layout<B: AutodiffBackend>(
    net: &Net<B>,
    seqs: &[TokenSeq],
    targets: &[u16],
    outcomes: &[f32],
    label_smoothing: f32,
    report_metrics: bool,
    layout: BatchLayout,
    device: &B::Device,
) -> BcForward<B> {
    let b = seqs.len();
    assert!(b > 0);

    let (logits, values) = forward_layout(net, seqs, layout, device);
    let valid = action_valid_mask::<B>(seqs, device); // true = legal
    let neg_inf = valid.clone().bool_not().float() * LOGIT_MASK;
    let masked_logits = logits + neg_inf;
    let log_p = activation::log_softmax(masked_logits, 1);

    let act_t = Tensor::<B, 2, Int>::from_data(
        TensorData::new(
            targets.iter().map(|&a| a as i32).collect::<Vec<_>>(),
            [b, 1],
        ),
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

    let policy_ce = report_metrics.then(|| hard.mean());
    let value_mse = report_metrics.then(|| value_loss.clone());
    // Terminal-outcome BC warm start, with the existing value weight.
    let total = policy_loss + value_loss * 0.3;
    BcForward {
        total,
        policy_ce,
        value_mse,
        sequences: b,
    }
}

pub struct BcBackward<B: AutodiffBackend> {
    pub grads: GradientsParams,
    pub policy_ce: Option<Tensor<B::InnerBackend, 1>>,
    pub value_mse: Option<Tensor<B::InnerBackend, 1>>,
    pub sequences: usize,
}

#[allow(clippy::too_many_arguments)]
fn bc_backward_layout<B: AutodiffBackend>(
    net: &Net<B>,
    seqs: &[TokenSeq],
    targets: &[u16],
    outcomes: &[f32],
    label_smoothing: f32,
    report_metrics: bool,
    layout: BatchLayout,
    device: &B::Device,
) -> BcBackward<B> {
    let fwd = bc_forward_layout(
        net,
        seqs,
        targets,
        outcomes,
        label_smoothing,
        report_metrics,
        layout,
        device,
    );
    // Inner handles are eager-backend scalars, never a download or detach.
    let policy_ce = fwd.policy_ce.map(|t| t.inner());
    let value_mse = fwd.value_mse.map(|t| t.inner());
    let grads = GradientsParams::from_grads(fwd.total.backward(), net);
    BcBackward {
        grads,
        policy_ce,
        value_mse,
        sequences: fwd.sequences,
    }
}

pub fn bc_apply_step<B: AutodiffBackend, O: Optimizer<Net<B>, B>>(
    net: Net<B>,
    optim: &mut O,
    lr: f64,
    grads: GradientsParams,
) -> Net<B> {
    optim.step(lr, net, grads)
}

// BC parameter pairing and reduction. IDs may be equal across replicas;
// paths, ranks and dimensions define the pairing, with separate grad maps.
pub const BC_PARAM_TENSORS: usize = 112;
pub const BC_PARAM_ELEMENTS: usize = 4_987_394;

#[derive(Clone, Debug)]
pub struct ParamSlot {
    pub path: String,
    pub id: ParamId,
    pub rank: u8,
    pub numel: usize,
    pub dims: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct PairedSlot {
    pub path: String,
    pub id0: ParamId,
    pub id1: ParamId,
    pub rank: u8,
    pub dims: Vec<usize>,
}

#[derive(Debug)]
pub struct PairError(pub String);
impl std::fmt::Display for PairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for PairError {}

pub fn float_param_slots<B: Backend>(net: &Net<B>) -> Vec<ParamSlot> {
    #[derive(Default)]
    struct Collector {
        stack: Vec<String>,
        slots: Vec<ParamSlot>,
    }
    impl<B: Backend> ModuleVisitor<B> for Collector {
        fn enter_module(&mut self, name: &str, _: &str) {
            self.stack.push(name.to_owned());
        }
        fn exit_module(&mut self, _: &str, _: &str) {
            self.stack.pop();
        }
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<B, D>>) {
            let dims = p.val().dims().to_vec();
            self.slots.push(ParamSlot {
                path: self.stack.join("."),
                id: p.id,
                rank: D as u8,
                numel: dims.iter().product(),
                dims,
            });
        }
        // A sentinel slot ensures pairing rejects future non-float params.
        fn visit_int<const D: usize>(&mut self, p: &Param<Tensor<B, D, Int>>) {
            self.slots.push(ParamSlot {
                path: self.stack.join("."),
                id: p.id,
                rank: 0,
                numel: 0,
                dims: vec![],
            });
        }
        fn visit_bool<const D: usize>(&mut self, p: &Param<Tensor<B, D, Bool>>) {
            self.slots.push(ParamSlot {
                path: self.stack.join("."),
                id: p.id,
                rank: 0,
                numel: 0,
                dims: vec![],
            });
        }
    }
    let mut c = Collector::default();
    net.visit(&mut c);
    c.slots
}

pub fn pair_slots(a: &[ParamSlot], b: &[ParamSlot]) -> Result<Vec<PairedSlot>, PairError> {
    let fail = |s: String| PairError(s);
    if a.len() != b.len() || a.len() != BC_PARAM_TENSORS {
        return Err(fail(format!(
            "BC pairing requires {BC_PARAM_TENSORS} slots, found {} and {}",
            a.len(),
            b.len()
        )));
    }
    let mut seen = std::collections::HashSet::new();
    let mut pairs = Vec::with_capacity(a.len());
    for (a, b) in a.iter().zip(b) {
        if a.path != b.path {
            return Err(fail(format!(
                "BC parameter path mismatch: {} vs {}",
                a.path, b.path
            )));
        }
        if !seen.insert(&a.path) {
            return Err(fail(format!("BC duplicate parameter path: {}", a.path)));
        }
        if !matches!(a.rank, 1 | 2)
            || a.rank != b.rank
            || a.dims != b.dims
            || a.dims.len() != a.rank as usize
            || a.numel != a.dims.iter().product::<usize>()
            || b.numel != a.numel
        {
            return Err(fail(format!(
                "BC rank/dimension mismatch or non-float parameter: {}",
                a.path
            )));
        }
        pairs.push(PairedSlot {
            path: a.path.clone(),
            id0: a.id,
            id1: b.id,
            rank: a.rank,
            dims: a.dims.clone(),
        });
    }
    let elements: usize = a.iter().map(|s| s.numel).sum();
    if elements != BC_PARAM_ELEMENTS {
        return Err(fail(format!(
            "BC pairing requires {BC_PARAM_ELEMENTS} elements, found {elements}"
        )));
    }
    Ok(pairs)
}

pub fn weighted_average<B: Backend, const D: usize>(
    grads: &[Tensor<B, D>],
    counts: &[usize],
) -> Tensor<B, D> {
    assert!(!grads.is_empty(), "gradient average must be nonempty");
    assert_eq!(grads.len(), counts.len(), "gradient/count length mismatch");
    assert!(
        counts.iter().all(|&n| n > 0),
        "sequence counts must be positive"
    );
    let total = counts
        .iter()
        .try_fold(0usize, |a, n| a.checked_add(*n))
        .expect("sequence count overflow");
    let dims = grads[0].dims();
    let device = grads[0].device();
    for grad in grads {
        assert_eq!(grad.dims(), dims, "gradient shape mismatch");
        assert_eq!(
            grad.device(),
            device,
            "gradients must be on the same device"
        );
    }
    let mut sum = grads[0].clone() * counts[0] as f32;
    for (g, &n) in grads.iter().zip(counts).skip(1) {
        sum = sum + g.clone() * n as f32;
    }
    sum / total as f32
}

fn average_slot<B: Backend, const D: usize>(
    slot: &PairedSlot,
    a: &mut GradientsParams,
    b: &mut GradientsParams,
    counts: [usize; 2],
    device: &B::Device,
    output: &mut GradientsParams,
) -> Result<(), PairError> {
    let ga = a
        .remove::<B, D>(slot.id0)
        .ok_or_else(|| PairError(format!("BC missing device-0 gradient: {}", slot.path)))?;
    let gb = b
        .remove::<B, D>(slot.id1)
        .ok_or_else(|| PairError(format!("BC missing device-1 gradient: {}", slot.path)))?;
    if ga.dims().as_slice() != slot.dims || gb.dims().as_slice() != slot.dims {
        return Err(PairError(format!(
            "BC gradient dimensions changed: {}",
            slot.path
        )));
    }
    if ga.device() != *device {
        return Err(PairError(format!(
            "BC device-0 gradient on wrong device: {}",
            slot.path
        )));
    }
    let avg = weighted_average(&[ga, gb.to_device(device)], &counts);
    output.register(slot.id0, avg);
    Ok(())
}

pub fn average_gradients<B: Backend>(
    pairs: &[PairedSlot],
    mut a: GradientsParams,
    mut b: GradientsParams,
    counts: [usize; 2],
    device: &B::Device,
) -> Result<GradientsParams, PairError> {
    let mut output = GradientsParams::new();
    for slot in pairs {
        match slot.rank {
            1 => average_slot::<B, 1>(slot, &mut a, &mut b, counts, device, &mut output)?,
            2 => average_slot::<B, 2>(slot, &mut a, &mut b, counts, device, &mut output)?,
            _ => return Err(PairError(format!("BC unsupported rank: {}", slot.path))),
        }
    }
    if !a.is_empty() || !b.is_empty() {
        return Err(PairError(
            "BC unexpected gradients outside pairing table".into(),
        ));
    }
    Ok(output)
}

/// Materialize every primary weight before cloning its parameter state.
/// Burn 0.21 clones lazy initializers independently; matching IDs alone do
/// not imply matching values until the source parameters are initialized.
/// This reads tensor metadata, not tensor data, and retains fork-disjoint tapes.
pub fn fork_bc_replica<B: Backend>(
    net: &Net<B>,
    device: &B::Device,
) -> Result<(Net<B>, Vec<PairedSlot>), PairError> {
    let primary_slots = float_param_slots(net);
    pair_slots(&primary_slots, &primary_slots)?;
    let replica = net.clone().fork(device);
    let pairs = pair_slots(&primary_slots, &float_param_slots(&replica))?;
    Ok((replica, pairs))
}

pub fn check_pair_ids<B: Backend>(
    net: &Net<B>,
    replica: &Net<B>,
    pairs: &[PairedSlot],
) -> Result<(), PairError> {
    let current = pair_slots(&float_param_slots(net), &float_param_slots(replica))?;
    for (now, old) in current.iter().zip(pairs) {
        if now.path != old.path || now.id0 != old.id0 || now.id1 != old.id1 {
            return Err(PairError(format!("BC parameter IDs changed: {}", old.path)));
        }
    }
    if current.len() != pairs.len() {
        return Err(PairError("BC pairing length changed".into()));
    }
    Ok(())
}

/// Copy live weights into the existing replica while preserving its IDs.
/// Both mapper passes use the same module-path stack as the slot visitor.
pub fn broadcast_weights<B: AutodiffBackend>(
    net: &Net<B>,
    replica: Net<B>,
    pairs: &[PairedSlot],
    device: &B::Device,
) -> Result<Net<B>, PairError> {
    struct Broadcast<'a, B: AutodiffBackend> {
        stack: Vec<String>,
        at: usize,
        pairs: &'a [PairedSlot],
        device: &'a B::Device,
        weights: GradientsParams,
        source: bool,
        error: Option<PairError>,
    }
    impl<B: AutodiffBackend> ModuleMapper<B> for Broadcast<'_, B> {
        fn enter_module(&mut self, name: &str, _: &str) {
            self.stack.push(name.to_owned());
        }
        fn exit_module(&mut self, _: &str, _: &str) {
            self.stack.pop();
        }
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let path = self.stack.join(".");
            let Some(slot) = self.pairs.get(self.at) else {
                self.error = Some(PairError(format!("BC broadcast unexpected path: {path}")));
                return param;
            };
            self.at += 1;
            let expected_id = if self.source { slot.id0 } else { slot.id1 };
            if path != slot.path
                || param.id != expected_id
                || D != slot.rank as usize
                || param.val().dims().as_slice() != slot.dims
            {
                self.error = Some(PairError(format!(
                    "BC broadcast path/id/shape mismatch: {path} vs {}",
                    slot.path
                )));
                return param;
            }
            if self.source {
                let tensor = param.val();
                let requires_grad = tensor.is_require_grad();
                let tensor = tensor.to_device(self.device).detach();
                let tensor = if requires_grad {
                    tensor.require_grad()
                } else {
                    tensor
                };
                self.weights.register(slot.id1, tensor);
                param
            } else {
                let Some(tensor) = self.weights.remove::<B, D>(slot.id1) else {
                    self.error = Some(PairError(format!("BC broadcast missing path: {path}")));
                    return param;
                };
                let (id, _, mapper) = param.consume();
                Param::from_mapped_value(id, tensor, mapper)
            }
        }
    }
    check_pair_ids(net, &replica, pairs)?;
    let mut mapper = Broadcast::<B> {
        stack: vec![],
        at: 0,
        pairs,
        device,
        weights: GradientsParams::new(),
        source: true,
        error: None,
    };
    let _ = net.clone().map(&mut mapper);
    if let Some(e) = mapper.error.take() {
        return Err(e);
    }
    if mapper.at != pairs.len() {
        return Err(PairError("BC broadcast missing source slots".into()));
    }
    mapper.source = false;
    mapper.at = 0;
    let replica = replica.map(&mut mapper);
    if let Some(e) = mapper.error {
        return Err(e);
    }
    if mapper.at != pairs.len() || !mapper.weights.is_empty() {
        return Err(PairError("BC broadcast missing replica slots".into()));
    }
    Ok(replica)
}

/// Per-device slots are written only by their worker; the host slot changes
/// only outside the launch window. No recovery is possible after CUDA OOM.
#[derive(Default)]
pub struct BcPhases {
    cuda0: std::sync::atomic::AtomicU8,
    cuda1: std::sync::atomic::AtomicU8,
    host: std::sync::atomic::AtomicU8,
}
impl BcPhases {
    fn store(slot: &std::sync::atomic::AtomicU8, phase: u8) {
        slot.store(phase, std::sync::atomic::Ordering::Relaxed);
    }
    pub fn install_hook(self: &std::sync::Arc<Self>) {
        let phases = self.clone();
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let load = |slot: &std::sync::atomic::AtomicU8| {
                slot.load(std::sync::atomic::Ordering::Relaxed)
            };
            let worker = |n| match n {
                1 => "forward-backward",
                2 => "returned",
                _ => "idle",
            };
            let host = match load(&phases.host) {
                1 => "grad-average",
                2 => "adam",
                3 => "broadcast",
                _ => "idle",
            };
            eprintln!("bc: fatal during two-device step, cuda:0 {}, cuda:1 {}, host {host}. No fallback to one GPU and no batch shrink. 256-per-card may OOM a 16 GB T4; rerun with --batch-size 128.", worker(load(&phases.cuda0)), worker(load(&phases.cuda1)));
            previous(info);
        }));
    }
}

#[allow(clippy::too_many_arguments)]
fn bc_backward_pair<B: AutodiffBackend>(
    net: &Net<B>,
    replica: &Net<B>,
    seqs: [&[TokenSeq]; 2],
    targets: [&[u16]; 2],
    outcomes: [&[f32]; 2],
    smoothing: f32,
    metrics: bool,
    layouts: [BatchLayout; 2],
    devices: &[B::Device],
    phases: &BcPhases,
) -> [BcBackward<B>; 2] {
    BcPhases::store(&phases.host, 0);
    std::thread::scope(|scope| {
        let a = scope.spawn(|| {
            BcPhases::store(&phases.cuda0, 1);
            let result = bc_backward_layout(
                net,
                seqs[0],
                targets[0],
                outcomes[0],
                smoothing,
                metrics,
                layouts[0],
                &devices[0],
            );
            BcPhases::store(&phases.cuda0, 2);
            result
        });
        let b = scope.spawn(|| {
            BcPhases::store(&phases.cuda1, 1);
            let result = bc_backward_layout(
                replica,
                seqs[1],
                targets[1],
                outcomes[1],
                smoothing,
                metrics,
                layouts[1],
                &devices[1],
            );
            BcPhases::store(&phases.cuda1, 2);
            result
        });
        // Join both before propagating a worker panic, and before any read/copy.
        let a = a.join();
        let b = b.join();
        let unwrap = |r: std::thread::Result<BcBackward<B>>| match r {
            Ok(v) => v,
            Err(e) => std::panic::resume_unwind(e),
        };
        [unwrap(a), unwrap(b)]
    })
}

#[allow(clippy::too_many_arguments)]
pub fn bc_two_device_step<B: AutodiffBackend, O: Optimizer<Net<B>, B>>(
    net: Net<B>,
    replica: Net<B>,
    pairs: &[PairedSlot],
    optim: &mut O,
    lr: f64,
    seqs: [&[TokenSeq]; 2],
    targets: [&[u16]; 2],
    outcomes: [&[f32]; 2],
    smoothing: f32,
    metrics: bool,
    devices: &[B::Device],
    phases: &BcPhases,
) -> Result<(Net<B>, Net<B>, Option<f32>, Option<f32>), PairError> {
    assert_eq!(devices.len(), 2);
    assert_eq!(seqs[0].len(), seqs[1].len());
    assert!(seqs[0].len() >= 2);
    let [a, b] = bc_backward_pair(
        &net,
        &replica,
        seqs,
        targets,
        outcomes,
        smoothing,
        metrics,
        [
            BatchLayout::for_batch(seqs[0]),
            BatchLayout::for_batch(seqs[1]),
        ],
        devices,
        phases,
    );
    BcPhases::store(&phases.host, 1);
    let counts = [a.sequences, b.sequences];
    let scalar =
        |a: Option<Tensor<B::InnerBackend, 1>>, b: Option<Tensor<B::InnerBackend, 1>>| match (a, b)
        {
            (Some(a), Some(b)) => {
                let a = download_f32(a)[0];
                let b = download_f32(b)[0];
                Some((counts[0] as f32 * a + counts[1] as f32 * b) / (counts[0] + counts[1]) as f32)
            }
            (None, None) => None,
            _ => panic!("BC inconsistent metric availability"),
        };
    let pl = scalar(a.policy_ce, b.policy_ce);
    let vl = scalar(a.value_mse, b.value_mse);
    let grads = average_gradients::<B::InnerBackend>(pairs, a.grads, b.grads, counts, &devices[0])?;
    BcPhases::store(&phases.host, 2);
    let net = bc_apply_step(net, optim, lr, grads);
    BcPhases::store(&phases.host, 3);
    let replica = broadcast_weights(&net, replica, pairs, &devices[1])?;
    Ok((net, replica, pl, vl))
}

/// Dense-layout, dropout-free gradient parity on three fork-disjoint graphs.
/// Does not step Adam, change live weights, or write a checkpoint.
#[cfg(any(test, feature = "tch"))]
pub fn bc_parity<B: AutodiffBackend>(devices: &[B::Device]) -> Result<(), PairError> {
    assert_eq!(devices.len(), 2);
    B::seed(&devices[0], 42);
    let net = Net::<B>::fresh_with_dropout(&devices[0], 0.0);
    let (replica, pairs) = fork_bc_replica(&net, &devices[1])?;
    let (reference, ref_pairs) = fork_bc_replica(&net, &devices[0])?;
    let samples = crate::recorder::benchmark_samples(42);
    assert!(!samples.is_empty());
    let batch: Vec<_> = samples.iter().cycle().take(4).collect();
    let seqs: Vec<_> = batch.iter().map(|s| s.tokens).collect();
    let targets: Vec<_> = batch.iter().map(|s| s.target).collect();
    let outcomes: Vec<_> = batch.iter().map(|s| s.outcome).collect();
    let reference_grads = bc_backward_layout(
        &reference,
        &seqs,
        &targets,
        &outcomes,
        0.05,
        false,
        BatchLayout::dense(),
        &devices[0],
    );
    let [a, b] = bc_backward_pair(
        &net,
        &replica,
        [&seqs[..2], &seqs[2..]],
        [&targets[..2], &targets[2..]],
        [&outcomes[..2], &outcomes[2..]],
        0.05,
        false,
        [BatchLayout::dense(); 2],
        devices,
        &BcPhases::default(),
    );
    let mut avg =
        average_gradients::<B::InnerBackend>(&pairs, a.grads, b.grads, [2, 2], &devices[0])?;
    let mut reference = reference_grads.grads;
    fn compare<B: Backend, const D: usize>(
        pair: &PairedSlot,
        reference_pair: &PairedSlot,
        avg: &mut GradientsParams,
        reference: &mut GradientsParams,
    ) -> Result<(f32, f32, f32, f32), PairError> {
        let g = avg
            .remove::<B, D>(pair.id0)
            .ok_or_else(|| PairError(format!("bc-parity missing average: {}", pair.path)))?;
        let r = reference
            .remove::<B, D>(reference_pair.id1)
            .ok_or_else(|| PairError(format!("bc-parity missing reference: {}", pair.path)))?;
        let diff = g - r.clone();
        let abs = download_f32(diff.clone().abs().max())[0];
        let dl2 = download_f32(diff.clone().mul(diff).sum().sqrt())[0];
        let rl2 = download_f32(r.clone().mul(r).sum().sqrt())[0];
        Ok((abs, dl2, rl2, dl2 / rl2.max(1e-8)))
    }
    let mut worst = (String::new(), 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut worst_score = -1.0f32;
    let mut failed = false;
    for (pair, rp) in pairs.iter().zip(&ref_pairs) {
        let (abs, dl2, rl2, rel) = match pair.rank {
            1 => compare::<B::InnerBackend, 1>(pair, rp, &mut avg, &mut reference)?,
            2 => compare::<B::InnerBackend, 2>(pair, rp, &mut avg, &mut reference)?,
            _ => unreachable!(),
        };
        let finite = [abs, dl2, rl2, rel].iter().all(|x| x.is_finite());
        failed |= !finite || abs > 1e-4 || rel > 1e-4;
        let score = if finite { abs.max(rel) } else { f32::INFINITY };
        if score > worst_score {
            worst_score = score;
            worst = (pair.path.clone(), abs, dl2, rl2, rel);
        }
    }
    println!("bc-parity: {} ({} tensors), worst={} max_abs={:.6e} diff_l2={:.6e} ref_l2={:.6e} relative_l2={:.6e}",
        if failed { "FAILED" } else { "passed" }, pairs.len(), worst.0, worst.1, worst.2, worst.3, worst.4);
    if failed {
        Err(PairError(format!(
            "bc-parity tolerance 1e-4 exceeded: {}",
            worst.0
        )))
    } else {
        Ok(())
    }
}

// ── TD-value training step (ASTRA7.md §4.2) ─────────────────────────────
//
// Frozen-trunk value regression: only value_hidden/value_out receive
// gradients. Mechanism (§8.2, verified against ml/src/net_burn.rs's
// `frozen_or_live_body`, the one other place this codebase freezes a burn
// trunk): run the trunk on `net.valid()` - the INNER, non-autodiff backend -
// then re-import the pooled features via `Tensor::from_inner`. This is NOT
// `detach()`: a detached autodiff tensor still carries a graph node that
// only `backward()` frees, so a per-batch detach leaks the whole trunk graph
// (measured ~5GB/epoch in v6). Computing on the inner backend never builds
// that graph in the first place. The value head's own Linear layers are
// still the LIVE autodiff `net`'s Params, so `loss.backward()` produces
// gradients for exactly value_hidden/value_out - GradientsParams::from_grads
// naturally has no entries for type_embed/field_embed/encoder/pointer_*
// because they were never touched by this forward pass, so optim.step is a
// value-head-only update without needing an explicit param-group API.
pub fn td_train_step<B: AutodiffBackend, O: Optimizer<Net<B>, B>>(
    net: Net<B>,
    optim: &mut O,
    lr: f64,
    seqs: &[TokenSeq],
    td_targets: &[f32],
    report_metrics: bool,
    device: &B::Device,
) -> (Net<B>, Option<f32>) {
    let b = seqs.len();
    assert!(b > 0);

    let vnet = net.valid(); // inner backend snapshot of the current weights
    let (enc_inner, pad_inner, layout) = encode_trunk(&vnet, seqs, device);
    let pooled_inner = pool_state(enc_inner, pad_inner, layout.state);
    let pooled: Tensor<B, 2> = Tensor::from_inner(pooled_inner); // fresh leaf, no grad path to the trunk

    let value = value_from_pooled(&net, pooled);
    let tgt = Tensor::<B, 2>::from_data(TensorData::new(td_targets.to_vec(), [b, 1]), device);
    let diff = value - tgt;
    let loss = (diff.clone() * diff).mean();
    let vl = if report_metrics {
        Some(download_f32(loss.clone()).first().copied().unwrap_or(f32::NAN))
    } else {
        None
    };

    let grads = GradientsParams::from_grads(loss.backward(), &net);
    (optim.step(lr, net, grads), vl)
}

// ── TD-value UNFROZEN training step (2026-07-10 postmortem) ─────────────
//
// `td_train_step` above (frozen trunk, head-only) hit a ceiling on the
// value-rank gate - two Fable 5 consults converged on the frozen BC trunk's
// pooled features being the binding constraint, not the TD(λ) target
// formula (see STATE.md's "TD-value stage" section for the full trail:
// λ=0.9 head-only scored 50.8%, λ=1.0 head-only scored 44.8% - WORSE, and
// statistically indistinguishable from an untrained net - against a 65%
// gate). This step unfreezes the whole net: `forward()` runs on the LIVE
// autodiff `net` (not `net.valid()`), so `loss.backward()` produces
// gradients for type_embed/field_embed/encoder/pointer_*/value_* alike -
// genuinely joint training, not the value-head-only update above.
//
// Unfreezing the trunk under a value-only loss would happily distort the
// pointer-policy features BC already spent a full training run shaping
// (nothing in a value-MSE loss cares about preserving policy quality) - so
// this adds a policy-distillation anchor: cross-entropy between the LIVE
// net's policy output and a FROZEN teacher snapshot's (masked) softmax
// distribution on the same states. The teacher is a `.valid()` snapshot
// taken once by the caller before this fine-tuning phase begins (passed in
// unchanged across every call, NOT re-derived per step - it's meant to
// anchor to "what BC produced," not to a chasing/moving target), imported
// via `Tensor::from_inner` for the same reason `td_train_step` uses that
// trick on the trunk above: `detach()` would still leak a graph node per
// call, `from_inner` never builds one.
pub fn td_unfrozen_train_step<B: AutodiffBackend, O: Optimizer<Net<B>, B>>(
    net: Net<B>,
    teacher: &Net<B::InnerBackend>,
    optim: &mut O,
    lr: f64,
    seqs: &[TokenSeq],
    td_targets: &[f32],
    distill_weight: f32,
    report_metrics: bool,
    device: &B::Device,
) -> (Net<B>, Option<f32>, Option<f32>) {
    let b = seqs.len();
    assert!(b > 0);

    let (logits, value) = forward(&net, seqs, device);
    let valid = action_valid_mask::<B>(seqs, device);
    let neg_inf = valid.clone().bool_not().float() * LOGIT_MASK;
    let masked_logits = logits + neg_inf;
    let log_p = activation::log_softmax(masked_logits, 1);

    let tgt = Tensor::<B, 2>::from_data(TensorData::new(td_targets.to_vec(), [b, 1]), device);
    let diff = value - tgt;
    let value_loss = (diff.clone() * diff).mean();

    let (teacher_logits, _teacher_value) = forward(teacher, seqs, device);
    let teacher_valid = action_valid_mask::<B::InnerBackend>(seqs, device);
    let teacher_neg_inf = teacher_valid.bool_not().float() * LOGIT_MASK;
    let teacher_masked = teacher_logits + teacher_neg_inf;
    let teacher_probs_inner = activation::softmax(teacher_masked, 1);
    let teacher_probs: Tensor<B, 2> = Tensor::from_inner(teacher_probs_inner); // fresh leaf, no grad path to the teacher
    let distill_loss = -(teacher_probs * log_p).sum_dim(1).mean();

    let (vl, dl) = if report_metrics {
        (
            Some(download_f32(value_loss.clone()).first().copied().unwrap_or(f32::NAN)),
            Some(download_f32(distill_loss.clone()).first().copied().unwrap_or(f32::NAN)),
        )
    } else {
        (None, None)
    };

    let total = value_loss + distill_loss * distill_weight;
    let grads = GradientsParams::from_grads(total.backward(), &net);
    (optim.step(lr, net, grads), vl, dl)
}

// ── AZ league training step (ASTRA7.md §4.3) ────────────────────────────
//
// Unlike BC's hard-index CE, the policy target here is the MCTS visit-count
// distribution (soft, already legal-set-normalized) for choose_action/
// chain-response decisions, or a one-hot on the net's own choice for
// decide/pick_targets/choose_discards (those three have no resumable
// engine continuation to search over - see mcts.rs's module doc). Cross-
// entropy against a full distribution rather than argmax + label smoothing:
// same shape v6's az_train_step used (ml/src/net_burn.rs). Trunk is NOT
// frozen here - AZ is meant to keep refining the whole network, unlike the
// TD-value stage which only warms up the value head.
pub fn az_train_step<B: AutodiffBackend, O: Optimizer<Net<B>, B>>(
    net: Net<B>,
    optim: &mut O,
    lr: f64,
    seqs: &[TokenSeq],
    visit_dists: &[[f32; ACTION_MAX]],
    outcomes: &[f32],
    report_metrics: bool,
    device: &B::Device,
) -> (Net<B>, Option<f32>, Option<f32>) {
    let b = seqs.len();
    assert!(b > 0);

    let (logits, values) = forward(&net, seqs, device);
    let valid = action_valid_mask::<B>(seqs, device);
    let neg_inf = valid.bool_not().float() * LOGIT_MASK;
    let log_p = activation::log_softmax(logits + neg_inf, 1);

    let visit_flat: Vec<f32> = visit_dists.iter().flat_map(|v| v.iter().copied()).collect();
    let visit_t = Tensor::<B, 2>::from_data(TensorData::new(visit_flat, [b, ACTION_MAX]), device);
    let policy_loss = -(visit_t * log_p).sum_dim(1).mean();

    let tgt = Tensor::<B, 2>::from_data(TensorData::new(outcomes.to_vec(), [b, 1]), device);
    let diff = values - tgt;
    let value_loss = (diff.clone() * diff).mean();

    let (pl, vl) = if report_metrics {
        (
            Some(download_f32(policy_loss.clone()).first().copied().unwrap_or(f32::NAN)),
            Some(download_f32(value_loss.clone()).first().copied().unwrap_or(f32::NAN)),
        )
    } else {
        (None, None)
    };

    let total = policy_loss + value_loss;
    let grads = GradientsParams::from_grads(total.backward(), &net);
    (optim.step(lr, net, grads), pl, vl)
}

/// Forward-only policy evaluation (argmax-correct count + summed hard CE).
/// Callers must pass `net.valid()` - forward passes on the autodiff backend
/// accumulate graph state that only `backward()` releases (ASTRA7.md §8.2).
/// The epoch loop uses `val_diag` instead; this stays for the top-1 count.
#[allow(dead_code)]
pub fn policy_eval<B: Backend>(net: &Net<B>, seqs: &[TokenSeq], targets: &[u16], device: &B::Device) -> (usize, f32) {
    let b = seqs.len();
    if b == 0 {
        return (0, 0.0);
    }
    let (logits, _values) = forward(net, seqs, device);
    let valid = action_valid_mask::<B>(seqs, device);
    let neg_inf = valid.bool_not().float() * LOGIT_MASK;
    let masked = download_f32(logits + neg_inf);

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
/// pointer index) instead of one aggregate number - answers "is any
/// decision kind (esp. chain responses, this project's single largest
/// measured lever - ASTRA7.md §1.1) lagging the others" directly, rather
/// than guessing from an average. Returns `[(correct, total, ce_sum); 16]`
/// indexed by `ActionKind as usize`. Same `net.valid()` calling convention
/// as `policy_eval` (§8.2) - this is a forward-only pass.
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
    let neg_inf = valid.bool_not().float() * LOGIT_MASK;
    let masked = download_f32(logits + neg_inf);

    let action_base = STATE_MAX + HISTORY_MAX;
    for i in 0..b {
        let row = &masked[i * ACTION_MAX..(i + 1) * ACTION_MAX];
        let target_idx = targets[i] as usize;
        let kind = seqs[i][action_base + target_idx].fields[0] as usize;
        if kind >= NUM_ACTION_KINDS {
            continue; // defensive - should never happen given the encoders
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

// ── Validation report ────────────────────────────────────────────────────
//
// One forward per validation batch. The extra numbers (top-k, the teacher's
// probability and rank, entropy, per-decision accuracy, value calibration,
// and accuracy by legal-count and history length) are scored on the host
// from logits the eval already downloads.

#[derive(Clone, Copy, Debug, Default)]
struct Count {
    hit: u64,
    n: u64,
}

#[derive(Clone, Debug, Default)]
pub struct ValDiag {
    n: u64,
    top1: u64,
    top3: u64,
    top5: u64,
    ce: f64,
    teacher_p: f64,
    teacher_rank: f64,
    teacher_margin: f64,
    entropy: f64,
    /// Play (including Pass), Chain, Decide, Target, Discard.
    groups: [Count; 5],
    se: f64,
    ae: f64,
    value_sum: f64,
    winner_sum: f64,
    winner_n: u64,
    loser_sum: f64,
    loser_n: u64,
    scores: Vec<f32>,
    labels: Vec<bool>,
    legal: [Count; 5],
    history: [Count; 5],
}

impl ValDiag {
    pub fn absorb(&mut self, other: ValDiag) {
        self.n += other.n;
        self.top1 += other.top1;
        self.top3 += other.top3;
        self.top5 += other.top5;
        self.ce += other.ce;
        self.teacher_p += other.teacher_p;
        self.teacher_rank += other.teacher_rank;
        self.teacher_margin += other.teacher_margin;
        self.entropy += other.entropy;
        for i in 0..5 {
            self.groups[i].hit += other.groups[i].hit;
            self.groups[i].n += other.groups[i].n;
            self.legal[i].hit += other.legal[i].hit;
            self.legal[i].n += other.legal[i].n;
            self.history[i].hit += other.history[i].hit;
            self.history[i].n += other.history[i].n;
        }
        self.se += other.se;
        self.ae += other.ae;
        self.value_sum += other.value_sum;
        self.winner_sum += other.winner_sum;
        self.winner_n += other.winner_n;
        self.loser_sum += other.loser_sum;
        self.loser_n += other.loser_n;
        self.scores.extend(other.scores);
        self.labels.extend(other.labels);
    }

    pub fn top1_count(&self) -> usize { self.top1 as usize }
    pub fn ce_sum(&self) -> f32 { self.ce as f32 }
    pub fn len(&self) -> usize { self.n as usize }

    pub fn to_json(&self) -> String {
        let mean = |sum: f64| -> Option<f64> {
            if self.n == 0 { None } else { Some(sum / self.n as f64) }
        };
        let rate = |hit: u64| mean(hit as f64);
        let bucket = |c: Count| serde_json::json!({
            "acc": json_num(if c.n == 0 { None } else { Some(c.hit as f64 / c.n as f64) }),
            "n": c.n,
        });
        let bins = |counts: [Count; 5], labels: [&str; 5]| {
            labels.iter().zip(counts).map(|(label, c)| {
                serde_json::json!({
                    "label": label,
                    "acc": json_num(if c.n == 0 { None } else { Some(c.hit as f64 / c.n as f64) }),
                    "n": c.n,
                })
            }).collect::<Vec<_>>()
        };
        let names = ["play", "chain", "decide", "target", "discard"];
        let mut groups = serde_json::Map::new();
        for (name, c) in names.iter().zip(self.groups) {
            groups.insert((*name).to_string(), bucket(c));
        }
        serde_json::json!({
            "n": self.n,
            "top1": json_num(rate(self.top1)),
            "top3": json_num(rate(self.top3)),
            "top5": json_num(rate(self.top5)),
            "teacher_p": json_num(mean(self.teacher_p)),
            "teacher_rank": json_num(mean(self.teacher_rank)),
            "teacher_margin": json_num(mean(self.teacher_margin)),
            "entropy": json_num(mean(self.entropy)),
            "groups": groups,
            "mse": json_num(mean(self.se)),
            "mae": json_num(mean(self.ae)),
            "mean_v": json_num(mean(self.value_sum)),
            "winner_v": json_num(if self.winner_n == 0 { None } else { Some(self.winner_sum / self.winner_n as f64) }),
            "loser_v": json_num(if self.loser_n == 0 { None } else { Some(self.loser_sum / self.loser_n as f64) }),
            "winner_n": self.winner_n,
            "loser_n": self.loser_n,
            "auc": json_num(roc_auc(&self.scores, &self.labels).map(|a| a as f64)),
            "legal": bins(self.legal, ["1", "2-3", "4-7", "8-15", "16+"]),
            "history": bins(self.history, ["0", "1-8", "9-24", "25-48", "49-64"]),
        }).to_string()
    }
}

fn json_num(v: Option<f64>) -> serde_json::Value {
    match v {
        Some(x) if x.is_finite() => serde_json::json!(x),
        _ => serde_json::Value::Null,
    }
}

/// ROC AUC of `scores` against binary `positive` labels. Ties share a rank.
pub(crate) fn roc_auc(scores: &[f32], positive: &[bool]) -> Option<f32> {
    if scores.len() != positive.len() || scores.is_empty() {
        return None;
    }
    let n_pos = positive.iter().filter(|p| **p).count();
    let n_neg = scores.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return None;
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut rank_sum_pos = 0.0f64;
    let mut i = 0;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && scores[order[j]] == scores[order[i]] {
            j += 1;
        }
        let avg = 0.5 * ((i + 1) as f64 + j as f64);
        for k in i..j {
            if positive[order[k]] {
                rank_sum_pos += avg;
            }
        }
        i = j;
    }
    let auc = (rank_sum_pos - n_pos as f64 * (n_pos as f64 + 1.0) / 2.0)
        / (n_pos as f64 * n_neg as f64);
    Some(auc as f32)
}

fn decision_group(kind: u8) -> Option<usize> {
    match kind {
        0..=10 => Some(0), // Pass and every Play*
        11 | 12 => Some(1),
        13 => Some(2),
        14 => Some(3),
        15 => Some(4),
        _ => None,
    }
}

fn legal_bin(n: usize) -> usize {
    match n {
        0..=1 => 0,
        2..=3 => 1,
        4..=7 => 2,
        8..=15 => 3,
        _ => 4,
    }
}

fn history_bin(n: usize) -> usize {
    match n {
        0 => 0,
        1..=8 => 1,
        9..=24 => 2,
        25..=48 => 3,
        _ => 4,
    }
}

fn history_len(seq: &TokenSeq) -> usize {
    seq[STATE_MAX..STATE_MAX + HISTORY_MAX]
        .iter()
        .filter(|t| t.ttype != TokenType::Pad as u8)
        .count()
}

fn legal_len(seq: &TokenSeq) -> usize {
    let base = STATE_MAX + HISTORY_MAX;
    seq[base..base + ACTION_MAX]
        .iter()
        .rposition(|t| t.ttype != TokenType::Pad as u8)
        .map_or(0, |i| i + 1)
}

/// Score one masked action row. `legal` is the packed prefix of real actions.
/// Rank is 1-based: 1 + the number of legal actions scored strictly higher.
fn observe_row(diag: &mut ValDiag, legal: &[f32], target: usize, value: f32, outcome: f32, kind: u8, history: usize) {
    if target >= legal.len() || !value.is_finite() || !outcome.is_finite() {
        return;
    }
    if legal.iter().any(|v| !v.is_finite()) {
        return;
    }
    let n_legal = legal.len();
    let teacher = legal[target];
    let better = legal.iter().filter(|v| **v > teacher).count();
    let m = legal.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = legal.iter().map(|v| (v - m).exp()).collect();
    let z = exps.iter().sum::<f32>();
    if !z.is_finite() || z <= 0.0 {
        return;
    }
    let p = exps[target] / z;
    let best_other = legal.iter().enumerate()
        .filter(|(i, _)| *i != target)
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let margin = if n_legal < 2 || !best_other.is_finite() { 0.0 } else { teacher - best_other };
    let entropy = if n_legal <= 1 {
        0.0
    } else {
        let h: f32 = exps.iter().map(|e| {
            let pi = e / z;
            if pi > 0.0 { -pi * pi.ln() } else { 0.0 }
        }).sum();
        h / (n_legal as f32).ln()
    };
    if !p.is_finite() || !entropy.is_finite() || !margin.is_finite() {
        return;
    }

    let top1 = better == 0;
    diag.n += 1;
    diag.top1 += top1 as u64;
    diag.top3 += (better < 3) as u64;
    diag.top5 += (better < 5) as u64;
    diag.ce += ((z.ln() + m) - teacher) as f64;
    diag.teacher_p += p as f64;
    diag.teacher_rank += (1 + better) as f64;
    diag.teacher_margin += margin as f64;
    diag.entropy += entropy as f64;
    if let Some(g) = decision_group(kind) {
        diag.groups[g].n += 1;
        diag.groups[g].hit += top1 as u64;
    }
    let err = value - outcome;
    diag.se += (err * err) as f64;
    diag.ae += err.abs() as f64;
    diag.value_sum += value as f64;
    if outcome >= 0.5 {
        diag.winner_sum += value as f64;
        diag.winner_n += 1;
        diag.labels.push(true);
    } else {
        diag.loser_sum += value as f64;
        diag.loser_n += 1;
        diag.labels.push(false);
    }
    diag.scores.push(value);
    let lb = legal_bin(n_legal);
    diag.legal[lb].n += 1;
    diag.legal[lb].hit += top1 as u64;
    let hb = history_bin(history);
    diag.history[hb].n += 1;
    diag.history[hb].hit += top1 as u64;
}

/// Forward-only validation report. Callers pass `net.valid()`.
pub fn val_diag<B: Backend>(
    net: &Net<B>,
    seqs: &[TokenSeq],
    targets: &[u16],
    outcomes: &[f32],
    device: &B::Device,
) -> ValDiag {
    let mut diag = ValDiag::default();
    let b = seqs.len();
    if b == 0 {
        return diag;
    }
    let (logits, values) = forward(net, seqs, device);
    let valid = action_valid_mask::<B>(seqs, device);
    let neg_inf = valid.bool_not().float() * LOGIT_MASK;
    let masked = download_f32(logits + neg_inf);
    let values = download_f32(values);
    let action_base = STATE_MAX + HISTORY_MAX;
    for i in 0..b {
        let n_legal = legal_len(&seqs[i]);
        let target = targets[i] as usize;
        let row = &masked[i * ACTION_MAX..i * ACTION_MAX + n_legal.min(ACTION_MAX)];
        let kind = if target < ACTION_MAX {
            seqs[i][action_base + target].fields[0]
        } else {
            255
        };
        let value = values.get(i).copied().unwrap_or(f32::NAN);
        observe_row(&mut diag, row, target, value, outcomes[i], kind, history_len(&seqs[i]));
    }
    diag
}

/// Compare the old dense layout with compact execution using identical weights
/// and a batch sampled across a real seeded teacher game. Kernel warmup and
/// explicit synchronization keep GPU compilation and queued work out of timings.
pub fn benchmark<B: AutodiffBackend>(samples: &[crate::recorder::Sample], batch_size: usize,
    steps: usize, rounds: usize, compact_only: bool, device: &B::Device) {
    assert!(steps > 0 && rounds > 0 && batch_size > 0 && !samples.is_empty());
    B::seed(device, 42);
    let initial = Net::<B>::fresh(device);
    let indices: Vec<_> = (0..batch_size).map(|i| i * samples.len() / batch_size).collect();
    let seqs: Vec<_> = indices.iter().map(|&i| samples[i].tokens).collect();
    let targets: Vec<_> = indices.iter().map(|&i| samples[i].target).collect();
    let outcomes: Vec<_> = indices.iter().map(|&i| samples[i].outcome).collect();
    let compact = BatchLayout::for_batch(&seqs);
    println!("{}", serde_json::json!({"backend": B::name(device), "batch_size": batch_size,
        "compact_tokens": compact.len(), "dense_tokens": BatchLayout::dense().len(),
        "sequence_bytes": std::mem::size_of::<TokenSeq>(), "steps": steps, "compact_only": compact_only}));
    let probe = initial.clone();
    let mut probe_optim = make_optimizer::<B>();
    let (_, policy_loss, value_loss) = bc_train_step_layout(
        probe, &mut probe_optim, 0.0003, &seqs, &targets, &outcomes, 0.05, true, compact, device,
    );
    println!("{}", serde_json::json!({"probe_policy_loss": policy_loss, "probe_value_loss": value_loss}));
    for round in 0..rounds {
        let order = if round % 2 == 0 { [false, true] } else { [true, false] };
        let modes: &[bool] = if compact_only { &[true] } else { &order };
        for &packed in modes {
            let layout = if packed { compact } else { BatchLayout::dense() };
            let mode = if packed { "compact" } else { "dense" };
            let vnet = initial.valid();
            let _ = forward_layout(&vnet, &seqs, layout, device);
            B::sync(device).expect("inference warmup failed");
            let t = std::time::Instant::now();
            for _ in 0..steps {
                let (p, v) = forward_layout(&vnet, &seqs, layout, device);
                std::hint::black_box((p.into_data(), v.into_data()));
            }
            B::sync(device).expect("inference synchronization failed");
            let inference = t.elapsed().as_secs_f64() / steps as f64;

            let mut net = initial.clone();
            let mut optim = make_optimizer::<B>();
            net = bc_train_step_layout(net, &mut optim, 0.0003, &seqs, &targets,
                &outcomes, 0.05, false, layout, device).0;
            B::sync(device).expect("training warmup failed");
            let t = std::time::Instant::now();
            for _ in 0..steps {
                net = bc_train_step_layout(net, &mut optim, 0.0003, &seqs, &targets,
                    &outcomes, 0.05, false, layout, device).0;
            }
            B::sync(device).expect("training synchronization failed");
            println!("{}", serde_json::json!({"mode": mode, "round": round,
                "inference_seconds_per_batch": inference,
                "training_seconds_per_batch": t.elapsed().as_secs_f64() / steps as f64}));
        }
    }
}

#[cfg(test)]
mod compact_tests {
    use super::*;
    use crate::tokens::{Token, SEQ_LEN};
    use burn::backend::Autodiff;
    use burn_ndarray::{NdArray, NdArrayDevice};

    fn fixture() -> Vec<TokenSeq> {
        [(3, 0, 2), (17, 11, 13)].into_iter().map(|(s, h, a)| {
            let mut seq = [Token { ttype: 0, fields: [0; NUM_FIELDS] }; SEQ_LEN];
            for (start, len, kind) in [(0, s, 1), (STATE_MAX, h, 6), (STATE_MAX + HISTORY_MAX, a, 7)] {
                for i in 0..len {
                    seq[start + i] = Token { ttype: kind, fields: [((i + 1) % FIELD_VOCAB) as u8; NUM_FIELDS] };
                }
            }
            // Internal padding must remain masked when trimming the block tail.
            seq[1].ttype = 0;
            seq
        }).collect()
    }

    fn model<B: Backend>(device: &B::Device) -> Net<B> {
        let mut net = Net::fresh(device);
        net.encoder = TransformerEncoderConfig::new(D_MODEL, 128, N_HEADS, 2)
            .with_dropout(0.0).with_norm_first(true).init(device);
        net
    }

    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        let max = a.iter().zip(b).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
        assert!(a.iter().chain(b).all(|x| x.is_finite()));
        assert!(max < 0.0001, "max absolute error {max}");
    }

    #[test]
    fn compact_matches_dense_outputs_and_full_capacity() {
        let d = NdArrayDevice::Cpu;
        let net = model::<NdArray>(&d);
        let mut seqs = fixture();
        assert!(BatchLayout::for_batch(&seqs).len() < SEQ_LEN);
        for full in [false, true] {
            if full {
                for t in &mut seqs[0] { t.ttype = 1; }
                assert_eq!(BatchLayout::for_batch(&seqs).len(), SEQ_LEN);
            }
            let (p, v) = forward(&net, &seqs, &d);
            let (old_p, old_v) = forward_layout(&net, &seqs, BatchLayout::dense(), &d);
            let valid = action_valid_mask::<NdArray>(&seqs, &d).float();
            close(&(p * valid.clone()).into_data().to_vec::<f32>().unwrap(),
                  &(old_p * valid).into_data().to_vec::<f32>().unwrap());
            close(&v.into_data().to_vec::<f32>().unwrap(), &old_v.into_data().to_vec::<f32>().unwrap());
        }
        assert_eq!(std::mem::size_of::<TokenSeq>(), SEQ_LEN * 8);
    }

    #[test]
    fn compact_matches_dense_gradients() {
        type B = Autodiff<NdArray>;
        let d = NdArrayDevice::Cpu;
        let net = model::<B>(&d);
        let seqs = fixture();
        let loss = |layout| {
            let (p, v) = forward_layout(&net, &seqs, layout, &d);
            let valid = action_valid_mask::<B>(&seqs, &d);
            let ce = -activation::log_softmax(p + valid.bool_not().float() * LOGIT_MASK, 1)
                .slice([0..2, 0..1]).mean();
            (ce + v.mean()).backward()
        };
        let dense = loss(BatchLayout::dense());
        let compact = loss(BatchLayout::for_batch(&seqs));
        for param in [&net.type_embed.weight, &net.encoder.layers[0].mha.query.weight,
                      &net.pointer_out.weight, &net.value_out.weight] {
            let a = param.val().grad(&dense).unwrap().into_data().to_vec::<f32>().unwrap();
            let b = param.val().grad(&compact).unwrap().into_data().to_vec::<f32>().unwrap();
            close(&a, &b);
        }
    }
}

#[cfg(test)]
mod diag_tests {
    use super::*;

    #[test]
    fn teacher_behind_two_actions_is_rank_three() {
        let mut diag = ValDiag::default();
        // target 0 is the worst of three legal logits.
        observe_row(&mut diag, &[1.0, 3.0, 2.0], 0, 0.4, 0.0, 12, 0);
        assert_eq!(diag.n, 1);
        assert_eq!(diag.top1, 0);
        assert_eq!(diag.top3, 1);
        assert_eq!(diag.top5, 1);
        assert!((diag.teacher_rank - 3.0).abs() < 1e-6);
        assert!((diag.teacher_margin - (1.0 - 3.0)).abs() < 1e-6);
        assert_eq!(diag.groups[1].n, 1); // ChainResponseCard
        assert_eq!(diag.groups[1].hit, 0);
        assert_eq!(diag.legal[1].n, 1); // 2-3
        assert_eq!(diag.history[0].n, 1);
        assert_eq!(diag.loser_n, 1);
        let p = diag.teacher_p;
        assert!(p > 0.05 && p < 0.15, "teacher probability {p}");
        let h = diag.entropy;
        assert!(h > 0.5 && h < 1.0, "normalized entropy {h}");
    }

    #[test]
    fn sole_top_action_is_rank_one() {
        let mut diag = ValDiag::default();
        observe_row(&mut diag, &[5.0, 0.0], 0, 0.8, 1.0, 1, 30);
        assert_eq!(diag.top1, 1);
        assert!((diag.teacher_rank - 1.0).abs() < 1e-6);
        assert!((diag.teacher_margin - 5.0).abs() < 1e-6);
        assert!(diag.teacher_p > 0.99);
        assert_eq!(diag.groups[0].hit, 1); // PlayStar
        assert_eq!(diag.winner_n, 1);
        assert_eq!(diag.history[3].n, 1); // 25-48
        assert!(diag.entropy < 0.1, "entropy {}", diag.entropy);
    }

    #[test]
    fn auc_of_a_known_ranking_is_three_quarters() {
        let scores = [0.1, 0.4, 0.35, 0.8];
        let positive = [false, false, true, true];
        let auc = roc_auc(&scores, &positive).unwrap();
        assert!((auc - 0.75).abs() < 1e-5, "{auc}");
    }

    #[test]
    fn auc_is_absent_when_one_class_is_missing() {
        assert!(roc_auc(&[0.2, 0.8], &[true, true]).is_none());
    }
}

#[cfg(test)]
mod bc_parallel_tests {
    use super::*;
    use burn::backend::Autodiff;
    use burn_ndarray::{NdArray, NdArrayDevice};

    fn tensor(v: [f32; 2]) -> Tensor<NdArray, 1> {
        Tensor::from_data(v, &NdArrayDevice::Cpu)
    }
    fn close(t: Tensor<NdArray, 1>, expected: [f32; 2]) {
        let actual = download_f32(t);
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn weighted_means_cover_equal_and_unequal_sequence_counts() {
        close(
            weighted_average(&[tensor([2., 4.]), tensor([6., 8.])], &[1, 1]),
            [4., 6.],
        );
        close(
            weighted_average(&[tensor([2., 4.]), tensor([6., 8.])], &[1, 3]),
            [5., 7.],
        );
    }
    #[test]
    #[should_panic(expected = "length mismatch")]
    fn weighted_mean_rejects_length_mismatch() {
        weighted_average(&[tensor([2., 4.]), tensor([6., 8.])], &[1]);
    }
    #[test]
    #[should_panic(expected = "positive")]
    fn weighted_mean_rejects_zero_count() {
        weighted_average(&[tensor([2., 4.]), tensor([6., 8.])], &[0, 1]);
    }
    #[test]
    #[should_panic(expected = "nonempty")]
    fn weighted_mean_rejects_empty_input() {
        weighted_average::<NdArray, 1>(&[], &[]);
    }
    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn weighted_mean_rejects_shape_mismatch() {
        weighted_average(
            &[
                tensor([2., 4.]),
                Tensor::<NdArray, 1>::zeros([3], &NdArrayDevice::Cpu),
            ],
            &[1, 1],
        );
    }

    #[test]
    fn freshly_forked_bc_models_have_identical_starting_weights() {
        type B = Autodiff<NdArray>;
        let device = NdArrayDevice::Cpu;
        let net = Net::<B>::fresh_with_dropout(&device, 0.0);
        // Call the production constructor before any other access to weights.
        let (replica, pairs) = fork_bc_replica(&net, &device).unwrap();
        let (reference, _) = fork_bc_replica(&net, &device).unwrap();
        check_pair_ids(&net, &replica, &pairs).unwrap();
        let a = download_f32(net.value_hidden.weight.val().inner());
        let b = download_f32(replica.value_hidden.weight.val().inner());
        let c = download_f32(reference.value_hidden.weight.val().inner());
        assert_eq!(a.len(), b.len());
        let replica_diff = a.iter().zip(&b).map(|(a, b)| (a-b).abs()).fold(0.0f32, f32::max);
        let reference_diff = a.iter().zip(&c).map(|(a, b)| (a-b).abs()).fold(0.0f32, f32::max);
        assert_eq!(replica_diff, 0.0, "replica weights differ before any forward");
        assert_eq!(reference_diff, 0.0, "reference weights differ before any forward");
    }

    #[test]
    fn cpu_split_gradients_match_the_unchanged_dense_parity_check() {
        bc_parity::<Autodiff<NdArray>>(&[NdArrayDevice::Cpu, NdArrayDevice::Cpu]).unwrap();
    }

    #[test]
    fn full_model_paths_shapes_and_ids_pair_without_id_deduplication() {
        type B = Autodiff<NdArray>;
        let d = NdArrayDevice::Cpu;
        let net = Net::<B>::fresh(&d);
        let a = float_param_slots(&net);
        assert_eq!(a.len(), BC_PARAM_TENSORS);
        assert_eq!(a.iter().map(|s| s.numel).sum::<usize>(), BC_PARAM_ELEMENTS);
        assert_eq!(
            a.iter()
                .map(|s| &s.path)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            BC_PARAM_TENSORS
        );
        assert!(a.iter().all(|s| matches!(s.rank, 1 | 2)));
        for path in [
            "field_embed.0.weight",
            "encoder.layers.0.mha.query.weight",
            "encoder.layers.0.norm_1.gamma",
            "pointer_out.bias",
            "value_out.weight",
        ] {
            assert!(a.iter().any(|s| s.path == path), "missing {path}");
        }
        let fresh = float_param_slots(&Net::<B>::fresh(&d));
        assert!(pair_slots(&a, &fresh)
            .unwrap()
            .iter()
            .all(|s| s.id0 != s.id1));
        let fork = net.clone().fork(&d);
        let b = float_param_slots(&fork);
        assert!(pair_slots(&a, &b).unwrap().iter().all(|s| s.id0 == s.id1));
        check_pair_ids(&net, &fork, &pair_slots(&a, &b).unwrap()).unwrap();
    }

    fn synthetic_slots() -> Vec<ParamSlot> {
        (0..BC_PARAM_TENSORS)
            .map(|i| {
                let n = if i == 0 {
                    BC_PARAM_ELEMENTS - (BC_PARAM_TENSORS - 1)
                } else {
                    1
                };
                ParamSlot {
                    path: format!("test.{i}"),
                    id: ParamId::new(),
                    rank: 1,
                    numel: n,
                    dims: vec![n],
                }
            })
            .collect()
    }
    #[test]
    fn pairing_rejects_paths_tail_dimensions_and_non_float_slots() {
        let a = synthetic_slots();
        pair_slots(&a, &a).unwrap();
        let mut b = a.clone();
        b.swap(0, 1);
        assert!(pair_slots(&a, &b).unwrap_err().0.contains("path mismatch"));
        assert!(pair_slots(&a, &a[..a.len() - 1]).is_err());
        let mut b = a.clone();
        b[0].dims[0] += 1;
        assert!(pair_slots(&a, &b).unwrap_err().0.contains("dimension"));
        let mut b = a.clone();
        b[0].rank = 0;
        assert!(pair_slots(&b, &b).is_err());
        let mut b = a.clone();
        b[1].path = b[0].path.clone();
        assert!(pair_slots(&b, &b).unwrap_err().0.contains("duplicate"));
        let mut b = a.clone();
        b[0].numel += 1;
        b[0].dims[0] += 1;
        assert!(pair_slots(&b, &b).unwrap_err().0.contains("elements"));
    }

    #[test]
    fn same_id_in_separate_containers_is_averaged_but_one_map_replaces() {
        let id = ParamId::new();
        let slot = PairedSlot {
            path: "test.bias".into(),
            id0: id,
            id1: id,
            rank: 1,
            dims: vec![2],
        };
        let mut a = GradientsParams::new();
        let mut b = GradientsParams::new();
        a.register(id, tensor([2., 4.]));
        b.register(id, tensor([6., 8.]));
        let avg = average_gradients::<NdArray>(&[slot], a, b, [1, 3], &NdArrayDevice::Cpu).unwrap();
        assert_eq!(avg.len(), 1);
        close(avg.get(id).unwrap(), [5., 7.]);
        let mut single = GradientsParams::new();
        single.register(id, tensor([2., 4.]));
        single.register(id, tensor([6., 8.]));
        assert_eq!(single.len(), 1);
        close(single.get(id).unwrap(), [6., 8.]);
    }

    #[test]
    fn missing_gradient_is_an_error_with_its_module_path() {
        for missing_device in [0, 1] {
            let id = ParamId::new();
            let slot = PairedSlot {
                path: "test.bias".into(),
                id0: id,
                id1: id,
                rank: 1,
                dims: vec![2],
            };
            let mut a = GradientsParams::new();
            let mut b = GradientsParams::new();
            if missing_device != 0 {
                a.register(id, tensor([2., 4.]));
            }
            if missing_device != 1 {
                b.register(id, tensor([6., 8.]));
            }
            let e = average_gradients::<NdArray>(&[slot], a, b, [1, 1], &NdArrayDevice::Cpu)
                .unwrap_err();
            assert!(e.0.contains("test.bias"));
            assert!(e.0.contains(&format!("device-{missing_device}")));
        }
    }

    #[test]
    fn rank_two_gradients_use_their_declared_rank() {
        let id0 = ParamId::new();
        let id1 = ParamId::new();
        let slot = PairedSlot {
            path: "test.weight".into(),
            id0,
            id1,
            rank: 2,
            dims: vec![1, 2],
        };
        let mut a = GradientsParams::new();
        let mut b = GradientsParams::new();
        a.register(id0, tensor([2., 4.]).reshape([1, 2]));
        b.register(id1, tensor([6., 8.]).reshape([1, 2]));
        let avg = average_gradients::<NdArray>(&[slot], a, b, [1, 3], &NdArrayDevice::Cpu).unwrap();
        close(avg.get::<NdArray, 2>(id0).unwrap().reshape([2]), [5., 7.]);
    }

    #[test]
    fn broadcast_preserves_ids_weights_and_trainability_across_steps() {
        type B = Autodiff<NdArray>;
        let d = NdArrayDevice::Cpu;
        let mut net = Net::<B>::fresh_with_dropout(&d, 0.0);
        let (mut replica, pairs) = fork_bc_replica(&net, &d).unwrap();
        let mut opt = make_optimizer::<B>();
        use crate::tokens::{Token, SEQ_LEN};
        let mut seq = [Token {
            ttype: 0,
            fields: [0; NUM_FIELDS],
        }; SEQ_LEN];
        seq[0] = Token {
            ttype: 1,
            fields: [1; NUM_FIELDS],
        };
        seq[STATE_MAX + HISTORY_MAX] = Token {
            ttype: 7,
            fields: [2; NUM_FIELDS],
        };
        seq[STATE_MAX + HISTORY_MAX + 1] = Token {
            ttype: 7,
            fields: [3; NUM_FIELDS],
        };
        let seqs = [seq; 2];
        for metrics in [true, false] {
            let old = download_f32(net.pointer_out.weight.val().inner());
            let (nn, rr, pl, vl) = bc_two_device_step(
                net,
                replica,
                &pairs,
                &mut opt,
                3e-4,
                [&seqs, &seqs],
                [&[0, 1], &[1, 0]],
                [&[0., 1.], &[1., 0.]],
                0.05,
                metrics,
                &[d.clone(), d.clone()],
                &BcPhases::default(),
            )
            .unwrap();
            net = nn;
            replica = rr;
            check_pair_ids(&net, &replica, &pairs).unwrap();
            assert_eq!(pl.is_some(), metrics);
            assert_eq!(vl.is_some(), metrics);
            if let Some(pl) = pl {
                assert!(pl.is_finite());
            }
            if let Some(vl) = vl {
                assert!(vl.is_finite());
            }
            assert!(net.pointer_out.weight.val().is_require_grad());
            assert!(replica.pointer_out.weight.val().is_require_grad());
            let now = download_f32(net.pointer_out.weight.val().inner());
            assert!(
                old.iter().zip(&now).any(|(a, b)| a != b),
                "Adam must update a live weight"
            );
            assert_eq!(now, download_f32(replica.pointer_out.weight.val().inner()));
            assert_eq!(
                download_f32(net.type_embed.weight.val().inner()),
                download_f32(replica.type_embed.weight.val().inner())
            );
        }
    }
}

#[cfg(all(test, feature = "tch"))]
mod cuda_bc_parity_tests {
    use super::*;
    #[test]
    fn two_cuda_devices_match_global_batch_gradients() {
        if tch::Cuda::device_count() < 2 {
            return;
        }
        bc_parity::<burn::backend::Autodiff<burn_tch::LibTorch<f32>>>(&[
            burn_tch::LibTorchDevice::Cuda(0),
            burn_tch::LibTorchDevice::Cuda(1),
        ])
        .unwrap();
    }
}


#[cfg(all(test, feature = "vulkan", not(feature = "tch")))]
mod vulkan_bc_regression {
    use super::*;
    use burn::backend::Autodiff;
    use burn::record::{BinFileRecorder, FullPrecisionSettings};
    use burn_wgpu::Wgpu;
    fn old_bc_step<B: AutodiffBackend, O: Optimizer<Net<B>, B>>(
        net: Net<B>,
        optim: &mut O,
        lr: f64,
        seqs: &[TokenSeq],
        targets: &[u16],
        outcomes: &[f32],
        label_smoothing: f32,
        report_metrics: bool,
        layout: BatchLayout,
        device: &B::Device,
    ) -> (Net<B>, Option<f32>, Option<f32>) {
        let b = seqs.len();
        assert!(b > 0);

        let (logits, values) = forward_layout(&net, seqs, layout, device);
        let valid = action_valid_mask::<B>(seqs, device); // true = legal
        let neg_inf = valid.clone().bool_not().float() * LOGIT_MASK;
        let masked_logits = logits + neg_inf;
        let log_p = activation::log_softmax(masked_logits, 1);

        let act_t = Tensor::<B, 2, Int>::from_data(
            TensorData::new(
                targets.iter().map(|&a| a as i32).collect::<Vec<_>>(),
                [b, 1],
            ),
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

        // Reading a scalar back from CUDA synchronizes the device. Training only
        // needs these diagnostics at heartbeat intervals, so avoid two device
        // barriers on every optimizer step.
        let (pl, vl) = if report_metrics {
            (
                Some(
                    download_f32(hard.mean())
                        .first()
                        .copied()
                        .unwrap_or(f32::NAN),
                ),
                Some(
                    download_f32(value_loss.clone())
                        .first()
                        .copied()
                        .unwrap_or(f32::NAN),
                ),
            )
        } else {
            (None, None)
        };

        // ASTRA7.md §10: value target is only a terminal-outcome warm start (the
        // real value signal comes from TD-value/AZ later) - down-weight so it
        // doesn't distract the policy gradient, which is what BC is actually for.
        const VALUE_LOSS_WEIGHT: f32 = 0.3;
        let total = policy_loss + value_loss * VALUE_LOSS_WEIGHT;
        let grads = GradientsParams::from_grads(total.backward(), &net);
        (optim.step(lr, net, grads), pl, vl)
    }

    struct TestFiles(std::path::PathBuf);
    impl Drop for TestFiles {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn heartbeat_wrapper_matches_original_weights_metrics_and_checkpoint_bytes() {
        type B = Autodiff<Wgpu<f32, i32>>;
        let device = crate::vulkan_device();
        B::seed(&device, 42);
        let mut initial = Net::<B>::fresh_with_dropout(&device, 0.0);
        initial.encoder = TransformerEncoderConfig::new(D_MODEL, 128, N_HEADS, 2)
            .with_dropout(0.0)
            .with_norm_first(true)
            .init(&device);
        // Reload twice to ensure distinct GPU storage even on the same device.
        // A same-device fork detaches graphs but may still share weight storage.
        let root = std::env::temp_dir().join(format!("novacana-vulkan-bc-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let _files = TestFiles(root.clone());
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        initial
            .clone()
            .save_file(root.join("initial"), &recorder)
            .unwrap();
        let mut current = initial
            .clone()
            .load_file(root.join("initial"), &recorder, &device)
            .unwrap()
            .fork(&device);
        let mut previous = initial
            .load_file(root.join("initial"), &recorder, &device)
            .unwrap()
            .fork(&device);
        let mut new_opt = make_optimizer::<B>();
        let mut old_opt = make_optimizer::<B>();
        use crate::tokens::{Token, SEQ_LEN};
        let mut seq = [Token {
            ttype: 0,
            fields: [0; NUM_FIELDS],
        }; SEQ_LEN];
        seq[0] = Token {
            ttype: 1,
            fields: [1; NUM_FIELDS],
        };
        seq[STATE_MAX + HISTORY_MAX] = Token {
            ttype: 7,
            fields: [2; NUM_FIELDS],
        };
        seq[STATE_MAX + HISTORY_MAX + 1] = Token {
            ttype: 7,
            fields: [3; NUM_FIELDS],
        };
        let seqs = [seq; 2];
        for metrics in [true, false, true] {
            let (nn, np, nv) = bc_train_step(
                current,
                &mut new_opt,
                3e-4,
                &seqs,
                &[0, 1],
                &[0., 1.],
                0.05,
                metrics,
                &device,
            );
            let (oo, op, ov) = old_bc_step(
                previous,
                &mut old_opt,
                3e-4,
                &seqs,
                &[0, 1],
                &[0., 1.],
                0.05,
                metrics,
                BatchLayout::for_batch(&seqs),
                &device,
            );
            current = nn;
            previous = oo;
            assert_eq!(np, op, "policy metric differs, report_metrics={metrics}");
            assert_eq!(nv, ov, "value metric differs, report_metrics={metrics}");
            current
                .clone()
                .save_file(root.join("new"), &recorder)
                .unwrap();
            previous
                .clone()
                .save_file(root.join("old"), &recorder)
                .unwrap();
            let a = std::fs::read(root.join("new.bin")).unwrap();
            let b = std::fs::read(root.join("old.bin")).unwrap();
            assert_eq!(a.len(), b.len());
            assert!(
                a == b,
                "single-device checkpoint bytes differ, metrics={metrics}"
            );
        }
    }
}
