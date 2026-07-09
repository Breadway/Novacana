// 486→512→512(+res)→512(+res)→512→{31,1}  +  candidate head [512+24→128→1]
// Input = 3 stacked GameView snapshots × 162 features each.
use ndarray::{Array1, Array2};
use ndarray_rand::{rand_distr::Normal, RandomExt};
use serde::{Deserialize, Serialize};

use crate::action::NUM_ACTIONS;
use crate::candidate::CAND_DIM;
use crate::feature::FEATURE_DIM;

pub const INPUT_DIM: usize = FEATURE_DIM * 3; // 486

const H1: usize = 512;
const H2: usize = 512;
const H3: usize = 512;
const H4: usize = 512;
const HC: usize = 128;

// ── Flat-weight serialisation ─────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone)]
pub struct Weights {
    pub w1: Vec<f32>, pub b1: Vec<f32>,
    pub w2: Vec<f32>, pub b2: Vec<f32>,
    pub w3: Vec<f32>, pub b3: Vec<f32>,
    pub w4: Vec<f32>, pub b4: Vec<f32>,
    pub wp: Vec<f32>, pub bp: Vec<f32>,
    pub wv: Vec<f32>, pub bv: Vec<f32>,
    pub wc1: Vec<f32>, pub bc1: Vec<f32>,
    pub wc2: Vec<f32>, pub bc2: Vec<f32>,
}

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

// ── Network (inference only; training lives in net_burn.rs) ──────────────────

pub struct Net {
    l1: Layer,
    l2: Layer,
    l3: Layer,
    l4: Layer,
    lp: Layer,
    lv: Layer,
    lc1: Layer,
    lc2: Layer,
}

impl Net {
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

    pub fn from_weights(w: &Weights) -> Self {
        let load2 = |data: &[f32], rows: usize, cols: usize| {
            Array2::from_shape_vec((rows, cols), data.to_vec()).unwrap()
        };
        let load1 = |data: &[f32]| Array1::from_vec(data.to_vec());
        Self {
            l1:  Layer { w: load2(&w.w1,  H1, INPUT_DIM),      b: load1(&w.b1)  },
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

    pub fn body_forward(&self, features: &[f32; INPUT_DIM]) -> Array1<f32> {
        let x  = ndarray::aview1(features);
        let a1 = relu1(self.l1.forward_single(x));
        let a2 = relu1(self.l2.forward_single(a1.view()) + &a1);
        let a3 = relu1(self.l3.forward_single(a2.view()) + &a2);
        relu1(self.l4.forward_single(a3.view()))
    }

    pub fn body_forward_batch(&self, features: &[[f32; INPUT_DIM]]) -> Array2<f32> {
        let b    = features.len();
        let flat: Vec<f32> = features.iter().flat_map(|f| f.iter().copied()).collect();
        let x    = Array2::from_shape_vec((b, INPUT_DIM), flat).unwrap();
        let a1   = relu2(self.l1.forward(&x).view());
        let a2   = relu2((self.l2.forward(&a1) + &a1).view());
        let a3   = relu2((self.l3.forward(&a2) + &a2).view());
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
        features:    &[f32; INPUT_DIM],
        mask:        u64,
        temperature: f32,
        u:           f32,
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

// ── Helpers ───────────────────────────────────────────────────────────────────

fn relu1(x: Array1<f32>) -> Array1<f32> { x.mapv(|v| v.max(0.0)) }
fn relu2(x: ndarray::ArrayView2<f32>) -> Array2<f32> { x.mapv(|v| v.max(0.0)) }
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

pub fn masked_softmax(logits: &[f32], mask: u64) -> Vec<f32> {
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

pub fn save_weights(w: &Weights, path: &str) -> std::io::Result<()> {
    let data = bincode::serialize(w).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(path, &data)
}

pub fn load_weights(path: &str) -> std::io::Result<Weights> {
    let data = std::fs::read(path)?;
    bincode::deserialize(&data).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}
