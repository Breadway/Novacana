use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::generate::{CandidateSample, Sample};
use crate::net::{Batch, CandidateSampleBatch, Net, Optimizer, ValueBatch,
                  candidate_train_step, train_step, value_pretrain_step, Weights};

pub struct TrainConfig {
    pub epochs:     usize,
    pub batch_size: usize,
    pub lr:         f32,
    pub out_path:   String,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs:     50,
            batch_size: 256,
            lr:         1e-3,
            out_path:   "models/net.bin".into(),
        }
    }
}

pub fn train(samples: &[Sample], cfg: &TrainConfig) -> std::io::Result<Net> {
    std::fs::create_dir_all(std::path::Path::new(&cfg.out_path).parent().unwrap_or(std::path::Path::new(".")))?;

    let mut net = Net::new();
    let mut opt = Optimizer::new(cfg.lr);
    let mut indices: Vec<usize> = (0..samples.len()).collect();
    let mut rng = thread_rng();

    println!("Training on {} samples  |  epochs={}  batch={}  lr={}",
        samples.len(), cfg.epochs, cfg.batch_size, cfg.lr);

    for epoch in 0..cfg.epochs {
        indices.shuffle(&mut rng);
        let mut epoch_policy = 0.0f64;
        let mut epoch_value  = 0.0f64;
        let mut num_batches  = 0usize;

        for chunk in indices.chunks(cfg.batch_size) {
            if chunk.len() < 2 { continue; } // skip tiny tail

            // Gather batch arrays.
            let mut features: Vec<Vec<f32>>           = Vec::with_capacity(chunk.len());
            let mut masks:    Vec<u64>                = Vec::with_capacity(chunk.len());
            let mut actions:  Vec<u8>                 = Vec::with_capacity(chunk.len());
            let mut outcomes: Vec<f32>                = Vec::with_capacity(chunk.len());

            for &i in chunk {
                let s = &samples[i];
                // Verify the recorded action is in the legal mask; skip if not.
                if s.legal_mask & (1u64 << s.action) == 0 { continue; }
                features.push(s.features.clone());
                masks.push(s.legal_mask);
                actions.push(s.action);
                outcomes.push(s.outcome);
            }
            if features.is_empty() { continue; }

            let batch = Batch { features: &features, masks: &masks, actions: &actions, outcomes: &outcomes };
            let (pl, vl) = train_step(&mut net, &mut opt, &batch);
            epoch_policy += pl as f64;
            epoch_value  += vl as f64;
            num_batches  += 1;
        }

        if num_batches > 0 {
            println!(
                "Epoch {:3}  policy_loss={:.4}  value_loss={:.4}",
                epoch + 1,
                epoch_policy / num_batches as f64,
                epoch_value  / num_batches as f64,
            );
        }

        // Save checkpoint after every epoch.
        save_weights(&net.to_weights(), &cfg.out_path)?;
    }

    println!("Model saved → {}", cfg.out_path);
    Ok(net)
}

pub fn pretrain_value(samples: &[Sample], cfg: &TrainConfig) -> std::io::Result<Net> {
    let weights = load_weights(&cfg.out_path)?;
    let mut net = Net::from_weights(&weights);
    let mut opt = Optimizer::new(cfg.lr);
    let mut indices: Vec<usize> = (0..samples.len()).collect();
    let mut rng = thread_rng();

    println!("Value pretraining on {} samples  |  epochs={}  batch={}  lr={}",
        samples.len(), cfg.epochs, cfg.batch_size, cfg.lr);

    for epoch in 0..cfg.epochs {
        indices.shuffle(&mut rng);
        let mut epoch_value = 0.0f64;
        let mut num_batches = 0usize;

        for chunk in indices.chunks(cfg.batch_size) {
            if chunk.len() < 2 { continue; }
            let mut features: Vec<Vec<f32>> = Vec::with_capacity(chunk.len());
            let mut outcomes: Vec<f32>      = Vec::with_capacity(chunk.len());
            for &i in chunk {
                features.push(samples[i].features.clone());
                outcomes.push(samples[i].outcome);
            }
            if features.is_empty() { continue; }
            let batch = ValueBatch { features: &features, outcomes: &outcomes };
            let vl = value_pretrain_step(&mut net, &mut opt, &batch);
            epoch_value += vl as f64;
            num_batches += 1;
        }

        if num_batches > 0 {
            println!("Epoch {:3}  value_loss={:.4}", epoch + 1, epoch_value / num_batches as f64);
        }
        save_weights(&net.to_weights(), &cfg.out_path)?;
    }

    println!("Value pretraining complete → {}", cfg.out_path);
    Ok(net)
}

pub fn pretrain_candidates(samples: &[CandidateSample], cfg: &TrainConfig) -> std::io::Result<Net> {
    let weights = load_weights(&cfg.out_path)?;
    let mut net = Net::from_weights(&weights);
    let mut opt = Optimizer::new(cfg.lr);
    let mut indices: Vec<usize> = (0..samples.len()).collect();
    let mut rng = thread_rng();

    println!("Candidate pretraining on {} samples  |  epochs={}  batch={}  lr={}",
        samples.len(), cfg.epochs, cfg.batch_size, cfg.lr);

    for epoch in 0..cfg.epochs {
        indices.shuffle(&mut rng);
        let mut epoch_loss  = 0.0f64;
        let mut num_batches = 0usize;

        for chunk in indices.chunks(cfg.batch_size) {
            if chunk.len() < 2 { continue; }
            let mut features:    Vec<Vec<f32>>         = Vec::with_capacity(chunk.len());
            let mut candidates:  Vec<Vec<[f32; crate::candidate::CAND_DIM]>> = Vec::with_capacity(chunk.len());
            let mut chosen_idxs: Vec<u8>               = Vec::with_capacity(chunk.len());
            for &i in chunk {
                let s = &samples[i];
                if s.candidates.is_empty() { continue; }
                if s.chosen_idx as usize >= s.candidates.len() { continue; }
                features.push(s.features.clone());
                candidates.push(s.candidates.clone());
                chosen_idxs.push(s.chosen_idx);
            }
            if features.is_empty() { continue; }

            let batch = CandidateSampleBatch {
                features:    &features,
                candidates:  &candidates,
                chosen_idxs: &chosen_idxs,
            };
            let loss = candidate_train_step(&mut net, &mut opt, &batch);
            epoch_loss  += loss as f64;
            num_batches += 1;
        }

        if num_batches > 0 {
            println!("Epoch {:3}  cand_loss={:.4}", epoch + 1, epoch_loss / num_batches as f64);
        }
        save_weights(&net.to_weights(), &cfg.out_path)?;
    }

    println!("Candidate pretraining complete → {}", cfg.out_path);
    Ok(net)
}

pub fn save_weights(w: &Weights, path: &str) -> std::io::Result<()> {
    let data = bincode::serialize(w).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(path, data)
}

pub fn load_weights(path: &str) -> std::io::Result<Weights> {
    let data = std::fs::read(path)?;
    bincode::deserialize(&data).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}
