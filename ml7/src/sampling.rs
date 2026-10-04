// Shared action-sampling helpers for the two self-play stages that read
// straight off the net's pointer logits without MCTS: TD-value's temperature
// sampler (ASTRA7.md §4.2) and AZ's decide/pick_targets/choose_discards
// decisions (mcts.rs's module doc explains why those three aren't searched).

use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;

/// Temperature-softmax sample over the first `n_legal` masked logits.
/// `temperature` <= ~1e-3 is treated as greedy (argmax) - matches
/// mcts.rs::finalize's convention so callers can share one "is this greedy"
/// threshold across both search and non-search decision kinds.
pub fn sample_temperature(masked_logits: &[f32], n_legal: usize, temperature: f32, rng: &mut impl Rng) -> usize {
    let logits = &masked_logits[..n_legal];
    if temperature <= 1e-3 {
        return argmax(logits);
    }
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f64> = logits.iter().map(|&x| (((x - m) / temperature) as f64).exp()).collect();
    match WeightedIndex::new(&weights) {
        Ok(dist) => dist.sample(rng),
        Err(_) => argmax(logits),
    }
}

pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Greedy top-k indices by raw logit value, for multi-pick decisions
/// (pick_targets count>1, multi-discard) - ASTRA7.md §3.2/§9: "start
/// greedy", the same simplification the BC recorder's supervision already
/// assumes (tokens.rs only ever labels the FIRST pick). One forward pass,
/// no autoregressive re-encoding.
pub fn top_k_indices(masked_logits: &[f32], n_legal: usize, k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n_legal).collect();
    idx.sort_by(|&a, &b| masked_logits[b].partial_cmp(&masked_logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(k.max(1).min(n_legal.max(1)));
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_temperature_matches_argmax() {
        let logits = vec![0.1, 5.0, -3.0, 2.0];
        let mut rng = rand::thread_rng();
        assert_eq!(sample_temperature(&logits, 4, 0.0, &mut rng), 1);
    }

    #[test]
    fn top_k_picks_highest_logits_in_order() {
        let logits = vec![1.0, 9.0, 3.0, 7.0, 0.0];
        let top2 = top_k_indices(&logits, 5, 2);
        assert_eq!(top2, vec![1, 3]);
    }

    #[test]
    fn top_k_clamped_to_n_legal() {
        let logits = vec![1.0, 2.0];
        let top = top_k_indices(&logits, 2, 5);
        assert_eq!(top.len(), 2);
    }
}
