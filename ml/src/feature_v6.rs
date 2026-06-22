use novacana_engine::{
    card::{CardKind, FlareName},
    player::GameView,
};

use crate::feature::{encode, FEATURE_DIM};

pub const FEATURE_DIM_V6: usize = 162;

// New features appended after the existing 104:
//   [104..124] Play history: 4 player seats × 5 card categories, count in recent_plays /5
//   [124..133] Threat flags: 3 opponent slots × 3 (has_played_flare, fracture, supernova)
//   [133..162] Cumulative play counts: 29 card types from view.discard_pile_counts /3

fn card_category(kind: &CardKind) -> usize {
    match kind {
        CardKind::Star(_)      => 0,
        CardKind::Discovery(_) => 1,
        CardKind::Flare(_)     => 2,
        CardKind::Fracture(_)  => 3,
        CardKind::Omen(_)      => 4,
    }
}

pub fn encode_v6(view: &GameView) -> [f32; FEATURE_DIM_V6] {
    let mut feat = [0.0f32; FEATURE_DIM_V6];

    // Copy the 104 base features.
    let base = encode(view);
    feat[..FEATURE_DIM].copy_from_slice(&base);

    // [104..124] Play history: per-player-seat × per-category counts.
    // Categories: 0=Star, 1=Discovery, 2=Flare, 3=Fracture, 4=Omen.
    // Counts over the whole recent_plays window (up to 20 entries), normalised /5.
    let mut history_counts = [[0u8; 5]; 4];
    for &(player_idx, ref kind) in &view.recent_plays {
        if player_idx < 4 {
            let cat = card_category(kind);
            history_counts[player_idx][cat] = history_counts[player_idx][cat].saturating_add(1);
        }
    }
    for seat in 0..4 {
        for cat in 0..5 {
            feat[104 + seat * 5 + cat] = history_counts[seat][cat] as f32 / 5.0;
        }
    }

    // [124..133] Threat flags for each opponent slot (up to 3).
    // For each opponent, scan recent_plays to check if that opponent played any
    // Flare, any Fracture, or any Supernova this game.
    for (slot, opp) in view.opponents.iter().enumerate().take(3) {
        let opp_idx = opp.idx;
        let mut has_flare    = false;
        let mut has_fracture = false;
        let mut has_supernova = false;
        for &(player_idx, ref kind) in &view.recent_plays {
            if player_idx != opp_idx { continue; }
            match kind {
                CardKind::Flare(f) => {
                    has_flare = true;
                    if f.is_supernova() { has_supernova = true; }
                }
                CardKind::Fracture(_) => has_fracture = true,
                _ => {}
            }
        }
        feat[124 + slot * 3]     = if has_flare     { 1.0 } else { 0.0 };
        feat[124 + slot * 3 + 1] = if has_fracture  { 1.0 } else { 0.0 };
        feat[124 + slot * 3 + 2] = if has_supernova { 1.0 } else { 0.0 };
    }

    // [133..162] Cumulative play counts per card type, normalised /3.
    for (i, &count) in view.discard_pile_counts.iter().enumerate() {
        feat[133 + i] = count as f32 / 3.0;
    }

    feat
}
