use novacana_engine::{
    card::{Card, CardKind, StarName},
    player::{GameView, Target, TargetOption},
    star_chart::SolarSystem,
};

use crate::feature::card_type_idx;

pub const CAND_DIM: usize = 24;

// ── Candidate encoding layout ─────────────────────────────────────────────────
//
//  [0]  is_hand_card     — candidate is a card in hand (discard or action disambiguation)
//  [1]  is_system        — candidate is a star chart system slot
//  [2]  is_player        — candidate is an opponent player target
//  [3]  is_star_class    — candidate is a star-class value (Dark Matter Void targeting)
//  [4]  is_chain_link    — candidate is a chain link (Cosmic Denial targeting)
//
//  card features (when is_hand_card == 1)
//  [5]  card_type / 28.0
//  [6]  is_star_card
//  [7]  is_discovery_card
//  [8]  is_flare_card
//  [9]  is_fracture_card
//  [10] is_omen_card
//  [11] card_star_class / 6.0  (0 if card is not a star)
//
//  system features (when is_system == 1, or target is a StarChartCard)
//  [12] star_name_idx / 6.0
//  [13] star_class / 6.0
//  [14] fill_fraction (discoveries.len / class)
//  [15] would_complete (1 if choosing this completes the system)
//  [16] is_eclipse_protected
//
//  player features (when is_player == 1)
//  [17] player_rel_idx / 3.0
//  [18] opp_complete_systems / 3.0
//  [19] opp_approaching_win (1 if 2+ complete)
//
//  shared value
//  [20] star_class_val / 6.0  (if is_star_class, the class value)
//  [21] is_on_opponent_chart  (1 if target is on an opponent's chart)
//
//  positional
//  [22] option_idx / 10.0
//  [23] option_count / 10.0

fn star_name_to_idx(name: StarName) -> usize {
    match name {
        StarName::Protostar     => 0,
        StarName::YellowDwarf   => 1,
        StarName::RedGiant      => 2,
        StarName::RedSupergiant => 3,
        StarName::WhiteDwarf    => 4,
        StarName::RedDwarf      => 5,
        StarName::BrownDwarf    => 6,
    }
}

fn set_pos(c: &mut [f32; CAND_DIM], slot: usize, total: usize) {
    c[22] = slot as f32 / (total as f32).max(1.0);
    c[23] = total as f32 / 10.0;
}

// ── Encode a card in hand (for choose_discards or concrete-action disambiguation)

pub fn encode_card_candidate(card: &Card, slot: usize, total: usize) -> [f32; CAND_DIM] {
    let mut c = [0.0f32; CAND_DIM];
    c[0] = 1.0;
    c[5] = card_type_idx(&card.kind) as f32 / 28.0;
    c[6]  = if matches!(&card.kind, CardKind::Star(_))      { 1.0 } else { 0.0 };
    c[7]  = if matches!(&card.kind, CardKind::Discovery(_)) { 1.0 } else { 0.0 };
    c[8]  = if matches!(&card.kind, CardKind::Flare(_))     { 1.0 } else { 0.0 };
    c[9]  = if matches!(&card.kind, CardKind::Fracture(_))  { 1.0 } else { 0.0 };
    c[10] = if matches!(&card.kind, CardKind::Omen(_))      { 1.0 } else { 0.0 };
    c[11] = card.star_class().unwrap_or(0) as f32 / 6.0;
    set_pos(&mut c, slot, total);
    c
}

// ── Encode a star chart system (for concrete-action disambiguation: which star to fill)

pub fn encode_system_candidate(sys: &SolarSystem, on_opponent: bool, slot: usize, total: usize) -> [f32; CAND_DIM] {
    let mut c = [0.0f32; CAND_DIM];
    c[1]  = 1.0;
    c[12] = star_name_to_idx(sys.star_name) as f32 / 6.0;
    c[13] = sys.class as f32 / 6.0;
    c[14] = sys.discoveries.len() as f32 / sys.class as f32;
    c[15] = if sys.discoveries.len() + 1 >= sys.class as usize { 1.0 } else { 0.0 };
    c[16] = if sys.is_eclipse_protected() { 1.0 } else { 0.0 };
    c[21] = if on_opponent { 1.0 } else { 0.0 };
    set_pos(&mut c, slot, total);
    c
}

// ── Encode a TargetOption (for pick_targets)

pub fn encode_target_candidate(
    opt:   &TargetOption,
    view:  &GameView,
    slot:  usize,
    total: usize,
) -> [f32; CAND_DIM] {
    let mut c = [0.0f32; CAND_DIM];

    match &opt.target {
        Target::Player(idx) => {
            c[2] = 1.0;
            c[17] = *idx as f32 / 3.0;
            if let Some(opp) = view.opponents.iter().find(|o| o.idx == *idx) {
                let nc = opp.star_chart.complete_systems().len();
                c[18] = nc as f32 / 3.0;
                c[19] = if nc >= 2 { 1.0 } else { 0.0 };
            }
        }

        Target::StarChartCard { player_idx, card_id } => {
            c[1] = 1.0;
            let on_opp = *player_idx != view.my_idx;
            c[21] = if on_opp { 1.0 } else { 0.0 };
            let chart = if !on_opp {
                Some(&view.my_star_chart)
            } else {
                view.opponents.iter().find(|o| o.idx == *player_idx).map(|o| &o.star_chart)
            };
            if let Some(ch) = chart {
                // Find the system that contains this card (as star or discovery)
                let sys = ch.systems.iter().find(|s| {
                    s.star_id == *card_id
                    || s.discoveries.iter().any(|d| d.card_id == *card_id)
                });
                if let Some(sys) = sys {
                    c[12] = star_name_to_idx(sys.star_name) as f32 / 6.0;
                    c[13] = sys.class as f32 / 6.0;
                    c[14] = sys.discoveries.len() as f32 / sys.class as f32;
                    c[15] = if sys.is_complete() { 1.0 } else { 0.0 };
                    c[16] = if sys.is_eclipse_protected() { 1.0 } else { 0.0 };
                }
            }
        }

        Target::HandCard { player_idx, card_id } => {
            c[0] = 1.0; // treat as a card
            c[21] = if *player_idx != view.my_idx { 1.0 } else { 0.0 };
            let card = if *player_idx == view.my_idx {
                view.my_hand.iter().find(|c| c.id == *card_id)
            } else {
                view.opponents.iter()
                    .find(|o| o.idx == *player_idx)
                    .and_then(|o| o.hand_revealed.iter().find(|c| c.id == *card_id))
            };
            if let Some(card) = card {
                c[5]  = card_type_idx(&card.kind) as f32 / 28.0;
                c[6]  = if matches!(&card.kind, CardKind::Star(_))      { 1.0 } else { 0.0 };
                c[7]  = if matches!(&card.kind, CardKind::Discovery(_)) { 1.0 } else { 0.0 };
                c[8]  = if matches!(&card.kind, CardKind::Flare(_))     { 1.0 } else { 0.0 };
                c[9]  = if matches!(&card.kind, CardKind::Fracture(_))  { 1.0 } else { 0.0 };
                c[10] = if matches!(&card.kind, CardKind::Omen(_))      { 1.0 } else { 0.0 };
            }
        }

        Target::StarClass(cls) => {
            c[3]  = 1.0;
            c[20] = *cls as f32 / 6.0;
        }

        Target::ChainLink(idx) => {
            c[4]  = 1.0;
            c[22] = *idx as f32 / 10.0; // overrides position — that's fine for chain links
        }
    }

    // Always set position (overrides only if ChainLink didn't)
    if !matches!(&opt.target, Target::ChainLink(_)) {
        set_pos(&mut c, slot, total);
    }
    c
}
