use novacana_engine::{
    card::{CardKind, DiscoveryName, FlareName, FractureName, OmenName, StarName},
    player::GameView,
    star_chart::StarChart,
};

pub const FEATURE_DIM: usize = 104;

// ── Card-type index (0-28) ────────────────────────────────────────────────────

pub fn card_type_idx(kind: &CardKind) -> usize {  // also used by candidate.rs
    match kind {
        CardKind::Star(StarName::Protostar)          =>  0,
        CardKind::Star(StarName::YellowDwarf)        =>  1,
        CardKind::Star(StarName::RedGiant)           =>  2,
        CardKind::Star(StarName::RedSupergiant)      =>  3,
        CardKind::Star(StarName::WhiteDwarf)         =>  4,
        CardKind::Star(StarName::RedDwarf)           =>  5,
        CardKind::Star(StarName::BrownDwarf)         =>  6,
        CardKind::Discovery(DiscoveryName::AsteroidBelt)  =>  7,
        CardKind::Discovery(DiscoveryName::DwarfPlanet)   =>  8,
        CardKind::Discovery(DiscoveryName::RockyPlanet)   =>  9,
        CardKind::Discovery(DiscoveryName::GasGiant)      => 10,
        CardKind::Discovery(DiscoveryName::Comet)         => 11,
        CardKind::Flare(FlareName::SupernovaFusion)   => 12,
        CardKind::Flare(FlareName::SupernovaGravity)  => 13,
        CardKind::Flare(FlareName::SupernovaGas)      => 14,
        CardKind::Flare(FlareName::CosmicDenial)      => 15,
        CardKind::Flare(FlareName::Ignition)          => 16,
        CardKind::Flare(FlareName::AstralProjection)  => 17,
        CardKind::Flare(FlareName::StellarAlignment)  => 18,
        CardKind::Flare(FlareName::Wormhole)          => 19,
        CardKind::Omen(OmenName::SolarEclipse)        => 20,
        CardKind::Omen(OmenName::PlanetOfLife)        => 21,
        CardKind::Fracture(FractureName::BlackHole)   => 22,
        CardKind::Fracture(FractureName::Athena)      => 23,
        CardKind::Fracture(FractureName::Pulsar)      => 24,
        CardKind::Fracture(FractureName::DarkMatterVoid)             => 25,
        CardKind::Fracture(FractureName::QuantumEntanglementFailure) => 26,
        CardKind::Fracture(FractureName::StellarNurseryCollapse)     => 27,
        CardKind::Fracture(FractureName::Shockwave)  => 28,
    }
}

// ── Star-type index (0-6, matching card_type_idx offsets 0-6) ────────────────

fn star_idx(name: StarName) -> usize {
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

// ── Star chart helper ─────────────────────────────────────────────────────────

fn encode_chart(chart: &StarChart, out: &mut [f32], base: usize) {
    // [base+0]: complete_systems / 3.0
    // [base+1]: incomplete_systems / 6.0
    // [base+2..+9]: has_star_type (7 binary)
    // [base+9]: overall fill fraction
    // [base+10]: has eclipse protected
    // [base+11]: approaching win (2+ complete)
    let complete = chart.complete_systems();
    let incomplete = chart.incomplete_systems();

    out[base]     = complete.len() as f32 / 3.0;
    out[base + 1] = incomplete.len() as f32 / 6.0;

    for sys in &chart.systems {
        let si = star_idx(sys.star_name);
        out[base + 2 + si] = 1.0;
    }

    let (total_filled, total_needed): (usize, usize) = chart.systems.iter()
        .map(|s| (s.discoveries.len(), s.class as usize))
        .fold((0, 0), |(af, an), (f, n)| (af + f, an + n));
    out[base + 9] = if total_needed > 0 { total_filled as f32 / total_needed as f32 } else { 0.0 };

    out[base + 10] = if chart.systems.iter().any(|s| s.is_eclipse_protected()) { 1.0 } else { 0.0 };
    out[base + 11] = if complete.len() >= 2 { 1.0 } else { 0.0 };
}

// ── Opponent features (12 per opponent) ──────────────────────────────────────

fn encode_opponent(hand_size: usize, chart: &StarChart, out: &mut [f32], base: usize) {
    out[base]     = hand_size as f32 / 10.0;
    out[base + 1] = chart.complete_systems().len() as f32 / 3.0;
    out[base + 2] = chart.incomplete_systems().len() as f32 / 6.0;
    out[base + 3] = if chart.complete_systems().len() >= 2 { 1.0 } else { 0.0 };
    out[base + 4] = if chart.systems.iter().any(|s| s.is_eclipse_protected()) { 1.0 } else { 0.0 };
    for sys in &chart.systems {
        let si = star_idx(sys.star_name);
        out[base + 5 + si] = 1.0;
    }
}

// ── Main encoder ──────────────────────────────────────────────────────────────

pub fn encode(view: &GameView) -> [f32; FEATURE_DIM] {
    let mut feat = [0.0f32; FEATURE_DIM];

    // ── [0..29] My hand card-type counts (normalised /3) ─────────────────────
    for card in &view.my_hand {
        let idx = card_type_idx(&card.kind);
        feat[idx] += 1.0 / 3.0;
    }

    // ── [29..52] My star chart (23 features) ─────────────────────────────────
    encode_chart(&view.my_star_chart, &mut feat, 29);

    // Incomplete system fill fractions per star type [41..48]
    for sys in view.my_star_chart.incomplete_systems() {
        let si = star_idx(sys.star_name);
        let fill = sys.discoveries.len() as f32 / sys.class as f32;
        feat[41 + si] = fill.max(feat[41 + si]);
    }

    // Supernova flags [48..52]
    let has_fusion  = view.my_hand.iter().any(|c| matches!(&c.kind, CardKind::Flare(FlareName::SupernovaFusion)));
    let has_gravity = view.my_hand.iter().any(|c| matches!(&c.kind, CardKind::Flare(FlareName::SupernovaGravity)));
    let has_gas     = view.my_hand.iter().any(|c| matches!(&c.kind, CardKind::Flare(FlareName::SupernovaGas)));
    feat[48] = if has_fusion  { 1.0 } else { 0.0 };
    feat[49] = if has_gravity { 1.0 } else { 0.0 };
    feat[50] = if has_gas     { 1.0 } else { 0.0 };
    feat[51] = if has_fusion && has_gravity && has_gas
                  && !view.my_star_chart.complete_systems().is_empty() { 1.0 } else { 0.0 };

    // ── [52..88] Opponents (up to 3 × 12 features) ───────────────────────────
    for (slot, opp) in view.opponents.iter().enumerate().take(3) {
        let hand_size = if opp.hand_revealed.is_empty() { opp.hand_size } else { opp.hand_revealed.len() };
        encode_opponent(hand_size, &opp.star_chart, &mut feat, 52 + slot * 12);
    }

    // ── [88..96] Global state ─────────────────────────────────────────────────
    feat[88] = view.deck_size as f32 / 90.0;
    feat[89] = view.turn_number as f32 / 400.0;

    if let Some(class) = view.dark_matter_void {
        feat[90] = 1.0;
        if (2u8..=6).contains(&class) {
            feat[90 + (class - 1) as usize] = 1.0;
        }
    }

    // ── [96..104] Derived signals for the value head ──────────────────────────
    // [96] My current hand size (distinct from type-count vector above).
    feat[96] = view.my_hand.len() as f32 / 7.0;
    // [97] Am I the active player this turn.
    feat[97] = if view.am_active() { 1.0 } else { 0.0 };
    // [98] Number of opponents (proxy for game size).
    feat[98] = view.opponents.len() as f32 / 3.0;
    // [99] Best opponent complete-system count (the strongest threat).
    let best_opp_complete = view.opponents.iter()
        .map(|o| o.star_chart.complete_systems().len())
        .max()
        .unwrap_or(0);
    feat[99] = best_opp_complete as f32 / 3.0;
    // [100] Game progress (inverse of deck remaining).
    feat[100] = 1.0 - feat[88];
    // [101] My hand is at or near the limit (pressure signal).
    feat[101] = if view.my_hand.len() >= 5 { 1.0 } else { 0.0 };
    // [102] I am leading in complete systems vs all opponents.
    let my_complete = view.my_star_chart.complete_systems().len();
    feat[102] = if my_complete > best_opp_complete { 1.0 } else { 0.0 };
    // [103] Deck exhausted.
    feat[103] = if view.deck_size == 0 { 1.0 } else { 0.0 };

    feat
}
