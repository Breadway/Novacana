use novacana_engine::{
    card::{CardKind, DiscoveryName, FlareName, FractureName, OmenName, StarName},
    player::GameView,
    star_chart::StarChart,
};

pub const FEATURE_DIM: usize = 162;
const BASE_DIM: usize = 104;

// ── Card-type index (0-28), also used by candidate.rs ────────────────────────

pub fn card_type_idx(kind: &CardKind) -> usize {
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

fn encode_chart(chart: &StarChart, out: &mut [f32], base: usize) {
    let complete   = chart.complete_systems();
    let incomplete = chart.incomplete_systems();
    out[base]     = complete.len() as f32 / 3.0;
    out[base + 1] = incomplete.len() as f32 / 6.0;
    for sys in &chart.systems {
        out[base + 2 + star_idx(sys.star_name)] = 1.0;
    }
    let (filled, needed): (usize, usize) = chart.systems.iter()
        .map(|s| (s.discoveries.len(), s.class as usize))
        .fold((0, 0), |(af, an), (f, n)| (af + f, an + n));
    out[base + 9]  = if needed > 0 { filled as f32 / needed as f32 } else { 0.0 };
    out[base + 10] = if chart.systems.iter().any(|s| s.is_eclipse_protected()) { 1.0 } else { 0.0 };
    out[base + 11] = if complete.len() >= 2 { 1.0 } else { 0.0 };
}

fn encode_opponent(hand_size: usize, chart: &StarChart, out: &mut [f32], base: usize) {
    out[base]     = hand_size as f32 / 10.0;
    out[base + 1] = chart.complete_systems().len() as f32 / 3.0;
    out[base + 2] = chart.incomplete_systems().len() as f32 / 6.0;
    out[base + 3] = if chart.complete_systems().len() >= 2 { 1.0 } else { 0.0 };
    out[base + 4] = if chart.systems.iter().any(|s| s.is_eclipse_protected()) { 1.0 } else { 0.0 };
    for sys in &chart.systems {
        out[base + 5 + star_idx(sys.star_name)] = 1.0;
    }
}

// 104-feature snapshot matching v5 layout (base for v6 extension)
fn encode_base(view: &GameView) -> [f32; BASE_DIM] {
    let mut f = [0.0f32; BASE_DIM];

    for card in &view.my_hand {
        f[card_type_idx(&card.kind)] += 1.0 / 3.0;
    }
    encode_chart(&view.my_star_chart, &mut f, 29);
    for sys in view.my_star_chart.incomplete_systems() {
        let si   = star_idx(sys.star_name);
        let fill = sys.discoveries.len() as f32 / sys.class as f32;
        f[41 + si] = fill.max(f[41 + si]);
    }
    let has_fusion  = view.my_hand.iter().any(|c| matches!(&c.kind, CardKind::Flare(FlareName::SupernovaFusion)));
    let has_gravity = view.my_hand.iter().any(|c| matches!(&c.kind, CardKind::Flare(FlareName::SupernovaGravity)));
    let has_gas     = view.my_hand.iter().any(|c| matches!(&c.kind, CardKind::Flare(FlareName::SupernovaGas)));
    f[48] = if has_fusion  { 1.0 } else { 0.0 };
    f[49] = if has_gravity { 1.0 } else { 0.0 };
    f[50] = if has_gas     { 1.0 } else { 0.0 };
    f[51] = if has_fusion && has_gravity && has_gas
               && !view.my_star_chart.complete_systems().is_empty() { 1.0 } else { 0.0 };

    for (slot, opp) in view.opponents.iter().enumerate().take(3) {
        let hs = if opp.hand_revealed.is_empty() { opp.hand_size } else { opp.hand_revealed.len() };
        encode_opponent(hs, &opp.star_chart, &mut f, 52 + slot * 12);
    }

    f[88] = view.deck_size as f32 / 90.0;
    f[89] = view.turn_number as f32 / 400.0;
    if let Some(class) = view.dark_matter_void {
        f[90] = 1.0;
        if (2u8..=6).contains(&class) { f[90 + (class - 1) as usize] = 1.0; }
    }

    f[96] = view.my_hand.len() as f32 / 7.0;
    f[97] = if view.am_active() { 1.0 } else { 0.0 };
    f[98] = view.opponents.len() as f32 / 3.0;
    let best_opp = view.opponents.iter().map(|o| o.star_chart.complete_systems().len()).max().unwrap_or(0);
    f[99]  = best_opp as f32 / 3.0;
    f[100] = 1.0 - f[88];
    f[101] = if view.my_hand.len() >= 5 { 1.0 } else { 0.0 };
    f[102] = if view.my_star_chart.complete_systems().len() > best_opp { 1.0 } else { 0.0 };
    f[103] = if view.deck_size == 0 { 1.0 } else { 0.0 };
    f
}

fn card_category(kind: &CardKind) -> usize {
    match kind {
        CardKind::Star(_)      => 0,
        CardKind::Discovery(_) => 1,
        CardKind::Flare(_)     => 2,
        CardKind::Fracture(_)  => 3,
        CardKind::Omen(_)      => 4,
    }
}

// ── Main encoder: 162-feature v6 snapshot ────────────────────────────────────

pub fn encode(view: &GameView) -> [f32; FEATURE_DIM] {
    let mut feat = [0.0f32; FEATURE_DIM];
    feat[..BASE_DIM].copy_from_slice(&encode_base(view));

    // [104..124] Play history: 4 seats × 5 card categories, counts /5.
    // Seat-relative: row 0 = my plays, rows 1-3 = opponents[0..3] by their idx.
    // The window is the last 20 global plays, so a single seat's count can
    // exceed 5 - clamp instead of just scaling.
    let mut history = [[0u8; 5]; 4];
    for &(player_idx, ref kind) in &view.recent_plays {
        let seat = if player_idx == view.my_idx {
            Some(0)
        } else {
            view.opponents.iter().position(|o| o.idx == player_idx).map(|slot| slot + 1)
        };
        if let Some(seat) = seat {
            history[seat][card_category(kind)] =
                history[seat][card_category(kind)].saturating_add(1);
        }
    }
    for seat in 0..4 {
        for cat in 0..5 {
            feat[104 + seat * 5 + cat] = (history[seat][cat] as f32 / 5.0).min(1.0);
        }
    }

    // [124..133] Threat flags: 3 opponent slots × (has_flare, has_fracture, has_supernova)
    for (slot, opp) in view.opponents.iter().enumerate().take(3) {
        let oi = opp.idx;
        let (mut hf, mut hfr, mut hs) = (false, false, false);
        for &(pi, ref kind) in &view.recent_plays {
            if pi != oi { continue; }
            match kind {
                CardKind::Flare(f) => { hf = true; if f.is_supernova() { hs = true; } }
                CardKind::Fracture(_) => hfr = true,
                _ => {}
            }
        }
        feat[124 + slot * 3]     = if hf  { 1.0 } else { 0.0 };
        feat[124 + slot * 3 + 1] = if hfr { 1.0 } else { 0.0 };
        feat[124 + slot * 3 + 2] = if hs  { 1.0 } else { 0.0 };
    }

    // [133..162] Cumulative discard pile counts per card type /3
    for (i, &count) in view.discard_pile_counts.iter().enumerate() {
        feat[133 + i] = count as f32 / 3.0;
    }

    feat
}
