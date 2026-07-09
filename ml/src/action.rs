use novacana_engine::{
    card::{Card, CardKind, FlareName, FractureName, OmenName, StarName},
    player::{Action, ChainAction, ProtostarChoice},
};

pub const NUM_ACTIONS: usize = 31;

// ── Abstract action layout ────────────────────────────────────────────────────
//
//  0        Pass
//  1- 6     PlayStar(YDwarf, RGiant, RSuper, WDwarf, RDwarf, BDwarf)
//  7        PlayDiscovery (any type; candidate head disambiguates type+system)
//  8-13     PlaySolarSystem(same 6 star kinds)
// 14-18     PlayFlare(StellarAlignment, CosmicDenial, Wormhole, Ignition, AstralProjection)
// 19        PlaySupernovas
// 20-26     PlayFracture(BlackHole, Athena, Pulsar, DMV, QEF, SNC, Shockwave)
// 27-28     PlayProtostar(DrawTwo, SearchWhiteDwarf)
// 29-30     PlayOmen(SolarEclipse, PlanetOfLife)

fn star_abstract(name: StarName) -> Option<usize> {
    Some(match name {
        StarName::YellowDwarf   => 1,
        StarName::RedGiant      => 2,
        StarName::RedSupergiant => 3,
        StarName::WhiteDwarf    => 4,
        StarName::RedDwarf      => 5,
        StarName::BrownDwarf    => 6,
        StarName::Protostar     => return None,
    })
}

fn disc_abstract(kind: &CardKind) -> Option<usize> {
    match kind {
        CardKind::Discovery(_) => Some(7),
        _ => None,
    }
}

fn flare_abstract(f: FlareName) -> Option<usize> {
    Some(match f {
        FlareName::StellarAlignment => 14,
        FlareName::CosmicDenial     => 15,
        FlareName::Wormhole         => 16,
        FlareName::Ignition         => 17,
        FlareName::AstralProjection => 18,
        _ => return None, // supernovas handled separately
    })
}

fn fracture_abstract(f: FractureName) -> usize {
    match f {
        FractureName::BlackHole                    => 20,
        FractureName::Athena                       => 21,
        FractureName::Pulsar                       => 22,
        FractureName::DarkMatterVoid               => 23,
        FractureName::QuantumEntanglementFailure   => 24,
        FractureName::StellarNurseryCollapse       => 25,
        FractureName::Shockwave                    => 26,
    }
}

// ── Convert concrete action → abstract index ──────────────────────────────────

pub fn action_to_abstract(action: &Action, hand: &[Card]) -> Option<usize> {
    let lookup_kind = |id| hand.iter().find(|c| c.id == id).map(|c| &c.kind);

    match action {
        Action::Pass => Some(0),

        Action::PlayStar(id) => {
            if let Some(CardKind::Star(name)) = lookup_kind(*id) {
                star_abstract(*name).map(|base| base) // PlayStar: 1-6
            } else { None }
        }

        Action::PlayDiscovery { card: id, .. } => {
            lookup_kind(*id).and_then(disc_abstract)
        }

        Action::PlaySolarSystem { star: id, .. } => {
            if let Some(CardKind::Star(name)) = lookup_kind(*id) {
                star_abstract(*name).map(|base| base + 7) // offset by 7 → 8-13
            } else { None }
        }

        Action::PlayFlare(id) => {
            if let Some(CardKind::Flare(f)) = lookup_kind(*id) {
                flare_abstract(*f)
            } else { None }
        }

        Action::PlaySupernovas { .. } => Some(19),

        Action::PlayFracture(id) => {
            if let Some(CardKind::Fracture(f)) = lookup_kind(*id) {
                Some(fracture_abstract(*f))
            } else { None }
        }

        Action::PlayProtostar { choice, .. } => Some(match choice {
            ProtostarChoice::DrawTwo         => 27,
            ProtostarChoice::SearchWhiteDwarf => 28,
        }),

        Action::PlayOmen { card: id, .. } => {
            Some(match lookup_kind(*id)? {
                CardKind::Omen(OmenName::SolarEclipse) => 29,
                CardKind::Omen(OmenName::PlanetOfLife) => 30,
                _ => return None,
            })
        }
    }
}

// ── Chain response → abstract index ──────────────────────────────────────────

pub fn chain_response_to_abstract(ca: &ChainAction, hand: &[Card]) -> Option<usize> {
    let kind = hand.iter().find(|c| c.id == ca.card).map(|c| &c.kind)?;
    match kind {
        CardKind::Flare(f) => flare_abstract(*f),
        CardKind::Fracture(f) => Some(fracture_abstract(*f)),
        CardKind::Omen(OmenName::SolarEclipse) => Some(29),
        CardKind::Omen(OmenName::PlanetOfLife) => Some(30),
        _ => None,
    }
}

// ── Build legal action mask (u64 bitfield) ────────────────────────────────────

pub fn legal_mask(actions: &[Action], hand: &[Card]) -> u64 {
    let mut mask = 0u64;
    for a in actions {
        if let Some(idx) = action_to_abstract(a, hand) {
            mask |= 1u64 << idx;
        }
    }
    mask
}

pub fn chain_legal_mask(responses: &[ChainAction], hand: &[Card]) -> u64 {
    let mut mask = 1u64; // bit 0 = pass is always legal
    for r in responses {
        if let Some(idx) = chain_response_to_abstract(r, hand) {
            mask |= 1u64 << idx;
        }
    }
    mask
}

// ── Map abstract index → first matching concrete action ───────────────────────

pub fn abstract_to_action<'a>(
    abstract_idx: usize,
    legal: &'a [Action],
    hand: &[Card],
) -> Option<&'a Action> {
    legal.iter().find(|a| action_to_abstract(a, hand) == Some(abstract_idx))
}

pub fn abstract_to_chain<'a>(
    abstract_idx: usize,
    legal: &'a [ChainAction],
    hand: &[Card],
) -> Option<&'a ChainAction> {
    legal.iter().find(|r| chain_response_to_abstract(r, hand) == Some(abstract_idx))
}

