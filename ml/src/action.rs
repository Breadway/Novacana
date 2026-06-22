use novacana_engine::{
    card::{Card, CardKind, FlareName, FractureName, OmenName, StarName},
    player::{Action, ChainAction, ProtostarChoice},
};

pub const NUM_ACTIONS: usize = 35;

// ── Abstract action layout ────────────────────────────────────────────────────
//
//  0        Pass
//  1- 6     PlayStar(YDwarf, RGiant, RSuper, WDwarf, RDwarf, BDwarf)
//  7-11     PlayDiscovery(AsteroidBelt, DwarfPlanet, RockyPlanet, GasGiant, Comet)
// 12-17     PlaySolarSystem(same 6 star kinds)
// 18-22     PlayFlare(StellarAlignment, CosmicDenial, Wormhole, Ignition, AstralProjection)
// 23        PlaySupernovas
// 24-30     PlayFracture(BlackHole, Athena, Pulsar, DMV, QEF, SNC, Shockwave)
// 31-32     PlayProtostar(DrawTwo, SearchWhiteDwarf)
// 33-34     PlayOmen(SolarEclipse, PlanetOfLife)

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
    use novacana_engine::card::DiscoveryName::*;
    Some(match kind {
        CardKind::Discovery(AsteroidBelt) =>  7,
        CardKind::Discovery(DwarfPlanet)  =>  8,
        CardKind::Discovery(RockyPlanet)  =>  9,
        CardKind::Discovery(GasGiant)     => 10,
        CardKind::Discovery(Comet)        => 11,
        _ => return None,
    })
}

fn flare_abstract(f: FlareName) -> Option<usize> {
    Some(match f {
        FlareName::StellarAlignment => 18,
        FlareName::CosmicDenial     => 19,
        FlareName::Wormhole         => 20,
        FlareName::Ignition         => 21,
        FlareName::AstralProjection => 22,
        _ => return None, // supernovas handled separately
    })
}

fn fracture_abstract(f: FractureName) -> usize {
    match f {
        FractureName::BlackHole                    => 24,
        FractureName::Athena                       => 25,
        FractureName::Pulsar                       => 26,
        FractureName::DarkMatterVoid               => 27,
        FractureName::QuantumEntanglementFailure   => 28,
        FractureName::StellarNurseryCollapse       => 29,
        FractureName::Shockwave                    => 30,
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
                star_abstract(*name).map(|base| base + 11) // offset by 11 → 12-17
            } else { None }
        }

        Action::PlayFlare(id) => {
            if let Some(CardKind::Flare(f)) = lookup_kind(*id) {
                flare_abstract(*f)
            } else { None }
        }

        Action::PlaySupernovas { .. } => Some(23),

        Action::PlayFracture(id) => {
            if let Some(CardKind::Fracture(f)) = lookup_kind(*id) {
                Some(fracture_abstract(*f))
            } else { None }
        }

        Action::PlayProtostar { choice, .. } => Some(match choice {
            ProtostarChoice::DrawTwo         => 31,
            ProtostarChoice::SearchWhiteDwarf => 32,
        }),

        Action::PlayOmen { card: id, .. } => {
            Some(match lookup_kind(*id)? {
                CardKind::Omen(OmenName::SolarEclipse) => 33,
                CardKind::Omen(OmenName::PlanetOfLife) => 34,
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
        CardKind::Omen(OmenName::SolarEclipse) => Some(33),
        CardKind::Omen(OmenName::PlanetOfLife) => Some(34),
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

