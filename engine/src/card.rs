use std::fmt;

pub type CardId = u32;

// ── Star names ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum StarName {
    Protostar,
    YellowDwarf,
    RedGiant,
    RedSupergiant,
    WhiteDwarf,
    RedDwarf,
    BrownDwarf,
}

impl StarName {
    /// Returns the Star's Class value (number of Discoveries needed to complete a system).
    /// Protostar has no Class and cannot build a system.
    pub fn class(self) -> Option<u8> {
        match self {
            StarName::RedDwarf | StarName::BrownDwarf => Some(2),
            StarName::WhiteDwarf => Some(3),
            StarName::YellowDwarf => Some(4),
            StarName::RedGiant => Some(5),
            StarName::RedSupergiant => Some(6),
            StarName::Protostar => None,
        }
    }
}

impl fmt::Display for StarName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StarName::Protostar => write!(f, "Protostar"),
            StarName::YellowDwarf => write!(f, "Yellow Dwarf"),
            StarName::RedGiant => write!(f, "Red Giant"),
            StarName::RedSupergiant => write!(f, "Red Supergiant"),
            StarName::WhiteDwarf => write!(f, "White Dwarf"),
            StarName::RedDwarf => write!(f, "Red Dwarf"),
            StarName::BrownDwarf => write!(f, "Brown Dwarf"),
        }
    }
}

// ── Discovery names ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DiscoveryName {
    AsteroidBelt,
    DwarfPlanet,
    RockyPlanet,
    GasGiant,
    Comet,
}

impl fmt::Display for DiscoveryName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiscoveryName::AsteroidBelt => write!(f, "Asteroid Belt"),
            DiscoveryName::DwarfPlanet => write!(f, "Dwarf Planet"),
            DiscoveryName::RockyPlanet => write!(f, "Rocky Planet"),
            DiscoveryName::GasGiant => write!(f, "Gas Giant"),
            DiscoveryName::Comet => write!(f, "Comet"),
        }
    }
}

// ── Flare names ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum FlareName {
    SupernovaFusion,
    SupernovaGravity,
    SupernovaGas,
    CosmicDenial,
    Ignition,
    AstralProjection,
    StellarAlignment,
    Wormhole,
}

impl FlareName {
    pub fn is_supernova(self) -> bool {
        matches!(
            self,
            FlareName::SupernovaFusion | FlareName::SupernovaGravity | FlareName::SupernovaGas
        )
    }
}

impl fmt::Display for FlareName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FlareName::SupernovaFusion => write!(f, "Supernova - Fusion"),
            FlareName::SupernovaGravity => write!(f, "Supernova - Gravity"),
            FlareName::SupernovaGas => write!(f, "Supernova - Gas"),
            FlareName::CosmicDenial => write!(f, "Cosmic Denial"),
            FlareName::Ignition => write!(f, "Ignition"),
            FlareName::AstralProjection => write!(f, "Astral Projection"),
            FlareName::StellarAlignment => write!(f, "Stellar Alignment"),
            FlareName::Wormhole => write!(f, "Wormhole"),
        }
    }
}

// ── Omen names ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum OmenName {
    SolarEclipse,
    PlanetOfLife,
}

impl fmt::Display for OmenName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OmenName::SolarEclipse => write!(f, "Solar Eclipse"),
            OmenName::PlanetOfLife => write!(f, "Planet of Life"),
        }
    }
}

// ── Fracture names ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum FractureName {
    BlackHole,
    Athena,
    Pulsar,
    DarkMatterVoid,
    QuantumEntanglementFailure,
    StellarNurseryCollapse,
    Shockwave,
}

impl fmt::Display for FractureName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FractureName::BlackHole => write!(f, "Black Hole"),
            FractureName::Athena => write!(f, "Athena"),
            FractureName::Pulsar => write!(f, "Pulsar"),
            FractureName::DarkMatterVoid => write!(f, "Dark Matter Void"),
            FractureName::QuantumEntanglementFailure => write!(f, "Quantum Entanglement Failure"),
            FractureName::StellarNurseryCollapse => write!(f, "Stellar Nursery Collapse"),
            FractureName::Shockwave => write!(f, "Shockwave"),
        }
    }
}

// ── Card kind ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CardKind {
    Star(StarName),
    Discovery(DiscoveryName),
    Flare(FlareName),
    Omen(OmenName),
    Fracture(FractureName),
}

impl CardKind {
    /// Whether this card can enter the Chain when played reactively.
    pub fn can_enter_chain(&self) -> bool {
        matches!(self, CardKind::Flare(_) | CardKind::Fracture(_) | CardKind::Omen(_))
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            CardKind::Star(_) => "Star",
            CardKind::Discovery(_) => "Discovery",
            CardKind::Flare(_) => "Flare",
            CardKind::Omen(_) => "Omen",
            CardKind::Fracture(_) => "Fracture",
        }
    }

    pub fn short_name(&self) -> String {
        match self {
            CardKind::Star(n) => n.to_string(),
            CardKind::Discovery(n) => n.to_string(),
            CardKind::Flare(n) => n.to_string(),
            CardKind::Omen(n) => n.to_string(),
            CardKind::Fracture(n) => n.to_string(),
        }
    }

    /// Stable index 0..29 mapping each unique card type to a slot.
    /// Stars 0-6, Discoveries 7-11, Flares 12-19, Omens 20-21, Fractures 22-28.
    pub fn type_idx(&self) -> usize {
        match self {
            CardKind::Star(n) => match n {
                StarName::Protostar     => 0,
                StarName::YellowDwarf   => 1,
                StarName::RedGiant      => 2,
                StarName::RedSupergiant => 3,
                StarName::WhiteDwarf    => 4,
                StarName::RedDwarf      => 5,
                StarName::BrownDwarf    => 6,
            },
            CardKind::Discovery(n) => match n {
                DiscoveryName::AsteroidBelt => 7,
                DiscoveryName::DwarfPlanet  => 8,
                DiscoveryName::RockyPlanet  => 9,
                DiscoveryName::GasGiant     => 10,
                DiscoveryName::Comet        => 11,
            },
            CardKind::Flare(n) => match n {
                FlareName::SupernovaFusion  => 12,
                FlareName::SupernovaGravity => 13,
                FlareName::SupernovaGas     => 14,
                FlareName::CosmicDenial     => 15,
                FlareName::Ignition         => 16,
                FlareName::AstralProjection => 17,
                FlareName::StellarAlignment => 18,
                FlareName::Wormhole         => 19,
            },
            CardKind::Omen(n) => match n {
                OmenName::SolarEclipse => 20,
                OmenName::PlanetOfLife => 21,
            },
            CardKind::Fracture(n) => match n {
                FractureName::BlackHole                 => 22,
                FractureName::Athena                    => 23,
                FractureName::Pulsar                    => 24,
                FractureName::DarkMatterVoid            => 25,
                FractureName::QuantumEntanglementFailure => 26,
                FractureName::StellarNurseryCollapse    => 27,
                FractureName::Shockwave                 => 28,
            },
        }
    }
}

impl fmt::Display for CardKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CardKind::Star(n) => write!(f, "{n} [Star]"),
            CardKind::Discovery(n) => write!(f, "{n} [Discovery]"),
            CardKind::Flare(n) => write!(f, "{n} [Flare]"),
            CardKind::Omen(n) => write!(f, "{n} [Omen]"),
            CardKind::Fracture(n) => write!(f, "{n} [Fracture]"),
        }
    }
}

// ── Card ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Card {
    pub id: CardId,
    pub kind: CardKind,
}

impl Card {
    pub fn new(id: CardId, kind: CardKind) -> Self {
        Self { id, kind }
    }

    pub fn is_supernova(&self) -> bool {
        matches!(&self.kind, CardKind::Flare(n) if n.is_supernova())
    }

    pub fn star_class(&self) -> Option<u8> {
        match &self.kind {
            CardKind::Star(n) => n.class(),
            _ => None,
        }
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}
