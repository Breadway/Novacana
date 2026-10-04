use std::fmt;

use crate::card::{Card, CardId, CardKind, DiscoveryName, StarName};

// ── Attached discovery ────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AttachedDiscovery {
    pub card_id: CardId,
    pub name: DiscoveryName,
    /// Planet of Life omen attached to this Rocky Planet.
    pub planet_of_life_id: Option<CardId>,
}

// ── Solar system ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SolarSystem {
    pub star_id: CardId,
    pub star_name: StarName,
    /// Discoveries needed to complete - equal to the star's class value.
    pub class: u8,
    pub discoveries: Vec<AttachedDiscovery>,
    /// Solar Eclipse omen attached to this (completed) system.
    pub eclipse_id: Option<CardId>,
}

impl SolarSystem {
    pub fn new(star_id: CardId, star_name: StarName, class: u8) -> Self {
        Self { star_id, star_name, class, discoveries: Vec::new(), eclipse_id: None }
    }

    pub fn is_complete(&self) -> bool {
        self.discoveries.len() as u8 == self.class
    }

    pub fn has_gas_giant(&self) -> bool {
        self.discoveries.iter().any(|d| d.name == DiscoveryName::GasGiant)
    }

    pub fn rocky_planets(&self) -> Vec<CardId> {
        self.discoveries
            .iter()
            .filter(|d| d.name == DiscoveryName::RockyPlanet)
            .map(|d| d.card_id)
            .collect()
    }

    pub fn is_eclipse_protected(&self) -> bool {
        self.eclipse_id.is_some()
    }

    pub fn add_discovery(&mut self, card_id: CardId, name: DiscoveryName) {
        self.discoveries.push(AttachedDiscovery { card_id, name, planet_of_life_id: None });
    }

    pub fn remove_discovery(&mut self, card_id: CardId) -> Option<AttachedDiscovery> {
        if let Some(pos) = self.discoveries.iter().position(|d| d.card_id == card_id) {
            Some(self.discoveries.remove(pos))
        } else {
            None
        }
    }

    pub fn attach_planet_of_life(&mut self, rocky_id: CardId, omen_id: CardId) -> bool {
        if let Some(disc) = self.discoveries.iter_mut().find(|d| d.card_id == rocky_id) {
            if disc.name == DiscoveryName::RockyPlanet && disc.planet_of_life_id.is_none() {
                disc.planet_of_life_id = Some(omen_id);
                return true;
            }
        }
        false
    }
}

impl fmt::Display for SolarSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.is_complete() {
            format!("COMPLETE [{}/{}]", self.discoveries.len(), self.class)
        } else {
            format!("{}/{}", self.discoveries.len(), self.class)
        };
        let eclipse = if self.eclipse_id.is_some() { " [Eclipse]" } else { "" };
        write!(f, "{} (Class {}){} - {}", self.star_name, self.class, eclipse, status)?;
        for disc in &self.discoveries {
            let pol = if disc.planet_of_life_id.is_some() { " [Planet of Life]" } else { "" };
            write!(f, "\n    · {}{}", disc.name, pol)?;
        }
        Ok(())
    }
}

// ── Star chart ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct StarChart {
    pub systems: Vec<SolarSystem>,
    /// Omens that survive board wipes and float without a host (Black Hole).
    pub floating_omens: Vec<CardId>,
}

impl StarChart {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn place_star(&mut self, card_id: CardId, name: StarName, class: u8) {
        self.systems.push(SolarSystem::new(card_id, name, class));
    }

    pub fn system_for_star(&self, star_id: CardId) -> Option<&SolarSystem> {
        self.systems.iter().find(|s| s.star_id == star_id)
    }

    pub fn system_for_star_mut(&mut self, star_id: CardId) -> Option<&mut SolarSystem> {
        self.systems.iter_mut().find(|s| s.star_id == star_id)
    }

    /// Attach a Discovery to an incomplete system identified by its star card ID.
    pub fn attach_discovery(&mut self, star_id: CardId, disc_id: CardId, name: DiscoveryName) -> bool {
        if let Some(sys) = self.system_for_star_mut(star_id) {
            if !sys.is_complete() {
                sys.add_discovery(disc_id, name);
                return true;
            }
        }
        false
    }

    pub fn complete_count(&self) -> usize {
        self.systems.iter().filter(|s| s.is_complete()).count()
    }

    pub fn complete_iter(&self) -> impl Iterator<Item = &SolarSystem> {
        self.systems.iter().filter(|s| s.is_complete())
    }

    pub fn incomplete_iter(&self) -> impl Iterator<Item = &SolarSystem> {
        self.systems.iter().filter(|s| !s.is_complete())
    }

    pub fn complete_systems(&self) -> Vec<&SolarSystem> {
        self.systems.iter().filter(|s| s.is_complete()).collect()
    }

    pub fn incomplete_systems(&self) -> Vec<&SolarSystem> {
        self.systems.iter().filter(|s| !s.is_complete()).collect()
    }

    /// Remove all Stars and Discoveries from this chart.
    ///
    /// Returns the card IDs that went to the deck (`to_deck`) and the IDs of Solar Eclipse
    /// omens that survived (rules: "Omen cards remain in play"). Planet of Life omens
    /// are destroyed when their host Rocky Planet leaves, so they are included in `to_deck`.
    /// Drain the Star Chart for a board-wipe effect (Black Hole, Pulsar).
    /// Eclipse-protected systems are completely untouched by any board wipe -
    /// the star, all discoveries, and the Solar Eclipse card all remain in place.
    /// Unprotected systems return their cards to the deck; Planet of Life is
    /// destroyed when its Rocky Planet host leaves.
    pub fn drain_for_board_wipe(&mut self) -> BoardWipeResult {
        let mut to_deck: Vec<CardId> = Vec::new();
        let mut kept: Vec<SolarSystem> = Vec::new();

        for sys in self.systems.drain(..) {
            if sys.is_eclipse_protected() {
                kept.push(sys);
                continue;
            }
            to_deck.push(sys.star_id);
            for disc in sys.discoveries {
                to_deck.push(disc.card_id);
                if let Some(pol_id) = disc.planet_of_life_id {
                    to_deck.push(pol_id);
                }
            }
            // An eclipse on an unprotected system shouldn't occur (eclipse makes it
            // protected), but if it somehow does, float it rather than wipe it.
            if let Some(eclipse_id) = sys.eclipse_id {
                self.floating_omens.push(eclipse_id);
            }
        }

        self.systems = kept;
        BoardWipeResult { to_deck, floating_eclipse_ids: vec![] }
    }

    /// Remove a single Discovery by card ID. Returns (star_name, disc_name,
    /// freed_planet_of_life_id) if found - the third element is `Some` when
    /// the removed Discovery was a Rocky Planet with a Planet of Life omen
    /// attached, so the caller can return that card to the deck instead of
    /// silently losing it (a real card-conservation bug this used to have:
    /// the omen id was dropped along with the rest of the `AttachedDiscovery`
    /// struct - caught by the engine test suite's card-conservation invariant
    /// checker). Skips eclipse-protected systems unless `bypass_eclipse` is true.
    pub fn remove_discovery(
        &mut self,
        disc_id: CardId,
        bypass_eclipse: bool,
    ) -> Option<(StarName, DiscoveryName, Option<CardId>)> {
        for sys in &mut self.systems {
            if sys.is_eclipse_protected() && !bypass_eclipse {
                continue;
            }
            if let Some(disc) = sys.remove_discovery(disc_id) {
                return Some((sys.star_name, disc.name, disc.planet_of_life_id));
            }
        }
        None
    }

    /// Remove a Gas Giant Discovery from a non-protected system, returning its card ID.
    pub fn take_gas_giant(&mut self, bypass_eclipse: bool) -> Option<(CardId, StarName)> {
        for sys in &mut self.systems {
            if sys.is_eclipse_protected() && !bypass_eclipse {
                continue;
            }
            let gas_pos = sys.discoveries.iter().position(|d| d.name == DiscoveryName::GasGiant);
            if let Some(pos) = gas_pos {
                let disc = sys.discoveries.remove(pos);
                return Some((disc.card_id, sys.star_name));
            }
        }
        None
    }

    /// Attach Solar Eclipse to a completed, unprotected system.
    pub fn attach_solar_eclipse(&mut self, star_id: CardId, omen_id: CardId) -> bool {
        if let Some(sys) = self.system_for_star_mut(star_id) {
            if sys.is_complete() && sys.eclipse_id.is_none() {
                sys.eclipse_id = Some(omen_id);
                return true;
            }
        }
        false
    }

    /// Detach and return the Solar Eclipse omen ID from a system.
    pub fn detach_solar_eclipse(&mut self, star_id: CardId) -> Option<CardId> {
        self.system_for_star_mut(star_id).and_then(|s| s.eclipse_id.take())
    }

    /// Attach Planet of Life to a Rocky Planet in this chart.
    pub fn attach_planet_of_life(&mut self, rocky_id: CardId, omen_id: CardId) -> bool {
        self.systems.iter_mut().any(|sys| sys.attach_planet_of_life(rocky_id, omen_id))
    }

    pub fn eclipse_protected_stars(&self) -> Vec<CardId> {
        self.systems.iter().filter(|s| s.is_eclipse_protected()).map(|s| s.star_id).collect()
    }

    pub fn find_discovery(&self, disc_id: CardId) -> Option<(&SolarSystem, &AttachedDiscovery)> {
        for sys in &self.systems {
            if let Some(disc) = sys.discoveries.iter().find(|d| d.card_id == disc_id) {
                return Some((sys, disc));
            }
        }
        None
    }

    /// All card IDs currently in this chart (stars + discoveries, not omens).
    pub fn all_field_ids(&self) -> Vec<CardId> {
        let mut ids = Vec::new();
        for sys in &self.systems {
            ids.push(sys.star_id);
            ids.extend(sys.discoveries.iter().map(|d| d.card_id));
        }
        ids
    }

    /// All omen IDs currently attached in this chart.
    pub fn attached_omen_ids(&self) -> Vec<CardId> {
        let mut ids = Vec::new();
        for sys in &self.systems {
            if let Some(e) = sys.eclipse_id {
                ids.push(e);
            }
            for d in &sys.discoveries {
                if let Some(p) = d.planet_of_life_id {
                    ids.push(p);
                }
            }
        }
        ids.extend(self.floating_omens.iter().copied());
        ids
    }
}

impl fmt::Display for StarChart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.systems.is_empty() && self.floating_omens.is_empty() {
            return write!(f, "  (empty)");
        }
        for (i, sys) in self.systems.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "  System {}: {}", i + 1, sys)?;
        }
        if !self.floating_omens.is_empty() {
            writeln!(f)?;
            write!(f, "  [Floating omens: {} Solar Eclipse(s)]", self.floating_omens.len())?;
        }
        Ok(())
    }
}

// ── Board wipe result ─────────────────────────────────────────────────────────

pub struct BoardWipeResult {
    /// Card IDs that should be shuffled into the Main Deck.
    pub to_deck: Vec<CardId>,
    /// Solar Eclipse IDs now floating in the chart (remain in play).
    pub floating_eclipse_ids: Vec<CardId>,
}

// ── Solar system play validation ──────────────────────────────────────────────

pub fn validate_solar_system_play(star: &Card, discoveries: &[Card]) -> Result<(), &'static str> {
    let class = match &star.kind {
        CardKind::Star(name) => name.class().ok_or("Protostar cannot anchor a Solar System")?,
        _ => return Err("first card must be a Star"),
    };
    if discoveries.len() as u8 != class {
        return Err("discovery count must match the star's class");
    }
    for card in discoveries {
        if !matches!(card.kind, CardKind::Discovery(_)) {
            return Err("all non-star cards must be Discoveries");
        }
    }
    Ok(())
}
