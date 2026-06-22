use std::collections::HashMap;

use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::card::{Card, CardId, CardKind, DiscoveryName, FlareName, FractureName, OmenName, StarName};

pub struct Deck {
    cards: Vec<Card>,
    next_id: CardId,
    /// Permanent lookup: every card ever created, for reconstruction after Black Hole / Pulsar.
    pub registry: HashMap<CardId, CardKind>,
}

impl Deck {
    /// Builds the 90-card Classic Mode deck.
    pub fn build_classic() -> Self {
        let mut deck = Deck {
            cards: Vec::with_capacity(90),
            next_id: 1,
            registry: HashMap::with_capacity(90),
        };

        // Stars (21)
        deck.add_many(CardKind::Star(StarName::Protostar), 3);
        deck.add_many(CardKind::Star(StarName::YellowDwarf), 3);
        deck.add_many(CardKind::Star(StarName::RedGiant), 3);
        deck.add_many(CardKind::Star(StarName::RedSupergiant), 2);
        deck.add_many(CardKind::Star(StarName::WhiteDwarf), 3);
        deck.add_many(CardKind::Star(StarName::RedDwarf), 4);
        deck.add_many(CardKind::Star(StarName::BrownDwarf), 3);

        // Discoveries (27)
        deck.add_many(CardKind::Discovery(DiscoveryName::AsteroidBelt), 5);
        deck.add_many(CardKind::Discovery(DiscoveryName::DwarfPlanet), 5);
        deck.add_many(CardKind::Discovery(DiscoveryName::RockyPlanet), 7);
        deck.add_many(CardKind::Discovery(DiscoveryName::GasGiant), 5);
        deck.add_many(CardKind::Discovery(DiscoveryName::Comet), 5);

        // Flares (19)
        deck.add_many(CardKind::Flare(FlareName::SupernovaFusion), 2);
        deck.add_many(CardKind::Flare(FlareName::SupernovaGravity), 2);
        deck.add_many(CardKind::Flare(FlareName::SupernovaGas), 2);
        deck.add_many(CardKind::Flare(FlareName::CosmicDenial), 3);
        deck.add_many(CardKind::Flare(FlareName::Ignition), 2);
        deck.add_many(CardKind::Flare(FlareName::AstralProjection), 2);
        deck.add_many(CardKind::Flare(FlareName::StellarAlignment), 4);
        deck.add_many(CardKind::Flare(FlareName::Wormhole), 2);

        // Omens (8)
        deck.add_many(CardKind::Omen(OmenName::SolarEclipse), 4);
        deck.add_many(CardKind::Omen(OmenName::PlanetOfLife), 4);

        // Fractures (15)
        deck.add_many(CardKind::Fracture(FractureName::BlackHole), 2);
        deck.add_many(CardKind::Fracture(FractureName::Athena), 3);
        deck.add_many(CardKind::Fracture(FractureName::Pulsar), 2);
        deck.add_many(CardKind::Fracture(FractureName::DarkMatterVoid), 2);
        deck.add_many(CardKind::Fracture(FractureName::QuantumEntanglementFailure), 2);
        deck.add_many(CardKind::Fracture(FractureName::StellarNurseryCollapse), 2);
        deck.add_many(CardKind::Fracture(FractureName::Shockwave), 2);

        assert_eq!(deck.cards.len(), 90, "classic deck must have exactly 90 cards");
        deck
    }

    fn add_many(&mut self, kind: CardKind, count: usize) {
        for _ in 0..count {
            let id = self.next_id;
            self.next_id += 1;
            self.registry.insert(id, kind.clone());
            self.cards.push(Card::new(id, kind.clone()));
        }
    }

    /// Reconstruct a Card from its ID using the registry.
    pub fn card_from_id(&self, id: CardId) -> Option<Card> {
        self.registry.get(&id).map(|kind| Card::new(id, kind.clone()))
    }

    pub fn shuffle(&mut self) {
        self.cards.shuffle(&mut thread_rng());
    }

    /// Draw from the top.
    pub fn draw(&mut self) -> Option<Card> {
        if self.cards.is_empty() {
            None
        } else {
            Some(self.cards.remove(0))
        }
    }

    /// Place a card face-down at the bottom (discard / post-resolution).
    pub fn push_bottom(&mut self, card: Card) {
        self.cards.push(card);
    }

    pub fn push_bottom_many(&mut self, cards: impl IntoIterator<Item = Card>) {
        for c in cards {
            self.cards.push(c);
        }
    }

    /// Shuffle a card back into the deck at a random position.
    pub fn shuffle_in(&mut self, card: Card) {
        let pos = rand::random::<usize>() % (self.cards.len() + 1);
        self.cards.insert(pos, card);
    }

    /// Shuffle all cards in a vec into the deck, then re-shuffle.
    pub fn shuffle_in_many(&mut self, cards: impl IntoIterator<Item = Card>) {
        for c in cards {
            self.cards.push(c);
        }
        self.cards.shuffle(&mut thread_rng());
    }

    /// Find and remove the first card matching the predicate, returning it.
    pub fn search_and_remove<F>(&mut self, predicate: F) -> Option<Card>
    where
        F: Fn(&Card) -> bool,
    {
        if let Some(pos) = self.cards.iter().position(|c| predicate(c)) {
            Some(self.cards.remove(pos))
        } else {
            None
        }
    }

    /// Peek at the top N cards without removing them (for Stellar Nursery Collapse).
    pub fn peek_top(&self, n: usize) -> &[Card] {
        let end = n.min(self.cards.len());
        &self.cards[..end]
    }

    /// Replace the top N cards in a given order (caller supplies card IDs in desired order).
    /// Cards not appearing in the supplied order are left at their current positions.
    pub fn reorder_top(&mut self, ordered_ids: &[CardId]) {
        let n = ordered_ids.len().min(self.cards.len());
        let mut top: Vec<Card> = self.cards.drain(..n).collect();
        let mut reordered = Vec::with_capacity(n);
        for &id in ordered_ids {
            if let Some(pos) = top.iter().position(|c| c.id == id) {
                reordered.push(top.remove(pos));
            }
        }
        // Any that weren't in ordered_ids (shouldn't happen, but defensive)
        reordered.extend(top);
        // Prepend them back
        for c in reordered.into_iter().rev() {
            self.cards.insert(0, c);
        }
        // Place on bottom instead — rules say "place them on the bottom in any order"
        // So drain the top n, then push_bottom in desired order
    }

    /// Stellar Nursery Collapse: shuffle deck, privately view top 5, then push them
    /// to the bottom in a player-specified order.
    pub fn nursery_collapse(&mut self, ordered_ids: Vec<CardId>) {
        self.cards.shuffle(&mut thread_rng());
        let n = 5.min(self.cards.len());
        let top: Vec<Card> = self.cards.drain(..n).collect();
        // Re-order per player choice
        let mut ordered = Vec::with_capacity(n);
        let mut remaining = top;
        for id in &ordered_ids {
            if let Some(pos) = remaining.iter().position(|c| c.id == *id) {
                ordered.push(remaining.remove(pos));
            }
        }
        ordered.extend(remaining);
        for c in ordered {
            self.cards.push(c);
        }
    }

    pub fn len(&self) -> usize {
        self.cards.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cards.is_empty()
    }
}
