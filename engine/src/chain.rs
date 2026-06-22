use std::fmt;

use crate::card::{Card, FlareName};
use crate::player::Target;

// ── Chain link ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ChainLink {
    pub card: Card,
    pub player_idx: usize,
    pub targets: Vec<Target>,
    pub negated: bool,
}

impl ChainLink {
    pub fn new(card: Card, player_idx: usize, targets: Vec<Target>) -> Self {
        Self { card, player_idx, targets, negated: false }
    }
}

// ── Chain ─────────────────────────────────────────────────────────────────────

/// The resolution queue for reactive plays.
/// Links are resolved in LIFO order (last added = first resolved).
#[derive(Debug, Default)]
pub struct Chain {
    pub links: Vec<ChainLink>,
}

impl Chain {
    pub fn new(first_link: ChainLink) -> Self {
        Self { links: vec![first_link] }
    }

    pub fn push(&mut self, link: ChainLink) {
        self.links.push(link);
    }

    /// Negate (but do not remove) a link by index.
    pub fn negate(&mut self, link_idx: usize) {
        if link_idx < self.links.len() {
            self.links[link_idx].negated = true;
        }
    }

    /// Indices in LIFO resolution order (last to first).
    pub fn resolution_order(&self) -> Vec<usize> {
        (0..self.links.len()).rev().collect()
    }

    /// Find all non-negated Flare or Fracture links that can be targeted by Cosmic Denial.
    pub fn deniable_links(&self) -> Vec<usize> {
        self.links
            .iter()
            .enumerate()
            .filter(|(_, l)| {
                !l.negated
                    && matches!(
                        &l.card.kind,
                        crate::card::CardKind::Flare(_) | crate::card::CardKind::Fracture(_)
                    )
                    && !matches!(&l.card.kind, crate::card::CardKind::Flare(f) if f.is_supernova())
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// True if the chain contains the three unique Supernovas (all non-negated).
    pub fn has_winning_supernovas(&self) -> bool {
        let active_supernovas: Vec<FlareName> = self
            .links
            .iter()
            .filter(|l| !l.negated && l.card.is_supernova())
            .filter_map(|l| {
                if let crate::card::CardKind::Flare(f) = &l.card.kind {
                    Some(*f)
                } else {
                    None
                }
            })
            .collect();
        active_supernovas.contains(&FlareName::SupernovaFusion)
            && active_supernovas.contains(&FlareName::SupernovaGravity)
            && active_supernovas.contains(&FlareName::SupernovaGas)
    }

    pub fn len(&self) -> usize {
        self.links.len()
    }

    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }
}

impl fmt::Display for Chain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  Chain ({} link{}):", self.links.len(), if self.links.len() == 1 { "" } else { "s" })?;
        for (i, link) in self.links.iter().enumerate() {
            let negated = if link.negated { " [NEGATED]" } else { "" };
            writeln!(f, "    CL{}: {} (player {}){}", i + 1, link.card, link.player_idx, negated)?;
        }
        Ok(())
    }
}

// ── Lightweight view for players ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ChainView {
    pub links: Vec<ChainLinkView>,
}

#[derive(Debug, Clone)]
pub struct ChainLinkView {
    pub index: usize,
    pub player_name: String,
    pub card: Card,
    pub negated: bool,
}

impl From<&Chain> for ChainView {
    fn from(chain: &Chain) -> Self {
        ChainView {
            links: chain
                .links
                .iter()
                .enumerate()
                .map(|(i, l)| ChainLinkView {
                    index: i,
                    player_name: format!("Player {}", l.player_idx),
                    card: l.card.clone(),
                    negated: l.negated,
                })
                .collect(),
        }
    }
}

impl fmt::Display for ChainView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for link in &self.links {
            let negated = if link.negated { " [NEGATED]" } else { "" };
            writeln!(f, "  CL{}: {} played by {}{}", link.index + 1, link.card, link.player_name, negated)?;
        }
        Ok(())
    }
}
