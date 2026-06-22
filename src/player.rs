use std::fmt;

use crate::card::{Card, CardId};
use crate::chain::ChainView;
use crate::star_chart::StarChart;

// ── Targets ───────────────────────────────────────────────────────────────────

/// A card or player that can be the target of an effect.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Target {
    /// A specific opponent player.
    Player(usize),
    /// A card in someone's Star Chart.
    StarChartCard { player_idx: usize, card_id: CardId },
    /// A card in an opponent's hand (only valid when hand is revealed via Astral Projection).
    HandCard { player_idx: usize, card_id: CardId },
    /// A Star class value (for Dark Matter Void).
    StarClass(u8),
    /// A specific chain link by index (for Cosmic Denial targeting).
    ChainLink(usize),
}

// ── Actions ───────────────────────────────────────────────────────────────────

/// What a player does during their Action Phase.
#[derive(Debug, Clone)]
pub enum Action {
    /// End the action phase voluntarily (or forced when hand ≤ 5).
    Pass,

    /// Place a Star into the Star Chart.
    PlayStar(CardId),

    /// Attach a Discovery to an incomplete Star.
    PlayDiscovery { card: CardId, onto_star: CardId },

    /// Play a Flare into the Chain (targets resolved during effect).
    PlayFlare(CardId),

    /// Play a Fracture into the Chain (targets resolved during effect).
    PlayFracture(CardId),

    /// Attach an Omen card.
    PlayOmen { card: CardId, attachment: OmenAttachment },

    /// Play a complete Solar System simultaneously (star + N discoveries).
    PlaySolarSystem { star: CardId, discoveries: Vec<CardId> },

    /// Win condition: play all three unique Supernovas at once.
    PlaySupernovas { fusion: CardId, gravity: CardId, gas: CardId },

    /// Protostar special: plays itself then chooses A or B.
    PlayProtostar { card: CardId, choice: ProtostarChoice },
}

#[derive(Debug, Clone)]
pub enum OmenAttachment {
    /// Solar Eclipse attaches to a completed Solar System identified by its star card.
    SolarEclipse { system_star: CardId },
    /// Planet of Life attaches to a Rocky Planet Discovery.
    PlanetOfLife { rocky_planet_disc: CardId },
}

#[derive(Debug, Clone)]
pub enum ProtostarChoice {
    /// Option A: draw 2 cards.
    DrawTwo,
    /// Option B: search Main Deck for a White Dwarf and place it into Star Chart.
    SearchWhiteDwarf,
}

/// A card played reactively into an open Chain.
#[derive(Debug, Clone)]
pub struct ChainAction {
    pub card: CardId,
    pub targets: Vec<Target>,
    /// If the reactive card is an Omen, where it attaches.
    pub omen_attachment: Option<OmenAttachment>,
}

// ── Decision requests ─────────────────────────────────────────────────────────

/// Mid-effect prompt that requires the player to pick from a numbered list.
#[derive(Debug, Clone)]
pub struct ChoiceRequest {
    pub prompt: String,
    /// Each option is a human-readable description.
    pub options: Vec<String>,
    /// How many to pick.
    pub pick: usize,
}

// ── Views ─────────────────────────────────────────────────────────────────────

/// What a player can observe about an opponent.
#[derive(Debug, Clone)]
pub struct OpponentView {
    pub idx: usize,
    pub name: String,
    pub hand_size: usize,
    /// Non-empty only when the player is under Astral Projection.
    pub hand_revealed: Vec<Card>,
    pub star_chart: StarChart,
}

/// The full observable game state from one player's perspective.
#[derive(Debug, Clone)]
pub struct GameView {
    pub my_idx: usize,
    pub my_name: String,
    pub my_hand: Vec<Card>,
    pub my_star_chart: StarChart,
    pub opponents: Vec<OpponentView>,
    pub deck_size: usize,
    pub active_player_idx: usize,
    pub turn_number: u32,
    /// If Some, this Star class cannot be played by any opponent until stated player's next turn.
    pub dark_matter_void: Option<u8>,
    /// Recent game log entries, oldest first. Shows what happened this game.
    pub history: Vec<String>,
}

impl GameView {
    pub fn am_active(&self) -> bool {
        self.my_idx == self.active_player_idx
    }
}

impl fmt::Display for GameView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // ── Recent history ────────────────────────────────────────────────────
        if !self.history.is_empty() {
            writeln!(f, "=== RECENT GAME EVENTS ===")?;
            for line in &self.history {
                writeln!(f, "  {}", line)?;
            }
        }

        // ── Turn / global state ───────────────────────────────────────────────
        writeln!(f, "=== GAME STATE (Turn {}) ===", self.turn_number)?;
        let active_name = if self.am_active() {
            "YOU".to_string()
        } else {
            self.opponents.iter()
                .find(|o| o.idx == self.active_player_idx)
                .map(|o| o.name.clone())
                .unwrap_or_else(|| "unknown".into())
        };
        writeln!(f, "Active player: {} | Deck: {} cards remaining", active_name, self.deck_size)?;

        if let Some(class) = self.dark_matter_void {
            writeln!(f, "DARK MATTER VOID: Class {} Stars cannot be played until the caster's next turn.", class)?;
        }

        // ── Your hand ─────────────────────────────────────────────────────────
        writeln!(f, "\nYOUR HAND [{} cards] (you are: {}):", self.my_hand.len(), self.my_name)?;
        if self.my_hand.is_empty() {
            writeln!(f, "  (empty)")?;
        } else {
            for (i, card) in self.my_hand.iter().enumerate() {
                writeln!(f, "  {}. {}", i + 1, card)?;
            }
        }

        // ── Your Star Chart ───────────────────────────────────────────────────
        let my_complete = self.my_star_chart.complete_systems().len();
        writeln!(f, "\nYOUR STAR CHART ({}/3 complete systems):", my_complete)?;
        if self.my_star_chart.systems.is_empty() {
            writeln!(f, "  (no systems placed yet)")?;
        } else {
            for sys in &self.my_star_chart.systems {
                let complete = sys.discoveries.len() >= sys.class as usize;
                let status = if complete { "COMPLETE" } else {
                    &format!("needs {} more discovery/ies", sys.class as usize - sys.discoveries.len())
                };
                let eclipse = if sys.eclipse_id.is_some() { " [Solar Eclipse — PROTECTED]" } else { "" };
                writeln!(f, "  • {} [class {}] {}/{} slots filled — {}{}",
                    sys.star_name, sys.class, sys.discoveries.len(), sys.class, status, eclipse)?;
                for disc in &sys.discoveries {
                    let pol = if disc.planet_of_life_id.is_some() { " [Planet of Life attached]" } else { "" };
                    writeln!(f, "      – {}{}", disc.name, pol)?;
                }
            }
        }
        if my_complete >= 2 {
            writeln!(f, "  *** You need {} more complete system(s) to win! ***", 3 - my_complete)?;
        }

        // ── Opponents ─────────────────────────────────────────────────────────
        for opp in &self.opponents {
            let opp_complete = opp.star_chart.complete_systems().len();
            let danger = if opp_complete >= 2 { " ⚠ ONE SYSTEM AWAY FROM WINNING" } else { "" };
            writeln!(f, "\nOPPONENT: {} | Hand: {} cards | {}/3 complete systems{}",
                opp.name, opp.hand_size, opp_complete, danger)?;

            if !opp.hand_revealed.is_empty() {
                writeln!(f, "  [HAND REVEALED via Astral Projection]:")?;
                for card in &opp.hand_revealed {
                    writeln!(f, "    – {}", card)?;
                }
            }

            if opp.star_chart.systems.is_empty() {
                writeln!(f, "  Star Chart: (empty)")?;
            } else {
                for sys in &opp.star_chart.systems {
                    let complete = sys.discoveries.len() >= sys.class as usize;
                    let eclipse = if sys.eclipse_id.is_some() { " [PROTECTED]" } else { "" };
                    writeln!(f, "  • {} [class {}] {}/{} — {}{}",
                        sys.star_name, sys.class, sys.discoveries.len(), sys.class,
                        if complete { "COMPLETE" } else { "incomplete" }, eclipse)?;
                }
            }
        }

        writeln!(f, "\n=== END OF STATE ===")?;
        Ok(())
    }
}

// ── Player trait ──────────────────────────────────────────────────────────────

/// Callback type for streaming token chunks from a live-generating player to an observer.
pub type ThinkingCallback = std::sync::Arc<dyn Fn(String) + Send + Sync>;

/// Implement this trait to create a player (human CLI, random bot, LLM, etc.).
///
/// All methods receive a `&GameView` so the implementation has full context.
/// For an LLM player, `GameView::to_string()` is the natural prompt preamble.
///
/// Future Ollama integration: wrap this trait with `async_trait` and replace
/// the synchronous body with an async HTTP call to the Ollama `/api/chat` endpoint.
pub trait Player: Send + Sync {
    fn name(&self) -> &str;

    /// Choose an action during the Action Phase.
    /// `legal_actions` is a pre-validated list — the player must return one of them.
    fn choose_action(&self, view: &GameView, legal_actions: &[Action]) -> usize;

    /// Decide whether to add a link to the open Chain.
    /// Returns `Some(index)` into `legal_responses`, or `None` to pass.
    fn respond_to_chain(
        &self,
        view: &GameView,
        chain: &ChainView,
        legal_responses: &[ChainAction],
    ) -> Option<usize>;

    /// Pick one option from a numbered list of choices (mid-effect decision point).
    fn decide(&self, view: &GameView, request: &ChoiceRequest) -> usize;

    /// Pick `count` targets from the supplied list.
    fn pick_targets(&self, view: &GameView, prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize>;

    /// Select which cards to discard to reach the 5-card hand limit.
    fn choose_discards(&self, view: &GameView, hand: &[Card], must_discard: usize) -> Vec<CardId>;

    /// Return and clear any reasoning text the player generated for its last
    /// decision. Called by the game engine immediately after each decision so
    /// the thought appears in the game log. Default: no reasoning.
    fn take_thinking(&self) -> Option<String> { None }

    /// Register a callback that receives streaming token chunks during generation.
    /// Called once by the game engine when a TUI observer is attached.
    fn set_thinking_callback(&self, _cb: ThinkingCallback) {}
}

/// A labelled target option presented to the player.
#[derive(Debug, Clone)]
pub struct TargetOption {
    pub label: String,
    pub target: Target,
}

impl TargetOption {
    pub fn new(label: impl Into<String>, target: Target) -> Self {
        Self { label: label.into(), target }
    }
}
