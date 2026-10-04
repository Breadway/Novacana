use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::Mutex;

use crate::card::{Card, CardId};
use crate::card_db;
use crate::chain::ChainView;
use crate::player::{
    Action, ChainAction, ChoiceRequest, GameView, OmenAttachment, Player, ProtostarChoice,
    TargetOption,
};

// ── Protocol ──────────────────────────────────────────────────────────────────

/// One card in the human's hand, ready for display.
///
/// SERVER PATH: serialised as part of `HumanPrompt` JSON.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HandCardDisplay {
    pub name: String,
    pub card_type: String,
    pub effect: String,
}

/// A prompt sent from the game engine to the human player's UI layer.
///
/// SERVER PATH: serialise as JSON and send over a WebSocket.
/// The remote client renders it and sends back a `HumanResponse`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HumanPrompt {
    pub kind: PromptKind,
    /// Current hand, with effect text pre-resolved from the card DB.
    pub hand_cards: Vec<HandCardDisplay>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum PromptKind {
    ChooseAction { options: Vec<String>, action_types: Vec<Option<String>> },
    RespondToChain { chain: Vec<String>, options: Vec<String> },
    Decide { prompt: String, options: Vec<String> },
    PickTarget { prompt: String, options: Vec<String> },
    ChooseDiscard { options: Vec<String>, step: usize, total: usize },
}

impl PromptKind {
    pub fn options(&self) -> &[String] {
        match self {
            Self::ChooseAction { options, .. } => options,
            Self::RespondToChain { options, .. } => options,
            Self::Decide { options, .. } => options,
            Self::PickTarget { options, .. } => options,
            Self::ChooseDiscard { options, .. } => options,
        }
    }

    /// Whether the player may "pass" (only chain responses).
    pub fn can_pass(&self) -> bool {
        matches!(self, Self::RespondToChain { .. })
    }

    pub fn title(&self) -> String {
        match self {
            Self::ChooseAction { .. } => "YOUR TURN".into(),
            Self::RespondToChain { .. } => "CHAIN OPEN - Respond or pass?".into(),
            Self::Decide { prompt, .. } => prompt.clone(),
            Self::PickTarget { prompt, .. } => prompt.clone(),
            Self::ChooseDiscard { step, total, .. } => {
                format!("DISCARD - choose {} of {}", step, total)
            }
        }
    }
}

/// Response sent from the human UI back to the game engine.
///
/// SERVER PATH: receive as JSON from the WebSocket client.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub enum HumanResponse {
    Pick(usize),
    Pass,
}

fn action_card_type(action: &Action) -> Option<&'static str> {
    match action {
        Action::Pass => None,
        Action::PlayStar(_) | Action::PlayProtostar { .. } | Action::PlaySolarSystem { .. } => Some("Star"),
        Action::PlayDiscovery { .. } => Some("Discovery"),
        Action::PlayFlare(_) | Action::PlaySupernovas { .. } => Some("Flare"),
        Action::PlayFracture(_) => Some("Fracture"),
        Action::PlayOmen { .. } => Some("Omen"),
    }
}

// ── Player ────────────────────────────────────────────────────────────────────

/// A human player that communicates via in-process channels.
///
/// SERVER PATH: Replace `prompt_tx`/`response_rx` with a WebSocket connection.
/// The `Player` trait implementation stays identical - no game-engine changes required.
pub struct HumanPlayer {
    name: String,
    prompt_tx: SyncSender<HumanPrompt>,
    response_rx: Mutex<Receiver<HumanResponse>>,
}

impl HumanPlayer {
    pub fn new(
        name: impl Into<String>,
        prompt_tx: SyncSender<HumanPrompt>,
        response_rx: Receiver<HumanResponse>,
    ) -> Self {
        Self { name: name.into(), prompt_tx, response_rx: Mutex::new(response_rx) }
    }

    fn send(&self, prompt: HumanPrompt) {
        let _ = self.prompt_tx.send(prompt);
    }

    fn recv_pick(&self) -> usize {
        match self.response_rx.lock().unwrap().recv() {
            Ok(HumanResponse::Pick(i)) => i,
            _ => 0,
        }
    }

    fn recv_chain(&self) -> Option<usize> {
        match self.response_rx.lock().unwrap().recv() {
            Ok(HumanResponse::Pick(i)) => Some(i),
            Ok(HumanResponse::Pass) | Err(_) => None,
        }
    }

    fn hand_cards(view: &GameView) -> Vec<HandCardDisplay> {
        view.my_hand.iter().map(|c| {
            let name = c.kind.short_name();
            let card_type = c.kind.type_name().to_string();
            let effect = card_db::lookup(&name)
                .map(|e| e.effect.clone())
                .unwrap_or_default();
            HandCardDisplay { name, card_type, effect }
        }).collect()
    }
}

impl Player for HumanPlayer {
    fn name(&self) -> &str {
        &self.name
    }

    fn choose_action(&self, view: &GameView, legal_actions: &[Action]) -> usize {
        let options = legal_actions.iter().map(|a| describe_action(a, view)).collect();
        let action_types = legal_actions.iter().map(|a| action_card_type(a).map(str::to_string)).collect();
        self.send(HumanPrompt {
            kind: PromptKind::ChooseAction { options, action_types },
            hand_cards: Self::hand_cards(view),
        });
        self.recv_pick().min(legal_actions.len().saturating_sub(1))
    }

    fn respond_to_chain(
        &self,
        view: &GameView,
        chain: &ChainView,
        legal_responses: &[ChainAction],
    ) -> Option<usize> {
        if legal_responses.is_empty() {
            return None;
        }
        let chain_lines = chain.links.iter().map(|l| {
            let neg = if l.negated { " (NEGATED)" } else { "" };
            format!("CL{} [{}] {} - {}{}", l.index + 1, l.player_name, l.card.kind.type_name(), l.card, neg)
        }).collect();
        let options = legal_responses.iter()
            .filter_map(|r| view.my_hand.iter().find(|c| c.id == r.card))
            .map(|c| c.to_string())
            .collect();
        self.send(HumanPrompt {
            kind: PromptKind::RespondToChain { chain: chain_lines, options },
            hand_cards: Self::hand_cards(view),
        });
        let pick = self.recv_chain()?;
        Some(pick.min(legal_responses.len().saturating_sub(1)))
    }

    fn decide(&self, view: &GameView, request: &ChoiceRequest) -> usize {
        self.send(HumanPrompt {
            kind: PromptKind::Decide {
                prompt: request.prompt.clone(),
                options: request.options.clone(),
            },
            hand_cards: Self::hand_cards(view),
        });
        self.recv_pick().min(request.options.len().saturating_sub(1))
    }

    fn pick_targets(
        &self,
        view: &GameView,
        prompt: &str,
        options: &[TargetOption],
        count: usize,
    ) -> Vec<usize> {
        let labels: Vec<String> = options.iter().map(|o| o.label.clone()).collect();
        let mut picked = Vec::new();
        let mut available: Vec<usize> = (0..labels.len()).collect();

        for i in 0..count.min(options.len()) {
            let opts: Vec<String> = available.iter().map(|&j| labels[j].clone()).collect();
            let p = if count > 1 {
                format!("{} ({} of {})", prompt, i + 1, count)
            } else {
                prompt.to_string()
            };
            self.send(HumanPrompt {
                kind: PromptKind::PickTarget { prompt: p, options: opts },
                hand_cards: Self::hand_cards(view),
            });
            let pick = self.recv_pick().min(available.len().saturating_sub(1));
            let original = available.remove(pick);
            picked.push(original);
        }
        picked
    }

    fn choose_discards(&self, view: &GameView, hand: &[Card], must_discard: usize) -> Vec<CardId> {
        let mut remaining: Vec<&Card> = hand.iter().collect();
        let mut result = Vec::new();

        for step in 0..must_discard {
            let options: Vec<String> = remaining.iter().map(|c| c.to_string()).collect();
            self.send(HumanPrompt {
                kind: PromptKind::ChooseDiscard {
                    options,
                    step: step + 1,
                    total: must_discard,
                },
                hand_cards: Self::hand_cards(view),
            });
            let pick = self.recv_pick().min(remaining.len().saturating_sub(1));
            let card_id = remaining[pick].id;
            result.push(card_id);
            remaining.remove(pick);
        }
        result
    }
}

// ── Action descriptions ────────────────────────────────────────────────────────

pub fn describe_action(action: &Action, view: &GameView) -> String {
    let card_name = |id: CardId| -> String {
        view.my_hand.iter().find(|c| c.id == id)
            .map(|c| c.to_string())
            .unwrap_or_default()
    };

    match action {
        Action::Pass => "Pass - end your action phase".into(),

        Action::PlayStar(id) => {
            let cls = view.my_hand.iter().find(|c| c.id == *id)
                .and_then(|c| c.star_class())
                .map(|c| format!(" [class {}]", c))
                .unwrap_or_default();
            format!("Place {}{} in Star Chart", card_name(*id), cls)
        }

        Action::PlayDiscovery { card, onto_star } => {
            let sys = view.my_star_chart.system_for_star(*onto_star);
            let name = sys.map(|s| s.star_name.to_string()).unwrap_or_default();
            let slot = sys.map(|s| {
                format!(" → [{}/{} slots filled]", s.discoveries.len() + 1, s.class)
            }).unwrap_or_default();
            format!("Attach {} to {} system{}", card_name(*card), name, slot)
        }

        Action::PlayFlare(id) => {
            format!("Play {} → opens a Chain (opponents may respond)", card_name(*id))
        }

        Action::PlayFracture(id) => {
            format!("Play {} → opens a Chain (opponents may respond)", card_name(*id))
        }

        Action::PlayOmen { card, attachment } => {
            let desc = match attachment {
                OmenAttachment::SolarEclipse { system_star } => {
                    view.my_star_chart.system_for_star(*system_star)
                        .map(|s| format!("protect {} system", s.star_name))
                        .unwrap_or_else(|| "protect a system".into())
                }
                OmenAttachment::PlanetOfLife { .. } => "attach to Rocky Planet (+1 Discovery slot)".into(),
            };
            format!("Play {} → {}", card_name(*card), desc)
        }

        Action::PlaySolarSystem { star, discoveries } => {
            let discs: Vec<String> = discoveries.iter().map(|&id| card_name(id)).collect();
            format!("★ Complete Solar System: {} + [{}]", card_name(*star), discs.join(", "))
        }

        Action::PlaySupernovas { .. } => {
            "★★★ PLAY ALL 3 SUPERNOVAS - Cosmic Cataclysm - YOU WIN!".into()
        }

        Action::PlayProtostar { card, choice } => match choice {
            ProtostarChoice::DrawTwo => {
                format!("Play {} → draw 2 cards immediately", card_name(*card))
            }
            ProtostarChoice::SearchWhiteDwarf => {
                format!("Play {} → search deck for White Dwarf → place in Star Chart", card_name(*card))
            }
        },
    }
}
