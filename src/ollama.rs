/// Ollama-backed LLM player.
///
/// Each decision point serialises the full game state (including recent history)
/// as plain text and asks the model to reply with a single number.
///
/// If the model replies with `LOOKUP: <card name>` instead of a number, the
/// effect text is fetched from the compiled-in card database and the model is
/// re-prompted (up to 2 lookup attempts per decision).
use std::sync::Mutex;
use std::time::Duration;

use crate::card::{Card, CardId};
use crate::card_db;
use crate::chain::ChainView;
use crate::player::{
    Action, ChainAction, ChoiceRequest, GameView, OmenAttachment, Player, ProtostarChoice,
    TargetOption, ThinkingCallback,
};

// ── System prompt ─────────────────────────────────────────────────────────────

const SYSTEM: &str = "\
You are playing a card game called Novacana. Make strategic decisions to win.

━━━ WIN CONDITIONS ━━━
A) ASTRONOMICAL DOMINATION: Control 3 complete Solar Systems at end of your turn.
B) COSMIC CATACLYSM: Play all 3 unique Supernova Flare cards simultaneously as one action.

━━━ TURN STRUCTURE ━━━
1. DRAW: Draw 1 card from the deck.
2. ACTION PHASE: If you have 6+ cards in hand, you may play one card or group.
   After each play, if your hand drops below 6, your action phase ends immediately.
   You may only play one Stellar Alignment Flare per turn.
3. END PHASE: Discard down to 5 cards if over the limit.

━━━ STAR CHART ━━━
Stars anchor Solar Systems. A Star's Class value = the number of Discovery cards needed to complete it.
Playing a Star starts a new system. Play Discoveries one at a time onto an incomplete Star.
A complete system counts toward the 3-system win.

STARS (Class → Discoveries needed to complete):
  Brown Dwarf   class 2 - needs 2 discoveries  [3 copies in deck]
  Red Dwarf     class 2 - needs 2 discoveries  [4 copies]
  White Dwarf   class 3 - needs 3 discoveries  [3 copies]
  Yellow Dwarf  class 4 - needs 4 discoveries  [3 copies]
  Red Giant     class 5 - needs 5 discoveries  [3 copies]
  Red Supergiant class 6 - needs 6 discoveries  [2 copies]
  Protostar     (special - see below)           [3 copies]

DISCOVERIES (fill one slot each when attached to a Star):
  Rocky Planet   ×7 - can host Planet of Life Omen
  Gas Giant      ×5 - can be targeted by Ignition and Wormhole specifically
  Asteroid Belt  ×5
  Dwarf Planet   ×5
  Comet          ×5
Total discoveries: 27

━━━ THE CHAIN ━━━
When a Flare, Fracture, or Omen is played it 'enters the Chain'.
After the active player plays one, opponents may respond with their own chainable cards.
The chain closes when all players pass consecutively. It then RESOLVES IN REVERSE ORDER
(last card played resolves first - LIFO).
Cosmic Denial can negate any Flare or Fracture currently in the chain.

━━━ ALL CARD EFFECTS ━━━

── FLARES (positive effects, enter the Chain) ──
Stellar Alignment [×4]
  Draw 2 cards. You may only play one Stellar Alignment per turn.

Wormhole [×2]
  Choose one:
  A) Target opponent fans hand face-down; you take 1 card blind.
  B) Take up to 2 Star/Discovery cards from target opponent's Star Chart into yours.
     You may not take a Star if its Discoveries would be left without a host.
     Attach taken Discoveries legally if possible; otherwise they go to your hand.

Cosmic Denial [×3]
  Negate a Flare or Fracture in the Chain. The negated card is destroyed.

Ignition [×2]
  Choose one:
  A) Destroy target Gas Giant in an opponent's Star Chart, then search deck for a
     Yellow Dwarf and add it to your hand; shuffle deck.
  B) Target opponent reveals hand. If they hold Gas Giants, choose one and place it
     at the bottom of the deck, then search for Yellow Dwarf and add to hand; shuffle.

Astral Projection [×2]
  Target opponent reveals their hand. This effect lasts 2 of their turns.
  While revealed, their hand cards can be targeted as if in a Star Chart
  (e.g., Wormhole can steal from a revealed hand).

Supernova - Gas [×2]
Supernova - Gravity [×2]
Supernova - Fusion [×2]
  You may only play all three Supernovas together as one simultaneous action.
  If you do, you immediately win the game (Cosmic Cataclysm).
  Collect them via Shockwave Fracture.

── FRACTURES (negative effects, enter the Chain) ──
Black Hole [×2]
  Each player shuffles ALL hand AND Star Chart cards into the Main Deck.
  You (the caster) skip your next Action Phase (still draw and discard).
  Omen cards (Solar Eclipse, Planet of Life) remain in play.
  Systems actively protected by Solar Eclipse are completely unaffected.

Pulsar [×2]
  Shuffle YOUR OWN hand AND Star Chart cards into the Main Deck.
  Then draw up to 7 cards. Omen cards remain in play.
  (Unlike Black Hole, only you are affected - useful to reset a bad hand.)

Athena [×3]
  Choose one:
  A) Destroy target Discovery card in an opponent's Star Chart.
  B) If an opponent controls a Gas Giant in their Star Chart: return it to their hand.

Dark Matter Void [×2]
  Choose one Star Class value (2–6). Until the start of YOUR next turn,
  no opponent may play Stars of the chosen class from their hand.

Quantum Entanglement Failure [×2]
  Choose two different players. Each secretly selects one card face-down from hand.
  Both cards are simultaneously revealed and exchanged.

Stellar Nursery Collapse [×2]
  Shuffle the Main Deck. Privately look at the top 5 cards, then place them on the
  bottom in any order you choose. (Useful to bury or set up key cards.)

Shockwave [×2]
  Search the Main Deck for any one Supernova card of your choice and add it to hand.
  Shuffle the deck. (The fastest path to Cosmic Cataclysm.)

── OMENS (permanent attachments, enter the Chain) ──
Solar Eclipse [×4]
  Attach to one of YOUR completed Solar Systems.
  At the start of each of your turns, discard 1 card from hand to maintain protection.
  If you cannot or choose not to pay, Solar Eclipse detaches (goes to bottom of deck).
  While active: that system and ALL cards within it cannot be targeted, stolen, or
  destroyed by ANY card effect. (Black Hole still shuffles unprotected systems.)

Planet of Life [×4]
  Attach to a Rocky Planet in your Star Chart.
  Does NOT count toward system completion.
  If the host Rocky Planet is destroyed, Planet of Life is also destroyed.
  Once per your turn (one-turn cooldown): target an opponent; they reveal their hand.

── SPECIAL STAR ──
Protostar [×3]
  Cannot anchor a Solar System. When played:
  1. Shuffle the Protostar itself back into the Main Deck.
  2. Choose one:
     A) Draw 2 cards.
     B) Search the deck for a White Dwarf (class 3); put it directly into your Star
        Chart as a new incomplete system; shuffle.

━━━ STRATEGY GUIDE ━━━
• Build cheap systems first: Red Dwarf/Brown Dwarf (class 2, needs only 2 discoveries).
• White Dwarf (class 3) is also fast - use Protostar option B to place one for free.
• Collect all 3 Supernovas via Shockwave for the instant-win Cosmic Cataclysm path.
• Use Cosmic Denial to stop Black Hole (which wipes everyone's boards) or Pulsar.
• Solar Eclipse on a complete system makes it near-invincible - prioritise it late game.
• Dark Matter Void on class 2 slows opponents who are rushing cheap stars.
• Astral Projection reveals a hand for 2 turns - combine with Wormhole to steal a
  specific card from the revealed hand.
• Athena destroys Discoveries mid-system, setting opponents back significantly.
• Black Hole resets the whole board - use when you're behind, not ahead.
• Pulsar resets YOUR board only - use it to trade a bad hand/board for a fresh 7.

━━━ CARD LOOKUP ━━━
If you need the exact effect text of any card, respond ONLY with:
LOOKUP: <exact card name>
Example: LOOKUP: Wormhole
You will receive the effect and be asked again. You may look up cards up to 2 times.

━━━ ANSWER FORMAT ━━━
When choosing an action or target: reply with ONLY the number (e.g. 3).
No explanation. No punctuation. Just the number.";

// ── OllamaPlayer ─────────────────────────────────────────────────────────────

pub struct OllamaPlayer {
    pub name: String,
    pub model: String,
    pub base_url: String,
    /// When true, suppresses all eprintln! debug output (use in TUI observer mode).
    pub silent: bool,
    /// HTTP agent with extended timeouts - large models can take several minutes.
    agent: ureq::Agent,
    /// Persistent conversation history. The system prompt is sent once via the
    /// `system` field; this vec holds the alternating user/assistant messages that
    /// accumulate across the entire game so the model remembers previous decisions.
    messages: Mutex<Vec<serde_json::Value>>,
    /// Reasoning text extracted from the last response. Cleared by `take_thinking()`.
    last_thought: Mutex<Option<String>>,
    /// Optional callback receiving raw token chunks as they stream in (for live TUI).
    thinking_cb: Mutex<Option<ThinkingCallback>>,
}

impl OllamaPlayer {
    pub fn new(name: impl Into<String>, model: impl Into<String>) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(10))
            .timeout_read(Duration::from_secs(600))
            .build();
        Self {
            name: name.into(),
            model: model.into(),
            base_url: "http://localhost:52625".into(),
            silent: false,
            agent,
            messages: Mutex::new(Vec::new()),
            last_thought: Mutex::new(None),
            thinking_cb: Mutex::new(None),
        }
    }

    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_silent(mut self) -> Self {
        self.silent = true;
        self
    }

    fn log(&self, msg: &str) {
        if !self.silent {
            eprintln!("[{}] {}", self.name, msg);
        }
    }

    // ── Persistent HTTP chat ──────────────────────────────────────────────────

    /// Split a raw model response into (thought, answer).
    /// Handles `<think>…</think>` blocks produced by DeepSeek-R1 and similar.
    /// The answer is what gets stored in conversation history and parsed for a number.
    /// The thought is returned separately so the caller can log it.
    fn extract_thinking(raw: &str) -> (Option<String>, String) {
        let text = raw.trim();

        // Case 1: complete <think>…</think> block
        if let Some(open) = text.find("<think>") {
            if let Some(close) = text.find("</think>") {
                let thought = text[open + 7..close].trim().to_string();
                let answer = text[close + 8..].trim().to_string();
                return (
                    if thought.is_empty() { None } else { Some(thought) },
                    if answer.is_empty() { "1".to_string() } else { answer },
                );
            }
            // Truncated: <think> with no closing tag - model ran out of token budget.
            // Capture the partial thought; there is no answer to parse.
            let thought = text[open + 7..].trim().to_string();
            return (
                if thought.is_empty() { None } else { Some(thought) },
                "1".to_string(),
            );
        }

        // Case 2: bare </think> with no opening tag (some models omit <think>)
        if let Some(close) = text.find("</think>") {
            let thought = text[..close].trim().to_string();
            let answer = text[close + 8..].trim().to_string();
            return (
                if thought.is_empty() { None } else { Some(thought) },
                if answer.is_empty() { "1".to_string() } else { answer },
            );
        }

        (None, text.to_string())
    }

    /// Send `user_content`, append to persistent history, return raw response.
    /// `history_content` is the compact form stored in history (omits the bulky
    /// game-state preamble so history doesn't balloon past the context window).
    /// Thoughts are extracted and stored in `last_thought`; the clean answer
    /// goes into the message history the model sees on the next call.
    fn chat(&self, user_content: String) -> String {
        self.chat_with_history(user_content.clone(), user_content, 4096)
    }

    /// Stateless single-shot call - no history read or write. Used for cheap yes/no questions.
    fn quick_ask(&self, prompt: &str, num_predict: u32) -> String {
        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                { "role": "system", "content": SYSTEM },
                { "role": "user",   "content": prompt }
            ],
            "stream": false,
            "options": { "temperature": 0.2, "num_predict": num_predict, "num_ctx": 32768 }
        });
        let url = format!("{}/api/chat", self.base_url);
        match self.agent.post(&url).send_json(body) {
            Ok(resp) => match resp.into_json::<serde_json::Value>() {
                Ok(json) => json["message"]["content"].as_str().unwrap_or("2").to_string(),
                Err(e) => { self.log(&format!("JSON parse error: {}", e)); "2".to_string() }
            },
            Err(e) => { self.log(&format!("Ollama unreachable: {}", e)); "2".to_string() }
        }
    }

    fn chat_with_history(&self, user_content: String, history_content: String, num_predict: u32) -> String {
        const MAX_HISTORY: usize = 20;

        let mut msgs = self.messages.lock().unwrap();
        // Push compact form so history stays small; trim old entries.
        msgs.push(serde_json::json!({ "role": "user", "content": history_content }));
        if msgs.len() > MAX_HISTORY {
            let trim = msgs.len() - MAX_HISTORY;
            msgs.drain(0..trim);
        }

        // Build the API message list: system + prior history + full current message.
        // The last entry in msgs is what we just pushed (compact); replace it with the
        // full user_content so the model sees the complete game state this turn.
        let prior = msgs[..msgs.len() - 1].to_vec();
        drop(msgs); // release lock before slow network call

        let mut all: Vec<serde_json::Value> = Vec::with_capacity(prior.len() + 2);
        all.push(serde_json::json!({ "role": "system", "content": SYSTEM }));
        all.extend(prior);
        all.push(serde_json::json!({ "role": "user", "content": user_content }));

        let body = serde_json::json!({
            "model": self.model,
            "messages": all,
            "stream": false,
            "options": {
                "temperature": 0.2,
                "num_predict": num_predict,
                "num_ctx": 32768
            }
        });

        let url = format!("{}/api/chat", self.base_url);
        let raw = match self.agent.post(&url).send_json(body) {
            Ok(resp) => match resp.into_json::<serde_json::Value>() {
                Ok(json) => json["message"]["content"]
                    .as_str()
                    .unwrap_or("1")
                    .to_string(),
                Err(e) => { self.log(&format!("JSON parse error: {}", e)); "1".to_string() }
            },
            Err(e) => { self.log(&format!("Ollama unreachable: {}", e)); "1".to_string() }
        };

        // Fire the full response as a single chunk so the TUI thinking panel updates.
        let cb = self.thinking_cb.lock().unwrap().clone();
        if let Some(ref f) = cb {
            f(raw.clone());
        }

        let (thought, answer) = Self::extract_thinking(&raw);

        // Store thought so the engine can drain it into the game log
        if let Some(t) = thought {
            let mut lt = self.last_thought.lock().unwrap();
            match lt.as_mut() {
                Some(existing) => { existing.push_str("\n---\n"); existing.push_str(&t); }
                None => *lt = Some(t),
            }
        }

        // Store only the clean answer in conversation history (no think tags)
        self.messages.lock().unwrap()
            .push(serde_json::json!({ "role": "assistant", "content": answer }));

        raw
    }

    // ── Response parsing ──────────────────────────────────────────────────────

    fn parse_pick(raw: &str, max: usize) -> usize {
        let (_, answer) = Self::extract_thinking(raw);
        let text = answer;
        for token in text.split_whitespace() {
            let digits: String = token.chars().filter(|c| c.is_ascii_digit()).collect();
            if let Ok(n) = digits.parse::<usize>() {
                if n >= 1 && n <= max { return n - 1; }
            }
        }
        let mut buf = String::new();
        for ch in text.chars() {
            if ch.is_ascii_digit() {
                buf.push(ch);
            } else if !buf.is_empty() {
                if let Ok(n) = buf.parse::<usize>() {
                    if n >= 1 && n <= max { return n - 1; }
                }
                buf.clear();
            }
        }
        0
    }

    // ── Pick with LOOKUP support ──────────────────────────────────────────────

    fn pick(&self, context: &str, prompt: &str, options: &[String]) -> usize {
        self.pick_n(context, prompt, options, 4096)
    }

    fn pick_n(&self, context: &str, prompt: &str, options: &[String], num_predict: u32) -> usize {
        let numbered: String = options
            .iter()
            .enumerate()
            .map(|(i, s)| format!("{}. {}", i + 1, s))
            .collect::<Vec<_>>()
            .join("\n");

        let full_msg = format!(
            "{}\n\n{}\n\n{}\n\nReply with ONLY the number.",
            context, prompt, numbered
        );
        // Store only the compact prompt+options in history, not the bulky game state.
        let compact_msg = format!("{}\n\n{}\n\nReply with ONLY the number.", prompt, numbered);
        let mut raw = self.chat_with_history(full_msg, compact_msg, num_predict);

        // Handle up to 2 LOOKUP requests
        for _ in 0..2 {
            let (_, answer) = Self::extract_thinking(&raw);
            let answer = answer.trim().to_string();
            if let Some(card_name) = answer.strip_prefix("LOOKUP:") {
                let card_name = card_name.trim();
                let info = card_db::lookup_text(card_name)
                    .unwrap_or_else(|| format!("'{}' not found - check spelling.", card_name));
                self.log(&format!("LOOKUP({}) → {}", card_name, info));
                let follow_up = format!(
                    "[Card lookup: {}]\n\nNow choose - reply with ONLY the number (1–{}):",
                    info, options.len()
                );
                raw = self.chat(follow_up);
            } else {
                break;
            }
        }

        let (_, answer) = Self::extract_thinking(&raw);
        self.log(&format!("chose: {:?}", answer.trim()));
        Self::parse_pick(&raw, options.len())
    }

    fn game_context(view: &GameView) -> String {
        view.to_string()
    }
}

// ── Player trait ──────────────────────────────────────────────────────────────

impl Player for OllamaPlayer {
    fn name(&self) -> &str {
        &self.name
    }

    fn take_thinking(&self) -> Option<String> {
        self.last_thought.lock().unwrap().take()
    }

    fn set_thinking_callback(&self, cb: ThinkingCallback) {
        *self.thinking_cb.lock().unwrap() = Some(cb);
    }

    fn choose_action(&self, view: &GameView, legal_actions: &[Action]) -> usize {
        let ctx = Self::game_context(view);
        let descriptions: Vec<String> = legal_actions
            .iter()
            .map(|a| describe_action(a, view))
            .collect();
        self.pick(&ctx, "It is your Action Phase. Choose your action:", &descriptions)
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

        // Step 1: cheap yes/no - does the player even want to respond?
        let card_names: Vec<String> = legal_responses
            .iter()
            .filter_map(|r| view.my_hand.iter().find(|c| c.id == r.card))
            .map(|c| c.to_string())
            .collect();

        let yn_prompt = format!(
            "THE OPEN CHAIN (resolves LIFO):\n{}\n\nYour chain-eligible cards: {}\n\nDo you want to respond?\n1. Yes\n2. No (pass)\n\nReply with ONLY the number.",
            chain,
            card_names.join(", ")
        );
        let yn_raw = self.quick_ask(&yn_prompt, 1024);
        if Self::parse_pick(&yn_raw, 2) != 0 {
            return None;
        }

        // Step 2: full context pick of which card to play.
        let ctx = format!(
            "{}\nTHE OPEN CHAIN (resolves LIFO - last played resolves first):\n{}",
            Self::game_context(view), chain
        );
        let mut opts: Vec<String> = legal_responses
            .iter()
            .filter_map(|r| view.my_hand.iter().find(|c| c.id == r.card))
            .map(|c| format!("Respond with {}", c))
            .collect();
        opts.push("Pass (do not respond to this chain)".into());

        let choice = self.pick_n(&ctx, "Which card do you want to play into the chain?", &opts, 4096);
        if choice == opts.len() - 1 { None } else { Some(choice) }
    }

    fn decide(&self, view: &GameView, request: &ChoiceRequest) -> usize {
        let ctx = Self::game_context(view);
        self.pick(&ctx, &request.prompt, &request.options)
    }

    fn pick_targets(
        &self,
        view: &GameView,
        prompt: &str,
        options: &[TargetOption],
        count: usize,
    ) -> Vec<usize> {
        if options.is_empty() {
            return vec![];
        }
        let ctx = Self::game_context(view);
        let labels: Vec<String> = options.iter().map(|o| o.label.clone()).collect();
        let mut picks = Vec::new();
        for i in 0..count.min(options.len()) {
            let full_prompt = if count > 1 {
                format!("{} (pick {} of {})", prompt, i + 1, count)
            } else {
                prompt.to_string()
            };
            picks.push(self.pick(&ctx, &full_prompt, &labels));
        }
        picks
    }

    fn choose_discards(
        &self,
        view: &GameView,
        hand: &[Card],
        must_discard: usize,
    ) -> Vec<CardId> {
        let ctx = Self::game_context(view);
        let mut discards: Vec<CardId> = Vec::new();
        let mut remaining: Vec<&Card> = hand.iter().collect();

        for i in 0..must_discard {
            if remaining.is_empty() { break; }
            let opts: Vec<String> = remaining.iter().map(|c| c.to_string()).collect();
            let prompt = format!(
                "You must discard {} card(s). Choose card {} of {} to discard.",
                must_discard, i + 1, must_discard
            );
            let choice = self.pick(&ctx, &prompt, &opts);
            discards.push(remaining[choice].id);
            remaining.remove(choice);
        }
        discards
    }
}

// ── Action descriptions ───────────────────────────────────────────────────────

fn describe_action(action: &Action, view: &GameView) -> String {
    let card_name = |id: CardId| {
        view.my_hand
            .iter()
            .find(|c| c.id == id)
            .map(|c| c.to_string())
            .unwrap_or_else(|| format!("Card#{}", id))
    };

    match action {
        Action::Pass => "Pass - end your action phase".into(),

        Action::PlayStar(id) => {
            let class = view.my_hand.iter().find(|c| c.id == *id)
                .and_then(|c| c.star_class())
                .map(|cl| format!(" (class {} - needs {} more discoveries to complete)", cl, cl))
                .unwrap_or_default();
            format!("Place {} in Star Chart{}", card_name(*id), class)
        }

        Action::PlayDiscovery { card, onto_star } => {
            let sys = view.my_star_chart.system_for_star(*onto_star);
            let sys_desc = sys.map(|s| {
                let remaining = s.class as usize - s.discoveries.len();
                format!("{} (class {}, needs {} more after this)", s.star_name, s.class, remaining.saturating_sub(1))
            }).unwrap_or_default();
            format!("Attach {} onto {} system", card_name(*card), sys_desc)
        }

        Action::PlayFlare(id) => format!("Play {} [Flare] - enters Chain", card_name(*id)),

        Action::PlayFracture(id) => format!("Play {} [Fracture] - enters Chain", card_name(*id)),

        Action::PlayOmen { card, attachment } => {
            let where_ = match attachment {
                OmenAttachment::SolarEclipse { system_star } => view
                    .my_star_chart
                    .system_for_star(*system_star)
                    .map(|s| format!("onto COMPLETE {} system [PROTECTS IT]", s.star_name))
                    .unwrap_or_default(),
                OmenAttachment::PlanetOfLife { .. } => "onto Rocky Planet [grants hand-peek ability]".into(),
            };
            format!("Play {} [Omen] → {}", card_name(*card), where_)
        }

        Action::PlaySolarSystem { star, discoveries } => {
            let discs: Vec<String> = discoveries.iter().map(|&id| card_name(id)).collect();
            let complete = view.my_star_chart.complete_systems().len() + 1;
            format!(
                "COMPLETE SYSTEM: {} + [{}] → {}/3 complete (need 3 to win!)",
                card_name(*star), discs.join(", "), complete
            )
        }

        Action::PlaySupernovas { .. } => {
            "*** WIN: Play all 3 Supernovas → COSMIC CATACLYSM - you win immediately! ***".into()
        }

        Action::PlayProtostar { card, choice } => match choice {
            ProtostarChoice::DrawTwo =>
                format!("Play {} → draw 2 cards", card_name(*card)),
            ProtostarChoice::SearchWhiteDwarf =>
                format!("Play {} → place White Dwarf (class 3) directly in Star Chart for free", card_name(*card)),
        },
    }
}
