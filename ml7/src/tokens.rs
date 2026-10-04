// ASTRA-7 token schema (ASTRA7.md §3.1).
//
// Every token is `embed(type) + sum(embed(field_i))`. There is one shared
// vocabulary of learned embeddings: one table for `type_id` (TYPE_VOCAB rows)
// and NUM_FIELDS tables for the generic field slots (FIELD_VOCAB rows each,
// reused across token types - the network disambiguates a field's meaning
// from the type embedding it's added to). This keeps the embedding surface
// small (8 tables total) while letting every token category - state, history,
// and *all five* decision kinds folded into one action-token shape - share
// the same encode/decode plumbing.
//
// Sequence layout is FIXED-OFFSET, not densely packed: state tokens always
// occupy positions [0, STATE_MAX), history tokens always occupy
// [STATE_MAX, STATE_MAX + HISTORY_MAX), action tokens always occupy
// [STATE_MAX + HISTORY_MAX, SEQ_LEN). Unused slots within a block are filled
// with Pad tokens. This means the action-token block for every sample in a
// batch starts at the same tensor offset, so net.rs can `narrow()` it out
// without any per-sample gather - and the padding mask for the encoder AND
// the legal-action mask for the pointer head are both directly derivable
// from `type_id == TokenType::Pad`, so EncodedSample doesn't need to carry
// separate mask arrays at all.

use novacana_engine::card::{Card, CardId, CardKind, StarName};
use novacana_engine::player::{Action, ChainAction, ChoiceRequest, GameView, Target, TargetOption};
use novacana_engine::star_chart::SolarSystem;

// ── Sequence layout ────────────────────────────────────────────────────────

pub const NUM_FIELDS: usize = 7;
pub const FIELD_VOCAB: usize = 64;

pub const STATE_GLOBAL: usize = 1;
pub const STATE_HAND_MAX: usize = 12;
pub const STATE_MY_SYS_MAX: usize = 9;
pub const STATE_OPP_MAX: usize = 3;
pub const STATE_OPP_SYS_MAX: usize = 9;
pub const STATE_MAX: usize =
    STATE_GLOBAL + STATE_HAND_MAX + STATE_MY_SYS_MAX + STATE_OPP_MAX * (1 + STATE_OPP_SYS_MAX);
// = 1 + 12 + 9 + 3*10 = 52

pub const HISTORY_MAX: usize = 64;
pub const ACTION_MAX: usize = 64;

pub const SEQ_LEN: usize = STATE_MAX + HISTORY_MAX + ACTION_MAX; // 180

// ── Token type vocabulary ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TokenType {
    Pad = 0,
    Global = 1,
    HandCard = 2,
    MySystem = 3,
    OppSystem = 4,
    Opponent = 5,
    History = 6,
    Action = 7,
}
pub const TYPE_VOCAB: usize = 8;

// ── Action-kind vocabulary (stored in an Action token's field[0]) ────────
//
// Unifies all five decision kinds (choose_action, chain response, decide,
// pick_targets, choose_discards) into one token shape - this is the entire
// point of the pointer-policy design (ASTRA7.md §3.1): the decision *kind*
// lives in the action tokens, not in a separate abstract-action space.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ActionKind {
    Pass = 0,
    PlayStar = 1,
    PlayDiscovery = 2,
    PlaySolarSystem = 3,
    PlayFlare = 4,
    PlayFracture = 5,
    PlayOmenEclipse = 6,
    PlayOmenPlanetOfLife = 7,
    PlaySupernovas = 8,
    PlayProtostarDrawTwo = 9,
    PlayProtostarSearchWhiteDwarf = 10,
    ChainResponsePass = 11,
    ChainResponseCard = 12,
    DecideOption = 13,
    PickTargetOption = 14,
    DiscardCandidate = 15,
}

pub const NUM_ACTION_KINDS: usize = 16;

/// Human-readable name for a raw ActionKind byte (as stored in an action
/// token's fields[0]) - used by the per-kind fidelity diagnostic
/// (ASTRA7.md §1.1/§10: is chain-response fidelity lagging other kinds?).
pub fn action_kind_name(kind: u8) -> &'static str {
    match kind {
        0 => "Pass",
        1 => "PlayStar",
        2 => "PlayDiscovery",
        3 => "PlaySolarSystem",
        4 => "PlayFlare",
        5 => "PlayFracture",
        6 => "PlayOmenEclipse",
        7 => "PlayOmenPlanetOfLife",
        8 => "PlaySupernovas",
        9 => "PlayProtostarDrawTwo",
        10 => "PlayProtostarSearchWhiteDwarf",
        11 => "ChainResponsePass",
        12 => "ChainResponseCard",
        13 => "DecideOption",
        14 => "PickTargetOption",
        15 => "DiscardCandidate",
        _ => "Unknown",
    }
}

// ── Token ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct Token {
    pub ttype: u8,
    pub fields: [u8; NUM_FIELDS],
}

impl Token {
    fn new(t: TokenType) -> Self {
        Self { ttype: t as u8, fields: [0; NUM_FIELDS] }
    }
    fn pad() -> Self {
        Self::new(TokenType::Pad)
    }
}

pub type TokenSeq = [Token; SEQ_LEN];

fn pad_seq() -> TokenSeq {
    [Token::pad(); SEQ_LEN]
}

// ── Field bucket helpers (shared by state, history, and action encoders) ──

fn card_type_field(kind: &CardKind) -> u8 {
    kind.type_idx() as u8 // engine-stable 0..28
}

fn star_type_field(name: StarName) -> u8 {
    match name {
        StarName::Protostar => 0,
        StarName::YellowDwarf => 1,
        StarName::RedGiant => 2,
        StarName::RedSupergiant => 3,
        StarName::WhiteDwarf => 4,
        StarName::RedDwarf => 5,
        StarName::BrownDwarf => 6,
    }
}

// 0 = not applicable/no class (Protostar), 1..5 = class 2..6.
fn class_bucket(class: u8) -> u8 {
    match class {
        2 => 1,
        3 => 2,
        4 => 3,
        5 => 4,
        6 => 5,
        _ => 0,
    }
}

// 0..6: number of discoveries currently filled (clamped to class).
fn fill_bucket(filled: usize, class: u8) -> u8 {
    (filled.min(class as usize)) as u8
}

fn bool_field(b: bool) -> u8 {
    b as u8
}

fn clip_bucket(v: usize, max: usize) -> u8 {
    v.min(max) as u8
}

/// Relative seat: 0 = me, 1..=3 = opponent slot (position in `view.opponents`) + 1.
/// Absolute seat indices are useless across games (v6 bug, ASTRA7.md §3.1) -
/// always resolve through the viewer's own perspective.
fn relative_seat(view: &GameView, player_idx: usize) -> u8 {
    if player_idx == view.my_idx {
        0
    } else {
        view.opponents
            .iter()
            .position(|o| o.idx == player_idx)
            .map(|slot| slot as u8 + 1)
            .unwrap_or(0)
    }
}

fn star_name_to_idx(name: StarName) -> u8 {
    star_type_field(name)
}

// ── State tokens (a set - no positional encoding) ─────────────────────────

pub fn encode_state_tokens(view: &GameView) -> Vec<Token> {
    let mut out = Vec::with_capacity(STATE_MAX);

    // Global
    let mut g = Token::new(TokenType::Global);
    g.fields[0] = clip_bucket(view.deck_size / 10, 9);
    g.fields[1] = clip_bucket(view.turn_number as usize / 20, 9);
    g.fields[2] = view.dark_matter_void.map(class_bucket).unwrap_or(0);
    g.fields[3] = bool_field(view.am_active());
    out.push(g);

    // Hand cards
    for card in view.my_hand.iter().take(STATE_HAND_MAX) {
        let mut t = Token::new(TokenType::HandCard);
        t.fields[0] = card_type_field(&card.kind);
        t.fields[1] = card.star_class().map(class_bucket).unwrap_or(0);
        out.push(t);
    }

    // My systems
    for sys in view.my_star_chart.systems.iter().take(STATE_MY_SYS_MAX) {
        out.push(encode_system_token(sys, TokenType::MySystem, 0));
    }

    // Opponents + their systems
    for (slot, opp) in view.opponents.iter().enumerate().take(STATE_OPP_MAX) {
        let mut t = Token::new(TokenType::Opponent);
        t.fields[0] = slot as u8 + 1;
        t.fields[1] = clip_bucket(opp.hand_size, 10);
        let complete = opp.star_chart.complete_count();
        t.fields[2] = clip_bucket(complete, 3);
        t.fields[3] = bool_field(complete >= 2);
        out.push(t);

        for sys in opp.star_chart.systems.iter().take(STATE_OPP_SYS_MAX) {
            out.push(encode_system_token(sys, TokenType::OppSystem, slot as u8 + 1));
        }
    }

    out
}

fn encode_system_token(sys: &SolarSystem, ttype: TokenType, opp_slot: u8) -> Token {
    let mut t = Token::new(ttype);
    t.fields[0] = star_name_to_idx(sys.star_name);
    t.fields[1] = class_bucket(sys.class);
    t.fields[2] = fill_bucket(sys.discoveries.len(), sys.class);
    t.fields[3] = bool_field(sys.is_eclipse_protected());
    t.fields[4] = bool_field(sys.is_complete());
    t.fields[5] = opp_slot; // 0 = mine
    t
}

// ── History tokens (sequence, most-recent-last, learned position field) ───
//
// `GameView::recent_plays` is `Vec<(usize, CardKind)>`, already capped at the
// last 64 plays (engine/src/game.rs `record_play`) and already includes
// chain-response plays alongside primary actions, oldest-first.
//
// Design decision (documented per ASTRA7.md deliverable-1 instructions):
// this build does NOT extend `record_play`'s signature to add an explicit
// event-type tag. `play_history` doesn't currently distinguish "primary
// play" from "chain response" at the call site, and threading that through
// engine/src/game.rs's ~11 call sites (concurrently edited by another agent
// today) wasn't judged worth the merge risk for v7.0: relative-seat +
// card-type is still real signal, and event_type field[0] is reserved
// (currently always 0 = "play") so a future pass can add the richer
// taxonomy from ASTRA7.md §3.1 (system-complete/destroy/discard-visible/
// omen-attach) purely by widening this encoder - no token-schema migration
// needed downstream.
pub fn encode_history_tokens(view: &GameView) -> Vec<Token> {
    let n = view.recent_plays.len().min(HISTORY_MAX);
    let start = view.recent_plays.len() - n;
    let mut out = Vec::with_capacity(n);
    for (i, (player_idx, kind)) in view.recent_plays[start..].iter().enumerate() {
        let mut t = Token::new(TokenType::History);
        t.fields[0] = 0; // event_type: "play" (only value populated today)
        t.fields[1] = card_type_field(kind);
        t.fields[2] = relative_seat(view, *player_idx);
        t.fields[6] = clip_bucket(i, HISTORY_MAX - 1); // position, oldest..newest
        out.push(t);
    }
    out
}

// ── Decide-prompt taxonomy (reused from v6 candidate.rs keyword buckets) ──

fn prompt_type_bucket(prompt: &str) -> u8 {
    let p = prompt.to_lowercase();
    if p.contains("wormhole") {
        1
    } else if p.contains("ignition") {
        2
    } else if p.contains("athena") {
        3
    } else if p.contains("shockwave") {
        4
    } else if p.contains("cosmic denial") {
        5
    } else if p.contains("dark matter") {
        6
    } else if p.contains("qef") {
        7
    } else if p.contains("solar eclipse") {
        8
    } else if p.contains("secretly choose") {
        9
    } else if p.contains("stellar nursery") {
        10
    } else if p.contains("astral projection") || p.contains("choose target opponent") {
        11
    } else {
        12
    }
}

// 6 boolean keyword flags packed into one 0..63 field.
fn option_flags_bucket(option: &str) -> u8 {
    let o = option.to_lowercase();
    let mut bits: u8 = 0;
    if o.contains("star chart") || o.contains("chart") {
        bits |= 1 << 0;
    }
    if o.contains("destroy") {
        bits |= 1 << 1;
    }
    if o.contains("gas giant") {
        bits |= 1 << 2;
    }
    if o.contains("pay") || o.contains("discard") {
        bits |= 1 << 3;
    }
    if o.contains("supernova") {
        bits |= 1 << 4;
    }
    if o.contains("blind") || o.contains("reveal") {
        bits |= 1 << 5;
    }
    bits
}

// ── Action-token encoders - one per decision kind, all emitting the same
// `Token` shape so the pointer head is decision-kind-agnostic. ─────────────

/// choose_action: one token per legal concrete `Action` - no abstraction
/// layer, unlike v6. Target index == the chosen index into `legal` directly.
pub fn encode_action_tokens_choose_action(view: &GameView, legal: &[Action]) -> Vec<Token> {
    legal.iter().map(|a| encode_one_action(view, a)).collect()
}

// Finds the system containing `card_id` either as its star or as one of its
// attached discoveries - covers both PlayDiscovery's `onto_star` (a star id)
// and PlayOmen::PlanetOfLife's `rocky_planet_disc` (a discovery id).
/// Same lookup, plus a stable position index across (my systems, then each
/// opponent's systems, in order) - disambiguates two legal actions whose
/// target systems bucket identically (same star_name/class/fill) but are
/// actually different systems. Found 2026-07-09 via the AdvCPU-training-
/// distribution collision scan: PlayOmenEclipse/PlayDiscovery collided this
/// way even after the hand-slot-index fix, since that fix only disambiguates
/// WHICH CARD is played, not WHICH TARGET it's played onto.
fn find_system_containing_with_pos<'a>(view: &'a GameView, card_id: CardId) -> Option<(&'a SolarSystem, usize)> {
    let hit = |sys: &&SolarSystem| {
        sys.star_id == card_id || sys.discoveries.iter().any(|d| d.card_id == card_id)
    };
    if let Some(pos) = view.my_star_chart.systems.iter().position(|s| hit(&s)) {
        return Some((&view.my_star_chart.systems[pos], pos));
    }
    let mut base = view.my_star_chart.systems.len();
    for o in &view.opponents {
        if let Some(pos) = o.star_chart.systems.iter().position(|s| hit(&s)) {
            return Some((&o.star_chart.systems[pos], base + pos));
        }
        base += o.star_chart.systems.len();
    }
    None
}

fn hand_card<'a>(view: &'a GameView, id: CardId) -> Option<&'a Card> {
    view.my_hand.iter().find(|c| c.id == id)
}

fn hand_card_slot(view: &GameView, id: CardId) -> Option<usize> {
    view.my_hand.iter().position(|c| c.id == id)
}

// field[6] = hand-slot index of the card this action plays (found
// 2026-07-09 via the token-collision scan, live-play only - every variant
// below previously encoded card TYPE via card_type_field but never card
// IDENTITY, so two legal actions playing different copies of the same card
// type collided into an identical token. 10.4% of live-play decisions hit
// this in testing. field[6] is unused by every variant here, safe to add
// uniformly. Same fix shape as encode_one_target's HandCard branch.
fn encode_one_action(view: &GameView, action: &Action) -> Token {
    let mut t = Token::new(TokenType::Action);
    match action {
        Action::Pass => {
            t.fields[0] = ActionKind::Pass as u8;
        }
        Action::PlayStar(id) => {
            t.fields[0] = ActionKind::PlayStar as u8;
            if let Some(card) = hand_card(view, *id) {
                t.fields[1] = card_type_field(&card.kind);
                if let CardKind::Star(name) = &card.kind {
                    t.fields[2] = star_name_to_idx(*name);
                    if let Some(class) = name.class() {
                        t.fields[3] = class_bucket(class);
                    }
                }
            }
            if let Some(slot) = hand_card_slot(view, *id) {
                t.fields[6] = clip_bucket(slot, FIELD_VOCAB - 1);
            }
        }
        Action::PlayDiscovery { card: id, onto_star } => {
            t.fields[0] = ActionKind::PlayDiscovery as u8;
            if let Some(card) = hand_card(view, *id) {
                t.fields[1] = card_type_field(&card.kind);
            }
            if let Some((sys, pos)) = find_system_containing_with_pos(view, *onto_star) {
                t.fields[2] = star_name_to_idx(sys.star_name);
                t.fields[3] = class_bucket(sys.class);
                t.fields[4] = fill_bucket(sys.discoveries.len(), sys.class);
                // Target-system position (2026-07-09, quantified via the
                // AdvCPU-training-distribution scan): two different systems
                // that happen to bucket identically (same star/class/fill)
                // otherwise collide even with the card identity fixed.
                t.fields[5] = clip_bucket(pos, FIELD_VOCAB - 1);
            }
            if let Some(slot) = hand_card_slot(view, *id) {
                t.fields[6] = clip_bucket(slot, FIELD_VOCAB - 1);
            }
        }
        Action::PlaySolarSystem { star, discoveries } => {
            t.fields[0] = ActionKind::PlaySolarSystem as u8;
            if let Some(card) = hand_card(view, *star) {
                t.fields[1] = card_type_field(&card.kind);
                if let CardKind::Star(name) = &card.kind {
                    t.fields[2] = star_name_to_idx(*name);
                    if let Some(class) = name.class() {
                        t.fields[3] = class_bucket(class);
                    }
                }
            }
            // Attached-discoveries count (2026-07-09, quantified via the
            // AdvCPU-training-distribution scan - this was THE dominant
            // collision source, 77% of all remaining collisions after the
            // card-identity fix): `discoveries: Vec<CardId>` was previously
            // never read at all, so playing the same star with genuinely
            // different discovery sets - a real difference in game outcome -
            // collided into one token.
            t.fields[5] = clip_bucket(discoveries.len(), FIELD_VOCAB - 1);
            if let Some(slot) = hand_card_slot(view, *star) {
                t.fields[6] = clip_bucket(slot, FIELD_VOCAB - 1);
            }
            // Discovery-set signature (2026-07-10): count alone still
            // collides whenever two same-size-but-different discovery
            // combinations are simultaneously legal for the same star -
            // confirmed the dominant residual (~439/568 collisions in the
            // AdvCPU-distribution scan even after the count fix). Repurposes
            // field[4] (previously a redundant duplicate of field[3] - "fill
            // == class" for a simultaneous-complete play, no real signal) as
            // an order-independent hash over the discoveries' card types.
            // Lossy (a hash collision between two genuinely different
            // combos remains theoretically possible) but drastically
            // narrows the collision surface versus count alone - a stronger
            // fix would need per-discovery identity fields, which the
            // current 7-field budget has no room for without dropping
            // something else already in use.
            //
            // 2026-07-10 correction: the original version summed
            // `(type+1) * K` for a fixed constant K - since multiplying by
            // a fixed odd constant mod 2^32 is a bijection, that carried
            // exactly as much information as a plain `sum(type+1)`, no
            // more. Two different combos with equal type-id sums (e.g.
            // types {1,4} vs {2,3}) collided with certainty, not by chance
            // - nowhere near as collision-resistant as a real hash. Fixed
            // by mixing each element's bits (fmix32, Murmur3's finalizer)
            // before summing, so equal-sum coincidences no longer
            // guarantee a collision.
            fn fmix32(mut h: u32) -> u32 {
                h ^= h >> 16;
                h = h.wrapping_mul(0x85ebca6b);
                h ^= h >> 13;
                h = h.wrapping_mul(0xc2b2ae35);
                h ^= h >> 16;
                h
            }
            let sig = discoveries.iter().filter_map(|id| hand_card(view, *id)).fold(0u32, |acc, c| {
                acc.wrapping_add(fmix32(card_type_field(&c.kind) as u32 + 1))
            });
            t.fields[4] = clip_bucket((sig % (FIELD_VOCAB as u32 - 1)) as usize, FIELD_VOCAB - 1);
        }
        Action::PlayFlare(id) => {
            t.fields[0] = ActionKind::PlayFlare as u8;
            if let Some(card) = hand_card(view, *id) {
                t.fields[1] = card_type_field(&card.kind);
            }
            if let Some(slot) = hand_card_slot(view, *id) {
                t.fields[6] = clip_bucket(slot, FIELD_VOCAB - 1);
            }
        }
        Action::PlayFracture(id) => {
            t.fields[0] = ActionKind::PlayFracture as u8;
            if let Some(card) = hand_card(view, *id) {
                t.fields[1] = card_type_field(&card.kind);
            }
            if let Some(slot) = hand_card_slot(view, *id) {
                t.fields[6] = clip_bucket(slot, FIELD_VOCAB - 1);
            }
        }
        Action::PlayOmen { card: id, attachment } => {
            use novacana_engine::player::OmenAttachment;
            t.fields[0] = match attachment {
                OmenAttachment::SolarEclipse { .. } => ActionKind::PlayOmenEclipse as u8,
                OmenAttachment::PlanetOfLife { .. } => ActionKind::PlayOmenPlanetOfLife as u8,
            };
            if let Some(card) = hand_card(view, *id) {
                t.fields[1] = card_type_field(&card.kind);
            }
            let target_card = match attachment {
                OmenAttachment::SolarEclipse { system_star } => *system_star,
                OmenAttachment::PlanetOfLife { rocky_planet_disc } => *rocky_planet_disc,
            };
            if let Some((sys, pos)) = find_system_containing_with_pos(view, target_card) {
                t.fields[2] = star_name_to_idx(sys.star_name);
                t.fields[3] = class_bucket(sys.class);
                t.fields[4] = fill_bucket(sys.discoveries.len(), sys.class);
                // Target-system position - same fix as PlayDiscovery above.
                t.fields[5] = clip_bucket(pos, FIELD_VOCAB - 1);
            }
            if let Some(slot) = hand_card_slot(view, *id) {
                t.fields[6] = clip_bucket(slot, FIELD_VOCAB - 1);
            }
        }
        Action::PlaySupernovas { fusion, gravity, gas } => {
            t.fields[0] = ActionKind::PlaySupernovas as u8;
            if let Some(card) = hand_card(view, *fusion) {
                t.fields[1] = card_type_field(&card.kind);
            }
            if let Some(slot) = hand_card_slot(view, *fusion) {
                t.fields[6] = clip_bucket(slot, FIELD_VOCAB - 1);
            }
            // gravity/gas (2026-07-09): previously never read at all - each
            // Supernova type is unique per ASTRA7.md's own docs (only one
            // copy exists), so this likely never actually collided in
            // practice (matches fidelity's 100% PlaySupernovas accuracy),
            // but the encoding was still silently dropping real action data.
            if let Some(slot) = hand_card_slot(view, *gravity) {
                t.fields[2] = clip_bucket(slot, FIELD_VOCAB - 1);
            }
            if let Some(slot) = hand_card_slot(view, *gas) {
                t.fields[3] = clip_bucket(slot, FIELD_VOCAB - 1);
            }
        }
        Action::PlayProtostar { card: id, choice } => {
            use novacana_engine::player::ProtostarChoice;
            t.fields[0] = match choice {
                ProtostarChoice::DrawTwo => ActionKind::PlayProtostarDrawTwo as u8,
                ProtostarChoice::SearchWhiteDwarf => ActionKind::PlayProtostarSearchWhiteDwarf as u8,
            };
            if let Some(card) = hand_card(view, *id) {
                t.fields[1] = card_type_field(&card.kind);
            }
            if let Some(slot) = hand_card_slot(view, *id) {
                t.fields[6] = clip_bucket(slot, FIELD_VOCAB - 1);
            }
        }
    }
    t
}

/// respond_to_chain: index 0 is always an explicit "pass" token (pass is
/// always legal for a chain response); indices 1..=len(legal) map 1:1 to
/// `legal`. Caller target index: `response.map(|i| i + 1).unwrap_or(0)`.
pub fn encode_action_tokens_chain(view: &GameView, legal: &[ChainAction]) -> Vec<Token> {
    let mut out = Vec::with_capacity(legal.len() + 1);
    let mut pass = Token::new(TokenType::Action);
    pass.fields[0] = ActionKind::ChainResponsePass as u8;
    out.push(pass);
    for ca in legal {
        let mut t = Token::new(TokenType::Action);
        t.fields[0] = ActionKind::ChainResponseCard as u8;
        if let Some(card) = hand_card(view, ca.card) {
            t.fields[1] = card_type_field(&card.kind);
        }
        // Hand-slot index (2026-07-09, same fix as encode_one_action): same
        // card-identity gap, just not hit in the live-play collision scan's
        // sample - duplicate-type cards rarely co-occur as chain response
        // options, but the same bug is there if they do.
        if let Some(slot) = hand_card_slot(view, ca.card) {
            t.fields[6] = clip_bucket(slot, FIELD_VOCAB - 1);
        }
        out.push(t);
    }
    out
}

/// decide: one token per `ChoiceRequest::options` entry, in order.
///
/// Fields 1-2 (slot index, option count) were added 2026-07-09 after the
/// per-ActionKind fidelity diagnostic (ASTRA7.md §1.1/§10) found DecideOption
/// at 60.0% accuracy on the single largest bucket (43% of all samples) -
/// far below every other kind. Root cause: without a positional field, two
/// options in the same decide() call that don't trip any of
/// `option_flags_bucket`'s 6 keyword checks produce BYTE-IDENTICAL tokens -
/// an information bottleneck no amount of data or training can fix. v6 hit
/// this exact problem and every one of its candidate encoders
/// (candidate.rs's `set_pos`) always carried a slot/total pair for exactly
/// this reason; this encoder was the one place that pattern didn't make it
/// into v7. `view` remains otherwise unused here - decide() prompts don't
/// currently need board-state context beyond the prompt/option text.
pub fn encode_action_tokens_decide(view: &GameView, req: &ChoiceRequest) -> Vec<Token> {
    let _ = view;
    let total = req.options.len();
    req.options
        .iter()
        .enumerate()
        .map(|(slot, opt)| {
            let mut t = Token::new(TokenType::Action);
            t.fields[0] = ActionKind::DecideOption as u8;
            t.fields[1] = clip_bucket(slot, FIELD_VOCAB - 1);
            t.fields[2] = clip_bucket(total, FIELD_VOCAB - 1);
            t.fields[5] = prompt_type_bucket(&req.prompt);
            t.fields[6] = option_flags_bucket(opt);
            t
        })
        .collect()
}

/// pick_targets: one token per `TargetOption`, in order. Multi-pick
/// (count>1) decisions are trained greedy: only the first chosen target is
/// used as the pointer-CE label (ASTRA7.md §3.2/§9 - "start greedy", the
/// same simplification v6's candidate head used for multi-select).
pub fn encode_action_tokens_targets(view: &GameView, options: &[TargetOption]) -> Vec<Token> {
    options.iter().map(|opt| encode_one_target(view, &opt.target)).collect()
}

fn encode_one_target(view: &GameView, target: &Target) -> Token {
    let mut t = Token::new(TokenType::Action);
    t.fields[0] = ActionKind::PickTargetOption as u8;
    match target {
        Target::Player(idx) => {
            t.fields[5] = 0; // target_kind: player
            t.fields[6] = relative_seat(view, *idx);
        }
        Target::StarChartCard { player_idx, card_id } => {
            t.fields[5] = 1; // target_kind: star chart card
            t.fields[6] = relative_seat(view, *player_idx);
            let chart = if *player_idx == view.my_idx {
                Some(&view.my_star_chart)
            } else {
                view.opponents.iter().find(|o| o.idx == *player_idx).map(|o| &o.star_chart)
            };
            if let Some(chart) = chart {
                if let Some(sys) = chart.systems.iter().find(|s| {
                    s.star_id == *card_id || s.discoveries.iter().any(|d| d.card_id == *card_id)
                }) {
                    t.fields[2] = star_name_to_idx(sys.star_name);
                    t.fields[3] = class_bucket(sys.class);
                    t.fields[4] = fill_bucket(sys.discoveries.len(), sys.class);
                    // Card-within-system slot (found 2026-07-09 via the fidelity
                    // diagnostic + independent review): without this, every
                    // discovery in the same system (and the system's own star,
                    // in chart_target_options) encodes to the identical token -
                    // same root cause as the DecideOption collision (§10.1).
                    // 0 = the star itself, 1+ = discovery position in list order.
                    t.fields[1] = if sys.star_id == *card_id {
                        0
                    } else {
                        clip_bucket(
                            1 + sys.discoveries.iter().position(|d| d.card_id == *card_id).unwrap_or(0),
                            FIELD_VOCAB - 1,
                        )
                    };
                }
            }
        }
        Target::HandCard { player_idx, card_id } => {
            t.fields[5] = 2; // target_kind: hand card
            t.fields[6] = relative_seat(view, *player_idx);
            let found = if *player_idx == view.my_idx {
                view.my_hand.iter().enumerate().find(|(_, c)| c.id == *card_id)
            } else {
                view.opponents
                    .iter()
                    .find(|o| o.idx == *player_idx)
                    .and_then(|o| o.hand_revealed.iter().enumerate().find(|(_, c)| c.id == *card_id))
            };
            if let Some((pos, card)) = found {
                t.fields[1] = card_type_field(&card.kind);
                // Hand-slot position (2026-07-09, same fix as StarChartCard
                // above): duplicate-type hand cards otherwise collide.
                t.fields[2] = clip_bucket(pos, FIELD_VOCAB - 1);
            }
        }
        Target::StarClass(cls) => {
            t.fields[5] = 3; // target_kind: star class value
            t.fields[3] = class_bucket(*cls);
        }
        Target::ChainLink(idx) => {
            t.fields[5] = 4; // target_kind: chain link
            t.fields[6] = clip_bucket(*idx, FIELD_VOCAB - 1);
        }
    }
    t
}

/// choose_discards: one token per hand card, in order. Multi-discard is
/// trained greedy (same simplification as pick_targets - see there).
pub fn encode_action_tokens_discards(hand: &[Card]) -> Vec<Token> {
    hand.iter()
        .enumerate()
        .map(|(slot, card)| {
            let mut t = Token::new(TokenType::Action);
            t.fields[0] = ActionKind::DiscardCandidate as u8;
            t.fields[1] = card_type_field(&card.kind);
            t.fields[2] = card.star_class().map(class_bucket).unwrap_or(0);
            // Hand-slot index (2026-07-09, same fix as encode_one_action):
            // duplicate-type hand cards otherwise collide as discard options.
            t.fields[6] = clip_bucket(slot, FIELD_VOCAB - 1);
            t
        })
        .collect()
}

// ── Assembly ────────────────────────────────────────────────────────────

/// A fully assembled, fixed-length token sequence for one decision, plus the
/// number of real (non-pad) action tokens - i.e. the size of the legal set
/// the pointer head must classify over for this decision.
pub struct EncodedDecision {
    pub tokens: TokenSeq,
    pub n_actions: usize,
}

/// Assemble state + history + action tokens into the fixed SEQ_LEN layout.
/// Returns `None` if `action_tokens` doesn't fit in ACTION_MAX (should be
/// exceedingly rare given the cap's headroom over v6's measured worst case -
/// callers should skip such a sample rather than corrupt the pointer target).
pub fn assemble(view: &GameView, action_tokens: &[Token]) -> Option<EncodedDecision> {
    if action_tokens.is_empty() || action_tokens.len() > ACTION_MAX {
        return None;
    }
    let mut seq = pad_seq();

    let state = encode_state_tokens(view);
    debug_assert!(state.len() <= STATE_MAX);
    for (i, t) in state.iter().take(STATE_MAX).enumerate() {
        seq[i] = *t;
    }

    let history = encode_history_tokens(view);
    debug_assert!(history.len() <= HISTORY_MAX);
    for (i, t) in history.iter().take(HISTORY_MAX).enumerate() {
        seq[STATE_MAX + i] = *t;
    }

    let action_base = STATE_MAX + HISTORY_MAX;
    for (i, t) in action_tokens.iter().enumerate() {
        seq[action_base + i] = *t;
    }

    Some(EncodedDecision { tokens: seq, n_actions: action_tokens.len() })
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use novacana_engine::card::{CardKind, DiscoveryName, FlareName, StarName};
    use novacana_engine::star_chart::StarChart;

    fn empty_view(my_idx: usize) -> GameView {
        GameView {
            my_idx,
            my_name: "Me".into(),
            my_hand: Vec::new(),
            my_star_chart: StarChart::new(),
            opponents: Vec::new(),
            deck_size: 90,
            active_player_idx: my_idx,
            turn_number: 0,
            dark_matter_void: None,
            history: Vec::new(),
            recent_plays: Vec::new(),
            discard_pile_counts: [0; 29],
        }
    }

    fn card(id: CardId, kind: CardKind) -> Card {
        Card::new(id, kind)
    }

    #[test]
    fn empty_hand_no_panic() {
        let view = empty_view(0);
        let state = encode_state_tokens(&view);
        assert!(state.len() <= STATE_MAX);
        let history = encode_history_tokens(&view);
        assert!(history.is_empty());
        // choose_action with only Pass legal (empty hand < 6 cards case).
        let legal = vec![Action::Pass];
        let actions = encode_action_tokens_choose_action(&view, &legal);
        assert_eq!(actions.len(), 1);
        let enc = assemble(&view, &actions).expect("assemble should succeed");
        assert_eq!(enc.n_actions, 1);
    }

    #[test]
    fn zero_legal_actions_returns_none() {
        let view = empty_view(0);
        let actions: Vec<Token> = Vec::new();
        assert!(assemble(&view, &actions).is_none());
    }

    #[test]
    fn max_hand_size_no_panic() {
        let mut view = empty_view(0);
        for i in 0..20u32 {
            view.my_hand.push(card(i, CardKind::Star(StarName::YellowDwarf)));
        }
        let state = encode_state_tokens(&view);
        // Global + hand cards capped at STATE_HAND_MAX (rest silently dropped).
        assert_eq!(state.len(), 1 + STATE_HAND_MAX);
    }

    #[test]
    fn action_token_count_matches_legal_action_count() {
        let mut view = empty_view(0);
        view.my_hand.push(card(1, CardKind::Star(StarName::YellowDwarf)));
        view.my_hand.push(card(2, CardKind::Discovery(DiscoveryName::Comet)));
        view.my_hand.push(card(3, CardKind::Flare(FlareName::Wormhole)));
        let legal = vec![
            Action::Pass,
            Action::PlayStar(1),
            Action::PlayFlare(3),
        ];
        let actions = encode_action_tokens_choose_action(&view, &legal);
        assert_eq!(actions.len(), legal.len());
        for (tok, act) in actions.iter().zip(legal.iter()) {
            match act {
                Action::Pass => assert_eq!(tok.fields[0], ActionKind::Pass as u8),
                Action::PlayStar(_) => assert_eq!(tok.fields[0], ActionKind::PlayStar as u8),
                Action::PlayFlare(_) => assert_eq!(tok.fields[0], ActionKind::PlayFlare as u8),
                _ => unreachable!(),
            }
        }
        let enc = assemble(&view, &actions).unwrap();
        assert_eq!(enc.n_actions, 3);
        // Action block padding beyond n_actions must be Pad.
        for i in 3..ACTION_MAX {
            assert_eq!(enc.tokens[STATE_MAX + HISTORY_MAX + i].ttype, TokenType::Pad as u8);
        }
    }

    #[test]
    fn chain_response_pass_is_index_zero() {
        let view = empty_view(0);
        let legal: Vec<ChainAction> = Vec::new();
        let actions = encode_action_tokens_chain(&view, &legal);
        assert_eq!(actions.len(), 1); // just the pass token
        assert_eq!(actions[0].fields[0], ActionKind::ChainResponsePass as u8);
    }

    #[test]
    fn discard_candidates_match_hand() {
        let mut hand = Vec::new();
        for i in 0..7u32 {
            hand.push(card(i, CardKind::Discovery(DiscoveryName::Comet)));
        }
        let actions = encode_action_tokens_discards(&hand);
        assert_eq!(actions.len(), hand.len());
    }

    #[test]
    fn history_truncates_to_max_and_preserves_recency_order() {
        let mut view = empty_view(0);
        for i in 0..100u32 {
            view.recent_plays.push((0, CardKind::Star(StarName::YellowDwarf)));
            let _ = i;
        }
        let hist = encode_history_tokens(&view);
        assert_eq!(hist.len(), HISTORY_MAX);
        // Position field should be strictly increasing (oldest-kept .. newest).
        for w in hist.windows(2) {
            assert!(w[0].fields[6] < w[1].fields[6]);
        }
    }

    #[test]
    fn relative_seat_is_viewer_relative_not_absolute() {
        use novacana_engine::player::OpponentView;
        let mut view = empty_view(2); // I am seat 2
        view.opponents.push(OpponentView {
            idx: 0,
            name: "A".into(),
            hand_size: 5,
            hand_revealed: Vec::new(),
            star_chart: StarChart::new(),
        });
        view.opponents.push(OpponentView {
            idx: 3,
            name: "B".into(),
            hand_size: 5,
            hand_revealed: Vec::new(),
            star_chart: StarChart::new(),
        });
        assert_eq!(relative_seat(&view, 2), 0); // me
        assert_eq!(relative_seat(&view, 0), 1); // opponents[0]
        assert_eq!(relative_seat(&view, 3), 2); // opponents[1]
    }

    #[test]
    fn field_values_stay_within_vocab() {
        let mut view = empty_view(0);
        for i in 0..20u32 {
            view.my_hand.push(card(i, CardKind::Star(StarName::YellowDwarf)));
        }
        for _ in 0..5 {
            view.my_star_chart.place_star(100, StarName::RedSupergiant, 6);
        }
        let seq = encode_state_tokens(&view);
        for t in &seq {
            assert!((t.ttype as usize) < TYPE_VOCAB);
            for f in t.fields {
                assert!((f as usize) < FIELD_VOCAB, "field {} out of vocab", f);
            }
        }
    }

    #[test]
    fn sequence_length_is_fixed() {
        let view = empty_view(0);
        let actions = vec![Token::new(TokenType::Action)];
        let enc = assemble(&view, &actions).unwrap();
        assert_eq!(enc.tokens.len(), SEQ_LEN);
    }
}
