use crate::card::{Card, CardId, CardKind, FlareName, FractureName, OmenName};
use crate::chain::{ChainLinkView, ChainView};
use crate::player::{
    Action, ChainAction, ChoiceRequest, GameView, OpponentView, Player, ProtostarChoice, Target,
    TargetOption,
};

pub struct CpuPlayer {
    name: String,
}

impl CpuPlayer {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl Player for CpuPlayer {
    fn name(&self) -> &str {
        &self.name
    }

    fn choose_action(&self, view: &GameView, legal_actions: &[Action]) -> usize {
        legal_actions
            .iter()
            .enumerate()
            .max_by_key(|(_, a)| score_action(a, view))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn respond_to_chain(
        &self,
        view: &GameView,
        chain: &ChainView,
        legal_responses: &[ChainAction],
    ) -> Option<usize> {
        let (best_idx, best_score) = legal_responses
            .iter()
            .enumerate()
            .map(|(i, r)| (i, score_chain_response(r, view, chain)))
            .max_by_key(|(_, s)| *s)
            .unwrap_or((0, 0));

        if best_score > 0 { Some(best_idx) } else { None }
    }

    fn decide(&self, view: &GameView, request: &ChoiceRequest) -> usize {
        best_decision(view, request)
    }

    fn pick_targets(
        &self,
        view: &GameView,
        _prompt: &str,
        options: &[TargetOption],
        count: usize,
    ) -> Vec<usize> {
        let mut scored: Vec<(usize, i32)> = options
            .iter()
            .enumerate()
            .map(|(i, opt)| (i, score_target(&opt.target, view)))
            .collect();
        scored.sort_by_key(|(_, s)| -s);
        scored.iter().take(count).map(|(i, _)| *i).collect()
    }

    fn choose_discards(
        &self,
        view: &GameView,
        hand: &[Card],
        must_discard: usize,
    ) -> Vec<CardId> {
        let mut scored: Vec<(usize, i32)> = hand
            .iter()
            .enumerate()
            .map(|(i, c)| (i, card_keep_value(c, view)))
            .collect();
        scored.sort_by_key(|(_, v)| *v); // ascending: lowest keep-value first
        scored.iter().take(must_discard).map(|(i, _)| hand[*i].id).collect()
    }
}

// ── Action scoring ────────────────────────────────────────────────────────────

fn score_action(action: &Action, view: &GameView) -> i32 {
    let my_complete = view.my_star_chart.complete_systems().len();
    let supernovas = view.my_hand.iter().filter(|c| c.is_supernova()).count();
    let leader_systems = view.opponents.iter()
        .map(|o| o.star_chart.complete_systems().len())
        .max()
        .unwrap_or(0);

    match action {
        Action::Pass => 0,

        // Instant win — always take it
        Action::PlaySupernovas { .. } => 1_000_000,

        // Play star + discoveries as a set
        Action::PlaySolarSystem { star, discoveries } => {
            let class = find_card(&view.my_hand, *star)
                .and_then(|c| c.star_class())
                .unwrap_or(3) as i32;
            let win_bonus = if my_complete + 1 >= 3 { 2000 } else if my_complete + 1 >= 2 { 400 } else { 0 };
            500 + discoveries.len() as i32 * 80 - class * 5 + win_bonus
        }

        Action::PlayDiscovery { onto_star, .. } => score_discovery(*onto_star, view, my_complete),

        Action::PlayStar(id) => score_star(*id, view),

        Action::PlayProtostar { choice, .. } => match choice {
            ProtostarChoice::DrawTwo => 130,
            ProtostarChoice::SearchWhiteDwarf => {
                if view.my_star_chart.systems.is_empty() { 170 } else { 55 }
            }
        },

        Action::PlayFlare(id) => {
            let Some(card) = find_card(&view.my_hand, *id) else { return 0 };
            match &card.kind {
                // Supernovas are only useful as a trio via PlaySupernovas
                CardKind::Flare(n) if n.is_supernova() => -100,
                // CosmicDenial is a reactive card — wasteful to proactively play
                CardKind::Flare(FlareName::CosmicDenial) => 15,
                CardKind::Flare(FlareName::StellarAlignment) => 115,
                CardKind::Flare(FlareName::Wormhole) => 105,
                CardKind::Flare(FlareName::Ignition) => 85,
                CardKind::Flare(FlareName::AstralProjection) => 80,
                _ => 30,
            }
        }

        Action::PlayFracture(id) => {
            let Some(card) = find_card(&view.my_hand, *id) else { return 0 };
            match &card.kind {
                CardKind::Fracture(FractureName::Shockwave) => match supernovas {
                    0 => 65,
                    1 => 190,
                    2 => 650, // one search away from Cosmic Cataclysm
                    _ => 0,
                },
                CardKind::Fracture(FractureName::BlackHole) => {
                    if leader_systems >= 2 { 145 } else { 70 }
                }
                CardKind::Fracture(FractureName::Athena) => {
                    60 + opponent_danger(view) / 3
                }
                CardKind::Fracture(FractureName::DarkMatterVoid) => 75,
                CardKind::Fracture(FractureName::StellarNurseryCollapse) => 70,
                CardKind::Fracture(FractureName::QuantumEntanglementFailure) => 55,
                CardKind::Fracture(FractureName::Pulsar) => 30, // risky hand wipe
                _ => 40,
            }
        }

        Action::PlayOmen { card, .. } => {
            let Some(c) = find_card(&view.my_hand, *card) else { return 0 };
            match &c.kind {
                CardKind::Omen(OmenName::SolarEclipse) => {
                    if my_complete > 0 { 195 } else { -30 }
                }
                CardKind::Omen(OmenName::PlanetOfLife) => 115,
                _ => 40,
            }
        }

    }
}

fn score_discovery(onto_star: CardId, view: &GameView, my_complete: usize) -> i32 {
    let Some(sys) = view.my_star_chart.system_for_star(onto_star) else { return 50 };
    let after = sys.discoveries.len() + 1;
    let needed = sys.class as usize;

    if after >= needed {
        // Completes the system — weight by how close we are to winning
        match my_complete {
            2 => 1600, // this wins the game
            1 => 750,
            0 => 550,
            _ => 200,
        }
    } else {
        let remaining = needed - after;
        match remaining {
            1 => 270, // one away from complete
            2 => 160,
            _ => 90,
        }
    }
}

fn score_star(id: CardId, view: &GameView) -> i32 {
    let Some(card) = find_card(&view.my_hand, id) else { return 0 };
    let class = card.star_class().unwrap_or(3);

    // Blocked by Dark Matter Void
    if view.dark_matter_void == Some(class) { return -50; }

    // Prefer lower class stars: faster to complete, cheaper win
    let class_score = match class {
        2 => 100,
        3 => 82,
        4 => 62,
        5 => 42,
        6 => 25,
        _ => 50,
    };

    // Penalise cluttering the star chart with many incomplete systems
    let incomplete = view.my_star_chart.systems.iter().filter(|s| !s.is_complete()).count();
    let clutter = (incomplete.saturating_sub(1)) as i32 * 18;

    class_score - clutter
}

// ── Chain response scoring ────────────────────────────────────────────────────

fn score_chain_response(resp: &ChainAction, view: &GameView, chain: &ChainView) -> i32 {
    let Some(card) = find_card(&view.my_hand, resp.card) else { return 0 };
    let supernovas = view.my_hand.iter().filter(|c| c.is_supernova()).count();
    let my_complete = view.my_star_chart.complete_systems().len();
    let leader_systems = view.opponents.iter()
        .map(|o| o.star_chart.complete_systems().len())
        .max()
        .unwrap_or(0);

    match &card.kind {
        CardKind::Flare(FlareName::CosmicDenial) => {
            // Only count opponent links as threats — never negate own cards
            let my_tag = format!("Player {}", view.my_idx);
            let opponent_link = |l: &crate::chain::ChainLinkView| !l.negated && l.player_name != my_tag;

            let shockwave_in_chain = chain.links.iter()
                .any(|l| opponent_link(l) && matches!(&l.card.kind, CardKind::Fracture(FractureName::Shockwave)));
            let pulsar_in_chain = chain.links.iter()
                .any(|l| opponent_link(l) && matches!(&l.card.kind, CardKind::Fracture(FractureName::Pulsar)));
            let supernova_hunt = chain.links.iter()
                .any(|l| opponent_link(l) && l.card.is_supernova());

            if shockwave_in_chain && !supernova_hunt {
                // Block the search before they get a supernova
                320
            } else if pulsar_in_chain {
                160
            } else {
                75
            }
        }

        CardKind::Fracture(FractureName::Shockwave) => match supernovas {
            0 => 70,
            1 => 220,
            2 => 750,
            _ => 0,
        },

        CardKind::Fracture(FractureName::BlackHole) => {
            if leader_systems >= 2 { 150 } else { 65 }
        }

        CardKind::Fracture(FractureName::Athena) => 90,
        CardKind::Fracture(FractureName::StellarNurseryCollapse) => 65,
        CardKind::Fracture(FractureName::DarkMatterVoid) => 60,
        CardKind::Fracture(FractureName::QuantumEntanglementFailure) => 55,

        CardKind::Flare(FlareName::Ignition) => 80,
        CardKind::Flare(FlareName::AstralProjection) => 70,

        CardKind::Omen(OmenName::SolarEclipse) => {
            if my_complete > 0 { 170 } else { 0 }
        }
        CardKind::Omen(OmenName::PlanetOfLife) => 85,

        _ => 25,
    }
}

// ── Mid-effect decisions ──────────────────────────────────────────────────────

fn best_decision(view: &GameView, request: &ChoiceRequest) -> usize {
    let prompt = request.prompt.to_lowercase();
    let opts = &request.options;

    if opts.is_empty() { return 0; }

    if prompt.contains("negate") || prompt.contains("cosmic denial") || prompt.contains("deny") {
        return best_negate_target(opts);
    }

    if (prompt.contains("class") || prompt.contains("star")) && prompt.contains("void") {
        return best_class_to_block(opts, view);
    }

    if prompt.contains("nursery") || prompt.contains("bottom") || prompt.contains("top 5") {
        return best_nursery_pick(opts);
    }

    if prompt.contains("remove") || prompt.contains("athena") || prompt.contains("destroy") {
        return best_removal_pick(opts, view);
    }

    0
}

fn best_negate_target(opts: &[String]) -> usize {
    // Prioritise negating the most dangerous fractures
    for keyword in &["Shockwave", "Pulsar", "Stellar Nursery", "Black Hole", "Athena", "Quantum"] {
        if let Some(i) = opts.iter().position(|o| o.contains(keyword)) {
            return i;
        }
    }
    0
}

fn best_class_to_block(opts: &[String], view: &GameView) -> usize {
    // Count how many opponents are building each class
    let mut by_class = [0i32; 8];
    for opp in &view.opponents {
        for sys in &opp.star_chart.systems {
            if !sys.is_complete() {
                let c = sys.class as usize;
                if c < by_class.len() {
                    by_class[c] += 1;
                }
            }
        }
    }
    // Pick whichever option matches the most-built class
    let mut best = (0usize, -1i32);
    for (i, opt) in opts.iter().enumerate() {
        for class in 2u8..=6 {
            if opt.contains(&class.to_string()) {
                let score = by_class[class as usize];
                if score > best.1 {
                    best = (i, score);
                }
            }
        }
    }
    best.0
}

fn best_nursery_pick(opts: &[String]) -> usize {
    // Bury supernovas — send them to the bottom so opponents can't Shockwave into them
    for (i, opt) in opts.iter().enumerate() {
        if opt.to_lowercase().contains("supernova") {
            return i;
        }
    }
    // Otherwise bury high-class stars (they're hardest to use)
    for keyword in &["Red Supergiant", "Red Giant", "Yellow Dwarf"] {
        if let Some(i) = opts.iter().position(|o| o.contains(keyword)) {
            return i;
        }
    }
    0
}

fn best_removal_pick(_opts: &[String], _view: &GameView) -> usize {
    0
}

// ── Target scoring ────────────────────────────────────────────────────────────

fn score_target(target: &Target, view: &GameView) -> i32 {
    match target {
        Target::Player(idx) => {
            if *idx == view.my_idx { return -100; }
            let opp = view.opponents.iter().find(|o| o.idx == *idx);
            opp.map(|o| o.star_chart.complete_systems().len() as i32 * 100).unwrap_or(0)
        }
        Target::StarChartCard { player_idx, .. } => {
            if *player_idx == view.my_idx { return -50; }
            let opp = view.opponents.iter().find(|o| o.idx == *player_idx);
            opp.map(|o| o.star_chart.complete_systems().len() as i32 * 80).unwrap_or(0)
        }
        Target::HandCard { player_idx, .. } => {
            if *player_idx == view.my_idx { return -50; }
            let opp = view.opponents.iter().find(|o| o.idx == *player_idx);
            opp.map(|o| o.star_chart.complete_systems().len() as i32 * 60).unwrap_or(0)
        }
        Target::StarClass(class) => {
            // Prefer blocking classes opponents are actively building
            view.opponents.iter()
                .flat_map(|o| &o.star_chart.systems)
                .filter(|s| s.class == *class && !s.is_complete())
                .count() as i32 * 50
        }
        Target::ChainLink(idx) => *idx as i32 * 5,
    }
}

// ── Discard value ─────────────────────────────────────────────────────────────

fn card_keep_value(card: &Card, view: &GameView) -> i32 {
    let supernovas = view.my_hand.iter().filter(|c| c.is_supernova()).count();
    let my_complete = view.my_star_chart.complete_systems().len();

    match &card.kind {
        // Never discard supernovas — they're part of an instant-win combo
        CardKind::Flare(n) if n.is_supernova() => 1000,

        CardKind::Fracture(FractureName::Shockwave) => match supernovas {
            0 => 70,
            1 => 160,
            2 => 450,
            _ => 50,
        },

        CardKind::Flare(FlareName::StellarAlignment) => 105,
        CardKind::Flare(FlareName::Wormhole) => 95,
        CardKind::Flare(FlareName::CosmicDenial) => 90,
        CardKind::Flare(FlareName::Ignition) => 85,
        CardKind::Flare(FlareName::AstralProjection) => 80,
        CardKind::Flare(_) => 70,

        CardKind::Omen(OmenName::SolarEclipse) => {
            if my_complete > 0 { 155 } else { 50 }
        }
        CardKind::Omen(_) => 80,

        CardKind::Fracture(FractureName::BlackHole) => 90,
        CardKind::Fracture(FractureName::Athena) => 80,
        CardKind::Fracture(_) => 65,

        CardKind::Star(n) => {
            let class = n.class().unwrap_or(4) as i32;
            120 - class * 10
        }

        CardKind::Discovery(_) => {
            // More valuable when we have an incomplete system to attach to
            let has_space = view.my_star_chart.systems.iter().any(|s| !s.is_complete());
            if has_space { 100 } else { 65 }
        }
    }
}

// ── Advanced CPU ──────────────────────────────────────────────────────────────

/// A strong CPU with full threat assessment, combo awareness, and context-sensitive
/// decisions for every card effect.
pub struct AdvancedCpuPlayer {
    name: String,
}

impl AdvancedCpuPlayer {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl Player for AdvancedCpuPlayer {
    fn name(&self) -> &str {
        &self.name
    }

    fn choose_action(&self, view: &GameView, legal_actions: &[Action]) -> usize {
        legal_actions
            .iter()
            .enumerate()
            .max_by_key(|(_, a)| adv_score_action(a, view))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn respond_to_chain(
        &self,
        view: &GameView,
        chain: &ChainView,
        legal_responses: &[ChainAction],
    ) -> Option<usize> {
        let (best_idx, best_score) = legal_responses
            .iter()
            .enumerate()
            .map(|(i, r)| (i, adv_score_chain_response(r, view, chain)))
            .max_by_key(|(_, s)| *s)
            .unwrap_or((0, 0));
        if best_score > 0 { Some(best_idx) } else { None }
    }

    fn decide(&self, view: &GameView, request: &ChoiceRequest) -> usize {
        adv_decision(view, request, &self.name)
    }

    fn pick_targets(
        &self,
        view: &GameView,
        prompt: &str,
        options: &[TargetOption],
        count: usize,
    ) -> Vec<usize> {
        let mut scored: Vec<(usize, i32)> = options
            .iter()
            .enumerate()
            .map(|(i, opt)| (i, adv_score_target(&opt.target, view, prompt, &opt.label)))
            .collect();
        scored.sort_by_key(|(_, s)| -s);
        scored.iter().take(count).map(|(i, _)| *i).collect()
    }

    fn choose_discards(
        &self,
        view: &GameView,
        hand: &[Card],
        must_discard: usize,
    ) -> Vec<CardId> {
        let mut scored: Vec<(usize, i32)> = hand
            .iter()
            .enumerate()
            .map(|(i, c)| (i, adv_card_keep_value(c, view)))
            .collect();
        scored.sort_by_key(|(_, v)| *v);
        scored.iter().take(must_discard).map(|(i, _)| hand[*i].id).collect()
    }
}

// ── Adv action scoring ────────────────────────────────────────────────────────

fn adv_score_action(action: &Action, view: &GameView) -> i32 {
    let my_complete = view.my_star_chart.complete_systems().len();
    let supernovas = view.my_hand.iter().filter(|c| c.is_supernova()).count();
    let leader_complete = view.opponents.iter()
        .map(|o| o.star_chart.complete_systems().len())
        .max()
        .unwrap_or(0);
    let max_threat = max_opp_threat(view);
    // Extra incentive to play disruptive cards when someone is one system from winning
    let threat_bonus = if leader_complete >= 2 { 200 } else { 0 };

    match action {
        Action::Pass => 0,
        Action::PlaySupernovas { .. } => 1_000_000,

        Action::PlaySolarSystem { star, discoveries } => {
            let class = find_card(&view.my_hand, *star)
                .and_then(|c| c.star_class())
                .unwrap_or(3) as i32;
            if my_complete + 1 >= 3 {
                100_000
            } else {
                let prox = match my_complete + 1 {
                    2 => 2500,
                    1 => 1000,
                    _ => 400,
                };
                prox + discoveries.len() as i32 * 60 - class * 3
            }
        }

        Action::PlayDiscovery { onto_star, .. } => {
            adv_score_discovery(*onto_star, view, my_complete)
        }

        Action::PlayStar(id) => adv_score_star(*id, view),

        Action::PlayProtostar { choice, .. } => {
            let hand_sz = view.my_hand.len() as i32;
            match choice {
                ProtostarChoice::DrawTwo => 100 + (7 - hand_sz.min(7)) * 15,
                ProtostarChoice::SearchWhiteDwarf => {
                    let needs_class3 = view.my_star_chart.systems.iter()
                        .any(|s| !s.is_complete() && s.class == 3);
                    if view.my_star_chart.systems.is_empty() { 200 }
                    else if needs_class3 { 170 }
                    else { 65 }
                }
            }
        }

        Action::PlayFlare(id) => adv_score_flare_proactive(*id, view, supernovas, max_threat),
        Action::PlayFracture(id) => adv_score_fracture_proactive(
            *id, view, supernovas, my_complete, leader_complete, max_threat, threat_bonus,
        ),

        Action::PlayOmen { card, .. } => {
            let Some(c) = find_card(&view.my_hand, *card) else { return 0 };
            match &c.kind {
                CardKind::Omen(OmenName::SolarEclipse) => match my_complete {
                    0 => -30,
                    1 => 160,
                    2 => 600,
                    _ => 200,
                },
                CardKind::Omen(OmenName::PlanetOfLife) => 85 + threat_bonus / 3,
                _ => 40,
            }
        }
    }
}

fn adv_score_discovery(onto_star: CardId, view: &GameView, my_complete: usize) -> i32 {
    let Some(sys) = view.my_star_chart.system_for_star(onto_star) else { return 55 };
    let after = sys.discoveries.len() + 1;
    let needed = sys.class as usize;

    if after >= needed {
        return match my_complete {
            2 => 100_000,
            1 => 2200,
            0 => 950,
            _ => 350,
        };
    }

    let remaining = needed - after;
    let disc_in_hand = view.my_hand.iter()
        .filter(|c| matches!(c.kind, CardKind::Discovery(_)))
        .count();
    let reachable_bonus = if disc_in_hand >= remaining { 80 } else { 0 };
    match remaining {
        1 => 400 + reachable_bonus,
        2 => 220 + reachable_bonus,
        _ => 110,
    }
}

fn adv_score_star(id: CardId, view: &GameView) -> i32 {
    let Some(card) = find_card(&view.my_hand, id) else { return 0 };
    let class = card.star_class().unwrap_or(3);

    if view.dark_matter_void == Some(class) { return -50; }

    let class_base: i32 = match class {
        2 => 115,
        3 => 95,
        4 => 72,
        5 => 50,
        6 => 32,
        _ => 60,
    };

    let disc_in_hand = view.my_hand.iter()
        .filter(|c| matches!(c.kind, CardKind::Discovery(_)))
        .count() as i32;
    let fill_potential = (disc_in_hand - class as i32 + 2).clamp(-3, 3) * 18;

    let incomplete = view.my_star_chart.systems.iter().filter(|s| !s.is_complete()).count() as i32;
    let clutter = if incomplete >= 2 { (incomplete - 1) * 22 } else { 0 };

    (class_base + fill_potential - clutter).max(-50)
}

fn adv_score_flare_proactive(id: CardId, view: &GameView, _supernovas: usize, max_threat: i32) -> i32 {
    let Some(card) = find_card(&view.my_hand, id) else { return 0 };
    match &card.kind {
        CardKind::Flare(n) if n.is_supernova() => -100,
        CardKind::Flare(FlareName::CosmicDenial) => 12, // save for reactions
        CardKind::Flare(FlareName::StellarAlignment) => {
            100 + (7 - view.my_hand.len().min(7)) as i32 * 18
        }
        CardKind::Flare(FlareName::Wormhole) => {
            let stealable: i32 = view.opponents.iter()
                .map(|o| {
                    o.star_chart.systems.iter()
                        .filter(|s| !s.is_eclipse_protected())
                        .map(|s| s.discoveries.len() as i32 * 55 + if !s.discoveries.is_empty() { 30 } else { 0 })
                        .sum::<i32>()
                })
                .max()
                .unwrap_or(0);
            70 + stealable.min(250)
        }
        CardKind::Flare(FlareName::Ignition) => {
            let has_gas = view.opponents.iter().any(|o| {
                o.star_chart.systems.iter().any(|s| s.has_gas_giant() && !s.is_eclipse_protected())
            });
            if has_gas { 160 } else { 55 }
        }
        CardKind::Flare(FlareName::AstralProjection) => 50 + max_threat / 6,
        _ => 30,
    }
}

fn adv_score_fracture_proactive(
    id: CardId,
    view: &GameView,
    supernovas: usize,
    my_complete: usize,
    leader_complete: usize,
    _max_threat: i32,
    threat_bonus: i32,
) -> i32 {
    let Some(card) = find_card(&view.my_hand, id) else { return 0 };
    match &card.kind {
        CardKind::Fracture(FractureName::Shockwave) => match supernovas {
            0 => 62,
            1 => 230,
            2 => 720,
            _ => 0,
        },

        CardKind::Fracture(FractureName::BlackHole) => {
            let opp_exposed: i32 = view.opponents.iter()
                .map(|o| {
                    o.star_chart.systems.iter()
                        .filter(|s| s.is_complete() && !s.is_eclipse_protected())
                        .count() as i32
                })
                .sum();
            let my_protected = view.my_star_chart.complete_systems().iter()
                .filter(|s| s.is_eclipse_protected())
                .count() as i32;
            let my_exposed = my_complete as i32 - my_protected;
            let net = opp_exposed - my_exposed;
            if net > 0 {
                100 + net * 100 + threat_bonus
            } else if leader_complete >= 2 {
                160 + threat_bonus
            } else {
                30
            }
        }

        CardKind::Fracture(FractureName::Athena) => {
            let has_target = view.opponents.iter().any(|o| {
                o.star_chart.systems.iter()
                    .any(|s| !s.is_eclipse_protected() && !s.discoveries.is_empty())
            });
            if has_target { 115 + threat_bonus / 3 } else { 35 }
        }

        CardKind::Fracture(FractureName::DarkMatterVoid) => {
            let opp_building = view.opponents.iter()
                .any(|o| !o.star_chart.incomplete_systems().is_empty());
            if opp_building { 95 + threat_bonus / 4 } else { 45 }
        }

        CardKind::Fracture(FractureName::StellarNurseryCollapse) => {
            if supernovas >= 1 || view.my_star_chart.systems.is_empty() { 100 } else { 65 }
        }

        CardKind::Fracture(FractureName::QuantumEntanglementFailure) => {
            let hand_v: i32 = view.my_hand.iter().map(|c| adv_card_keep_value(c, view)).sum();
            let avg_v = hand_v / view.my_hand.len().max(1) as i32;
            if avg_v < 70 { 85 } else { 45 }
        }

        CardKind::Fracture(FractureName::Pulsar) => {
            let hand_v: i32 = view.my_hand.iter()
                .filter(|c| !c.is_supernova())
                .map(|c| adv_card_keep_value(c, view))
                .sum();
            let chart_v: i32 = view.my_star_chart.systems.iter()
                .filter(|s| !s.is_eclipse_protected())
                .map(|s| s.discoveries.len() as i32 * 45 + if s.is_complete() { 120 } else { 0 })
                .sum();
            let loss = hand_v + chart_v;
            let expected_gain = 455; // 7 cards × ~65 avg keep-value
            if expected_gain > loss + 100 { 80 } else { (80 - (loss - expected_gain) / 8).max(-60) }
        }

        _ => 40,
    }
}

// ── Adv chain response ────────────────────────────────────────────────────────

fn adv_score_chain_response(resp: &ChainAction, view: &GameView, chain: &ChainView) -> i32 {
    let Some(card) = find_card(&view.my_hand, resp.card) else { return 0 };
    let supernovas = view.my_hand.iter().filter(|c| c.is_supernova()).count();
    let my_complete = view.my_star_chart.complete_systems().len();
    let leader_complete = view.opponents.iter()
        .map(|o| o.star_chart.complete_systems().len())
        .max()
        .unwrap_or(0);

    let my_tag = format!("Player {}", view.my_idx);
    let is_opp_link = |l: &ChainLinkView| !l.negated && l.player_name != my_tag;

    let opp_shockwave = chain.links.iter().any(|l| {
        is_opp_link(l) && matches!(&l.card.kind, CardKind::Fracture(FractureName::Shockwave))
    });
    let opp_pulsar = chain.links.iter().any(|l| {
        is_opp_link(l) && matches!(&l.card.kind, CardKind::Fracture(FractureName::Pulsar))
    });
    let opp_blackhole = chain.links.iter().any(|l| {
        is_opp_link(l) && matches!(&l.card.kind, CardKind::Fracture(FractureName::BlackHole))
    });
    let active_supernovas = chain.links.iter()
        .filter(|l| !l.negated && l.card.is_supernova())
        .count();
    let win_imminent = active_supernovas >= 2; // one more would win

    match &card.kind {
        CardKind::Flare(FlareName::CosmicDenial) => {
            if win_imminent { return 600; }
            if opp_shockwave { 380 }
            else if opp_pulsar { 220 }
            else if opp_blackhole && my_complete > 0 { 190 }
            else { 55 }
        }

        CardKind::Fracture(FractureName::Shockwave) => {
            if win_imminent { return 0; } // don't help chain heading to a win
            match supernovas {
                0 => 68,
                1 => 270,
                2 => 850,
                _ => 0,
            }
        }

        CardKind::Fracture(FractureName::BlackHole) => {
            let opp_exposed: i32 = view.opponents.iter()
                .map(|o| {
                    o.star_chart.systems.iter()
                        .filter(|s| s.is_complete() && !s.is_eclipse_protected())
                        .count() as i32
                })
                .sum();
            let my_protected = view.my_star_chart.complete_systems().iter()
                .filter(|s| s.is_eclipse_protected())
                .count() as i32;
            let my_exposed = my_complete as i32 - my_protected;
            let net = opp_exposed - my_exposed;
            if net > 0 { 130 + net * 90 }
            else if leader_complete >= 2 { 170 }
            else { 40 }
        }

        CardKind::Fracture(FractureName::Athena) => {
            let has_gas = view.opponents.iter().any(|o| {
                o.star_chart.systems.iter().any(|s| s.has_gas_giant() && !s.is_eclipse_protected())
            });
            if has_gas { 145 } else { 60 }
        }

        CardKind::Fracture(FractureName::DarkMatterVoid) => 75,
        CardKind::Fracture(FractureName::StellarNurseryCollapse) => 60,
        CardKind::Fracture(FractureName::QuantumEntanglementFailure) => 50,

        CardKind::Flare(FlareName::Ignition) => {
            let has_gas = view.opponents.iter().any(|o| {
                o.star_chart.systems.iter().any(|s| s.has_gas_giant() && !s.is_eclipse_protected())
            });
            if has_gas { 125 } else { 58 }
        }
        CardKind::Flare(FlareName::AstralProjection) => 68,
        CardKind::Flare(FlareName::Wormhole) => {
            let stealable = view.opponents.iter().any(|o| {
                o.star_chart.systems.iter()
                    .any(|s| !s.is_eclipse_protected() && !s.discoveries.is_empty())
            });
            if stealable { 95 } else { 40 }
        }

        CardKind::Omen(OmenName::SolarEclipse) => match my_complete {
            0 => 0,
            1 => 150,
            2 => 480,
            _ => 200,
        },
        CardKind::Omen(OmenName::PlanetOfLife) => 75,

        _ => 20,
    }
}

// ── Adv mid-effect decisions ──────────────────────────────────────────────────

fn adv_decision(view: &GameView, request: &ChoiceRequest, player_name: &str) -> usize {
    let prompt = request.prompt.to_lowercase();
    let opts = &request.options;
    if opts.is_empty() { return 0; }

    if prompt.contains("wormhole") { return adv_wormhole_mode(view); }
    if prompt.contains("ignition") { return adv_ignition_mode(view); }
    if prompt.contains("athena") { return adv_athena_mode(view); }
    if prompt.contains("shockwave") { return adv_shockwave_target(opts, view); }
    if prompt.contains("cosmic denial") || (prompt.contains("negate") && !prompt.contains("pay")) {
        return adv_best_negate_target(opts);
    }
    if prompt.contains("dark matter void") || (prompt.contains("class") && prompt.contains("restrict")) {
        return best_class_to_block(opts, view);
    }
    if prompt.contains("qef: choose first") {
        return adv_qef_first_player(opts, player_name);
    }
    if prompt.contains("qef: choose second")
        || prompt.contains("choose target opponent")
        || prompt.contains("astral projection")
    {
        return best_opponent_pick(opts, view);
    }
    if prompt.contains("secretly choose") {
        return adv_worst_card_idx(view);
    }
    if prompt.contains("stellar nursery") {
        return adv_nursery_pick(opts, view);
    }
    // Solar Eclipse maintenance: pay if we have cards and the system is valuable
    if prompt.contains("solar eclipse") && prompt.contains("pay") {
        let my_complete = view.my_star_chart.complete_systems().len();
        return if !view.my_hand.is_empty() && my_complete >= 1 { 0 } else { 1 };
    }

    0
}

fn adv_wormhole_mode(view: &GameView) -> usize {
    // Mode 1 (steal from star chart) is almost always better — we pick what to take
    let chart_stealable = view.opponents.iter().any(|o| {
        o.star_chart.systems.iter()
            .any(|s| !s.is_eclipse_protected() && !s.discoveries.is_empty())
    });
    if chart_stealable { 1 } else { 0 }
}

fn adv_ignition_mode(view: &GameView) -> usize {
    // Mode 0: destroy Gas Giant from star chart (better — destroys it)
    let gas_in_chart = view.opponents.iter().any(|o| {
        o.star_chart.systems.iter().any(|s| s.has_gas_giant() && !s.is_eclipse_protected())
    });
    if gas_in_chart { 0 } else { 1 }
}

fn adv_athena_mode(view: &GameView) -> usize {
    // Mode 0: destroy any Discovery (default — more flexible)
    // Mode 1: return Gas Giant to hand — only if taking it breaks a complete system
    let breaks_complete = view.opponents.iter().any(|o| {
        o.star_chart.systems.iter()
            .any(|s| s.has_gas_giant() && !s.is_eclipse_protected() && s.is_complete())
    });
    if breaks_complete { 1 } else { 0 }
}

fn adv_shockwave_target(opts: &[String], view: &GameView) -> usize {
    // Search for the supernova we don't yet hold
    let has_fusion = view.my_hand.iter()
        .any(|c| matches!(&c.kind, CardKind::Flare(FlareName::SupernovaFusion)));
    let has_gravity = view.my_hand.iter()
        .any(|c| matches!(&c.kind, CardKind::Flare(FlareName::SupernovaGravity)));

    if !has_fusion && !opts.is_empty() { return 0; }        // "Supernova - Fusion"
    if !has_gravity && opts.len() > 1 { return 1; }         // "Supernova - Gravity"
    if opts.len() > 2 { return 2; }                          // "Supernova - Gas"
    0
}

fn adv_best_negate_target(opts: &[String]) -> usize {
    for keyword in &[
        "Shockwave", "Supernova", "Pulsar", "Stellar Nursery", "Black Hole",
        "Wormhole", "Ignition", "Athena", "Quantum", "Dark Matter",
    ] {
        if let Some(i) = opts.iter().position(|o| o.contains(keyword)) {
            return i;
        }
    }
    0
}

fn best_opponent_pick(opts: &[String], view: &GameView) -> usize {
    opts.iter()
        .enumerate()
        .max_by_key(|(_, name)| {
            view.opponents.iter()
                .find(|o| o.name.as_str() == name.as_str())
                .map(|o| opp_threat(o))
                .unwrap_or(0)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn adv_qef_first_player(opts: &[String], player_name: &str) -> usize {
    // Pick ourselves so we control which card we offer
    opts.iter().position(|n| n == player_name).unwrap_or(0)
}

fn adv_worst_card_idx(view: &GameView) -> usize {
    // Give away the card with the lowest keep value
    view.my_hand
        .iter()
        .enumerate()
        .min_by_key(|(_, c)| adv_card_keep_value(c, view))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn adv_nursery_pick(opts: &[String], view: &GameView) -> usize {
    let supernovas = view.my_hand.iter().filter(|c| c.is_supernova()).count();
    // Bury supernovas to deny Shockwave access — unless we already have 2 (then keep near top)
    if supernovas < 2 {
        if let Some(i) = opts.iter().position(|o| o.to_lowercase().contains("supernova")) {
            return i;
        }
    }
    // Otherwise bury high-class stars (hardest to complete)
    for keyword in &["Red Supergiant", "Red Giant", "Yellow Dwarf"] {
        if let Some(i) = opts.iter().position(|o| o.contains(keyword)) {
            return i;
        }
    }
    0
}

// ── Adv target scoring ────────────────────────────────────────────────────────

fn adv_score_target(target: &Target, view: &GameView, _prompt: &str, label: &str) -> i32 {
    match target {
        Target::Player(idx) => {
            if *idx == view.my_idx { return -1000; }
            view.opponents.iter().find(|o| o.idx == *idx)
                .map(|o| opp_threat(o))
                .unwrap_or(0)
        }
        Target::StarChartCard { player_idx, .. } => {
            if *player_idx == view.my_idx { return -1000; }
            let base = view.opponents.iter().find(|o| o.idx == *player_idx)
                .map(|o| opp_threat(o) / 3)
                .unwrap_or(0);
            // Extra value if removing this card would set back a completing system
            let completion_bonus = if label.to_lowercase().contains("complete") { 120 } else { 0 };
            base + completion_bonus
        }
        Target::HandCard { player_idx, .. } => {
            if *player_idx == view.my_idx { return -1000; }
            view.opponents.iter().find(|o| o.idx == *player_idx)
                .map(|o| opp_threat(o) / 2)
                .unwrap_or(0)
        }
        Target::StarClass(class) => {
            view.opponents.iter()
                .flat_map(|o| &o.star_chart.systems)
                .filter(|s| s.class == *class && !s.is_complete())
                .count() as i32 * 75
                - if view.dark_matter_void == Some(*class) { 150 } else { 0 }
        }
        Target::ChainLink(_) => {
            for (score, keyword) in &[
                (1000i32, "Supernova"),
                (550,     "Shockwave"),
                (420,     "Pulsar"),
                (380,     "Stellar Nursery"),
                (320,     "Black Hole"),
                (270,     "Wormhole"),
                (230,     "Ignition"),
                (200,     "Athena"),
                (170,     "Quantum"),
                (150,     "Dark Matter"),
            ] {
                if label.contains(keyword) { return *score; }
            }
            50
        }
    }
}

// ── Adv discard value ─────────────────────────────────────────────────────────

fn adv_card_keep_value(card: &Card, view: &GameView) -> i32 {
    let supernovas = view.my_hand.iter().filter(|c| c.is_supernova()).count();
    let my_complete = view.my_star_chart.complete_systems().len();
    let leader_complete = view.opponents.iter()
        .map(|o| o.star_chart.complete_systems().len())
        .max()
        .unwrap_or(0);
    let has_incomplete = !view.my_star_chart.incomplete_systems().is_empty();

    match &card.kind {
        CardKind::Flare(n) if n.is_supernova() => 1000,

        CardKind::Fracture(FractureName::Shockwave) => match supernovas {
            0 => 65,
            1 => 180,
            2 => 550,
            _ => 55,
        },

        CardKind::Flare(FlareName::StellarAlignment) => 115,
        CardKind::Flare(FlareName::Wormhole) => 105,
        CardKind::Flare(FlareName::CosmicDenial) => 100,
        CardKind::Flare(FlareName::Ignition) => {
            let has_gas = view.opponents.iter()
                .any(|o| o.star_chart.systems.iter().any(|s| s.has_gas_giant()));
            if has_gas { 140 } else { 75 }
        }
        CardKind::Flare(FlareName::AstralProjection) => 85,
        CardKind::Flare(_) => 70,

        CardKind::Omen(OmenName::SolarEclipse) => match my_complete {
            0 => 45,
            1 => 165,
            2 => 450,
            _ => 200,
        },
        CardKind::Omen(_) => 80,

        CardKind::Fracture(FractureName::BlackHole) => {
            if leader_complete >= 2 { 160 } else { 85 }
        }
        CardKind::Fracture(FractureName::Athena) => {
            let has_gas = view.opponents.iter()
                .any(|o| o.star_chart.systems.iter().any(|s| s.has_gas_giant()));
            if has_gas { 130 } else { 65 }
        }
        CardKind::Fracture(_) => 70,

        CardKind::Star(n) => {
            let class = n.class().unwrap_or(4) as i32;
            let base = 120 - class * 10;
            let disc_count = view.my_hand.iter()
                .filter(|c| matches!(c.kind, CardKind::Discovery(_)))
                .count() as i32;
            if disc_count >= class { base + 45 } else { base }
        }

        CardKind::Discovery(_) => {
            if has_incomplete { 115 } else { 58 }
        }
    }
}

// ── Adv helpers ───────────────────────────────────────────────────────────────

fn opp_threat(opp: &OpponentView) -> i32 {
    let complete = opp.star_chart.complete_systems().len() as i32 * 300;
    let near: i32 = opp.star_chart.incomplete_systems().iter()
        .map(|s| {
            let rem = s.class as usize - s.discoveries.len();
            match rem { 1 => 120, 2 => 50, _ => 15 }
        })
        .sum();
    complete + near + opp.hand_size as i32 * 8
}

fn max_opp_threat(view: &GameView) -> i32 {
    view.opponents.iter().map(opp_threat).max().unwrap_or(0)
}

// ── Easy CPU (onboarding difficulty) ─────────────────────────────────────────

/// A weaker CPU opponent for new players: 40% random actions, rarely chain-responds,
/// and doesn't recognise win threats as urgently as the full CPU.
pub struct EasyCpuPlayer {
    name: String,
}

impl EasyCpuPlayer {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl Player for EasyCpuPlayer {
    fn name(&self) -> &str {
        &self.name
    }

    fn choose_action(&self, view: &GameView, legal_actions: &[Action]) -> usize {
        // Always take an instant win if available
        if let Some(i) = legal_actions.iter().position(|a| matches!(a, Action::PlaySupernovas { .. })) {
            return i;
        }
        // 40% random, 60% simplified heuristic
        if rand::random::<u8>() % 10 < 4 {
            return rand::random::<usize>() % legal_actions.len();
        }
        legal_actions.iter().enumerate()
            .max_by_key(|(_, a)| easy_score_action(a, view))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn respond_to_chain(
        &self,
        _view: &GameView,
        _chain: &ChainView,
        legal_responses: &[ChainAction],
    ) -> Option<usize> {
        if legal_responses.is_empty() { return None; }
        // Responds only 20% of the time, always randomly
        if rand::random::<u8>() % 10 < 2 {
            Some(rand::random::<usize>() % legal_responses.len())
        } else {
            None
        }
    }

    fn decide(&self, _view: &GameView, request: &ChoiceRequest) -> usize {
        rand::random::<usize>() % request.options.len().max(1)
    }

    fn pick_targets(
        &self,
        _view: &GameView,
        _prompt: &str,
        options: &[TargetOption],
        count: usize,
    ) -> Vec<usize> {
        (0..count.min(options.len())).collect()
    }

    fn choose_discards(
        &self,
        _view: &GameView,
        hand: &[Card],
        must_discard: usize,
    ) -> Vec<CardId> {
        hand.iter().take(must_discard).map(|c| c.id).collect()
    }
}

fn easy_score_action(action: &Action, view: &GameView) -> i32 {
    match action {
        Action::Pass => 0,
        Action::PlaySupernovas { .. } => 1_000_000,
        Action::PlaySolarSystem { discoveries, .. } => 150 + discoveries.len() as i32 * 30,
        Action::PlayDiscovery { onto_star, .. } => {
            view.my_star_chart.system_for_star(*onto_star)
                .map(|s| if s.discoveries.len() + 1 >= s.class as usize { 140 } else { 70 })
                .unwrap_or(60)
        }
        Action::PlayStar(id) => {
            view.my_hand.iter().find(|c| c.id == *id)
                .and_then(|c| c.star_class())
                .map(|cl| 80_i32 - cl as i32 * 5)
                .unwrap_or(50)
        }
        Action::PlayProtostar { .. } => 85,
        Action::PlayFlare(_) | Action::PlayFracture(_) => (rand::random::<u8>() % 40) as i32 + 10,
        Action::PlayOmen { .. } => 45,
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn find_card<'a>(hand: &'a [Card], id: CardId) -> Option<&'a Card> {
    hand.iter().find(|c| c.id == id)
}

/// Rough measure of how close the most threatening opponent is to winning.
fn opponent_danger(view: &GameView) -> i32 {
    view.opponents.iter().map(|o| {
        let complete = o.star_chart.complete_systems().len() as i32 * 100;
        let progress: i32 = o.star_chart.systems.iter()
            .filter(|s| !s.is_complete())
            .map(|s| s.discoveries.len() as i32 * 10 / s.class.max(1) as i32)
            .sum();
        complete + progress
    }).max().unwrap_or(0)
}
