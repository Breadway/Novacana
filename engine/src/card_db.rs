/// Compile-time-embedded card database parsed from the project CSV.
///
/// The CSV lives one directory above the crate root and is embedded with
/// `include_str!` so it needs no runtime I/O. Lookup is O(1) via a
/// `OnceLock<HashMap>` that is initialised on first call.
use std::collections::HashMap;
use std::sync::OnceLock;

static RAW_CSV: &str = include_str!("cards.csv");
static DB: OnceLock<HashMap<String, CardEntry>> = OnceLock::new();

pub struct CardEntry {
    pub card_type: String,
    pub class_value: Option<u8>,
    pub quantity: u8,
    pub effect: String,
}

/// A card entry with its name included - used by the card reference UI.
pub struct CardRef {
    pub name: String,
    pub card_type: String,
    pub effect: String,
}

pub fn lookup(name: &str) -> Option<&'static CardEntry> {
    DB.get_or_init(build).get(&name.to_lowercase())
}

/// Returns a formatted string for the LLM: "NAME (Type, classN) - effect"
pub fn lookup_text(name: &str) -> Option<String> {
    lookup(name).map(|e| {
        let class_str = e.class_value
            .map(|c| format!(", class {}", c))
            .unwrap_or_default();
        format!("{} ({}{}×{}) - {}", name, e.card_type, class_str, e.quantity, e.effect)
    })
}

/// Returns all cards sorted by name, with name included.
pub fn all_cards() -> Vec<CardRef> {
    let db = DB.get_or_init(build);
    let mut cards: Vec<CardRef> = db.iter().map(|(name, entry)| CardRef {
        name: title_case(name),
        card_type: entry.card_type.clone(),
        effect: entry.effect.clone(),
    }).collect();
    cards.sort_by(|a, b| a.card_type.cmp(&b.card_type).then(a.name.cmp(&b.name)));
    cards
}

fn title_case(s: &str) -> String {
    s.split_whitespace()
        .map(|word| {
            let mut c = word.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn build() -> HashMap<String, CardEntry> {
    let mut map = HashMap::new();
    let mut lines = RAW_CSV.lines();
    lines.next(); // skip header

    for line in lines {
        let line = line.trim_start_matches('\u{feff}'); // strip BOM if present
        if line.trim().is_empty() {
            continue;
        }
        let fields = parse_csv_row(line);
        if fields.len() < 6 {
            continue;
        }
        let name = fields[0].trim().to_string();
        let card_type = fields[2].trim().to_string();
        let class_value = fields[3].trim().parse::<u8>().ok();
        let quantity = fields[4].trim().parse::<u8>().unwrap_or(0);
        let effect = fields[5].trim().to_string();

        map.insert(name.to_lowercase(), CardEntry { card_type, class_value, quantity, effect });
    }
    map
}

fn parse_csv_row(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    chars.next(); // escaped double-quote → single "
                    current.push('"');
                } else {
                    in_quotes = false;
                }
            }
            '"' => in_quotes = true,
            ',' if !in_quotes => {
                fields.push(std::mem::take(&mut current));
            }
            other => current.push(other),
        }
    }
    fields.push(current);
    fields
}
