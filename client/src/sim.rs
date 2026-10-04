use rayon::prelude::*;

use novacana_engine::{
    cpu::{AdvancedCpuPlayer, CpuPlayer, EasyCpuPlayer},
    deck::Deck,
    game::{GameEngine, GameResult, WinCondition},
    player::Player,
};

const MAX_TURNS: u32 = 1000;

#[derive(Clone)]
struct GameRecord {
    winner_idx: usize,
    condition: Cond,
    turns: u32,
    timed_out: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum Cond { AstroDom, CosmicCataclysm }

impl From<&WinCondition> for Cond {
    fn from(w: &WinCondition) -> Self {
        match w {
            WinCondition::AstronomicalDomination => Cond::AstroDom,
            WinCondition::CosmicCataclysm => Cond::CosmicCataclysm,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum CpuKind { Easy, Normal, Advanced }

impl CpuKind {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "easy" => Some(Self::Easy),
            "normal" | "cpu" => Some(Self::Normal),
            "advanced" | "adv" | "pro" => Some(Self::Advanced),
            _ => None,
        }
    }

    fn make_player(&self, name: impl Into<String>) -> Box<dyn Player> {
        match self {
            Self::Easy => Box::new(EasyCpuPlayer::new(name)),
            Self::Normal => Box::new(CpuPlayer::new(name)),
            Self::Advanced => Box::new(AdvancedCpuPlayer::new(name)),
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Easy => "Easy",
            Self::Normal => "Normal",
            Self::Advanced => "Advanced",
        }
    }
}

fn run_one(num_players: usize, cpu: CpuKind) -> GameRecord {
    let players: Vec<Box<dyn Player>> = (0..num_players)
        .map(|i| cpu.make_player(format!("P{}", i + 1)))
        .collect();
    let deck = Deck::build_classic();
    let mut engine = GameEngine::new(deck, players);
    engine.set_silent();
    engine.set_max_turns(MAX_TURNS);
    let result: GameResult = engine.run();
    GameRecord {
        winner_idx: result.winner_idx,
        condition: Cond::from(&result.condition),
        timed_out: result.turns_elapsed >= MAX_TURNS,
        turns: result.turns_elapsed,
    }
}

pub fn run(num_games: usize, num_players: usize, cpu: CpuKind) {
    assert!(num_players >= 2 && num_players <= 4);
    println!(
        "Running {} games · {} players · {} CPU  (turn limit: {})…",
        num_games, num_players, cpu.label(), MAX_TURNS
    );

    let records: Vec<GameRecord> = (0..num_games)
        .into_par_iter()
        .map(|_| run_one(num_players, cpu))
        .collect();

    print_report(&records, num_players, cpu);
}

fn mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

fn stddev(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let m = mean(data);
    let variance = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    variance.sqrt()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn print_report(records: &[GameRecord], num_players: usize, cpu: CpuKind) {
    let n = records.len();
    if n == 0 {
        println!("No games recorded.");
        return;
    }

    let timeouts = records.iter().filter(|r| r.timed_out).count();

    // ── Win rates by position ─────────────────────────────────────────────────
    let mut wins = vec![0usize; num_players];
    for r in records { wins[r.winner_idx] += 1; }

    let expected = 1.0 / num_players as f64;
    let z95 = 1.96_f64;

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  WIN RATE BY TURN ORDER  ({} games, {} CPU)", n, cpu.label());
    if timeouts > 0 {
        println!("  ⚠ {}/{} games hit the {}-turn limit and were force-ended", timeouts, n, MAX_TURNS);
    }
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  {:<10}  {:>6}  {:>6}  {:>20}  {}", "Player", "Wins", "Win%", "95% CI", "Flag");
    println!("  {}", "─".repeat(58));

    let mut position_flags: Vec<String> = Vec::new();

    for (i, &w) in wins.iter().enumerate() {
        let p = w as f64 / n as f64;
        let margin = z95 * (p * (1.0 - p) / n as f64).sqrt();
        let lo = (p - margin).max(0.0);
        let hi = (p + margin).min(1.0);
        let ci = format!("[{:.1}%–{:.1}%]", lo * 100.0, hi * 100.0);

        // Flag if the expected win rate falls outside the 95% CI.
        let flag = if expected < lo || expected > hi {
            let delta = (p - expected) * 100.0;
            let dir = if p > expected { "↑ OVER" } else { "↓ UNDER" };
            format!("{} ({:+.1}pp)", dir, delta)
        } else {
            "ok".into()
        };

        if flag != "ok" {
            position_flags.push(format!("Player {} (seat {}): {}", i + 1, i + 1, flag));
        }

        println!(
            "  {:<10}  {:>6}  {:>5.1}%  {:>20}  {}",
            format!("P{} (seat {})", i + 1, i + 1),
            w,
            p * 100.0,
            ci,
            flag
        );
    }

    // Chi-squared test for positional uniformity.
    let expected_per_pos = n as f64 / num_players as f64;
    let chi2: f64 = wins.iter().map(|&w| {
        let diff = w as f64 - expected_per_pos;
        diff * diff / expected_per_pos
    }).sum();
    // Critical value at p=0.05 for (num_players-1) degrees of freedom.
    // df=1: 3.84, df=2: 5.99, df=3: 7.82
    let critical = match num_players - 1 {
        1 => 3.84,
        2 => 5.99,
        _ => 7.82,
    };
    let significant = chi2 > critical;

    println!("  {}", "─".repeat(58));
    println!("  Expected win rate per seat: {:.1}%", expected * 100.0);
    println!("  χ²={:.2}, df={}, critical={:.2} → {}",
        chi2, num_players - 1, critical,
        if significant { "SIGNIFICANT POSITIONAL BIAS (p<0.05)" } else { "no significant positional bias" }
    );

    // ── Win conditions ────────────────────────────────────────────────────────
    let astro = records.iter().filter(|r| r.condition == Cond::AstroDom).count();
    let cataclysm = n - astro;

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  WIN CONDITIONS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Astronomical Domination : {:5}  ({:.1}%)", astro, astro as f64 / n as f64 * 100.0);
    println!("  Cosmic Cataclysm        : {:5}  ({:.1}%)", cataclysm, cataclysm as f64 / n as f64 * 100.0);

    if cataclysm as f64 / (n as f64) > 0.40 {
        println!("  ⚠ Cosmic Cataclysm rate >40% - Supernova may be too accessible");
    } else if cataclysm as f64 / (n as f64) < 0.05 {
        println!("  ⚠ Cosmic Cataclysm rate <5% - Supernova may be too rare");
    }

    // ── Game length ───────────────────────────────────────────────────────────
    let mut turn_data: Vec<f64> = records.iter().map(|r| r.turns as f64).collect();
    turn_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let turns_mean = mean(&turn_data);
    let turns_sd = stddev(&turn_data);
    let p25 = percentile(&turn_data, 25.0);
    let p50 = percentile(&turn_data, 50.0);
    let p75 = percentile(&turn_data, 75.0);
    let p95 = percentile(&turn_data, 95.0);
    let t_min = turn_data.first().copied().unwrap_or(0.0);
    let t_max = turn_data.last().copied().unwrap_or(0.0);

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  GAME LENGTH (turns)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Min   : {:.0}", t_min);
    println!("  p25   : {:.0}", p25);
    println!("  Median: {:.0}", p50);
    println!("  Mean  : {:.1} ± {:.1} (σ)", turns_mean, turns_sd);
    println!("  p75   : {:.0}", p75);
    println!("  p95   : {:.0}", p95);
    println!("  Max   : {:.0}", t_max);

    // Build a simple ASCII histogram.
    let bucket_width = ((t_max - t_min) / 10.0).ceil().max(1.0);
    let mut buckets = vec![0usize; 10];
    for &t in &turn_data {
        let idx = ((t - t_min) / bucket_width) as usize;
        let idx = idx.min(9);
        buckets[idx] += 1;
    }
    let max_bucket = *buckets.iter().max().unwrap_or(&1);

    println!();
    println!("  Distribution:");
    for (i, &count) in buckets.iter().enumerate() {
        let lo = t_min + i as f64 * bucket_width;
        let hi = lo + bucket_width;
        let bar_len = (count * 30 / max_bucket.max(1)).max(if count > 0 { 1 } else { 0 });
        let bar = "█".repeat(bar_len);
        println!("  {:4.0}–{:<4.0}  {:<30} {}", lo, hi, bar, count);
    }

    // ── Win-by-condition × position heatmap ───────────────────────────────────
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  WIN CONDITION BY SEAT");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  {:<12}  {:>8}  {:>8}", "Seat", "AstroDom", "Cataclysm");
    println!("  {}", "─".repeat(32));
    for i in 0..num_players {
        let a: usize = records.iter().filter(|r| r.winner_idx == i && r.condition == Cond::AstroDom).count();
        let c: usize = records.iter().filter(|r| r.winner_idx == i && r.condition == Cond::CosmicCataclysm).count();
        println!("  P{} (seat {})  {:>8}  {:>8}", i + 1, i + 1, a, c);
    }

    // ── Early-finish outliers ─────────────────────────────────────────────────
    let very_short = records.iter().filter(|r| r.turns < 30).count();
    let very_long = records.iter().filter(|r| r.turns > 400).count();

    if very_short > 0 || very_long > 0 {
        println!();
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  OUTLIERS");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        if very_short > 0 {
            println!("  ⚠ {} game(s) ended in <30 turns - possible runaway win or rules edge case", very_short);
        }
        if very_long > 0 {
            println!("  ⚠ {} game(s) lasted >400 turns - possible deadlock or deck exhaustion issue", very_long);
        }
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  BALANCE SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut issues = Vec::new();
    if significant {
        issues.push(format!("Turn-order bias detected (χ²={:.2})", chi2));
    }
    for f in &position_flags {
        issues.push(f.clone());
    }
    if cataclysm as f64 / (n as f64) > 0.40 {
        issues.push("Cosmic Cataclysm win rate unusually high".into());
    } else if cataclysm as f64 / (n as f64) < 0.05 {
        issues.push("Cosmic Cataclysm win rate unusually low".into());
    }
    if very_short > 0 {
        issues.push(format!("{} game(s) ended <30 turns", very_short));
    }
    if very_long > 0 {
        issues.push(format!("{} game(s) lasted >400 turns", very_long));
    }
    if timeouts > 0 {
        let pct = timeouts as f64 / n as f64 * 100.0;
        issues.push(format!("{} game(s) ({:.1}%) hit the {}-turn limit - games may not be terminating naturally", timeouts, pct, MAX_TURNS));
    }

    if issues.is_empty() {
        println!("  No balance issues detected in {} games.", n);
    } else {
        println!("  Issues found:");
        for issue in &issues {
            println!("  • {}", issue);
        }
    }
    println!();
}
