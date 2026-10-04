//! Repeatable simulation workloads. Run both binaries with identical arguments.
use std::{alloc::{GlobalAlloc, Layout, System}, hint::black_box, sync::atomic::{AtomicBool, AtomicU64, Ordering}, time::Instant};
use novacana_engine::{cpu::{AdvancedCpuPlayer, ExpertCpuPlayer}, deck::Deck, game::GameEngine, player::Player, random::with_seed};

struct CountAlloc;
static COUNT: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNT.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) { System.dealloc(ptr, layout); }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if COUNT.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(size as u64, Ordering::Relaxed);
        }
        System.realloc(ptr, layout, size)
    }
}
#[global_allocator]
static ALLOCATOR: CountAlloc = CountAlloc;

fn players(expert: bool) -> Vec<Box<dyn Player>> {
    (0..4).map(|i| {
        let name = format!("p{i}");
        if expert { Box::new(ExpertCpuPlayer::new(name)) as Box<dyn Player> }
        else { Box::new(AdvancedCpuPlayer::new(name)) as Box<dyn Player> }
    }).collect()
}

fn mix(hash: &mut u64, data: &[u8]) {
    for byte in data { *hash = (*hash ^ u64::from(*byte)).wrapping_mul(1099511628211); }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("expert");
    let n: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(20);
    let seed: u64 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(42);
    assert!(["expert", "advanced", "view", "legal", "clone"].contains(&mode));
    let corpus: Vec<_> = (0..16).map(|i| with_seed(seed + i, || {
        let mut engine = GameEngine::new(Deck::build_classic(), players(false));
        engine.set_silent();
        engine.set_max_turns(4 + i as u32);
        engine.run();
        engine
    })).collect();
    let mut checksum = 14695981039346656037u64;
    let mut turns = 0u64;
    COUNT.store(args.iter().any(|s| s == "alloc"), Ordering::Relaxed);
    let start = Instant::now();
    for i in 0..n {
        match mode {
            "expert" | "advanced" => with_seed(seed + i as u64, || {
                let mut engine = GameEngine::new(Deck::build_classic(), players(mode == "expert"));
                engine.set_silent();
                engine.set_max_turns(1000);
                let result = engine.run();
                turns += result.turns_elapsed as u64;
                mix(&mut checksum, serde_json::to_string(&result).unwrap().as_bytes());
                mix(&mut checksum, format!("{:?}", engine.state.deck.all_ids()).as_bytes());
                for p in &engine.state.players {
                    mix(&mut checksum, format!("{:?}{:?}", p.hand, p.star_chart).as_bytes());
                }
            }),
            "view" => { black_box(corpus[i % corpus.len()].build_view(i % 4)); },
            "legal" => { black_box(corpus[i % corpus.len()].compute_legal_actions(i % 4)); },
            "clone" => { black_box(corpus[i % corpus.len()].state.clone()); },
            _ => unreachable!(),
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    COUNT.store(false, Ordering::Relaxed);
    println!("{}", serde_json::json!({"mode": mode, "iterations": n, "seed": seed,
        "seconds": elapsed, "per_second": n as f64 / elapsed, "turns": turns,
        "checksum": format!("{checksum:016x}"), "allocations": ALLOCS.load(Ordering::Relaxed),
        "allocated_bytes": BYTES.load(Ordering::Relaxed)}));
}
