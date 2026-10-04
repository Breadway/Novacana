// Micro-batched GPU inference for self-play/search (2026-07-10 - see
// STATE.md's "value-rank gate" postmortem for the measurement that
// motivated this). Every self-play/MCTS caller used to hit
// `net::forward_single` (batch-of-1) directly from its own rayon worker
// thread. Measured on the TD-value pod run: only ~3.6/13-14 usable vCPUs
// and ~30% GPU util were ever active at once, with most of the process's
// OS threads parked on `futex_wait_queue` - i.e. neither CPU nor GPU was
// the bottleneck, dozens of threads serializing one batch-of-1 CUDA call
// at a time through the driver was. AZ's MCTS (>=400 leaf evals per real
// decision, mostly falling through to a value-head forward pass - see
// mcts.rs's module doc) multiplies this same problem by ~2-3 orders of
// magnitude over TD-value's one-call-per-decision self-play, so it needed
// fixing before that stage launches, not after.
//
// This coalesces concurrent single-decision requests - which naturally
// arrive close together in time since ~13-14 games/threads run in
// parallel - into one batched `net::forward` call. Sims *within* one
// decision stay genuinely sequential (PUCT needs each sim's updated
// visit/value stats before picking the next candidate - see
// mcts.rs::run_puct), so this does not parallelize a single decision's
// search; it only removes the artificial serialization *across*
// concurrently-running games/threads. That's the actual bottleneck this
// found, not a deeper MCTS-internal one (no virtual-loss/leaf-parallelism
// redesign here).
//
// CRITICAL FOLLOW-UP (same day): the outer game loop must NOT be rayon
// (`.into_par_iter()`) once a `BatchedNet` is in the picture. `ndarray`
// (which `burn-ndarray` wraps) depends on `rayon` for its own tensor ops,
// using the SAME GLOBAL rayon pool. If the outer per-game loop is also
// rayon, every worker thread blocks on `BatchedNet::forward_single`'s
// channel `recv()` while still holding its pool slot - and once enough
// concurrent games saturate the pool this way, the batcher thread's OWN
// `net::forward()` call can't get a free rayon worker for its internal
// `ndarray` computation either. Deadlock: the batcher can't finish, so it
// never replies, so the blocked callers never free their slots. Reproduced
// directly (5 tests running concurrently hung indefinitely, confirmed via
// `/proc/<pid>/task/*/wchan` showing every thread parked on
// `futex_do_wait` with zero CPU time advancing over a 15s+ window; an
// isolated single low-concurrency test did NOT hang, consistent with the
// pool only being exhausted once concurrent demand is high enough).
// `parallel_for_each` below runs the outer loop on a small bounded pool of
// plain `std::thread`s instead - genuinely independent of rayon's global
// pool, so no nesting hazard regardless of how saturated either pool gets.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use burn::tensor::backend::Backend;

use crate::net::{self, Net};
use crate::tokens::TokenSeq;

/// How long the batcher waits after its first queued request before
/// draining - small enough to be invisible next to the CPU-side game-
/// simulation work between decisions, large enough to let concurrently
/// running threads' requests land in the same batch instead of each
/// forcing its own round trip.
const GRACE_PERIOD: Duration = Duration::from_micros(300);

/// How long the batcher blocks on the condvar between checks of the stop
/// flag when the queue is empty - just a periodic-wakeup bound, not a
/// throughput-relevant knob.
const IDLE_POLL: Duration = Duration::from_millis(50);

/// Default worker count for `parallel_for_each` callers generating self-
/// play games - adapts to whatever machine this actually runs on. Used to
/// be a hardcoded 32 (comfortably above the pod's ~13-14 usable vCPUs so
/// the batcher always had enough concurrent callers to coalesce); that's
/// badly wrong for the brother's PC's WSL2 instance, capped at 6 logical
/// processors by `dot-wslconfig` - 32 concurrent MCTS-playout threads on 6
/// cores thrash rather than help (each playout is CPU-heavy: a full engine
/// clone + simulated turn, see mcts.rs). 2x cores (not 1x) because these
/// threads spend real time blocked on `BatchedNet::forward_single`'s
/// channel recv, not just computing - some oversubscription keeps the
/// batcher fed. Every call site takes this as an explicit `concurrency`
/// parameter now (plumbed from `--gen-concurrency` in main.rs) rather than
/// reading this directly, so it's overridable per box; this function is
/// just the shared default (Fable 5 consult, 2026-07-16).
pub fn default_concurrency() -> usize {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    (2 * cores).max(8)
}

type Reply = mpsc::Sender<(Vec<f32>, f32)>;

/// Shared micro-batching wrapper around `net::forward`/`forward_single`.
/// Spawn once per self-play/search/eval call (`generate_td_samples`,
/// `generate_az_games`, `eval_win_rate`, `run_value_rank`, ...), share the
/// returned `Arc` across every concurrent game/decision-scorer, and let it
/// drop at the end of that call - `Drop` stops and joins the batcher
/// thread cleanly.
pub struct BatchedNet<B: Backend> {
    queue: Arc<Mutex<VecDeque<(TokenSeq, Reply)>>>,
    cvar: Arc<Condvar>,
    stop: Arc<AtomicBool>,
    worker: Mutex<Option<JoinHandle<()>>>,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> BatchedNet<B>
where
    B::Device: Clone + Send + Sync + 'static,
{
    /// `max_batch` caps how many requests one `net::forward` call absorbs
    /// at once - set comfortably above the expected concurrent-caller count
    /// (game/thread count) so a full burst coalesces into a single batch.
    pub fn spawn(net: Arc<Net<B>>, device: B::Device, max_batch: usize) -> Arc<Self> {
        let queue: Arc<Mutex<VecDeque<(TokenSeq, Reply)>>> = Arc::new(Mutex::new(VecDeque::new()));
        let cvar = Arc::new(Condvar::new());
        let stop = Arc::new(AtomicBool::new(false));

        let worker = {
            let queue = Arc::clone(&queue);
            let cvar = Arc::clone(&cvar);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || batcher_loop(net, device, max_batch, queue, cvar, stop))
        };

        Arc::new(Self { queue, cvar, stop, worker: Mutex::new(Some(worker)), _marker: std::marker::PhantomData })
    }

    /// Drop-in replacement for `net::forward_single` - identical semantics
    /// (masked policy logits + scalar value), transparently batched.
    pub fn forward_single(&self, seq: &TokenSeq) -> (Vec<f32>, f32) {
        // Thread-local, reused channel instead of allocating a fresh
        // mpsc::channel() every call (2026-07-16, Fable 5 consult) - only
        // matters at AZ scale (~6M calls/iteration, mcts.rs's leaf evals),
        // where millions of per-call channel allocations add up against a
        // playout-dominated budget. Safe to reuse: each call is a strictly
        // synchronous send-then-block-on-recv round trip (never two
        // in-flight forward_single calls on the same thread at once), and
        // `Sender` is `Clone` so the batcher's reply target is a fresh
        // clone of the same underlying channel each time, not a shared
        // handle that could race across threads (each thread gets its own
        // thread_local instance).
        thread_local! {
            static CHANNEL: (mpsc::Sender<(Vec<f32>, f32)>, mpsc::Receiver<(Vec<f32>, f32)>) = mpsc::channel();
        }
        CHANNEL.with(|(tx, rx)| {
            self.queue.lock().unwrap().push_back((*seq, tx.clone()));
            self.cvar.notify_one();
            // The batcher always sends exactly one reply per queued
            // request before dropping its Sender clone, so this never
            // panics under normal operation - a disconnect here would mean
            // the batcher thread died, which is a bug worth surfacing loudly.
            rx.recv().expect("BatchedNet worker thread died before replying")
        })
    }
}

impl<B: Backend> Drop for BatchedNet<B> {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.cvar.notify_all();
        if let Some(handle) = self.worker.lock().unwrap().take() {
            let _ = handle.join();
        }
    }
}

/// Runs `total` units of work (each producing one `T`) across a bounded
/// pool of plain `std::thread`s, NOT rayon - see this module's doc for why
/// rayon here nests badly with `BatchedNet`'s own use of `net::forward`
/// (which goes through `ndarray`'s rayon-based internals). `concurrency`
/// caps how many threads run at once (workers pull from a shared atomic
/// counter until `total` is exhausted, so it's safe to pass a `total`
/// larger than `concurrency`). Order of returned results matches input
/// index order.
pub fn parallel_for_each<T, F>(total: usize, concurrency: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    if total == 0 {
        return Vec::new();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: Mutex<Vec<(usize, T)>> = Mutex::new(Vec::with_capacity(total));
    let workers = concurrency.clamp(1, total);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            let next = &next;
            let results = &results;
            let f = &f;
            scope.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= total {
                    return;
                }
                let r = f(i);
                results.lock().unwrap().push((i, r));
            });
        }
    });
    let mut v = results.into_inner().unwrap();
    v.sort_by_key(|(i, _)| *i);
    v.into_iter().map(|(_, r)| r).collect()
}

fn batcher_loop<B: Backend>(
    net: Arc<Net<B>>,
    device: B::Device,
    max_batch: usize,
    queue: Arc<Mutex<VecDeque<(TokenSeq, Reply)>>>,
    cvar: Arc<Condvar>,
    stop: Arc<AtomicBool>,
) where
    B::Device: Clone,
{
    loop {
        let mut guard = queue.lock().unwrap();
        loop {
            if !guard.is_empty() || stop.load(Ordering::Relaxed) {
                break;
            }
            let (g, _) = cvar.wait_timeout(guard, IDLE_POLL).unwrap();
            guard = g;
        }
        if guard.is_empty() {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            continue;
        }
        drop(guard);

        std::thread::sleep(GRACE_PERIOD);

        let mut batch = Vec::with_capacity(max_batch);
        {
            let mut guard = queue.lock().unwrap();
            while batch.len() < max_batch {
                match guard.pop_front() {
                    Some(item) => batch.push(item),
                    None => break,
                }
            }
        }
        if batch.is_empty() {
            continue;
        }

        let seqs: Vec<TokenSeq> = batch.iter().map(|(s, _)| *s).collect();
        let (logits, values) = net::forward(&net, &seqs, &device);
        let valid = net::action_valid_mask::<B>(&seqs, &device);
        let neg_inf = valid.bool_not().float() * net::LOGIT_MASK;
        let masked = net::download_f32(logits + neg_inf);
        let vals = net::download_f32(values);
        let action_max = masked.len() / seqs.len().max(1);

        for (i, (_, tx)) in batch.into_iter().enumerate() {
            let row = masked[i * action_max..(i + 1) * action_max].to_vec();
            let v = *vals.get(i).unwrap_or(&0.5);
            let _ = tx.send((row, v));
        }
    }
}
