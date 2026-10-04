//! Scoped reproducibility for simulation and benchmarks. Ordinary games retain
//! the thread RNG; a seed covers the game and all search simulations it calls.
use std::cell::RefCell;
use rand::{rngs::StdRng, Rng, RngCore, SeedableRng};

thread_local! {
    static SEEDED: RefCell<Option<StdRng>> = const { RefCell::new(None) };
}

pub fn with_seed<T>(seed: u64, f: impl FnOnce() -> T) -> T {
    struct Restore(Option<StdRng>);
    impl Drop for Restore {
        fn drop(&mut self) {
            SEEDED.with(|slot| *slot.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(SEEDED.with(|slot| slot.replace(Some(StdRng::seed_from_u64(seed)))));
    f()
}

pub(crate) fn with_rng<T>(f: impl FnOnce(&mut dyn RngCore) -> T) -> T {
    SEEDED.with(|slot| match slot.borrow_mut().as_mut() {
        Some(rng) => f(rng),
        None => f(&mut rand::thread_rng()),
    })
}

pub(crate) fn random<T>() -> T
where
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    with_rng(|rng| rng.gen())
}
