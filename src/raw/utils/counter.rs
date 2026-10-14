use std::sync::{
    atomic::{AtomicIsize, Ordering},
    OnceLock,
};

use super::CachePadded;
use std::num::NonZeroUsize;

// A sharded atomic counter.
//
// Sharding the length counter of `HashMap` is extremely important,
// as a single point of contention for insertions/deletions significantly
// degrades concurrent performance.
//
// We can take advantage of the fact that `seize::Guard`
pub struct Counter(Box<[CachePadded<AtomicIsize>]>);

impl Default for Counter {
    /// Create a new `Counter`.
    fn default() -> Counter {
        // available_parallelism is quite slow (microseconds).
        static CPUS: OnceLock<usize> = OnceLock::new();
        let num_cpus = *CPUS.get_or_init(|| {
            std::thread::available_parallelism()
                .map(Into::into)
                .unwrap_or(1)
        });

        // Round up to the next power-of-two for fast modulo.
        Self::with_shards(NonZeroUsize::new(num_cpus.next_power_of_two()).unwrap())
    }
}

impl Counter {
    pub fn with_shards(shards: NonZeroUsize) -> Self {
        assert!(
            shards.get().is_power_of_two(),
            "counter shards must be a power of two"
        );
        Counter((0..shards.get()).map(|_| Default::default()).collect())
    }

    pub fn shards(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.0.len()).unwrap()
    }

    // Return the shard for the given thread ID.
    #[inline]
    pub fn get(&self, guard: &impl seize::Guard) -> &AtomicIsize {
        // Guard thread IDs are essentially perfectly sharded due to
        // the internal thread ID allocator, which makes contention
        // very unlikely even with the exact number of shards as CPUs.
        let shard = guard.thread_id() & (self.0.len() - 1);

        &self.0[shard].value
    }

    // Returns the sum of all counter shards.
    #[inline]
    pub fn sum(&self) -> usize {
        self.0
            .iter()
            .map(|x| x.value.load(Ordering::Relaxed))
            .sum::<isize>()
            .try_into()
            // Depending on the order of deletion/insertions this might be negative,
            // in which case we assume the map is empty.
            .unwrap_or(0)
    }
}
