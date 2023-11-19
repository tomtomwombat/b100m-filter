//! A very fast bloom filter for Rust.
//! Implemented with L1 cache friendly blocks and efficient hashing.
//!
//! # Examples
//! ```
//! use b100m_filter::BloomFilter;
//!
//! let num_blocks = 4; // by default, each block is 512 bits
//! let values = vec!["42", "qwerty", "bloom"];
//!
//! let filter = BloomFilter::builder(num_blocks).items(values.iter());
//! assert!(filter.contains("42"));
//! assert!(filter.contains("bloom"));
//! ```

use std::{
    hash::{BuildHasher, Hash, Hasher},
    ops::Range,
};

mod hasher;
use hasher::{CloneBuildHasher, DefaultHasher};
mod builder;
pub use builder::Builder;
mod bit_vector;
use bit_vector::BlockedBitVector;

/// Produces a new hash efficiently from two orignal hashes and a new seed.
#[inline]
fn seeded_hash_from_hashes(h1: &mut u64, h2: &mut u64, seed: u64) -> u64 {
    *h1 = h1.wrapping_add(*h2).rotate_left(5);
    *h2 = h2.wrapping_add(seed);
    *h1
}

/// A space efficient approximate membership set data structure.
/// False positives from `contains` are possible, but false negatives
/// are not, i.e. `contains` for all items in the set is guaranteed to return
/// true, while `contains` for all items not in the set probably return false.
///
/// `BloomFilter` is supported by an underlying bit vector, chunked into 512, 256, 128, or 64 bit "blocks", to track item membership.
/// To insert, a number of bits, based on the item's hash, are set in the underlying bit vector.
/// To check membership, a number of bits, based on the item's hash, are checked in the underlying bit vector.
///
/// Once constructed, neither the bloom filter's underlying memory usage nor number of bits per item change.
///
/// # Examples
/// ```
/// use b100m_filter::BloomFilter;
///
/// let num_blocks = 4; // the default for each block is 512 bits
/// let values = vec!["42", "bloom"];
///
/// let mut filter = BloomFilter::builder(num_blocks).items(values.iter());
/// filter.insert("qwerty");
/// assert!(filter.contains("42"));
/// assert!(filter.contains("bloom"));
/// assert!(filter.contains("qwerty"));
/// ```
#[derive(Debug, Clone)]
pub struct BloomFilter<const BLOCK_SIZE_BITS: usize = 512, S = DefaultHasher> {
    bits: BlockedBitVector<BLOCK_SIZE_BITS>,
    num_hashes: u64,
    hasher: S,
}

impl BloomFilter {
    pub(crate) fn new_builder<const BLOCK_SIZE_BITS: usize>(
        num_blocks: usize,
    ) -> Builder<BLOCK_SIZE_BITS> {
        Builder::<BLOCK_SIZE_BITS> {
            num_blocks,
            hasher: Default::default(),
        }
    }

    /// Creates a new instance of `Builder` to construct a `BloomFilter`
    /// with `num_blocks` number of blocks for tracking item membership.
    /// **Each block is 512 bits of memory.**
    ///
    /// Use `builder256`, `builder128`, or `builder64` for more speed
    /// but slightly higher false positive rates.
    ///
    /// # Examples
    ///
    /// ```
    /// use b100m_filter::BloomFilter;
    ///
    /// let bloom = BloomFilter::builder(16).hashes(4);
    /// ```
    pub fn builder(num_blocks: usize) -> Builder<512> {
        Self::builder512(num_blocks)
    }

    /// Creates a new instance of `Builder` to construct a `BloomFilter`
    /// with `num_blocks` number of blocks for tracking item membership.
    /// **Each block is 512 bits of memory.**
    ///
    /// Use `builder256`, `builder128`, or `builder64` for more speed
    /// but slightly higher false positive rates.
    ///
    /// # Examples
    ///
    /// ```
    /// use b100m_filter::BloomFilter;
    ///
    /// let bloom = BloomFilter::builder512(16).hashes(4);
    /// ```
    pub fn builder512(num_blocks: usize) -> Builder<512> {
        Self::new_builder::<512>(num_blocks)
    }

    /// Creates a new instance of `Builder` to construct a `BloomFilter`
    /// with `num_blocks` number of blocks for tracking item membership.
    /// **Each block is 256 bits of memory.**
    ///
    /// `Builder<256>` is faster but less accurate than `Builder<512>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use b100m_filter::BloomFilter;
    ///
    /// let bloom = BloomFilter::builder256(16).hashes(4);
    /// ```
    pub fn builder256(num_blocks: usize) -> Builder<256> {
        Self::new_builder::<256>(num_blocks)
    }

    /// Creates a new instance of `Builder` to construct a `BloomFilter`
    /// with `num_blocks` number of blocks for tracking item membership.
    /// **Each block is 128 bits of memory.**
    ///
    /// `Builder<128>` is faster but less accurate than `Builder<256>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use b100m_filter::BloomFilter;
    ///
    /// let bloom = BloomFilter::builder128(16).hashes(8);
    /// ```
    pub fn builder128(num_blocks: usize) -> Builder<128> {
        Self::new_builder::<128>(num_blocks)
    }

    /// Creates a new instance of `Builder` to construct a `BloomFilter`
    /// with `num_blocks` number of blocks for tracking item membership.
    /// **Each block is 64 bits of memory.**
    ///
    /// `Builder<64>` is faster but less accurate than `Builder<128>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use b100m_filter::BloomFilter;
    ///
    /// let bloom = BloomFilter::builder64(16).hashes(8);
    /// ```
    pub fn builder64(num_blocks: usize) -> Builder<64> {
        Self::new_builder::<64>(num_blocks)
    }
}

impl<const BLOCK_SIZE_BITS: usize, S: BuildHasher> BloomFilter<BLOCK_SIZE_BITS, S> {
    const LOG2_BLOCK_SIZE_BITS: u32 = u32::ilog2(BLOCK_SIZE_BITS as u32);

    /// Used to calculate block index
    const BLOCK_MASK: u64 = {
        let log2_block_size_u64s = Self::LOG2_BLOCK_SIZE_BITS - 6;
        u64::MAX >> (32 + log2_block_size_u64s)
    };
    /// Used to calculate bit index inside a block
    const BIT_MASK: u64 = (1 << Self::LOG2_BLOCK_SIZE_BITS) - 1;

    /// Number of coordinates (i.e. bits in our bloom filter) that can be derived by one hash.
    /// One hash is u64 bits, and we only need 9 bits (LOG2_U64_BITS + LOG2_BLOCK_SIZE) from
    /// the hash for a bit index. For more runtime performance we can cheaply copy an index
    /// from the hash, instead of computing the next hash.
    ///
    /// From experiments, powers of 2 coordinates from the hash provides the best performance
    /// for `contains` for existing and non-existing values.
    const NUM_COORDS_PER_HASH: u32 = 2u32.pow(u32::ilog2(64 / Self::LOG2_BLOCK_SIZE_BITS));

    #[inline]
    fn floor_round(x: f64) -> u64 {
        let floored = x.floor() as u64;
        let thresh = Self::NUM_COORDS_PER_HASH as u64;
        if floored < thresh {
            thresh
        } else {
            floored - (floored % thresh)
        }
    }

    #[inline]
    fn optimal_hashes(num_items: usize) -> u64 {
        let m = BLOCK_SIZE_BITS as f64;
        let n = std::cmp::max(num_items, 1) as f64;
        let num_hashes = m / n * f64::ln(2.0f64);
        Self::floor_round(num_hashes)
    }

    /// Returns a `usize` within the range of `0..self.mem.len()`
    /// A more performant alternative to `hash % self.mem.len()`
    #[inline]
    fn block_index(&self, hash: u64) -> usize {
        (((hash & Self::BLOCK_MASK) as usize * self.bits.len()) >> 32) as usize
    }

    /// Return the bit indexes with a block for an item's two orginal hashes
    #[inline]
    fn bit_indexes(hash1: &mut u64, hash2: &mut u64, seed: u64) -> impl Iterator<Item = u64> {
        let h = seeded_hash_from_hashes(hash1, hash2, seed);
        (0..Self::NUM_COORDS_PER_HASH).map(move |j| {
            // shr: remove right bits from previous bit index (j - 1)
            // and: remove left bits to keep bit index in range of a block's bit size
            h.wrapping_shr(j * Self::LOG2_BLOCK_SIZE_BITS) & Self::BIT_MASK
        })
    }

    /// Returns all seeds that should be used by the hasher
    #[inline]
    fn hash_seeds(size: u64) -> impl Iterator<Item = u64> {
        (0..size).step_by(Self::NUM_COORDS_PER_HASH as usize)
    }

    /// Adds a value to the bloom filter.
    ///
    /// # Examples
    /// ```
    /// use b100m_filter::BloomFilter;
    ///
    /// let mut bloom = BloomFilter::builder(4).hashes(4);
    /// bloom.insert(&2);
    /// assert!(bloom.contains(&2));
    /// ```
    #[inline]
    pub fn insert(&mut self, val: &(impl Hash + ?Sized)) {
        let [mut h1, mut h2] = self.get_orginal_hashes(val);
        let block_index = self.block_index(h1);
        for i in Self::hash_seeds(self.num_hashes) {
            for bit_index in Self::bit_indexes(&mut h1, &mut h2, i) {
                self.bits.set(block_index, bit_index);
            }
        }
    }

    /// Returns `false` if the bloom filter definitely does not contain a value.
    /// Returns `true` if the bloom filter may contain a value, with a degree of certainty.
    ///
    /// # Examples
    ///
    /// ```
    /// use b100m_filter::BloomFilter;
    ///
    /// let bloom = BloomFilter::builder(4).items([1, 2, 3].iter());
    /// assert!(bloom.contains(&1));
    /// ```
    #[inline]
    pub fn contains(&self, val: &(impl Hash + ?Sized)) -> bool {
        let [mut h1, mut h2] = self.get_orginal_hashes(val);
        let block = &self.bits.get_block(self.block_index(h1));
        Self::hash_seeds(self.num_hashes).into_iter().all(|i| {
            BlockedBitVector::<BLOCK_SIZE_BITS>::check_all_for_block(
                block,
                Self::bit_indexes(&mut h1, &mut h2, i),
            )
        })
    }

    /// Returns the effective number of hashes per item. In other words,
    /// the number of bits derived per item.
    ///
    /// For performance reasons, the number of bits is rounded to down to a power of 2, depending on `BLOCK_SIZE_BITS`.
    pub fn num_hashes(&self) -> u64 {
        self.num_hashes
    }

    /// The first two hashes of the value to be inserted or checked.
    ///
    /// Subsequent hashes are efficiently derived from these two using `seeded_hash_from_hashes`,
    /// generating many "random" values for the single value.
    #[inline]
    fn get_orginal_hashes(&self, val: &(impl Hash + ?Sized)) -> [u64; 2] {
        let mut state = self.hasher.build_hasher();
        val.hash(&mut state);
        let hash = state.finish();
        [hash, hash.wrapping_shr(32)]
    }
}

impl<T, const BLOCK_SIZE_BITS: usize, S: BuildHasher> Extend<T> for BloomFilter<BLOCK_SIZE_BITS, S>
where
    T: Hash,
{
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for val in iter {
            self.insert(&val);
        }
    }
}

impl PartialEq for BloomFilter {
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits && self.num_hashes == other.num_hashes
    }
}
impl Eq for BloomFilter {}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::collections::HashSet;

    fn random_strings(num: usize, min_repeat: u32, max_repeat: u32, seed: u64) -> Vec<String> {
        let mut rng = StdRng::seed_from_u64(seed);
        let gen = rand_regex::Regex::compile(r"[a-zA-Z]+", max_repeat).unwrap();
        (&mut rng)
            .sample_iter(&gen)
            .filter(|s: &String| s.len() >= min_repeat as usize)
            .take(num)
            .collect()
    }

    #[test]
    fn random_inserts_always_contained() {
        fn random_inserts_always_contained_<const N: usize>() {
            for mag in 1..6 {
                let size = 10usize.pow(mag);
                for bloom_size_mag in 6..10 {
                    let num_blocks_bytes = 1 << bloom_size_mag;
                    let sample_vals = random_strings(size, 16, 32, 52323);
                    let num_blocks = num_blocks_bytes / (N >> 3);
                    let filter =
                        BloomFilter::new_builder::<N>(num_blocks).items(sample_vals.iter());
                    for x in &sample_vals {
                        assert!(filter.contains(x));
                    }
                }
            }
        }
        random_inserts_always_contained_::<512>();
        random_inserts_always_contained_::<256>();
        random_inserts_always_contained_::<128>();
        random_inserts_always_contained_::<64>();
    }

    #[test]
    fn seeded_is_same() {
        let mag = 3;
        let size = 10usize.pow(mag);
        let bloom_size_bytes = 1 << 10;
        let sample_vals = random_strings(size, 16, 32, 53226);
        let block_size = bloom_size_bytes / 64;
        for x in 0u8..4 {
            let seed = [x; 16];
            let filter1 = BloomFilter::builder(block_size)
                .seed(&seed)
                .items(sample_vals.iter());
            let filter2 = BloomFilter::builder(block_size)
                .seed(&seed)
                .items(sample_vals.iter());
            assert_eq!(filter1, filter2);
        }
    }

    fn false_pos_rate<const N: usize>(filter: &BloomFilter<N>, control: &HashSet<String>) -> f64 {
        let sample_anti_vals = random_strings(10000, 16, 32, 11);
        let mut total = 0;
        let mut false_positives = 0;
        for x in &sample_anti_vals {
            if !control.contains(x) {
                total += 1;
                if filter.contains(x) {
                    false_positives += 1;
                }
            }
        }
        (false_positives as f64) / (total as f64)
    }

    #[test]
    fn false_pos_decrease_with_size() {
        for mag in 1..5 {
            let size = 10usize.pow(mag);
            let mut prev_fp = 1.0;
            let mut prev_prev_fp = 1.0;
            for bloom_size_mag in 6..18 {
                let bloom_size_bytes = 1 << bloom_size_mag;
                let num_blocks = bloom_size_bytes / 64;
                let sample_vals = random_strings(size, 16, 32, 5234);
                let filter = BloomFilter::builder512(num_blocks)
                    .seed(&[1u8; 16])
                    .items(sample_vals.iter());
                let control: HashSet<String> = sample_vals.into_iter().collect();

                let fp = false_pos_rate(&filter, &control);

                println!(
                    "{:?}, {:?}, {:.6}, {:?}",
                    size,
                    bloom_size_bytes,
                    fp,
                    filter.num_hashes(),
                );
                assert!(fp <= prev_fp || prev_fp <= prev_prev_fp); // allows 1 data point to be higher
                prev_prev_fp = prev_fp;
                prev_fp = fp;
            }
        }
    }

    #[test]
    fn test_floor_round() {
        fn assert_floor_round<const N: usize>() {
            let hashes = BloomFilter::<N>::NUM_COORDS_PER_HASH;
            for i in 0..hashes {
                assert_eq!(hashes as u64, BloomFilter::<N>::floor_round(i as f64));
            }
            for i in (hashes as u64..100).step_by(hashes as usize) {
                for j in 0..(hashes as u64) {
                    let x = (i + j) as f64;
                    assert_eq!(i, BloomFilter::<N>::floor_round(x));
                    assert_eq!(i, BloomFilter::<N>::floor_round(x + 0.9999));
                    assert_eq!(i, BloomFilter::<N>::floor_round(x + 0.0001));
                }
            }
        }
        assert_floor_round::<512>();
        assert_floor_round::<256>();
        assert_floor_round::<128>();
        assert_floor_round::<64>();
    }

    fn assert_even_distribution(distr: &[u64], err: f64) {
        assert!(err > 0.0 && err < 1.0);
        let expected: i64 = (distr.iter().sum::<u64>() / (distr.len() as u64)) as i64;
        let thresh = (expected as f64 * err) as i64;
        for x in distr {
            let diff = (*x as i64 - expected).abs();
            assert!(
                diff <= thresh,
                "{x:?} deviates from {expected:?} (err: {err:?})"
            );
        }
    }

    #[test]
    fn block_hash_distribution() {
        fn block_hash_distribution_<const N: usize>(mut filter: BloomFilter<N>) {
            let mut rng = StdRng::seed_from_u64(1);
            let iterations = 1000000;
            let mut buckets = vec![0; filter.bits.num_blocks()];
            for _ in 0..iterations {
                let h1 = (&mut rng).gen_range(0..u64::MAX);
                buckets[filter.block_index(h1)] += 1;
            }
            assert_even_distribution(&buckets, 0.05);
        }
        let num_blocks = 100;
        let seed = [0; 16];
        block_hash_distribution_::<512>(BloomFilter::builder512(num_blocks).seed(&seed).hashes(1));
        block_hash_distribution_::<256>(BloomFilter::builder256(num_blocks).seed(&seed).hashes(1));
        block_hash_distribution_::<128>(BloomFilter::builder128(num_blocks).seed(&seed).hashes(1));
        block_hash_distribution_::<64>(BloomFilter::builder64(num_blocks).seed(&seed).hashes(1));
    }

    #[test]
    fn test_seeded_hash_from_hashes() {
        let mut rng = StdRng::seed_from_u64(524323);
        let mut h1 = (&mut rng).gen_range(0..u64::MAX);
        let mut h2 = (&mut rng).gen_range(0..u64::MAX);
        let size = 1000;
        let mut seeded_hash_counts = vec![0; size];
        let iterations = 10000000;
        for i in 0..iterations {
            let hi = seeded_hash_from_hashes(&mut h1, &mut h2, i);
            seeded_hash_counts[(hi as usize) % size] += 1;
        }
        assert_even_distribution(&seeded_hash_counts, 0.05);
    }

    #[test]
    fn index_hash_distribution() {
        fn index_hash_distribution_<const N: usize>(filter: BloomFilter<N>, thresh_pct: f64) {
            let [mut h1, mut h2] = filter.get_orginal_hashes("qwerty");
            let mut counts = vec![0; N];
            let iterations = 100000;
            for i in 0..iterations {
                for bit_index in BloomFilter::<N>::bit_indexes(&mut h1, &mut h2, i) {
                    let index = bit_index as usize % N;
                    counts[index] += 1;
                }
            }
            assert_even_distribution(&counts, thresh_pct);
        }
        let seed = [0; 16];
        index_hash_distribution_::<512>(BloomFilter::builder512(1).seed(&seed).hashes(1), 0.2);
        index_hash_distribution_::<256>(BloomFilter::builder256(1).seed(&seed).hashes(1), 0.05);
        index_hash_distribution_::<128>(BloomFilter::builder128(1).seed(&seed).hashes(1), 0.05);
        index_hash_distribution_::<64>(BloomFilter::builder64(1).seed(&seed).hashes(1), 0.05);
    }

    #[test]
    fn test_debug() {
        let filter = BloomFilter::builder64(1).hashes(1);
        assert!(!format!("{:?}", filter).is_empty());
    }

    #[test]
    fn test_clone() {
        let filter = BloomFilter::builder(4).hashes(4);
        assert_eq!(filter, filter.clone());
    }
}
