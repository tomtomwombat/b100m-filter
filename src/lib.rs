//! A very fast bloom filter for Rust.
//! Implemented with L1 cache friendly array blocks and efficient hashing.
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

use getrandom::getrandom;
use siphasher::sip::SipHasher13;
use std::{
    hash::{Hash, Hasher},
    iter::ExactSizeIterator,
    ops::Range,
};

#[cfg(test)]
pub(crate) mod test_util;

/// Grab the lower 32 bits of a u64
const U32_MASK_LOWER: u64 = 0b0000000000000000000000000000000011111111111111111111111111111111;

/// u64 have 64 bits, and therefore are used to store 64 elements in the bloom filter.
/// We use a bitmaks with a single bit set to interpret a number as a bit index.
/// 2^6 - 1 = 63 = the max bit index of a u64.
const LOG2_U64_BITS: u32 = u32::ilog2(u64::BITS);

/// Gets 6 last bits from the hash
const BIT_MASK: u64 = (1 << LOG2_U64_BITS) - 1;

/// A bloom filter builder
///
/// This type can be used to construct an instance of `BloomFilter`
/// through a builder-like pattern.
#[derive(Debug, Clone)]
pub struct Builder<const BLOCK_SIZE_BITS: usize = 512> {
    num_blocks: usize,
    seed: [u8; 16],
}

impl<const BLOCK_SIZE_BITS: usize> Builder<BLOCK_SIZE_BITS> {
    /// Sets the seed for this builder. The later constructed `BloomFilter`
    /// will use this seed when hashing items.
    ///
    /// # Examples
    ///
    /// ```
    /// use b100m_filter::BloomFilter;
    ///
    /// let bloom = BloomFilter::builder(4).seed(&[0u8; 16]).hashes(4);
    /// ```
    pub fn seed(mut self, seed: &[u8; 16]) -> Self {
        self.seed.copy_from_slice(seed);
        self
    }

    /// "Consumes" this builder, using the provided `num_hashes` to return an
    /// empty `BloomFilter`. For performance, the actual number of
    /// hashes performed internally will be rounded to down to a power of 2.
    ///
    /// # Examples
    ///
    /// ```
    /// use b100m_filter::BloomFilter;
    ///
    /// let bloom = BloomFilter::builder(4).hashes(4);
    /// ```
    pub fn hashes(self, num_hashes: u64) -> BloomFilter {
        BloomFilter {
            mem: vec![0u64; BLOCK_SIZE_BITS / 64 * self.num_blocks],
            num_hashes,
            seed: self.seed,
            hasher: SipHasher13::new_with_key(&self.seed),
        }
    }

    /// "Consumes" this builder, using the provided `expected_num_items` to return an
    /// empty `BloomFilter`. More or less than `expected_num_items` may be inserted into
    /// `BloomFilter`, but the number of hashes per item is intially calculated
    /// to minimize false positive rate for exactly `expected_num_items`.
    ///
    /// # Examples
    ///
    /// ```
    /// use b100m_filter::BloomFilter;
    ///
    /// let bloom = BloomFilter::builder(4).expected_items(500);
    /// ```
    pub fn expected_items(self, expected_num_items: usize) -> BloomFilter {
        let num_hashes =
            BloomFilter::<BLOCK_SIZE_BITS>::optimal_hashes(expected_num_items / self.num_blocks);
        self.hashes(num_hashes)
    }

    /// "Consumes" this builder and constructs a `BloomFilter` containing
    /// all values in `items`. The number of hashes per item is calculated
    /// based on `items.len()` to minimize false positive rate.
    ///
    /// # Examples
    ///
    /// ```
    /// use b100m_filter::BloomFilter;
    ///
    /// let bloom = BloomFilter::builder(4).items([1, 2, 3].iter());
    /// ```
    pub fn items<I: ExactSizeIterator<Item = impl Hash>>(self, items: I) -> BloomFilter {
        let mut filter = self.expected_items(items.len());
        filter.extend(items);
        filter
    }
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
pub struct BloomFilter<const BLOCK_SIZE_BITS: usize = 512> {
    mem: Vec<u64>,
    num_hashes: u64,
    seed: [u8; 16],
    hasher: SipHasher13,
}

impl BloomFilter {
    fn random_seed() -> [u8; 16] {
        let mut seed = [0u8; 16];
        getrandom(&mut seed).unwrap();
        seed
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
    pub fn builder(num_blocks: usize) -> Builder {
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
        Builder::<512> {
            num_blocks,
            seed: Self::random_seed(),
        }
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
        Builder::<256> {
            num_blocks,
            seed: Self::random_seed(),
        }
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
    /// let bloom = BloomFilter::builder128(16).hashes(4);
    /// ```
    pub fn builder128(num_blocks: usize) -> Builder<128> {
        Builder::<128> {
            num_blocks,
            seed: Self::random_seed(),
        }
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
    /// let bloom = BloomFilter::builder64(16).hashes(4);
    /// ```
    pub fn builder64(num_blocks: usize) -> Builder<64> {
        Builder::<64> {
            num_blocks,
            seed: Self::random_seed(),
        }
    }
}

impl<const BLOCK_SIZE_BITS: usize> BloomFilter<BLOCK_SIZE_BITS> {
    const BLOCK_SIZE: usize = BLOCK_SIZE_BITS / 64;

    /// Used to shift u64 index
    const LOG2_BLOCK_SIZE: u32 = u32::ilog2(Self::BLOCK_SIZE as u32);

    /// Gets 3 last bits from the shifted hash
    const U64_MASK: u64 = (1 << Self::LOG2_BLOCK_SIZE) - 1;

    /// Number of coordinates (i.e. bits in our bloom filter) that can be derived by one hash.
    /// One hash is u64 bits, and we only need 9 bits (LOG2_U64_BITS + LOG2_BLOCK_SIZE) from
    /// the hash for a bit index. For more runtime performance we can cheaply copy an index
    /// from the hash, instead of computing the next hash.
    ///
    /// From experiments, powers of 2 coordinates from the hash provides the best performance
    /// for `contains` for existing and non-existing values.
    const NUM_COORDS_PER_HASH: u32 =
        2u32.pow(u32::ilog2(64 / (LOG2_U64_BITS + Self::LOG2_BLOCK_SIZE)));

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
        let m = (u64::BITS as usize * Self::BLOCK_SIZE) as f64;
        let n = std::cmp::max(num_items, 1) as f64;
        let num_hashes = m / n * f64::ln(2.0f64);
        Self::floor_round(num_hashes)
    }

    /// Returns the number of bits derived per item for the bloom filter.
    /// This number is effectivly the number of hashes per item, but
    /// all hashes are not actually performed.
    ///
    /// The returned value is always a multiple of 4 due to internal
    /// optimizations.
    pub fn num_hashes(&self) -> u64 {
        self.num_hashes
    }

    /// Returns the first u64 and bit index pair from 9 bits of the hash.
    /// Which 9 bits is determined from `seed`.
    /// From the those 9 bits:
    /// The u64 index is the first 3 bits, shifted right.
    /// The bit index is the last 6 bits.
    #[inline]
    const fn coordinate(hash: u64, seed: u32) -> (usize, u64) {
        let offset = seed * (LOG2_U64_BITS + Self::LOG2_BLOCK_SIZE);
        let index = hash.wrapping_shr(LOG2_U64_BITS + offset) & Self::U64_MASK;
        let bit = 1u64 << (hash.wrapping_shr(offset) & BIT_MASK);
        (index as usize, bit)
    }

    /// Produces a new hash efficiently from two orignal hashes and a new seed.
    #[inline]
    fn seeded_hash_from_hashes(h1: &mut u64, h2: &mut u64, seed: u64) -> u64 {
        *h1 = h1.wrapping_add(*h2);
        *h2 = h2.wrapping_add(seed);
        *h1
    }

    /// Returns a `usize` within the range of `0..self.mem.len()`
    /// A more performant alternative to `hash % self.mem.len()`
    #[inline]
    fn to_index(&self, hash: u64) -> usize {
        ((((hash & U32_MASK_LOWER) as usize * self.mem.len()) >> 32) as usize)
            >> Self::LOG2_BLOCK_SIZE
    }

    #[inline]
    fn block_range(&self, hash: u64) -> Range<usize> {
        let block_index = self.to_index(hash) * Self::BLOCK_SIZE;
        block_index..(block_index + Self::BLOCK_SIZE)
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
        let block_index = self.block_range(h1);
        let block = &mut self.mem[block_index];
        for i in (0..self.num_hashes).step_by(Self::NUM_COORDS_PER_HASH as usize) {
            let h = Self::seeded_hash_from_hashes(&mut h1, &mut h2, i);
            for j in 0..Self::NUM_COORDS_PER_HASH {
                let (index, bit) = Self::coordinate(h, j);
                block[index] |= bit;
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
        let block_index = self.block_range(h1);
        let block = &self.mem[block_index];
        (0..self.num_hashes)
            .step_by(Self::NUM_COORDS_PER_HASH as usize)
            .into_iter()
            .all(|i| {
                let h = Self::seeded_hash_from_hashes(&mut h1, &mut h2, i);
                (0..Self::NUM_COORDS_PER_HASH).all(|j| {
                    let (index, bit) = Self::coordinate(h, j);
                    block[index] & bit > 0
                })
            })
    }

    #[inline]
    fn get_orginal_hashes(&self, val: &(impl Hash + ?Sized)) -> [u64; 2] {
        let hasher = &mut self.hasher.clone();
        val.hash(hasher);
        let hash = hasher.finish();
        [hash, hash.wrapping_shr(32)]
    }
}

impl<T> Extend<T> for BloomFilter
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
        self.mem == other.mem && self.seed == other.seed && self.num_hashes == other.num_hashes
    }
}
impl Eq for BloomFilter {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::*;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;

    #[test]
    fn random_inserts_always_contained() {
        for mag in 1..7 {
            let size = 10usize.pow(mag);
            for bloom_size_mag in 6..10 {
                let bloom_size_bytes = 1 << bloom_size_mag;
                let sample_vals = random_strings(size, 16, 32, *b"seedseedseedseed");
                let mut control: HashSet<String> = HashSet::new();
                let block_size = bloom_size_bytes / 64;
                let filter = BloomFilter::builder(block_size).items(sample_vals.iter());
                for x in &sample_vals {
                    control.insert(x.clone());
                    assert!(filter.contains(x));
                }
            }
        }
    }

    #[test]
    fn seeded_is_same() {
        let mag = 3;
        let size = 10usize.pow(mag);
        let bloom_size_bytes = 1 << 10;
        let sample_vals = random_strings(size, 16, 32, *b"seedseedseedseed");
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

    #[test]
    fn false_pos_decrease_with_size() {
        for mag in 1..5 {
            let size = 10usize.pow(mag);
            let mut prev_fp = 1.0;
            let mut prev_prev_fp = 1.0;
            for bloom_size_mag in 6..18 {
                let bloom_size_bytes = 1 << bloom_size_mag;
                let block_size = bloom_size_bytes / 64;
                let sample_vals = random_strings(size, 16, 32, *b"seedseedseedseed");
                let sample_anti_vals = random_strings(100000, 16, 32, *b"antiantiantianti");
                let mut control: HashSet<String> = HashSet::new();
                let seed = [1u8; 16];
                let filter = BloomFilter::builder(block_size)
                    .seed(&seed)
                    .items(sample_vals.iter());

                for x in &sample_vals {
                    control.insert(x.clone());
                    assert!(filter.contains(x));
                }

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

                let fp = (false_positives as f64) / (total as f64);
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
        let hashes = BloomFilter::<512>::NUM_COORDS_PER_HASH;
        for i in 0..hashes {
            assert_eq!(hashes as u64, BloomFilter::<512>::floor_round(i as f64));
        }
        for i in (hashes as u64..100).step_by(hashes as usize) {
            for j in 0..(hashes as u64) {
                let x = (i + j) as f64;
                assert_eq!(i, BloomFilter::<512>::floor_round(x));
                assert_eq!(i, BloomFilter::<512>::floor_round(x + 0.9999));
                assert_eq!(i, BloomFilter::<512>::floor_round(x + 0.0001));
            }
        }
    }

    fn assert_even_distribution(distr: Vec<u64>, expected: u64, threshold: u64) {
        for x in distr {
            let diff = ((x as i64) - (expected as i64)).abs();
            assert!(
                diff <= threshold as i64,
                "{x:?} deviates from {expected:?} (threshold: {threshold:?})"
            );
        }
    }

    #[test]
    fn block_hash_distribution() {
        let mut rng = rand_xorshift::XorShiftRng::from_seed(*b"seedseedseedseed");

        let num_blocks = 100;
        let filter = BloomFilter::builder(num_blocks).items(vec![1].iter());
        let mut counts = vec![0u64; num_blocks];

        let iterations = 10000000;
        for _ in 0..iterations {
            let h1 = (&mut rng).gen_range(0..u64::MAX);
            let block_index = filter.to_index(h1);
            println!("{:?}", block_index);
            counts[block_index] += 1;
        }
        let thresh = ((1.025 * (iterations / num_blocks) as f64)
            - ((iterations / num_blocks) as f64)) as u64;
        assert_even_distribution(counts, (iterations / num_blocks) as u64, thresh);
    }

    #[test]
    fn index_hash_distribution() {
        let mut rng = rand_xorshift::XorShiftRng::from_seed(*b"seedseedseedseed");
        let mut h1 = (&mut rng).gen_range(0..u64::MAX);
        let mut h2 = (&mut rng).gen_range(0..u64::MAX);

        let mut counts = vec![vec![0u64; 64]; BloomFilter::<512>::BLOCK_SIZE];

        let iterations = 10000000;
        for i in 0..iterations {
            let hi = BloomFilter::<512>::seeded_hash_from_hashes(&mut h1, &mut h2, i);
            for i in 1..BloomFilter::<512>::NUM_COORDS_PER_HASH + 1 {
                let (index, bit) = BloomFilter::<512>::coordinate(hi, i);
                let bit_index = u64::ilog2(bit) as usize;
                counts[index][bit_index] += 1;
            }
        }
        let total_iterations = (iterations * BloomFilter::<512>::NUM_COORDS_PER_HASH as u64) as u64;
        let total_coords = (BloomFilter::<512>::BLOCK_SIZE as u64 * 64) as u64;
        let thresh = ((1.025 * (total_iterations / total_coords) as f64)
            - ((total_iterations / total_coords) as f64)) as u64;
        println!("{:?}", counts);
        assert_even_distribution(
            counts.into_iter().flatten().collect(),
            (total_iterations / total_coords) as u64,
            thresh,
        );
    }
}
