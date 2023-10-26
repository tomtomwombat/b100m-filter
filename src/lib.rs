use getrandom::getrandom;
use siphasher::sip::SipHasher13;
use std::hash::{Hash, Hasher};

pub(crate) mod test_util;

/// u64 have 64 bits, and therefore are used to store 64 elements in the bloom filter.
/// We use a bitmaks with a single bit set to interpret a number as a bit index.
/// 2^6 - 1 = 63 = the max bit index of a u64.
const LOG2_U64_BITS: u32 = u32::ilog2(u64::BITS);

/// Number of bytes per block, matching a typical L1 cache line size.
const BLOCK_SIZE_BYTES: usize = 64;

/// Number of u64's (8 bytes per u64) per block, matching a typical L1 cache line size.
const BLOCK_SIZE: usize = BLOCK_SIZE_BYTES / 8;

/// Used to shift u64 index
const LOG2_BLOCK_SIZE: u32 = u32::ilog2(BLOCK_SIZE as u32);

/// Gets 6 last bits from the hash
const BIT_MASK: u64 = 0b0000000000000000000000000000000000000000000000000000000000111111;
/// Gets 3 last bits from the shifted hash
const U64_MASK: u64 = 0b0000000000000000000000000000000000000000000000000000000000000111;

/// Returns the first u64 and bit index pair from the last 9 bits of the hash.
/// From the last 9 bits:
/// The u64 index is the first 3 bits, shifted right.
/// The bit index is the last 6 bits.
#[inline]
fn coordinate_1(hash: u64) -> (usize, u64) {
    let index = ((hash >> LOG2_U64_BITS) & U64_MASK) as usize;
    let bit = 1u64 << (hash & BIT_MASK);
    (index, bit)
}

/// Returns the first u64 and bit index pair from the first 9 bits of the hash.
/// From the first 9 bits:
/// The bit index is the first 6 bits.
/// The u64 index is the last 3 bits, shifted right.
#[inline]
const fn coordinate_2(hash: u64) -> (usize, u64) {
    let index = (hash.rotate_left(LOG2_U64_BITS + LOG2_BLOCK_SIZE) & U64_MASK) as usize;
    let bit = 1u64 << (hash.rotate_left(LOG2_U64_BITS) & BIT_MASK);
    (index, bit)
}

#[inline]
fn optimal_hashes(num_bits: usize, num_items: usize) -> u64 {
    let m = num_bits as f64;
    let n = std::cmp::max(num_items, 1) as f64;
    let num_hashes = m / n * f64::ln(2.0f64);
    floor_round_2(num_hashes)
}

#[inline]
fn floor_round_2(x: f64) -> u64 {
    match (2.0 * x).floor() as u64 {
        0..=3 => 2,
        x => 2 * (x / 4),
    }
}

#[inline]
fn hashes_for_approximate_rate(_num_bits: usize, _num_items: usize, rate: f64) -> u64 {
    assert!(rate < 1.0 && rate > 0.0);
    assert!(_num_items > 0);
    floor_round_2(-f64::log2(rate))
}

pub struct BloomFilter {
    mem: Vec<[u64; BLOCK_SIZE]>,
    num_hashes: u64,
    hashers: [SipHasher13; 2],
}

impl BloomFilter {
    /// Constructs a bloom filter from the number of blocks (where each block is 512 bits),
    /// and the number of hashes. For performance, the actual number of hashes performed internally will be
    /// rounded to down to the nearest even number.
    pub fn with_num_hashes(num_blocks: usize, num_hashes: u64) -> Self {
        // assert!(num_hashes % 2 == 0);
        let mut seed1 = [0u8; 16];
        let mut seed2 = [0u8; 16];
        getrandom(&mut seed1).unwrap();
        getrandom(&mut seed2).unwrap();
        let hashers = [
            SipHasher13::new_with_key(&seed1),
            SipHasher13::new_with_key(&seed2),
        ];
        Self {
            mem: vec![[0u64; BLOCK_SIZE]; num_blocks],
            num_hashes,
            hashers,
        }
    }

    /// Constructs a bloom filter from the number of blocks (where each block is 512 bits),
    /// and all the items from `vals`. The number of hashes is set for the optimal false
    /// positive rate based on the number on `vals.len()`.
    pub fn from_vec(num_blocks: usize, vals: &[impl Hash]) -> Self {
        let num_hashes = optimal_hashes(BLOCK_SIZE * 64, vals.len() / num_blocks);
        let mut filter = Self::with_num_hashes(num_blocks, num_hashes);
        filter.insert_all(vals);
        filter
    }

    /*
    // TODO, test this
    pub fn with_false_negative_rate(num_blocks: usize, num_items: usize, rate: f64) -> Self {
        let num_hashes = hashes_for_approximate_rate(BLOCK_SIZE * num_blocks * 64, num_items, rate);
        Self::with_num_hashes(num_blocks, num_hashes)
    }
    */

    /// Returns the number of hashes per element in the bloom filter.
    /// The returned value is always a multiple of two due to internal
    /// optimizations.
    pub fn num_hashes(&self) -> u64 {
        self.num_hashes
    }

    /// Produces a hash efficiently from two orignal hashes.
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
        (((hash >> 32) as usize * self.mem.len()) >> 32) as usize
    }

    /// Adds a value to the bloom filter.
    ///
    /// # Examples
    /// ```
    /// use bloom_filter::BloomFilter;
    ///
    /// let mut bloom = BloomFilter::from_vec(4, &[1]);
    /// bloom.insert(&2);
    /// assert!(bloom.contains(&1));
    /// assert!(bloom.contains(&2));
    /// ```
    #[inline]
    pub fn insert(&mut self, val: &(impl Hash + ?Sized)) {
        let [mut h1, mut h2] = self.get_orginal_hashes(val);
        let block_index = self.to_index(h1);
        for i in (0..self.num_hashes).step_by(2) {
            let h = Self::seeded_hash_from_hashes(&mut h1, &mut h2, i);
            let (index1, bit1) = coordinate_1(h);
            let (index2, bit2) = coordinate_2(h);
            self.mem[block_index][index1] |= bit1;
            self.mem[block_index][index2] |= bit2;
        }
    }

    /// Adds all the values to the bloom filter.
    ///
    /// # Examples
    /// ```
    /// use bloom_filter::BloomFilter;
    ///
    /// let mut bloom = BloomFilter::from_vec(4, &[1]);
    /// bloom.insert_all(&[2, 3]);
    /// assert!(bloom.contains(&1));
    /// assert!(bloom.contains(&2));
    /// assert!(bloom.contains(&3));
    /// ```
    #[inline]
    pub fn insert_all(&mut self, vals: &[impl Hash]) {
        for val in vals {
            self.insert(&val);
        }
    }

    /// Returns `false` if the bloom filter definitely does not contain a value.
    /// Returns `true` if the bloom filter may contain a value, with a degree of certainty.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloom_filter::BloomFilter;
    ///
    /// let bloom = BloomFilter::from_vec(4, &[1, 2, 3]);
    /// assert!(bloom.contains(&1));
    /// ```
    #[inline]
    pub fn contains(&self, val: &(impl Hash + ?Sized)) -> bool {
        let [mut h1, mut h2] = self.get_orginal_hashes(val);
        let block_index = self.to_index(h1);
        let cached_block = self.mem[block_index];
        (0..self.num_hashes).step_by(2).into_iter().all(|i| {
            let h = Self::seeded_hash_from_hashes(&mut h1, &mut h2, i);
            let (index1, bit1) = coordinate_1(h);
            let (index2, bit2) = coordinate_2(h);
            (cached_block[index1] & bit1 > 0) && (cached_block[index2] & bit2 > 0)
        })
    }

    #[inline]
    fn get_orginal_hashes(&self, val: &(impl Hash + ?Sized)) -> [u64; 2] {
        let mut hashes = [0u64, 0u64];
        for k_i in 0..2u64 {
            let sip = &mut self.hashers[k_i as usize].clone();
            val.hash(sip);
            let hash = sip.finish();
            hashes[k_i as usize] = hash;
        }
        hashes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::*;
    use bloomfilter;
    use std::collections::HashSet;

    #[test]
    fn random_stuff() {
        // lets look at num items per block and num false pos per block
        let size = 10;
        let sample_vals = random_strings(size, 16, 32, *b"seedseedseedseed");
        let sample_anti_vals = random_strings(1000000, 16, 32, *b"antiantiantianti");
        let mut control: HashSet<String> = HashSet::new();
        let bloom_size_bytes = 4096 * 2;
        let block_size = bloom_size_bytes / 64;
        let mut bloom1: bloomfilter::Bloom<String> =
            bloomfilter::Bloom::new(bloom_size_bytes, sample_vals.len());
        let mut filter = BloomFilter::from_vec(block_size, &sample_vals);
        println!("num hashes: {:?}", filter.num_hashes as u64);
        println!(
            "num hashes bloomfilter::Bloom: {:?}",
            bloom1.number_of_hash_functions()
        );
        for x in &sample_vals {
            control.insert(x.clone());
            bloom1.set(x);
        }
        for x in sample_vals {
            assert!(filter.contains(&x));
        }

        let mut total = 0;
        let mut false_positives = 0;
        let mut false_positives_control = 0;
        for x in &sample_anti_vals {
            if !control.contains(x) {
                total += 1;
                if filter.contains(x) {
                    false_positives += 1;
                }
                if bloom1.check(x) {
                    false_positives_control += 1;
                }
            }
        }
        println!(
            "False Positives: {:?}",
            (false_positives as f64) / (total as f64)
        );
        println!(
            "False Positives Control: {:?}",
            (false_positives_control as f64) / (total as f64)
        );
    }

    // #[test] TODO make this into a real test
    fn compare_false_positives() {
        for mag in 1..7 {
            let size = 10usize.pow(mag);
            for bloom_size_mag in 6..22 {
                let bloom_size_bytes = 1 << bloom_size_mag;
                let block_size = bloom_size_bytes / 64;
                let sample_vals = random_strings(size, 16, 32, *b"seedseedseedseed");
                let sample_anti_vals = random_strings(100000, 16, 32, *b"antiantiantianti");
                let mut control: HashSet<String> = HashSet::new();

                let mut bloom1: bloomfilter::Bloom<String> =
                    bloomfilter::Bloom::new(bloom_size_bytes, sample_vals.len());
                let mut filter = BloomFilter::from_vec(block_size, &sample_vals);

                for x in &sample_vals {
                    control.insert(x.clone());
                    bloom1.set(x);
                }
                for x in sample_vals {
                    assert!(filter.contains(&x));
                }

                let mut total = 0;
                let mut false_positives = 0;
                let mut false_positives_control = 0;
                for x in &sample_anti_vals {
                    if !control.contains(x) {
                        total += 1;
                        if filter.contains(x) {
                            false_positives += 1;
                        }
                        if bloom1.check(x) {
                            false_positives_control += 1;
                        }
                    }
                }

                let fp = (false_positives as f64) / (total as f64);
                if fp <= 0.15 {
                    println!(
                        "{:?}, {:?}, {:.6}, {:.6}, {:?}, {:?}, {:.6}",
                        size,
                        bloom_size_bytes,
                        (false_positives_control as f64) / (total as f64),
                        fp,
                        bloom1.number_of_hash_functions(),
                        filter.num_hashes(),
                        ((false_positives_control as f64) / (total as f64)) - fp
                    );
                }
            }
        }
    }

    #[test]
    fn test_floor_round_2() {
        for x in [0.0, 0.01, 0.1, 1.0, 1.1, 1.99999, 2.0001, 2.0] {
            assert_eq!(floor_round_2(x), 2, "Wrong for {x:?}");
        }

        for x in [4.0, 4.01, 4.1, 4.0, 4.1, 4.99999, 5.0001, 5.0] {
            assert_eq!(floor_round_2(x), 4, "Wrong for {x:?}");
        }
    }
}
