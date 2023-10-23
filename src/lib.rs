use getrandom::getrandom;
use siphasher::sip::SipHasher13;
use std::cmp;
use std::hash::Hash;
use std::hash::Hasher;

const U64_RANGE: u32 = 6;

const BLOCK_SIZE: usize = 512 / 64; // 8
const MAG: u32 = 3;
const ROUND_MASK: u64 = (1 << (U64_RANGE + MAG) as u64) - 1;
const BIT_MASK_1: u64 = (1 << 6) - 1;
const INDEX_MASK_1: u64 = !BIT_MASK_1 & ROUND_MASK;
const BIT_MASK_2: u64 = BIT_MASK_1.reverse_bits();
const INDEX_MASK_2: u64 = INDEX_MASK_1.reverse_bits();

pub mod test_util;

pub struct BloomFilter {
    mem: Vec<[u64; BLOCK_SIZE]>,
    pub num_hashes: u64,
    hashers: [SipHasher13; 2],
}

impl BloomFilter {
    pub fn from_vec(num_blocks: usize, vals: &Vec<impl Hash>) -> Self {
        let total_size = BLOCK_SIZE * num_blocks;
        let mut seed1 = [0u8; 16];
        let mut seed2 = [0u8; 16];
        getrandom(&mut seed1).unwrap();
        getrandom(&mut seed2).unwrap();
        let hashers = [
            SipHasher13::new_with_key(&seed1),
            SipHasher13::new_with_key(&seed2),
        ];
        let mut filter = Self {
            mem: vec![[0u64; BLOCK_SIZE]; num_blocks],
            num_hashes: Self::optimal_hashes(total_size, vals.len()),
            hashers,
        };
        for val in vals {
            filter.insert(&val);
        }
        filter
    }

    // TODO: check this
    pub fn optimal_hashes(size: usize, items_count: usize) -> u64 {
        let m = (size * 64) as f64;
        let n = items_count as f64;
        let k_num = (m / n * f64::ln(2.0f64)).ceil() as u64;
        cmp::max(k_num, 1)
    }

    #[inline]
    fn reduce_range(&self, i: u64) -> usize {
        (((i >> 32) as usize * self.mem.len()) >> 32) as usize
    }

    #[inline]
    fn coordinate_1(&self, hash: u64) -> (usize, u64) {
        let index1 = ((hash & INDEX_MASK_1) >> U64_RANGE) as usize;
        let bit1 = 1u64 << (hash & BIT_MASK_1);
        (index1, bit1)
    }

    #[inline]
    fn coordinate_2(&self, hash: u64) -> (usize, u64) {
        let index2 = ((hash & INDEX_MASK_2).wrapping_shl(U64_RANGE + MAG)) as usize;
        let bit2 = 1u64 << (hash & BIT_MASK_2).wrapping_shl(U64_RANGE);
        (index2, bit2)
    }

    #[inline]
    fn seeded_hash_from_hashes(&self, h1: &mut u64, h2: &mut u64, seed: u64) -> u64 {
        *h1 = h1.wrapping_add(*h2);
        *h2 = h2.wrapping_add(seed);
        *h1
    }

    #[inline]
    pub fn insert(&mut self, val: &(impl Hash + ?Sized)) {
        let [mut h1, mut h2] = self.get_orginal_hashes(val);
        let block_index = self.reduce_range(h1);
        for i in (0..self.num_hashes).step_by(2) {
            let h = self.seeded_hash_from_hashes(&mut h1, &mut h2, i);
            let (index1, bit1) = self.coordinate_1(h);
            let (index2, bit2) = self.coordinate_2(h);
            self.mem[block_index][index1] |= bit1;
            self.mem[block_index][index2] |= bit2;
        }
    }

    #[inline]
    pub fn contains(&self, val: &(impl Hash + ?Sized)) -> bool {
        let [mut h1, mut h2] = self.get_orginal_hashes(val);
        let block_index = self.reduce_range(h1);
        let cached_block = self.mem[block_index];
        (0..self.num_hashes).step_by(2).into_iter().all(|i| {
            let h = self.seeded_hash_from_hashes(&mut h1, &mut h2, i);
            let (index1, bit1) = self.coordinate_1(h);
            let (index2, bit2) = self.coordinate_2(h);
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
    use std::collections::HashSet;

    #[test]
    fn random_stuff() {
        let size = 1000;
        let sample_vals = random_strings(size, 32, *b"seedseedseedseed");
        let sample_anti_vals = random_strings(size, 32, *b"antiantiantianti");
        let mut control: HashSet<String> = HashSet::new();

        let filter = BloomFilter::from_vec(1024, &sample_vals);
        println!("num hashes: {:?}", filter.num_hashes as u64);

        for x in &sample_vals {
            control.insert(x.clone());
        }
        for x in sample_vals {
            assert!(filter.contains(&x));
        }

        let mut false_positives = 0;
        for x in sample_anti_vals {
            if filter.contains(&x) && !control.contains(&x) {
                false_positives += 1;
            }
        }
        println!("false_positives: {false_positives:?}");
    }
}
