use std::ops::Range;

/// u64s have 64 bits, and therefore are used to store 64 elements in the bloom filter.
/// We use a bitmask with a single bit set to interpret a number as a bit index.
/// 2^6 - 1 = 63 = the max bit index of a u64.
const BIT_MASK_LEN: u32 = u32::ilog2(u64::BITS);

/// Gets 6 last bits from the index
const BIT_MASK: u64 = (1 << BIT_MASK_LEN) - 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockedBitVector<const BLOCK_SIZE_BITS: usize> {
    pub(crate) bits: Vec<u64>,
}

impl<const BLOCK_SIZE_BITS: usize> BlockedBitVector<BLOCK_SIZE_BITS> {
    /// Block size in u64s
    const BLOCK_SIZE: usize = BLOCK_SIZE_BITS / 64;
    /// Used to shift u64 index
    const LOG2_BLOCK_SIZE: u32 = u32::ilog2(Self::BLOCK_SIZE as u32);

    pub fn new(num_blocks: usize) -> Self {
        Self {
            bits: vec![0u64; num_blocks * Self::BLOCK_SIZE],
        }
    }

    #[inline]
    pub(crate) const fn block_range(index: usize) -> Range<usize> {
        let block_index = index * Self::BLOCK_SIZE;
        block_index..(block_index + Self::BLOCK_SIZE)
    }

    #[allow(dead_code)]
    #[inline]
    pub fn bits(&self) -> &[u64] {
        &self.bits
    }

    #[allow(dead_code)]
    #[inline]
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    #[allow(dead_code)]
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.bits.len() >> Self::LOG2_BLOCK_SIZE
    }

    #[inline]
    pub fn get_block(&self, index: usize) -> &[u64] {
        &self.bits[Self::block_range(index)]
    }

    #[inline]
    pub fn get_block_mut(&mut self, index: usize) -> &mut [u64] {
        &mut self.bits[Self::block_range(index)]
    }

    /// Returns the ith bit coordinate (u64 and bit index pair) from the hash.
    /// The `usize` is used to get the corresponding u64 from `self.mem`,
    /// the u64 is a mask used to get the corresponding bit from that u64.
    #[inline]
    pub(crate) const fn coordinate(bit_index: u64) -> (usize, u64) {
        let index = bit_index.wrapping_shr(BIT_MASK_LEN);
        let bit = 1u64 << (bit_index & BIT_MASK);
        (index as usize, bit)
    }

    #[inline]
    pub fn set(&mut self, block_index: usize, bit_index: u64) {
        let block_range = Self::block_range(block_index);
        let block = &mut self.bits[block_range];
        let (index, bit) = Self::coordinate(bit_index);
        block[index] |= bit;
    }

    #[inline]
    pub fn set_all_for_block(block: &mut [u64], bit_indexes: impl Iterator<Item = u64>) {
        for bit_index in bit_indexes {
            let (index, bit) = Self::coordinate(bit_index);
            block[index] |= bit;
        }
    }

    #[inline]
    pub fn check(&self, block_index: usize, bit_index: u64) -> bool {
        let block_range = Self::block_range(block_index);
        let block = &self.bits[block_range];
        let (index, bit) = Self::coordinate(bit_index);
        block[index] & bit > 0
    }

    #[inline]
    pub fn check_all_for_block(block: &[u64], mut bit_indexes: impl Iterator<Item = u64>) -> bool {
        bit_indexes.all(|bit_index| {
            let (index, bit) = Self::coordinate(bit_index);
            block[index] & bit > 0
        })
    }
}
