use crate::BlockedBitVector;
use crate::BloomFilter;
use crate::BuildHasher;
use crate::DefaultHasher;
use std::hash::Hash;

/// A bloom filter builder.
///
/// This type can be used to construct an instance of `BloomFilter`
/// via the builder pattern.
#[derive(Debug, Clone)]
pub struct Builder<const BLOCK_SIZE_BITS: usize = 512, S = DefaultHasher> {
    pub(crate) num_blocks: usize,
    pub(crate) hasher: S,
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
        self.hasher = DefaultHasher::seeded(seed);
        self
    }
}

impl<const BLOCK_SIZE_BITS: usize, S: BuildHasher> Builder<BLOCK_SIZE_BITS, S> {
    /// Sets the hasher for this builder. The later constructed `BloomFilter` will use
    /// this hasher when inserting and checking items.
    ///
    /// # Examples
    ///
    /// ```
    /// use b100m_filter::BloomFilter;
    /// use ahash::RandomState;
    ///
    /// let bloom = BloomFilter::builder(4).hasher(RandomState::default()).hashes(4);
    /// ```
    pub fn hasher<H: BuildHasher>(self, hasher: H) -> Builder<BLOCK_SIZE_BITS, H> {
        Builder::<BLOCK_SIZE_BITS, H> {
            num_blocks: self.num_blocks,
            hasher,
        }
    }

    /// "Consumes" this builder, using the provided `num_hashes` to return an
    /// empty `BloomFilter`. For performance, the actual number of
    /// hashes performed internally will be rounded to down to a power of 2,
    /// depending on `BLOCK_SIZE_BITS`.
    ///
    /// # Examples
    ///
    /// ```
    /// use b100m_filter::BloomFilter;
    ///
    /// let bloom = BloomFilter::builder(4).hashes(4);
    /// ```
    pub fn hashes(self, num_hashes: u64) -> BloomFilter<BLOCK_SIZE_BITS, S> {
        BloomFilter {
            bits: BlockedBitVector::<BLOCK_SIZE_BITS>::new(self.num_blocks),
            num_hashes,
            hasher: self.hasher,
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
    pub fn expected_items(self, expected_num_items: usize) -> BloomFilter<BLOCK_SIZE_BITS, S> {
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
    pub fn items<I: ExactSizeIterator<Item = impl Hash>>(
        self,
        items: I,
    ) -> BloomFilter<BLOCK_SIZE_BITS, S> {
        let mut filter = self.expected_items(items.len());
        filter.extend(items);
        filter
    }
}

#[cfg(test)]
mod tests {
    use crate::BloomFilter;
    use ahash::RandomState;

    #[test]
    fn api() {
        let _bloom = BloomFilter::builder128(10)
            .hasher(RandomState::default())
            .hashes(4);
    }
}
