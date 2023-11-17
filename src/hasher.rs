use getrandom::getrandom;
use siphasher::sip::SipHasher13;
use std::hash::BuildHasher;

/// The default hasher for `BloomFilter`.
///
/// `DefaultHasher` has a faster `build_hasher` than `std::collections::hash_map::RandomState` or `SipHasher13`.
/// This is important because `build_hasher` is called once for every actual hash.
#[derive(Debug, Clone)]
pub struct DefaultHasher {
    hasher: SipHasher13,
}

impl DefaultHasher {
    pub fn seeded(seed: &[u8; 16]) -> Self {
        Self {
            hasher: SipHasher13::new_with_key(seed),
        }
    }
}

impl Default for DefaultHasher {
    fn default() -> Self {
        let mut seed = [0u8; 16];
        getrandom(&mut seed).unwrap();
        Self::seeded(&seed)
    }
}

impl BuildHasher for DefaultHasher {
    type Hasher = SipHasher13;
    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        self.hasher.clone()
    }
}
