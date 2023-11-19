use rand::Rng;
use siphasher::sip::SipHasher13;
use std::hash::{BuildHasher, Hasher};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CloneBuildHasher<H: Hasher + Clone> {
    hasher: H,
}

impl<H: Hasher + Clone> CloneBuildHasher<H> {
    fn new(hasher: H) -> Self {
        Self { hasher }
    }
}

impl<H: Hasher + Clone> BuildHasher for CloneBuildHasher<H> {
    type Hasher = H;
    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        self.hasher.clone()
    }
}

/// The default hasher for `BloomFilter`.
///
/// `DefaultHasher` has a faster `build_hasher` than `std::collections::hash_map::RandomState` or `SipHasher13`.
/// This is important because `build_hasher` is called once for every actual hash.
pub type DefaultHasher = CloneBuildHasher<RandomDefaultHasher>;

impl DefaultHasher {
    pub fn seeded(seed: &[u8; 16]) -> Self {
        Self {
            hasher: RandomDefaultHasher::seeded(seed),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RandomDefaultHasher(SipHasher13);

impl RandomDefaultHasher {
    #[inline]
    pub fn seeded(seed: &[u8; 16]) -> Self {
        Self(SipHasher13::new_with_key(seed))
    }
}

impl Default for RandomDefaultHasher {
    #[inline]
    fn default() -> Self {
        let mut rng = rand::thread_rng();
        let mut seed = [0u8; 16];
        for b in seed.iter_mut() {
            *b = rng.gen();
        }
        Self::seeded(&seed)
    }
}

impl Hasher for RandomDefaultHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0.finish()
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes)
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.0.write_u8(i)
    }
    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.0.write_u16(i)
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.0.write_u32(i)
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0.write_u64(i)
    }
    #[inline]
    fn write_u128(&mut self, i: u128) {
        self.0.write_u128(i)
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.0.write_usize(i)
    }
    #[inline]
    fn write_i8(&mut self, i: i8) {
        self.0.write_i8(i)
    }
    #[inline]
    fn write_i16(&mut self, i: i16) {
        self.0.write_i16(i)
    }
    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.0.write_i32(i)
    }
    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.0.write_i64(i)
    }
    #[inline]
    fn write_i128(&mut self, i: i128) {
        self.0.write_i128(i)
    }
    #[inline]
    fn write_isize(&mut self, i: isize) {
        self.0.write_isize(i)
    }
}
