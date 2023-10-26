use rand::{Rng, SeedableRng};
use std::iter::repeat;

#[allow(dead_code)]
pub(crate) fn random_strings(
    num: usize,
    min_repeat: u32,
    max_repeat: u32,
    seed: [u8; 16],
) -> Vec<String> {
    let mut rng = rand_xorshift::XorShiftRng::from_seed(seed);
    let gen = rand_regex::Regex::compile(r"[a-zA-Z]+", max_repeat).unwrap();
    (&mut rng)
        .sample_iter(&gen)
        .filter(|s: &String| s.len() >= min_repeat as usize)
        .take(num)
        .collect::<Vec<String>>()
}

#[allow(dead_code)]
pub(crate) fn random_u32s(num: usize) -> Vec<u32> {
    let mut rng = rand_xorshift::XorShiftRng::from_seed(*b"seedseedseedseed");
    repeat(())
        .map(|_| (&mut rng).gen_range(0..1000))
        .take(num)
        .collect::<Vec<u32>>()
}
