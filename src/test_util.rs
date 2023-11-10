use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::iter::repeat;

#[allow(dead_code)]
pub(crate) fn random_strings(
    num: usize,
    min_repeat: u32,
    max_repeat: u32,
    seed: u64,
) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let gen = rand_regex::Regex::compile(r"[a-zA-Z]+", max_repeat).unwrap();
    (&mut rng)
        .sample_iter(&gen)
        .filter(|s: &String| s.len() >= min_repeat as usize)
        .take(num)
        .collect::<Vec<String>>()
}
