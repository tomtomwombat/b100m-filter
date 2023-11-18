use b100m_filter::{CloneBuildHasher, DefaultHasher};
use criterion::{black_box, criterion_group, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashSet;

use fxhash::FxBuildHasher;
use fxhash::FxHasher;

#[allow(dead_code)]
fn random_strings(num: usize, max_repeat: u32, seed: u64) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let gen = rand_regex::Regex::compile(r"[a-zA-Z]+", max_repeat).unwrap();
    (&mut rng)
        .sample_iter(&gen)
        .take(num)
        .collect::<Vec<String>>()
}

fn bench_hashset(c: &mut Criterion) {
    let sample_vals = random_strings(1000, 12, 1234);
    // let set: HashSet<String> = sample_vals.clone().into_iter().collect();
    let set: HashSet<String, DefaultHasher> = sample_vals.clone().into_iter().collect();

    c.bench_function("Bench HashSet: get existing 1000", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let val = &sample_vals[i];
                let _ = black_box(set.contains(val));
            }
        })
    });
}

criterion_group!(bench_hasher, bench_hashset,);
