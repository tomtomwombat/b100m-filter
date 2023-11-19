use b100m_filter::BloomFilter;
use criterion::{black_box, criterion_group, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;

use ahash;
use fxhash::FxBuildHasher;
use fxhash::FxHasher;

use fastbloom_rs::{self, Membership};

fn random_strings(num: usize, max_repeat: u32, seed: u64) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let gen = rand_regex::Regex::compile(r"[a-zA-Z]+", max_repeat).unwrap();
    (&mut rng).sample_iter(&gen).take(num).collect()
}

fn bench_get(c: &mut Criterion) {
    for (num_items, bloom_size_bytes) in [(1000, 1 << 16), (1000, 2097152)] {
        let sample_vals = random_strings(num_items, 12, 1234);
        let num_blocks = bloom_size_bytes / 64;

        let bloom = BloomFilter::builder512(num_blocks)
            .hasher(FxBuildHasher::default())
            .items(sample_vals.iter());

        let mut fast_bloom = fastbloom_rs::FilterBuilder::from_size_and_hashes(
            bloom_size_bytes as u64 * 8,
            bloom.num_hashes() as u32,
        )
        .build_bloom_filter();
        let mut fast_bloom_fp = 0;

        for v in sample_vals.iter() {
            fast_bloom.add(v.as_bytes());
        }

        let control: HashSet<String> = sample_vals.clone().into_iter().collect();

        let sample_anti_vals = random_strings(100000, 16, 9876);
        let mut total = 0;
        let mut false_positives = 0;
        for x in &sample_anti_vals {
            if !control.contains(x) {
                total += 1;
                if bloom.contains(x) {
                    false_positives += 1;
                }

                if fast_bloom.contains(x.as_bytes()) {
                    fast_bloom_fp += 1;
                }
            }
        }
        println!("Other FP: {:?}", (fast_bloom_fp as f64) / (total as f64));
        let fp = (false_positives as f64) / (total as f64);
        println!("Number of hashes: {:?}", bloom.num_hashes());
        println!("Sampled False Postive Rate: {:.6}%", 100.0 * fp);
        c.bench_function(
            &format!(
                "BloomFilter ({:?} items, {:?} bytes): get existing 1000",
                num_items, bloom_size_bytes
            ),
            |b| {
                b.iter(|| {
                    for i in 0..1000 {
                        let val = &sample_vals[i];
                        // let _ = black_box(bloom.contains(val));
                        let _ = black_box(fast_bloom.contains(val.as_bytes()));
                    }
                })
            },
        );
        println!("Sampled False Postive Rate: {:.6}%", 100.0 * fp);
        c.bench_function(
            &format!(
                "BloomFilter ({:?} items, {:?} bytes): get non-existing 1000",
                num_items, bloom_size_bytes
            ),
            |b| {
                b.iter(|| {
                    for i in 0..1000 {
                        let val = &sample_anti_vals[i];
                        // let _ = black_box(bloom.contains(val));
                        let _ = black_box(fast_bloom.contains(val.as_bytes()));
                    }
                })
            },
        );
    }
}

criterion_group!(bench_bloom_filter, bench_get,);
