use bloom_filter::BloomFilter;
use bloomfilter;
use criterion::{black_box, criterion_group, Criterion};
use rand::{Rng, SeedableRng};
use std::collections::HashSet;

#[allow(dead_code)]
fn random_strings(num: usize, max_repeat: u32, seed: [u8; 16]) -> Vec<String> {
    let mut rng = rand_xorshift::XorShiftRng::from_seed(seed);
    let gen = rand_regex::Regex::compile(r"[a-zA-Z]+", max_repeat).unwrap();
    (&mut rng)
        .sample_iter(&gen)
        .take(num)
        .collect::<Vec<String>>()
}

fn bench_some_get(c: &mut Criterion) {
    for (num_items, bloom_size_bytes) in [(1000, 1 << 16), (1000, 2097152)] {
        let sample_vals = random_strings(num_items, 12, *b"seedseedseedseed");
        let block_size = bloom_size_bytes / 64;
        let mut bloom1: bloomfilter::Bloom<String> =
            bloomfilter::Bloom::new(bloom_size_bytes, sample_vals.len());
        let bloom = BloomFilter::from_vec(block_size, &sample_vals);
        let mut control: HashSet<String> = HashSet::new();

        println!("{:?}", bloom.num_hashes());
        println!("{:?}", bloom1.number_of_hash_functions());
        for i in 0..num_items {
            bloom1.set(&sample_vals[i]);
            control.insert(sample_vals[i].clone());
        }
        let sample_anti_vals = random_strings(1000, 16, *b"antiantiantianti");
        let mut total = 0;
        let mut false_positives = 0;
        let mut false_positives_control = 0;
        for x in &sample_anti_vals {
            if !control.contains(x) {
                total += 1;
                if bloom.contains(x) {
                    false_positives += 1;
                }
                if bloom1.check(x) {
                    false_positives_control += 1;
                }
            }
        }
        println!(
            "Sampled False Postive Rate: {:.6}% (vs {:.6}%)",
            100 * false_positives,
            100 * false_positives_control
        );
        c.bench_function(
            &format!(
                "BloomFilter ({:?} items, {:?} bytes): get existing 1000",
                num_items, bloom_size_bytes
            ),
            |b| {
                b.iter(|| {
                    for i in 0..1000 {
                        let val = &sample_vals[i];
                        let _ = black_box(bloom.contains(val));
                        // let _ = black_box(bloom1.check(val));
                    }
                })
            },
        );
    }
}

fn bench_none_get(c: &mut Criterion) {
    for (num_items, bloom_size_bytes) in [(1000, 1 << 16), (1000, 2097152)] {
        let sample_vals = random_strings(num_items, 12, *b"seedseedseedseed");
        let block_size = bloom_size_bytes / 64;
        let mut bloom1: bloomfilter::Bloom<String> =
            bloomfilter::Bloom::new(bloom_size_bytes, sample_vals.len());
        let bloom = BloomFilter::from_vec(block_size, &sample_vals);
        let mut control: HashSet<String> = HashSet::new();

        for i in 0..num_items {
            bloom1.set(&sample_vals[i]);
            control.insert(sample_vals[i].clone());
        }

        let sample_anti_vals = random_strings(1000, 16, *b"antiantiantianti");
        let mut total = 0;
        let mut false_positives = 0;
        let mut false_positives_control = 0;
        for x in &sample_anti_vals {
            if !control.contains(x) {
                total += 1;
                if bloom.contains(x) {
                    false_positives += 1;
                }
                if bloom1.check(x) {
                    false_positives_control += 1;
                }
            }
        }
        println!(
            "Sampled False Postive Rate: {:.6}% (vs {:.6}%)",
            100 * false_positives,
            100 * false_positives_control
        );
        c.bench_function(
            &format!(
                "BloomFilter ({:?} items, {:?} bytes): get non-existing 1000",
                num_items, bloom_size_bytes
            ),
            |b| {
                b.iter(|| {
                    for i in 0..1000 {
                        let val = &sample_anti_vals[i];
                        let _ = black_box(bloom.contains(val));
                        // let _ = black_box(bloom1.check(val));
                    }
                })
            },
        );
    }
}

criterion_group!(bench_bloom_filter, bench_some_get, bench_none_get,);
