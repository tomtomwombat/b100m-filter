use bloom_filter::test_util::*;
use bloom_filter::BloomFilter;
use bloomfilter;
use criterion::{black_box, criterion_group, Criterion}; // comparison

fn bench_some_get(c: &mut Criterion) {
    let size = 1000;
    let sample_vals = random_strings(size, 12, *b"seedseedseedseed");
    let bloom_size_bytes = 1 << 16;
    let block_size = bloom_size_bytes / 64;
    let mut bloom1: bloomfilter::Bloom<String> =
        bloomfilter::Bloom::new(bloom_size_bytes, sample_vals.len());
    let bloom = BloomFilter::from_vec(block_size, &sample_vals);

    println!("{:?}", bloom.num_hashes);
    println!("{:?}", bloom1.number_of_hash_functions());
    for i in 0..size {
        bloom1.set(&sample_vals[i]);
    }

    c.bench_function("BloomFilter: get existing 1000", |b| {
        b.iter(|| {
            for i in 0..size {
                let val = &sample_vals[i];
                let _ = black_box(bloom.contains(val));
                // let _ = black_box(bloom1.check(val));
            }
        })
    });
}

fn bench_none_get(c: &mut Criterion) {
    let size = 1000;
    let sample_vals = random_strings(size, 12, *b"seedseedseedseed");
    let bloom_size_bytes = 1 << 16;
    let block_size = bloom_size_bytes / 64;
    let mut bloom1: bloomfilter::Bloom<String> =
        bloomfilter::Bloom::new(bloom_size_bytes, sample_vals.len());
    let bloom = BloomFilter::from_vec(block_size, &sample_vals);
    for i in 0..size {
        bloom1.set(&sample_vals[i]);
    }
    let sample_anti_vals = random_strings(1000, 16, *b"antiantiantianti");
    c.bench_function("BloomFilter: get non-existing 1000", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let val = &sample_anti_vals[i];
                let _ = black_box(bloom.contains(val));
                // let _ = black_box(bloom1.check(val));
            }
        })
    });
}

criterion_group!(bench_bloom_filter, bench_some_get, bench_none_get,);
