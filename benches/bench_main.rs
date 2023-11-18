use criterion::criterion_main;

mod bench_bloom_filter;
use crate::bench_bloom_filter::bench_bloom_filter;
mod bench_hasher;
use crate::bench_hasher::bench_hasher;

criterion_main!(
    bench_bloom_filter,
    // bench_hasher,
);
