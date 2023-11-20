# b100m-filter
[![Crates.io](https://img.shields.io/crates/v/b100m-filter.svg)](https://crates.io/crates/b100m-filter)
[![docs.rs](https://docs.rs/bloomfilter/badge.svg)](https://docs.rs/b100m-filter)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/tomtomwombat/bloom-filter/blob/main/LICENSE-MIT)
[![License: APACHE](https://img.shields.io/badge/License-Apache-blue.svg)](https://github.com/tomtomwombat/bloom-filter/blob/main/LICENSE-Apache)
<a href="https://codecov.io/gh/tomtomwombat/b100m-filter">
    <img src="https://codecov.io/gh/tomtomwombat/b100m-filter/branch/main/graph/badge.svg">
</a>

The fastest bloom filter in Rust. No accuracy compromises. Use any hasher.


### Usage

```toml
# Cargo.toml
[dependencies]
b100m-filter = "0.3.0"
```

```rust
use b100m_filter::BloomFilter;

let num_blocks = 4; // by default, each block is 512 bits
let values = vec!["42", "🦀"];

let filter = BloomFilter::builder(num_blocks).items(values.iter());
assert!(filter.contains("42"));
assert!(filter.contains("🦀"));
```

```rust
use b100m_filter::BloomFilter;
use ahash::RandomState;

let num_blocks = 4; // by default, each block is 512 bits

let filter = BloomFilter::builder(num_blocks)
    .hasher(RandomState::default())
    .items(["42", "🦀"].iter());
```

### Background
Bloom filters are a space efficient approximate membership set data structure. False positives from `contains` are possible, but false negatives are not, i.e. `contains` for all items in the set is guaranteed to return true, while `contains` for all items not in the set probably return false.

Blocked bloom filters are supported by an underlying bit vector, chunked into 512, 256, 128, or 64 bit "blocks", to track item membership. To insert, a number of bits, based on the item's hash, are set in the underlying bit vector. To check membership, a number of bits, based on the item's hash, are checked in the underlying bit vector.

Once constructed, neither the bloom filter's underlying memory usage nor number of bits per item change.


### Implementation

`b100m-filter` is blazingly fast because it uses L1 cache friendly blocks and efficiently derives many index bits from only one hash per value. Compared to traditional implementations, `b100m-filter` is 2-5 times faster for small sets of items, and hundreds of times faster for larger item sets. In all cases, `b100m-filter` maintains competitive false positive rates.

### Runtime Performance

Runtime comparison to other bloom filter crates:
- Bloom memory size = 16Kb
- 1000 contained items
- 364 hashes per item
  
|  | Check Non-Existing (ns) | Check Existing (ns) |
| --- | --- | --- |
| b100m-filter | 16.900 | 139.62 |
| *fastbloom-rs | 35.358 | 485.81 |
| bloom | 66.136 | 749.27 |
| bloomfilter | 68.912 | 996.56 |
| probabilistic-collections | 83.385 | 974.67 |

*fastbloom-rs uses XXHash, which is faster than SipHash.

### False Positive Performance

`b100m-filter` does not compromise accuracy. Below is a comparison false positive rate with other bloom filter crates:
> ![bloom-crate-fp](https://github.com/tomtomwombat/b100m-filter/assets/45644087/1043c30b-3de8-44ec-b868-88625c7aad09)



### Scaling

`b100m-filter` scales very well.

As the number of bits and set size increase, traditional bloom filters need to perform more hashes per item to keep false positive rates low. However, `b100m-filter`'s optimal number of hashes is bounded while keeping near zero rates even for many items:
> ![bloom_perf](https://github.com/thomaspendock/bloom-filter/assets/45644087/ebe424cf-d8f1-4401-ac10-a4879123565f)
>
> Bloom filter speed is directly proportional to number of hashes.

## References
- [Bloom Filter](https://brilliant.org/wiki/bloom-filter/)
- [Less hashing, same performance: Building a better Bloom filter](https://dl.acm.org/doi/10.5555/1400123.1400125)
- [A fast alternative to the modulo reduction](https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/)

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
