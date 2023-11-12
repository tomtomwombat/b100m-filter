# b100m-filter
[![Crates.io](https://img.shields.io/crates/v/b100m-filter.svg)](https://crates.io/crates/b100-filter)
[![docs.rs](https://docs.rs/bloomfilter/badge.svg)](https://docs.rs/b100m-filter)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/tomtomwombat/bloom-filter/blob/main/LICENSE-MIT)
[![License: APACHE](https://img.shields.io/badge/License-Apache-blue.svg)](https://github.com/tomtomwombat/bloom-filter/blob/main/LICENSE-Apache)
<a href="https://codecov.io/gh/tomtomwombat/bloom-filter">
    <img src="https://codecov.io/gh/tomtomwombat/bloom-filter/branch/main/graph/badge.svg">
</a>

A very fast and accurate Bloom Filter implementation in Rust.


### Usage

```toml
# Cargo.toml
[dependencies]
b100m-filter = "0.2.0"
```

```rust
use b100m_filter::BloomFilter;

let num_blocks = 4; // by default, each block is 512 bits
let values = vec!["42", "qwerty", "bloom"];

let filter = BloomFilter::builder(num_blocks).items(values.iter());
assert!(filter.contains("42"));
assert!(filter.contains("bloom"));
```

`b100m-filter` is blazingly fast because it uses L1 cache friendly blocks and efficiently derives many index bits from only one hash per value. Compared to traditional implementations, this bloom filter is 5-13 times faster for a small number of contained items, and hundreds of times faster for more items. In all cases, `b100m-filter` maintains competitive false positive rates.

### Runtime Performance

Runtime comparison to a popular traditonal bloom filter crate:
```diff
Sampled False Postive Rate: 0.000000%

BloomFilter (1000 items, 65536 bytes): get existing 1000
  time:   [72.823 µs 72.856 µs 72.893 µs]
+ change: [-92.273% -92.252% -92.210%] (p = 0.00 < 0.05)

BloomFilter (1000 items, 65536 bytes): get non-existing 1000
  time:   [14.631 µs 14.826 µs 15.051 µs]
+ change: [-80.346% -80.186% -79.991%] (p = 0.00 < 0.05)

BloomFilter (1000 items, 2097152 bytes): get existing 1000
  time:   [75.010 µs 75.439 µs 75.902 µs]
+ change: [-99.832% -99.832% -99.831%] (p = 0.00 < 0.05)

BloomFilter (1000 items, 2097152 bytes): get non-existing 1000
  time:   [14.532 µs 14.560 µs 14.591 µs]
+ change: [-99.470% -99.468% -99.467%] (p = 0.00 < 0.05)
```
As the memory size and set size increase, traditional bloom filters need to perform more hashes to keep false positive rates low. Bloom filter speed is directly proportional to number of hashes. However, `b100m-filter`'s optimal number of hashes is bounded while keeping near zero rates even for many items:
> ![bloom_perf](https://github.com/thomaspendock/bloom-filter/assets/45644087/ebe424cf-d8f1-4401-ac10-a4879123565f)


### False Positive Performance

`b100m-filter` does not sacrifice accuracy. Below is a comparison false positive rate with a traditional bloom filter:
> ![bloom_fp](https://github.com/thomaspendock/bloom-filter/assets/45644087/03687bcd-412b-434f-9cc4-c844395c0f42)

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
