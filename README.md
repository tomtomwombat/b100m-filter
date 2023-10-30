# b100m-filter
A very fast and accurate Bloom Filter implementation in Rust.


### Usage

```toml
# Cargo.toml
[dependencies]
b100m_filter = "0.1.0"
```

```rust
use b100m_filter::BloomFilter;

let num_blocks = 4; // each block is 64 bytes, 512 bits
let values = vec!["42", "qwerty", "bloom"];

let filter = BloomFilter::builder(num_blocks).items(values.iter());
assert!(filter.contains("42"));
assert!(filter.contains("bloom"));
```

`b100m-filter` is blazingly fast because it uses L1 cache friendly blocks and efficiently derives many index bits from only two hashes per value. Compared to traditional implementations, this bloom filter is 2.5-6.5 faster for a small number of contained items, and hundreds of times faster for more items. In all cases `b100m_filter` maintains competitive false positive rates.

### Runtime Performance

Runtime comparison to a popular traditonal bloom filter crate:
```diff
Sampled False Postive Rate: 0.000000%
BloomFilter (1000 items, 65536 bytes): get existing 1000
                        time:   [140.84 µs 141.30 µs 141.85 µs]
+                       change: [-85.090% -85.025% -84.936%] (p = 0.00 < 0.05)

Sampled False Postive Rate: 0.000000%
BloomFilter (1000 items, 2097152 bytes): get existing 1000
                        time:   [141.30 µs 141.46 µs 141.65 µs]
+                       change: [-99.686% -99.686% -99.686%] (p = 0.00 < 0.05)

Sampled False Postive Rate: 0.000000%
BloomFilter (1000 items, 65536 bytes): get non-existing 1000
                        time:   [27.722 µs 27.754 µs 27.787 µs]
+                       change: [-62.268% -62.027% -61.757%] (p = 0.00 < 0.05)

Sampled False Postive Rate: 0.000000%
BloomFilter (1000 items, 2097152 bytes): get non-existing 1000
                        time:   [27.209 µs 27.223 µs 27.241 µs]
+                       change: [-99.002% -99.000% -98.996%] (p = 0.00 < 0.05)

```
As the memory size and set size increase, bloom filters need to perform more hashes to keep false positive rates low. Bloom filter speed is directly proportional to number of hashes. `b100m-filter`'s optimal number of hashes is bounded and keeps near zero rates even for large sizes:
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
