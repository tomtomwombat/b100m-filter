# bloom-filter
A very fast Bloom Filter implementation in Rust.

Runtime comparison to the most popular [bloomfilter](https://crates.io/crates/bloomfilter) crate:
```
BloomFilter: get existing 1000
  time:   [148.59 µs 148.84 µs 149.17 µs]
  change: [-84.667% -84.629% -84.585%] (p = 0.00 < 0.05)

BloomFilter: get non-existing 1000
  time:   [27.746 µs 27.767 µs 27.800 µs]
  change: [-63.278% -63.072% -62.902%] (p = 0.00 < 0.05)       
```
