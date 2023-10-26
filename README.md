# bloom-filter
A very fast Bloom Filter implementation in Rust.


## Runtime Performance
This Bloom Filter runs 2.5-6.5 times faster than a traditional Bloom Filter implementation, and even hundreds of times faster for larger sizes.

Runtime comparison to a popular [bloomfilter](https://crates.io/crates/bloomfilter) crate:
```
Sampled False Postive Rate: 0%
BloomFilter (1000 items, 65536 bytes): get existing 1000
                        time:   [160.30 µs 161.42 µs 162.83 µs]
                        change: [-83.393% -83.315% -83.230%] (p = 0.00 < 0.05)
                        Performance has improved.

Sampled False Postive Rate: 0%
BloomFilter (1000 items, 2097152 bytes): get existing 1000
                        time:   [160.35 µs 161.73 µs 163.35 µs]
                        change: [-99.649% -99.647% -99.645%] (p = 0.00 < 0.05)
                        Performance has improved.

Sampled False Postive Rate: 0%
BloomFilter (1000 items, 65536 bytes): get non-existing 1000
                        time:   [30.359 µs 30.573 µs 30.789 µs]
                        change: [-59.666% -59.378% -59.067%] (p = 0.00 < 0.05)
                        Performance has improved.

Sampled False Postive Rate: 0%
BloomFilter (1000 items, 2097152 bytes): get non-existing 1000
                        time:   [28.621 µs 28.725 µs 28.831 µs]
                        change: [-98.997% -98.993% -98.989%] (p = 0.00 < 0.05)
                        Performance has improved.
```
## False Positive Performance

This Bloom Filter does not sacrifice false positive rate. Below we compare false positive rate with a traditional (control) bloom filter:
![Figure_1](https://github.com/thomaspendock/bloom-filter/assets/45644087/45bdd45e-1993-46d7-ad29-d0f13c2e729b)
