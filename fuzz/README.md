# Fuzzer for smallvec

Based on fuzzing in [rust-bitcoin](https://github.com/rust-bitcoin/rust-bitcoin/tree/c8ac25219a09bf9d017f1b05abe3e746e2136f73/fuzz)

## Running manually with afl

```
cargo afl build --release --bin smallvec_ops --features afl   && cargo afl fuzz -i in -o out target/release/smallvec_ops
```

# Useful links:
* https://rust-fuzz.github.io/book/afl.html
