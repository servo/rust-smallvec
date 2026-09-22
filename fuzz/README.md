# SmallVec Differential Fuzzer

Differential fuzzer comparing `smallvec::SmallVec` invariants and operations directly against `std::vec::Vec`.

It tests structural equivalence across multiple inline capacity configurations (`N = 0, 1, 2, 7, 8`) to stress inline-to-heap spilling boundaries, buffer alignment and iterator drop mechanics.

## Quick start

This fuzzer uses `libFuzzer` via `cargo-fuzz`.

### Prerequisites

Install `cargo-fuzz` (requires a nightly Rust toolchain):

```sh
cargo +nightly install cargo-fuzz
```

### Running the fuzzer

Run the target with standard `libFuzzer` options:

```sh
cargo +nightly fuzz run smallvec_ops
```

### Reproducing a Ccash

If the fuzzer finds an invariant mismatch or panic, reproduce it against a saved crash artifact:

```sh
cargo +nightly fuzz run smallvec_ops artifacts/smallvec_ops/crash-<hash>
```

### Generating Coverage Reports

Generating coverage requires the `llvm-tools-preview` component:

```sh
cargo +nightly rustup component add llvm-tools-preview
```

Then run coverage against the target:

```sh
cargo +nightly fuzz coverage smallvec_ops
```
