#!/usr/bin/env bash

set -e
. ./test-common.sh --source-only

# All nightly features individually

NIGHTLY_FEATURES=(
    may_dangle
    specialization
    union
)

for feature in "${NIGHTLY_FEATURES[@]}"; do
    test_with_feature $feature
done

# All nightly features individually with constant generics

# FIXME: Remove CARGO_INCREMENTAL. See https://github.com/rust-lang/rust/issues/61338
# FIXME: Remove --lib once const_generics is default. Doctests are not using constant generics

test_lib_with_feature() {
    local feature=$1
    /bin/echo -e "\e[0;33m***** Testing with feature '${feature}' *****\e[0m\n"
    CARGO_INCREMENTAL=0 cargo test --features "${feature}" --lib --no-default-features --verbose
}

for feature in "${STABLE_FEATURES[@]}"; do
    test_lib_with_feature "const_generics, $feature"
done

for feature in "${NIGHTLY_FEATURES[@]}"; do
    test_lib_with_feature "const_generics, $feature"
done

# All features

/bin/echo -e "\e[0;33m***** Testing all features *****\e[0m\n"
CARGO_INCREMENTAL=0 cargo test --all-features --lib --no-default-features --verbose

# Run bench

/bin/echo -e "\e[0;33m***** Running bench *****\e[0m\n"
cargo bench --verbose bench
