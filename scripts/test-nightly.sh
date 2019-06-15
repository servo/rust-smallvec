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

# for feature in "${STABLE_FEATURES[@]}"; do
#     test_with_feature "const_generic, $feature"
# done

# for feature in "${NIGHTLY_FEATURES[@]}"; do
#     test_with_feature "const_generic, $feature"
# done

# All features

# /bin/echo -e "\e[0;33m***** Testing all features *****\e[0m\n"
# cargo test --all-features --no-default-features --verbose

# Run bench

/bin/echo -e "\e[0;33m***** Running bench *****\e[0m\n"
cargo bench --verbose bench
