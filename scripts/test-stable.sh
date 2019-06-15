#!/usr/bin/env bash

set -e
. ./test-common.sh --source-only

# No features

/bin/echo -e "\e[0;33m***** Testing without features *****\e[0m\n"
cargo test --no-default-features --verbose

# All stable features individually

for feature in "${STABLE_FEATURES[@]}"; do
    test_with_feature $feature
done

# All stable features at once

test_with_feature $(IFS=, ; echo "${STABLE_FEATURES[*]}")
