STABLE_FEATURES=(
    serde
    std
)

test_with_feature() {
    local feature=$1
    /bin/echo -e "\e[0;33m***** Testing with feature '${feature}' *****\e[0m\n"
    CARGO_INCREMENTAL=0 cargo test --features "${feature}" --no-default-features --verbose
}