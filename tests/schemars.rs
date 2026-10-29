#![cfg(feature = "schemars")]
use {
    schemars::schema_for,
    smallvec::SmallVec
};

#[test]
fn schema_matches_vec() {
    assert_eq!(schema_for!(SmallVec<u32, 4>), schema_for!(Vec<u32>));
}
