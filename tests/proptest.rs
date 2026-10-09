#[cfg(all(feature = "proptest", not(feature = "allocator-api2")))]
use proptest::prelude::*;
#[cfg(all(feature = "proptest", not(feature = "allocator-api2")))]
use smallvec::SmallVec;

#[cfg(all(feature = "proptest", not(feature = "allocator-api2")))]
proptest! {
    #[test]
    fn test_proptest_arbitrary_smallvec_u32(v: SmallVec<u32, 4>) {
        let _ = v.len();
    }

    #[test]
    fn test_proptest_arbitrary_smallvec_string(v: SmallVec<String, 2>) {
        let _ = v.len();
    }

    #[test]
    fn test_proptest_arbitrary_with_params(v: SmallVec<u8, 3>) {
        let _ = v.len();
    }
}
