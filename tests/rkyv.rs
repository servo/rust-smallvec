#![cfg(feature = "rkyv")]
use {
    rkyv::{
        Archived,
        rancor::Error,
        vec::ArchivedVec
    },
    smallvec::{
        SmallVec,
        smallvec
    }
};

#[test]
fn roundtrip() {
    for sv in [smallvec![1u32, 2], smallvec![1u32, 2, 3, 4, 5, 6]] {
        let sv: SmallVec<u32, 4> = sv;
        let bytes = rkyv::to_bytes::<Error>(&sv).unwrap();
        let archived = unsafe { rkyv::access_unchecked::<ArchivedVec<Archived<u32>>>(&bytes) };
        assert!(*archived == sv);
        let back = rkyv::deserialize::<SmallVec<u32, 4>, Error>(archived).unwrap();
        assert_eq!(back, sv);
    }
}
