#!cfg(not(miri))

use {
    rkyv::{
        Archived,
        rancor::Error,
        vec::ArchivedVec
    },
    smallvec::SmallVec
};

#[test]
fn roundtrip() {
    // Empty, inline (2 <= 4) and spilled (6 > 4).
    for sv in [
        SmallVec::<u32, 4>::new(),
        SmallVec::from([1u32, 2]),
        SmallVec::from([1u32, 2, 3, 4, 5, 6])
    ] {
        let bytes = rkyv::to_bytes::<Error>(&sv).unwrap();
        let archived = unsafe { rkyv::access_unchecked::<ArchivedVec<Archived<u32>>>(&bytes) };
        assert!(*archived == sv);
        assert!(sv == *archived);
        assert!(*archived <= sv);
        assert!(*archived >= sv);
        let back = rkyv::deserialize::<SmallVec<u32, 4>, Error>(archived).unwrap();
        assert_eq!(back, sv);
    }
}

#[test]
fn roundtrip_zst() {
    for length in [0, 1, 4, 100] {
        let sv: SmallVec<(), 4> = core::iter::repeat_n((), length).collect();
        let bytes = rkyv::to_bytes::<Error>(&sv).unwrap();
        let archived = unsafe { rkyv::access_unchecked::<ArchivedVec<Archived<()>>>(&bytes) };
        assert_eq!(archived.len(), length);
        assert!(*archived == sv);
        assert!(sv == *archived);
        assert!(*archived <= sv);
        let back = rkyv::deserialize::<SmallVec<(), 4>, Error>(archived).unwrap();
        assert_eq!(back, sv);
    }
}
