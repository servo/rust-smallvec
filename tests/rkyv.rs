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
    for sv in [
        SmallVec::<u32, 4>::new(),
        smallvec![1u32, 2],
        smallvec![1u32, 2, 3, 4, 5, 6]
    ] {
        let bytes = rkyv::to_bytes::<Error>(&sv).unwrap();
        let archived = unsafe { rkyv::access_unchecked::<ArchivedVec<Archived<u32>>>(&bytes) };
        assert!(*archived == sv);
        assert!(sv == *archived);
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
        let back = rkyv::deserialize::<SmallVec<(), 4>, Error>(archived).unwrap();
        assert_eq!(back, sv);
    }
}


