use {
    borsh::{
        BorshDeserialize,
        to_vec
    },
    smallvec::SmallVec
};

#[test]
fn round_trip() {
    let smallvec = SmallVec::<u8, 6>::from([1, 2, 3]);
    let bytes = to_vec(&smallvec).unwrap();
    let new = SmallVec::<u8, 6>::deserialize(&mut bytes.as_ref()).unwrap();
    assert_eq!(new, smallvec);
}

#[test]
fn round_trip_zst() {
    let smallvec = SmallVec::<(), 5>::from([(); 0x100000]);
    let bytes = to_vec(&smallvec).unwrap();
    let new = SmallVec::<(), 100>::deserialize(&mut bytes.as_ref()).unwrap();
    assert_eq!(new, smallvec);
}
