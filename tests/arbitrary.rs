use {
    arbitrary::{
        Arbitrary,
        Unstructured
    },
    smallvec::SmallVec
};

#[test]
fn arbitrary() {
    // Deterministic for fixed input bytes; assert it builds a consistent
    // SmallVec.
    let mut u = Unstructured::new(&[0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    let v = SmallVec::<u8, 4>::arbitrary(&mut u).unwrap();
    assert_eq!(v.len(), v.iter().count());
}
