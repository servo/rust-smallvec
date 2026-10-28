#![cfg(feature = "bytemuck")]
use {
    bytemuck::Zeroable,
    smallvec::SmallVec
};

#[test]
fn zeroed_is_empty_and_inline() {
    let mut v: SmallVec<String, 4> = Zeroable::zeroed(); // String isn't Zeroable; that's fine
    assert!(v.is_empty());
    assert!(!v.spilled());
    assert_eq!(v.capacity(), SmallVec::<String, 4>::inline_size());
    for i in 0..10 {
        v.push(i.to_string()); // also exercises spilling
    }
    assert_eq!(v.len(), 10);
}
