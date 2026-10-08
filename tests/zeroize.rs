use {
    smallvec::SmallVec,
    zeroize::Zeroize
};

fn assert_fully_zeroed<const N: usize>(v: &mut SmallVec<u8, N>) {
    assert!(v.is_empty());
    assert!(
        v.spare_capacity_mut()
            .iter()
            .all(|slot| unsafe { slot.assume_init() } == 0)
    );
}

#[test]
fn zeroize_inline() {
    let mut v: SmallVec<u8, 8> = SmallVec::from([1, 2, 3, 4]);
    v.push(5);
    assert!(!v.spilled());

    v.zeroize();

    assert_eq!(v.capacity(), 8);
    assert_fully_zeroed(&mut v);

    // It is still usable.
    v.extend_from_slice(&[1, 2, 3]);
    assert_eq!(v.as_slice(), &[1, 2, 3]);
}

#[test]
fn zeroize_spilled() {
    let mut v: SmallVec<u8, 8> = SmallVec::from([1, 2, 3, 4, 5, 6, 7, 8]);
    v.push(9);
    assert!(v.spilled());
    let capacity = v.capacity();

    v.zeroize();

    assert!(v.spilled());
    assert_eq!(v.capacity(), capacity);
    assert_fully_zeroed(&mut v);

    v.extend_from_slice(&[10, 11]);
    assert_eq!(v.as_slice(), &[10, 11]);
}

#[test]
fn zeroize_empty() {
    let mut v: SmallVec<u8, 4> = SmallVec::new();

    v.zeroize();

    assert_eq!(v.capacity(), 4);
    assert_fully_zeroed(&mut v);
}
