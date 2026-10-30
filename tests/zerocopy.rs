use {
    smallvec::SmallVec,
    zerocopy::FromZeros
};

#[test]
fn zeroed_is_empty_and_inline() {
    let mut v: SmallVec<String, 4> = FromZeros::new_zeroed();
    assert!(v.is_empty());
    assert!(!v.spilled());
    assert_eq!(v.capacity(), SmallVec::<String, 4>::inline_size());
    for i in 0..10 {
        v.push(i.to_string());
    }
    assert_eq!(v.len(), 10);
}

#[test]
fn zeroed_primitive() {
    let v: SmallVec<u8, 8> = FromZeros::new_zeroed();
    assert!(v.is_empty());
    assert!(!v.spilled());
}

#[test]
fn zeroed_zst() {
    let v: SmallVec<(), 0> = FromZeros::new_zeroed();
    assert!(v.is_empty());
    assert!(!v.spilled());
}

#[test]
fn zero_method() {
    let mut v: SmallVec<u8, 8> = SmallVec::new();
    v.push(42);
    v.zero();
    assert!(v.is_empty());
    assert!(!v.spilled());

    // Verify all bytes are 0
    let bytes = unsafe {
        core::slice::from_raw_parts(&v as *const _ as *const u8, core::mem::size_of_val(&v))
    };
    assert!(bytes.iter().all(|&b| b == 0));
}

#[test]
fn zero_method_zst() {
    let mut v: SmallVec<(), 4> = SmallVec::new();
    for _ in 0..10 {
        v.push(());
    }
    assert_eq!(v.len(), 10);

    v.zero();
    assert!(v.is_empty());
    assert_eq!(v.len(), 0);
}
