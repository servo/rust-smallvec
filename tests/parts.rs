#![cfg(feature = "internals")]

use {
    smallvec::SmallVec,
    std::rc::Rc
};

#[test]
fn into_from_parts_spilled() {
    let v: SmallVec<i32, 1> = SmallVec::from([-1, 0, 1]);
    assert!(v.spilled());

    let (raw, length) = v.into_parts();
    assert!(length.on_heap());
    assert_eq!(length.len(), 3);

    // SAFETY: `length` says on heap, so `heap` is the active member.
    let (ptr, cap) = unsafe { raw.heap };
    assert!(cap > 1);

    // SAFETY: parts come straight from `into_parts`.
    let rebuilt = unsafe { SmallVec::<i32, 1>::from_parts(raw, length) };
    assert_eq!(rebuilt, [-1, 0, 1]);
    assert_eq!(rebuilt.as_ptr(), ptr.as_ptr() as *const i32);
}

#[test]
fn into_from_parts_inline() {
    let v: SmallVec<i32, 4> = SmallVec::from([1, 2, 3]);
    assert!(!v.spilled());

    let (raw, length) = v.into_parts();
    assert!(!length.on_heap());
    assert_eq!(length.len(), 3);

    // SAFETY: parts come straight from `into_parts`.
    let rebuilt = unsafe { SmallVec::<i32, 4>::from_parts(raw, length) };
    assert_eq!(rebuilt, [1, 2, 3]);
}

#[test]
fn parts_do_not_drop_or_double_drop() {
    let rc = Rc::new(());

    for spilled in [false, true] {
        // INLINE = 2: two items stay inline, three spill
        let count = if spilled { 3 } else { 2 };
        let mut v: SmallVec<Rc<()>, 2> = SmallVec::new();
        for _ in 0..count {
            v.push(rc.clone());
        }
        assert_eq!(v.spilled(), spilled);
        assert_eq!(Rc::strong_count(&rc), 1 + count);

        let (raw, length) = v.into_parts();
        // nothing was dropped by `into_parts`
        assert_eq!(Rc::strong_count(&rc), 1 + count);

        // SAFETY: parts come straight from `into_parts`.
        let rebuilt = unsafe { SmallVec::<Rc<()>, 2>::from_parts(raw, length) };
        assert_eq!(Rc::strong_count(&rc), 1 + count);

        drop(rebuilt);
        // dropped exactly once
        assert_eq!(Rc::strong_count(&rc), 1);
    }
}

#[test]
fn into_from_parts_empty() {
    let v: SmallVec<i32, 2> = SmallVec::new();

    let (raw, length) = v.into_parts();
    assert_eq!(length.len(), 0);
    assert!(!length.on_heap());

    // SAFETY: parts come straight from `into_parts`.
    let rebuilt = unsafe { SmallVec::<i32, 2>::from_parts(raw, length) };
    assert!(rebuilt.is_empty());
}

#[test]
fn into_from_parts_zst() {
    let v: SmallVec<(), 2> = SmallVec::from([(), (), ()]);

    let (raw, length) = v.into_parts();
    assert!(!length.on_heap());
    assert_eq!(length.len(), 3);

    // SAFETY: parts come straight from `into_parts`.
    let rebuilt = unsafe { SmallVec::<(), 2>::from_parts(raw, length) };
    assert_eq!(rebuilt.len(), 3);
}
