extern crate alloc;

use crate::SmallVec;
use alloc::{boxed::Box, rc::Rc, vec, vec::Vec};
use core::iter::FromIterator;

#[test]
pub fn test_zero() {
    let mut v = SmallVec::<[_; 0]>::new();
    assert!(!v.spilled());
    v.push(0usize);
    assert!(v.spilled());
    assert_eq!(&*v, &[0]);
}

// We heap allocate all these strings so that double frees will show up under valgrind.

#[test]
pub fn test_inline() {
    let mut v = SmallVec::<[_; 16]>::new();
    v.push("hello");
    v.push("there");
    assert_eq!(&*v, &["hello", "there",][..]);
}

#[test]
pub fn test_spill() {
    let mut v = SmallVec::<[_; 2]>::new();
    v.push("hello");
    assert_eq!(v[0], "hello");
    v.push("there");
    v.push("burma");
    assert_eq!(v[0], "hello");
    v.push("shave");
    assert_eq!(&*v, &["hello", "there", "burma", "shave",][..]);
}

#[test]
pub fn test_double_spill() {
    let mut v = SmallVec::<[_; 2]>::new();
    v.push("hello");
    v.push("there");
    v.push("burma");
    v.push("shave");
    v.push("hello");
    v.push("there");
    v.push("burma");
    v.push("shave");
    assert_eq!(
        &*v,
        &["hello", "there", "burma", "shave", "hello", "there", "burma", "shave",][..]
    );
}

/// https://github.com/servo/rust-smallvec/issues/4
#[test]
fn issue_4() {
    SmallVec::<[Box<u32>; 2]>::new();
}

/// https://github.com/servo/rust-smallvec/issues/5
#[test]
fn issue_5() {
    assert!(Some(SmallVec::<[&u32; 2]>::new()).is_some());
}

#[test]
fn test_with_capacity() {
    let v: SmallVec<[u8; 3]> = SmallVec::with_capacity(1);
    assert!(v.is_empty());
    assert!(!v.spilled());
    assert_eq!(v.capacity(), 3);

    let v: SmallVec<[u8; 3]> = SmallVec::with_capacity(10);
    assert!(v.is_empty());
    assert!(v.spilled());
    assert_eq!(v.capacity(), 10);
}

#[test]
fn drain() {
    let mut v: SmallVec<[u8; 2]> = SmallVec::new();
    v.push(3);
    assert_eq!(v.drain().collect::<Vec<_>>(), &[3]);

    // spilling the vec
    v.push(3);
    v.push(4);
    v.push(5);
    assert_eq!(v.drain().collect::<Vec<_>>(), &[3, 4, 5]);
}

#[test]
fn drain_rev() {
    let mut v: SmallVec<[u8; 2]> = SmallVec::new();
    v.push(3);
    assert_eq!(v.drain().rev().collect::<Vec<_>>(), &[3]);

    // spilling the vec
    v.push(3);
    v.push(4);
    v.push(5);
    assert_eq!(v.drain().rev().collect::<Vec<_>>(), &[5, 4, 3]);
}

#[test]
fn into_iter() {
    let mut v: SmallVec<[u8; 2]> = SmallVec::new();
    v.push(3);
    assert_eq!(v.into_iter().collect::<Vec<_>>(), &[3]);

    // spilling the vec
    let mut v: SmallVec<[u8; 2]> = SmallVec::new();
    v.push(3);
    v.push(4);
    v.push(5);
    assert_eq!(v.into_iter().collect::<Vec<_>>(), &[3, 4, 5]);
}

#[test]
fn into_iter_rev() {
    let mut v: SmallVec<[u8; 2]> = SmallVec::new();
    v.push(3);
    assert_eq!(v.into_iter().rev().collect::<Vec<_>>(), &[3]);

    // spilling the vec
    let mut v: SmallVec<[u8; 2]> = SmallVec::new();
    v.push(3);
    v.push(4);
    v.push(5);
    assert_eq!(v.into_iter().rev().collect::<Vec<_>>(), &[5, 4, 3]);
}

#[test]
fn into_iter_drop() {
    use core::cell::Cell;

    struct DropCounter<'a>(&'a Cell<i32>);

    impl<'a> Drop for DropCounter<'a> {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    {
        let cell = Cell::new(0);
        let mut v: SmallVec<[DropCounter; 2]> = SmallVec::new();
        v.push(DropCounter(&cell));
        v.into_iter();
        assert_eq!(cell.get(), 1);
    }

    {
        let cell = Cell::new(0);
        let mut v: SmallVec<[DropCounter; 2]> = SmallVec::new();
        v.push(DropCounter(&cell));
        v.push(DropCounter(&cell));
        assert!(v.into_iter().next().is_some());
        assert_eq!(cell.get(), 2);
    }

    {
        let cell = Cell::new(0);
        let mut v: SmallVec<[DropCounter; 2]> = SmallVec::new();
        v.push(DropCounter(&cell));
        v.push(DropCounter(&cell));
        v.push(DropCounter(&cell));
        assert!(v.into_iter().next().is_some());
        assert_eq!(cell.get(), 3);
    }
    {
        let cell = Cell::new(0);
        let mut v: SmallVec<[DropCounter; 2]> = SmallVec::new();
        v.push(DropCounter(&cell));
        v.push(DropCounter(&cell));
        v.push(DropCounter(&cell));
        {
            let mut it = v.into_iter();
            assert!(it.next().is_some());
            assert!(it.next_back().is_some());
        }
        assert_eq!(cell.get(), 3);
    }
}

#[test]
fn test_capacity() {
    let mut v: SmallVec<[u8; 2]> = SmallVec::new();
    v.reserve(1);
    assert_eq!(v.capacity(), 2);
    assert!(!v.spilled());

    v.reserve_exact(0x100);
    assert!(v.capacity() >= 0x100);

    v.push(0);
    v.push(1);
    v.push(2);
    v.push(3);

    v.shrink_to_fit();
    assert!(v.capacity() < 0x100);
}

#[test]
fn test_truncate() {
    let mut v: SmallVec<[Box<u8>; 8]> = SmallVec::new();

    for x in 0..8 {
        v.push(Box::new(x));
    }
    v.truncate(4);

    assert_eq!(v.len(), 4);
    assert!(!v.spilled());

    assert_eq!(*v.swap_remove(1), 1);
    assert_eq!(*v.remove(1), 3);
    v.insert(1, Box::new(3));

    assert_eq!(&v.iter().map(|v| **v).collect::<Vec<_>>(), &[0, 3, 2]);
}

#[test]
fn test_insert_many() {
    let mut v: SmallVec<[u8; 8]> = SmallVec::new();
    for x in 0..4 {
        v.push(x);
    }
    assert_eq!(v.len(), 4);
    v.insert_many(1, [5, 6].iter().cloned());
    assert_eq!(
        &v.iter().map(|v| *v).collect::<Vec<_>>(),
        &[0, 5, 6, 1, 2, 3]
    );
}

struct MockHintIter<T: Iterator> {
    x: T,
    hint: usize,
}
impl<T: Iterator> Iterator for MockHintIter<T> {
    type Item = T::Item;
    fn next(&mut self) -> Option<Self::Item> {
        self.x.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.hint, None)
    }
}

#[test]
fn test_insert_many_short_hint() {
    let mut v: SmallVec<[u8; 8]> = SmallVec::new();
    for x in 0..4 {
        v.push(x);
    }
    assert_eq!(v.len(), 4);
    v.insert_many(
        1,
        MockHintIter {
            x: [5, 6].iter().cloned(),
            hint: 5,
        },
    );
    assert_eq!(
        &v.iter().map(|v| *v).collect::<Vec<_>>(),
        &[0, 5, 6, 1, 2, 3]
    );
}

#[test]
fn test_insert_many_long_hint() {
    let mut v: SmallVec<[u8; 8]> = SmallVec::new();
    for x in 0..4 {
        v.push(x);
    }
    assert_eq!(v.len(), 4);
    v.insert_many(
        1,
        MockHintIter {
            x: [5, 6].iter().cloned(),
            hint: 1,
        },
    );
    assert_eq!(
        &v.iter().map(|v| *v).collect::<Vec<_>>(),
        &[0, 5, 6, 1, 2, 3]
    );
}

#[cfg(feature = "std")]
#[test]
// https://github.com/servo/rust-smallvec/issues/96
fn test_insert_many_panic() {
    struct PanicOnDoubleDrop {
        dropped: Box<bool>,
    }

    impl Drop for PanicOnDoubleDrop {
        fn drop(&mut self) {
            assert!(!*self.dropped, "already dropped");
            *self.dropped = true;
        }
    }

    struct BadIter;
    impl Iterator for BadIter {
        type Item = PanicOnDoubleDrop;
        fn size_hint(&self) -> (usize, Option<usize>) {
            (1, None)
        }
        fn next(&mut self) -> Option<Self::Item> {
            panic!()
        }
    }

    let mut vec: SmallVec<[PanicOnDoubleDrop; 0]> = vec![
        PanicOnDoubleDrop {
            dropped: Box::new(false),
        },
        PanicOnDoubleDrop {
            dropped: Box::new(false),
        },
    ]
    .into();
    let result = std::panic::catch_unwind(move || {
        vec.insert_many(0, BadIter);
    });
    assert!(result.is_err());
}

#[test]
#[should_panic]
fn test_invalid_grow() {
    let mut v: SmallVec<[u8; 8]> = SmallVec::new();
    v.extend(0..8);
    v.grow(5);
}

#[test]
fn test_insert_from_slice() {
    let mut v: SmallVec<[u8; 8]> = SmallVec::new();
    for x in 0..4 {
        v.push(x);
    }
    assert_eq!(v.len(), 4);
    v.insert_from_slice(1, &[5, 6]);
    assert_eq!(
        &v.iter().map(|v| *v).collect::<Vec<_>>(),
        &[0, 5, 6, 1, 2, 3]
    );
}

#[test]
fn test_extend_from_slice() {
    let mut v: SmallVec<[u8; 8]> = SmallVec::new();
    for x in 0..4 {
        v.push(x);
    }
    assert_eq!(v.len(), 4);
    v.extend_from_slice(&[5, 6]);
    assert_eq!(
        &v.iter().map(|v| *v).collect::<Vec<_>>(),
        &[0, 1, 2, 3, 5, 6]
    );
}

#[test]
#[should_panic]
fn test_drop_panic_smallvec() {
    // This test should only panic once, and not double panic,
    // which would mean a double drop
    struct DropPanic;

    impl Drop for DropPanic {
        fn drop(&mut self) {
            panic!("drop");
        }
    }

    let mut v = SmallVec::<[_; 1]>::new();
    v.push(DropPanic);
}

#[test]
fn test_eq() {
    let mut a: SmallVec<[u32; 2]> = SmallVec::new();
    let mut b: SmallVec<[u32; 2]> = SmallVec::new();
    let mut c: SmallVec<[u32; 2]> = SmallVec::new();
    // a = [1, 2]
    a.push(1);
    a.push(2);
    // b = [1, 2]
    b.push(1);
    b.push(2);
    // c = [3, 4]
    c.push(3);
    c.push(4);

    assert!(a == b);
    assert!(a != c);
}

#[test]
fn test_ord() {
    let mut a: SmallVec<[u32; 2]> = SmallVec::new();
    let mut b: SmallVec<[u32; 2]> = SmallVec::new();
    let mut c: SmallVec<[u32; 2]> = SmallVec::new();
    // a = [1]
    a.push(1);
    // b = [1, 1]
    b.push(1);
    b.push(1);
    // c = [1, 2]
    c.push(1);
    c.push(2);

    assert!(a < b);
    assert!(b > a);
    assert!(b < c);
    assert!(c > b);
}

#[cfg(feature = "std")]
#[test]
fn test_hash() {
    use std::{collections::hash_map::DefaultHasher, hash::Hash};

    {
        let mut a: SmallVec<[u32; 2]> = SmallVec::new();
        let b = [1, 2];
        a.extend(b.iter().cloned());
        let mut hasher = DefaultHasher::new();
        assert_eq!(a.hash(&mut hasher), b.hash(&mut hasher));
    }
    {
        let mut a: SmallVec<[u32; 2]> = SmallVec::new();
        let b = [1, 2, 11, 12];
        a.extend(b.iter().cloned());
        let mut hasher = DefaultHasher::new();
        assert_eq!(a.hash(&mut hasher), b.hash(&mut hasher));
    }
}

#[test]
fn test_as_ref() {
    let mut a: SmallVec<[u32; 2]> = SmallVec::new();
    a.push(1);
    assert_eq!(a.as_ref(), [1]);
    a.push(2);
    assert_eq!(a.as_ref(), [1, 2]);
    a.push(3);
    assert_eq!(a.as_ref(), [1, 2, 3]);
}

#[test]
fn test_as_mut() {
    let mut a: SmallVec<[u32; 2]> = SmallVec::new();
    a.push(1);
    assert_eq!(a.as_mut(), [1]);
    a.push(2);
    assert_eq!(a.as_mut(), [1, 2]);
    a.push(3);
    assert_eq!(a.as_mut(), [1, 2, 3]);
    a.as_mut()[1] = 4;
    assert_eq!(a.as_mut(), [1, 4, 3]);
}

#[test]
fn test_borrow() {
    use core::borrow::Borrow;

    let mut a: SmallVec<[u32; 2]> = SmallVec::new();
    a.push(1);
    assert_eq!(a.borrow(), [1]);
    a.push(2);
    assert_eq!(a.borrow(), [1, 2]);
    a.push(3);
    assert_eq!(a.borrow(), [1, 2, 3]);
}

#[test]
fn test_borrow_mut() {
    use core::borrow::BorrowMut;

    let mut a: SmallVec<[u32; 2]> = SmallVec::new();
    a.push(1);
    assert_eq!(a.borrow_mut(), [1]);
    a.push(2);
    assert_eq!(a.borrow_mut(), [1, 2]);
    a.push(3);
    assert_eq!(a.borrow_mut(), [1, 2, 3]);
    BorrowMut::<[u32]>::borrow_mut(&mut a)[1] = 4;
    assert_eq!(a.borrow_mut(), [1, 4, 3]);
}

#[test]
fn test_from() {
    assert_eq!(&SmallVec::<[u32; 2]>::from(&[1][..])[..], [1]);
    assert_eq!(&SmallVec::<[u32; 2]>::from(&[1, 2, 3][..])[..], [1, 2, 3]);

    let vec = vec![];
    let small_vec: SmallVec<[u8; 3]> = SmallVec::from(vec);
    assert_eq!(&*small_vec, &[]);
    drop(small_vec);

    let vec = vec![1, 2, 3, 4, 5];
    let small_vec: SmallVec<[u8; 3]> = SmallVec::from(vec);
    assert_eq!(&*small_vec, &[1, 2, 3, 4, 5]);
    drop(small_vec);

    let vec = vec![1, 2, 3, 4, 5];
    let small_vec: SmallVec<[u8; 1]> = SmallVec::from(vec);
    assert_eq!(&*small_vec, &[1, 2, 3, 4, 5]);
    drop(small_vec);

    let array = [1];
    let small_vec: SmallVec<[u8; 1]> = SmallVec::from(array);
    assert_eq!(&*small_vec, &[1]);
    drop(small_vec);

    let array = [99; 128];
    let small_vec: SmallVec<[u8; 128]> = SmallVec::from(array);
    assert_eq!(&*small_vec, vec![99u8; 128].as_slice());
    drop(small_vec);
}

#[test]
fn test_from_slice() {
    assert_eq!(&SmallVec::<[u32; 2]>::from_slice(&[1][..])[..], [1]);
    assert_eq!(
        &SmallVec::<[u32; 2]>::from_slice(&[1, 2, 3][..])[..],
        [1, 2, 3]
    );
}

#[test]
fn test_exact_size_iterator() {
    let mut vec = SmallVec::<[u32; 2]>::from(&[1, 2, 3][..]);
    assert_eq!(vec.clone().into_iter().len(), 3);
    assert_eq!(vec.drain().len(), 3);
}

#[test]
fn shrink_to_fit_unspill() {
    let mut vec = SmallVec::<[u8; 2]>::from_iter(0..3);
    vec.pop();
    assert!(vec.spilled());
    vec.shrink_to_fit();
    assert!(!vec.spilled(), "shrink_to_fit will un-spill if possible");
}

#[test]
fn test_into_vec() {
    let vec = SmallVec::<[u8; 2]>::from_iter(0..2);
    assert_eq!(vec.into_vec(), vec![0, 1]);

    let vec = SmallVec::<[u8; 2]>::from_iter(0..3);
    assert_eq!(vec.into_vec(), vec![0, 1, 2]);
}

#[test]
fn test_into_inner() {
    let vec = SmallVec::<[u8; 2]>::from_iter(0..2);
    assert_eq!(vec.into_inner(), Ok([0, 1]));

    let vec = SmallVec::<[u8; 2]>::from_iter(0..1);
    assert_eq!(vec.clone().into_inner(), Err(vec));

    let vec = SmallVec::<[u8; 2]>::from_iter(0..3);
    assert_eq!(vec.clone().into_inner(), Err(vec));
}

#[test]
fn test_from_vec() {
    let vec = vec![];
    let small_vec: SmallVec<[u8; 3]> = SmallVec::from_vec(vec);
    assert_eq!(&*small_vec, &[]);
    drop(small_vec);

    let vec = vec![];
    let small_vec: SmallVec<[u8; 1]> = SmallVec::from_vec(vec);
    assert_eq!(&*small_vec, &[]);
    drop(small_vec);

    let vec = vec![1];
    let small_vec: SmallVec<[u8; 3]> = SmallVec::from_vec(vec);
    assert_eq!(&*small_vec, &[1]);
    drop(small_vec);

    let vec = vec![1, 2, 3];
    let small_vec: SmallVec<[u8; 3]> = SmallVec::from_vec(vec);
    assert_eq!(&*small_vec, &[1, 2, 3]);
    drop(small_vec);

    let vec = vec![1, 2, 3, 4, 5];
    let small_vec: SmallVec<[u8; 3]> = SmallVec::from_vec(vec);
    assert_eq!(&*small_vec, &[1, 2, 3, 4, 5]);
    drop(small_vec);

    let vec = vec![1, 2, 3, 4, 5];
    let small_vec: SmallVec<[u8; 1]> = SmallVec::from_vec(vec);
    assert_eq!(&*small_vec, &[1, 2, 3, 4, 5]);
    drop(small_vec);
}

#[test]
fn test_retain() {
    // Test inline data storate
    let mut sv: SmallVec<[i32; 5]> = SmallVec::from_slice(&[1, 2, 3, 3, 4]);
    sv.retain(|&mut i| i != 3);
    assert_eq!(sv.pop(), Some(4));
    assert_eq!(sv.pop(), Some(2));
    assert_eq!(sv.pop(), Some(1));
    assert_eq!(sv.pop(), None);

    // Test spilled data storage
    let mut sv: SmallVec<[i32; 3]> = SmallVec::from_slice(&[1, 2, 3, 3, 4]);
    sv.retain(|&mut i| i != 3);
    assert_eq!(sv.pop(), Some(4));
    assert_eq!(sv.pop(), Some(2));
    assert_eq!(sv.pop(), Some(1));
    assert_eq!(sv.pop(), None);

    // Test that drop implementations are called for inline.
    let one = Rc::new(1);
    let mut sv: SmallVec<[Rc<i32>; 3]> = SmallVec::new();
    sv.push(Rc::clone(&one));
    assert_eq!(Rc::strong_count(&one), 2);
    sv.retain(|_| false);
    assert_eq!(Rc::strong_count(&one), 1);

    // Test that drop implementations are called for spilled data.
    let mut sv: SmallVec<[Rc<i32>; 1]> = SmallVec::new();
    sv.push(Rc::clone(&one));
    sv.push(Rc::new(2));
    assert_eq!(Rc::strong_count(&one), 2);
    sv.retain(|_| false);
    assert_eq!(Rc::strong_count(&one), 1);
}

#[test]
fn test_dedup() {
    let mut dupes: SmallVec<[i32; 5]> = SmallVec::from_slice(&[1, 1, 2, 3, 3]);
    dupes.dedup();
    assert_eq!(&*dupes, &[1, 2, 3]);

    let mut empty: SmallVec<[i32; 5]> = SmallVec::new();
    empty.dedup();
    assert!(empty.is_empty());

    let mut all_ones: SmallVec<[i32; 5]> = SmallVec::from_slice(&[1, 1, 1, 1, 1]);
    all_ones.dedup();
    assert_eq!(all_ones.len(), 1);

    let mut no_dupes: SmallVec<[i32; 5]> = SmallVec::from_slice(&[1, 2, 3, 4, 5]);
    no_dupes.dedup();
    assert_eq!(no_dupes.len(), 5);
}

#[test]
fn test_resize() {
    let mut v: SmallVec<[i32; 8]> = SmallVec::new();
    v.push(1);
    v.resize(5, 0);
    assert_eq!(v[..], [1, 0, 0, 0, 0][..]);

    v.resize(2, -1);
    assert_eq!(v[..], [1, 0][..]);
}

#[cfg(feature = "std")]
#[test]
fn test_write() {
    use std::io::Write;

    let data = [1, 2, 3, 4, 5];

    let mut small_vec: SmallVec<[u8; 2]> = SmallVec::new();
    let len = small_vec.write(&data[..]).unwrap();
    assert_eq!(len, 5);
    assert_eq!(small_vec.as_ref(), data.as_ref());

    let mut small_vec: SmallVec<[u8; 2]> = SmallVec::new();
    small_vec.write_all(&data[..]).unwrap();
    assert_eq!(small_vec.as_ref(), data.as_ref());
}

#[cfg(feature = "serde")]
extern crate bincode;

#[cfg(feature = "serde")]
#[test]
fn test_serde() {
    use self::bincode::{config, deserialize};
    let mut small_vec: SmallVec<[i32; 2]> = SmallVec::new();
    small_vec.push(1);
    let encoded = config().limit(100).serialize(&small_vec).unwrap();
    let decoded: SmallVec<[i32; 2]> = deserialize(&encoded).unwrap();
    assert_eq!(small_vec, decoded);
    small_vec.push(2);
    // Spill the vec
    small_vec.push(3);
    small_vec.push(4);
    // Check again after spilling.
    let encoded = config().limit(100).serialize(&small_vec).unwrap();
    let decoded: SmallVec<[i32; 2]> = deserialize(&encoded).unwrap();
    assert_eq!(small_vec, decoded);
}

#[test]
fn grow_to_shrink() {
    let mut v: SmallVec<[u8; 2]> = SmallVec::new();
    v.push(1);
    v.push(2);
    v.push(3);
    assert!(v.spilled());
    v.clear();
    // Shrink to inline.
    v.grow(2);
    assert!(!v.spilled());
    assert_eq!(v.capacity(), 2);
    assert_eq!(v.len(), 0);
    v.push(4);
    assert_eq!(v[..], [4]);
}

#[test]
fn resumable_extend() {
    let s = "a b c";
    // This iterator yields: (Some('a'), None, Some('b'), None, Some('c')), None
    let it = s
        .chars()
        .scan(0, |_, ch| if ch.is_whitespace() { None } else { Some(ch) });
    let mut v: SmallVec<[char; 4]> = SmallVec::new();
    v.extend(it);
    assert_eq!(v[..], ['a']);
}

#[test]
fn grow_spilled_same_size() {
    let mut v: SmallVec<[u8; 2]> = SmallVec::new();
    v.push(0);
    v.push(1);
    v.push(2);
    assert!(v.spilled());
    assert_eq!(v.capacity(), 4);
    // grow with the same capacity
    v.grow(4);
    assert_eq!(v.capacity(), 4);
    assert_eq!(v[..], [0, 1, 2]);
}
