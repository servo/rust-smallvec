use {
    core::{
        borrow::{
            Borrow,
            BorrowMut
        },
        cell::Cell,
        hash::{
            Hash,
            Hasher
        },
        iter::FromIterator
    },
    smallvec::SmallVec,
    std::{
        borrow::ToOwned,
        boxed::Box,
        hash::DefaultHasher,
        rc::Rc,
        vec::Vec
    }
};

#[test]
pub fn zero() {
    let mut v = SmallVec::<_, 0>::new();
    assert!(!v.spilled());
    v.push(0usize);
    assert!(v.spilled());
    assert_eq!(&*v, &[0]);
}

// We heap allocate all these strings so that double frees will show up under
// valgrind.

#[test]
pub fn push_mut() {
    let mut v = SmallVec::<_, 16>::new();

    let first_elem = v.push_mut("hello".to_owned());
    assert_eq!(&*first_elem, &"hello".to_owned());

    *first_elem = "hi".to_owned();
    assert_eq!(&*first_elem, &"hi".to_owned());

    v.push("there".to_owned());
    assert_eq!(&*v, &["hi".to_owned(), "there".to_owned(),][..]);
}

#[test]
pub fn insert_mut() {
    let mut v = SmallVec::<_, 16>::new();
    v.push("hello".to_owned());
    v.push("there".to_owned());

    let second_elem = v.insert_mut(1, ",".to_owned());
    assert_eq!(&*second_elem, &",".to_owned());

    *second_elem = ";".to_owned();
    assert_eq!(&*second_elem, &";".to_owned());

    assert_eq!(
        &*v,
        &["hello".to_owned(), ";".to_owned(), "there".to_owned(),][..]
    );
}

#[test]
pub fn inline() {
    let mut v = SmallVec::<_, 16>::new();
    v.push("hello".to_owned());
    v.push("there".to_owned());
    assert_eq!(&*v, &["hello".to_owned(), "there".to_owned(),][..]);
}

#[test]
pub fn spill() {
    let mut v = SmallVec::<_, 2>::new();
    v.push("hello".to_owned());
    assert_eq!(v[0], "hello");
    v.push("there".to_owned());
    v.push("burma".to_owned());
    assert_eq!(v[0], "hello");
    v.push("shave".to_owned());
    assert_eq!(
        &*v,
        &[
            "hello".to_owned(),
            "there".to_owned(),
            "burma".to_owned(),
            "shave".to_owned(),
        ][..]
    );
}

#[test]
pub fn double_spill() {
    let mut v = SmallVec::<_, 2>::new();
    v.push("hello".to_owned());
    v.push("there".to_owned());
    v.push("burma".to_owned());
    v.push("shave".to_owned());
    v.push("hello".to_owned());
    v.push("there".to_owned());
    v.push("burma".to_owned());
    v.push("shave".to_owned());
    assert_eq!(
        &*v,
        &[
            "hello".to_owned(),
            "there".to_owned(),
            "burma".to_owned(),
            "shave".to_owned(),
            "hello".to_owned(),
            "there".to_owned(),
            "burma".to_owned(),
            "shave".to_owned(),
        ][..]
    );
}

// https://github.com/servo/rust-smallvec/issues/4
#[test]
fn issue_4() {
    SmallVec::<Box<u32>, 2>::new();
}

// https://github.com/servo/rust-smallvec/issues/5
#[test]
fn issue_5() {
    assert!(Some(SmallVec::<&u32, 2>::new()).is_some());
}

#[test]
fn with_capacity() {
    let v: SmallVec<u8, 3> = SmallVec::with_capacity(1);
    assert!(v.is_empty());
    assert!(!v.spilled());
    assert_eq!(v.capacity(), 3);

    let v: SmallVec<u8, 3> = SmallVec::with_capacity(10);
    assert!(v.is_empty());
    assert!(v.spilled());
    assert_eq!(v.capacity(), 10);
}

#[test]
fn drain() {
    let mut v: SmallVec<u8, 2> = SmallVec::new();
    v.push(3);
    assert_eq!(v.drain(..).collect::<Vec<_>>(), &[3]);

    // spilling the vec
    v.push(3);
    v.push(4);
    v.push(5);
    let old_capacity = v.capacity();
    assert_eq!(v.drain(1..).collect::<Vec<_>>(), &[4, 5]);
    // drain should not change the capacity
    assert_eq!(v.capacity(), old_capacity);

    // Exercise the tail-shifting code when in the inline state
    // This has the potential to produce UB due to aliasing
    let mut v: SmallVec<u8, 2> = SmallVec::new();
    v.push(1);
    v.push(2);
    assert_eq!(v.drain(..1).collect::<Vec<_>>(), &[1]);
}

#[test]
fn drain_rev() {
    let mut v: SmallVec<u8, 2> = SmallVec::new();
    v.push(3);
    assert_eq!(v.drain(..).rev().collect::<Vec<_>>(), &[3]);

    // spilling the vec
    v.push(3);
    v.push(4);
    v.push(5);
    assert_eq!(v.drain(..).rev().collect::<Vec<_>>(), &[5, 4, 3]);
}

#[test]
fn drain_forget() {
    let mut v: SmallVec<u8, 1> = SmallVec::from([0, 1, 2, 3, 4, 5, 6, 7]);
    std::mem::forget(v.drain(2..5));
    assert_eq!(v.len(), 2);
}

#[test]
fn splice() {
    // The range starts right before the end.
    let mut v: SmallVec<u8, 1> = SmallVec::from([0, 1, 2, 3, 4, 5, 6]);
    let new = [7, 8, 9, 10];
    let u: SmallVec<u8, 1> = v.splice(6.., new).collect();
    assert_eq!(v, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]);
    assert_eq!(u, [6]);

    // The range is empty.
    let mut v: SmallVec<u8, 1> = SmallVec::from([0, 1, 2, 3, 4, 5, 6]);
    let new = [7, 8, 9, 10];
    let u: SmallVec<u8, 1> = v.splice(1..1, new).collect();
    assert_eq!(v, [0, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6]);
    assert_eq!(u, [0u8; 0]);

    // The range is at the beginning and nonempty.
    let mut v: SmallVec<u8, 1> = SmallVec::from([0, 1, 2, 3, 4, 5, 6]);
    let new = [7, 8, 9, 10];
    let u: SmallVec<u8, 1> = v.splice(..3, new).collect();
    assert_eq!(v, [7, 8, 9, 10, 3, 4, 5, 6]);
    assert_eq!(u, [0, 1, 2]);
}

#[test]
fn splice_inline_fill_then_move_tail_ub_test() {
    let mut v: SmallVec<Box<usize>, 16> = (0..6).map(Box::new).collect();
    assert!(!v.spilled());
    let out: Vec<usize> = v
        .splice(1..3, (100..103).map(Box::new))
        .map(|b| *b)
        .collect();
    assert_eq!(out, [1, 2]);
    assert_eq!(
        v.iter().map(|b| **b).collect::<Vec<usize>>(),
        [0, 100, 101, 102, 3, 4, 5]
    );
    assert!(!v.spilled());
}

#[test]
fn into_iter() {
    let mut v: SmallVec<u8, 2> = SmallVec::new();
    v.push(3);
    assert_eq!(v.into_iter().collect::<Vec<_>>(), &[3]);

    // spilling the vec
    let mut v: SmallVec<u8, 2> = SmallVec::new();
    v.push(3);
    v.push(4);
    v.push(5);
    assert_eq!(v.into_iter().collect::<Vec<_>>(), &[3, 4, 5]);
}

#[test]
fn into_iter_rev() {
    let mut v: SmallVec<u8, 2> = SmallVec::new();
    v.push(3);
    assert_eq!(v.into_iter().rev().collect::<Vec<_>>(), &[3]);

    // spilling the vec
    let mut v: SmallVec<u8, 2> = SmallVec::new();
    v.push(3);
    v.push(4);
    v.push(5);
    assert_eq!(v.into_iter().rev().collect::<Vec<_>>(), &[5, 4, 3]);
}

#[test]
fn into_iter_drop() {
    struct DropCounter<'a>(&'a Cell<i32>);

    impl<'a> Drop for DropCounter<'a> {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    {
        let cell = Cell::new(0);
        let mut v: SmallVec<DropCounter<'_>, 2> = SmallVec::new();
        v.push(DropCounter(&cell));
        v.into_iter();
        assert_eq!(cell.get(), 1);
    }

    {
        let cell = Cell::new(0);
        let mut v: SmallVec<DropCounter<'_>, 2> = SmallVec::new();
        v.push(DropCounter(&cell));
        v.push(DropCounter(&cell));
        assert!(v.into_iter().next().is_some());
        assert_eq!(cell.get(), 2);
    }

    {
        let cell = Cell::new(0);
        let mut v: SmallVec<DropCounter<'_>, 2> = SmallVec::new();
        v.push(DropCounter(&cell));
        v.push(DropCounter(&cell));
        v.push(DropCounter(&cell));
        assert!(v.into_iter().next().is_some());
        assert_eq!(cell.get(), 3);
    }
    {
        let cell = Cell::new(0);
        let mut v: SmallVec<DropCounter<'_>, 2> = SmallVec::new();
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
fn capacity() {
    let mut v: SmallVec<u8, 2> = SmallVec::new();
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
fn truncate() {
    let mut v: SmallVec<Box<u8>, 8> = SmallVec::new();

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
fn truncate_references() {
    let mut v = Vec::from([0, 1, 2, 3, 4, 5, 6, 7]);
    let mut i = 8;
    let mut v: SmallVec<&mut u8, 8> = v.iter_mut().collect();

    v.truncate(4);

    assert_eq!(v.len(), 4);
    assert!(!v.spilled());

    assert_eq!(*v.swap_remove(1), 1);
    assert_eq!(*v.remove(1), 3);
    v.insert(1, &mut i);

    assert_eq!(
        &v.iter_mut().map(|v| &mut **v).collect::<Vec<_>>(),
        &[&mut 0, &mut 8, &mut 2]
    );
}

#[test]
fn split_off() {
    let mut vec: SmallVec<u32, 4> = SmallVec::from([1, 2, 3, 4, 5, 6]);
    let orig_ptr = vec.as_ptr();
    let orig_capacity = vec.capacity();

    let split_off = vec.split_off(4);
    assert_eq!(&vec[..], &[1, 2, 3, 4]);
    assert_eq!(&split_off[..], &[5, 6]);
    assert_eq!(vec.capacity(), orig_capacity);
    assert_eq!(vec.as_ptr(), orig_ptr);
}

#[test]
fn split_off_take_all() {
    // Allocate enough capacity that we can tell whether the split-off vector's
    // capacity is based on its size, or (incorrectly) on the original capacity.
    let mut vec = SmallVec::<u32, 4>::with_capacity(1000);
    vec.extend([1, 2, 3, 4, 5, 6]);
    let orig_ptr = vec.as_ptr();
    let orig_capacity: usize = vec.capacity();

    let split_off = vec.split_off(0);
    assert_eq!(&vec[..], &[0u32; 0]);
    assert_eq!(&split_off[..], &[1, 2, 3, 4, 5, 6]);
    assert_eq!(vec.capacity(), orig_capacity);
    assert_eq!(vec.as_ptr(), orig_ptr);

    // The split-off vector should be newly-allocated, and should not have
    // stolen the original vector's allocation.
    assert!(split_off.capacity() < orig_capacity);
    assert_ne!(split_off.as_ptr(), orig_ptr);
}

#[test]
fn append() {
    let mut v: SmallVec<u8, 8> = SmallVec::new();
    for x in 0..4 {
        v.push(x);
    }
    assert_eq!(v.len(), 4);

    let mut n: SmallVec<u8, 2> = SmallVec::from_buf([5, 6]);
    v.append(&mut n);
    assert_eq!(v.len(), 6);
    assert_eq!(n.len(), 0);

    assert_eq!(v.iter().copied().collect::<Vec<_>>(), [0, 1, 2, 3, 5, 6]);
}

#[test]
#[should_panic(expected = "new_capacity >= len")]
fn invalid_grow() {
    let mut v: SmallVec<u8, 8> = SmallVec::new();
    v.extend(0..8);
    v.grow(5);
}

#[test]
#[should_panic(expected = "attempted to index slice up to maximum usize")]
fn drain_overflow() {
    let mut v: SmallVec<u8, 8> = SmallVec::from([0]);
    v.drain(..=usize::MAX);
}

#[test]
fn extend_from_slice() {
    let mut v: SmallVec<u8, 8> = SmallVec::new();
    for x in 0..4 {
        v.push(x);
    }
    assert_eq!(v.len(), 4);
    v.extend_from_slice(&[5, 6]);
    assert_eq!(v.iter().copied().collect::<Vec<_>>(), [0, 1, 2, 3, 5, 6]);
}

#[test]
fn extend_from_within() {
    let mut v: SmallVec<u8, 8> = SmallVec::from([0, 1, 2, 3]);
    v.extend_from_within(1..3);
    assert_eq!(v.iter().copied().collect::<Vec<_>>(), [0, 1, 2, 3, 1, 2],);
}

#[test]
#[should_panic(expected = "drop")]
fn drop_panic_smallvec() {
    // This test should only panic once, and not double panic,
    // which would mean a double drop
    struct DropPanic;

    impl Drop for DropPanic {
        fn drop(&mut self) {
            panic!("drop");
        }
    }

    let mut v = SmallVec::<_, 1>::new();
    v.push(DropPanic);
}

#[test]
fn eq() {
    let mut a: SmallVec<u32, 2> = SmallVec::new();
    let mut b: SmallVec<u32, 2> = SmallVec::new();
    let mut c: SmallVec<u32, 2> = SmallVec::new();
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
fn ord() {
    let mut a: SmallVec<u32, 2> = SmallVec::new();
    let mut b: SmallVec<u32, 2> = SmallVec::new();
    let mut c: SmallVec<u32, 2> = SmallVec::new();
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

#[test]
fn hash() {
    fn hash(value: impl Hash) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    {
        let mut a: SmallVec<u32, 2> = SmallVec::new();
        let b = [1, 2];
        a.extend(b.iter().cloned());
        assert_eq!(hash(a), hash(b));
    }

    {
        let mut a: SmallVec<u32, 2> = SmallVec::new();
        let b = [1, 2, 11, 12];
        a.extend(b.iter().cloned());
        assert_eq!(hash(a), hash(b));
    }
}

#[test]
fn as_ref() {
    let mut a: SmallVec<u32, 2> = SmallVec::new();
    a.push(1);
    assert_eq!(a.as_ref(), [1]);
    a.push(2);
    assert_eq!(a.as_ref(), [1, 2]);
    a.push(3);
    assert_eq!(a.as_ref(), [1, 2, 3]);
}

#[test]
fn as_mut() {
    let mut a: SmallVec<u32, 2> = SmallVec::new();
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
fn borrow() {
    let mut a: SmallVec<u32, 2> = SmallVec::new();
    a.push(1);
    assert_eq!(a.borrow(), [1]);
    a.push(2);
    assert_eq!(a.borrow(), [1, 2]);
    a.push(3);
    assert_eq!(a.borrow(), [1, 2, 3]);
}

#[test]
fn borrow_mut() {
    let mut a: SmallVec<u32, 2> = SmallVec::new();
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
fn from() {
    assert_eq!(&SmallVec::<u32, 2>::from(&[1][..])[..], [1]);
    assert_eq!(&SmallVec::<u32, 2>::from(&[1, 2, 3][..])[..], [1, 2, 3]);

    let vec = Vec::new();
    let small_vec: SmallVec<u8, 3> = SmallVec::from(vec);
    assert_eq!(&*small_vec, &[0u8; 0]);
    drop(small_vec);

    let vec = Vec::from([1, 2, 3, 4, 5]);
    let small_vec: SmallVec<u8, 3> = SmallVec::from(vec);
    assert_eq!(&*small_vec, &[1, 2, 3, 4, 5]);
    drop(small_vec);

    let vec = Vec::from([1, 2, 3, 4, 5]);
    let small_vec: SmallVec<u8, 1> = SmallVec::from(vec);
    assert_eq!(&*small_vec, &[1, 2, 3, 4, 5]);
    drop(small_vec);

    let array = [1];
    let small_vec: SmallVec<u8, 1> = SmallVec::from(array);
    assert_eq!(&*small_vec, &[1]);
    drop(small_vec);

    let array = [99; 128];
    let small_vec: SmallVec<u8, 128> = SmallVec::from(array);
    assert_eq!(&*small_vec, Vec::from([99u8; 128]).as_slice());
    drop(small_vec);

    #[derive(PartialEq, Eq, Debug)]
    struct NoClone(u8);
    let array = [NoClone(42)];
    let small_vec: SmallVec<NoClone, 1> = SmallVec::from(array);
    assert_eq!(&*small_vec, &[NoClone(42)]);
    drop(small_vec);

    let vec = Vec::from([NoClone(42)]);
    let small_vec: SmallVec<NoClone, 1> = SmallVec::from(vec);
    assert_eq!(&*small_vec, &[NoClone(42)]);
    drop(small_vec);

    let array = [1; 128];
    let small_vec: SmallVec<u8, 1> = SmallVec::from(array);
    assert_eq!(&*small_vec, Vec::from([1; 128]).as_slice());
    drop(small_vec);

    let array = [99];
    let small_vec: SmallVec<u8, 128> = SmallVec::from(array);
    assert_eq!(&*small_vec, &[99u8]);
    drop(small_vec);
}

#[test]
fn from_slice() {
    assert_eq!(&SmallVec::<u32, 2>::from(&[1][..])[..], [1]);
    assert_eq!(&SmallVec::<u32, 2>::from(&[1, 2, 3][..])[..], [1, 2, 3]);
}

#[test]
fn exact_size_iterator() {
    let mut vec = SmallVec::<u32, 2>::from(&[1, 2, 3][..]);
    assert_eq!(vec.clone().into_iter().len(), 3);
    assert_eq!(vec.drain(..2).len(), 2);
    assert_eq!(vec.into_iter().len(), 1);
}

#[test]
fn into_iter_as_slice() {
    let vec = SmallVec::<u32, 2>::from(&[1, 2, 3][..]);
    let mut iter = vec.clone().into_iter();
    assert_eq!(iter.as_slice(), &[1, 2, 3]);
    assert_eq!(iter.as_mut_slice(), &[1, 2, 3]);
    iter.next();
    assert_eq!(iter.as_slice(), &[2, 3]);
    assert_eq!(iter.as_mut_slice(), &[2, 3]);
    iter.next_back();
    assert_eq!(iter.as_slice(), &[2]);
    assert_eq!(iter.as_mut_slice(), &[2]);
}

#[test]
fn into_iter_clone() {
    // Test that the cloned iterator yields identical elements and that it owns
    // its own copy (i.e. no use after move errors).
    let iter = SmallVec::<u8, 2>::from_iter(0..3).into_iter();
    let mut clone_iter = iter.clone();
    for x in iter {
        assert_eq!(x, clone_iter.next().unwrap());
    }
    assert_eq!(clone_iter.next(), None);
}

#[test]
fn into_iter_clone_partially_consumed_iterator() {
    // Test that the cloned iterator only contains the remaining elements of the
    // original iterator.
    let iter = SmallVec::<u8, 2>::from_iter(0..3).into_iter().skip(1);
    let mut clone_iter = iter.clone();
    for x in iter {
        assert_eq!(x, clone_iter.next().unwrap());
    }
    assert_eq!(clone_iter.next(), None);
}

#[test]
fn into_iter_clone_empty_smallvec() {
    let mut iter = SmallVec::<u8, 2>::new().into_iter();
    let mut clone_iter = iter.clone();
    assert_eq!(iter.next(), None);
    assert_eq!(clone_iter.next(), None);
}

#[test]
fn shrink_to_fit_unspill() {
    let mut vec = SmallVec::<u8, 2>::from_iter(0..3);
    vec.pop();
    assert!(vec.spilled());
    vec.shrink_to_fit();
    assert!(!vec.spilled(), "shrink_to_fit will un-spill if possible");
}

#[test]
fn shrink_after_from_empty_vec() {
    let mut v = SmallVec::<u8, 2>::from_vec(Vec::new());
    v.shrink_to_fit();
    assert!(!v.spilled())
}

#[test]
fn into_vec() {
    let vec = SmallVec::<u8, 2>::from_iter(0..2);
    assert_eq!(vec.into_vec(), Vec::from([0, 1]));

    let vec = SmallVec::<u8, 2>::from_iter(0..3);
    assert_eq!(vec.into_vec(), Vec::from([0, 1, 2]));
}

#[test]
fn into_inner() {
    let vec = SmallVec::<u8, 2>::from_iter(0..2);
    assert_eq!(vec.try_into(), Ok([0, 1]));

    let vec = SmallVec::<u8, 2>::from_iter(0..1);
    assert_eq!(vec.clone().try_into(), Err::<[u8; 7], SmallVec<u8, 2>>(vec));

    let vec = SmallVec::<u8, 2>::from_iter(0..3);
    assert_eq!(vec.clone().try_into(), Err::<[u8; 1], SmallVec<u8, 2>>(vec));
}

#[test]
fn try_into_array() {
    // Inline < capacity
    let vec = SmallVec::<u8, 2>::from_iter(0..1);
    assert_eq!(<[u8; 0]>::try_from(vec.clone()), Err(vec.clone()));
    assert_eq!(<[u8; 1]>::try_from(vec.clone()), Ok([0]));
    assert_eq!(<[u8; 2]>::try_from(vec.clone()), Err(vec));

    // Inline == capacity
    let vec = SmallVec::<u8, 2>::from_iter(0..2);
    assert_eq!(<[u8; 1]>::try_from(vec.clone()), Err(vec.clone()));
    assert_eq!(<[u8; 2]>::try_from(vec.clone()), Ok([0, 1]));
    assert_eq!(<[u8; 3]>::try_from(vec.clone()), Err(vec));

    // Heap
    let vec = SmallVec::<u8, 2>::from_iter(0..3);
    assert_eq!(<[u8; 2]>::try_from(vec.clone()), Err(vec.clone()));
    assert_eq!(<[u8; 3]>::try_from(vec.clone()), Ok([0, 1, 2]));
    assert_eq!(<[u8; 4]>::try_from(vec.clone()), Err(vec));
}

#[test]
fn from_vec() {
    let vec = Vec::new();
    let small_vec: SmallVec<u8, 3> = SmallVec::from_vec(vec);
    assert_eq!(&*small_vec, &[0u8; 0]);
    drop(small_vec);

    let vec = Vec::new();
    let small_vec: SmallVec<u8, 1> = SmallVec::from_vec(vec);
    assert_eq!(&*small_vec, &[0u8; 0]);
    drop(small_vec);

    let vec = Vec::from([1]);
    let small_vec: SmallVec<u8, 3> = SmallVec::from_vec(vec);
    assert_eq!(&*small_vec, &[1]);
    drop(small_vec);

    let vec = Vec::from([1, 2, 3]);
    let small_vec: SmallVec<u8, 3> = SmallVec::from_vec(vec);
    assert_eq!(&*small_vec, &[1, 2, 3]);
    drop(small_vec);

    let vec = Vec::from([1, 2, 3, 4, 5]);
    let small_vec: SmallVec<u8, 3> = SmallVec::from_vec(vec);
    assert_eq!(&*small_vec, &[1, 2, 3, 4, 5]);
    drop(small_vec);

    let vec = Vec::from([1, 2, 3, 4, 5]);
    let small_vec: SmallVec<u8, 1> = SmallVec::from_vec(vec);
    assert_eq!(&*small_vec, &[1, 2, 3, 4, 5]);
    drop(small_vec);
}

#[test]
fn retain() {
    // Test inline data storage
    let mut sv: SmallVec<i32, 5> = SmallVec::from(&[1, 2, 3, 3, 4]);
    sv.retain(|&i| i != 3);
    assert_eq!(sv.pop(), Some(4));
    assert_eq!(sv.pop(), Some(2));
    assert_eq!(sv.pop(), Some(1));
    assert_eq!(sv.pop(), None);

    // Test spilled data storage
    let mut sv: SmallVec<i32, 3> = SmallVec::from(&[1, 2, 3, 3, 4]);
    sv.retain(|&i| i != 3);
    assert_eq!(sv.pop(), Some(4));
    assert_eq!(sv.pop(), Some(2));
    assert_eq!(sv.pop(), Some(1));
    assert_eq!(sv.pop(), None);

    // Test that drop implementations are called for inline.
    let one = Rc::new(1);
    let mut sv: SmallVec<Rc<i32>, 3> = SmallVec::new();
    sv.push(Rc::clone(&one));
    assert_eq!(Rc::strong_count(&one), 2);
    sv.retain(|_| false);
    assert_eq!(Rc::strong_count(&one), 1);

    // Test that drop implementations are called for spilled data.
    let mut sv: SmallVec<Rc<i32>, 1> = SmallVec::new();
    sv.push(Rc::clone(&one));
    sv.push(Rc::new(2));
    assert_eq!(Rc::strong_count(&one), 2);
    sv.retain(|_| false);
    assert_eq!(Rc::strong_count(&one), 1);
}

#[test]
fn dedup() {
    let mut dupes: SmallVec<i32, 5> = SmallVec::from(&[1, 1, 2, 3, 3]);
    dupes.dedup();
    assert_eq!(&*dupes, &[1, 2, 3]);

    let mut empty: SmallVec<i32, 5> = SmallVec::new();
    empty.dedup();
    assert!(empty.is_empty());

    let mut all_ones: SmallVec<i32, 5> = SmallVec::from(&[1, 1, 1, 1, 1]);
    all_ones.dedup();
    assert_eq!(all_ones.len(), 1);

    let mut no_dupes: SmallVec<i32, 5> = SmallVec::from(&[1, 2, 3, 4, 5]);
    no_dupes.dedup();
    assert_eq!(no_dupes.len(), 5);
}

#[test]
fn resize() {
    let mut v: SmallVec<i32, 8> = SmallVec::new();
    v.push(1);
    v.resize(5, 0);
    assert_eq!(v[..], [1, 0, 0, 0, 0][..]);

    v.resize(2, -1);
    assert_eq!(v[..], [1, 0][..]);
}

#[test]
fn grow_to_shrink() {
    let mut v: SmallVec<u8, 2> = SmallVec::new();
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
    let mut v: SmallVec<char, 4> = SmallVec::new();
    v.extend(it);
    assert_eq!(v[..], ['a']);
}

// #139
#[test]
fn uninhabited() {
    enum Void {}
    let _sv = SmallVec::<Void, 8>::new();
}

#[test]
fn grow_spilled_same_size() {
    let mut v: SmallVec<u8, 2> = SmallVec::new();
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

#[test]
fn const_generics() {
    let _v = SmallVec::<i32, 987>::default();
}

#[test]
fn const_new() {
    let v = const_new_inner();
    assert_eq!(v.capacity(), 4);
    assert_eq!(v.len(), 0);
    let v = const_new_inline_sized();
    assert_eq!(v.capacity(), 4);
    assert_eq!(v.len(), 4);
    assert_eq!(v[0], 1);
    let v = const_new_inline_args();
    assert_eq!(v.capacity(), 2);
    assert_eq!(v.len(), 2);
    assert_eq!(v[0], 1);
    assert_eq!(v[1], 4);
}
const fn const_new_inner() -> SmallVec<i32, 4> {
    SmallVec::<i32, 4>::new()
}
const fn const_new_inline_sized() -> SmallVec<i32, 4> {
    SmallVec::from_buf([1; 4])
}
const fn const_new_inline_args() -> SmallVec<i32, 2> {
    SmallVec::from_buf([1, 4])
}

#[test]
fn zero_size_items() {
    SmallVec::<(), 0>::new().push(());
}

#[test]
fn clone_from() {
    let mut a: SmallVec<u8, 2> = SmallVec::new();
    a.push(1);
    a.push(2);
    a.push(3);

    let mut b: SmallVec<u8, 2> = SmallVec::new();
    b.push(10);

    let mut c: SmallVec<u8, 2> = SmallVec::new();
    c.push(20);
    c.push(21);
    c.push(22);

    a.clone_from(&b);
    assert_eq!(&*a, &[10]);

    b.clone_from(&c);
    assert_eq!(&*b, &[20, 21, 22]);
}

#[test]
fn extract_if() {
    let mut a: SmallVec<u8, 2> = SmallVec::from([0, 1u8, 2, 3, 4, 5, 6, 7, 8, 0]);

    let b: SmallVec<u8, 2> = a.extract_if(1..9, |x| *x % 3 == 0).collect();

    assert_eq!(a, SmallVec::<u8, 2>::from(&[0, 1u8, 2, 4, 5, 7, 8, 0]));
    assert_eq!(b, SmallVec::<u8, 2>::from(&[3u8, 6]));
}

/// This assortment of tests, in combination with miri, verifies we handle UB on
/// fishy arguments given to SmallVec. Draining and extending the allocation are
/// fairly well-tested earlier, but `smallvec.insert(usize::MAX, val)` once
/// slipped by!
///
/// All code that indexes into SmallVecs should be tested with such "trivially
/// wrong" args.
#[test]
fn max_dont_panic() {
    let mut sv: SmallVec<i32, 2> = SmallVec::from([0]);
    let _ = sv.get(usize::MAX);
    sv.truncate(usize::MAX);
}

#[test]
#[should_panic(expected = "removal index")]
fn max_remove() {
    let mut sv: SmallVec<i32, 2> = SmallVec::from([0]);
    sv.remove(usize::MAX);
}

#[test]
#[should_panic(expected = "swap_remove index")]
fn max_swap_remove() {
    let mut sv: SmallVec<i32, 2> = SmallVec::from([0]);
    sv.swap_remove(usize::MAX);
}

#[test]
#[should_panic(expected = "insertion index")]
fn max_insert() {
    let mut sv: SmallVec<i32, 2> = SmallVec::from([0]);
    sv.insert(usize::MAX, 0);
}

#[test]
fn collect_from_iter() {
    // Regression test for https://github.com/servo/rust-smallvec/issues/353
    struct IterNoHint<I: Iterator>(I);

    impl<I: Iterator> Iterator for IterNoHint<I> {
        type Item = I::Item;

        fn next(&mut self) -> Option<Self::Item> {
            self.0.next()
        }

        // no implementation of size_hint means it returns (0, None) - which
        // forces from_iter to grow the allocated space iteratively.
    }

    // A length of 3 is fine to trigger this bug under valgrind, but making the
    // vector 1 million elements makes it crash - which is much easier to
    // detect.
    #[cfg(miri)]
    const ELEMENTS: usize = 1000;
    #[cfg(not(miri))]
    const ELEMENTS: usize = 1_000_000;
    let iter = IterNoHint(std::iter::repeat_n(1u8, ELEMENTS));

    let _y: SmallVec<u8, 1> = SmallVec::from_iter(iter);
}

#[test]
fn collect_with_spill() {
    let input = "0123456";
    let collected: SmallVec<char, 4> = input.chars().collect();
    assert_eq!(collected, &['0', '1', '2', '3', '4', '5', '6']);
}

#[test]
fn spare_capacity_mut() {
    let mut v: SmallVec<u8, 2> = SmallVec::new();
    assert!(!v.spilled());
    let spare = v.spare_capacity_mut();
    assert_eq!(spare.len(), 2);
    assert_eq!(spare.as_ptr().cast::<u8>(), v.as_ptr());

    v.push(1);
    assert!(!v.spilled());
    let spare = v.spare_capacity_mut();
    assert_eq!(spare.len(), 1);
    assert_eq!(spare.as_ptr().cast::<u8>(), unsafe { v.as_ptr().add(1) });

    v.push(2);
    assert!(!v.spilled());
    let spare = v.spare_capacity_mut();
    assert_eq!(spare.len(), 0);
    assert_eq!(spare.as_ptr().cast::<u8>(), unsafe { v.as_ptr().add(2) });

    v.push(3);
    assert!(v.spilled());
    let spare = v.spare_capacity_mut();
    assert!(!spare.is_empty());
    assert_eq!(spare.as_ptr().cast::<u8>(), unsafe { v.as_ptr().add(3) });
}
