/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Small vectors in various sizes. These store a certain number of elements inline and fall back
//! to the heap for larger allocations.

use std::cmp;
use std::fmt;
use std::iter::{IntoIterator, FromIterator};
use std::mem;
use std::ops;
use std::ptr;
use std::slice;

use SmallVecData::{Inline, Heap};


pub trait VecLike<T>:
        ops::Index<usize, Output=T> +
        ops::IndexMut<usize> +
        ops::Index<ops::Range<usize>, Output=[T]> +
        ops::IndexMut<ops::Range<usize>> +
        ops::Index<ops::RangeFrom<usize>, Output=[T]> +
        ops::IndexMut<ops::RangeFrom<usize>> +
        ops::Index<ops::RangeTo<usize>, Output=[T]> +
        ops::IndexMut<ops::RangeTo<usize>> +
        ops::Index<ops::RangeFull, Output=[T]> +
        ops::IndexMut<ops::RangeFull> +
        ops::Deref +
        ops::DerefMut {

    fn len(&self) -> usize;
    fn push(&mut self, value: T);
}

impl<T> VecLike<T> for Vec<T> {
    #[inline]
    fn len(&self) -> usize {
        Vec::len(self)
    }

    #[inline]
    fn push(&mut self, value: T) {
        Vec::push(self, value);
    }
}

unsafe fn deallocate<T>(ptr: *mut T, capacity: usize) {
    let _vec: Vec<T> = Vec::from_raw_parts(ptr, 0, capacity);
    // Let it drop.
}

pub struct SmallVecMoveIterator<'a, T: 'a> {
    iter: slice::IterMut<'a,T>,
}

impl<'a, T: 'a> Iterator for SmallVecMoveIterator<'a,T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        match self.iter.next() {
            None => None,
            Some(reference) => {
                unsafe {
                    Some(ptr::read(reference))
                }
            }
        }
    }
}

impl<'a, T: 'a> Drop for SmallVecMoveIterator<'a,T> {
    fn drop(&mut self) {
        // Destroy the remaining elements.
        for _ in self.by_ref() {}
    }
}

enum SmallVecData<A: Array> {
    Inline { array: A },
    Heap { ptr: *mut A::Item, capacity: usize },
}


pub struct SmallVec<A: Array> {
    len: usize,
    data: SmallVecData<A>,
}

impl<A: Array> SmallVec<A> {
    unsafe fn set_len(&mut self, new_len: usize) {
        self.len = new_len
    }

    pub fn inline_size(&self) -> usize {
        A::size()
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn capacity(&self) -> usize {
        match self.data {
            Inline { .. } => A::size(),
            Heap { capacity, .. } => capacity,
        }
    }

    pub fn spilled(&self) -> bool {
        match self.data {
            Inline { .. } => false,
            Heap { .. } => true,
        }
    }

    /// NB: For efficiency reasons (avoiding making a second copy of the inline elements), this
    /// actually clears out the original array instead of moving it.
    /// FIXME: Rename this to `drain`? It’s more like `Vec::drain` than `Vec::into_iter`.
    pub fn into_iter<'a>(&'a mut self) -> SmallVecMoveIterator<'a, A::Item> {
        unsafe {
            self.set_len(0);
            SmallVecMoveIterator {
                iter: self.iter_mut(),
            }
        }
    }

    pub fn push(&mut self, value: A::Item) {
        let cap = self.capacity();
        if self.len == cap {
            self.grow(cmp::max(cap * 2, 1))
        }
        unsafe {
            let end = self.as_mut_ptr().offset(self.len as isize);
            ptr::write(end, value);
            let len = self.len;
            self.set_len(len + 1)
        }
    }

    pub fn push_all_move<V: IntoIterator<Item=A::Item>>(&mut self, other: V) {
        for value in other {
            self.push(value)
        }
    }

    pub fn pop(&mut self) -> Option<A::Item> {
        if self.len == 0 {
            return None
        }
        let last_index = self.len - 1;
        if (last_index as isize) < 0 {
            panic!("overflow")
        }
        unsafe {
            let end_ptr = self.as_mut_ptr().offset(last_index as isize);
            let value = ptr::replace(end_ptr, mem::uninitialized());
            self.set_len(last_index);
            Some(value)
        }
    }

    pub fn grow(&mut self, new_cap: usize) {
        let mut vec: Vec<A::Item> = Vec::with_capacity(new_cap);
        let new_alloc = vec.as_mut_ptr();
        unsafe {
            mem::forget(vec);
            ptr::copy_nonoverlapping(self.as_ptr(), new_alloc, self.len);

            match self.data {
                Inline { .. } => {}
                Heap { ptr, capacity } => deallocate(ptr, capacity),
            }
            ptr::write(&mut self.data, Heap {
                ptr: new_alloc,
                capacity: new_cap,
            });
        }
    }
}

impl<A: Array> ops::Deref for SmallVec<A> {
    type Target = [A::Item];
    #[inline]
    fn deref(&self) -> &[A::Item] {
        let ptr: *const _ = match self.data {
            Inline { ref array } => array.ptr(),
            Heap { ptr, .. } => ptr,
        };
        unsafe {
            slice::from_raw_parts(ptr, self.len)
        }
    }
}

impl<A: Array> ops::DerefMut for SmallVec<A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [A::Item] {
        let ptr = match self.data {
            Inline { ref mut array } => array.ptr_mut(),
            Heap { ptr, .. } => ptr,
        };
        unsafe {
            slice::from_raw_parts_mut(ptr, self.len)
        }
    }
}

macro_rules! impl_index {
    ($index_type: ty, $output_type: ty) => {
        impl<A: Array> ops::Index<$index_type> for SmallVec<A> {
            type Output = $output_type;
            #[inline]
            fn index(&self, index: $index_type) -> &$output_type {
                &(&**self)[index]
            }
        }

        impl<A: Array> ops::IndexMut<$index_type> for SmallVec<A> {
            #[inline]
            fn index_mut(&mut self, index: $index_type) -> &mut $output_type {
                &mut (&mut **self)[index]
            }
        }
    }
}

impl_index!(usize, A::Item);
impl_index!(ops::Range<usize>, [A::Item]);
impl_index!(ops::RangeFrom<usize>, [A::Item]);
impl_index!(ops::RangeTo<usize>, [A::Item]);
impl_index!(ops::RangeFull, [A::Item]);


impl<A: Array> VecLike<A::Item> for SmallVec<A> {
    #[inline]
    fn len(&self) -> usize {
        SmallVec::len(self)
    }

    #[inline]
    fn push(&mut self, value: A::Item) {
        SmallVec::push(self, value);
    }
}

impl<A: Array> FromIterator<A::Item> for SmallVec<A> {
    fn from_iter<I: IntoIterator<Item=A::Item>>(iterable: I) -> SmallVec<A> {
        let mut v = SmallVec::new();
        v.extend(iterable);
        v
    }
}

impl<A: Array> SmallVec<A> {
    pub fn extend<I: IntoIterator<Item=A::Item>>(&mut self, iterable: I) {
        let iter = iterable.into_iter();
        let (lower_size_bound, _) = iter.size_hint();

        let target_len = self.len + lower_size_bound;

        if target_len > self.capacity() {
           self.grow(target_len);
        }

        for elem in iter {
            self.push(elem);
        }
    }
}

impl<A: Array> fmt::Debug for SmallVec<A> where A::Item: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", &**self)
    }
}

impl<A: Array> SmallVec<A> {
    #[inline]
    pub fn new() -> SmallVec<A> {
        unsafe {
            SmallVec {
                len: 0,
                data: Inline { array: mem::zeroed() },
            }
        }
    }
}

impl<A: Array> Drop for SmallVec<A> {
    fn drop(&mut self) {
        unsafe {
            let ptr = self.as_ptr();
            for i in 0 .. self.len {
                ptr::read(ptr.offset(i as isize));
            }

            match self.data {
                Inline { .. } => {
                    // Inhibit the array destructor.
                    ptr::write(&mut self.data, Heap {
                        ptr: ptr::null_mut(),
                        capacity: 0,
                    });
                }
                Heap { ptr, capacity } => deallocate(ptr, capacity),
            }
        }
    }
}

impl<A: Array> Clone for SmallVec<A> where A::Item: Clone {
    fn clone(&self) -> SmallVec<A> {
        let mut new_vector = SmallVec::new();
        for element in self.iter() {
            new_vector.push((*element).clone())
        }
        new_vector
    }
}

unsafe impl<A: Array> Send for SmallVec<A> where A::Item: Send {}

// TODO: Remove these and its users.

/// Deprecated alias to ease transition from an earlier version.
pub type SmallVec1<T> = SmallVec<[T; 1]>;

/// Deprecated alias to ease transition from an earlier version.
pub type SmallVec2<T> = SmallVec<[T; 2]>;

/// Deprecated alias to ease transition from an earlier version.
pub type SmallVec4<T> = SmallVec<[T; 4]>;

/// Deprecated alias to ease transition from an earlier version.
pub type SmallVec8<T> = SmallVec<[T; 8]>;

/// Deprecated alias to ease transition from an earlier version.
pub type SmallVec16<T> = SmallVec<[T; 16]>;

/// Deprecated alias to ease transition from an earlier version.
pub type SmallVec24<T> = SmallVec<[T; 24]>;

/// Deprecated alias to ease transition from an earlier version.
pub type SmallVec32<T> = SmallVec<[T; 32]>;


pub trait Array {
    type Item;
    fn size() -> usize;
    fn ptr(&self) -> *const Self::Item;
    fn ptr_mut(&mut self) -> *mut Self::Item;
}

macro_rules! impl_array(
    ($($size:expr),+) => {
        $(
            impl<T> Array for [T; $size] {
                type Item = T;
                fn size() -> usize { $size }
                fn ptr(&self) -> *const T { &self[0] }
                fn ptr_mut(&mut self) -> *mut T { &mut self[0] }
            }
        )+
    }
);

impl_array!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 24, 32);

#[cfg(test)]
pub mod tests {
    use SmallVec2;
    use SmallVec16;
    use std::borrow::ToOwned;

    // We heap allocate all these strings so that double frees will show up under valgrind.

    #[test]
    pub fn test_inline() {
        let mut v = SmallVec16::new();
        v.push("hello".to_owned());
        v.push("there".to_owned());
        assert_eq!(&*v, &[
            "hello".to_owned(),
            "there".to_owned(),
        ][..]);
    }

    #[test]
    pub fn test_spill() {
        let mut v = SmallVec2::new();
        v.push("hello".to_owned());
        assert_eq!(v[0], "hello");
        v.push("there".to_owned());
        v.push("burma".to_owned());
        assert_eq!(v[0], "hello");
        v.push("shave".to_owned());
        assert_eq!(&*v, &[
            "hello".to_owned(),
            "there".to_owned(),
            "burma".to_owned(),
            "shave".to_owned(),
        ][..]);
    }

    #[test]
    pub fn test_double_spill() {
        let mut v = SmallVec2::new();
        v.push("hello".to_owned());
        v.push("there".to_owned());
        v.push("burma".to_owned());
        v.push("shave".to_owned());
        v.push("hello".to_owned());
        v.push("there".to_owned());
        v.push("burma".to_owned());
        v.push("shave".to_owned());
        assert_eq!(&*v, &[
            "hello".to_owned(),
            "there".to_owned(),
            "burma".to_owned(),
            "shave".to_owned(),
            "hello".to_owned(),
            "there".to_owned(),
            "burma".to_owned(),
            "shave".to_owned(),
        ][..]);
    }

    /// https://github.com/servo/rust-smallvec/issues/4
    #[test]
    fn issue_4() {
        SmallVec2::<Box<u32>>::new();
    }

    /// https://github.com/servo/rust-smallvec/issues/5
    #[test]
    fn issue_5() {
        assert!(Some(SmallVec2::<&u32>::new()).is_some());
    }
}
