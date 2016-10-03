/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Small vectors in various sizes. These store a certain number of elements inline and fall back
//! to the heap for larger allocations.

use std::borrow::{Borrow, BorrowMut};
use std::cmp;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter::{IntoIterator, FromIterator};
use std::mem;
use std::ops;
use std::ptr;
use std::slice;

use SmallVecData::{Inline, Heap};


unsafe fn deallocate<T>(ptr: *mut T, capacity: usize) {
    let _vec: Vec<T> = Vec::from_raw_parts(ptr, 0, capacity);
    // Let it drop.
}

pub struct Drain<'a, T: 'a> {
    iter: slice::IterMut<'a,T>,
}

impl<'a, T: 'a> Iterator for Drain<'a,T> {
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

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T: 'a> DoubleEndedIterator for Drain<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        match self.iter.next_back() {
            None => None,
            Some(reference) => {
                unsafe {
                    Some(ptr::read(reference))
                }
            }
        }
    }
}

impl<'a, T> ExactSizeIterator for Drain<'a, T> { }

impl<'a, T: 'a> Drop for Drain<'a,T> {
    fn drop(&mut self) {
        // Destroy the remaining elements.
        for _ in self.by_ref() {}
    }
}

enum SmallVecData<A: Array> {
    Inline { array: A },
    Heap { ptr: *mut A::Item, capacity: usize },
}

impl<A: Array> SmallVecData<A> {
    fn ptr_mut(&mut self) -> *mut A::Item {
        match *self {
            Inline { ref mut array } => array.ptr_mut(),
            Heap { ptr, .. } => ptr,
        }
    }
}

unsafe impl<A: Array + Send> Send for SmallVecData<A> {}
unsafe impl<A: Array + Sync> Sync for SmallVecData<A> {}

impl<A: Array> Drop for SmallVecData<A> {
    fn drop(&mut self) {
        unsafe {
            match *self {
                ref mut inline @ Inline { .. } => {
                    // Inhibit the array destructor.
                    ptr::write(inline, Heap {
                        ptr: ptr::null_mut(),
                        capacity: 0,
                    });
                }
                Heap { ptr, capacity } => deallocate(ptr, capacity),
            }
        }
    }
}


pub struct SmallVec<A: Array> {
    len: usize,
    data: SmallVecData<A>,
}

impl<A: Array> SmallVec<A> {
    pub unsafe fn set_len(&mut self, new_len: usize) {
        self.len = new_len
    }

    pub fn inline_size(&self) -> usize {
        A::size()
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

    pub fn drain(&mut self) -> Drain<A::Item> {
        unsafe {
            let current_len = self.len();
            self.set_len(0);

            let ptr = self.data.ptr_mut();

            let slice = slice::from_raw_parts_mut(ptr, current_len);

            Drain {
                iter: slice.iter_mut(),
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

    pub fn reserve(&mut self, additional: usize) {
        let len = self.len();
        if self.capacity() - len < additional {
            match len.checked_add(additional).and_then(usize::checked_next_power_of_two) {
                Some(cap) => self.grow(cap),
                None => self.grow(usize::max_value()),
            }
        }
    }

    pub fn reserve_exact(&mut self, additional: usize) {
        let len = self.len();
        if self.capacity() - len < additional {
            match len.checked_add(additional) {
                Some(cap) => self.grow(cap),
                None => panic!("reserve_exact overflow"),
            }
        }
    }

    pub fn shrink_to_fit(&mut self) {
        let len = self.len;
        if self.spilled() && self.capacity() > len {
            self.grow(len);
        }
    }

    pub fn truncate(&mut self, len: usize) {
        let end_ptr = self.as_ptr();
        while len < self.len {
            unsafe {
                let last_index = self.len - 1;
                self.set_len(last_index);
                ptr::read(end_ptr.offset(last_index as isize));
            }
        }
    }

    pub fn swap_remove(&mut self, index: usize) -> A::Item {
        let len = self.len;
        self.swap(len - 1, index);
        self.pop().unwrap()
    }

    pub fn clear(&mut self) {
        self.truncate(0);
    }

    pub fn remove(&mut self, index: usize) -> A::Item {
        let len = self.len();

        assert!(index < len);

        unsafe {
            let ptr = self.as_mut_ptr().offset(index as isize);
            let item = ptr::read(ptr);
            ptr::copy(ptr.offset(1), ptr, len - index - 1);
            self.set_len(len - 1);
            item
        }
    }

    pub fn insert(&mut self, index: usize, element: A::Item) {
        self.reserve(1);

        let len = self.len;
        assert!(index <= len);

        unsafe {
            let ptr = self.as_mut_ptr().offset(index as isize);
            ptr::copy(ptr, ptr.offset(1), len - index);
            ptr::write(ptr, element);
            self.set_len(len + 1);
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
        let ptr = self.data.ptr_mut();
        unsafe {
            slice::from_raw_parts_mut(ptr, self.len)
        }
    }
}

impl<A: Array> AsRef<[A::Item]> for SmallVec<A> {
    #[inline]
    fn as_ref(&self) -> &[A::Item] {
        self
    }
}

impl<A: Array> AsMut<[A::Item]> for SmallVec<A> {
    #[inline]
    fn as_mut(&mut self) -> &mut [A::Item] {
        self
    }
}

impl<A: Array> Borrow<[A::Item]> for SmallVec<A> {
    #[inline]
    fn borrow(&self) -> &[A::Item] {
        self
    }
}

impl<A: Array> BorrowMut<[A::Item]> for SmallVec<A> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [A::Item] {
        self
    }
}

impl<'a, A: Array> From<&'a [A::Item]> for SmallVec<A> where A::Item: Clone {
    #[inline]
    fn from(slice: &'a [A::Item]) -> SmallVec<A> {
        slice.into_iter().cloned().collect()
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


impl<A: Array> FromIterator<A::Item> for SmallVec<A> {
    fn from_iter<I: IntoIterator<Item=A::Item>>(iterable: I) -> SmallVec<A> {
        let mut v = SmallVec::new();
        v.extend(iterable);
        v
    }
}

impl<A: Array> Extend<A::Item> for SmallVec<A> {
    fn extend<I: IntoIterator<Item=A::Item>>(&mut self, iterable: I) {
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

impl<A: Array> Default for SmallVec<A> {
    #[inline]
    fn default() -> SmallVec<A> {
        SmallVec::new()
    }
}

impl<A: Array> Drop for SmallVec<A> {
    fn drop(&mut self) {
        // Note on panic safety: dropping an element may panic,
        // but the inner SmallVecData destructor will still run
        unsafe {
            let ptr = self.as_ptr();
            for i in 0 .. self.len {
                ptr::read(ptr.offset(i as isize));
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

impl<A: Array, B: Array> PartialEq<SmallVec<B>> for SmallVec<A>
    where A::Item: PartialEq<B::Item> {
    #[inline]
    fn eq(&self, other: &SmallVec<B>) -> bool { self[..] == other[..] }
    #[inline]
    fn ne(&self, other: &SmallVec<B>) -> bool { self[..] != other[..] }
}

impl<A: Array> Eq for SmallVec<A> where A::Item: Eq {}

impl<A: Array> PartialOrd for SmallVec<A> where A::Item: PartialOrd {
    #[inline]
    fn partial_cmp(&self, other: &SmallVec<A>) -> Option<cmp::Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

impl<A: Array> Ord for SmallVec<A> where A::Item: Ord {
    #[inline]
    fn cmp(&self, other: &SmallVec<A>) -> cmp::Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<A: Array> Hash for SmallVec<A> where A::Item: Hash {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

unsafe impl<A: Array> Send for SmallVec<A> where A::Item: Send {}

pub struct IntoIter<A: Array> {
    data: SmallVecData<A>,
    current: usize,
    end: usize,
}

impl<A: Array> Drop for IntoIter<A> {
    fn drop(&mut self) {
        for _ in self { }
    }
}

impl<A: Array> Iterator for IntoIter<A> {
    type Item = A::Item;
    
    #[inline]
    fn next(&mut self) -> Option<A::Item> {
        if self.current == self.end {
            None    
        }
        else {
            unsafe {
                let current = self.current as isize;
                self.current += 1;
                Some(ptr::read(self.data.ptr_mut().offset(current)))
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.end - self.current;
        (size, Some(size))
    }
}

impl<A: Array> DoubleEndedIterator for IntoIter<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A::Item> {
        if self.current == self.end {
            None    
        }
        else {
            unsafe {
                self.end -= 1;
                Some(ptr::read(self.data.ptr_mut().offset(self.end as isize)))
            }
        }
    }
}

impl<A: Array> ExactSizeIterator for IntoIter<A> { }

impl<A: Array> IntoIterator for SmallVec<A> {
    type IntoIter = IntoIter<A>;
    type Item = A::Item;
    fn into_iter(mut self) -> Self::IntoIter {
        let len = self.len();
        unsafe {
            // Only grab the `data` field, the `IntoIter` type handles dropping of the elements
            let data = ptr::read(&mut self.data);
            mem::forget(self);
            IntoIter {
                data: data,
                current: 0,
                end: len,
            }
        }
    }
}

impl<'a, A: Array> IntoIterator for &'a SmallVec<A> {
    type IntoIter = slice::Iter<'a, A::Item>;
    type Item = &'a A::Item;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, A: Array> IntoIterator for &'a mut SmallVec<A> {
    type IntoIter = slice::IterMut<'a, A::Item>;
    type Item = &'a mut A::Item;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

// TODO: Remove these and its users.

/// Deprecated alias to ease transition from an earlier version.
#[deprecated]
pub type SmallVec1<T> = SmallVec<[T; 1]>;

/// Deprecated alias to ease transition from an earlier version.
#[deprecated]
pub type SmallVec2<T> = SmallVec<[T; 2]>;

/// Deprecated alias to ease transition from an earlier version.
#[deprecated]
pub type SmallVec4<T> = SmallVec<[T; 4]>;

/// Deprecated alias to ease transition from an earlier version.
#[deprecated]
pub type SmallVec8<T> = SmallVec<[T; 8]>;

/// Deprecated alias to ease transition from an earlier version.
#[deprecated]
pub type SmallVec16<T> = SmallVec<[T; 16]>;

/// Deprecated alias to ease transition from an earlier version.
#[deprecated]
pub type SmallVec24<T> = SmallVec<[T; 24]>;

/// Deprecated alias to ease transition from an earlier version.
#[deprecated]
pub type SmallVec32<T> = SmallVec<[T; 32]>;


pub unsafe trait Array {
    type Item;
    fn size() -> usize;
    fn ptr(&self) -> *const Self::Item;
    fn ptr_mut(&mut self) -> *mut Self::Item;
}

macro_rules! impl_array(
    ($($size:expr),+) => {
        $(
            unsafe impl<T> Array for [T; $size] {
                type Item = T;
                fn size() -> usize { $size }
                fn ptr(&self) -> *const T { &self[0] }
                fn ptr_mut(&mut self) -> *mut T { &mut self[0] }
            }
        )+
    }
);

impl_array!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 24, 32,
            0x40, 0x80, 0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000,
            0x10000, 0x20000, 0x40000, 0x80000, 0x100000);

#[cfg(test)]
pub mod tests {
    use SmallVec;
    use std::borrow::ToOwned;

    // We heap allocate all these strings so that double frees will show up under valgrind.

    #[test]
    pub fn test_inline() {
        let mut v = SmallVec::<[_; 16]>::new();
        v.push("hello".to_owned());
        v.push("there".to_owned());
        assert_eq!(&*v, &[
            "hello".to_owned(),
            "there".to_owned(),
        ][..]);
    }

    #[test]
    pub fn test_spill() {
        let mut v = SmallVec::<[_; 2]>::new();
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
        let mut v = SmallVec::<[_; 2]>::new();
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
        SmallVec::<[Box<u32>; 2]>::new();
    }

    /// https://github.com/servo/rust-smallvec/issues/5
    #[test]
    fn issue_5() {
        assert!(Some(SmallVec::<[&u32; 2]>::new()).is_some());
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
        use std::cell::Cell;

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

    #[test]
    fn test_hash() {
        use std::hash::{Hash, SipHasher};

        {
            let mut a: SmallVec<[u32; 2]> = SmallVec::new();
            let b = [1, 2];
            a.extend(b.iter().cloned());
            let mut hasher = SipHasher::new();
            assert_eq!(a.hash(&mut hasher), b.hash(&mut hasher));
        }
        {
            let mut a: SmallVec<[u32; 2]> = SmallVec::new();
            let b = [1, 2, 11, 12];
            a.extend(b.iter().cloned());
            let mut hasher = SipHasher::new();
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
        use std::borrow::Borrow;

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
        use std::borrow::BorrowMut;

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
    }

    #[test]
    fn test_exact_size_iterator() {
        let mut vec = SmallVec::<[u32; 2]>::from(&[1, 2, 3][..]);
        assert_eq!(vec.clone().into_iter().len(), 3);
        assert_eq!(vec.drain().len(), 3);
    }
}
