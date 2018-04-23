/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Small vectors in various sizes. These store a certain number of elements inline, and fall back
//! to the heap for larger allocations.  This can be a useful optimization for improving cache
//! locality and reducing allocator traffic for workloads that fit within the inline buffer.
//!
//! ## no_std support
//!
//! By default, `smallvec` depends on `libstd`. However, it can be configured to use the unstable
//! `liballoc` API instead, for use on platforms that have `liballoc` but not `libstd`.  This
//! configuration is currently unstable and is not guaranteed to work on all versions of Rust.
//!
//! To depend on `smallvec` without `libstd`, use `default-features = false` in the `smallvec`
//! section of Cargo.toml to disable its `"std"` feature.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(not(feature = "std"), feature(alloc))]
#![cfg_attr(feature = "union", feature(untagged_unions))]
#![deny(missing_docs)]

#[cfg(not(feature = "std"))]
#[cfg_attr(test, macro_use)]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::Vec;

#[cfg(feature = "serde")]
extern crate serde;

#[cfg(not(feature = "std"))]
mod std {
    pub use core::*;
}

#[cfg(feature = "serde")]
use serde::de::{Deserialize, Deserializer, SeqAccess, Visitor};
#[cfg(feature = "serde")]
use serde::ser::{Serialize, SerializeSeq, Serializer};
use std::borrow::{Borrow, BorrowMut};
use std::cmp;
use std::fmt;
use std::hash::{Hash, Hasher};
#[cfg(feature = "std")]
use std::io;
use std::iter::{repeat, FromIterator, IntoIterator};
#[cfg(feature = "serde")]
use std::marker::PhantomData;
use std::mem;
#[cfg(not(feature = "union"))]
use std::mem::ManuallyDrop;
use std::ops;
use std::ptr;
use std::slice;

/// Creates a [`SmallVec`] containing the arguments.
///
/// `smallvec!` allows `SmallVec`s to be defined with the same syntax as array expressions.
/// There are two forms of this macro:
///
/// - Create a [`SmallVec`] containing a given list of elements:
///
/// ```
/// # #[macro_use] extern crate smallvec;
/// # use smallvec::SmallVec;
/// # fn main() {
/// let v: SmallVec<[_; 128]> = smallvec![1, 2, 3];
/// assert_eq!(v[0], 1);
/// assert_eq!(v[1], 2);
/// assert_eq!(v[2], 3);
/// # }
/// ```
///
/// - Create a [`SmallVec`] from a given element and size:
///
/// ```
/// # #[macro_use] extern crate smallvec;
/// # use smallvec::SmallVec;
/// # fn main() {
/// let v: SmallVec<[_; 0x8000]> = smallvec![1; 3];
/// assert_eq!(v, SmallVec::from_buf([1, 1, 1]));
/// # }
/// ```
///
/// Note that unlike array expressions this syntax supports all elements
/// which implement [`Clone`] and the number of elements doesn't have to be
/// a constant.
///
/// This will use `clone` to duplicate an expression, so one should be careful
/// using this with types having a nonstandard `Clone` implementation. For
/// example, `smallvec![Rc::new(1); 5]` will create a vector of five references
/// to the same boxed integer value, not five references pointing to independently
/// boxed integers.

#[macro_export]
macro_rules! smallvec {
    ($elem:expr; $n:expr) => ({
        SmallVec::from_elem($elem, $n)
    });
    ($($x:expr),*) => ({
        SmallVec::from_slice(&[$($x),*])
    });
}

/// Common operations implemented by both `Vec` and `SmallVec`.
///
/// This can be used to write generic code that works with both `Vec` and `SmallVec`.
///
/// ## Example
///
/// ```rust
/// use smallvec::{VecLike, SmallVec};
///
/// fn initialize<V: VecLike<u8>>(v: &mut V) {
///     for i in 0..5 {
///         v.push(i);
///     }
/// }
///
/// let mut vec = Vec::new();
/// initialize(&mut vec);
///
/// let mut small_vec = SmallVec::<[u8; 8]>::new();
/// initialize(&mut small_vec);
/// ```
#[deprecated(note = "Use `Extend` and `Deref<[T]>` instead")]
pub trait VecLike<T>:
    ops::Index<usize, Output = T>
    + ops::IndexMut<usize>
    + ops::Index<ops::Range<usize>, Output = [T]>
    + ops::IndexMut<ops::Range<usize>>
    + ops::Index<ops::RangeFrom<usize>, Output = [T]>
    + ops::IndexMut<ops::RangeFrom<usize>>
    + ops::Index<ops::RangeTo<usize>, Output = [T]>
    + ops::IndexMut<ops::RangeTo<usize>>
    + ops::Index<ops::RangeFull, Output = [T]>
    + ops::IndexMut<ops::RangeFull>
    + ops::DerefMut<Target = [T]>
    + Extend<T>
{
    /// Append an element to the vector.
    fn push(&mut self, value: T);
}

#[allow(deprecated)]
impl<T> VecLike<T> for Vec<T> {
    #[inline]
    fn push(&mut self, value: T) {
        Vec::push(self, value);
    }
}

/// Trait to be implemented by a collection that can be extended from a slice
///
/// ## Example
///
/// ```rust
/// use smallvec::{ExtendFromSlice, SmallVec};
///
/// fn initialize<V: ExtendFromSlice<u8>>(v: &mut V) {
///     v.extend_from_slice(b"Test!");
/// }
///
/// let mut vec = Vec::new();
/// initialize(&mut vec);
/// assert_eq!(&vec, b"Test!");
///
/// let mut small_vec = SmallVec::<[u8; 8]>::new();
/// initialize(&mut small_vec);
/// assert_eq!(&small_vec as &[_], b"Test!");
/// ```
pub trait ExtendFromSlice<T> {
    /// Extends a collection from a slice of its element type
    fn extend_from_slice(&mut self, other: &[T]);
}

impl<T: Clone> ExtendFromSlice<T> for Vec<T> {
    fn extend_from_slice(&mut self, other: &[T]) {
        Vec::extend_from_slice(self, other)
    }
}

unsafe fn deallocate<T>(ptr: *mut T, capacity: usize) {
    let _vec: Vec<T> = Vec::from_raw_parts(ptr, 0, capacity);
    // Let it drop.
}

/// An iterator that removes the items from a `SmallVec` and yields them by value.
///
/// Returned from [`SmallVec::drain`][1].
///
/// [1]: struct.SmallVec.html#method.drain
pub struct Drain<'a, T: 'a> {
    iter: slice::IterMut<'a, T>,
}

impl<'a, T: 'a> Iterator for Drain<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        match self.iter.next() {
            None => None,
            Some(reference) => unsafe { Some(ptr::read(reference)) },
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
            Some(reference) => unsafe { Some(ptr::read(reference)) },
        }
    }
}

impl<'a, T> ExactSizeIterator for Drain<'a, T> {}

impl<'a, T: 'a> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        // Destroy the remaining elements.
        for _ in self.by_ref() {}
    }
}

#[cfg(feature = "union")]
#[allow(unions_with_drop_fields)]
union SmallVecData<A: Array> {
    inline: A,
    heap: (*mut A::Item, usize),
}

#[cfg(feature = "union")]
impl<A: Array> SmallVecData<A> {
    #[inline]
    unsafe fn inline(&self) -> &A {
        &self.inline
    }
    #[inline]
    unsafe fn inline_mut(&mut self) -> &mut A {
        &mut self.inline
    }
    #[inline]
    fn from_inline(inline: A) -> SmallVecData<A> {
        SmallVecData { inline }
    }
    #[inline]
    unsafe fn heap(&self) -> (*mut A::Item, usize) {
        self.heap
    }
    #[inline]
    unsafe fn heap_mut(&mut self) -> &mut (*mut A::Item, usize) {
        &mut self.heap
    }
    #[inline]
    fn from_heap(ptr: *mut A::Item, len: usize) -> SmallVecData<A> {
        SmallVecData { heap: (ptr, len) }
    }
}

#[cfg(not(feature = "union"))]
enum SmallVecData<A: Array> {
    Inline(ManuallyDrop<A>),
    Heap((*mut A::Item, usize)),
}

#[cfg(not(feature = "union"))]
impl<A: Array> SmallVecData<A> {
    #[inline]
    unsafe fn inline(&self) -> &A {
        match *self {
            SmallVecData::Inline(ref a) => a,
            _ => unreachable!(),
        }
    }
    #[inline]
    unsafe fn inline_mut(&mut self) -> &mut A {
        match *self {
            SmallVecData::Inline(ref mut a) => a,
            _ => unreachable!(),
        }
    }
    #[inline]
    fn from_inline(inline: A) -> SmallVecData<A> {
        SmallVecData::Inline(ManuallyDrop::new(inline))
    }
    #[inline]
    unsafe fn heap(&self) -> (*mut A::Item, usize) {
        match *self {
            SmallVecData::Heap(data) => data,
            _ => unreachable!(),
        }
    }
    #[inline]
    unsafe fn heap_mut(&mut self) -> &mut (*mut A::Item, usize) {
        match *self {
            SmallVecData::Heap(ref mut data) => data,
            _ => unreachable!(),
        }
    }
    #[inline]
    fn from_heap(ptr: *mut A::Item, len: usize) -> SmallVecData<A> {
        SmallVecData::Heap((ptr, len))
    }
}

unsafe impl<A: Array + Send> Send for SmallVecData<A> {}
unsafe impl<A: Array + Sync> Sync for SmallVecData<A> {}

/// A `Vec`-like container that can store a small number of elements inline.
///
/// `SmallVec` acts like a vector, but can store a limited amount of data inline within the
/// `Smallvec` struct rather than in a separate allocation.  If the data exceeds this limit, the
/// `SmallVec` will "spill" its data onto the heap, allocating a new buffer to hold it.
///
/// The amount of data that a `SmallVec` can store inline depends on its backing store. The backing
/// store can be any type that implements the `Array` trait; usually it is a small fixed-sized
/// array.  For example a `SmallVec<[u64; 8]>` can hold up to eight 64-bit integers inline.
///
/// ## Example
///
/// ```rust
/// use smallvec::SmallVec;
/// let mut v = SmallVec::<[u8; 4]>::new(); // initialize an empty vector
///
/// // The vector can hold up to 4 items without spilling onto the heap.
/// v.extend(0..4);
/// assert_eq!(v.len(), 4);
/// assert!(!v.spilled());
///
/// // Pushing another element will force the buffer to spill:
/// v.push(4);
/// assert_eq!(v.len(), 5);
/// assert!(v.spilled());
/// ```
pub struct SmallVec<A: Array> {
    capacity: usize,
    data: SmallVecData<A>,
}

impl<A: Array> SmallVec<A> {
    /// Construct an empty vector
    #[inline]
    pub fn new() -> SmallVec<A> {
        unsafe {
            SmallVec {
                capacity: 0,
                data: SmallVecData::from_inline(mem::uninitialized()),
            }
        }
    }

    /// Construct an empty vector with enough capacity pre-allocated to store at least `n`
    /// elements.
    ///
    /// Will create a heap allocation only if `n` is larger than the inline capacity.
    ///
    /// ```
    /// # use smallvec::SmallVec;
    ///
    /// let v: SmallVec<[u8; 3]> = SmallVec::with_capacity(100);
    ///
    /// assert!(v.is_empty());
    /// assert!(v.capacity() >= 100);
    /// ```
    #[inline]
    pub fn with_capacity(n: usize) -> Self {
        let mut v = SmallVec::new();
        v.reserve_exact(n);
        v
    }

    /// Construct a new `SmallVec` from a `Vec<A::Item>` without copying
    /// elements.
    ///
    /// ```rust
    /// use smallvec::SmallVec;
    ///
    /// let vec = vec![1, 2, 3, 4, 5];
    /// let small_vec: SmallVec<[_; 3]> = SmallVec::from_vec(vec);
    ///
    /// assert_eq!(&*small_vec, &[1, 2, 3, 4, 5]);
    /// ```
    #[inline]
    pub fn from_vec(mut vec: Vec<A::Item>) -> SmallVec<A> {
        if vec.capacity() <= A::size() {
            unsafe {
                let mut data = SmallVecData::<A>::from_inline(mem::uninitialized());
                let len = vec.len();
                vec.set_len(0);
                ptr::copy_nonoverlapping(vec.as_ptr(), data.inline_mut().ptr_mut(), len);

                SmallVec {
                    capacity: len,
                    data,
                }
            }
        } else {
            let (ptr, cap, len) = (vec.as_mut_ptr(), vec.capacity(), vec.len());
            mem::forget(vec);

            SmallVec {
                capacity: cap,
                data: SmallVecData::from_heap(ptr, len),
            }
        }
    }

    /// Constructs a new `SmallVec` on the stack from an `A` without
    /// copying elements.
    ///
    /// ```rust
    /// use smallvec::SmallVec;
    ///
    /// let buf = [1, 2, 3, 4, 5];
    /// let small_vec: SmallVec<_> = SmallVec::from_buf(buf);
    ///
    /// assert_eq!(&*small_vec, &[1, 2, 3, 4, 5]);
    /// ```
    #[inline]
    pub fn from_buf(buf: A) -> SmallVec<A> {
        SmallVec {
            capacity: A::size(),
            data: SmallVecData::from_inline(buf),
        }
    }

    /// Sets the length of a vector.
    ///
    /// This will explicitly set the size of the vector, without actually
    /// modifying its buffers, so it is up to the caller to ensure that the
    /// vector is actually the specified size.
    pub unsafe fn set_len(&mut self, new_len: usize) {
        if self.spilled() {
            self.data.heap_mut().1 = new_len;
        } else {
            self.capacity = new_len;
        }
    }

    /// The maximum number of elements this vector can hold inline
    #[inline]
    pub fn inline_size(&self) -> usize {
        A::size()
    }

    /// The number of elements stored in the vector
    #[inline]
    pub fn len(&self) -> usize {
        if self.spilled() {
            unsafe { self.data.heap().1 }
        } else {
            self.capacity
        }
    }

    /// Returns `true` if the vector is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of items the vector can hold without reallocating
    #[inline]
    pub fn capacity(&self) -> usize {
        if self.spilled() {
            self.capacity
        } else {
            A::size()
        }
    }

    /// Returns `true` if the data has spilled into a separate heap-allocated buffer.
    #[inline]
    pub fn spilled(&self) -> bool {
        self.capacity > A::size()
    }

    /// Empty the vector and return an iterator over its former contents.
    pub fn drain(&mut self) -> Drain<A::Item> {
        unsafe {
            let ptr = self.as_mut_ptr();

            let current_len = self.len();
            self.set_len(0);

            let slice = slice::from_raw_parts_mut(ptr, current_len);

            Drain {
                iter: slice.iter_mut(),
            }
        }
    }

    /// Append an item to the vector.
    #[inline]
    pub fn push(&mut self, value: A::Item) {
        let cap = self.capacity();
        if self.len() == cap {
            self.grow(cmp::max(cap * 2, 1))
        }
        unsafe {
            let end = self.as_mut_ptr().offset(self.len() as isize);
            ptr::write(end, value);
            let len = self.len();
            self.set_len(len + 1)
        }
    }

    /// Remove an item from the end of the vector and return it, or None if empty.
    #[inline]
    pub fn pop(&mut self) -> Option<A::Item> {
        if self.len() == 0 {
            return None;
        }
        let last_index = self.len() - 1;
        if (last_index as isize) < 0 {
            panic!("overflow")
        }
        unsafe {
            let end_ptr = self.as_ptr().offset(last_index as isize);
            let value = ptr::read(end_ptr);
            self.set_len(last_index);
            Some(value)
        }
    }

    /// Re-allocate to set the capacity to `max(new_cap, inline_size())`.
    ///
    /// Panics if `new_cap` is less than the vector's length.
    pub fn grow(&mut self, new_cap: usize) {
        assert!(new_cap >= self.len());
        unsafe {
            if new_cap <= self.inline_size() {
                if !self.spilled() {
                    return;
                }
                let (ptr, len) = self.data.heap();
                let mut vec = Vec::from_raw_parts(ptr, len, self.capacity);
                self.data = SmallVecData::from_inline(mem::uninitialized());
                ptr::copy_nonoverlapping(vec.as_ptr(), self.as_mut_ptr(), len);
                vec.set_len(0);
            } else if new_cap != self.capacity {
                let mut vec = Vec::with_capacity(new_cap);
                let new_alloc = vec.as_mut_ptr();
                mem::forget(vec);
                let len = self.len();
                ptr::copy_nonoverlapping(self.as_ptr(), new_alloc, len);
                if self.spilled() {
                    deallocate(self.data.heap().0, self.capacity);
                }
                self.data = SmallVecData::from_heap(new_alloc, len);
                self.capacity = new_cap;
            }
        }
    }

    /// Reserve capacity for `additional` more elements to be inserted.
    ///
    /// May reserve more space to avoid frequent reallocations.
    ///
    /// If the new capacity would overflow `usize` then it will be set to `usize::max_value()`
    /// instead. (This means that inserting `additional` new elements is not guaranteed to be
    /// possible after calling this function.)
    pub fn reserve(&mut self, additional: usize) {
        let len = self.len();
        if self.capacity() - len < additional {
            match len.checked_add(additional)
                .and_then(usize::checked_next_power_of_two)
            {
                Some(cap) => self.grow(cap),
                None => self.grow(usize::max_value()),
            }
        }
    }

    /// Reserve the minumum capacity for `additional` more elements to be inserted.
    ///
    /// Panics if the new capacity overflows `usize`.
    pub fn reserve_exact(&mut self, additional: usize) {
        let len = self.len();
        if self.capacity() - len < additional {
            match len.checked_add(additional) {
                Some(cap) => self.grow(cap),
                None => panic!("reserve_exact overflow"),
            }
        }
    }

    /// Shrink the capacity of the vector as much as possible.
    ///
    /// When possible, this will move data from an external heap buffer to the vector's inline
    /// storage.
    pub fn shrink_to_fit(&mut self) {
        if !self.spilled() {
            return;
        }
        let len = self.len();
        if self.inline_size() >= len {
            unsafe {
                let (ptr, len) = self.data.heap();
                self.data = SmallVecData::from_inline(mem::uninitialized());
                ptr::copy_nonoverlapping(ptr, self.data.inline_mut().ptr_mut(), len);
                deallocate(ptr, self.capacity);
                self.capacity = len;
            }
        } else if self.capacity() > len {
            self.grow(len);
        }
    }

    /// Shorten the vector, keeping the first `len` elements and dropping the rest.
    ///
    /// If `len` is greater than or equal to the vector's current length, this has no
    /// effect.
    ///
    /// This does not re-allocate.  If you want the vector's capacity to shrink, call
    /// `shrink_to_fit` after truncating.
    pub fn truncate(&mut self, len: usize) {
        let end_ptr = self.as_ptr();
        while len < self.len() {
            unsafe {
                let last_index = self.len() - 1;
                self.set_len(last_index);
                ptr::read(end_ptr.offset(last_index as isize));
            }
        }
    }

    /// Extracts a slice containing the entire vector.
    ///
    /// Equivalent to `&s[..]`.
    pub fn as_slice(&self) -> &[A::Item] {
        self
    }

    /// Extracts a mutable slice of the entire vector.
    ///
    /// Equivalent to `&mut s[..]`.
    pub fn as_mut_slice(&mut self) -> &mut [A::Item] {
        self
    }

    /// Remove the element at position `index`, replacing it with the last element.
    ///
    /// This does not preserve ordering, but is O(1).
    ///
    /// Panics if `index` is out of bounds.
    #[inline]
    pub fn swap_remove(&mut self, index: usize) -> A::Item {
        let len = self.len();
        self.swap(len - 1, index);
        self.pop().unwrap()
    }

    /// Remove all elements from the vector.
    #[inline]
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Remove and return the element at position `index`, shifting all elements after it to the
    /// left.
    ///
    /// Panics if `index` is out of bounds.
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

    /// Insert an element at position `index`, shifting all elements after it to the right.
    ///
    /// Panics if `index` is out of bounds.
    pub fn insert(&mut self, index: usize, element: A::Item) {
        self.reserve(1);

        let len = self.len();
        assert!(index <= len);

        unsafe {
            let ptr = self.as_mut_ptr().offset(index as isize);
            ptr::copy(ptr, ptr.offset(1), len - index);
            ptr::write(ptr, element);
            self.set_len(len + 1);
        }
    }

    /// Insert multiple elements at position `index`, shifting all following elements toward the
    /// back.
    pub fn insert_many<I: IntoIterator<Item = A::Item>>(&mut self, index: usize, iterable: I) {
        let iter = iterable.into_iter();
        let (lower_size_bound, _) = iter.size_hint();
        assert!(lower_size_bound <= std::isize::MAX as usize); // Ensure offset is indexable
        assert!(index + lower_size_bound >= index); // Protect against overflow
        self.reserve(lower_size_bound);

        unsafe {
            let old_len = self.len();
            assert!(index <= old_len);
            let ptr = self.as_mut_ptr().offset(index as isize);
            ptr::copy(ptr, ptr.offset(lower_size_bound as isize), old_len - index);
            for (off, element) in iter.enumerate() {
                if off < lower_size_bound {
                    ptr::write(ptr.offset(off as isize), element);
                    let len = self.len() + 1;
                    self.set_len(len);
                } else {
                    // Iterator provided more elements than the hint.
                    assert!(index + off >= index); // Protect against overflow.
                    self.insert(index + off, element);
                }
            }
            let num_added = self.len() - old_len;
            if num_added < lower_size_bound {
                // Iterator provided fewer elements than the hint
                ptr::copy(
                    ptr.offset(lower_size_bound as isize),
                    ptr.offset(num_added as isize),
                    old_len - index,
                );
            }
        }
    }

    /// Convert a SmallVec to a Vec, without reallocating if the SmallVec has already spilled onto
    /// the heap.
    pub fn into_vec(self) -> Vec<A::Item> {
        if self.spilled() {
            unsafe {
                let (ptr, len) = self.data.heap();
                let v = Vec::from_raw_parts(ptr, len, self.capacity);
                mem::forget(self);
                v
            }
        } else {
            self.into_iter().collect()
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` such that `f(&e)` returns `false`.
    /// This method operates in place and preserves the order of the retained
    /// elements.
    pub fn retain<F: FnMut(&mut A::Item) -> bool>(&mut self, mut f: F) {
        let mut del = 0;
        let len = self.len();
        for i in 0..len {
            if !f(&mut self[i]) {
                del += 1;
            } else if del > 0 {
                self.swap(i - del, i);
            }
        }
        self.truncate(len - del);
    }

    /// Removes consecutive duplicate elements.
    pub fn dedup(&mut self)
    where
        A::Item: PartialEq<A::Item>,
    {
        self.dedup_by(|a, b| a == b);
    }

    /// Removes consecutive duplicate elements using the given equality relation.
    pub fn dedup_by<F>(&mut self, mut same_bucket: F)
    where
        F: FnMut(&mut A::Item, &mut A::Item) -> bool,
    {
        // See the implementation of Vec::dedup_by in the
        // standard library for an explanation of this algorithm.
        let len = self.len();
        if len <= 1 {
            return;
        }

        let ptr = self.as_mut_ptr();
        let mut w: usize = 1;

        unsafe {
            for r in 1..len {
                let p_r = ptr.offset(r as isize);
                let p_wm1 = ptr.offset((w - 1) as isize);
                if !same_bucket(&mut *p_r, &mut *p_wm1) {
                    if r != w {
                        let p_w = p_wm1.offset(1);
                        mem::swap(&mut *p_r, &mut *p_w);
                    }
                    w += 1;
                }
            }
        }

        self.truncate(w);
    }

    /// Removes consecutive elements that map to the same key.
    pub fn dedup_by_key<F, K>(&mut self, mut key: F)
    where
        F: FnMut(&mut A::Item) -> K,
        K: PartialEq<K>,
    {
        self.dedup_by(|a, b| key(a) == key(b));
    }
}

impl<A: Array> SmallVec<A>
where
    A::Item: Copy,
{
    /// Copy the elements from a slice into a new `SmallVec`.
    ///
    /// For slices of `Copy` types, this is more efficient than `SmallVec::from(slice)`.
    pub fn from_slice(slice: &[A::Item]) -> Self {
        let mut vec = Self::new();
        vec.extend_from_slice(slice);
        vec
    }

    /// Copy elements from a slice into the vector at position `index`, shifting any following
    /// elements toward the back.
    ///
    /// For slices of `Copy` types, this is more efficient than `insert`.
    pub fn insert_from_slice(&mut self, index: usize, slice: &[A::Item]) {
        self.reserve(slice.len());

        let len = self.len();
        assert!(index <= len);

        unsafe {
            let slice_ptr = slice.as_ptr();
            let ptr = self.as_mut_ptr().offset(index as isize);
            ptr::copy(ptr, ptr.offset(slice.len() as isize), len - index);
            ptr::copy_nonoverlapping(slice_ptr, ptr, slice.len());
            self.set_len(len + slice.len());
        }
    }

    /// Copy elements from a slice and append them to the vector.
    ///
    /// For slices of `Copy` types, this is more efficient than `extend`.
    #[inline]
    pub fn extend_from_slice(&mut self, slice: &[A::Item]) {
        let len = self.len();
        self.insert_from_slice(len, slice);
    }
}

impl<A: Array> SmallVec<A>
where
    A::Item: Clone,
{
    /// Resizes the vector so that its length is equal to `len`.
    ///
    /// If `len` is less than the current length, the vector simply truncated.
    ///
    /// If `len` is greater than the current length, `value` is appended to the
    /// vector until its length equals `len`.
    pub fn resize(&mut self, len: usize, value: A::Item) {
        let old_len = self.len();

        if len > old_len {
            self.extend(repeat(value).take(len - old_len));
        } else {
            self.truncate(len);
        }
    }

    /// Creates a `SmallVec` with `n` copies of `elem`.
    ///
    /// ```
    /// use smallvec::SmallVec;
    ///
    /// let v = SmallVec::<[char; 128]>::from_elem('d', 2);
    /// assert_eq!(v, SmallVec::from_buf(['d', 'd']));
    /// ```
    pub fn from_elem(elem: A::Item, n: usize) -> Self {
        let mut v = SmallVec::with_capacity(n);
        v.insert_many(0, (0..n).map(|_| elem.clone()));
        v
    }
}

impl<A: Array> ops::Deref for SmallVec<A> {
    type Target = [A::Item];
    #[inline]
    fn deref(&self) -> &[A::Item] {
        unsafe {
            let ptr = if self.spilled() {
                self.data.heap().0
            } else {
                self.data.inline().ptr()
            };
            slice::from_raw_parts(ptr, self.len())
        }
    }
}

impl<A: Array> ops::DerefMut for SmallVec<A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [A::Item] {
        unsafe {
            let ptr = if self.spilled() {
                self.data.heap().0
            } else {
                self.data.inline_mut().ptr_mut()
            };
            slice::from_raw_parts_mut(ptr, self.len())
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

#[cfg(feature = "std")]
impl<A: Array<Item = u8>> io::Write for SmallVec<A> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.extend_from_slice(buf);
        Ok(buf.len())
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.extend_from_slice(buf);
        Ok(())
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(feature = "serde")]
impl<A: Array> Serialize for SmallVec<A>
where
    A::Item: Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_seq(Some(self.len()))?;
        for item in self {
            state.serialize_element(&item)?;
        }
        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, A: Array> Deserialize<'de> for SmallVec<A>
where
    A::Item: Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_seq(SmallVecVisitor {
            phantom: PhantomData,
        })
    }
}

#[cfg(feature = "serde")]
struct SmallVecVisitor<A> {
    phantom: PhantomData<A>,
}

#[cfg(feature = "serde")]
impl<'de, A: Array> Visitor<'de> for SmallVecVisitor<A>
where
    A::Item: Deserialize<'de>,
{
    type Value = SmallVec<A>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a sequence")
    }

    fn visit_seq<B>(self, mut seq: B) -> Result<Self::Value, B::Error>
    where
        B: SeqAccess<'de>,
    {
        let len = seq.size_hint().unwrap_or(0);
        let mut values = SmallVec::with_capacity(len);

        while let Some(value) = seq.next_element()? {
            values.push(value);
        }

        Ok(values)
    }
}

impl<'a, A: Array> From<&'a [A::Item]> for SmallVec<A>
where
    A::Item: Clone,
{
    #[inline]
    fn from(slice: &'a [A::Item]) -> SmallVec<A> {
        slice.into_iter().cloned().collect()
    }
}

impl<A: Array> From<Vec<A::Item>> for SmallVec<A> {
    #[inline]
    fn from(vec: Vec<A::Item>) -> SmallVec<A> {
        SmallVec::from_vec(vec)
    }
}

impl<A: Array> From<A> for SmallVec<A> {
    #[inline]
    fn from(array: A) -> SmallVec<A> {
        SmallVec::from_buf(array)
    }
}

macro_rules! impl_index {
    ($index_type:ty, $output_type:ty) => {
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
    };
}

impl_index!(usize, A::Item);
impl_index!(ops::Range<usize>, [A::Item]);
impl_index!(ops::RangeFrom<usize>, [A::Item]);
impl_index!(ops::RangeTo<usize>, [A::Item]);
impl_index!(ops::RangeFull, [A::Item]);

impl<A: Array> ExtendFromSlice<A::Item> for SmallVec<A>
where
    A::Item: Copy,
{
    fn extend_from_slice(&mut self, other: &[A::Item]) {
        SmallVec::extend_from_slice(self, other)
    }
}

#[allow(deprecated)]
impl<A: Array> VecLike<A::Item> for SmallVec<A> {
    #[inline]
    fn push(&mut self, value: A::Item) {
        SmallVec::push(self, value);
    }
}

impl<A: Array> FromIterator<A::Item> for SmallVec<A> {
    fn from_iter<I: IntoIterator<Item = A::Item>>(iterable: I) -> SmallVec<A> {
        let mut v = SmallVec::new();
        v.extend(iterable);
        v
    }
}

impl<A: Array> Extend<A::Item> for SmallVec<A> {
    fn extend<I: IntoIterator<Item = A::Item>>(&mut self, iterable: I) {
        let iter = iterable.into_iter();
        let (lower_size_bound, _) = iter.size_hint();

        let target_len = self.len() + lower_size_bound;

        if target_len > self.capacity() {
            self.grow(target_len);
        }

        for elem in iter {
            self.push(elem);
        }
    }
}

impl<A: Array> fmt::Debug for SmallVec<A>
where
    A::Item: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", &**self)
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
        unsafe {
            if self.spilled() {
                let (ptr, len) = self.data.heap();
                Vec::from_raw_parts(ptr, len, self.capacity);
            } else {
                let ptr = self.as_ptr();
                for i in 0..self.len() {
                    ptr::read(ptr.offset(i as isize));
                }
            }
        }
    }
}

impl<A: Array> Clone for SmallVec<A>
where
    A::Item: Clone,
{
    fn clone(&self) -> SmallVec<A> {
        let mut new_vector = SmallVec::with_capacity(self.len());
        for element in self.iter() {
            new_vector.push((*element).clone())
        }
        new_vector
    }
}

impl<A: Array, B: Array> PartialEq<SmallVec<B>> for SmallVec<A>
where
    A::Item: PartialEq<B::Item>,
{
    #[inline]
    fn eq(&self, other: &SmallVec<B>) -> bool {
        self[..] == other[..]
    }
    #[inline]
    fn ne(&self, other: &SmallVec<B>) -> bool {
        self[..] != other[..]
    }
}

impl<A: Array> Eq for SmallVec<A>
where
    A::Item: Eq,
{
}

impl<A: Array> PartialOrd for SmallVec<A>
where
    A::Item: PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, other: &SmallVec<A>) -> Option<cmp::Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

impl<A: Array> Ord for SmallVec<A>
where
    A::Item: Ord,
{
    #[inline]
    fn cmp(&self, other: &SmallVec<A>) -> cmp::Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<A: Array> Hash for SmallVec<A>
where
    A::Item: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

unsafe impl<A: Array> Send for SmallVec<A>
where
    A::Item: Send,
{
}

/// An iterator that consumes a `SmallVec` and yields its items by value.
///
/// Returned from [`SmallVec::into_iter`][1].
///
/// [1]: struct.SmallVec.html#method.into_iter
pub struct IntoIter<A: Array> {
    data: SmallVec<A>,
    current: usize,
    end: usize,
}

impl<A: Array> Drop for IntoIter<A> {
    fn drop(&mut self) {
        for _ in self {}
    }
}

impl<A: Array> Iterator for IntoIter<A> {
    type Item = A::Item;

    #[inline]
    fn next(&mut self) -> Option<A::Item> {
        if self.current == self.end {
            None
        } else {
            unsafe {
                let current = self.current as isize;
                self.current += 1;
                Some(ptr::read(self.data.as_ptr().offset(current)))
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
        } else {
            unsafe {
                self.end -= 1;
                Some(ptr::read(self.data.as_ptr().offset(self.end as isize)))
            }
        }
    }
}

impl<A: Array> ExactSizeIterator for IntoIter<A> {}

impl<A: Array> IntoIterator for SmallVec<A> {
    type IntoIter = IntoIter<A>;
    type Item = A::Item;
    fn into_iter(mut self) -> Self::IntoIter {
        unsafe {
            let len = self.len();
            self.set_len(0);
            IntoIter {
                data: self,
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

/// Types that can be used as the backing store for a SmallVec
pub unsafe trait Array {
    /// The type of the array's elements.
    type Item;
    /// Returns the number of items the array can hold.
    fn size() -> usize;
    /// Returns a pointer to the first element of the array.
    fn ptr(&self) -> *const Self::Item;
    /// Returns a mutable pointer to the first element of the array.
    fn ptr_mut(&mut self) -> *mut Self::Item;
}

macro_rules! impl_array(
    ($($size:expr),+) => {
        $(
            unsafe impl<T> Array for [T; $size] {
                type Item = T;
                fn size() -> usize { $size }
                fn ptr(&self) -> *const T { self.as_ptr() }
                fn ptr_mut(&mut self) -> *mut T { self.as_mut_ptr() }
            }
        )+
    }
);

impl_array!(
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 24, 32, 36, 0x40, 0x80, 0x100,
    0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000, 0x10000, 0x20000, 0x40000, 0x80000,
    0x100000
);

#[cfg(test)]
mod tests {
    use SmallVec;

    use std::iter::FromIterator;

    #[cfg(not(feature = "std"))]
    use alloc::borrow::ToOwned;
    #[cfg(not(feature = "std"))]
    use alloc::boxed::Box;
    #[cfg(not(feature = "std"))]
    use alloc::rc::Rc;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::borrow::ToOwned;
    #[cfg(feature = "std")]
    use std::rc::Rc;

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
        v.push("hello".to_owned());
        v.push("there".to_owned());
        assert_eq!(&*v, &["hello".to_owned(), "there".to_owned()][..]);
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
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hash;

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
    #[allow(deprecated)]
    fn veclike_deref_slice() {
        use super::VecLike;

        fn test<T: VecLike<i32>>(vec: &mut T) {
            assert!(!vec.is_empty());
            assert_eq!(vec.len(), 3);

            vec.sort();
            assert_eq!(&vec[..], [1, 2, 3]);
        }

        let mut vec = SmallVec::<[i32; 2]>::from(&[3, 1, 2][..]);
        test(&mut vec);
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
        use io::Write;

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
        use self::bincode::{deserialize, serialize, Bounded};
        let mut small_vec: SmallVec<[i32; 2]> = SmallVec::new();
        small_vec.push(1);
        let encoded = serialize(&small_vec, Bounded(100)).unwrap();
        let decoded: SmallVec<[i32; 2]> = deserialize(&encoded).unwrap();
        assert_eq!(small_vec, decoded);
        small_vec.push(2);
        // Spill the vec
        small_vec.push(3);
        small_vec.push(4);
        // Check again after spilling.
        let encoded = serialize(&small_vec, Bounded(100)).unwrap();
        let decoded: SmallVec<[i32; 2]> = deserialize(&encoded).unwrap();
        assert_eq!(small_vec, decoded);
    }
}
