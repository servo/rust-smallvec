// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Small vectors in various sizes. These store a certain number of elements inline, and fall back
//! to the heap for larger allocations.  This can be a useful optimization for improving cache
//! locality and reducing allocator traffic for workloads that fit within the inline buffer.
//!
//! ## `no_std` support
//!
//! By default, `smallvec` does not depend on `std`.  However, the optional
//! `write` feature implements the `std::io::Write` trait for vectors of `u8`.
//! When this feature is enabled, `smallvec` depends on `std`.
//!
//! ## Optional features
//!
//! ### `serde`
//!
//! When this optional dependency is enabled, `SmallVec` implements the `serde::Serialize` and
//! `serde::Deserialize` traits.
//!
//! ### `write`
//!
//! When this feature is enabled, `SmallVec<[u8; _]>` implements the `std::io::Write` trait.
//! This feature is not compatible with `#![no_std]` programs.
//!
//! ### `specialization`
//!
//! **This feature is unstable and requires a nightly build of the Rust toolchain.**
//!
//! When this feature is enabled, `SmallVec::from(slice)` has improved performance for slices
//! of `Copy` types.  (Without this feature, you can use `SmallVec::from_slice` to get optimal
//! performance for `Copy` types.)
//!
//! Tracking issue: [rust-lang/rust#31844](https://github.com/rust-lang/rust/issues/31844)
//!
//! ### `may_dangle`
//!
//! **This feature is unstable and requires a nightly build of the Rust toolchain.**
//!
//! This feature makes the Rust compiler less strict about use of vectors that contain borrowed
//! references. For details, see the
//! [Rustonomicon](https://doc.rust-lang.org/1.42.0/nomicon/dropck.html#an-escape-hatch).
//!
//! Tracking issue: [rust-lang/rust#34761](https://github.com/rust-lang/rust/issues/34761)

#![no_std]

#[doc(hidden)]
pub extern crate alloc;

// #[cfg(any(test, feature = "write"))]
extern crate std;

#[cfg(test)]
mod tests;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use alloc::alloc::Layout;
use core::borrow::Borrow;
use core::borrow::BorrowMut;
use core::fmt::Debug;
use core::mem::align_of;
use core::mem::size_of;
use core::mem::ManuallyDrop;
use core::mem::MaybeUninit;
use core::ptr::addr_of;
use core::ptr::addr_of_mut;
use core::ptr::copy_nonoverlapping;

#[cfg(feature = "serde")]
use core::marker::PhantomData;
#[cfg(feature = "serde")]
use serde::{
    de::{Deserialize, Deserializer, SeqAccess, Visitor},
    ser::{Serialize, SerializeSeq, Serializer},
};
#[cfg(feature = "write")]
use std::io;

/// Error type for APIs with fallible heap allocation
#[derive(Debug)]
pub enum CollectionAllocErr {
    /// Overflow `usize::MAX` or other error during size computation
    CapacityOverflow,
    /// The allocator return an error
    AllocErr {
        /// The layout that was passed to the allocator
        layout: Layout,
    },
}
impl core::fmt::Display for CollectionAllocErr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Allocation error: {:?}", self)
    }
}

/// Either a stack array with `length <= N` or a heap array
/// whose pointer and capacity are stored here.
///
/// We store a `*const T` instead of a `*mut T` so that the type is covariant
/// with respect to `T`.
#[repr(C)]
pub union RawSmallVec<T, const N: usize> {
    inline: ManuallyDrop<MaybeUninit<[T; N]>>,
    heap: (*const T, usize),
}

#[inline]
fn infallible<T>(result: Result<T, CollectionAllocErr>) -> T {
    match result {
        Ok(x) => x,
        Err(CollectionAllocErr::CapacityOverflow) => panic!("capacity overflow"),
        Err(CollectionAllocErr::AllocErr { layout }) => alloc::alloc::handle_alloc_error(layout),
    }
}

impl<T, const N: usize> RawSmallVec<T, N> {
    #[inline]
    const fn is_zst() -> bool {
        size_of::<T>() == 0
    }

    #[inline]
    const fn new() -> Self {
        Self::new_inline(MaybeUninit::uninit())
    }
    #[inline]
    const fn new_inline(inline: MaybeUninit<[T; N]>) -> Self {
        Self {
            inline: ManuallyDrop::new(inline),
        }
    }
    #[inline]
    const fn new_heap(ptr: *mut T, capacity: usize) -> Self {
        Self {
            heap: (ptr, capacity),
        }
    }

    #[inline]
    const fn as_ptr_inline(&self) -> *const T {
        // SAFETY: This is safe because we don't read the value. We only get a pointer to the data.
        // Dereferencing the pointer is unsafe so unsafe code is required to misuse the return
        // value.
        (unsafe { addr_of!(self.inline) }) as *const T
    }

    #[inline]
    fn as_mut_ptr_inline(&mut self) -> *mut T {
        // SAFETY: See above.
        (unsafe { addr_of_mut!(self.inline) }) as *mut T
    }

    #[inline]
    const unsafe fn as_ptr_heap(&self) -> *const T {
        self.heap.0
    }

    #[inline]
    unsafe fn as_mut_ptr_heap(&mut self) -> *mut T {
        self.heap.0 as *mut T
    }

    unsafe fn try_grow(
        &mut self,
        len: TaggedLen,
        new_capacity: usize,
    ) -> Result<(), CollectionAllocErr> {
        use alloc::alloc::{alloc, realloc};
        debug_assert!(new_capacity > 0);

        if Self::is_zst() {
            debug_assert_eq!(len.value(Self::is_zst()), usize::MAX);
            Err(CollectionAllocErr::CapacityOverflow)
        } else {
            let was_on_heap = len.on_heap(Self::is_zst());
            let ptr = if was_on_heap {
                self.as_mut_ptr_heap()
            } else {
                self.as_mut_ptr_inline()
            };
            let len = len.value(Self::is_zst());

            let new_layout = Layout::array::<T>(new_capacity)
                .map_err(|_| CollectionAllocErr::CapacityOverflow)?;
            if new_layout.size() > isize::MAX as usize {
                return Err(CollectionAllocErr::CapacityOverflow);
            }

            if len == 0 || !was_on_heap {
                // get a fresh allocation

                // layout has non zero size
                let new_ptr = alloc(new_layout) as *mut T;
                if new_ptr.is_null() {
                    Err(CollectionAllocErr::AllocErr { layout: new_layout })
                } else {
                    copy_nonoverlapping(ptr, new_ptr, len);
                    *self = Self::new_heap(new_ptr, new_capacity);
                    Ok(())
                }
            } else {
                // use realloc

                // this can't overflow since we already constructed an equivalent layout during
                // the previous allocation
                let old_layout = Layout::from_size_align_unchecked(
                    self.heap.1 * size_of::<T>(),
                    align_of::<T>(),
                );

                // SAFETY: ptr was allocated with this allocator
                // old_layout is the same as the layout used to allocate the previous memory block
                // new_layout.size() is greater than zero
                // does not overflow when rounded up to alignment. since it was constructed
                // with Layout::array
                let new_ptr = realloc(ptr as *mut u8, old_layout, new_layout.size()) as *mut T;
                if new_ptr.is_null() {
                    Err(CollectionAllocErr::AllocErr { layout: new_layout })
                } else {
                    *self = Self::new_heap(new_ptr, new_capacity);
                    Ok(())
                }
            }
        }
    }
}

/// Vec guarantees that its length is always less than isize::MAX in *bytes*.
///
/// For a non ZST, this means that the length is less than isize::MAX objects, which implies we
/// have at least one free bit we can use. We use the least significant bit for the tag. And store
/// the length in the `usize::BITS - 1` most significant bits.
///
/// For a ZST, we never use the heap, so we just store the length directly.
#[repr(transparent)]
#[derive(Clone, Copy)]
struct TaggedLen(usize);

impl TaggedLen {
    #[inline]
    pub const fn new(len: usize, on_heap: bool, is_zst: bool) -> Self {
        if is_zst {
            debug_assert!(!on_heap);
            TaggedLen(len)
        } else {
            debug_assert!(len < isize::MAX as usize);
            TaggedLen((len << 1) | on_heap as usize)
        }
    }

    #[inline]
    #[must_use]
    pub const fn on_heap(self, is_zst: bool) -> bool {
        if is_zst {
            false
        } else {
            (self.0 & 1_usize) == 1
        }
    }

    #[inline]
    pub const fn value(self, is_zst: bool) -> usize {
        if is_zst {
            self.0
        } else {
            self.0 >> 1
        }
    }
}

#[repr(C)]
pub struct SmallVec<T, const N: usize> {
    len: TaggedLen,
    raw: RawSmallVec<T, N>,
}

/// An iterator that removes the items from a `SmallVec` and yields them by value.
///
/// Returned from [`SmallVec::drain`][1].
///
/// [1]: struct.SmallVec.html#method.drain
pub struct Drain<'a, T: 'a, const N: usize> {
    tail_start: usize,
    tail_len: usize,
    iter: core::slice::Iter<'a, T>,
    vec: core::ptr::NonNull<SmallVec<T, N>>,
}

impl<'a, T: 'a, const N: usize> Iterator for Drain<'a, T, N> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.iter
            .next()
            .map(|reference| unsafe { core::ptr::read(reference) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T: 'a, const N: usize> DoubleEndedIterator for Drain<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.iter
            .next_back()
            .map(|reference| unsafe { core::ptr::read(reference) })
    }
}

impl<'a, T, const N: usize> ExactSizeIterator for Drain<'a, T, N> {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, T, const N: usize> core::iter::FusedIterator for Drain<'a, T, N> {}

impl<'a, T: 'a, const N: usize> Drop for Drain<'a, T, N> {
    fn drop(&mut self) {
        self.for_each(drop);

        if self.tail_len > 0 {
            unsafe {
                let source_vec = self.vec.as_mut();

                // memmove back untouched tail, update to new length
                let start = source_vec.len();
                let tail = self.tail_start;
                if tail != start {
                    // as_mut_ptr creates a &mut, invalidating other pointers.
                    // This pattern avoids calling it with a pointer already present.
                    let ptr = source_vec.as_mut_ptr();
                    let src = ptr.add(tail);
                    let dst = ptr.add(start);
                    core::ptr::copy(src, dst, self.tail_len);
                }
                source_vec.set_len(start + self.tail_len);
            }
        }
    }
}

/// An iterator that consumes a `SmallVec` and yields its items by value.
///
/// Returned from [`SmallVec::into_iter`][1].
///
/// [1]: struct.SmallVec.html#method.into_iter
pub struct IntoIter<T, const N: usize> {
    raw: RawSmallVec<T, N>,
    begin: usize,
    end: TaggedLen,
}

impl<T, const N: usize> IntoIter<T, N> {
    #[inline]
    const fn is_zst() -> bool {
        size_of::<T>() == 0
    }

    #[inline]
    const fn as_ptr(&self) -> *const T {
        let on_heap = self.end.on_heap(Self::is_zst());
        if on_heap {
            unsafe { self.raw.as_ptr_heap() }
        } else {
            self.raw.as_ptr_inline()
        }
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T {
        let on_heap = self.end.on_heap(Self::is_zst());
        if on_heap {
            unsafe { self.raw.as_mut_ptr_heap() }
        } else {
            self.raw.as_mut_ptr_inline()
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            let ptr = self.as_ptr();
            core::slice::from_raw_parts(
                ptr.add(self.begin),
                self.end.value(Self::is_zst()) - self.begin,
            )
        }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            let ptr = self.as_mut_ptr();
            core::slice::from_raw_parts_mut(
                ptr.add(self.begin),
                self.end.value(Self::is_zst()) - self.begin,
            )
        }
    }
}

impl<T, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.begin == self.end.value(Self::is_zst()) {
            None
        } else {
            unsafe {
                let ptr = self.as_mut_ptr();
                let value = ptr.add(self.begin).read();
                self.begin += 1;
                Some(value)
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.end.value(Self::is_zst()) - self.begin;
        (size, Some(size))
    }
}

impl<T, const N: usize> DoubleEndedIterator for IntoIter<T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let mut end = self.end.value(Self::is_zst());
        if self.begin == end {
            None
        } else {
            unsafe {
                let ptr = self.as_mut_ptr();
                let on_heap = self.end.on_heap(Self::is_zst());
                end -= 1;
                self.end = TaggedLen::new(end, on_heap, Self::is_zst());
                let value = ptr.add(end).read();
                Some(value)
            }
        }
    }
}
impl<T, const N: usize> ExactSizeIterator for IntoIter<T, N> {}
impl<T, const N: usize> core::iter::FusedIterator for IntoIter<T, N> {}

impl<T, const N: usize> SmallVec<T, N> {
    #[inline]
    const fn is_zst() -> bool {
        size_of::<T>() == 0
    }

    #[inline]
    pub const fn new() -> SmallVec<T, N> {
        Self {
            len: TaggedLen::new(0, false, Self::is_zst()),
            raw: RawSmallVec::new(),
        }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        let mut this = Self::new();
        this.reserve_exact(capacity);
        this
    }

    #[inline]
    pub fn from_vec(vec: Vec<T>) -> Self {
        if Self::is_zst() {
            // "Move" elements to stack buffer. They're ZST so we don't actually have to do
            // anything. Just make sure they're not dropped.
            // We don't wrap the vector in ManuallyDrop so that when it's dropped, the memory is
            // deallocated, if it needs to be.
            let mut vec = vec;
            let len = vec.len();

            // SAFETY: `0` is less than the vector's capacity.
            // old_len..new_len is an empty range. So there are no uninitialized elements
            unsafe { vec.set_len(0) };
            Self {
                len: TaggedLen::new(len, false, Self::is_zst()),
                raw: RawSmallVec::new(),
            }
        } else {
            let mut vec = ManuallyDrop::new(vec);
            let len = vec.len();
            let cap = vec.capacity();
            let ptr = vec.as_mut_ptr();

            Self {
                len: TaggedLen::new(len, true, Self::is_zst()),
                raw: RawSmallVec::new_heap(ptr, cap),
            }
        }
    }

    #[inline]
    pub const fn from_buf(buf: [T; N]) -> Self {
        Self {
            len: TaggedLen::new(N, false, Self::is_zst()),
            raw: RawSmallVec::new_inline(MaybeUninit::new(buf)),
        }
    }

    #[inline]
    pub fn from_buf_and_len(buf: [T; N], len: usize) -> Self {
        assert!(len <= N);
        let mut vec = Self {
            len: TaggedLen::new(len, false, Self::is_zst()),
            raw: RawSmallVec::new_inline(MaybeUninit::new(buf)),
        };
        // Deallocate the remaining elements so no memory is leaked.
        unsafe {
            // SAFETY: both the input and output pointers are in range of the stack allocation
            let remainder_ptr = addr_of_mut!(vec.raw.inline).add(len);
            let remainder_len = N - len;

            // SAFETY: the values are initialized, so dropping them here is fine.
            core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(
                remainder_ptr,
                remainder_len,
            ));
        }

        vec
    }

    #[inline]
    pub const unsafe fn from_buf_and_len_unchecked(buf: MaybeUninit<[T; N]>, len: usize) -> Self {
        debug_assert!(len <= N);
        Self {
            len: TaggedLen::new(len, false, Self::is_zst()),
            raw: RawSmallVec::new_inline(buf),
        }
    }

    #[inline]
    unsafe fn set_on_heap(&mut self) {
        self.len = TaggedLen::new(self.len(), true, Self::is_zst());
    }
    #[inline]
    unsafe fn set_inline(&mut self) {
        self.len = TaggedLen::new(self.len(), false, Self::is_zst());
    }

    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.capacity());
        let on_heap = self.len.on_heap(Self::is_zst());
        self.len = TaggedLen::new(new_len, on_heap, Self::is_zst());
    }

    #[inline]
    const fn inline_capacity() -> usize {
        if Self::is_zst() {
            usize::MAX
        } else {
            N
        }
    }

    #[inline]
    pub const fn inline_size(&self) -> usize {
        Self::inline_capacity()
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.len.value(Self::is_zst())
    }

    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub const fn capacity(&self) -> usize {
        if self.len.on_heap(Self::is_zst()) {
            unsafe { self.raw.heap.1 }
        } else {
            self.inline_size()
        }
    }

    #[inline]
    pub const fn spilled(&self) -> bool {
        self.len.on_heap(Self::is_zst())
    }

    pub fn drain<R>(&mut self, range: R) -> Drain<'_, T, N>
    where
        R: core::ops::RangeBounds<usize>,
    {
        use core::ops::Bound::*;

        let len = self.len();
        let start = match range.start_bound() {
            Included(&n) => n,
            Excluded(&n) => n.checked_add(1).expect("Range start out of bounds"),
            Unbounded => 0,
        };
        let end = match range.end_bound() {
            Included(&n) => n.checked_add(1).expect("Range end out of bounds"),
            Excluded(&n) => n,
            Unbounded => len,
        };

        assert!(start <= end);
        assert!(end <= len);

        unsafe {
            self.set_len(start);

            let range_slice = core::slice::from_raw_parts(self.as_ptr().add(start), end - start);

            Drain {
                tail_start: end,
                tail_len: len - end,
                iter: range_slice.iter(),
                // Since self is a &mut, passing it to a function would invalidate the slice iterator.
                vec: core::ptr::NonNull::new_unchecked(self as *mut _),
            }
        }
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        let len = self.len();
        if len == self.capacity() {
            self.reserve(1);
        }
        // SAFETY: both the input and output are within the allocation
        let ptr = unsafe { self.as_mut_ptr().add(len) };
        // SAFETY: we allocated enough space in case it wasn't enough, so the address is valid for
        // writes.
        unsafe { ptr.write(value) };
        unsafe { self.set_len(len + 1) }
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            let len = self.len() - 1;
            // SAFETY: len < old_len since this can't overflow, because the old length is non zero
            unsafe { self.set_len(len) };
            let value = unsafe { self.as_mut_ptr().add(len).read() };
            Some(value)
        }
    }

    #[inline]
    pub fn append<const M: usize>(&mut self, other: &mut SmallVec<T, M>) {
        // can't overflow since both are smaller than isize::MAX and 2 * isize::MAX < usize::MAX
        let len = self.len();
        let other_len = other.len();
        let total_len = len + other_len;
        if total_len > self.capacity() {
            self.reserve(other_len);
        }

        // SAFETY: see `Self::push`
        let ptr = unsafe { self.as_mut_ptr().add(len) };
        // SAFETY: we have a mutable reference to each vector and each uniquely owns its memory.
        // so the ranges can't overlap
        unsafe { copy_nonoverlapping(other.as_ptr(), ptr, other_len) };
        unsafe { other.set_len(0) }
    }

    #[inline]
    pub fn grow(&mut self, new_capacity: usize) {
        infallible(self.try_grow(new_capacity));
    }

    #[inline]
    pub fn try_grow(&mut self, new_capacity: usize) -> Result<(), CollectionAllocErr> {
        let len = self.len();
        assert!(new_capacity >= len);

        if new_capacity > self.inline_size() {
            let result = unsafe { self.raw.try_grow(self.len, new_capacity) };
            if result.is_ok() {
                unsafe { self.set_on_heap() };
            }
            result
        } else {
            if self.spilled() {
                unsafe {
                    let (ptr, old_cap) = self.raw.heap;
                    copy_nonoverlapping(ptr, self.raw.as_mut_ptr_inline(), len);
                    {
                        let _drop_dealloc = DropDealloc {
                            ptr: ptr as *mut u8,
                            size_bytes: old_cap * size_of::<T>(),
                            align: align_of::<T>(),
                        };
                    }
                    self.set_inline();
                }
            }
            Ok(())
        }
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        // can't overflow since len <= capacity
        if additional > self.capacity() - self.len() {
            let new_capacity = infallible(
                self.len()
                    .checked_add(additional)
                    .and_then(usize::checked_next_power_of_two)
                    .ok_or(CollectionAllocErr::CapacityOverflow),
            );
            self.grow(new_capacity);
        }
    }

    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), CollectionAllocErr> {
        if additional > self.capacity() - self.len() {
            let new_capacity = self
                .len()
                .checked_add(additional)
                .and_then(usize::checked_next_power_of_two)
                .ok_or(CollectionAllocErr::CapacityOverflow)?;
            self.try_grow(new_capacity)
        } else {
            Ok(())
        }
    }

    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        // can't overflow since len <= capacity
        if additional > self.capacity() - self.len() {
            let new_capacity = infallible(
                self.len()
                    .checked_add(additional)
                    .ok_or(CollectionAllocErr::CapacityOverflow),
            );
            self.grow(new_capacity);
        }
    }

    #[inline]
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), CollectionAllocErr> {
        if additional > self.capacity() - self.len() {
            let new_capacity = self
                .len()
                .checked_add(additional)
                .ok_or(CollectionAllocErr::CapacityOverflow)?;
            self.try_grow(new_capacity)
        } else {
            Ok(())
        }
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        if !self.spilled() {
            return;
        }
        let len = self.len();
        if len <= self.inline_size() {
            let (ptr, capacity) = unsafe { self.raw.heap };
            self.raw = RawSmallVec::new_inline(MaybeUninit::uninit());
            unsafe { copy_nonoverlapping(ptr, self.raw.as_mut_ptr_inline(), len) };
            unsafe { self.set_inline() };
            unsafe {
                alloc::alloc::dealloc(
                    ptr as *mut T as *mut u8,
                    Layout::from_size_align_unchecked(capacity * size_of::<T>(), align_of::<T>()),
                )
            };
        } else if len < self.capacity() {
            self.grow(len);
        }
    }

    #[inline]
    pub fn truncate(&mut self, len: usize) {
        let old_len = self.len();
        if len < old_len {
            unsafe {
                self.set_len(len);
                core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(
                    self.as_mut_ptr().add(len),
                    old_len - len,
                ))
            }
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        let len = self.len();
        let ptr = self.as_ptr();
        unsafe { core::slice::from_raw_parts(ptr, len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let len = self.len();
        let ptr = self.as_mut_ptr();
        unsafe { core::slice::from_raw_parts_mut(ptr, len) }
    }

    #[inline]
    pub fn swap_remove(&mut self, index: usize) -> T {
        let len = self.len();
        assert!(index < len);
        let new_len = len - 1;
        unsafe {
            self.set_len(new_len);
            let ptr = self.as_mut_ptr();
            let last = ptr.add(new_len);
            let ith = ptr.add(index);
            let last_item = last.read();
            let ith_item = ith.read();
            ith.write(last_item);
            ith_item
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    #[inline]
    pub fn remove(&mut self, index: usize) -> T {
        let len = self.len();
        assert!(index < len);
        let new_len = len - 1;
        unsafe {
            self.set_len(new_len);
            let ptr = self.as_mut_ptr();
            let ith = ptr.add(index);
            let ith_item = ith.read();
            core::ptr::copy(ith.add(1), ith, new_len - index);
            ith_item
        }
    }

    #[inline]
    pub fn insert(&mut self, index: usize, value: T) {
        let len = self.len();
        assert!(index <= len);
        self.reserve(1);
        let ptr = self.as_mut_ptr();
        unsafe {
            if index < len {
                core::ptr::copy(ptr.add(index), ptr.add(index + 1), len - index);
            }
            ptr.add(index).write(value);
            self.set_len(len + 1);
        }
    }

    fn insert_many_impl<I: Iterator<Item = T>>(&mut self, mut index: usize, mut iter: I) {
        let len = self.len();
        if index == len {
            return self.extend(iter);
        }

        let (lower_bound, _) = iter.size_hint();
        self.reserve(lower_bound);

        let count = unsafe {
            let ptr = self.as_mut_ptr();
            let count = insert_many_batch_phase(ptr, index, lower_bound, len, &mut iter);
            self.set_len(len + count);
            count
        };

        index += count;
        iter.enumerate()
            .for_each(|(i, item)| self.insert(index + i, item));
    }

    #[inline]
    pub fn insert_many<I: IntoIterator<Item = T>>(&mut self, index: usize, iterable: I) {
        self.insert_many_impl(index, iterable.into_iter());
    }

    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        if self.len.on_heap(Self::is_zst()) {
            unsafe { self.raw.as_ptr_heap() }
        } else {
            self.raw.as_ptr_inline()
        }
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        if self.len.on_heap(Self::is_zst()) {
            unsafe { self.raw.as_mut_ptr_heap() }
        } else {
            self.raw.as_mut_ptr_inline()
        }
    }

    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        let len = self.len();
        if !self.spilled() {
            let mut vec = Vec::with_capacity(len);
            let this = ManuallyDrop::new(self);
            unsafe {
                copy_nonoverlapping(this.raw.as_ptr_inline(), vec.as_mut_ptr(), len);
                vec.set_len(len);
            }
            vec
        } else {
            let this = ManuallyDrop::new(self);
            unsafe {
                let (ptr, cap) = this.raw.heap;
                Vec::from_raw_parts(ptr as *mut T, len, cap)
            }
        }
    }

    #[inline]
    pub fn into_boxed_slice(self) -> Box<[T]> {
        self.into_vec().into_boxed_slice()
    }

    #[inline]
    pub fn into_inner(self) -> Result<[T; N], Self> {
        if self.len() != N {
            Err(self)
        } else {
            let this = ManuallyDrop::new(self);
            let ptr = this.as_ptr() as *const [T; N];
            unsafe { Ok(ptr.read()) }
        }
    }

    pub fn retain<F: FnMut(&mut T) -> bool>(&mut self, mut f: F) {
        let mut del = 0;
        let len = self.len();
        for i in 0..len {
            unsafe {
                let ptr = self.as_mut_ptr().add(i);
                if !f(&mut *ptr) {
                    del += 1;
                } else if del > 0 {
                    core::mem::swap(&mut *ptr, &mut *ptr.sub(del));
                }
            }
        }
        self.truncate(len - del);
    }

    #[inline]
    pub fn retain_mut<F: FnMut(&mut T) -> bool>(&mut self, f: F) {
        self.retain(f)
    }

    #[inline]
    pub fn dedup(&mut self)
    where
        T: PartialEq,
    {
        self.dedup_by(|a, b| a == b);
    }

    #[inline]
    pub fn dedup_by_key<F, K>(&mut self, mut key: F)
    where
        F: FnMut(&mut T) -> K,
        K: PartialEq<K>,
    {
        self.dedup_by(|a, b| key(a) == key(b));
    }

    #[inline]
    pub fn dedup_by<F>(&mut self, mut same_bucket: F)
    where
        F: FnMut(&mut T, &mut T) -> bool,
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
                let p_r = ptr.add(r);
                let p_wm1 = ptr.add(w - 1);
                if !same_bucket(&mut *p_r, &mut *p_wm1) {
                    if r != w {
                        let p_w = p_wm1.add(1);
                        core::mem::swap(&mut *p_r, &mut *p_w);
                    }
                    w += 1;
                }
            }
        }

        self.truncate(w);
    }

    pub fn resize_with<F>(&mut self, new_len: usize, f: F)
    where
        F: FnMut() -> T,
    {
        let old_len = self.len();
        if old_len < new_len {
            let mut f = f;
            let additional = new_len - old_len;
            self.reserve(additional);
            for _ in 0..additional {
                self.push(f());
            }
        } else if old_len > new_len {
            self.truncate(new_len);
        }
    }

    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut T, length: usize, capacity: usize) -> SmallVec<T, N> {
        assert!(!Self::is_zst());
        SmallVec {
            len: TaggedLen::new(length, true, Self::is_zst()),
            raw: RawSmallVec::new_heap(ptr, capacity),
        }
    }

    fn extend_impl<I: Iterator<Item = T>>(&mut self, mut iter: I) {
        let len = self.len();
        let (lower_bound, _) = iter.size_hint();
        self.reserve(lower_bound);
        unsafe {
            let ptr = self.as_mut_ptr();
            let count = extend_batch_phase(ptr, lower_bound, len, &mut iter);
            self.set_len(len + count);
        }
        iter.for_each(|item| self.push(item));
    }
}

impl<T, const N: usize> Default for SmallVec<T, N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy, const N: usize> SmallVec<T, N> {
    #[inline]
    pub fn from_slice(slice: &[T]) -> Self {
        let len = slice.len();

        let mut this = Self::with_capacity(len);
        let ptr = this.as_mut_ptr();
        unsafe {
            copy_nonoverlapping(slice.as_ptr(), ptr, len);
            this.set_len(len);
        }
        this
    }

    #[inline]
    pub fn insert_from_slice(&mut self, index: usize, slice: &[T]) {
        let len = self.len();
        let other_len = slice.len();
        assert!(index <= len);
        self.reserve(other_len);
        unsafe {
            let base_ptr = self.as_mut_ptr();
            let ith_ptr = base_ptr.add(index);
            let shifted_ptr = base_ptr.add(index + other_len);
            core::ptr::copy(ith_ptr, shifted_ptr, len - index);
            copy_nonoverlapping(slice.as_ptr(), ith_ptr, other_len);
            self.set_len(len + other_len);
        }
    }

    #[inline]
    pub fn extend_from_slice(&mut self, slice: &[T]) {
        let len = self.len();
        let other_len = slice.len();
        self.reserve(other_len);
        unsafe {
            let base_ptr = self.as_mut_ptr();
            let end_ptr = base_ptr.add(len);
            copy_nonoverlapping(slice.as_ptr(), end_ptr, other_len);
            self.set_len(len + other_len);
        }
    }
}

impl<T: Clone, const N: usize> SmallVec<T, N> {
    #[inline]
    pub fn resize(&mut self, len: usize, value: T) {
        let old_len = self.len();
        if len > old_len {
            self.extend(core::iter::repeat(value).take(len - old_len));
        } else {
            self.truncate(len);
        }
    }

    #[inline]
    pub fn from_elem(elem: T, n: usize) -> Self {
        if n > Self::inline_capacity() {
            Self::from_vec(vec![elem; n])
        } else {
            let mut v = Self::new();

            unsafe {
                let ptr = v.as_mut_ptr();
                let mut guard = DropGuard { ptr, len: 0 };

                // assume T is expensive to clone
                for i in 0..n {
                    guard.len = i;
                    ptr.add(i).write(elem.clone());
                }
                core::mem::forget(guard);
                v.set_len(n);
            }
            v
        }
    }
}

struct DropShiftGuard<T> {
    ptr: *mut T,
    len: usize,
    shifted_ptr: *const T,
    shifted_len: usize,
}
impl<T> Drop for DropShiftGuard<T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            core::ptr::slice_from_raw_parts_mut(self.ptr, self.len).drop_in_place();
            core::ptr::copy(self.shifted_ptr, self.ptr, self.shifted_len);
        }
    }
}

struct DropGuard<T> {
    ptr: *mut T,
    len: usize,
}
impl<T> Drop for DropGuard<T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            core::ptr::slice_from_raw_parts_mut(self.ptr, self.len).drop_in_place();
        }
    }
}

unsafe fn insert_many_batch_phase<T, I: Iterator<Item = T>>(
    ptr: *mut T,
    index: usize,
    lower_bound: usize,
    len: usize,
    iter: &mut I,
) -> usize {
    // shift elements to the right to make space for the initial elements from the iterator
    core::ptr::copy(ptr.add(index), ptr.add(index + lower_bound), len - index);
    let ptr_ith = ptr.add(index);
    let mut guard = DropShiftGuard {
        ptr: ptr_ith,
        len: 0,
        shifted_ptr: ptr_ith.add(lower_bound),
        shifted_len: len - index,
    };
    iter.take(lower_bound).enumerate().for_each(|(i, item)| {
        ptr_ith.add(i).write(item);
        guard.len = i + 1;
    });
    let count = guard.len;
    core::mem::forget(guard);

    if count < lower_bound {
        core::ptr::copy(ptr_ith.add(lower_bound), ptr_ith.add(count), len - index);
    }
    count
}

unsafe fn extend_batch_phase<T, I: Iterator<Item = T>>(
    ptr: *mut T,
    lower_bound: usize,
    len: usize,
    iter: &mut I,
) -> usize {
    let ptr_end = ptr.add(len);
    let mut guard = DropGuard {
        ptr: ptr_end,
        len: 0,
    };
    iter.take(lower_bound).enumerate().for_each(|(i, item)| {
        ptr_end.add(i).write(item);
        guard.len = i + 1;
    });
    let count = guard.len;
    core::mem::forget(guard);
    count
}

impl<T, const N: usize> Extend<T> for SmallVec<T, N> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iterable: I) {
        self.extend_impl(iterable.into_iter());
    }
}

struct DropDealloc {
    ptr: *mut u8,
    size_bytes: usize,
    align: usize,
}

impl Drop for DropDealloc {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            if self.size_bytes > 0 {
                alloc::alloc::dealloc(
                    self.ptr,
                    Layout::from_size_align_unchecked(self.size_bytes, self.align),
                );
            }
        }
    }
}

impl<T, const N: usize> Drop for SmallVec<T, N> {
    fn drop(&mut self) {
        let on_heap = self.spilled();
        let len = self.len();
        let ptr = self.as_mut_ptr();
        unsafe {
            let _drop_dealloc = if on_heap {
                let capacity = self.capacity();
                Some(DropDealloc {
                    ptr: ptr as *mut u8,
                    size_bytes: capacity * size_of::<T>(),
                    align: align_of::<T>(),
                })
            } else {
                None
            };
            core::ptr::slice_from_raw_parts_mut(ptr, len).drop_in_place();
        }
    }
}

impl<T, const N: usize> Drop for IntoIter<T, N> {
    fn drop(&mut self) {
        unsafe {
            let is_zst = size_of::<T>() == 0;
            let on_heap = self.end.on_heap(is_zst);
            let begin = self.begin;
            let end = self.end.value(is_zst);
            let ptr = self.as_mut_ptr();
            let _drop_dealloc = if on_heap {
                let capacity = self.raw.heap.1;
                Some(DropDealloc {
                    ptr: ptr as *mut u8,
                    size_bytes: capacity * size_of::<T>(),
                    align: align_of::<T>(),
                })
            } else {
                None
            };
            core::ptr::slice_from_raw_parts_mut(ptr.add(begin), end - begin).drop_in_place();
        }
    }
}

impl<T, const N: usize> core::ops::Deref for SmallVec<T, N> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}
impl<T, const N: usize> core::ops::DerefMut for SmallVec<T, N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T, const N: usize> core::iter::FromIterator<T> for SmallVec<T, N> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iterable: I) -> Self {
        let mut vec = Self::new();
        vec.extend_impl(iterable.into_iter());
        vec
    }
}

impl<'a, T: Clone, const N: usize> From<&'a [T]> for SmallVec<T, N> {
    fn from(slice: &'a [T]) -> Self {
        slice.iter().cloned().collect()
    }
}
impl<T: Clone, const N: usize> From<[T; N]> for SmallVec<T, N> {
    fn from(array: [T; N]) -> Self {
        Self::from_buf(array)
    }
}
impl<T: Clone, const N: usize> From<Vec<T>> for SmallVec<T, N> {
    fn from(array: Vec<T>) -> Self {
        Self::from_vec(array)
    }
}

impl<T: Clone, const N: usize> Clone for SmallVec<T, N> {
    #[inline]
    fn clone(&self) -> SmallVec<T, N> {
        SmallVec::from(self.as_slice())
    }

    fn clone_from(&mut self, source: &Self) {
        // Inspired from `impl Clone for Vec`.

        // drop anything that will not be overwritten
        self.truncate(source.len());

        // self.len <= other.len due to the truncate above, so the
        // slices here are always in-bounds.
        let init = unsafe { source.get_unchecked(..self.len()) };
        let tail = unsafe { source.get_unchecked(self.len()..) };

        // reuse the contained values' allocations/resources.
        self.clone_from_slice(init);
        self.extend(tail.iter().cloned());
    }
}

impl<T: Clone, const N: usize> Clone for IntoIter<T, N> {
    #[inline]
    fn clone(&self) -> IntoIter<T, N> {
        SmallVec::from(self.as_slice()).into_iter()
    }
}

#[macro_export]
macro_rules! smallvec {
    // count helper: transform any expression into 1
    (@one $x:expr) => (1usize);
    ($elem:expr; $n:expr) => ({
        $crate::SmallVec::from_elem($elem, $n)
    });
    ($($x:expr),*$(,)?) => ({
        let count = 0usize $(+ $crate::smallvec!(@one $x))*;
        #[allow(unused_mut)]
        let mut vec = $crate::SmallVec::new();
        if count <= vec.inline_size() {
            $(vec.push($x);)*
            vec
        } else {
            $crate::SmallVec::from_vec($crate::alloc::vec![$($x,)*])
        }
    });
}

#[macro_export]
macro_rules! smallvec_inline {
    // count helper: transform any expression into 1
    (@one $x:expr) => (1usize);
    ($elem:expr; $n:expr) => ({
        $crate::SmallVec::<_, $n>::from_buf([$elem; $n])
    });
    ($($x:expr),+ $(,)?) => ({
        const N: usize = 0usize $(+ $crate::smallvec_inline!(@one $x))*;
        $crate::SmallVec::<_, N>::from_buf([$($x,)*])
    });
}

impl<T, const N: usize> IntoIterator for SmallVec<T, N> {
    type IntoIter = IntoIter<T, N>;
    type Item = T;
    fn into_iter(self) -> Self::IntoIter {
        unsafe {
            // Set SmallVec len to zero as `IntoIter` drop handles dropping of the elements
            let this = ManuallyDrop::new(self);
            IntoIter {
                raw: (&this.raw as *const RawSmallVec<T, N>).read(),
                begin: 0,
                end: this.len,
            }
        }
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a SmallVec<T, N> {
    type IntoIter = core::slice::Iter<'a, T>;
    type Item = &'a T;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut SmallVec<T, N> {
    type IntoIter = core::slice::IterMut<'a, T>;
    type Item = &'a mut T;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, U, const N: usize, const M: usize> PartialEq<SmallVec<U, M>> for SmallVec<T, N>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &SmallVec<U, M>) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}
impl<T, const N: usize> Eq for SmallVec<T, N> where T: Eq {}

impl<T, const N: usize> PartialOrd for SmallVec<T, N>
where
    T: PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, other: &SmallVec<T, N>) -> Option<core::cmp::Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<T, const N: usize> Ord for SmallVec<T, N>
where
    T: Ord,
{
    #[inline]
    fn cmp(&self, other: &SmallVec<T, N>) -> core::cmp::Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<T, const N: usize> Borrow<[T]> for SmallVec<T, N> {
    #[inline]
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, const N: usize> BorrowMut<[T]> for SmallVec<T, N> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, const N: usize> AsRef<[T]> for SmallVec<T, N> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, const N: usize> AsMut<[T]> for SmallVec<T, N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Debug, const N: usize> Debug for SmallVec<T, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: Debug, const N: usize> Debug for IntoIter<T, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("IntoIter").field(&self.as_slice()).finish()
    }
}

impl<'a, T: Debug, const N: usize> Debug for Drain<'a, T, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Drain").field(&self.iter.as_slice()).finish()
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<T, const N: usize> Serialize for SmallVec<T, N>
where
    T: Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_seq(Some(self.len()))?;
        for item in self {
            state.serialize_element(item)?;
        }
        state.end()
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<'de, T, const N: usize> Deserialize<'de> for SmallVec<T, N>
where
    T: Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_seq(SmallVecVisitor {
            phantom: PhantomData,
        })
    }
}

#[cfg(feature = "serde")]
struct SmallVecVisitor<T, const N: usize> {
    phantom: PhantomData<T>,
}

#[cfg(feature = "serde")]
impl<'de, T, const N: usize> Visitor<'de> for SmallVecVisitor<T, N>
where
    T: Deserialize<'de>,
{
    type Value = SmallVec<T, N>;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("a sequence")
    }

    fn visit_seq<B>(self, mut seq: B) -> Result<Self::Value, B::Error>
    where
        B: SeqAccess<'de>,
    {
        use serde::de::Error;
        let len = seq.size_hint().unwrap_or(0);
        let mut values = SmallVec::new();
        values.try_reserve(len).map_err(B::Error::custom)?;

        while let Some(value) = seq.next_element()? {
            values.push(value);
        }

        Ok(values)
    }
}

#[cfg(feature = "write")]
#[cfg_attr(docsrs, doc(cfg(feature = "write")))]
impl<const N: usize> io::Write for SmallVec<u8, N> {
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
