// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(feature = "specialization", allow(incomplete_features))]
#![cfg_attr(feature = "specialization", feature(specialization, trusted_len))]
#![cfg_attr(
    all(feature = "allocator-api", not(feature = "allocator-api2")),
    feature(allocator_api)
)]

extern crate alloc;

mod allocator;
#[cfg(feature = "borsh")]
mod borsh;
mod comparisons;
mod conversions;
mod errors;
mod iterators;
pub use iterators::{
    drain::Drain,
    extractif::ExtractIf,
    intoiter::IntoIter,
    splice::Splice
};
mod macros;
#[cfg(feature = "malloc_size_of")]
mod mallocsizeof;
mod newrange;
mod rawsmallvec;
mod references;
#[cfg(feature = "serde")]
mod serde;
#[cfg(feature = "specialization")]
mod specialization;
mod taggedlen;

#[cfg(feature = "bytes")]
use bytes::{
    BufMut,
    buf::UninitSlice
};
#[cfg(feature = "defmt")]
use defmt::{
    Format,
    Formatter as DeFormatter,
    write as dewrite
};
pub use errors::SmallVecError;
#[cfg(feature = "std")]
use std::io;
use {
    allocator::{
        Allocator,
        Box,
        Global,
        Vec,
        vec
    },
    core::{
        alloc::Layout,
        fmt::Debug,
        hash::{
            Hash,
            Hasher
        },
        mem::{
            ManuallyDrop,
            MaybeUninit,
            align_of,
            size_of
        },
        ptr::{
            NonNull,
            copy,
            copy_nonoverlapping,
            drop_in_place
        }
    },
    newrange::NewRange
};
#[cfg(feature = "internals")]
pub use {
    rawsmallvec::RawSmallVec,
    taggedlen::TaggedLen
};
#[cfg(not(feature = "internals"))]
use {
    rawsmallvec::RawSmallVec,
    taggedlen::TaggedLen
};

#[repr(C)]
pub struct SmallVec<T, const N: usize, A: Allocator = Global> {
    length: TaggedLen<T>,
    raw: RawSmallVec<T, N>,
    allocator: A
}

unsafe impl<T: Send, const N: usize, A: Allocator + Send> Send for SmallVec<T, N, A> {}
unsafe impl<T: Sync, const N: usize, A: Allocator + Sync> Sync for SmallVec<T, N, A> {}

impl<T, const N: usize> Default for SmallVec<T, N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> SmallVec<T, N> {
    #[inline]
    pub const fn new() -> SmallVec<T, N> {
        Self::new_in(Global)
    }

    pub fn try_with_capacity(capacity: usize) -> Result<Self, SmallVecError> {
        Self::try_with_capacity_in(capacity, Global)
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, Global)
    }

    #[inline]
    pub const fn from_buf<const S: usize>(elements: [T; S]) -> Self {
        const {
            assert!(S <= N);
        }

        // Although we create a new buffer, since S and N are known at compile
        // time, even with `-C opt-level=1`, it gets optimized as best
        // as it could be. (Checked with <godbolt.org>)
        let mut buf: MaybeUninit<[T; N]> = MaybeUninit::uninit();

        // SAFETY: buf and elements do not overlap, are aligned and have space
        // for at least S elements since S <= N.
        // We will drop the elements only once since we do forget(elements).
        unsafe {
            copy_nonoverlapping(elements.as_ptr(), buf.as_mut_ptr() as *mut T, S);
        }

        // `elements` have been moved into buf and will be dropped by SmallVec
        core::mem::forget(elements);

        // SAFETY: all the members in 0..S are initialized
        Self {
            length: TaggedLen::new(S, false),
            raw: RawSmallVec::new_inline(buf),
            allocator: Global
        }
    }

    #[inline]
    pub fn from_buf_and_len(buf: [T; N], length: usize) -> Self {
        assert!(length <= N);
        // SAFETY: all the members in 0..length are initialized
        let mut vec = Self {
            length: TaggedLen::new(length, false),
            raw: RawSmallVec::new_inline(MaybeUninit::new(buf)),
            allocator: Global
        };
        // Deallocate the remaining elements so no memory is leaked.
        unsafe {
            // SAFETY: both the input and output pointers are in range of the
            // stack allocation
            let remainder_ptr = vec.raw.as_mut_ptr_inline().add(length);
            let remainder_len = N - length;

            // SAFETY: the values are initialized, so dropping them here is
            // fine.
            core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(
                remainder_ptr,
                remainder_len
            ));
        }

        vec
    }

    /// Constructs a new `SmallVec` on the stack from a buffer without copying
    /// elements. Also sets the length. The user is responsible for ensuring
    /// that `length <= N`.
    ///
    /// # Examples
    ///
    /// ```
    /// use {
    ///     smallvec::SmallVec,
    ///     std::mem::MaybeUninit
    /// };
    ///
    /// let buf = [1, 2, 3, 4, 5, 0, 0, 0];
    /// let small_vec = unsafe { SmallVec::from_buf_and_len_unchecked(MaybeUninit::new(buf), 5) };
    ///
    /// assert_eq!(&*small_vec, &[1, 2, 3, 4, 5]);
    /// ```
    ///
    /// # Safety
    ///
    /// `length <= N`, and all the elements in `buf[..length]` must be
    /// initialized
    #[inline]
    pub const unsafe fn from_buf_and_len_unchecked(
        buf: MaybeUninit<[T; N]>,
        length: usize
    ) -> Self {
        debug_assert!(length <= N);
        Self {
            length: TaggedLen::new(length, false),
            raw: RawSmallVec::new_inline(buf),
            allocator: Global
        }
    }

    #[inline]
    pub fn from_vec(vec: Vec<T>) -> Self {
        if vec.capacity() == 0 {
            return Self::new();
        }

        if Self::IS_ZST {
            // "Move" elements to stack buffer. They're ZST so we don't actually
            // have to do anything. Just make sure they're not
            // dropped. We don't wrap the vector in ManuallyDrop so
            // that when it's dropped, the memory is deallocated, if
            // it needs to be.
            let mut vec = vec;
            let length = vec.len();

            // SAFETY: `0` is less than the vector's capacity.
            // old_len..new_len is an empty range. So there are no uninitialized
            // elements
            unsafe { vec.set_len(0) };
            Self {
                length: TaggedLen::new(length, false),
                raw: RawSmallVec::new(),
                allocator: Global
            }
        } else {
            let mut vec = ManuallyDrop::new(vec);
            let length = vec.len();
            let cap = vec.capacity();
            // SAFETY: vec.capacity is not `0` (checked above), so the pointer
            // can not dangle and thus specifically cannot be null.
            let ptr = unsafe { NonNull::new_unchecked(vec.as_mut_ptr()) };

            Self {
                length: TaggedLen::new(length, true),
                raw: RawSmallVec::new_heap(ptr, cap),
                allocator: Global
            }
        }
    }

    pub fn splice<R, I>(&mut self, range: R, replace_with: I) -> Splice<'_, I::IntoIter, N>
    where
        R: core::ops::RangeBounds<usize>,
        I: IntoIterator<Item = T>
    {
        Splice::new(self.drain(range), replace_with.into_iter())
    }

    /// Creates a `SmallVec` directly from the raw components of another
    /// `SmallVec`.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren’t
    /// checked:
    ///
    /// - `ptr` needs to have been previously allocated via `SmallVec` from its
    ///   spilled storage (at least, it’s highly likely to be incorrect if it
    ///   wasn’t).
    /// - `ptr`’s `T` type needs to be the same size and alignment that it was
    ///   allocated with
    /// - `length` needs to be less than or equal to `capacity`.
    /// - `capacity` needs to be the capacity that the pointer was allocated
    ///   with.
    ///
    /// Violating these may cause problems like corrupting the allocator’s
    /// internal data structures.
    ///
    /// Additionally, `capacity` must be greater than `N`; that is, the new
    /// `SmallVec` must need to spill over into heap allocated storage. This
    /// condition is asserted against.
    ///
    /// The ownership of `ptr` is effectively transferred to the `SmallVec`
    /// which may then deallocate, reallocate or change the contents of memory
    /// pointed to by the pointer at will. Ensure that nothing else uses the
    /// pointer after calling this function.
    ///
    /// # Examples
    ///
    /// ```
    /// use smallvec::SmallVec;
    ///
    /// let mut v: SmallVec<_, 1> = SmallVec::from([1, 2, 3]);
    ///
    /// // Pull out the important parts of `v`.
    /// let p = v.as_mut_ptr();
    /// let length = v.len();
    /// let cap = v.capacity();
    /// let spilled = v.spilled();
    ///
    /// unsafe {
    ///     // Forget all about `v`. The heap allocation that stored the
    ///     // three values won't be deallocated.
    ///     std::mem::forget(v);
    ///
    ///     // Overwrite memory with [4, 5, 6].
    ///     //
    ///     // This is only safe if `spilled` is true! Otherwise, we are
    ///     // writing into the old `SmallVec`'s inline storage on the
    ///     // stack.
    ///     assert!(spilled);
    ///     for i in 0..length {
    ///         std::ptr::write(p.add(i), 4 + i);
    ///     }
    ///
    ///     // Put everything back together into a SmallVec with a different
    ///     // amount of inline storage, but which is still less than `cap`.
    ///     let rebuilt = SmallVec::<_, 2>::from_raw_parts(p, length, cap);
    ///     assert_eq!(&*rebuilt, &[4, 5, 6]);
    /// }
    /// ```
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut T, length: usize, capacity: usize) -> SmallVec<T, N> {
        assert!(!Self::IS_ZST);

        // SAFETY: We require caller to provide same ptr as we alloc
        // and we never alloc null pointer.
        let ptr = unsafe {
            debug_assert!(!ptr.is_null(), "Called `from_raw_parts` with null pointer.");
            NonNull::new_unchecked(ptr)
        };

        SmallVec {
            length: TaggedLen::new(length, true),
            raw: RawSmallVec::new_heap(ptr, capacity),
            allocator: Global
        }
    }
}

impl<T: Clone, const N: usize> SmallVec<T, N> {
    /// A function for creating [`SmallVec`] values out of slices
    /// for types with the [`Copy`] trait.
    pub fn from_slice_copy(slice: &[T]) -> Self
    where T: Copy {
        let src = slice.as_ptr();
        let length = slice.len();
        let mut result = Self::with_capacity(length);

        // SAFETY: By using `with_capacity`, the pointer will point to valid
        // memory.
        unsafe {
            let dst = result.as_mut_ptr();
            copy_nonoverlapping(src, dst, length);
            result.set_len(length);
        }

        result
    }
}

impl<T, const N: usize, A: Allocator> SmallVec<T, N, A> {
    const IS_ZST: bool = size_of::<T>() == 0;

    /// Sets the length of a vector.
    ///
    /// This will explicitly set the size of the vector, without actually
    /// modifying its buffers, so it is up to the caller to ensure that the
    /// vector is actually the specified size.
    ///
    /// # Safety
    ///
    /// `new_len <= self.capacity()` must be true, and all the elements in the
    /// range `..self.length` must be initialized.
    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.capacity());
        self.length = TaggedLen::new(new_len, self.length.on_heap());
    }

    #[inline]
    pub const fn inline_size() -> usize {
        RawSmallVec::<T, N>::INLINE_CAP
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.length.len()
    }

    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub const fn capacity(&self) -> usize {
        // SAFETY: the tag tells which member is active
        unsafe { self.raw.capacity(self.length.on_heap()) }
    }

    #[inline]
    pub const fn spilled(&self) -> bool {
        self.length.on_heap()
    }

    pub fn drain<R>(&mut self, range: R) -> Drain<'_, T, N, A>
    where R: core::ops::RangeBounds<usize> {
        let length = self.len();
        let core::ops::Range {
            start,
            end
        } = core::ops::Range::new(range, length);

        unsafe {
            // SAFETY: `start <= length`
            self.set_len(start);

            // SAFETY: all the elements in `start..end` are initialized
            let range_slice = core::slice::from_raw_parts(self.as_ptr().add(start), end - start);

            // SAFETY: all the elements in `end..length` are initialized
            Drain {
                tail_start: end,
                tail_len: length - end,
                iter: range_slice.iter(),
                // Since self is a &mut, passing it to a function would invalidate the slice
                // iterator.
                vec: core::ptr::NonNull::new_unchecked(self as *mut _)
            }
        }
    }

    /// Creates an iterator which uses a closure to determine if element in the
    /// range should be removed.
    ///
    /// If the closure returns true, then the element is removed and yielded.
    /// If the closure returns false, the element will remain in the vector and
    /// will not be yielded by the iterator.
    ///
    /// Only elements that fall in the provided range are considered for
    /// extraction, but any elements after the range will still have to be
    /// moved if any element has been extracted.
    ///
    /// If the returned `ExtractIf` is not exhausted, e.g. because it is dropped
    /// without iterating or the iteration short-circuits, then the
    /// remaining elements will be retained. Use [`retain`] with a negated
    /// predicate if you do not need the returned iterator.
    ///
    /// [`retain`]: SmallVec::retain
    ///
    /// Using this method is equivalent to the following code:
    /// ```
    /// # use smallvec::SmallVec;
    /// # use std::cmp::min;
    /// # let some_predicate = |x: &mut i32| { *x == 2 || *x == 3 || *x == 6 };
    /// # let mut vec: SmallVec<i32, 8> = SmallVec::from(&[1i32, 2, 3, 4, 5, 6]);
    /// # let range = 1..4;
    /// let mut i = 0;
    /// while i < min(vec.len(), range.end) {
    ///     if some_predicate(&mut vec[i]) {
    ///         let val = vec.remove(i);
    ///         // your code here
    ///     } else {
    ///         i += 1;
    ///     }
    /// }
    ///
    /// # assert_eq!(vec, SmallVec::<i32, 8>::from(&[1i32, 4, 5]));
    /// ```
    ///
    /// But `extract_if` is easier to use. `extract_if` is also more efficient,
    /// because it can backshift the elements of the array in bulk.
    ///
    /// Note that `extract_if` also lets you mutate the elements passed to the
    /// filter closure, regardless of whether you choose to keep or remove
    /// them.
    ///
    /// # Panics
    ///
    /// If `range` is out of bounds.
    ///
    /// # Examples
    ///
    /// Splitting an array into evens and odds, reusing the original allocation:
    ///
    /// ```
    /// # use smallvec::SmallVec;
    /// let mut numbers: SmallVec<i32, 16> =
    ///     SmallVec::from(&[1i32, 2, 3, 4, 5, 6, 8, 9, 11, 13, 14, 15]);
    ///
    /// let evens = numbers
    ///     .extract_if(.., |x| *x % 2 == 0)
    ///     .collect::<SmallVec<i32, 16>>();
    /// let odds = numbers;
    ///
    /// assert_eq!(evens, SmallVec::<i32, 16>::from(&[2i32, 4, 6, 8, 14]));
    /// assert_eq!(
    ///     odds,
    ///     SmallVec::<i32, 16>::from(&[1i32, 3, 5, 9, 11, 13, 15])
    /// );
    /// ```
    ///
    /// Using the range argument to only process a part of the vector:
    ///
    /// ```
    /// # use smallvec::SmallVec;
    /// let mut items: SmallVec<i32, 16> = SmallVec::from(&[0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2]);
    /// let ones = items
    ///     .extract_if(7.., |x| *x == 1)
    ///     .collect::<SmallVec<i32, 16>>();
    /// assert_eq!(
    ///     items,
    ///     SmallVec::<i32, 16>::from(&[0, 0, 0, 0, 0, 0, 0, 2, 2, 2])
    /// );
    /// assert_eq!(ones.len(), 3);
    /// ```
    pub fn extract_if<F, R>(&mut self, range: R, filter: F) -> ExtractIf<'_, T, N, F, A>
    where
        F: FnMut(&mut T) -> bool,
        R: core::ops::RangeBounds<usize>
    {
        let old_len = self.len();
        let core::ops::Range {
            start,
            end
        } = core::ops::Range::new(range, old_len);

        // Guard against us getting leaked (leak amplification)
        unsafe {
            self.set_len(0);
        }

        ExtractIf {
            vec: self,
            idx: start,
            end,
            del: 0,
            old_len,
            pred: filter
        }
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        _ = self.push_mut(value);
    }

    #[inline]
    #[must_use]
    pub fn push_mut(&mut self, value: T) -> &mut T {
        let length = self.len();
        if length == self.capacity() {
            self.reserve(1);
        }
        debug_assert!(length < self.capacity());

        // SAFETY: `length < capacity` after the reserve,
        //         so the offset stays in bounds of the allocation.
        let ptr = unsafe { self.as_mut_ptr().add(length) };
        // SAFETY: we allocated enough space in case it wasn't enough, so the
        // address is valid for writes.
        unsafe { ptr.write(value) };

        // LEGAL: all elements in `0..length + 1` are initialized.
        {
            // This block is an exact copy of `self.set_len`.
            // We have to do this so that Miri doesn't report a "Stacked
            // Borrows" rule violation. See PR/406

            // SAFETY: we have wrote the value to the address already
            unsafe {
                self.length.add(1);
            }
        }

        // SAFETY: `ptr` is aligned, non-null and points to the element
        // initialized         above; the borrow is tied to `&mut self`,
        // so it is exclusive.
        unsafe { &mut *ptr }
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        let length = self.len();
        if length == 0 {
            return None;
        }
        let new_len = length - 1;
        // SAFETY: new_len < length since length is non-zero and
        // we are returning ownership of the current value.
        unsafe {
            self.length.sub(1);
        }
        // SAFETY: this element was initialized and we just gave up ownership of
        // it, so we can give it away
        let value = unsafe { self.as_mut_ptr().add(new_len).read() };
        Some(value)
    }

    #[inline]
    pub fn pop_if(&mut self, predicate: impl FnOnce(&mut T) -> bool) -> Option<T> {
        let last = self.last_mut()?;
        if predicate(last) { self.pop() } else { None }
    }

    #[inline]
    pub fn append<const M: usize>(&mut self, other: &mut SmallVec<T, M, A>) {
        // can't overflow since both are smaller than isize::MAX and 2 *
        // isize::MAX < usize::MAX
        let length = self.len();
        let other_len = other.len();
        let total_len = length + other_len;
        if total_len > self.capacity() {
            self.reserve(other_len);
        }

        // SAFETY: see `Self::push`
        let ptr = unsafe { self.as_mut_ptr().add(length) };
        unsafe { other.set_len(0) }
        // SAFETY: we have a mutable reference to each vector and each uniquely
        // owns its memory. so the ranges can't overlap
        unsafe { copy_nonoverlapping(other.as_ptr(), ptr, other_len) };
        unsafe { self.length.add(other_len) }
    }

    #[inline]
    pub fn grow(&mut self, new_capacity: usize) {
        self.try_grow(new_capacity)
            .unwrap_or_else(SmallVecError::handle);
    }

    #[cold]
    pub fn try_grow(&mut self, new_capacity: usize) -> Result<(), SmallVecError> {
        if Self::IS_ZST {
            return Ok(());
        }

        let (length, on_heap) = self.length.parts();
        assert!(new_capacity >= length);

        if new_capacity > Self::inline_size() {
            // SAFETY: we checked all the preconditions
            let result = unsafe {
                self.raw
                    .try_grow_raw(self.length, new_capacity, &self.allocator)
            };

            if result.is_ok() {
                // SAFETY: the allocation succeeded, so self.raw.heap is now
                // active
                self.length.set_location::<true>();
            }
            result
        } else {
            // new_capacity <= Self::inline_size()
            if on_heap {
                unsafe {
                    // SAFETY: heap member is active
                    let (ptr, old_cap) = self.raw.heap;
                    // inline member is now active

                    // SAFETY: length <= new_capacity <= Self::inline_size()
                    // so the copy is within bounds of the inline member
                    copy_nonoverlapping(ptr.as_ptr(), self.raw.as_mut_ptr_inline(), length);
                    drop(DropDealloc {
                        ptr: ptr.cast(),
                        size_bytes: old_cap * size_of::<T>(),
                        align: align_of::<T>(),
                        allocator: &self.allocator
                    });
                    self.length.set_location::<false>();
                }
            }
            Ok(())
        }
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.try_reserve(additional)
            .unwrap_or_else(SmallVecError::handle);
    }

    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), SmallVecError> {
        if additional > self.capacity() - self.len() {
            let new_capacity = self
                .len()
                .checked_add(additional)
                .and_then(usize::checked_next_power_of_two)
                .ok_or(SmallVecError::CapacityOverflow)?;
            self.try_grow(new_capacity)
        } else {
            Ok(())
        }
    }

    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.try_reserve_exact(additional)
            .unwrap_or_else(SmallVecError::handle);
    }

    #[inline]
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), SmallVecError> {
        if additional > self.capacity() - self.len() {
            let new_capacity = self
                .len()
                .checked_add(additional)
                .ok_or(SmallVecError::CapacityOverflow)?;
            self.try_grow(new_capacity)
        } else {
            Ok(())
        }
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        let (length, on_heap) = self.length.parts();
        if !on_heap {
            return;
        }
        if length <= Self::inline_size() {
            // SAFETY: on_heap is true, so we're on the heap
            unsafe {
                let (ptr, capacity) = self.raw.heap;
                copy_nonoverlapping(ptr.as_ptr(), self.raw.as_mut_ptr_inline(), length);
                self.length.set_location::<false>();
                self.allocator.deallocate(
                    ptr.cast(),
                    Layout::from_size_align_unchecked(capacity * size_of::<T>(), align_of::<T>())
                );
            }
        } else if length < self.capacity() {
            // SAFETY: length > Self::inline_size() >= 0
            // so new capacity is non zero, it is equal to the length
            // T can't be a ZST because SmallVec<ZST, N> is never spilled.
            unsafe {
                self.raw
                    .try_grow_raw(self.length, length, &self.allocator)
                    .unwrap_or_else(SmallVecError::handle)
            };
        }
    }

    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        let (length, on_heap) = self.length.parts();
        if !on_heap {
            return;
        }
        // SAFETY: the vector is on the heap
        let capacity = unsafe { self.raw.heap.1 };
        if capacity > min_capacity {
            let target = core::cmp::max(length, min_capacity);
            if target <= Self::inline_size() {
                // SAFETY: on_heap is true, so we're on the heap
                unsafe {
                    let (ptr, capacity) = self.raw.heap;
                    copy_nonoverlapping(ptr.as_ptr(), self.raw.as_mut_ptr_inline(), length);
                    self.length.set_location::<false>();
                    self.allocator.deallocate(
                        ptr.cast(),
                        Layout::from_size_align_unchecked(
                            capacity * size_of::<T>(),
                            align_of::<T>()
                        )
                    );
                }
            } else if target < capacity {
                // SAFETY: length > Self::inline_size() >= 0
                // so new capacity is non zero, it is equal to the length
                // T can't be a ZST because SmallVec<ZST, N> is never spilled.
                unsafe {
                    self.raw
                        .try_grow_raw(self.length, target, &self.allocator)
                        .unwrap_or_else(SmallVecError::handle)
                };
            }
        }
    }

    #[inline]
    pub fn truncate(&mut self, length: usize) {
        let old_len = self.len();
        if length < old_len {
            // SAFETY: we set `length` to a smaller value
            // then we drop the previously initialized elements
            unsafe {
                self.set_len(length);
                core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(
                    self.as_mut_ptr().add(length),
                    old_len - length
                ))
            }
        }
    }

    #[inline]
    pub fn swap_remove(&mut self, index: usize) -> T {
        #[cold]
        #[inline(never)]
        #[track_caller]
        fn assert_failed(index: usize, length: usize) -> ! {
            panic!("swap_remove index (is {index}) should be < length (is {length})");
        }

        let length = self.len();
        if index >= length {
            assert_failed(index, length);
        }
        // This can't overflow since `length > index >= 0`
        let new_len = length - 1;
        unsafe {
            // We replace self[index] with the last element. Note that if the
            // bounds check above succeeds there must be a last element (which
            // can be self[index] itself).
            let value = core::ptr::read(self.as_ptr().add(index));
            let base_ptr = self.as_mut_ptr();
            core::ptr::copy(base_ptr.add(new_len), base_ptr.add(index), 1);
            self.length.sub(1);
            value
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        // SAFETY: we set `length` to a smaller value
        // then we drop the previously initialized elements
        unsafe {
            let old_len = self.len();
            self.set_len(0);
            core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(
                self.as_mut_ptr(),
                old_len
            ));
        }
    }

    #[inline]
    pub fn remove(&mut self, index: usize) -> T {
        #[cold]
        #[inline(never)]
        #[track_caller]
        fn assert_failed(index: usize, length: usize) -> ! {
            panic!("removal index (is {index}) should be < length (is {length})");
        }

        let length = self.len();
        if index >= length {
            assert_failed(index, length);
        }
        let new_len = length - 1;
        unsafe {
            // SAFETY: new_len < length
            self.length.sub(1);
            let ptr = self.as_mut_ptr();
            let ith = ptr.add(index);
            // This item is initialized since index < length
            let ith_item = ith.read();
            copy(ith.add(1), ith, new_len - index);
            ith_item
        }
    }

    #[inline]
    pub fn insert(&mut self, index: usize, value: T) {
        _ = self.insert_mut(index, value);
    }

    #[inline]
    #[must_use]
    pub fn insert_mut(&mut self, index: usize, value: T) -> &mut T {
        #[cold]
        #[inline(never)]
        #[track_caller]
        fn assert_failed(index: usize, length: usize) -> ! {
            panic!("insertion index (is {index}) should be <= length (is {length})");
        }

        let length = self.len();
        if index > length {
            assert_failed(index, length);
        }
        self.reserve(1);
        debug_assert!(length < self.capacity());

        // SAFETY: `index <= length <= capacity`,
        //         so the offset stays in bounds of the allocation.
        let ptr = unsafe { self.as_mut_ptr().add(index) };

        if index < length {
            // SAFETY: `reserve(1)` guarantees capacity for `length + 1`
            // elements,         so shifting `length - index`
            // elements one slot up stays in bounds.         Source
            // and destination overlap, hence `copy` instead of
            // `copy_nonoverlapping`.
            unsafe { copy(ptr, ptr.add(1), length - index) };
        }

        // SAFETY: the slot at `index` is free and properly aligned for `T`.
        unsafe { ptr.write(value) };

        // LEGAL: all elements in `0..length + 1` are initialized.
        {
            // This block is an exact copy of `self.set_len`.
            // We have to do this so that Miri doesn't report a "Stacked
            // Borrows" rule violation. See PR/406

            // SAFETY: we have wrote the value to the address already
            unsafe {
                self.length.add(1);
            }
        }

        // SAFETY: `ptr` is aligned, non-null and points to the element
        // initialized         above; the borrow is tied to `&mut self`,
        // so it is exclusive.
        unsafe { &mut *ptr }
    }

    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        let (length, on_heap) = self.length.parts();
        // SAFETY: all the elements in `..length` are initialized
        unsafe { core::slice::from_raw_parts(self.raw.as_ptr(on_heap), length) }
    }

    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        let (length, on_heap) = self.length.parts();
        // SAFETY: see above
        unsafe { core::slice::from_raw_parts_mut(self.raw.as_mut_ptr(on_heap), length) }
    }

    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        // SAFETY: the tag tells which member is active
        unsafe { self.raw.as_ptr(self.length.on_heap()) }
    }

    #[inline]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        // SAFETY: see above
        unsafe { self.raw.as_mut_ptr(self.length.on_heap()) }
    }

    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        let (length, on_heap) = self.length.parts();
        if !on_heap {
            let mut vec = Vec::with_capacity(length);
            let this = ManuallyDrop::new(self);
            // SAFETY: we create a new vector with sufficient capacity, copy our
            // elements into it to transfer ownership and then set
            // the length we don't drop the elements we previously
            // held
            unsafe {
                copy_nonoverlapping(this.raw.as_ptr_inline(), vec.as_mut_ptr(), length);
                vec.set_len(length);
            }
            vec
        } else {
            let this = ManuallyDrop::new(self);
            // SAFETY:
            // - `ptr` was created with the global allocator
            // - `ptr` was created with the appropriate alignment for `T`
            // - the allocation pointed to by ptr is exactly cap * sizeof(T)
            // - `length` is less than or equal to `cap`
            // - the first `length` entries are proper `T`-values
            // - the allocation is not larger than `isize::MAX`
            unsafe {
                let (ptr, cap) = this.raw.heap;
                Vec::from_raw_parts(ptr.as_ptr(), length, cap)
            }
        }
    }

    #[inline]
    pub fn into_boxed_slice(self) -> Box<[T]> {
        self.into_vec().into_boxed_slice()
    }

    #[inline]
    #[deprecated(
        since = "2.0.0-alpha.13",
        note = "use `TryInto::<[T; N]>::try_into` instead"
    )]
    pub fn into_inner(self) -> Result<[T; N], Self> {
        if self.len() != N {
            Err(self)
        } else {
            // when `this` is dropped, the memory is released if it's on the
            // heap.
            let mut this = self;
            // SAFETY: we release ownership of the elements we hold
            unsafe {
                this.set_len(0);
            }
            let ptr = this.as_ptr() as *const [T; N];
            // SAFETY: these elements are initialized since the length was `N`
            unsafe { Ok(ptr.read()) }
        }
    }

    #[inline]
    pub fn retain<F: FnMut(&T) -> bool>(&mut self, mut f: F) {
        self.retain_mut(|elem| f(elem))
    }

    #[inline]
    pub fn retain_mut<F: FnMut(&mut T) -> bool>(&mut self, mut f: F) {
        let original_len = self.len();

        if original_len == 0 {
            // Empty case: explicit return allows better optimization, vs
            // letting compiler infer it
            return;
        }

        // Vec: [Kept, Kept, Hole, Hole, Hole, Hole, Unchecked, Unchecked]
        //      |            ^- write                ^- read             |
        //      |<-              original_len                          ->|
        // Kept: Elements which predicate returns true on.
        // Hole: Moved or dropped element slot.
        // Unchecked: Unchecked valid elements.
        //
        // This drop guard will be invoked when predicate or `drop` of element
        // panicked. It shifts unchecked elements to cover holes and
        // `set_len` to the correct length. In cases when predicate and
        // `drop` never panic, it will be optimized out.
        struct PanicGuard<'a, T, const N: usize, A: Allocator> {
            v: &'a mut SmallVec<T, N, A>,
            read: usize,
            write: usize,
            original_len: usize
        }

        impl<T, const N: usize, A: Allocator> Drop for PanicGuard<'_, T, N, A> {
            #[cold]
            fn drop(&mut self) {
                let remaining = self.original_len - self.read;
                // SAFETY: Trailing unchecked items must be valid since we never
                // touch them.
                unsafe {
                    let ptr = self.v.as_mut_ptr();
                    copy(ptr.add(self.read), ptr.add(self.write), remaining);
                }
                // SAFETY: After filling holes, all items are in contiguous
                // memory.
                unsafe {
                    self.v.set_len(self.write + remaining);
                }
            }
        }

        let mut read = 0;
        loop {
            // SAFETY: read < original_len
            let cur = unsafe { self.get_unchecked_mut(read) };
            if !f(cur) {
                break;
            }
            read += 1;
            if read == original_len {
                // All elements are kept, return early.
                return;
            }
        }

        // Critical section starts here and at least one element is going to be
        // removed. Advance `g.read` early to avoid double drop if
        // `drop_in_place` panicked.
        let mut g = PanicGuard {
            v: self,
            read: read + 1,
            write: read,
            original_len
        };
        // SAFETY: previous `read` is always less than original_len.
        unsafe { drop_in_place(g.v.as_mut_ptr().add(read)) }

        let ptr = g.v.as_mut_ptr();
        while g.read < g.original_len {
            // SAFETY: `read` is always less than original_len.
            let cur = unsafe { &mut *ptr.add(g.read) };
            if !f(cur) {
                // Advance `read` early to avoid double drop if `drop_in_place`
                // panicked.
                g.read += 1;
                // SAFETY: We never touch this element again after dropped.
                unsafe { drop_in_place(cur) };
            } else {
                // SAFETY: `read` > `write`, so the slots don't overlap.
                // We use copy for move, and never touch the source element
                // again.
                unsafe {
                    let hole = ptr.add(g.write);
                    copy_nonoverlapping(cur, hole, 1);
                }
                g.write += 1;
                g.read += 1;
            }
        }

        // We are leaving the critical section and no panic happened,
        // Commit the length change and forget the guard.
        // SAFETY: `write` is always less than or equal to original_len.
        unsafe { g.v.set_len(g.write) };
        core::mem::forget(g);
    }

    #[inline]
    pub fn dedup(&mut self)
    where T: PartialEq {
        self.dedup_by(|a, b| a == b);
    }

    #[inline]
    pub fn dedup_by_key<F, K>(&mut self, mut key: F)
    where
        F: FnMut(&mut T) -> K,
        K: PartialEq<K>
    {
        self.dedup_by(|a, b| key(a) == key(b));
    }

    #[inline]
    pub fn dedup_by<F>(&mut self, mut same_bucket: F)
    where F: FnMut(&mut T, &mut T) -> bool {
        // See the implementation of Vec::dedup_by in the
        // standard library for an explanation of this algorithm.
        let length = self.len();
        if length <= 1 {
            return;
        }

        let ptr = self.as_mut_ptr();
        let mut w: usize = 1;

        unsafe {
            for r in 1..length {
                let p_r = ptr.add(r);
                let p_wm1 = ptr.add(w - 1);
                if !same_bucket(&mut *p_r, &mut *p_wm1) {
                    if r != w {
                        let p_w = p_wm1.add(1);
                        core::ptr::swap(p_r, p_w);
                    }
                    w += 1;
                }
            }
        }

        self.truncate(w);
    }

    pub fn resize_with<F>(&mut self, new_len: usize, f: F)
    where F: FnMut() -> T {
        let old_len = self.len();
        if old_len < new_len {
            self.extend(core::iter::repeat_with(f).take(new_len - old_len));
        } else if old_len > new_len {
            self.truncate(new_len);
        }
    }

    pub fn leak<'a>(self) -> &'a mut [T] {
        let (length, on_heap) = self.length.parts();
        if !on_heap {
            panic!(
                "SmallVec::leak() called on inline (stack) SmallVec, which cannot be safely leaked"
            );
        }
        let mut me = ManuallyDrop::new(self);
        unsafe { core::slice::from_raw_parts_mut(me.raw.as_mut_ptr(true), length) }
    }

    /// Returns the remaining spare capacity of the vector as a slice of
    /// `MaybeUninit<T>`.
    ///
    /// The returned slice can be used to fill the vector with data (e.g. by
    /// reading from a file) before marking the data as initialized using the
    /// [`set_len`](Self::set_len) method.
    #[inline]
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        let (length, on_heap) = self.length.parts();
        unsafe {
            let capacity = self.raw.capacity(on_heap);
            core::slice::from_raw_parts_mut(
                self.raw.as_mut_ptr(on_heap).add(length) as *mut MaybeUninit<T>,
                capacity - length
            )
        }
    }

    /// Decomposes a `SmallVec<T, N>` into its raw components: `(pointer,
    /// length, capacity)`.
    ///
    /// Returns the raw pointer to the underlying data, the length of
    /// the vector (in elements), and the allocated capacity of the
    /// data (in elements). These are the same arguments in the same
    /// order as the arguments to [`from_raw_parts`].
    ///
    /// After calling this function, the caller is responsible for the
    /// memory previously managed by the `SmallVec`. Most often, one does
    /// this by converting the raw pointer, length, and capacity back
    /// into a `SmallVec` with the [`from_raw_parts`] function; more generally,
    /// if `T` is non-zero-sized and the capacity is nonzero, one may use
    /// any method that calls [`dealloc`] with a layout of
    /// `Layout::array::<T>(capacity)`; if `T` is zero-sized or the
    /// capacity is zero, nothing needs to be done.
    ///
    /// [`from_raw_parts`]: SmallVec::from_raw_parts
    /// [`dealloc`]: alloc::alloc::GlobalAlloc::dealloc
    ///
    /// # Panics
    ///
    /// This function will panic if the contents are not spilled on the heap.
    ///
    /// # Examples
    ///
    /// ```
    /// # use smallvec::SmallVec;
    ///
    /// let v: SmallVec<i32, 1> = SmallVec::from([-1, 0, 1]);
    ///
    /// let (ptr, length, cap) = v.into_raw_parts();
    ///
    /// let rebuilt = unsafe {
    ///     // We can now make changes to the components, such as
    ///     // transmuting the raw pointer to a compatible type.
    ///     let ptr = ptr as *mut u32;
    ///
    ///     SmallVec::<u32, 5>::from_raw_parts(ptr, length, cap)
    /// };
    /// assert_eq!(rebuilt, [4294967295, 0, 1]);
    /// ```
    #[inline]
    pub fn into_raw_parts(self) -> (*mut T, usize, usize) {
        #[cold]
        #[inline(never)]
        #[track_caller]
        fn assert_failed() -> ! {
            panic!(
                "SmallVec::into_raw_parts() called on inline (stack) SmallVec, \
                 which cannot be safely leaked"
            );
        }
        if !self.spilled() {
            assert_failed();
        }

        let mut me = ManuallyDrop::new(self);
        (me.as_mut_ptr(), me.len(), me.capacity())
    }

    #[inline]
    pub fn extend_from_slice_copy(&mut self, other: &[T])
    where T: Copy {
        let length = other.len();
        let src = other.as_ptr();

        let l = self.len();
        self.reserve(length);

        // SAFETY: Additional memory has been reserved,
        // therefore the pointer access is valid.
        unsafe {
            let dst = self.as_mut_ptr().add(l);
            copy_nonoverlapping(src, dst, length);
            self.length.add(length);
        }
    }

    pub fn extend_from_within_copy<R>(&mut self, src: R)
    where
        R: core::ops::RangeBounds<usize>,
        T: Copy
    {
        let src = core::ops::Range::new(src, self.len());
        let core::ops::Range {
            start,
            end
        } = src;
        let length = end - start;
        self.reserve(length);

        // SAFETY: The call to `reserve` ensures that the capacity is large
        // enough. The range is within bounds through the use of
        // `core::slice::range`.
        unsafe {
            let l = self.len();
            let ptr = self.as_mut_ptr();
            copy_nonoverlapping(ptr.add(start), ptr.add(l), length);
            self.length.add(length);
        }
    }

    pub fn insert_from_slice_copy(&mut self, index: usize, other: &[T])
    where T: Copy {
        let l = self.len();
        let length = other.len();
        assert!(index <= l);
        self.reserve(length);
        unsafe {
            let base_ptr = self.as_mut_ptr();
            let ith_ptr = base_ptr.add(index);
            let shifted_ptr = base_ptr.add(index + length);
            // elements at `index + other_len..length + other_len` are now
            // initialized
            copy(ith_ptr, shifted_ptr, l - index);
            // elements at `index..index + other_len` are now initialized
            copy_nonoverlapping(other.as_ptr(), ith_ptr, length);

            // SAFETY: all the elements are initialized
            self.length.add(length);
        }
    }

    pub const fn new_in(allocator: A) -> SmallVec<T, N, A> {
        Self {
            length: TaggedLen::new(0, false),
            raw: RawSmallVec::new(),
            allocator
        }
    }

    pub fn try_with_capacity_in(capacity: usize, allocator: A) -> Result<Self, SmallVecError> {
        let mut this = Self::new_in(allocator);
        if capacity > Self::inline_size() && !Self::IS_ZST {
            // SAFETY: we checked all the preconditions
            unsafe {
                this.raw
                    .try_grow_raw(TaggedLen::new(0, false), capacity, &this.allocator)
            }?;

            // SAFETY: the allocation succeeded, so self.raw.heap is now active
            this.length.set_location::<true>();
        }
        Ok(this)
    }

    pub fn with_capacity_in(capacity: usize, allocator: A) -> Self {
        Self::try_with_capacity_in(capacity, allocator).unwrap_or_else(SmallVecError::handle)
    }
}

impl<T: Clone, const N: usize, A: Allocator> SmallVec<T, N, A> {
    #[inline]
    pub fn resize(&mut self, length: usize, value: T) {
        let old_len = self.len();
        if length > old_len {
            self.extend(core::iter::repeat_n(value, length - old_len));
        } else {
            self.truncate(length);
        }
    }

    #[inline]
    pub fn extend_from_slice(&mut self, other: &[T]) {
        self.extend(other.iter())
    }

    pub fn extend_from_within<R>(&mut self, src: R)
    where R: core::ops::RangeBounds<usize> {
        let src = core::ops::Range::new(src, self.len());
        self.reserve(src.len());

        // SAFETY: The call to `reserve` ensures that the capacity is large
        // enough. The range is within bounds through the use of
        // `core::slice::range`.
        unsafe {
            #[cfg(feature = "specialization")]
            {
                <Self as specialization::SpecExtendFromWithin<T>>::spec_extend_from_within(
                    self, src
                );
            }

            #[cfg(not(feature = "specialization"))]
            {
                self.extend_from_within_fallback(src);
            }
        }
    }
}

impl<T, const N: usize, A: Allocator + Clone> SmallVec<T, N, A> {
    /// Splits the collection into two at the given index.
    ///
    /// Returns a newly allocated vector containing the elements in the range
    /// `[at, length)`. After the call, the original vector will be left
    /// containing the elements `[0, at)` with its previous capacity
    /// unchanged.
    ///
    /// - If you want to take ownership of the entire contents and capacity of
    ///   the vector, see [`core::mem::take`] or [`core::mem::replace`].
    /// - If you don't need the returned vector at all, see
    ///   [`SmallVec::truncate`].
    /// - If you want to take ownership of an arbitrary subslice, or you don't
    ///   necessarily want to store the removed items in a vector, see
    ///   [`SmallVec::drain`].
    ///
    /// # Panics
    ///
    /// Panics if `at > length`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1, 2, 3];
    /// let vec2 = vec.split_off(1);
    /// assert_eq!(vec, [1]);
    /// assert_eq!(vec2, [2, 3]);
    /// ```
    #[inline]
    pub fn split_off(&mut self, at: usize) -> Self {
        let length = self.len();
        assert!(at <= length);

        let other_len = length - at;
        let mut other = Self::with_capacity_in(other_len, self.allocator.clone());

        // Unsafely `set_len` and copy items to `other`.
        unsafe {
            self.set_len(at);
            other.set_len(other_len);

            core::ptr::copy_nonoverlapping(self.as_ptr().add(at), other.as_mut_ptr(), other_len);
        }
        other
    }
}

struct DropGuard<T> {
    ptr: *mut T,
    length: usize
}
impl<T> Drop for DropGuard<T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            core::ptr::slice_from_raw_parts_mut(self.ptr, self.length).drop_in_place();
        }
    }
}

struct DropDealloc<'a, A: Allocator> {
    ptr: NonNull<u8>,
    size_bytes: usize,
    align: usize,
    allocator: &'a A
}

impl<A: Allocator> Drop for DropDealloc<'_, A> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            if self.size_bytes > 0 {
                self.allocator.deallocate(
                    self.ptr,
                    Layout::from_size_align_unchecked(self.size_bytes, self.align)
                );
            }
        }
    }
}

impl<T, const N: usize, A: Allocator> Drop for SmallVec<T, N, A> {
    fn drop(&mut self) {
        let (length, on_heap) = self.length.parts();
        // SAFETY: the tag tells which member is active
        let ptr = unsafe { self.raw.as_mut_ptr(on_heap) };
        // SAFETY: see above
        unsafe {
            let _drop_dealloc = if on_heap {
                let capacity = self.raw.heap.1;
                Some(DropDealloc {
                    ptr: NonNull::new_unchecked(ptr as *mut u8),
                    size_bytes: capacity * size_of::<T>(),
                    align: align_of::<T>(),
                    allocator: &self.allocator
                })
            } else {
                None
            };
            core::ptr::slice_from_raw_parts_mut(ptr, length).drop_in_place();
        }
    }
}

/// This function is used in the [`smallvec`] macro.
/// It is recommended to use the macro instead of using this function.
#[doc(hidden)]
#[track_caller]
pub fn from_elem<T: Clone, const N: usize>(elem: T, n: usize) -> SmallVec<T, N> {
    if n > SmallVec::<T, N>::inline_size() {
        // Standard Rust vectors are already specialized.
        SmallVec::from_vec(vec![elem; n])
    } else {
        #[cfg(feature = "specialization")]
        {
            // SAFETY: The precondition is checked in the initial comparison
            // above.
            unsafe { <SmallVec<T, N> as specialization::SpecFromElem<T>>::spec_from_elem(elem, n) }
        }

        #[cfg(not(feature = "specialization"))]
        {
            // SAFETY: The precondition is checked in the initial comparison
            // above.
            unsafe { SmallVec::<T, N>::from_elem_fallback(elem, n) }
        }
    }
}

/// Fallback functions for various specialized methods. These are kept in
/// a separate implementation block for easy access whenever specialization is
/// disabled.
impl<T, const N: usize> SmallVec<T, N> {
    /// Creates a `Smallvec` value where `elem` is repeated `n` times.
    /// This will use the inline storage, not the heap.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `n <= Self::inline_size()`.
    unsafe fn from_elem_fallback(elem: T, n: usize) -> Self
    where T: Clone {
        let mut result = Self::new();

        if n > 0 {
            let ptr = result.raw.as_mut_ptr_inline();
            let mut guard = DropGuard {
                ptr,
                length: 0
            };

            // SAFETY: The caller ensures that the first `n`
            // is smaller than the inline size.
            unsafe {
                for i in 0..(n - 1) {
                    ptr.add(i).write(elem.clone());
                    guard.length += 1;
                }
                core::mem::forget(guard);
                ptr.add(n - 1).write(elem);
            }
        }

        // SAFETY: The first `n` elements of the vector
        // have been initialized in the loop above.
        unsafe {
            result.set_len(n);
        }

        result
    }

    fn from_iter_fallback<I>(iter: I) -> Self
    where I: Iterator<Item = T> {
        let (size, _) = iter.size_hint();
        let mut v = Self::with_capacity(size);
        v.extend_fallback(iter);
        v
    }

    /// Creates a `SmallVec` value based on the contents of `slice`.
    /// This will use the inline storage, not the heap.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `slice.len() <= Self::inline_size()`.
    unsafe fn from_slice_fallback(slice: &[T]) -> Self
    where T: Clone {
        let mut v = Self::new();

        let src = slice.as_ptr();
        let length = slice.len();
        let dst = v.as_mut_ptr();

        // SAFETY: The caller ensures that the slice length is smaller
        // than or equal to the inline length.
        unsafe {
            let mut guard = DropGuard {
                ptr: dst,
                length: 0
            };
            for i in 0..length {
                let val = (*src.add(i)).clone();
                dst.add(i).write(val);
                guard.length += 1;
            }
            core::mem::forget(guard);
        }

        // SAFETY: The elements were initialized in the loop above.
        unsafe {
            v.set_len(length);
        }

        v
    }
}

impl<T, const N: usize, A: Allocator> SmallVec<T, N, A> {
    fn extend_fallback<I>(&mut self, iter: I)
    where I: IntoIterator<Item = T> {
        let mut iterator = iter.into_iter();
        while let Some(element) = iterator.next() {
            let length = self.len();
            if length == self.capacity() {
                let (lower, _) = iterator.size_hint();
                self.reserve(lower.saturating_add(1));
            }
            unsafe {
                core::ptr::write(self.as_mut_ptr().add(length), element);
                // Since next() executes user code which can panic we have to
                // bump the length after each step.
                // NB can't overflow since we would have had to alloc the
                // address space
                self.length.add(1);
            }
        }
    }

    /// Main worker for [`extend_from_within`].
    ///
    /// # Safety
    ///
    /// * The length of the vector is larger than or equal to `src.len()`.
    /// * The spare capacity of the vector is larger than or equal to
    ///   `src.len()`.
    ///
    /// [`extend_from_within`]: SmallVec::extend_from_within
    unsafe fn extend_from_within_fallback(&mut self, src: core::ops::Range<usize>)
    where T: Clone {
        let old_len = self.len();

        let start = src.start;
        let length = src.len();

        // SAFETY: The caller ensures that the vector has spare capacity
        // for at least `src.len()` elements. This implies that the loop
        // operates on valid memory.
        unsafe {
            let ptr = self.as_mut_ptr();
            let dst = ptr.add(old_len);
            let src = ptr.add(start);

            let mut guard = DropGuard {
                ptr: dst,
                length: 0
            };
            for i in 0..length {
                let val = (*src.add(i)).clone();
                dst.add(i).write(val);
                guard.length += 1;
            }
            core::mem::forget(guard);
        }

        // SAFETY: The elements were initialized in the loop above.
        unsafe {
            self.length.add(length);
        }
    }

    fn clone_from_fallback(&mut self, source: &[T])
    where T: Clone {
        // Inspired from `impl Clone for Vec`.

        // Drop anything that will not be overwritten.
        self.truncate(source.len());

        // SAFETY: self.length <= other.length due to the truncate above, so the
        // slices here are always in-bounds.
        let (init, tail) = unsafe { source.split_at_unchecked(self.len()) };

        // Reuse the contained values' allocations/resources.
        self.clone_from_slice(init);
        self.extend(tail.iter().cloned());
    }
}

impl<T: Clone, const N: usize, A: Allocator + Clone> Clone for SmallVec<T, N, A> {
    #[inline]
    fn clone(&self) -> SmallVec<T, N, A> {
        let mut vec = SmallVec {
            length: TaggedLen::new(0, false),
            raw: RawSmallVec::new(),
            allocator: self.allocator.clone()
        };

        vec.extend(self);

        vec
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        #[cfg(feature = "specialization")]
        {
            <Self as specialization::SpecCloneFrom<T>>::spec_clone_from(self, source);
        }

        #[cfg(not(feature = "specialization"))]
        {
            self.clone_from_fallback(source);
        }
    }
}

impl<T, const N: usize, A: Allocator> Extend<T> for SmallVec<T, N, A> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        #[cfg(feature = "specialization")]
        {
            specialization::SpecExtend::<T, _>::spec_extend(self, iter.into_iter());
        }

        #[cfg(not(feature = "specialization"))]
        {
            self.extend_fallback(iter);
        }
    }
}

impl<'a, T: Clone + 'a, const N: usize, A: Allocator> Extend<&'a T> for SmallVec<T, N, A> {
    #[inline]
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        #[cfg(feature = "specialization")]
        {
            specialization::SpecExtend::<&'a T, _>::spec_extend(self, iter.into_iter());
        }

        #[cfg(not(feature = "specialization"))]
        {
            self.extend_fallback(iter.into_iter().cloned());
        }
    }
}

impl<T, const N: usize> core::iter::FromIterator<T> for SmallVec<T, N> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        #[cfg(feature = "specialization")]
        {
            specialization::SpecFromIterator::<T, _>::spec_from_iter(iter.into_iter())
        }

        #[cfg(not(feature = "specialization"))]
        {
            Self::from_iter_fallback(iter.into_iter())
        }
    }
}

impl<'a, T, const N: usize, A: Allocator> IntoIterator for &'a SmallVec<T, N, A> {
    type IntoIter = core::slice::Iter<'a, T>;
    type Item = &'a T;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const N: usize, A: Allocator> IntoIterator for &'a mut SmallVec<T, N, A> {
    type IntoIter = core::slice::IterMut<'a, T>;
    type Item = &'a mut T;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T: Hash, const N: usize, A: Allocator> Hash for SmallVec<T, N, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state)
    }
}

impl<T: Debug, const N: usize, A: Allocator> Debug for SmallVec<T, N, A> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

#[cfg(feature = "arbitrary")]
#[cfg_attr(docsrs, doc(cfg(feature = "arbitrary")))]
impl<'a, T, const N: usize> arbitrary::Arbitrary<'a> for SmallVec<T, N>
where T: arbitrary::Arbitrary<'a>
{
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        u.arbitrary_iter()?.collect()
    }

    fn arbitrary_take_rest(u: arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        u.arbitrary_take_rest_iter()?.collect()
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        arbitrary::size_hint::and(<usize as arbitrary::Arbitrary>::size_hint(depth), (0, None))
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
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

#[cfg(feature = "bytes")]
unsafe impl<const N: usize> BufMut for SmallVec<u8, N> {
    #[inline]
    fn remaining_mut(&self) -> usize {
        // A vector can never have more than isize::MAX bytes
        isize::MAX as usize - self.len()
    }

    #[inline]
    unsafe fn advance_mut(&mut self, cnt: usize) {
        let length = self.len();
        let remaining = self.capacity() - length;

        if remaining < cnt {
            panic!("advance out of bounds: the length is {remaining} but advancing by {cnt}");
        }

        // Addition will not overflow since the sum is at most the capacity.
        unsafe { self.length.add(cnt) };
    }

    #[inline]
    fn chunk_mut(&mut self) -> &mut UninitSlice {
        if self.capacity() == self.len() {
            self.reserve(64); // Grow the smallvec
        }

        let (length, on_heap) = self.length.parts();
        // SAFETY: the tag tells which member is active
        let cap = unsafe { self.raw.capacity(on_heap) };
        let ptr = unsafe { self.raw.as_mut_ptr(on_heap) };
        // SAFETY: Since `ptr` is valid for `cap` bytes, `ptr.add(length)` must
        // be valid for `cap - length` bytes. The subtraction will not
        // underflow since `length <= cap`.
        unsafe { UninitSlice::from_raw_parts_mut(ptr.add(length), cap - length) }
    }

    // Specialize these methods so they can skip checking `remaining_mut`
    // and `advance_mut`.
    #[inline]
    fn put<T: bytes::Buf>(&mut self, mut src: T)
    where Self: Sized {
        // In case the src isn't contiguous, reserve upfront.
        self.reserve(src.remaining());

        while src.has_remaining() {
            let s = src.chunk();
            let l = s.len();
            self.extend_from_slice(s);
            src.advance(l);
        }
    }

    #[inline]
    fn put_slice(&mut self, src: &[u8]) {
        self.extend_from_slice(src);
    }

    #[inline]
    fn put_bytes(&mut self, val: u8, cnt: usize) {
        // If the addition overflows, then the `resize` will fail.
        let new_len = self.len().saturating_add(cnt);
        self.resize(new_len, val);
    }
}

#[cfg(feature = "defmt")]
impl<T: Format, const N: usize> Format for SmallVec<T, N> {
    fn format(&self, fmt: DeFormatter) {
        dewrite!(fmt, "{=[?]}", self.as_ref());
    }
}

#[cfg(feature = "encase")]
encase::rts_array::impl_rts_array!(SmallVec<T, N>; (T, const N: usize); using len truncate);
