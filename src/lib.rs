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
//! ### `std`
//!
//! When this feature is enabled, traits available from `std` are implemented:
//!
//! * `SmallVec<u8, _>` implements the [`std::io::Write`] trait.
//! * [`CollectionAllocErr`] implements [`std::error::Error`].
//!
//! This feature is not compatible with `#![no_std]` programs.
//!
//! ### `serde`
//!
//! When this optional dependency is enabled, `SmallVec` implements the `serde::Serialize` and
//! `serde::Deserialize` traits.
//!
//! ### `extract_if`
//!
//! **This feature is unstable.** It may change to match the unstable `extract_if` method in libstd.
//!
//! Enables the `extract_if` method, which produces an iterator that calls a user-provided
//! closure to determine which elements of the vector to remove and yield from the iterator.
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
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(feature = "allocator_api", feature(allocator_api))]
#![cfg_attr(feature = "specialization", allow(incomplete_features))]
#![cfg_attr(feature = "specialization", feature(specialization, trusted_len))]
#![cfg_attr(feature = "may_dangle", feature(dropck_eyepatch))]

#[doc(hidden)]
pub extern crate alloc;

#[cfg(any(test, feature = "std"))]
extern crate std;

#[cfg(test)]
mod tests;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

#[cfg(feature = "allocator_api")]
use alloc::alloc::{Global, Allocator};

use alloc::alloc::Layout;
use core::borrow::Borrow;
use core::borrow::BorrowMut;
use core::fmt::Debug;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::mem::align_of;
use core::mem::size_of;
use core::mem::ManuallyDrop;
use core::mem::MaybeUninit;
use core::ptr::addr_of;
use core::ptr::addr_of_mut;
use core::ptr::copy;
use core::ptr::copy_nonoverlapping;
use core::ptr::NonNull;

#[cfg(feature = "bytes")]
use bytes::{buf::UninitSlice, BufMut};
#[cfg(feature = "malloc_size_of")]
use malloc_size_of::{MallocShallowSizeOf, MallocSizeOf, MallocSizeOfOps};
#[cfg(feature = "serde")]
use serde_core::{
    de::{Deserialize, Deserializer, SeqAccess, Visitor},
    ser::{Serialize, SerializeSeq, Serializer},
};
#[cfg(feature = "std")]
use std::io;

/// A macro that conditionally discards the last type parameter of its input depending on the "allocator_api" feature.
/// This is used to control the presence of the allocator parameter.
#[cfg(not(feature = "allocator_api"))]
macro_rules! alloc_param {
    // While a more generic version of this macro is likely possible, it isn't needed for the puporses of this crate.
    ($type:ident<$arg0:tt>) => {
        $type
    };
    ($type:ident<$arg0:tt, $arg1:tt $(,)?>) => {
        $type<$arg0>
    };
    ($type:ident<$arg0:tt, $arg1:tt, $arg2:tt $(,)?>) => {
        $type<$arg0, $arg1>
    };
    ($type:ident<$arg0:tt, $arg1:tt, $arg2:tt, $arg3:tt $(,)?>) => {
        $type<$arg0, $arg1, $arg2>
    };
    ($type:ident<$arg0:tt, $path:ident::$arg1:tt, $arg2:tt, $arg3:tt $(,)?>) => {
        $type<$arg0, $path::$arg1, $arg2>
    };
    ($type:ident<$arg0:tt, $arg1:tt, $arg2:tt, $arg3:tt, $arg4:tt $(,)?>) => {
        $type<$arg0, $arg1, $arg2, $arg3>
    };
}

#[cfg(feature = "allocator_api")]
macro_rules! alloc_param {
    // This will return the tokens without change.
    ($($args:tt)+) => {
        $($args)*
    };
}

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
impl core::error::Error for CollectionAllocErr {}

#[inline]
#[track_caller]
fn infallible<T>(result: Result<T, CollectionAllocErr>) -> T {
    match result {
        Ok(x) => x,
        Err(CollectionAllocErr::CapacityOverflow) => panic!("capacity overflow"),
        Err(CollectionAllocErr::AllocErr { layout }) => alloc::alloc::handle_alloc_error(layout),
    }
}

#[inline]
/// A local copy of [`core::slice::range`]. The latter function is unstable
/// and thus cannot be used yet.
fn slice_range<R>(range: R, bounds: core::ops::RangeTo<usize>) -> core::ops::Range<usize>
where
    R: core::ops::RangeBounds<usize>,
{
    let len = bounds.end;

    let start = match range.start_bound() {
        core::ops::Bound::Included(&start) => start,
        core::ops::Bound::Excluded(start) => start
            .checked_add(1)
            .unwrap_or_else(|| panic!("attempted to index slice from after maximum usize")),
        core::ops::Bound::Unbounded => 0,
    };

    let end = match range.end_bound() {
        core::ops::Bound::Included(end) => end
            .checked_add(1)
            .unwrap_or_else(|| panic!("attempted to index slice up to maximum usize")),
        core::ops::Bound::Excluded(&end) => end,
        core::ops::Bound::Unbounded => len,
    };

    if start > end {
        panic!("slice index starts at {start} but ends at {end}");
    }
    if end > len {
        panic!("range end index {end} out of range for slice of length {len}");
    }

    core::ops::Range { start, end }
}

#[inline(always)]
const fn is_zst<T>() -> bool {
    size_of::<T>() == 0
}

#[inline(always)]
const fn inline_size<T, const N: usize>() -> usize {
    if is_zst::<T>() {
        usize::MAX
    } else {
        N
    }
}

/// Creates a [`Layout`] values for arrays of length `n`
/// for a given type without checking preconditions.
///
/// # Safety
///
/// The caller must ensure that an array of length `n` results
/// in a valid layout.
#[inline(always)]
const unsafe fn array_layout_unchecked<T>(n: usize) -> Layout {
    // SAFETY: The caller ensures that the an array of length `n` is possible
    // which means that the multiplication can't overflow.
    // The value returned by `align_of` will fulfill the safety conditions for
    // `Layout::from_size_align_unchecked`.
    unsafe {
        Layout::from_size_align_unchecked(size_of::<T>().unchecked_mul(n), align_of::<T>())
    }
}

/// A ZST from which memory allocation methods are exposed.
/// The methods correspond to the `allocator_api` Rust nightly feature.
#[derive(Clone, Copy)]
#[cfg(not(feature = "allocator_api"))]
struct A;

#[cfg(not(feature = "allocator_api"))]
impl A {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, CollectionAllocErr> {
        let err = CollectionAllocErr::AllocErr { layout };
        if layout.size() == 0 {
            Err(err)
        } else {
            // SAFETY: The size of the layout has been checked in the comparison above.
            unsafe {
                let ptr = alloc::alloc::alloc(layout);
                let ptr = NonNull::new(ptr).ok_or(err)?;
                Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
            }
        }
    }

    #[inline(always)]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() > 0 {
            // SAFETY: Safety conditions are identical.
            unsafe {
                alloc::alloc::dealloc(ptr.as_ptr(), layout);
            }
        }
    }

    /// # Safety
    ///
    /// The new layout must have a size that is larger than or equal to
    /// the size of the old layout.
    #[inline(always)]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, CollectionAllocErr> {
        debug_assert!(
            new_layout.size() >= old_layout.size(),
            "`new_layout.size()` must be greater than or equal to `old_layout.size()`"
        );

        let old_size = old_layout.size();
        let old_align = old_layout.align();
        let new_size = new_layout.size();
        let new_align = new_layout.align();

        // SAFETY: Safety condition is upheld by the caller.
        unsafe {
            core::hint::assert_unchecked(new_size >= old_size);
        }

        let err = CollectionAllocErr::AllocErr { layout: new_layout };

        if old_size == 0 {
            self.allocate(new_layout)
        } else if old_align == new_align {
            // SAFETY: The caller ensures that the layout is valid.
            // The pointers returned by the allocation methods are valid up to `new_size`.
            unsafe {
                let new_ptr = alloc::alloc::realloc(ptr.as_ptr(), old_layout, new_size);
                if new_ptr.is_null() {
                    return Err(err);
                }
                let new_ptr = NonNull::new(new_ptr).ok_or(err)?;
                Ok(NonNull::slice_from_raw_parts(new_ptr, new_size))
            }
        } else {
            // SAFETY: The caller ensures that the layout is valid.
            // The pointers returned by the allocation methods are valid up to `new_size`.
            unsafe {
                let new_ptr = self.allocate(new_layout)?;
                let raw = new_ptr.as_ptr().cast::<u8>();
                copy_nonoverlapping(ptr.as_ptr(), raw, old_size);
                Ok(new_ptr)
            }
        }
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, CollectionAllocErr> {
        debug_assert!(
            new_layout.size() <= old_layout.size(),
            "`new_layout.size()` must be smaller than or equal to `old_layout.size()`"
        );

        let old_size = old_layout.size();
        let old_align = old_layout.align();
        let new_size = new_layout.size();
        let new_align = new_layout.align();

        // SAFETY: The safety condition is upheld by the caller.
        unsafe {
            core::hint::assert_unchecked(new_size <= old_size);
        }

        let err = CollectionAllocErr::AllocErr { layout: new_layout };

        if new_size == 0 {
            self.deallocate(ptr, old_layout);
            // FIXME: Replace with `Layout::dangling` once the latter is stable.
            // This version should work for now wrt. provenance.
            let data = core::mem::transmute(new_align);
            Ok(NonNull::slice_from_raw_parts(data, 0))
        } else if old_align == new_align {
            // SAFETY: The caller ensures that the layout is valid.
            // The pointers returned by the allocation methods are valid up to `new_size`.
            unsafe {
                let new_ptr = alloc::alloc::realloc(ptr.as_ptr(), old_layout, new_size);
                if new_ptr.is_null() {
                    return Err(err);
                }
                let new_ptr = NonNull::new(new_ptr).ok_or(err)?;
                Ok(NonNull::slice_from_raw_parts(new_ptr, new_size))
            }
        } else {
            // SAFETY: The caller ensures that the layout is valid.
            // The pointers returned by the allocation methods are valid up to `new_size`.
            unsafe {
                let new_ptr = self.allocate(new_layout)?;
                let raw = new_ptr.as_ptr().cast::<u8>();
                copy_nonoverlapping(ptr.as_ptr(), raw, new_size);
                Ok(new_ptr)
            }
        }
    }
}

#[cfg(not(feature = "allocator_api"))]
const GLOBAL: A = A;

#[cfg(feature = "allocator_api")]
const GLOBAL: Global = Global;

struct DropDealloc<'a, #[cfg(feature = "allocator_api")] A: Allocator> {
    ptr: NonNull<u8>,
    layout: Layout,
    allocator: &'a A,
}
impl<#[cfg(feature = "allocator_api")] A: Allocator> Drop for alloc_param!(DropDealloc<'_, A>) {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.allocator.deallocate(self.ptr, self.layout);
        }
    }
}

/// Either a stack array with `length <= N` or a heap array
/// whose pointer and capacity are stored here.
///
/// We store a `NonNull<T>` instead of a `*mut T`, so that
/// niche-optimization can be performed and the type is covariant
/// with respect to `T`.
#[repr(C)]
pub union RawSmallVecUnion<T, const N: usize> {
    inline: ManuallyDrop<MaybeUninit<[T; N]>>,
    heap: (NonNull<T>, usize),
}

impl<T, const N: usize> RawSmallVecUnion<T, N> {
    #[inline]
    const fn new_inline(inline: MaybeUninit<[T; N]>) -> Self {
        Self {
            inline: ManuallyDrop::new(inline),
        }
    }

    #[inline]
    const fn new_heap(ptr: NonNull<T>, capacity: usize) -> Self {
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
    const fn as_mut_ptr_inline(&mut self) -> *mut T {
        // SAFETY: See above.
        (unsafe { addr_of_mut!(self.inline) }) as *mut T
    }

    /// # Safety
    ///
    /// The vector must be on the heap.
    #[inline]
    const unsafe fn as_ptr_heap(&self) -> *const T {
        // SAFETY: Safety conditions are identical.
        unsafe {
            self.heap.0.as_ptr()
        }
    }

    /// # Safety
    ///
    /// The vector must be on the heap.
    #[inline]
    const unsafe fn as_mut_ptr_heap(&mut self) -> *mut T {
        // SAFETY: Safety conditions are identical.
        unsafe {
            self.heap.0.as_ptr()
        }
    }
}

pub struct RawSmallVec<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator = Global> {
    pub inner: RawSmallVecUnion<T, N>,
    allocator: A,
}

impl<T, const N: usize> alloc_param!(RawSmallVec<T, N, Global>) {
    #[inline]
    pub const fn new() -> Self {
        Self::new_in(GLOBAL)
    }

    #[inline]
    pub const fn new_inline(inline: MaybeUninit<[T; N]>) -> Self {
        Self::new_inline_in(inline, GLOBAL)
    }

    #[inline]
    pub const fn new_heap(ptr: NonNull<T>, capacity: usize) -> Self {
        Self::new_heap_in(ptr, capacity, GLOBAL)
    }

    #[inline]
    #[track_caller]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, GLOBAL)
    }

    #[inline]
    pub fn try_with_capacity(capacity: usize) -> Result<Self, CollectionAllocErr> {
        Self::try_with_capacity_in(capacity, GLOBAL)
    }
}

impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> alloc_param!(RawSmallVec<T, N, A>) {
    /// Turn a generic allocation error returned by a parametric allocator into a [`CollectionAllocErr`].
    #[cfg(feature = "allocator_api")]
    #[inline(always)]
    fn handle_alloc_error<U>(r: Result<U, alloc::alloc::AllocError>, layout: Layout) -> Result<U, CollectionAllocErr> {
        r.map_err(|_| CollectionAllocErr::AllocErr { layout })
    }

    /// A fallback used whenever the "allocator_api" is disabled. All errors are already [`CollectionAllocErr`] values,
    /// so nothing needs to be done.
    #[cfg(not(feature = "allocator_api"))]
    #[inline(always)]
    fn handle_alloc_error<U>(r: Result<U, CollectionAllocErr>, _layout: Layout) -> Result<U, CollectionAllocErr> {
        r
    }

    #[inline]
    const fn new_in(allocator: A) -> Self {
        Self::new_inline_in(MaybeUninit::uninit(), allocator)
    }

    #[inline]
    const fn new_inline_in(inline: MaybeUninit<[T; N]>, allocator: A) -> Self {
        Self {
            inner: RawSmallVecUnion::new_inline(inline),
            allocator,
        }
    }

    #[inline]
    const fn new_heap_in(ptr: NonNull<T>, capacity: usize, allocator: A) -> Self {
        Self {
            inner: RawSmallVecUnion::new_heap(ptr, capacity),
            allocator,
        }
    }

    #[inline]
    #[track_caller]
    fn with_capacity_in(capacity: usize, allocator: A) -> Self {
        infallible(Self::try_with_capacity_in(capacity, allocator))
    }

    #[inline]
    fn try_with_capacity_in(capacity: usize, allocator: A) -> Result<Self, CollectionAllocErr> {
        if capacity <= const { inline_size::<T, N>() } {
            Ok(Self::new_inline_in(MaybeUninit::uninit(), allocator))
        } else {
            let layout = Layout::array::<T>(capacity).map_err(|_| CollectionAllocErr::CapacityOverflow)?;
            let ptr = Self::handle_alloc_error(allocator.allocate(layout), layout)?;
            let inner = RawSmallVecUnion {
                heap: (ptr.cast(), capacity)
            };
            Ok(Self {
                inner,
                allocator,
            })
        }
    }

    /// Gets a pointer to the contents of the vector, under the assumption
    /// that the content is stored inline.
    #[inline]
    pub const fn as_ptr_inline(&self) -> *const T {
        self.inner.as_ptr_inline()
    }

    /// Gets a pointer to the contents of the vector, under the assumption
    /// that the content is stored inline.
    #[inline]
    pub const fn as_mut_ptr_inline(&mut self) -> *mut T {
        self.inner.as_mut_ptr_inline()
    }

    /// Gets a pointer to the contents of the vector, under the assumption
    /// that the content is stored on the heap.
    ///
    /// # Safety
    ///
    /// The vector must be on the heap.
    #[inline]
    pub const unsafe fn as_ptr_heap(&self) -> *const T {
        // SAFETY: The safety requirements are identical.
        unsafe {
            self.inner.as_ptr_heap()
        }
    }

    /// Gets a pointer to the contents of the vector, under the assumption
    /// that the content is stored on the heap.
    ///
    /// # Safety
    ///
    /// The vector must be on the heap.
    #[inline]
    pub const unsafe fn as_mut_ptr_heap(&mut self) -> *mut T {
        // SAFETY: The safety requirements are identical.
        unsafe {
            self.inner.as_mut_ptr_heap()
        }
    }

    /// Returns `true` if the elements are stored on the heap, and `false` otherwise.
    ///
    /// # Safety
    ///
    /// The way elements are stored in `self` must correspond to the tag in `len`.
    unsafe fn try_reserve(
        &mut self,
        len: TaggedLen,
        additional: usize,
    ) -> Result<bool, CollectionAllocErr> {
        debug_assert!(!is_zst::<T>());

        let on_heap = len.on_heap(is_zst::<T>());
        let len = len.value(is_zst::<T>());

        if additional == 0 {
            return Ok(on_heap);
        }

        let new_capacity = len.checked_add(additional).ok_or(CollectionAllocErr::CapacityOverflow)?;

        if on_heap {
            // SAFETY: The caller ensures that the tag corresponds to the
            // way in which data is stored.
            let (old_ptr, old_capacity) = unsafe { self.inner.heap };

            // Nothing needs to be done if the capacity is already sufficient.
            if old_capacity >= new_capacity {
                return Ok(true);
            }

            // Ensure capacity growth is exponential.
            let new_capacity = new_capacity.max(2 * old_capacity);

            // SAFETY: The stored capacity corresponds always to a valid layout.
            let old_layout = unsafe { array_layout_unchecked::<T>(old_capacity) };

            let new_layout = Layout::array::<T>(new_capacity).map_err(|_| CollectionAllocErr::CapacityOverflow)?;
            let ptr = Self::handle_alloc_error(self.allocator.grow(old_ptr.cast(), old_layout, new_layout), new_layout)?;

            self.inner = RawSmallVecUnion::new_heap(ptr.cast(), new_capacity);
            Ok(true)
        } else if new_capacity > inline_size::<T, N>() {
            // Ensure capacity growth is exponential.
            let new_capacity = (2 * N).max(new_capacity);

            let layout = Layout::array::<T>(new_capacity).map_err(|_| CollectionAllocErr::CapacityOverflow)?;
            let ptr = Self::handle_alloc_error(self.allocator.allocate(layout), layout)?;

            // SAFETY: The pointer returned by `allocate` is valid and its own memory region.
            unsafe {
                copy_nonoverlapping(self.as_mut_ptr_inline(), ptr.cast().as_ptr(), len);
            }

            self.inner = RawSmallVecUnion::new_heap(ptr.cast(), new_capacity);
            Ok(true)
        } else {
            Ok(on_heap)
        }
    }

    /// Returns `true` if the elements are stored on the heap, and `false` otherwise.
    ///
    /// # Safety
    ///
    /// The way elements are stored in `self` must correspond to the tag in `len`.
    unsafe fn try_reserve_exact(
        &mut self,
        len: TaggedLen,
        additional: usize,
    ) -> Result<bool, CollectionAllocErr> {
        debug_assert!(!is_zst::<T>());

        let on_heap = len.on_heap(is_zst::<T>());
        let len = len.value(is_zst::<T>());

        if additional == 0 {
            return Ok(on_heap);
        }

        let new_capacity = len.checked_add(additional).ok_or(CollectionAllocErr::CapacityOverflow)?;

        if on_heap {
            // SAFETY: The caller ensures that the tag corresponds to the
            // way in which data is stored.
            let (old_ptr, old_capacity) = unsafe { self.inner.heap };

            // Nothing needs to be done if the capacity is already sufficient.
            if old_capacity >= new_capacity {
                return Ok(true);
            }

            // SAFETY: The stored capacity corresponds always to a valid layout.
            let old_layout = unsafe { array_layout_unchecked::<T>(old_capacity) };

            let new_layout = Layout::array::<T>(new_capacity).map_err(|_| CollectionAllocErr::CapacityOverflow)?;
            let ptr = Self::handle_alloc_error(self.allocator.grow(old_ptr.cast(), old_layout, new_layout), new_layout)?;

            self.inner = RawSmallVecUnion::new_heap(ptr.cast(), new_capacity);

            Ok(true)
        } else if new_capacity > inline_size::<T, N>() {
            let layout = Layout::array::<T>(new_capacity).map_err(|_| CollectionAllocErr::CapacityOverflow)?;
            let ptr = Self::handle_alloc_error(self.allocator.allocate(layout), layout)?;

            // SAFETY: The pointer returned by `allocate` is valid and its own memory region.
            unsafe {
                copy_nonoverlapping(self.as_mut_ptr_inline(), ptr.cast().as_ptr(), len);
            }

            self.inner = RawSmallVecUnion::new_heap(ptr.cast(), new_capacity);

            Ok(true)
        } else {
            Ok(on_heap)
        }
    }

    /// Returns `true` if the elements are still stored on the heap, and `false` otherwise.
    ///
    /// # Safety
    ///
    /// The way elements are stored in `self` must correspond to `on_heap`.
    unsafe fn shrink_to_fit(
        &mut self,
        on_heap: bool,
        cap: usize,
    ) -> Result<bool, CollectionAllocErr> {
        debug_assert!(!is_zst::<T>());

        if on_heap {
            // SAFETY: The caller ensures that the tag corresponds to the
            // way in which data is stored.
            let (old_ptr, old_capacity) = unsafe { self.inner.heap };

            // SAFETY: The stored capacity corresponds always to a valid layout.
            let layout = unsafe { array_layout_unchecked::<T>(old_capacity) };

            if cap <= N {
                self.inner = RawSmallVecUnion::new_inline(MaybeUninit::uninit());

                // SAFETY: The memory regions don't overlap because one pointer is recently created inline storage.
                // By taking the minimum value of both capabilities, the copying will only touch valid memory.
                unsafe {
                    let count = cap.min(old_capacity);
                    copy_nonoverlapping(old_ptr.cast().as_ptr(), self.as_mut_ptr_inline(), count);
                }

                self.allocator.deallocate(old_ptr.cast(), layout);

                Ok(false)
            } else if cap < old_capacity {
                // SAFETY: The new capacity is smaller than the old capacity,
                // and it is already possible to construct a valid layout with the old capacity.
                let new_layout = unsafe { array_layout_unchecked::<T>(cap) };

                let ptr = Self::handle_alloc_error(self.allocator.shrink(old_ptr.cast(), layout, new_layout), new_layout)?;
                self.inner = RawSmallVecUnion::new_heap(ptr.cast(), cap);

                Ok(true)
            } else {
                Ok(true)
            }
        } else {
            Ok(on_heap)
        }
    }
}

/// Vec guarantees that its length is always less than [`isize::MAX`] in *bytes*.
///
/// For a non ZST, this means that the length is less than `isize::MAX` objects, which implies we
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
pub struct SmallVec<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator = Global> {
    len: TaggedLen,
    raw: alloc_param!(RawSmallVec<T, N, A>),
    _marker: PhantomData<T>,
}

unsafe impl<T: Send, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Send for alloc_param!(SmallVec<T, N, A>) {}
unsafe impl<T: Sync, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Sync for alloc_param!(SmallVec<T, N, A>) {}

impl<T, const N: usize> Default for SmallVec<T, N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// An iterator that removes the items from a `SmallVec` and yields them by value.
///
/// Returned from [`SmallVec::drain`][1].
///
/// [1]: struct.SmallVec.html#method.drain
pub struct Drain<'a, T: 'a, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> {
    // `vec` points to a valid object within its lifetime.
    // This is ensured by the fact that we're holding an iterator to its items.
    //
    // # Safety
    //
    // Members in vec[tail_start..tail_start + tail_len] are initialized
    // even though vec has length < tail_start
    tail_start: usize,
    tail_len: usize,
    iter: core::slice::Iter<'a, T>,
    vec: core::ptr::NonNull<alloc_param!(SmallVec<T, N, A>)>,
}

impl<'a, T: 'a, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Iterator for alloc_param!(Drain<'a, T, N, A>) {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        // SAFETY: we shrunk the length of the vector so it no longer owns these items, and we can
        // take ownership of them.
        self.iter
            .next()
            .map(|reference| unsafe { core::ptr::read(reference) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T: 'a, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> DoubleEndedIterator for alloc_param!(Drain<'a, T, N, A>) {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        // SAFETY: see above
        self.iter
            .next_back()
            .map(|reference| unsafe { core::ptr::read(reference) })
    }
}

impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> ExactSizeIterator for alloc_param!(Drain<'_, T, N, A>) {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> core::iter::FusedIterator for alloc_param!(Drain<'_, T, N, A>) {}

impl<'a, T: 'a, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Drop for alloc_param!(Drain<'a, T, N, A>) {
    fn drop(&mut self) {
        /// Moves back the un-`Drain`ed elements to restore the original `Vec`.
        struct DropGuard<'r, 'a, T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator>(&'r mut alloc_param!(Drain<'a, T, N, A>));

        impl<'r, 'a, T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Drop for alloc_param!(DropGuard<'r, 'a, T, N, A>) {
            fn drop(&mut self) {
                if self.0.tail_len > 0 {
                    unsafe {
                        let source_vec = self.0.vec.as_mut();
                        // memmove back untouched tail, update to new length
                        let start = source_vec.len();
                        let tail = self.0.tail_start;
                        if tail != start {
                            let ptr = source_vec.as_mut_ptr();
                            let src = ptr.add(tail);
                            let dst = ptr.add(start);
                            core::ptr::copy(src, dst, self.0.tail_len);
                        }
                        source_vec.set_len(start + self.0.tail_len);
                    }
                }
            }
        }

        let iter = core::mem::take(&mut self.iter);
        let drop_len = iter.len();

        let mut vec = self.vec;

        if is_zst::<T>() {
            // ZSTs have no identity, so we don't need to move them around, we only need to drop the correct amount.
            // this can be achieved by manipulating the Vec length instead of moving values out from `iter`.
            unsafe {
                let vec = vec.as_mut();
                let old_len = vec.len();
                vec.set_len(old_len + drop_len + self.tail_len);
                vec.truncate(old_len + self.tail_len);
            }

            return;
        }

        // ensure elements are moved back into their appropriate places, even when drop_in_place panics
        let _guard = DropGuard(self);

        if drop_len == 0 {
            return;
        }

        // as_slice() must only be called when iter.len() is > 0 because
        // it also gets touched by vec::Splice which may turn it into a dangling pointer
        // which would make it and the vec pointer point to different allocations which would
        // lead to invalid pointer arithmetic below.
        let drop_ptr = iter.as_slice().as_ptr();

        unsafe {
            // drop_ptr comes from a slice::Iter which only gives us a &[T] but for drop_in_place
            // a pointer with mutable provenance is necessary. Therefore we must reconstruct
            // it from the original vec but also avoid creating a &mut to the front since that could
            // invalidate raw pointers to it which some unsafe code might rely on.
            let vec_ptr = vec.as_mut().as_mut_ptr();
            // May be replaced with the line below later, once this crate's MSRV is >= 1.87.
            //let drop_offset = drop_ptr.offset_from_unsigned(vec_ptr);
            let drop_offset = drop_ptr.offset_from(vec_ptr) as usize;
            let to_drop = core::ptr::slice_from_raw_parts_mut(vec_ptr.add(drop_offset), drop_len);
            core::ptr::drop_in_place(to_drop);
        }
    }
}

impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> alloc_param!(Drain<'_, T, N, A>) {
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        self.iter.as_slice()
    }

    /// The range from `self.vec.len` to `self.tail_start` contains elements
    /// that have been moved out.
    /// Fill that range as much as possible with new elements from the `replace_with` iterator.
    /// Returns `true` if we filled the entire range. (`replace_with.next()` didn’t return `None`.)
    unsafe fn fill<I: Iterator<Item = T>>(&mut self, replace_with: &mut I) -> bool {
        let vec = unsafe { self.vec.as_mut() };
        let range_start = vec.len();
        let range_end = self.tail_start;
        let range_slice = unsafe {
            core::slice::from_raw_parts_mut(vec.as_mut_ptr().add(range_start), range_end - range_start)
        };

        for place in range_slice {
            if let Some(new_item) = replace_with.next() {
                unsafe { core::ptr::write(place, new_item) };
                vec.set_len(vec.len() + 1);
            } else {
                return false;
            }
        }
        true
    }

    /// Makes room for inserting more elements before the tail.
    #[track_caller]
    unsafe fn move_tail(&mut self, additional: usize) {
        let vec = unsafe { self.vec.as_mut() };
        let len = self.tail_start + self.tail_len;

        // Test
        let old_len = vec.len();
        vec.set_len(len);
        vec.reserve(additional);
        vec.set_len(old_len);

        let new_tail_start = self.tail_start + additional;
        unsafe {
            let src = vec.as_ptr().add(self.tail_start);
            let dst = vec.as_mut_ptr().add(new_tail_start);
            core::ptr::copy(src, dst, self.tail_len);
        }
        self.tail_start = new_tail_start;
    }
}

#[cfg(feature = "extract_if")]
/// An iterator which uses a closure to determine if an element should be removed.
///
/// Returned from [`SmallVec::extract_if`][1].
///
/// [1]: struct.SmallVec.html#method.extract_if
pub struct ExtractIf<'a, T, const N: usize, F, #[cfg(feature = "allocator_api")] A: Allocator>
where
    F: FnMut(&mut T) -> bool,
{
    vec: &'a mut alloc_param!(SmallVec<T, N, A>),
    /// The index of the item that will be inspected by the next call to `next`.
    idx: usize,
    /// Elements at and beyond this point will be retained. Must be equal or smaller than `old_len`.
    end: usize,
    /// The number of items that have been drained (removed) thus far.
    del: usize,
    /// The original length of `vec` prior to draining.
    old_len: usize,
    /// The filter test predicate.
    pred: F,
}

#[cfg(feature = "extract_if")]
impl<T, const N: usize, F, #[cfg(feature = "allocator_api")] A: Allocator> core::fmt::Debug for alloc_param!(ExtractIf<'_, T, N, F, A>)
where
    F: FnMut(&mut T) -> bool,
    T: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("ExtractIf")
            .field(&self.vec.as_slice())
            .finish()
    }
}

#[cfg(feature = "extract_if")]
impl<T, F, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Iterator for alloc_param!(ExtractIf<'_, T, N, F, A>)
where
    F: FnMut(&mut T) -> bool,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        unsafe {
            while self.idx < self.end {
                let i = self.idx;
                let v = core::slice::from_raw_parts_mut(self.vec.as_mut_ptr(), self.old_len);
                let drained = (self.pred)(&mut v[i]);
                // Update the index *after* the predicate is called. If the index
                // is updated prior and the predicate panics, the element at this
                // index would be leaked.
                self.idx += 1;
                if drained {
                    self.del += 1;
                    return Some(core::ptr::read(&v[i]));
                } else if self.del > 0 {
                    let del = self.del;
                    let src: *const T = &v[i];
                    let dst: *mut T = &mut v[i - del];
                    core::ptr::copy_nonoverlapping(src, dst, 1);
                }
            }
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.end - self.idx))
    }
}

#[cfg(feature = "extract_if")]
impl<T, F, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Drop for alloc_param!(ExtractIf<'_, T, N, F, A>)
where
    F: FnMut(&mut T) -> bool,
{
    fn drop(&mut self) {
        unsafe {
            if self.idx < self.old_len && self.del > 0 {
                // This is a pretty messed up state, and there isn't really an
                // obviously right thing to do. We don't want to keep trying
                // to execute `pred`, so we just backshift all the unprocessed
                // elements and tell the vec that they still exist. The backshift
                // is required to prevent a double-drop of the last successfully
                // drained item prior to a panic in the predicate.
                let ptr = self.vec.as_mut_ptr();
                let src = ptr.add(self.idx);
                let dst = src.sub(self.del);
                let tail_len = self.old_len - self.idx;
                src.copy_to(dst, tail_len);
            }
            self.vec.set_len(self.old_len - self.del);
        }
    }
}

pub struct Splice<'a, I: Iterator + 'a, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> {
    drain: alloc_param!(Drain<'a, I::Item, N, A>),
    //drain: Drain<'a, I::Item, N>,
    replace_with: I,
}

impl<'a, I, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> core::fmt::Debug for alloc_param!(Splice<'a, I, N, A>)
where
    I: Debug + Iterator + 'a,
    <I as Iterator>::Item: Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Splice").field(&self.drain).finish()
    }
}

impl<I: Iterator, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Iterator for alloc_param!(Splice<'_, I, N, A>) {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.drain.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.drain.size_hint()
    }
}

impl<I: Iterator, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> DoubleEndedIterator for alloc_param!(Splice<'_, I, N, A>) {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.drain.next_back()
    }
}

impl<I: Iterator, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> ExactSizeIterator for alloc_param!(Splice<'_, I, N, A>) {}

impl<I: Iterator, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Drop for alloc_param!(Splice<'_, I, N, A>) {
    fn drop(&mut self) {
        self.drain.by_ref().for_each(drop);
        // At this point draining is done and the only remaining tasks are splicing
        // and moving things into the final place.
        // Which means we can replace the slice::Iter with pointers that won't point to deallocated
        // memory, so that Drain::drop is still allowed to call iter.len(), otherwise it would break
        // the ptr.sub_ptr contract.
        self.drain.iter = [].iter();

        unsafe {
            if self.drain.tail_len == 0 {
                self.drain.vec.as_mut().extend(self.replace_with.by_ref());
                return;
            }

            // First fill the range left by drain().
            if !self.drain.fill(&mut self.replace_with) {
                return;
            }

            // There may be more elements. Use the lower bound as an estimate.
            // FIXME: Is the upper bound a better guess? Or something else?
            let (lower_bound, _upper_bound) = self.replace_with.size_hint();
            if lower_bound > 0 {
                self.drain.move_tail(lower_bound);
                if !self.drain.fill(&mut self.replace_with) {
                    return;
                }
            }

            // Collect any remaining elements.
            let mut collected = self.replace_with.by_ref().collect::<SmallVec<I::Item, N>>().into_iter();
            // Now we have an exact count.
            if collected.len() > 0 {
                self.drain.move_tail(collected.len());
                let filled = self.drain.fill(&mut collected);
                debug_assert!(filled);
                debug_assert_eq!(collected.len(), 0);
            }
        }
        // Let `Drain::drop` move the tail back if necessary and restore `vec.len`.
    }
}

/// An iterator that consumes a `SmallVec` and yields its items by value.
///
/// Returned from [`SmallVec::into_iter`][1].
///
/// [1]: struct.SmallVec.html#method.into_iter
pub struct IntoIter<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> {
    // # Safety
    //
    // `end` decides whether the data lives on the heap or not
    //
    // The members from begin..end are initialized
    raw: alloc_param!(RawSmallVec<T, N, A>),
    begin: usize,
    end: TaggedLen,
    _marker: PhantomData<T>,
}

// SAFETY: IntoIter has unique ownership of its contents.  Sending (or sharing) an `IntoIter<T, N>`
// is equivalent to sending (or sharing) a `SmallVec<T, N>`.
unsafe impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Send for alloc_param!(IntoIter<T, N, A>) where T: Send {}
unsafe impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Sync for alloc_param!(IntoIter<T, N, A>) where T: Sync {}

impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> alloc_param!(IntoIter<T, N, A>) {
    #[inline]
    const fn is_zst() -> bool {
        size_of::<T>() == 0
    }

    #[inline]
    const fn as_ptr(&self) -> *const T {
        let on_heap = self.end.on_heap(Self::is_zst());
        if on_heap {
            // SAFETY: vector is on the heap
            unsafe { self.raw.as_ptr_heap() }
        } else {
            self.raw.as_ptr_inline()
        }
    }

    #[inline]
    const fn as_mut_ptr(&mut self) -> *mut T {
        let on_heap = self.end.on_heap(Self::is_zst());
        if on_heap {
            // SAFETY: vector is on the heap
            unsafe { self.raw.as_mut_ptr_heap() }
        } else {
            self.raw.as_mut_ptr_inline()
        }
    }

    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        // SAFETY: The members in self.begin..self.end.value() are all initialized
        // So the pointer arithmetic is valid, and so is the construction of the slice
        unsafe {
            let ptr = self.as_ptr();
            core::slice::from_raw_parts(
                ptr.add(self.begin),
                self.end.value(Self::is_zst()) - self.begin,
            )
        }
    }

    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: see above
        unsafe {
            let ptr = self.as_mut_ptr();
            core::slice::from_raw_parts_mut(
                ptr.add(self.begin),
                self.end.value(Self::is_zst()) - self.begin,
            )
        }
    }

    /// Returns a reference to the underlying allocator.
    #[cfg(feature = "allocator_api")]
    pub fn allocator(&self) -> &A {
        &self.raw.allocator
    }
}

impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Iterator for alloc_param!(IntoIter<T, N, A>) {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.begin == self.end.value(Self::is_zst()) {
            None
        } else {
            // SAFETY: see above
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

impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> DoubleEndedIterator for alloc_param!(IntoIter<T, N, A>) {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let mut end = self.end.value(Self::is_zst());
        if self.begin == end {
            None
        } else {
            // SAFETY: see above
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
impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> ExactSizeIterator for alloc_param!(IntoIter<T, N, A>) {}
impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> core::iter::FusedIterator for alloc_param!(IntoIter<T, N, A>) {}

impl<T, const N: usize> SmallVec<T, N> {
    /// A stub for `Self::new_in` that can be used whenever the "allocator_api" feature is disabled.
    /// This makes it possible to develop some functions independent from the feature,
    /// reduceing code complexity.
    #[cfg(not(feature = "allocator_api"))]
    const fn new_in(_alloc: A) -> Self {
        const { Self::new() }
    }

    /// A stub for `Self::with_capacity_in` that can be used whenever the "allocator_api" feature is disabled.
    /// This makes it possible to develop some functions independent from the feature,
    /// reduceing code complexity.
    #[cfg(not(feature = "allocator_api"))]
    fn with_capacity_in(capacity: usize, _alloc: A) -> Self {
        Self::with_capacity(capacity)
    }

    #[inline]
    pub const fn new() -> SmallVec<T, N> {
        Self {
            len: TaggedLen::new(0, false, Self::is_zst()),
            raw: RawSmallVec::new(),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        let on_heap = capacity > inline_size::<T, N>();
        Self {
            len: TaggedLen::new(0, on_heap, Self::is_zst()),
            raw: RawSmallVec::with_capacity(capacity),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub const fn from_buf<const S: usize>(elements: [T; S]) -> Self {
        const { assert!(S <= N); }

        // Althought we create a new buffer, since S and N are known at compile time,
        // even with `-C opt-level=1`, it gets optimized as best as it could be. (Checked with <godbolt.org>)
        let mut buf: MaybeUninit<[T; N]> = MaybeUninit::uninit();

        // SAFETY: buf and elements do not overlap, are aligned and have space
        // for at least S elements since S <= N.
        // We will drop the elements only once since we do forget(elements).
        unsafe {
            copy_nonoverlapping(elements.as_ptr(), buf.as_mut_ptr() as *mut T, S);
        }

        // `elements` have been moved into buf and will be droped by SmallVec
        core::mem::forget(elements);

        // SAFETY: all the members in 0..S are initialized
        Self {
            len: TaggedLen::new(S, false, Self::is_zst()),
            raw: RawSmallVec::new_inline(buf),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn from_buf_and_len(buf: [T; N], len: usize) -> Self {
        assert!(len <= N);
        // SAFETY: all the members in 0..len are initialized
        let mut vec = Self {
            len: TaggedLen::new(len, false, Self::is_zst()),
            raw: RawSmallVec::new_inline(MaybeUninit::new(buf)),
            _marker: PhantomData,
        };
        // Deallocate the remaining elements so no memory is leaked.
        unsafe {
            // SAFETY: both the input and output pointers are in range of the stack allocation
            let remainder_ptr = vec.raw.as_mut_ptr_inline().add(len);
            let remainder_len = N - len;

            // SAFETY: the values are initialized, so dropping them here is fine.
            core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(
                remainder_ptr,
                remainder_len,
            ));
        }

        vec
    }

    /// Constructs a new `SmallVec` on the stack from an A without copying elements. Also sets the length. The user is responsible for ensuring that `len <= A::size()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use smallvec::SmallVec;
    /// use std::mem::MaybeUninit;
    ///
    /// let buf = [1, 2, 3, 4, 5, 0, 0, 0];
    /// let small_vec = unsafe {
    ///     SmallVec::from_buf_and_len_unchecked(MaybeUninit::new(buf), 5)
    /// };
    ///
    /// assert_eq!(&*small_vec, &[1, 2, 3, 4, 5]);
    /// ```
    ///
    /// # Safety
    ///
    /// `len <= N`, and all the elements in `buf[..len]` must be initialized
    #[inline]
    pub const unsafe fn from_buf_and_len_unchecked(buf: MaybeUninit<[T; N]>, len: usize) -> Self {
        debug_assert!(len <= N);
        Self {
            len: TaggedLen::new(len, false, Self::is_zst()),
            raw: RawSmallVec::new_inline(buf),
            _marker: PhantomData,
        }
    }

    /// Creates a `SmallVec` directly from the raw components of another `SmallVec`.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren’t checked:
    ///
    /// - `ptr` needs to have been previously allocated via `SmallVec` from its spilled storage (at least, it’s highly likely to be incorrect if it wasn’t).
    /// - `ptr`’s `A::Item` type needs to be the same size and alignment that it was allocated with
    /// - `length` needs to be less than or equal to `capacity`.
    /// - `capacity` needs to be the capacity that the pointer was allocated with.
    ///
    /// Violating these may cause problems like corrupting the allocator’s internal data structures.
    ///
    /// Additionally, `capacity` must be greater than the amount of inline storage `A` has; that is, the new `SmallVec` must need to spill over into heap allocated storage. This condition is asserted against.
    ///
    /// The ownership of `ptr` is effectively transferred to the `SmallVec` which may then deallocate, reallocate or change the contents of memory pointed to by the pointer at will. Ensure that nothing else uses the pointer after calling this function.
    ///
    /// # Examples
    ///
    /// ```
    /// use smallvec::{SmallVec, smallvec};
    ///
    /// let mut v: SmallVec<_, 1> = smallvec![1, 2, 3];
    ///
    /// // Pull out the important parts of `v`.
    /// let p = v.as_mut_ptr();
    /// let len = v.len();
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
    ///     for i in 0..len {
    ///         std::ptr::write(p.add(i), 4 + i);
    ///     }
    ///
    ///     // Put everything back together into a SmallVec with a different
    ///     // amount of inline storage, but which is still less than `cap`.
    ///     let rebuilt = SmallVec::<_, 2>::from_raw_parts(p, len, cap);
    ///     assert_eq!(&*rebuilt, &[4, 5, 6]);
    /// }
    /// ```
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut T, length: usize, capacity: usize) -> SmallVec<T, N> {
        assert!(!Self::is_zst());

        // SAFETY: We require caller to provide same ptr as we alloc
        // and we never alloc null pointer.
        let ptr = unsafe {
            debug_assert!(!ptr.is_null(), "Called `from_raw_parts` with null pointer.");
            NonNull::new_unchecked(ptr)
        };

        SmallVec {
            len: TaggedLen::new(length, true, is_zst::<T>()),
            raw: RawSmallVec::new_heap(ptr, capacity),
            _marker: PhantomData,
        }
    }
}

#[cfg(feature = "allocator_api")]
impl<T, const N: usize, A: Allocator> SmallVec<T, N, A> {
    #[inline(always)]
    pub const fn allocator(&self) -> &A {
        &self.raw.allocator
    }

    #[inline]
    pub const fn new_in(alloc: A) -> Self {
        Self {
            len: TaggedLen::new(0, false, Self::is_zst()),
            raw: RawSmallVec::new_in(alloc),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        let on_heap = capacity > inline_size::<T, N>();
        Self {
            len: TaggedLen::new(0, on_heap, Self::is_zst()),
            raw: RawSmallVec::with_capacity_in(capacity, alloc),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn try_with_capacity_in(
        capacity: usize,
        alloc: A,
    ) -> Result<Self, CollectionAllocErr> {
        let on_heap = capacity > inline_size::<T, N>();
        Ok(Self {
            len: TaggedLen::new(0, on_heap, Self::is_zst()),
            raw: RawSmallVec::try_with_capacity_in(capacity, alloc)?,
            _marker: PhantomData,
        })
    }

    #[inline]
    pub unsafe fn from_raw_parts_in(
        ptr: *mut T,
        length: usize,
        capacity: usize,
        alloc: A,
    ) -> Self {
        assert!(!Self::is_zst());

        // SAFETY: We require caller to provide same ptr as we alloc
        // and we never alloc null pointer.
        let ptr = unsafe {
            debug_assert!(!ptr.is_null(), "Called `from_raw_parts` with null pointer.");
            NonNull::new_unchecked(ptr)
        };

        Self {
            len: TaggedLen::new(length, true, is_zst::<T>()),
            raw: RawSmallVec::new_heap_in(ptr, capacity, alloc),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub unsafe fn from_parts_in(
        ptr: NonNull<T>,
        length: usize,
        capacity: usize,
        alloc: A,
    ) -> Self {
        assert!(!Self::is_zst());

        Self {
            len: TaggedLen::new(length, true, is_zst::<T>()),
            raw: RawSmallVec::new_heap_in(ptr, capacity, alloc),
            _marker: PhantomData,
        }
    }
}

impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> alloc_param!(SmallVec<T, N, A>) {
    #[inline]
    const fn is_zst() -> bool {
        size_of::<T>() == 0
    }

    /// Sets the tag to be on the heap
    ///
    /// # Safety
    ///
    /// The active union member must be the self.raw.heap
    #[inline]
    unsafe fn set_on_heap(&mut self) {
        self.len = TaggedLen::new(self.len(), true, Self::is_zst());
    }

    /// Sets the tag to be inline
    ///
    /// # Safety
    ///
    /// The active union member must be the self.raw.inline
    #[inline]
    unsafe fn set_inline(&mut self) {
        self.len = TaggedLen::new(self.len(), false, Self::is_zst());
    }

    /// Sets the length of a vector.
    ///
    /// This will explicitly set the size of the vector, without actually modifying its buffers, so
    /// it is up to the caller to ensure that the vector is actually the specified size.
    ///
    /// # Safety
    ///
    /// `new_len <= self.capacity()` must be true, and all the elements in the range `..self.len`
    /// must be initialized.
    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.capacity());
        let on_heap = self.len.on_heap(Self::is_zst());
        self.len = TaggedLen::new(new_len, on_heap, Self::is_zst());
    }

    #[inline]
    pub const fn inline_size() -> usize {
        if Self::is_zst() {
            usize::MAX
        } else {
            N
        }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.len.value(Self::is_zst())
    }

    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub const fn capacity(&self) -> usize {
        if self.len.on_heap(Self::is_zst()) {
            // SAFETY: raw.heap is active
            unsafe { self.raw.inner.heap.1 }
        } else {
            Self::inline_size()
        }
    }

    #[inline]
    pub const fn spilled(&self) -> bool {
        self.len.on_heap(Self::is_zst())
    }

    #[inline]
    pub fn from_vec(vec: alloc_param!(Vec<T, A>)) -> Self {
        let mut vec = ManuallyDrop::new(vec);
        let cap = vec.capacity();
        let len = vec.len();
        #[cfg(feature = "allocator_api")]
        // SAFETY: casting a reference to a pointer results
        // in a valid pointer. Since the value is wrapped inside a
        // `ManuallyDrop`, the value is not dropped twice.
        let alloc = unsafe { core::ptr::read(vec.allocator() as *const A) };
        #[cfg(not(feature = "allocator_api"))]
        let alloc = A;

        if Self::is_zst() {
            // Standard Rust ZST vecs don't allocate, so no memory management needs to be done.
            Self {
                len: TaggedLen::new(len, false, Self::is_zst()),
                raw: RawSmallVec::new_in(alloc),
                _marker: PhantomData,
            }
        } else if cap > 0 {
            // FIXME: Replace with `Vec::as_non_null` once it is stable.
            // SAFETY: The pointer of a vector is never null.
            let ptr = unsafe { NonNull::new_unchecked(vec.as_mut_ptr()) };

            Self {
                len: TaggedLen::new(len, true, Self::is_zst()),
                raw: RawSmallVec::new_heap_in(ptr, cap, alloc),
                _marker: PhantomData,
            }
        } else {
            Self {
                len: TaggedLen::new(0, false, Self::is_zst()),
                raw: RawSmallVec::new_inline_in(MaybeUninit::uninit(), alloc),
                _marker: PhantomData,
            }
        }
    }

    pub fn drain<R>(&mut self, range: R) -> alloc_param!(Drain<'_, T, N, A>)
    where
        R: core::ops::RangeBounds<usize>,
    {
        let len = self.len();
        let core::ops::Range { start, end } = slice_range(range, ..len);

        unsafe {
            // SAFETY: `start <= len`
            self.set_len(start);

            // SAFETY: all the elements in `start..end` are initialized
            let range_slice = core::slice::from_raw_parts(self.as_ptr().add(start), end - start);

            // SAFETY: all the elements in `end..len` are initialized
            Drain {
                tail_start: end,
                tail_len: len - end,
                iter: range_slice.iter(),
                // Since self is a &mut, passing it to a function would invalidate the slice iterator.
                vec: core::ptr::NonNull::new_unchecked(self as *mut _),
                //vec: core::ptr::NonNull::from(self),
            }
        }
    }

    #[cfg(feature = "extract_if")]
    /// Creates an iterator which uses a closure to determine if element in the range should be removed.
    ///
    /// If the closure returns true, then the element is removed and yielded.
    /// If the closure returns false, the element will remain in the vector and will not be yielded
    /// by the iterator.
    ///
    /// Only elements that fall in the provided range are considered for extraction, but any elements
    /// after the range will still have to be moved if any element has been extracted.
    ///
    /// If the returned `ExtractIf` is not exhausted, e.g. because it is dropped without iterating
    /// or the iteration short-circuits, then the remaining elements will be retained.
    /// Use [`retain`] with a negated predicate if you do not need the returned iterator.
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
    /// Note that `extract_if` also lets you mutate the elements passed to the filter closure,
    /// regardless of whether you choose to keep or remove them.
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
    /// let mut numbers: SmallVec<i32, 16> = SmallVec::from(&[1i32, 2, 3, 4, 5, 6, 8, 9, 11, 13, 14, 15]);
    ///
    /// let evens = numbers.extract_if(.., |x| *x % 2 == 0).collect::<SmallVec<i32, 16>>();
    /// let odds = numbers;
    ///
    /// assert_eq!(evens, SmallVec::<i32, 16>::from(&[2i32, 4, 6, 8, 14]));
    /// assert_eq!(odds, SmallVec::<i32, 16>::from(&[1i32, 3, 5, 9, 11, 13, 15]));
    /// ```
    ///
    /// Using the range argument to only process a part of the vector:
    ///
    /// ```
    /// # use smallvec::SmallVec;
    /// let mut items: SmallVec<i32, 16> = SmallVec::from(&[0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2]);
    /// let ones = items.extract_if(7.., |x| *x == 1).collect::<SmallVec<i32, 16>>();
    /// assert_eq!(items, SmallVec::<i32, 16>::from(&[0, 0, 0, 0, 0, 0, 0, 2, 2, 2]));
    /// assert_eq!(ones.len(), 3);
    /// ```
    pub fn extract_if<F, R>(&mut self, range: R, filter: F) -> alloc_param!(ExtractIf<'_, T, N, F, A>)
    where
        F: FnMut(&mut T) -> bool,
        R: core::ops::RangeBounds<usize>,
    {
        let old_len = self.len();
        let core::ops::Range { start, end } = slice_range(range, ..old_len);

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
            pred: filter,
        }
    }

    pub fn splice<R, I>(&mut self, range: R, replace_with: I) -> alloc_param!(Splice<'_, I::IntoIter, N, A>)
    where
        R: core::ops::RangeBounds<usize>,
        I: IntoIterator<Item = T>,
    {
        Splice { drain: self.drain(range), replace_with: replace_with.into_iter() }
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
            // SAFETY: this element was initialized and we just gave up ownership of it, so we can
            // give it away
            let value = unsafe { self.as_mut_ptr().add(len).read() };
            Some(value)
        }
    }

    #[inline]
    pub fn pop_if(&mut self, predicate: impl FnOnce(&mut T) -> bool) -> Option<T> {
        let last = self.last_mut()?;
        if predicate(last) { self.pop() } else { None }
    }

    #[inline]
    pub fn append<const M: usize>(&mut self, other: &mut alloc_param!(SmallVec<T, M, A>)) {
        // can't overflow since both are smaller than isize::MAX and 2 * isize::MAX < usize::MAX
        let len = self.len();
        let other_len = other.len();
        let total_len = len + other_len;
        if total_len > self.capacity() {
            self.reserve(other_len);
        }

        // SAFETY: see `Self::push`
        let ptr = unsafe { self.as_mut_ptr().add(len) };
        unsafe { other.set_len(0) }
        // SAFETY: we have a mutable reference to each vector and each uniquely owns its memory.
        // so the ranges can't overlap
        unsafe { copy_nonoverlapping(other.as_ptr(), ptr, other_len) };
        unsafe { self.set_len(total_len) }
    }

    #[inline]
    pub fn grow(&mut self, new_capacity: usize) {
        infallible(self.try_grow(new_capacity));
    }

    #[cold]
    pub fn try_grow(&mut self, new_capacity: usize) -> Result<(), CollectionAllocErr> {
        if Self::is_zst() {
            return Ok(());
        }

        let len = self.len();
        let on_heap = self.spilled();
        let capacity = self.capacity();
        assert!(new_capacity >= len);

        if new_capacity <= Self::inline_size() && self.spilled() {
            // If the desired capacity is smaller than the inline size,
            // then store all elements inline.

            // SAFETY: The tag inside the length of the vector corresponds to the way
            // elements are stored inside the vector.
            // All elements will be stored inline because the vector will
            // be shrunk to a capacity smaller than the inline size.
            unsafe {
                let on_heap = self.raw.shrink_to_fit(on_heap, len)?;
                debug_assert!(!on_heap);
                self.set_inline();
            }
            Ok(())
        } else if new_capacity > capacity {
            self.try_reserve(new_capacity - capacity)
        } else {
            Ok(())
        }
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        infallible(self.try_reserve(additional))
    }

    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), CollectionAllocErr> {
        if Self::is_zst() {
            return Ok(());
        }

        // SAFETY: The tag inside the length of the vector corresponds to the way
        // elements are stored inside the vector. The same goes for the return value
        // of the function.
        unsafe {
            let on_heap = self.raw.try_reserve(self.len, additional)?;
            if on_heap {
                self.set_on_heap();
            } else {
                self.set_inline();
            }
        };

        Ok(())
    }

    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        infallible(self.try_reserve_exact(additional))
    }

    #[inline]
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), CollectionAllocErr> {
        if is_zst::<T>() {
            return Ok(());
        }

        // SAFETY: The tag inside the length of the vector corresponds to the way
        // elements are stored inside the vector. The same goes for the return value
        // of the function.
        unsafe {
            let on_heap = self.raw.try_reserve_exact(self.len, additional)?;
            if on_heap {
                self.set_on_heap();
            } else {
                self.set_inline();
            }
        };

        Ok(())
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        if is_zst::<T>() {
            return;
        }

        let len = self.len();
        let on_heap = self.spilled();

        // SAFETY: The tag inside the length of the vector corresponds to the way
        // elements are stored inside the vector.
        let on_heap = unsafe {
            infallible(self.raw.shrink_to_fit(on_heap, len))
        };
        self.len = TaggedLen::new(len, on_heap, is_zst::<T>());
    }

    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        if is_zst::<T>() {
            return;
        }

        let len = self.len();
        let min_capacity = len.max(min_capacity);
        let on_heap = self.spilled();

        // SAFETY: The tag inside the length of the vector corresponds to the way
        // elements are stored inside the vector.
        let on_heap = unsafe {
            infallible(self.raw.shrink_to_fit(on_heap, min_capacity))
        };
        self.len = TaggedLen::new(len, on_heap, is_zst::<T>());
    }

    #[inline]
    pub fn truncate(&mut self, len: usize) {
        let old_len = self.len();
        if len < old_len {
            // SAFETY: we set `len` to a smaller value
            // then we drop the previously initialized elements
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
    pub fn swap_remove(&mut self, index: usize) -> T {
        let len = self.len();
        assert!(index < len, "swap_remove index (is {index}) should be < len (is {len})");
        // This can't overflow since `len > index >= 0`
        let new_len = len - 1;
        unsafe {
            // We replace self[index] with the last element. Note that if the
            // bounds check above succeeds there must be a last element (which
            // can be self[index] itself).
            let value = core::ptr::read(self.as_ptr().add(index));
            let base_ptr = self.as_mut_ptr();
            core::ptr::copy(base_ptr.add(new_len), base_ptr.add(index), 1);
            self.set_len(new_len);
            value
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        // SAFETY: we set `len` to a smaller value
        // then we drop the previously initialized elements
        unsafe {
            let old_len = self.len();
            self.set_len(0);
            core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(
                self.as_mut_ptr(),
                old_len,
            ));
        }
    }

    #[inline]
    pub fn remove(&mut self, index: usize) -> T {
        let len = self.len();
        assert!(index < len, "removal index (is {index}) should be < len (is {len})");
        let new_len = len - 1;
        unsafe {
            // SAFETY: new_len < len
            self.set_len(new_len);
            let ptr = self.as_mut_ptr();
            let ith = ptr.add(index);
            // This item is initialized since index < len
            let ith_item = ith.read();
            copy(ith.add(1), ith, new_len - index);
            ith_item
        }
    }

    #[inline]
    pub fn insert(&mut self, index: usize, value: T) {
        let len = self.len();
        assert!(index <= len, "insertion index (is {index}) should be <= len (is {len})");
        self.reserve(1);
        let ptr = self.as_mut_ptr();
        unsafe {
            // the elements at `index + 1..len + 1` are now initialized
            if index < len {
                copy(ptr.add(index), ptr.add(index + 1), len - index);
            }
            // the element at `index` is now initialized
            ptr.add(index).write(value);

            // SAFETY: all the elements are initialized
            self.set_len(len + 1);
        }
    }

    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        let len = self.len();
        let ptr = self.as_ptr();
        // SAFETY: all the elements in `..len` are initialized
        unsafe { core::slice::from_raw_parts(ptr, len) }
    }

    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        let len = self.len();
        let ptr = self.as_mut_ptr();
        // SAFETY: see above
        unsafe { core::slice::from_raw_parts_mut(ptr, len) }
    }

    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        if self.len.on_heap(Self::is_zst()) {
            // SAFETY: heap member is active
            unsafe { self.raw.as_ptr_heap() }
        } else {
            self.raw.as_ptr_inline()
        }
    }

    #[inline]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        if self.len.on_heap(Self::is_zst()) {
            // SAFETY: see above
            unsafe { self.raw.as_mut_ptr_heap() }
        } else {
            self.raw.as_mut_ptr_inline()
        }
    }

    #[inline]
    pub fn into_vec(self) -> alloc_param!(Vec<T, A>) {
        let len = self.len();
        let on_heap = self.spilled();
        let this = ManuallyDrop::new(self);

        // SAFETY: casting a reference to a pointer results
        // in a valid pointer. Since the value is wrapped inside a
        // `ManuallyDrop`, the value is not dropped twice.
        #[cfg(feature = "allocator_api")]
        let alloc = unsafe {
            core::ptr::read(&this.raw.allocator as *const A)
        };

        if !on_heap {
            #[cfg(not(feature = "allocator_api"))]
            let mut vec = Vec::with_capacity(len);
            #[cfg(feature = "allocator_api")]
            let mut vec = Vec::with_capacity_in(len, alloc);

            // SAFETY: we create a new vector with sufficient capacity, copy our elements into it
            // to transfer ownership and then set the length
            // we don't drop the elements we previously held
            unsafe {
                copy_nonoverlapping(this.raw.as_ptr_inline(), vec.as_mut_ptr(), len);
                vec.set_len(len);
            }
            vec
        } else {
            // SAFETY:
            // - `ptr` was created with the appropriate allocator
            // - `ptr` was created with the appropriate alignment for `T`
            // - the allocation pointed to by ptr is exactly cap * sizeof(T)
            // - `len` is less than or equal to `cap`
            // - the first `len` entries are proper `T`-values
            // - the allocation is not larger than `isize::MAX`
            unsafe {
                let (ptr, cap) = this.raw.inner.heap;

                #[cfg(not(feature = "allocator_api"))]
                {
                    Vec::from_raw_parts(ptr.as_ptr(), len, cap)
                }

                #[cfg(feature = "allocator_api")]
                {
                    Vec::from_raw_parts_in(ptr.as_ptr(), len, cap, alloc)
                }
            }
        }
    }

    #[inline]
    pub fn into_boxed_slice(self) -> alloc_param!(Box<[T], A>) {
        self.into_vec().into_boxed_slice()
    }

    #[inline]
    pub fn into_inner(self) -> Result<[T; N], Self> {
        if self.len() != N {
            Err(self)
        } else {
            // when `this` is dropped, the memory is released if it's on the heap.
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
        let mut del = 0;
        let len = self.len();
        let ptr = self.as_mut_ptr();
        for i in 0..len {
            // SAFETY: all the pointers are in bounds
            // `i - del` never overflows since `del <= i` is a maintained invariant
            unsafe {
                if !f(&mut *ptr.add(i)) {
                    del += 1;
                } else if del > 0 {
                    core::ptr::swap(ptr.add(i), ptr.add(i - del));
                }
            }
        }
        self.truncate(len - del);
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
                        core::ptr::swap(p_r, p_w);
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

    pub fn leak<'a>(self) -> &'a mut [T] {
        if !self.spilled() {
            panic!("SmallVec::leak() called on inline (stack) SmallVec, which cannot be safely leaked");
        }
        let mut me = ManuallyDrop::new(self);
        unsafe { core::slice::from_raw_parts_mut(me.as_mut_ptr(), me.len()) }
    }

    /// Returns the remaining spare capacity of the vector as a slice of
    /// `MaybeUninit<T>`.
    ///
    /// The returned slice can be used to fill the vector with data (e.g. by
    /// reading from a file) before marking the data as initialized using the
    /// [`set_len`](Self::set_len) method.
    #[inline]
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        unsafe {
            core::slice::from_raw_parts_mut(
                self.as_mut_ptr().add(self.len()) as *mut MaybeUninit<T>,
                self.capacity() - self.len(),
            )
        }
    }
}

impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator + Clone> alloc_param!(SmallVec<T, N, A>) {
    /// Splits the collection into two at the given index.
    ///
    /// Returns a newly allocated vector containing the elements in the range
    /// `[at, len)`. After the call, the original vector will be left containing
    /// the elements `[0, at)` with its previous capacity unchanged.
    ///
    /// - If you want to take ownership of the entire contents and capacity of
    ///   the vector, see [`core::mem::take`] or [`core::mem::replace`].
    /// - If you don't need the returned vector at all, see [`SmallVec::truncate`].
    /// - If you want to take ownership of an arbitrary subslice, or you don't
    ///   necessarily want to store the removed items in a vector, see [`SmallVec::drain`].
    ///
    /// # Panics
    ///
    /// Panics if `at > len`.
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
        let len = self.len();
        assert!(at <= len);

        let other_len = len - at;

        let mut other = Self::with_capacity_in(other_len, self.raw.allocator.clone());

        // Unsafely `set_len` and copy items to `other`.
        unsafe {
            self.set_len(at);
            other.set_len(other_len);

            core::ptr::copy_nonoverlapping(self.as_ptr().add(at), other.as_mut_ptr(), other_len);
        }
        other
    }
}

impl<T: Clone, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> alloc_param!(SmallVec<T, N, A>) {
    /// Creates a [`SmallVec`] value from the slice `slice` with the specified allocator.
    #[cfg(feature = "allocator_api")]
    pub fn from_slice_in(slice: &[T], alloc: A) -> Self {
        if slice.len() > Self::inline_size() {
            // Standard Rust vectors are already specialized.
            Self::from_vec(slice.to_vec_in(alloc))
        } else {
            // SAFETY: The precondition is checked in the initial comparison above.
            unsafe {
                #[cfg(feature = "specialization")]
                {
                    <Self as spec_traits::SpecFromSlice<T, alloc_param!(A)>>::spec_from(slice, alloc)
                }

                #[cfg(not(feature = "specialization"))]
                {
                    Self::from_slice_fallback(slice, alloc)
                }
            }
        }
    }

    /// A stub for `Self::from_slice_in` that can be used whenever the "allocator_api" feature is disabled.
    /// This makes it possible to develop some functions independent from the feature,
    /// reduceing code complexity.
    #[cfg(not(feature = "allocator_api"))]
    #[inline(always)]
    fn from_slice_in(slice: &[T], _alloc: A) -> Self {
        Self::from(slice)
    }

    #[inline]
    pub fn resize(&mut self, len: usize, value: T) {
        let old_len = self.len();
        if len > old_len {
            self.extend(core::iter::repeat_n(value, len - old_len));
        } else {
            self.truncate(len);
        }
    }

    #[inline]
    pub fn extend_from_slice(&mut self, other: &[T]) {
        self.extend(other.iter())
    }

    pub fn extend_from_within<R>(&mut self, src: R)
    where
        R: core::ops::RangeBounds<usize>,
    {
        let src = slice_range(src, ..self.len());
        self.reserve(src.len());

        // SAFETY: The call to `reserve` ensures that the capacity is large enough.
        // The range is within bounds through the use of `core::slice::range`.
        unsafe {
            #[cfg(feature = "specialization")]
            {
                <Self as spec_traits::SpecExtendFromWithin<T>>::spec_extend_from_within(self, src);
            }

            #[cfg(not(feature = "specialization"))]
            {
                self.extend_from_within_fallback(src);
            }
        }
    }

    #[inline]
    pub fn extend_from_slice_copy(&mut self, other: &[T])
    where
        T: Copy
    {
        
        let len = other.len();
        let src = other.as_ptr();
        
        let l = self.len();
        self.reserve(len);

        // SAFETY: Additional memory has been reserved,
        // therefore the pointer access is valid.
        unsafe {
            let dst = self.as_mut_ptr().add(l);
            copy_nonoverlapping(src, dst, len);
            self.set_len(l + len);
        }
    }

    pub fn extend_from_within_copy<R>(&mut self, src: R)
    where
        R: core::ops::RangeBounds<usize>,
        T: Copy
    {
        let src = slice_range(src, ..self.len());
        let core::ops::Range { start, end } = src;
        let len = end - start;
        self.reserve(len);

        // SAFETY: The call to `reserve` ensures that the capacity is large enough.
        // The range is within bounds through the use of `core::slice::range`.
        unsafe {
            let l = self.len();
            let ptr = self.as_mut_ptr();
            copy_nonoverlapping(ptr.add(start), ptr.add(l), len);
            self.set_len(l + len);
        }
    }

    pub fn insert_from_slice_copy(&mut self, index: usize, other: &[T])
    where
        T: Copy
    {
        let l = self.len();
        let len = other.len();
        assert!(index <= l);
        self.reserve(len);
        unsafe {
            let base_ptr = self.as_mut_ptr();
            let ith_ptr = base_ptr.add(index);
            let shifted_ptr = base_ptr.add(index + len);
            // elements at `index + other_len..len + other_len` are now initialized
            copy(ith_ptr, shifted_ptr, l - index);
            // elements at `index..index + other_len` are now initialized
            copy_nonoverlapping(other.as_ptr(), ith_ptr, len);

            // SAFETY: all the elements are initialized
            self.set_len(l + len);
        }
    }

    #[cfg(feature = "allocator_api")]
    /// A function for creating [`SmallVec`] values out of slices
    /// for types with the [`Copy`] trait.
    pub fn from_slice_copy_in(slice: &[T], alloc: A) -> Self
    where
        T: Copy
    {
        let src = slice.as_ptr();
        let len = slice.len();
        let mut result = Self::with_capacity_in(len, alloc);

        // SAFETY: By using `with_capacity_in`, the pointer will point to valid memory.
        unsafe {
            let dst = result.as_mut_ptr();
            copy_nonoverlapping(src, dst, len);
            result.set_len(len);
        }

        result
    }
}

impl<T: Copy, const N: usize> SmallVec<T, N> {
    /// A function for creating [`SmallVec`] values out of slices
    /// for types with the [`Copy`] trait.
    pub fn from_slice_copy(slice: &[T]) -> Self {
        let src = slice.as_ptr();
        let len = slice.len();
        let mut result = Self::with_capacity(len);

        // SAFETY: By using `with_capacity`, the pointer will point to valid memory.
        unsafe {
            let dst = result.as_mut_ptr();
            copy_nonoverlapping(src, dst, len);
            result.set_len(len);
        }

        result
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

#[cfg(feature = "may_dangle")]
unsafe impl<#[may_dangle] T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Drop for alloc_param!(SmallVec<T, N, A>) {
    fn drop(&mut self) {
        let on_heap = self.spilled();
        let len = self.len();
        let ptr = self.as_mut_ptr();
        // SAFETY: we first drop the elements, then `_drop_dealloc` is dropped, releasing memory we
        // used to own
        unsafe {
            let _drop_dealloc = if on_heap {
                let capacity = self.capacity();
                Some(DropDealloc {
                    ptr: NonNull::new_unchecked(ptr as *mut u8),
                    layout: array_layout_unchecked::<T>(capacity),
                    allocator: &self.raw.allocator,
                })
            } else {
                None
            };
            core::ptr::slice_from_raw_parts_mut(ptr, len).drop_in_place();
        }
    }
}

#[cfg(not(feature = "may_dangle"))]
impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Drop for alloc_param!(SmallVec<T, N, A>) {
    fn drop(&mut self) {
        let on_heap = self.spilled();
        let len = self.len();
        let ptr = self.as_mut_ptr();
        // SAFETY: see above
        unsafe {
            let _drop_dealloc = if on_heap {
                let capacity = self.capacity();
                Some(DropDealloc {
                    ptr: NonNull::new_unchecked(ptr as *mut u8),
                    layout: array_layout_unchecked::<T>(capacity),
                    allocator: &self.raw.allocator,
                })
            } else {
                None
            };
            core::ptr::slice_from_raw_parts_mut(ptr, len).drop_in_place();
        }
    }
}

impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Drop for alloc_param!(IntoIter<T, N, A>) {
    fn drop(&mut self) {
        // SAFETY: see above
        unsafe {
            let is_zst = size_of::<T>() == 0;
            let on_heap = self.end.on_heap(is_zst);
            let begin = self.begin;
            let end = self.end.value(is_zst);
            let ptr = self.as_mut_ptr();
            let _drop_dealloc = if on_heap {
                let capacity = self.raw.inner.heap.1;
                Some(DropDealloc {
                    ptr: NonNull::new_unchecked(ptr as *mut u8),
                    layout: array_layout_unchecked::<T>(capacity),
                    allocator: &self.raw.allocator,
                })
            } else {
                None
            };
            core::ptr::slice_from_raw_parts_mut(ptr.add(begin), end - begin).drop_in_place();
        }
    }
}

impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> core::ops::Deref for alloc_param!(SmallVec<T, N, A>) {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}
impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> core::ops::DerefMut for alloc_param!(SmallVec<T, N, A>) {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

/// This function is used in the [`smallvec`] macro.
/// It is recommended to use the macro instead of using thís function.
#[doc(hidden)]
#[track_caller]
pub fn from_elem<T: Clone, const N: usize>(elem: T, n: usize) -> SmallVec<T, N> {
    if n > SmallVec::<T, N>::inline_size() {
        // Standard Rust vectors are already specialized.
        SmallVec::<T, N>::from_vec(vec![elem; n])
    } else {
        #[cfg(feature = "specialization")]
        {
            // SAFETY: The precondition is checked in the initial comparison above.
            #[cfg(feature = "allocator_api")]
            unsafe {
                <SmallVec<T, N> as spec_traits::SpecFromElem<T, Global>>::spec_from_elem(elem, n, GLOBAL)
            }

            // SAFETY: The precondition is checked in the initial comparison above.
            #[cfg(not(feature = "allocator_api"))]
            unsafe {
                <SmallVec<T, N> as spec_traits::SpecFromElem<T>>::spec_from_elem(elem, n, GLOBAL)
            }
        }

        #[cfg(not(feature = "specialization"))]
        {
            // SAFETY: The precondition is checked in the initial comparison above.
            unsafe { SmallVec::<T, N>::from_elem_fallback(elem, n, GLOBAL) }
        }
    }
}

#[cfg(feature = "allocator_api")]
#[doc(hidden)]
#[track_caller]
pub fn from_elem_in<T: Clone, const N: usize, A: Allocator>(elem: T, n: usize, alloc: A) -> SmallVec<T, N, A> {
    if n > SmallVec::<T, N, A>::inline_size() {
        // Standard Rust vectors are already specialized.
        SmallVec::<T, N, A>::from_vec(alloc::vec::from_elem_in(elem, n, alloc))
    } else {
        #[cfg(feature = "specialization")]
        {
            // SAFETY: The precondition is checked in the initial comparison above.
            unsafe { <SmallVec<T, N, A> as spec_traits::SpecFromElem<T, A>>::spec_from_elem(elem, n, alloc) }
        }

        #[cfg(not(feature = "specialization"))]
        {
            // SAFETY: The precondition is checked in the initial comparison above.
            unsafe { SmallVec::<T, N, A>::from_elem_fallback(elem, n, alloc) }
        }
    }
}

#[cfg(feature = "specialization")]
mod spec_traits {
    use super::*;

    /// A trait for specializing the implementation of [`from_elem`].
    ///
    /// [`from_elem`]: crate::from_elem
    pub(crate) trait SpecFromElem<T, #[cfg(feature = "allocator_api")] A: Allocator> {
        /// Creates a `Smallvec` value where `elem` is repeated `n` times.
        /// This will use the inline storage, not the heap.
        ///
        /// # Safety
        ///
        /// The caller must ensure that `n <= Self::inline_size()`.
        unsafe fn spec_from_elem(elem: T, n: usize, alloc: A) -> Self;
    }

    #[cfg(not(feature = "allocator_api"))]
    impl<T: Clone, const N: usize> SpecFromElem<T> for SmallVec<T, N> {
        #[inline]
        default unsafe fn spec_from_elem(elem: T, n: usize, alloc: A) -> Self {
            // SAFETY: Safety conditions are identical.
            unsafe { SmallVec::from_elem_fallback(elem, n, alloc) }
        }
    }

    #[cfg(feature = "allocator_api")]
    impl<T: Clone, const N: usize, A: Allocator> SpecFromElem<T, A> for SmallVec<T, N, A> {
        #[inline]
        default unsafe fn spec_from_elem(elem: T, n: usize, alloc: A) -> Self {
            // SAFETY: Safety conditions are identical.
            unsafe { SmallVec::from_elem_fallback(elem, n, alloc) }
        }
    }

    #[cfg(not(feature = "allocator_api"))]
    impl<T: Copy, const N: usize> SpecFromElem<T> for SmallVec<T, N> {
        unsafe fn spec_from_elem(elem: T, n: usize, alloc: A) -> Self {
            let mut result = Self::new_in(alloc);

            if n > 0 {
                let ptr = result.raw.as_mut_ptr_inline();

                // SAFETY: The caller ensures that the first `n`
                // is smaller than the inline size.
                unsafe {
                    for i in 0..n {
                        ptr.add(i).write(elem);
                    }
                }
            }

            // SAFETY: The first `n` elements of the vector
            // have been initialized in the loop above.
            unsafe {
                result.set_len(n);
            }

            result
        }
    }

    #[cfg(feature = "allocator_api")]
    impl<T: Copy, const N: usize, A: Allocator> SpecFromElem<T, A> for SmallVec<T, N, A> {
        unsafe fn spec_from_elem(elem: T, n: usize, alloc: A) -> Self {
            let mut result = Self::new_in(alloc);

            if n > 0 {
                let ptr = result.raw.as_mut_ptr_inline();

                // SAFETY: The caller ensures that the first `n`
                // is smaller than the inline size.
                unsafe {
                    for i in 0..n {
                        ptr.add(i).write(elem);
                    }
                }
            }

            // SAFETY: The first `n` elements of the vector
            // have been initialized in the loop above.
            unsafe {
                result.set_len(n);
            }

            result
        }
    }

    /// A trait for specializing the implementations of [`Extend`] and [`extend_from_slice`].
    ///
    /// [`extend_from_slice`]: crate::SmallVec::extend_from_slice
    pub(crate) trait SpecExtend<T, I> {
        fn spec_extend(&mut self, iter: I);
    }

    impl<T, I, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> SpecExtend<T, I> for alloc_param!(SmallVec<T, N, A>)
    where
        I: Iterator<Item = T>,
    {
        #[inline]
        default fn spec_extend(&mut self, iter: I) {
            self.extend_fallback(iter);
        }
    }

    impl<T, I, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> SpecExtend<T, I> for alloc_param!(SmallVec<T, N, A>)
    where
        I: core::iter::TrustedLen<Item = T>,
    {
        fn spec_extend(&mut self, iter: I) {
            let (_, Some(additional)) = iter.size_hint() else {
                panic!("capacity overflow")
            };
            self.reserve(additional);

            // SAFETY: A `TrustedLen` iterator provides accurate information
            // about its size, which was used to reserve additional memory.
            // This ensures that the access operations inside the loop always
            // operate on valid memory.
            unsafe {
                let len = self.len();
                let ptr = self.as_mut_ptr().add(len);
                let mut guard = DropGuard { ptr, len: 0 };

                for x in iter {
                    ptr.add(guard.len).write(x);
                    guard.len += 1;
                }

                // The elements have been initialized in the loop above.
                self.set_len(len + guard.len);
                core::mem::forget(guard);
            }
        }
    }

    impl<T, const N: usize, const M: usize, #[cfg(feature = "allocator_api")] A: Allocator> SpecExtend<T, alloc_param!(IntoIter<T, M, A>)> for alloc_param!(SmallVec<T, N, A>) {
        fn spec_extend(&mut self, mut iter: alloc_param!(IntoIter<T, M, A>)) {
            let slice = iter.as_slice();
            let len = slice.len();
            let old_len = self.len();

            self.reserve(len);

            // SAFETY: Additional memory has been reserved above.
            // Therefore, the copy operates on valid memory.
            unsafe {
                let dst = self.as_mut_ptr().add(old_len);
                let src = slice.as_ptr();
                copy_nonoverlapping(src, dst, len);
            }

            // SAFETY: The elements were initialized above.
            unsafe {
                self.set_len(old_len + len);
            }

            // Mark the iterator as fully consumed.
            iter.begin = iter.end.value(Self::is_zst());
        }
    }

    impl<'a, T: 'a, const N: usize, I, #[cfg(feature = "allocator_api")] A: Allocator> SpecExtend<&'a T, I> for alloc_param!(SmallVec<T, N, A>)
    where
        I: Iterator<Item = &'a T>,
        T: Clone,
    {
        #[inline]
        default fn spec_extend(&mut self, iterator: I) {
            self.spec_extend(iterator.cloned())
        }
    }

    impl<'a, T: 'a, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> SpecExtend<&'a T, core::slice::Iter<'a, T>> for alloc_param!(SmallVec<T, N, A>)
    where
        T: Copy,
    {
        fn spec_extend(&mut self, iter: core::slice::Iter<'a, T>) {
            let slice = iter.as_slice();
            let len = slice.len();
            let old_len = self.len();

            self.reserve(len);

            // SAFETY: Additional memory has been reserved above.
            // Therefore, the copy operates on valid memory.
            unsafe {
                let dst = self.as_mut_ptr().add(old_len);
                let src = slice.as_ptr();
                copy_nonoverlapping(src, dst, len);
            }

            // SAFETY: The elements were initialized above.
            unsafe {
                self.set_len(old_len + len);
            }
        }
    }

    /// A trait for specializing the implementation of [`extend_from_within`].
    ///
    /// [`extend_from_within`]: crate::SmallVec::extend_from_within
    pub(crate) trait SpecExtendFromWithin<T> {
        /// Main worker for [`extend_from_within`].
        ///
        /// # Safety
        ///
        /// * The length of the vector is larger than or equal to `src.len()`.
        /// * The spare capacity of the vector is larger than or equal to `src.len()`.
        ///
        /// [`extend_from_within`]: SmallVec::extend_from_within
        unsafe fn spec_extend_from_within(&mut self, src: core::ops::Range<usize>);
    }

    impl<T: Clone, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> SpecExtendFromWithin<T> for alloc_param!(SmallVec<T, N, A>) {
        default unsafe fn spec_extend_from_within(&mut self, src: core::ops::Range<usize>) {
            // SAFETY: Safety conditions are identical.
            unsafe {
                self.extend_from_within_fallback(src);
            }
        }
    }

    impl<T: Copy, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> SpecExtendFromWithin<T> for alloc_param!(SmallVec<T, N, A>) {
        unsafe fn spec_extend_from_within(&mut self, src: core::ops::Range<usize>) {
            let old_len = self.len();

            let start = src.start;
            let len = src.len();

            // SAFETY: The caller ensures that the vector has spare capacity
            // for at least `src.len()` elements. This is alse the amount of memory
            // accessed when the data is copied.
            unsafe {
                let ptr = self.as_mut_ptr();
                let dst = ptr.add(old_len);
                let src = ptr.add(start);
                copy_nonoverlapping(src, dst, len);
            }

            // SAFETY: The elements were initialized above.
            unsafe {
                self.set_len(old_len + len);
            }
        }
    }

    /// A trait for specializing the implementation of [`FromIterator`].
    ///
    /// [`clone_from`]: Clone::clone_from
    pub(crate) trait SpecFromIterator<T, I> {
        fn spec_from_iter(iter: I) -> Self;
    }

    impl<T, I, const N: usize> SpecFromIterator<T, I> for SmallVec<T, N>
    where
        I: Iterator<Item = T>,
    {
        #[inline]
        default fn spec_from_iter(iter: I) -> Self {
            Self::from_iter_fallback(iter, GLOBAL)
        }
    }

    impl<T, I, const N: usize> SpecFromIterator<T, I> for SmallVec<T, N>
    where
        I: core::iter::TrustedLen<Item = T>,
    {
        fn spec_from_iter(iter: I) -> Self {
            let mut v = match iter.size_hint() {
                (_, Some(upper)) => SmallVec::with_capacity(upper),
                // TrustedLen contract guarantees that `size_hint() == (_, None)` means that there
                // are more than `usize::MAX` elements.
                // Since the previous branch would eagerly panic if the capacity is too large
                // (via `with_capacity`) we do the same here.
                _ => panic!("capacity overflow"),
            };
            // Reuse the extend specialization for TrustedLen.
            v.spec_extend(iter);
            v
        }
    }

    /// A trait for specializing the implementation of [`clone_from`].
    ///
    /// [`clone_from`]: Clone::clone_from
    pub(crate) trait SpecCloneFrom<T> {
        fn spec_clone_from(&mut self, source: &[T]);
    }

    impl<T: Clone, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> SpecCloneFrom<T> for alloc_param!(SmallVec<T, N, A>) {
        #[inline]
        default fn spec_clone_from(&mut self, source: &[T]) {
            self.clone_from_fallback(source);
        }
    }

    impl<T: Copy, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> SpecCloneFrom<T> for alloc_param!(SmallVec<T, N, A>) {
        fn spec_clone_from(&mut self, source: &[T]) {
            self.clear();
            self.extend_from_slice(source);
        }
    }

    /// A trait for specializing the implementation of [`From`]
    /// with the source type being slices.
    pub(crate) trait SpecFromSlice<T, #[cfg(feature = "allocator_api")] A> {
        /// Creates a `SmallVec` value based on the contents of `slice`.
        /// This will use the inline storage, not the heap.
        ///
        /// # Safety
        ///
        /// The caller must ensure that `slice.len() <= Self::inline_size()`.
        unsafe fn spec_from(slice: &[T], alloc: A) -> Self;
    }

    #[cfg(not(feature = "allocator_api"))]
    impl<T: Clone, const N: usize> SpecFromSlice<T> for SmallVec<T, N> {
        default unsafe fn spec_from(slice: &[T], alloc: A) -> Self {
            // SAFETY: Safety conditions are identical.
            unsafe { Self::from_slice_fallback(slice, alloc) }
        }
    }

    #[cfg(feature = "allocator_api")]
    impl<T: Clone, const N: usize, A: Allocator> SpecFromSlice<T, A> for SmallVec<T, N, A> {
        default unsafe fn spec_from(slice: &[T], alloc: A) -> Self {
            // SAFETY: Safety conditions are identical.
            unsafe { Self::from_slice_fallback(slice, alloc) }
        }
    }

    #[cfg(not(feature = "allocator_api"))]
    impl<T: Copy, const N: usize> SpecFromSlice<T> for SmallVec<T, N> {
        unsafe fn spec_from(slice: &[T], alloc: A) -> Self {
            let mut v = Self::new_in(alloc);

            let src = slice.as_ptr();
            let len = slice.len();
            let dst = v.as_mut_ptr();

            // SAFETY: The caller ensures that the slice length is smaller
            // than or equal to the inline length.
            unsafe {
                copy_nonoverlapping(src, dst, len);
            }

            // SAFETY: The elements were initialized above.
            unsafe {
                v.set_len(len);
            }

            v
        }
    }

    #[cfg(feature = "allocator_api")]
    impl<T: Copy, const N: usize, A: Allocator> SpecFromSlice<T, A> for SmallVec<T, N, A> {
        unsafe fn spec_from(slice: &[T], alloc: A) -> Self {
            let mut v = Self::new_in(alloc);

            let src = slice.as_ptr();
            let len = slice.len();
            let dst = v.as_mut_ptr();

            // SAFETY: The caller ensures that the slice length is smaller
            // than or equal to the inline length.
            unsafe {
                copy_nonoverlapping(src, dst, len);
            }

            // SAFETY: The elements were initialized above.
            unsafe {
                v.set_len(len);
            }

            v
        }
    }
}

/// Fallback functions for various specialized methods. These are kept in
/// a separate implementation block for easy access whenever specialization is disabled.
impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> alloc_param!(SmallVec<T, N, A>) {
    /// Creates a `Smallvec` value where `elem` is repeated `n` times.
    /// This will use the inline storage, not the heap.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `n <= Self::inline_size()`.
    unsafe fn from_elem_fallback(elem: T, n: usize, alloc: A) -> Self
    where
        T: Clone,
    {
        let mut result = Self::new_in(alloc);

        if n > 0 {
            let ptr = result.raw.as_mut_ptr_inline();
            let mut guard = DropGuard { ptr, len: 0 };

            // SAFETY: The caller ensures that the first `n`
            // is smaller than the inline size.
            unsafe {
                for i in 0..(n - 1) {
                    ptr.add(i).write(elem.clone());
                    guard.len += 1;
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

    fn extend_fallback<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        let iter = iter.into_iter();
        let (size, _) = iter.size_hint();
        self.reserve(size);
        for x in iter {
            self.push(x);
        }
    }

    /// Main worker for [`extend_from_within`].
    ///
    /// # Safety
    ///
    /// * The length of the vector is larger than or equal to `src.len()`.
    /// * The spare capacity of the vector is larger than or equal to `src.len()`.
    ///
    /// [`extend_from_within`]: SmallVec::extend_from_within
    unsafe fn extend_from_within_fallback(&mut self, src: core::ops::Range<usize>)
    where
        T: Clone,
    {
        let old_len = self.len();

        let start = src.start;
        let len = src.len();

        // SAFETY: The caller ensures that the vector has spare capacity
        // for at least `src.len()` elements. This implies that the loop
        // operates on valid memory.
        unsafe {
            let ptr = self.as_mut_ptr();
            let dst = ptr.add(old_len);
            let src = ptr.add(start);

            let mut guard = DropGuard { ptr: dst, len: 0 };
            for i in 0..len {
                let val = (*src.add(i)).clone();
                dst.add(i).write(val);
                guard.len += 1;
            }
            core::mem::forget(guard);
        }

        // SAFETY: The elements were initialized in the loop above.
        unsafe {
            self.set_len(old_len + len);
        }
    }

    fn from_iter_fallback<I>(iter: I, alloc: A) -> Self
    where
        I: Iterator<Item = T>,
    {
        let (size, _) = iter.size_hint();
        let mut v = Self::with_capacity_in(size, alloc);
        for x in iter {
            v.push(x);
        }
        v
    }

    fn clone_from_fallback(&mut self, source: &[T])
    where
        T: Clone,
    {
        // Inspired from `impl Clone for Vec`.

        // Drop anything that will not be overwritten.
        self.truncate(source.len());

        // SAFETY: self.len <= other.len due to the truncate above, so the
        // slices here are always in-bounds.
        let (init, tail) = unsafe { source.split_at_unchecked(self.len()) };

        // Reuse the contained values' allocations/resources.
        self.clone_from_slice(init);
        self.extend(tail.iter().cloned());
    }

    /// Creates a `SmallVec` value based on the contents of `slice`.
    /// This will use the inline storage, not the heap.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `slice.len() <= Self::inline_size()`.
    unsafe fn from_slice_fallback(slice: &[T], alloc: A) -> Self
    where
        T: Clone,
    {
        let mut v = Self::new_in(alloc);

        let src = slice.as_ptr();
        let len = slice.len();
        let dst = v.as_mut_ptr();

        // SAFETY: The caller ensures that the slice length is smaller
        // than or equal to the inline length.
        unsafe {
            let mut guard = DropGuard { ptr: dst, len: 0 };
            for i in 0..len {
                let val = (*src.add(i)).clone();
                dst.add(i).write(val);
                guard.len += 1;
            }
            core::mem::forget(guard);
        }

        // SAFETY: The elements were initialized in the loop above.
        unsafe {
            v.set_len(len);
        }

        v
    }
}

impl<T: Clone, const N: usize> From<&[T]> for SmallVec<T, N> {
    #[inline]
    fn from(slice: &[T]) -> Self {
        if slice.len() > Self::inline_size() {
            // Standard Rust vectors are already specialized.
            Self::from_vec(Vec::from(slice))
        } else {
            // SAFETY: The precondition is checked in the initial comparison above.
            unsafe {
                #[cfg(feature = "specialization")]
                {
                    #[cfg(feature = "allocator_api")]
                    {
                        <Self as spec_traits::SpecFromSlice<T, Global>>::spec_from(slice, GLOBAL)
                    }

                    #[cfg(not(feature = "allocator_api"))]
                    {
                        <Self as spec_traits::SpecFromSlice<T>>::spec_from(slice, GLOBAL)
                    }
                }

                #[cfg(not(feature = "specialization"))]
                {
                    Self::from_slice_fallback(slice, GLOBAL)
                }
            }
        }
    }
}

impl<T: Clone, const N: usize> From<&mut [T]> for SmallVec<T, N> {
    #[inline]
    fn from(slice: &mut [T]) -> Self {
        Self::from(slice as &[T])
    }
}

impl<T: Clone, const M: usize, const N: usize> From<&[T; M]> for SmallVec<T, N> {
    #[inline]
    fn from(slice: &[T; M]) -> Self {
        Self::from(slice as &[T])
    }
}

impl<T: Clone, const M: usize, const N: usize> From<&mut [T; M]> for SmallVec<T, N> {
    #[inline]
    fn from(slice: &mut [T; M]) -> Self {
        Self::from(slice as &[T])
    }
}

impl<T, const N: usize, const M: usize> From<[T; M]> for SmallVec<T, N> {
    fn from(array: [T; M]) -> Self {
        if M > N {
            // If M > N, we'd have to heap allocate anyway,
            // so delegate for Vec for the allocation.
            Self::from(Vec::from(array))
        } else {
            // M <= N
            let mut this = Self::new();
            debug_assert!(M <= this.capacity());
            let array = ManuallyDrop::new(array);
            // SAFETY: M <= this.capacity()
            unsafe {
                copy_nonoverlapping(array.as_ptr(), this.as_mut_ptr(), M);
                this.set_len(M);
            }
            this
        }
    }
}

impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> From<alloc_param!(Vec<T, A>)> for alloc_param!(SmallVec<T, N, A>) {
    fn from(v: alloc_param!(Vec<T, A>)) -> Self {
        Self::from_vec(v)
    }
}

impl<T: Clone, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator + Clone> Clone for alloc_param!(SmallVec<T, N, A>) {
    #[inline]
    fn clone(&self) -> Self {
        let alloc = self.raw.allocator.clone();
        Self::from_slice_in(self.as_slice(), alloc)
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        #[cfg(feature = "specialization")]
        {
            <Self as spec_traits::SpecCloneFrom<T>>::spec_clone_from(self, source);
        }

        #[cfg(not(feature = "specialization"))]
        {
            self.clone_from_fallback(&*source);
        }
    }
}

impl<T: Clone, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator + Clone> Clone for alloc_param!(IntoIter<T, N, A>) {
    #[inline]
    fn clone(&self) -> Self {
        let alloc = self.raw.allocator.clone();
        SmallVec::from_slice_in(self.as_slice(), alloc).into_iter()
    }
}

impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Extend<T> for alloc_param!(SmallVec<T, N, A>) {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        #[cfg(feature = "specialization")]
        {
            spec_traits::SpecExtend::<T, _>::spec_extend(self, iter.into_iter());
        }

        #[cfg(not(feature = "specialization"))]
        {
            self.extend_fallback(iter);
        }
    }
}

impl<'a, T: Clone + 'a, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Extend<&'a T> for alloc_param!(SmallVec<T, N, A>) {
    #[inline]
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        #[cfg(feature = "specialization")]
        {
            spec_traits::SpecExtend::<&'a T, _>::spec_extend(self, iter.into_iter());
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
            spec_traits::SpecFromIterator::<T, _>::spec_from_iter(iter.into_iter())
        }

        #[cfg(not(feature = "specialization"))]
        {
            Self::from_iter_fallback(iter.into_iter(), GLOBAL)
        }
    }
}

#[macro_export]
macro_rules! smallvec {
    // count helper: transform any expression into 1
    (@one $x:expr) => (1usize);
    () => (
        $crate::SmallVec::new()
    );
    ($elem:expr; $n:expr) => ({
        $crate::from_elem($elem, $n)
    });
    ($($x:expr),+$(,)?) => ({
        const COUNT: usize = 0usize $(+ $crate::smallvec!(@one $x))+;
        let mut vec = $crate::SmallVec::new();
        if COUNT <= vec.capacity() {
            $(vec.push($x);)*
            vec
        } else {
            $crate::SmallVec::from_vec($crate::alloc::vec![$($x,)+])
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

impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> IntoIterator for alloc_param!(SmallVec<T, N, A>) {
    type IntoIter = alloc_param!(IntoIter<T, N, A>);
    type Item = T;
    fn into_iter(self) -> Self::IntoIter {
        // SAFETY: we move out of this.raw by reading the value at its address, which is fine since
        // we don't drop it
        unsafe {
            // Set SmallVec len to zero as `IntoIter` drop handles dropping of the elements
            let this = ManuallyDrop::new(self);
            IntoIter {
                raw: (&this.raw as *const alloc_param!(RawSmallVec<T, N, A>)).read(),
                begin: 0,
                end: this.len,
                _marker: PhantomData,
            }
        }
    }
}

impl<'a, T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> IntoIterator for &'a alloc_param!(SmallVec<T, N, A>) {
    type IntoIter = core::slice::Iter<'a, T>;
    type Item = &'a T;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> IntoIterator for &'a mut alloc_param!(SmallVec<T, N, A>) {
    type IntoIter = core::slice::IterMut<'a, T>;
    type Item = &'a mut T;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, U, const N: usize, const M: usize,
    #[cfg(feature = "allocator_api")] A1: Allocator,
    #[cfg(feature = "allocator_api")] A2: Allocator,
> PartialEq<alloc_param!(SmallVec<U, M, A2>)> for alloc_param!(SmallVec<T, N, A1>)
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &alloc_param!(SmallVec<U, M, A2>)) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}
impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Eq for alloc_param!(SmallVec<T, N, A>) where T: Eq {}

impl<T, U, const N: usize, const M: usize, #[cfg(feature = "allocator_api")] A: Allocator> PartialEq<[U; M]> for alloc_param!(SmallVec<T, N, A>)
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &[U; M]) -> bool {
        self[..] == other[..]
    }
}

impl<T, U, const N: usize, const M: usize, #[cfg(feature = "allocator_api")] A: Allocator> PartialEq<&[U; M]> for alloc_param!(SmallVec<T, N, A>)
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &&[U; M]) -> bool {
        self[..] == other[..]
    }
}

impl<T, U, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> PartialEq<[U]> for alloc_param!(SmallVec<T, N, A>)
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &[U]) -> bool {
        self[..] == other[..]
    }
}

impl<T, U, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> PartialEq<&[U]> for alloc_param!(SmallVec<T, N, A>)
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &&[U]) -> bool {
        self[..] == other[..]
    }
}

impl<T, U, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> PartialEq<&mut [U]> for alloc_param!(SmallVec<T, N, A>)
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &&mut [U]) -> bool {
        self[..] == other[..]
    }
}

impl<T, const N: usize, const M: usize,
    #[cfg(feature = "allocator_api")] A1: Allocator,
    #[cfg(feature = "allocator_api")] A2: Allocator,
    > PartialOrd<alloc_param!(SmallVec<T, M, A2>)> for alloc_param!(SmallVec<T, N, A1>)
where
    T: PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, other: &alloc_param!(SmallVec<T, M, A2>)) -> Option<core::cmp::Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Ord for alloc_param!(SmallVec<T, N, A>)
where
    T: Ord,
{
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<T: Hash, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Hash for alloc_param!(SmallVec<T, N, A>) {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state)
    }
}

impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Borrow<[T]> for alloc_param!(SmallVec<T, N, A>) {
    #[inline]
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> BorrowMut<[T]> for alloc_param!(SmallVec<T, N, A>) {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> AsRef<[T]> for alloc_param!(SmallVec<T, N, A>) {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> AsMut<[T]> for alloc_param!(SmallVec<T, N, A>) {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Debug, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Debug for alloc_param!(SmallVec<T, N, A>) {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: Debug, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Debug for alloc_param!(IntoIter<T, N, A>) {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("IntoIter").field(&self.as_slice()).finish()
    }
}

impl<T: Debug, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Debug for alloc_param!(Drain<'_, T, N, A>) {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Drain").field(&self.iter.as_slice()).finish()
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> Serialize for alloc_param!(SmallVec<T, N, A>)
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
        use serde_core::de::Error;
        let len = seq.size_hint().unwrap_or(0);
        let mut values = SmallVec::new();
        values.try_reserve(len).map_err(B::Error::custom)?;

        while let Some(value) = seq.next_element()? {
            values.push(value);
        }

        Ok(values)
    }
}

#[cfg(feature = "malloc_size_of")]
impl<T, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> MallocShallowSizeOf for alloc_param!(SmallVec<T, N, A>) {
    fn shallow_size_of(&self, ops: &mut MallocSizeOfOps) -> usize {
        if self.spilled() {
            unsafe { ops.malloc_size_of(self.as_ptr()) }
        } else {
            0
        }
    }
}

#[cfg(feature = "malloc_size_of")]
impl<T: MallocSizeOf, const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> MallocSizeOf for alloc_param!(SmallVec<T, N, A>) {
    fn size_of(&self, ops: &mut MallocSizeOfOps) -> usize {
        let mut n = self.shallow_size_of(ops);
        for elem in self.iter() {
            n += elem.size_of(ops);
        }
        n
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> io::Write for alloc_param!(SmallVec<u8, N, A>) {
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
unsafe impl<const N: usize, #[cfg(feature = "allocator_api")] A: Allocator> BufMut for alloc_param!(SmallVec<u8, N, A>) {
    #[inline]
    fn remaining_mut(&self) -> usize {
        // A vector can never have more than isize::MAX bytes
        isize::MAX as usize - self.len()
    }

    #[inline]
    unsafe fn advance_mut(&mut self, cnt: usize) {
        let len = self.len();
        let remaining = self.capacity() - len;

        if remaining < cnt {
            panic!("advance out of bounds: the len is {remaining} but advancing by {cnt}");
        }

        // Addition will not overflow since the sum is at most the capacity.
        self.set_len(len + cnt);
    }

    #[inline]
    fn chunk_mut(&mut self) -> &mut UninitSlice {
        if self.capacity() == self.len() {
            self.reserve(64); // Grow the smallvec
        }

        let cap = self.capacity();
        let len = self.len();

        let ptr = self.as_mut_ptr();
        // SAFETY: Since `ptr` is valid for `cap` bytes, `ptr.add(len)` must be
        // valid for `cap - len` bytes. The subtraction will not underflow since
        // `len <= cap`.
        unsafe { UninitSlice::from_raw_parts_mut(ptr.add(len), cap - len) }
    }

    // Specialize these methods so they can skip checking `remaining_mut`
    // and `advance_mut`.
    #[inline]
    fn put<T: bytes::Buf>(&mut self, mut src: T)
    where
        Self: Sized,
    {
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
