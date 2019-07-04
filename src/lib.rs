// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![no_std]
#![cfg_attr(feature = "may_dangle", feature(dropck_eyepatch))]

extern crate alloc;

mod array;
mod heap;
mod repr;
mod tagged_union;

pub use array::Array;
pub use heap::TryReserveError;
pub use repr::SmallVec;

use alloc::vec::Vec;

impl<A: Array> SmallVec<A> {
    const INLINE_CAPACITY: usize = A::LEN;

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn capacity(&self) -> usize {
        match self.as_inline() {
            Ok(_) => Self::INLINE_CAPACITY,
            Err(heap) => heap.capacity,
        }
    }

    pub fn as_ptr(&self) -> *const A::Item {
        match self.as_inline() {
            Ok(inline) => inline.as_ptr(),
            Err(heap) => heap.ptr.as_ptr(),
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut A::Item {
        match self.as_inline_mut() {
            Ok(inline) => inline.as_mut_ptr(),
            Err(heap) => heap.ptr.as_ptr(),
        }
    }

    pub fn as_slice(&self) -> &[A::Item] {
        unsafe { core::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [A::Item] {
        unsafe { core::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }

    pub fn shrink_to_fit(&mut self) {
        self.shrink_to(0)
    }

    pub fn shrink_to(&mut self, new_capacity: usize) {
        let len = self.len();
        let new_capacity = new_capacity.max(len);
        if let Err(heap) = self.as_inline_mut() {
            if new_capacity <= Self::INLINE_CAPACITY {
                let ptr = heap.ptr;
                let capacity = heap.capacity;
                // Do accesses to `heap` above here, so we can borrow `self` again below:
                unsafe {
                    let inline = self.set_inline_tag(len);
                    core::ptr::copy_nonoverlapping(ptr.as_ptr(), inline.as_mut_ptr(), len);
                    heap::dealloc_array(ptr, capacity)
                }
            } else if new_capacity < heap.capacity {
                let new_ptr = unsafe {
                    heap::realloc_array(heap.ptr, heap.capacity, new_capacity)
                        .unwrap_or_else(|err| err.bail())
                };
                heap.ptr = new_ptr;
                heap.capacity = new_capacity;
            }
        }
    }

    fn try_reserve_internal(&mut self, extra: usize, exact: bool) -> Result<(), TryReserveError> {
        let len = self.len();
        let capacity = self.capacity();
        let requested_capacity = len
            .checked_add(extra)
            .ok_or(TryReserveError::CapacityOverflow)?;
        if requested_capacity <= capacity {
            return Ok(());
        }
        let new_capacity = if exact {
            requested_capacity
        } else {
            requested_capacity.max(capacity * 2)
        };
        unsafe {
            match self.as_inline_mut() {
                Ok(inline) => {
                    let new_ptr = heap::alloc_array(new_capacity)?;
                    core::ptr::copy_nonoverlapping(inline.as_ptr(), new_ptr.as_ptr(), len);
                    let heap = self.set_heap_tag(len);
                    heap.ptr = new_ptr;
                    heap.capacity = new_capacity;
                }
                Err(heap) => {
                    let new_ptr = heap::realloc_array(heap.ptr, capacity, new_capacity)?;
                    heap.ptr = new_ptr;
                    heap.capacity = new_capacity;
                }
            }
        }
        Ok(())
    }

    pub fn reserve(&mut self, extra: usize) {
        self.try_reserve_internal(extra, false)
            .unwrap_or_else(|err| err.bail())
    }

    pub fn reserve_exact(&mut self, extra: usize) {
        self.try_reserve_internal(extra, true)
            .unwrap_or_else(|err| err.bail())
    }

    pub fn try_reserve(&mut self, extra: usize) -> Result<(), TryReserveError> {
        self.try_reserve_internal(extra, false)
    }

    pub fn try_reserve_exact(&mut self, extra: usize) -> Result<(), TryReserveError> {
        self.try_reserve_internal(extra, true)
    }

    pub fn push(&mut self, value: A::Item) {
        self.reserve(1);
        let len = self.len();
        unsafe {
            self.as_mut_ptr().add(len).write(value);
            self.set_len(len + 1)
        }
    }

    pub fn clear(&mut self) {
        self.truncate(0)
    }

    pub fn truncate(&mut self, new_len: usize) {
        if let Some(to_drop) = self.get_mut(new_len..) {
            let to_drop: *mut [A::Item] = to_drop;
            unsafe {
                self.set_len(new_len);
                to_drop.drop_in_place()
            }
        }
    }
}

#[cfg(feature = "may_dangle")]
unsafe impl<#[may_dangle] A: Array> Drop for SmallVec<A> {
    fn drop(&mut self) {
        drop_impl(self)
    }
}

#[cfg(not(feature = "may_dangle"))]
impl<A: Array> Drop for SmallVec<A> {
    fn drop(&mut self) {
        drop_impl(self)
    }
}

fn drop_impl<A: Array>(s: &mut SmallVec<A>) {
    unsafe {
        core::ptr::drop_in_place(s.as_mut_slice());
        match s.as_inline() {
            Ok(_) => {}
            Err(heap) => heap::dealloc_array(heap.ptr, heap.capacity),
        }
    }
}

impl<A: Array> core::ops::Deref for SmallVec<A> {
    type Target = [A::Item];
    fn deref(&self) -> &[A::Item] {
        self.as_slice()
    }
}

impl<A: Array> core::ops::DerefMut for SmallVec<A> {
    fn deref_mut(&mut self) -> &mut [A::Item] {
        self.as_mut_slice()
    }
}

impl<A: Array, I> core::ops::Index<I> for SmallVec<A>
where
    I: core::slice::SliceIndex<[A::Item]>,
{
    type Output = I::Output;
    fn index(&self, index: I) -> &I::Output {
        &self.as_slice()[index]
    }
}

impl<A: Array, I> core::ops::IndexMut<I> for SmallVec<A>
where
    I: core::slice::SliceIndex<[A::Item]>,
{
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        &mut self.as_mut_slice()[index]
    }
}

impl<A: Array> From<Vec<A::Item>> for SmallVec<A> {
    fn from(mut vec: Vec<A::Item>) -> Self {
        let ptr = vec.as_mut_ptr();
        let len = vec.len();
        let cap = vec.capacity();
        core::mem::forget(vec);
        let mut s = Self::new();
        unsafe {
            let heap = s.set_heap_tag(len);
            heap.ptr = core::ptr::NonNull::new_unchecked(ptr);
            heap.capacity = cap;
        }
        s
    }
}
