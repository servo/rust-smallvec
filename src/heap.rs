// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use self::TryReserveError::CapacityOverflow as CO;
use alloc::alloc::{self, Layout};
use core::mem;
use core::ptr::NonNull;

fn array_layout<T>(capacity: usize) -> Result<Layout, TryReserveError> {
    let size = mem::size_of::<T>().checked_mul(capacity).ok_or(CO)?;
    if size > crate::repr::MAX_LEN {
        return Err(CO);
    }
    Layout::from_size_align(size, mem::align_of::<T>()).map_err(|_| CO)
}

pub(crate) unsafe fn alloc_array<T>(capacity: usize) -> Result<NonNull<T>, TryReserveError> {
    let layout = array_layout::<T>(capacity)?;
    let ptr = alloc::alloc(layout);
    NonNull::new(ptr)
        .map(NonNull::cast)
        .ok_or(TryReserveError::AllocError { layout })
}

pub(crate) unsafe fn realloc_array<T>(
    ptr: NonNull<T>,
    old: usize,
    new: usize,
) -> Result<NonNull<T>, TryReserveError> {
    let old = array_layout::<T>(old)?;
    let new = array_layout::<T>(new)?;
    let ptr = alloc::realloc(ptr.cast().as_ptr(), old, new.size());
    NonNull::new(ptr)
        .map(NonNull::cast)
        .ok_or(TryReserveError::AllocError { layout: new })
}

pub(crate) unsafe fn dealloc_array<T>(ptr: NonNull<T>, capacity: usize) {
    let layout = array_layout::<T>(capacity).unwrap();
    alloc::dealloc(ptr.cast().as_ptr(), layout)
}

#[derive(Debug)]
pub enum TryReserveError {
    CapacityOverflow,
    AllocError { layout: Layout },
}

impl TryReserveError {
    pub(crate) fn bail(&self) -> ! {
        match *self {
            TryReserveError::CapacityOverflow => panic!("SmallVec capacity overflow"),
            TryReserveError::AllocError { layout } => alloc::handle_alloc_error(layout),
        }
    }
}
