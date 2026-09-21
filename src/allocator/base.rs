pub use alloc::{
    boxed::Box,
    vec::Vec
};
use {
    super::Allocator,
    alloc::alloc::{
        alloc,
        dealloc,
        realloc
    },
    core::{
        alloc::Layout,
        ptr::NonNull
    }
};

#[derive(Clone)]
pub struct Global;

impl Allocator for Global {
    #[inline]
    fn allocate(&self, layout: Layout) -> Option<NonNull<[u8]>> {
        Some(NonNull::slice_from_raw_parts(
            NonNull::new(unsafe { alloc(layout) })?,
            layout.size()
        ))
    }

    #[inline]
    unsafe fn deallocate(&self, pointer: NonNull<u8>, layout: Layout) -> () {
        unsafe { dealloc(pointer.as_ptr(), layout) };
    }

    #[inline]
    unsafe fn grow(&self, pointer: NonNull<u8>, old: Layout, new: Layout) -> Option<NonNull<[u8]>> {
        Some(NonNull::slice_from_raw_parts(
            NonNull::new(unsafe { realloc(pointer.as_ptr(), old, new.size()) })?,
            new.size()
        ))
    }

    #[inline]
    unsafe fn shrink(
        &self,
        pointer: NonNull<u8>,
        old: Layout,
        new: Layout
    ) -> Option<NonNull<[u8]>> {
        Some(NonNull::slice_from_raw_parts(
            NonNull::new(unsafe { realloc(pointer.as_ptr(), old, new.size()) })?,
            new.size()
        ))
    }
}
