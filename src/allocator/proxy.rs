#[cfg(all(feature = "allocator-api", not(feature = "allocator-api2")))]
use alloc::alloc::AllocError;
#[cfg(feature = "allocator-api2")]
use allocator_api2::alloc::AllocError;
#[cfg(feature = "allocator-api")]
use core::{
    alloc::Layout,
    ptr::NonNull
};
use super::{
    Allocator,
    requirement::Requirement
};

#[repr(transparent)]
pub struct Proxy<'valid, Heap: Allocator>(pub &'valid Heap);

#[cfg(not(feature = "allocator-api"))]
impl<'valid, Heap: Allocator> Requirement for Proxy<'valid, Heap> {}
#[cfg(all(feature = "allocator-api", not(feature = "allocator-api2")))]
unsafe impl<'valid, Heap: Allocator> Requirement for Proxy<'valid, Heap> {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Requirement::allocate(&self, layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe {
            Requirement::deallocate(&self, ptr, layout);
        }
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout
    ) -> Result<NonNull<[u8]>, AllocError> {
        unsafe { Requirement::grow(&self, ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout
    ) -> Result<NonNull<[u8]>, AllocError> {
        unsafe { Requirement::shrink(&self, ptr, old_layout, new_layout) }
    }
}
#[cfg(feature = "allocator-api2")]
unsafe impl<'valid, Heap: Allocator> Requirement for Proxy<'valid, Heap> {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Requirement::allocate(&self, layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe {
            Requirement::deallocate(&self, ptr, layout);
        }
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout
    ) -> Result<NonNull<[u8]>, AllocError> {
        unsafe { Requirement::grow(&self, ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout
    ) -> Result<NonNull<[u8]>, AllocError> {
        unsafe { Requirement::shrink(&self, ptr, old_layout, new_layout) }
    }
}
