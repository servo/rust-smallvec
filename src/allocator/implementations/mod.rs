use {
    super::{
        global::Global,
        requirement::Requirement,
        Allocator
    },
    core::{
        alloc::Layout,
        ptr::NonNull
    },
    alloc::alloc::{
        alloc,
        dealloc,
        realloc
    }
};

#[cfg(not(feature = "allocator-api"))]
impl Requirement for Global {}

#[cfg(not(feature = "allocator-api"))]
impl<Type: Requirement> Allocator for Type {
    #[inline]
    fn allocate(&self, layout: Layout) -> Option<NonNull<[u8]>> {
        Some(NonNull::slice_from_raw_parts(
            NonNull::new(unsafe { alloc(layout) })?,
            layout.size()
        ))
    }

    #[inline]
    unsafe fn deallocate(&self, pointer: NonNull<u8>, layout: Layout) {
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

#[cfg(feature = "allocator-api")]
impl<Type: Requirement> Allocator for Type {
    #[inline]
    fn allocate(&self, layout: Layout) -> Option<NonNull<[u8]>> {
        Requirement::allocate(&self, layout).ok()
    }

    #[inline]
    unsafe fn deallocate(&self, pointer: NonNull<u8>, layout: Layout) {
        unsafe { Requirement::deallocate(&self, pointer, layout) }
    }

    #[inline]
    unsafe fn grow(&self, pointer: NonNull<u8>, old: Layout, new: Layout) -> Option<NonNull<[u8]>> {
        unsafe { Requirement::grow(&self, pointer, old, new).ok() }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        pointer: NonNull<u8>,
        old: Layout,
        new: Layout
    ) -> Option<NonNull<[u8]>> {
        unsafe { Requirement::shrink(&self, pointer, old, new).ok() }
    }
}
