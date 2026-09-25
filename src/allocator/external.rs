use {
    super::Allocator,
    allocator_api2::alloc::Allocator as External,
    core::{
        alloc::Layout,
        ptr::NonNull
    }
};

impl<Type: External> Allocator for Type {
    #[inline]
    fn allocate(&self, layout: Layout) -> Option<NonNull<[u8]>> {
        External::allocate(&self, layout).ok()
    }

    #[inline]
    unsafe fn deallocate(&self, pointer: NonNull<u8>, layout: Layout) {
        unsafe { External::deallocate(&self, pointer, layout) }
    }

    #[inline]
    unsafe fn grow(&self, pointer: NonNull<u8>, old: Layout, new: Layout) -> Option<NonNull<[u8]>> {
        unsafe { External::grow(&self, pointer, old, new).ok() }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        pointer: NonNull<u8>,
        old: Layout,
        new: Layout
    ) -> Option<NonNull<[u8]>> {
        unsafe { External::shrink(&self, pointer, old, new).ok() }
    }
}
