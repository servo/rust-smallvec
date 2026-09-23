use {
    super::Allocator,
    alloc::alloc::Allocator as Native,
    core::{
        alloc::Layout,
        ptr::NonNull
    }
};

impl<Type: Native> Allocator for Type {
    #[inline]
    fn allocate(&self, layout: Layout) -> Option<NonNull<[u8]>> {
        Native::allocate(&self, layout).ok()
    }

    #[inline]
    unsafe fn deallocate(&self, pointer: NonNull<u8>, layout: Layout) {
        unsafe { Native::deallocate(&self, pointer, layout) }
    }

    #[inline]
    unsafe fn grow(&self, pointer: NonNull<u8>, old: Layout, new: Layout) -> Option<NonNull<[u8]>> {
        unsafe { Native::grow(&self, pointer, old, new).ok() }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        pointer: NonNull<u8>,
        old: Layout,
        new: Layout
    ) -> Option<NonNull<[u8]>> {
        unsafe { Native::shrink(&self, pointer, old, new).ok() }
    }
}
