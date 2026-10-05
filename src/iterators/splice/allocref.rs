use {
    super::Allocator,
    core::{
        alloc::Layout,
        ptr::NonNull
    }
};

pub struct Allocref<'valid, Heap: Allocator>(pub &'valid Heap);

impl<'valid, Heap: Allocator> Allocator for Allocref<'valid, Heap> {
    #[inline]
    fn allocate(&self, layout: Layout) -> Option<NonNull<[u8]>> {
        self.0.allocate(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, pointer: NonNull<u8>, layout: Layout) {
        unsafe { self.0.deallocate(pointer, layout) };
    }

    #[inline]
    unsafe fn grow(&self, pointer: NonNull<u8>, old: Layout, new: Layout) -> Option<NonNull<[u8]>> {
        unsafe { self.0.grow(pointer, old, new) }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        pointer: NonNull<u8>,
        old: Layout,
        new: Layout
    ) -> Option<NonNull<[u8]>> {
        unsafe { self.0.shrink(pointer, old, new) }
    }
}
