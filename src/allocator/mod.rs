pub mod boxed;
pub mod global;
mod implementations;
pub mod like;
pub mod proxy;
mod requirement;
pub mod vec;

use {
    core::{
        alloc::Layout,
        ptr::NonNull
    },
    requirement::Requirement
};

pub trait Allocator: Requirement {
    fn allocate(&self, layout: Layout) -> Option<NonNull<[u8]>>;
    unsafe fn deallocate(&self, pointer: NonNull<u8>, layout: Layout);
    unsafe fn grow(&self, pointer: NonNull<u8>, old: Layout, new: Layout) -> Option<NonNull<[u8]>>;
    unsafe fn shrink(
        &self,
        pointer: NonNull<u8>,
        old: Layout,
        new: Layout
    ) -> Option<NonNull<[u8]>>;
}
