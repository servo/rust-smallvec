#[cfg(not(feature = "allocator-api"))]
mod base;
#[cfg(feature = "allocator-api2")]
mod external;
#[cfg(all(feature = "allocator-api", not(feature = "allocator-api2")))]
mod native;

#[cfg(not(feature = "allocator-api"))]
pub use base::{
    Box,
    Global,
    Vec
};
use core::{
    alloc::Layout,
    ptr::NonNull
};
#[cfg(feature = "allocator-api2")]
pub use external::{
    Box,
    Global,
    Vec
};
#[cfg(all(feature = "allocator-api", not(feature = "allocator-api2")))]
pub use native::{
    Box,
    Global,
    Vec
};

pub trait Allocator {
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
