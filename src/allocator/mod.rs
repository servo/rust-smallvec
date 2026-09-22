#[cfg(not(feature = "allocator-api"))]
mod base;
#[cfg(all(feature = "allocator-api", not(feature = "allocator-api2")))]
mod native;
#[cfg(feature = "allocator-api2")]
mod external;

use core::{
    alloc::Layout,
    ptr::NonNull
};

#[cfg(not(feature = "allocator-api"))]
#[rustfmt::skip]
pub use {
    alloc::{
        boxed::Box,
        vec::Vec,
        vec
    },
    base::Global
};
#[cfg(all(feature = "allocator-api", not(feature = "allocator-api2")))]
#[rustfmt::skip]
pub use alloc::{
    alloc::Global,
    boxed::Box,
    vec::Vec,
    vec
};
#[cfg(feature = "allocator-api2")]
#[rustfmt::skip]
pub use allocator_api2::{
    alloc::Global,
    boxed::Box,
    vec::Vec,
    vec
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
