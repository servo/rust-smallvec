#[cfg(not(feature = "allocator-api2"))]
pub use alloc::boxed::Box;
#[cfg(feature = "allocator-api2")]
pub use allocator_api2::boxed::Box;
