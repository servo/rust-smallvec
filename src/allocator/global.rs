#[cfg(not(feature = "allocator-api"))]
pub struct Global;

#[cfg(all(feature = "allocator-api", not(feature = "allocator-api2")))]
pub use alloc::alloc::Global;
#[cfg(feature = "allocator-api2")]
pub use allocator_api2::alloc::Global;
