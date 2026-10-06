#[cfg(all(feature = "allocator-api", not(feature = "allocator-api2")))]
pub use alloc::alloc::Allocator as Requirement;
#[cfg(feature = "allocator-api2")]
pub use allocator_api2::alloc::Allocator as Requirement;
#[cfg(not(feature = "allocator-api"))]
pub use core::marker::Sized as Requirement;
