#[cfg(not(feature = "allocator-api"))]
mod base;
#[cfg(all(feature = "allocator-api", not(feature = "allocator-api2")))]
mod native;
#[cfg(feature = "allocator-api2")]
mod external;