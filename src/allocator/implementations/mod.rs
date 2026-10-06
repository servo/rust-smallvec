#[cfg(not(feature = "allocator-api"))]
mod base;
#[cfg(feature = "allocator-api2")]
mod external;
#[cfg(all(feature = "allocator-api", not(feature = "allocator-api2")))]
mod native;
