#[cfg(not(feature = "allocator-api2"))]
#[rustfmt::skip]
pub use alloc::{
    vec,
    vec::Vec
};

#[cfg(feature = "allocator-api2")]
#[rustfmt::skip]
pub use allocator_api2::{
    vec,
    vec::Vec
};
