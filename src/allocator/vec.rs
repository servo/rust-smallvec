#[cfg(not(feature = "allocator-api2"))]
pub use alloc::vec;
#[cfg(feature = "allocator-api2")]
pub use allocator_api2::vec;
use {
    super::{
        Allocator,
        like::Like,
        requirement::Requirement
    },
    core::{
        convert::Infallible,
        marker::PhantomData
    }
};

pub struct Vec<Item, Heap: Allocator + Requirement>(Infallible, PhantomData<(Heap, Item)>);

impl<Item, Heap: Allocator + Requirement> Like for Vec<Item, Heap> {
    #[cfg(not(feature = "allocator-api"))]
    type Type = alloc::vec::Vec<Item>;
    #[cfg(all(feature = "allocator-api", not(feature = "allocator-api2")))]
    type Type = alloc::vec::Vec<Item, Heap>;
    #[cfg(feature = "allocator-api2")]
    type Type = allocator_api2::vec::Vec<Item, Heap>;
}
