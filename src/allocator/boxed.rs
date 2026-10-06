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

pub struct Box<Item: ?Sized, Heap: Allocator + Requirement>(Infallible, PhantomData<(Heap, Item)>);

impl<Item: ?Sized, Heap: Allocator + Requirement> Like for Box<Item, Heap> {
    #[cfg(not(feature = "allocator-api"))]
    type Type = alloc::boxed::Box<Item>;
    #[cfg(all(feature = "allocator-api", not(feature = "allocator-api2")))]
    type Type = alloc::boxed::Box<Item, Heap>;
    #[cfg(feature = "allocator-api2")]
    type Type = allocator_api2::boxed::Box<Item, Heap>;
}
