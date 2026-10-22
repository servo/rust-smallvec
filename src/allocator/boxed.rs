use {
    super::{
        Allocator,
        like::Like,
        vec::Vec
    },
    core::{
        marker::PhantomData,
        ptr::NonNull,
        mem::MaybeUninit
    }
};

pub struct Box<Item: ?Sized, Heap: Allocator>(PhantomData<(Heap, Item)>);

impl<Item, Heap: Allocator> Like for Box<[Item], Heap> {
    type Item = Item;
    type Heap = Heap;
    #[cfg(not(feature = "allocator-api"))]
    type Type = alloc::boxed::Box<[Item]>;
    #[cfg(all(feature = "allocator-api", not(feature = "allocator-api2")))]
    type Type = alloc::boxed::Box<[Item], Heap>;
    #[cfg(feature = "allocator-api2")]
    type Type = allocator_api2::boxed::Box<[Item], Heap>;
    fn destruct(value: Self::Type) -> (Option<NonNull<[MaybeUninit<Self::Item>]>>, usize, Self::Heap) {
        <Vec<Item, Heap> as Like>::destruct(value.into_vec())
    }
    fn construct(pointer: Option<NonNull<[MaybeUninit<Self::Item>]>>, length: usize, allocator: Self::Heap) -> Self::Type {
        <Vec::<Item, Heap> as Like>::construct(pointer, length, allocator).into_boxed_slice()
    }
}