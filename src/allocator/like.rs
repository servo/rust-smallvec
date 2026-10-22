use {
    super::Allocator,
    core::{
        ptr::NonNull,
        mem::MaybeUninit
    }
};

pub trait Like {
    type Type;
    type Item;
    type Heap: Allocator;
    fn destruct(value: Self::Type) -> (Option<NonNull<[MaybeUninit<Self::Item>]>>, usize, Self::Heap);
    fn construct(pointer: Option<NonNull<[MaybeUninit<Self::Item>]>>, length: usize, allocator: Self::Heap) -> Self::Type;
}
