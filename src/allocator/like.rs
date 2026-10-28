use {
    super::Allocator,
    core::{
        mem::MaybeUninit,
        ptr::NonNull
    }
};

pub trait Like {
    type Type;
    type Item;
    type Heap: Allocator;
    unsafe fn destruct(
        value: Self::Type
    ) -> (
        Option<NonNull<[MaybeUninit<Self::Item>]>>,
        usize,
        Self::Heap
    );
    unsafe fn construct(
        pointer: Option<NonNull<[MaybeUninit<Self::Item>]>>,
        length: usize,
        allocator: Self::Heap
    ) -> Self::Type;
}
