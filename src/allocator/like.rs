use {
    super::Allocator,
    core::{
        mem::{
            MaybeUninit,
            ManuallyDrop
        },
        ptr::NonNull
    }
};

pub trait Like {
    type Type;
    type Item;
    type Heap: Allocator;
    fn destruct(
        value: ManuallyDrop<Self::Type>
    ) -> (
        Option<NonNull<[MaybeUninit<Self::Item>]>>,
        usize,
        Self::Heap
    );
    fn construct(
        pointer: Option<NonNull<[MaybeUninit<Self::Item>]>>,
        length: usize,
        allocator: Self::Heap
    ) -> Self::Type;
}
