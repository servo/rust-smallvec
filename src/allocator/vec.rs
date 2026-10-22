#[cfg(not(feature = "allocator-api2"))]
pub use alloc::vec;
#[cfg(feature = "allocator-api2")]
pub use allocator_api2::vec;
use {
    super::{
        Allocator,
        like::Like
    },
    core::{
        marker::PhantomData,
        mem::MaybeUninit,
        ptr::{
            NonNull,
            slice_from_raw_parts_mut
        }
    }
};

pub struct Vec<Item, Heap: Allocator>(PhantomData<(Heap, Item)>);

impl<Item, Heap: Allocator> Like for Vec<Item, Heap> {
    type Heap = Heap;
    type Item = Item;
    #[cfg(not(feature = "allocator-api"))]
    type Type = alloc::vec::Vec<Item>;
    #[cfg(all(feature = "allocator-api", not(feature = "allocator-api2")))]
    type Type = alloc::vec::Vec<Item, Heap>;
    #[cfg(feature = "allocator-api2")]
    type Type = allocator_api2::vec::Vec<Item, Heap>;

    #[cfg(not(feature = "allocator-api"))]
    fn destruct(
        mut value: Self::Type
    ) -> (
        Option<NonNull<[MaybeUninit<Self::Item>]>>,
        usize,
        Self::Heap
    ) {
        let pointer = value.as_mut_ptr();
        let length = value.len();
        let capacity = value.capacity();
        (
            NonNull::new(slice_from_raw_parts_mut(pointer.cast(), capacity)),
            length,
            unsafe { MaybeUninit::uninit().assume_init() }
        )
    }

    #[cfg(all(feature = "allocator-api", not(feature = "allocator-api2")))]
    fn destruct(
        value: Self::Type
    ) -> (
        Option<NonNull<[MaybeUninit<Self::Item>]>>,
        usize,
        Self::Heap
    ) {
        let (pointer, length, capacity, allocator) = value.into_raw_parts_with_allocator();
        (
            NonNull::new(slice_from_raw_parts_mut(pointer.cast(), capacity)),
            length,
            allocator
        )
    }

    #[cfg(feature = "allocator-api2")]
    fn destruct(
        value: Self::Type
    ) -> (
        Option<NonNull<[MaybeUninit<Self::Item>]>>,
        usize,
        Self::Heap
    ) {
        let (pointer, length, capacity, allocator) = value.into_raw_parts_with_alloc();
        (
            NonNull::new(slice_from_raw_parts_mut(pointer.cast(), capacity)),
            length,
            allocator
        )
    }

    #[cfg(not(feature = "allocator-api"))]
    fn construct(
        pointer: Option<NonNull<[MaybeUninit<Self::Item>]>>,
        length: usize,
        _allocator: Self::Heap
    ) -> Self::Type {
        if let Some(pointer) = pointer {
            unsafe { Self::Type::from_raw_parts(pointer.as_ptr().cast(), length, pointer.len()) }
        } else {
            let mut vector = Self::Type::new();
            if size_of::<Self::Item>() == 0 {
                // SAFETY: we can change the length freely as copying the item
                // because it is a ZST
                unsafe { vector.set_len(length) }
            };
            vector
        }
    }

    #[cfg(all(feature = "allocator-api", not(feature = "allocator-api2")))]
    fn construct(
        pointer: Option<NonNull<[MaybeUninit<Self::Item>]>>,
        length: usize,
        allocator: Self::Heap
    ) -> Self::Type {
        if let Some(pointer) = pointer {
            unsafe {
                Self::Type::from_raw_parts_in(
                    pointer.as_ptr().cast(),
                    length,
                    pointer.len(),
                    allocator
                )
            }
        } else {
            let mut vector = Self::Type::new_in(allocator);
            if size_of::<Self::Item>() == 0 {
                // SAFETY: we can change the length freely as copying the item
                // because it is a ZST
                unsafe { vector.set_len(length) }
            };
            vector
        }
    }

    #[cfg(feature = "allocator-api2")]
    fn construct(
        pointer: Option<NonNull<[MaybeUninit<Self::Item>]>>,
        length: usize,
        allocator: Self::Heap
    ) -> Self::Type {
        if let Some(pointer) = pointer {
            unsafe {
                Self::Type::from_raw_parts_in(
                    pointer.as_ptr().cast(),
                    length,
                    pointer.len(),
                    allocator
                )
            }
        } else {
            let mut vector = Self::Type::new_in(allocator);
            if size_of::<Self::Item>() == 0 {
                // SAFETY: we can change the length freely as copying the item
                // because it is a ZST
                unsafe { vector.set_len(length) }
            };
            vector
        }
    }
}
