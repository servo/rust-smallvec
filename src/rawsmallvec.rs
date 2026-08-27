use super::{Allocator, Global};
use core::mem::{ManuallyDrop, MaybeUninit};
use core::ptr::NonNull;

/// Either a stack array with `length <= N` or a heap array
/// whose pointer and capacity are stored here.
///
/// We store a `NonNull<T>` instead of a `*mut T` so that type is covariant
/// with respect to `T`, and since the heap pointer is never null.
#[repr(C)]
pub union RawSmallVecUnion<T, const N: usize> {
    pub inline: ManuallyDrop<MaybeUninit<[T; N]>>,
    pub heap: (NonNull<T>, usize),
}

/// A wrapper around a [`RawSmallVecUnion`] and an allocator.
/// It is assumed that any pointer inside the `heap` field
/// was allocated using this allocator.
#[repr(C)]
pub struct RawSmallVec<T, const N: usize, A: Allocator = Global> {
    pub inner: RawSmallVecUnion<T, N>,
    pub allocator: A,
}
