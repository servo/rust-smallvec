use core::mem::{ManuallyDrop, MaybeUninit};
use core::ptr::NonNull;

/// Either a stack array with `length <= N` or a heap array
/// whose pointer and capacity are stored here.
///
/// We store a `NonNull<T>` instead of a `*mut T` so that type is covariant
/// with respect to `T`, and since the heap pointer is never null.
#[repr(C)]
pub union RawSmallVec<T, const N: usize> {
    pub inline: ManuallyDrop<MaybeUninit<[T; N]>>,
    pub heap: (NonNull<T>, usize),
}
